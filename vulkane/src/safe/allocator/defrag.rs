// SPDX-License-Identifier: MIT OR Apache-2.0
//! Defragmentation planning and application on [`Allocator`].

use super::*;

/// One source -> destination move computed by [`Allocator::build_defragmentation_plan`].
///
/// The user is responsible for issuing the GPU work that copies the
/// data from `(src_memory, src_offset)` to `(dst_memory, dst_offset)`,
/// destroying the old `VkBuffer`/`VkImage`, creating a new one bound to
/// `(dst_memory, dst_offset)`, and waiting for the copy to complete
/// before calling [`Allocator::apply_defragmentation_plan`]. The
/// `user_data` field surfaces the value the caller attached at
/// allocation time, so apps can map a move back to whichever buffer or
/// image the allocation was bound to.
#[derive(Debug, Clone)]
pub struct DefragmentationMove {
    pub allocation_id: u64,
    pub user_data: u64,
    pub size: u64,
    pub src_memory: VkDeviceMemory,
    pub src_offset: u64,
    pub dst_memory: VkDeviceMemory,
    pub dst_offset: u64,
}

/// A complete defragmentation plan returned by
/// [`Allocator::build_defragmentation_plan`].
#[derive(Debug, Clone, Default)]
pub struct DefragmentationPlan {
    /// The list of moves the user must execute on the GPU side. Each
    /// entry has `src != dst`. The user issues a copy from src to dst
    /// for each.
    pub moves: Vec<DefragmentationMove>,
    /// The complete target layout for every live allocation in the
    /// pool, including ones that are already in the right place. The
    /// allocator uses this internally to rebuild its TLSF state in
    /// `apply_defragmentation_plan`. Exposed read-only via
    /// [`total_layout`](Self::total_layout) for diagnostics.
    pub(crate) total_layout: Vec<DefragmentationMove>,
    /// The pool the plan was built for. `apply_defragmentation_plan`
    /// uses this to know which pool to mutate.
    pub(crate) pool_id: u64,
    /// Estimated bytes that will be freed if every move is applied.
    pub bytes_freed: u64,
}

impl DefragmentationPlan {
    /// The complete target layout for every live allocation in the
    /// pool, including allocations that are already in the right place.
    /// Mostly useful for diagnostics; the [`moves`](Self::moves) field
    /// is what most callers want.
    pub fn total_layout(&self) -> &[DefragmentationMove] {
        &self.total_layout
    }
}

impl Allocator {
    /// Compute a defragmentation plan for the given custom pool.
    ///
    /// The plan compacts every live allocation in the pool to the start
    /// of the lowest-numbered block, ordered by `(block_index, offset)`.
    ///
    /// `pool` must be a `FreeList` (TLSF) custom pool. Linear pools
    /// don't need defragmentation — call [`reset_pool`](Self::reset_pool)
    /// to reclaim them in bulk.
    ///
    /// Returns an empty plan if the pool is unknown, is a Linear pool,
    /// or has no live allocations.
    pub fn build_defragmentation_plan(&self, pool: PoolHandle) -> DefragmentationPlan {
        let mut state = self.inner.pools.lock().unwrap();
        let p = match state.custom_pools.get_mut(&pool.0) {
            Some(p) if p.strategy == AllocationStrategy::FreeList => p,
            _ => return DefragmentationPlan::default(),
        };

        // Prune dead weak refs first.
        p.live_allocations.retain(|w| w.upgrade().is_some());

        // Snapshot every live allocation.
        struct LiveAlloc {
            arc: Arc<AllocationInner>,
            current_memory: VkDeviceMemory,
            current_offset: u64,
            block_index: u32,
            size: u64,
        }

        let mut lives: Vec<LiveAlloc> = Vec::new();
        for w in &p.live_allocations {
            let Some(arc) = w.upgrade() else {
                continue;
            };
            let loc = arc.location.lock().unwrap();
            let block_index = match &loc.kind {
                AllocationKind::CustomPool { block_index, .. } => *block_index,
                _ => continue,
            };
            lives.push(LiveAlloc {
                current_memory: loc.memory,
                current_offset: loc.offset,
                block_index,
                size: arc.size,
                arc: Arc::clone(&arc),
            });
        }

        // Sort by (block, offset) so the compaction is stable.
        lives.sort_by_key(|l| (l.block_index, l.current_offset));

        if p.blocks.is_empty() {
            return DefragmentationPlan::default();
        }
        let target_memory = p.blocks[0].memory;

        // Lay everything out at the start of block 0 in sequence,
        // 256-byte aligned (TLSF's SMALL_BLOCK_SIZE).
        let mut total_layout = Vec::new();
        let mut moves = Vec::new();
        let mut next_offset: u64 = 0;
        let mut bytes_freed: u64 = 0;
        const ALIGN: u64 = 256;

        for live in &lives {
            let aligned_next = (next_offset + ALIGN - 1) & !(ALIGN - 1);
            let entry = DefragmentationMove {
                allocation_id: live.arc.id,
                user_data: live.arc.user_data,
                size: live.size,
                src_memory: live.current_memory,
                src_offset: live.current_offset,
                dst_memory: target_memory,
                dst_offset: aligned_next,
            };
            let unchanged =
                aligned_next == live.current_offset && target_memory == live.current_memory;
            if !unchanged {
                moves.push(entry.clone());
                bytes_freed += live.size;
            }
            total_layout.push(entry);
            next_offset = aligned_next + live.size;
        }

        DefragmentationPlan {
            moves,
            total_layout,
            pool_id: pool.0,
            bytes_freed,
        }
    }

    /// Apply a previously computed defragmentation plan.
    ///
    /// **Preconditions** (the user must guarantee these):
    /// - The user has issued GPU copy commands (e.g. via
    ///   `vkCmdCopyBuffer`) for every entry in `plan.moves`.
    /// - The user has destroyed their old `Buffer`/`Image` handles and
    ///   created new ones bound to the `(dst_memory, dst_offset)` of
    ///   each move (or rebound them via the existing handle if their
    ///   API permits).
    /// - The user has waited for that GPU work to complete (e.g. via
    ///   `Fence::wait`).
    /// - No other thread is racing on the affected `Allocation`s.
    ///
    /// After this call, every live `Allocation` in the pool returns its
    /// new `(memory, offset)` from `Allocation::memory()` /
    /// `Allocation::offset()`. The plan is consumed.
    pub fn apply_defragmentation_plan(&self, plan: DefragmentationPlan) {
        if plan.total_layout.is_empty() {
            return;
        }
        let pool_id = plan.pool_id;
        let mut state = self.inner.pools.lock().unwrap();

        // Build an id -> Arc<AllocationInner> lookup so we can update
        // every live allocation in this pool.
        let mut by_id: HashMap<u64, Arc<AllocationInner>> = HashMap::new();
        if let Some(pool) = state.custom_pools.get(&pool_id) {
            for w in &pool.live_allocations {
                if let Some(arc) = w.upgrade() {
                    by_id.insert(arc.id, arc);
                }
            }
        } else {
            return;
        }

        // Reset every TLSF block in the pool. This invalidates every
        // tlsf_block_id but leaves the underlying VkDeviceMemory alone.
        if let Some(pool) = state.custom_pools.get_mut(&pool_id) {
            for block in pool.blocks.iter_mut() {
                if let BlockStrategy::Tlsf(ref mut t) = block.strategy {
                    let cap = t.capacity();
                    *t = Tlsf::new(cap);
                }
            }
        }

        // Re-allocate each entry in `total_layout` (both moved and
        // unmoved) at its target position. The plan was built so the
        // entries are sorted in ascending offset order, so TLSF will
        // hand them back exactly the requested offsets.
        for entry in &plan.total_layout {
            // Find which block index the dst_memory corresponds to.
            let target_block_index = if let Some(pool) = state.custom_pools.get(&pool_id) {
                pool.blocks
                    .iter()
                    .position(|b| b.memory == entry.dst_memory)
                    .map(|i| i as u32)
            } else {
                None
            };
            let Some(block_index) = target_block_index else {
                continue;
            };

            // Allocate `size` bytes with the same 256-byte alignment
            // build_defragmentation_plan used. TLSF will pick the
            // lowest free offset, which matches what the plan computed.
            const ALIGN: u64 = 256;
            if let Some(pool) = state.custom_pools.get_mut(&pool_id)
                && let BlockStrategy::Tlsf(ref mut t) = pool.blocks[block_index as usize].strategy
                && let Some(ta) = t.allocate(entry.size, ALIGN)
                && let Some(arc) = by_id.get(&entry.allocation_id)
            {
                let mut loc = arc.location.lock().unwrap();
                loc.memory = entry.dst_memory;
                loc.offset = ta.offset;
                loc.kind = AllocationKind::CustomPool {
                    pool_id,
                    block_index,
                    tlsf_block_id: ta.block_id,
                };
                // mapped_ptr stays the same — same block, same parent
                // mapping.
            }
        }

        self.refresh_free_region_count(&mut state);

        // Resolve the pool's heap before dropping `state` so we can
        // fire pressure with the lock released. Defrag doesn't actually
        // free VRAM (it just relocates inside existing blocks), but
        // users typically destroy no-longer-needed blocks after
        // applying a plan — evaluating callbacks here lets them see
        // the resulting state.
        let affected_heap = state
            .custom_pools
            .get(&pool_id)
            .map(|p| self.inner.heap_index_for_memory_type(p.memory_type_index))
            .unwrap_or(u32::MAX);
        drop(state);
        if affected_heap != u32::MAX {
            self.inner.fire_pressure_for_heap(affected_heap, None);
        }
    }
}
