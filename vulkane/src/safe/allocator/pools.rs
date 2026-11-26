// SPDX-License-Identifier: MIT OR Apache-2.0
//! Custom pools on [`Allocator`].

use super::*;

impl Allocator {
    /// Create a custom user pool. Returns an opaque [`PoolHandle`] you
    /// can pass to [`AllocationCreateInfo::pool`] to allocate from this
    /// pool, and to [`destroy_pool`](Self::destroy_pool),
    /// [`reset_pool`](Self::reset_pool), and
    /// [`pool_statistics`](Self::pool_statistics) to manage it.
    ///
    /// `Linear` pools always grow to exactly one block. `FreeList`
    /// pools grow on demand up to `info.max_block_count` (or unlimited
    /// when `max_block_count == 0`).
    pub fn create_pool(&self, info: PoolCreateInfo) -> Result<PoolHandle> {
        let mut state = self.inner.pools.lock().unwrap();
        let id = state.next_pool_id;
        state.next_pool_id += 1;
        let block_size = self.heap_block_size_for_type(info.memory_type_index);
        let pool = Pool::new_custom(info, block_size, false);
        state.custom_pools.insert(id, pool);
        Ok(PoolHandle(id))
    }

    /// Destroy a custom pool, freeing all its underlying
    /// `VkDeviceMemory` blocks. The caller must ensure no live
    /// allocations from this pool are still in use.
    pub fn destroy_pool(&self, handle: PoolHandle) {
        let affected_heap = {
            let mut state = self.inner.pools.lock().unwrap();
            let Some(mut pool) = state.custom_pools.remove(&handle.0) else {
                return;
            };
            let heap = self
                .inner
                .heap_index_for_memory_type(pool.memory_type_index);
            for block in pool.blocks.drain(..) {
                self.free_block_memory(&block);
                state.statistics.block_bytes =
                    state.statistics.block_bytes.saturating_sub(block.capacity);
                state.statistics.block_count = state.statistics.block_count.saturating_sub(1);
                // Subtract any still-counted allocation bytes.
                let alloc_count = block.strategy.allocation_count();
                state.statistics.allocation_count = state
                    .statistics
                    .allocation_count
                    .saturating_sub(alloc_count);
            }
            self.refresh_free_region_count(&mut state);
            heap
        };
        if affected_heap != u32::MAX {
            self.inner.fire_pressure_for_heap(affected_heap, None);
        }
    }

    /// Reset a Linear-strategy pool. All allocations from this pool
    /// become invalid; the user is responsible for ensuring no live
    /// references remain. No-op for `FreeList` pools.
    pub fn reset_pool(&self, handle: PoolHandle) {
        let affected_heap = {
            let mut state = self.inner.pools.lock().unwrap();
            let Some(pool) = state.custom_pools.get_mut(&handle.0) else {
                return;
            };
            if pool.strategy != AllocationStrategy::Linear {
                return;
            }
            let heap = self
                .inner
                .heap_index_for_memory_type(pool.memory_type_index);
            // Count all the allocations we're about to invalidate.
            let mut total_count = 0u32;
            let mut total_bytes = 0u64;
            for block in pool.blocks.iter_mut() {
                total_count += block.strategy.allocation_count();
                if let BlockStrategy::Linear(ref mut l) = block.strategy {
                    total_bytes += l.used_bytes();
                    l.reset();
                }
            }
            state.statistics.allocation_count = state
                .statistics
                .allocation_count
                .saturating_sub(total_count);
            state.statistics.allocation_bytes = state
                .statistics
                .allocation_bytes
                .saturating_sub(total_bytes);
            self.refresh_free_region_count(&mut state);
            heap
        };
        // Resetting a linear pool doesn't free `VkDeviceMemory`, so the
        // driver-reported budget `usage` typically doesn't move — but
        // we still evaluate pressure so callbacks that watch only
        // the allocator's own state (via `statistics`) have a hook if
        // future helpers surface that view.
        if affected_heap != u32::MAX {
            self.inner.fire_pressure_for_heap(affected_heap, None);
        }
    }

    /// Snapshot statistics for one custom pool. Returns `None` if the
    /// handle is unknown (or has been destroyed).
    pub fn pool_statistics(&self, handle: PoolHandle) -> Option<AllocationStatistics> {
        let state = self.inner.pools.lock().unwrap();
        let pool = state.custom_pools.get(&handle.0)?;
        let mut stats = AllocationStatistics::default();
        for block in &pool.blocks {
            stats.block_count += 1;
            stats.block_bytes += block.capacity;
            stats.allocation_count += block.strategy.allocation_count();
            stats.free_region_count += block.strategy.free_region_count();
            match &block.strategy {
                BlockStrategy::Tlsf(t) => stats.allocation_bytes += t.used_bytes(),
                BlockStrategy::Linear(l) => stats.allocation_bytes += l.used_bytes(),
            }
        }
        if stats.allocation_bytes > stats.peak_allocation_bytes {
            stats.peak_allocation_bytes = stats.allocation_bytes;
        }
        Some(stats)
    }
}
