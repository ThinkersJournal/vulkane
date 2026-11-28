// SPDX-License-Identifier: MIT OR Apache-2.0
//! VMA-style sub-allocator for Vulkan device memory.
//!
//! Real Vulkan applications allocate thousands of buffers and images. The
//! "one `vkAllocateMemory` per resource" pattern hits the
//! `maxMemoryAllocationCount` limit (~4096 on most drivers) very quickly,
//! and is wasteful besides — every individual `VkDeviceMemory` carries
//! per-allocation overhead in the driver.
//!
//! [`Allocator`] solves this by sub-allocating from larger
//! [`Block`](Block)s (one `VkDeviceMemory` each) using a
//! [Two-Level Segregated Fit](tlsf::Tlsf) free-list strategy. The API is
//! deliberately modeled after AMD's
//! [VulkanMemoryAllocator](https://gpuopen.com/vulkan-memory-allocator/) so
//! that users coming from C++ Vulkan find the concepts familiar:
//!
//! - **General-purpose pool** (TLSF) — long-lived mixed-size allocations.
//! - **Linear pool** (bump allocator) — stack/ring/single-frame uploads.
//! - **Dedicated allocations** — single-resource `vkAllocateMemory` for
//!   resources that are too large for the pool block size, that the driver
//!   requires to be dedicated, or that the user explicitly opts into.
//! - **Memory-type selection** by `(required, preferred)` property flag
//!   pair.
//! - **Statistics** — per-pool, per-type, and total: bytes used, bytes
//!   free, allocation count, free-region count, peak usage.
//! - **Budget queries** — when `VK_EXT_memory_budget` is enabled and
//!   loadable, [`Allocator::query_budget`] returns the per-heap soft
//!   budget the driver suggests respecting.
//!
//! The default block size is **256 MiB** for "large" heaps (≥ 4 GiB) and
//! **64 MiB** for "small" heaps. Allocations larger than half the block
//! size automatically fall through to a dedicated allocation. These
//! thresholds match VMA's defaults.
//!
//! ## Example
//!
//! ```ignore
//! use vulkane::safe::{Allocator, AllocationCreateInfo, AllocationUsage,
//!                   Buffer, BufferCreateInfo, BufferUsage};
//!
//! let allocator = Allocator::new(&device, &physical)?;
//!
//! // A device-local storage buffer for compute output.
//! let (buffer, alloc) = allocator.create_buffer(
//!     BufferCreateInfo {
//!         size: 4 * 1024 * 1024,
//!         usage: BufferUsage::STORAGE_BUFFER,
//!     },
//!     AllocationCreateInfo {
//!         usage: AllocationUsage::DeviceLocal,
//!         ..Default::default()
//!     },
//! )?;
//!
//! // ... use the buffer ...
//!
//! // The Drop impl on `alloc` returns the sub-allocation to its TLSF
//! // pool; the buffer's own Drop calls vkDestroyBuffer.
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::raw::bindings::*;

use super::buffer::{Buffer, BufferCreateInfo, MemoryRequirements};
use super::device_inner::DeviceInner;
use super::image::{Image, Image2dCreateInfo};
use super::physical::PhysicalDevice;
use super::{Device, Error, Result, check};

mod budget;
mod defrag;
mod linear;
mod memory_type;
mod pools;
mod raw;
mod tlsf;

pub use self::defrag::{DefragmentationMove, DefragmentationPlan};
use self::linear::Linear;
use self::memory_type::pick_memory_type_in;
use self::tlsf::{Tlsf, TlsfAllocation};

/// Default block size for small heaps (< 4 GiB), in bytes.
const SMALL_HEAP_BLOCK_SIZE: u64 = 64 * 1024 * 1024;
/// Default block size for large heaps (>= 4 GiB), in bytes.
const LARGE_HEAP_BLOCK_SIZE: u64 = 256 * 1024 * 1024;
/// Heap-size threshold for picking the larger block size.
const LARGE_HEAP_THRESHOLD: u64 = 4 * 1024 * 1024 * 1024;

/// Hint about how the user intends to access the allocation. This drives
/// memory-type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationUsage {
    /// "I don't care, pick whatever works." Mostly equivalent to
    /// `DeviceLocal` on discrete GPUs.
    #[default]
    Auto,
    /// Prefer `DEVICE_LOCAL`. Fastest GPU access; not directly mappable on
    /// discrete GPUs.
    ///
    /// This is a preference, not a guarantee: if the resource's
    /// `memoryTypeBits` admits no device-local type, the allocation
    /// succeeds on whatever type is available rather than failing — which
    /// makes it indistinguishable from [`Auto`](Self::Auto) in that case.
    /// Check [`Allocation::memory_type_index`] against
    /// [`PhysicalDevice::memory_properties`] if you need to know what you
    /// actually got.
    DeviceLocal,
    /// `HOST_VISIBLE | HOST_COHERENT`, preferring a type that is *not*
    /// `DEVICE_LOCAL`. Mappable from the host; suitable for staging
    /// buffers and persistent uploads.
    ///
    /// The device-local avoidance keeps staging out of the PCIe BAR
    /// window, whose exhaustion surfaces as a driver-level failure rather
    /// than a catchable `VK_ERROR_OUT_OF_DEVICE_MEMORY`. It falls back to
    /// a device-local host-visible type when the device offers no other —
    /// on UMA parts there is nothing else to pick.
    ///
    /// Note this never requests `HOST_CACHED`, so CPU *reads* from the
    /// result are uncached. For a readback pool, pin the memory type
    /// explicitly with [`PhysicalDevice::find_memory_type`] and
    /// [`Allocator::create_pool`], which bypasses usage-based selection.
    HostVisible,
    /// `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` (BAR / ReBAR /
    /// integrated GPUs) — the deliberate way to ask for BAR memory. Falls
    /// back to `HostVisible` when not available.
    HostVisibleDeviceLocal,
}

/// Strategy used to lay out allocations within a pool's memory blocks.
///
/// `FreeList` (TLSF) is the default and the right choice for general
/// long-lived mixed-size allocations: O(1) alloc/free, low fragmentation.
///
/// `Linear` is a bump pointer that only frees on
/// [`Allocator::reset_pool`]; suitable for transient per-frame or
/// per-pass allocations where you batch-reset everything at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationStrategy {
    /// Two-Level Segregated Fit free-list (TLSF) — general purpose.
    #[default]
    FreeList,
    /// Linear bump allocator — supports `reset_pool` only.
    Linear,
}

/// Opaque handle to a custom user pool created via
/// [`Allocator::create_pool`]. Pass to
/// [`AllocationCreateInfo::pool`] to allocate from this pool, and to
/// [`Allocator::destroy_pool`] / [`Allocator::reset_pool`] /
/// [`Allocator::pool_statistics`] to manage it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolHandle(u64);

/// Opaque identifier for a registered budget-pressure callback.
/// Returned by [`Allocator::register_pressure_callback`]; pass to
/// [`Allocator::unregister_pressure_callback`] to remove it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PressureCallbackId(u64);

/// What triggered a [`PressureEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureKind {
    /// The heap's usage-vs-budget fraction just **rose past** the
    /// registered threshold. Fired after an allocation succeeded.
    Crossed,
    /// The heap's usage-vs-budget fraction just **fell below**
    /// `threshold - hysteresis`. Fired after a free,
    /// [`Allocator::reset_pool`], or
    /// [`Allocator::apply_defragmentation_plan`].
    Relieved,
    /// A proactive check (via [`Allocator::would_fit`]) found that the
    /// pending allocation would push the heap past the registered
    /// threshold **if it were attempted**. Does not imply the
    /// allocation was actually issued, and does not latch the
    /// callback's internal "currently over" flag — a subsequent
    /// successful allocation will still fire [`PressureKind::Crossed`].
    Predictive,
}

/// Payload delivered to a pressure-callback closure.
///
/// See [`Allocator::register_pressure_callback`] for the contract.
#[derive(Debug, Clone, Copy)]
pub struct PressureEvent {
    /// Which memory heap crossed the threshold. Matches the indices
    /// used by [`crate::safe::MemoryBudget`].
    pub heap_index: u32,
    /// The heap's bytes-in-use at the moment of firing.
    pub current_usage: u64,
    /// The driver's soft budget for this heap at the moment of firing.
    pub budget: u64,
    /// `current_usage / budget`, or 0.0 when `budget == 0`.
    pub fraction: f64,
    /// The threshold that fired. For [`PressureKind::Crossed`] and
    /// [`PressureKind::Predictive`] this is the registered threshold;
    /// for [`PressureKind::Relieved`] it is also the registered
    /// threshold (the callback fires once when usage drops below
    /// `threshold - hysteresis`).
    pub threshold: f64,
    /// What caused the event to fire.
    pub kind: PressureKind,
    /// For [`PressureKind::Predictive`] only: the byte size the caller
    /// asked about in [`Allocator::would_fit`]. Zero for
    /// [`PressureKind::Crossed`] and [`PressureKind::Relieved`].
    pub predictive_size: u64,
}

/// Result of [`Allocator::would_fit`]: the proactive check that
/// predicts whether a forthcoming allocation will violate the driver's
/// soft budget on the relevant heap.
#[derive(Debug, Clone, Copy)]
pub struct FitStatus {
    /// Heap that would back this allocation.
    pub heap_index: u32,
    /// Current bytes in use on this heap per the driver's estimate.
    /// (Zero when the budget query is unsupported — compare with
    /// [`Allocator::has_memory_budget_support`] to distinguish "heap
    /// is empty" from "no query support".)
    pub current_usage: u64,
    /// Driver-provided soft budget on this heap.
    pub budget: u64,
    /// Byte size the caller asked about.
    pub requested: u64,
    /// `current_usage + requested` — what usage would become.
    pub projected_usage: u64,
    /// `true` iff `projected_usage <= budget`. Also `true` when the
    /// budget query is unsupported (we cannot prove the allocation
    /// would fail).
    pub fits: bool,
    /// `projected_usage / budget`, or 0.0 when `budget == 0`.
    pub projected_fraction: f64,
}

/// Parameters for [`Allocator::create_pool`]. Custom pools let users
/// segregate allocations (e.g. all transient buffers from one frame go
/// into a single linear pool that gets reset together) and pick a
/// non-default strategy.
#[derive(Debug, Clone, Copy)]
pub struct PoolCreateInfo {
    /// Memory type the pool will allocate from. Use
    /// [`PhysicalDevice::find_memory_type`] to pick this.
    pub memory_type_index: u32,
    /// Strategy to use within each block.
    pub strategy: AllocationStrategy,
    /// Initial / per-block size in bytes. Set to 0 to use the
    /// allocator's default for the heap (256 MiB on heaps ≥ 4 GiB,
    /// 64 MiB on smaller heaps).
    pub block_size: u64,
    /// Maximum number of blocks the pool may grow to. Set to 0 for
    /// unlimited (the default). Linear pools always have exactly 1 block.
    pub max_block_count: u32,
}

/// Per-allocation knobs the user can set when creating a resource.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocationCreateInfo {
    /// Hint about how the allocation will be used.
    pub usage: AllocationUsage,
    /// If `true`, force a dedicated `vkAllocateMemory` even when the
    /// pool could satisfy the request. Useful for resources whose contents
    /// are owned by an external API (e.g. swapchain images, exportable
    /// memory).
    pub dedicated: bool,
    /// If `true`, the underlying block is mapped persistently and the
    /// allocation exposes a host pointer for the lifetime of the
    /// allocation. Only meaningful for host-visible memory.
    pub mapped: bool,
    /// Allocate from a specific custom pool (created via
    /// [`Allocator::create_pool`]). When `None`, the default per-memory-type
    /// pool is used.
    pub pool: Option<PoolHandle>,
    /// Opaque caller data attached to the allocation. Surfaced via
    /// [`Allocation::user_data`] and on each
    /// [`DefragmentationMove::user_data`] entry, so apps can map an
    /// allocation back to whatever buffer/image they bound it to. Common
    /// patterns: a `u64` cast of an `Arc<MyResource>` raw pointer, or
    /// a slot index into the app's own resource table.
    pub user_data: u64,
    /// Device-group mask: which physical devices in the
    /// [`Device`] group this allocation should live on. `None` (the
    /// default) lets the driver pick — typically all devices in the
    /// group.
    ///
    /// Setting this to `Some(mask)` forces a dedicated `vkAllocateMemory`
    /// (since each `VkDeviceMemory` block is allocated with one mask),
    /// chains a `VkMemoryAllocateFlagsInfo` with
    /// `VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT`, and sets `deviceMask` to
    /// `mask`. Setting `Some(0)` is invalid; use `None` to mean
    /// "default".
    ///
    /// Only meaningful when the [`Device`] was created from a multi-GPU
    /// [`PhysicalDeviceGroup`](super::PhysicalDeviceGroup) via
    /// [`PhysicalDeviceGroup::create_device`](super::PhysicalDeviceGroup::create_device).
    /// On a single-device group it's a no-op (the driver always uses
    /// device 0).
    pub device_mask: Option<u32>,
}

/// Aggregate statistics for the [`Allocator`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocationStatistics {
    /// Total `VkDeviceMemory` bytes the allocator currently owns
    /// (allocated from Vulkan, regardless of whether they're sub-allocated
    /// to user resources).
    pub block_bytes: u64,
    /// Bytes the allocator has handed out as user allocations.
    pub allocation_bytes: u64,
    /// Number of `VkDeviceMemory` blocks currently held.
    pub block_count: u32,
    /// Number of currently outstanding user allocations.
    pub allocation_count: u32,
    /// Number of separate free regions across all blocks (a measure of
    /// fragmentation: lower is better).
    pub free_region_count: u32,
    /// Peak `allocation_bytes` over the lifetime of this allocator.
    pub peak_allocation_bytes: u64,
    /// Number of allocations that fell through to dedicated
    /// `vkAllocateMemory` rather than being sub-allocated.
    pub dedicated_allocation_count: u32,
}

/// Strategy state stored alongside a block of device memory.
enum BlockStrategy {
    Tlsf(Tlsf),
    Linear(Linear),
}

impl BlockStrategy {
    fn allocation_count(&self) -> u32 {
        match self {
            Self::Tlsf(t) => t.allocation_count(),
            Self::Linear(l) => l.allocation_count(),
        }
    }
    fn free_region_count(&self) -> u32 {
        match self {
            Self::Tlsf(t) => t.free_region_count(),
            // A linear allocator has at most one free region (the
            // unused tail).
            Self::Linear(l) => {
                if l.free_bytes() == 0 {
                    0
                } else {
                    1
                }
            }
        }
    }
}

/// One block of `VkDeviceMemory` plus its allocation strategy and
/// persistent mapping pointer.
#[allow(dead_code)] // capacity / memory_type_index are diagnostic-only fields.
struct Block {
    memory: VkDeviceMemory,
    capacity: u64,
    memory_type_index: u32,
    strategy: BlockStrategy,
    /// Persistent mapped pointer, if the block was created with `mapped`.
    mapped_ptr: *mut std::ffi::c_void,
}

// Safety: Block is owned by Allocator (which is Send) and accessed only
// behind a Mutex.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

/// One pool of [`Block`]s, all backed by the same memory type. The
/// per-memory-type default pools are TLSF only; user-created custom
/// pools may use either strategy.
#[allow(dead_code)] // memory_type_index / block_size / max_block_count are diagnostic-only.
struct Pool {
    memory_type_index: u32,
    blocks: Vec<Block>,
    block_size: u64,
    max_block_count: u32,
    strategy: AllocationStrategy,
    mapped: bool,
    /// Weak references to every allocation handed out from this pool.
    /// Used by defragmentation to walk live allocations. Periodically
    /// pruned of dead entries.
    live_allocations: Vec<std::sync::Weak<AllocationInner>>,
}

impl Pool {
    fn new_default(memory_type_index: u32, block_size: u64) -> Self {
        Self {
            memory_type_index,
            blocks: Vec::new(),
            block_size,
            max_block_count: 0,
            strategy: AllocationStrategy::FreeList,
            mapped: false,
            live_allocations: Vec::new(),
        }
    }

    fn new_custom(info: PoolCreateInfo, default_block_size: u64, mapped: bool) -> Self {
        let block_size = if info.block_size == 0 {
            default_block_size
        } else {
            info.block_size
        };
        // Linear pools are always a single block.
        let max = if info.strategy == AllocationStrategy::Linear {
            1
        } else {
            info.max_block_count
        };
        Self {
            memory_type_index: info.memory_type_index,
            blocks: Vec::new(),
            block_size,
            max_block_count: max,
            strategy: info.strategy,
            mapped,
            live_allocations: Vec::new(),
        }
    }
}

/// Allocator-wide options set once at construction via
/// [`Allocator::new_with_options`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocatorOptions {
    /// When `true`, every `VkDeviceMemory` block this allocator allocates
    /// is created with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`, so any
    /// buffer sub-allocated from it (and created with
    /// [`BufferUsage::SHADER_DEVICE_ADDRESS`](super::BufferUsage::SHADER_DEVICE_ADDRESS))
    /// returns a valid GPU virtual address from
    /// [`Buffer::device_address`](super::Buffer::device_address).
    ///
    /// The flag lives on the *block*, not the buffer, because one block
    /// backs many sub-allocations — so it must be requested allocator-wide
    /// rather than per-buffer. This mirrors VMA's
    /// `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT`.
    ///
    /// Requires the `bufferDeviceAddress` device feature to have been
    /// enabled at device creation
    /// ([`DeviceFeatures::with_buffer_device_address`](super::DeviceFeatures::with_buffer_device_address));
    /// setting this without the feature is a validation error at the first
    /// allocation. Leave `false` if you don't use buffer device addresses —
    /// the bit has a small cost and some drivers reserve address space for
    /// it.
    pub buffer_device_address: bool,
}

/// Top-level allocator. Owns one pool per memory type and synthesizes
/// dedicated allocations for resources that don't fit the pool model.
///
/// `Allocator` is `Send + Sync`; internal state is protected by a single
/// `Mutex`. For most workloads this is plenty — Vulkan memory operations
/// are infrequent and the critical section is small. Highly contended
/// workloads can use multiple `Allocator` instances per device.
pub struct Allocator {
    inner: Arc<AllocatorInner>,
}

pub(crate) struct AllocatorInner {
    device: Arc<DeviceInner>,
    /// Kept for budget queries via VK_EXT_memory_budget.
    physical: PhysicalDevice,
    /// Cached `VkPhysicalDeviceMemoryProperties` so we don't re-query.
    memory_properties: VkPhysicalDeviceMemoryProperties,
    /// One pool per memory type. None until first use.
    pools: Mutex<PoolState>,
    /// Registered budget-pressure callbacks + their per-heap latch state.
    /// Held behind its own `Mutex` rather than folded into [`PoolState`]
    /// so we can invoke callbacks *after* releasing the pool lock,
    /// preventing callback code from re-entering the allocator and
    /// deadlocking on itself.
    pressure: Mutex<PressureState>,
    /// When set, every block is allocated with
    /// `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`. See
    /// [`AllocatorOptions::buffer_device_address`].
    buffer_device_address: bool,
}

/// Erased-type alias for a registered pressure callback. Used in a
/// few places that would otherwise hit `clippy::type_complexity`.
type PressureCallbackFn = Arc<dyn Fn(PressureEvent) + Send + Sync>;

/// Internal state for the pressure-callback system.
struct PressureState {
    next_id: u64,
    callbacks: Vec<PressureCallbackEntry>,
}

struct PressureCallbackEntry {
    id: u64,
    threshold: f64,
    hysteresis: f64,
    callback: PressureCallbackFn,
    /// Per-heap latch: `latched[h]` is `true` iff this callback has
    /// fired a `Crossed` event on heap `h` and has not yet fired the
    /// matching `Relieved` event. Vulkan caps heap count at
    /// `VK_MAX_MEMORY_HEAPS = 16`.
    latched: [bool; 16],
}

impl AllocatorInner {
    /// Accessor for the set of extensions enabled on the underlying
    /// [`Device`]. Thin shim used by [`Allocator::has_memory_budget_support`]
    /// so the allocator can ask "did VK_EXT_memory_budget get enabled?"
    /// without re-exporting the full `Device` public surface.
    fn device_enabled_extensions(&self) -> &[String] {
        &self.device.enabled_extensions
    }

    /// Which memory heap backs the given memory type. Returns
    /// `u32::MAX` for out-of-range indices (should never happen in
    /// correct code — the allocator itself generates every
    /// `memory_type_index` from a Vulkan query).
    fn heap_index_for_memory_type(&self, memory_type_index: u32) -> u32 {
        if memory_type_index >= self.memory_properties.memoryTypeCount {
            return u32::MAX;
        }
        self.memory_properties.memoryTypes[memory_type_index as usize].heapIndex
    }

    /// Evaluate every registered pressure callback against the current
    /// budget state for `heap_index`, latch the transitions, and invoke
    /// any callbacks whose threshold was just crossed or relieved.
    ///
    /// `predictive_size`: when `Some`, this is a
    /// [`PressureKind::Predictive`] check triggered from
    /// [`Allocator::would_fit`] — we evaluate the transitions against
    /// `current_usage + predictive_size` but **do not** update the
    /// `latched` bits (predictive events are advisory and must not
    /// suppress a real `Crossed` event on a subsequent allocation).
    /// When `None`, this is a reactive pass after an allocate/free —
    /// normal latch transitions apply.
    ///
    /// Never holds the [`PoolState`] lock; callers are expected to
    /// have dropped it before entering.
    fn fire_pressure_for_heap(&self, heap_index: u32, predictive_size: Option<u64>) {
        if heap_index >= self.memory_properties.memoryHeapCount || heap_index >= 16 {
            return;
        }
        // Snapshot budget. Safe to do without holding pools lock.
        let Some(budget) = self.physical.memory_budget() else {
            return;
        };
        let h = heap_index as usize;
        if h >= budget.heap_count as usize {
            return;
        }
        let raw_usage = budget.usage[h];
        let raw_budget = budget.budget[h];
        let projected_usage = match predictive_size {
            Some(n) => raw_usage.saturating_add(n),
            None => raw_usage,
        };
        let fraction = if raw_budget == 0 {
            0.0
        } else {
            projected_usage as f64 / raw_budget as f64
        };

        // Collect events to fire under the pressure lock, update latches,
        // then release the lock before invoking the callbacks so they
        // can call back into the allocator without deadlocking.
        let mut to_fire: Vec<(PressureCallbackFn, PressureEvent)> = Vec::new();
        {
            let mut ps = self.pressure.lock().unwrap();
            for cb in ps.callbacks.iter_mut() {
                let above = fraction >= cb.threshold;
                let below_low = fraction < (cb.threshold - cb.hysteresis).max(0.0);
                match predictive_size {
                    Some(n) => {
                        // Predictive: fire only when the projection
                        // would push us above the threshold AND we're
                        // not already latched (otherwise we'd be
                        // notifying about a state the caller already
                        // knows about via Crossed).
                        if above && !cb.latched[h] {
                            to_fire.push((
                                Arc::clone(&cb.callback),
                                PressureEvent {
                                    heap_index,
                                    current_usage: raw_usage,
                                    budget: raw_budget,
                                    fraction,
                                    threshold: cb.threshold,
                                    kind: PressureKind::Predictive,
                                    predictive_size: n,
                                },
                            ));
                        }
                    }
                    None => {
                        if above && !cb.latched[h] {
                            cb.latched[h] = true;
                            to_fire.push((
                                Arc::clone(&cb.callback),
                                PressureEvent {
                                    heap_index,
                                    current_usage: raw_usage,
                                    budget: raw_budget,
                                    fraction,
                                    threshold: cb.threshold,
                                    kind: PressureKind::Crossed,
                                    predictive_size: 0,
                                },
                            ));
                        } else if below_low && cb.latched[h] {
                            cb.latched[h] = false;
                            to_fire.push((
                                Arc::clone(&cb.callback),
                                PressureEvent {
                                    heap_index,
                                    current_usage: raw_usage,
                                    budget: raw_budget,
                                    fraction,
                                    threshold: cb.threshold,
                                    kind: PressureKind::Relieved,
                                    predictive_size: 0,
                                },
                            ));
                        }
                    }
                }
            }
        }
        for (cb, event) in to_fire {
            cb(event);
        }
    }

    /// Fire pressure for a memory-type index (resolves heap and
    /// delegates). Used by the allocate/free hot paths.
    fn fire_pressure_for_memory_type(&self, memory_type_index: u32, predictive_size: Option<u64>) {
        let heap = self.heap_index_for_memory_type(memory_type_index);
        if heap == u32::MAX {
            return;
        }
        self.fire_pressure_for_heap(heap, predictive_size);
    }

    /// Refresh the free-region count statistic across every pool. Called
    /// after any free/destroy that may have merged regions.
    fn refresh_free_region_count(&self, state: &mut PoolState) {
        let mut total = 0u32;
        for pool in state.pools.iter().flatten() {
            for block in &pool.blocks {
                total += block.strategy.free_region_count();
            }
        }
        for pool in state.custom_pools.values() {
            for block in &pool.blocks {
                total += block.strategy.free_region_count();
            }
        }
        state.statistics.free_region_count = total;
    }

    /// Unmap (if mapped) and `vkFreeMemory` a block. Used by `destroy_pool`,
    /// `free_allocation_internal`, and the allocator's own `Drop`.
    fn free_block_memory(&self, block: &Block) {
        self.free_raw_memory(block.memory, block.mapped_ptr);
    }

    /// Unmap (if `mapped_ptr` is non-null) and `vkFreeMemory` a device
    /// memory handle that is not — or is no longer — owned by a [`Block`].
    ///
    /// Needed on the failure paths between `vkAllocateMemory` and the point
    /// where the block is recorded in `PoolState`: an early return in that
    /// window strands the handle, because nothing holds it and so neither
    /// `Drop` nor `destroy_pool` can reclaim it. That window is reached
    /// exactly when `vkMapMemory` fails, which is what a driver reports
    /// when it cannot satisfy a host mapping — so leaking there would make
    /// each retry likelier to fail than the last.
    pub(crate) fn free_raw_memory(
        &self,
        memory: VkDeviceMemory,
        mapped_ptr: *mut std::ffi::c_void,
    ) {
        if !mapped_ptr.is_null()
            && let Some(unmap) = self.device.dispatch.vkUnmapMemory
        {
            // Safety: handle is valid; we are about to free it.
            unsafe { unmap(self.device.handle, memory) };
        }
        if let Some(free) = self.device.dispatch.vkFreeMemory {
            // Safety: handle is valid; we are the sole owner.
            unsafe { free(self.device.handle, memory, std::ptr::null()) };
        }
    }

    /// Return a single allocation to its pool. Called both from
    /// [`Allocator::free`] (explicitly) and from
    /// [`AllocationInner::Drop`] (implicitly when the last clone goes
    /// out of scope).
    pub(crate) fn free_allocation_internal(&self, location: &AllocationLocation, alloc_size: u64) {
        let mut state = self.pools.lock().unwrap();
        match location.kind {
            AllocationKind::DefaultPool {
                memory_type_index,
                block_index,
                tlsf_block_id,
            } => {
                if let Some(Some(pool)) = state.pools.get_mut(memory_type_index as usize)
                    && let Some(block) = pool.blocks.get_mut(block_index as usize)
                    && let BlockStrategy::Tlsf(ref mut t) = block.strategy
                {
                    t.free(TlsfAllocation {
                        offset: location.offset,
                        size: alloc_size,
                        block_id: tlsf_block_id,
                    });
                }
                state.statistics.allocation_count =
                    state.statistics.allocation_count.saturating_sub(1);
                state.statistics.allocation_bytes =
                    state.statistics.allocation_bytes.saturating_sub(alloc_size);
                self.refresh_free_region_count(&mut state);
            }
            AllocationKind::CustomPool {
                pool_id,
                block_index,
                tlsf_block_id,
            } => {
                if let Some(pool) = state.custom_pools.get_mut(&pool_id)
                    && let Some(block) = pool.blocks.get_mut(block_index as usize)
                    && let BlockStrategy::Tlsf(ref mut t) = block.strategy
                {
                    // Linear pool allocations cannot be freed
                    // individually — caller must use reset_pool. We
                    // silently no-op the per-alloc free in that case.
                    t.free(TlsfAllocation {
                        offset: location.offset,
                        size: alloc_size,
                        block_id: tlsf_block_id,
                    });
                }
                state.statistics.allocation_count =
                    state.statistics.allocation_count.saturating_sub(1);
                state.statistics.allocation_bytes =
                    state.statistics.allocation_bytes.saturating_sub(alloc_size);
                self.refresh_free_region_count(&mut state);
            }
            AllocationKind::Dedicated { id } => {
                if let Some(pos) = state.dedicated_blocks.iter().position(|d| d.id == id) {
                    let dedicated = state.dedicated_blocks.swap_remove(pos);
                    // Unmap if needed and free.
                    if !dedicated.mapped_ptr.is_null()
                        && let Some(unmap) = self.device.dispatch.vkUnmapMemory
                    {
                        // Safety: handle is valid.
                        unsafe { unmap(self.device.handle, dedicated.memory) };
                    }
                    if let Some(free) = self.device.dispatch.vkFreeMemory {
                        // Safety: handle is valid; we are the sole owner.
                        unsafe { free(self.device.handle, dedicated.memory, std::ptr::null()) };
                    }
                    state.statistics.block_count = state.statistics.block_count.saturating_sub(1);
                    state.statistics.block_bytes =
                        state.statistics.block_bytes.saturating_sub(dedicated.size);
                    state.statistics.dedicated_allocation_count = state
                        .statistics
                        .dedicated_allocation_count
                        .saturating_sub(1);
                    state.statistics.allocation_count =
                        state.statistics.allocation_count.saturating_sub(1);
                    state.statistics.allocation_bytes =
                        state.statistics.allocation_bytes.saturating_sub(alloc_size);
                }
            }
        }
    }
}

struct PoolState {
    /// Per-memory-type default TLSF pools, indexed by memory type.
    pools: Vec<Option<Pool>>,
    /// User-created custom pools, keyed by `PoolHandle`.
    custom_pools: HashMap<u64, Pool>,
    next_pool_id: u64,
    /// Stable ids for allocations, used by defragmentation to identify
    /// individual allocations across moves.
    next_alloc_id: u64,
    statistics: AllocationStatistics,
    dedicated_blocks: Vec<DedicatedBlock>,
}

/// Build an `Allocation` from its parts and register it in the
/// appropriate pool's live-allocation list (so defragmentation can walk
/// it later). Centralised so all allocation construction goes through
/// one path.
#[allow(clippy::too_many_arguments)] // every parameter is structural metadata
fn make_allocation(
    state: &mut PoolState,
    memory: VkDeviceMemory,
    offset: u64,
    size: u64,
    memory_type_index: u32,
    mapped_ptr: *mut std::ffi::c_void,
    kind: AllocationKind,
    user_data: u64,
    allocator: std::sync::Weak<AllocatorInner>,
) -> Allocation {
    let id = state.next_alloc_id;
    state.next_alloc_id += 1;
    let alloc = Allocation {
        inner: Arc::new(AllocationInner {
            location: Mutex::new(AllocationLocation {
                memory,
                offset,
                mapped_ptr,
                kind: kind.clone(),
            }),
            size,
            memory_type_index,
            user_data,
            id,
            allocator,
        }),
    };
    // Register the weak ref in the owning pool so defragmentation can
    // find it. Dedicated allocations are not registered (defrag only
    // operates on pool allocations).
    let weak = Arc::downgrade(&alloc.inner);
    match &kind {
        AllocationKind::DefaultPool {
            memory_type_index, ..
        } => {
            if let Some(Some(pool)) = state.pools.get_mut(*memory_type_index as usize) {
                pool.live_allocations.push(weak);
            }
        }
        AllocationKind::CustomPool { pool_id, .. } => {
            if let Some(pool) = state.custom_pools.get_mut(pool_id) {
                pool.live_allocations.push(weak);
            }
        }
        AllocationKind::Dedicated { .. } => {}
    }
    alloc
}

/// A `vkAllocateMemory`'d block that holds exactly one user allocation.
#[allow(dead_code)] // memory_type_index is diagnostic-only.
struct DedicatedBlock {
    memory: VkDeviceMemory,
    size: u64,
    memory_type_index: u32,
    mapped_ptr: *mut std::ffi::c_void,
    /// Slot id used to free this dedicated allocation. Stable for the
    /// lifetime of the allocator.
    id: u64,
}

// Safety: DedicatedBlock is only accessed inside the Allocator's Mutex,
// and the raw pointer is only used while the underlying VkDeviceMemory
// is still mapped (which we control).
unsafe impl Send for DedicatedBlock {}
unsafe impl Sync for DedicatedBlock {}

/// A live allocation handed out by the [`Allocator`]. Free it explicitly
/// by passing it to [`Allocator::free`], or by dropping the
/// [`Buffer`]/[`Image`] returned by `create_buffer` / `create_image_2d`.
///
/// `Allocation` is `Clone` (cheap — internally a single `Arc`) so apps
/// can share references freely. After a defragmentation pass, every
/// clone of an allocation automatically reflects the new memory location:
/// the next call to [`memory`](Self::memory) and [`offset`](Self::offset)
/// returns the post-defrag values.
#[derive(Debug, Clone)]
pub struct Allocation {
    pub(crate) inner: Arc<AllocationInner>,
}

#[derive(Debug)]
pub(crate) struct AllocationInner {
    /// Mutable: changes if a defragmentation pass moves this allocation.
    pub(crate) location: Mutex<AllocationLocation>,
    pub(crate) size: u64,
    pub(crate) memory_type_index: u32,
    pub(crate) user_data: u64,
    pub(crate) id: u64,
    /// Back-reference to the owning [`AllocatorInner`]. `Weak` (not
    /// `Arc`) so that an `Allocation` outliving its `Allocator` is a
    /// no-op-on-drop instead of a use-after-free — the
    /// `AllocatorInner::Drop` already releases all underlying memory
    /// when the allocator itself goes away.
    pub(crate) allocator: std::sync::Weak<AllocatorInner>,
}

impl Drop for AllocationInner {
    fn drop(&mut self) {
        // If the allocator is still alive, return our slot to the pool.
        // If the allocator was dropped first, its own Drop has already
        // freed the underlying VkDeviceMemory blocks — nothing to do.
        let Some(allocator) = self.allocator.upgrade() else {
            return;
        };
        let location = self.location.lock().unwrap().clone();
        allocator.free_allocation_internal(&location, self.size);
        // The pool lock is released inside free_allocation_internal;
        // safe to re-enter the allocator now to evaluate callbacks.
        allocator.fire_pressure_for_memory_type(self.memory_type_index, None);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AllocationLocation {
    pub(crate) memory: VkDeviceMemory,
    pub(crate) offset: u64,
    /// Persistent mapped pointer for the *containing block*, if any.
    pub(crate) mapped_ptr: *mut std::ffi::c_void,
    /// How to free this allocation: default pool, custom pool, or
    /// dedicated.
    pub(crate) kind: AllocationKind,
}

// Safety: the raw pointer inside AllocationLocation is only safe to deref
// while the parent Allocator is alive. The Mutex makes any defrag update
// happen-before subsequent reads.
unsafe impl Send for AllocationInner {}
unsafe impl Sync for AllocationInner {}

#[derive(Debug, Clone)]
pub(crate) enum AllocationKind {
    /// Sub-allocation from the default per-memory-type pool.
    DefaultPool {
        memory_type_index: u32,
        block_index: u32,
        /// TLSF side-table id used to free the sub-allocation.
        tlsf_block_id: u32,
    },
    /// Sub-allocation from a custom user pool.
    CustomPool {
        pool_id: u64,
        block_index: u32,
        /// TLSF side-table id (only meaningful for FreeList strategy
        /// pools; ignored for Linear pools whose blocks free in bulk
        /// via reset_pool).
        tlsf_block_id: u32,
    },
    /// Standalone `vkAllocateMemory` allocation.
    Dedicated { id: u64 },
}

impl Allocation {
    /// Returns the raw `VkDeviceMemory` handle this allocation lives in.
    /// May change between calls if a defragmentation pass has moved the
    /// allocation; the safe wrapper synchronizes the read internally.
    pub fn memory(&self) -> VkDeviceMemory {
        self.inner.location.lock().unwrap().memory
    }

    /// Byte offset of this allocation within its current `VkDeviceMemory`.
    pub fn offset(&self) -> u64 {
        self.inner.location.lock().unwrap().offset
    }

    /// Size of this allocation in bytes. Stable for the lifetime of the
    /// allocation — defragmentation never changes the size.
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    /// Memory type index this allocation lives in. Stable for the
    /// lifetime of the allocation.
    pub fn memory_type_index(&self) -> u32 {
        self.inner.memory_type_index
    }

    /// Opaque user data the caller attached to this allocation via
    /// [`AllocationCreateInfo::user_data`].
    pub fn user_data(&self) -> u64 {
        self.inner.user_data
    }

    /// Allocator-internal stable identifier. Survives defragmentation
    /// and is unique within an [`Allocator`].
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    /// Returns the host-visible mapped pointer for this allocation, if
    /// the allocation was created `mapped: true` and the underlying
    /// memory type is host-visible. The pointer is into the parent
    /// `VkDeviceMemory` at this allocation's current `offset`.
    pub fn mapped_ptr(&self) -> Option<*mut std::ffi::c_void> {
        let loc = self.inner.location.lock().unwrap();
        if loc.mapped_ptr.is_null() {
            None
        } else {
            // Safety: the parent block stays mapped for as long as this
            // allocation lives.
            unsafe { Some(loc.mapped_ptr.add(loc.offset as usize)) }
        }
    }
}

impl Allocator {
    /// Create a new allocator for the given device with default options.
    ///
    /// Equivalent to [`new_with_options`](Self::new_with_options) with
    /// [`AllocatorOptions::default()`]. To allocate buffers whose
    /// [`device_address`](super::Buffer::device_address) you intend to read
    /// (e.g. for `buffer_reference` in a shader), use
    /// [`new_with_options`](Self::new_with_options) with
    /// `buffer_device_address: true` instead.
    pub fn new(device: &Device, physical: &PhysicalDevice) -> Result<Self> {
        Self::new_with_options(device, physical, AllocatorOptions::default())
    }

    /// Create a new allocator with explicit [`AllocatorOptions`].
    ///
    /// The main reason to use this over [`new`](Self::new) is to set
    /// [`buffer_device_address`](AllocatorOptions::buffer_device_address),
    /// which makes every block device-address-capable so sub-allocated
    /// buffers can return a valid [`Buffer::device_address`](super::Buffer::device_address).
    pub fn new_with_options(
        device: &Device,
        physical: &PhysicalDevice,
        options: AllocatorOptions,
    ) -> Result<Self> {
        let get = physical
            .instance()
            .vkGetPhysicalDeviceMemoryProperties
            .ok_or(Error::MissingFunction(
                "vkGetPhysicalDeviceMemoryProperties",
            ))?;

        // Safety: handle is valid; output is fully overwritten.
        let mut props: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
        unsafe { get(physical.raw(), &mut props) };

        let pools: Vec<Option<Pool>> = (0..props.memoryTypeCount).map(|_| None).collect();

        Ok(Self {
            inner: Arc::new(AllocatorInner {
                device: Arc::clone(&device.inner),
                physical: physical.clone(),
                memory_properties: props,
                pools: Mutex::new(PoolState {
                    pools,
                    custom_pools: HashMap::new(),
                    next_pool_id: 1,
                    next_alloc_id: 1,
                    statistics: AllocationStatistics::default(),
                    dedicated_blocks: Vec::new(),
                }),
                pressure: Mutex::new(PressureState {
                    next_id: 1,
                    callbacks: Vec::new(),
                }),
                buffer_device_address: options.buffer_device_address,
            }),
        })
    }

    /// Cheap clone — the underlying state is shared via `Arc`.
    pub fn clone_inner(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Free a previously returned allocation.
    ///
    /// Equivalent to dropping `allocation` (every `Allocation` now
    /// `Drop`s back into the allocator automatically). Kept as an
    /// explicit method for callers who prefer the imperative style or
    /// want a clear "free here" line in their code.
    pub fn free(&self, allocation: Allocation) {
        // The Drop impl on `AllocationInner` does the actual work; we
        // just need to release our handle to it. If this was the last
        // outstanding clone the slot is reclaimed immediately.
        drop(allocation);
    }

    /// Allocate `size` bytes meeting `requirements` and the `info` hints,
    /// returning an [`Allocation`]. The caller is responsible for binding
    /// the result to a resource via `vkBindBufferMemory` or
    /// `vkBindImageMemory` (use [`create_buffer`](Self::create_buffer) /
    /// [`create_image_2d`](Self::create_image_2d) for the bound case).
    ///
    /// When `info.pool` is `Some`, the allocation comes from the
    /// designated custom pool and `info.usage` is ignored (the pool
    /// already determined the memory type at create time). Otherwise the
    /// per-memory-type default pool is used.
    pub fn allocate(
        &self,
        requirements: MemoryRequirements,
        info: AllocationCreateInfo,
    ) -> Result<Allocation> {
        let result = self.allocate_inner(requirements, info);
        if let Ok(ref alloc) = result {
            // `allocate_inner` already dropped the pool lock before
            // returning the Allocation; safe to fire pressure here.
            self.inner
                .fire_pressure_for_memory_type(alloc.memory_type_index(), None);
        }
        result
    }

    fn allocate_inner(
        &self,
        requirements: MemoryRequirements,
        info: AllocationCreateInfo,
    ) -> Result<Allocation> {
        if let Some(pool_handle) = info.pool {
            // Per-allocation device_mask is incompatible with custom pools:
            // a custom pool's blocks are allocated up front with one mask,
            // and we'd silently sub-allocate from a block whose mask
            // doesn't match the user's request. Reject explicitly.
            if info.device_mask.is_some() {
                return Err(Error::InvalidArgument(
                    "AllocationCreateInfo.device_mask cannot be combined with a custom pool; \
                     omit `pool` to get a dedicated allocation with the requested device mask",
                ));
            }
            return self.allocate_in_custom_pool(pool_handle, &requirements, info);
        }

        let memory_type_index = pick_memory_type_in(
            &self.inner.memory_properties,
            requirements.memory_type_bits,
            info.usage,
        )
        .ok_or(Error::Vk(VkResult::ERROR_FEATURE_NOT_PRESENT))?;

        let mut state = self.inner.pools.lock().unwrap();
        let block_size = self.heap_block_size_for_type(memory_type_index);

        // Force dedicated for very large allocations, when explicitly asked,
        // or when the user supplied a non-default device_mask (each
        // VkDeviceMemory block is allocated with one mask, so per-allocation
        // mask control implies one allocation per block).
        let force_dedicated =
            info.dedicated || info.device_mask.is_some() || requirements.size > block_size / 2;

        if force_dedicated {
            return self.allocate_dedicated(&mut state, memory_type_index, &requirements, info);
        }

        // Lazily create the pool.
        if state.pools[memory_type_index as usize].is_none() {
            state.pools[memory_type_index as usize] =
                Some(Pool::new_default(memory_type_index, block_size));
        }

        // Try existing blocks.
        let pool = state.pools[memory_type_index as usize].as_mut().unwrap();
        for (block_index, block) in pool.blocks.iter_mut().enumerate() {
            let BlockStrategy::Tlsf(ref mut t) = block.strategy else {
                continue;
            };
            if let Some(ta) = t.allocate(requirements.size, requirements.alignment) {
                let memory = block.memory;
                let mapped_ptr = block.mapped_ptr;
                let kind = AllocationKind::DefaultPool {
                    memory_type_index,
                    block_index: block_index as u32,
                    tlsf_block_id: ta.block_id,
                };
                state.statistics.allocation_count += 1;
                state.statistics.allocation_bytes += ta.size;
                if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
                    state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
                }
                self.refresh_free_region_count(&mut state);
                let alloc = make_allocation(
                    &mut state,
                    memory,
                    ta.offset,
                    ta.size,
                    memory_type_index,
                    mapped_ptr,
                    kind,
                    info.user_data,
                    Arc::downgrade(&self.inner),
                );
                return Ok(alloc);
            }
        }
        // Need to grow: add a new block.
        let new_block_size = block_size.max(
            requirements
                .size
                .next_power_of_two()
                .max(SMALL_HEAP_BLOCK_SIZE),
        );
        let memory = self.raw_allocate(new_block_size, memory_type_index)?;
        // From here to `pool.blocks.push(block)` below, `memory` is owned by
        // nothing — every early return must free it explicitly.
        let mapped_ptr = if info.mapped && self.is_host_visible(memory_type_index) {
            match self.raw_map_persistent(memory) {
                Ok(p) => p,
                Err(e) => {
                    self.inner.free_raw_memory(memory, std::ptr::null_mut());
                    return Err(e);
                }
            }
        } else {
            std::ptr::null_mut()
        };
        let mut tlsf = Tlsf::new(new_block_size);
        let Some(ta) = tlsf.allocate(requirements.size, requirements.alignment) else {
            self.inner.free_raw_memory(memory, mapped_ptr);
            return Err(Error::Vk(VkResult::ERROR_OUT_OF_DEVICE_MEMORY));
        };
        let block = Block {
            memory,
            capacity: new_block_size,
            memory_type_index,
            strategy: BlockStrategy::Tlsf(tlsf),
            mapped_ptr,
        };
        let pool = state.pools[memory_type_index as usize].as_mut().unwrap();
        pool.blocks.push(block);
        let block_index = pool.blocks.len() as u32 - 1;

        state.statistics.block_bytes += new_block_size;
        state.statistics.block_count += 1;
        state.statistics.allocation_count += 1;
        state.statistics.allocation_bytes += ta.size;
        if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
            state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
        }
        self.refresh_free_region_count(&mut state);

        Ok(make_allocation(
            &mut state,
            memory,
            ta.offset,
            ta.size,
            memory_type_index,
            mapped_ptr,
            AllocationKind::DefaultPool {
                memory_type_index,
                block_index,
                tlsf_block_id: ta.block_id,
            },
            info.user_data,
            Arc::downgrade(&self.inner),
        ))
    }

    /// Allocate from a specific custom pool.
    fn allocate_in_custom_pool(
        &self,
        handle: PoolHandle,
        requirements: &MemoryRequirements,
        info: AllocationCreateInfo,
    ) -> Result<Allocation> {
        let mut state = self.inner.pools.lock().unwrap();
        // Snapshot what we need from the pool before we mutate it (to
        // avoid double borrows).
        let (memory_type_index, strategy, block_size, max_blocks) = {
            let pool = state
                .custom_pools
                .get(&handle.0)
                .ok_or(Error::Vk(VkResult::ERROR_OUT_OF_POOL_MEMORY))?;
            (
                pool.memory_type_index,
                pool.strategy,
                pool.block_size,
                pool.max_block_count,
            )
        };

        // First try existing blocks. We use indexed iteration so we can
        // re-borrow `state` immutably between mutating the pool and
        // calling make_allocation.
        let block_count = state
            .custom_pools
            .get(&handle.0)
            .map(|p| p.blocks.len())
            .unwrap_or(0);
        for block_index in 0..block_count {
            // Borrow the block briefly, attempt the allocation, capture
            // the result, and drop the borrow before touching state.
            let attempt: Option<(VkDeviceMemory, *mut std::ffi::c_void, u64, u64, u32)> = {
                let pool = state.custom_pools.get_mut(&handle.0).unwrap();
                let block = &mut pool.blocks[block_index];
                match &mut block.strategy {
                    BlockStrategy::Tlsf(t) => {
                        if let Some(ta) = t.allocate(requirements.size, requirements.alignment) {
                            Some((
                                block.memory,
                                block.mapped_ptr,
                                ta.offset,
                                ta.size,
                                ta.block_id,
                            ))
                        } else {
                            None
                        }
                    }
                    BlockStrategy::Linear(l) => {
                        if let Some(la) = l.allocate(requirements.size, requirements.alignment) {
                            Some((block.memory, block.mapped_ptr, la.offset, la.size, 0))
                        } else {
                            None
                        }
                    }
                }
            };

            if let Some((memory, mapped_ptr, off, sz, tlsf_id)) = attempt {
                state.statistics.allocation_count += 1;
                state.statistics.allocation_bytes += sz;
                if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
                    state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
                }
                self.refresh_free_region_count(&mut state);
                return Ok(make_allocation(
                    &mut state,
                    memory,
                    off,
                    sz,
                    memory_type_index,
                    mapped_ptr,
                    AllocationKind::CustomPool {
                        pool_id: handle.0,
                        block_index: block_index as u32,
                        tlsf_block_id: tlsf_id,
                    },
                    info.user_data,
                    Arc::downgrade(&self.inner),
                ));
            }
        }

        // Out of room — allocate a new block if we're allowed.
        let pool = state.custom_pools.get_mut(&handle.0).unwrap();
        if max_blocks > 0 && pool.blocks.len() as u32 >= max_blocks {
            return Err(Error::Vk(VkResult::ERROR_OUT_OF_DEVICE_MEMORY));
        }
        let new_block_size = block_size.max(requirements.size.next_power_of_two().max(64));
        let memory = self.raw_allocate(new_block_size, memory_type_index)?;
        // From here to `pool.blocks.push(block)` below, `memory` is owned by
        // nothing — every early return must free it explicitly.
        let mapped_ptr = if info.mapped && self.is_host_visible(memory_type_index) {
            match self.raw_map_persistent(memory) {
                Ok(p) => p,
                Err(e) => {
                    self.inner.free_raw_memory(memory, std::ptr::null_mut());
                    return Err(e);
                }
            }
        } else {
            std::ptr::null_mut()
        };

        // Build the strategy + first allocation in one shot.
        let carved = match strategy {
            AllocationStrategy::FreeList => {
                let mut t = Tlsf::new(new_block_size);
                t.allocate(requirements.size, requirements.alignment)
                    .map(|ta| (BlockStrategy::Tlsf(t), ta.offset, ta.size, ta.block_id))
            }
            AllocationStrategy::Linear => {
                let mut l = Linear::new(new_block_size);
                l.allocate(requirements.size, requirements.alignment)
                    .map(|la| (BlockStrategy::Linear(l), la.offset, la.size, 0))
            }
        };
        let Some((block_strategy, alloc_offset, alloc_size, tlsf_block_id)) = carved else {
            self.inner.free_raw_memory(memory, mapped_ptr);
            return Err(Error::Vk(VkResult::ERROR_OUT_OF_DEVICE_MEMORY));
        };

        let block = Block {
            memory,
            capacity: new_block_size,
            memory_type_index,
            strategy: block_strategy,
            mapped_ptr,
        };
        let pool = state.custom_pools.get_mut(&handle.0).unwrap();
        pool.blocks.push(block);
        let block_index = pool.blocks.len() as u32 - 1;

        state.statistics.block_bytes += new_block_size;
        state.statistics.block_count += 1;
        state.statistics.allocation_count += 1;
        state.statistics.allocation_bytes += alloc_size;
        if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
            state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
        }
        self.refresh_free_region_count(&mut state);

        Ok(make_allocation(
            &mut state,
            memory,
            alloc_offset,
            alloc_size,
            memory_type_index,
            mapped_ptr,
            AllocationKind::CustomPool {
                pool_id: handle.0,
                block_index,
                tlsf_block_id,
            },
            info.user_data,
            Arc::downgrade(&self.inner),
        ))
    }

    /// Convenience: allocate memory and bind it to a freshly-created
    /// buffer in one call.
    pub fn create_buffer(
        &self,
        info: BufferCreateInfo,
        alloc_info: AllocationCreateInfo,
    ) -> Result<(Buffer, Allocation)> {
        // Build a Device handle from our cached Arc.
        let device_for_buffer = Device {
            inner: Arc::clone(&self.inner.device),
        };
        let buffer = Buffer::new(&device_for_buffer, info)?;
        let req = buffer.memory_requirements();
        let allocation = self.allocate(req, alloc_info)?;
        // Bind via the buffer's bind_memory using a transient DeviceMemory
        // wrapper. We don't actually own the memory through DeviceMemory
        // here — the Allocator owns the lifetime — so we use the raw
        // vkBindBufferMemory directly.
        let bind = self
            .inner
            .device
            .dispatch
            .vkBindBufferMemory
            .ok_or(Error::MissingFunction("vkBindBufferMemory"))?;
        // Safety: device, buffer, and memory handles are all valid; we
        // computed offset to satisfy the buffer's alignment.
        check(unsafe {
            bind(
                self.inner.device.handle,
                buffer.raw(),
                allocation.memory(),
                allocation.offset(),
            )
        })?;
        Ok((buffer, allocation))
    }

    /// Convenience: allocate memory and bind it to a freshly-created 2D
    /// image in one call.
    pub fn create_image_2d(
        &self,
        info: Image2dCreateInfo,
        alloc_info: AllocationCreateInfo,
    ) -> Result<(Image, Allocation)> {
        let device_for_image = Device {
            inner: Arc::clone(&self.inner.device),
        };
        let image = Image::new_2d(&device_for_image, info)?;
        let req = image.memory_requirements();
        let allocation = self.allocate(req, alloc_info)?;
        let bind = self
            .inner
            .device
            .dispatch
            .vkBindImageMemory
            .ok_or(Error::MissingFunction("vkBindImageMemory"))?;
        // Safety: device, image, and memory handles are all valid.
        check(unsafe {
            bind(
                self.inner.device.handle,
                image.raw(),
                allocation.memory(),
                allocation.offset(),
            )
        })?;
        Ok((image, allocation))
    }
}

impl Allocator {
    fn is_host_visible(&self, memory_type_index: u32) -> bool {
        let mp = &self.inner.memory_properties;
        let flags = mp.memoryTypes[memory_type_index as usize].propertyFlags;
        (flags & 0x0002) != 0 // HOST_VISIBLE
    }

    fn heap_block_size_for_type(&self, memory_type_index: u32) -> u64 {
        let mp = &self.inner.memory_properties;
        let heap_index = mp.memoryTypes[memory_type_index as usize].heapIndex;
        let heap_size = mp.memoryHeaps[heap_index as usize].size;
        if heap_size >= LARGE_HEAP_THRESHOLD {
            LARGE_HEAP_BLOCK_SIZE
        } else {
            SMALL_HEAP_BLOCK_SIZE
        }
    }

    /// Forwarder — these helpers live on [`AllocatorInner`] now so the
    /// `AllocationInner::Drop` impl can call them. Kept here as
    /// `pub(crate)` shim so the rest of the file's call sites compile
    /// unchanged.
    fn refresh_free_region_count(&self, state: &mut PoolState) {
        self.inner.refresh_free_region_count(state);
    }

    fn free_block_memory(&self, block: &Block) {
        self.inner.free_block_memory(block);
    }
}

impl Drop for AllocatorInner {
    fn drop(&mut self) {
        // Free all per-memory-type pool blocks, custom pool blocks, and
        // dedicated allocations. Inlined the unmap+free logic since the
        // helper method on Allocator wants &Allocator, not
        // &AllocatorInner.
        let mut state = self.pools.lock().unwrap();
        let unmap = self.device.dispatch.vkUnmapMemory;
        let free = self.device.dispatch.vkFreeMemory;
        let dh = self.device.handle;

        let free_block = |block: &Block| {
            if !block.mapped_ptr.is_null()
                && let Some(u) = unmap
            {
                // Safety: handle is valid; we are about to free it.
                unsafe { u(dh, block.memory) };
            }
            if let Some(f) = free {
                // Safety: handle is valid; we are the sole owner.
                unsafe { f(dh, block.memory, std::ptr::null()) };
            }
        };

        for pool in state.pools.iter_mut().flatten() {
            for block in pool.blocks.drain(..) {
                free_block(&block);
            }
        }
        let custom_keys: Vec<u64> = state.custom_pools.keys().copied().collect();
        for k in custom_keys {
            if let Some(mut pool) = state.custom_pools.remove(&k) {
                for block in pool.blocks.drain(..) {
                    free_block(&block);
                }
            }
        }
        for dedicated in state.dedicated_blocks.drain(..) {
            if !dedicated.mapped_ptr.is_null()
                && let Some(u) = unmap
            {
                unsafe { u(dh, dedicated.memory) };
            }
            if let Some(f) = free {
                unsafe { f(dh, dedicated.memory, std::ptr::null()) };
            }
        }
    }
}
