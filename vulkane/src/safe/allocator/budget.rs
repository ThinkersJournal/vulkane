// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statistics, budget queries and pressure callbacks on [`Allocator`].

use super::*;

impl Allocator {
    /// Snapshot the current statistics.
    pub fn statistics(&self) -> AllocationStatistics {
        self.inner.pools.lock().unwrap().statistics
    }

    /// Query the per-heap memory budget via `VK_EXT_memory_budget`.
    ///
    /// Returns `None` only when `vkGetPhysicalDeviceMemoryProperties2`
    /// is not loaded (Vulkan 1.0 without
    /// `VK_KHR_get_physical_device_properties2`). The budget/usage values
    /// inside the returned [`crate::safe::MemoryBudget`] are only
    /// meaningful when `VK_EXT_memory_budget` was enabled on the device
    /// (which [`Device`] does automatically when the physical device
    /// supports it — see [`Allocator::has_memory_budget_support`]).
    pub fn query_budget(&self) -> Option<crate::safe::MemoryBudget> {
        self.inner.physical.memory_budget()
    }

    /// `true` iff this allocator can return meaningful budget numbers.
    ///
    /// Requires both `vkGetPhysicalDeviceMemoryProperties2` to be loaded
    /// (Vulkan 1.1+ core, or `VK_KHR_get_physical_device_properties2`
    /// as an instance extension) **and** `VK_EXT_memory_budget` to be
    /// enabled on the device. The safe wrapper auto-enables the device
    /// extension when the physical device supports it, so in practice
    /// this returns `true` on any reasonably modern driver.
    ///
    /// When this returns `false`, [`query_budget`](Self::query_budget),
    /// [`vram_budget`](Self::vram_budget), and
    /// [`vram_used`](Self::vram_used) will still return values, but
    /// the budget numbers come from the driver's fallback reporting and
    /// should not be treated as authoritative.
    pub fn has_memory_budget_support(&self) -> bool {
        let get2_loaded = self
            .inner
            .physical
            .instance()
            .vkGetPhysicalDeviceMemoryProperties2
            .is_some();
        let ext_enabled = self
            .inner
            .device_enabled_extensions()
            .iter()
            .any(|n| n == "VK_EXT_memory_budget");
        get2_loaded && ext_enabled
    }

    /// Total soft VRAM budget, in bytes, summed across every
    /// `DEVICE_LOCAL` memory heap.
    ///
    /// This is the number ML schedulers typically mean when they say
    /// "how much VRAM do I have to play with": the driver's suggested
    /// cap across every heap that lives on the GPU. Does not include
    /// system-memory heaps (CPU RAM exposed as host-visible memory on
    /// integrated or UMA systems appears on a separate, non-device-local
    /// heap).
    ///
    /// Returns `0` if the budget query fails for any reason (old driver,
    /// extension not loaded). Check
    /// [`has_memory_budget_support`](Self::has_memory_budget_support)
    /// first if you need to distinguish "no budget" from "no query
    /// support".
    pub fn vram_budget(&self) -> u64 {
        self.sum_device_local_heaps(|budget, heap| budget.budget[heap])
    }

    /// Total VRAM currently in use, in bytes, summed across every
    /// `DEVICE_LOCAL` memory heap. The driver's estimate — includes
    /// allocations from this process, other processes, and driver
    /// internal state.
    ///
    /// Returns `0` if the budget query fails. Pair with
    /// [`vram_budget`](Self::vram_budget) for a ratio:
    ///
    /// ```ignore
    /// let frac = allocator.vram_used() as f64 / allocator.vram_budget() as f64;
    /// if frac > 0.85 {
    ///     scheduler.evict_least_recently_used();
    /// }
    /// ```
    pub fn vram_used(&self) -> u64 {
        self.sum_device_local_heaps(|budget, heap| budget.usage[heap])
    }

    /// Internal helper: sum `f(budget, heap_idx)` over every
    /// `DEVICE_LOCAL` heap. Returns 0 if the budget query returns `None`.
    fn sum_device_local_heaps<F>(&self, f: F) -> u64
    where
        F: Fn(&crate::safe::MemoryBudget, usize) -> u64,
    {
        let Some(budget) = self.query_budget() else {
            return 0;
        };
        let props = self.inner.physical.memory_properties();
        let mut total: u64 = 0;
        for heap_idx in 0..(budget.heap_count as usize) {
            if props
                .memory_heap(heap_idx as u32)
                .flags()
                .contains(crate::safe::MemoryHeapFlags::DEVICE_LOCAL)
            {
                total = total.saturating_add(f(&budget, heap_idx));
            }
        }
        total
    }

    /// Returns the cached `PhysicalDevice` this allocator was created
    /// with. Useful for follow-up queries that the allocator does not
    /// proxy directly.
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.inner.physical
    }

    /// Register a budget-pressure callback that fires when the
    /// fraction `usage / budget` on any memory heap crosses a
    /// threshold.
    ///
    /// # Arguments
    ///
    /// - `threshold` — the fraction (typically between 0.5 and 0.99)
    ///   at which the callback fires. For example, `0.85` fires when a
    ///   heap reaches 85 % of its driver-reported budget.
    /// - `hysteresis` — the relief gap. After firing
    ///   [`PressureKind::Crossed`], the matching
    ///   [`PressureKind::Relieved`] event will not fire until usage
    ///   drops below `threshold - hysteresis`. This prevents rapid
    ///   re-firing as usage oscillates around the threshold. A
    ///   reasonable default is `0.05` (5 % of budget).
    /// - `callback` — the closure to invoke. Receives a
    ///   [`PressureEvent`] describing which heap, the threshold, and
    ///   the kind of transition.
    ///
    /// # When callbacks fire
    ///
    /// - After every successful allocation — reactive, fires
    ///   [`PressureKind::Crossed`] if usage just rose past `threshold`.
    /// - After every free, [`reset_pool`](Self::reset_pool),
    ///   [`destroy_pool`](Self::destroy_pool), and
    ///   [`apply_defragmentation_plan`](Self::apply_defragmentation_plan)
    ///   — fires [`PressureKind::Relieved`] if usage dropped below
    ///   `threshold - hysteresis`.
    /// - From every call to [`would_fit`](Self::would_fit) — proactive,
    ///   fires [`PressureKind::Predictive`] if the projected usage
    ///   (current + requested) would cross `threshold`.
    ///
    /// # Re-entrancy
    ///
    /// Callbacks are invoked **after** every internal allocator lock
    /// is released, so they may call back into this `Allocator` (to
    /// query statistics, free other allocations, or trigger a defrag
    /// cycle) without deadlocking. They must still be `Send + Sync`.
    ///
    /// Callbacks run on whatever thread issued the allocation or free.
    /// If your callback does nontrivial work, hand off to a worker
    /// thread rather than blocking the allocation path.
    ///
    /// # Meaningful budget requires the extension
    ///
    /// The budget numbers are only authoritative when
    /// [`has_memory_budget_support`](Self::has_memory_budget_support)
    /// is `true`. When it is `false`, callbacks still fire based on
    /// whatever numbers the driver returns, but those numbers may not
    /// reflect reality.
    pub fn register_pressure_callback<F>(
        &self,
        threshold: f64,
        hysteresis: f64,
        callback: F,
    ) -> PressureCallbackId
    where
        F: Fn(PressureEvent) + Send + Sync + 'static,
    {
        let mut ps = self.inner.pressure.lock().unwrap();
        let id = ps.next_id;
        ps.next_id += 1;
        ps.callbacks.push(PressureCallbackEntry {
            id,
            threshold,
            hysteresis,
            callback: Arc::new(callback),
            latched: [false; 16],
        });
        PressureCallbackId(id)
    }

    /// Remove a previously registered pressure callback. Returns
    /// `true` if the callback was found and removed, `false` if the
    /// id was already unregistered.
    pub fn unregister_pressure_callback(&self, id: PressureCallbackId) -> bool {
        let mut ps = self.inner.pressure.lock().unwrap();
        let before = ps.callbacks.len();
        ps.callbacks.retain(|c| c.id != id.0);
        ps.callbacks.len() != before
    }

    /// Proactively check whether allocating `size` bytes from
    /// `memory_type_index` would fit under the driver's soft budget
    /// for the backing heap.
    ///
    /// Returns a [`FitStatus`] with the current heap usage, budget,
    /// projected usage, and a boolean `fits`. Fires
    /// [`PressureKind::Predictive`] events for any registered
    /// callback whose threshold would be crossed by the projected
    /// usage — letting a scheduler free resources *before* the
    /// allocation is attempted, rather than reacting to a Crossed
    /// event after the fact.
    ///
    /// Pick `memory_type_index` the same way you would for a real
    /// allocation — e.g. via
    /// [`PhysicalDevice::find_memory_type`](crate::safe::PhysicalDevice::find_memory_type).
    /// When the budget query is unsupported, `fits` is `true` (we
    /// cannot prove failure) and `current_usage`/`budget` are `0`.
    ///
    /// This method never actually allocates and has no effect on the
    /// allocator state beyond firing callbacks.
    pub fn would_fit(&self, size: u64, memory_type_index: u32) -> FitStatus {
        let heap_index = self.inner.heap_index_for_memory_type(memory_type_index);
        let (current_usage, budget) = match self.query_budget() {
            Some(b) if (heap_index as usize) < b.heap_count as usize => {
                let h = heap_index as usize;
                (b.usage[h], b.budget[h])
            }
            _ => (0u64, 0u64),
        };
        let projected_usage = current_usage.saturating_add(size);
        let fits = budget == 0 || projected_usage <= budget;
        let projected_fraction = if budget == 0 {
            0.0
        } else {
            projected_usage as f64 / budget as f64
        };

        // Fire predictive callbacks for thresholds the projection
        // would cross. Only do this when the heap index is a real heap
        // and we have a meaningful budget number to compare against.
        if heap_index != u32::MAX && budget != 0 {
            self.inner.fire_pressure_for_heap(heap_index, Some(size));
        }

        FitStatus {
            heap_index,
            current_usage,
            budget,
            requested: size,
            projected_usage,
            fits,
            projected_fraction,
        }
    }
}
