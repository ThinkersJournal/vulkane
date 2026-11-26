// SPDX-License-Identifier: MIT OR Apache-2.0
//! Raw `vkAllocateMemory` / `vkMapMemory` calls behind [`Allocator`].

use super::*;

impl Allocator {
    // ----- internals -----

    pub(super) fn allocate_dedicated(
        &self,
        state: &mut PoolState,
        memory_type_index: u32,
        requirements: &MemoryRequirements,
        info: AllocationCreateInfo,
    ) -> Result<Allocation> {
        let memory =
            self.raw_allocate_with_mask(requirements.size, memory_type_index, info.device_mask)?;
        // From here to `state.dedicated_blocks.push(..)` below, `memory` is
        // owned by nothing — an early return must free it explicitly.
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
        let id = state
            .dedicated_blocks
            .iter()
            .map(|d| d.id)
            .max()
            .map_or(0, |m| m + 1);
        state.dedicated_blocks.push(DedicatedBlock {
            memory,
            size: requirements.size,
            memory_type_index,
            mapped_ptr,
            id,
        });
        state.statistics.block_bytes += requirements.size;
        state.statistics.block_count += 1;
        state.statistics.dedicated_allocation_count += 1;
        state.statistics.allocation_count += 1;
        state.statistics.allocation_bytes += requirements.size;
        if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
            state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
        }

        Ok(make_allocation(
            state,
            memory,
            0,
            requirements.size,
            memory_type_index,
            mapped_ptr,
            AllocationKind::Dedicated { id },
            info.user_data,
            Arc::downgrade(&self.inner),
        ))
    }

    pub(super) fn raw_allocate(&self, size: u64, memory_type_index: u32) -> Result<VkDeviceMemory> {
        self.raw_allocate_with_mask(size, memory_type_index, None)
    }

    pub(super) fn raw_allocate_with_mask(
        &self,
        size: u64,
        memory_type_index: u32,
        device_mask: Option<u32>,
    ) -> Result<VkDeviceMemory> {
        let allocate = self
            .inner
            .device
            .dispatch
            .vkAllocateMemory
            .ok_or(Error::MissingFunction("vkAllocateMemory"))?;

        // Chain a VkMemoryAllocateFlagsInfo when either:
        //   * the allocator was created with `buffer_device_address`, so the
        //     block must carry DEVICE_ADDRESS_BIT for `device_address()` to
        //     be valid on buffers sub-allocated from it, and/or
        //   * the user supplied a device mask, pinning the allocation to a
        //     subset of physical devices in the group.
        // Both bits can be set on the same allocation.
        let mut chain = crate::safe::PNextChain::new();
        let mut flags: VkMemoryAllocateFlags = 0;
        if self.inner.buffer_device_address {
            flags |= MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        }
        if device_mask.is_some() {
            flags |= MEMORY_ALLOCATE_DEVICE_MASK_BIT;
        }
        if flags != 0 {
            chain.push(VkMemoryAllocateFlagsInfo {
                sType: VkStructureType::STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                pNext: std::ptr::null_mut(),
                flags,
                // deviceMask is ignored by the driver unless DEVICE_MASK_BIT
                // is set, so 0 is correct for the address-only case.
                deviceMask: device_mask.unwrap_or(0),
            });
        }

        let info = VkMemoryAllocateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: chain.head(),
            allocationSize: size,
            memoryTypeIndex: memory_type_index,
        };

        let mut handle: VkDeviceMemory = 0;
        // Safety: info is valid for the duration of the call; the chain
        // lives until dropped below.
        let r = check(unsafe {
            allocate(
                self.inner.device.handle,
                &info,
                std::ptr::null(),
                &mut handle,
            )
        });
        drop(chain);
        r?;
        Ok(handle)
    }

    pub(super) fn raw_map_persistent(
        &self,
        memory: VkDeviceMemory,
    ) -> Result<*mut std::ffi::c_void> {
        let map = self
            .inner
            .device
            .dispatch
            .vkMapMemory
            .ok_or(Error::MissingFunction("vkMapMemory"))?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // Safety: handle is valid; we map the entire range.
        check(unsafe {
            map(
                self.inner.device.handle,
                memory,
                0,
                u64::MAX, // VK_WHOLE_SIZE
                0,
                &mut ptr,
            )
        })?;
        Ok(ptr)
    }
}
