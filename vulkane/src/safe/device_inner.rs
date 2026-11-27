// SPDX-License-Identifier: MIT OR Apache-2.0
//! The state shared between a device and its child handles (leaf module: depends on nothing in `safe`).

use crate::raw::bindings::*;
use std::sync::Arc;

/// Internal state shared between `Device` and its child handles.
pub(crate) struct DeviceInner {
    pub(crate) handle: VkDevice,
    pub(crate) dispatch: VkDeviceDispatchTable,
    /// Keep the parent instance alive as long as any device child is alive.
    /// Field is unread but its `Drop` semantics are essential.
    #[allow(dead_code)]
    pub(crate) instance: Arc<super::instance::InstanceInner>,
    /// All physical devices in the device group this `Device` was
    /// created from. Always at least one element. Length 1 for the
    /// overwhelmingly common case of `physical.create_device(...)`.
    pub(crate) physical_devices: Vec<VkPhysicalDevice>,
    /// Names of device-level extensions that were passed to
    /// `vkCreateDevice`. Includes both extensions the caller explicitly
    /// requested via `DeviceCreateInfo::enabled_extensions` and any
    /// the safe wrapper auto-enabled when supported (currently only
    /// `VK_EXT_memory_budget`).
    pub(crate) enabled_extensions: Vec<String>,
}

// Safety: VkDevice is documented by the Vulkan spec as safe to share between
// threads. Individual function calls have their own external synchronization
// requirements (queue submission, command buffer recording, etc. — these are
// the caller's responsibility), but the handle itself is thread-safe.
unsafe impl Send for DeviceInner {}
unsafe impl Sync for DeviceInner {}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        if let Some(destroy) = self.dispatch.vkDestroyDevice {
            // Safety: handle is valid (constructed by Device::new), and
            // by the Arc invariant we are the last owner.
            unsafe { destroy(self.handle, std::ptr::null()) };
        }
    }
}
