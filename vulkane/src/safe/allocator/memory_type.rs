// SPDX-License-Identifier: MIT OR Apache-2.0
//! Memory-type selection: resolve an [`AllocationUsage`] against a device's memory layout.

use crate::raw::bindings::*;

use super::AllocationUsage;

// Memory-property bits, spelled out so the selection rules below read as
// prose. These mirror `VkMemoryPropertyFlagBits`.
const MP_DEVICE_LOCAL: u32 = 0x0001;
const MP_HOST_VISIBLE: u32 = 0x0002;
const MP_HOST_COHERENT: u32 = 0x0004;

/// Resolve an [`AllocationUsage`] against a device's memory layout.
///
/// Split out of [`Allocator`] so it can be exercised against synthetic
/// layouts — the interesting cases are ones no single machine exhibits.
pub(super) fn pick_memory_type_in(
    mp: &VkPhysicalDeviceMemoryProperties,
    type_bits: u32,
    usage: AllocationUsage,
) -> Option<u32> {
    // Required / preferred / avoided property flags per usage. `avoid`
    // applies only to the preference pass — see `find_type_in`.
    let (required_a, preferred_a, avoid_a, fallback_required, fallback_preferred) = match usage {
        AllocationUsage::Auto | AllocationUsage::DeviceLocal => (
            MP_DEVICE_LOCAL,
            MP_DEVICE_LOCAL,
            0,
            // No device-local type in `type_bits` — take anything.
            0,
            0,
        ),
        // Staging memory must avoid a DEVICE_LOCAL | HOST_VISIBLE type
        // whenever a plain host-visible one exists: that type is the
        // PCIe BAR window, and filling it with staging buffers starves
        // the CPU host aperture — which surfaces as a driver-level
        // failure, not a catchable VK_ERROR_OUT_OF_DEVICE_MEMORY.
        //
        // The spec's memory-type ordering (a type whose propertyFlags
        // are a strict subset of another's must come first) already
        // puts the plain type ahead of the BAR type, so on most layouts
        // this pass changes nothing. It matters where the flag sets are
        // *incomparable* — e.g. HOST_VISIBLE | HOST_COHERENT |
        // HOST_CACHED versus DEVICE_LOCAL | HOST_VISIBLE |
        // HOST_COHERENT, neither a subset of the other — because there
        // the spec imposes no order at all and a conforming driver may
        // enumerate the BAR type first. Callers who *want* the BAR ask
        // for it by name with `HostVisibleDeviceLocal`.
        AllocationUsage::HostVisible => (
            MP_HOST_VISIBLE | MP_HOST_COHERENT,
            MP_HOST_VISIBLE | MP_HOST_COHERENT,
            MP_DEVICE_LOCAL, // avoid the BAR
            MP_HOST_VISIBLE,
            MP_HOST_VISIBLE,
        ),
        AllocationUsage::HostVisibleDeviceLocal => (
            MP_DEVICE_LOCAL | MP_HOST_VISIBLE | MP_HOST_COHERENT,
            MP_DEVICE_LOCAL | MP_HOST_VISIBLE | MP_HOST_COHERENT,
            0,
            MP_HOST_VISIBLE | MP_HOST_COHERENT,
            MP_HOST_VISIBLE | MP_HOST_COHERENT,
        ),
    };
    // First pass: required + preferred, honouring `avoid`.
    if let Some(i) = find_type_in(mp, type_bits, required_a, preferred_a, avoid_a) {
        return Some(i);
    }
    // Second pass: fallback.
    find_type_in(mp, type_bits, fallback_required, fallback_preferred, 0)
}

/// Pick a memory type index satisfying `required`, preferring one that
/// also has every `preferred` flag and *none* of the `avoid` flags.
///
/// `avoid` is a preference, not a constraint: if no type satisfies it, the
/// second loop returns any type meeting `required`. That keeps a device
/// whose only host-visible memory is also DEVICE_LOCAL (UMA, or a fully
/// host-visible VRAM layout) working rather than failing to allocate.
fn find_type_in(
    mp: &VkPhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: u32,
    preferred: u32,
    avoid: u32,
) -> Option<u32> {
    // First try types that have all `preferred` and no `avoid` flags...
    for i in 0..mp.memoryTypeCount {
        if (type_bits & (1 << i)) == 0 {
            continue;
        }
        let flags = mp.memoryTypes[i as usize].propertyFlags;
        if (flags & required) == required
            && (flags & preferred) == preferred
            && (flags & avoid) == 0
        {
            return Some(i);
        }
    }
    // ...otherwise any type that meets `required`.
    for i in 0..mp.memoryTypeCount {
        if (type_bits & (1 << i)) == 0 {
            continue;
        }
        let flags = mp.memoryTypes[i as usize].propertyFlags;
        if (flags & required) == required {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const MP_HOST_CACHED: u32 = 0x0008;

    /// Build a `VkPhysicalDeviceMemoryProperties` from a list of property
    /// flag words, in the order a driver would enumerate them.
    fn layout(types: &[u32]) -> VkPhysicalDeviceMemoryProperties {
        let mut mp = VkPhysicalDeviceMemoryProperties {
            memoryTypeCount: types.len() as u32,
            memoryHeapCount: 1,
            ..Default::default()
        };
        for (i, &flags) in types.iter().enumerate() {
            mp.memoryTypes[i].propertyFlags = flags;
            mp.memoryTypes[i].heapIndex = 0;
        }
        mp.memoryHeaps[0].size = 8 * 1024 * 1024 * 1024;
        mp
    }

    fn flags_of(mp: &VkPhysicalDeviceMemoryProperties, index: u32) -> u32 {
        mp.memoryTypes[index as usize].propertyFlags
    }

    const ALL: u32 = u32::MAX;

    /// The case the spec's ordering rules do *not* cover: a HOST_CACHED
    /// host type and a DEVICE_LOCAL host type are incomparable (neither
    /// flag set is a subset of the other), so a conforming driver may
    /// enumerate the BAR first. Staging must still avoid it.
    #[test]
    fn host_visible_avoids_the_bar_when_the_bar_is_enumerated_first() {
        let mp = layout(&[
            MP_DEVICE_LOCAL | MP_HOST_VISIBLE | MP_HOST_COHERENT, // 0: ReBAR
            MP_HOST_VISIBLE | MP_HOST_COHERENT | MP_HOST_CACHED,  // 1: system RAM
        ]);
        let picked = pick_memory_type_in(&mp, ALL, AllocationUsage::HostVisible)
            .expect("a host-visible type exists");
        // Assert the property that matters, not merely that we got an index.
        assert_eq!(
            flags_of(&mp, picked) & MP_DEVICE_LOCAL,
            0,
            "staging resolved to a DEVICE_LOCAL type (index {picked}) — that is the PCIe BAR",
        );
        assert_eq!(picked, 1);
    }

    /// The opposite request must still reach the BAR — the avoidance is
    /// scoped to `HostVisible`, and `HostVisibleDeviceLocal` is how a
    /// caller asks for BAR memory on purpose.
    #[test]
    fn host_visible_device_local_still_selects_the_bar() {
        let mp = layout(&[
            MP_DEVICE_LOCAL | MP_HOST_VISIBLE | MP_HOST_COHERENT,
            MP_HOST_VISIBLE | MP_HOST_COHERENT | MP_HOST_CACHED,
        ]);
        let picked = pick_memory_type_in(&mp, ALL, AllocationUsage::HostVisibleDeviceLocal)
            .expect("a host-visible device-local type exists");
        assert_eq!(flags_of(&mp, picked) & MP_DEVICE_LOCAL, MP_DEVICE_LOCAL);
        assert_eq!(picked, 0);
    }

    /// Avoidance is a preference, not a constraint. On a layout whose only
    /// host-visible type is also DEVICE_LOCAL (UMA, or fully host-visible
    /// VRAM) there is nothing to prefer, so the allocation must still
    /// succeed rather than fail.
    #[test]
    fn host_visible_falls_back_when_every_host_type_is_device_local() {
        let mp = layout(&[
            MP_DEVICE_LOCAL,
            MP_DEVICE_LOCAL | MP_HOST_VISIBLE | MP_HOST_COHERENT,
        ]);
        let picked = pick_memory_type_in(&mp, ALL, AllocationUsage::HostVisible)
            .expect("must fall back rather than fail");
        assert_eq!(picked, 1);
    }

    /// A realistic conformant desktop layout: the subset-ordering rule
    /// already puts plain system RAM ahead of the BAR. Locks in that the
    /// avoidance pass does not perturb the common case.
    #[test]
    fn conformant_desktop_layout_picks_plain_system_ram() {
        let mp = layout(&[
            MP_DEVICE_LOCAL,                                      // 0: VRAM
            MP_HOST_VISIBLE | MP_HOST_COHERENT,                   // 1: system RAM
            MP_HOST_VISIBLE | MP_HOST_COHERENT | MP_HOST_CACHED,  // 2: cached system RAM
            MP_DEVICE_LOCAL | MP_HOST_VISIBLE | MP_HOST_COHERENT, // 3: ReBAR
        ]);
        let picked = pick_memory_type_in(&mp, ALL, AllocationUsage::HostVisible).unwrap();
        assert_eq!(picked, 1);
        assert_eq!(flags_of(&mp, picked) & MP_DEVICE_LOCAL, 0);
        assert_eq!(
            pick_memory_type_in(&mp, ALL, AllocationUsage::DeviceLocal).unwrap(),
            0
        );
    }

    /// `type_bits` still wins: when a resource's requirements exclude every
    /// non-device-local host type, the BAR is the only candidate left.
    #[test]
    fn type_bits_mask_overrides_the_preference() {
        let mp = layout(&[
            MP_HOST_VISIBLE | MP_HOST_COHERENT,
            MP_DEVICE_LOCAL | MP_HOST_VISIBLE | MP_HOST_COHERENT,
        ]);
        let picked = pick_memory_type_in(&mp, 0b10, AllocationUsage::HostVisible)
            .expect("the masked-in type satisfies the requirement");
        assert_eq!(picked, 1);
    }

    /// No host-visible type at all — an honest `None`, not an arbitrary index.
    #[test]
    fn host_visible_reports_none_when_no_host_type_exists() {
        let mp = layout(&[MP_DEVICE_LOCAL]);
        assert_eq!(
            pick_memory_type_in(&mp, ALL, AllocationUsage::HostVisible),
            None
        );
    }
}
