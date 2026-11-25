// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 8c, `ExecId::{LaneId, SubgroupId, SubgroupSize}` on SPIR-V, on a device.
//!
//! The ids are the `SubgroupLocalInvocationId`, `SubgroupId` and `SubgroupSize` builtins. SPIR-V for
//! Vulkan cannot pin the size: a launch's `LaunchShape::subgroup` range is the contract the host
//! meets (`required_subgroup_size` where the device exposes size control; otherwise the device's
//! one size must lie in the range). This test applies the second rule: a device whose size is outside
//! the range is a decline by name, and a range no device can have is refused by the emitter.

use vulkane_test_support::common;
use vulkane_test_support::gpu;
use vulkane_test_support::subgroup_kernel;

use unpopped::backend::{Backend, LowerError};
use vulkane_spirv_emit::SpirV;

/// The size the launch will run at: `Ok` when the device's one size is in `min..=max`, else a
/// decline naming both (the host cannot pin it on this harness's device).
fn runnable_size(device_size: u32, min: u32, max: u32) -> Result<u32, String> {
    if (min..=max).contains(&device_size) {
        Ok(device_size)
    } else {
        Err(format!(
            "the kernel requires a subgroup size in {min}..={max} and the device runs {device_size}"
        ))
    }
}

#[test]
fn lane_subgroup_and_size_ids_match_the_device_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("spirv_kernel_subgroup") else {
        return;
    };
    let size = gpu.subgroup_size();
    let k = SpirV
        .lower_kernel(&subgroup_kernel::subgroup_kernel(1, 128))
        .expect("lowers");
    let size = runnable_size(size, 1, 128).expect("every device's size is in 1..=128");
    let got = gpu.run_grid(
        &k.name,
        k.words().expect("spirv words"),
        subgroup_kernel::BLOCK as usize,
        [1, 1, 1],
    );
    for (t, &g) in got.iter().enumerate() {
        let want = subgroup_kernel::expected(t as u32, size);
        assert_eq!(g, want, "thread {t} (subgroup size {size})");
    }
    eprintln!(
        "subgroup size {size}: {} subgroups in the block",
        subgroup_kernel::BLOCK / size.min(subgroup_kernel::BLOCK)
    );
}

#[test]
fn a_device_size_outside_the_required_range_is_a_decline_naming_both() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("spirv_kernel_subgroup_decline") else {
        return;
    };
    let size = gpu.subgroup_size();
    // A range that excludes the device's size, and one that includes it (the control).
    let (lo, hi) = (size + 1, size * 2);
    let e = runnable_size(size, lo, hi).expect_err("excluded");
    assert!(
        e.contains(&format!("{lo}..={hi}")) && e.contains(&size.to_string()),
        "{e}"
    );
    assert!(runnable_size(size, size, size).is_ok());
}

#[test]
fn a_range_no_vulkan_device_can_have_is_refused_by_the_emitter() {
    for (min, max) in [(256, 512), (3, 3), (5, 7)] {
        match SpirV.lower_kernel(&subgroup_kernel::subgroup_kernel(min, max)) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(
                    detail.contains("holds no power of two up to 128"),
                    "{detail}"
                );
            }
            other => panic!("{min}..={max}: wanted a decline, got {other:?}"),
        }
    }
    // `check_kernel` refuses a zero minimum before the emitter's own check is reached.
    match SpirV.lower_kernel(&subgroup_kernel::subgroup_kernel(0, 8)) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("BadLaunch"), "{detail}")
        }
        other => panic!("0..=8: wanted a decline, got {other:?}"),
    }
    // Control: a range that holds a power of two lowers.
    assert!(
        SpirV
            .lower_kernel(&subgroup_kernel::subgroup_kernel(4, 7))
            .is_ok()
    );
}

#[test]
fn the_module_declares_the_capability_and_the_three_builtins() {
    let k = SpirV
        .lower_kernel(&subgroup_kernel::subgroup_kernel(1, 128))
        .expect("lowers");
    for w in [
        "OpCapability GroupNonUniform",
        "BuiltIn SubgroupLocalInvocationId",
        "BuiltIn SubgroupId",
        "BuiltIn SubgroupSize",
    ] {
        assert!(k.source.contains(w), "{}", k.source);
    }
}
