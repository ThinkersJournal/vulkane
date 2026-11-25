// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F16` as a compute dtype on a live device (`tests/support/half_kernel.rs`): half arithmetic rounds
//! to half after each operation (a multiply feeding an add must not fuse: `NoContraction`),
//! subnormals survive (`DenormPreserve` at 16 bits), and NaN, infinities and signed zeros come out
//! as IEEE gives them. Each case proves its mutant (fused, or flushed) differs from the expected
//! output, so a wrong lowering could not pass. Without `shaderFloat16` + `storageBuffer16BitAccess`
//! or the 16-bit float controls a `SKIP (unsupported)` line names the capability.

use vulkane_test_support::common;
use vulkane_test_support::float_kernel;
use vulkane_test_support::gpu;
use vulkane_test_support::half_kernel;

use float_kernel::{Case, float_kernel};
use half_kernel::{data, same};
use unpopped::backend::Backend;
use unpopped_vocab::ElementKind::F16;
use vulkane::safe::DeviceFeatures;
use vulkane_spirv_emit::{SpirV, required_features};

fn run(tag: &str, case: Case) -> Option<(Vec<u32>, Vec<u32>)> {
    common::require_serialization_lock();
    let features = DeviceFeatures::new()
        .with_shader_float16()
        .with_storage_buffer16_bit_access();
    let gpu = gpu::Gpu::open_with(
        tag,
        Some((&features, "shaderFloat16 + storageBuffer16BitAccess")),
    )?;
    let g = SpirV
        .lower_kernel(&float_kernel(F16, case))
        .expect("the SPIR-V backend lowers the F16 kernel");
    let words = g.words().expect("words").to_vec();
    let need = required_features(&words).expect("a module");
    let fc = gpu.float_controls();
    eprintln!("F16 kernel needs {need:?}; device float controls: {fc:?}");
    if !fc.is_some_and(|c| c.denorm_preserve_f16 && c.signed_zero_inf_nan_preserve_f16) {
        common::skip(common::Missing::capability(
            "DenormPreserve + SignedZeroInfNanPreserve at 16 bits (VkPhysicalDeviceFloatControlsProperties)",
        ));
        return None;
    }
    let (a, b, want, _) = data(case);
    Some((gpu.run(&g.name, &words, &[&a, &b]), want))
}

fn check(tag: &str, case: Case) {
    let (_, _, want, mutant) = data(case);
    assert!(
        same(&mutant, &want).is_err(),
        "{case:?}: the mutant is indistinguishable, so the case proves nothing"
    );
    let Some((got, want)) = run(tag, case) else {
        return; // skip already declared
    };
    same(&got, &want).unwrap_or_else(|e| panic!("F16 {case:?}: {e}"));
}

#[test]
fn f16_mul_add_rounds_each_step_and_is_unfused_on_a_device() {
    check("spirv_f16_muladd", Case::MulAdd);
}

#[test]
fn f16_max_min_floor_match_ieee_on_a_device() {
    check("spirv_f16_max", Case::Max);
    check("spirv_f16_min", Case::Min);
    check("spirv_f16_floor", Case::Floor);
}

/// Control for the comparison: it rejects a wrong half and a flipped zero, accepts two NaNs.
#[test]
fn the_half_comparison_rejects_wrong_values_and_accepts_any_nan() {
    assert!(same(&[0x3c00_3c00], &[0x3c00_4000]).is_err());
    assert!(same(&[0x8000], &[0x0000]).is_err());
    assert!(same(&[0x7e00], &[0x7e01]).is_ok());
    assert!(same(&[0x3c00], &[0x3c00]).is_ok());
}

/// The `MulAdd` oracle holds a subnormal result (`2^-24`), so a device that flushed denormals
/// (`DenormPreserve` not honoured) would fail the check above: flushing the expected output differs.
#[test]
fn the_expected_output_holds_a_subnormal_a_flushing_device_would_lose() {
    let (_, _, want, _) = data(Case::MulAdd);
    let flushed: Vec<u32> = want
        .iter()
        .map(|w| {
            let f = |h: u32| if h & 0x7c00 == 0 { h & 0x8000 } else { h };
            f(w & 0xffff) | (f(w >> 16) << 16)
        })
        .collect();
    assert!(same(&flushed, &want).is_err());
}
