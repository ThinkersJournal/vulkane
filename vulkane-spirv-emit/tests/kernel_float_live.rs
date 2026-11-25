// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F32`/`F64` kernels on a live device (`tests/support/float_kernel.rs`): NaN, infinities, signed
//! zeros and a denormal result must come out bit-exact as native IEEE arithmetic gives them
//! (`SignedZeroInfNanPreserve` and `DenormPreserve` are declared for the kernel), a multiply feeding
//! an add must not fuse (`NoContraction`), and values only `f64` holds (`2^24 + 1`, `3.4e38 * 10`)
//! must survive an `F64` kernel. The device is created with `shaderFloat64` for `F64`; the float
//! controls the module declares are read from the device and printed; without a feature a
//! `SKIP (unsupported)` line names it.

use vulkane_test_support::common;
use vulkane_test_support::float_kernel;
use vulkane_test_support::gpu;

use float_kernel::{Case, THREADS, data, float_kernel, same};
use unpopped::backend::Backend;
use unpopped_vocab::ElementKind::{self, F32, F64};
use vulkane::safe::DeviceFeatures;
use vulkane_spirv_emit::{SpirV, required_features};

fn run(tag: &str, ty: ElementKind, case: Case) -> Option<(Vec<u32>, Vec<u32>)> {
    common::require_serialization_lock();
    let f64_kernel = ty == F64;
    let features = if f64_kernel {
        DeviceFeatures::new().with_shader_float64()
    } else {
        DeviceFeatures::new()
    };
    let gpu = gpu::Gpu::open_with(tag, Some((&features, "shaderFloat64")))?;
    let g = SpirV
        .lower_kernel(&float_kernel(ty, case))
        .expect("the SPIR-V backend lowers the float kernel");
    let words = g.words().expect("words").to_vec();
    let need = required_features(&words).expect("a module");
    let fc = gpu.float_controls();
    eprintln!("float kernel needs {need:?}; device float controls: {fc:?}");
    let ok = match (f64_kernel, fc) {
        (false, Some(c)) => c.denorm_preserve_f32 && c.signed_zero_inf_nan_preserve_f32,
        (true, Some(c)) => c.denorm_preserve_f64 && c.signed_zero_inf_nan_preserve_f64,
        (_, None) => false,
    };
    if !ok {
        common::skip(common::Missing::capability(
            "DenormPreserve + SignedZeroInfNanPreserve at this width (VkPhysicalDeviceFloatControlsProperties)",
        ));
        return None;
    }
    let (a, b, want) = if f64_kernel {
        data::<f64>(case)
    } else {
        data::<f32>(case)
    };
    // `run` dispatches by input length; the kernel indexes by ThreadIdx, so any extra group repeats.
    let got = gpu.run(&g.name, &words, &[&a, &b]);
    // `run` returns as many words as one input has: that is every output word.
    Some((got, want))
}

fn check(tag: &str, ty: ElementKind, case: Case) {
    let Some((got, want)) = run(tag, ty, case) else {
        return; // skip already declared
    };
    same(ty == F64, &got, &want).unwrap_or_else(|e| panic!("{ty:?} {case:?}: {e}"));
    assert_eq!(got.len(), want.len());
    let _ = THREADS;
}

#[test]
fn f32_mul_add_is_unfused_and_ieee_on_a_device() {
    check("spirv_f32_muladd", F32, Case::MulAdd);
}

#[test]
fn f64_mul_add_is_unfused_and_ieee_on_a_device() {
    check("spirv_f64_muladd", F64, Case::MulAdd);
}

#[test]
fn nan_propagating_max_min_and_floor_match_ieee_on_a_device() {
    for (ty, tag) in [(F32, "spirv_f32_ops"), (F64, "spirv_f64_ops")] {
        for case in [Case::Max, Case::Min, Case::Floor] {
            check(tag, ty, case);
        }
    }
}

/// Mutant: running the `F64` values through `f32` arithmetic is wrong where only `f64` holds the
/// value (`2^24 + 1`, `3.4e38 * 10`, the denormal), so the check above would fail it.
#[test]
fn an_f32_lowering_of_an_f64_kernel_would_fail_the_check() {
    let (_, _, want64) = data::<f64>(Case::MulAdd);
    let (_, _, want32) = data::<f32>(Case::MulAdd);
    // Re-read the f32 oracle as f64 bit patterns: every cell it rounds differently is a cell the
    // mutant gets wrong.
    let widened: Vec<u32> = want32
        .iter()
        .flat_map(|w| (f64::from(f32::from_bits(*w))).to_bits().to_le_bytes())
        .collect::<Vec<u8>>()
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(
        same(true, &widened, &want64).is_err(),
        "f32 and f64 agree everywhere"
    );
    // Control: the oracle agrees with itself, so the failure is the width.
    assert!(same(true, &want64, &want64).is_ok());
}

/// Control for the comparison itself: it must reject a wrong value, a flipped zero sign and a
/// denormal flushed to zero, and accept NaNs of different payload, or every pass above means nothing.
#[test]
fn the_comparison_rejects_wrong_values_and_signed_zero_and_flushed_denormals() {
    let one = 1.0f32.to_bits();
    assert!(same(false, &[one], &[2.0f32.to_bits()]).is_err());
    assert!(same(false, &[(-0.0f32).to_bits()], &[0.0f32.to_bits()]).is_err());
    assert!(same(false, &[0], &[1e-40f32.to_bits()]).is_err());
    assert!(same(false, &[f32::NAN.to_bits()], &[0x7fc0_0001]).is_ok());
    assert!(same(false, &[one], &[one]).is_ok());
}
