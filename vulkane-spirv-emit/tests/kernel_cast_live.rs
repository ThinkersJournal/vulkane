// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Cast` on a live device (`tests/support/cast_kernel.rs`): int to float and `f64` to `f32` round to
//! nearest even (ties at `2^24 + 1`), float to int rounds per the mode, maps NaN to 0 and clamps, and
//! `f32` to `f64` is exact, each against native Rust arithmetic, which is `unpopped`'s `cast_ref`
//! contract. Every case also proves its own mutant (truncating, wrapping, through-`f32`) differs from
//! the expected output, so a wrong lowering could not pass. The device is created with
//! `shaderFloat64` where a case uses `f64`; the float controls the module declares (including
//! `RoundingModeRTE`) are checked against the device's and a missing one is a `SKIP (unsupported)`.

#[path = "support/cast_kernel.rs"]
mod cast_kernel;
#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;

use cast_kernel::{CastCase, THREADS, agree, cases, kernel, words};
use unpopped::backend::Backend;
use unpopped_vocab::ElementKind::{F16, F64};
use vulkane::safe::DeviceFeatures;
use vulkane_spirv_emit::{RequiredFeatures, SpirV, required_features};

fn device_ok(need: &RequiredFeatures, fc: Option<vulkane::safe::FloatControlsProperties>) -> bool {
    let Some(c) = fc else {
        return false;
    };
    (!need.denorm_preserve_f32 || c.denorm_preserve_f32)
        && (!need.denorm_preserve_f64 || c.denorm_preserve_f64)
        && (!need.signed_zero_inf_nan_preserve_f32 || c.signed_zero_inf_nan_preserve_f32)
        && (!need.signed_zero_inf_nan_preserve_f64 || c.signed_zero_inf_nan_preserve_f64)
        && (!need.denorm_preserve_f16 || c.denorm_preserve_f16)
        && (!need.signed_zero_inf_nan_preserve_f16 || c.signed_zero_inf_nan_preserve_f16)
        && (!need.rounding_mode_rte_f16 || c.rounding_mode_rte_f16)
        && (!need.rounding_mode_rte_f32 || c.rounding_mode_rte_f32)
        && (!need.rounding_mode_rte_f64 || c.rounding_mode_rte_f64)
}

fn run_case(c: &CastCase) -> Option<(Vec<u32>, Vec<u32>)> {
    common::require_serialization_lock();
    let uses_f64 = [c.src, c.dst, c.compute].contains(&F64);
    let uses_f16 = [c.src, c.dst, c.compute].contains(&F16);
    let mut features = DeviceFeatures::new();
    if uses_f64 {
        features = features.with_shader_float64();
    }
    if uses_f16 {
        features = features
            .with_shader_float16()
            .with_storage_buffer16_bit_access();
    }
    let gpu = gpu::Gpu::open_with(
        c.name,
        Some((
            &features,
            "shaderFloat64 / shaderFloat16 + storageBuffer16BitAccess",
        )),
    )?;
    let g = SpirV.lower_kernel(&kernel(c)).expect("lowers");
    let words_ = g.words().expect("words").to_vec();
    let need = required_features(&words_).expect("a module");
    let fc = gpu.float_controls();
    eprintln!("{}: needs {need:?}; device {fc:?}", c.name);
    if !device_ok(&need, fc) {
        common::skip(common::Missing::capability(
            "the float controls this cast module declares (VkPhysicalDeviceFloatControlsProperties)",
        ));
        return None;
    }
    // `run` returns as many words as one input has: pad the input so every output word is read.
    let in_words = words(c.src, &c.input, 2 * THREADS);
    let got = gpu.run(&g.name, &words_, &[&in_words]);
    Some((got, words(c.dst, &c.want, 0)))
}

#[test]
fn every_cast_matches_the_oracle_on_a_device_and_its_mutant_would_not() {
    for c in cases() {
        // The mutant differs from the expected output somewhere, or this case proves nothing.
        let (w, m) = (words(c.dst, &c.want, 0), words(c.dst, &c.mutant, 0));
        assert!(
            agree(c.dst, &m, &w).is_err(),
            "{}: the mutant is indistinguishable",
            c.name
        );
        let Some((got, want)) = run_case(&c) else {
            continue; // skip already declared
        };
        let n = want.len();
        agree(c.dst, &got[..n], &want).unwrap_or_else(|e| panic!("{}: {e}", c.name));
    }
}

/// Control for the comparison: it rejects a wrong integer and accepts two NaNs.
#[test]
fn the_cast_comparison_rejects_wrong_values_and_accepts_nan_payloads() {
    use unpopped_vocab::ElementKind::{F32, I32};
    assert!(agree(I32, &[1], &[2]).is_err());
    assert!(agree(F32, &[f32::NAN.to_bits()], &[0x7fc0_0001]).is_ok());
    assert!(agree(F32, &[(-0.0f32).to_bits()], &[0.0f32.to_bits()]).is_err());
}
