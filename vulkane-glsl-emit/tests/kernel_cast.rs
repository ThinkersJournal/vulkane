// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Cast` in float kernels through GLSL (ROADMAP row 8, the float arm), the cases of
//! `vulkane-spirv-emit`'s `cast_kernel` support that stay within `F32`, `F64`, `I32` and `U32`: int to
//! float and `f64` to `f32` round to nearest even (ties at `2^24 + 1`), float to int rounds per the
//! mode, maps NaN to 0 and clamps, and `f32` to `f64` is exact, each against native Rust arithmetic
//! (`unpopped`'s `cast_ref` contract). The source declares `RoundingModeRTE` where it rounds; the
//! module's requirements are read back from the glslang output and checked against the device's
//! float controls, a missing one is a `SKIP (unsupported)`. Every case proves its mutant (truncating,
//! wrapping, through-`f32`) differs from the expected output.

#[path = "../../vulkane-spirv-emit/tests/support/cast_kernel.rs"]
#[allow(dead_code)]
mod cast_kernel;
#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/glslang.rs"]
mod glslang;
#[path = "../../vulkane-spirv-emit/tests/support/gpu.rs"]
mod gpu;

use cast_kernel::{CastCase, THREADS, agree, cases, kernel, words};
use glslang::{glslang_words, validates};
use unpopped::backend::Backend;
use unpopped_vocab::ElementKind::{F32, F64, I32, U32};
use vulkane::safe::{DeviceFeatures, FloatControlsProperties};
use vulkane_glsl_emit::Glsl;
use vulkane_spirv_emit::{RequiredFeatures, required_features};

/// The cases this arm lowers: every type among `F32`, `F64`, `I32` and `U32`, with one float width
/// (glslang keeps one float-control execution mode per kind, so `F32` with `F64` is a decline).
fn mine() -> Vec<CastCase> {
    cases()
        .into_iter()
        .filter(|c| {
            !mixed(c)
                && [c.src, c.dst, c.compute]
                    .iter()
                    .all(|t| matches!(t, F32 | F64 | I32 | U32))
        })
        .collect()
}

fn known(c: &CastCase) -> bool {
    [c.src, c.dst, c.compute]
        .iter()
        .all(|t| matches!(t, F32 | F64 | I32 | U32))
}

fn mixed(c: &CastCase) -> bool {
    let ts = [c.src, c.dst, c.compute];
    known(c) && ts.contains(&F32) && ts.contains(&F64)
}

fn device_ok(need: &RequiredFeatures, fc: Option<FloatControlsProperties>) -> bool {
    let Some(c) = fc else {
        return false;
    };
    (!need.denorm_preserve_f32 || c.denorm_preserve_f32)
        && (!need.denorm_preserve_f64 || c.denorm_preserve_f64)
        && (!need.signed_zero_inf_nan_preserve_f32 || c.signed_zero_inf_nan_preserve_f32)
        && (!need.signed_zero_inf_nan_preserve_f64 || c.signed_zero_inf_nan_preserve_f64)
        && (!need.rounding_mode_rte_f32 || c.rounding_mode_rte_f32)
        && (!need.rounding_mode_rte_f64 || c.rounding_mode_rte_f64)
}

fn run_case(c: &CastCase) -> Option<(Vec<u32>, Vec<u32>)> {
    common::require_serialization_lock();
    let uses_f64 = [c.src, c.dst, c.compute].contains(&F64);
    let features = if uses_f64 {
        DeviceFeatures::new().with_shader_float64()
    } else {
        DeviceFeatures::new()
    };
    let gpu = gpu::Gpu::open_with(c.name, Some((&features, "shaderFloat64")))?;
    let g = Glsl.lower_kernel(&kernel(c)).expect("GLSL lowers the cast");
    let spv = glslang_words(&g)?;
    if let Some((ok, err)) = validates(&spv) {
        assert!(ok, "spirv-val rejected {}:\n{err}\n{}", c.name, g.source);
    }
    let need = required_features(&spv).expect("a module");
    let fc = gpu.float_controls();
    eprintln!("{}: needs {need:?}; device {fc:?}", c.name);
    if !device_ok(&need, fc) {
        common::skip(common::Missing::capability(
            "the float controls this cast module declares (VkPhysicalDeviceFloatControlsProperties)",
        ));
        return None;
    }
    let in_words = words(c.src, &c.input, 2 * THREADS);
    let got = gpu.run("main", &spv, &[&in_words]);
    Some((got, words(c.dst, &c.want, 0)))
}

#[test]
fn every_cast_matches_the_oracle_on_a_device_through_glsl_and_its_mutant_would_not() {
    let all = mine();
    assert!(all.len() >= 6, "only {} cases are in scope", all.len());
    for c in all {
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

/// `RoundingModeRTE` is declared where (and only where) the kernel rounds a conversion.
#[test]
fn the_rounding_mode_is_declared_only_for_a_rounding_conversion() {
    let mut rounding = 0;
    let mut exact = 0;
    for c in mine() {
        let g = Glsl.lower_kernel(&kernel(&c)).expect("lowers");
        let Some(spv) = glslang_words(&g) else {
            return;
        };
        let need = required_features(&spv).expect("a module");
        let rte = need.rounding_mode_rte_f32 || need.rounding_mode_rte_f64;
        let int_src = matches!(c.src, I32 | U32);
        let rounds = matches!(c.dst, F32 | F64) && (int_src || (c.src == F64 && c.dst == F32));
        assert_eq!(rte, rounds, "{}: {need:?}\n{}", c.name, g.source);
        if rte {
            rounding += 1;
        } else {
            exact += 1;
        }
    }
    // Control: both kinds occurred, so the assertion above discriminated.
    assert!(
        rounding > 0 && exact > 0,
        "{rounding} rounding, {exact} exact"
    );
}

/// A conversion between `F32` and `F64` needs float controls at two widths, which glslang cannot
/// carry: a typed decline naming it. Control: the same cases lower as soon as one width is left.
#[test]
fn an_f32_f64_conversion_is_a_typed_decline_and_the_other_casts_lower() {
    let all = cases();
    let (mixed_cases, rest): (Vec<_>, Vec<_>) = all.iter().partition(|c| mixed(c));
    assert!(!mixed_cases.is_empty() && !rest.is_empty());
    for c in mixed_cases {
        let err = Glsl.lower_kernel(&kernel(c)).expect_err("declined");
        assert!(
            format!("{err:?}").contains("both F32 and F64"),
            "{}: {err:?}",
            c.name
        );
    }
    for c in rest.iter().filter(|c| {
        [c.src, c.dst, c.compute]
            .iter()
            .all(|t| matches!(t, F32 | F64 | I32 | U32))
    }) {
        Glsl.lower_kernel(&kernel(c)).expect("lowers");
    }
}

#[test]
fn a_rounding_the_oracle_defines_but_glsl_leaves_to_the_implementation_is_a_typed_decline() {
    use unpopped::ir::{RoundMode, ScalarExpr as E};
    // `NearestAway` (ties away from zero): GLSL `round()` leaves ties to the implementation.
    let c = mine()
        .into_iter()
        .find(|c| matches!(c.src, F32) && matches!(c.dst, I32))
        .expect("an f32 to i32 case");
    let mut k = kernel(&c);
    if let Some(unpopped::kernel::Stmt::Store {
        value: E::Cast(_, spec),
        ..
    }) = k.body.0.last_mut()
    {
        spec.round = RoundMode::NearestAway;
    }
    let err = Glsl.lower_kernel(&k).expect_err("declined");
    assert!(format!("{err:?}").contains("NearestAway"), "{err:?}");
    // Control: the unmodified case lowers.
    Glsl.lower_kernel(&kernel(&c)).expect("lowers");
}
