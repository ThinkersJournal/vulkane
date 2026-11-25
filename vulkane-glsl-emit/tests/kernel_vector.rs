// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packed values (`vec2`, `vec4`) in an `F32` kernel through GLSL (the kernel of `vulkane-spirv-emit`'s
//! `vector_kernel` support): `Pack`, lane-wise `Add`/`Mul`/`Max` and `Extract`, with a lane swap that
//! reads the lanes it overwrites. Packed values are scalarized before lowering (`unpopped`'s `lower_dag`
//! has no seam for them), so the source has no `vec` type. One device run per lane, bit-exact against
//! native `f32`; the fused, unswapped and swap-in-place mutants each differ from the reference.

use vulkane_test_support::common;
use vulkane_test_support::glslang;
use vulkane_test_support::gpu;
use vulkane_test_support::vector_kernel;

use glslang::{glslang_words, validates};
use unpopped::backend::Backend;
use unpopped::ir::{ScalarExpr as E, UnaryOp};
use unpopped::kernel::Stmt;
use vector_kernel::{Variant, data, same, vector_kernel};
use vulkane_glsl_emit::Glsl;

fn run(tag: &str, lanes: u8, store: u8) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let gpu = gpu::Gpu::open(tag)?;
    let g = Glsl
        .lower_kernel(&vector_kernel(lanes, store))
        .expect("GLSL lowers the vector kernel");
    let words = glslang_words(&g)?;
    if let Some((ok, err)) = validates(&words) {
        assert!(ok, "spirv-val rejected the module:\n{err}\n{}", g.source);
    }
    if !gpu
        .float_controls()
        .is_some_and(|c| c.denorm_preserve_f32 && c.signed_zero_inf_nan_preserve_f32)
    {
        common::skip(common::Missing::capability(
            "DenormPreserve + SignedZeroInfNanPreserve at 32 bits (VkPhysicalDeviceFloatControlsProperties)",
        ));
        return None;
    }
    let (a, b, _) = data(lanes, store, Variant::Right);
    Some(gpu.run("main", &words, &[&a, &b]))
}

#[test]
fn every_lane_of_vec2_and_vec4_matches_native_f32_on_a_device_through_glsl() {
    for lanes in [2u8, 4] {
        for store in 0..lanes {
            let Some(got) = run(&format!("glsl_vec{lanes}_lane{store}"), lanes, store) else {
                return; // skip already declared
            };
            let (_, _, want) = data(lanes, store, Variant::Right);
            same(&got, &want).unwrap_or_else(|e| panic!("vec{lanes} lane {store}: {e}"));
        }
    }
}

#[test]
fn the_fused_the_unswapped_and_the_in_place_kernels_would_fail_the_check() {
    for (lanes, variants) in [
        (2u8, vec![Variant::Fused, Variant::NoSwap, Variant::InPlace]),
        (4, vec![Variant::NoSwap, Variant::InPlace]),
    ] {
        for v in variants {
            let differs = (0..lanes).any(|s| {
                let (_, _, wrong) = data(lanes, s, v);
                let (_, _, right) = data(lanes, s, Variant::Right);
                same(&wrong, &right).is_err()
            });
            assert!(differs, "vec{lanes} {v:?} is indistinguishable");
        }
    }
}

/// The source is scalar (no `vec` type): each lane is a local, and the swap goes through temporaries
/// (each lane evaluated before any is written). A control kernel with no packed local is untouched.
#[test]
fn packed_values_become_scalar_lanes_with_the_swap_through_temporaries() {
    let g = Glsl.lower_kernel(&vector_kernel(2, 0)).expect("lowers");
    assert!(
        !g.source.contains("vec2") && !g.source.contains("vec4"),
        "{}",
        g.source
    );
    // Lane locals plus two temporaries for each of the two assignments.
    assert!(
        g.source.matches("precise float l").count() >= 7,
        "{}",
        g.source
    );
    assert!(g.source.contains("const float"), "{}", g.source);
}

/// A packed operand of an operator that is not lane-wise has no rewrite: a typed decline naming it.
#[test]
fn a_non_lane_wise_operator_on_a_packed_value_is_a_typed_decline() {
    let mut k = vector_kernel(2, 0);
    for s in &mut k.body.0 {
        if let Stmt::Assign { value, .. } = s {
            if !matches!(value, E::Pack(_)) {
                *value = E::Unary(UnaryOp::Neg, Box::new(E::Local(2)));
                break;
            }
        }
    }
    let err = Glsl.lower_kernel(&k).expect_err("declined");
    let text = format!("{err:?}");
    assert!(
        text.contains("Unary") && text.contains("lane-wise"),
        "{text}"
    );
}
