// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packed values (`vec2`, `vec4`) in an `F32` kernel on a live device (`tests/support/vector_kernel.rs`):
//! `Pack`, lane-wise `Add`/`Mul`/`Max` and `Extract`, with a lane swap that reads the lanes it
//! overwrites. One run per lane, each bit-exact against native `f32` (NaN equal to NaN). The mutants
//! (fused, no swap, swap in place) each differ from the reference somewhere. A device without the float
//! controls prints a `SKIP (unsupported)` line naming the capability.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(
    dead_code,
    unreachable_pub,
    missing_docs,
    reason = "shared test module: each test binary compiles it whole but uses part of it"
)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;
#[path = "support/vector_kernel.rs"]
#[allow(
    dead_code,
    reason = "shared test module: each test binary compiles it whole but uses part of it"
)]
mod vector_kernel;

use unpopped::backend::Backend;
use vector_kernel::{Variant, data, same, vector_kernel};
use vulkane_spirv_emit::SpirV;

fn run(tag: &str, lanes: u8, store: u8) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let gpu = gpu::Gpu::open(tag)?;
    let g = SpirV
        .lower_kernel(&vector_kernel(lanes, store))
        .expect("the SPIR-V backend lowers the vector kernel");
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
    Some(gpu.run(&g.name, g.words().expect("words"), &[&a, &b]))
}

#[test]
fn every_lane_of_vec2_and_vec4_matches_native_f32_on_a_device() {
    for lanes in [2u8, 4] {
        for store in 0..lanes {
            let Some(got) = run(&format!("spirv_vec{lanes}_lane{store}"), lanes, store) else {
                return; // skip already declared
            };
            let (_, _, want) = data(lanes, store, Variant::Right);
            same(&got, &want).unwrap_or_else(|e| panic!("vec{lanes} lane {store}: {e}"));
        }
    }
}

/// Mutants: every wrong kernel differs from the reference in some lane (so the device check above would
/// fail it), and the comparison accepts the reference against itself.
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
        let (_, _, r) = data(lanes, 0, Variant::Right);
        assert!(same(&r, &r).is_ok());
    }
}
