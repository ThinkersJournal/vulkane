// SPDX-License-Identifier: MIT OR Apache-2.0
//! Control flow in `F32` kernels on a live device (`tests/support/float_flow_kernel.rs`): `If` and
//! `While` take float conditions, a `For` counts with an integer local (`U32` up, `I32` down), `Break`
//! leaves it, and the multiply-add inside is not fused. Results are bit-exact against native `f32`
//! (NaN equal to NaN); each wrong kernel (fused, `break` ignored) must differ from the reference
//! somewhere, or the check could not catch it. A device without the float controls prints a
//! `SKIP (unsupported)` line naming the capability.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(
    dead_code,
    unreachable_pub,
    missing_docs,
    reason = "shared test module: each test binary compiles it whole but uses part of it"
)]
mod common;
#[path = "support/float_flow_kernel.rs"]
#[allow(
    dead_code,
    reason = "shared test module: each test binary compiles it whole but uses part of it"
)]
mod float_flow_kernel;
#[path = "support/gpu.rs"]
mod gpu;

use float_flow_kernel::{Variant, data, float_flow_kernel, same};
use unpopped::backend::Backend;
use unpopped_vocab::ElementKind::{I32, U32};
use vulkane_spirv_emit::SpirV;

fn run(tag: &str, counter: unpopped_vocab::ElementKind) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let gpu = gpu::Gpu::open(tag)?;
    let g = SpirV
        .lower_kernel(&float_flow_kernel(counter))
        .expect("the SPIR-V backend lowers the float control-flow kernel");
    let ok = gpu
        .float_controls()
        .is_some_and(|c| c.denorm_preserve_f32 && c.signed_zero_inf_nan_preserve_f32);
    if !ok {
        common::skip(common::Missing::capability(
            "DenormPreserve + SignedZeroInfNanPreserve at 32 bits (VkPhysicalDeviceFloatControlsProperties)",
        ));
        return None;
    }
    let (x, w, _) = data(counter, Variant::Right);
    Some(gpu.run(&g.name, g.words().expect("words"), &[&x, &w]))
}

#[test]
fn float_conditions_and_integer_counters_match_native_f32_on_a_device() {
    for (counter, tag) in [(U32, "spirv_flow_u32"), (I32, "spirv_flow_i32")] {
        let Some(got) = run(tag, counter) else {
            continue; // skip already declared
        };
        let (_, _, want) = data(counter, Variant::Right);
        same(&got, &want).unwrap_or_else(|e| panic!("{counter:?} counter: {e}"));
    }
}

/// Mutants: a fused multiply-add and an ignored `break` each give a different output somewhere (so the
/// device check above would fail them), and the comparison accepts the reference against itself.
#[test]
fn the_fused_and_the_no_break_kernels_would_fail_the_check() {
    for counter in [U32, I32] {
        let (_, _, right) = data(counter, Variant::Right);
        for v in [Variant::Fused, Variant::NoBreak] {
            let (_, _, wrong) = data(counter, v);
            assert!(
                same(&wrong, &right).is_err(),
                "{counter:?} {v:?} is indistinguishable"
            );
        }
        assert!(same(&right, &right).is_ok());
    }
    // The two counter directions add in a different order, so they differ too.
    let (_, _, up) = data(U32, Variant::Right);
    let (_, _, down) = data(I32, Variant::Right);
    assert!(same(&up, &down).is_err());
}
