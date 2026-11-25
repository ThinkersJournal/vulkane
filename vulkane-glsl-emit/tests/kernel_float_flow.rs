// SPDX-License-Identifier: MIT OR Apache-2.0
//! Control flow in `F32` kernels through GLSL (the kernel of `vulkane-spirv-emit`'s
//! `float_flow_kernel` support): `If` and `While` take float conditions, a `For` counts with an
//! integer local (`U32` up, `I32` down), `Break` leaves it, and the multiply-add inside is not fused.
//! Lowered by `Glsl::lower_kernel`, compiled by `glslang`, validated by `spirv-val`, run on a device,
//! bit-exact against native `f32`. A device without the float controls prints a `SKIP (unsupported)`
//! line naming the capability.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "../../vulkane-spirv-emit/tests/support/float_flow_kernel.rs"]
#[allow(dead_code)]
mod float_flow_kernel;
#[path = "support/glslang.rs"]
mod glslang;
#[path = "../../vulkane-spirv-emit/tests/support/gpu.rs"]
mod gpu;

use float_flow_kernel::{Variant, data, float_flow_kernel, same};
use glslang::{glslang_words, validates};
use unpopped::backend::{Backend, GeneratedKernel};
use unpopped_vocab::ElementKind::{self, I32, U32};
use vulkane_glsl_emit::Glsl;

fn compile(counter: ElementKind) -> Option<(GeneratedKernel, Vec<u32>)> {
    let g = Glsl
        .lower_kernel(&float_flow_kernel(counter))
        .expect("GLSL lowers the float control-flow kernel");
    let words = glslang_words(&g)?;
    if let Some((ok, err)) = validates(&words) {
        assert!(ok, "spirv-val rejected the module:\n{err}\n{}", g.source);
    }
    Some((g, words))
}

fn run(tag: &str, counter: ElementKind) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let gpu = gpu::Gpu::open(tag)?;
    let (_, words) = compile(counter)?;
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
    Some(gpu.run("main", &words, &[&x, &w]))
}

#[test]
fn float_conditions_and_integer_counters_match_native_f32_on_a_device_through_glsl() {
    for (counter, tag) in [(U32, "glsl_flow_u32"), (I32, "glsl_flow_i32")] {
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
}

/// The source holds the control flow, and the module keeps every multiply and add `NoContraction`
/// (the stored value, the `if` test and the counter's bounds go through `precise` temporaries).
/// Mutant: the same source without `precise` carries none.
#[test]
fn the_flow_is_spelled_and_the_arithmetic_stays_no_contraction() {
    for counter in [U32, I32] {
        let Some((g, words)) = compile(counter) else {
            return;
        };
        for needle in ["for (l3 =", "while (true)", "break;", "precise const float"] {
            assert!(
                g.source.contains(needle),
                "wanted `{needle}` in\n{}",
                g.source
            );
        }
        let decorated = |w: &[u32]| {
            let (mut at, mut n) = (5, 0);
            while at < w.len() {
                let (count, op) = ((w[at] >> 16) as usize, w[at] & 0xffff);
                if op == 71 && w[at + 2] == 42 {
                    n += 1;
                }
                at += count.max(1);
            }
            n
        };
        assert!(decorated(&words) >= 4, "{counter:?}\n{}", g.source);
        let loose =
            GeneratedKernel::new("float_flow".to_string(), g.source.replace("precise ", ""));
        let loose = glslang_words(&loose).expect("the mutant compiles");
        assert_eq!(decorated(&loose), 0);
    }
}

/// An integer comparison condition has no spelling (the shared walker lowers no integer test inside a
/// float kernel): a typed decline.
#[test]
fn an_integer_condition_in_a_float_kernel_is_a_typed_decline() {
    use unpopped::ir::{BinaryOp, ScalarExpr as E};
    use unpopped::kernel::Stmt;
    let mut k = float_flow_kernel(U32);
    let Stmt::For { body, .. } = &mut k.body.0[2] else {
        panic!("the third statement is the loop");
    };
    let Stmt::If { cond, .. } = &mut body.0[1] else {
        panic!("the second loop statement is the first `if`");
    };
    *cond = E::Binary(
        BinaryOp::CmpLt,
        Box::new(E::Local(3)),
        Box::new(E::ConstInt(2, U32)),
    );
    let err = Glsl.lower_kernel(&k).expect_err("declined");
    let text = format!("{err:?}");
    assert!(
        text.contains("Integer") && text.contains("int_op"),
        "{text}"
    );
}
