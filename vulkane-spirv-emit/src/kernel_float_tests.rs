// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for `F32`/`F64` kernels in `Backend::lower_kernel`: the float capabilities and execution
//! modes are declared only for a float kernel (and `Float64` only at `F64`), every multiply, add
//! and subtract is `NoContraction`, an op whose Vulkan accuracy bound is looser than its contract
//! tier is declined by name, and the plan path is unchanged.

use unpopped::backend::Backend;
use unpopped_vocab::ElementKind::{F32, F64};

use crate::tests::assert_valid;
use crate::{SpirV, required_features};

#[path = "../tests/support/float_kernel.rs"]
#[allow(
    dead_code,
    reason = "shared test module: each test binary compiles it whole but uses part of it"
)]
pub(crate) mod float_kernel;
use float_kernel::{Case, float_kernel};

fn count(src: &str, needle: &str) -> usize {
    src.lines().filter(|l| l.contains(needle)).count()
}

#[test]
fn a_float_kernel_declares_ieee_controls_and_float64_only_at_f64() {
    for (ty, width, f64k) in [(F32, 32, false), (F64, 64, true)] {
        let g = SpirV
            .lower_kernel(&float_kernel(ty, Case::MulAdd))
            .expect("lowers");
        assert_valid(&g);
        let s = &g.source;
        assert!(s.contains(&format!("DenormPreserve {width}")), "{s}");
        assert!(
            s.contains(&format!("SignedZeroInfNanPreserve {width}")),
            "{s}"
        );
        assert_eq!(s.contains("OpCapability Float64"), f64k, "{s}");
        assert!(!s.contains("RelaxedPrecision"), "{s}");
        let need = required_features(g.words().expect("words")).expect("a module");
        assert_eq!(need.shader_float64, f64k, "{need:?}");
        assert_eq!(need.denorm_preserve_f32, !f64k, "{need:?}");
        assert_eq!(need.denorm_preserve_f64, f64k, "{need:?}");
        assert_eq!(need.signed_zero_inf_nan_preserve_f32, !f64k, "{need:?}");
        assert_eq!(need.signed_zero_inf_nan_preserve_f64, f64k, "{need:?}");
        assert!(
            !need.shader_int64 && !need.shader_int8 && !need.shader_int16,
            "{need:?}"
        );
    }
}

/// Every `OpFMul`/`OpFAdd` carries `NoContraction` (so no driver fuses them into an FMA the source
/// does not ask for, fuel#568). Control: the same query counts the two arithmetic ops themselves.
#[test]
fn every_float_multiply_and_add_is_no_contraction() {
    let g = SpirV
        .lower_kernel(&float_kernel(F32, Case::MulAdd))
        .expect("lowers");
    let s = &g.source;
    assert_eq!(count(s, "OpFMul"), 1, "{s}");
    assert_eq!(count(s, "OpFAdd"), 1, "{s}");
    assert_eq!(count(s, "NoContraction"), 2, "{s}");
}

/// A 32-bit integer kernel declares no float control (golden-pinned elsewhere); the same queries
/// find them in the float kernel above.
#[test]
fn an_integer_kernel_declares_no_float_control() {
    let k = crate::kernel_tests::reverse_kernel::reverse(64, true, false);
    let g = SpirV.lower_kernel(&k).expect("lowers");
    for needle in [
        "DenormPreserve",
        "SignedZeroInfNanPreserve",
        "Float64",
        "NoContraction",
    ] {
        assert!(!g.source.contains(needle), "{needle}");
    }
    let need = required_features(g.words().expect("words")).expect("a module");
    assert_eq!(need, crate::RequiredFeatures::default());
}

#[test]
fn ops_whose_vulkan_bound_is_looser_than_the_contract_tier_are_declined_by_name() {
    for (case, needle) in [
        (Case::Div, "2.5 ULP"),
        (Case::Sqrt, "inversesqrt"),
        (Case::Exp, "3 + 2 x |x| ULP"),
    ] {
        for ty in [F32, F64] {
            match SpirV.lower_kernel(&float_kernel(ty, case)) {
                Err(e) => {
                    let text = format!("{e:?}");
                    assert!(text.contains(needle), "{case:?} {ty:?}: {text}");
                }
                Ok(_) => panic!("{case:?} {ty:?}: expected a decline"),
            }
        }
    }
    // Control: the allowed cases lower, so the declines are the ops and not the kernel shape.
    for case in [Case::MulAdd, Case::Max, Case::Min, Case::Floor] {
        SpirV
            .lower_kernel(&float_kernel(F32, case))
            .expect("lowers");
    }
}

#[path = "../tests/support/float_flow_kernel.rs"]
#[allow(dead_code)]
mod float_flow_kernel;

/// `If`, `While` and `For` in a float kernel give valid modules for both counter types, with the
/// multiply and the add still `NoContraction`.
#[test]
fn float_control_flow_lowers_to_valid_modules_for_both_counters() {
    for counter in [
        unpopped_vocab::ElementKind::U32,
        unpopped_vocab::ElementKind::I32,
    ] {
        let g = SpirV
            .lower_kernel(&float_flow_kernel::float_flow_kernel(counter))
            .expect("lowers");
        assert_valid(&g);
        let s = &g.source;
        // Two multiplies (`y * w` in the test and in the add), the add and the subtract, and the
        // final halving multiply: every float add, subtract and multiply is decorated.
        let arith = count(s, "OpFMul") + count(s, "OpFAdd") + count(s, "OpFSub");
        assert_eq!(count(s, "NoContraction"), arith, "{counter:?}\n{s}");
        assert!(arith >= 4, "{arith}");
        assert!(
            s.contains("OpLoopMerge") && s.contains("OpSelectionMerge"),
            "{s}"
        );
    }
}

/// An integer comparison in a float kernel has no spelling (the shared `lower_dag` has no integer
/// seam here): a typed decline, never a module that compares integers as floats.
#[test]
fn an_integer_condition_in_a_float_kernel_is_a_typed_decline() {
    use unpopped::ir::{BinaryOp, ScalarExpr as E};
    use unpopped::kernel::Stmt;
    let mut k = float_flow_kernel::float_flow_kernel(unpopped_vocab::ElementKind::U32);
    let Stmt::For { body, .. } = &mut k.body.0[2] else {
        panic!("the third statement is the loop");
    };
    let Stmt::If { cond, .. } = &mut body.0[1] else {
        panic!("the second loop statement is the first `if`");
    };
    *cond = E::Binary(
        BinaryOp::CmpLt,
        Box::new(E::Local(3)),
        Box::new(E::ConstInt(2, unpopped_vocab::ElementKind::U32)),
    );
    let err = SpirV.lower_kernel(&k).expect_err("declined");
    assert!(
        format!("{err:?}").contains("not lowered") || format!("{err:?}").contains("Integer"),
        "{err:?}"
    );
}
