// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for `F16` as a compute dtype in `Backend::lower_kernel`: the half capabilities and float
//! controls are declared exactly once, ops whose Vulkan bound is looser than the contract tier (or
//! unchecked at 16 bits) are declined by name, and a literal that is not exactly a half is declined
//! rather than rounded.

use unpopped::backend::Backend;
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::F16;

use crate::tests::assert_valid;
use crate::{SpirV, required_features};

use crate::kernel_float_tests::float_kernel::{Case, float_kernel};

fn count(src: &str, needle: &str) -> usize {
    src.lines().filter(|l| l.contains(needle)).count()
}

#[test]
fn an_f16_kernel_declares_half_capabilities_once_and_no_float64() {
    for case in [Case::MulAdd, Case::Max, Case::Min, Case::Floor] {
        let g = SpirV
            .lower_kernel(&float_kernel(F16, case))
            .expect("lowers");
        assert_valid(&g);
        let s = &g.source;
        assert_eq!(count(s, "OpCapability Float16"), 1, "{s}");
        assert_eq!(count(s, "OpCapability StorageBuffer16BitAccess"), 1, "{s}");
        assert!(s.contains("DenormPreserve 16"), "{s}");
        assert!(s.contains("SignedZeroInfNanPreserve 16"), "{s}");
        assert!(
            !s.contains("Float64") && !s.contains("RoundingModeRTE"),
            "{s}"
        );
        let need = required_features(g.words().expect("words")).expect("a module");
        assert!(need.shader_float16 && need.storage_buffer_16bit, "{need:?}");
        assert!(
            need.denorm_preserve_f16 && !need.denorm_preserve_f32,
            "{need:?}"
        );
    }
    // The multiply feeding the add stays two rounded half operations.
    let g = SpirV
        .lower_kernel(&float_kernel(F16, Case::MulAdd))
        .expect("lowers");
    assert_eq!(count(&g.source, "NoContraction"), 2, "{}", g.source);
}

#[test]
fn f16_ops_with_a_looser_or_unchecked_bound_are_declined_by_name() {
    for (case, needle) in [
        (Case::Div, "2.5 ULP"),
        (Case::Sqrt, "inversesqrt"),
        (Case::Exp, "3 + 2 x |x| ULP"),
    ] {
        let err = SpirV
            .lower_kernel(&float_kernel(F16, case))
            .expect_err("declined");
        let text = format!("{err:?}");
        assert!(text.contains(needle), "{case:?}: {text}");
    }
}

fn literal_kernel(v: f64) -> KernelFn {
    let at = Place::Logical {
        buffer: 0,
        coords: vec![E::Exec(unpopped::exec::ExecId::ThreadIdx(0))],
    };
    let mut k = KernelFn::new(
        "lit",
        F16,
        vec![
            Param::buffer("a", F16, BufferRole::Read, Some(1)),
            Param::buffer("out", F16, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F16)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at.clone(),
            },
            Stmt::Store {
                place: Place::Logical {
                    buffer: 1,
                    coords: vec![E::Exec(unpopped::exec::ExecId::ThreadIdx(0))],
                },
                value: E::Add(Box::new(E::Local(0)), Box::new(E::Const(v))),
            },
        ]),
    );
    k.launch = Some(unpopped::exec::LaunchShape::new([64, 1, 1], [1, 1, 1]));
    k
}

#[test]
fn an_f16_literal_must_be_exactly_a_half() {
    // 0.5 and 65504 are halves.
    for v in [0.5, 65504.0, -0.0] {
        assert_valid(&SpirV.lower_kernel(&literal_kernel(v)).expect("exact"));
    }
    // 0.1 is not, and is declined instead of rounded; the control above shows the kernel shape lowers.
    let err = SpirV
        .lower_kernel(&literal_kernel(0.1))
        .expect_err("0.1 is not a half");
    assert!(format!("{err:?}").contains("not exactly an F16"), "{err:?}");
}
