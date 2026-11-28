// SPDX-License-Identifier: MIT OR Apache-2.0
//! The seven dtypes unpopped 0.60 added to `ElementKind` (`F128`, `I128`, `U128`, `I256`, `U256`, `F8E3M4`,
//! `F8UE4M3`; KISS 0.24.0, plan E1a) are declined by name by the SpirV emitter: as the compute dtype of a kernel
//! and as the dtype `supports_dtype` is asked about. Measured against unpopped 0.60.1: `check_kernel` refuses
//! them first (`KernelError::DtypeNotComputed`), so the refusal text names the dtype and the checker's variant;
//! the emitter's own allow-list says `Unsupported` for each (a wildcard arm, not an exhaustive match), so a future
//! dtype is refused too unless someone adds it to a supported list. Every `ElementKind` is walked
//! (`ElementKind::ALL`, 35 at 0.60.1), so a dtype added to the enum later cannot be lowered by accident or panic.
//!
//! **What is pinned, and what is not:** the first test pins UNPOPPED's checker text (`DtypeNotComputed`), which is not this emitter's gate: the
//! checker refuses before the emitter is asked, so nothing the emitter does can change that text. The emitter's own decline is pinned by the
//! `supports_dtype` test alone (same rule as rockfish and baracuda).

use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::{ElementKind, TargetId};
use vulkane_spirv_emit::SpirV;

/// The dtypes unpopped 0.60 added.
const NEW: [ElementKind; 7] = [
    ElementKind::F128,
    ElementKind::I128,
    ElementKind::U128,
    ElementKind::I256,
    ElementKind::U256,
    ElementKind::F8E3M4,
    ElementKind::F8UE4M3,
];

fn place() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![ScalarExpr::Exec(ExecId::ThreadIdx(0))],
    }
}

/// The dtypes the CHECKER computes since unpopped 0.62 (the four wide integers, ROADMAP E2) and 0.64 (`F128`, the soft binary128 of E3),
/// so it no longer declines them and each emitter's own dtype gate must.
fn is_wide(t: ElementKind) -> bool {
    matches!(
        t,
        ElementKind::I128
            | ElementKind::U128
            | ElementKind::I256
            | ElementKind::U256
            | ElementKind::F128
    )
}

fn kernel(t: ElementKind) -> KernelFn {
    KernelFn::new(
        "probe",
        t,
        vec![Param::buffer("out", t, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(t)],
        // A load and a store of a local, not a constant: since unpopped 0.62 the four wide integers are computed dtypes,
        // and an integer constant of one (`ConstInt(1, I128)`) is an illegal leaf the checker refuses first.
        Block(vec![
            Stmt::Load {
                local: 0,
                place: place(),
            },
            Stmt::Store {
                place: place(),
                value: ScalarExpr::Local(0),
            },
        ]),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]))
}

fn target() -> TargetId {
    TargetId::parse("vulkan:sg32.arith-f16-i8").expect("a vulkan target token")
}

#[test]
fn each_new_dtype_is_declined_by_name_as_a_kernel_dtype() {
    for t in NEW {
        match SpirV.lower_kernel(&kernel(t)) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                if is_wide(t) {
                    assert!(
                        detail.contains(&format!("{t:?}"))
                            && !detail.contains("check_kernel refused")
                            && !detail.contains("checker refuses"),
                        "{t:?}: the EMITTER's own gate must name a wide integer, got: {detail}"
                    );
                } else {
                    assert!(
                        detail.contains("DtypeNotComputed") && detail.contains(&format!("{t:?}")),
                        "{t:?}: the refusal must name the dtype and the checker's variant, got: {detail}"
                    );
                }
            }
            other => panic!("{t:?}: expected the typed UnsupportedKernel decline, got {other:?}"),
        }
    }
    // The control: the same kernel at a supported dtype lowers, so the refusals above are about the dtype.
    assert!(
        SpirV.lower_kernel(&kernel(ElementKind::U32)).is_ok(),
        "control: a U32 kernel lowers"
    );
}

#[test]
fn each_new_dtype_is_unsupported_to_supports_dtype() {
    for t in NEW {
        assert!(
            !SpirV.supports_dtype(t, target()).is_supported(),
            "{t:?} must be Unsupported"
        );
    }
    assert!(
        SpirV
            .supports_dtype(ElementKind::U32, target())
            .is_supported(),
        "control: U32 is supported"
    );
}

#[test]
fn no_dtype_panics_and_every_new_one_is_among_the_refused() {
    let mut refused_new = 0;
    for t in ElementKind::ALL {
        // Must not panic whatever the dtype; the result is a lowering or a typed decline.
        let r = SpirV.lower_kernel(&kernel(t));
        if NEW.contains(&t) {
            assert!(r.is_err(), "{t:?} must be refused");
            refused_new += 1;
        }
    }
    assert_eq!(
        refused_new,
        NEW.len(),
        "every new dtype is in ElementKind::ALL"
    );
}
