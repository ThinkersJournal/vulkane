// SPDX-License-Identifier: MIT OR Apache-2.0
//! unpopped 0.61 added `UnaryOp::{Popc, Clz, Brev}`. This emitter does not lower them yet: each is a typed `UnsupportedKernel` that names
//! the op and comes from THIS emitter (not from `check_kernel`, which accepts them at an integer type), for `I32` and `U32` kernels. The facts
//! (what each target says, and the IR rule) are in `docs/design/unaryop-bitops-0.61.md`. A kernel that lowered one would pass the checker and
//! be written as text, so the refusal is the only guard.

use unpopped::backend::{Backend, LowerError};
use unpopped_vocab::ElementKind;
use vulkane_test_support::bitop_kernel;
use vulkane_wgsl_emit::Wgsl;

#[test]
fn popc_clz_and_brev_are_declined_by_name_by_this_emitter() {
    for (name, op) in bitop_kernel::OPS {
        for t in [ElementKind::I32, ElementKind::U32] {
            let k = bitop_kernel::kernel(op, t);
            unpopped::kernel::check_kernel(&k).expect("the checker accepts an integer bit op");
            match Wgsl.lower_kernel(&k) {
                // The shared walker's decline is `UnsupportedOp`; a front end that declines first (HLSL, through fishstringer) says
                // `UnsupportedKernel`; the SPIR-V emitter's own unary gate says `UnsupportedDtype`. Each is a typed refusal that names the op.
                Err(
                    LowerError::UnsupportedOp { detail } | LowerError::UnsupportedKernel { detail },
                ) => {
                    assert!(
                        detail.contains(name),
                        "{name} at {t:?}: the refusal must name the op: {detail}"
                    );
                    assert!(
                        !detail.contains("check_kernel refused"),
                        "{name} at {t:?}: the refusal must be the emitter's own: {detail}"
                    );
                }
                other => panic!("{name} at {t:?}: expected the typed refusal, got {other:?}"),
            }
        }
    }
}
