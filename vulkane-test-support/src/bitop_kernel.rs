// SPDX-License-Identifier: MIT OR Apache-2.0
//! The unpopped 0.61 integer bit operations (`UnaryOp::Popc`, `Clz`, `Brev`) as a test kernel: 64 threads, `out[t] = op(in[t])` over an
//! `I32` or `U32` kernel. Used by every emitter's decline test (the three are declined by name until each is lowered; the facts are in
//! `docs/design/unaryop-bitops-0.61.md`).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{ScalarExpr as E, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

/// Threads launched.
pub const THREADS: u32 = 64;

/// The three operations by name.
pub const OPS: [(&str, UnaryOp); 3] = [
    ("Popc", UnaryOp::Popc),
    ("Clz", UnaryOp::Clz),
    ("Brev", UnaryOp::Brev),
];

/// `out[t] = op(in[t])` at integer type `t` (`I32` or `U32`).
#[must_use]
pub fn kernel(op: UnaryOp, t: ElementKind) -> KernelFn {
    let tid = || E::Exec(ExecId::ThreadIdx(0));
    let place = |buffer| Place::Logical {
        buffer,
        coords: vec![tid()],
    };
    KernelFn::new(
        "bitop",
        t,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(t)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: place(0),
            },
            Stmt::Store {
                place: place(1),
                value: E::Unary(op, Box::new(E::Local(0))),
            },
        ]),
    )
    .with_launch(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernels_are_checker_valid_at_both_integer_types_and_a_float_one_is_refused() {
        for (name, op) in OPS {
            for t in [ElementKind::I32, ElementKind::U32] {
                unpopped::kernel::check_kernel(&kernel(op, t))
                    .unwrap_or_else(|e| panic!("{name} at {t:?}: {e:?}"));
            }
            // The control: the checker refuses the same op on a float, so the emitters' own declines are what the tests below read.
            let float = unpopped::kernel::check_kernel(&kernel(op, ElementKind::F32));
            assert!(
                format!("{float:?}").contains("IntegerOnlyOperand"),
                "{name}: {float:?}"
            );
        }
    }
}
