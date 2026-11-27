// SPDX-License-Identifier: MIT OR Apache-2.0
//! The row 9e test kernel: predicated access (`Stmt::MaskedLoad`, `Stmt::MaskedStore`), one block of
//! `THREADS` threads over an input `a` and an output `out` of `THREADS` cells each. Thread `t`:
//!
//! - `l2 = t + SHIFT`, `l1 = MaskedLoad(a[l2], mask = t < VALID, fill = FILL)`: `SHIFT` is chosen so `l2` is a
//!   valid index exactly when the mask holds (`t + SHIFT < THREADS` iff `t < VALID`), so a load that is not skipped
//!   under a false mask reads out of range (a fault, or zero under robust buffer access, which is not `FILL`);
//! - `out[t] = l1`;
//! - `MaskedStore(out[t], l1 + ADD, mask = t & 1)`: only odd threads add; an even thread's masked-off store must
//!   leave the `l1` the plain store just wrote (an unconditional store would write `l1 + ADD`).
//!
//! The reference is [`expected`].

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::BinaryOp::{BitAnd, CmpLt};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::U32;

/// Threads launched (one block) and the cells of each buffer.
pub const THREADS: u32 = 256;
/// Threads whose mask holds.
pub const VALID: u32 = 100;
/// Index shift of the masked load.
pub const SHIFT: u32 = THREADS - VALID;
/// The value a masked-off load takes (not zero, so a robust out-of-range read is told from it).
pub const FILL: u32 = 7;
/// What an odd thread's masked store adds.
pub const ADD: u32 = 5;

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

fn at(buffer: u32, index: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![E::Local(index)],
    }
}

/// The kernel.
pub fn masked_kernel() -> KernelFn {
    let body = vec![
        Stmt::Let {
            local: 0,
            init: E::Exec(ExecId::ThreadIdx(0)),
        },
        Stmt::Let {
            local: 2,
            init: E::Add(Box::new(E::Local(0)), Box::new(c(i64::from(SHIFT)))),
        },
        Stmt::MaskedLoad {
            local: 1,
            place: at(0, 2),
            mask: E::Binary(CmpLt, Box::new(E::Local(0)), Box::new(c(i64::from(VALID)))),
            fill: c(i64::from(FILL)),
        },
        Stmt::Store {
            place: at(1, 0),
            value: E::Local(1),
        },
        Stmt::MaskedStore {
            place: at(1, 0),
            value: E::Add(Box::new(E::Local(1)), Box::new(c(i64::from(ADD)))),
            mask: E::Binary(BitAnd, Box::new(E::Local(0)), Box::new(c(1))),
        },
    ];
    let mut k = KernelFn::new(
        "masked",
        U32,
        vec![
            Param::buffer("a", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::ReadWrite, Some(1)),
        ],
        vec![LocalDecl::scalar(U32); 3],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]));
    k
}

/// The input cells (`THREADS` of them), each distinct and never `FILL`.
pub fn input() -> Vec<u32> {
    (0..THREADS).map(|i| 1000 + 3 * i).collect()
}

/// What `out[t]` must hold.
pub fn expected(t: u32) -> u32 {
    let base = if t < VALID {
        1000 + 3 * (t + SHIFT)
    } else {
        FILL
    };
    if t & 1 == 1 { base + ADD } else { base }
}
