// SPDX-License-Identifier: MIT OR Apache-2.0
//! The row 8d test kernel, shared by the unit tests and the live test (`tests/kernel_atomic_live.rs`).
//! Every thread of 16 blocks of 64 updates a four-cell `ReadWrite` buffer pre-filled with the
//! harness sentinel `0xAAAA_AAAA`:
//!
//! - cell 0: `AtomicRmw Add 1`, so the cell ends at `sentinel + 1024` only if no update is lost;
//! - cell 1: `AtomicRmw Min gid`, ending at `0` (unsigned: thread 0's id beats the sentinel);
//! - cell 2: `AtomicCas(expected = sentinel, desired = gid + 1)`, which exactly one thread wins; its
//!   `old` local feeds cell 3;
//! - cell 3: `AtomicRmw Add (old == sentinel)`, the count of CAS winners, which must be 1.
//!
//! With `racy`, cell 0 is a non-atomic `Load` / `Add` / `Store` instead: the control, which loses
//! updates on a device that interleaves the invocations.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::BinaryOp::CmpEq;
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind::U32;

pub(crate) const BLOCK: u32 = 64;
pub(crate) const BLOCKS: u32 = 16;
#[allow(dead_code)] // the unit tests do not use it; the live test does
pub(crate) const THREADS: u32 = BLOCK * BLOCKS;
pub(crate) const SENTINEL: u32 = 0xAAAA_AAAA;

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

fn cell(i: i64) -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![c(i)],
    }
}

fn rmw(op: AtomicOp, i: i64, value: E, old: Option<u32>) -> Stmt {
    Stmt::AtomicRmw {
        op,
        place: cell(i),
        value,
        old,
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    }
}

/// The kernel; `racy` replaces cell 0's atomic add by a plain read-modify-write.
pub(crate) fn atomic_kernel(racy: bool) -> KernelFn {
    let gid = E::Add(
        Box::new(E::Mul(
            Box::new(E::Exec(ExecId::BlockIdx(0))),
            Box::new(c(i64::from(BLOCK))),
        )),
        Box::new(E::Exec(ExecId::ThreadIdx(0))),
    );
    let mut body = vec![Stmt::Let {
        local: 0,
        init: gid,
    }];
    if racy {
        body.push(Stmt::Load {
            local: 1,
            place: cell(0),
        });
        body.push(Stmt::Store {
            place: cell(0),
            value: E::Add(Box::new(E::Local(1)), Box::new(c(1))),
        });
    } else {
        body.push(rmw(AtomicOp::Add, 0, c(1), None));
    }
    body.push(rmw(AtomicOp::Min, 1, E::Local(0), None));
    body.push(Stmt::AtomicCas {
        place: cell(2),
        expected: c(i64::from(SENTINEL)),
        desired: E::Add(Box::new(E::Local(0)), Box::new(c(1))),
        old: Some(2),
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    });
    let won = E::Binary(
        CmpEq,
        Box::new(E::Local(2)),
        Box::new(c(i64::from(SENTINEL))),
    );
    body.push(rmw(AtomicOp::Add, 3, won, None));
    let params = vec![Param::buffer("cells", U32, BufferRole::ReadWrite, Some(1))];
    let mut k = KernelFn::new(
        "atomics",
        U32,
        params,
        vec![LocalDecl::scalar(U32); 3],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([BLOCK, 1, 1], [BLOCKS, 1, 1]));
    k
}
