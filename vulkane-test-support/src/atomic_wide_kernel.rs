// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 64-bit atomic test kernel (ROADMAP 8d), shared by the unit tests and
//! `tests/kernel_wide_atomic_live.rs`. 16 blocks of 64 threads update a four-cell 64-bit `ReadWrite`
//! buffer pre-filled with the harness fill `0xAAAA_AAAA` in every 32-bit word, so a cell starts at
//! `SENTINEL` (read as `I64`, a negative number); every update crosses the 32-bit boundary:
//!
//! - cell 0: `Add (2^32 + 1)` from every thread: `SENTINEL + 1024 * (2^32 + 1)` only if no update is
//!   lost AND the add is 64 bits wide (a 32-bit atomic loses the high halves);
//! - cell 1: `Max (gid + 2^40)`: ends at `1023 + 2^40`;
//! - cell 2: `AtomicCas(expected = SENTINEL, desired = gid + 1 + 2^33)`, which exactly one thread
//!   wins; its `old` local feeds cell 3;
//! - cell 3: `Add (old == SENTINEL)`, the count of CAS winners: `SENTINEL + 1`.
//!
//! With `racy`, cell 0 is a non-atomic `Load`/`Add`/`Store`: the control.

use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::BinaryOp::CmpEq;
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr as E};
use unpopped::kernel::{
    AtomicOp, Block, BufferRole, KernelFn, LocalDecl, MemoryOrder, Param, Place, Stmt,
};
use unpopped_vocab::ElementKind;

/// Threads per workgroup.
pub const BLOCK: u32 = 64;
/// Workgroups launched.
pub const BLOCKS: u32 = 16;
/// Threads launched in total.
pub const THREADS: u32 = BLOCK * BLOCKS;
/// The 64-bit value the harness fill leaves in a cell, as `I64`.
pub const SENTINEL: u64 = 0xAAAA_AAAA_AAAA_AAAA;
/// One above the 32-bit range: each add is `1 << 32`, so a lost 64-bit carry shows.
pub const B: i64 = 1 << 32;

fn add(a: E, b: E) -> E {
    E::Add(Box::new(a), Box::new(b))
}

/// The 64-bit atomic-add kernel over `ty`; `racy` swaps the atomic for plain memory traffic (the control).
pub fn atomic_wide_kernel(ty: ElementKind, racy: bool) -> KernelFn {
    let c = |v: i64| E::ConstInt(v, ty);
    let cell = |i: i64| Place::Logical {
        buffer: 0,
        coords: vec![c(i)],
    };
    let rmw = |op, i, value, old| Stmt::AtomicRmw {
        op,
        place: cell(i),
        value,
        old,
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    };
    // The sentinel as the dtype's constant: `ConstInt` takes an `i64`, so only `I64` can name the
    // harness fill (negative). `U64` gets a value in its range so the kernel still lowers; the live
    // test runs `I64`.
    let sentinel = if ty == ElementKind::U64 {
        (SENTINEL >> 1) as i64
    } else {
        SENTINEL as i64
    };
    let cast = |e: E| {
        E::Cast(
            Box::new(e),
            CastSpec {
                to: ty,
                round: RoundMode::NearestEven,
                sat: Saturation::Wrap,
            },
        )
    };
    let gid = add(
        E::Mul(
            Box::new(cast(E::Exec(ExecId::BlockIdx(0)))),
            Box::new(c(i64::from(BLOCK))),
        ),
        cast(E::Exec(ExecId::ThreadIdx(0))),
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
            value: add(E::Local(1), c(B + 1)),
        });
    } else {
        body.push(rmw(AtomicOp::Add, 0, c(B + 1), None));
    }
    body.push(rmw(AtomicOp::Max, 1, add(E::Local(0), c(1 << 40)), None));
    body.push(Stmt::AtomicCas {
        place: cell(2),
        expected: c(sentinel),
        desired: add(add(E::Local(0), c(1)), c(1 << 33)),
        old: Some(2),
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    });
    let won = E::Binary(CmpEq, Box::new(E::Local(2)), Box::new(c(sentinel)));
    body.push(rmw(AtomicOp::Add, 3, won, None));
    let mut k = KernelFn::new(
        "atomics64",
        ty,
        vec![Param::buffer("cells", ty, BufferRole::ReadWrite, Some(1))],
        vec![LocalDecl::scalar(ty); 3],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([BLOCK, 1, 1], [BLOCKS, 1, 1]));
    k
}
