// SPDX-License-Identifier: MIT OR Apache-2.0
//! The row 8f test kernel: every thread of a non-cubic 3-D launch writes its own ids, packed into
//! one word, to the cell its *global* position names. The cell is computed from `BlockDim` and
//! `GridDim` too, so a wrong dimension scatters the writes and fails the permutation check as
//! surely as a wrong index fails a value.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::U32;

/// Threads per block, per axis (24 threads: no axis equals another).
pub(crate) const BLOCK: [u32; 3] = [4, 3, 2];
/// Blocks per grid, per axis (30 blocks).
pub(crate) const GRID: [u32; 3] = [3, 2, 5];

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

fn mul(a: E, b: E) -> E {
    E::Mul(Box::new(a), Box::new(b))
}

fn add(a: E, b: E) -> E {
    E::Add(Box::new(a), Box::new(b))
}

fn local(i: u32) -> E {
    E::Local(i)
}

/// `block_idx * block_dim + thread_idx` on `axis`: the thread's global coordinate.
fn global(axis: u8) -> E {
    add(
        mul(
            E::Exec(ExecId::BlockIdx(axis)),
            E::Exec(ExecId::BlockDim(axis)),
        ),
        E::Exec(ExecId::ThreadIdx(axis)),
    )
}

/// The extent of the whole launch on `axis`, in threads.
fn extent(axis: u8) -> E {
    mul(
        E::Exec(ExecId::BlockDim(axis)),
        E::Exec(ExecId::GridDim(axis)),
    )
}

/// The six ids as nibbles of one word: thread xyz in bits 0..12, block xyz in bits 12..24.
fn packed() -> E {
    let ids = [
        ExecId::ThreadIdx(0),
        ExecId::ThreadIdx(1),
        ExecId::ThreadIdx(2),
        ExecId::BlockIdx(0),
        ExecId::BlockIdx(1),
        ExecId::BlockIdx(2),
    ];
    ids.iter().enumerate().fold(c(0), |acc, (k, id)| {
        add(acc, mul(E::Exec(*id), c(1 << (4 * k))))
    })
}

pub(crate) fn grid_kernel() -> KernelFn {
    let body = vec![
        Stmt::Let {
            local: 0,
            init: global(0),
        },
        Stmt::Let {
            local: 1,
            init: global(1),
        },
        Stmt::Let {
            local: 2,
            init: global(2),
        },
        Stmt::Let {
            local: 3,
            init: add(
                mul(add(mul(local(2), extent(1)), local(1)), extent(0)),
                local(0),
            ),
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![local(3)],
            },
            value: packed(),
        },
    ];
    let params = vec![Param::buffer("output", U32, BufferRole::Write, Some(1))];
    let mut k = KernelFn::new(
        "grid3d",
        U32,
        params,
        vec![LocalDecl::scalar(U32); 4],
        Block(body),
    );
    k.launch = Some(LaunchShape::new(BLOCK, GRID));
    k
}

/// The word cell `(x, y, z)` of the launch must hold: the ids of the thread that owns it.
#[allow(dead_code)]
pub(crate) fn expected(x: u32, y: u32, z: u32) -> u32 {
    let (g, t) = ([x, y, z], BLOCK);
    let (mut w, mut shift) = (0, 0);
    for a in 0..3 {
        w |= (g[a] % t[a]) << shift;
        shift += 4;
    }
    for a in 0..3 {
        w |= (g[a] / t[a]) << shift;
        shift += 4;
    }
    w
}

/// Cells in the launch, and the dispatch shape that covers it.
#[allow(dead_code)]
pub(crate) fn cells() -> usize {
    (0..3).map(|a| (BLOCK[a] * GRID[a]) as usize).product()
}
