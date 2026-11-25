// SPDX-License-Identifier: MIT OR Apache-2.0
//! The row 8k test kernel, shared by the unit tests (`src/kernel_tests.rs`) and the live test
//! (`tests/kernel_private_live.rs`): each thread fills its own four-element `Private` array with
//! `t + 10*k`, then reads it at the *data-dependent* index `in[t]` (always below 4), so the result
//! is `t + 10 * in[t]`. A private array shared between threads would corrupt it.

use unpopped::exec::{AllocDecl, ExecId, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::U32;

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

/// `priv[k] = t + 10 * k`.
fn fill(k: i64) -> Stmt {
    Stmt::Store {
        place: Place::Alloc {
            alloc: 0,
            index: c(k),
        },
        value: E::Add(Box::new(E::Local(0)), Box::new(c(10 * k))),
    }
}

/// The kernel over `n` threads, with the private array in `space` (the negative test passes
/// `Constant`).
pub fn private_index(n: u32, space: MemorySpace) -> KernelFn {
    let mut body = vec![
        Stmt::Let {
            local: 0,
            init: E::Exec(ExecId::ThreadIdx(0)),
        },
        Stmt::Load {
            local: 1,
            place: Place::Logical {
                buffer: 0,
                coords: vec![E::Local(0)],
            },
        },
    ];
    body.extend((0..4).map(fill));
    body.push(Stmt::Load {
        local: 2,
        place: Place::Alloc {
            alloc: 0,
            index: E::Local(1),
        },
    });
    body.push(Stmt::Store {
        place: Place::Logical {
            buffer: 1,
            coords: vec![E::Local(0)],
        },
        value: E::Local(2),
    });
    let params = vec![
        Param::buffer("input", U32, BufferRole::Read, Some(1)),
        Param::buffer("output", U32, BufferRole::Write, Some(1)),
    ];
    let mut k = KernelFn::new(
        "private_index",
        U32,
        params,
        vec![LocalDecl::scalar(U32); 3],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([n, 1, 1], [1, 1, 1]));
    k.allocs = vec![AllocDecl::new("scratch", space, U32, Extent::Static(4))];
    k
}

/// The kernel with a read-only `Constant` allocation (its fills dropped, so `check_kernel` admits
/// it and the refusal under test is the emitter's own).
#[allow(
    dead_code,
    reason = "shared test module: each test binary compiles it whole but uses part of it"
)]
pub fn read_only_constant(n: u32) -> KernelFn {
    let mut k = private_index(n, MemorySpace::Constant);
    k.body.0.retain(|s| {
        !matches!(
            s,
            Stmt::Store {
                place: Place::Alloc { .. },
                ..
            }
        )
    });
    k
}
