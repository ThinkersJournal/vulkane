// SPDX-License-Identifier: MIT OR Apache-2.0
//! A one-statement kernel for the "this statement is not lowered, and here is why" tests (ROADMAP row 9f): a `U32`
//! kernel of one `ReadWrite` buffer whose body is `stmt` alone, launched as one block of one thread.

use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Place, Stmt};
use unpopped_vocab::ElementKind::U32;

/// The kernel whose whole body is `stmt`.
pub fn kernel_with(stmt: Stmt) -> KernelFn {
    let mut k = KernelFn::new(
        "one_stmt",
        U32,
        vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(vec![stmt]),
    );
    k.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    k
}

/// The `NVIDIA-specific` statements that carry no operands, with the phrase their refusal must contain.
pub fn nvidia_unit_statements() -> Vec<(&'static str, Stmt, &'static str)> {
    vec![
        ("GridDepWait", Stmt::GridDepWait, "dependent launch"),
        (
            "GridDepLaunchDependents",
            Stmt::GridDepLaunchDependents,
            "dependent launch",
        ),
        ("WgmmaCommit", Stmt::WgmmaCommit, "wgmma"),
    ]
}

/// One-store kernels through the byte-addressed places `Place::Raw` and `Place::AllocRaw` (unpopped 0.57, ROADMAP U9c): no emitter here
/// lowers them, and each must refuse by naming the place rather than spell a wrong address. The kernels pass the kernel checker (a declared
/// allocation admits `AllocRaw`), so the refusal under test is the emitter's own. (Recovered 2026-11-27 from the `take-unpopped-0.57.0`
/// branch: main had no test of this refusal.)
#[must_use]
pub fn raw_place_kernels() -> Vec<(&'static str, KernelFn)> {
    let at = |place| {
        let mut k = kernel_with(Stmt::Store {
            place,
            value: ScalarExpr::ConstInt(1, U32),
        });
        k.allocs = vec![AllocDecl::new(
            "tile",
            MemorySpace::Shared,
            U32,
            Extent::Static(4),
        )];
        k
    };
    vec![
        (
            "Raw",
            at(Place::Raw {
                buffer: 0,
                byte_offset: ScalarExpr::ConstInt(0, U32),
                elem: U32,
            }),
        ),
        (
            "AllocRaw",
            at(Place::AllocRaw {
                alloc: 0,
                byte_offset: ScalarExpr::ConstInt(0, U32),
                elem: U32,
            }),
        ),
    ]
}
