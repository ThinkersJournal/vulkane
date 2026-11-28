// SPDX-License-Identifier: MIT OR Apache-2.0
//! A one-statement kernel for the "this statement is not lowered, and here is why" tests (ROADMAP row 9f): a `U32`
//! kernel of one `ReadWrite` buffer whose body is `stmt` alone, launched as one block of one thread.

use unpopped::exec::ExecId;
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace};
use unpopped::ir::{BinaryOp, ScalarExpr};
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

/// `Stmt::Assume` (unpopped 0.64, U50) is a pure value fact that every emitter OMITS: undefined behaviour on a device when false, so emitting
/// nothing cannot change defined behaviour. The pairs are `(label, without, with)`: the same kernel with and without the statement(s), each
/// `with` kernel checker-valid. The test of an emitter is byte-equal emission of the two, plus a control that two different kernels do NOT
/// emit equal text (so the equality is not vacuous).
#[must_use]
pub fn assume_pairs() -> Vec<(&'static str, KernelFn, KernelFn)> {
    let store = |v: i64| Stmt::Store {
        place: Place::Logical {
            buffer: 0,
            coords: vec![ScalarExpr::ConstInt(0, U32)],
        },
        value: ScalarExpr::ConstInt(v, U32),
    };
    // a thread index below the launch's 32 (the unpopped fixture's fact): true on every thread
    let fact = || Stmt::Assume {
        cond: ScalarExpr::Binary(
            BinaryOp::CmpLt,
            Box::new(ScalarExpr::Exec(ExecId::ThreadIdx(0))),
            Box::new(ScalarExpr::ConstInt(32, U32)),
        ),
    };
    let kernel = |body: Vec<Stmt>| {
        let mut k = KernelFn::new(
            "assume_probe",
            U32,
            vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
            vec![],
            Block(body),
        );
        k.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
        k
    };
    vec![
        (
            "first",
            kernel(vec![store(7)]),
            kernel(vec![fact(), store(7)]),
        ),
        (
            "between and last",
            kernel(vec![store(1), store(2)]),
            kernel(vec![store(1), fact(), store(2), fact()]),
        ),
        ("alone", kernel(vec![]), kernel(vec![fact()])),
    ]
}
