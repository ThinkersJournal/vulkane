// SPDX-License-Identifier: MIT OR Apache-2.0
//! A one-statement kernel for the "this statement is not lowered, and here is why" tests (ROADMAP row 9f): a `U32`
//! kernel of one `ReadWrite` buffer whose body is `stmt` alone, launched as one block of one thread.

use unpopped::exec::LaunchShape;
use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Stmt};
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
