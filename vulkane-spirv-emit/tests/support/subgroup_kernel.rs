// SPDX-License-Identifier: MIT OR Apache-2.0
//! The row 8c ExecId test kernel: one block of 128 threads (two subgroups on a device of size 64); thread `t` stores
//! `lane + 256 * subgroup_id + 65536 * subgroup_size` at cell `t`, the three subgroup ids packed so a
//! swapped or wrong id changes the value. The launch names a subgroup size range, which
//! `check_kernel` requires of a kernel reading these ids.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::U32;

pub(crate) const BLOCK: u32 = 128;

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

fn mul(a: E, k: i64) -> E {
    E::Mul(Box::new(a), Box::new(c(k)))
}

/// The kernel for a launch requiring a subgroup size in `min..=max`.
pub(crate) fn subgroup_kernel(min: u32, max: u32) -> KernelFn {
    let packed = E::Add(
        Box::new(E::Add(
            Box::new(E::Exec(ExecId::LaneId)),
            Box::new(mul(E::Exec(ExecId::SubgroupId), 256)),
        )),
        Box::new(mul(E::Exec(ExecId::SubgroupSize), 65536)),
    );
    let body = vec![
        Stmt::Let {
            local: 0,
            init: E::Exec(ExecId::ThreadIdx(0)),
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![E::Local(0)],
            },
            value: packed,
        },
    ];
    let params = vec![Param::buffer("cells", U32, BufferRole::ReadWrite, Some(1))];
    let mut k = KernelFn::new(
        "subgroup_ids",
        U32,
        params,
        vec![LocalDecl::scalar(U32)],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([BLOCK, 1, 1], [1, 1, 1]).with_subgroup(min, max));
    k
}

#[allow(dead_code, reason = "the WGSL test only needs the kernel")]
/// What thread `t` must store when the device runs subgroups of `size` lanes.
pub(crate) fn expected(t: u32, size: u32) -> u32 {
    t % size + 256 * (t / size) + 65536 * size
}
