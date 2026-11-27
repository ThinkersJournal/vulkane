// SPDX-License-Identifier: MIT OR Apache-2.0
//! The board 506 A subgroup-collective device kernels and their oracle (ROADMAP 8t).
//!
//! One kernel per ruled `CollectiveKind`, all of the same shape: one block of [`BLOCK`] `U32` threads, a launch that states
//! a subgroup size of [`SUBGROUP`], `out[t] = collective(in[t])`. The expected values come from Unpopped's own CPU oracle
//! (`run_launch_with_subgroup`), the reference the IR defines these statements by, NOT from a second implementation written
//! here: a device that disagrees with it disagrees with the IR. [`input`] is shaped so that no two kinds agree on it (each
//! subgroup holds distinct values; the votes see an all-true, a mixed, an all-false and a mixed subgroup), which
//! [`kinds`]'s own test pins.
//!
//! Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call; "outside as read" is a reading of public documents, not a
//! clearance; no attorney; findings, not advice.

use unpopped::TypedBuffer;
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr as E};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind,
};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::ElementKind::U32;

/// Threads per workgroup (four subgroups of [`SUBGROUP`]).
pub const BLOCK: u32 = 128;
/// The subgroup size the launch states (min and max): the device must run this width for a case to apply.
pub const SUBGROUP: u32 = 32;

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

fn tid() -> E {
    E::Exec(ExecId::ThreadIdx(0))
}

/// Every ruled kind, by a stable case name.
#[must_use]
pub fn kinds() -> Vec<(&'static str, CollectiveKind)> {
    use CollectiveKind as C;
    vec![
        ("reduce_sum", C::Reduce(ReduceOp::Sum)),
        ("reduce_prod", C::Reduce(ReduceOp::Prod)),
        ("reduce_max", C::Reduce(ReduceOp::Max)),
        ("reduce_min", C::Reduce(ReduceOp::Min)),
        ("reduce_and", C::Reduce(ReduceOp::And)),
        ("reduce_or", C::Reduce(ReduceOp::Or)),
        ("reduce_xor", C::Reduce(ReduceOp::Xor)),
        ("vote_all", C::Vote(VoteKind::All)),
        ("vote_any", C::Vote(VoteKind::Any)),
        ("broadcast_3", C::Broadcast { lane: c(3) }),
        (
            "shuffle_idx",
            C::Shuffle {
                mode: ShuffleMode::Idx,
                operand: E::Mul(Box::new(tid()), Box::new(c(3))),
            },
        ),
        (
            "shuffle_up_1",
            C::Shuffle {
                mode: ShuffleMode::Up,
                operand: c(1),
            },
        ),
        (
            "shuffle_down_2",
            C::Shuffle {
                mode: ShuffleMode::Down,
                operand: c(2),
            },
        ),
        (
            "shuffle_xor_5",
            C::Shuffle {
                mode: ShuffleMode::Xor,
                operand: c(5),
            },
        ),
    ]
}

/// `out[t] = kind(in[t])` over [`BLOCK`] threads, the launch stating a subgroup size of [`SUBGROUP`].
#[must_use]
pub fn kernel(kind: CollectiveKind) -> KernelFn {
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![tid()],
    };
    let k = KernelFn::new(
        "collective",
        U32,
        vec![
            Param::buffer("a", U32, BufferRole::Read, Some(1)),
            Param::buffer("out", U32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(U32), LocalDecl::scalar(U32)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: buf(0),
            },
            Stmt::Collective {
                kind,
                value: E::Local(0),
                result: 1,
                scope: Scope::Subgroup,
            },
            Stmt::Store {
                place: buf(1),
                value: E::Local(1),
            },
        ]),
    );
    k.with_launch(LaunchShape::new([BLOCK, 1, 1], [1, 1, 1]).with_subgroup(SUBGROUP, SUBGROUP))
}

/// The input words. Subgroup 0 is all non-zero, 1 holds one zero, 2 is all zero, 3 holds one non-zero; the non-zero values
/// differ lane by lane, always carry bits 0 and 1 (so an AND is not a vote), and are not powers of two so every reduction and shuffle is told apart.
#[must_use]
pub fn input() -> Vec<u32> {
    (0..BLOCK)
        .map(|t| {
            let (sg, lane) = (t / SUBGROUP, t % SUBGROUP);
            let v = t.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9) | 3;
            match sg {
                0 => v,
                1 if lane == 5 => 0,
                1 => v,
                2 => 0,
                _ if lane == 9 => v,
                _ => 0,
            }
        })
        .collect()
}

/// What the IR says `out` holds for `kind` over [`input`], by Unpopped's CPU oracle at subgroup size [`SUBGROUP`].
///
/// # Errors
///
/// The oracle's trap text, e.g. when the kernel is not checker-valid.
pub fn expected(kind: &CollectiveKind) -> Result<Vec<u32>, String> {
    let k = kernel(kind.clone());
    let n = i64::from(BLOCK);
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_u32(&[n], &input())),
        KernelArg::Buffer(TypedBuffer::from_u32(&[n], &vec![0; BLOCK as usize])),
    ];
    run_launch_with_subgroup(&k, &mut args, Some(SUBGROUP)).map_err(|e| format!("{e:?}"))?;
    let KernelArg::Buffer(out) = &args[1] else {
        return Err("the output argument is not a buffer".to_string());
    };
    Ok(out
        .bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_is_checker_valid_and_has_an_oracle_result() {
        for (name, kind) in kinds() {
            unpopped::kernel::check_kernel(&kernel(kind.clone()))
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let out = expected(&kind).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(out.len(), BLOCK as usize, "{name}");
        }
    }

    #[test]
    fn the_oracle_tells_every_pair_of_kinds_apart_on_the_input() {
        // Non-vacuity: a device (or a mutant spelling) that computed another kind's function would match the wrong column.
        let all: Vec<(&str, Vec<u32>)> = kinds()
            .into_iter()
            .map(|(n, k)| (n, expected(&k).expect(n)))
            .collect();
        for (i, (a, ra)) in all.iter().enumerate() {
            for (b, rb) in &all[i + 1..] {
                assert_ne!(ra, rb, "{a} and {b} must differ on the input");
            }
        }
    }

    #[test]
    fn the_votes_see_an_all_true_a_mixed_an_all_false_and_a_mixed_subgroup() {
        let all = expected(&CollectiveKind::Vote(VoteKind::All)).unwrap();
        let any = expected(&CollectiveKind::Vote(VoteKind::Any)).unwrap();
        let per_sg =
            |v: &[u32]| -> Vec<u32> { (0..4).map(|s| v[(s * SUBGROUP) as usize]).collect() };
        assert_eq!(per_sg(&all), vec![1, 0, 0, 0]);
        assert_eq!(per_sg(&any), vec![1, 1, 0, 1]);
    }
}
