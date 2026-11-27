// SPDX-License-Identifier: MIT OR Apache-2.0
//! The board 506 A float subgroup-reduction device kernels, their input and their comparator (ROADMAP 8t).
//!
//! `out[t] = reduce(in[t])` over one block of [`BLOCK`] `F32` threads (eight subgroups of [`SUBGROUP`]), the launch stating a subgroup size
//! of [`SUBGROUP`]. The expected value is Unpopped's CPU oracle (`run_launch_with_subgroup`). The input gives every subgroup a different
//! case: finite values; a NaN in the first, the last and a middle lane; all NaN; NaN with +inf; NaN with +inf and -inf; and signed zeros.
//!
//! **Comparators (the clauses read verbatim from KISS `spec/ops.md` at origin/main 8828b19):**
//! * `Max` and `Min` are exact-byte (KISS-OPS-6.0-0002) and MUST propagate NaN (KISS-OPS-6.11-0002): every lane of a subgroup that holds a
//!   NaN must be NaN (any NaN: the payload is not compared), and every other cell must equal the oracle's bits. **The +-0 cells are
//!   EXCLUDED from the byte compare** (the subgroup that holds signed zeros): KISS does not pin the +-0 bits of a float reduce(max|min)
//!   (6.15-0001 pins a tie only for binary minmax; a possible spec gap, not a ruling), so nothing is claimed about them.
//! * `Sum` is order-invariant/nondeterministic (KISS-OPS-6.0-0004), compared under a tolerance. **That clause, and 6.17-0005 and -0008,
//!   state no numeric bound, so the bound here is OURS, NOT KISS's:** `|device - oracle| <= GAMMA * sum(|x_i|)` with `GAMMA = (n-1) * u`,
//!   `n = 32`, `u = 2^-24` (the standard worst-case bound of recursive summation in f32, valid for any order of the additions). A NaN or an
//!   infinity must match in kind (NaN with NaN, +inf with +inf, -inf with -inf), not within a tolerance.
//!
//! Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call; "outside as read" is a reading of public documents, not a clearance; no
//! attorney; findings, not advice.

use unpopped::TypedBuffer;
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr as E};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, Stmt,
};
use unpopped::oracle::kernel_eval::KernelArg;
use unpopped::oracle::kernel_launch::run_launch_with_subgroup;
use unpopped_vocab::ElementKind::F32;

/// Threads per workgroup (eight subgroups).
pub const BLOCK: u32 = 256;
/// The subgroup size the launch states (min and max).
pub const SUBGROUP: u32 = 32;
/// Our (not KISS's) relative summation bound: `(n - 1) * 2^-24` for `n = 32`.
pub const GAMMA: f64 = 31.0 / 16_777_216.0;

fn tid() -> E {
    E::Exec(ExecId::ThreadIdx(0))
}

/// The three lowered kinds, by case name.
#[must_use]
pub fn kinds() -> Vec<(&'static str, CollectiveKind)> {
    vec![
        ("sum", CollectiveKind::Reduce(ReduceOp::Sum)),
        ("max", CollectiveKind::Reduce(ReduceOp::Max)),
        ("min", CollectiveKind::Reduce(ReduceOp::Min)),
    ]
}

/// `out[t] = kind(in[t])` over [`BLOCK`] threads.
#[must_use]
pub fn kernel(kind: CollectiveKind) -> KernelFn {
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![tid()],
    };
    let k = KernelFn::new(
        "fcollective",
        F32,
        vec![
            Param::buffer("a", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32)],
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

/// What each subgroup of [`input`] holds, in order (the labels the device log prints).
pub const SUBGROUPS: [&str; 8] = [
    "finite values",
    "NaN in the first lane",
    "NaN in the last lane",
    "NaN in a middle lane",
    "all NaN",
    "NaN and +inf",
    "NaN, +inf and -inf",
    "signed zeros (+-0 cells excluded from the byte compare)",
];

/// The index of the subgroup that holds signed zeros (its Max and Min bytes are not compared).
pub const ZERO_SUBGROUP: usize = 7;

/// The input bits, [`BLOCK`] words.
#[must_use]
pub fn input() -> Vec<u32> {
    let nan = f32::NAN.to_bits();
    (0..BLOCK)
        .map(|t| {
            let (sg, lane) = ((t / SUBGROUP) as usize, t % SUBGROUP);
            // Distinct, mixed-sign, mixed-magnitude finite values.
            let finite = {
                let m = 1.0 + f32::from((t * 37 % 101) as u8) / 7.0;
                let s = if t % 3 == 0 { -1.0 } else { 1.0 };
                (s * m * 2.0f32.powi((t % 5) as i32 - 2)).to_bits()
            };
            match sg {
                0 => finite,
                1 if lane == 0 => nan,
                2 if lane == SUBGROUP - 1 => nan,
                3 if lane == 13 => nan,
                4 => nan,
                5 if lane == 4 => nan,
                5 if lane == 20 => f32::INFINITY.to_bits(),
                6 if lane == 4 => nan,
                6 if lane == 20 => f32::INFINITY.to_bits(),
                6 if lane == 21 => f32::NEG_INFINITY.to_bits(),
                7 => match lane % 4 {
                    0 => 0.0f32.to_bits(),
                    1 => (-0.0f32).to_bits(),
                    2 => 0.0f32.to_bits(),
                    _ => (-0.0f32).to_bits(),
                },
                _ => finite,
            }
        })
        .collect()
}

/// The oracle's `out` for `kind` over [`input`], at subgroup size [`SUBGROUP`].
///
/// # Errors
///
/// The oracle's trap text.
pub fn expected(kind: &CollectiveKind) -> Result<Vec<u32>, String> {
    let n = i64::from(BLOCK);
    let a: Vec<f32> = input().into_iter().map(f32::from_bits).collect();
    let mut args = vec![
        KernelArg::Buffer(TypedBuffer::from_f32(&[n], &a)),
        KernelArg::Buffer(TypedBuffer::from_f32(&[n], &vec![0.0; BLOCK as usize])),
    ];
    run_launch_with_subgroup(&kernel(kind.clone()), &mut args, Some(SUBGROUP))
        .map_err(|e| format!("{e:?}"))?;
    let KernelArg::Buffer(out) = &args[1] else {
        return Err("the output argument is not a buffer".to_string());
    };
    Ok(out
        .bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Compare the device's words with the oracle's under the comparators in the module docs. `Err` names the first wrong cell.
///
/// # Errors
///
/// The first cell that violates its comparator.
pub fn check(kind: &CollectiveKind, got: &[u32]) -> Result<(), String> {
    let want = expected(kind)?;
    if got.len() != want.len() {
        return Err(format!("{} cells, want {}", got.len(), want.len()));
    }
    let inp: Vec<f32> = input().into_iter().map(f32::from_bits).collect();
    let is_sum = matches!(kind, CollectiveKind::Reduce(ReduceOp::Sum));
    for t in 0..BLOCK as usize {
        let sg = t / SUBGROUP as usize;
        let (g, w) = (f32::from_bits(got[t]), f32::from_bits(want[t]));
        let where_ = format!("cell {t} (subgroup {sg}: {})", SUBGROUPS[sg]);
        if is_sum {
            if w.is_nan() || g.is_nan() {
                if !(w.is_nan() && g.is_nan()) {
                    return Err(format!("{where_}: NaN mismatch: got {g} want {w}"));
                }
            } else if w.is_infinite() || g.is_infinite() {
                if g != w {
                    return Err(format!("{where_}: infinity mismatch: got {g} want {w}"));
                }
            } else {
                let lo = sg * SUBGROUP as usize;
                let abs_sum: f64 = inp[lo..lo + SUBGROUP as usize]
                    .iter()
                    .map(|x| f64::from(x.abs()))
                    .sum();
                let err = (f64::from(g) - f64::from(w)).abs();
                if err > GAMMA * abs_sum {
                    return Err(format!(
                        "{where_}: got {g} want {w}: error {err:e} exceeds OUR bound {:e} (not a KISS bound)",
                        GAMMA * abs_sum
                    ));
                }
            }
        } else if sg == ZERO_SUBGROUP {
            // +-0 cells are not pinned by KISS for a float reduce(max|min): only that the result is a zero.
            if g != 0.0 {
                return Err(format!("{where_}: not a zero: got {g}"));
            }
        } else if w.is_nan() {
            if !g.is_nan() {
                return Err(format!(
                    "{where_}: a NaN lane must propagate (KISS-OPS-6.11-0002): got {g}"
                ));
            }
        } else if got[t] != want[t] {
            return Err(format!(
                "{where_}: got {:#010x} want {:#010x}",
                got[t], want[t]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_oracle_propagates_nan_and_the_kernels_are_checker_valid() {
        for (name, kind) in kinds() {
            unpopped::kernel::check_kernel(&kernel(kind.clone()))
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let want = expected(&kind).unwrap_or_else(|e| panic!("{name}: {e}"));
            // The oracle's own answer for every NaN-bearing subgroup is a NaN (the bar the device must meet).
            for sg in 1..=6 {
                assert!(
                    f32::from_bits(want[sg * SUBGROUP as usize]).is_nan(),
                    "{name}: the oracle must give NaN for subgroup {sg}"
                );
            }
            // Self-check: the oracle's own words pass the comparator.
            check(&kind, &want).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    #[test]
    fn the_comparator_rejects_a_device_that_ignores_nan() {
        // Vacuity: the measured behaviour of the bare `simd_max` (it IGNORES a NaN lane) must FAIL the comparator.
        let kind = CollectiveKind::Reduce(ReduceOp::Max);
        let mut got = expected(&kind).unwrap();
        // Replace subgroup 1's answer with the max of its non-NaN lanes (what an ignoring device returns).
        let lo = SUBGROUP as usize;
        let inp: Vec<f32> = input().into_iter().map(f32::from_bits).collect();
        let m = inp[lo..lo + SUBGROUP as usize]
            .iter()
            .copied()
            .filter(|x| !x.is_nan())
            .fold(f32::NEG_INFINITY, f32::max);
        for c in &mut got[lo..lo + SUBGROUP as usize] {
            *c = m.to_bits();
        }
        let e = check(&kind, &got).expect_err("an ignoring device must fail");
        assert!(e.contains("must propagate"), "{e}");
    }

    #[test]
    fn the_sum_comparator_has_a_bound_that_is_neither_zero_nor_infinite() {
        let kind = CollectiveKind::Reduce(ReduceOp::Sum);
        let want = expected(&kind).unwrap();
        // A tiny relative error inside the bound passes; a large one fails.
        let mut ok = want.clone();
        ok[0] = (f32::from_bits(want[0]) * (1.0 + 1.0e-7)).to_bits();
        check(&kind, &ok).expect("a 1e-7 relative error is inside the bound");
        let mut bad = want.clone();
        bad[0] = (f32::from_bits(want[0]) * 1.1).to_bits();
        assert!(check(&kind, &bad).is_err(), "a 10% error must fail");
    }
}
