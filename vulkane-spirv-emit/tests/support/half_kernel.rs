// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `F16` compute-kernel oracle (ROADMAP 8e, half as a compute dtype), shared by the unit tests
//! and `tests/kernel_half_live.rs`. The kernels are `float_kernel(F16, case)`; the host oracle works
//! on the exact `f64` values of the half inputs and rounds once to half after each IEEE operation
//! (`+`, `-`, `*` of two halves are exact in `f64`, so that is the correctly rounded half result).
//! Each case also names a mutant (a fused multiply-add, a NaN-ignoring max, a rounding floor) that must differ from
//! the expected output somewhere, or the case could not catch it.

use crate::cast_kernel::{f16_decode, f16_round};
use crate::float_kernel::Case;

pub(crate) const THREADS: usize = 256;

/// `(a, b)` as `f64`; each is rounded to half for the input buffers (all are exact halves or NaN).
pub(crate) const PAIRS: [(f64, f64); 14] = [
    (0.1, 3.0),
    (65504.0, 2.0),
    // 2^-24 * 0.5 ties to zero, then + a = the smallest subnormal (flushed to 0 by FTZ).
    (5.960_464_477_539_063e-8, 0.5),
    (f64::NAN, 1.0),
    (f64::INFINITY, 0.0),
    (-0.0, 1.0),
    (0.0, -1.0),
    (f64::INFINITY, -1.0),
    (1.0, f64::NAN),
    (-0.5, 2.0),
    (1.5, -2.5),
    // (1 + 2^-10)^2 + (1 + 2^-10): a fused multiply-add rounds differently.
    (1.000_976_562_5, 1.000_976_562_5),
    (6.103_515_625e-5, 0.5),
    (-0.0, 0.0),
];

fn val(bits: u16) -> f64 {
    f64::from(f16_decode(bits))
}

/// The expected half bits of `case` for one input pair, or (`mutant`) what a wrong lowering gives:
/// a fused multiply-add, a max/min that ignores NaN, a floor that rounds to nearest.
fn one(case: Case, a: u16, b: u16, mutant: bool) -> u16 {
    let (x, y) = (val(a), val(b));
    let round = |v: f64| f16_round(v, false);
    match case {
        Case::MulAdd if mutant => round(x * y + x),
        Case::MulAdd => round(val(round(x * y)) + x),
        Case::Floor if mutant => round(x.round_ties_even()),
        Case::Floor => round(x.floor()),
        Case::Max | Case::Min => {
            let is_max = matches!(case, Case::Max);
            if x.is_nan() {
                if mutant { b } else { a }
            } else if y.is_nan() {
                if mutant { a } else { b }
            } else if (is_max && x >= y) || (!is_max && x <= y) {
                a
            } else {
                b
            }
        }
        _ => unreachable!("a declined case has no oracle"),
    }
}

fn pack(halves: &[u16]) -> Vec<u32> {
    halves
        .chunks(2)
        .map(|c| u32::from(c[0]) | (u32::from(c[1]) << 16))
        .collect()
}

/// `(a words, b words, expected words, mutant words)` for `case`: the case's mutant.
pub(crate) fn data(case: Case) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut a, mut b, mut want, mut mutant) = (vec![], vec![], vec![], vec![]);
    for k in 0..THREADS {
        let (x, y) = PAIRS[k % PAIRS.len()];
        let (ha, hb) = (f16_round(x, false), f16_round(y, false));
        a.push(ha);
        b.push(hb);
        want.push(one(case, ha, hb, false));
        mutant.push(one(case, ha, hb, true));
    }
    (pack(&a), pack(&b), pack(&want), pack(&mutant))
}

/// Whether two half word streams agree: equal bits, or both NaN (a NaN's payload is unspecified).
pub(crate) fn same(got: &[u32], want: &[u32]) -> Result<(), String> {
    let nan = |h: u16| h & 0x7c00 == 0x7c00 && h & 0x03ff != 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        for half in 0..2 {
            let (gh, wh) = ((g >> (16 * half)) as u16, (w >> (16 * half)) as u16);
            if gh != wh && !(nan(gh) && nan(wh)) {
                return Err(format!(
                    "element {}: got {gh:#06x}, want {wh:#06x}",
                    2 * i + half
                ));
            }
        }
    }
    Ok(())
}
