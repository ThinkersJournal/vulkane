// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Bf16` compute-kernel oracle (ROADMAP 8p), the bfloat16 twin of [`crate::half_kernel`]. The kernels are
//! `float_kernel(Bf16, case)`; the host oracle works on the exact `f64` value of each bfloat16 input and rounds ONCE to
//! bfloat16 (round to nearest even) after each IEEE operation. `+`, `-` and `*` of two bfloat16 values are exact in `f64`
//! (8 significant bits each), so that is the correctly rounded bfloat16 result. Each case also names a mutant (a fused
//! multiply-add, a NaN-ignoring max, a rounding floor) that must differ from the expected output somewhere, or the case
//! could not catch it.

use crate::float_kernel::Case;

/// Threads launched.
pub const THREADS: usize = 256;

/// `(a, b)` as `f64`; each is rounded to bfloat16 for the input buffers.
pub const PAIRS: [(f64, f64); 14] = [
    (0.1, 3.0),
    // 3e38 * 2 is past the largest bfloat16 (about 3.39e38): the product rounds to infinity.
    (3.0e38, 2.0),
    (3.0e-38, 0.5),
    (f64::NAN, 1.0),
    (f64::INFINITY, 0.0),
    (-0.0, 1.0),
    (0.0, -1.0),
    (f64::INFINITY, -1.0),
    (1.0, f64::NAN),
    (-0.5, 2.0),
    (1.5, -2.5),
    // (1 + 2^-7) * (1 + 3 * 2^-7) + (1 + 2^-7): rounded after the product it is 2.03125, fused it is 2.046875.
    (1.007_812_5, 1.023_437_5),
    (1.2e-38, 1.0),
    (-0.0, 0.0),
];

/// Pairs whose operands or intermediate product are SUBNORMAL bfloat16 (below 2^-126): `MSL 8.1` lets a device flush these to
/// zero, so they are a separate case ([`data_subnormal`]) that accepts the exact result or the flushed one.
/// `1.1e-40` is 2^-133 (the smallest subnormal): times 0.5 it ties to zero, and zero plus it is itself. `1.2e-38` is normal but
/// half of it is subnormal.
pub const SUBNORMAL_PAIRS: [(f64, f64); 2] = [(1.1e-40, 0.5), (1.2e-38, 0.5)];

/// The bfloat16 bits nearest to `v` (ties to even); every NaN is the canonical quiet NaN.
pub fn bf16_round(v: f64) -> u16 {
    if v.is_nan() {
        return 0x7fc0;
    }
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() {
        return sign | 0x7f80;
    }
    // The spacing of bfloat16 values near `a`: 2^(e-7) for a normal value, 2^-133 below 2^-126 (the subnormals).
    let e = (((a.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-126);
    let quantum = 2f64.powi(e - 7);
    // `a / quantum` is exact (a power of two); `round_ties_even` is the one rounding.
    let r = (a / quantum).round_ties_even() * quantum;
    if r >= 2f64.powi(128) {
        return sign | 0x7f80;
    }
    // `r` is exactly a bfloat16, hence exactly an f32: its top 16 bits are the encoding.
    sign | ((r as f32).to_bits() >> 16) as u16
}

/// The exact value of the bfloat16 `bits`.
pub fn bf16_decode(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

fn val(bits: u16) -> f64 {
    bf16_decode(bits)
}

/// The expected bfloat16 bits of `case` for one input pair, or (`mutant`) what a wrong lowering gives: a fused
/// multiply-add, a max/min that ignores NaN, a floor that rounds to nearest.
fn one(case: Case, a: u16, b: u16, mutant: bool) -> u16 {
    let (x, y) = (val(a), val(b));
    match case {
        Case::MulAdd if mutant => bf16_round(x * y + x),
        Case::MulAdd => bf16_round(val(bf16_round(x * y)) + x),
        Case::Floor if mutant => bf16_round(x.round_ties_even()),
        Case::Floor => bf16_round(x.floor()),
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
pub fn data(case: Case) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    data_over(case, &PAIRS)
}

fn data_over(case: Case, pairs: &[(f64, f64)]) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut a, mut b, mut want, mut mutant) = (vec![], vec![], vec![], vec![]);
    for k in 0..THREADS {
        let (x, y) = pairs[k % pairs.len()];
        let (ha, hb) = (bf16_round(x), bf16_round(y));
        a.push(ha);
        b.push(hb);
        want.push(one(case, ha, hb, false));
        mutant.push(one(case, ha, hb, true));
    }
    (pack(&a), pack(&b), pack(&want), pack(&mutant))
}

/// The flush-to-zero value of one bfloat16: a subnormal is zero (sign kept).
fn flush(h: u16) -> u16 {
    if h & 0x7f80 == 0 { h & 0x8000 } else { h }
}

/// `(a words, b words, exact words, flushed words)` for the multiply-add over [`SUBNORMAL_PAIRS`]: `exact` is IEEE with
/// subnormals, `flushed` is what a device that flushes subnormal inputs and results to zero (MSL 8.1) computes.
pub fn data_subnormal() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let (a, b, exact, _) = data_over(Case::MulAdd, &SUBNORMAL_PAIRS);
    let unpack = |w: &[u32]| -> Vec<u16> {
        w.iter()
            .flat_map(|x| [*x as u16, (*x >> 16) as u16])
            .collect()
    };
    let (ha, hb) = (unpack(&a), unpack(&b));
    let flushed: Vec<u16> = ha
        .iter()
        .zip(&hb)
        .map(|(&x, &y)| {
            let (x, y) = (flush(x), flush(y));
            let p = flush(bf16_round(val(x) * val(y)));
            flush(bf16_round(val(p) + val(x)))
        })
        .collect();
    (a, b, exact, pack(&flushed))
}

/// Whether two bfloat16 word streams agree: equal bits, or both NaN (a NaN's payload is unspecified).
pub fn same(got: &[u32], want: &[u32]) -> Result<(), String> {
    let nan = |h: u16| h & 0x7f80 == 0x7f80 && h & 0x007f != 0;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_nearest_even_with_subnormals_and_overflow() {
        assert_eq!(bf16_round(1.0), 0x3f80);
        // 1 + 2^-8 is the tie between 1.0 (even) and 1 + 2^-7: it goes to 1.0; 1 + 3 * 2^-8 ties up to the even one.
        assert_eq!(bf16_round(1.0 + 2f64.powi(-8)), 0x3f80);
        assert_eq!(bf16_round(1.0 + 3.0 * 2f64.powi(-8)), 0x3f82);
        // The smallest subnormal is 2^-133; half of it ties to zero (even), 1.5 of it ties to 2 of it.
        assert_eq!(bf16_round(2f64.powi(-133)), 0x0001);
        assert_eq!(bf16_round(2f64.powi(-134)), 0x0000);
        assert_eq!(bf16_round(1.5 * 2f64.powi(-133)), 0x0002);
        // Past the largest value (0x7f7f, about 3.3895e38) plus half a spacing it is infinity.
        assert_eq!(bf16_round(3.3895e38), 0x7f7f);
        assert_eq!(bf16_round(3.4e38), 0x7f80);
        assert_eq!(bf16_round(-0.0), 0x8000);
        assert_eq!(bf16_decode(0x3f80), 1.0);
        for bits in [0x0001u16, 0x007f, 0x0080, 0x3f80, 0x7f7f, 0x8001, 0xff7f] {
            assert_eq!(
                bf16_round(bf16_decode(bits)),
                bits,
                "{bits:#06x} round-trips"
            );
        }
    }

    #[test]
    fn the_subnormal_case_tells_exact_from_flushed() {
        let (_, _, exact, flushed) = data_subnormal();
        assert!(same(&exact, &exact).is_ok() && same(&flushed, &flushed).is_ok());
        assert!(
            same(&exact, &flushed).is_err(),
            "a flushing device must be told from an exact one"
        );
        // The smallest subnormal times 0.5 plus itself: exact keeps it, flushed is zero.
        assert_eq!(exact[0] as u16, 0x0001);
        assert_eq!(flushed[0] as u16, 0x0000);
    }

    #[test]
    fn every_case_has_a_mutant_that_differs_and_matches_where_it_should() {
        for case in [Case::MulAdd, Case::Max, Case::Min, Case::Floor] {
            let (_, _, want, mutant) = data(case);
            assert!(same(&want, &want).is_ok());
            assert!(
                same(&mutant, &want).is_err(),
                "{case:?}: the mutant is indistinguishable"
            );
        }
    }
}
