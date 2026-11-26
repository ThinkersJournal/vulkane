// SPDX-License-Identifier: MIT OR Apache-2.0
//! IEEE half-precision constants. A kernel literal at `F16` is lowered only when it is exactly a
//! half; anything else would be a silent rounding the IR did not ask for, so it is declined.

/// The IEEE binary16 bits of `v` when `v` is exactly representable (zero and its sign, the two
/// infinities, normals and subnormals), else `None`. NaN is `None`: its payload cannot be carried
/// through an `f64` literal faithfully.
pub(crate) fn exact_bits(v: f64) -> Option<u16> {
    if v.is_nan() {
        return None;
    }
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return Some(sign);
    }
    if a.is_infinite() {
        return Some(sign | 0x7c00);
    }
    let bits = a.to_bits();
    let exp = i32::try_from((bits >> 52) & 0x7ff).expect("11 bits") - 1023;
    if (-14..=15).contains(&exp) {
        // Normal: the low 42 of the 52 fraction bits must be zero.
        if bits & ((1u64 << 42) - 1) != 0 {
            return None;
        }
        let frac = u16::try_from((bits >> 42) & 0x3ff).expect("10 bits");
        let biased = u16::try_from(exp + 15).expect("1..=30");
        return Some(sign | (biased << 10) | frac);
    }
    if (-24..-14).contains(&exp) {
        // Subnormal: `k * 2^-24` with `k` in 1..1024 (the scaling by a power of two is exact).
        let k = a * 16_777_216.0;
        if k.fract() != 0.0 {
            return None;
        }
        return Some(sign | k as u16);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::exact_bits;

    #[test]
    fn exact_halves_round_trip_and_inexact_values_are_refused() {
        assert_eq!(exact_bits(1.0), Some(0x3c00));
        assert_eq!(exact_bits(-1.0), Some(0xbc00));
        assert_eq!(exact_bits(0.0), Some(0));
        assert_eq!(exact_bits(-0.0), Some(0x8000));
        assert_eq!(exact_bits(65504.0), Some(0x7bff));
        assert_eq!(exact_bits(2f64.powi(-24)), Some(1));
        assert_eq!(exact_bits(2f64.powi(-14)), Some(0x0400));
        assert_eq!(exact_bits(f64::NEG_INFINITY), Some(0xfc00));
        // Controls: a value one bit too fine, too large, too small, and NaN.
        assert_eq!(exact_bits(0.1), None);
        assert_eq!(exact_bits(1.0 + 2f64.powi(-11)), None);
        assert_eq!(exact_bits(65536.0), None);
        assert_eq!(exact_bits(2f64.powi(-25)), None);
        assert_eq!(exact_bits(f64::NAN), None);
        // The ten-bit fraction boundary is exact.
        assert_eq!(exact_bits(1.0 + 2f64.powi(-10)), Some(0x3c01));
    }
}
