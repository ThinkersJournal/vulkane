// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Cast` for kernel bodies: integer to integer, integer to float, float to float and float to
//! integer, against `unpopped`'s reference (`cast_ref.rs`, read at `origin/main` 08f6780d):
//!
//! - **int to int**: `Wrap` truncates two's complement; zero- or sign-extends by the source. Only
//!   `Wrap` is spelled; a clamping integer cast is declined.
//! - **int to float**: rounded per `RoundMode`. Only `NearestEven` is spelled; the module then
//!   declares the `RoundingModeRTE` execution mode at the result's width, so the device must round
//!   to nearest even (Vulkan's table calls conversions "correctly rounded"; the mode removes any
//!   doubt about which rounding). Ties (`2^24 + 1`) are tested on a device.
//! - **integer to and from half**: integer to half is `NearestEven` through `f32` (see `int_to_float`),
//!   half to integer is the float-to-integer rule below with the half's own bounds.
//! - **float to float**: widening (`F32` to `F64`) is exact. Narrowing (`F64` to `F32`) is
//!   `NearestEven` with `Wrap` (overflow to infinity, NaN stays NaN) under `RoundingModeRTE`;
//!   `ClampFinite` and the other rounding modes are declined.
//! - **float to int**: the oracle rounds per `RoundMode`, maps NaN to 0 and clamps to the
//!   destination range (the IR demands `Saturation::Clamp`). `OpConvertFToS`/`OpConvertFToU` are
//!   undefined out of range, so the cast is composed: round with an exact GLSL.std.450 op
//!   (`RoundEven`, `Trunc`, `Floor`, `Ceil`), keep the converted value only where it is in range,
//!   and `Select` the clamp bound or `0` otherwise. `NearestAway` is declined (`Round` leaves ties
//!   to the implementation) and so is a 64-bit destination (not yet spelled).

use rspirv::spirv::{Capability, Word};
use unpopped::backend::{Decline, DeclinedOp, LowerError, Spelling};
use unpopped::ir::{CastSpec, RoundMode, Saturation};
use unpopped_vocab::ElementKind;

use crate::family::Family;
use crate::ops::{Em, lo};

fn declined(why: String) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op: DeclinedOp::Cast,
        why,
    }))
}

fn spelled(w: Word) -> Result<Spelling, LowerError> {
    Ok(Spelling::Spelled(w.to_string()))
}

/// The `[lo, hi + 1)` range of an integer destination of at most 32 bits, as exact floats.
fn int_range(dst: Family) -> (f64, f64) {
    let bits = dst.bits();
    if dst.is_signed() {
        (-(2f64.powi(bits as i32 - 1)), 2f64.powi(bits as i32 - 1))
    } else {
        (0.0, 2f64.powi(bits as i32))
    }
}

impl Em {
    fn int_type(&mut self, f: Family) -> Word {
        self.b.type_int(f.bits(), u32::from(f.is_signed()))
    }

    fn float_type(&mut self, f: Family) -> Word {
        self.b.type_float(f.bits(), None)
    }

    fn float_const(&mut self, f: Family, v: f64) -> Word {
        let ty = self.float_type(f);
        if f == Family::F64 {
            self.b.constant_bit64(ty, v.to_bits())
        } else if f == Family::F16 {
            let bits = crate::half::exact_bits(v).expect("a half bound the cast makes is exact");
            self.b.constant_bit32(ty, u32::from(bits))
        } else {
            self.b.constant_bit32(ty, (v as f32).to_bits())
        }
    }

    /// `x` (of type `from`) converted per `spec`, or a decline naming why not.
    pub(crate) fn cast(
        &mut self,
        x: Word,
        from: ElementKind,
        spec: CastSpec,
    ) -> Result<Spelling, LowerError> {
        let (Some(src), Some(dst)) = (Family::of(from), Family::of(spec.to)) else {
            return declined(format!("a Cast {from:?} -> {:?} is not lowered", spec.to));
        };
        if src == dst {
            return spelled(x);
        }
        match (src.is_float(), dst.is_float()) {
            (false, false) => self.int_to_int(x, src, dst, spec),
            (false, true) => self.int_to_float(x, src, dst, spec),
            (true, true) => self.float_to_float(x, src, dst, spec),
            (true, false) => self.float_to_int(x, src, dst, spec),
        }
    }

    fn int_to_int(
        &mut self,
        x: Word,
        src: Family,
        dst: Family,
        spec: CastSpec,
    ) -> Result<Spelling, LowerError> {
        if spec.sat != Saturation::Wrap {
            return declined(format!(
                "an integer Cast with {:?} is not lowered: only a wrapping one is",
                spec.sat
            ));
        }
        let t = self.int_type(dst);
        if src.bits() == dst.bits() {
            return spelled(lo(self.b.bitcast(t, None, x))?);
        }
        // SPIR-V types the result of `OpSConvert` signed and of `OpUConvert` unsigned: sign-extend a
        // signed source, otherwise zero-extend or truncate, then bitcast to the destination's own
        // signedness.
        let signed_op = src.bits() < dst.bits() && src.is_signed();
        let mid = self.b.type_int(dst.bits(), u32::from(signed_op));
        let w = if signed_op {
            lo(self.b.s_convert(mid, None, x))?
        } else {
            lo(self.b.u_convert(mid, None, x))?
        };
        if signed_op == dst.is_signed() {
            spelled(w)
        } else {
            spelled(lo(self.b.bitcast(t, None, w))?)
        }
    }

    fn int_to_float(
        &mut self,
        x: Word,
        src: Family,
        dst: Family,
        spec: CastSpec,
    ) -> Result<Spelling, LowerError> {
        if spec.round != RoundMode::NearestEven || spec.sat != Saturation::Wrap {
            return declined(format!(
                "an int to float Cast with {:?}/{:?} is not lowered: only NearestEven with Wrap is (the module declares RoundingModeRTE)",
                spec.round, spec.sat
            ));
        }
        let t = self.float_type(dst);
        self.need_rte(dst);
        if dst.is_half() {
            // Through `f32`, then a narrowing `OpFConvert`: exact, because an integer below 2^24 is
            // exact in `f32` (one rounding, to half), and one at or above 65520 is infinity in half
            // whichever way `f32` rounded it. Measured on the harness device: a direct
            // `OpConvertSToF` to half rounds toward zero (2051 gives 2050, not 2052) although
            // `RoundingModeRTE` at 16 bits is declared and the device reports it, while the
            // `OpFConvert` narrowing honours it.
            let f32_t = self.b.type_float(32, None);
            let mid = lo(if src.is_signed() {
                self.b.convert_s_to_f(f32_t, None, x)
            } else {
                self.b.convert_u_to_f(f32_t, None, x)
            })?;
            return spelled(lo(self.b.f_convert(t, None, mid))?);
        }
        spelled(lo(if src.is_signed() {
            self.b.convert_s_to_f(t, None, x)
        } else {
            self.b.convert_u_to_f(t, None, x)
        })?)
    }

    fn float_to_float(
        &mut self,
        x: Word,
        src: Family,
        dst: Family,
        spec: CastSpec,
    ) -> Result<Spelling, LowerError> {
        let t = self.float_type(dst);
        if src.bits() < dst.bits() {
            // Widening is exact: no rounding, no overflow.
            return spelled(lo(self.b.f_convert(t, None, x))?);
        }
        if spec.round != RoundMode::NearestEven || spec.sat != Saturation::Wrap {
            return declined(format!(
                "a narrowing float Cast with {:?}/{:?} is not lowered: only NearestEven with Wrap is (the module declares RoundingModeRTE)",
                spec.round, spec.sat
            ));
        }
        self.need_rte(dst);
        spelled(lo(self.b.f_convert(t, None, x))?)
    }

    fn float_to_int(
        &mut self,
        x: Word,
        src: Family,
        dst: Family,
        spec: CastSpec,
    ) -> Result<Spelling, LowerError> {
        use crate::ops::GLOp;
        if spec.sat != Saturation::Clamp || dst.bits() > 32 {
            return declined(format!(
                "a float to {:?} Cast with {:?} is not lowered: only Clamp into a destination of at most 32 bits is",
                spec.to, spec.sat
            ));
        }
        let round = match spec.round {
            RoundMode::NearestEven => GLOp::RoundEven,
            RoundMode::TowardZero => GLOp::Trunc,
            RoundMode::TowardNegInf => GLOp::Floor,
            RoundMode::TowardPosInf => GLOp::Ceil,
            other => {
                return declined(format!(
                    "a float to int Cast rounding {other:?} is not lowered (GLSL.std.450 `Round` leaves ties to the implementation)"
                ));
            }
        };
        let ft = self.float_type(src);
        let it = self.int_type(dst);
        let bl = self.boolean;
        let r = self.gl_on(ft, round, &[x])?;
        // A half tops out at 65504, below every 32-bit range end, so its bounds are the finite
        // extremes themselves (inclusive); only an infinity is out of range.
        let half = src.is_half();
        let (lo_f, hi_f) = if half {
            (if dst.is_signed() { -65504.0 } else { 0.0 }, 65504.0)
        } else {
            int_range(dst)
        };
        let (lo_c, hi_c) = (self.float_const(src, lo_f), self.float_const(src, hi_f));
        let zero_f = self.float_const(src, 0.0);
        let ge_lo = lo(self.b.f_ord_greater_than_equal(bl, None, r, lo_c))?;
        let lt_hi = lo(if half {
            self.b.f_ord_less_than_equal(bl, None, r, hi_c)
        } else {
            self.b.f_ord_less_than(bl, None, r, hi_c)
        })?;
        let in_range = lo(self.b.logical_and(bl, None, ge_lo, lt_hi))?;
        // Convert only an in-range value: out of range `OpConvertF*` is undefined.
        let safe = lo(self.b.select(ft, None, in_range, r, zero_f))?;
        let conv = lo(if dst.is_signed() {
            self.b.convert_f_to_s(it, None, safe)
        } else {
            self.b.convert_f_to_u(it, None, safe)
        })?;
        let (lo_i, hi_i) = if dst.is_signed() {
            (-(1i64 << (dst.bits() - 1)), (1i64 << (dst.bits() - 1)) - 1)
        } else {
            (0, (1i64 << dst.bits()) - 1)
        };
        let (lo_k, hi_k) = (
            self.b.constant_bit32(it, lo_i as u32),
            self.b.constant_bit32(it, hi_i as u32),
        );
        let zero_i = self.b.constant_bit32(it, 0);
        let ge_hi = lo(if half {
            self.b.f_ord_greater_than(bl, None, r, hi_c)
        } else {
            self.b.f_ord_greater_than_equal(bl, None, r, hi_c)
        })?;
        let lt_lo = lo(self.b.f_ord_less_than(bl, None, r, lo_c))?;
        let below = lo(self.b.select(it, None, lt_lo, lo_k, conv))?;
        let above = lo(self.b.select(it, None, ge_hi, hi_k, below))?;
        let nan = lo(self.b.is_nan(bl, None, x))?;
        spelled(lo(self.b.select(it, None, nan, zero_i, above))?)
    }

    /// Record that a conversion needs the device to round to nearest even at `dst`'s width.
    fn need_rte(&mut self, dst: Family) {
        self.rte[match dst.bits() {
            16 => 0,
            32 => 1,
            _ => 2,
        }] = true;
    }

    /// The capabilities a kernel that did rounding conversions needs, declared once at the end.
    pub(crate) fn declare_rte(&mut self, main: Word) {
        use rspirv::spirv::ExecutionMode;
        for (i, width) in [16u32, 32, 64].into_iter().enumerate() {
            if self.rte[i] {
                self.b.capability(Capability::RoundingModeRTE);
                self.b
                    .execution_mode(main, ExecutionMode::RoundingModeRTE, [width]);
            }
        }
    }
}
