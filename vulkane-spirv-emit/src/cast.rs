// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integer-to-integer `Cast` for kernel bodies. An execution id is a `U32` in the IR, so a kernel at
//! another integer dtype (the 64-bit ones) reaches its thread index through this cast; `I32`/`U32`
//! kernels can use it too. Only a wrapping cast to the kernel's own compute dtype is spelled; every
//! other pair (float, clamping, a different destination) declines by name, as it did before.

use rspirv::spirv::Word;
use unpopped::backend::{Decline, DeclinedOp, LowerError, Spelling};
use unpopped::ir::{CastSpec, Saturation};
use unpopped_vocab::ElementKind;

use crate::ops::{Em, Family, lo};

fn declined(why: String) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op: DeclinedOp::Cast,
        why,
    }))
}

impl Em {
    /// `x` (of type `from`) converted to the compute dtype: a bitcast between same-width
    /// signedness, an extension (sign- or zero- by the source's signedness) when widening, and a
    /// truncation when narrowing.
    pub(crate) fn int_cast(
        &mut self,
        x: Word,
        from: ElementKind,
        spec: CastSpec,
    ) -> Result<Spelling, LowerError> {
        let (Some(src), Some(dst)) = (Family::of(from), Family::of(spec.to)) else {
            return declined(format!("a Cast {from:?} -> {:?} is not lowered", spec.to));
        };
        if src.is_float() || dst.is_float() || spec.sat != Saturation::Wrap || spec.to != self.dtype
        {
            return declined(format!(
                "a Cast {from:?} -> {:?} ({:?}) is not lowered: only a wrapping integer cast to the compute dtype {:?} is",
                spec.to, spec.sat, self.dtype
            ));
        }
        let t = self.t;
        if src == dst {
            return Ok(Spelling::Spelled(x.to_string()));
        }
        let r = if src.bits() == dst.bits() {
            lo(self.b.bitcast(t, None, x))?
        } else {
            // SPIR-V types the result of `OpSConvert` signed and of `OpUConvert` unsigned: sign-extend
            // a signed source, otherwise zero-extend or truncate, then bitcast to the destination's
            // own signedness.
            let signed_op = src.bits() < dst.bits() && src.is_signed();
            let mid = self.b.type_int(dst.bits(), u32::from(signed_op));
            let w = if signed_op {
                lo(self.b.s_convert(mid, None, x))?
            } else {
                lo(self.b.u_convert(mid, None, x))?
            };
            if signed_op == dst.is_signed() {
                w
            } else {
                lo(self.b.bitcast(t, None, w))?
            }
        };
        Ok(Spelling::Spelled(r.to_string()))
    }
}
