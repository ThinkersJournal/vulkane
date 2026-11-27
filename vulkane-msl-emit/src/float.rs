// SPDX-License-Identifier: MIT OR Apache-2.0
//! The MSL spellings of an `F32` or `F16` kernel (ROADMAP row 8o): the float arm of the shared walker.
//!
//! Metal compiles with fast math on by default (spec 1.6: no NaNs, no INFs, no signed zeros, reassociation and
//! fused contraction allowed). The source therefore opens with `#pragma METAL fp math_mode(safe)`
//! and `#pragma METAL fp contract(off)` (Metal 3.2), and every float operation is also pinned in a
//! `const` temporary, as the GLSL arm pins it in a `precise` one. **Not compiled here:** CI has no
//! Metal compiler, so this text is pinned, not run (the oracle is a macOS self-hosted runner).
//!
//! **Hardware caveat, stated not hidden:** MSL 8.1 says "Denormalized single-precision, half-precision, or brain
//! floating-point numbers passed as input to or produced as the output of ... arithmetic operations may be flushed
//! to zero" and 8.2 says "Either round ties to even or round toward zero rounding mode may be supported", so no
//! source pragma makes a Metal float kernel bit-exact IEEE: a device oracle must treat subnormals and the last bit
//! as the device's, as the WGSL arm does.

use std::fmt::Write;

use unpopped::exec::{ExecId, Scope};
use unpopped::ir::ReduceOp;
use unpopped::ir::{BinaryOp, RoundMode, UnaryOp};
use unpopped::kernel::{BufferRole, CollectiveKind, MemoryOrder};
use unpopped_vocab::ElementKind;
use vulkane_kernel_text::{BIT_OPS, COMPARISONS, Dialect, Layout, SubgroupCollective, operator};

use crate::dialect::{Ids, barrier, fence};

/// The MSL dialect of an `F32`, `F16` or `Bf16` kernel; the integer one is [`crate::Msl`].
#[derive(Debug, Default)]
pub(crate) struct MslFloat {
    /// `true` for an `F16` kernel (`half`), `false` for `F32`.
    pub(crate) half: bool,
    /// `true` for a `Bf16` kernel (`bfloat`, Metal 3.1 and later).
    pub(crate) bf16: bool,
    /// The built-in ids the body read.
    pub(crate) ids: Ids,
}

fn elem(t: ElementKind) -> Option<(&'static str, &'static str)> {
    match t {
        ElementKind::F32 => Some(("float", "0.0f")),
        ElementKind::F16 => Some(("half", "0.0h")),
        ElementKind::Bf16 => Some(("bfloat", "static_cast<bfloat>(0.0f)")),
        ElementKind::I32 => Some(("int", "0")),
        ElementKind::U32 => Some(("uint", "0u")),
        _ => None,
    }
}

/// The exact bounds `[lo, hi)` of the 32-bit integer `to` as floats, and the integers it clamps to.
fn bounds(to: ElementKind) -> [&'static str; 4] {
    if to == ElementKind::I32 {
        [
            "-2147483648.0f",
            "2147483648.0f",
            "(-2147483647 - 1)",
            "2147483647",
        ]
    } else {
        ["0.0f", "4294967296.0f", "0u", "4294967295u"]
    }
}

/// A unary op on a `bfloat`. The math functions of MSL 6.6 take `half` and `float` only, so the exact ops go through
/// `float`: widening a bfloat is exact, `fabs`, `floor`, `ceil`, `trunc` and `rint` of it need no more bits than it
/// has (the result is an integer no larger than the input, or a power of two), so the narrowing back is exact.
fn bfloat_unary(op: UnaryOp, x: &str) -> Option<String> {
    let through_float = |f: &str| format!("static_cast<bfloat>({f}(static_cast<float>({x})))");
    Some(match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Sqr => format!("({x} * {x})"),
        UnaryOp::Abs => through_float("fabs"),
        UnaryOp::Floor => through_float("floor"),
        UnaryOp::Ceil => through_float("ceil"),
        UnaryOp::Trunc => through_float("trunc"),
        UnaryOp::Round => through_float("rint"),
        _ => return None,
    })
}

/// The MSL expression of one float subgroup reduction (board 506 A; pm-kernels ruling 2026-11-27: Sum first, then Max and Min with an
/// explicit NaN select), or the reason it is declined by name.
///
/// MEASURED on the hosted macOS paravirtual device, NOT Apple hardware, under `math_mode(safe)` (vulkane#189, run 45972534688);
/// fast-math was NOT measured and a kernel compiled with it may differ. `simd_sum` propagated NaN. `simd_max` and `simd_min`
/// IGNORED a NaN operand, and KISS-OPS-6.11-0002 requires a float reduce(max|min) to propagate it, so each is wrapped with a NaN
/// indicator via a Sum reduction because `simd_max`/`simd_min` ignore NaN:
/// `select(simd_max(v), NAN, simd_sum(isnan(v) ? 1.0f : 0.0f) > 0.0f)` (`select` is a function, so both operands are evaluated and the
/// collectives stay under uniform control). The indicator count is exact while the lane count is below 2^24 (a float holds every
/// integer up to 2^24); the hosted device reports a SIMD-group width of 32, and the MSL text read states no maximum width, so the bound is
/// the float's, not the specification's. The +-0 bits of a float max or min are NOT pinned by KISS (unpinned; kiss options note pending): nothing is
/// claimed about them. A float sum's order is unpinned too (KISS-OPS-6.0-0004: class order-invariant/nondeterministic, compared
/// under a tolerance; the clause states no numeric bound). `Prod`, `Mean`, shuffles, votes, broadcasts, half and bfloat stay declined
/// (unmeasured or outside the ruling). Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call; "outside as read" is a reading of
/// public documents, not a clearance; no attorney; findings, not advice.
fn spell_float_collective(c: &SubgroupCollective<'_>) -> Result<String, String> {
    let v = &c.value;
    match c.kind {
        CollectiveKind::Reduce(ReduceOp::Sum) => Ok(format!("simd_sum({v})")),
        CollectiveKind::Reduce(ReduceOp::Max) => Ok(format!(
            "select(simd_max({v}), NAN, simd_sum(isnan({v}) ? 1.0f : 0.0f) > 0.0f)"
        )),
        CollectiveKind::Reduce(ReduceOp::Min) => Ok(format!(
            "select(simd_min({v}), NAN, simd_sum(isnan({v}) ? 1.0f : 0.0f) > 0.0f)"
        )),
        other => Err(format!(
            "the float subgroup collective {other:?} is declined: only Sum, Max and Min of an F32 kernel are lowered (Prod, Mean, shuffles, votes, broadcasts and half or bfloat are unmeasured or outside the board 506 A ruling)"
        )),
    }
}

impl Dialect for MslFloat {
    fn name(&self) -> &'static str {
        "msl"
    }

    fn float(&self) -> bool {
        true
    }

    fn subgroup_collectives(&self) -> bool {
        // F32 only: half and bfloat reductions are unmeasured (board 506 A float ruling, pm-kernels 2026-11-27).
        !self.half && !self.bf16
    }

    fn subgroup_collective(&self, c: &SubgroupCollective<'_>) -> Result<String, String> {
        spell_float_collective(c)
    }

    fn half(&self) -> bool {
        self.half
    }

    fn bfloat(&self) -> bool {
        self.bf16
    }

    fn lit_float(&self, v: f64) -> Option<String> {
        // The walker has checked `v` is exact at the kernel's width, so `as f32` loses nothing for a half or a bfloat.
        if self.bf16 {
            // MSL 2.24: neither float nor half converts implicitly to bfloat, so the literal is an explicit cast.
            return Some(format!("static_cast<bfloat>({:?}f)", v as f32));
        }
        let suffix = if self.half { 'h' } else { 'f' };
        let text = format!("{:?}{suffix}", v as f32);
        Some(if text.starts_with('-') {
            format!("({text})")
        } else {
            text
        })
    }

    fn float_unary(&self, op: UnaryOp, x: &str) -> Option<String> {
        if self.bf16 {
            return bfloat_unary(op, x);
        }
        Some(match op {
            UnaryOp::Neg => format!("(-{x})"),
            UnaryOp::Abs => format!("fabs({x})"),
            UnaryOp::Sqr => format!("({x} * {x})"),
            UnaryOp::Floor => format!("floor({x})"),
            UnaryOp::Ceil => format!("ceil({x})"),
            UnaryOp::Trunc => format!("trunc({x})"),
            // `rint` rounds half to even; Metal's `round` rounds half away from zero.
            UnaryOp::Round => format!("rint({x})"),
            _ => return None,
        })
    }

    fn float_minmax(&self, is_max: bool, a: &str, b: &str) -> Option<String> {
        let cmp = if is_max { ">=" } else { "<=" };
        Some(format!(
            "(isnan({a}) ? {a} : (isnan({b}) ? {b} : ({a} {cmp} {b} ? {a} : {b})))"
        ))
    }

    fn truthy_of(&self, value: &str, t: ElementKind) -> String {
        let zero = elem(t).map_or("0", |(_, z)| z);
        format!("({value} != {zero})")
    }

    fn type_name(&self, t: ElementKind) -> Option<String> {
        elem(t).map(|(n, _)| n.to_string())
    }

    fn convert(
        &self,
        x: &str,
        from: ElementKind,
        to: ElementKind,
        round: RoundMode,
    ) -> Option<String> {
        let (ty, _) = elem(to)?;
        let is_float = |t| matches!(t, ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16);
        if is_float(from) && is_float(to) {
            // Widening is exact; narrowing rounds to nearest even (MSL 2.23, `static_cast`).
            return Some(format!("(({ty})({x}))"));
        }
        if matches!(from, ElementKind::F16 | ElementKind::Bf16)
            || matches!(to, ElementKind::F16 | ElementKind::Bf16)
        {
            // Half to or from an integer is not spelled: declined by name, not guessed.
            return None;
        }
        if from != ElementKind::F32 || to == ElementKind::F32 {
            // int to int wraps; int to float rounds to nearest even; both are Metal's own conversion.
            return Some(format!("(({ty})({x}))"));
        }
        let round_fn = match round {
            RoundMode::NearestEven => "rint",
            RoundMode::TowardZero => "trunc",
            RoundMode::TowardNegInf => "floor",
            RoundMode::TowardPosInf => "ceil",
            _ => return None,
        };
        // NaN to 0, clamp to the range; the conversion is undefined out of range, so it only sees
        // an in-range value.
        let [lo, hi, lo_i, hi_i] = bounds(to);
        let zero_i = if to == ElementKind::I32 { "0" } else { "0u" };
        let r = format!("{round_fn}({x})");
        Some(format!(
            "(isnan({x}) ? {zero_i} : ({r} >= {hi} ? {hi_i} : ({r} < {lo} ? {lo_i} : {ty}(({r} >= {lo} && {r} < {hi}) ? {r} : 0.0f))))"
        ))
    }

    fn temp_typed(&self, ty: &str, name: &str, rhs: &str) -> String {
        format!("const {ty} {name} = {rhs};")
    }

    fn prologue(&self, layout: &Layout) -> String {
        let mut s = String::from(
            "// Generated by vulkane-msl-emit (lower_kernel): do not edit.\n#include <metal_stdlib>\nusing namespace metal;\n#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n",
        );
        let [x, y, z] = layout.block;
        let _ = writeln!(s, "// threadsPerThreadgroup: {x} x {y} x {z}");
        let _ = writeln!(s, "kernel void {}(", layout.name);
        for b in &layout.buffers {
            let access = if b.role == BufferRole::Read {
                "const "
            } else {
                ""
            };
            let t = elem(b.elem).map_or("float", |(n, _)| n);
            let _ = writeln!(
                s,
                "    device {access}{t}* {} [[buffer({})]],",
                b.name, b.binding
            );
        }
        s.push_str("    uint3 tid [[thread_position_in_threadgroup]],\n");
        s.push_str("    uint3 gid [[threadgroup_position_in_grid]]");
        s.push_str(&self.ids.params());
        s.push_str(")\n{\n");
        for a in &layout.shared {
            let class = if a.private { "" } else { "threadgroup " };
            let t = elem(a.elem).map_or("float", |(n, _)| n);
            let _ = writeln!(s, "    {class}{t} {}[{}];", a.name, a.len);
        }
        for i in 0..layout.locals {
            let (t, zero) = layout
                .local_elems
                .get(i)
                .copied()
                .and_then(elem)
                .unwrap_or(("float", "0.0f"));
            let _ = writeln!(s, "    {t} l{i} = {zero};");
        }
        s
    }

    fn epilogue(&self) -> &'static str {
        "}\n"
    }

    fn lit(&self, v: i64, _signed: bool) -> String {
        self.lit_float(v as f64).unwrap_or_default()
    }

    fn index(&self, name: &str, idx: &str, _is_buffer: bool) -> String {
        format!("{name}[{idx}]")
    }

    fn exec(&self, id: ExecId, _signed: bool) -> Option<String> {
        // An execution id stays the `u32` the IR types it as.
        self.ids.exec(id, false)
    }

    fn binary(&self, op: BinaryOp, a: &str, b: &str, signed: bool) -> Option<String> {
        if let Some(sym) = operator(COMPARISONS, op) {
            let (one, zero) = (self.lit(1, signed), self.lit(0, signed));
            return Some(format!("({a} {sym} {b} ? {one} : {zero})"));
        }
        operator(BIT_OPS, op).map(|sym| format!("({a} {sym} {b})"))
    }

    fn select(&self, cond: &str, a: &str, b: &str, signed: bool) -> String {
        format!("({} ? {a} : {b})", self.truthy(cond, signed))
    }

    fn temp(&self, name: &str, rhs: &str, _signed: bool) -> String {
        let t = if self.half { "half" } else { "float" };
        format!("const {t} {name} = {rhs};")
    }

    fn bound(&self, name: &str, rhs: &str, signed: bool) -> String {
        self.temp(name, rhs, signed)
    }

    fn barrier(&self, scope: Scope) -> Option<String> {
        barrier(scope)
    }

    fn fence(&self, scope: Scope, order: MemoryOrder) -> Option<String> {
        fence(scope, order)
    }
}
