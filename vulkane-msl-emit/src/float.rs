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
use unpopped::ir::{BinaryOp, RoundMode, UnaryOp};
use unpopped::kernel::{BufferRole, MemoryOrder};
use unpopped_vocab::ElementKind;
use vulkane_kernel_text::{BIT_OPS, COMPARISONS, Dialect, Layout, operator};

use crate::dialect::{axis, barrier, fence};

/// The MSL dialect of an `F32` or `F16` kernel; the integer one is [`crate::Msl`].
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct MslFloat {
    /// `true` for an `F16` kernel (`half`), `false` for `F32`.
    pub(crate) half: bool,
}

fn elem(t: ElementKind) -> Option<(&'static str, &'static str)> {
    match t {
        ElementKind::F32 => Some(("float", "0.0f")),
        ElementKind::F16 => Some(("half", "0.0h")),
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

impl Dialect for MslFloat {
    fn name(&self) -> &'static str {
        "msl"
    }

    fn float(&self) -> bool {
        true
    }

    fn half(&self) -> bool {
        self.half
    }

    fn lit_float(&self, v: f64) -> Option<String> {
        // The walker has checked `v` is exact at the kernel's width, so `as f32` loses nothing for a half.
        let suffix = if self.half { 'h' } else { 'f' };
        let text = format!("{:?}{suffix}", v as f32);
        Some(if text.starts_with('-') {
            format!("({text})")
        } else {
            text
        })
    }

    fn float_unary(&self, op: UnaryOp, x: &str) -> Option<String> {
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
        let is_float = |t| matches!(t, ElementKind::F32 | ElementKind::F16);
        if is_float(from) && is_float(to) {
            // Widening is exact; narrowing rounds to nearest even (MSL 2.23, `static_cast`).
            return Some(format!("(({ty})({x}))"));
        }
        if from == ElementKind::F16 || to == ElementKind::F16 {
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
        s.push_str("    uint3 gid [[threadgroup_position_in_grid]])\n{\n");
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
        let (var, a) = match id {
            ExecId::ThreadIdx(a) => ("tid", a),
            ExecId::BlockIdx(a) => ("gid", a),
            _ => return None,
        };
        // An execution id stays the `u32` the IR types it as.
        Some(format!("{var}.{}", axis(a)?))
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
