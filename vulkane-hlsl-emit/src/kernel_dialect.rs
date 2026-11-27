// SPDX-License-Identifier: MIT OR Apache-2.0
//! The HLSL spellings the shared walker (`vulkane-kernel-text`) asks for, for the one family
//! `fishstringer-slang-emit` declines: a 32-bit integer kernel holding 16-bit integer buffers or
//! locals (`I16`/`U16`, ROADMAP row 8e).
//!
//! `dxc` compiles `int16_t`/`uint16_t` with `-enable-16bit-types` (SPIR-V `Int16` and
//! `StorageBuffer16BitAccess`), and has no 8-bit integer type at all, so an `I8`/`U8` kernel stays a
//! typed decline, named, with the reason. The compute dtype stays 32-bit: a 16-bit value is reached
//! and left through a wrapping `Cast`, exactly as in the GLSL and SPIR-V arms.
//!
//! **Gaps stated, not hidden:** no atomics, no subgroup ids, no scalar parameters (a specialization
//! constant would be declared here but is untested, so it is refused before any text is written),
//! no float, no 64-bit; a `Fence` at block or device scope is spelled, others decline.

use std::fmt::Write;

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::exec::{ExecId, Scope};
use unpopped::ir::{BinaryOp, RoundMode};
use unpopped::kernel::{BufferRole, KernelFn, MemoryOrder, ParamKind};
use unpopped_vocab::ElementKind;
use vulkane_kernel_text::{BIT_OPS, COMPARISONS, Dialect, Layout, operator};

/// The `dxc` flag a 16-bit integer kernel needs: without it `int16_t` is a 32-bit alias.
const FLAG_16BIT: &str = "-enable-16bit-types";

/// The HLSL dialect of [`vulkane_kernel_text::lower_kernel`] for 32-bit integer kernels with 16-bit storage.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct HlslKernel;

/// The 8- and 16-bit integer element types the kernel's buffers and locals hold.
pub(crate) fn narrow_elems(kernel: &KernelFn) -> Vec<ElementKind> {
    let narrow = |t: ElementKind| {
        matches!(
            t,
            ElementKind::I8 | ElementKind::U8 | ElementKind::I16 | ElementKind::U16
        )
    };
    kernel
        .params
        .iter()
        .filter_map(|p| match &p.kind {
            ParamKind::Buffer { elem, .. } => Some(*elem),
            _ => None,
        })
        .chain(kernel.locals.iter().map(|l| l.elem))
        .filter(|t| narrow(*t))
        .collect()
}

/// The extra `dxc` arguments `kernel` needs (beyond `-T cs_6_2 -E <entry> -spirv`): `-enable-16bit-types`
/// when it holds a 16-bit integer buffer or local, none otherwise.
pub(crate) fn dxc_args(kernel: &KernelFn) -> &'static [&'static str] {
    if narrow_elems(kernel)
        .iter()
        .any(|t| matches!(t, ElementKind::I16 | ElementKind::U16))
    {
        &[FLAG_16BIT]
    } else {
        &[]
    }
}

/// Lower a kernel that holds 8- or 16-bit integers (`narrow` is non-empty), or decline it by name.
pub(crate) fn lower(
    backend: &str,
    kernel: &KernelFn,
    narrow: &[ElementKind],
) -> Result<GeneratedKernel, LowerError> {
    let decline = |detail: String| LowerError::UnsupportedKernel {
        detail: format!("{backend} backend: {detail} (kernel `{}`)", kernel.name),
    };
    if let Some(e) = narrow
        .iter()
        .find(|t| matches!(t, ElementKind::I8 | ElementKind::U8))
    {
        return Err(decline(format!(
            "a {e:?} buffer or local is not lowered: dxc has no 8-bit integer type (`int8_t` and `uint8_t` are undeclared, with or without `-enable-16bit-types`)"
        )));
    }
    if kernel
        .params
        .iter()
        .any(|p| matches!(p.kind, ParamKind::Scalar { .. }))
    {
        return Err(decline(
            "a scalar parameter beside a 16-bit integer buffer is not lowered in HLSL".to_string(),
        ));
    }
    vulkane_kernel_text::lower_kernel(&HlslKernel, kernel)
}

fn ty(signed: bool) -> &'static str {
    if signed { "int" } else { "uint" }
}

/// The HLSL type of a 16-bit integer element.
fn narrow_ty(elem: ElementKind) -> Option<&'static str> {
    match elem {
        ElementKind::I16 => Some("int16_t"),
        ElementKind::U16 => Some("uint16_t"),
        _ => None,
    }
}

fn axis(a: u8) -> Option<char> {
    ["x", "y", "z"]
        .get(usize::from(a))
        .and_then(|s| s.chars().next())
}

impl HlslKernel {
    fn declarations(layout: &Layout) -> String {
        let t = ty(layout.signed);
        let mut s = String::new();
        for b in &layout.buffers {
            let class = if b.role == BufferRole::Read {
                "StructuredBuffer"
            } else {
                "RWStructuredBuffer"
            };
            let bt = narrow_ty(b.elem).unwrap_or(t);
            let _ = writeln!(
                s,
                "[[vk::binding({}, 0)]] {class}<{bt}> {};",
                b.binding, b.name
            );
        }
        for a in &layout.shared {
            let class = if a.private { "static" } else { "groupshared" };
            let _ = writeln!(s, "{class} {t} {}[{}];", a.name, a.len);
        }
        s
    }
}

impl Dialect for HlslKernel {
    fn name(&self) -> &'static str {
        "hlsl"
    }

    fn narrow_ints(&self) -> bool {
        true
    }

    fn convert(
        &self,
        x: &str,
        _from: ElementKind,
        to: ElementKind,
        _round: RoundMode,
    ) -> Option<String> {
        // A C-style cast between integer widths: truncating (wrapping) to 16 bits, sign-extending a
        // signed 16-bit source and zero-extending an unsigned one when widening.
        let target = match to {
            ElementKind::I32 => "int",
            ElementKind::U32 => "uint",
            other => narrow_ty(other)?,
        };
        Some(format!("(({target})({x}))"))
    }

    fn prologue(&self, layout: &Layout) -> String {
        let t = ty(layout.signed);
        let [x, y, z] = layout.block;
        let mut s = String::from(
            "// Generated by vulkane-hlsl-emit (lower_kernel): do not edit.\n// dxc: -enable-16bit-types\n",
        );
        s.push_str(&Self::declarations(layout));
        let _ = writeln!(s, "[numthreads({x}, {y}, {z})]");
        let _ = writeln!(
            s,
            "void {}(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)\n{{",
            layout.name
        );
        for i in 0..layout.locals {
            if let Some(nt) = layout.local_elems.get(i).copied().and_then(narrow_ty) {
                let _ = writeln!(s, "    {nt} l{i} = ({nt})0;");
            } else {
                let _ = writeln!(s, "    {t} l{i} = {};", self.lit(0, layout.signed));
            }
        }
        s
    }

    fn epilogue(&self) -> &'static str {
        "}\n"
    }

    fn lit(&self, v: i64, signed: bool) -> String {
        match (signed, v) {
            (true, i64::MIN..=-2_147_483_648) => "(-2147483647 - 1)".to_string(),
            (true, ..=-1) => format!("({v})"),
            (true, _) => v.to_string(),
            (false, _) => format!("{v}u"),
        }
    }

    fn index(&self, name: &str, idx: &str, _is_buffer: bool) -> String {
        format!("{name}[{idx}]")
    }

    fn exec(&self, id: ExecId, signed: bool) -> Option<String> {
        let (var, a) = match id {
            ExecId::ThreadIdx(a) => ("gtid", a),
            ExecId::BlockIdx(a) => ("gid", a),
            _ => return None,
        };
        let v = format!("{var}.{}", axis(a)?);
        Some(if signed { format!("((int){v})") } else { v })
    }

    fn binary(&self, op: BinaryOp, a: &str, b: &str, signed: bool) -> Option<String> {
        if let Some(sym) = operator(COMPARISONS, op) {
            let (one, zero) = (self.lit(1, signed), self.lit(0, signed));
            return Some(format!("({a} {sym} {b} ? {one} : {zero})"));
        }
        if let Some(sym) = operator(BIT_OPS, op) {
            return Some(format!("({a} {sym} {b})"));
        }
        let sym = match op {
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            _ => return None,
        };
        Some(format!("({a} {sym} ({b} & {}))", self.lit(31, signed)))
    }

    fn select(&self, cond: &str, a: &str, b: &str, signed: bool) -> String {
        format!("({} ? {a} : {b})", self.truthy(cond, signed))
    }

    fn temp(&self, name: &str, rhs: &str, signed: bool) -> String {
        format!("const {} {name} = {rhs};", ty(signed))
    }

    fn bound(&self, name: &str, rhs: &str, signed: bool) -> String {
        self.temp(name, rhs, signed)
    }

    fn barrier(&self, scope: Scope) -> Option<String> {
        (scope == Scope::Block).then(|| "AllMemoryBarrierWithGroupSync();".to_string())
    }

    fn fence(&self, scope: Scope, _order: MemoryOrder) -> Option<String> {
        match scope {
            Scope::Block => Some("GroupMemoryBarrier();".to_string()),
            Scope::Device => Some("DeviceMemoryBarrier();".to_string()),
            _ => None,
        }
    }
}
