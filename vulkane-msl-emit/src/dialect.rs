// SPDX-License-Identifier: MIT OR Apache-2.0
//! The MSL spellings the shared walker asks for.

use std::cell::Cell;
use std::fmt::Write;

use unpopped::exec::{ExecId, Scope};
use unpopped::ir::{BinaryOp, RoundMode};
use unpopped::kernel::{AtomicOp, BufferRole, MemoryOrder};
use unpopped_vocab::ElementKind;
use vulkane_kernel_text::{BIT_OPS, COMPARISONS, Dialect, Layout, operator};

/// The MSL backend of Unpopped: [`Backend::lower_kernel`](unpopped::backend::Backend::lower_kernel) is the entry point.
#[derive(Copy, Clone, Debug, Default)]
pub struct Msl;

/// The built-in ids a kernel body read (`BlockDim` and `GridDim` never reach the dialect: the walker folds them to launch
/// constants), so the signature declares exactly those (a bit per id). The walker lowers the
/// body before it asks for the prologue, so a spelling asked for here is known by then.
#[derive(Debug, Default)]
pub(crate) struct Ids(Cell<u8>);

const LANE: u8 = 1;
const SUBGROUP: u8 = 2;
const SIZE: u8 = 4;

impl Ids {
    /// The spelling of an execution id (marking it for the signature), or `None` when MSL has none.
    pub(crate) fn exec(&self, id: ExecId, signed: bool) -> Option<String> {
        let (bit, name) = match id {
            ExecId::ThreadIdx(a) => (0, format!("tid.{}", axis(a)?)),
            ExecId::BlockIdx(a) => (0, format!("gid.{}", axis(a)?)),
            ExecId::LaneId => (LANE, "lane".to_string()),
            ExecId::SubgroupId => (SUBGROUP, "sgid".to_string()),
            ExecId::SubgroupSize => (SIZE, "sgsize".to_string()),
            _ => return None,
        };
        self.0.set(self.0.get() | bit);
        Some(if signed {
            format!("((int){name})")
        } else {
            name
        })
    }

    /// The extra signature parameters (each preceded by `,`) for the ids the body read.
    pub(crate) fn params(&self) -> String {
        let used = self.0.get();
        let mut s = String::new();
        for (bit, decl) in [
            (LANE, "uint lane [[thread_index_in_simdgroup]]"),
            (SUBGROUP, "uint sgid [[simdgroup_index_in_threadgroup]]"),
            (SIZE, "uint sgsize [[threads_per_simdgroup]]"),
        ] {
            if used & bit != 0 {
                let _ = write!(s, ",\n    {decl}");
            }
        }
        s
    }
}

/// The MSL dialect of [`vulkane_kernel_text::lower_kernel`] for `I32`/`U32` kernels.
#[derive(Debug, Default)]
pub(crate) struct IntKernel {
    /// The built-in ids the body read.
    pub(crate) ids: Ids,
}

fn ty(signed: bool) -> &'static str {
    if signed { "int" } else { "uint" }
}

/// The Metal type of an 8- or 16-bit integer element (Metal has all four natively).
fn narrow_ty(elem: ElementKind) -> Option<&'static str> {
    match elem {
        ElementKind::I8 => Some("char"),
        ElementKind::U8 => Some("uchar"),
        ElementKind::I16 => Some("short"),
        ElementKind::U16 => Some("ushort"),
        _ => None,
    }
}

/// The Metal type of a buffer or array element: `atomic_int`/`atomic_uint` when the kernel applies
/// an atomic to it (Metal atomics act on atomic types in `device` or `threadgroup` memory only).
fn elem_ty(signed: bool, atomic: bool) -> &'static str {
    match (atomic, signed) {
        (false, s) => ty(s),
        (true, true) => "atomic_int",
        (true, false) => "atomic_uint",
    }
}

/// `x`, `y` or `z`.
pub(crate) fn axis(a: u8) -> Option<char> {
    ["x", "y", "z"]
        .get(usize::from(a))
        .and_then(|s| s.chars().next())
}

/// The workgroup-and-buffer barrier, at block scope only.
pub(crate) fn barrier(scope: Scope) -> Option<String> {
    let call = match scope {
        Scope::Block => "threadgroup_barrier",
        Scope::Subgroup => "simdgroup_barrier",
        _ => return None,
    };
    Some(format!(
        "{call}(mem_flags::mem_threadgroup | mem_flags::mem_device);"
    ))
}

/// A fence at block or device scope; every order collapses to `memory_order_seq_cst`.
pub(crate) fn fence(scope: Scope, _order: MemoryOrder) -> Option<String> {
    let (flags, thread_scope) = match scope {
        Scope::Block => ("mem_threadgroup", "thread_scope_threadgroup"),
        Scope::Device => ("mem_device", "thread_scope_device"),
        _ => return None,
    };
    Some(format!(
        "atomic_thread_fence(mem_flags::{flags}, memory_order_seq_cst, {thread_scope});"
    ))
}

impl Dialect for IntKernel {
    fn name(&self) -> &'static str {
        "msl"
    }

    fn narrow_ints(&self) -> bool {
        true
    }

    /// A C-style cast between integer widths: modular when narrowing (Metal is C++ based; the result of narrowing a
    /// signed value is implementation-defined before C++20 and two's complement in practice, not compiled here),
    /// sign-extending a signed narrow source and zero-extending an unsigned one when widening.
    fn convert(
        &self,
        x: &str,
        _from: ElementKind,
        to: ElementKind,
        _round: RoundMode,
    ) -> Option<String> {
        let target = match to {
            ElementKind::I32 => "int",
            ElementKind::U32 => "uint",
            other => narrow_ty(other)?,
        };
        Some(format!("(({target})({x}))"))
    }

    fn prologue(&self, layout: &Layout) -> String {
        let t = ty(layout.signed);
        let mut s = String::from(
            "// Generated by vulkane-msl-emit (lower_kernel): do not edit.\n#include <metal_stdlib>\nusing namespace metal;\n",
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
            let _ = writeln!(
                s,
                "    device {access}{}* {} [[buffer({})]],",
                narrow_ty(b.elem).unwrap_or_else(|| elem_ty(layout.signed, b.atomic)),
                b.name,
                b.binding
            );
        }
        s.push_str("    uint3 tid [[thread_position_in_threadgroup]],\n");
        s.push_str("    uint3 gid [[threadgroup_position_in_grid]]");
        s.push_str(&self.ids.params());
        s.push_str(")\n{\n");
        for a in &layout.shared {
            let class = if a.private { "" } else { "threadgroup " };
            let _ = writeln!(
                s,
                "    {class}{} {}[{}];",
                elem_ty(layout.signed, a.atomic && !a.private),
                a.name,
                a.len
            );
        }
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
        self.ids.exec(id, signed)
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
        barrier(scope)
    }

    fn fence(&self, scope: Scope, order: MemoryOrder) -> Option<String> {
        fence(scope, order)
    }

    fn typed_atomics(&self) -> bool {
        true
    }

    /// Metal atomics act on `device` and `threadgroup` memory; a subgroup or cluster scope has no form.
    fn atomic_scope_ok(&self, scope: Scope) -> bool {
        matches!(scope, Scope::Block | Scope::Device)
    }

    fn atomic_load(&self, elem: &str) -> Option<String> {
        Some(format!(
            "atomic_load_explicit(&{elem}, memory_order_relaxed)"
        ))
    }

    fn atomic_store(&self, elem: &str, value: &str) -> Option<String> {
        Some(format!(
            "atomic_store_explicit(&{elem}, {value}, memory_order_relaxed);"
        ))
    }

    fn atomic_rmw(&self, op: AtomicOp, elem: &str, value: &str) -> Option<String> {
        let f = match op {
            AtomicOp::Add => "atomic_fetch_add_explicit",
            AtomicOp::Min => "atomic_fetch_min_explicit",
            AtomicOp::Max => "atomic_fetch_max_explicit",
            AtomicOp::And => "atomic_fetch_and_explicit",
            AtomicOp::Or => "atomic_fetch_or_explicit",
            AtomicOp::Xor => "atomic_fetch_xor_explicit",
            AtomicOp::Exchange => "atomic_exchange_explicit",
            _ => return None,
        };
        Some(format!("{f}(&{elem}, {value}, memory_order_relaxed)"))
    }

    /// Metal has only `atomic_compare_exchange_weak_explicit`, which may fail with the element equal to
    /// `expected` (spuriously) and then writes the value it saw to `expected`. The IR's compare-exchange
    /// (the old value tells whether it swapped) cannot express a spurious failure, so the exchange is
    /// retried while the value seen still equals the original `expected`.
    fn atomic_cas_lines(
        &self,
        elem: &str,
        expected: &str,
        desired: &str,
        dest: Option<&str>,
    ) -> Option<Vec<String>> {
        let mut lines = vec![
            "{".to_string(),
            format!("    const auto cas_x = {expected};"),
            "    auto cas_e = cas_x;".to_string(),
            format!("    const auto cas_d = {desired};"),
            format!(
                "    while (!atomic_compare_exchange_weak_explicit(&{elem}, &cas_e, cas_d, memory_order_relaxed, memory_order_relaxed) && cas_e == cas_x) {{}}"
            ),
        ];
        if let Some(l) = dest {
            lines.push(format!("    {l} = cas_e;"));
        }
        lines.push("}".to_string());
        Some(lines)
    }
}
