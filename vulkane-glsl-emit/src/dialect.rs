// SPDX-License-Identifier: MIT OR Apache-2.0
//! The GLSL spellings the shared walker asks for.

use std::fmt::Write;

use unpopped::exec::{ExecId, Scope};
use unpopped::ir::{BinaryOp, RoundMode, UnaryOp};
use unpopped::kernel::{AtomicOp, BufferRole, MemoryOrder};
use unpopped_vocab::ElementKind;
use vulkane_kernel_text::{BIT_OPS, COMPARISONS, Dialect, Layout, operator};

/// The GLSL dialect of [`vulkane_kernel_text::lower_kernel`].
#[derive(Copy, Clone, Debug, Default)]
pub struct Glsl;

/// The GLSL spellings for one kernel: 32-bit (`int`/`uint`) or, for an `I64`/`U64` kernel, 64-bit
/// (`int64_t`/`uint64_t`, `GL_ARB_gpu_shader_int64`).
#[derive(Copy, Clone, Debug)]
pub(crate) struct Text {
    pub(crate) wide: bool,
    /// Whether the compute dtype is signed (`I64`).
    pub(crate) signed: bool,
    /// Whether the compute dtype is a float (`float`, or `double` when `double`; every operation
    /// `precise`).
    pub(crate) float: bool,
    /// Whether that float is `F64` (`GL_ARB_gpu_shader_fp64`).
    pub(crate) double: bool,
}

impl Text {
    /// The scalar type name.
    fn ty(self, signed: bool) -> &'static str {
        if self.float {
            return if self.double { "double" } else { "float" };
        }
        match (self.wide, signed) {
            (false, true) => "int",
            (false, false) => "uint",
            (true, true) => "int64_t",
            (true, false) => "uint64_t",
        }
    }
}

/// One `spirv_execution_mode` line declaring `capability` and setting `mode` at `width`.
fn mode(capability: u32, mode: u32, width: u32) -> String {
    format!(
        "spirv_execution_mode(extensions = [\"SPV_KHR_float_controls\"], capabilities = [{capability}], {mode}, {width});\n"
    )
}

/// `x`, `y` or `z`.
fn axis(a: u8) -> Option<char> {
    ["x", "y", "z"]
        .get(usize::from(a))
        .and_then(|s| s.chars().next())
}

/// The declared type and zero of a float-kernel element type.
fn float_elem(elem: ElementKind) -> Option<(&'static str, &'static str)> {
    Some(match elem {
        ElementKind::F32 => ("float", "0.0"),
        ElementKind::F64 => ("double", "0.0lf"),
        ElementKind::I32 => ("int", "0"),
        ElementKind::U32 => ("uint", "0u"),
        _ => return None,
    })
}

/// A float conversion's range test: the exact bounds `[lo, hi)` of the 32-bit integer `to`, spelled
/// in the source float type, and the integer bounds the result clamps to.
fn int_bounds(to: ElementKind, double: bool) -> [String; 4] {
    let f = |v: &str| {
        if double {
            format!("{v}lf")
        } else {
            v.to_string()
        }
    };
    if to == ElementKind::I32 {
        [
            f("-2147483648.0"),
            f("2147483648.0"),
            "(-2147483647 - 1)".into(),
            "2147483647".into(),
        ]
    } else {
        [
            f("0.0"),
            f("4294967296.0"),
            "0u".into(),
            "4294967295u".into(),
        ]
    }
}

impl Text {
    /// The `layout`, buffer and shared declarations.
    fn declarations(self, layout: &Layout) -> String {
        let t = self.ty(layout.signed);
        let [x, y, z] = layout.block;
        let mut s =
            format!("layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;\n");
        for b in &layout.buffers {
            if b.spec_local.is_some() {
                let _ = writeln!(
                    s,
                    "layout(constant_id = {}) const {t} {} = {};",
                    b.binding,
                    b.name,
                    if layout.signed { "1" } else { "1u" }
                );
                continue;
            }
            let access = match b.role {
                BufferRole::Read => "readonly ",
                BufferRole::Write => "writeonly ",
                _ => "",
            };
            let n = &b.name;
            let bt = float_elem(b.elem)
                .filter(|_| self.float)
                .map_or(t, |(ty, _)| ty);
            let _ = writeln!(
                s,
                "layout(std430, set = 0, binding = {}) {access}buffer B_{n} {{ {bt} data[]; }} {n};",
                b.binding
            );
        }
        for a in &layout.shared {
            let class = if a.private { "" } else { "shared " };
            let len = a.len_spec.clone().unwrap_or_else(|| a.len.to_string());
            let _ = writeln!(s, "{class}{t} {}[{len}];", a.name);
        }
        s
    }
}

impl Text {
    /// The extension and float-control lines of a float kernel: `GL_ARB_gpu_shader_fp64` when a
    /// double is used, `DenormPreserve` and `SignedZeroInfNanPreserve` at each float width the kernel
    /// uses (its compute dtype, buffers and locals), and `RoundingModeRTE` where it rounds a
    /// conversion (`SPV_KHR_float_controls`, as the SPIR-V emitter declares them).
    fn float_header(self, layout: &Layout) -> String {
        let elems = layout
            .buffers
            .iter()
            .map(|b| b.elem)
            .chain(layout.local_elems.iter().copied());
        let mut widths: Vec<u32> = vec![if self.double { 64 } else { 32 }];
        for e in elems {
            let w = match e {
                ElementKind::F32 => 32,
                ElementKind::F64 => 64,
                _ => continue,
            };
            if !widths.contains(&w) {
                widths.push(w);
            }
        }
        widths.sort_unstable();
        let mut s = String::new();
        if widths.contains(&64) {
            s.push_str("#extension GL_ARB_gpu_shader_fp64 : require\n");
        }
        s.push_str("#extension GL_EXT_spirv_intrinsics : require\n");
        for w in &widths {
            s.push_str(&mode(4464, 4459, *w));
            s.push_str(&mode(4466, 4461, *w));
        }
        for w in &layout.rte {
            s.push_str(&mode(4467, 4462, *w));
        }
        s
    }
}

impl Dialect for Text {
    fn name(&self) -> &'static str {
        "glsl"
    }

    fn wide(&self) -> bool {
        self.wide
    }

    fn float(&self) -> bool {
        self.float
    }

    fn double(&self) -> bool {
        self.double
    }

    fn mixed_float_widths(&self) -> bool {
        // glslang keeps one `spirv_execution_mode` per mode, so a second width would silently replace
        // the first (measured with glslang 16.2: `DenormPreserve 32` then `64` leaves only 64).
        false
    }

    fn lit_float(&self, v: f64) -> Option<String> {
        // `{:?}` of an `f32` is the shortest text that reads back to the same value, with a `.` or an
        // exponent, so it is a GLSL `float` literal.
        let text = if self.double {
            format!("{v:?}lf")
        } else {
            format!("{:?}", v as f32)
        };
        Some(if text.starts_with('-') {
            format!("({text})")
        } else {
            text
        })
    }

    fn float_unary(&self, op: UnaryOp, x: &str) -> Option<String> {
        // Exact in Vulkan (`OpFNegate`, `FAbs`, `Floor`, `Ceil`, `Trunc`, `RoundEven`) or one
        // correctly rounded multiply (`Sqr`).
        Some(match op {
            UnaryOp::Neg => format!("(-{x})"),
            UnaryOp::Abs => format!("abs({x})"),
            UnaryOp::Sqr => format!("({x} * {x})"),
            UnaryOp::Floor => format!("floor({x})"),
            UnaryOp::Ceil => format!("ceil({x})"),
            UnaryOp::Trunc => format!("trunc({x})"),
            UnaryOp::Round => format!("roundEven({x})"),
            _ => return None,
        })
    }

    fn truthy_of(&self, value: &str, t: ElementKind) -> String {
        let zero = match t {
            ElementKind::F64 => "0.0lf",
            ElementKind::F32 => "0.0",
            ElementKind::I32 => "0",
            _ => "0u",
        };
        format!("({value} != {zero})")
    }

    fn float_minmax(&self, is_max: bool, a: &str, b: &str) -> Option<String> {
        // NaN-propagating, `a` on ties: the KISS-Ops `max_prop`/`min_prop` decomposition.
        let cmp = if is_max { ">=" } else { "<=" };
        Some(format!(
            "(isnan({a}) ? {a} : (isnan({b}) ? {b} : ({a} {cmp} {b} ? {a} : {b})))"
        ))
    }

    fn convert(
        &self,
        x: &str,
        from: ElementKind,
        to: ElementKind,
        round: RoundMode,
    ) -> Option<String> {
        let (ty, _) = float_elem(to)?;
        let src_float = matches!(from, ElementKind::F32 | ElementKind::F64);
        if !src_float || matches!(to, ElementKind::F32 | ElementKind::F64) {
            // int to int (wraps), int to float and float to float (rounded to nearest even under the
            // module's `RoundingModeRTE`, or exact when widening): GLSL's own conversion.
            return Some(format!("{ty}({x})"));
        }
        // Float to integer: round per the mode, NaN to 0, clamp to the range; the conversion itself is
        // undefined out of range, so it only ever sees an in-range value.
        let round_fn = match round {
            RoundMode::NearestEven => "roundEven",
            RoundMode::TowardZero => "trunc",
            RoundMode::TowardNegInf => "floor",
            RoundMode::TowardPosInf => "ceil",
            _ => return None,
        };
        let [lo, hi, lo_i, hi_i] = int_bounds(to, from == ElementKind::F64);
        let zero_f = if from == ElementKind::F64 {
            "0.0lf"
        } else {
            "0.0"
        };
        let zero_i = if to == ElementKind::I32 { "0" } else { "0u" };
        let r = format!("{round_fn}({x})");
        Some(format!(
            "(isnan({x}) ? {zero_i} : ({r} >= {hi} ? {hi_i} : ({r} < {lo} ? {lo_i} : {ty}(({r} >= {lo} && {r} < {hi}) ? {r} : {zero_f}))))"
        ))
    }

    fn type_name(&self, t: ElementKind) -> Option<String> {
        match t {
            ElementKind::F32 => Some("float".to_string()),
            ElementKind::F64 => Some("double".to_string()),
            ElementKind::U32 => Some("uint".to_string()),
            ElementKind::I32 => Some("int".to_string()),
            _ => None,
        }
    }

    fn temp_typed(&self, ty: &str, name: &str, rhs: &str) -> String {
        // A float temporary is `precise`: glslang then decorates every operation contributing to it
        // `NoContraction`, so a driver cannot fuse `a * b + c` into an FMA the source does not ask for.
        let precise = if ty == "float" || ty == "double" {
            "precise "
        } else {
            ""
        };
        format!("{precise}const {ty} {name} = {rhs};")
    }

    fn cast_to_compute(&self, x: &str, from: ElementKind) -> Option<String> {
        let integer = matches!(
            from,
            ElementKind::I32 | ElementKind::U32 | ElementKind::I64 | ElementKind::U64
        );
        // GLSL converts an integer to another width and signedness modulo 2^N, so this wraps.
        integer.then(|| format!("{}({x})", self.ty(self.signed)))
    }

    fn prologue(&self, layout: &Layout) -> String {
        let t = self.ty(layout.signed);
        let mut s = String::from(
            "// Generated by vulkane-glsl-emit (lower_kernel): do not edit.\n#version 450\n",
        );
        if self.wide {
            s.push_str(
                "#extension GL_ARB_gpu_shader_int64 : require
",
            );
        }
        if self.float {
            s.push_str(&self.float_header(layout));
        }
        if layout.subgroup {
            s.push_str("#extension GL_KHR_shader_subgroup_basic : enable\n");
        }
        s.push_str(&self.declarations(layout));
        s.push_str("void main()\n{\n");
        for i in 0..layout.locals {
            if self.float {
                let (ty, zero) = layout
                    .local_elems
                    .get(i)
                    .copied()
                    .and_then(float_elem)
                    .unwrap_or(("float", "0.0"));
                let precise = if ty == "float" || ty == "double" {
                    "precise "
                } else {
                    ""
                };
                let _ = writeln!(s, "    {precise}{ty} l{i} = {zero};");
                continue;
            }
            let init = layout
                .spec_init(i)
                .map_or_else(|| self.lit(0, layout.signed), str::to_string);
            let _ = writeln!(s, "    {t} l{i} = {init};");
        }
        s
    }

    fn epilogue(&self) -> &'static str {
        "}\n"
    }

    fn lit(&self, v: i64, signed: bool) -> String {
        if self.float {
            return self.lit_float(v as f64).unwrap_or_default();
        }
        if self.wide {
            return match (signed, v) {
                (true, i64::MIN) => "(-9223372036854775807l - 1l)".to_string(),
                (true, ..=-1) => format!("({v}l)"),
                (true, _) => format!("{v}l"),
                (false, _) => format!("{v}ul"),
            };
        }
        match (signed, v) {
            (true, i64::MIN..=-2_147_483_648) => "(-2147483647 - 1)".to_string(),
            (true, ..=-1) => format!("({v})"),
            (true, _) => v.to_string(),
            (false, _) => format!("{v}u"),
        }
    }

    fn index(&self, name: &str, idx: &str, is_buffer: bool) -> String {
        // GLSL indexes with a 32-bit integer: a 64-bit index is narrowed (the array has far fewer
        // than 2^31 elements, so the narrowing is exact for every in-range index).
        let idx = &if self.wide {
            format!("int({idx})")
        } else {
            idx.to_string()
        };
        if is_buffer {
            format!("{name}.data[{idx}]")
        } else {
            format!("{name}[{idx}]")
        }
    }

    fn exec(&self, id: ExecId, signed: bool) -> Option<String> {
        let subgroup = match id {
            ExecId::LaneId => Some("gl_SubgroupInvocationID"),
            ExecId::SubgroupId => Some("gl_SubgroupID"),
            ExecId::SubgroupSize => Some("gl_SubgroupSize"),
            _ => None,
        };
        if let Some(v) = subgroup {
            return Some(if signed && !self.wide {
                format!("int({v})")
            } else {
                v.to_string()
            });
        }
        let (var, a) = match id {
            ExecId::ThreadIdx(a) => ("gl_LocalInvocationID", a),
            ExecId::BlockIdx(a) => ("gl_WorkGroupID", a),
            _ => return None,
        };
        let v = format!("{var}.{}", axis(a)?);
        // A wide kernel keeps an execution id the `u32` the IR types it as.
        Some(if signed && !self.wide {
            format!("int({v})")
        } else {
            v
        })
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
        let mask = if self.wide { 63 } else { 31 };
        Some(format!("({a} {sym} ({b} & {}))", self.lit(mask, signed)))
    }

    fn select(&self, cond: &str, a: &str, b: &str, signed: bool) -> String {
        format!("({} ? {a} : {b})", self.truthy(cond, signed))
    }

    fn temp(&self, name: &str, rhs: &str, signed: bool) -> String {
        format!("const {} {name} = {rhs};", self.ty(signed))
    }

    fn bound(&self, name: &str, rhs: &str, signed: bool) -> String {
        self.temp(name, rhs, signed)
    }

    fn barrier(&self, scope: Scope) -> Option<String> {
        match scope {
            Scope::Block => {
                Some("memoryBarrierShared(); memoryBarrierBuffer(); barrier();".to_string())
            }
            Scope::Subgroup => Some("subgroupMemoryBarrier(); subgroupBarrier();".to_string()),
            _ => None,
        }
    }

    fn atomic_rmw(&self, op: AtomicOp, elem: &str, value: &str) -> Option<String> {
        if self.wide || self.float {
            return None; // 64-bit atomics need GL_EXT_shader_atomic_int64 and a device gate: not lowered
        }
        let f = match op {
            AtomicOp::Add => "atomicAdd",
            AtomicOp::Min => "atomicMin",
            AtomicOp::Max => "atomicMax",
            AtomicOp::And => "atomicAnd",
            AtomicOp::Or => "atomicOr",
            AtomicOp::Xor => "atomicXor",
            AtomicOp::Exchange => "atomicExchange",
            _ => return None,
        };
        Some(format!("{f}({elem}, {value})"))
    }

    fn atomic_cas(&self, elem: &str, expected: &str, desired: &str) -> Option<String> {
        if self.wide || self.float {
            return None;
        }
        Some(format!("atomicCompSwap({elem}, {expected}, {desired})"))
    }

    fn fence(&self, scope: Scope, _order: MemoryOrder) -> Option<String> {
        let call = match scope {
            Scope::Block => "groupMemoryBarrier",
            Scope::Device => "memoryBarrier",
            Scope::Subgroup => "subgroupMemoryBarrier",
            _ => return None,
        };
        Some(format!("{call}();"))
    }
}
