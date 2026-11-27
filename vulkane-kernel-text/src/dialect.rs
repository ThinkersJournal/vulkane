// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a target language supplies to the walker.

use unpopped::exec::{ExecId, Scope};
use unpopped::ir::{ArithOp, BinaryOp, RoundMode, UnaryOp};
use unpopped::kernel::{AtomicOp, BufferRole, MemoryOrder};
use unpopped_vocab::ElementKind;

/// The comparisons both dialects spell with the same operator text; each gives `1` or `0` in the dtype.
pub const COMPARISONS: &[(BinaryOp, &str)] = &[
    (BinaryOp::CmpEq, "=="),
    (BinaryOp::CmpNe, "!="),
    (BinaryOp::CmpLt, "<"),
    (BinaryOp::CmpLe, "<="),
    (BinaryOp::CmpGt, ">"),
    (BinaryOp::CmpGe, ">="),
];

/// The bitwise operators both dialects spell with the same operator text.
pub const BIT_OPS: &[(BinaryOp, &str)] = &[
    (BinaryOp::BitAnd, "&"),
    (BinaryOp::BitOr, "|"),
    (BinaryOp::BitXor, "^"),
];

/// The operator text `op` has in `table`, if any.
#[must_use]
pub fn operator(table: &[(BinaryOp, &'static str)], op: BinaryOp) -> Option<&'static str> {
    table.iter().find(|(o, _)| *o == op).map(|(_, s)| *s)
}

/// A buffer or shared array the prologue declares.
#[derive(Clone, Debug)]
pub struct Decl {
    /// The identifier the body uses (`b0`, `a0`).
    pub name: String,
    /// The binding slot of a buffer; unused for a shared array.
    pub binding: u32,
    /// The access of a buffer; `ReadWrite` for a shared array.
    pub role: BufferRole,
    /// The element type: the compute dtype, except in a float kernel, whose buffers may also hold
    /// the other float width and the 32-bit integers a `Cast` converts.
    pub elem: ElementKind,
    /// The element count of a shared array; `0` for a buffer.
    pub len: u32,
    /// `true` for a per-thread (`Private`) array, `false` for a buffer or a workgroup array.
    pub private: bool,
    /// `true` when the dialect types atomics ([`Dialect::typed_atomics`]) and the kernel applies an
    /// atomic to this buffer or array: every access to it is then an atomic one.
    pub atomic: bool,
    /// For a scalar parameter lowered as a specialization constant (`SpecId` = `binding`): the local
    /// that reads it. `None` for a buffer or an array.
    pub spec_local: Option<u32>,
    /// For an array whose extent is a scalar parameter: the identifier of that specialization
    /// constant (`len` is then `0`).
    pub len_spec: Option<String>,
}

/// Everything the prologue needs.
#[derive(Clone, Debug)]
pub struct Layout {
    /// The kernel's name.
    pub name: String,
    /// `true` for an `I32` kernel, `false` for `U32`.
    pub signed: bool,
    /// The workgroup size.
    pub block: [u32; 3],
    /// The storage buffers, in parameter order.
    pub buffers: Vec<Decl>,
    /// The workgroup and per-thread arrays, in allocation order (`Decl::private` tells them apart).
    pub shared: Vec<Decl>,
    /// The `(local, constant identifier)` of each scalar parameter lowered as a specialization
    /// constant; the local starts at its value instead of zero.
    pub spec_locals: Vec<(u32, String)>,
    /// How many locals the body declares (`l0..`).
    pub locals: usize,
    /// The element type of each local (the compute dtype, except in a float kernel).
    pub local_elems: Vec<ElementKind>,
    /// The widths (16/32/64) at which the body does a rounding float conversion, so the module must
    /// declare the round-to-nearest-even mode there.
    pub rte: Vec<u32>,
    /// Whether any statement is a subgroup barrier or fence (a GLSL extension line).
    pub subgroup: bool,
}

impl Layout {
    /// The specialization constant `local` starts at, when it is a scalar parameter's local.
    #[must_use]
    pub fn spec_init(&self, local: usize) -> Option<&str> {
        self.spec_locals
            .iter()
            .find(|(l, _)| usize::try_from(*l).is_ok_and(|l| l == local))
            .map(|(_, n)| n.as_str())
    }
}

/// The spellings of one target language.
///
/// A method that cannot spell something returns `None` (or an `Err` text), and the walker turns that
/// into a typed decline naming it; nothing is approximated.
pub trait Dialect {
    /// The backend name used in decline texts (`glsl`, `wgsl`).
    fn name(&self) -> &'static str;
    /// Whether this dialect instance lowers 64-bit integer kernels (`I64`/`U64`). The default is no:
    /// such a kernel is then a typed decline naming the dialect. A dialect that says yes types every
    /// spelling at 64 bits and spells [`Self::cast_to_compute`]; an execution id stays the `u32` the IR
    /// types it as, reached through a `Cast`.
    fn wide(&self) -> bool {
        false
    }
    /// Why this dialect cannot lower a float kernel at all, named in the decline of one (`None`: no
    /// reason beyond the generic one). Only asked of a dialect whose [`Self::float`] is false.
    fn float_refusal(&self) -> Option<&'static str> {
        None
    }
    /// Why this dialect has no 8- or 16-bit integer buffers or locals at all, named in the decline of a
    /// kernel that holds one (`None`: no reason beyond the generic one). Only asked of a dialect whose
    /// [`Self::narrow_ints`] is false.
    fn narrow_refusal(&self) -> Option<&'static str> {
        None
    }
    /// Whether an `I32`/`U32` kernel may hold `I8`/`U8`/`I16`/`U16` buffers and locals (reached by a
    /// wrapping `Cast`, spelled by [`Self::convert`]). The default is no: such a kernel is then a typed
    /// decline naming the buffer.
    fn narrow_ints(&self) -> bool {
        false
    }
    /// Whether this dialect instance lowers float kernels (`F32`, or `F64` when [`Self::double`]). The default is no: such a kernel is then a
    /// typed decline naming the dialect. A dialect that says yes spells [`Self::lit_float`],
    /// [`Self::float_unary`] and [`Self::float_minmax`], keeps every float add, subtract and multiply
    /// unfused ([`Self::temp_typed`]) and keeps an execution id the `u32` the IR types it as.
    fn float(&self) -> bool {
        false
    }
    /// Whether the float kernel this instance lowers is `F64` rather than `F32` (only asked when
    /// [`Self::float`] is true).
    fn double(&self) -> bool {
        false
    }
    /// Whether the float kernel this instance lowers is `F16` (only asked when [`Self::float`] is
    /// true): literals must then be exact halves.
    fn half(&self) -> bool {
        false
    }
    /// Whether one kernel may use float types of two widths (`F32` and `F64` together). The default
    /// is yes; a dialect whose float-control declarations cannot name two widths says no, and such a
    /// kernel is a typed decline.
    fn mixed_float_widths(&self) -> bool {
        true
    }
    /// A float literal for `v`, already known to be a finite value exact at the kernel's float width, or `None`.
    fn lit_float(&self, _v: f64) -> Option<String> {
        None
    }
    /// `x` of type `from` converted to `to` (both `F32`, `F64`, `I32` or `U32` in a float kernel), the
    /// rounding of a float to integer conversion being `round`; the walker has already checked the
    /// combination is one it lowers. A float to integer conversion maps NaN to 0 and clamps to the
    /// destination range, an integer to float or float narrowing rounds to nearest even.
    fn convert(
        &self,
        _x: &str,
        _from: ElementKind,
        _to: ElementKind,
        _round: RoundMode,
    ) -> Option<String> {
        None
    }
    /// A float unary op on `x`, or `None` when there is no exact spelling (the walker declines by name).
    fn float_unary(&self, _op: UnaryOp, _x: &str) -> Option<String> {
        None
    }
    /// NaN-propagating max (`is_max`) or min of `a` and `b`, `a` on ties; `None` when not spelled.
    fn float_minmax(&self, _is_max: bool, _a: &str, _b: &str) -> Option<String> {
        None
    }
    /// The type name of `t` in a float kernel (`float`, `uint`, `int`), or `None`.
    fn type_name(&self, _t: ElementKind) -> Option<String> {
        None
    }
    /// A boolean spelling of `value != 0` for a value of element type `t` in a float kernel (a float
    /// compares against a float zero, an integer leaf against an integer zero).
    fn truthy_of(&self, value: &str, _t: ElementKind) -> String {
        format!("({value} != 0)")
    }
    /// A hoisted temporary of type `ty` (from [`Self::type_name`]) holding `rhs`. A float one must be
    /// one a compiler may not fuse or reassociate the operations of (GLSL `precise`): the value of a
    /// float expression is pinned in such a temporary before it is stored.
    fn temp_typed(&self, ty: &str, name: &str, rhs: &str) -> String {
        format!("const {ty} {name} = {rhs};")
    }
    /// `x`, of integer type `from`, converted (wrapping) to the compute dtype, or `None` when there is
    /// no spelling. Only asked of a dialect whose [`Self::wide`] is true.
    fn cast_to_compute(&self, _x: &str, _from: ElementKind) -> Option<String> {
        None
    }
    /// The header, buffer and shared declarations and the opening of the entry function, with the
    /// locals declared (one line each) at the top of the entry.
    fn prologue(&self, layout: &Layout) -> String;
    /// The closing of the entry function.
    fn epilogue(&self) -> &'static str;
    /// An integer literal in the compute dtype.
    fn lit(&self, v: i64, signed: bool) -> String;
    /// The element `idx` of the buffer or shared array `name`.
    fn index(&self, name: &str, idx: &str, is_buffer: bool) -> String;
    /// The spelling of an execution id, or `None` when the dialect has none.
    fn exec(&self, id: ExecId, signed: bool) -> Option<String>;
    /// A binary op on two integers, or `None` when there is no spelling.
    fn binary(&self, op: BinaryOp, a: &str, b: &str, signed: bool) -> Option<String>;
    /// `a + b`, `a - b`, `a * b` (wrapping); `None` for a division.
    fn arith(&self, op: ArithOp, a: &str, b: &str) -> Option<String> {
        let sym = match op {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            _ => return None,
        };
        Some(format!("({a} {sym} {b})"))
    }
    /// `cond != 0 ? a : b`.
    fn select(&self, cond: &str, a: &str, b: &str, signed: bool) -> String;
    /// A boolean spelling of `value != 0`.
    fn truthy(&self, value: &str, signed: bool) -> String {
        format!("({value} != {})", self.lit(0, signed))
    }
    /// A hoisted temporary declaration line (no indentation, no newline).
    fn temp(&self, name: &str, rhs: &str, signed: bool) -> String;
    /// The `For` counter's bound and step constants, declared once before the loop.
    fn bound(&self, name: &str, rhs: &str, signed: bool) -> String;
    /// A statement that synchronizes `scope` (and orders memory), or `None` when there is none.
    fn barrier(&self, scope: Scope) -> Option<String>;
    /// An expression applying atomic `op` to the element `elem` with `value`, giving the old value;
    /// `None` when the dialect has no spelling (the default).
    fn atomic_rmw(&self, _op: AtomicOp, _elem: &str, _value: &str) -> Option<String> {
        None
    }
    /// An expression compare-exchanging the element `elem`, giving the value found; `None` when
    /// the dialect has no spelling (the default).
    fn atomic_cas(&self, _elem: &str, _expected: &str, _desired: &str) -> Option<String> {
        None
    }
    /// Whether the dialect declares an atomically accessed buffer or array with a different type (WGSL
    /// `atomic<u32>`), so plain access to it goes through [`Self::atomic_load`] / [`Self::atomic_store`].
    fn typed_atomics(&self) -> bool {
        false
    }
    /// Whether an atomic at `scope` can be spelled (the default: every scope).
    fn atomic_scope_ok(&self, _scope: Scope) -> bool {
        true
    }
    /// An expression reading the element `elem` of an atomically typed buffer or array.
    fn atomic_load(&self, _elem: &str) -> Option<String> {
        None
    }
    /// A statement writing `value` to the element `elem` of an atomically typed buffer or array.
    fn atomic_store(&self, _elem: &str, _value: &str) -> Option<String> {
        None
    }
    /// The lines compare-exchanging `elem` and writing the value found to `dest`, if any. The default
    /// is one assignment of [`Self::atomic_cas`]; a dialect whose exchange can fail spuriously
    /// overrides it with a retry loop, because the IR's compare-exchange cannot.
    fn atomic_cas_lines(
        &self,
        elem: &str,
        expected: &str,
        desired: &str,
        dest: Option<&str>,
    ) -> Option<Vec<String>> {
        let call = self.atomic_cas(elem, expected, desired)?;
        Some(vec![match dest {
            Some(l) => format!("{l} = {call};"),
            None => format!("{call};"),
        }])
    }
    /// A memory fence at `scope`, or `None` when the dialect has none without a barrier.
    fn fence(&self, scope: Scope, order: MemoryOrder) -> Option<String>;
}
