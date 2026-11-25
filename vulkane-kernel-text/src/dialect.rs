// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a target language supplies to the walker.

use unpopped::exec::{ExecId, Scope};
use unpopped::ir::{ArithOp, BinaryOp};
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
