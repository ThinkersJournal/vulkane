// SPDX-License-Identifier: MIT OR Apache-2.0
//! The scalar family a dtype lowers to: a leaf module (it reads only `unpopped_vocab`), so `accuracy` and `ops` do not
//! depend on each other (CodeRipper MOD-002, ROADMAP row 9d).

use unpopped_vocab::ElementKind;

/// Which scalar family a dtype lowers to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    F32,
    F64,
    I32,
    U32,
    /// 64-bit integers: kernels only (`Backend::lower_kernel`); a plan never lowers them.
    I64,
    U64,
    /// 8- and 16-bit integers: storage and local types of a kernel (a `Cast` widens them to, and
    /// narrows them from, the 32-bit compute dtype); never a compute dtype and never a plan dtype.
    I8,
    U8,
    I16,
    U16,
    /// IEEE half: a storage and local type of a float kernel (reached through a float `Cast`); never
    /// a compute dtype and never a plan dtype.
    F16,
}

impl Family {
    pub(crate) fn of(dt: ElementKind) -> Option<Family> {
        match dt {
            ElementKind::F32 | ElementKind::F32Strict => Some(Family::F32),
            ElementKind::F64 => Some(Family::F64),
            ElementKind::I32 => Some(Family::I32),
            ElementKind::U32 => Some(Family::U32),
            ElementKind::I64 => Some(Family::I64),
            ElementKind::U64 => Some(Family::U64),
            ElementKind::I8 => Some(Family::I8),
            ElementKind::U8 => Some(Family::U8),
            ElementKind::I16 => Some(Family::I16),
            ElementKind::U16 => Some(Family::U16),
            ElementKind::F16 => Some(Family::F16),
            _ => None,
        }
    }
    pub(crate) fn is_float(self) -> bool {
        matches!(self, Family::F32 | Family::F64 | Family::F16)
    }
    /// The 16-bit float, which needs `Float16` and is never a compute dtype.
    pub(crate) fn is_half(self) -> bool {
        self == Family::F16
    }
    pub(crate) fn bytes(self) -> u32 {
        self.bits() / 8
    }
    /// The width of the compute dtype in bits.
    pub(crate) fn bits(self) -> u32 {
        match self {
            Family::F64 | Family::I64 | Family::U64 => 64,
            Family::I8 | Family::U8 => 8,
            Family::I16 | Family::U16 | Family::F16 => 16,
            _ => 32,
        }
    }
    /// An 8- or 16-bit integer, which needs `Int8`/`Int16` (and a storage capability in a buffer).
    pub(crate) fn is_narrow(self) -> bool {
        matches!(self, Family::I8 | Family::U8 | Family::I16 | Family::U16)
    }
    /// Whether an `ExecId` (typed `U32` in the IR) stays a raw `u32` here: the 64-bit integer and the
    /// float kernels, which reach their dtype through a `Cast` or index with the id directly.
    pub(crate) fn keeps_u32_ids(self) -> bool {
        self.is_wide() || self.is_float()
    }
    /// A 64-bit integer, which needs the `Int64` capability.
    pub(crate) fn is_wide(self) -> bool {
        matches!(self, Family::I64 | Family::U64)
    }
    /// A signed integer family.
    pub(crate) fn is_signed(self) -> bool {
        matches!(self, Family::I32 | Family::I64 | Family::I8 | Family::I16)
    }
}
