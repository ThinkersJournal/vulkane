// SPDX-License-Identifier: MIT OR Apache-2.0
//! Expressions: lowered through the language-neutral `lower_dag` with the dialect's spellers.

use unpopped::backend::{Decline, DeclinedOp, LowerError, Lowering, Spelling, lower_dag};
use unpopped::exec::ExecId;
use unpopped::ir::{
    ArithOp, BinaryOp, CastSpec, ExprDag, RoundMode, Saturation, ScalarExpr, UnaryOp,
};
use unpopped::kernel::KernelFn;
use unpopped::types::Ty;
use unpopped_vocab::ElementKind;

use crate::dialect::{COMPARISONS, Dialect, operator};
use crate::scope::Scope;

fn declined(why: impl Into<String>) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp {
        op: DeclinedOp::Kernel,
        why: why.into(),
    }))
}

/// The seams `lower_dag` calls, over one dialect and one kernel.
struct Spell<'a> {
    d: &'a dyn Dialect,
    kernel: &'a KernelFn,
    sc: &'a Scope,
}

impl Spell<'_> {
    fn exec(&self, id: ExecId) -> Result<Spelling, LowerError> {
        let signed = self.sc.signed;
        match launch_const(self.sc, id) {
            // A wide or float kernel keeps an execution id the `u32` the IR types it as.
            Some(v) if self.d.wide() || self.sc.float => Ok(Spelling::Spelled(format!("{v}u"))),
            Some(v) => Ok(Spelling::Spelled(self.d.lit(i64::from(v), signed))),
            None => match self.d.exec(id, signed) {
                Some(s) => Ok(Spelling::Spelled(s)),
                None => declined(format!("execution id {id:?} is not lowered")),
            },
        }
    }

    fn binary(&self, op: BinaryOp, a: &str, b: &str) -> Result<Spelling, LowerError> {
        if self.sc.float {
            return self.float_binary(op, a, b);
        }
        match self.d.binary(op, a, b, self.sc.signed) {
            Some(s) => Ok(Spelling::Spelled(s)),
            None => declined(format!(
                "binary op {op:?} has no integer spelling in {}",
                self.d.name()
            )),
        }
    }

    /// A float comparison (`1.0`/`0.0`, `!=` unordered, the rest ordered, as C) or NaN-propagating
    /// `Max`/`Min`; anything else is a decline by name.
    fn float_binary(&self, op: BinaryOp, a: &str, b: &str) -> Result<Spelling, LowerError> {
        if let Some(sym) = operator(COMPARISONS, op) {
            let (one, zero) = (self.d.lit_float(1.0), self.d.lit_float(0.0));
            if let (Some(one), Some(zero)) = (one, zero) {
                return Ok(Spelling::Spelled(format!(
                    "({a} {sym} {b} ? {one} : {zero})"
                )));
            }
        }
        let minmax = match op {
            BinaryOp::Max => self.d.float_minmax(true, a, b),
            BinaryOp::Min => self.d.float_minmax(false, a, b),
            _ => None,
        };
        match minmax {
            Some(s) => Ok(Spelling::Spelled(s)),
            None => declined(format!(
                "float binary op {op:?} is not lowered in {} (compares and NaN-propagating Max/Min only: the others have no Vulkan accuracy bound shown no looser than the contract tier)",
                self.d.name()
            )),
        }
    }

    fn unary(&self, op: UnaryOp, x: &str) -> Result<Spelling, LowerError> {
        match self.sc.float.then(|| self.d.float_unary(op, x)).flatten() {
            Some(s) => Ok(Spelling::Spelled(s)),
            None if self.sc.float => declined(format!(
                "float unary op {op:?} is not lowered in {} (exact ops only: Neg, Abs, Sqr, Floor, Ceil, Trunc, Round; Sqrt, Recip, Rsqrt and the transcendentals have a Vulkan bound looser than the contract tier)",
                self.d.name()
            )),
            None => declined(format!("unary op {op:?} is not lowered (integer kernels)")),
        }
    }

    /// A float literal, exact at `f32`, or a decline.
    fn constant(&self, v: f64) -> Result<Spelling, LowerError> {
        let at_width = if self.d.half() {
            exact_half(v)
        } else {
            self.d.double() || f64::from(v as f32) == v
        };
        let exact = self.sc.float && v.is_finite() && at_width;
        match exact.then(|| self.d.lit_float(v)).flatten() {
            Some(s) => Ok(Spelling::Spelled(s)),
            None if self.sc.float => declined(format!(
                "the constant {v} is not a finite value exact at the kernel's float width"
            )),
            None => declined("a float constant in an integer kernel; use ConstInt"),
        }
    }

    fn arith(&self, op: ArithOp, a: &str, b: &str) -> Result<Spelling, LowerError> {
        if self.sc.float && !matches!(op, ArithOp::Add | ArithOp::Sub | ArithOp::Mul) {
            return declined(format!(
                "float arithmetic {op:?} is not lowered (Div is bounded at 2.5 ULP by Vulkan, looser than the oracle's correctly rounded `/`)"
            ));
        }
        match self.d.arith(op, a, b) {
            Some(s) => Ok(Spelling::Spelled(s)),
            None => declined(format!(
                "arithmetic {op:?} is not lowered (integer division is not total here)"
            )),
        }
    }

    /// A wrapping integer `Cast` to the compute dtype, or a decline.
    fn cast(&self, x: &str, from: ElementKind, spec: CastSpec) -> Result<Spelling, LowerError> {
        if self.sc.float {
            return self.float_cast(x, from, spec);
        }
        if self.sc.narrow && !self.d.wide() {
            return self.narrow_cast(x, from, spec);
        }
        if spec.to != self.kernel.compute
            || spec.sat != Saturation::Wrap
            || spec.round != RoundMode::NearestEven
        {
            return declined(format!(
                "a Cast to {:?} with {:?}/{:?} is not lowered (wrapping, to the compute dtype only)",
                spec.to, spec.round, spec.sat
            ));
        }
        match self.d.cast_to_compute(x, from) {
            Some(s) => Ok(Spelling::Spelled(s)),
            None => declined(format!(
                "a Cast from {from:?} is not lowered in {}",
                self.d.name()
            )),
        }
    }

    /// A wrapping `Cast` between the 32-bit compute dtype, its 32-bit twin and the 8/16-bit integers.
    fn narrow_cast(
        &self,
        x: &str,
        from: ElementKind,
        spec: CastSpec,
    ) -> Result<Spelling, LowerError> {
        use ElementKind::{I8, I16, I32, U8, U16, U32};
        let known = |t| matches!(t, I8 | U8 | I16 | U16 | I32 | U32);
        if !known(from) || !known(spec.to) {
            return declined(format!(
                "a Cast {from:?} -> {:?} is not lowered in {} (8-, 16- and 32-bit integers only)",
                spec.to,
                self.d.name()
            ));
        }
        if spec.sat != Saturation::Wrap {
            return declined(format!(
                "an integer Cast with {:?} is not lowered: only a wrapping one is",
                spec.sat
            ));
        }
        match self.d.convert(x, from, spec.to, spec.round) {
            Some(s) => Ok(Spelling::Spelled(s)),
            None => declined(format!(
                "a Cast {from:?} -> {:?} is not lowered in {}",
                spec.to,
                self.d.name()
            )),
        }
    }

    /// A `Cast` in a float kernel, against `unpopped`'s `cast_ref.rs`: integer to integer wraps, integer
    /// to float and `F64` to `F32` round to nearest even, `F32` to `F64` is exact, and float to
    /// integer rounds per the mode, maps NaN to 0 and clamps. Anything else is a decline by name.
    fn float_cast(
        &self,
        x: &str,
        from: ElementKind,
        spec: CastSpec,
    ) -> Result<Spelling, LowerError> {
        use ElementKind::{F32, F64, I32, U32};
        let to = spec.to;
        let known = |t| matches!(t, F32 | F64 | I32 | U32);
        if !known(from) || !known(to) {
            return declined(format!(
                "a Cast {from:?} -> {to:?} is not lowered in {} (F32, F64, I32 and U32 only)",
                self.d.name()
            ));
        }
        if from == to {
            return Ok(Spelling::Spelled(x.to_string()));
        }
        let float = |t| matches!(t, F32 | F64);
        let nearest = spec.round == RoundMode::NearestEven && spec.sat == Saturation::Wrap;
        let why = match (float(from), float(to)) {
            (false, false) => (spec.sat != Saturation::Wrap)
                .then(|| "an integer Cast with a clamp is not lowered: only a wrapping one is".to_string()),
            (false, true) => (!nearest).then(|| format!("an int to float Cast with {:?}/{:?} is not lowered: only NearestEven with Wrap is", spec.round, spec.sat)),
            (true, true) => (from == F64 && !nearest).then(|| format!("a narrowing float Cast with {:?}/{:?} is not lowered: only NearestEven with Wrap is", spec.round, spec.sat)),
            (true, false) => (spec.sat != Saturation::Clamp
                || !matches!(
                    spec.round,
                    RoundMode::NearestEven
                        | RoundMode::TowardZero
                        | RoundMode::TowardNegInf
                        | RoundMode::TowardPosInf
                ))
            .then(|| format!("a float to {to:?} Cast rounding {:?} with {:?} is not lowered: only Clamp with NearestEven, TowardZero, TowardNegInf or TowardPosInf is", spec.round, spec.sat)),
        };
        if let Some(why) = why {
            return declined(why);
        }
        // A conversion that rounds needs the module to say how, at the result's width.
        if float(to) && !(float(from) && from == F32) {
            self.sc
                .rte
                .borrow_mut()
                .insert(if to == F64 { 64 } else { 32 });
        }
        match self.d.convert(x, from, to, spec.round) {
            Some(s) => Ok(Spelling::Spelled(s)),
            None => declined(format!(
                "a Cast {from:?} -> {to:?} is not lowered in {}",
                self.d.name()
            )),
        }
    }

    fn const_int(&self, v: i64, t: ElementKind) -> Result<Spelling, LowerError> {
        if self.sc.float {
            // A float kernel's integers are the `u32` index arithmetic around its buffers.
            return match (t, u32::try_from(v), i32::try_from(v)) {
                (ElementKind::U32, Ok(n), _) => Ok(Spelling::Spelled(format!("{n}u"))),
                (ElementKind::I32, _, Ok(n)) if n >= 0 => Ok(Spelling::Spelled(n.to_string())),
                (ElementKind::I32, _, Ok(n)) => Ok(Spelling::Spelled(format!("({n})"))),
                _ => declined(format!(
                    "ConstInt({v}, {t:?}) is not a 32-bit integer leaf in a float kernel"
                )),
            };
        }
        let signed = self.sc.signed;
        let fits = match (self.d.wide(), signed) {
            (true, true) => true,
            (true, false) => v >= 0,
            (false, true) => i32::try_from(v).is_ok(),
            (false, false) => u32::try_from(v).is_ok(),
        };
        if t == self.kernel.compute && fits {
            Ok(Spelling::Spelled(self.d.lit(v, signed)))
        } else {
            declined(format!(
                "ConstInt({v}, {t:?}) is not exact at {:?}",
                self.kernel.compute
            ))
        }
    }
}

/// Whether `v` is exactly an IEEE half: within 65504, a multiple of 2^-24 (the smallest subnormal)
/// and with at most 11 significant bits.
fn exact_half(v: f64) -> bool {
    let a = v.abs();
    if a == 0.0 {
        return true;
    }
    if a > 65504.0 || (a * 16_777_216.0).fract() != 0.0 {
        return false;
    }
    let ulp = 2f64.powf(a.log2().floor().max(-14.0) - 10.0);
    (a / ulp).fract() == 0.0
}

/// The hoisted declarations (in order) and the root spelling of `e`.
pub(crate) fn lower(
    d: &dyn Dialect,
    kernel: &KernelFn,
    sc: &Scope,
    e: &ScalarExpr,
) -> Result<(Vec<String>, String), LowerError> {
    let (sp, signed) = (Spell { d, kernel, sc }, sc.signed);
    let tys: Vec<Ty> = kernel.locals.iter().map(|l| l.ty()).collect();
    let leaf = |k: u8| {
        declined(format!(
            "an operand leaf Input({k}) is not a kernel expression"
        ))
    };
    let local = |i: u32| Ok(Spelling::Spelled(format!("l{i}")));
    let unary = |op, x: String| sp.unary(op, &x);
    let binary = |op, a: String, b: String| sp.binary(op, &a, &b);
    let arith = |op, a: String, b: String| sp.arith(op, &a, &b);
    let select =
        |c: String, a: String, b: String| Ok(Spelling::Spelled(d.select(&c, &a, &b, signed)));
    let constant = |v: f64| sp.constant(v);
    let const_int = |v: i64, t: ElementKind| sp.const_int(v, t);
    let exec = |id| sp.exec(id);
    let temp = |ty: &str, name: &str, rhs: &str| {
        if sc.float || sc.narrow {
            d.temp_typed(ty, name, rhs)
        } else {
            d.temp(name, rhs, signed)
        }
    };
    let cast = |x: String, from: ElementKind, spec: CastSpec| sp.cast(&x, from, spec);
    let type_name = |t: ElementKind| {
        if sc.float || sc.narrow {
            d.type_name(t)
        } else {
            (t == kernel.compute).then(String::new)
        }
    };
    let builder = Lowering::builder(&leaf, &unary, &binary)
        .arith(&arith)
        .select(&select)
        .constant(&constant)
        .const_int(&const_int)
        .temp(&temp)
        .local(&local)
        .exec(&exec)
        .local_types(&tys)
        .compute_dtype(kernel.compute);
    // Only a 64-bit kernel has a `Cast` seam: its execution ids and narrower operands are reached
    // through one. A 32-bit kernel keeps the walker's typed decline.
    let lowering = if d.wide() || sc.float || sc.narrow {
        builder.cast(&cast).type_name(&type_name).build()
    } else {
        builder.build()
    };
    lower_dag(&ExprDag::from_expr(e), d.name(), &lowering)
}

/// `BlockDim` and `GridDim` are constants of the launch.
fn launch_const(sc: &Scope, id: ExecId) -> Option<u32> {
    use unpopped::exec::ExecId::{BlockDim, GridDim};
    match id {
        BlockDim(a) if a < 3 => Some(sc.block[usize::from(a)]),
        GridDim(a) if a < 3 => Some(sc.grid[usize::from(a)]),
        _ => None,
    }
}
