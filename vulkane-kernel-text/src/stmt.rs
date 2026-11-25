// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statements and structured control flow.
//!
//! `If` and `While` map to the native statements; a `While` condition that needs hoisted temporaries
//! is written `while (true) { temps; if (!cond) { break; } body }` so they run every pass. `For` is a
//! native `for (init; test; update)` inside its own scope that declares the bounds once, so a
//! `continue` still runs the update. Statements after a `Break`, `Continue` or `Return` in a block
//! are dead and are not written.

use unpopped::backend::LowerError;
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BarrierKind, Block, KernelFn, ParamKind, Place, Stmt};
use unpopped::types::{Ty, infer_ty_in};
use unpopped_vocab::ElementKind;

use crate::dialect::Dialect;
use crate::expr;
use crate::scope::{Scope, place_name, place_ok, refuse};

mod atomic;

/// The text being written.
struct W<'a> {
    d: &'a dyn Dialect,
    kernel: &'a KernelFn,
    sc: &'a Scope,
    out: &'a mut String,
    depth: usize,
    loops: u32,
    pins: u32,
}

/// Write the body of `kernel` at one level of indentation.
pub(crate) fn emit_body(
    d: &dyn Dialect,
    kernel: &KernelFn,
    sc: &Scope,
    out: &mut String,
) -> Result<(), LowerError> {
    let mut w = W {
        d,
        kernel,
        sc,
        out,
        depth: 1,
        loops: 0,
        pins: 0,
    };
    w.block(&kernel.body).map(drop)
}

/// The variant name of a statement: the `Debug` text up to the first non-identifier character.
fn stmt_name(s: &Stmt) -> String {
    let text = format!("{s:?}");
    text.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

impl W<'_> {
    fn line(&mut self, text: &str) {
        for _ in 0..self.depth {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// [`Self::line`] as a statement result.
    fn emit(&mut self, text: &str) -> Result<(), LowerError> {
        self.line(text);
        Ok(())
    }

    fn no(&self, what: impl std::fmt::Display) -> LowerError {
        refuse(self.sc.backend, self.kernel, what)
    }

    /// The root spelling of `e`, its hoisted declarations written first.
    fn expr(&mut self, e: &ScalarExpr) -> Result<String, LowerError> {
        let (prelude, root) = expr::lower(self.d, self.kernel, self.sc, e)?;
        for p in prelude.iter().filter(|p| !p.is_empty()) {
            self.line(p);
        }
        Ok(root)
    }

    /// In a float kernel, `value` held in a temporary no compiler may fuse the operations of (a
    /// stored `a * b + a` must round twice, as the oracle does); otherwise `value` itself.
    fn pinned(&mut self, value: String, place: &Place) -> Result<String, LowerError> {
        if !self.sc.float {
            return Ok(value);
        }
        // The value has the element type of the place it is stored to.
        let elem = match place {
            Place::Logical { buffer, .. } => {
                match &self.kernel.params[usize::try_from(*buffer).expect("buffer id fits usize")]
                    .kind
                {
                    ParamKind::Buffer { elem, .. } => *elem,
                    _ => self.kernel.compute,
                }
            }
            _ => self.kernel.compute,
        };
        self.pinned_elem(value, elem)
    }

    /// `value`, of element type `elem`, held in a temporary (see [`Self::pinned`]).
    fn pinned_elem(&mut self, value: String, elem: ElementKind) -> Result<String, LowerError> {
        let ty = self.d.type_name(elem);
        let Some(ty) = ty else {
            return Err(self.no("a float kernel needs a float type name"));
        };
        let name = format!("p{}", self.pins);
        self.pins += 1;
        let decl = self.d.temp_typed(&ty, &name, &value);
        self.line(&decl);
        Ok(name)
    }

    /// The element type of `e` in this kernel.
    fn elem_of(&self, e: &ScalarExpr) -> Result<ElementKind, LowerError> {
        let tys: Vec<Ty> = self.kernel.locals.iter().map(|l| l.ty()).collect();
        match infer_ty_in(e, self.kernel.compute, &tys) {
            Ok(Ty::Scalar(t)) => Ok(t),
            _ => Err(self.no("a condition whose type cannot be inferred")),
        }
    }

    /// The boolean text of `cond != 0`. A float kernel's condition is a float (pinned first, so its
    /// multiplies and adds are not fused) or an integer leaf (a loop counter, an execution id).
    fn condition(&mut self, cond: &ScalarExpr) -> Result<String, LowerError> {
        let c = self.expr(cond)?;
        if !self.sc.float {
            return Ok(self.d.truthy(&c, self.sc.signed));
        }
        let elem = self.elem_of(cond)?;
        let c = if matches!(elem, ElementKind::F32 | ElementKind::F64) {
            self.pinned_elem(c, elem)?
        } else {
            c
        };
        Ok(self.d.truthy_of(&c, elem))
    }

    /// Lower a block; `true` when it ended in a terminator.
    fn block(&mut self, block: &Block) -> Result<bool, LowerError> {
        for s in &block.0 {
            self.stmt(s)?;
            if matches!(s, Stmt::Break | Stmt::Continue | Stmt::Return) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `body` one level deeper.
    fn nested(&mut self, body: &Block) -> Result<(), LowerError> {
        self.depth += 1;
        let r = self.block(body);
        self.depth -= 1;
        r.map(drop)
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { .. } | Stmt::Assign { .. } | Stmt::Load { .. } | Stmt::Store { .. } => self.data(s),
            Stmt::MaskedLoad { .. } | Stmt::MaskedStore { .. } => self.masked(s),
            Stmt::Barrier { .. } | Stmt::Fence { .. } => self.sync(s),
            Stmt::AtomicRmw { .. } | Stmt::AtomicCas { .. } => self.atomic(s),
            Stmt::If { .. } | Stmt::For { .. } | Stmt::While { .. } => self.control(s),
            Stmt::Break => self.emit("break;"),
            Stmt::Continue => self.emit("continue;"),
            Stmt::Return => self.emit("return;"),
            other => Err(self.no(format!(
                "`Stmt::{}` is not lowered (v1: Let, Load, Assign, Store, Barrier, Fence, atomics where the dialect spells them, control flow)",
                stmt_name(other)
            ))),
        }
    }

    /// The element `place` names, as an lvalue or rvalue.
    fn element(&mut self, place: &Place) -> Result<String, LowerError> {
        if !place_ok(place) {
            return Err(self.no(format!(
                "place {place:?} is not lowered (v1: one-coordinate buffer or shared element)"
            )));
        }
        let (name, idx, is_buffer) = match place {
            Place::Logical { buffer, coords } => (format!("b{buffer}"), &coords[0], true),
            Place::Alloc { alloc, index } => (format!("a{alloc}"), index, false),
            _ => unreachable!("place_ok admits two forms"),
        };
        let idx = self.expr(idx)?;
        Ok(self.d.index(&name, &idx, is_buffer))
    }

    /// Whether `place` addresses an atomically typed buffer or array.
    fn is_atomic(&self, place: &Place) -> bool {
        place_name(place).is_some_and(|n| self.sc.atomic.contains(&n))
    }

    /// The read of `elem`: itself, or the dialect's atomic load when the buffer is atomically typed.
    fn read_of(&self, place: &Place, elem: String) -> Result<String, LowerError> {
        if !self.is_atomic(place) {
            return Ok(elem);
        }
        self.d
            .atomic_load(&elem)
            .ok_or_else(|| self.no("an atomically typed buffer has no atomic load here"))
    }

    /// The write of `value` to `elem`: a plain assignment, or the dialect's atomic store.
    fn write_of(&self, place: &Place, elem: &str, value: &str) -> Result<String, LowerError> {
        if !self.is_atomic(place) {
            return Ok(format!("{elem} = {value};"));
        }
        self.d
            .atomic_store(elem, value)
            .ok_or_else(|| self.no("an atomically typed buffer has no atomic store here"))
    }

    fn data(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init: v } | Stmt::Assign { local, value: v } => {
                let v = self.expr(v)?;
                self.emit(&format!("l{local} = {v};"))
            }
            Stmt::Load { local, place } => {
                let e = self.element(place)?;
                let e = self.read_of(place, e)?;
                self.emit(&format!("l{local} = {e};"))
            }
            Stmt::Store { place, value } => {
                let e = self.element(place)?;
                let v = self.expr(value)?;
                let v = self.pinned(v, place)?;
                let text = self.write_of(place, &e, &v)?;
                self.emit(&text)
            }
            _ => unreachable!("`stmt` routes only data statements here"),
        }
    }

    /// Predicated access (ROADMAP 9e): a guarded plain access, so the memory is not touched where the mask is false
    /// (no bounds check, no race record: exactly the IR's meaning). `MaskedLoad` is
    /// `if (mask) { l = element; } else { l = fill; }`, `MaskedStore` is `if (mask) { element = value; }`; each goes
    /// through the plain `Load` / `Store` path, so an atomically typed buffer and a pinned float value behave as there.
    fn masked(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::MaskedLoad {
                local,
                place,
                mask,
                fill,
            } => {
                let t = self.condition(mask)?;
                self.line(&format!("if {t} {{"));
                self.guarded(&Stmt::Load {
                    local: *local,
                    place: place.clone(),
                })?;
                self.line("} else {");
                self.guarded(&Stmt::Assign {
                    local: *local,
                    value: fill.clone(),
                })?;
                self.emit("}")
            }
            Stmt::MaskedStore { place, value, mask } => {
                let t = self.condition(mask)?;
                self.line(&format!("if {t} {{"));
                self.guarded(&Stmt::Store {
                    place: place.clone(),
                    value: value.clone(),
                })?;
                self.emit("}")
            }
            _ => unreachable!("`stmt` routes only masked statements here"),
        }
    }

    /// One data statement one level deeper (the depth is restored on an error too).
    fn guarded(&mut self, s: &Stmt) -> Result<(), LowerError> {
        self.depth += 1;
        let r = self.data(s);
        self.depth -= 1;
        r
    }

    fn sync(&mut self, s: &Stmt) -> Result<(), LowerError> {
        let text = match s {
            Stmt::Barrier {
                scope,
                kind: BarrierKind::Full,
            } => self.d.barrier(*scope),
            Stmt::Fence { scope, order } => self.d.fence(*scope, *order),
            _ => None,
        };
        let Some(text) = text else {
            return Err(self.no(format!(
                "`Stmt::{}` at this scope or kind has no {} spelling",
                stmt_name(s),
                self.d.name()
            )));
        };
        self.emit(&text)
    }

    fn control(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::If { cond, then, els } => self.lower_if(cond, then, els),
            Stmt::While { cond, body } => self.lower_while(cond, body),
            Stmt::For {
                local,
                start,
                end,
                step,
                body,
                ..
            } => self.lower_for(*local, [start, end, step], body),
            _ => unreachable!("`stmt` routes only control statements here"),
        }
    }

    fn lower_if(&mut self, cond: &ScalarExpr, then: &Block, els: &Block) -> Result<(), LowerError> {
        let t = self.condition(cond)?;
        self.line(&format!("if {t} {{"));
        self.nested(then)?;
        if !els.0.is_empty() {
            self.line("} else {");
            self.nested(els)?;
        }
        self.emit("}")
    }

    fn lower_while(&mut self, cond: &ScalarExpr, body: &Block) -> Result<(), LowerError> {
        self.line("while (true) {");
        self.depth += 1;
        let c = self.condition(cond);
        let r = c.and_then(|t| {
            self.line(&format!("if (!{t}) {{ break; }}"));
            self.block(body).map(drop)
        });
        self.depth -= 1;
        r?;
        self.emit("}")
    }

    /// The loop test: `i < end` for a positive step, `i > end` for a negative one (`I32` only); a
    /// zero step is false.
    fn counted_test(&self, i: &str, e: &str, s: &str, signed: bool) -> String {
        let zero = if self.sc.float {
            // A float kernel's counter is an integer local, not the compute dtype.
            if signed { "0" } else { "0u" }.to_string()
        } else {
            self.d.lit(0, signed)
        };
        if signed {
            format!("(({s} > {zero} && {i} < {e}) || ({s} < {zero} && {i} > {e}))")
        } else {
            format!("({s} != {zero} && {i} < {e})")
        }
    }

    fn lower_for(
        &mut self,
        local: u32,
        [start, end, step]: [&ScalarExpr; 3],
        body: &Block,
    ) -> Result<(), LowerError> {
        let n = self.loops;
        self.loops += 1;
        self.line("{");
        self.depth += 1;
        let r = self.for_in_scope(local, n, [start, end, step], body);
        self.depth -= 1;
        r?;
        self.emit("}")
    }

    /// The counter local's element type and whether it is signed: the compute dtype's own, except in
    /// a float kernel, whose counters are `I32`/`U32` locals.
    fn counter(&self, local: u32) -> (ElementKind, bool) {
        if !self.sc.float {
            return (self.kernel.compute, self.sc.signed);
        }
        let kind = self.kernel.locals[usize::try_from(local).expect("local id fits usize")].elem;
        (kind, kind == ElementKind::I32)
    }

    fn for_in_scope(
        &mut self,
        local: u32,
        n: u32,
        [start, end, step]: [&ScalarExpr; 3],
        body: &Block,
    ) -> Result<(), LowerError> {
        let (kind, signed) = self.counter(local);
        let (start, end, step) = (self.expr(start)?, self.expr(end)?, self.expr(step)?);
        let (eb, sb) = (format!("e{n}"), format!("s{n}"));
        let (end_decl, step_decl) = if self.sc.float {
            let ty = self.d.type_name(kind).unwrap_or_default();
            (
                self.d.temp_typed(&ty, &eb, &end),
                self.d.temp_typed(&ty, &sb, &step),
            )
        } else {
            (
                self.d.bound(&eb, &end, signed),
                self.d.bound(&sb, &step, signed),
            )
        };
        self.line(&end_decl);
        self.line(&step_decl);
        let (i, e, s) = (format!("l{local}"), format!("e{n}"), format!("s{n}"));
        let test = self.counted_test(&i, &e, &s, signed);
        self.line(&format!(
            "for ({i} = {start}; {test}; {i} = ({i} + {s})) {{"
        ));
        self.nested(body)?;
        self.emit("}")
    }
}
