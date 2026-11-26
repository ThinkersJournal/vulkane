// SPDX-License-Identifier: MIT OR Apache-2.0
//! Atomics (ROADMAP row 8d): `AtomicRmw` and `AtomicCas` on a buffer or shared element, spelled by
//! the dialect. A dialect with no spelling (the default) leaves a typed decline naming it, and one
//! whose atomics carry no memory order declines every non-`Relaxed` order rather than dropping it.

use unpopped::backend::LowerError;
use unpopped::kernel::{MemoryOrder, Stmt};

use super::W;

impl W<'_> {
    /// One atomic statement.
    pub(super) fn atomic(&mut self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::AtomicRmw {
                op,
                place,
                value,
                old,
                order,
                scope,
            } => {
                self.relaxed_only(*order)?;
                self.scope_ok(*scope)?;
                let elem = self.element(place)?;
                let v = self.expr(value)?;
                let Some(call) = self.d.atomic_rmw(*op, &elem, &v) else {
                    return Err(self.no(format!("atomic op {op:?} has no spelling here")));
                };
                self.atomic_result(*old, &call)
            }
            Stmt::AtomicCas {
                place,
                expected,
                desired,
                old,
                order,
                scope,
            } => {
                self.relaxed_only(*order)?;
                self.scope_ok(*scope)?;
                let elem = self.element(place)?;
                let (e, d) = (self.expr(expected)?, self.expr(desired)?);
                let dest = old.map(|l| format!("l{l}"));
                let Some(lines) = self.d.atomic_cas_lines(&elem, &e, &d, dest.as_deref()) else {
                    return Err(self.no("compare-exchange has no spelling here"));
                };
                lines.iter().try_for_each(|l| self.emit(l))
            }
            _ => unreachable!("`stmt` routes only atomics here"),
        }
    }

    /// The statement that keeps (or drops) the old value.
    fn atomic_result(&mut self, old: Option<u32>, call: &str) -> Result<(), LowerError> {
        match old {
            Some(l) => self.emit(&format!("l{l} = {call};")),
            None => self.emit(&format!("{call};")),
        }
    }

    /// An atomic scope the dialect can spell, or a decline naming it.
    fn scope_ok(&self, scope: unpopped::exec::Scope) -> Result<(), LowerError> {
        if self.d.atomic_scope_ok(scope) {
            Ok(())
        } else {
            Err(self.no(format!("an atomic at {scope:?} scope is not lowered")))
        }
    }

    /// A dialect's atomics are relaxed; a stronger order is a decline, never a silent drop.
    fn relaxed_only(&self, order: MemoryOrder) -> Result<(), LowerError> {
        if order == MemoryOrder::Relaxed {
            Ok(())
        } else {
            Err(self.no(format!(
                "a {order:?} atomic is not lowered (the text atomics carry no memory order; use a Fence)"
            )))
        }
    }
}
