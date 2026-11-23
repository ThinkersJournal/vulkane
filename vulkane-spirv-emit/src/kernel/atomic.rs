// SPDX-License-Identifier: MIT OR Apache-2.0
//! Atomics for `Backend::lower_kernel` (ROADMAP row 8d, 32-bit integer slice): `AtomicRmw` and
//! `AtomicCas` on a storage-buffer element or a workgroup-array element.
//!
//! `AtomicOp` maps to `OpAtomic{IAdd,SMin,UMin,SMax,UMax,And,Or,Xor,Exchange}` (the signed or
//! unsigned form follows the compute dtype) and `AtomicCas` to `OpAtomicCompareExchange`, whose
//! result is the value found either way, as in the IR. The compute dtype is 32-bit by construction
//! here, so no `Int64Atomics` capability is involved. Scope maps `Subgroup`, `Block` and `Device`
//! to the SPIR-V scopes of the same name; anything wider is a decline. `Relaxed` carries no
//! ordering bits; `Acquire`, `Release` and `AcqRel`/`SeqCst` (collapsed to `AcquireRelease`, as the
//! fence does) add the memory class of the pointer. A `Private` array is no atomic target in SPIR-V
//! and is declined.

use rspirv::spirv::{MemorySemantics, Scope as SpvScope, Word};
use unpopped::backend::LowerError;
use unpopped::exec::{MemorySpace, Scope};
use unpopped::kernel::{AtomicOp, MemoryOrder, Place, Stmt};

use super::{Kc, refuse};
use crate::ops::lo;

impl Kc<'_> {
    /// The SPIR-V scope of an atomic's IR scope, or a decline.
    fn atomic_scope(&self, scope: Scope) -> Result<SpvScope, LowerError> {
        match scope {
            Scope::Subgroup => Ok(SpvScope::Subgroup),
            Scope::Block => Ok(SpvScope::Workgroup),
            Scope::Device => Ok(SpvScope::Device),
            other => Err(refuse(
                self.kernel,
                format!("an atomic at scope {other:?} is not lowered"),
            )),
        }
    }

    /// The memory-semantics bits of `order` for the memory `place` lives in.
    fn atomic_semantics(&self, place: &Place, order: MemoryOrder) -> Result<u32, LowerError> {
        let class = match place {
            Place::Alloc { alloc, .. } => {
                let slot = usize::try_from(*alloc).expect("alloc id fits usize");
                match self.kernel.allocs[slot].space {
                    MemorySpace::Shared => MemorySemantics::WORKGROUP_MEMORY,
                    other => {
                        return Err(refuse(
                            self.kernel,
                            format!("an atomic on a {other:?} array is not lowered"),
                        ));
                    }
                }
            }
            _ => MemorySemantics::UNIFORM_MEMORY,
        };
        let ord = match order {
            MemoryOrder::Relaxed => return Ok(MemorySemantics::empty().bits()),
            MemoryOrder::Acquire => MemorySemantics::ACQUIRE,
            MemoryOrder::Release => MemorySemantics::RELEASE,
            MemoryOrder::AcqRel | MemoryOrder::SeqCst => MemorySemantics::ACQUIRE_RELEASE,
            other => {
                return Err(refuse(
                    self.kernel,
                    format!("a {other:?} atomic is not lowered"),
                ));
            }
        };
        Ok((ord | class).bits())
    }

    /// One atomic read-modify-write or compare-exchange statement.
    pub(super) fn atomic(&self, s: &Stmt) -> Result<(), LowerError> {
        if self.em.borrow().fam.is_wide() {
            return Err(refuse(
                self.kernel,
                "a 64-bit atomic needs Int64Atomics, which is not lowered yet",
            ));
        }
        match s {
            Stmt::AtomicRmw {
                op,
                place,
                value,
                old,
                order,
                scope,
            } => {
                let found = self.atomic_rmw(*op, place, value, *order, *scope)?;
                old.map_or(Ok(()), |l| self.store_local(l, found))
            }
            Stmt::AtomicCas {
                place,
                expected,
                desired,
                old,
                order,
                scope,
            } => {
                let found = self.atomic_cas(place, expected, desired, *order, *scope)?;
                old.map_or(Ok(()), |l| self.store_local(l, found))
            }
            _ => unreachable!("`stmt` routes only atomics here"),
        }
    }

    fn atomic_rmw(
        &self,
        op: AtomicOp,
        place: &Place,
        value: &unpopped::ir::ScalarExpr,
        order: MemoryOrder,
        scope: Scope,
    ) -> Result<Word, LowerError> {
        let spv = self.atomic_scope(scope)?;
        let sem = self.atomic_semantics(place, order)?;
        let ptr = self.pointer(place)?;
        let v = self.expr(value)?;
        let signed = self.em.borrow().fam.is_signed();
        let (t, sc, sm) = (self.t(), self.u32_const(spv as u32), self.u32_const(sem));
        let mut e = self.em.borrow_mut();
        let b = &mut e.b;
        lo(match (op, signed) {
            (AtomicOp::Add, _) => b.atomic_i_add(t, None, ptr, sc, sm, v),
            (AtomicOp::Max, true) => b.atomic_s_max(t, None, ptr, sc, sm, v),
            (AtomicOp::Max, false) => b.atomic_u_max(t, None, ptr, sc, sm, v),
            (AtomicOp::Min, true) => b.atomic_s_min(t, None, ptr, sc, sm, v),
            (AtomicOp::Min, false) => b.atomic_u_min(t, None, ptr, sc, sm, v),
            (AtomicOp::And, _) => b.atomic_and(t, None, ptr, sc, sm, v),
            (AtomicOp::Or, _) => b.atomic_or(t, None, ptr, sc, sm, v),
            (AtomicOp::Xor, _) => b.atomic_xor(t, None, ptr, sc, sm, v),
            (AtomicOp::Exchange, _) => b.atomic_exchange(t, None, ptr, sc, sm, v),
            _ => {
                drop(e);
                return Err(refuse(
                    self.kernel,
                    format!("atomic op {op:?} is not lowered"),
                ));
            }
        })
    }

    fn atomic_cas(
        &self,
        place: &Place,
        expected: &unpopped::ir::ScalarExpr,
        desired: &unpopped::ir::ScalarExpr,
        order: MemoryOrder,
        scope: Scope,
    ) -> Result<Word, LowerError> {
        let spv = self.atomic_scope(scope)?;
        let sem = self.atomic_semantics(place, order)?;
        let ptr = self.pointer(place)?;
        let (cmp, new) = (self.expr(expected)?, self.expr(desired)?);
        let (t, sc, sm) = (self.t(), self.u32_const(spv as u32), self.u32_const(sem));
        // The failure order may not be stronger than the success order, nor a release.
        let fail = self.u32_const(sem & !MemorySemantics::RELEASE.bits());
        let mut e = self.em.borrow_mut();
        lo(e.b.atomic_compare_exchange(t, None, ptr, sc, sm, fail, new, cmp))
    }
}
