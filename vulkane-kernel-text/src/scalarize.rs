// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packed values (`LocalDecl::packed`, `ScalarExpr::Pack` / `Extract`) lowered by scalarization.
//!
//! `unpopped`'s shared `lower_dag` has no seam for a packed value (it returns "packed values have no
//! lowering seam yet"), so an emitter cannot spell a vector type through it. A packed value of the
//! kernel IR is a lane-wise scalar computation, though: `Add`, `Sub`, `Mul`, `Div`, `Min`/`Max`,
//! `Select` (scalar condition) and `Cast` apply to each lane alone, with exactly the scalar rules. So a
//! packed local of `n` lanes becomes `n` scalar locals, `Pack(xs)` is `xs[k]` in lane `k`, `Extract(v, k)`
//! is lane `k` of `v`, and a lane-wise operator is that operator applied to each lane. The result is
//! the same bits as a native vector would give, with the same float controls (they are per dtype).
//!
//! An assignment to a packed local evaluates **every** lane into a fresh temporary before it writes
//! any lane, so `v = Pack(Extract(v, 1), Extract(v, 0))` swaps the lanes rather than copying one.
//!
//! What is not rewritten (a packed `Bitcast`, a `Qualified` packed operation, a packed operand of an
//! operator that does not apply lane-wise, an `Extract` inside a statement this pass does not walk) is
//! left alone or refused by name, and the emitter declines it as it did before.

use unpopped::backend::LowerError;
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, KernelFn, LocalDecl, Place, Stmt};

fn refuse(kernel: &KernelFn, what: impl std::fmt::Display) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!("packed values: {what} (kernel `{}`)", kernel.name),
    }
}

struct Pass<'k> {
    kernel: &'k KernelFn,
    locals: Vec<LocalDecl>,
    /// For a packed local, the scalar local of each lane (lane 0 is the local itself).
    lanes: std::collections::BTreeMap<u32, Vec<u32>>,
}

/// `kernel` with every packed local, `Pack` and `Extract` replaced by scalars, or `None` when it has
/// no packed local (the common case: nothing to do).
///
/// # Errors
///
/// [`LowerError::UnsupportedKernel`] naming a packed construct that has no lane-wise rewrite.
pub fn scalarize(kernel: &KernelFn) -> Result<Option<KernelFn>, LowerError> {
    if kernel.locals.iter().all(|l| l.lanes == 1) {
        return Ok(None);
    }
    let mut pass = Pass {
        kernel,
        locals: kernel.locals.clone(),
        lanes: std::collections::BTreeMap::new(),
    };
    for (i, l) in kernel.locals.iter().enumerate() {
        if l.lanes > 1 {
            let id = u32::try_from(i).expect("local id fits u32");
            let mut ids = vec![id];
            for _ in 1..l.lanes {
                ids.push(pass.fresh(l.elem));
            }
            pass.locals[i] = LocalDecl::scalar(l.elem);
            pass.lanes.insert(id, ids);
        }
    }
    let body = pass.block(&kernel.body)?;
    let mut out = kernel.clone();
    out.locals = pass.locals;
    out.body = body;
    Ok(Some(out))
}

impl Pass<'_> {
    fn fresh(&mut self, elem: unpopped_vocab::ElementKind) -> u32 {
        self.locals.push(LocalDecl::scalar(elem));
        u32::try_from(self.locals.len() - 1).expect("local id fits u32")
    }

    fn no(&self, what: impl std::fmt::Display) -> LowerError {
        refuse(self.kernel, what)
    }

    /// Lane `k` of the packed expression `e`.
    fn lane(&self, e: &E, k: usize) -> Result<E, LowerError> {
        let b = |x: &E| self.lane(x, k).map(Box::new);
        Ok(match e {
            E::Local(v) => match self.lanes.get(v) {
                Some(ids) => E::Local(*ids.get(k).ok_or_else(|| self.no("a lane out of range"))?),
                None => return Err(self.no(format!("local {v} is not packed"))),
            },
            E::Pack(xs) => self.scalar(xs.get(k).ok_or_else(|| self.no("a lane out of range"))?)?,
            E::Add(x, y) => E::Add(b(x)?, b(y)?),
            E::Sub(x, y) => E::Sub(b(x)?, b(y)?),
            E::Mul(x, y) => E::Mul(b(x)?, b(y)?),
            E::Div(x, y) => E::Div(b(x)?, b(y)?),
            E::Binary(op, x, y) => E::Binary(*op, b(x)?, b(y)?),
            E::Fma(x, y, z) => E::Fma(b(x)?, b(y)?, b(z)?),
            E::Select(c, x, y) => E::Select(Box::new(self.scalar(c)?), b(x)?, b(y)?),
            E::Cast(x, spec) => E::Cast(b(x)?, *spec),
            other => {
                let name = format!("{other:?}");
                let name = name.split(['(', '{']).next().unwrap_or("an expression");
                return Err(self.no(format!(
                    "`{name}` of a packed value has no lane-wise lowering"
                )));
            }
        })
    }

    /// `e` (scalar) with every `Extract` replaced by the lane it names.
    fn scalar(&self, e: &E) -> Result<E, LowerError> {
        let b = |x: &E| self.scalar(x).map(Box::new);
        Ok(match e {
            E::Extract(x, i) => self.lane(x, usize::from(*i))?,
            E::Add(x, y) => E::Add(b(x)?, b(y)?),
            E::Sub(x, y) => E::Sub(b(x)?, b(y)?),
            E::Mul(x, y) => E::Mul(b(x)?, b(y)?),
            E::Div(x, y) => E::Div(b(x)?, b(y)?),
            E::Unary(op, x) => E::Unary(*op, b(x)?),
            E::Binary(op, x, y) => E::Binary(*op, b(x)?, b(y)?),
            E::Fma(x, y, z) => E::Fma(b(x)?, b(y)?, b(z)?),
            E::Select(c, x, y) => E::Select(b(c)?, b(x)?, b(y)?),
            E::Cast(x, spec) => E::Cast(b(x)?, *spec),
            other => other.clone(),
        })
    }

    fn place(&self, p: &Place) -> Result<Place, LowerError> {
        Ok(match p {
            Place::Logical { buffer, coords } => Place::Logical {
                buffer: *buffer,
                coords: coords
                    .iter()
                    .map(|c| self.scalar(c))
                    .collect::<Result<_, _>>()?,
            },
            Place::Alloc { alloc, index } => Place::Alloc {
                alloc: *alloc,
                index: self.scalar(index)?,
            },
            other => other.clone(),
        })
    }

    fn block(&mut self, block: &Block) -> Result<Block, LowerError> {
        let mut out = Vec::new();
        for s in &block.0 {
            self.stmt(s, &mut out)?;
        }
        Ok(Block(out))
    }

    fn stmt(&mut self, s: &Stmt, out: &mut Vec<Stmt>) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init } if self.lanes.contains_key(local) => {
                let ids = self.lanes[local].clone();
                for (k, id) in ids.into_iter().enumerate() {
                    out.push(Stmt::Let {
                        local: id,
                        init: self.lane(init, k)?,
                    });
                }
            }
            Stmt::Assign { local, value } if self.lanes.contains_key(local) => {
                // Every lane into a temporary first: the value may read the lanes it overwrites.
                let ids = self.lanes[local].clone();
                let elem = self.kernel.locals[usize::try_from(*local).expect("local id")].elem;
                let mut temps = Vec::new();
                for k in 0..ids.len() {
                    let t = self.fresh(elem);
                    out.push(Stmt::Let {
                        local: t,
                        init: self.lane(value, k)?,
                    });
                    temps.push(t);
                }
                for (id, t) in ids.into_iter().zip(temps) {
                    out.push(Stmt::Assign {
                        local: id,
                        value: E::Local(t),
                    });
                }
            }
            Stmt::Let { local, init } => out.push(Stmt::Let {
                local: *local,
                init: self.scalar(init)?,
            }),
            Stmt::Assign { local, value } => out.push(Stmt::Assign {
                local: *local,
                value: self.scalar(value)?,
            }),
            Stmt::Load { local, place } => out.push(Stmt::Load {
                local: *local,
                place: self.place(place)?,
            }),
            Stmt::Store { place, value } => out.push(Stmt::Store {
                place: self.place(place)?,
                value: self.scalar(value)?,
            }),
            Stmt::If { cond, then, els } => out.push(Stmt::If {
                cond: self.scalar(cond)?,
                then: self.block(then)?,
                els: self.block(els)?,
            }),
            Stmt::While { cond, body } => out.push(Stmt::While {
                cond: self.scalar(cond)?,
                body: self.block(body)?,
            }),
            Stmt::For {
                local,
                start,
                end,
                step,
                unroll,
                body,
            } => out.push(Stmt::For {
                local: *local,
                start: self.scalar(start)?,
                end: self.scalar(end)?,
                step: self.scalar(step)?,
                unroll: *unroll,
                body: self.block(body)?,
            }),
            other => out.push(other.clone()),
        }
        Ok(())
    }
}
