// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 64-bit integer test kernel (ROADMAP row 8, wider dtypes), at `I64` or `U64`. Thread `t` of
//! `THREADS` writes `K` results to `out[t * K + k]`, every one of them across the 32-bit boundary, so
//! a lowering that goes through 32 bits cannot match [`expected`]:
//!
//! | k | result (`x = t`, `B = 2^32`) |
//! |---|---|
//! | 0 | `x + B + (B - 1)`: an add carrying into the high word |
//! | 1 | `(x + 1) * (B + 1)`: a mul whose high word depends on the low |
//! | 2 | `(x + 1) << 40`: a shift past 32 |
//! | 3 | `(x + 1) << 99`: the count masked to `99 & 63 = 35` (a 32-bit mask would give 3) (rule 3 of `int-div-shift-semantics.md`) |
//! | 4 | `((x + 1) << 62) >> 29`: arithmetic (`I64`) or logical (`U64`) by the dtype |
//! | 5 | `(x + 2B) / 0 = x + 2B` (rule 1/2, zero divisor gives the dividend) |
//! | 6 | `I64`: `MIN / -1 = MIN`; `U64`: `(x + 2B) / (x + 1)` |
//! | 7 | `Rem` as `Select(b == 0, 0, a - (a / b) * b)` with `a = x + 2B + 5`, `b = x & 3` |
//! | 8 | `I64`: `MIN % -1 = 0`; `U64`: `(x + 2B) % 0 = 0` |

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, CastSpec, RoundMode, Saturation, ScalarExpr as E};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const THREADS: u32 = 64;
pub(crate) const K: usize = 9;
const B: i64 = 1 << 32;

fn bx(e: E) -> Box<E> {
    Box::new(e)
}

fn add(a: E, b: E) -> E {
    E::Add(bx(a), bx(b))
}

fn mul(a: E, b: E) -> E {
    E::Mul(bx(a), bx(b))
}

fn bin(op: BinaryOp, a: E, b: E) -> E {
    E::Binary(op, bx(a), bx(b))
}

/// `a - (a / b) * b`, zero for `b == 0`: the composition the ruling gives integer `Rem`.
fn rem(a: E, b: E, ty: ElementKind) -> E {
    let q = E::Div(bx(a.clone()), bx(b.clone()));
    let r = E::Sub(bx(a), bx(mul(q, b.clone())));
    let zero = || E::ConstInt(0, ty);
    E::Select(bx(bin(BinaryOp::CmpEq, b, zero())), bx(zero()), bx(r))
}

/// The kernel at `ty` (`I64` or `U64`).
pub(crate) fn wide_kernel(ty: ElementKind) -> KernelFn {
    let c = |v: i64| E::ConstInt(v, ty);
    let (x, zero, a2) = (E::Local(0), E::Local(1), E::Local(2));
    let signed = ty == ElementKind::I64;
    let x1 = add(x.clone(), c(1));
    let results = [
        add(add(x.clone(), c(B)), c(B - 1)),
        mul(x1.clone(), c(B + 1)),
        bin(BinaryOp::Shl, x1.clone(), c(40)),
        bin(BinaryOp::Shl, x1.clone(), c(99)),
        bin(BinaryOp::Shr, bin(BinaryOp::Shl, x1.clone(), c(62)), c(29)),
        E::Div(bx(a2.clone()), bx(zero.clone())),
        if signed {
            E::Div(bx(c(i64::MIN)), bx(c(-1)))
        } else {
            E::Div(bx(a2.clone()), bx(x1))
        },
        rem(
            add(a2.clone(), c(5)),
            bin(BinaryOp::BitAnd, x.clone(), c(3)),
            ty,
        ),
        if signed {
            rem(c(i64::MIN), c(-1), ty)
        } else {
            rem(a2.clone(), zero, ty)
        },
    ];
    let mut body = vec![
        Stmt::Let {
            local: 0,
            // An execution id is a `U32`; the cast carries it to the kernel's dtype.
            init: E::Cast(
                bx(E::Exec(ExecId::ThreadIdx(0))),
                CastSpec {
                    to: ty,
                    round: RoundMode::NearestEven,
                    sat: Saturation::Wrap,
                },
            ),
        },
        Stmt::Let {
            local: 1,
            init: E::Sub(bx(x.clone()), bx(x.clone())),
        },
        Stmt::Let {
            local: 2,
            init: add(x.clone(), c(2 * B)),
        },
    ];
    for (k, r) in results.into_iter().enumerate() {
        let at = add(mul(x.clone(), c(K as i64)), c(k as i64));
        body.push(Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![at],
            },
            value: r,
        });
    }
    let mut kern = KernelFn::new(
        "wide",
        ty,
        vec![Param::buffer("out", ty, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(ty); 3],
        Block(body),
    );
    kern.launch = Some(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]));
    kern
}

/// The host result of cell `(t, k)` computed at `width` bits (64 for the kernel, 32 for the
/// mutant: a lowering that goes through 32 bits, which wraps every value, constant and count at 2^32),
/// as the bits the device holds, by the ruled semantics.
pub(crate) fn expected_at(width: u32, signed: bool, t: u64, k: usize) -> u64 {
    let m = if width == 64 {
        u64::MAX
    } else {
        u64::from(u32::MAX)
    };
    let w = |v: u64| v & m;
    // The value as a signed number of `width` bits, for the arithmetic shift.
    let sext = |v: u64| ((v << (64 - width)) as i64) >> (64 - width);
    let (bb, x) = (w(1 << 32), w(t));
    let x1 = w(x + 1);
    let a2 = w(x.wrapping_add(w(2u64 << 32)));
    let count = |c: u64| (c & u64::from(width - 1)) as u32;
    match k {
        0 => w(x.wrapping_add(bb).wrapping_add(w((1 << 32) - 1))),
        1 => w(x1.wrapping_mul(w((1 << 32) + 1))),
        2 => w(x1 << count(40)),
        3 => w(x1 << count(99)),
        4 => {
            let up = w(x1 << count(62));
            if signed {
                w((sext(up) >> count(29)) as u64)
            } else {
                up >> count(29)
            }
        }
        5 => a2,
        6 if signed => w(1 << (width - 1)),
        6 => a2 / x1,
        7 => {
            let (a, b) = (w(a2 + 5), x & 3);
            a.checked_div(b).map_or(0, |q| w(a - q * b))
        }
        _ => 0,
    }
}

/// The host result at the kernel's own 64 bits.
pub(crate) fn expected(signed: bool, t: u64, k: usize) -> u64 {
    expected_at(64, signed, t, k)
}

/// The device words (little-endian halves) of every cell, in cell order.
pub(crate) fn expected_words(signed: bool) -> Vec<u32> {
    (0..u64::from(THREADS))
        .flat_map(|t| (0..K).map(move |k| expected(signed, t, k)))
        .flat_map(|v| [v as u32, (v >> 32) as u32])
        .collect()
}
