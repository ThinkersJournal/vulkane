// SPDX-License-Identifier: MIT OR Apache-2.0
//! Control flow in a float kernel (ROADMAP 8g over `F32`), shared by the SPIR-V and GLSL tests:
//!
//! ```text
//! y = x[t]; acc = 0.0
//! for i in 0..4 (U32 counter, or 3 down to 0 with an I32 one):
//!     w = wts[i]
//!     if y * w > 0.0 { acc = acc + y * w } else { acc = acc - w }
//!     if acc > 100.0 { break }
//! while acc > 8.0 { acc = acc * 0.5 }
//! out[t] = acc
//! ```
//!
//! The conditions are float tests, the counter is an integer local, `y * w` feeding `acc +` is a
//! fusable pair, and the data holds NaN (every compare false), a denormal, a tie at 2^24+1, signs and
//! zeros, but no infinity (`while` would not end). The host reference is native `f32`; each mutant is
//! the output of a *wrong* kernel and must differ from it somewhere.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr as E};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, Unroll};
use unpopped_vocab::ElementKind::{self, F32, I32, U32};

pub const THREADS: usize = 256;
pub const WEIGHTS: [f32; 4] = [1.5, -2.0, 0.25, 3.0];
pub const XS: [f32; 24] = [
    0.1,
    -0.1,
    3.0,
    -3.0,
    16_777_217.0,
    1e-30,
    -1e-30,
    0.5,
    2.5,
    100.0,
    1e6,
    f32::NAN,
    0.0,
    -0.0,
    7.999_999_5,
    1e-40,
    // Values whose product with a weight is not exact in `f32`, so a fused multiply-add differs.
    1.000_000_1,
    1.234_567_9,
    0.333_333_34,
    3.141_593_1,
    12_345.678,
    0.007_812_51,
    5.000_000_5,
    1.999_999_9,
];

fn b(e: E) -> Box<E> {
    Box::new(e)
}

fn at(buffer: u32, index: E) -> Place {
    Place::Logical {
        buffer,
        coords: vec![index],
    }
}

fn cmp_gt(l: E, r: E) -> E {
    E::Binary(BinaryOp::CmpGt, b(l), b(r))
}

/// The kernel: `counter` is `U32` (up from 0) or `I32` (down from 3, step -1).
pub fn float_flow_kernel(counter: ElementKind) -> KernelFn {
    let (y, acc, w, i) = (E::Local(0), E::Local(1), E::Local(2), E::Local(3));
    let ci = |v: i64| E::ConstInt(v, counter);
    let (start, end, step) = if counter == I32 {
        (ci(3), ci(-1), ci(-1))
    } else {
        (ci(0), ci(4), ci(1))
    };
    let prod = || E::Mul(b(y.clone()), b(w.clone()));
    let body = vec![
        Stmt::Load {
            local: 0,
            place: at(0, E::Exec(ExecId::ThreadIdx(0))),
        },
        Stmt::Let {
            local: 1,
            init: E::Const(0.0),
        },
        Stmt::For {
            local: 3,
            start,
            end,
            step,
            unroll: Unroll::None,
            body: Block(vec![
                Stmt::Load {
                    local: 2,
                    place: at(1, i.clone()),
                },
                Stmt::If {
                    cond: cmp_gt(prod(), E::Const(0.0)),
                    then: Block(vec![Stmt::Assign {
                        local: 1,
                        value: E::Add(b(acc.clone()), b(prod())),
                    }]),
                    els: Block(vec![Stmt::Assign {
                        local: 1,
                        value: E::Sub(b(acc.clone()), b(w.clone())),
                    }]),
                },
                Stmt::If {
                    cond: cmp_gt(acc.clone(), E::Const(100.0)),
                    then: Block(vec![Stmt::Break]),
                    els: Block(vec![]),
                },
            ]),
        },
        Stmt::While {
            cond: cmp_gt(acc.clone(), E::Const(8.0)),
            body: Block(vec![Stmt::Assign {
                local: 1,
                value: E::Mul(b(acc.clone()), b(E::Const(0.5))),
            }]),
        },
        Stmt::Store {
            place: at(2, E::Exec(ExecId::ThreadIdx(0))),
            value: acc,
        },
    ];
    let mut k = KernelFn::new(
        "float_flow",
        F32,
        vec![
            Param::buffer("x", F32, BufferRole::Read, Some(1)),
            Param::buffer("wts", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(F32),
            LocalDecl::scalar(counter),
        ],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([THREADS as u32, 1, 1], [1, 1, 1]));
    k
}

/// What a variant of the kernel computes on the host: the reference, or one wrong kernel.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Variant {
    /// The kernel as written.
    Right,
    /// A fused multiply-add where the kernel does a multiply then an add.
    Fused,
    /// The `break` ignored.
    NoBreak,
}

/// The result for input `y`, the weights visited in `order`.
pub fn host(y: f32, order: [usize; 4], v: Variant) -> f32 {
    let mut acc = 0.0f32;
    for &i in &order {
        let w = WEIGHTS[i];
        if y * w > 0.0 {
            acc = if v == Variant::Fused {
                y.mul_add(w, acc)
            } else {
                acc + y * w
            };
        } else {
            acc -= w;
        }
        if acc > 100.0 && v != Variant::NoBreak {
            break;
        }
    }
    while acc > 8.0 {
        acc *= 0.5;
    }
    acc
}

/// The weight order of the kernel with this counter type.
pub fn order(counter: ElementKind) -> [usize; 4] {
    if counter == U32 {
        [0, 1, 2, 3]
    } else {
        [3, 2, 1, 0]
    }
}

/// Input words (`x`, `wts` padded to the same length), and the expected output for `variant`.
pub fn data(counter: ElementKind, v: Variant) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut xs = Vec::new();
    let mut want = Vec::new();
    for t in 0..THREADS {
        let y = XS[t % XS.len()];
        xs.push(y.to_bits());
        want.push(host(y, order(counter), v).to_bits());
    }
    let mut w: Vec<u32> = WEIGHTS.iter().map(|w| w.to_bits()).collect();
    w.resize(THREADS, 0);
    (xs, w, want)
}

/// Whether two outputs agree: bit-exact except that every NaN equals every NaN.
pub fn same(got: &[u32], want: &[u32]) -> Result<(), String> {
    for (t, (g, w)) in got.iter().zip(want).enumerate() {
        let nan = |x: u32| f32::from_bits(x).is_nan();
        if !(nan(*g) && nan(*w)) && g != w {
            return Err(format!("thread {t}: got {g:#010x} want {w:#010x}"));
        }
    }
    Ok(())
}
