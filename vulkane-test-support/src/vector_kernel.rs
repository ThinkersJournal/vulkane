// SPDX-License-Identifier: MIT OR Apache-2.0
//! Packed values (`vec2` / `vec4`: `LocalDecl::packed`, `Pack`, `Extract`) in an `F32` kernel, shared by
//! the SPIR-V and GLSL tests:
//!
//! ```text
//! x = a[t]; y = b[t]
//! v = Pack(x, y)                       // lanes 2;  lanes 4: Pack(x, y, x + y, x - y)
//! v = v * v + v                        // lanes 2;  lanes 4: Max(v * v, v), lane-wise, NaN-propagating
//! v = Pack(v[1], v[0])                 // lanes 4: Pack(v[1], v[0], v[3], v[2]): a lane swap that reads
//!                                      // the lanes it overwrites
//! out[t] = v[LANE]
//! ```
//!
//! The kernel stores one lane, so a device run per lane reads all of them. Data holds NaN, infinities,
//! signed zeros, a denormal and the fusion pair; the host reference is native `f32`. Mutants (each must
//! differ from the reference somewhere): a fused multiply-add, no lane swap, and the swap done lane by
//! lane in place (both lanes end up `w[1]`).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr as E};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::F32;

/// Threads launched.
pub const THREADS: usize = 256;

/// The `(a, b)` float input pairs, cycled over the threads.
pub const PAIRS: [(f32, f32); 14] = [
    (0.1, 3.0),
    (1e-30, 1e-10),
    (16_777_217.0, 1.0),
    (f32::NAN, 1.0),
    (f32::INFINITY, 0.0),
    (-0.0, 1.0),
    (1.0, f32::NAN),
    (-0.5, 2.0),
    (1.5, -2.5),
    (3.4e38, 10.0),
    (1.000_000_1, 1.000_000_1),
    (0.333_333_34, -0.333_333_34),
    (1e-40, 2.0),
    (2.0, 1e-40),
];

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![E::Exec(ExecId::ThreadIdx(0))],
    }
}

fn lane(local: u32, k: u8) -> E {
    E::Extract(Box::new(E::Local(local)), k)
}

/// The kernel for `lanes` (2 or 4) storing lane `store` of the swapped vector.
pub fn vector_kernel(lanes: u8, store: u8) -> KernelFn {
    let (x, y) = (E::Local(0), E::Local(1));
    let packed = LocalDecl::packed(F32, lanes);
    let (init, w, order): (Vec<E>, E, Vec<u8>) = if lanes == 2 {
        (
            vec![x, y],
            E::Add(
                Box::new(E::Mul(Box::new(E::Local(2)), Box::new(E::Local(2)))),
                Box::new(E::Local(2)),
            ),
            vec![1, 0],
        )
    } else {
        (
            vec![
                x.clone(),
                y.clone(),
                E::Add(Box::new(x.clone()), Box::new(y.clone())),
                E::Sub(Box::new(x), Box::new(y)),
            ],
            E::Binary(
                BinaryOp::Max,
                Box::new(E::Mul(Box::new(E::Local(2)), Box::new(E::Local(2)))),
                Box::new(E::Local(2)),
            ),
            vec![1, 0, 3, 2],
        )
    };
    let body = vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Load {
            local: 1,
            place: at(1),
        },
        Stmt::Let {
            local: 2,
            init: E::Pack(init),
        },
        Stmt::Assign { local: 2, value: w },
        Stmt::Assign {
            local: 2,
            value: E::Pack(order.iter().map(|&k| lane(2, k)).collect()),
        },
        Stmt::Store {
            place: at(2),
            value: lane(2, store),
        },
    ];
    let mut k = KernelFn::new(
        "vector",
        F32,
        vec![
            Param::buffer("a", F32, BufferRole::Read, Some(1)),
            Param::buffer("b", F32, BufferRole::Read, Some(1)),
            Param::buffer("out", F32, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(F32), LocalDecl::scalar(F32), packed],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([THREADS as u32, 1, 1], [1, 1, 1]));
    k
}

/// What a variant computes on the host.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Variant {
    Right,
    /// A fused multiply-add in the `lanes == 2` arithmetic.
    Fused,
    /// The vector is not swapped.
    NoSwap,
    /// The swap is written lane by lane in place: each pair's first lane takes the second, the second
    /// then takes the new first (both end up the old second lane).
    InPlace,
}

fn max_prop(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if a >= b {
        a
    } else {
        b
    }
}

/// The lanes of `w` for input `(x, y)`.
fn w_lanes(lanes: u8, x: f32, y: f32, fused: bool) -> Vec<f32> {
    if lanes == 2 {
        let f = |v: f32| if fused { v.mul_add(v, v) } else { v * v + v };
        vec![f(x), f(y)]
    } else {
        [x, y, x + y, x - y]
            .iter()
            .map(|&v| max_prop(v * v, v))
            .collect()
    }
}

/// The value the kernel stores in lane `store` for `(x, y)` under `variant`.
pub fn host(lanes: u8, store: u8, x: f32, y: f32, variant: Variant) -> f32 {
    let w = w_lanes(lanes, x, y, variant == Variant::Fused);
    let order: Vec<usize> = if lanes == 2 {
        vec![1, 0]
    } else {
        vec![1, 0, 3, 2]
    };
    let mut v = w.clone();
    match variant {
        Variant::NoSwap => {}
        Variant::InPlace => {
            for pair in 0..usize::from(lanes) / 2 {
                v[2 * pair] = w[2 * pair + 1];
                v[2 * pair + 1] = v[2 * pair];
            }
        }
        _ => {
            for (k, &src) in order.iter().enumerate() {
                v[k] = w[src];
            }
        }
    }
    v[usize::from(store)]
}

/// `(a words, b words, expected words)` for `lanes`, `store` and `variant`.
pub fn data(lanes: u8, store: u8, variant: Variant) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut a, mut bw, mut want) = (vec![], vec![], vec![]);
    for t in 0..THREADS {
        let (x, y) = PAIRS[t % PAIRS.len()];
        a.push(x.to_bits());
        bw.push(y.to_bits());
        want.push(host(lanes, store, x, y, variant).to_bits());
    }
    (a, bw, want)
}

/// Bit-exact, every NaN equal to every NaN.
pub fn same(got: &[u32], want: &[u32]) -> Result<(), String> {
    for (t, (g, w)) in got.iter().zip(want).enumerate() {
        let nan = |x: u32| f32::from_bits(x).is_nan();
        if !(nan(*g) && nan(*w)) && g != w {
            return Err(format!("thread {t}: got {g:#010x} want {w:#010x}"));
        }
    }
    Ok(())
}
