// SPDX-License-Identifier: MIT OR Apache-2.0
//! The float kernel test cases (ROADMAP 8, `F32`/`F64` kernels), shared by the unit tests and
//! `tests/kernel_float_live.rs`. 256 threads each read `a[t]` and `b[t]` and write one result: the
//! pairs cycle through values that distinguish `f32` from `f64` (`0.1`, `1e-30 * 1e-10`, `2^24 + 1`,
//! `3.4e38 * 10`), a denormal result, NaN, infinities and signed zeros. The host oracle computes the
//! same expression in native `f32`/`f64`, which is IEEE and does not fuse.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{ArithOp, BinaryOp, ScalarExpr as E, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

/// Threads launched.
pub const THREADS: u32 = 256;

/// The expression under test.
#[derive(Copy, Clone, Debug)]
pub enum Case {
    /// `a * b + a`: a multiply feeding an add, which a driver may fuse (so `NoContraction`).
    MulAdd,
    /// `Max(a, b)`, NaN-propagating, `a` on ties.
    Max,
    /// `Min(a, b)`, NaN-propagating, `a` on ties.
    Min,
    /// `Floor(a)`.
    Floor,
    /// `a / b`: declined (the spec bound on `OpFDiv` is looser than the oracle's).
    Div,
    /// `Sqrt(a)`: declined.
    Sqrt,
    /// `Exp(a)`: declined.
    Exp,
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![E::Exec(ExecId::ThreadIdx(0))],
    }
}

/// The kernel applying `case` over `ty` to two input buffers.
pub fn float_kernel(ty: ElementKind, case: Case) -> KernelFn {
    let (a, b) = (E::Local(0), E::Local(1));
    let value = match case {
        Case::MulAdd => E::Add(
            Box::new(E::Mul(Box::new(a.clone()), Box::new(b))),
            Box::new(a),
        ),
        Case::Max => E::Binary(BinaryOp::Max, Box::new(a), Box::new(b)),
        Case::Min => E::Binary(BinaryOp::Min, Box::new(a), Box::new(b)),
        Case::Floor => E::Unary(UnaryOp::Floor, Box::new(a)),
        Case::Div => E::Div(Box::new(a), Box::new(b)),
        Case::Sqrt => E::Unary(UnaryOp::Sqrt, Box::new(a)),
        Case::Exp => E::Unary(UnaryOp::Exp, Box::new(a)),
    };
    let _ = ArithOp::Add;
    let body = vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Load {
            local: 1,
            place: at(1),
        },
        Stmt::Store {
            place: at(2),
            value,
        },
    ];
    let mut k = KernelFn::new(
        "float",
        ty,
        vec![
            Param::buffer("a", ty, BufferRole::Read, Some(1)),
            Param::buffer("b", ty, BufferRole::Read, Some(1)),
            Param::buffer("out", ty, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(ty); 2],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]));
    k
}

/// The `(a, b)` pairs, as `f64` (rounded to `f32` for an `F32` kernel).
pub const PAIRS: [(f64, f64); 14] = [
    (0.1, 3.0),
    (1e-30, 1e-10),
    (16_777_217.0, 1.0),
    (1e-30, 1e-30),
    (f64::NAN, 1.0),
    (f64::INFINITY, 0.0),
    (-0.0, 1.0),
    (0.0, -1.0),
    (f64::INFINITY, -1.0),
    (1.0, f64::NAN),
    (-0.5, 2.0),
    (1.5, -2.5),
    (3.4e38, 10.0),
    (-0.0, 0.0),
];

/// A host float: the same IEEE operations as the kernel, in native arithmetic.
pub trait Fl: Copy {
    fn from_f64(v: f64) -> Self;
    fn mul_add_unfused(self, b: Self) -> Self;
    fn floor_(self) -> Self;
    fn is_nan_(self) -> bool;
    fn ge_(self, o: Self) -> bool;
    fn le_(self, o: Self) -> bool;
    /// The value as little-endian 32-bit words (one for `f32`, two for `f64`).
    fn words(self) -> Vec<u32>;
}

macro_rules! fl {
    ($t:ty, $bits:ty) => {
        impl Fl for $t {
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn mul_add_unfused(self, b: Self) -> Self {
                self * b + self
            }
            fn floor_(self) -> Self {
                self.floor()
            }
            fn is_nan_(self) -> bool {
                self.is_nan()
            }
            fn ge_(self, o: Self) -> bool {
                self >= o
            }
            fn le_(self, o: Self) -> bool {
                self <= o
            }
            fn words(self) -> Vec<u32> {
                let b: $bits = self.to_bits();
                let b = u64::from(b);
                if std::mem::size_of::<$t>() == 4 {
                    vec![b as u32]
                } else {
                    vec![b as u32, (b >> 32) as u32]
                }
            }
        }
    };
}
fl!(f32, u32);
fl!(f64, u64);

/// The expected result for `(a, b)` under `case`, in the kernel's own width.
pub fn expect<T: Fl>(case: Case, a: T, b: T) -> T {
    // NaN-propagating, `a` on ties (the `max_prop`/`min_prop` decomposition).
    let prop = |is_max: bool| {
        if a.is_nan_() {
            a
        } else if b.is_nan_() {
            b
        } else if (is_max && a.ge_(b)) || (!is_max && a.le_(b)) {
            a
        } else {
            b
        }
    };
    match case {
        Case::MulAdd => a.mul_add_unfused(b),
        Case::Max => prop(true),
        Case::Min => prop(false),
        Case::Floor => a.floor_(),
        _ => unreachable!("a declined case has no oracle"),
    }
}

/// Input words for `a` and `b`, and the expected output words, for `THREADS` threads.
pub fn data<T: Fl>(case: Case) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut a, mut b, mut want) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..THREADS as usize {
        let (x, y) = PAIRS[t % PAIRS.len()];
        let (x, y) = (T::from_f64(x), T::from_f64(y));
        a.extend(x.words());
        b.extend(y.words());
        want.extend(expect(case, x, y).words());
    }
    (a, b, want)
}

/// Whether two result words agree: bit-exact, except that every NaN equals every NaN (payload and
/// sign of a NaN are not part of the contract) and a `-0.0` must stay `-0.0` (bit-exact).
pub fn same(width64: bool, got: &[u32], want: &[u32]) -> Result<(), String> {
    let step = if width64 { 2 } else { 1 };
    for (i, (g, w)) in got.chunks(step).zip(want.chunks(step)).enumerate() {
        let nan = |c: &[u32]| {
            if width64 {
                f64::from_bits(u64::from(c[0]) | (u64::from(c[1]) << 32)).is_nan()
            } else {
                f32::from_bits(c[0]).is_nan()
            }
        };
        if !(nan(g) && nan(w)) && g != w {
            return Err(format!("thread {i}: got {g:08x?} want {w:08x?}"));
        }
    }
    Ok(())
}
