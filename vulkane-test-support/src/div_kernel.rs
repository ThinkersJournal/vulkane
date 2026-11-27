// SPDX-License-Identifier: MIT OR Apache-2.0
//! The integer `Div` test kernel (Unpopped's total rule, `docs/int-div-shift-semantics.md`, rules 1 and 2): `out[t] = a[t] / b[t]`
//! over `I32` or `U32`, one thread per `(dividend, divisor)` pair of the cross product of [`DIVIDENDS`] and [`DIVISORS`]
//! (zero, one, minus one, `MIN`, `MAX` and mixed values). The rule: a zero divisor gives the dividend, and signed
//! `MIN / -1` gives `MIN`; C++ and SPIR-V leave both undefined, so a lowering must guard them.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

/// Dividends: zero, one, minus one, `MIN`, `MAX`, and mixed patterns.
pub const DIVIDENDS: [u32; 9] = [
    0,
    1,
    0xFFFF_FFFF,
    0x8000_0000,
    0x7FFF_FFFF,
    7,
    0xFFFF_FFF9,
    0xDEAD_BEEF,
    0x1234_5678,
];
/// Divisors: zero, one, minus one, `MIN`, `MAX`, and small mixed values.
pub const DIVISORS: [u32; 8] = [
    0,
    1,
    0xFFFF_FFFF,
    0x8000_0000,
    0x7FFF_FFFF,
    2,
    3,
    0xFFFF_FFFD,
];

/// Threads launched: every pair, padded to a multiple of 8 (the padding repeats the last pair).
pub const THREADS: u32 = 72;

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![E::Exec(ExecId::ThreadIdx(0))],
    }
}

/// The kernel: `out[t] = a[t] / b[t]` in compute type `ty` (`I32` or `U32`).
pub fn div_kernel(ty: ElementKind) -> KernelFn {
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
            value: E::Div(Box::new(E::Local(0)), Box::new(E::Local(1))),
        },
    ];
    let mut k = KernelFn::new(
        "div",
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

/// The input buffers: pair `t` is `(DIVIDENDS[t / 8], DIVISORS[t % 8])`.
pub fn inputs() -> (Vec<u32>, Vec<u32>) {
    (0..THREADS as usize)
        .map(|t| (DIVIDENDS[t / DIVISORS.len()], DIVISORS[t % DIVISORS.len()]))
        .unzip()
}

/// The total rule's quotient.
pub fn expected(signed: bool, a: u32, b: u32) -> u32 {
    if b == 0 {
        return a;
    }
    if signed {
        (a as i32).wrapping_div(b as i32) as u32
    } else {
        a / b
    }
}

/// What a lowering that leaves the C++ cases unguarded would give where it can be named: a zero divisor leaves the
/// dividend's complement (any value would do; this one differs from the rule at every `b == 0` pair).
pub fn unguarded_zero(a: u32) -> u32 {
    !a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_total_rule_at_the_corners() {
        assert_eq!(
            expected(true, 0x8000_0000, 0xFFFF_FFFF),
            0x8000_0000,
            "MIN / -1"
        );
        assert_eq!(expected(true, 7, 0), 7);
        assert_eq!(expected(false, 0xFFFF_FFFF, 0), 0xFFFF_FFFF);
        assert_eq!(
            expected(true, 0xFFFF_FFF9, 2),
            0xFFFF_FFFD,
            "-7 / 2 truncates toward zero to -3"
        );
        assert_eq!(expected(false, 0xFFFF_FFF9, 2), 0x7FFF_FFFC);
        let (a, b) = inputs();
        assert_eq!((a.len(), b.len()), (THREADS as usize, THREADS as usize));
        assert_eq!(DIVIDENDS.len() * DIVISORS.len(), THREADS as usize);
    }
}
