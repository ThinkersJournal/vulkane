// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 8/16-bit integer storage test kernels (ROADMAP 8e): 256 threads read one `elem`-typed element
//! each (`elem` one of `I8`/`U8`/`I16`/`U16`) and the kernel computes at 32 bits, since an 8/16-bit
//! integer is not a compute dtype (its constants are illegal in the IR); a wrapping `Cast` widens and
//! narrows:
//!
//! - [`widen_kernel`]: `out[t] = (compute) in[t]`, a 32-bit result: sign-extended for `I8`/`I16`,
//!   zero-extended for `U8`/`U16`;
//! - [`wrap_kernel`]: `out[t] = (elem) ((compute) in[t] + 1)`: the 32-bit add narrowed back, so the
//!   type's maximum wraps to its minimum (`127 -> -128`, `255 -> 0`).

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr as E};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;

pub(crate) const THREADS: u32 = 256;

/// The compute dtype that matches `elem`'s signedness.
pub(crate) fn compute_of(elem: ElementKind) -> ElementKind {
    match elem {
        ElementKind::I8 | ElementKind::I16 => ElementKind::I32,
        _ => ElementKind::U32,
    }
}

fn cast(e: E, to: ElementKind) -> E {
    E::Cast(
        Box::new(e),
        CastSpec {
            to,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    )
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![E::Local(0)],
    }
}

/// `tid` as a local, `in[tid]` as a local of the narrow type, then `store`'s statement.
fn kernel(
    name: &str,
    elem: ElementKind,
    out: ElementKind,
    store: impl Fn(ElementKind) -> E,
) -> KernelFn {
    let compute = compute_of(elem);
    let body = vec![
        Stmt::Let {
            local: 0,
            init: cast(E::Exec(ExecId::ThreadIdx(0)), compute),
        },
        Stmt::Load {
            local: 1,
            place: at(0),
        },
        Stmt::Store {
            place: at(1),
            value: store(compute),
        },
    ];
    let mut k = KernelFn::new(
        name,
        compute,
        vec![
            Param::buffer("input", elem, BufferRole::Read, Some(1)),
            Param::buffer("output", out, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(compute), LocalDecl::scalar(elem)],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([THREADS, 1, 1], [1, 1, 1]));
    k
}

pub(crate) fn widen_kernel(elem: ElementKind) -> KernelFn {
    kernel("widen", elem, compute_of(elem), |compute| {
        cast(E::Local(1), compute)
    })
}

pub(crate) fn wrap_kernel(elem: ElementKind) -> KernelFn {
    kernel("wrap", elem, elem, |compute| {
        cast(
            E::Add(
                Box::new(cast(E::Local(1), compute)),
                Box::new(E::ConstInt(1, compute)),
            ),
            elem,
        )
    })
}

/// The input element `t` as a signed number: a repeating pattern that includes the type's extremes.
pub(crate) fn input(elem: ElementKind, t: u32) -> i64 {
    let (lo, hi): (i64, i64) = match elem {
        ElementKind::I8 => (-128, 127),
        ElementKind::U8 => (0, 255),
        ElementKind::I16 => (-32768, 32767),
        _ => (0, 65535),
    };
    // lo, hi, lo+1, hi-1, 0 or midpoint, -1 or 1 ... then a stride through the range.
    let span = hi - lo + 1;
    match t % 6 {
        0 => lo,
        1 => hi,
        2 => lo + 1,
        3 => hi - 1,
        _ => lo + (i64::from(t) * 37) % span,
    }
}

/// The bytes the host writes for the input: element `t` little-endian at `width` bytes, padded with
/// zero words to `words` words.
pub(crate) fn input_words(elem: ElementKind, words: usize) -> Vec<u32> {
    let width = width(elem);
    let mut bytes = Vec::new();
    for t in 0..THREADS {
        let v = input(elem, t);
        bytes.extend_from_slice(&v.to_le_bytes()[..width]);
    }
    bytes.resize(words * 4, 0);
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub(crate) fn width(elem: ElementKind) -> usize {
    match elem {
        ElementKind::I8 | ElementKind::U8 => 1,
        _ => 2,
    }
}

/// Wrap `v` to `elem`'s width, signedness as `elem`: the value a narrowing cast gives.
pub(crate) fn wrap(elem: ElementKind, v: i64) -> i64 {
    match elem {
        ElementKind::I8 => i64::from(v as i8),
        ElementKind::U8 => i64::from(v as u8),
        ElementKind::I16 => i64::from(v as i16),
        _ => i64::from(v as u16),
    }
}

/// The expected 32-bit result of [`widen_kernel`] at `t` (the sign- or zero-extension of the input).
pub(crate) fn widened(elem: ElementKind, t: u32) -> u32 {
    wrap(elem, input(elem, t)) as u32
}

/// The wrong widening, the mutant: the *other* extension (zero-extend a signed element, sign-extend
/// an unsigned one).
pub(crate) fn widened_wrong(elem: ElementKind, t: u32) -> u32 {
    let bits = width(elem) * 8;
    let v = wrap(elem, input(elem, t));
    let raw = (v as u32) & ((1u64 << bits) - 1) as u32;
    if matches!(elem, ElementKind::I8 | ElementKind::I16) {
        raw
    } else {
        // Sign-extend the unsigned element's bit pattern.
        (((raw << (32 - bits)) as i32) >> (32 - bits)) as u32
    }
}

/// The expected narrow result of [`wrap_kernel`] at `t`, as the element's own bits.
pub(crate) fn wrapped(elem: ElementKind, t: u32) -> u32 {
    (wrap(elem, input(elem, t) + 1) as u32) & ((1u64 << (width(elem) * 8)) - 1) as u32
}
