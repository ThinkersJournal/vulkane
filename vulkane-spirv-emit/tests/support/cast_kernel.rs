// SPDX-License-Identifier: MIT OR Apache-2.0
//! The `Cast` test cases (int <-> float, `F32` <-> `F64`), shared by the unit tests and
//! `tests/kernel_cast_live.rs`. Every case is a kernel `out[t] = (dst) in[t]` over 256 threads and the
//! host oracle in native Rust arithmetic, which follows the same rules as `unpopped`'s `cast_ref.rs`:
//! int to float and `f64` to `f32` round to nearest even; float to int rounds per the mode, maps NaN to
//! 0 and clamps (Rust's `as` is exactly that for floats). Each case also carries a `mutant`: the
//! output a *wrong* lowering would give (truncating instead of rounding, wrapping instead of
//! clamping), which must differ from `want` somewhere or the case could not catch it.

use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr as E};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::{self, F16, F32, F64, I32, U32};

pub(crate) const THREADS: usize = 256;

pub(crate) struct CastCase {
    pub(crate) name: &'static str,
    pub(crate) src: ElementKind,
    pub(crate) dst: ElementKind,
    pub(crate) compute: ElementKind,
    pub(crate) round: RoundMode,
    pub(crate) sat: Saturation,
    /// Per-element bit patterns of the input, cycled over the threads.
    pub(crate) input: Vec<u64>,
    /// The result for each input element, as bit patterns of `dst`.
    pub(crate) want: Vec<u64>,
    /// What a wrong lowering would give for each input element.
    pub(crate) mutant: Vec<u64>,
}

fn bits32(v: f32) -> u64 {
    u64::from(v.to_bits())
}

fn bits64(v: f64) -> u64 {
    v.to_bits()
}

fn i(v: i32) -> u64 {
    u64::from(v as u32)
}

const INTS: [i32; 14] = [
    0,
    1,
    -1,
    16_777_216,
    16_777_217,
    16_777_219,
    16_777_221,
    33_554_435,
    -16_777_217,
    123_456_789,
    i32::MAX,
    i32::MIN,
    i32::MAX - 63,
    -33_554_435,
];

const UINTS: [u32; 10] = [
    0,
    1,
    16_777_217,
    16_777_219,
    33_554_435,
    2_147_483_649,
    u32::MAX,
    u32::MAX - 127,
    4_000_000_001,
    123_456_789,
];

const FLOATS: [f32; 24] = [
    0.0,
    -0.0,
    0.5,
    1.5,
    2.5,
    3.5,
    -0.5,
    -1.5,
    -2.5,
    2.7,
    -2.7,
    1e10,
    -1e10,
    f32::MAX,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    2_147_483_520.0,
    -2_147_483_648.0,
    2_147_483_648.0,
    4_294_967_040.0,
    4_294_967_296.0,
    -1.0e-3,
    16_777_216.0,
];

const DOUBLES: [f64; 14] = [
    1.0 + 1.0 / 16_777_216.0,
    1.0 + 3.0 / 16_777_216.0,
    0.1,
    1e39,
    -1e39,
    1e-40,
    1e-46,
    f64::NAN,
    f64::INFINITY,
    -0.0,
    3.402_823_567_797_336_6e38,
    16_777_217.0,
    1.0e-310,
    -2.5,
];

/// `x` rounded to an integer toward zero, ignoring the clamp (the wrapping mutant of float to int).
fn wrap_i32(x: f64) -> i32 {
    (x.trunc() as i64) as i32
}

pub(crate) fn cases() -> Vec<CastCase> {
    let ne = RoundMode::NearestEven;
    let mut v = Vec::new();
    v.push(CastCase {
        name: "i32_to_f32",
        src: I32,
        dst: F32,
        compute: F32,
        round: ne,
        sat: Saturation::Wrap,
        input: INTS.iter().map(|&x| i(x)).collect(),
        want: INTS.iter().map(|&x| bits32(x as f32)).collect(),
        // Round toward zero instead of to nearest even: wrong on 16_777_219 and the like.
        mutant: INTS
            .iter()
            .map(|&x| bits32(rtz_f32(f64::from(x))))
            .collect(),
    });
    v.push(CastCase {
        name: "u32_to_f32",
        src: U32,
        dst: F32,
        compute: F32,
        round: ne,
        sat: Saturation::Wrap,
        input: UINTS.iter().map(|&x| u64::from(x)).collect(),
        want: UINTS.iter().map(|&x| bits32(x as f32)).collect(),
        // Read the unsigned value as signed: wrong above 2^31.
        mutant: UINTS.iter().map(|&x| bits32(x as i32 as f32)).collect(),
    });
    v.push(CastCase {
        name: "i32_to_f64",
        src: I32,
        dst: F64,
        compute: F64,
        round: ne,
        sat: Saturation::Wrap,
        input: INTS.iter().map(|&x| i(x)).collect(),
        want: INTS.iter().map(|&x| bits64(f64::from(x))).collect(),
        // Through f32: loses the low bits of values past 2^24.
        mutant: INTS.iter().map(|&x| bits64(f64::from(x as f32))).collect(),
    });
    for (name, round, f) in [
        ("f32_to_i32_nearest_even", ne, rte as fn(f64) -> f64),
        ("f32_to_i32_toward_zero", RoundMode::TowardZero, f64::trunc),
    ] {
        v.push(CastCase {
            name,
            src: F32,
            dst: I32,
            compute: F32,
            round,
            sat: Saturation::Clamp,
            input: FLOATS.iter().map(|&x| bits32(x)).collect(),
            want: FLOATS.iter().map(|&x| i(f(f64::from(x)) as i32)).collect(),
            // Wrap instead of clamp (and a NaN that is not zero).
            mutant: FLOATS
                .iter()
                .map(|&x| i(wrap_i32(f(f64::from(x)))))
                .collect(),
        });
    }
    v.push(CastCase {
        name: "f32_to_u32_nearest_even",
        src: F32,
        dst: U32,
        compute: F32,
        round: ne,
        sat: Saturation::Clamp,
        input: FLOATS.iter().map(|&x| bits32(x)).collect(),
        want: FLOATS
            .iter()
            .map(|&x| u64::from(rte(f64::from(x)) as u32))
            .collect(),
        // Negative values wrap instead of clamping to 0.
        mutant: FLOATS
            .iter()
            .map(|&x| u64::from(wrap_i32(rte(f64::from(x))) as u32))
            .collect(),
    });
    v.push(CastCase {
        name: "f64_to_f32",
        src: F64,
        dst: F32,
        compute: F32,
        round: ne,
        sat: Saturation::Wrap,
        input: DOUBLES.iter().map(|&x| bits64(x)).collect(),
        want: DOUBLES.iter().map(|&x| bits32(x as f32)).collect(),
        mutant: DOUBLES.iter().map(|&x| bits32(rtz_f32(x))).collect(),
    });
    v.push(CastCase {
        name: "f32_to_f64",
        src: F32,
        dst: F64,
        compute: F64,
        round: ne,
        sat: Saturation::Wrap,
        input: FLOATS.iter().map(|&x| bits32(x)).collect(),
        want: FLOATS.iter().map(|&x| bits64(f64::from(x))).collect(),
        // Widening that drops the sign of zero.
        mutant: FLOATS.iter().map(|&x| bits64(f64::from(x) + 0.0)).collect(),
    });
    let tie16: f32 = 1.0 / 2048.0;
    let tie16_64: f64 = 1.0 / 2048.0;
    let f16_in32: Vec<f32> = vec![
        1.0 + tie16,
        1.0 + 3.0 * tie16,
        65_504.0,
        65_519.0,
        65_520.0,
        1e-8,
        2f32.powi(-25),
        2f32.powi(-24),
        3.0 * 2f32.powi(-25),
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.1,
        6.1035e-5,
        1e10,
    ];
    v.push(CastCase {
        name: "f32_to_f16",
        src: F32,
        dst: F16,
        compute: F32,
        round: ne,
        sat: Saturation::Wrap,
        input: f16_in32.iter().map(|&x| bits32(x)).collect(),
        want: f16_in32
            .iter()
            .map(|&x| u64::from(f16_round(f64::from(x), false)))
            .collect(),
        mutant: f16_in32
            .iter()
            .map(|&x| u64::from(f16_round(f64::from(x), true)))
            .collect(),
    });
    let f16_bits: Vec<u16> = vec![
        0x3c00, 0x7bff, 0x0001, 0x03ff, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x0400, 0x3555, 0x8001,
        0xbc00,
    ];
    v.push(CastCase {
        name: "f16_to_f32",
        src: F16,
        dst: F32,
        compute: F32,
        round: ne,
        sat: Saturation::Wrap,
        input: f16_bits.iter().map(|&b| u64::from(b)).collect(),
        want: f16_bits.iter().map(|&b| bits32(f16_decode(b))).collect(),
        // Subnormals flushed to zero.
        mutant: f16_bits
            .iter()
            .map(|&b| bits32(if b & 0x7c00 == 0 { 0.0 } else { f16_decode(b) }))
            .collect(),
    });
    let f64_in16: Vec<f64> = vec![
        1.0 + tie16_64 + 2f64.powi(-40),
        1.0 + tie16_64,
        1.0 + 3.0 * tie16_64 + 2f64.powi(-40),
        65_519.999_999,
        65_520.0,
        2f64.powi(-25) + 2f64.powi(-60),
        0.1,
        f64::NAN,
    ];
    v.push(CastCase {
        name: "f64_to_f16",
        src: F64,
        dst: F16,
        compute: F32,
        round: ne,
        sat: Saturation::Wrap,
        input: f64_in16.iter().map(|&x| bits64(x)).collect(),
        want: f64_in16
            .iter()
            .map(|&x| u64::from(f16_round(x, false)))
            .collect(),
        // Rounded twice, through f32: wrong where the first rounding creates a tie.
        mutant: f64_in16
            .iter()
            .map(|&x| u64::from(f16_round(f64::from(x as f32), false)))
            .collect(),
    });
    // Integer <-> half: a half tops out at 65504, so ties sit at 2049 / 2051 and the overflow edge at
    // 65519 / 65520; i32::MAX and u32::MAX overflow to infinity.
    let half_ints: [i32; 14] = [
        0,
        1,
        -1,
        2049,
        2051,
        4097,
        -2049,
        65_504,
        65_519,
        65_520,
        65_535,
        -65_520,
        i32::MAX,
        i32::MIN,
    ];
    v.push(CastCase {
        name: "i32_to_f16",
        src: I32,
        dst: F16,
        compute: F32,
        round: ne,
        sat: Saturation::Wrap,
        input: half_ints.iter().map(|&x| i(x)).collect(),
        want: half_ints
            .iter()
            .map(|&x| u64::from(f16_round(f64::from(x), false)))
            .collect(),
        // Round toward zero: wrong at 2051 and at the overflow edge (65520 is infinity, not 65504).
        mutant: half_ints
            .iter()
            .map(|&x| u64::from(f16_round(f64::from(x), true)))
            .collect(),
    });
    let half_uints: [u32; 10] = [
        0,
        1,
        2049,
        2051,
        4097,
        65_504,
        65_519,
        65_520,
        4_000_000_001,
        u32::MAX,
    ];
    v.push(CastCase {
        name: "u32_to_f16",
        src: U32,
        dst: F16,
        compute: F32,
        round: ne,
        sat: Saturation::Wrap,
        input: half_uints.iter().map(|&x| u64::from(x)).collect(),
        want: half_uints
            .iter()
            .map(|&x| u64::from(f16_round(f64::from(x), false)))
            .collect(),
        mutant: half_uints
            .iter()
            .map(|&x| u64::from(f16_round(f64::from(x), true)))
            .collect(),
    });
    // Half to integer: the 12 half bit patterns of `f16_bits` plus rounding cases, a negative, infinities and NaN.
    let h2i_bits: Vec<u16> = vec![
        0x0000, 0x8000, 0x3800, 0x3e00, 0x4100, 0x4300, 0xb800, 0xbe00, 0xc100, 0x4166, 0xc166,
        0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x1000, 0x63d0,
    ];
    for (name, dst, round, f, signed) in [
        (
            "f16_to_i32_nearest_even",
            I32,
            ne,
            rte as fn(f64) -> f64,
            true,
        ),
        (
            "f16_to_i32_toward_zero",
            I32,
            RoundMode::TowardZero,
            f64::trunc,
            true,
        ),
        (
            "f16_to_u32_nearest_even",
            U32,
            ne,
            rte as fn(f64) -> f64,
            false,
        ),
    ] {
        let val = |b: u16| f(f64::from(f16_decode(b)));
        v.push(CastCase {
            name,
            src: F16,
            dst,
            compute: F32,
            round,
            sat: Saturation::Clamp,
            input: h2i_bits.iter().map(|&b| u64::from(b)).collect(),
            want: h2i_bits
                .iter()
                .map(|&b| {
                    if signed {
                        i(val(b) as i32)
                    } else {
                        u64::from(val(b) as u32)
                    }
                })
                .collect(),
            // Wrap instead of clamp: wrong for the infinities and, unsigned, the negatives.
            mutant: h2i_bits
                .iter()
                .map(|&b| {
                    let w = wrap_i32(val(b));
                    if signed { i(w) } else { u64::from(w as u32) }
                })
                .collect(),
        });
    }
    v
}

/// Round half to even, `f64` to integral `f64`.
fn rte(x: f64) -> f64 {
    x.round_ties_even()
}

/// `x` converted to `f32` rounding toward zero (the mutant of round-to-nearest-even).
fn rtz_f32(x: f64) -> f32 {
    let r = x as f32;
    if r.is_finite() && f64::from(r).abs() > x.abs() {
        f32::from_bits(r.to_bits() - 1)
    } else if r.is_infinite() && x.is_finite() {
        f32::MAX.copysign(r)
    } else {
        r
    }
}

fn width_bytes(t: ElementKind) -> usize {
    match t {
        F64 => 8,
        F16 => 2,
        _ => 4,
    }
}

/// `x` as IEEE half bits, one rounding (nearest even, or toward zero for the mutant).
pub(crate) fn f16_round(x: f64, toward_zero: bool) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    if x.is_nan() {
        return 0x7e00;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return sign | 0x7c00;
    }
    if ax == 0.0 {
        return sign;
    }
    let round = |s: f64| {
        if toward_zero {
            s.floor()
        } else {
            s.round_ties_even()
        }
    };
    let mut e = ((ax.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    if e < -14 {
        // Subnormal range: units of 2^-24 (an exact scaling).
        return sign | round(ax * 2f64.powi(24)) as u16;
    }
    let mut q = round(ax * 2f64.powi(10 - e)) as u32;
    if q == 2048 {
        e += 1;
        q = 1024;
    }
    if e > 15 {
        return sign | if toward_zero { 0x7bff } else { 0x7c00 };
    }
    sign | (((e + 15) as u16) << 10) | (q - 1024) as u16
}

/// The exact `f32` value of half bits `b`.
pub(crate) fn f16_decode(b: u16) -> f32 {
    let sign = if b & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let (e, m) = (i32::from((b >> 10) & 0x1f), f32::from(b & 0x3ff));
    match e {
        0 => sign * m * 2f32.powi(-24),
        31 if m == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The words of `elems` (cycled to `THREADS` elements, little-endian, `width_bytes` each), padded
/// with zero words to `min_words`.
pub(crate) fn words(t: ElementKind, elems: &[u64], min_words: usize) -> Vec<u32> {
    let w = width_bytes(t);
    let mut bytes = Vec::new();
    for k in 0..THREADS {
        bytes.extend_from_slice(&elems[k % elems.len()].to_le_bytes()[..w]);
    }
    bytes.resize(bytes.len().div_ceil(4).max(min_words) * 4, 0);
    bytes
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn at(buffer: u32) -> Place {
    Place::Logical {
        buffer,
        coords: vec![E::Exec(ExecId::ThreadIdx(0))],
    }
}

pub(crate) fn kernel(c: &CastCase) -> KernelFn {
    let value = E::Cast(
        Box::new(E::Local(0)),
        CastSpec {
            to: c.dst,
            round: c.round,
            sat: c.sat,
        },
    );
    let mut k = KernelFn::new(
        "cast",
        c.compute,
        vec![
            Param::buffer("in", c.src, BufferRole::Read, Some(1)),
            Param::buffer("out", c.dst, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(c.src)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value,
            },
        ]),
    );
    k.launch = Some(LaunchShape::new([THREADS as u32, 1, 1], [1, 1, 1]));
    k
}

/// Whether two result word sequences agree element by element: integers exactly, floats bit-exactly
/// except that every NaN equals every NaN.
pub(crate) fn agree(t: ElementKind, got: &[u32], want: &[u32]) -> Result<(), String> {
    let w = width_bytes(t);
    let bytes = |x: &[u32]| x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let (g, h) = (bytes(got), bytes(want));
    let val = |c: &[u8]| c.iter().rev().fold(0u64, |a, &b| (a << 8) | u64::from(b));
    for (k, (a, b)) in g.chunks(w).zip(h.chunks(w)).enumerate() {
        let (a, b) = (val(a), val(b));
        let nan = |x: u64| match t {
            F32 => f32::from_bits(x as u32).is_nan(),
            F64 => f64::from_bits(x).is_nan(),
            F16 => (x & 0x7c00) == 0x7c00 && (x & 0x3ff) != 0,
            _ => false,
        };
        if !(nan(a) && nan(b)) && a != b {
            return Err(format!("element {k}: got {a:#x} want {b:#x}"));
        }
    }
    Ok(())
}
