// SPDX-License-Identifier: MIT OR Apache-2.0
//! Which float ops a *kernel* may lower: only those whose Vulkan-guaranteed accuracy is no looser
//! than the contract tier Unpopped rates the op at (`contract.rs`, `unary_ulp`/`binary_ulp`).
//! Anything else is declined by name rather than lowered under a tier the device does not promise.
//! The table is in the crate docs; the source is the Vulkan specification, appendix "Vulkan
//! Environment for SPIR-V", section "Precision of Individual Operations", Tables "Precision of Core
//! SPIR-V Instructions" (3) and "Precision of GLSL.std.450 Instructions" (4).
//!
//! The plan path (`Backend::lower`) is unchanged: this gate applies to `lower_kernel` only.

use unpopped::ir::{ArithOp, BinaryOp, UnaryOp};

use crate::family::Family;

const CITE: &str = "Vulkan spec, Vulkan Environment for SPIR-V, Precision of Individual Operations";

/// Why a kernel may not lower float unary `op` at `fam`, or `None` when it may.
pub(crate) fn unary_refusal(op: UnaryOp, fam: Family) -> Option<String> {
    match op {
        // Exact or correctly rounded (Table 3 `OpFNegate`, `OpFMul`; Table 4 `FAbs`, `Floor`, `Ceil`,
        // `RoundEven`, `Trunc`) or built from compares and selects, which are exact; the contract
        // tier is 0 and the spec bound is 0.
        UnaryOp::Neg
        | UnaryOp::Abs
        | UnaryOp::Sqr
        | UnaryOp::Relu
        | UnaryOp::Sign
        | UnaryOp::Step
        | UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round
        | UnaryOp::Trunc => None,
        // Table 4 `InverseSqrt`: "2 ULP", equal to the contract's tier 2 for `Rsqrt`; the spec
        // lists 32- and 16-bit only, so not at 64 bits.
        UnaryOp::Rsqrt if fam == Family::F32 => None,
        UnaryOp::Rsqrt => Some(format!(
            "Rsqrt is declined at {fam:?}: {CITE} gives `inversesqrt()` \"2 ULP\" for 32-bit (and 16-bit) only, and no bound-to-tier match was checked for the other widths, so only F32 lowers it"
        )),
        UnaryOp::Sqrt => Some(format!(
            "Sqrt is rated 0 ulp by the contract but {CITE} gives `sqrt()` only \"Inherited from 1.0 / inversesqrt()\" (and `inversesqrt()` is 2 ULP), looser than correctly rounded"
        )),
        UnaryOp::Recip => Some(format!(
            "Recip is rated 0 ulp by the contract but is `OpFDiv`, which {CITE} bounds at 2.5 ULP (single, |y| in [2^-126, 2^126])"
        )),
        _ => Some(match spec_bound(op) {
            Some((bound, tier)) => format!(
                "{op:?} is declined at {fam:?}: {CITE} gives {bound}, looser than the contract tier of {tier} ulp (spirvenv.adoc read 2026-11-23); not lowered under a tier the device does not promise"
            ),
            None => format!(
                "{op:?} is declined at {fam:?}: it is built on `Exp`, `Log`, `OpFDiv` or an \"inherited from\" GLSL.std.450 formula whose bound ({CITE}) is looser than its contract tier; not lowered under a tier the device does not promise"
            ),
        }),
    }
}

/// The Vulkan bound (quoted from the spec source) and the contract tier of the ops that are one
/// GLSL.std.450 call and looser than their tier. Source: `appendices/spirvenv.adoc` in
/// `KhronosGroup/Vulkan-Docs` (`main`), "Precision of GLSL.std.450 Instructions", read 2026-11-23.
fn spec_bound(op: UnaryOp) -> Option<(&'static str, f64)> {
    match op {
        UnaryOp::Exp | UnaryOp::Exp2 => Some(("\"3 + 2 x |x| ULP\" (single)", 2.0)),
        UnaryOp::Log | UnaryOp::Log2 => Some((
            "\"3 ULP outside the range [0.5, 2.0]. Absolute error < 2^-21 inside\" (single)",
            1.0,
        )),
        UnaryOp::Sin | UnaryOp::Cos => Some((
            "\"Absolute error <= 2^-11 inside the range [-pi, pi]\" (no ULP bound at all)",
            2.0,
        )),
        UnaryOp::Atan => Some(("\"4096 ULP\" (single)", 2.0)),
        UnaryOp::Tan => Some(("\"Inherited from sin()/cos()\"", 4.0)),
        _ => None,
    }
}

/// Why a kernel may not lower float arithmetic `op`, or `None`.
pub(crate) fn arith_refusal(op: ArithOp) -> Option<String> {
    match op {
        // Table 3: `OpFAdd`, `OpFSub`, `OpFMul` "Correctly rounded".
        ArithOp::Add | ArithOp::Sub | ArithOp::Mul => None,
        ArithOp::Div => Some(format!(
            "float Div is correctly rounded in the oracle (C `/`) but {CITE} bounds `OpFDiv` at 2.5 ULP for |y| in [2^-126, 2^126]"
        )),
        other => Some(format!("arithmetic {other:?} has no float kernel lowering")),
    }
}

/// Why a kernel may not lower float binary `op`, or `None`.
pub(crate) fn binary_refusal(op: BinaryOp) -> Option<String> {
    if op.is_cmp()
        || matches!(
            op,
            BinaryOp::Max | BinaryOp::Min | BinaryOp::FmaxIeee | BinaryOp::FminIeee
        )
    {
        // Compares are "Correct result"; `Max`/`Min` are `IsNan` plus compares plus selects; `NMax`/
        // `NMin` are "Correct result" (Table 4).
        return None;
    }
    Some(format!(
        "{op:?} is declined in a kernel: its lowering uses `OpFDiv`, `Pow`, `FRem` or an unbounded GLSL.std.450 call, none shown no looser than the contract tier in {CITE}"
    ))
}
