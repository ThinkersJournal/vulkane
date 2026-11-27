// SPDX-License-Identifier: MIT OR Apache-2.0
//! The shared text walker for Vulkane's kernel emitters (ROADMAP rows 8a, 8b and 8g for GLSL and WGSL).
//!
//! [`lower_kernel`] takes an Unpopped kernel-layer [`KernelFn`] and a
//! [`Dialect`] and writes shader source. The walk (scope checks, statements, control flow, the
//! structure of a `For`) is written once; a dialect supplies only the spellings that differ between
//! GLSL and WGSL: declarations, the entry point, literals, operators, barriers.
//!
//! ## What it lowers (v1)
//!
//! Integer kernels (`I32`/`U32`): buffer parameters, static `Shared` arrays, locals, `Let`, `Load`,
//! `Assign`, `Store`, `MaskedLoad` and `MaskedStore` (ROADMAP 9e: a guarded plain access, so the memory is not touched
//! where the mask is false), `Barrier` and `Fence`, `If`, `For`, `While`, `Break`, `Continue`, `Return`, and
//! the expressions `Local`, `ConstInt`, `Exec` (thread, block, `BlockDim`, `GridDim`), `Add`, `Sub`,
//! `Mul`, the comparisons, `BitAnd`/`BitOr`/`BitXor`, `Shl`/`Shr` (the count masked to the width, as
//! the SPIR-V emitter does) and `Select`. Anything else is a typed
//! [`UnsupportedKernel`](unpopped::backend::LowerError::UnsupportedKernel) naming it.
//!
//! ## Semantics shared with the SPIR-V emitter
//!
//! A comparison gives `1` or `0` in the compute dtype. A `For` evaluates its bounds once; a zero step
//! leaves the loop at once (a shader cannot trap). A barrier under thread-divergent control is refused
//! by `check_kernel` (`BarrierDivergent`) before any text is written.
//!
//! ## Refused by name
//!
//! The twelve NVIDIA-only statements of board 434 C and its extension (`CopyAsync`, `Collective`,
//! `MbarInit`, `WgmmaWait`, ...) are refused by name, at any nesting depth, before the walk starts.

mod coverage;
mod decline;
mod dialect;
mod expr;
mod scalarize;
mod scope;
mod stmt;

pub use coverage::{Coverage, coverage, not_lowered_note};
pub use decline::{
    DECLINED_434C, DECLINED_434C_EXT, collective_is_ruled, declined_by_name, declined_by_name_with,
};
pub use dialect::{BIT_OPS, COMPARISONS, Decl, Dialect, Layout, SubgroupCollective, operator};
pub use scalarize::scalarize;

use unpopped::backend::{GeneratedKernel, LowerError};
use unpopped::kernel::KernelFn;

/// Lower `kernel` to source in `dialect`, or say why it cannot.
///
/// # Errors
///
/// [`LowerError::UnsupportedKernel`] naming what is not lowered, with the by-name patent refusal
/// first.
pub fn lower_kernel(
    dialect: &dyn Dialect,
    kernel: &KernelFn,
) -> Result<GeneratedKernel, LowerError> {
    if let Some(e) = declined_by_name_with(dialect.subgroup_collectives(), dialect.name(), kernel) {
        return Err(e);
    }
    // Packed values (vectors) become scalars first: `lower_dag` has no seam for them.
    let scalarized = scalarize(kernel)?;
    let kernel = scalarized.as_ref().unwrap_or(kernel);
    let sc = scope::check(dialect, kernel)?;
    // The body first: a rounding conversion in it adds a declaration to the prologue.
    let mut body = String::new();
    stmt::emit_body(dialect, kernel, &sc, &mut body)?;
    let mut out = dialect.prologue_for(&sc.layout(kernel), stmt::uses_collective(&kernel.body));
    out.push_str(&body);
    out.push_str(dialect.epilogue());
    Ok(GeneratedKernel::new(kernel.name.clone(), out))
}
