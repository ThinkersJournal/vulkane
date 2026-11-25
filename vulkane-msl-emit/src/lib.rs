// SPDX-License-Identifier: MIT OR Apache-2.0
//! IR to Metal Shading Language compute-kernel lowering for [Unpopped](https://crates.io/crates/unpopped).
//!
//! Vulkane owns MSL in the CireSnave ecosystem (ROADMAP row 8o). [`Msl`] lowers a kernel-layer
//! [`KernelFn`] through the shared walker in `vulkane-kernel-text`; this crate supplies only the MSL
//! spellings (the [`Dialect`](vulkane_kernel_text::Dialect)).
//!
//! ## Shape of the output
//!
//! `#include <metal_stdlib>`, `using namespace metal;`, then one `kernel void <name>(...)`: a
//! `device const T* b<N> [[buffer(N)]]` parameter per `Read` buffer (`device T*` otherwise), then
//! `uint3 tid [[thread_position_in_threadgroup]]` and `uint3 gid [[threadgroup_position_in_grid]]`.
//! The body opens with the `threadgroup T a<K>[N];` arrays (Metal requires them at kernel scope), the
//! per-thread arrays and the locals. The workgroup size is not in the source: Metal takes it at dispatch
//! (`threadsPerThreadgroup`), so the host must launch with the kernel's block size.
//!
//! ## Lowered
//!
//! `I32`/`U32` kernels: buffers, static `Shared` and `Private` arrays, locals, `Let`, `Load`,
//! `Assign`, `Store`, `If`, `For`, `While`, `Break`, `Continue`, `Return`, `ThreadIdx` and `BlockIdx`,
//! the arithmetic, the comparisons, the bitwise operators and shifts (the count masked at 31),
//! `Select`. `Barrier { Block }` is `threadgroup_barrier(mem_flags::mem_threadgroup |
//! mem_flags::mem_device)`, the full workgroup-and-buffer barrier of the other emitters. A `Fence`
//! is `atomic_thread_fence` at threadgroup or device scope with `memory_order_seq_cst`: **every
//! `MemoryOrder` collapses to the strongest**, stronger than asked, never weaker.
//!
//! `F32` kernels lower through `float.rs`: `#pragma METAL fp math_mode(safe)` and `contract(off)` (Metal 3.2;
//! fast math is Metal's default and would flush denormals and fuse multiply-add), every operation pinned in a
//! `const` temporary, the exact unary ops only (`rint` for round-half-even), NaN-propagating max/min, and
//! guarded float-to-integer conversions. Text pinned, **not compiled** (no Metal compiler in CI).
//!
//! ## Declined by name
//!
//! `F64` (Metal has no `double`), `Div`, `Sqrt` and the transcendentals (looser bound than the contract tier), other
//! 64-bit, 8/16-bit integer and vector kernels, atomics, scalar parameters, subgroup and
//! `BlockDim`/`GridDim` ids, and a `Barrier` or `Fence` at a scope other than block (and device for a
//! fence): each a typed `UnsupportedKernel` from the shared walker naming it. The twelve NVIDIA-only
//! statements of board 434 C are refused by name first. The plan path (`Backend::lower`) is not
//! implemented.

mod dialect;
mod float;

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::kernel::{KernelFn, ParamKind};
use unpopped::plan::KernelPlan;
use unpopped_vocab::{ElementKind, TargetId};

pub use dialect::Msl;

impl Backend for Msl {
    fn name(&self) -> &str {
        "msl"
    }
    fn provider(&self) -> &str {
        "vulkane"
    }
    fn supports_dtype(&self, dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        match dtype {
            ElementKind::I32 | ElementKind::U32 | ElementKind::F32 => DtypeSupport::Supported,
            _ => DtypeSupport::Unsupported,
        }
    }
    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Err(LowerError::UnsupportedSchedule {
            detail: format!(
                "msl backend: the kernel layer only; the plan path is not implemented (op `{}`)",
                plan.op_name
            ),
        })
    }
    fn lower_kernel(&self, kernel: &KernelFn) -> Result<GeneratedKernel, LowerError> {
        if kernel
            .params
            .iter()
            .any(|p| matches!(p.kind, ParamKind::Scalar { .. }))
        {
            return Err(LowerError::UnsupportedKernel {
                detail: format!(
                    "msl backend: a scalar parameter is not lowered (a Metal function constant or `constant` buffer is not spelled yet; kernel `{}`)",
                    kernel.name
                ),
            });
        }
        if kernel.compute == ElementKind::F32 {
            return vulkane_kernel_text::lower_kernel(&float::MslFloat, kernel);
        }
        vulkane_kernel_text::lower_kernel(&Msl, kernel)
    }
}
