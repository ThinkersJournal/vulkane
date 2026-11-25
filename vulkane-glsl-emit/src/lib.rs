// SPDX-License-Identifier: MIT OR Apache-2.0
//! IR to GLSL compute-shader lowering for [Unpopped](https://crates.io/crates/unpopped).
//!
//! Vulkane owns GLSL in the CireSnave ecosystem. [`Glsl`] lowers a kernel-layer
//! [`KernelFn`] to GLSL 4.50 through the shared walker in
//! `vulkane-kernel-text` (ROADMAP rows 8a, 8b, 8g); this crate supplies only the GLSL spellings (the
//! [`Dialect`](vulkane_kernel_text::Dialect)).
//!
//! ## Shape of the output
//!
//! `#version 450`, a `layout(local_size_x = .., ..) in;` line, one `layout(std430, set = 0, binding = N)
//! buffer` block per parameter (`readonly` for a `Read` buffer, `writeonly` for a `Write` one) named
//! `b<N>` holding `data[]`, `shared T a<K>[N];` per allocation, and `void main()` with the locals
//! declared at its top. The entry is `main`, as GLSL requires: [`GeneratedKernel::name`] is the
//! kernel's own name.
//!
//! ## Barriers and fences
//!
//! `Barrier { Block }` is `memoryBarrierShared(); memoryBarrierBuffer(); barrier();`: GLSL's
//! `barrier()` orders execution only, so the memory barriers make it the full workgroup-and-buffer
//! barrier the SPIR-V emitter emits. `Barrier { Subgroup }` is `subgroupMemoryBarrier();
//! subgroupBarrier();` and adds `#extension GL_KHR_shader_subgroup_basic : enable`. A `Fence` is
//! `groupMemoryBarrier()` (block), `memoryBarrier()` (device) or `subgroupMemoryBarrier()`; GLSL
//! has no memory order on them, so **every `MemoryOrder` collapses to the one full fence**: stronger
//! than asked, never weaker. A device or cluster barrier is a typed decline.
//!
//! ## 64-bit integers
//!
//! An `I64`/`U64` kernel is spelled `int64_t`/`uint64_t` with `#extension GL_ARB_gpu_shader_int64 :
//! require` (literals `5l`/`5ul`), shift counts masked at 63, and an execution id kept the `u32` the
//! IR types it as and carried to the dtype by a wrapping `Cast` (`int64_t(x)`). A buffer index is
//! narrowed to `int` (GLSL indexes with 32 bits). The device must have `shaderInt64`: the host's
//! check, as for SPIR-V. 64-bit atomics, a specialization-constant extent and integer division are
//! typed declines (the shared walker lowers no `Div`).
//!
//! The plan path (`Backend::lower`) is not implemented: this emitter lowers kernels only.

mod dialect;

use unpopped::backend::{Backend, DtypeSupport, GeneratedKernel, LowerError};
use unpopped::kernel::KernelFn;
use unpopped::plan::KernelPlan;
use unpopped_vocab::{ElementKind, TargetId};

pub use dialect::Glsl;

impl Backend for Glsl {
    fn name(&self) -> &str {
        "glsl"
    }
    fn provider(&self) -> &str {
        "vulkane"
    }
    fn supports_dtype(&self, dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        match dtype {
            ElementKind::I32 | ElementKind::U32 | ElementKind::I64 | ElementKind::U64 => {
                DtypeSupport::Supported
            }
            _ => DtypeSupport::Unsupported,
        }
    }
    fn lower(&self, plan: &KernelPlan<'_>) -> Result<GeneratedKernel, LowerError> {
        Err(LowerError::UnsupportedSchedule {
            detail: format!(
                "glsl backend: the kernel layer only; the plan path is not implemented (op `{}`)",
                plan.op_name
            ),
        })
    }
    fn lower_kernel(&self, kernel: &KernelFn) -> Result<GeneratedKernel, LowerError> {
        let wide = matches!(kernel.compute, ElementKind::I64 | ElementKind::U64);
        let signed = matches!(kernel.compute, ElementKind::I64);
        vulkane_kernel_text::lower_kernel(&dialect::Text { wide, signed }, kernel)
    }
}
