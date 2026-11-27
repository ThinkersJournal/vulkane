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
//! `F32` and `F16` kernels lower through `float.rs` (`half` buffers, locals and literals `1.5h`; half to or from `F32` is
//! `static_cast`, round to nearest even per MSL 2.23; half to or from an integer is declined, not guessed): `#pragma METAL fp math_mode(safe)` and `contract(off)` (Metal 3.2;
//! fast math is Metal's default: no NaN, INF or signed zeros, reassociation and fusion allowed; `safe` leaves FP
//! contraction *on*, hence `contract(off)` as well), every operation pinned in a
//! `const` temporary, the exact unary ops only (`rint` for round-half-even), NaN-propagating max/min, and
//! guarded float-to-integer conversions. Text pinned, **not compiled** (no Metal compiler in CI). Subnormal handling and
//! the rounding mode are the device's (MSL 8.1, 8.2: denormals "may be flushed to zero"; "either round ties to even
//! or round toward zero ... may be supported"): no pragma makes a Metal float kernel bit-exact IEEE.
//!
//! **Exactness is NOT claimed for `F32` or `Bf16` (KISS ops spec, the bit-stable clause: "a kernel that flushes subnormal inputs
//! or results to zero (FTZ/DAZ) does not round at the storage dtype's full precision, so it is non-conformant under
//! bit-stable and MUST decline `Exact` rather than advertise the value").** Measured on the hosted paravirtual Metal
//! device (run 45748530618): `F32` and `Bf16` flush subnormals to zero, `F16` preserves them. So no output of this emitter
//! may be advertised as `Exact`/bit-stable for `F32` or `Bf16` on that device, and `F16` can be (for the value pairs
//! tested there). The emitter has no exactness or fidelity declaration to make decline: `Backend` carries none, the plan
//! path (`lower`, and so `lower_variants`' default of no variants) is not implemented, and a `GeneratedKernel` states none.
//! A caller that needs `Exact` must decline `F32`/`Bf16` MSL itself; the subnormal device cases print a `DEVICE note` with
//! what a device does.
//!
//! Atomics (`AtomicRmw` Add/Min/Max/And/Or/Xor/Exchange, `AtomicCas`) on `I32`/`U32` buffers and `threadgroup` arrays: the
//! target is declared `atomic_int`/`atomic_uint`, every access is `atomic_*_explicit` with `memory_order_relaxed`, and the
//! compare-exchange (Metal has only the weak form, whose spurious failure the IR cannot express) is retried while the value
//! seen still equals `expected`. Scope Block and Device only; a non-`Relaxed` order is a decline (Metal 4.1 adds
//! acquire/release, not spelled yet). Text pinned, **not compiled**.
//!
//! Subgroup ids: `LaneId`, `SubgroupId` and `SubgroupSize` are `thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup` and
//! `threads_per_simdgroup`, declared in the signature only when the body reads them; a subgroup `Barrier` is
//! `simdgroup_barrier`. `BlockDim` and `GridDim` are launch constants the walker folds. A launch's required subgroup size
//! range cannot be requested in Metal, so the source carries it as a `// requires a subgroup size in a..=b` comment.
//!
//! 8- and 16-bit integer buffers and locals (`I8`/`U8`/`I16`/`U16`) in an `I32`/`U32` kernel are `char`, `uchar`, `short`
//! and `ushort`, reached by C-style casts (modular narrowing, sign- or zero-extending widening); an atomic beside one is a
//! decline (the atomic target would have to be shown 32-bit). Text pinned, **not compiled**.
//!
//! `Bf16` kernels (Metal 3.1 and later; MSL 2.1, 2.24) are `bfloat` with explicit `static_cast<bfloat>` literals (neither
//! `float` nor `half` converts implicitly to it); the unary ops go through `float` because the math functions of MSL 6.6 take
//! `half` and `float` only (the widening and the exact result make that exact); `isnan` and the comparisons are defined on
//! `bfloat` (6.5). Whether a device rounds a bfloat add or multiply once is the device's (see the float caveat above).
//! Packed (vector) locals reach the dialect already scalarized by the walker, so a `vec2`/`vec4` kernel lowers lane by lane
//! with the same bits; `float2`, `half4` and `uchar4` are not spelled. FP8 is declined by name: MSL 4.1 2.21 has the FP8
//! formats only as `packed_numeric_type` and tensor format tags.
//!
//! ## SIMD-group functions: not lowered, on a ruling
//!
//! MSL 4.1 6.10.2 has a function for most `Stmt::Collective` kinds: `simd_shuffle`, `simd_shuffle_up`, `simd_shuffle_down`,
//! `simd_shuffle_xor`, `simd_all`, `simd_any`, `simd_ballot`, `simd_sum`, `simd_product`, `simd_max`, `simd_min`,
//! `simd_and`, `simd_or`, `simd_xor`, `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum` and `simd_broadcast`.
//! None is emitted. `Collective` and `CollectiveTuple` are two of the twelve statements of board 434 C: CireSnave
//! 2026-11-13, verbatim, *"emitters may lower AsyncWait/CopyAsync/AsyncCommit/Collective for NVIDIA devices ... and decline
//! by name for non-NVIDIA devices"*. Metal is not an NVIDIA device, so every kind is refused by name before lowering
//! (`tests/patent_decline.rs` pins each kind, at six nesting depths). Only CireSnave can change that ruling. Were it
//! lifted, the mapping would still not be one to one: the MSL functions act on the *active* threads (6.10.2: "A thread
//! needs to only read data from another active thread"; `simd_ballot` "sets the bits that correspond to inactive threads
//! to 0") where the IR's collective is over every lane of its scope, and `T` excludes `bool`, `bfloat`, `long` and `ulong`.
//! What is lowered is not a collective: `LaneId`, `SubgroupId`, `SubgroupSize` and `simdgroup_barrier`.
//!
//! ## Declined by name
//!
//! A kernel named with an MSL keyword or type name (`float`, `kernel`, `tid`, ...: `kernel void float(` cannot compile),
//! `Bf16` with a non-bfloat constant, `Fp8*`, `F64` (Metal has no `double`), float `Div` (integer `Div` is lowered, total: a zero divisor gives the dividend), `Sqrt` and the transcendentals (looser bound than the contract tier),
//! 64-bit atomics (only max/min on `atomic_ulong` are permitted by the specification and the hosted device cannot create a pipeline for them: ROADMAP 8r), float atomics, a subgroup-scope atomic, scalar parameters, the device ids
//! (`DeviceRank`, `DeviceCount`), and a `Barrier` or `Fence` at a scope other than block or subgroup (a barrier) / block or device (a
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

/// MSL and C++ words a kernel function cannot be named (the entry point is emitted as `kernel void <name>(`).
const RESERVED: &[&str] = &[
    "bool",
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "long",
    "ulong",
    "half",
    "bfloat",
    "float",
    "double",
    "void",
    "kernel",
    "vertex",
    "fragment",
    "device",
    "constant",
    "thread",
    "threadgroup",
    "main",
    "metal",
    "auto",
    "class",
    "struct",
    "union",
    "enum",
    "template",
    "namespace",
    "using",
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "return",
    "const",
    "static",
    "volatile",
    "sizeof",
    "typedef",
    "true",
    "false",
    "this",
    "new",
    "delete",
    "operator",
    "public",
    "private",
    "protected",
    "friend",
    "virtual",
    "typename",
    "tid",
    "gid",
    "lane",
    "sgid",
    "sgsize",
];

impl Backend for Msl {
    fn name(&self) -> &str {
        "msl"
    }
    fn provider(&self) -> &str {
        "vulkane"
    }
    fn supports_dtype(&self, dtype: ElementKind, _target: TargetId) -> DtypeSupport {
        match dtype {
            ElementKind::I32
            | ElementKind::U32
            | ElementKind::F32
            | ElementKind::F16
            | ElementKind::Bf16 => DtypeSupport::Supported,
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
        if RESERVED.contains(&kernel.name.as_str()) {
            return Err(LowerError::UnsupportedKernel {
                detail: format!(
                    "msl backend: the kernel name `{}` is an MSL keyword or type name, so `kernel void {}(` would not compile; rename the kernel",
                    kernel.name, kernel.name
                ),
            });
        }
        if kernel
            .params
            .iter()
            .any(|p| matches!(p.kind, ParamKind::Scalar { .. }))
        {
            return Err(LowerError::UnsupportedKernel {
                detail: format!(
                    "msl backend: a scalar parameter is not lowered: the shared walker lowers one only as the extent of a shared array, and the MSL specification (5.8, function constants) does not say that a function constant may bound an array, which cannot be checked without a Metal compiler (kernel `{}`)",
                    kernel.name
                ),
            });
        }
        let generated = if matches!(
            kernel.compute,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16
        ) {
            let d = float::MslFloat {
                half: kernel.compute == ElementKind::F16,
                bf16: kernel.compute == ElementKind::Bf16,
                ..Default::default()
            };
            vulkane_kernel_text::lower_kernel(&d, kernel)?
        } else {
            let wide = matches!(kernel.compute, ElementKind::I64 | ElementKind::U64);
            let signed = matches!(kernel.compute, ElementKind::I32 | ElementKind::I64);
            if wide && holds_atomic(&kernel.body) {
                return Err(LowerError::UnsupportedKernel {
                    detail: format!(
                        "msl backend: a 64-bit atomic is not lowered: Metal has no signed 64-bit atomic type; the MSL specification (4.1, 6.16.4.6) permits only `atomic_max_explicit` and `atomic_min_explicit` on `atomic_ulong` (device memory, relaxed order, void result); the `atomic_fetch_*`, load, store and compare-exchange forms are rejected by the Apple compiler (run 45751376095), and the permitted max/min compile but the hosted paravirtual device fails to create a pipeline for them (run 45752252131), so no 64-bit atomic is measured on hardware (ROADMAP 8r) (kernel `{}`)",
                        kernel.name
                    ),
                });
            }
            let d = dialect::IntKernel {
                wide,
                signed,
                ..Default::default()
            };
            vulkane_kernel_text::lower_kernel(&d, kernel)?
        };
        Ok(with_subgroup_note(kernel, generated))
    }
}

/// A launch's required subgroup size range cannot be requested in Metal (the device's `threads_per_simdgroup` is what a
/// kernel reads), so the source says what the kernel asked for, beside the `threadsPerThreadgroup` line, rather than
/// dropping it silently.
fn with_subgroup_note(kernel: &KernelFn, g: GeneratedKernel) -> GeneratedKernel {
    let Some(r) = kernel.launch.and_then(|l| l.subgroup) else {
        return g;
    };
    let note = format!(
        "// requires a subgroup size in {}..={}; Metal cannot request one (threads_per_simdgroup is the device's): check it at dispatch\n",
        r.min, r.max
    );
    let source = g.source.replacen(
        "// threadsPerThreadgroup:",
        &format!("{note}// threadsPerThreadgroup:"),
        1,
    );
    GeneratedKernel::new(g.name, source)
}

/// Whether `block` holds an atomic read-modify-write or compare-exchange at any depth.
fn holds_atomic(block: &unpopped::kernel::Block) -> bool {
    use unpopped::kernel::Stmt;
    block.0.iter().any(|s| match s {
        Stmt::AtomicRmw { .. } | Stmt::AtomicCas { .. } => true,
        Stmt::If { then, els, .. } => holds_atomic(then) || holds_atomic(els),
        Stmt::For { body, .. } | Stmt::While { body, .. } => holds_atomic(body),
        _ => false,
    })
}
