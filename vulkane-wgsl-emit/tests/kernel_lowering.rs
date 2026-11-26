// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP rows 8a, 8b and 8g for WGSL: the block-reverse kernel and the nested break/continue/early
//! return kernel of `vulkane-spirv-emit`'s tests, lowered by `Wgsl::lower_kernel`, parsed and
//! validated by `naga`, compiled to SPIR-V by naga and run on a device against the same host
//! references the SPIR-V emitter's tests use.
//!
//! Without a device, `glslangValidator` or `spirv-val` the tests follow `VULKANE_REQUIRE_DEVICE`,
//! `VULKANE_REQUIRE_GLSLANG` and `VULKANE_REQUIRE_SPIRV_VAL` rather than passing silently.

#[path = "../../vulkane-spirv-emit/tests/support/atomic_kernel.rs"]
mod atomic_kernel;
#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "../../vulkane-spirv-emit/tests/support/flow_kernel.rs"]
mod flow_kernel;
#[path = "../../vulkane-spirv-emit/tests/support/gpu.rs"]
mod gpu;
#[path = "../../vulkane-spirv-emit/tests/support/grid_kernel.rs"]
mod grid_kernel;
#[path = "../../vulkane-spirv-emit/tests/support/private_kernel.rs"]
mod private_kernel;
#[path = "../../vulkane-spirv-emit/tests/support/reverse_kernel.rs"]
mod reverse_kernel;
#[path = "../../vulkane-spirv-emit/tests/support/subgroup_kernel.rs"]
mod subgroup_kernel;

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::exec::MemorySpace;
use unpopped::kernel::KernelFn;
use vulkane_spirv_emit::WORKGROUP_SIZE;
use vulkane_wgsl_emit::Wgsl;

fn lower(k: &KernelFn) -> GeneratedKernel {
    Wgsl.lower_kernel(k)
        .expect("the WGSL backend lowers this kernel")
}

fn reverse(with_barrier: bool) -> GeneratedKernel {
    lower(&reverse_kernel::reverse(
        WORKGROUP_SIZE,
        with_barrier,
        false,
    ))
}

/// Parse and validate WGSL `source` with naga; the SPIR-V words naga writes for it.
fn naga_words(name: &str, source: &str) -> Result<Vec<u32>, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("naga validation of {name}: {e:?}"))?;
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|e| format!("naga SPIR-V output for {name}: {e:?}"))
}

/// The SPIR-V words naga writes for `source` once its `@id(id)` override is fixed to `value`. naga's
/// SPIR-V writer refuses an unresolved override, so the extent is fixed at translation here; a WebGPU
/// consumer sets it at pipeline creation.
fn naga_words_with_override(source: &str, id: u32, value: u32) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).expect("parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validates");
    let mut constants = naga::back::PipelineConstants::default();
    constants.insert(id.to_string(), f64::from(value));
    let (module, info) =
        naga::back::pipeline_constants::process_overrides(&module, &info, None, &constants)
            .expect("fixes the override");
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .expect("SPIR-V")
}

fn words(k: &GeneratedKernel) -> Vec<u32> {
    naga_words(&k.name, &k.source).unwrap_or_else(|e| {
        panic!(
            "naga rejects `{}`:
{e}
{}",
            k.name, k.source
        )
    })
}

#[test]
fn the_reverse_kernel_has_the_barrier_and_its_control_does_not() {
    let (with, without) = (reverse(true), reverse(false));
    assert_eq!(
        with.source.matches("workgroupBarrier();").count(),
        1,
        "{}",
        with.source
    );
    assert_eq!(
        without.source.matches("workgroupBarrier").count(),
        0,
        "{}",
        without.source
    );
    assert!(
        with.source.contains("var<workgroup> a0: array<u32, 256>;"),
        "{}",
        with.source
    );
    assert!(
        with.source.contains("var<storage, read> b0"),
        "{}",
        with.source
    );
    assert!(
        with.source.contains("@compute @workgroup_size(256, 1, 1)"),
        "{}",
        with.source
    );
}

#[test]
fn naga_accepts_both_kernels() {
    for k in [
        reverse(true),
        reverse(false),
        lower(&flow_kernel::flow_kernel()),
    ] {
        words(&k);
    }
}

/// ROADMAP 8b's "naga negative" did not hold when measured: naga 30.0 (all validation flags, all
/// capabilities) ACCEPTS a `workgroupBarrier()` under thread-dependent control flow, although the
/// WGSL spec requires uniform control flow there (Tint rejects it). So the guard is the walker's
/// own: `check_kernel`'s `BarrierDivergent` stops the kernel before any text is written.
#[test]
fn a_barrier_under_divergent_control_is_declined_before_any_text_is_written() {
    use unpopped::ir::{BinaryOp::CmpEq, ScalarExpr as E};
    use unpopped::kernel::{BarrierKind, Block, Stmt};
    let thread0 = E::Binary(
        CmpEq,
        Box::new(E::Exec(unpopped::exec::ExecId::ThreadIdx(0))),
        Box::new(E::ConstInt(0, unpopped_vocab::ElementKind::U32)),
    );
    let mut k = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    k.body.0.insert(
        3,
        Stmt::If {
            cond: thread0,
            then: Block(vec![Stmt::Barrier {
                scope: unpopped::exec::Scope::Block,
                kind: BarrierKind::Full,
            }]),
            els: Block(vec![]),
        },
    );
    declined(&k, "BarrierDivergent");
    // The measurement above, kept honest: if naga starts rejecting it this line says so.
    let src = "@group(0) @binding(0) var<storage, read_write> b: array<u32>;
               @compute @workgroup_size(64)
               fn k(@builtin(local_invocation_id) lid: vec3<u32>) {
                   if (b[lid.x] != 0u) { workgroupBarrier(); }
}
";
    eprintln!(
        "naga 30.0 on a divergent barrier: accepted = {}",
        naga_words("k", src).is_ok()
    );
}

#[test]
fn a_block_reversed_through_workgroup_memory_is_exact_with_the_barrier() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("wgsl_kernel_barrier") else {
        return; // skip already declared
    };
    let input: Vec<u32> = (0..WORKGROUP_SIZE).map(|i| i * 7 + 1).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();
    let k = reverse(true);
    assert_eq!(gpu.run(&k.name, &words(&k), &[&input]), want);
    // Control, reported not asserted (a missing barrier is a race a device may not show).
    let racy = reverse(false);
    let got = gpu.run(&racy.name, &words(&racy), &[&input]);
    let wrong = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    eprintln!(
        "wgsl kernel_barrier control: without the barrier {wrong} of {} cells differ",
        want.len()
    );
}

#[test]
fn nested_loops_break_continue_and_return_match_the_host_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("wgsl_kernel_flow") else {
        return;
    };
    let k = lower(&flow_kernel::flow_kernel());
    let got = gpu.run(&k.name, &words(&k), &[&[0u32; 256]]);
    for (t, &g) in got.iter().enumerate() {
        match flow_kernel::expected(t as u32) {
            Some(w) => assert_eq!(g, w, "thread {t}"),
            None => assert_eq!(g, gpu::SENTINEL, "thread {t} returned before its store"),
        }
    }
}

/// Row 8f: the shared 3-D kernel (non-cubic block and grid) through WGSL, every cell checked.
#[test]
fn a_non_cubic_3d_launch_writes_every_cell_its_own_ids_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("wgsl_kernel_grid") else {
        return;
    };
    let k = lower(&grid_kernel::grid_kernel());
    let got = gpu.run_grid(&k.name, &words(&k), grid_kernel::cells(), grid_kernel::GRID);
    let nx = (grid_kernel::BLOCK[0] * grid_kernel::GRID[0]) as usize;
    let ny = (grid_kernel::BLOCK[1] * grid_kernel::GRID[1]) as usize;
    for (i, &w) in got.iter().enumerate() {
        let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
        let want = grid_kernel::expected(x as u32, y as u32, z as u32);
        assert_eq!(
            w, want,
            "cell ({x},{y},{z})
{}",
            k.source
        );
    }
}

fn declined(k: &KernelFn, needle: &str) {
    match Wgsl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
        }
        other => panic!("wanted an UnsupportedKernel naming `{needle}`, got {other:?}"),
    }
}

/// Row 8a: the reversal with the shared array sized by an `override` (id 2 is the scalar parameter's
/// index), fixed to the block size.
#[test]
fn a_block_reversed_through_a_specialization_sized_array_is_exact_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("wgsl_kernel_dynamic") else {
        return;
    };
    let k = lower(&reverse_kernel::reverse(WORKGROUP_SIZE, true, true));
    assert!(k.source.contains("@id(2) override"), "{}", k.source);
    let input: Vec<u32> = (0..WORKGROUP_SIZE).map(|i| i * 7 + 1).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();
    let words = naga_words_with_override(&k.source, 2, WORKGROUP_SIZE);
    assert_eq!(gpu.run(&k.name, &words, &[&input]), want, "{}", k.source);
}

/// Row 8k: a per-thread `Private` array indexed by data, through WGSL; each thread reads its own value.
#[test]
fn a_private_array_with_data_dependent_indexing_is_per_thread_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("wgsl_kernel_private") else {
        return;
    };
    let n = WORKGROUP_SIZE;
    let k = lower(&private_kernel::private_index(n, MemorySpace::Private));
    let words = words(&k);
    let input: Vec<u32> = (0..n).map(|t| (t * 5 + 1) % 4).collect();
    let want: Vec<u32> = (0..n).map(|t| t + 10 * input[t as usize]).collect();
    assert_eq!(gpu.run(&k.name, &words, &[&input]), want, "{}", k.source);
    // Control: the indices differ between threads, so a wrong-index lowering could not pass.
    assert!(input.iter().any(|&i| i != input[0]));
}

/// Row 8k decline: a `Constant` allocation has no initialiser in the IR.
#[test]
fn a_constant_allocation_is_a_typed_decline() {
    declined(
        &private_kernel::read_only_constant(WORKGROUP_SIZE),
        "Constant space",
    );
}

#[test]
fn what_wgsl_cannot_say_is_a_typed_decline_naming_it() {
    use unpopped::exec::Scope;
    use unpopped::kernel::{BarrierKind, MemoryOrder, Stmt};
    let mut fence = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    fence.body.0.insert(
        3,
        Stmt::Fence {
            scope: Scope::Block,
            order: MemoryOrder::AcqRel,
        },
    );
    declined(&fence, "Fence");

    let mut subgroup = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    subgroup.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Subgroup,
            kind: BarrierKind::Full,
        },
    );
    declined(&subgroup, "Barrier");

    let mut float = reverse_kernel::reverse(WORKGROUP_SIZE, true, false);
    float.compute = unpopped_vocab::ElementKind::F32;
    declined(&float, "a F32 kernel is not lowered");
}

/// Row 8d: 1024 contended atomic updates through WGSL (`array<atomic<u32>>`, naga, a device); the
/// atomic kernel is exact and the plain load/add/store control loses updates on the same device
/// (reported, not asserted).
#[test]
fn contended_atomics_are_exact_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("wgsl_kernel_atomic") else {
        return;
    };
    let run = |racy: bool| -> Vec<u32> {
        let k = lower(&atomic_kernel::atomic_kernel(racy));
        gpu.run_grid(&k.name, &words(&k), 4, [atomic_kernel::BLOCKS, 1, 1])
    };
    let got = run(false);
    let (s, n) = (atomic_kernel::SENTINEL, atomic_kernel::THREADS);
    assert_eq!(got[0], s.wrapping_add(n), "lost updates");
    assert_eq!(got[1], 0, "unsigned minimum of the global ids");
    assert!((1..=n).contains(&got[2]), "cas winner wrote {}", got[2]);
    assert_eq!(
        got[3],
        s.wrapping_add(1),
        "exactly one compare-exchange winner"
    );
    eprintln!(
        "wgsl kernel_atomic control: a plain increment reached {} of {n}",
        run(true)[0].wrapping_sub(s)
    );
}

/// The atomic spellings and the atomic buffer type are in the source; the control keeps the type
/// (the same buffer still takes atomics) but reads and writes cell 0 through `atomicLoad/Store`.
#[test]
fn the_atomic_buffer_is_typed_and_accessed_atomically() {
    let a = lower(&atomic_kernel::atomic_kernel(false)).source;
    let r = lower(&atomic_kernel::atomic_kernel(true)).source;
    assert!(
        a.contains("var<storage, read_write> b0: array<atomic<u32>>;"),
        "{a}"
    );
    for f in [
        "atomicAdd(&b0[",
        "atomicMin(&b0[",
        "atomicCompareExchangeWeak(&b0[",
    ] {
        assert!(a.contains(f), "{a}");
    }
    assert_eq!(a.matches("atomicAdd").count(), 2, "{a}");
    assert_eq!(r.matches("atomicAdd").count(), 1, "{r}");
    assert_eq!(
        (
            a.matches("atomicLoad").count(),
            a.matches("atomicStore").count()
        ),
        (0, 0),
        "{a}"
    );
    assert_eq!(
        (
            r.matches("atomicLoad").count(),
            r.matches("atomicStore").count()
        ),
        (1, 1),
        "{r}"
    );
}

/// Mutation: the control's `atomicLoad` left as a plain read of the atomic buffer. naga must refuse
/// it (positive control: the unmutated source is accepted).
#[test]
fn a_plain_read_of_an_atomic_buffer_is_rejected_by_naga() {
    let r = lower(&atomic_kernel::atomic_kernel(true)).source;
    assert!(naga_words("atomics", &r).is_ok(), "{r}");
    let anchor = "atomicLoad(&b0[0u])";
    assert_eq!(r.matches(anchor).count(), 1, "{r}");
    let mutant = r.replace(anchor, "b0[0u]");
    assert!(naga_words("atomics", &mutant).is_err(), "{mutant}");
}

/// A compare-exchange that fails spuriously must not report the expected value as the old one: the
/// lowering retries until it swaps or sees a different value.
#[test]
fn the_weak_compare_exchange_is_retried_in_a_loop() {
    let a = lower(&atomic_kernel::atomic_kernel(false)).source;
    assert_eq!(a.matches("loop {").count(), 1, "{a}");
    assert!(
        a.contains("cas_r.exchanged || cas_r.old_value != cas_e"),
        "{a}"
    );
}

/// Conditions of the ruling: only relaxed, and only block or device scope.
#[test]
fn a_stronger_order_or_a_subgroup_scope_is_a_typed_decline() {
    use unpopped::exec::Scope;
    use unpopped::kernel::{MemoryOrder, Stmt};
    let mut k = atomic_kernel::atomic_kernel(false);
    if let Some(Stmt::AtomicRmw { order, .. }) = k.body.0.get_mut(1) {
        *order = MemoryOrder::AcqRel;
    } else {
        panic!("the kernel's second statement is the atomic add");
    }
    declined(&k, "AcqRel atomic is not lowered");
    let mut k = atomic_kernel::atomic_kernel(false);
    if let Some(Stmt::AtomicRmw { scope, .. }) = k.body.0.get_mut(1) {
        *scope = Scope::Subgroup;
    }
    declined(&k, "Subgroup scope is not lowered");
}

/// Row 8c: the subgroup ids need WGSL `enable subgroups;`, which naga 30 (this emitter's validity
/// oracle) rejects as "not yet implemented", so a kernel reading them is a typed decline naming the
/// id; the probe below pins the reason so the decline is lifted when naga accepts it.
#[test]
fn the_subgroup_ids_are_a_typed_decline_while_naga_rejects_enable_subgroups() {
    match Wgsl.lower_kernel(&subgroup_kernel::subgroup_kernel(1, 128)) {
        Err(LowerError::UnsupportedOp { detail }) => {
            assert!(detail.contains("execution id LaneId"), "{detail}");
        }
        other => panic!("wanted a decline naming LaneId, got {other:?}"),
    }
    let probe = "enable subgroups;
@compute @workgroup_size(1)
fn k() {}
";
    let err = naga::front::wgsl::parse_str(probe).expect_err("naga 30 rejects it");
    assert!(
        err.emit_to_string(probe).contains("not yet supported"),
        "naga now accepts `enable subgroups;`? lift the decline"
    );
}
