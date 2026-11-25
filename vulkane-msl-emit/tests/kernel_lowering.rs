// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 8o, first slice: the MSL text of the shared integer kernels (the block-reverse kernel,
//! the nested break/continue/return kernel, the grid-id kernel, the `Private`-array kernel) and the
//! typed declines for what this emitter does not spell yet.
//!
//! The text is pinned, not compiled: a Metal compiler needs macOS, which no runner here has (the
//! README says so). Each assertion names the spelling it pins.

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::exec::{MemorySpace, Scope};
use unpopped::kernel::{KernelFn, Stmt};
use unpopped_vocab::ElementKind;
use vulkane_msl_emit::Msl;
use vulkane_test_support::{
    atomic_kernel, float_kernel, flow_kernel, grid_kernel, private_kernel, reverse_kernel,
    subgroup_kernel, wide_kernel,
};

const BLOCK: u32 = 64;

fn lower(k: &KernelFn) -> GeneratedKernel {
    Msl.lower_kernel(k)
        .expect("the MSL backend lowers this kernel")
}

fn text(k: &KernelFn) -> String {
    lower(k).source
}

fn decline(k: &KernelFn) -> String {
    match Msl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail } | LowerError::UnsupportedOp { detail }) => {
            detail
        }
        other => panic!("expected a typed decline, got {other:?}"),
    }
}

fn reverse(with_barrier: bool) -> KernelFn {
    reverse_kernel::reverse(BLOCK, with_barrier, reverse_kernel::SharedLen::Static)
}

#[test]
fn the_signature_binds_buffers_and_the_thread_and_threadgroup_builtins() {
    let t = text(&reverse(true));
    assert!(
        t.contains("#include <metal_stdlib>\nusing namespace metal;"),
        "{t}"
    );
    assert!(t.contains("kernel void reverse("), "{t}");
    assert!(
        t.contains("device const uint* b0 [[buffer(0)]],"),
        "a Read buffer is const: {t}"
    );
    assert!(
        t.contains("device uint* b1 [[buffer(1)]],"),
        "a Write buffer is not: {t}"
    );
    assert!(
        t.contains("uint3 tid [[thread_position_in_threadgroup]],"),
        "{t}"
    );
    assert!(
        t.contains("uint3 gid [[threadgroup_position_in_grid]])"),
        "{t}"
    );
    assert!(
        t.contains("// threadsPerThreadgroup: 64 x 1 x 1"),
        "the size is a dispatch fact: {t}"
    );
}

#[test]
fn a_shared_array_is_a_threadgroup_array_at_kernel_scope() {
    let t = text(&reverse(true));
    let decl = t
        .find("threadgroup uint a0[64];")
        .expect("threadgroup array");
    let first_stmt = t.find("tid.x").expect("a thread id read");
    assert!(
        decl < first_stmt,
        "Metal needs the declaration at the top of the kernel: {t}"
    );
}

#[test]
fn the_barrier_is_the_full_one_and_the_racy_control_has_none() {
    let with = text(&reverse(true));
    let without = text(&reverse(false));
    let barrier = "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);";
    assert_eq!(with.matches(barrier).count(), 1, "{with}");
    assert_eq!(
        without.matches("threadgroup_barrier").count(),
        0,
        "{without}"
    );
}

#[test]
fn a_private_array_is_a_plain_per_thread_array() {
    let t = text(&private_kernel::private_index(8, MemorySpace::Private));
    assert!(
        !t.contains("threadgroup uint a0"),
        "a Private array is not shared: {t}"
    );
    assert!(t.contains("uint a0["), "{t}");
}

#[test]
fn nested_loops_with_break_continue_and_return_lower() {
    let t = text(&flow_kernel::flow_kernel());
    for spelling in ["break;", "continue;", "return;"] {
        assert!(t.contains(spelling), "{spelling} missing: {t}");
    }
}

#[test]
fn block_ids_read_the_threadgroup_position() {
    let t = text(&grid_kernel::grid_kernel());
    assert!(t.contains("gid.x"), "{t}");
}

#[test]
fn a_fence_is_spelled_at_block_and_device_scope_and_collapses_to_seq_cst() {
    let mut k = reverse(true);
    k.body.0.push(Stmt::Fence {
        scope: Scope::Block,
        order: unpopped::kernel::MemoryOrder::AcqRel,
    });
    k.body.0.push(Stmt::Fence {
        scope: Scope::Device,
        order: unpopped::kernel::MemoryOrder::SeqCst,
    });
    let t = text(&k);
    assert!(
        t.contains("atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);"),
        "{t}"
    );
    assert!(
        t.contains(
            "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);"
        ),
        "{t}"
    );
}

#[test]
fn what_is_not_spelled_is_declined_by_name() {
    // Floats, 64-bit, atomics, subgroup ids, a scalar parameter and an out-of-scope barrier.
    // (An `F32` kernel lowers; Metal has no `double`, so `F64` is declined.)
    let float = decline(&float_kernel::float_kernel(
        ElementKind::F64,
        float_kernel::Case::MulAdd,
    ));
    assert!(float.contains("msl"), "{float}");
    let sqrt = decline(&float_kernel::float_kernel(
        ElementKind::F32,
        float_kernel::Case::Sqrt,
    ));
    assert!(sqrt.contains("msl"), "{sqrt}");
    let wide = decline(&wide_kernel::wide_kernel(ElementKind::I64));
    assert!(wide.contains("msl"), "{wide}");
    let subgroup = decline(&subgroup_kernel::subgroup_kernel(1, 128));
    assert!(subgroup.contains("LaneId"), "{subgroup}");
    let scalar = decline(&reverse_kernel::reverse(
        BLOCK,
        true,
        reverse_kernel::SharedLen::Spec,
    ));
    assert!(scalar.contains("scalar parameter"), "{scalar}");
    let mut device = reverse(true);
    device.body.0.push(Stmt::Barrier {
        scope: Scope::Device,
        kind: unpopped::kernel::BarrierKind::Full,
    });
    assert!(decline(&device).contains("msl"));
}

#[test]
fn an_f32_kernel_pins_safe_math_and_no_contraction_and_keeps_every_op_unfused() {
    let src = text(&float_kernel::float_kernel(
        ElementKind::F32,
        float_kernel::Case::MulAdd,
    ));
    assert!(src.contains("#pragma METAL fp math_mode(safe)"), "{src}");
    assert!(src.contains("#pragma METAL fp contract(off)"), "{src}");
    assert!(src.contains("device const float*"), "{src}");
    assert!(src.contains("const float "), "{src}");
}

#[test]
fn float_max_propagates_nan_and_floor_is_spelled() {
    let max = text(&float_kernel::float_kernel(
        ElementKind::F32,
        float_kernel::Case::Max,
    ));
    assert!(max.contains("isnan("), "{max}");
    let floor = text(&float_kernel::float_kernel(
        ElementKind::F32,
        float_kernel::Case::Floor,
    ));
    assert!(floor.contains("floor("), "{floor}");
}

#[test]
fn atomics_use_atomic_types_relaxed_explicit_calls_and_a_retried_weak_exchange() {
    let src = text(&atomic_kernel::atomic_kernel(false));
    assert!(src.contains("device atomic_uint* b0"), "{src}");
    assert!(
        src.contains("atomic_fetch_add_explicit(&b0[0u], 1u, memory_order_relaxed)"),
        "{src}"
    );
    assert!(src.contains("atomic_fetch_min_explicit(&b0[1u]"), "{src}");
    // The weak exchange is retried while the value seen still equals `expected`.
    assert!(
        src.contains("atomic_compare_exchange_weak_explicit(&b0[2u], &cas_e, cas_d, memory_order_relaxed, memory_order_relaxed) && cas_e == cas_x"),
        "{src}"
    );
    assert!(!src.contains("atomicAdd"), "{src}");
}

#[test]
fn the_racy_control_reads_and_writes_the_atomic_cell_through_atomic_load_and_store() {
    let src = text(&atomic_kernel::atomic_kernel(true));
    assert!(src.contains("atomic_load_explicit(&b0[0u]"), "{src}");
    assert!(src.contains("atomic_store_explicit(&b0[0u]"), "{src}");
}

#[test]
fn an_atomic_at_subgroup_scope_or_with_a_stronger_order_is_declined() {
    use unpopped::kernel::MemoryOrder;
    let mut k = atomic_kernel::atomic_kernel(false);
    for st in &mut k.body.0 {
        if let Stmt::AtomicRmw { scope, .. } = st {
            *scope = Scope::Subgroup;
        }
    }
    assert!(decline(&k).contains("Subgroup"));
    let mut k = atomic_kernel::atomic_kernel(false);
    for st in &mut k.body.0 {
        if let Stmt::AtomicRmw { order, .. } = st {
            *order = MemoryOrder::SeqCst;
        }
    }
    assert!(decline(&k).contains("msl"));
}

#[test]
fn the_plan_path_is_declined() {
    assert_eq!(Msl.name(), "msl");
    assert_eq!(Msl.provider(), "vulkane");
}
