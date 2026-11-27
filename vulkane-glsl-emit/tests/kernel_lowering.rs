// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP rows 8a, 8b and 8g for GLSL: the block-reverse kernel and the nested break/continue/early
//! return kernel of `vulkane-spirv-emit`'s tests, lowered by `Glsl::lower_kernel`, compiled by
//! `glslangValidator` to SPIR-V, validated by `spirv-val` and run on a device against the same host
//! references the SPIR-V emitter's tests use.
//!
//! Without a device, `glslangValidator` or `spirv-val` the tests follow `VULKANE_REQUIRE_DEVICE`,
//! `VULKANE_REQUIRE_GLSLANG` and `VULKANE_REQUIRE_SPIRV_VAL` rather than passing silently.

use vulkane_test_support::atomic_kernel;
use vulkane_test_support::common;
use vulkane_test_support::flow_kernel;
use vulkane_test_support::glslang;
use vulkane_test_support::gpu;
use vulkane_test_support::grid_kernel;
use vulkane_test_support::private_kernel;
use vulkane_test_support::reverse_kernel;
use vulkane_test_support::subgroup_kernel;

use glslang::{glslang_words, validates};
use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::exec::MemorySpace;
use unpopped::kernel::KernelFn;
use vulkane_glsl_emit::Glsl;
use vulkane_spirv_emit::WORKGROUP_SIZE;

fn lower(k: &KernelFn) -> GeneratedKernel {
    Glsl.lower_kernel(k)
        .expect("the GLSL backend lowers this kernel")
}

fn reverse(with_barrier: bool) -> GeneratedKernel {
    lower(&reverse_kernel::reverse(
        WORKGROUP_SIZE,
        with_barrier,
        reverse_kernel::SharedLen::Static,
    ))
}

#[test]
fn the_reverse_kernel_has_the_barrier_and_its_control_does_not() {
    let (with, without) = (reverse(true), reverse(false));
    let barrier = "memoryBarrierShared(); memoryBarrierBuffer(); barrier();";
    assert_eq!(with.source.matches(barrier).count(), 1, "{}", with.source);
    assert_eq!(
        without.source.matches("barrier()").count(),
        0,
        "{}",
        without.source
    );
    assert!(
        with.source.contains("shared uint a0[256];"),
        "{}",
        with.source
    );
    assert!(with.source.contains("readonly buffer"), "{}", with.source);
    assert!(with.source.contains("writeonly buffer"), "{}", with.source);
    assert!(with.source.contains("#version 450"), "{}", with.source);
}

#[test]
fn glslang_compiles_both_kernels_and_spirv_val_accepts_them() {
    for k in [reverse(true), lower(&flow_kernel::flow_kernel())] {
        let Some(words) = glslang_words(&k) else {
            return;
        };
        if let Some((ok, err)) = validates(&words) {
            assert!(ok, "spirv-val rejected `{}`:\n{err}\n{}", k.name, k.source);
        }
    }
}

#[test]
fn a_block_reversed_through_shared_memory_is_exact_with_the_barrier() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_barrier") else {
        return; // skip already declared
    };
    let input: Vec<u32> = (0..WORKGROUP_SIZE).map(|i| i * 7 + 1).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();
    let k = reverse(true);
    let Some(words) = glslang_words(&k) else {
        return;
    };
    assert_eq!(gpu.run("main", &words, &[&input]), want);
    // Control, reported not asserted (a missing barrier is a race a device may not show).
    let racy = reverse(false);
    if let Some(words) = glslang_words(&racy) {
        let got = gpu.run("main", &words, &[&input]);
        let wrong = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        eprintln!(
            "glsl kernel_barrier control: without the barrier {wrong} of {} cells differ",
            want.len()
        );
    }
}

#[test]
fn nested_loops_break_continue_and_return_match_the_host_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_flow") else {
        return;
    };
    let k = lower(&flow_kernel::flow_kernel());
    let Some(words) = glslang_words(&k) else {
        return;
    };
    let got = gpu.run("main", &words, &[&[0u32; 256]]);
    for (t, &g) in got.iter().enumerate() {
        match flow_kernel::expected(t as u32) {
            Some(w) => assert_eq!(g, w, "thread {t}"),
            None => assert_eq!(g, gpu::SENTINEL, "thread {t} returned before its store"),
        }
    }
}

/// Row 8f: the shared 3-D kernel (non-cubic block and grid) through GLSL, every cell checked.
#[test]
fn a_non_cubic_3d_launch_writes_every_cell_its_own_ids_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_grid") else {
        return;
    };
    let k = lower(&grid_kernel::grid_kernel());
    let Some(words) = glslang_words(&k) else {
        return;
    };
    let got = gpu.run_grid("main", &words, grid_kernel::cells(), grid_kernel::GRID);
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

/// A refusal that is the kernel CHECKER's by design (the emitter surfaces `check_kernel`'s answer): pinned as such, so
/// `declined` can stay strict about emitter-owned refusals.
fn declined_by_checker(k: &KernelFn, needle: &str) {
    match Glsl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(
                detail.contains("check_kernel refused") && detail.contains(needle),
                "wanted the checker's `{needle}` in `{detail}`"
            );
        }
        other => panic!("wanted a checker refusal naming `{needle}`, got {other:?}"),
    }
}

fn declined(k: &KernelFn, needle: &str) {
    match Glsl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
            // STRICT (audit): the refusal is the emitter's own, not the kernel checker's, whose text can already name a construct.
            assert!(
                needle.contains("check_kernel")
                    || needle.contains("Divergent")
                    || !detail.contains("check_kernel refused"),
                "STRICT: the checker refused it, not the emitter: {detail}"
            );
        }
        other => panic!("wanted an UnsupportedKernel naming `{needle}`, got {other:?}"),
    }
}

/// Row 9e: predicated access through GLSL on a device. The masked-off lanes' index is out of range, so a load that
/// ran would fault or read zero (not `FILL`), and the masked-off store must leave the plain store's value.
#[test]
fn masked_access_skips_the_memory_where_the_mask_is_false_on_a_device() {
    use vulkane_test_support::masked_kernel;
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_masked") else {
        return;
    };
    let k = lower(&masked_kernel::masked_kernel());
    let Some(words) = glslang_words(&k) else {
        return;
    };
    let got = gpu.run("main", &words, &[&masked_kernel::input()]);
    for (t, &w) in got.iter().enumerate() {
        let want = masked_kernel::expected(t as u32);
        assert_eq!(
            w, want,
            "thread {t}
{}",
            k.source
        );
    }
}

/// Row 8d: 1024 contended atomic updates through GLSL; the atomic kernel is exact and the plain
/// load/add/store control loses updates on the same device (reported, not asserted).
#[test]
fn contended_atomics_are_exact_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_atomic") else {
        return;
    };
    let run = |racy: bool| -> Option<Vec<u32>> {
        let k = lower(&atomic_kernel::atomic_kernel(racy));
        let words = glslang_words(&k)?;
        Some(gpu.run_grid("main", &words, 4, [atomic_kernel::BLOCKS, 1, 1]))
    };
    let Some(got) = run(false) else {
        return;
    };
    let (s, n) = (atomic_kernel::SENTINEL, atomic_kernel::THREADS);
    assert_eq!(got[0], s.wrapping_add(n), "lost updates");
    assert_eq!(got[1], 0, "unsigned minimum of the global ids");
    assert!((1..=n).contains(&got[2]), "cas winner wrote {}", got[2]);
    assert_eq!(
        got[3],
        s.wrapping_add(1),
        "exactly one compare-exchange winner"
    );
    if let Some(racy) = run(true) {
        eprintln!(
            "glsl kernel_atomic control: a plain increment reached {} of {n}",
            racy[0].wrapping_sub(s)
        );
    }
}

/// Row 8d decline: GLSL atomics carry no memory order, so a non-relaxed one is refused by name.
#[test]
fn a_non_relaxed_atomic_is_a_typed_decline() {
    use unpopped::kernel::{MemoryOrder, Stmt};
    let mut k = atomic_kernel::atomic_kernel(false);
    if let Some(Stmt::AtomicRmw { order, .. }) = k.body.0.get_mut(1) {
        *order = MemoryOrder::AcqRel;
    } else {
        panic!("the kernel's second statement is the atomic add");
    }
    declined(&k, "AcqRel atomic is not lowered");
}

/// The atomic spellings are in the source (the same query finds the plain-store control without them).
#[test]
fn the_atomic_kernel_spells_the_glsl_atomics_and_the_control_does_not() {
    let a = lower(&atomic_kernel::atomic_kernel(false)).source;
    let r = lower(&atomic_kernel::atomic_kernel(true)).source;
    for f in ["atomicAdd", "atomicMin", "atomicCompSwap"] {
        assert!(a.contains(f), "{a}");
    }
    assert_eq!(a.matches("atomicAdd").count(), 2, "{a}");
    assert_eq!(r.matches("atomicAdd").count(), 1, "{r}");
}

/// Row 8a: the reversal with the shared array sized by a specialization constant (SpecId 2 is the
/// scalar parameter's index), set to the block size when the pipeline is created.
#[test]
fn a_block_reversed_through_a_specialization_sized_array_is_exact_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_dynamic") else {
        return;
    };
    let k = lower(&reverse_kernel::reverse(
        WORKGROUP_SIZE,
        true,
        reverse_kernel::SharedLen::Spec,
    ));
    assert!(k.source.contains("layout(constant_id = 2)"), "{}", k.source);
    let Some(words) = glslang_words(&k) else {
        return;
    };
    let input: Vec<u32> = (0..WORKGROUP_SIZE).map(|i| i * 7 + 1).collect();
    let want: Vec<u32> = input.iter().rev().copied().collect();
    let spec = vulkane::safe::SpecializationConstants::new().add_u32(2, WORKGROUP_SIZE);
    assert_eq!(
        gpu.run_spec("main", &words, &[&input], Some(&spec)),
        want,
        "{}",
        k.source
    );
}

/// Row 8k: a per-thread `Private` array indexed by data, through GLSL; each thread reads its own value.
#[test]
fn a_private_array_with_data_dependent_indexing_is_per_thread_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_private") else {
        return;
    };
    let n = WORKGROUP_SIZE;
    let k = lower(&private_kernel::private_index(n, MemorySpace::Private));
    let Some(words) = glslang_words(&k) else {
        return;
    };
    let input: Vec<u32> = (0..n).map(|t| (t * 5 + 1) % 4).collect();
    let want: Vec<u32> = (0..n).map(|t| t + 10 * input[t as usize]).collect();
    assert_eq!(gpu.run("main", &words, &[&input]), want, "{}", k.source);
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
fn what_the_walker_does_not_lower_is_a_typed_decline_naming_it() {
    use unpopped::exec::{Extent, Scope};
    use unpopped::kernel::{BarrierKind, Stmt};
    let mut f = reverse_kernel::reverse(WORKGROUP_SIZE, true, reverse_kernel::SharedLen::Static);
    f.compute = unpopped_vocab::ElementKind::Bf16;
    declined(&f, "integer kernels only");

    // A dynamic extent from a local that no scalar parameter declares: `check_kernel` refuses it.
    let mut dynamic =
        reverse_kernel::reverse(WORKGROUP_SIZE, true, reverse_kernel::SharedLen::Spec);
    dynamic.allocs[0].extent = Extent::Param(1);
    declined_by_checker(&dynamic, "AllocExtent");

    let mut device =
        reverse_kernel::reverse(WORKGROUP_SIZE, true, reverse_kernel::SharedLen::Static);
    device.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Device,
            kind: BarrierKind::Full,
        },
    );
    declined(&device, "Barrier");

    let mut grid = reverse_kernel::reverse(WORKGROUP_SIZE, true, reverse_kernel::SharedLen::Static);
    grid.body.0.push(Stmt::GridDepWait);
    declined(&grid, "GridDepWait");
}

/// Row 8c: `LaneId`, `SubgroupId` and `SubgroupSize` through GLSL (`gl_Subgroup*`); 128 threads, so
/// two subgroups on a device of size 64. Each cell is checked against the device's reported size.
#[test]
fn lane_subgroup_and_size_ids_match_the_device_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("glsl_kernel_subgroup") else {
        return;
    };
    let size = gpu.subgroup_size();
    assert!((1..=128).contains(&size), "device subgroup size {size}");
    let k = lower(&subgroup_kernel::subgroup_kernel(1, 128));
    let Some(words) = glslang_words(&k) else {
        return;
    };
    let got = gpu.run_grid("main", &words, subgroup_kernel::BLOCK as usize, [1, 1, 1]);
    for (t, &g) in got.iter().enumerate() {
        let want = subgroup_kernel::expected(t as u32, size);
        assert_eq!(
            g, want,
            "thread {t} (subgroup size {size})
{}",
            k.source
        );
    }
}

/// The three spellings and the extension are in the source.
#[test]
fn the_subgroup_kernel_spells_the_builtins_and_the_extension() {
    let src = lower(&subgroup_kernel::subgroup_kernel(1, 128)).source;
    for w in [
        "#extension GL_KHR_shader_subgroup_basic : enable",
        "gl_SubgroupInvocationID",
        "gl_SubgroupID",
        "gl_SubgroupSize",
    ] {
        assert!(src.contains(w), "{src}");
    }
}

/// A range no Vulkan device can have is refused by name (the control lowers).
/// ROADMAP 9e: a masked access is a guarded access, so the element is not touched where the mask is false.
#[test]
fn masked_access_is_an_if_around_the_access_and_the_fill_is_the_else_branch() {
    use vulkane_test_support::masked_kernel;
    let src = lower(&masked_kernel::masked_kernel()).source;
    let lines: Vec<&str> = src.lines().map(str::trim).collect();
    let at = |pred: &dyn Fn(&str) -> bool, what: &str| {
        lines.iter().position(|l| pred(l)).unwrap_or_else(|| {
            panic!(
                "{what} missing:
{src}"
            )
        })
    };
    // The load of `a` is the assignment to `l1` between an `if` line and its `} else {`; the fill follows the else.
    let load = at(&|l| l.starts_with("l1 = "), "the assignment to l1");
    let guard = lines[..load]
        .iter()
        .rposition(|l| l.starts_with("if "))
        .expect("an if before the load");
    let els = at(&|l| l.contains("} else {"), "the else");
    assert!(guard < load && load < els, "{src}");
    assert!(
        lines[els + 1].starts_with("l1 = ") && lines[els + 1].contains('7'),
        "{src}"
    );
    // The masked store (value `l1 + 5`) is inside its own `if`, with no else, after the load's.
    let store = at(&|l| l.contains("+ 5"), "the masked store value");
    let sguard = lines[..store]
        .iter()
        .rposition(|l| l.starts_with("if "))
        .expect("an if before the store");
    assert!(
        sguard > els,
        "the store has its own guard after the load's: {src}"
    );
    assert_eq!(
        masked_kernel::THREADS - masked_kernel::VALID,
        masked_kernel::SHIFT
    );
}

#[test]
fn a_subgroup_range_no_device_can_have_is_a_typed_decline() {
    for (min, max) in [(256, 512), (3, 3), (5, 7)] {
        declined(
            &subgroup_kernel::subgroup_kernel(min, max),
            "holds no power of two up to 128",
        );
    }
    assert!(
        Glsl.lower_kernel(&subgroup_kernel::subgroup_kernel(4, 7))
            .is_ok()
    );
}

/// A walker decline with no earlier test (an audit mutant that removed it passed every suite): a parameter-dependent grid.
/// The kernel passes the kernel checker, so the refusal is the walker's. (The walker's other `place_ok` decline, a place with
/// more than one coordinate, is unreachable by a checker-valid kernel on this unpopped: the checker refuses it with `BadPlace`
/// and a rank-2 buffer is refused earlier by the walker, so no test can reach it; the reachable form, `Place::Raw`, is pinned
/// where unpopped 0.57 is taken.)
#[test]
fn a_parameter_dependent_grid_is_declined_by_the_walker() {
    use unpopped::exec::{GridDim, LaunchShape};
    use unpopped::ir::ScalarExpr as E;
    use unpopped::kernel::{
        Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt, check_kernel,
    };
    use unpopped_vocab::ElementKind::U32;
    let c = |v| E::ConstInt(v, U32);
    // A grid whose x extent is the scalar parameter `n`.
    let mut grid = KernelFn::new(
        "grid_param",
        U32,
        vec![
            Param::buffer("out", U32, BufferRole::ReadWrite, Some(1)),
            Param::scalar("n", U32, 0),
        ],
        vec![LocalDecl::scalar(U32)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![c(0)],
            },
            value: c(1),
        }]),
    );
    let mut launch = LaunchShape::new([1, 1, 1], [1, 1, 1]);
    launch.grid_dims = Some([GridDim::Param(0), GridDim::Static(1), GridDim::Static(1)]);
    grid.launch = Some(launch);
    assert!(check_kernel(&grid).is_ok(), "{:?}", check_kernel(&grid));
    declined(&grid, "parameter-dependent grid");
}

/// A fence at a scope GLSL has no barrier for (cluster) is declined by name, never approximated by a block fence; found by
/// an audit mutant that spelled every scope and passed every suite. The kernel passes the kernel checker.
#[test]
fn a_cluster_scope_fence_is_declined_by_the_dialect() {
    use unpopped::exec::Scope;
    use unpopped::kernel::{MemoryOrder, Stmt, check_kernel};
    let mut k = reverse_kernel::reverse(WORKGROUP_SIZE, true, reverse_kernel::SharedLen::Static);
    k.launch = k.launch.map(|mut l| {
        l.grid = [2, 1, 1];
        l.cluster = Some([2, 1, 1]);
        l
    });
    k.body.0.insert(
        3,
        Stmt::Fence {
            scope: Scope::Cluster,
            order: MemoryOrder::SeqCst,
        },
    );
    assert!(check_kernel(&k).is_ok(), "{:?}", check_kernel(&k));
    declined(&k, "Fence");
}

/// Two walker declines no earlier test noticed losing (audit mutants that accepted a non-wrapping cast and an integer unary
/// op passed every suite). Each kernel passes the kernel checker, so the refusal is the walker's.
#[test]
fn a_non_wrapping_cast_and_an_integer_unary_op_are_declined_by_the_walker() {
    use unpopped::exec::LaunchShape;
    use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr as E, UnaryOp};
    use unpopped::kernel::{Block, BufferRole, KernelFn, Param, Place, Stmt, check_kernel};
    use unpopped_vocab::ElementKind::U32;
    let one_store = |value: E| {
        let mut k = KernelFn::new(
            "walker_decline",
            U32,
            vec![Param::buffer("out", U32, BufferRole::ReadWrite, Some(1))],
            vec![],
            Block(vec![Stmt::Store {
                place: Place::Logical {
                    buffer: 0,
                    coords: vec![E::ConstInt(0, U32)],
                },
                value,
            }]),
        );
        k.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
        assert!(check_kernel(&k).is_ok(), "{:?}", check_kernel(&k));
        k
    };
    let either = |k: &KernelFn, needle: &str| match Glsl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail } | LowerError::UnsupportedOp { detail }) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
            assert!(!detail.contains("check_kernel refused"), "{detail}");
        }
        other => panic!("wanted a decline naming `{needle}`, got {other:?}"),
    };
    // A clamping cast is refused by unpopped's own `cast` seam rule before the walker's wrapping-only guard (which a
    // mutant removing it showed is masked: an equivalent mutant), and the refusal is still not the checker's.
    either(
        &one_store(E::Cast(
            Box::new(E::ConstInt(1, U32)),
            CastSpec {
                to: U32,
                round: RoundMode::NearestEven,
                sat: Saturation::Clamp,
            },
        )),
        "a Cast U32 -> U32 reached an emitter with no `cast` seam",
    );
    // An integer kernel spells no unary op.
    either(
        &one_store(E::Unary(UnaryOp::Abs, Box::new(E::ConstInt(1, U32)))),
        "unary op Abs is not lowered",
    );
}
