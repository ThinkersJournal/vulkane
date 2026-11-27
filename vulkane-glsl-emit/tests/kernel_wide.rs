// SPDX-License-Identifier: MIT OR Apache-2.0
//! `I64`/`U64` kernels through GLSL (`GL_ARB_gpu_shader_int64`), the row 8d wider-dtype arm: the
//! 64-bit kernel of `vulkane-spirv-emit`'s tests, cut to the columns that need no integer division
//! (the shared walker does not lower `Div`), lowered by `Glsl::lower_kernel`, compiled by
//! `glslangValidator`, and run on a device that has `shaderInt64`; every result crosses the 32-bit
//! boundary and is compared at 64 bits. A device without the feature prints a `SKIP (unsupported)`
//! line naming it. Atomics at 64 bits are a typed decline.

use vulkane_test_support::common;
use vulkane_test_support::glslang;
use vulkane_test_support::gpu;
use vulkane_test_support::reverse_kernel;
use vulkane_test_support::wide_kernel;

use glslang::{glslang_words, validates};
use unpopped::backend::Backend;
use unpopped::kernel::KernelFn;
use unpopped_vocab::ElementKind::{self, I64, U64};
use vulkane::safe::DeviceFeatures;
use vulkane_glsl_emit::Glsl;
use wide_kernel::{K, THREADS};

/// Columns 0..=4 of the shared kernel: add carry, mul, shift past 32, masked shift, `>>` by dtype.
const COLUMNS: usize = 5;

fn kernel(ty: ElementKind) -> KernelFn {
    let mut k = wide_kernel::wide_kernel(ty);
    // Three `Let`s, then one `Store` per column: keep the first five stores.
    k.body.0.truncate(3 + COLUMNS);
    k
}

fn run(tag: &str, ty: ElementKind) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let features = DeviceFeatures::new().with_shader_int64();
    let gpu = gpu::Gpu::open_with(tag, Some((&features, "shaderInt64")))?;
    let g = Glsl.lower_kernel(&kernel(ty)).expect("GLSL lowers it");
    assert!(g.source.contains("GL_ARB_gpu_shader_int64"), "{}", g.source);
    let words = glslang_words(&g)?;
    if let Some((ok, err)) = validates(&words) {
        assert!(ok, "spirv-val rejected the module:\n{err}\n{}", g.source);
    }
    Some(gpu.run_grid("main", &words, THREADS as usize * K * 2, [1, 1, 1]))
}

fn check(got: &[u32], signed: bool) {
    let want = wide_kernel::expected_words(signed);
    for t in 0..THREADS as usize {
        for k in 0..COLUMNS {
            let cell = (t * K + k) * 2;
            assert_eq!(
                got[cell..cell + 2],
                want[cell..cell + 2],
                "thread {t} column {k} (low, high words), signed={signed}"
            );
        }
    }
}

#[test]
fn i64_values_across_the_32_bit_boundary_match_on_a_device_through_glsl() {
    if let Some(got) = run("glsl_kernel_i64", I64) {
        check(&got, true);
    }
}

#[test]
fn u64_values_across_the_32_bit_boundary_match_on_a_device_through_glsl() {
    if let Some(got) = run("glsl_kernel_u64", U64) {
        check(&got, false);
    }
}

/// Mutant: a lowering through 32 bits differs from the 64-bit result in every column checked, so the
/// device check above would fail it.
#[test]
fn a_32_bit_lowering_would_fail_every_checked_column() {
    for signed in [true, false] {
        for k in 0..COLUMNS {
            assert!(
                (0..u64::from(THREADS)).any(|t| wide_kernel::expected_at(64, signed, t, k)
                    != wide_kernel::expected_at(32, signed, t, k)),
                "column {k} (signed={signed}) would not catch a 32-bit lowering"
            );
        }
    }
}

#[test]
fn the_wide_source_spells_64_bit_types_masks_and_keeps_32_bit_kernels_unchanged() {
    let g = Glsl.lower_kernel(&kernel(I64)).expect("lowers");
    for needle in ["int64_t", "& 63l", "int64_t(gl_LocalInvocationID.x)"] {
        assert!(
            g.source.contains(needle),
            "wanted `{needle}` in\n{}",
            g.source
        );
    }
    // Control: a 32-bit kernel declares no 64-bit extension and masks at 31.
    let narrow = Glsl
        .lower_kernel(&reverse_kernel::reverse(
            64,
            true,
            reverse_kernel::SharedLen::Static,
        ))
        .expect("a 32-bit kernel lowers");
    assert!(!narrow.source.contains("int64"), "{}", narrow.source);
}

#[test]
fn a_dtype_with_no_spelling_is_a_typed_decline_naming_the_lowered_ones() {
    let mut k = unpopped::kernel::KernelFn::new(
        "f",
        ElementKind::Bf16,
        vec![unpopped::kernel::Param::buffer(
            "o",
            ElementKind::Bf16,
            unpopped::kernel::BufferRole::Write,
            Some(1),
        )],
        vec![],
        unpopped::kernel::Block(vec![]),
    );
    k.launch = Some(unpopped::exec::LaunchShape::new([1, 1, 1], [1, 1, 1]));
    let err = Glsl.lower_kernel(&k).expect_err("Bf16 is not lowered");
    assert!(
        format!("{err:?}").contains("integer kernels only"),
        "{err:?}"
    );
}

/// An `I64` kernel of one `ReadWrite` buffer cell whose body is `body`, for the decline tests.
fn one_cell_kernel(body: Vec<unpopped::kernel::Stmt>) -> KernelFn {
    use unpopped::exec::LaunchShape;
    use unpopped::kernel::{Block, BufferRole, Param};
    let mut k = KernelFn::new(
        "wide_decline",
        I64,
        vec![Param::buffer("out", I64, BufferRole::ReadWrite, Some(1))],
        vec![],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([1, 1, 1], [1, 1, 1]));
    k
}

fn declined_naming(k: &KernelFn, needle: &str) {
    match Glsl.lower_kernel(k) {
        Err(
            unpopped::backend::LowerError::UnsupportedKernel { detail }
            | unpopped::backend::LowerError::UnsupportedOp { detail },
        ) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
            // The refusal must come from the emitter or the walker, not the kernel checker (whose text can already
            // name the construct, so a test of the checker would pass without reaching the emitter's arm).
            assert!(!detail.contains("check_kernel refused"), "{detail}");
        }
        other => panic!("wanted an UnsupportedKernel naming `{needle}`, got {other:?}"),
    }
}

/// The CHANGELOG promises that a wide GLSL kernel declines integer division, atomics and a specialization-constant extent
/// by name; the shared wide kernel is truncated above (its `Div` columns cut away), so these pin the declines so a later
/// rung cannot start lowering one silently (review of #170, Low).
#[test]
fn a_wide_kernel_declines_division_atomics_and_a_specialization_constant_extent() {
    use unpopped::exec::{AllocDecl, Extent, MemorySpace, Scope};
    use unpopped::ir::ScalarExpr as E;
    use unpopped::kernel::{AtomicOp, LocalDecl, MemoryOrder, Param, Place, Stmt};
    // Division: the untruncated shared kernel holds `Div` columns.
    declined_naming(&wide_kernel::wide_kernel(I64), "Div");
    // An atomic: GLSL has no 64-bit atomic spelling here.
    let cell = Place::Logical {
        buffer: 0,
        coords: vec![E::ConstInt(0, I64)],
    };
    let atomic = one_cell_kernel(vec![Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: cell.clone(),
        value: E::ConstInt(1, I64),
        old: None,
        order: MemoryOrder::Relaxed,
        scope: Scope::Device,
    }]);
    declined_naming(&atomic, "atomic");
    // A specialization-constant extent: the scalar parameter `n` of the compute dtype sizes a shared array.
    let mut spec = one_cell_kernel(vec![Stmt::Store {
        place: cell,
        value: E::ConstInt(1, I64),
    }]);
    spec.params.push(Param::scalar("n", I64, 0));
    spec.locals.push(LocalDecl::scalar(I64));
    spec.allocs = vec![AllocDecl::new(
        "tile",
        MemorySpace::Shared,
        I64,
        Extent::Param(0),
    )];
    declined_naming(&spec, "specialization-constant extent");
}
