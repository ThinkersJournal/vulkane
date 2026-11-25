// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F16` kernels through GLSL (`GL_EXT_shader_explicit_arithmetic_types_float16` and
//! `GL_EXT_shader_16bit_storage`), the half cases of `vulkane-spirv-emit`'s `half_kernel` support:
//! half arithmetic rounds to half after each operation (a multiply feeding an add must not fuse),
//! subnormals survive (`DenormPreserve` at 16 bits, declared through `GL_EXT_spirv_intrinsics`), NaN,
//! infinities and signed zeros are IEEE. Lowered by `Glsl::lower_kernel`, compiled by `glslang`,
//! validated by `spirv-val`, run on a device. Without `shaderFloat16` + `storageBuffer16BitAccess` or the
//! 16-bit float controls a `SKIP (unsupported)` line names the capability.

use vulkane_test_support::common;
use vulkane_test_support::float_kernel;
use vulkane_test_support::glslang;
use vulkane_test_support::gpu;
use vulkane_test_support::half_kernel;

use float_kernel::{Case, float_kernel};
use glslang::{glslang_words, validates};
use half_kernel::{data, same};
use unpopped::backend::{Backend, GeneratedKernel};
use unpopped_vocab::ElementKind::{F16, F32};
use vulkane::safe::DeviceFeatures;
use vulkane_glsl_emit::Glsl;

fn compile(case: Case) -> Option<(GeneratedKernel, Vec<u32>)> {
    let g = Glsl
        .lower_kernel(&float_kernel(F16, case))
        .expect("GLSL lowers the F16 kernel");
    let words = glslang_words(&g)?;
    if let Some((ok, err)) = validates(&words) {
        assert!(ok, "spirv-val rejected the module:\n{err}\n{}", g.source);
    }
    Some((g, words))
}

fn run(tag: &str, case: Case) -> Option<(Vec<u32>, Vec<u32>)> {
    common::require_serialization_lock();
    let features = DeviceFeatures::new()
        .with_shader_float16()
        .with_storage_buffer16_bit_access();
    let gpu = gpu::Gpu::open_with(
        tag,
        Some((&features, "shaderFloat16 + storageBuffer16BitAccess")),
    )?;
    let (_, words) = compile(case)?;
    if !gpu
        .float_controls()
        .is_some_and(|c| c.denorm_preserve_f16 && c.signed_zero_inf_nan_preserve_f16)
    {
        common::skip(common::Missing::capability(
            "DenormPreserve + SignedZeroInfNanPreserve at 16 bits (VkPhysicalDeviceFloatControlsProperties)",
        ));
        return None;
    }
    let (a, b, want, _) = data(case);
    Some((gpu.run("main", &words, &[&a, &b]), want))
}

fn check(tag: &str, case: Case) {
    let (_, _, want, mutant) = data(case);
    assert!(
        same(&mutant, &want).is_err(),
        "{case:?}: the mutant is indistinguishable, so the case proves nothing"
    );
    let Some((got, want)) = run(tag, case) else {
        return; // skip already declared
    };
    same(&got, &want).unwrap_or_else(|e| panic!("F16 {case:?}: {e}"));
}

#[test]
fn f16_mul_add_rounds_each_step_and_is_unfused_on_a_device_through_glsl() {
    check("glsl_f16_muladd", Case::MulAdd);
}

#[test]
fn f16_max_min_floor_match_ieee_on_a_device_through_glsl() {
    check("glsl_f16_max", Case::Max);
    check("glsl_f16_min", Case::Min);
    check("glsl_f16_floor", Case::Floor);
}

/// The instructions of `words` with `opcode`, as their operand words.
fn instructions(words: &[u32], opcode: u32) -> Vec<&[u32]> {
    let (mut at, mut out) = (5, Vec::new());
    while at < words.len() {
        let (count, op) = ((words[at] >> 16) as usize, words[at] & 0xffff);
        if op == opcode {
            out.push(&words[at + 1..at + count]);
        }
        at += count.max(1);
    }
    out
}

/// The module declares the half capabilities and the float controls at 16 bits, keeps the multiply
/// and add `NoContraction`, and an `F32` kernel declares none of the half ones (control).
#[test]
fn the_module_declares_half_storage_and_float_controls_at_16_bits() {
    let Some((g, words)) = compile(Case::MulAdd) else {
        return;
    };
    let caps: Vec<u32> = instructions(&words, 17).iter().map(|o| o[0]).collect();
    // Float16 (9), StorageBuffer16BitAccess (4433), DenormPreserve (4464), SignedZeroInfNanPreserve (4466).
    for cap in [9, 4433, 4464, 4466] {
        assert!(
            caps.contains(&cap),
            "capability {cap} in {caps:?}\n{}",
            g.source
        );
    }
    let modes: Vec<(u32, u32)> = instructions(&words, 16)
        .iter()
        .filter(|o| o.len() == 3)
        .map(|o| (o[1], o[2]))
        .collect();
    for want in [(4459, 16), (4461, 16)] {
        assert!(modes.contains(&want), "{modes:?}");
    }
    let no_contraction = instructions(&words, 71)
        .iter()
        .filter(|o| o[1] == 42)
        .count();
    assert_eq!(no_contraction, 2, "{}", g.source);
    // Control: the same kernel at F32 declares no half capability and 32-bit modes.
    let f32g = Glsl
        .lower_kernel(&float_kernel(F32, Case::MulAdd))
        .expect("lowers");
    let f32w = glslang_words(&f32g).expect("compiles");
    let f32caps: Vec<u32> = instructions(&f32w, 17).iter().map(|o| o[0]).collect();
    assert!(
        !f32caps.contains(&9) && !f32caps.contains(&4433),
        "{f32caps:?}"
    );
}

/// Declined by name: what the Vulkan bound does not cover (as at `F32`), a literal that is not exactly a
/// half, an integer or `F32` element in a half kernel; an exact literal lowers and is spelled `hf`.
#[test]
fn what_a_half_kernel_cannot_say_exactly_is_a_typed_decline() {
    use unpopped::ir::ScalarExpr as E;
    use unpopped::kernel::Stmt;
    for (case, name) in [(Case::Div, "Div"), (Case::Sqrt, "Sqrt"), (Case::Exp, "Exp")] {
        let err = Glsl
            .lower_kernel(&float_kernel(F16, case))
            .expect_err("declined");
        let text = format!("{err:?}");
        assert!(
            text.contains(name) && text.contains("not lowered"),
            "{text}"
        );
    }
    let with_literal = |v: f64| {
        let mut k = float_kernel(F16, Case::MulAdd);
        if let Some(Stmt::Store { value, .. }) = k.body.0.last_mut() {
            *value = E::Add(Box::new(E::Local(0)), Box::new(E::Const(v)));
        }
        k
    };
    let err = Glsl
        .lower_kernel(&with_literal(0.1))
        .expect_err("0.1 is no half");
    assert!(
        format!("{err:?}").contains("not a finite value exact"),
        "{err:?}"
    );
    let ok = Glsl
        .lower_kernel(&with_literal(0.5))
        .expect("0.5 is a half");
    assert!(ok.source.contains("0.5hf"), "{}", ok.source);
    let ok = Glsl
        .lower_kernel(&with_literal(6.103_515_625e-5))
        .expect("2^-14 is a half");
    assert!(ok.source.contains("hf"), "{}", ok.source);
    // An F32 buffer in a half kernel: two float widths, not lowered.
    let mut k = float_kernel(F16, Case::MulAdd);
    k.params[0] =
        unpopped::kernel::Param::buffer("a", F32, unpopped::kernel::BufferRole::Read, Some(1));
    assert!(Glsl.lower_kernel(&k).is_err());
}
