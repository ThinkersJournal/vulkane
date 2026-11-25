// SPDX-License-Identifier: MIT OR Apache-2.0
//! `F32` kernels through GLSL (ROADMAP row 8, the float arm), the cases of `vulkane-spirv-emit`'s
//! `float_kernel` support: lowered by `Glsl::lower_kernel`, compiled by `glslangValidator`, validated
//! by `spirv-val`, and run on a device. NaN, infinities, signed zeros and a denormal result must come
//! out bit-exact as native `f32` arithmetic gives them, and a multiply feeding an add must not fuse.
//! The source declares the float controls through `GL_EXT_spirv_intrinsics`; a device without them at
//! 32 bits prints a `SKIP (unsupported)` line naming the capability. Everything the table of
//! `vulkane-spirv-emit` rates looser than its contract tier is a typed decline.

use vulkane_test_support::common;
use vulkane_test_support::float_kernel;
use vulkane_test_support::glslang;
use vulkane_test_support::gpu;

use float_kernel::{Case, data, float_kernel, same};
use glslang::{glslang_words, validates};
use unpopped::backend::{Backend, GeneratedKernel};
use unpopped_vocab::ElementKind::{self, Bf16, F32, F64};
use vulkane::safe::DeviceFeatures;
use vulkane_glsl_emit::Glsl;

/// The `GeneratedKernel` and its SPIR-V words (through glslang and `spirv-val`), or `None` when a tool
/// is absent and not required.
fn compile(ty: ElementKind, case: Case) -> Option<(GeneratedKernel, Vec<u32>)> {
    let g = Glsl
        .lower_kernel(&float_kernel(ty, case))
        .expect("GLSL lowers the float kernel");
    let words = glslang_words(&g)?;
    if let Some((ok, err)) = validates(&words) {
        assert!(ok, "spirv-val rejected the module:\n{err}\n{}", g.source);
    }
    Some((g, words))
}

fn run(tag: &str, ty: ElementKind, case: Case) -> Option<(Vec<u32>, Vec<u32>)> {
    common::require_serialization_lock();
    let wide = ty == F64;
    let features = if wide {
        DeviceFeatures::new().with_shader_float64()
    } else {
        DeviceFeatures::new()
    };
    let gpu = gpu::Gpu::open_with(tag, Some((&features, "shaderFloat64")))?;
    let (_, words) = compile(ty, case)?;
    let ok = gpu.float_controls().is_some_and(|c| {
        if wide {
            c.denorm_preserve_f64 && c.signed_zero_inf_nan_preserve_f64
        } else {
            c.denorm_preserve_f32 && c.signed_zero_inf_nan_preserve_f32
        }
    });
    if !ok {
        common::skip(common::Missing::capability(
            "DenormPreserve + SignedZeroInfNanPreserve at this width (VkPhysicalDeviceFloatControlsProperties)",
        ));
        return None;
    }
    let (a, b, want) = if wide {
        data::<f64>(case)
    } else {
        data::<f32>(case)
    };
    Some((gpu.run("main", &words, &[&a, &b]), want))
}

fn check(tag: &str, ty: ElementKind, case: Case) {
    if let Some((got, want)) = run(tag, ty, case) {
        same(ty == F64, &got, &want).unwrap_or_else(|e| panic!("{ty:?} {case:?}: {e}"));
    }
}

#[test]
fn f32_mul_add_is_unfused_and_ieee_on_a_device_through_glsl() {
    check("glsl_f32_muladd", F32, Case::MulAdd);
}

#[test]
fn f64_mul_add_is_unfused_and_ieee_on_a_device_through_glsl() {
    check("glsl_f64_muladd", F64, Case::MulAdd);
}

/// Mutant: running the `F64` values through `f32` arithmetic is wrong where only `f64` holds the
/// value (`2^24 + 1`, `3.4e38 * 10`, the denormal), so the device check above would fail it; the
/// comparison accepts the oracle against itself (control).
#[test]
fn an_f32_lowering_of_an_f64_kernel_would_fail_the_check() {
    let (_, _, want64) = data::<f64>(Case::MulAdd);
    let (_, _, want32) = data::<f32>(Case::MulAdd);
    let widened: Vec<u32> = want32
        .iter()
        .flat_map(|w| f64::from(f32::from_bits(*w)).to_bits().to_le_bytes())
        .collect::<Vec<u8>>()
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(same(true, &widened, &want64).is_err());
    assert!(same(true, &want64, &want64).is_ok());
}

#[test]
fn nan_propagating_max_min_and_floor_match_ieee_on_a_device_through_glsl() {
    for (ty, tag) in [(F32, "glsl_f32_ops"), (F64, "glsl_f64_ops")] {
        for case in [Case::Max, Case::Min, Case::Floor] {
            check(tag, ty, case);
        }
    }
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

/// `OpDecorate` of `NoContraction` (decoration 42).
fn no_contraction(words: &[u32]) -> usize {
    instructions(words, 71)
        .iter()
        .filter(|o| o[1] == 42)
        .count()
}

/// The multiply and the add of `a * b + a` carry `NoContraction`, and the module declares the float
/// controls. Mutant: the same source without `precise` carries none, so the count discriminates.
#[test]
fn the_multiply_and_add_are_no_contraction_and_the_float_controls_are_declared() {
    for (ty, width) in [(F32, 32), (F64, 64)] {
        no_contraction_at(ty, width);
    }
}

fn no_contraction_at(ty: ElementKind, width: u32) {
    let Some((g, words)) = compile(ty, Case::MulAdd) else {
        return;
    };
    assert_eq!(no_contraction(&words), 2, "{}", g.source);
    // `Float64` (capability 10) is declared for `F64` only.
    let caps: Vec<u32> = instructions(&words, 17).iter().map(|o| o[0]).collect();
    assert_eq!(caps.contains(&10), width == 64, "{caps:?}");
    let modes: Vec<(u32, u32)> = instructions(&words, 16)
        .iter()
        .filter(|o| o.len() == 3)
        .map(|o| (o[1], o[2]))
        .collect();
    for want in [(4459, width), (4461, width)] {
        assert!(modes.contains(&want), "{modes:?}\n{}", g.source);
    }
    let loose = GeneratedKernel::new("float".to_string(), g.source.replace("precise ", ""));
    let loose = glslang_words(&loose).expect("the mutant compiles");
    assert_eq!(no_contraction(&loose), 0, "the mutant is not `precise`");
}

#[test]
fn the_float_source_spells_float_types_and_exact_literals() {
    let g = Glsl
        .lower_kernel(&float_kernel(F32, Case::MulAdd))
        .expect("lowers");
    for needle in [
        "precise float l0",
        "float data[]",
        "precise const float p0",
        "gl_LocalInvocationID.x",
    ] {
        assert!(
            g.source.contains(needle),
            "wanted `{needle}` in\n{}",
            g.source
        );
    }
    // Control: an integer kernel declares no float control and no `precise`.
    let int = Glsl
        .lower_kernel(&vulkane_glsl_lowering_int())
        .expect("an integer kernel lowers");
    assert!(!int.source.contains("precise") && !int.source.contains("spirv_intrinsics"));
}

fn vulkane_glsl_lowering_int() -> unpopped::kernel::KernelFn {
    let mut k = float_kernel(unpopped_vocab::ElementKind::U32, Case::Floor);
    k.body.0.truncate(1);
    k
}

#[test]
fn what_the_vulkan_bounds_do_not_cover_is_a_typed_decline_naming_it() {
    for ty in [F32, F64] {
        for (case, name) in [(Case::Div, "Div"), (Case::Sqrt, "Sqrt"), (Case::Exp, "Exp")] {
            let err = Glsl
                .lower_kernel(&float_kernel(ty, case))
                .expect_err("declined");
            let text = format!("{err:?}");
            assert!(
                text.contains(name) && text.contains("not lowered"),
                "{text}"
            );
        }
    }
    let err = Glsl
        .lower_kernel(&float_kernel(Bf16, Case::MulAdd))
        .expect_err("Bf16 is not lowered");
    assert!(format!("{err:?}").contains("Bf16"), "{err:?}");
    // Control: the same kernel at F64 lowers and spells `double`.
    let g = Glsl
        .lower_kernel(&float_kernel(F64, Case::MulAdd))
        .expect("F64 lowers");
    assert!(g.source.contains("double data[]") && g.source.contains("GL_ARB_gpu_shader_fp64"));
}
