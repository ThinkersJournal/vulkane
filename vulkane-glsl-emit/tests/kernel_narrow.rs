// SPDX-License-Identifier: MIT OR Apache-2.0
//! 8- and 16-bit integer buffers (`I8`/`U8`/`I16`/`U16`) through GLSL (`GL_EXT_shader_8bit_storage`,
//! `GL_EXT_shader_16bit_storage` and the explicit arithmetic types), the kernels of `vulkane-spirv-emit`'s
//! `narrow_kernel` support: a 32-bit compute kernel reads, widens (sign- or zero-extending), narrows
//! (wrapping at the type's maximum) and writes them through wrapping `Cast`s. The inputs include each
//! type's extremes, so a sign/zero-extension mix-up or a missing wrap shows. Lowered by `Glsl`, compiled by
//! `glslang`, validated by `spirv-val`, run on a device created with `shaderInt8` + `storageBuffer8BitAccess`
//! or `shaderInt16` + `storageBuffer16BitAccess`; without them a `SKIP (unsupported)` line names them.

use vulkane_test_support::common;
use vulkane_test_support::glslang;
use vulkane_test_support::gpu;
use vulkane_test_support::narrow_kernel;
use vulkane_test_support::reverse_kernel;

use glslang::{glslang_words, validates};
use narrow_kernel::{THREADS, input_words, width};
use unpopped::backend::Backend;
use unpopped::kernel::KernelFn;
use unpopped_vocab::ElementKind::{self, I8, I16, U8, U16};
use vulkane::safe::DeviceFeatures;
use vulkane_glsl_emit::Glsl;
use vulkane_spirv_emit::{WORKGROUP_SIZE, required_features};

fn features(elem: ElementKind) -> (DeviceFeatures, &'static str) {
    if width(elem) == 1 {
        (
            DeviceFeatures::new()
                .with_shader_int8()
                .with_storage_buffer8_bit_access(),
            "shaderInt8 + storageBuffer8BitAccess",
        )
    } else {
        (
            DeviceFeatures::new()
                .with_shader_int16()
                .with_storage_buffer16_bit_access(),
            "shaderInt16 + storageBuffer16BitAccess",
        )
    }
}

fn compile(k: &KernelFn) -> Option<Vec<u32>> {
    let g = Glsl.lower_kernel(k).expect("GLSL lowers the narrow kernel");
    let words = glslang_words(&g)?;
    if let Some((ok, err)) = validates(&words) {
        assert!(ok, "spirv-val rejected the module:\n{err}\n{}", g.source);
    }
    Some(words)
}

fn run(tag: &str, elem: ElementKind, k: &KernelFn) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let (f, label) = features(elem);
    let gpu = gpu::Gpu::open_with(tag, Some((&f, label)))?;
    let words = compile(k)?;
    let need = required_features(&words).expect("a module");
    assert_eq!(need.shader_int8, width(elem) == 1, "{need:?}");
    assert_eq!(need.storage_buffer_8bit, width(elem) == 1, "{need:?}");
    assert_eq!(need.shader_int16, width(elem) == 2, "{need:?}");
    assert_eq!(need.storage_buffer_16bit, width(elem) == 2, "{need:?}");
    Some(gpu.run("main", &words, &[&input_words(elem, 256)]))
}

fn widen_on_device(elem: ElementKind, tag: &str) {
    let Some(got) = run(tag, elem, &narrow_kernel::widen_kernel(elem)) else {
        return; // skip already declared
    };
    for t in 0..THREADS {
        assert_eq!(
            got[t as usize],
            narrow_kernel::widened(elem, t),
            "{elem:?} t={t}"
        );
    }
    // Mutant: the other extension disagrees somewhere, so this check catches a mix-up.
    assert!(
        (0..THREADS)
            .any(|t| narrow_kernel::widened(elem, t) != narrow_kernel::widened_wrong(elem, t)),
        "{elem:?}: the wrong extension is indistinguishable"
    );
}

fn wrap_on_device(elem: ElementKind, tag: &str) {
    let Some(got) = run(tag, elem, &narrow_kernel::wrap_kernel(elem)) else {
        return;
    };
    let w = width(elem);
    let bytes: Vec<u8> = got.iter().flat_map(|x| x.to_le_bytes()).collect();
    for t in 0..THREADS as usize {
        let mut raw = [0u8; 4];
        raw[..w].copy_from_slice(&bytes[t * w..t * w + w]);
        assert_eq!(
            u32::from_le_bytes(raw),
            narrow_kernel::wrapped(elem, t as u32),
            "{elem:?} t={t}"
        );
    }
    // Mutant: a lowering that keeps the 32-bit sum (no wrap) differs where the input is the maximum.
    assert!(
        (0..THREADS).any(
            |t| narrow_kernel::wrap(elem, narrow_kernel::input(elem, t) + 1)
                != narrow_kernel::input(elem, t) + 1
        ),
        "{elem:?}: the wrap is not exercised"
    );
}

#[test]
fn i8_widens_with_sign_and_wraps_at_127_on_a_device_through_glsl() {
    widen_on_device(I8, "glsl_narrow_i8_widen");
    wrap_on_device(I8, "glsl_narrow_i8_wrap");
}

#[test]
fn u8_widens_with_zeros_and_wraps_at_255_on_a_device_through_glsl() {
    widen_on_device(U8, "glsl_narrow_u8_widen");
    wrap_on_device(U8, "glsl_narrow_u8_wrap");
}

#[test]
fn i16_widens_with_sign_and_wraps_at_32767_on_a_device_through_glsl() {
    widen_on_device(I16, "glsl_narrow_i16_widen");
    wrap_on_device(I16, "glsl_narrow_i16_wrap");
}

#[test]
fn u16_widens_with_zeros_and_wraps_at_65535_on_a_device_through_glsl() {
    widen_on_device(U16, "glsl_narrow_u16_widen");
    wrap_on_device(U16, "glsl_narrow_u16_wrap");
}

/// The extensions are declared only for the widths used, and a 32-bit kernel declares none (control).
#[test]
fn the_narrow_extensions_are_declared_only_for_the_widths_used() {
    let g8 = Glsl
        .lower_kernel(&narrow_kernel::widen_kernel(I8))
        .expect("lowers");
    assert!(
        g8.source.contains("GL_EXT_shader_8bit_storage") && g8.source.contains("int8_t data[]")
    );
    assert!(!g8.source.contains("16bit"), "{}", g8.source);
    let g16 = Glsl
        .lower_kernel(&narrow_kernel::widen_kernel(U16))
        .expect("lowers");
    assert!(
        g16.source.contains("GL_EXT_shader_16bit_storage")
            && g16.source.contains("uint16_t data[]")
    );
    assert!(!g16.source.contains("8bit"), "{}", g16.source);
    let plain = Glsl
        .lower_kernel(&reverse_kernel::reverse(
            WORKGROUP_SIZE,
            true,
            reverse_kernel::SharedLen::Static,
        ))
        .expect("lowers");
    assert!(!plain.source.contains("_storage"), "{}", plain.source);
}

/// An atomic in a kernel that holds narrow buffers is a typed decline (the target would have to be
/// proved 32-bit); an 8-bit `Cast` to a type with no spelling too.
#[test]
fn an_atomic_beside_a_narrow_buffer_is_a_typed_decline() {
    let mut k = narrow_kernel::widen_kernel(I8);
    k.body.0.push(unpopped::kernel::Stmt::AtomicRmw {
        op: unpopped::kernel::AtomicOp::Add,
        place: unpopped::kernel::Place::Logical {
            buffer: 1,
            coords: vec![unpopped::ir::ScalarExpr::Local(0)],
        },
        value: unpopped::ir::ScalarExpr::Local(0),
        old: None,
        order: unpopped::kernel::MemoryOrder::Relaxed,
        scope: unpopped::exec::Scope::Device,
    });
    let err = Glsl.lower_kernel(&k).expect_err("declined");
    assert!(format!("{err:?}").contains("atomic"), "{err:?}");
}
