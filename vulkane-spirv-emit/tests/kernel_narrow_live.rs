// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP 8e on a live device: 8- and 16-bit integer buffers (`I8`/`U8`/`I16`/`U16`) read, widened,
//! narrowed and written by a kernel that computes at 32 bits (`tests/support/narrow_kernel.rs`).
//! The inputs include each type's extremes, so a sign/zero-extension mix-up or a missing wrap shows.
//! The device is created with `shaderInt8`/`storageBuffer8BitAccess` or `shaderInt16`/
//! `storageBuffer16BitAccess` as the type needs; without them a `SKIP (unsupported)` line names them.

use vulkane_test_support::common;
use vulkane_test_support::gpu;
use vulkane_test_support::narrow_kernel;

use narrow_kernel::{THREADS, input_words, width};
use unpopped::backend::Backend;
use unpopped::kernel::KernelFn;
use unpopped_vocab::ElementKind::{self, I8, I16, U8, U16};
use vulkane::safe::DeviceFeatures;
use vulkane_spirv_emit::{SpirV, required_features};

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

/// Lower `k`, check the module names the features the test enabled, run it on `elem`'s input.
fn run(tag: &str, elem: ElementKind, k: &KernelFn) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let (f, label) = features(elem);
    let gpu = gpu::Gpu::open_with(tag, Some((&f, label)))?;
    let g = SpirV.lower_kernel(k).expect("lowers");
    let words = g.words().expect("words").to_vec();
    let need = required_features(&words).expect("a module");
    assert_eq!(need.shader_int8, width(elem) == 1, "{need:?}");
    assert_eq!(need.storage_buffer_8bit, width(elem) == 1, "{need:?}");
    assert_eq!(need.shader_int16, width(elem) == 2, "{need:?}");
    assert_eq!(need.storage_buffer_16bit, width(elem) == 2, "{need:?}");
    Some(gpu.run(&g.name, &words, &[&input_words(elem, 256)]))
}

fn widen_on_device(elem: ElementKind, tag: &str) {
    let k = narrow_kernel::widen_kernel(elem);
    let Some(got) = run(tag, elem, &k) else {
        return; // skip already declared
    };
    for t in 0..THREADS {
        assert_eq!(
            got[t as usize],
            narrow_kernel::widened(elem, t),
            "{elem:?} t={t}"
        );
    }
    // Mutant: the other extension must disagree somewhere, so this check catches a mix-up.
    assert!(
        (0..THREADS)
            .any(|t| narrow_kernel::widened(elem, t) != narrow_kernel::widened_wrong(elem, t)),
        "{elem:?}: the wrong extension is indistinguishable"
    );
}

fn wrap_on_device(elem: ElementKind, tag: &str) {
    let k = narrow_kernel::wrap_kernel(elem);
    let Some(got) = run(tag, elem, &k) else {
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
    let maxed = (0..THREADS).any(|t| {
        narrow_kernel::wrapped(elem, t)
            != ((narrow_kernel::input(elem, t) + 1) as u32) & ((1u64 << (w * 8)) - 1) as u32
            || narrow_kernel::wrap(elem, narrow_kernel::input(elem, t) + 1)
                != narrow_kernel::input(elem, t) + 1
    });
    assert!(maxed, "{elem:?}: the wrap is not exercised");
}

#[test]
fn i8_widens_with_sign_and_wraps_at_127_on_a_device() {
    widen_on_device(I8, "spirv_narrow_i8_widen");
    wrap_on_device(I8, "spirv_narrow_i8_wrap");
}

#[test]
fn u8_widens_with_zeros_and_wraps_at_255_on_a_device() {
    widen_on_device(U8, "spirv_narrow_u8_widen");
    wrap_on_device(U8, "spirv_narrow_u8_wrap");
}

#[test]
fn i16_widens_with_sign_and_wraps_at_32767_on_a_device() {
    widen_on_device(I16, "spirv_narrow_i16_widen");
    wrap_on_device(I16, "spirv_narrow_i16_wrap");
}

#[test]
fn u16_widens_with_zeros_and_wraps_at_65535_on_a_device() {
    widen_on_device(U16, "spirv_narrow_u16_widen");
    wrap_on_device(U16, "spirv_narrow_u16_wrap");
}
