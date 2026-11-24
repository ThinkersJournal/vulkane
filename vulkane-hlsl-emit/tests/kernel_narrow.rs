// SPDX-License-Identifier: MIT OR Apache-2.0
//! 8- and 16-bit integer buffers through HLSL. `fishstringer-slang-emit` declines each element type by name
//! ("not lowered yet"), so `Hlsl::lower_kernel` lowers a kernel holding a 16-bit integer (`I16`/`U16`) with this
//! crate's own dialect of the shared walker: `dxc` compiles `int16_t`/`uint16_t` with `-enable-16bit-types`
//! (`kernel_dxc_args`), and the SPIR-V needs `shaderInt16` + `storageBuffer16BitAccess`. `dxc` has no 8-bit
//! integer type (`int8_t` is undeclared), so `I8`/`U8` stay a typed decline naming the element and the reason.
//! The kernels are `vulkane-spirv-emit`'s `narrow_kernel` support (widen, and add-one-and-wrap), lowered,
//! compiled by `dxc`, validated by `spirv-val` and run on a device. Two mutants run for real: the sign/zero
//! extension swapped in the emitted text, and the same text compiled without the flag (a 32-bit `int16_t`).
//! Without `dxc`, `spirv-val` or a device the tests follow `VULKANE_REQUIRE_DXC`, `VULKANE_REQUIRE_SPIRV_VAL`
//! and `VULKANE_REQUIRE_DEVICE`; a device without the features prints a `SKIP (unsupported)` line naming them.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "../../vulkane-spirv-emit/tests/support/gpu.rs"]
mod gpu;
#[path = "../../vulkane-spirv-emit/tests/support/narrow_kernel.rs"]
mod narrow_kernel;

use std::process::Command;

use narrow_kernel::{THREADS, input_words, widen_kernel, width, wrap_kernel};
use unpopped::backend::{Backend, GeneratedKernel};
use unpopped_vocab::ElementKind;
use unpopped_vocab::ElementKind::{I8, I16, U8, U16};
use vulkane::safe::DeviceFeatures;
use vulkane_hlsl_emit::{Hlsl, kernel_dxc_args};
use vulkane_spirv_emit::required_features;

#[test]
fn an_8_bit_integer_kernel_is_a_typed_decline_naming_the_element_and_the_reason() {
    for e in [I8, U8] {
        let err = Hlsl.lower_kernel(&widen_kernel(e)).expect_err("declined");
        let text = format!("{err:?}");
        assert!(
            text.contains("UnsupportedKernel")
                && text.contains(&format!("{e:?}"))
                && text.contains("no 8-bit integer type"),
            "{e:?}: {text}"
        );
    }
    // Control: the 16-bit twin of the same kernel lowers.
    for e in [I16, U16] {
        Hlsl.lower_kernel(&widen_kernel(e))
            .expect("a 16-bit kernel lowers");
    }
}

/// `Some(true)` when `dxc` compiles a kernel reading a `ty` buffer, `Some(false)` when it rejects it,
/// `None` when `dxc` is absent (and not required).
fn dxc_compiles(ty: &str, flags: &[&str]) -> Option<bool> {
    let dir = tempfile::tempdir().unwrap();
    let (src, spv) = (dir.path().join("k.hlsl"), dir.path().join("k.spv"));
    std::fs::write(
        &src,
        format!(
            "[[vk::binding(0, 0)]] StructuredBuffer<{ty}> a;\n[[vk::binding(1, 0)]] RWStructuredBuffer<int> o;\n[shader(\"compute\")]\n[numthreads(64, 1, 1)]\nvoid main(uint3 t : SV_GroupThreadID) {{ o[t.x] = (int)a[t.x]; }}\n"
        ),
    )
    .unwrap();
    let out = Command::new("dxc")
        .args([
            "-T",
            "cs_6_2",
            "-E",
            "main",
            "-spirv",
            "-fspv-target-env=vulkan1.1",
        ])
        .args(flags)
        .arg(&src)
        .arg("-Fo")
        .arg(&spv)
        .output();
    match out {
        Ok(o) => Some(o.status.success()),
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_DXC").is_none(),
                "VULKANE_REQUIRE_DXC is set but `dxc` could not be run: {e}"
            );
            None
        }
    }
}

#[test]
fn dxc_has_16_bit_integers_with_the_flag_and_no_8_bit_integers() {
    let Some(with) = dxc_compiles("int16_t", &["-enable-16bit-types"]) else {
        return;
    };
    assert!(
        with,
        "dxc no longer compiles int16_t with -enable-16bit-types"
    );
    assert_eq!(
        dxc_compiles("uint16_t", &["-enable-16bit-types"]),
        Some(true)
    );
    // The 8-bit names do not exist, flag or no flag.
    for ty in ["int8_t", "uint8_t"] {
        assert_eq!(
            dxc_compiles(ty, &["-enable-16bit-types"]),
            Some(false),
            "dxc now accepts {ty}: an HLSL 8-bit arm can be written"
        );
    }
}

// ---- the 16-bit arm: text, `dxc`, `spirv-val`, a device ----

/// `dxc` to SPIR-V words for `source` (entry `entry`) with `args`; `None` when `dxc` is absent (a failure
/// under `VULKANE_REQUIRE_DXC`). A compile failure fails the test with the source.
fn dxc_words(entry: &str, source: &str, args: &[&str]) -> Option<Vec<u32>> {
    let dir = tempfile::tempdir().unwrap();
    let (src, spv) = (dir.path().join("k.hlsl"), dir.path().join("k.spv"));
    std::fs::write(&src, source).unwrap();
    let out = Command::new("dxc")
        .args([
            "-T",
            "cs_6_2",
            "-E",
            entry,
            "-spirv",
            "-fspv-target-env=vulkan1.1",
        ])
        .args(args)
        .arg(&src)
        .arg("-Fo")
        .arg(&spv)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            assert!(
                std::env::var_os("VULKANE_REQUIRE_DXC").is_none(),
                "VULKANE_REQUIRE_DXC is set but `dxc` could not be run: {e}"
            );
            return None;
        }
    };
    assert!(
        out.status.success(),
        "dxc failed:\n{}\n--- source ---\n{source}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(
        std::fs::read(&spv)
            .unwrap()
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect(),
    )
}

/// `spirv-val --target-env vulkan1.1` over `words`; a rejection fails the test; absent: skipped unless required.
fn validate(words: &[u32], source: &str) {
    use std::io::Write;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.1", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            let _ = c.stdin.take().expect("piped stdin").write_all(&bytes);
            c.wait_with_output()
        });
    match child {
        Ok(o) => assert!(
            o.status.success(),
            "spirv-val rejected the module:\n{}\n{source}",
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(e) => assert!(
            std::env::var_os("VULKANE_REQUIRE_SPIRV_VAL").is_none(),
            "VULKANE_REQUIRE_SPIRV_VAL is set but `spirv-val` could not be run: {e}"
        ),
    }
}

fn features(elem: ElementKind) -> (DeviceFeatures, &'static str) {
    assert_eq!(width(elem), 2, "the HLSL arm is 16-bit only");
    (
        DeviceFeatures::new()
            .with_shader_int16()
            .with_storage_buffer16_bit_access(),
        "shaderInt16 + storageBuffer16BitAccess",
    )
}

fn lower(elem: ElementKind, k: &unpopped::kernel::KernelFn) -> GeneratedKernel {
    let g = Hlsl.lower_kernel(k).expect("HLSL lowers the 16-bit kernel");
    assert_eq!(kernel_dxc_args(k), ["-enable-16bit-types"], "{elem:?}");
    assert!(
        g.source.contains("// dxc: -enable-16bit-types\n"),
        "{}",
        g.source
    );
    g
}

/// Compile (checking the module needs exactly the 16-bit features), validate and run `source` on a device.
fn run(
    tag: &str,
    elem: ElementKind,
    g: &GeneratedKernel,
    source: &str,
    args: &[&str],
) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let (f, label) = features(elem);
    let gpu = gpu::Gpu::open_with(tag, Some((&f, label)))?;
    let words = dxc_words(&g.name, source, args)?;
    validate(&words, source);
    let need = required_features(&words).expect("a module");
    assert!(need.shader_int16 && need.storage_buffer_16bit, "{need:?}");
    assert!(!need.shader_int8 && !need.storage_buffer_8bit, "{need:?}");
    Some(gpu.run(&g.name, &words, &[&input_words(elem, 256)]))
}

fn widen_on_device(elem: ElementKind, tag: &str) {
    let g = lower(elem, &widen_kernel(elem));
    let Some(got) = run(tag, elem, &g, &g.source, &["-enable-16bit-types"]) else {
        return;
    };
    for t in 0..THREADS {
        assert_eq!(
            got[t as usize],
            narrow_kernel::widened(elem, t),
            "{elem:?} t={t}\n{}",
            g.source
        );
    }
    // Mutant, run for real: the other extension, applied to the emitted text, must be caught by the same check.
    let (from, to) = if elem == ElementKind::I16 {
        ("((int)(l1))", "((int)((uint16_t)(l1)))")
    } else {
        ("((uint)(l1))", "((uint)((int16_t)(l1)))")
    };
    assert_eq!(
        g.source.matches(from).count(),
        1,
        "the mutation anchor is absent:\n{}",
        g.source
    );
    let mutant = g.source.replace(from, to);
    let bad = run(
        &format!("{tag}_mutant"),
        elem,
        &g,
        &mutant,
        &["-enable-16bit-types"],
    )
    .expect("device opened above");
    for t in 0..THREADS {
        assert_eq!(
            bad[t as usize],
            narrow_kernel::widened_wrong(elem, t),
            "{elem:?} t={t}: the mutant is the other extension"
        );
    }
    assert!(
        (0..THREADS).any(|t| bad[t as usize] != narrow_kernel::widened(elem, t)),
        "{elem:?}: the wrong extension was not caught"
    );
}

fn wrap_on_device(elem: ElementKind, tag: &str) {
    let g = lower(elem, &wrap_kernel(elem));
    let Some(got) = run(tag, elem, &g, &g.source, &["-enable-16bit-types"]) else {
        return;
    };
    let bytes: Vec<u8> = got.iter().flat_map(|x| x.to_le_bytes()).collect();
    for t in 0..THREADS as usize {
        let raw = u16::from_le_bytes([bytes[t * 2], bytes[t * 2 + 1]]);
        assert_eq!(
            u32::from(raw),
            narrow_kernel::wrapped(elem, t as u32),
            "{elem:?} t={t}\n{}",
            g.source
        );
    }
    // The wrap is exercised: some input is the type's maximum, whose successor differs unwrapped.
    assert!(
        (0..THREADS).any(
            |t| narrow_kernel::wrap(elem, narrow_kernel::input(elem, t) + 1)
                != narrow_kernel::input(elem, t) + 1
        ),
        "{elem:?}: the wrap is not exercised"
    );
}

#[test]
fn i16_widens_with_sign_and_wraps_at_32767_on_a_device_through_hlsl() {
    widen_on_device(I16, "hlsl_narrow_i16_widen");
    wrap_on_device(I16, "hlsl_narrow_i16_wrap");
}

#[test]
fn u16_widens_with_zeros_and_wraps_at_65535_on_a_device_through_hlsl() {
    widen_on_device(U16, "hlsl_narrow_u16_widen");
    wrap_on_device(U16, "hlsl_narrow_u16_wrap");
}

/// The wrong-width mutant: the same text compiled WITHOUT `-enable-16bit-types`. `dxc` then has no `int16_t`
/// at all (undeclared identifier), so the flag is not optional: the emitted `// dxc:` line and
/// `kernel_dxc_args` are what make the text compile, and the real build needs the 16-bit capabilities
/// (the check `run` makes, shown here to be able to fail).
#[test]
fn without_the_flag_dxc_rejects_the_text_and_with_it_the_module_needs_16_bit_storage() {
    for elem in [I16, U16] {
        let g = lower(elem, &widen_kernel(elem));
        let Some(real) = dxc_words(&g.name, &g.source, kernel_dxc_args(&widen_kernel(elem))) else {
            return;
        };
        let r = required_features(&real).unwrap();
        assert!(r.shader_int16 && r.storage_buffer_16bit, "{elem:?} {r:?}");
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("k.hlsl");
        std::fs::write(&src, &g.source).unwrap();
        let bare = Command::new("dxc")
            .args([
                "-T",
                "cs_6_2",
                "-E",
                &g.name,
                "-spirv",
                "-fspv-target-env=vulkan1.1",
            ])
            .arg(&src)
            .output()
            .expect("dxc ran above");
        assert!(
            !bare.status.success(),
            "{elem:?}: dxc accepted int16_t without the flag"
        );
        assert!(
            String::from_utf8_lossy(&bare.stderr).contains("int16_t"),
            "{}",
            String::from_utf8_lossy(&bare.stderr)
        );
    }
}

/// Text pins: 16-bit element types in the declarations and locals, a wrapping cast, and none of the
/// Slang emitter's strided-view `ConstantBuffer` (this arm has no `shaderInt64` need).
#[test]
fn the_16_bit_text_declares_16_bit_buffers_and_no_int64_view() {
    let g = lower(I16, &wrap_kernel(I16));
    for want in [
        "RWStructuredBuffer<int16_t>",
        "StructuredBuffer<int16_t>",
        "int16_t l1 = (int16_t)0;",
        "((int16_t)(",
        "[numthreads(256, 1, 1)]",
    ] {
        assert!(g.source.contains(want), "missing `{want}`:\n{}", g.source);
    }
    assert!(
        !g.source.contains("int64") && !g.source.contains("ConstantBuffer"),
        "{}",
        g.source
    );
    let u = lower(U16, &widen_kernel(U16));
    assert!(
        u.source.contains("StructuredBuffer<uint16_t>"),
        "{}",
        u.source
    );
}
