// SPDX-License-Identifier: MIT OR Apache-2.0
//! 8- and 16-bit integer buffers through HLSL: `Hlsl::lower_kernel` hands them to the Slang emitter, which
//! declines each element type by name ("not lowered yet"; that is `fishstringer-slang-emit`'s). This file
//! pins what `dxc` itself can do, so the decline can be told from a real limit: it compiles `int16_t` and
//! `uint16_t` buffers with `-enable-16bit-types` (a 16-bit arm is possible, once a lowering writes it) and
//! has no 8-bit integer type (`int8_t` is undeclared: an 8-bit arm is not). Without `dxc` the pins follow
//! `VULKANE_REQUIRE_DXC`.

use std::process::Command;

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr as E};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;
use unpopped_vocab::ElementKind::{I8, I16, U8, U16};
use vulkane_hlsl_emit::Hlsl;

/// `out[t] = (compute) in[t]` over 256 threads, the input buffer and its local of the narrow type `elem`.
fn widen_kernel(elem: ElementKind) -> KernelFn {
    let compute = if matches!(elem, I8 | I16) {
        ElementKind::I32
    } else {
        ElementKind::U32
    };
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![E::Exec(ExecId::ThreadIdx(0))],
    };
    let widen = E::Cast(
        Box::new(E::Local(0)),
        CastSpec {
            to: compute,
            round: RoundMode::NearestEven,
            sat: Saturation::Wrap,
        },
    );
    let mut k = KernelFn::new(
        "widen",
        compute,
        vec![
            Param::buffer("input", elem, BufferRole::Read, Some(1)),
            Param::buffer("output", compute, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(elem)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: at(0),
            },
            Stmt::Store {
                place: at(1),
                value: widen,
            },
        ]),
    );
    k.launch = Some(LaunchShape::new([256, 1, 1], [1, 1, 1]));
    k
}

#[test]
fn an_8_or_16_bit_integer_kernel_is_a_typed_decline_naming_the_element() {
    for e in [I8, U8, I16, U16] {
        let err = Hlsl.lower_kernel(&widen_kernel(e)).expect_err("declined");
        let text = format!("{err:?}");
        assert!(
            text.contains("UnsupportedKernel") && text.contains(&format!("{e:?}")),
            "{e:?}: {text}"
        );
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
