// SPDX-License-Identifier: MIT OR Apache-2.0
//! Float kernels through HLSL are a typed decline, and this file pins why against `dxc` itself: the
//! compiler turns `precise` into `NoContraction` and `-denorm preserve` into `DenormPreserve`, but
//! nothing in a source file or on the command line asks for `SignedZeroInfNanPreserve`, which the
//! SPIR-V and GLSL float arms declare so NaN, infinity and signed zero follow the oracle. When a `dxc`
//! can spell it, the pin test fails and the arm can be written. Without `dxc` the pin test follows
//! `VULKANE_REQUIRE_DXC`.

use vulkane_test_support::reverse_kernel;

use std::process::Command;

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{BinaryOp, ScalarExpr as E, UnaryOp};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::{self, F16, F32, F64};
use vulkane_hlsl_emit::Hlsl;
use vulkane_spirv_emit::WORKGROUP_SIZE;

/// The expression a float kernel stores: `a * b + a` (a fusable pair), `Max(a, b)` or `Floor(a)`.
#[derive(Copy, Clone, Debug)]
enum Case {
    MulAdd,
    Max,
    Floor,
}

/// 256 threads each load `a[t]` and `b[t]` and store one result.
fn float_kernel(ty: ElementKind, case: Case) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![E::Exec(ExecId::ThreadIdx(0))],
    };
    let (a, b) = (E::Local(0), E::Local(1));
    let value = match case {
        Case::MulAdd => E::Add(
            Box::new(E::Mul(Box::new(a.clone()), Box::new(b))),
            Box::new(a),
        ),
        Case::Max => E::Binary(BinaryOp::Max, Box::new(a), Box::new(b)),
        Case::Floor => E::Unary(UnaryOp::Floor, Box::new(a)),
    };
    let body = vec![
        Stmt::Load {
            local: 0,
            place: at(0),
        },
        Stmt::Load {
            local: 1,
            place: at(1),
        },
        Stmt::Store {
            place: at(2),
            value,
        },
    ];
    let mut k = KernelFn::new(
        "float",
        ty,
        vec![
            Param::buffer("a", ty, BufferRole::Read, Some(1)),
            Param::buffer("b", ty, BufferRole::Read, Some(1)),
            Param::buffer("out", ty, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(ty); 2],
        Block(body),
    );
    k.launch = Some(LaunchShape::new([256, 1, 1], [1, 1, 1]));
    k
}

#[test]
fn a_float_kernel_is_a_typed_decline_naming_the_missing_float_control() {
    for ty in [F16, F32, F64] {
        for case in [Case::MulAdd, Case::Max, Case::Floor] {
            let err = Hlsl
                .lower_kernel(&float_kernel(ty, case))
                .expect_err("declined");
            let text = format!("{err:?}");
            assert!(
                text.contains("SignedZeroInfNanPreserve") && text.contains(&format!("{ty:?}")),
                "{ty:?} {case:?}: {text}"
            );
        }
    }
    // Control: an integer kernel still lowers, so the decline is the float type.
    Hlsl.lower_kernel(&reverse_kernel::reverse(WORKGROUP_SIZE, true, false))
        .expect("an integer kernel lowers");
}

const PROBE: &str = r#"
[[vk::binding(0, 0)]] StructuredBuffer<float> b0;
[[vk::binding(1, 0)]] StructuredBuffer<float> b1;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> b2;
[[vk::ext_capability(4464)]] [[vk::ext_capability(4466)]] [[vk::ext_extension("SPV_KHR_float_controls")]]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid_ : SV_GroupThreadID)
{
    precise float l0 = b0[tid_.x];
    precise float l1 = b1[tid_.x];
    precise float p0 = ((l0 * l1) + l0);
    b2[tid_.x] = p0;
}
"#;

fn dxc(source: &str, extra: &[&str]) -> Option<Result<Vec<u32>, String>> {
    let dir = tempfile::tempdir().unwrap();
    let (src, spv) = (dir.path().join("k.hlsl"), dir.path().join("k.spv"));
    std::fs::write(&src, source).unwrap();
    let out = Command::new("dxc")
        .args([
            "-T",
            "cs_6_2",
            "-E",
            "main",
            "-spirv",
            "-fspv-target-env=vulkan1.1",
        ])
        .args(extra)
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
    if !out.status.success() {
        return Some(Err(String::from_utf8_lossy(&out.stderr).into_owned()));
    }
    let bytes = std::fs::read(&spv).unwrap();
    Some(Ok(bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()))
}

/// The operand words of every instruction of `opcode`.
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

#[test]
fn dxc_spells_no_contraction_and_denorm_preserve_but_not_signed_zero_inf_nan() {
    let Some(compiled) = dxc(PROBE, &["-denorm", "preserve"]) else {
        return;
    };
    let words = compiled.expect("the probe compiles");
    let no_contraction = instructions(&words, 71)
        .iter()
        .filter(|o| o[1] == 42)
        .count();
    assert_eq!(no_contraction, 2, "`precise` gives NoContraction");
    let modes: Vec<u32> = instructions(&words, 16).iter().map(|o| o[1]).collect();
    assert!(
        modes.contains(&4459),
        "-denorm preserve gives DenormPreserve: {modes:?}"
    );
    assert!(
        !modes.contains(&4461),
        "dxc now emits SignedZeroInfNanPreserve: write the HLSL float arm ({modes:?})"
    );
    // Control: dropping `precise` loses the NoContraction decorations, so the count discriminates.
    let loose = PROBE.replace("precise ", "");
    let words = dxc(&loose, &["-denorm", "preserve"])
        .expect("dxc ran above")
        .expect("compiles");
    assert_eq!(
        instructions(&words, 71)
            .iter()
            .filter(|o| o[1] == 42)
            .count(),
        0
    );
}

/// The attribute that would declare the mode is a parse error in this `dxc`.
#[test]
fn dxc_rejects_an_execution_mode_attribute() {
    let with_mode = PROBE.replace(
        "[numthreads(256, 1, 1)]",
        "[numthreads(256, 1, 1)]\n[[vk::ext_execution_mode(4461, 32)]]",
    );
    let Some(result) = dxc(&with_mode, &[]) else {
        return;
    };
    assert!(
        result.is_err(),
        "dxc now accepts vk::ext_execution_mode: the HLSL float arm can declare SignedZeroInfNanPreserve"
    );
}
