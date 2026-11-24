// SPDX-License-Identifier: MIT OR Apache-2.0
//! 8- and 16-bit integer buffers through WGSL are a typed decline naming why, pinned against naga: WGSL
//! has no such types, so the SPIR-V and GLSL arms' `I8`/`U8`/`I16`/`U16` storage has no spelling here.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::{CastSpec, RoundMode, Saturation, ScalarExpr as E};
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind;
use unpopped_vocab::ElementKind::{I8, I16, U8, U16};
use vulkane_wgsl_emit::Wgsl;

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
fn an_8_or_16_bit_integer_kernel_is_a_typed_decline_naming_why() {
    for e in [I8, U8, I16, U16] {
        let err = Wgsl.lower_kernel(&widen_kernel(e)).expect_err("declined");
        let text = format!("{err:?}");
        assert!(
            text.contains(&format!("{e:?}, an 8- or 16-bit integer"))
                && text.contains("no 8- or 16-bit integer types"),
            "{e:?}: {text}"
        );
    }
}

/// naga rejects the type names, so there is nothing to lower to. Control: the same module with `u32`
/// parses.
#[test]
fn naga_has_no_8_or_16_bit_integer_type() {
    let src = |ty: &str| {
        format!(
            "@group(0) @binding(0) var<storage, read> a: array<{ty}>;\n@compute @workgroup_size(1)\nfn main() {{ let x = a[0]; }}\n"
        )
    };
    assert!(naga::front::wgsl::parse_str(&src("u32")).is_ok());
    for ty in ["i8", "u8", "i16", "u16"] {
        assert!(
            naga::front::wgsl::parse_str(&src(ty)).is_err(),
            "naga now accepts {ty}: the WGSL 8/16-bit arm can be written"
        );
    }
}
