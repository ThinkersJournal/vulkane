// SPDX-License-Identifier: MIT OR Apache-2.0
//! Float kernels through WGSL are a typed decline, and this file pins why against naga (the one WGSL
//! compiler used here): WGSL has no spelling for what the SPIR-V and GLSL float arms declare, so the
//! SPIR-V naga writes for `a * b + a` carries no `NoContraction` and no float-control mode. WGSL can
//! therefore not be held to the oracle's bit-exact IEEE contract and nothing is lowered, so there is
//! nothing to run on a device.

use unpopped::backend::Backend;
use unpopped::exec::{ExecId, LaunchShape};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{Block, BufferRole, KernelFn, LocalDecl, Param, Place, Stmt};
use unpopped_vocab::ElementKind::{self, F16, F32, F64};
use vulkane_wgsl_emit::Wgsl;

/// 256 threads each load `a[t]` and `b[t]` and store `a * b + a`.
fn float_kernel(ty: ElementKind) -> KernelFn {
    let at = |buffer| Place::Logical {
        buffer,
        coords: vec![E::Exec(ExecId::ThreadIdx(0))],
    };
    let (a, b) = (E::Local(0), E::Local(1));
    let value = E::Add(
        Box::new(E::Mul(Box::new(a.clone()), Box::new(b))),
        Box::new(a),
    );
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
        let err = Wgsl.lower_kernel(&float_kernel(ty)).expect_err("declined");
        let text = format!("{err:?}");
        assert!(
            text.contains("SignedZeroInfNanPreserve") && text.contains(&format!("a {ty:?} kernel")),
            "{ty:?}: {text}"
        );
    }
    // Control: an integer kernel (the same shape at U32) still lowers, so the decline is the dtype.
    let mut k = float_kernel(ElementKind::U32);
    k.body = Block(vec![
        Stmt::Load {
            local: 0,
            place: Place::Logical {
                buffer: 0,
                coords: vec![E::Exec(ExecId::ThreadIdx(0))],
            },
        },
        Stmt::Store {
            place: Place::Logical {
                buffer: 2,
                coords: vec![E::Exec(ExecId::ThreadIdx(0))],
            },
            value: E::Local(0),
        },
    ]);
    Wgsl.lower_kernel(&k).expect("an integer kernel lowers");
}

const PROBE: &str = "@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    o[t] = a[t] * b[t] + a[t];
}
";

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
fn naga_writes_no_no_contraction_and_no_float_control_for_a_multiply_add() {
    let module = naga::front::wgsl::parse_str(PROBE).expect("parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validates");
    let words =
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .expect("SPIR-V");
    // Control: the query sees the multiply, the add and the workgroup size, so a zero below means
    // absent, not unseen.
    assert_eq!(instructions(&words, 133).len(), 1, "OpFMul");
    assert_eq!(instructions(&words, 129).len(), 1, "OpFAdd");
    let modes: Vec<u32> = instructions(&words, 16).iter().map(|o| o[1]).collect();
    assert!(modes.contains(&17), "LocalSize: {modes:?}");
    let no_contraction = instructions(&words, 71)
        .iter()
        .filter(|o| o[1] == 42)
        .count();
    assert_eq!(
        no_contraction, 0,
        "naga now writes NoContraction: the WGSL float arm can be written"
    );
    for mode in [4459, 4460, 4461, 4462, 4463] {
        assert!(
            !modes.contains(&mode),
            "naga now writes float-control mode {mode}: the WGSL float arm can be written"
        );
    }
}
