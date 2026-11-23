// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for 64-bit integer kernels (`I64`/`U64`) in `Backend::lower_kernel`: the capability and type
//! appear only for a kernel that uses 64-bit values, the plan path still declines the dtype, floats
//! stay declined, and 64-bit atomics are a decline by name until their own slice.

use unpopped::backend::{Backend, DtypeSupport, LowerError};
use unpopped::kernel::{AtomicOp, Block, BufferRole, MemoryOrder, Param, Place, Stmt};
use unpopped_vocab::ElementKind::{F32, F64, I64, U64};
use unpopped_vocab::{ArchSku, TargetId};

use crate::SpirV;
use crate::tests::assert_valid;

use crate::kernel_tests::reverse_kernel;

#[path = "../tests/support/wide_kernel.rs"]
#[allow(dead_code)]
mod wide_kernel;

#[test]
fn a_64_bit_kernel_declares_int64_and_a_64_bit_type_and_validates() {
    for (ty, sign) in [(I64, "1"), (U64, "0")] {
        let g = SpirV
            .lower_kernel(&wide_kernel::wide_kernel(ty))
            .expect("lowers");
        assert_valid(&g);
        assert!(g.source.contains("OpCapability Int64"), "{}", g.source);
        assert!(
            g.source.contains(&format!("OpTypeInt 64 {sign}")),
            "{}",
            g.source
        );
        // The array stride is 8 bytes, not 4.
        assert!(g.source.contains("ArrayStride 8"), "{}", g.source);
    }
}

/// Control for the above: the same queries find nothing in a 32-bit kernel (which the goldens in
/// `tests/golden_unchanged.rs` pin byte for byte).
#[test]
fn a_32_bit_kernel_declares_no_int64() {
    let k = reverse_kernel::reverse(64, true, false);
    let g = SpirV.lower_kernel(&k).expect("lowers");
    assert!(!g.source.contains("Int64"), "{}", g.source);
    assert!(!g.source.contains("OpTypeInt 64"), "{}", g.source);
}

#[test]
fn the_plan_path_still_declines_64_bit_integers() {
    for dt in [I64, U64] {
        assert_eq!(
            SpirV.supports_dtype(dt, TargetId::from(ArchSku::Sm89)),
            DtypeSupport::Unsupported,
            "{dt:?}"
        );
    }
}

#[test]
fn float_kernels_stay_a_typed_decline() {
    for ty in [F32, F64] {
        let mut k = unpopped::kernel::KernelFn::new(
            "float",
            ty,
            vec![Param::buffer("out", ty, BufferRole::Write, Some(1))],
            vec![],
            Block(vec![]),
        );
        k.launch = Some(unpopped::exec::LaunchShape::new([1, 1, 1], [1, 1, 1]));
        match SpirV.lower_kernel(&k) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(detail.contains("integer kernels only"), "{detail}");
            }
            other => panic!("expected a decline at {ty:?}, got {other:?}"),
        }
    }
}

#[test]
fn a_64_bit_atomic_is_a_decline_by_name() {
    let mut k = wide_kernel::wide_kernel(U64);
    k.params[0] = Param::buffer("out", U64, BufferRole::ReadWrite, Some(1));
    k.body.0.push(Stmt::AtomicRmw {
        op: AtomicOp::Add,
        place: Place::Logical {
            buffer: 0,
            coords: vec![unpopped::ir::ScalarExpr::ConstInt(0, U64)],
        },
        value: unpopped::ir::ScalarExpr::ConstInt(1, U64),
        old: None,
        order: MemoryOrder::Relaxed,
        scope: unpopped::exec::Scope::Device,
    });
    match SpirV.lower_kernel(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("Int64Atomics"), "{detail}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}
