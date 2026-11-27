// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for 64-bit integer kernels (`I64`/`U64`) in `Backend::lower_kernel`: the capability and type
//! appear only for a kernel that uses 64-bit values, the plan path still declines the dtype, floats
//! stay declined, and 64-bit atomics are a decline by name until their own slice.

use unpopped::backend::{Backend, DtypeSupport, LowerError};
use unpopped::kernel::{Block, BufferRole, Param};
use unpopped_vocab::ElementKind::{Bf16, I64, U64};
use unpopped_vocab::{ArchSku, TargetId};

use crate::SpirV;
use crate::tests::assert_valid;

use crate::kernel_tests::reverse_kernel;

use vulkane_test_support::atomic_wide_kernel;
use vulkane_test_support::wide_kernel;

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
    let k = reverse_kernel::reverse(64, true, reverse_kernel::SharedLen::Static);
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
fn kernels_at_an_unlowered_dtype_stay_a_typed_decline() {
    let ty = Bf16;
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
            assert!(detail.contains("kernels only"), "{detail}");
        }
        other => panic!("expected a decline at {ty:?}, got {other:?}"),
    }
}

#[test]
fn a_32_bit_kernel_requires_no_int64_feature() {
    let g = SpirV
        .lower_kernel(&reverse_kernel::reverse(
            64,
            true,
            reverse_kernel::SharedLen::Static,
        ))
        .expect("lowers");
    let need = crate::required_features(g.words().expect("words")).expect("a module");
    assert_eq!(need, crate::RequiredFeatures::default());
}

/// A 64-bit atomic declares `Int64Atomics` (once, only for such a kernel), uses the signed or
/// unsigned instruction by the dtype, validates, and is named by `required_features`.
#[test]
fn a_64_bit_atomic_declares_int64_atomics_only_for_a_kernel_that_has_one() {
    for (ty, max_op) in [(I64, "OpAtomicSMax"), (U64, "OpAtomicUMax")] {
        let g = SpirV
            .lower_kernel(&atomic_wide_kernel::atomic_wide_kernel(ty, false))
            .expect("lowers");
        assert_valid(&g);
        assert_eq!(
            g.source.matches("OpCapability Int64Atomics").count(),
            1,
            "{}",
            g.source
        );
        assert!(g.source.contains("OpAtomicIAdd"), "{}", g.source);
        assert!(g.source.contains(max_op), "{max_op}\n{}", g.source);
        let need = crate::required_features(g.words().expect("words")).expect("a module");
        assert!(need.shader_int64 && need.int64_atomics, "{need:?}");
    }
    // Control: the same queries find nothing in a 64-bit kernel with no atomic.
    let plain = SpirV
        .lower_kernel(&wide_kernel::wide_kernel(I64))
        .expect("lowers");
    assert!(!plain.source.contains("Int64Atomics"), "{}", plain.source);
    assert!(!plain.source.contains("OpAtomic"), "{}", plain.source);
}
