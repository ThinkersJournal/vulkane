// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for 8/16-bit integer storage in `Backend::lower_kernel` (ROADMAP 8e): each capability is
//! declared only for the width a kernel uses, the module validates, `required_features` names it,
//! and the plan path still declines these dtypes.

use unpopped::backend::{Backend, DtypeSupport};
use unpopped_vocab::ElementKind::{I8, I16, U8, U16};
use unpopped_vocab::{ArchSku, TargetId};

use crate::tests::assert_valid;
use crate::{SpirV, required_features};

use vulkane_test_support::narrow_kernel;

const CAPS: [(&str, &str); 4] = [
    ("OpCapability Int8", "SPV_KHR_8bit_storage"),
    (
        "OpCapability StorageBuffer8BitAccess",
        "SPV_KHR_8bit_storage",
    ),
    ("OpCapability Int16", "SPV_KHR_16bit_storage"),
    (
        "OpCapability StorageBuffer16BitAccess",
        "SPV_KHR_16bit_storage",
    ),
];

#[test]
fn each_width_declares_only_its_own_capabilities_and_validates() {
    for (elem, wide) in [(I8, false), (U8, false), (I16, true), (U16, true)] {
        for k in [
            narrow_kernel::widen_kernel(elem),
            narrow_kernel::wrap_kernel(elem),
        ] {
            let g = SpirV.lower_kernel(&k).expect("lowers");
            assert_valid(&g);
            for (i, (cap, ext)) in CAPS.iter().enumerate() {
                let want = (i >= 2) == wide;
                assert_eq!(g.source.contains(cap), want, "{elem:?} {cap}\n{}", g.source);
                assert_eq!(
                    g.source.contains(&format!("OpExtension \"{ext}\"")),
                    want,
                    "{elem:?} {ext}"
                );
            }
            let need = required_features(g.words().expect("words")).expect("a module");
            assert_eq!(
                (need.shader_int8, need.storage_buffer_8bit),
                (!wide, !wide),
                "{need:?}"
            );
            assert_eq!(
                (need.shader_int16, need.storage_buffer_16bit),
                (wide, wide),
                "{need:?}"
            );
            assert!(!need.shader_int64, "{need:?}");
        }
    }
}

/// Control: the same queries find none of the narrow capabilities in a 32-bit kernel (which
/// `tests/golden_unchanged.rs` pins byte for byte).
#[test]
fn a_32_bit_kernel_declares_no_narrow_capability() {
    let k = crate::kernel_tests::reverse_kernel::reverse(
        64,
        true,
        crate::kernel_tests::reverse_kernel::SharedLen::Static,
    );
    let g = SpirV.lower_kernel(&k).expect("lowers");
    for (cap, ext) in CAPS {
        assert!(!g.source.contains(cap) && !g.source.contains(ext), "{cap}");
    }
}

#[test]
fn the_plan_path_still_declines_8_and_16_bit_integers() {
    for dt in [I8, U8, I16, U16] {
        assert_eq!(
            SpirV.supports_dtype(dt, TargetId::from(ArchSku::Sm89)),
            DtypeSupport::Unsupported,
            "{dt:?}"
        );
    }
}
