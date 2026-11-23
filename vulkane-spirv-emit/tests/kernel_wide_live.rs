// SPDX-License-Identifier: MIT OR Apache-2.0
//! 64-bit integer kernels (`I64`/`U64`) on a live device (`tests/support/wide_kernel.rs`): every
//! result crosses the 32-bit boundary, and the ruled integer semantics (zero divisor gives the
//! dividend, `MIN / -1 = MIN`, `Rem` as the `Select` composition, shift counts masked mod 64,
//! arithmetic vs logical `>>`) are checked cell by cell at 64 bits. Without `shaderInt64` the test
//! prints the capability readout in a `SKIP (unsupported)` line and returns; without a device it
//! follows `VULKANE_REQUIRE_DEVICE` like every other live test.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;
#[path = "support/wide_kernel.rs"]
mod wide_kernel;

use unpopped::backend::Backend;
use unpopped_vocab::ElementKind::{I64, U64};
use vulkane::safe::DeviceFeatures;
use vulkane_spirv_emit::SpirV;
use wide_kernel::{K, THREADS};

fn run(tag: &str, signed: bool) -> Option<Vec<u32>> {
    common::require_serialization_lock();
    let features = DeviceFeatures::new().with_shader_int64();
    let gpu = gpu::Gpu::open_with(tag, Some((&features, "shaderInt64")))?;
    eprintln!("capability readout: shaderInt64={}", gpu.shader_int64());
    let g = SpirV
        .lower_kernel(&wide_kernel::wide_kernel(if signed { I64 } else { U64 }))
        .expect("the SPIR-V backend lowers the 64-bit kernel");
    let words = g.words().expect("words").to_vec();
    Some(gpu.run_grid(&g.name, &words, THREADS as usize * K * 2, [1, 1, 1]))
}

fn check(got: &[u32], signed: bool) {
    let want = wide_kernel::expected_words(signed);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let (cell, half) = (i / 2, if i % 2 == 0 { "low" } else { "high" });
        assert_eq!(
            g,
            w,
            "thread {} result {} ({half} word), signed={signed}",
            cell / K,
            cell % K
        );
    }
}

#[test]
fn i64_values_across_the_32_bit_boundary_match_the_ruled_semantics_on_a_device() {
    if let Some(got) = run("spirv_kernel_i64", true) {
        check(&got, true);
    }
}

#[test]
fn u64_values_across_the_32_bit_boundary_match_the_ruled_semantics_on_a_device() {
    if let Some(got) = run("spirv_kernel_u64", false) {
        check(&got, false);
    }
}

/// The mutant: a lowering that goes through 32 bits wraps every value, constant and count at 2^32.
/// Its host result must differ from the 64-bit one in every result column the 32-bit boundary
/// reaches (all but the `x / 0` ... zero column 8), so the device check above would fail it.
#[test]
fn a_lowering_through_32_bits_fails_the_check_in_every_result_column() {
    for signed in [true, false] {
        for k in (0..K).filter(|&k| k != 8) {
            let differs = (0..u64::from(THREADS)).any(|t| {
                wide_kernel::expected_at(64, signed, t, k)
                    != wide_kernel::expected_at(32, signed, t, k)
            });
            assert!(
                differs,
                "column {k} (signed={signed}) would not catch a 32-bit lowering"
            );
        }
        // Control: the same function at the same width agrees with itself, so a difference above is
        // the width and not the oracle.
        assert!((0..u64::from(THREADS)).all(|t| (0..K).all(|k| {
            wide_kernel::expected_at(64, signed, t, k) == wide_kernel::expected(signed, t, k)
        })));
    }
}
