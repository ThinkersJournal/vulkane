// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 8d, 64-bit integer atomics on a live device (`tests/support/atomic_wide_kernel.rs`):
//! 1024 contended updates from 16 workgroups, each crossing the 32-bit boundary, so a 32-bit atomic
//! (which would drop the high halves) cannot pass. The device is created with `shaderInt64` and
//! `shaderBufferInt64Atomics`; without them the test prints a `SKIP (unsupported)` line naming them.
//! The control (the same add as a plain load/add/store) is reported, not asserted.

#[path = "support/atomic_wide_kernel.rs"]
mod atomic_wide_kernel;
#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;

use atomic_wide_kernel::{B, BLOCKS, SENTINEL, THREADS, atomic_wide_kernel};
use unpopped::backend::Backend;
use unpopped_vocab::ElementKind::I64;
use vulkane::safe::DeviceFeatures;
use vulkane_spirv_emit::SpirV;

fn run(gpu: &gpu::Gpu, racy: bool) -> Vec<u64> {
    let g = SpirV
        .lower_kernel(&atomic_wide_kernel(I64, racy))
        .expect("the SPIR-V backend lowers the 64-bit atomic kernel");
    let words = g.words().expect("words").to_vec();
    let got = gpu.run_grid(&g.name, &words, 8, [BLOCKS, 1, 1]);
    got.chunks(2)
        .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
        .collect()
}

#[test]
fn contended_64_bit_atomics_are_exact_on_a_device() {
    common::require_serialization_lock();
    let features = DeviceFeatures::new()
        .with_shader_int64()
        .with_shader_buffer_int64_atomics();
    let Some(gpu) = gpu::Gpu::open_with(
        "spirv_kernel_atomic64",
        Some((&features, "shaderInt64 + shaderBufferInt64Atomics")),
    ) else {
        return; // skip already declared
    };
    let got = run(&gpu, false);
    let adds = u64::from(THREADS) * (B as u64 + 1);
    assert_eq!(
        got[0],
        SENTINEL.wrapping_add(adds),
        "lost or 32-bit updates"
    );
    assert_eq!(
        got[1],
        1023 + (1 << 40),
        "signed maximum of the shifted ids"
    );
    assert!(got[2] > (1 << 33), "cas winner wrote {:#x}", got[2]);
    assert_eq!(
        got[3],
        SENTINEL.wrapping_add(1),
        "exactly one compare-exchange winner"
    );
    // Mutant check on the oracle: a 32-bit add would have dropped the high halves.
    assert_ne!(
        SENTINEL.wrapping_add(adds),
        SENTINEL.wrapping_add(u64::from(THREADS)),
        "the expected sum must need the high word"
    );
    let racy = run(&gpu, true);
    eprintln!(
        "kernel_atomic64 control: a plain 64-bit add left cell 0 at {:#x} of {:#x}",
        racy[0].wrapping_sub(SENTINEL),
        adds
    );
}
