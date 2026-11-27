// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 8d on a live device: 1024 contended atomic updates from 16 workgroups. The atomic
//! kernel must sum exactly, take the unsigned minimum, and let exactly one thread win a
//! compare-exchange. The control (the same count as a plain load/add/store) is run too and its loss
//! is **reported, not asserted**: whether a device interleaves the invocations is not promised.
//!
//! Without a device the test follows `VULKANE_REQUIRE_DEVICE` like every other live test.

use vulkane_test_support::atomic_kernel;
use vulkane_test_support::common;
use vulkane_test_support::gpu;

use atomic_kernel::{BLOCKS, SENTINEL, THREADS, atomic_kernel};
use unpopped::backend::Backend;
use vulkane_spirv_emit::SpirV;

fn run(gpu: &gpu::Gpu, racy: bool) -> Vec<u32> {
    let g = SpirV
        .lower_kernel(&atomic_kernel(racy))
        .expect("the SPIR-V backend lowers the atomic kernel");
    let words = g.words().expect("words").to_vec();
    gpu.run_grid(&g.name, &words, 4, [BLOCKS, 1, 1])
}

#[test]
fn contended_atomics_are_exact_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("kernel_atomic") else {
        return; // skip already declared
    };
    let got = run(&gpu, false);
    assert_eq!(got[0], SENTINEL.wrapping_add(THREADS), "lost updates");
    assert_eq!(got[1], 0, "unsigned minimum of the global ids");
    assert!(
        (1..=THREADS).contains(&got[2]),
        "cas winner wrote {}",
        got[2]
    );
    assert_eq!(
        got[3],
        SENTINEL.wrapping_add(1),
        "exactly one compare-exchange winner"
    );

    // Control, reported not asserted (see the module docs).
    let racy = run(&gpu, true);
    eprintln!(
        "kernel_atomic control: a plain increment reached {} of {THREADS} on this device",
        racy[0].wrapping_sub(SENTINEL)
    );
}
