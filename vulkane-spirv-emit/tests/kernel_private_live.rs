// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 8k on a live device: a per-thread `Private` array indexed by data. Every thread
//! fills its own four cells and reads one back at `in[t]`, so a private array that leaked between
//! threads would return another thread's value. Without a device the test follows
//! `VULKANE_REQUIRE_DEVICE` like every other live test.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;
#[path = "support/private_kernel.rs"]
mod private_kernel;

use unpopped::backend::Backend;
use unpopped::exec::MemorySpace;
use vulkane_spirv_emit::{SpirV, WORKGROUP_SIZE};

#[test]
fn a_private_array_with_data_dependent_indexing_is_per_thread() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("kernel_private") else {
        return; // skip already declared
    };
    let n = WORKGROUP_SIZE;
    let k = private_kernel::private_index(n, MemorySpace::Private);
    let g = SpirV.lower_kernel(&k).expect("lowers");
    let words = g.words().expect("words").to_vec();
    let input: Vec<u32> = (0..n).map(|t| (t * 5 + 1) % 4).collect();
    let want: Vec<u32> = (0..n).map(|t| t + 10 * input[t as usize]).collect();
    let got = gpu.run_spec(&g.name, &words, &[&input], None);
    assert_eq!(got, want);
    // Control: the indices are not all equal, so a wrong-index lowering could not pass.
    assert!(input.iter().any(|&i| i != input[0]));
}
