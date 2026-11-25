// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 9e on a live device: predicated access (`Stmt::MaskedLoad`, `Stmt::MaskedStore`) through the SPIR-V
//! emitter. The shared kernel's masked-off lanes index out of range, so a load that ran would fault or read zero (not
//! the fill); its masked-off store must leave the value the plain store just wrote.
//!
//! Without a device the test follows `VULKANE_REQUIRE_DEVICE` like every other live test.

use vulkane_test_support::common;
use vulkane_test_support::glslang;
use vulkane_test_support::gpu;
use vulkane_test_support::masked_kernel;

use unpopped::backend::Backend;
use vulkane_spirv_emit::SpirV;

#[test]
fn masked_access_skips_the_memory_where_the_mask_is_false_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("kernel_masked") else {
        return; // skip already declared
    };
    let g = SpirV
        .lower_kernel(&masked_kernel::masked_kernel())
        .expect("the SPIR-V backend lowers the masked kernel");
    let words = g.words().expect("words").to_vec();
    if let Some((ok, err)) = glslang::validates(&words) {
        assert!(
            ok,
            "spirv-val rejected the module:
{err}"
        );
    }
    let got = gpu.run(&g.name, &words, &[&masked_kernel::input()]);
    for (t, &w) in got.iter().enumerate() {
        assert_eq!(w, masked_kernel::expected(t as u32), "thread {t}");
    }
}
