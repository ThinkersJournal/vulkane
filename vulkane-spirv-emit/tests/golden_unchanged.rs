// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP 8d/8c condition: a 32-bit kernel lowers to byte-identical SPIR-V to what it did before
//! 64-bit integer kernels existed (so no `Int64` capability or 64-bit type leaks into it). The goldens
//! in `tests/golden/` were written by the emitter BEFORE the 64-bit change;
//! `SPIRV_GOLDEN_WRITE=1` rewrites them (only for a deliberate change).

use vulkane_test_support::atomic_kernel;
use vulkane_test_support::flow_kernel;
use vulkane_test_support::grid_kernel;
use vulkane_test_support::private_kernel;
use vulkane_test_support::reverse_kernel;
use vulkane_test_support::subgroup_kernel;

use unpopped::backend::Backend;
use unpopped::exec::MemorySpace;
use unpopped::kernel::KernelFn;
use vulkane_spirv_emit::{SpirV, WORKGROUP_SIZE};

fn kernels() -> Vec<(&'static str, KernelFn)> {
    let n = WORKGROUP_SIZE;
    vec![
        ("reverse_barrier", reverse_kernel::reverse(n, true, false)),
        ("reverse_dynamic", reverse_kernel::reverse(n, true, true)),
        ("flow", flow_kernel::flow_kernel()),
        ("grid", grid_kernel::grid_kernel()),
        ("atomic", atomic_kernel::atomic_kernel(false)),
        ("subgroup", subgroup_kernel::subgroup_kernel(1, 128)),
        (
            "private",
            private_kernel::private_index(n, MemorySpace::Private),
        ),
    ]
}

#[test]
fn kernels_without_64_bit_values_lower_to_the_same_module_as_before() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let write = std::env::var_os("SPIRV_GOLDEN_WRITE").is_some();
    for (name, k) in kernels() {
        let got = SpirV.lower_kernel(&k).expect("lowers").source;
        let path = dir.join(format!("{name}.spvasm"));
        if write {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &got).unwrap();
        }
        let want = std::fs::read_to_string(&path).expect("golden present");
        assert_eq!(
            got.replace('\r', ""),
            want.replace('\r', ""),
            "{name} changed"
        );
        assert!(
            !got.contains("Int64") && !got.contains("OpTypeInt 64"),
            "{name}"
        );
    }
}
