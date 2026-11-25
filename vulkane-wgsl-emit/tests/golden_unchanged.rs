// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP 8d condition: a kernel WITHOUT atomics lowers to byte-identical WGSL to what it did before
//! atomic buffers existed. The goldens in `tests/golden/` were written by the emitter before the
//! atomic typing change; `WGSL_GOLDEN_WRITE=1` rewrites them (only for a deliberate change).

#[path = "../../vulkane-spirv-emit/tests/support/flow_kernel.rs"]
#[allow(dead_code)]
mod flow_kernel;
#[path = "../../vulkane-spirv-emit/tests/support/grid_kernel.rs"]
#[allow(dead_code)]
mod grid_kernel;
#[path = "../../vulkane-spirv-emit/tests/support/private_kernel.rs"]
#[allow(dead_code)]
mod private_kernel;
#[path = "../../vulkane-spirv-emit/tests/support/reverse_kernel.rs"]
#[allow(dead_code)]
mod reverse_kernel;

use unpopped::backend::Backend;
use unpopped::exec::MemorySpace;
use unpopped::kernel::KernelFn;
use vulkane_spirv_emit::WORKGROUP_SIZE;
use vulkane_wgsl_emit::Wgsl;

fn kernels() -> Vec<(&'static str, KernelFn)> {
    let n = WORKGROUP_SIZE;
    vec![
        ("reverse_barrier", reverse_kernel::reverse(n, true, false)),
        ("reverse_racy", reverse_kernel::reverse(n, false, false)),
        ("reverse_dynamic", reverse_kernel::reverse(n, true, true)),
        ("flow", flow_kernel::flow_kernel()),
        ("grid", grid_kernel::grid_kernel()),
        (
            "private",
            private_kernel::private_index(n, MemorySpace::Private),
        ),
    ]
}

#[test]
fn kernels_without_atomics_lower_to_the_same_bytes_as_before() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let write = std::env::var_os("WGSL_GOLDEN_WRITE").is_some();
    for (name, k) in kernels() {
        let got = Wgsl.lower_kernel(&k).expect("lowers").source;
        let path = dir.join(format!("{name}.wgsl"));
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
        assert!(!got.contains("atomic"), "{name}");
    }
}
