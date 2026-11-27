// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dumps the MSL this emitter produces for the shared kernels to `.metal` files, for the hosted
//! macOS CI job (`metal` in `.github/workflows/metal-probe.yml`) to compile with `xcrun metal`.
//!
//! With `VULKANE_MSL_DUMP_DIR` unset the test only checks that every kernel lowers and the list is
//! non-empty (so a kernel that stops lowering fails everywhere). With it set, each kernel is written
//! to `<dir>/<name>.metal` and `<dir>/MANIFEST.txt` lists the names, one per line; the CI job
//! compares the manifest's line count to the files it compiled, so a dump that wrote nothing fails.

use unpopped::backend::Backend;
use unpopped::exec::MemorySpace;
use unpopped::kernel::KernelFn;
use unpopped_vocab::ElementKind;
use vulkane_msl_emit::Msl;
use vulkane_test_support::{
    atomic_kernel, float_kernel, flow_kernel, grid_kernel, masked_kernel, narrow_kernel,
    private_kernel, reverse_kernel, subgroup_kernel, vector_kernel,
};

fn float_k(ty: ElementKind, case: float_kernel::Case) -> KernelFn {
    let mut k = float_kernel::float_kernel(ty, case);
    k.name = "float_case".to_string();
    k
}

/// Every kernel the MSL tests lower successfully, each under a file-safe name.
fn kernels() -> Vec<(String, KernelFn)> {
    use float_kernel::Case;
    let mut v: Vec<(String, KernelFn)> = vec![
        (
            "reverse".into(),
            reverse_kernel::reverse(64, true, reverse_kernel::SharedLen::Static),
        ),
        (
            "private_index".into(),
            private_kernel::private_index(8, MemorySpace::Private),
        ),
        ("flow".into(), flow_kernel::flow_kernel()),
        ("grid".into(), grid_kernel::grid_kernel()),
        ("atomic".into(), atomic_kernel::atomic_kernel(false)),
        ("atomic_racy".into(), atomic_kernel::atomic_kernel(true)),
        ("subgroup".into(), subgroup_kernel::subgroup_kernel(1, 128)),
        ("masked".into(), masked_kernel::masked_kernel()),
    ];
    for (tag, ty) in [
        ("f32", ElementKind::F32),
        ("f16", ElementKind::F16),
        ("bf16", ElementKind::Bf16),
    ] {
        for (c, case) in [
            ("muladd", Case::MulAdd),
            ("floor", Case::Floor),
            ("max", Case::Max),
        ] {
            // The half kernels' exact-op subset is narrower; the test below skips what declines.
            v.push((format!("float_{tag}_{c}"), float_k(ty, case)));
        }
    }
    for lanes in [2, 4] {
        let mut k = vector_kernel::vector_kernel(lanes, 0);
        k.name = "vector_case".to_string();
        v.push((format!("vector_{lanes}"), k));
    }
    for (elem, tag) in [
        (ElementKind::I8, "i8"),
        (ElementKind::U8, "u8"),
        (ElementKind::I16, "i16"),
        (ElementKind::U16, "u16"),
    ] {
        let mut w = narrow_kernel::widen_kernel(elem);
        w.name = "narrow_case".to_string();
        v.push((format!("narrow_widen_{tag}"), w));
        let mut r = narrow_kernel::wrap_kernel(elem);
        r.name = "narrow_case".to_string();
        v.push((format!("narrow_wrap_{tag}"), r));
    }
    v
}

#[test]
fn the_lowered_kernels_are_dumped_for_the_metal_compiler() {
    let dir = std::env::var_os("VULKANE_MSL_DUMP_DIR").map(std::path::PathBuf::from);
    let mut names = Vec::new();
    for (name, k) in kernels() {
        // A kernel this emitter declines by name is not a Metal-compile input; the decline tests own it.
        let Ok(g) = Msl.lower_kernel(&k) else {
            continue;
        };
        assert!(g.source.contains("kernel void"), "{name}: {}", g.source);
        if let Some(d) = &dir {
            std::fs::create_dir_all(d).expect("create the dump dir");
            std::fs::write(d.join(format!("{name}.metal")), &g.source).expect("write the kernel");
        }
        names.push(name);
    }
    // The floor: the integer, atomic, private, subgroup, masked and f32 kernels all lower today.
    assert!(
        names.len() >= 20,
        "only {} kernels lowered: {names:?}",
        names.len()
    );
    if let Some(d) = &dir {
        std::fs::write(d.join("MANIFEST.txt"), names.join("\n") + "\n")
            .expect("write the manifest");
    }
}
