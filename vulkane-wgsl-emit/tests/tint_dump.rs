// SPDX-License-Identifier: MIT OR Apache-2.0
//! The WGSL the emitter writes for the board 506 A subgroup collectives, dumped for an independent validator (ROADMAP 8t).
//!
//! With `VULKANE_WGSL_DUMP_DIR` set, writes one `.wgsl` file per case plus `MANIFEST.txt` (one `name expectation` line per file);
//! the advisory CI job `.github/workflows/tint-validate.yml` feeds each file to Dawn's `tint` and checks the answer against the
//! expectation. Without the variable the test only checks that every case lowers, so a case that stops lowering fails everywhere.
//!
//! Three kinds of case, so a validator that accepts or rejects everything is caught: `collective_*` (the 14 ruled kinds; expect `accept`),
//! `control_plain` (a kernel with no collective; expect `accept`) and `control_broken` (this file's own text with one token damaged;
//! expect `reject`). Nothing here validates anything: naga 30.0.2 rejects `enable subgroups;`, and Tint is the one validator this
//! repository can run on a hosted runner.
//!
//! Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call; "outside as read" is a reading of public documents, not a
//! clearance; no attorney; findings, not advice.

use std::fmt::Write as _;
use std::path::PathBuf;

use unpopped::backend::Backend;
use unpopped::kernel::KernelFn;
use vulkane_test_support::{collective_kernel, reverse_kernel};
use vulkane_wgsl_emit::Wgsl;

struct Case {
    name: String,
    expect: &'static str,
    text: String,
}

fn lower(name: &str, k: &KernelFn) -> String {
    match Wgsl.lower_kernel(k) {
        Ok(g) => g.source,
        Err(e) => panic!("{name} must lower: {e:?}"),
    }
}

fn cases() -> Vec<Case> {
    let mut v: Vec<Case> = collective_kernel::kinds()
        .into_iter()
        .map(|(name, kind)| {
            let name = format!("collective_{name}");
            let text = lower(&name, &collective_kernel::kernel(kind));
            assert!(
                text.contains("enable subgroups;"),
                "{name}: a collective kernel must enable subgroups"
            );
            Case {
                name,
                expect: "accept",
                text,
            }
        })
        .collect();
    let plain = lower(
        "control_plain",
        &reverse_kernel::reverse(64, true, reverse_kernel::SharedLen::Static),
    );
    assert!(!plain.contains("enable subgroups;"));
    let broken = {
        let src = lower(
            "control_broken",
            &collective_kernel::kernel(unpopped::kernel::CollectiveKind::Reduce(
                unpopped::ir::ReduceOp::Sum,
            )),
        );
        assert!(src.contains("subgroupAdd("));
        // One damaged token: a function Tint has no such name for.
        src.replace("subgroupAdd(", "subgroupNoSuchFunction(")
    };
    v.push(Case {
        name: "control_plain".into(),
        expect: "accept",
        text: plain,
    });
    v.push(Case {
        name: "control_broken".into(),
        expect: "reject",
        text: broken,
    });
    v
}

#[test]
fn every_case_lowers_and_is_written_when_asked() {
    let all = cases();
    assert_eq!(all.len(), 16, "14 collectives and two controls");
    let Ok(dir) = std::env::var("VULKANE_WGSL_DUMP_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("dump dir");
    let mut manifest = String::new();
    for c in &all {
        std::fs::write(dir.join(format!("{}.wgsl", c.name)), &c.text).expect("write");
        let _ = writeln!(manifest, "{} {}", c.name, c.expect);
    }
    std::fs::write(dir.join("MANIFEST.txt"), manifest).expect("manifest");
}
