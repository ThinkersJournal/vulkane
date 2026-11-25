// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 8n: shared test helpers come from this crate, never from a `#[path]` include.
//!
//! A `#[path]` include compiles the helper again into every test binary (hence a per-file
//! `#[allow(dead_code)]`) and ties a crate's tests to another crate's directory layout.

use std::path::{Path, PathBuf};

/// Does this line include a shared helper by path (`tests/common/mod.rs` or a `support/*.rs`)?
fn is_shared_include(line: &str) -> bool {
    let l = line.trim_start();
    l.starts_with("#[path") && (l.contains("common/mod.rs") || l.contains("support/"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn the_scanner_flags_a_shared_include_and_passes_a_plain_one() {
    // Positive control: the predicate fires on the two shapes in the tree, and not on a local `mod`.
    assert!(is_shared_include(
        "#[path = \"../../vulkane/tests/common/mod.rs\"]"
    ));
    assert!(is_shared_include("  #[path = \"support/gpu.rs\"]"));
    assert!(!is_shared_include("mod gpu;"));
    assert!(!is_shared_include(
        "#[path = \"../examples/emit_vocabulary_manifest.rs\"]"
    ));
}

#[test]
fn no_test_includes_a_shared_helper_by_path() {
    let root = workspace_root();
    let mut files = Vec::new();
    let mut crates = 0;
    for e in std::fs::read_dir(&root)
        .expect("read workspace root")
        .flatten()
    {
        let p = e.path();
        if p.join("Cargo.toml").is_file() && p.file_name().is_some_and(|n| n != "target") {
            crates += 1;
            for sub in ["tests", "src", "examples"] {
                rs_files(&p.join(sub), &mut files);
            }
        }
    }
    assert!(
        crates >= 10 && files.len() > 100,
        "scan saw {crates} crates, {} files",
        files.len()
    );
    let mut hits = Vec::new();
    for f in &files {
        let src = std::fs::read_to_string(f)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", f.display()));
        for (i, line) in src.lines().enumerate() {
            if is_shared_include(line) {
                hits.push(format!(
                    "{}:{}",
                    f.strip_prefix(&root).unwrap_or(f).display(),
                    i + 1
                ));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "{} shared `#[path]` includes remain (of {} files):\n{}",
        hits.len(),
        files.len(),
        hits.join("\n")
    );
}
