// SPDX-License-Identifier: MIT OR Apache-2.0
//! The README's infringement note (CireSnave 2026-11-16) must not disappear: it names all thirteen declined
//! statements, the two basis rulings and the patents, and tells emitter builders not to lower them.

const README: &str = include_str!("../README.md");

const NAMES: [&str; 13] = [
    "CopyAsync",
    "AsyncCommit",
    "AsyncWait",
    "Collective",
    "CollectiveTuple",
    "MbarInit",
    "MbarArrive",
    "MbarCompleteTx",
    "MbarWait",
    "CopyBulk",
    "Tcgen05Commit",
    "WgmmaWait",
    "CopyUnpack",
];

#[test]
fn readme_carries_the_infringement_note() {
    for needle in [
        "## Declined: possible patent exposure",
        "risks infringing NVIDIA patents; do not",
        "434 C",
        "US11294713B2",
        "US12118382B2",
        "US11803380B2",
        "US12204897B2",
        "PATENT-SWEEP",
    ] {
        assert!(README.contains(needle), "README.md lost `{needle}`");
    }
}

#[test]
fn readme_names_every_declined_statement() {
    for name in NAMES {
        assert!(
            README.contains(&format!("`{name}`")),
            "README.md no longer names `{name}`"
        );
    }
}
