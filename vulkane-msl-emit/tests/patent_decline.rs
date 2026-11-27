// SPDX-License-Identifier: MIT OR Apache-2.0
//! The statements declined for possible patent exposure (board 434 C extended; NVIDIA US11803380B2, US12204897B2)
//! are refused **by name** with a typed error, at the top level and at any `If`/`For`/`While` depth.
//!
//! Controls: a kernel that holds no declined statement is not refused with the declined marker (the walk would
//! otherwise refuse everything), so the assertions below can tell a declined statement from an undeclined one; and the
//! walk visits every name, so a name dropped from the table fails here.
//!
//! Every kind of `Stmt::Collective` is pinned separately (ROADMAP 8c, 8o): Metal has a SIMD-group function for most of
//! them (MSL 4.1 6.10.2). **Board 506 "A" (narrowed; CireSnave 2026-11-26) lifted the subgroup-scope reductions, shuffles, votes and
//! broadcasts** (`vulkane_kernel_text::collective_is_ruled`): those are lowered and pinned in `tests/subgroup_collectives.rs`; here, scans, user
//! combines and every block-scope collective stay refused by name with the 434 C marker, and the ruled kinds are asserted NOT refused by name.
//! Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call: "outside as read" is a reading of public documents, not a clearance; no
//! attorney; findings, not advice.

use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{AllocDecl, Extent, LaunchShape, MemorySpace, Scope};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind,
};
use unpopped_vocab::ElementKind;
use vulkane_msl_emit::Msl;

const MARKER_434C: &str = "declined: possible exposure, board 434 C";
const MARKER_EXT: &str = "declined: possible exposure, board 434 C extended";

/// A nesting shape: its label and how it wraps a statement.
type Shape = (&'static str, Box<dyn Fn(Stmt) -> Vec<Stmt>>);

fn u32c(v: u64) -> ScalarExpr {
    ScalarExpr::ConstInt(v as _, ElementKind::U32)
}

/// A shared `U32` element (the destination of a copy).
fn tile() -> Place {
    Place::Alloc {
        alloc: 1,
        index: u32c(0),
    }
}

/// A global buffer element (the only source the checker admits for a copy).
fn global() -> Place {
    Place::Logical {
        buffer: 0,
        coords: vec![u32c(0)],
    }
}

fn bar() -> Place {
    Place::Alloc {
        alloc: 0,
        index: u32c(0),
    }
}

fn kernel(body: Vec<Stmt>) -> KernelFn {
    let mut k = KernelFn::new(
        "patent",
        ElementKind::U32,
        vec![Param::buffer(
            "out",
            ElementKind::U32,
            BufferRole::ReadWrite,
            Some(1),
        )],
        // Two `U32` locals: the `For` variable and the collectives' results (the checker refuses an undeclared local).
        vec![LocalDecl::scalar(ElementKind::U32); 2],
        Block(body),
    )
    .with_launch(LaunchShape::new([2, 1, 1], [1, 1, 1]));
    k.allocs = vec![
        AllocDecl::new(
            "bar",
            MemorySpace::Shared,
            ElementKind::U64,
            Extent::Static(1),
        ),
        // The destination of the copies (the source is a global buffer place: the checker refuses any other).
        AllocDecl::new(
            "tile",
            MemorySpace::Shared,
            ElementKind::U32,
            Extent::Static(4),
        ),
    ];
    k
}

/// A declined statement: the name the refusal must carry, the ruling text it must carry, and the statement.
type Case = (&'static str, &'static str, Stmt);

/// The five of board 434 C.
fn declined_434c() -> Vec<Case> {
    vec![
        (
            "CopyAsync",
            MARKER_434C,
            Stmt::CopyAsync {
                dst: tile(),
                src: global(),
                elems: 1,
                valid: None,
                cache: Default::default(),
                l2_prefetch: None,
            },
        ),
        ("AsyncCommit", MARKER_434C, Stmt::AsyncCommit),
        ("AsyncWait", MARKER_434C, Stmt::AsyncWait { pending: 0 }),
        (
            "CollectiveTuple",
            MARKER_434C,
            Stmt::CollectiveTuple {
                kind: CollectiveKind::ReduceWith { combine: 0 },
                values: vec![u32c(1), u32c(2)],
                results: vec![0, 1],
                scope: Scope::Subgroup,
            },
        ),
        (
            "Collective",
            MARKER_434C,
            Stmt::Collective {
                // A scan: board 506 A lifted the reductions, shuffles, votes and broadcasts, not the scans.
                kind: CollectiveKind::Scan {
                    op: ReduceOp::Sum,
                    exclusive: false,
                },
                value: u32c(1),
                result: 1,
                scope: Scope::Subgroup,
            },
        ),
    ]
}

/// The seven of the 2026-11-16 extension.
fn declined_ext() -> Vec<Case> {
    let ext = MARKER_EXT;
    vec![
        (
            "MbarInit",
            ext,
            Stmt::MbarInit {
                bar: bar(),
                count: u32c(1),
            },
        ),
        (
            "MbarArrive",
            ext,
            Stmt::MbarArrive {
                bar: bar(),
                tx: None,
            },
        ),
        (
            "MbarCompleteTx",
            ext,
            Stmt::MbarCompleteTx {
                bar: bar(),
                bytes: u32c(4),
            },
        ),
        (
            "MbarWait",
            ext,
            Stmt::MbarWait {
                bar: bar(),
                parity: u32c(0),
            },
        ),
        (
            "CopyBulk",
            ext,
            Stmt::CopyBulk {
                dst: tile(),
                src: global(),
                elems: 4,
                bar: bar(),
            },
        ),
        ("Tcgen05Commit", ext, Stmt::Tcgen05Commit { bar: bar() }),
        ("WgmmaWait", ext, Stmt::WgmmaWait { pending: 0 }),
    ]
}

/// Every declined statement.
fn declined() -> Vec<Case> {
    let mut all = declined_434c();
    all.extend(declined_ext());
    all
}

fn refusal(body: Vec<Stmt>) -> String {
    match Msl.lower_kernel(&kernel(body)) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected the typed UnsupportedKernel refusal, got {other:?}"),
    }
}

fn nest_if_then(s: Stmt) -> Stmt {
    Stmt::If {
        cond: u32c(1),
        then: Block(vec![s]),
        els: Block(vec![]),
    }
}

fn nest_if_else(s: Stmt) -> Stmt {
    Stmt::If {
        cond: u32c(1),
        then: Block(vec![]),
        els: Block(vec![s]),
    }
}

fn nest_for(s: Stmt) -> Stmt {
    Stmt::For {
        local: 0,
        start: u32c(0),
        end: u32c(2),
        step: u32c(1),
        unroll: Default::default(),
        body: Block(vec![s]),
    }
}

fn nest_while(s: Stmt) -> Stmt {
    Stmt::While {
        cond: u32c(1),
        body: Block(vec![s]),
    }
}

/// The nesting shapes a declined statement is hidden in.
fn shapes() -> Vec<Shape> {
    vec![
        ("top level", Box::new(|s| vec![s])),
        ("if then", Box::new(|s| vec![nest_if_then(s)])),
        ("if else", Box::new(|s| vec![nest_if_else(s)])),
        ("for", Box::new(|s| vec![nest_for(s)])),
        ("while", Box::new(|s| vec![nest_while(s)])),
        (
            "for>if>while",
            Box::new(|s| vec![nest_for(nest_if_else(nest_while(s)))]),
        ),
    ]
}

#[test]
fn each_declined_statement_is_refused_by_name_at_every_depth() {
    assert_eq!(
        declined().len(),
        12,
        "the declined list is twelve statements"
    );
    for (name, marker, _) in declined() {
        for (where_, build) in shapes() {
            let stmt = declined()
                .into_iter()
                .find(|(n, _, _)| *n == name)
                .map(|(_, _, s)| s)
                .expect("the statement exists");
            let detail = refusal(build(stmt));
            assert!(
                marker == MARKER_EXT || !detail.contains("extended"),
                "{name}: a board 434 C refusal must not claim the extension: {detail}"
            );
            // The refusal is the emitter's own table, not the kernel checker's (whose text can already name a construct).
            assert!(
                !detail.contains("check_kernel refused"),
                "{name} at {where_}: the checker refused it, so the emitter's arm was not reached: {detail}"
            );
            assert!(
                detail.contains(&format!("Stmt::{name}")) && detail.contains(marker),
                "{name} at {where_}: the refusal must name it and carry the marker, got: {detail}"
            );
        }
    }
}

#[test]
fn a_kernel_without_a_declined_statement_is_refused_without_the_marker() {
    // The control: the same walk, no declined statement. It must not be mistaken for one.
    let detail = refusal(vec![nest_for(nest_if_then(Stmt::Return))]);
    assert!(
        !detail.contains(MARKER_434C) && !detail.contains("Stmt::"),
        "an innocent kernel must not carry the declined marker: {detail}"
    );
    let empty = refusal(vec![]);
    assert!(!empty.contains(MARKER_434C), "{empty}");
}

/// Every `CollectiveKind` the IR has, with the MSL 4.1 6.10.2 function that would spell it (so a reader sees what is NOT
/// lowered): all are refused by name, because the statement is one of board 434 C.
fn every_collective_kind() -> Vec<(&'static str, CollectiveKind)> {
    use ReduceOp::{And, Max, Min, Or, Prod, Sum, Xor};
    let sh = |mode| CollectiveKind::Shuffle {
        mode,
        operand: u32c(1),
    };
    vec![
        ("simd_shuffle", sh(ShuffleMode::Idx)),
        ("simd_shuffle_up", sh(ShuffleMode::Up)),
        ("simd_shuffle_down", sh(ShuffleMode::Down)),
        ("simd_shuffle_xor", sh(ShuffleMode::Xor)),
        ("simd_all", CollectiveKind::Vote(VoteKind::All)),
        ("simd_any", CollectiveKind::Vote(VoteKind::Any)),
        ("simd_ballot", CollectiveKind::Vote(VoteKind::Ballot)),
        ("simd_sum", CollectiveKind::Reduce(Sum)),
        ("simd_product", CollectiveKind::Reduce(Prod)),
        ("simd_max", CollectiveKind::Reduce(Max)),
        ("simd_min", CollectiveKind::Reduce(Min)),
        ("simd_and", CollectiveKind::Reduce(And)),
        ("simd_or", CollectiveKind::Reduce(Or)),
        ("simd_xor", CollectiveKind::Reduce(Xor)),
        (
            "simd_prefix_inclusive_sum",
            CollectiveKind::Scan {
                op: Sum,
                exclusive: false,
            },
        ),
        (
            "simd_prefix_exclusive_sum",
            CollectiveKind::Scan {
                op: Sum,
                exclusive: true,
            },
        ),
        (
            "simd_broadcast",
            CollectiveKind::Broadcast { lane: u32c(0) },
        ),
        // No MSL function takes a user combine function.
        (
            "(none: user combine)",
            CollectiveKind::ReduceWith { combine: 0 },
        ),
        (
            "(none: user combine)",
            CollectiveKind::ScanWith { combine: 0 },
        ),
    ]
}

#[test]
fn only_the_kinds_outside_the_ruled_set_are_refused_by_name_whatever_msl_function_would_spell_them()
{
    let (mut ruled_n, mut declined_n) = (0, 0);
    for (msl_fn, kind) in every_collective_kind() {
        let ruled = vulkane_kernel_text::collective_is_ruled(&kind, Scope::Subgroup);
        let stmt = Stmt::Collective {
            kind: kind.clone(),
            value: u32c(1),
            result: 1,
            scope: Scope::Subgroup,
        };
        for (shape, wrap) in shapes() {
            let result = Msl.lower_kernel(&kernel(wrap(stmt.clone())));
            if ruled {
                // Lowered, or refused for some OTHER reason of this probe kernel (no launch subgroup size, no locals): never by name.
                if let Err(LowerError::UnsupportedKernel { detail }) = &result {
                    assert!(
                        !detail.contains(MARKER_434C),
                        "{kind:?} (MSL {msl_fn}) at {shape} is in the ruled set and must not be refused by name: {detail}"
                    );
                }
            } else {
                let Err(LowerError::UnsupportedKernel { detail }) = result else {
                    panic!("{kind:?} (MSL {msl_fn}) at {shape} must be refused");
                };
                assert!(
                    !detail.contains("check_kernel refused"),
                    "{kind:?} at {shape}: the checker refused it: {detail}"
                );
                assert!(
                    detail.contains("Stmt::Collective") && detail.contains(MARKER_434C),
                    "{kind:?} (MSL {msl_fn}) at {shape}: {detail}"
                );
            }
        }
        if ruled {
            ruled_n += 1;
        } else {
            declined_n += 1;
        }
    }
    // The table is complete: 15 ruled kinds (7 reductions, 4 shuffles, 3 votes, 1 broadcast) and 4 that stay declined (2 scans, 2 combines).
    assert_eq!((ruled_n, declined_n), (15, 4));
}

#[test]
fn a_ruled_kind_over_a_block_is_refused_by_name() {
    // Board 506 A lifted SUBGROUP scope only: every block-scope collective stays declined, whatever the kind.
    for (msl_fn, kind) in every_collective_kind() {
        let stmt = Stmt::Collective {
            kind: kind.clone(),
            value: u32c(1),
            result: 0,
            scope: Scope::Block,
        };
        let detail = refusal(vec![stmt]);
        assert!(
            detail.contains("Stmt::Collective") && detail.contains(MARKER_434C),
            "{kind:?} (MSL {msl_fn}) over a Block: {detail}"
        );
    }
}

/// The probe kernels must pass the kernel checker, at every nesting depth, so a refusal can only be the emitter's table
/// (see the SPIR-V, GLSL, HLSL and WGSL suites). Exceptions: a statement whose combine function only a module can resolve
/// (`CollectiveTuple`, and `Collective` with `ReduceWith`/`ScanWith`): the test pins that this is the ONLY reason.
#[test]
fn every_probe_kernel_passes_the_kernel_checker() {
    let module_only = |checked: &Result<(), unpopped::kernel::KernelError>| {
        format!("{checked:?}").contains("outside a module")
    };
    for (name, _, stmt) in declined() {
        for (where_, build) in shapes() {
            let checked = unpopped::kernel::check_kernel(&kernel(build(stmt.clone())));
            if name == "CollectiveTuple" {
                assert!(module_only(&checked), "{name} at {where_}: {checked:?}");
            } else {
                assert!(checked.is_ok(), "{name} at {where_}: {checked:?}");
            }
        }
    }
    for (msl_fn, kind) in every_collective_kind() {
        let user_combine = matches!(
            kind,
            CollectiveKind::ReduceWith { .. } | CollectiveKind::ScanWith { .. }
        );
        // (unpopped before 0.60 refused a product fold in the checker, so it needed its own arm here; the checker at the taken
        // 0.60.26 accepts it, and board 506 A lowers it, so a product is checked like every other kind.)
        let stmt = Stmt::Collective {
            kind: kind.clone(),
            value: u32c(1),
            result: 1,
            scope: Scope::Subgroup,
        };
        for (shape, wrap) in shapes() {
            let checked = unpopped::kernel::check_kernel(&kernel(wrap(stmt.clone())));
            if user_combine {
                assert!(module_only(&checked), "{kind:?} at {shape}: {checked:?}");
            } else {
                assert!(
                    checked.is_ok(),
                    "{kind:?} (MSL {msl_fn}) at {shape}: {checked:?}"
                );
            }
        }
    }
}
