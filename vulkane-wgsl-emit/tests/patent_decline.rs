// SPDX-License-Identifier: MIT OR Apache-2.0
//! The statements declined for possible patent exposure (board 434 C and its extension, thirteen names; NVIDIA
//! US11294713B2, US12118382B2, US11803380B2, US12204897B2, US12710961B2) are refused **by name** with a typed error,
//! at the top level and at any `If`/`For`/`While` depth.
//!
//! The fixtures are the PUBLISHED ones, `unpopped::fixtures` (unpopped 0.58.5, feature `fixtures`; ROADMAP 9h): one
//! checker-valid kernel or module per `Stmt` variant. Every test asserts the fixture passes the checker first, so the
//! refusal seen here is the EMITTER's own text and never `check_kernel refused ...`.
//!
//! Controls: every statement that is NOT declined is run through the same emitter and its result carries no declined
//! marker; and the walk visits all thirteen names, so a name dropped from the table fails here.

use unpopped::backend::{Backend, LowerError};
use unpopped::fixtures::{Fixture, fixture, fixtures, variant_name};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, CollectiveKind, KernelFn, LocalDecl, Stmt, check_kernel, check_module,
};
use unpopped_vocab::ElementKind;
use vulkane_wgsl_emit::Wgsl;

const MARKER_434C: &str = "declined: possible exposure, board 434 C";
const MARKER_EXT: &str = "declined: possible exposure, board 434 C extended";

/// A nesting shape: its label and how it wraps a body (the extra local is the loop variable of a `For`).
type Shape = (&'static str, fn(Vec<Stmt>, u32) -> Vec<Stmt>);

/// A declined statement: the name the refusal must carry and the ruling text it must carry.
type Case = (&'static str, &'static str);

/// The five of board 434 C, then the eight of its extension (`CopyUnpack` since unpopped 0.58).
const DECLINED: [Case; 13] = [
    ("CopyAsync", MARKER_434C),
    ("AsyncCommit", MARKER_434C),
    ("AsyncWait", MARKER_434C),
    ("CollectiveTuple", MARKER_434C),
    ("Collective", MARKER_434C),
    ("MbarInit", MARKER_EXT),
    ("MbarArrive", MARKER_EXT),
    ("MbarCompleteTx", MARKER_EXT),
    ("MbarWait", MARKER_EXT),
    ("CopyBulk", MARKER_EXT),
    ("Tcgen05Commit", MARKER_EXT),
    ("WgmmaWait", MARKER_EXT),
    ("CopyUnpack", MARKER_EXT),
];

fn u32c(v: u64) -> ScalarExpr {
    ScalarExpr::ConstInt(v as _, ElementKind::U32)
}

/// The function of the fixture that holds the statement: the one kernel, or the module's entry.
fn entry(f: &Fixture) -> KernelFn {
    match f {
        Fixture::Kernel(k) => k.clone(),
        Fixture::Module { module, entry } => module.fns[*entry as usize].clone(),
        other => panic!("a fixture shape this test does not know: {other:?}"),
    }
}

/// The fixture with its entry function replaced.
fn with_entry(f: &Fixture, k: KernelFn) -> Fixture {
    match f {
        Fixture::Kernel(_) => Fixture::Kernel(k),
        Fixture::Module { module, entry } => {
            let mut module = module.clone();
            module.fns[*entry as usize] = k;
            Fixture::Module {
                module,
                entry: *entry,
            }
        }
        other => panic!("a fixture shape this test does not know: {other:?}"),
    }
}

/// Run the checker that suits the fixture, and fail with its text when it refuses.
fn assert_valid(what: &str, f: &Fixture) {
    let r = match f {
        Fixture::Kernel(k) => check_kernel(k),
        Fixture::Module { module, entry } => check_module(module, *entry),
        other => panic!("a fixture shape this test does not know: {other:?}"),
    };
    assert!(
        r.is_ok(),
        "{what}: the fixture must be checker-valid: {r:?}"
    );
}

fn contains(block: &Block, name: &str) -> bool {
    block.0.iter().any(|s| {
        variant_name(s) == name
            || match s {
                Stmt::If { then, els, .. } => contains(then, name) || contains(els, name),
                Stmt::For { body, .. } | Stmt::While { body, .. } => contains(body, name),
                _ => false,
            }
    })
}

fn refusal_of(k: &KernelFn) -> String {
    match Wgsl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected the typed UnsupportedKernel refusal, got {other:?}"),
    }
}

fn nest_if_then(b: Vec<Stmt>, _: u32) -> Vec<Stmt> {
    vec![Stmt::If {
        cond: u32c(1),
        then: Block(b),
        els: Block(vec![]),
    }]
}

fn nest_if_else(b: Vec<Stmt>, _: u32) -> Vec<Stmt> {
    vec![Stmt::If {
        cond: u32c(1),
        then: Block(vec![]),
        els: Block(b),
    }]
}

fn nest_for(b: Vec<Stmt>, local: u32) -> Vec<Stmt> {
    vec![Stmt::For {
        local,
        start: u32c(0),
        end: u32c(2),
        step: u32c(1),
        unroll: Default::default(),
        body: Block(b),
    }]
}

fn nest_while(b: Vec<Stmt>, _: u32) -> Vec<Stmt> {
    vec![Stmt::While {
        cond: u32c(1),
        body: Block(b),
    }]
}

fn nest_deep(b: Vec<Stmt>, local: u32) -> Vec<Stmt> {
    nest_for(nest_if_else(nest_while(b, local), local), local)
}

const SHAPES: [Shape; 6] = [
    ("top level", |b, _| b),
    ("if then", nest_if_then),
    ("if else", nest_if_else),
    ("for", nest_for),
    ("while", nest_while),
    ("for>if>while", nest_deep),
];

/// The fixture with the whole entry body wrapped by `shape`; the loop variable is a fresh `U32` local.
fn nested(f: &Fixture, shape: fn(Vec<Stmt>, u32) -> Vec<Stmt>) -> Fixture {
    let mut k = entry(f);
    let local = k.locals.len() as u32;
    k.locals.push(LocalDecl::scalar(ElementKind::U32));
    k.body = Block(shape(k.body.0.clone(), local));
    with_entry(f, k)
}

#[test]
fn each_declined_statement_is_refused_by_name_at_every_depth() {
    assert_eq!(
        DECLINED.len(),
        13,
        "the declined list is thirteen statements (twelve, and `CopyUnpack` since unpopped 0.58)"
    );
    for (name, marker) in DECLINED {
        let base = fixture(name).unwrap_or_else(|| panic!("unpopped::fixtures has no `{name}`"));
        assert_valid(name, &base);
        assert!(
            contains(&entry(&base).body, name),
            "{name}: the fixture's entry function must hold the statement itself"
        );
        for (where_, shape) in SHAPES {
            let f = nested(&base, shape);
            assert_valid(&format!("{name} at {where_}"), &f);
            let detail = refusal_of(&entry(&f));
            assert!(
                marker == MARKER_EXT || !detail.contains("extended"),
                "{name}: a board 434 C refusal must not claim the extension: {detail}"
            );
            assert!(
                !detail.contains("check_kernel refused"),
                "{name} at {where_}: the refusal must be the emitter's own, got: {detail}"
            );
            assert!(
                detail.contains(&format!("Stmt::{name}")) && detail.contains(marker),
                "{name} at {where_}: the refusal must name it and carry the marker, got: {detail}"
            );
        }
    }
}

#[test]
fn every_collective_kind_is_refused_by_name_the_new_exclusive_scans_included() {
    // unpopped 0.58.4 added exclusive Max/Min/Prod scans and a Prod fold. A non-NVIDIA emitter declines the WHOLE
    // `Collective` statement by name whatever its kind (board 434 C; board 506 holds subgroup scans for non-NVIDIA).
    let base = fixture("Collective").expect("a Collective fixture");
    let mut kinds: Vec<(String, CollectiveKind)> = Vec::new();
    for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod] {
        kinds.push((format!("Reduce({op:?})"), CollectiveKind::Reduce(op)));
        for exclusive in [false, true] {
            kinds.push((
                format!("Scan({op:?}, exclusive={exclusive})"),
                CollectiveKind::Scan { op, exclusive },
            ));
        }
    }
    for (label, kind) in kinds {
        let mut k = entry(&base);
        let mut replaced = 0;
        for s in &mut k.body.0 {
            if let Stmt::Collective { kind: slot, .. } = s {
                *slot = kind.clone();
                replaced += 1;
            }
        }
        assert_eq!(replaced, 1, "the fixture holds exactly one Collective");
        let f = with_entry(&base, k);
        assert_valid(&label, &f);
        let detail = refusal_of(&entry(&f));
        assert!(
            detail.contains("Stmt::Collective") && detail.contains(MARKER_434C),
            "{label}: must be refused by name, got: {detail}"
        );
    }
}

#[test]
fn a_statement_that_is_not_declined_never_carries_the_marker() {
    // The control: the same emitter on every OTHER statement's fixture. It may lower or refuse them for its own
    // reasons, but it must never mistake one for a declined statement.
    let declined: Vec<&str> = DECLINED.iter().map(|(n, _)| *n).collect();
    let mut seen = 0;
    for (name, f) in fixtures() {
        if declined.contains(&name) {
            continue;
        }
        assert_valid(name, &f);
        seen += 1;
        if let Err(LowerError::UnsupportedKernel { detail }) = Wgsl.lower_kernel(&entry(&f)) {
            assert!(
                !detail.contains("declined: possible exposure"),
                "{name} is not declined but its refusal carries the marker: {detail}"
            );
        }
    }
    assert!(
        seen >= 30,
        "the control must run over the other statements, ran over {seen}"
    );
}
