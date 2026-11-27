// SPDX-License-Identifier: MIT OR Apache-2.0
//! Board 506 "A" (narrowed; CireSnave 2026-11-26): the ruled subgroup collectives (reductions, shuffles, votes, broadcasts) lower to WGSL
//! `subgroup*` built-ins; scans, user combines, `CollectiveTuple` and block scope stay refused by name with the board 434 C marker.
//!
//! Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call: "outside as read" is a reading of public documents, not a clearance; no
//! attorney; findings, not advice.
//!
//! What this file pins: the emitted TEXT of each ruled kind (the function beside its `CollectiveKind`), the by-name declines, and the
//! checker's divergent-placement refusals (from `unpopped::fixtures::divergent_fixtures()`, unpopped 0.60). What it does NOT pin: a WGSL
//! compiler's verdict (naga 30.0.2 rejects `enable subgroups;`: see the last test; Tint is not installed) and behaviour on a device (the
//! device cases are a separate test).

use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::fixtures::{divergent_fixtures, fixture};
use unpopped::ir::{ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind, check_kernel,
};
use unpopped_vocab::ElementKind;
use vulkane_wgsl_emit::Wgsl;

const MARKER_434C: &str = "declined: possible exposure, board 434 C";

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn c(v: i64, t: ElementKind) -> ScalarExpr {
    ScalarExpr::ConstInt(v as _, t)
}

/// `out[tid] = collective(a[tid])` over one 32-thread block with a stated subgroup size; `t` is the element type of both buffers and of
/// the locals, `result_ty` the type of the result local (U32 for a vote).
fn kernel_with(
    t: ElementKind,
    kind: CollectiveKind,
    scope: Scope,
    result_ty: ElementKind,
    subgroup: Option<(u32, u32)>,
) -> KernelFn {
    let buf = |b| Place::Logical {
        buffer: b,
        coords: vec![tid()],
    };
    let mut k = KernelFn::new(
        "coll",
        t,
        vec![
            Param::buffer("a", t, BufferRole::Read, Some(1)),
            Param::buffer("out", t, BufferRole::Write, Some(1)),
        ],
        vec![LocalDecl::scalar(t), LocalDecl::scalar(result_ty)],
        Block(vec![
            Stmt::Load {
                local: 0,
                place: buf(0),
            },
            Stmt::Collective {
                kind,
                value: ScalarExpr::Local(0),
                result: 1,
                scope,
            },
            Stmt::Store {
                place: buf(1),
                value: ScalarExpr::Local(1),
            },
        ]),
    );
    let mut launch = LaunchShape::new([32, 1, 1], [1, 1, 1]);
    if let Some((lo, hi)) = subgroup {
        launch = launch.with_subgroup(lo, hi);
    }
    k = k.with_launch(launch);
    k
}

fn kernel(t: ElementKind, kind: CollectiveKind) -> KernelFn {
    kernel_with(t, kind, Scope::Subgroup, t, Some((32, 32)))
}

fn source(k: &KernelFn) -> String {
    match Wgsl.lower_kernel(k) {
        Ok(g) => g.source,
        Err(e) => panic!("expected WGSL for {:?}, got {e:?}", k.name),
    }
}

fn refusal(k: &KernelFn) -> String {
    match Wgsl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected a typed UnsupportedKernel, got {other:?}"),
    }
}

/// Parse, validate and lower to SPIR-V in naga (the validity oracle; test only).
fn naga_accepts(src: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("{e:?}"))?;
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|e| format!("spv: {e:?}"))?;
    Ok(())
}

const U: ElementKind = ElementKind::U32;
const I: ElementKind = ElementKind::I32;

/// Every ruled kind beside the text it must produce, for an unsigned kernel.
fn ruled_unsigned() -> Vec<(&'static str, CollectiveKind, ElementKind, &'static str)> {
    use CollectiveKind as C;
    vec![
        (
            "Reduce Sum",
            C::Reduce(ReduceOp::Sum),
            U,
            "l1 = subgroupAdd(l0);",
        ),
        (
            "Reduce Prod",
            C::Reduce(ReduceOp::Prod),
            U,
            "l1 = subgroupMul(l0);",
        ),
        (
            "Reduce Max",
            C::Reduce(ReduceOp::Max),
            U,
            "l1 = subgroupMax(l0);",
        ),
        (
            "Reduce Min",
            C::Reduce(ReduceOp::Min),
            U,
            "l1 = subgroupMin(l0);",
        ),
        (
            "Reduce And",
            C::Reduce(ReduceOp::And),
            U,
            "l1 = subgroupAnd(l0);",
        ),
        (
            "Reduce Or",
            C::Reduce(ReduceOp::Or),
            U,
            "l1 = subgroupOr(l0);",
        ),
        (
            "Reduce Xor",
            C::Reduce(ReduceOp::Xor),
            U,
            "l1 = subgroupXor(l0);",
        ),
        (
            "Vote All",
            C::Vote(VoteKind::All),
            U,
            "l1 = select(0u, 1u, subgroupAll((l0 != 0u)));",
        ),
        (
            "Vote Any",
            C::Vote(VoteKind::Any),
            U,
            "l1 = select(0u, 1u, subgroupAny((l0 != 0u)));",
        ),
        (
            "Vote Ballot",
            C::Vote(VoteKind::Ballot),
            U,
            "l1 = subgroupBallot((l0 != 0u)).x;",
        ),
        (
            "Broadcast const 3",
            C::Broadcast { lane: c(3, U) },
            U,
            "l1 = subgroupBroadcast(l0, 3u);",
        ),
        (
            "Shuffle Idx (tid)",
            C::Shuffle {
                mode: ShuffleMode::Idx,
                operand: tid(),
            },
            U,
            "l1 = subgroupShuffle(l0, (lid.x % sg_size));",
        ),
        (
            "Shuffle Up 1",
            C::Shuffle {
                mode: ShuffleMode::Up,
                operand: c(1, U),
            },
            U,
            "l1 = select(l0, subgroupShuffleUp(l0, 1u), sg_id >= 1u);",
        ),
        (
            "Shuffle Down 2",
            C::Shuffle {
                mode: ShuffleMode::Down,
                operand: c(2, U),
            },
            U,
            "l1 = select(l0, subgroupShuffleDown(l0, 2u), 2u < (sg_size - sg_id));",
        ),
        (
            "Shuffle Xor 5",
            C::Shuffle {
                mode: ShuffleMode::Xor,
                operand: c(5, U),
            },
            U,
            "l1 = select(l0, subgroupShuffleXor(l0, 5u), (sg_id ^ 5u) < sg_size);",
        ),
    ]
}

#[test]
fn each_ruled_kind_lowers_to_its_subgroup_builtin() {
    for (label, kind, result_ty, want) in ruled_unsigned() {
        let k = kernel_with(U, kind, Scope::Subgroup, result_ty, Some((32, 32)));
        let src = source(&k);
        assert!(src.contains(want), "{label}: expected `{want}` in:\n{src}");
        assert!(
            src.contains("enable subgroups;")
                && src.contains("@builtin(subgroup_invocation_id) sg_id: u32")
                && src.contains("@builtin(subgroup_size) sg_size: u32"),
            "{label}: the enable directive and the built-ins are missing:\n{src}"
        );
    }
}

#[test]
fn a_signed_kernel_spells_the_idx_modulo_and_every_reduction_in_i32() {
    use CollectiveKind as C;
    let src = source(&kernel(I, C::Reduce(ReduceOp::Sum)));
    assert!(src.contains("l1 = subgroupAdd(l0);"), "{src}");
    let idx = source(&kernel(
        I,
        C::Shuffle {
            mode: ShuffleMode::Idx,
            operand: tid(),
        },
    ));
    // Euclidean modulo, as the oracle (`rem_euclid`) reads lane `operand mod n`.
    assert!(
        idx.contains(
            "subgroupShuffle(l0, u32(((i32(lid.x) % i32(sg_size)) + i32(sg_size)) % i32(sg_size)))"
        ),
        "{idx}"
    );
    // An I32 kernel's locals are I32, so a vote's U32 result has no local. The kernel is checker-valid (the result stays in a U32 local, the
    // I32 value is what is stored), so the refusal below comes from the walker or this dialect, by name, not from a type error.
    let mut k = kernel_with(
        I,
        C::Vote(VoteKind::All),
        Scope::Subgroup,
        U,
        Some((32, 32)),
    );
    if let Some(Stmt::Store { value, .. }) = k.body.0.last_mut() {
        *value = ScalarExpr::Local(0);
    }
    check_kernel(&k).expect("the signed vote kernel is checker-valid");
    let why = refusal(&k);
    // The WALKER refuses first (its locals check: every local of an integer kernel has the compute dtype); this dialect's own signed-vote
    // message is defensive and not reachable through `lower_kernel`. So a vote lowers in a `U32` kernel only.
    assert!(
        why.contains("is not a scalar of the compute dtype"),
        "a signed kernel's vote must be declined (the vote result is U32), got: {why}"
    );
}

#[test]
fn a_kernel_without_a_collective_is_unchanged_and_has_no_enable_directive() {
    // The control for the prologue: nothing about subgroups appears unless a collective is used.
    let t = U;
    let k = KernelFn::new(
        "plain",
        t,
        vec![Param::buffer("out", t, BufferRole::Write, Some(1))],
        vec![LocalDecl::scalar(t)],
        Block(vec![Stmt::Store {
            place: Place::Logical {
                buffer: 0,
                coords: vec![tid()],
            },
            value: c(1, t),
        }]),
    )
    .with_launch(LaunchShape::new([32, 1, 1], [1, 1, 1]).with_subgroup(32, 32));
    let src = source(&k);
    assert!(
        !src.contains("enable subgroups") && !src.contains("sg_id") && !src.contains("sg_size"),
        "{src}"
    );
    // And naga accepts it: the control that the validity oracle works on this kind of module.
    assert!(naga_accepts(&src).is_ok(), "{src}");
}

#[test]
fn naga_cannot_validate_subgroup_wgsl_yet_so_the_text_is_pinned_and_not_validated() {
    // MEASURED with naga 30.0.2 (the newest on crates.io, 2026-11-26): its WGSL front end rejects `enable subgroups;` with
    // "the `subgroups` enable-extension is not yet supported". So no module in this file is validated by a WGSL compiler here: the text is
    // pinned, and the subgroup built-ins are UNMEASURED against naga and Tint. This test fails the day naga accepts the extension,
    // which is the day the validation half of every test above can be added.
    let src = source(&kernel(U, CollectiveKind::Reduce(ReduceOp::Sum)));
    let e = naga_accepts(&src).expect_err("naga 30.0.2 rejects the subgroups enable-extension");
    assert!(
        e.contains("subgroups") && e.contains("not yet supported"),
        "naga's refusal changed: {e}"
    );
}

#[test]
fn what_stays_declined_is_refused_by_name_with_the_434c_marker() {
    use CollectiveKind as C;
    let mut cases: Vec<(String, KernelFn)> = Vec::new();
    for op in [
        ReduceOp::Sum,
        ReduceOp::Prod,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::And,
        ReduceOp::Or,
        ReduceOp::Xor,
    ] {
        for exclusive in [false, true] {
            cases.push((
                format!("Scan {op:?} exclusive={exclusive} (subgroup scope)"),
                kernel(U, C::Scan { op, exclusive }),
            ));
        }
    }
    // Every ruled kind, but over a Block: declined.
    for (label, kind, result_ty, _) in ruled_unsigned() {
        cases.push((
            format!("{label} over a Block"),
            kernel_with(U, kind, Scope::Block, result_ty, Some((32, 32))),
        ));
    }
    cases.push((
        "Reduce Mean".to_string(),
        kernel(U, C::Reduce(ReduceOp::Mean)),
    ));
    cases.push((
        "ReduceWith (a user combine)".to_string(),
        kernel(U, C::ReduceWith { combine: 0 }),
    ));
    cases.push((
        "ScanWith (a user combine)".to_string(),
        kernel(U, C::ScanWith { combine: 0 }),
    ));
    assert!(
        cases.len() >= 14 + 15 + 3,
        "the table is complete: {}",
        cases.len()
    );
    for (label, k) in &cases {
        let detail = refusal(k);
        assert!(
            detail.contains("Stmt::Collective") && detail.contains(MARKER_434C),
            "{label}: must be refused by name with the 434 C marker, got: {detail}"
        );
    }
}

#[test]
fn a_collective_tuple_is_refused_by_name() {
    let Some(unpopped::fixtures::Fixture::Module { module, entry }) = fixture("CollectiveTuple")
    else {
        panic!("the CollectiveTuple fixture is a module");
    };
    let k = module.fns[entry as usize].clone();
    let detail = refusal(&k);
    assert!(
        detail.contains("Stmt::CollectiveTuple") && detail.contains(MARKER_434C),
        "{detail}"
    );
}

#[test]
fn the_unlifted_by_name_decline_still_refuses_every_collective() {
    // The control for the lift: the walker's own by-name decline, asked WITHOUT the lift, still names a ruled kind. A dialect that does
    // not say `subgroup_collectives()` is unchanged (this is what GLSL and SPIR-V keep).
    let k = kernel(U, CollectiveKind::Reduce(ReduceOp::Sum));
    let e = vulkane_kernel_text::declined_by_name("glsl", &k).expect("refused without the lift");
    let LowerError::UnsupportedKernel { detail } = e else {
        panic!("{e:?}");
    };
    assert!(
        detail.contains("Stmt::Collective") && detail.contains(MARKER_434C),
        "{detail}"
    );
    assert!(
        vulkane_kernel_text::declined_by_name_with(true, "wgsl", &k).is_none(),
        "with the lift the ruled kind is not refused by name"
    );
}

#[test]
fn the_checkers_divergent_placements_never_reach_the_dialect() {
    // unpopped 0.60 `divergent_fixtures()`: a subgroup collective under lane-divergent control, and after a lane-divergent Return.
    // The checker refuses both (`check_kernel` is the walker's first step), before any text is written. "Acts on active threads; the
    // checker guard admits only uniform placements."
    let fixtures = divergent_fixtures();
    assert_eq!(fixtures.len(), 2, "two refused kernels");
    for (name, k, expected) in fixtures {
        assert_eq!(
            check_kernel(&k).err(),
            Some(expected.clone()),
            "{name}: the checker's own refusal"
        );
        let detail = refusal(&k);
        assert!(
            detail.contains("check_kernel refused") && detail.contains(&format!("{expected:?}")),
            "{name}: the emitter must surface the checker's refusal, got: {detail}"
        );
    }
}

#[test]
fn shuffle_up_down_xor_need_a_constant_and_broadcast_a_constant_below_the_stated_size() {
    use CollectiveKind as C;
    for mode in [ShuffleMode::Up, ShuffleMode::Down, ShuffleMode::Xor] {
        // A non-constant delta or mask: declined by name (WGSL requires it uniform; a constant is what we can prove).
        let why = refusal(&kernel(
            U,
            C::Shuffle {
                mode,
                operand: tid(),
            },
        ));
        assert!(why.contains("CONSTANT"), "{mode:?}: {why}");
        // A constant above 127 is a WGSL shader-creation error.
        let why = refusal(&kernel(
            U,
            C::Shuffle {
                mode,
                operand: c(128, U),
            },
        ));
        assert!(why.contains("127"), "{mode:?}: {why}");
    }
    // Broadcast: a lane that is not a constant, a lane at or above the stated minimum size, no stated size.
    assert!(refusal(&kernel(U, C::Broadcast { lane: tid() })).contains("constant"));
    assert!(refusal(&kernel(U, C::Broadcast { lane: c(32, U) })).contains("32"));
    let why = refusal(&kernel_with(
        U,
        C::Broadcast { lane: c(3, U) },
        Scope::Subgroup,
        U,
        None,
    ));
    assert!(
        why.contains("constant below the launch's stated subgroup size"),
        "{why}"
    );
}

#[test]
fn ballot_needs_a_stated_subgroup_size_of_at_most_32() {
    let ballot = CollectiveKind::Vote(VoteKind::Ballot);
    for sg in [None, Some((32, 64)), Some((64, 64))] {
        let why = refusal(&kernel_with(U, ballot.clone(), Scope::Subgroup, U, sg));
        assert!(why.contains("at most 32"), "{sg:?}: {why}");
    }
    assert!(
        source(&kernel_with(U, ballot, Scope::Subgroup, U, Some((16, 32))))
            .contains("subgroupBallot")
    );
}
