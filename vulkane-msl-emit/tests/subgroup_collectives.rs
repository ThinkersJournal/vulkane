// SPDX-License-Identifier: MIT OR Apache-2.0
//! Board 506 "A" (narrowed; CireSnave 2026-11-26): the ruled subgroup collectives (reductions, shuffles, votes, broadcasts) lower to MSL
//! `simd_*` functions for integer kernels; scans, user combines, `CollectiveTuple` and block scope stay refused by name with the board
//! 434 C marker. Float kernels still refuse every collective by name (the float reduction order and NaN rules are not stated).
//!
//! Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call: "outside as read" is a reading of public documents, not a clearance; no
//! attorney; findings, not advice.
//!
//! What this file pins: the emitted TEXT of each ruled kind (the function beside its `CollectiveKind`; MSL 4.1 6.10.2, read from the
//! specification PDF on 2026-11-26), the signature (the lane and SIMD-group size parameters appear only when read), the by-name declines, and
//! the checker's divergent-placement refusals (from `unpopped::fixtures::divergent_fixtures()`, unpopped 0.60). What it does NOT pin:
//! behaviour on a device (the hosted Apple probe compiles and runs the same kernels; see `.github/workflows/metal-probe.yml`).

use unpopped::backend::{Backend, LowerError};
use unpopped::exec::{ExecId, LaunchShape, Scope};
use unpopped::fixtures::{Fixture, divergent_fixtures, fixture};
use unpopped::ir::{BinaryOp, ReduceOp, ScalarExpr};
use unpopped::kernel::{
    Block, BufferRole, CollectiveKind, KernelFn, LocalDecl, Param, Place, ShuffleMode, Stmt,
    VoteKind, check_kernel,
};
use unpopped_vocab::ElementKind;
use vulkane_msl_emit::Msl;

const MARKER_434C: &str = "declined: possible exposure, board 434 C";
const U: ElementKind = ElementKind::U32;
const I: ElementKind = ElementKind::I32;

fn tid() -> ScalarExpr {
    ScalarExpr::Exec(ExecId::ThreadIdx(0))
}

fn c(v: i64, t: ElementKind) -> ScalarExpr {
    ScalarExpr::ConstInt(v as _, t)
}

/// `out[tid] = collective(a[tid])` over one 32-thread block with a stated subgroup size; `t` is the element type of both buffers and of the
/// locals, `result_ty` the type of the result local (U32 for a vote).
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
    let k = KernelFn::new(
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
    k.with_launch(launch)
}

fn kernel(t: ElementKind, kind: CollectiveKind) -> KernelFn {
    kernel_with(t, kind, Scope::Subgroup, t, Some((32, 32)))
}

fn source(k: &KernelFn) -> String {
    match Msl.lower_kernel(k) {
        Ok(g) => g.source,
        Err(e) => panic!("expected MSL for {:?}, got {e:?}", k.name),
    }
}

fn refusal(k: &KernelFn) -> String {
    match Msl.lower_kernel(k) {
        Err(LowerError::UnsupportedKernel { detail }) => detail,
        other => panic!("expected a typed UnsupportedKernel, got {other:?}"),
    }
}

/// Every ruled kind beside the text it must produce, for an unsigned kernel.
fn ruled_unsigned() -> Vec<(&'static str, CollectiveKind, &'static str)> {
    use CollectiveKind as C;
    vec![
        ("Reduce Sum", C::Reduce(ReduceOp::Sum), "l1 = simd_sum(l0);"),
        (
            "Reduce Prod",
            C::Reduce(ReduceOp::Prod),
            "l1 = simd_product(l0);",
        ),
        ("Reduce Max", C::Reduce(ReduceOp::Max), "l1 = simd_max(l0);"),
        ("Reduce Min", C::Reduce(ReduceOp::Min), "l1 = simd_min(l0);"),
        ("Reduce And", C::Reduce(ReduceOp::And), "l1 = simd_and(l0);"),
        ("Reduce Or", C::Reduce(ReduceOp::Or), "l1 = simd_or(l0);"),
        ("Reduce Xor", C::Reduce(ReduceOp::Xor), "l1 = simd_xor(l0);"),
        (
            "Vote All",
            C::Vote(VoteKind::All),
            "l1 = (simd_all(l0 != 0u) ? 1u : 0u);",
        ),
        (
            "Vote Any",
            C::Vote(VoteKind::Any),
            "l1 = (simd_any(l0 != 0u) ? 1u : 0u);",
        ),
        (
            "Broadcast const 3",
            C::Broadcast { lane: c(3, U) },
            "l1 = simd_broadcast(l0, (ushort)3);",
        ),
        (
            "Shuffle Idx (tid)",
            C::Shuffle {
                mode: ShuffleMode::Idx,
                operand: tid(),
            },
            "l1 = simd_shuffle(l0, (ushort)((tid.x) % sgsize));",
        ),
        (
            "Shuffle Up 1",
            C::Shuffle {
                mode: ShuffleMode::Up,
                operand: c(1, U),
            },
            "l1 = simd_shuffle_up(l0, (ushort)1);",
        ),
        (
            "Shuffle Down 2",
            C::Shuffle {
                mode: ShuffleMode::Down,
                operand: c(2, U),
            },
            "l1 = simd_shuffle_down(l0, (ushort)2);",
        ),
        (
            "Shuffle Xor 5",
            C::Shuffle {
                mode: ShuffleMode::Xor,
                operand: c(5, U),
            },
            "l1 = select(l0, simd_shuffle_xor(l0, (ushort)5), ((lane ^ 5u) < sgsize));",
        ),
    ]
}

#[test]
fn each_ruled_kind_lowers_to_its_simd_function() {
    for (label, kind, want) in ruled_unsigned() {
        let src = source(&kernel(U, kind));
        assert!(src.contains(want), "{label}: expected `{want}` in:\n{src}");
    }
}

#[test]
fn the_lane_and_size_parameters_are_declared_only_when_the_body_reads_them() {
    use CollectiveKind as C;
    let lane = "uint lane [[thread_index_in_simdgroup]]";
    let size = "uint sgsize [[threads_per_simdgroup]]";
    // A plain reduction reads neither.
    let src = source(&kernel(U, C::Reduce(ReduceOp::Sum)));
    assert!(!src.contains(lane) && !src.contains(size), "{src}");
    // The xor shuffle reads both (its in-range select); the Idx shuffle reads the size (its modulo).
    let xor = source(&kernel(
        U,
        C::Shuffle {
            mode: ShuffleMode::Xor,
            operand: c(5, U),
        },
    ));
    assert!(xor.contains(lane) && xor.contains(size), "{xor}");
    let idx = source(&kernel(
        U,
        C::Shuffle {
            mode: ShuffleMode::Idx,
            operand: tid(),
        },
    ));
    assert!(!idx.contains(lane) && idx.contains(size), "{idx}");
    // Up and Down need neither (the specification's own edge rule is the IR's).
    for mode in [ShuffleMode::Up, ShuffleMode::Down] {
        let s = source(&kernel(
            U,
            C::Shuffle {
                mode,
                operand: c(1, U),
            },
        ));
        assert!(!s.contains(lane) && !s.contains(size), "{mode:?}: {s}");
    }
}

#[test]
fn a_signed_kernel_spells_the_reductions_and_the_idx_modulo_in_i32() {
    use CollectiveKind as C;
    let sum = source(&kernel(I, C::Reduce(ReduceOp::Sum)));
    assert!(sum.contains("l1 = simd_sum(l0);"), "{sum}");
    let idx = source(&kernel(
        I,
        C::Shuffle {
            mode: ShuffleMode::Idx,
            operand: tid(),
        },
    ));
    // Euclidean modulo (the oracle's `rem_euclid`): ((op % n) + n) % n in int, then a ushort lane.
    assert!(
        idx.contains("simd_shuffle(l0, (ushort)(((((int)tid.x)) % (int)sgsize) + (int)sgsize) % (int)sgsize))")
            || idx.contains("% (int)sgsize) + (int)sgsize) % (int)sgsize"),
        "{idx}"
    );
    // An I32 kernel's locals are I32, so a vote's U32 result has no local: the WALKER refuses first. The kernel is checker-valid.
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
    assert!(
        why.contains("is not a scalar of the compute dtype"),
        "{why}"
    );
}

#[test]
fn ballot_is_declined_until_the_simd_vote_conversion_is_measured() {
    let why = refusal(&kernel(U, CollectiveKind::Vote(VoteKind::Ballot)));
    assert!(
        why.contains("Ballot") && why.contains("simd_vote") && !why.contains(MARKER_434C),
        "{why}"
    );
}

/// A float kernel (F32) with `kind` over one subgroup of 32.
fn float_kernel(kind: CollectiveKind) -> KernelFn {
    let f = ElementKind::F32;
    kernel_with(f, kind, Scope::Subgroup, f, Some((32, 32)))
}

#[test]
fn an_f32_kernel_lowers_sum_max_and_min_with_the_nan_select_on_max_and_min() {
    // Board 506 A float ruling (pm-kernels 2026-11-27). MEASURED on the hosted macOS paravirtual device, NOT Apple hardware, in safe
    // math (vulkane#189, run 45972534688; fast-math not measured): `simd_sum` propagated NaN, `simd_max` and `simd_min` IGNORED it,
    // and KISS-OPS-6.11-0002 requires a float max or min to propagate NaN, so each carries an explicit select.
    let sum = source(&float_kernel(CollectiveKind::Reduce(ReduceOp::Sum)));
    assert!(sum.contains("l1 = simd_sum(l0);"), "{sum}");
    assert!(sum.contains("#pragma METAL fp math_mode(safe)"), "{sum}");
    for (op, f) in [(ReduceOp::Max, "simd_max"), (ReduceOp::Min, "simd_min")] {
        let s = source(&float_kernel(CollectiveKind::Reduce(op)));
        let want = format!("l1 = select({f}(l0), NAN, simd_any(isnan(l0)));");
        assert!(
            s.contains(&want),
            "{op:?}: expected `{want}` in:
{s}"
        );
    }
}

#[test]
fn an_f32_kernel_declines_everything_outside_the_float_ruling_by_name() {
    use CollectiveKind as C;
    // Prod is unmeasured (simd_product was not probed); the shuffles, votes and broadcast are outside the ruling: each is a typed
    // refusal from the float dialect that says so (no 434 C marker: these are not patent-declined kinds).
    let mut dialect_declined: Vec<(String, CollectiveKind)> = vec![
        ("Reduce Prod".into(), C::Reduce(ReduceOp::Prod)),
        ("Vote All".into(), C::Vote(VoteKind::All)),
        ("Broadcast".into(), C::Broadcast { lane: c(3, U) }),
    ];
    for mode in [
        ShuffleMode::Idx,
        ShuffleMode::Up,
        ShuffleMode::Down,
        ShuffleMode::Xor,
    ] {
        dialect_declined.push((
            format!("Shuffle {mode:?}"),
            C::Shuffle {
                mode,
                operand: c(1, U),
            },
        ));
    }
    for (label, kind) in &dialect_declined {
        let result_ty = if matches!(kind, C::Vote(_)) {
            U
        } else {
            ElementKind::F32
        };
        let mut k = kernel_with(
            ElementKind::F32,
            kind.clone(),
            Scope::Subgroup,
            result_ty,
            Some((32, 32)),
        );
        if matches!(kind, C::Vote(_)) {
            // A vote's result is a U32 local, which cannot be stored to the F32 buffer: store the loaded value instead, so the kernel
            // is checker-valid and the refusal under test is the dialect's.
            if let Some(Stmt::Store { value, .. }) = k.body.0.last_mut() {
                *value = ScalarExpr::Local(0);
            }
        }
        check_kernel(&k)
            .unwrap_or_else(|e| panic!("{label}: the probe kernel must be checker-valid: {e:?}"));
        let why = refusal(&k);
        assert!(
            why.contains("only Sum, Max and Min of an F32 kernel are lowered"),
            "{label}: {why}"
        );
    }
    // Scans, Mean and a user combine are not in the ruled set at all: the walker's by-name decline with the 434 C marker.
    for kind in [
        C::Reduce(ReduceOp::Mean),
        C::Scan {
            op: ReduceOp::Sum,
            exclusive: false,
        },
        C::ReduceWith { combine: 0 },
    ] {
        let why = refusal(&float_kernel(kind.clone()));
        assert!(
            why.contains("Stmt::Collective") && why.contains(MARKER_434C),
            "{kind:?}: {why}"
        );
    }
    // Half and bfloat kernels: the float dialect does not lift the subgroup collectives for them (unmeasured), so the by-name decline
    // applies even to a Sum.
    for t in [ElementKind::F16, ElementKind::Bf16] {
        let k = kernel_with(
            t,
            CollectiveKind::Reduce(ReduceOp::Sum),
            Scope::Subgroup,
            t,
            Some((32, 32)),
        );
        let why = refusal(&k);
        assert!(
            why.contains("Stmt::Collective") && why.contains(MARKER_434C),
            "{t:?}: {why}"
        );
    }
    // Block scope stays declined for a float kernel too.
    let k = kernel_with(
        ElementKind::F32,
        CollectiveKind::Reduce(ReduceOp::Sum),
        Scope::Block,
        ElementKind::F32,
        Some((32, 32)),
    );
    let why = refusal(&k);
    assert!(
        why.contains("Stmt::Collective") && why.contains(MARKER_434C),
        "{why}"
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
    // Every ruled kind, but over a Block: declined (Ballot included: the by-name decline comes first).
    let mut over_block = ruled_unsigned()
        .into_iter()
        .map(|(l, k, _)| (l, k))
        .collect::<Vec<_>>();
    over_block.push(("Vote Ballot", C::Vote(VoteKind::Ballot)));
    for (label, kind) in over_block {
        cases.push((
            format!("{label} over a Block"),
            kernel_with(U, kind, Scope::Block, U, Some((32, 32))),
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
    let Some(Fixture::Module { module, entry }) = fixture("CollectiveTuple") else {
        panic!("the CollectiveTuple fixture is a module");
    };
    let detail = refusal(&module.fns[entry as usize].clone());
    assert!(
        detail.contains("Stmt::CollectiveTuple") && detail.contains(MARKER_434C),
        "{detail}"
    );
}

#[test]
fn the_unlifted_by_name_decline_still_refuses_every_collective() {
    // The control for the lift: asked WITHOUT the lift, the walker's by-name decline still names a ruled kind (what GLSL and SPIR-V
    // keep); with the lift it does not.
    let k = kernel(U, CollectiveKind::Reduce(ReduceOp::Sum));
    let e = vulkane_kernel_text::declined_by_name("glsl", &k).expect("refused without the lift");
    let LowerError::UnsupportedKernel { detail } = e else {
        panic!("{e:?}");
    };
    assert!(
        detail.contains("Stmt::Collective") && detail.contains(MARKER_434C),
        "{detail}"
    );
    assert!(vulkane_kernel_text::declined_by_name_with(true, "msl", &k).is_none());
}

#[test]
fn the_checkers_divergent_placements_never_reach_the_dialect() {
    // unpopped 0.60 `divergent_fixtures()`: a subgroup collective under lane-divergent control, and after a lane-divergent Return. The
    // checker refuses both before any text is written. "Acts on active threads; the checker guard admits only uniform placements."
    let fixtures = divergent_fixtures();
    assert_eq!(fixtures.len(), 2, "two refused kernels");
    for (name, k, expected) in &fixtures {
        assert_eq!(
            check_kernel(k).err(),
            Some(expected.clone()),
            "{name}: the checker's own refusal"
        );
    }
    // The fixtures are F32 kernels, which MSL sends to the FLOAT dialect (it does not lift collectives: refused by name first), so the
    // emitter's surface of the checker's refusal is pinned on INTEGER analogs of the same two placements.
    let odd = || ScalarExpr::Binary(BinaryOp::BitAnd, Box::new(tid()), Box::new(c(1, U)));
    let sum = || Stmt::Collective {
        kind: CollectiveKind::Reduce(ReduceOp::Sum),
        value: ScalarExpr::Local(0),
        result: 1,
        scope: Scope::Subgroup,
    };
    let load = Stmt::Load {
        local: 0,
        place: Place::Logical {
            buffer: 0,
            coords: vec![tid()],
        },
    };
    let store = Stmt::Store {
        place: Place::Logical {
            buffer: 1,
            coords: vec![tid()],
        },
        value: ScalarExpr::Local(1),
    };
    let when_odd = |then: Vec<Stmt>| Stmt::If {
        cond: odd(),
        then: Block(then),
        els: Block(vec![]),
    };
    let build = |body: Vec<Stmt>| {
        let mut k = kernel(U, CollectiveKind::Reduce(ReduceOp::Sum));
        k.body = Block(body);
        k
    };
    let analogs = [
        (
            "CollectiveUnderDivergentIf",
            // (the result local lives inside the `If`, as in the fixture, so nothing reads it after)
            build(vec![load.clone(), when_odd(vec![sum()])]),
            "BarrierDivergent",
        ),
        (
            "CollectiveAfterDivergentReturn",
            build(vec![load, when_odd(vec![Stmt::Return]), sum(), store]),
            "CollectiveAfterReturn",
        ),
    ];
    for (name, k, expected) in analogs {
        assert_eq!(
            format!(
                "{:?}",
                check_kernel(&k).expect_err("the checker refuses it")
            ),
            expected,
            "{name}: the checker's own refusal of the integer analog"
        );
        let detail = refusal(&k);
        assert!(
            detail.contains("check_kernel refused") && detail.contains(expected),
            "{name}: the emitter must surface the checker's refusal, got: {detail}"
        );
    }
}

#[test]
fn shuffle_up_down_xor_need_a_constant_below_the_stated_size_and_broadcast_a_constant() {
    use CollectiveKind as C;
    for mode in [ShuffleMode::Up, ShuffleMode::Down, ShuffleMode::Xor] {
        // A non-constant delta or mask: declined (MSL: the same for every thread; a constant is what we can prove).
        let why = refusal(&kernel(
            U,
            C::Shuffle {
                mode,
                operand: tid(),
            },
        ));
        assert!(why.contains("CONSTANT"), "{mode:?}: {why}");
        // At or above the stated minimum subgroup size: the specification states no result beyond the SIMD-group.
        let why = refusal(&kernel(
            U,
            C::Shuffle {
                mode,
                operand: c(32, U),
            },
        ));
        assert!(
            why.contains("outside the launch's stated minimum subgroup size 32"),
            "{mode:?}: {why}"
        );
        // No stated subgroup size: nothing bounds it.
        let why = refusal(&kernel_with(
            U,
            C::Shuffle {
                mode,
                operand: c(1, U),
            },
            Scope::Subgroup,
            U,
            None,
        ));
        assert!(why.contains("states a subgroup size"), "{mode:?}: {why}");
        // 31 is the last accepted delta of a 32-lane subgroup.
        assert!(
            source(&kernel(
                U,
                C::Shuffle {
                    mode,
                    operand: c(31, U)
                }
            ))
            .contains("(ushort)31")
        );
    }
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
