// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests for `Backend::lower_kernel` (ROADMAP rows 8a and 8b, first slice): a producer/consumer
//! through workgroup memory, with and without the barrier, and the typed declines.

use unpopped::backend::{Backend, GeneratedKernel, LowerError};
use unpopped::exec::{ExecId, Extent, Scope};
use unpopped::ir::ScalarExpr as E;
use unpopped::kernel::{BarrierKind, Block, KernelFn, LocalDecl, MemoryOrder, Param, Stmt};
use unpopped_vocab::ElementKind::U32;

use crate::SpirV;
use crate::tests::assert_valid;

const N: u32 = 64;

pub(crate) use vulkane_test_support::reverse_kernel;

/// The block-reverse kernel over `N` threads with a static shared array.
fn reverse(with_barrier: bool) -> KernelFn {
    reverse_kernel::reverse(N, with_barrier, reverse_kernel::SharedLen::Static)
}

fn tid() -> E {
    E::Exec(ExecId::ThreadIdx(0))
}

fn c(v: i64) -> E {
    E::ConstInt(v, U32)
}

fn lower(k: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    SpirV.lower_kernel(k)
}

fn count(src: &str, needle: &str) -> usize {
    src.lines().filter(|l| l.contains(needle)).count()
}

#[test]
fn the_reverse_kernel_validates_and_has_the_workgroup_shape() {
    let g = lower(&reverse(true)).expect("lowers");
    assert_valid(&g);
    assert_eq!(count(&g.source, "OpControlBarrier"), 1, "{}", g.source);
    assert!(g.source.contains("Workgroup"), "{}", g.source);
    assert!(
        g.source.contains(&format!("LocalSize {N} 1 1")),
        "{}",
        g.source
    );
    // A `Read` buffer is `NonWritable`, the `Write` one is not.
    assert_eq!(count(&g.source, "NonWritable"), 1, "{}", g.source);
}

/// Control: the instruction check sees the barrier, so its absence in the other module is a
/// real absence (the same query finds the one in the barriered module).
#[test]
fn dropping_the_barrier_drops_exactly_the_control_barrier() {
    let with = lower(&reverse(true)).expect("lowers");
    let without = lower(&reverse(false)).expect("lowers");
    assert_valid(&without);
    assert_eq!(count(&with.source, "OpControlBarrier"), 1);
    assert_eq!(count(&without.source, "OpControlBarrier"), 0);
    assert_ne!(with.words(), without.words());
}

#[test]
fn the_barrier_orders_workgroup_and_buffer_memory_at_workgroup_scope() {
    let g = lower(&reverse(true)).expect("lowers");
    let line = g
        .source
        .lines()
        .find(|l| l.contains("OpControlBarrier"))
        .unwrap();
    let ids: Vec<&str> = line
        .split_whitespace()
        .filter(|w| w.starts_with('%'))
        .collect();
    let konst = |id: &str| -> u32 {
        let def = g
            .source
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{id} = ")))
            .unwrap();
        def.split_whitespace().last().unwrap().parse().unwrap()
    };
    assert_eq!(konst(ids[0]), 2, "execution scope Workgroup");
    assert_eq!(konst(ids[1]), 2, "memory scope Workgroup");
    // AcquireRelease (0x8) | UniformMemory (0x40) | WorkgroupMemory (0x100)
    assert_eq!(konst(ids[2]), 0x148, "semantics");
}

#[test]
fn a_fence_validates_and_seqcst_collapses_to_acquire_release() {
    for (order, bits) in [
        (MemoryOrder::Acquire, 0x2 | 0x140),
        (MemoryOrder::Release, 0x4 | 0x140),
        (MemoryOrder::AcqRel, 0x8 | 0x140),
        (MemoryOrder::SeqCst, 0x8 | 0x140),
    ] {
        let mut k = reverse(true);
        k.body.0.insert(
            3,
            Stmt::Fence {
                scope: Scope::Block,
                order,
            },
        );
        let g = lower(&k).expect("lowers");
        assert_valid(&g);
        let line = g
            .source
            .lines()
            .find(|l| l.contains("OpMemoryBarrier"))
            .unwrap();
        let sem = line.split_whitespace().last().unwrap();
        let def = g
            .source
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{sem} = ")))
            .unwrap();
        assert_eq!(
            def.split_whitespace()
                .last()
                .unwrap()
                .parse::<u32>()
                .unwrap(),
            bits,
            "{order:?}"
        );
    }
}

fn declined(k: &KernelFn, needle: &str) {
    match lower(k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
            // STRICT (audit): the refusal is the emitter's own, not the kernel checker's, whose text can already name a construct.
            assert!(
                needle.contains("check_kernel")
                    || needle.contains("Divergent")
                    || !detail.contains("check_kernel refused"),
                "STRICT: the checker refused it, not the emitter: {detail}"
            );
        }
        other => panic!("wanted an UnsupportedKernel naming `{needle}`, got {other:?}"),
    }
}

#[test]
fn what_this_slice_does_not_lower_is_a_typed_decline_naming_it() {
    let mut f = reverse(true);
    f.compute = unpopped_vocab::ElementKind::Bf16;
    declined(&f, "kernels only");

    let mut d = reverse(true);
    d.locals.push(LocalDecl::scalar(U32));
    d.allocs[0].extent = Extent::Param(4);
    declined(&d, "check_kernel refused");

    let mut s = reverse(true);
    s.params.push(Param::scalar("n", U32, 4));
    s.locals.push(LocalDecl::scalar(U32));
    declined(&s, "is not lowered");

    let mut dev = reverse(true);
    dev.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Device,
            kind: BarrierKind::Full,
        },
    );
    declined(&dev, "Barrier");

    let mut named = reverse(true);
    named.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Block,
            kind: BarrierKind::Named { id: 1, count: N },
        },
    );
    declined(&named, "Barrier");

    // Row 8f: thread-block clusters have no SPIR-V form, so a cluster barrier on a launch that
    // declares clusters is a decline naming the barrier, never an approximation by a block barrier.
    let mut cluster = reverse(true);
    cluster.launch = cluster.launch.map(|mut l| {
        l.grid = [2, 1, 1];
        l.cluster = Some([2, 1, 1]);
        l
    });
    cluster.body.0.insert(
        3,
        Stmt::Barrier {
            scope: Scope::Cluster,
            kind: BarrierKind::Full,
        },
    );
    declined(&cluster, "Barrier");

    let mut grid_dep = reverse(true);
    grid_dep.body.0.push(Stmt::GridDepWait);
    declined(&grid_dep, "GridDepWait");
}

use flow_kernel::{bin, flow_kernel};
use vulkane_test_support::flow_kernel;

#[test]
fn nested_loops_with_break_continue_and_early_return_validate() {
    let g = lower(&flow_kernel()).expect("lowers");
    assert_valid(&g);
    assert_eq!(count(&g.source, "OpLoopMerge"), 2, "{}", g.source);
    assert_eq!(count(&g.source, "OpSelectionMerge"), 3, "{}", g.source);
    assert_eq!(count(&g.source, "OpReturn"), 2, "{}", g.source);
}

/// Control: the validator sees a structural fault. Swapping the merge and continue operands of the
/// first `OpLoopMerge` makes a module `spirv-val` must reject, while the intact one is accepted.
#[test]
fn a_seeded_invalid_merge_is_rejected_by_the_validator() {
    let g = lower(&flow_kernel()).expect("lowers");
    assert_valid(&g);
    let mut words = g.words().expect("words").to_vec();
    let at = (5..words.len())
        .find(|&n| words[n] & 0xFFFF == 246 && words[n] >> 16 == 4)
        .expect("an OpLoopMerge");
    words.swap(at + 1, at + 2);
    if let Some((ok, err)) = crate::tests::validate(&words) {
        assert!(
            !ok,
            "spirv-val accepted a loop whose merge and continue are swapped"
        );
        assert!(!err.is_empty());
    }
}

#[test]
fn a_barrier_under_thread_divergent_control_is_a_decline_with_the_checkers_answer() {
    use unpopped::ir::BinaryOp::CmpEq;
    let mut k = reverse(true);
    k.body.0.insert(
        3,
        Stmt::If {
            cond: bin(CmpEq, tid(), c(0)),
            then: Block(vec![Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            }]),
            els: Block(vec![]),
        },
    );
    declined(&k, "BarrierDivergent");
}

/// `reverse` with the shared array sized by a scalar parameter (local 4).
fn reverse_dynamic() -> KernelFn {
    let mut k = reverse(true);
    k.params.push(Param::scalar("n", U32, 4));
    k.locals.push(LocalDecl::scalar(U32));
    k.allocs[0].extent = Extent::Param(4);
    k
}

#[test]
fn a_dynamic_extent_is_a_specialization_constant_sized_array() {
    let g = lower(&reverse_dynamic()).expect("lowers");
    assert_valid(&g);
    assert_eq!(count(&g.source, "OpSpecConstant"), 1, "{}", g.source);
    // The scalar parameter is parameter 2: its SpecId is 2.
    assert!(g.source.contains("SpecId 2"), "{}", g.source);
    let spec = g
        .source
        .lines()
        .find(|l| l.contains("OpSpecConstant"))
        .and_then(|l| l.split_whitespace().next())
        .expect("a spec constant id");
    let array = g
        .source
        .lines()
        .find(|l| l.contains("OpTypeArray"))
        .expect("an array type");
    assert!(
        array.trim_end().ends_with(spec),
        "the array length is the spec constant: {array}"
    );
    // The static kernel has no spec constant at all (the control for the query above).
    let fixed = lower(&reverse(true)).expect("lowers");
    assert_eq!(count(&fixed.source, "OpSpecConstant"), 0);
}

use vulkane_test_support::private_kernel;

/// Row 8k: a `Private` array is a module-scope `Private` variable, not a `Workgroup` one.
#[test]
fn a_private_array_validates_and_is_private_storage() {
    let k = private_kernel::private_index(N, unpopped::exec::MemorySpace::Private);
    let g = lower(&k).expect("lowers");
    assert_valid(&g);
    assert_eq!(count(&g.source, "OpTypePointer Private"), 2, "{}", g.source);
    assert_eq!(
        count(&g.source, "OpTypePointer Workgroup"),
        0,
        "{}",
        g.source
    );
    // Control: the same queries find Workgroup storage in the shared-array kernel, and not here.
    let shared = lower(&reverse(true)).expect("lowers");
    assert_eq!(
        count(&shared.source, "OpTypePointer Workgroup"),
        2,
        "{}",
        shared.source
    );
    assert_eq!(
        count(&shared.source, "OpTypePointer Private"),
        0,
        "{}",
        shared.source
    );
}

/// Row 8k decline: a `Constant` allocation has no initialiser in the IR, so it is refused by name.
#[test]
fn a_constant_allocation_is_a_typed_decline() {
    // Read-only, so `check_kernel` admits it and the decline is this emitter's own.
    let k = private_kernel::read_only_constant(N);
    match lower(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("Constant space"), "{detail}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}

use vulkane_test_support::atomic_kernel;

/// Row 8d: the atomic kernel validates and holds each atomic instruction; the racy control swaps
/// exactly one `OpAtomicIAdd` for plain memory traffic (the same query finds both counts).
#[test]
fn the_atomic_kernel_validates_and_the_racy_control_has_one_fewer_atomic_add() {
    let atomic = lower(&atomic_kernel::atomic_kernel(false)).expect("lowers");
    assert_valid(&atomic);
    assert_eq!(
        count(&atomic.source, "OpAtomicIAdd"),
        2,
        "{}",
        atomic.source
    );
    assert_eq!(
        count(&atomic.source, "OpAtomicUMin"),
        1,
        "{}",
        atomic.source
    );
    assert_eq!(
        count(&atomic.source, "OpAtomicCompareExchange"),
        1,
        "{}",
        atomic.source
    );
    let racy = lower(&atomic_kernel::atomic_kernel(true)).expect("lowers");
    assert_valid(&racy);
    assert_eq!(count(&racy.source, "OpAtomicIAdd"), 1, "{}", racy.source);
}

/// Row 8d decline: no atomic on a `Private` array; `check_kernel` refuses it before this emitter does
/// (the emitter keeps its own guard for the day that check moves).
#[test]
fn an_atomic_on_a_private_array_is_a_typed_decline() {
    use unpopped::exec::{MemorySpace, Scope};
    use unpopped::kernel::{AtomicOp, Place};
    let mut k = private_kernel::private_index(N, MemorySpace::Private);
    k.body.0.insert(
        2,
        Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: Place::Alloc {
                alloc: 0,
                index: c(0),
            },
            value: c(1),
            old: None,
            order: MemoryOrder::Relaxed,
            scope: Scope::Block,
        },
    );
    match lower(&k) {
        Err(LowerError::UnsupportedKernel { detail }) => {
            assert!(detail.contains("AtomicUnsupported"), "{detail}");
        }
        other => panic!("expected a decline, got {other:?}"),
    }
}

/// Declines with no earlier test (an audit mutant that removed each passed every suite): a fence at a scope
/// SPIR-V has no form for, an execution id the backend does not spell, and a cast that is not a wrapping integer cast (the cast pin was rewritten on 2026-11-27 for the current decline text and an identity-cast shortcut).
/// Every fixture passes the kernel checker, so each refusal is this emitter's own.
#[test]
fn an_unspellable_fence_exec_id_and_cast_are_typed_declines() {
    use unpopped::ir::{CastSpec, RoundMode, Saturation};
    use unpopped::kernel::check_kernel;
    let valid = |k: &KernelFn| assert!(check_kernel(k).is_ok(), "{:?}", check_kernel(k));
    // The walker reports an expression it cannot spell as `UnsupportedOp`, a statement as `UnsupportedKernel`.
    let either = |k: &KernelFn, needle: &str| match lower(k) {
        Err(LowerError::UnsupportedKernel { detail } | LowerError::UnsupportedOp { detail }) => {
            assert!(detail.contains(needle), "wanted `{needle}` in `{detail}`");
            assert!(!detail.contains("check_kernel refused"), "{detail}");
        }
        other => panic!("wanted a decline naming `{needle}`, got {other:?}"),
    };
    // A fence at cluster scope (the launch declares a cluster): no SPIR-V form.
    let mut cluster = reverse(true);
    cluster.launch = cluster.launch.map(|mut l| {
        l.grid = [2, 1, 1];
        l.cluster = Some([2, 1, 1]);
        l
    });
    cluster.body.0.insert(
        3,
        Stmt::Fence {
            scope: Scope::Cluster,
            order: MemoryOrder::SeqCst,
        },
    );
    valid(&cluster);
    declined(&cluster, "a fence at scope");
    // (A relaxed fence is refused by the kernel checker itself, `SyncUnsupported`, so the emitter's own order decline is
    // unreachable by a checker-valid kernel: an audit mutant that removed it is an equivalent mutant, not an untested decline.)
    // An execution id with no SPIR-V spelling (a device rank), and a clamping cast, in place of the first stored value.
    for (label, make, needle) in [
        (
            "exec id",
            (|| E::Exec(ExecId::DeviceRank)) as fn() -> E,
            "is not lowered by the spirv backend",
        ),
        (
            "cast",
            // A clamping I32 -> U32 cast (an identity U32 -> U32 cast is spelled as the operand itself, so the clamp is only
            // reached across types): the inner wrapping U32 -> I32 cast is lowered, the outer clamping one is declined.
            (|| {
                E::Cast(
                    Box::new(E::Cast(
                        Box::new(c(1)),
                        CastSpec {
                            to: unpopped_vocab::ElementKind::I32,
                            round: RoundMode::NearestEven,
                            sat: Saturation::Wrap,
                        },
                    )),
                    CastSpec {
                        to: U32,
                        round: RoundMode::NearestEven,
                        sat: Saturation::Clamp,
                    },
                )
            }) as fn() -> E,
            "only a wrapping one is",
        ),
    ] {
        let mut k = reverse(false);
        let mut done = false;
        for st in &mut k.body.0 {
            if let Stmt::Store { value, .. } = st {
                if !done {
                    *value = make();
                    done = true;
                }
            }
        }
        assert!(done, "{label}: the reverse kernel has a Store");
        valid(&k);
        either(&k, needle);
    }
}

/// An atomic at cluster scope has no SPIR-V form and is declined by name (an audit mutant that approximated it by a
/// workgroup atomic passed every suite). The kernel passes the checker, so the refusal is the emitter's own. (The other
/// declines in `atomic.rs`, an atomic on a non-shared array, a memory order or an op outside the IR's current variants,
/// are unreachable by a checker-valid kernel: equivalent mutants, not untested declines.)
#[test]
fn an_atomic_at_cluster_scope_is_a_typed_decline() {
    use unpopped::kernel::{AtomicOp, Place, check_kernel};
    let mut k = reverse(true);
    k.launch = k.launch.map(|mut l| {
        l.grid = [2, 1, 1];
        l.cluster = Some([2, 1, 1]);
        l
    });
    k.body.0.insert(
        3,
        Stmt::AtomicRmw {
            op: AtomicOp::Add,
            place: Place::Alloc {
                alloc: 0,
                index: c(0),
            },
            value: c(1),
            old: None,
            order: MemoryOrder::Relaxed,
            scope: Scope::Cluster,
        },
    );
    assert!(check_kernel(&k).is_ok(), "{:?}", check_kernel(&k));
    declined(&k, "an atomic at scope");
}
