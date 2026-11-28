// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statements this emitter refuses **by name** because lowering them to a hardware operation on non-NVIDIA
//! hardware risks infringing NVIDIA's patents. Twelve names: the four of board 434 C plus `CollectiveTuple` (CireSnave 2026-11-13) and the
//! seven of its extension (CireSnave 2026-11-16).
//!
//! - Board 434 C, *"emitters may lower AsyncWait/CopyAsync/AsyncCommit/Collective for NVIDIA devices ... and decline by
//!   name for non-NVIDIA devices"* (CireSnave 2026-11-13, "434: C."): `CopyAsync`, `AsyncCommit`, `AsyncWait` (NVIDIA
//!   US11294713B2 and US12118382B2, an API that blocks a thread's instructions around an asynchronous shared-memory
//!   copy; `unpopped` `docs/PATENT-SWEEP.md` hit 22) and `Collective`, and `CollectiveTuple`. `CollectiveTuple` is the tuple form of `Collective`: 434 C read as covering it; not named in the
//!   ruling's list (pm-kernels 2026-11-17, PM no objection; rockfish, bonefish and fishstringer decline the family).
//!
//! CireSnave 2026-11-16, verbatim: *"For that patent, we add a note about the potential for infringement if
//! non-Nvidia-specific emitters lower those operations to hardware ops and we ensure our projects' emitters for
//! other manufacturers' hardware know about that and we consider ourselves not infringing."* and, for the second
//! patent, *"I think we treat that patent like the others that are Nvidia-only. Add a note and specifically notify
//! our non-Nvidia-emitter builders about the potential for infringement and not to."*
//!
//! Board 434 C extended (2026-11-16):
//! - NVIDIA US11803380B2 (a synchronization barrier with an arrival counter, a phase and a hardware copy operator):
//!   `MbarInit`, `MbarArrive`, `MbarCompleteTx`, `MbarWait`, `CopyBulk` (completes on an mbarrier) and
//!   `Tcgen05Commit` (arrives on an mbarrier).
//! - NVIDIA US12204897B2 and its continuation US12710961B2 (a wait on a count of pending matrix-multiply
//!   operations): `WgmmaWait`.
//!
//! **Do not lower any of these to a hardware operation in this crate.** The walk below matches every [`Stmt`]
//! variant explicitly and has no wildcard arm, so a new statement fails to compile here until someone decides which
//! side of this list it is on.

use unpopped::backend::LowerError;
use unpopped::exec::Scope;
use unpopped::ir::ReduceOp;
use unpopped::kernel::{Block, CollectiveKind, KernelFn, Stmt};

/// The reason text the four board 434 C refusals carry, so a caller and a test can recognise it.
pub const DECLINED_434C: &str = "declined: possible exposure, board 434 C";
/// The reason text the seven board 434 C extended refusals carry.
pub const DECLINED_434C_EXT: &str = "declined: possible exposure, board 434 C extended";

/// The name of the first declined statement in `block` and the ruling text it carries, looking inside `If`, `For`
/// and `While` at any depth.
fn find(block: &Block, lift: bool) -> Option<Declined> {
    block.0.iter().find_map(|s| find_stmt(s, lift))
}

/// A declined statement's name and the ruling text its refusal carries.
type Declined = (&'static str, &'static str);

fn find_stmt(stmt: &Stmt, lift: bool) -> Option<Declined> {
    declined_434c(stmt, lift)
        .or_else(|| declined_extension(stmt))
        .or_else(|| nested(stmt, lift))
}

/// The five of board 434 C. The wildcard is fine here: exhaustiveness is enforced by [`nested`].
fn declined_434c(stmt: &Stmt, lift: bool) -> Option<Declined> {
    // Board 506 "A" (narrowed): a dialect that lowers subgroup collectives leaves the RULED kinds alone; every other `Collective`
    // (a scan, a user combine, a block scope) keeps the 434 C marker, and so does every `CollectiveTuple`.
    if let Stmt::Collective { kind, scope, .. } = stmt {
        if lift && collective_is_ruled(kind, *scope) {
            return None;
        }
    }
    let name = match stmt {
        Stmt::CopyAsync { .. } => "CopyAsync",
        Stmt::AsyncCommit => "AsyncCommit",
        Stmt::AsyncWait { .. } => "AsyncWait",
        Stmt::Collective { .. } => "Collective",
        Stmt::CollectiveTuple { .. } => "CollectiveTuple",
        _ => return None,
    };
    Some((name, DECLINED_434C))
}

/// The seven of the 2026-11-16 extension, and `CopyUnpack` (unpopped 0.58: its own doc says the `CopyBulk` patent note applies and
/// emitters for other manufacturers decline it).
fn declined_extension(stmt: &Stmt) -> Option<Declined> {
    let name = match stmt {
        Stmt::MbarInit { .. } => "MbarInit",
        Stmt::MbarArrive { .. } => "MbarArrive",
        Stmt::MbarCompleteTx { .. } => "MbarCompleteTx",
        Stmt::MbarWait { .. } => "MbarWait",
        Stmt::CopyBulk { .. } => "CopyBulk",
        Stmt::Tcgen05Commit { .. } => "Tcgen05Commit",
        Stmt::WgmmaWait { .. } => "WgmmaWait",
        Stmt::CopyUnpack { .. } => "CopyUnpack",
        _ => return None,
    };
    Some((name, DECLINED_434C_EXT))
}

/// Look inside `If`, `For` and `While`. **This match has no wildcard arm on purpose**: it names every [`Stmt`] variant,
/// so a new statement in `unpopped` fails to compile here until someone decides which side of the declined list it is
/// on. The declined ones are listed (and answered `None`: they were already found by the two functions above).
fn nested(stmt: &Stmt, lift: bool) -> Option<Declined> {
    match stmt {
        Stmt::If { then, els, .. } => find(then, lift).or_else(|| find(els, lift)),
        Stmt::For { body, .. } | Stmt::While { body, .. } => find(body, lift),
        // The declined statements (found above, and none holds a block).
        Stmt::CopyAsync { .. }
        | Stmt::AsyncCommit
        | Stmt::AsyncWait { .. }
        | Stmt::Collective { .. }
        | Stmt::CollectiveTuple { .. }
        | Stmt::MbarInit { .. }
        | Stmt::MbarArrive { .. }
        | Stmt::MbarCompleteTx { .. }
        | Stmt::MbarWait { .. }
        | Stmt::CopyBulk { .. }
        | Stmt::Tcgen05Commit { .. }
        | Stmt::WgmmaWait { .. }
        | Stmt::CopyUnpack { .. }
        // Every other statement is not declined by name here.
        | Stmt::Let { .. }
        | Stmt::Load { .. }
        | Stmt::Assign { .. }
        | Stmt::Store { .. }
        | Stmt::MaskedLoad { .. }
        | Stmt::MaskedStore { .. }
        | Stmt::Break
        | Stmt::Continue
        | Stmt::Return
        | Stmt::Barrier { .. }
        | Stmt::AtomicRmw { .. }
        | Stmt::AtomicCas { .. }
        | Stmt::MulticastLoadReduce { .. }
        | Stmt::GridDepWait
        | Stmt::GridDepLaunchDependents
        | Stmt::ProxyFence { .. }
        | Stmt::Fence { .. }
        | Stmt::Assume { .. }
        | Stmt::Mma { .. }
        | Stmt::TileLoad { .. }
        | Stmt::TileMma { .. }
        | Stmt::TileFill { .. }
        | Stmt::TileStore { .. }
        | Stmt::MatrixLoad { .. }
        | Stmt::MatrixStore { .. }
        | Stmt::Wgmma { .. }
        | Stmt::WgmmaCommit
        | Stmt::Tcgen05Mma { .. }
        | Stmt::Tcgen05Ld { .. }
        | Stmt::Tcgen05Alloc { .. }
        | Stmt::Tcgen05Dealloc { .. }
        | Stmt::Tcgen05RelinquishPermit
        | Stmt::Call { .. }
        | Stmt::CallTuple { .. }
        | Stmt::ExternCall { .. } => None,
    }
}

/// Whether a `Collective` of `kind` over `scope` is in the set CireSnave ruled liftable on non-NVIDIA targets (board 506 "A", narrowed,
/// 2026-11-26): subgroup-scope **reductions** (sum, product, max, min, and, or, xor), **shuffles**, **votes** and **broadcasts**.
/// Everything else stays declined by name with the 434 C marker: scans (inclusive or exclusive, any op), user combines
/// (`ReduceWith`, `ScanWith`), `CollectiveTuple`, and every `Block`-scope collective.
///
/// Lifted by CireSnave 2026-11-26 (board 506 A) as his risk call: "outside as read" is a reading of public documents, not a clearance;
/// no attorney; findings, not advice.
#[must_use]
pub fn collective_is_ruled(kind: &CollectiveKind, scope: Scope) -> bool {
    if scope != Scope::Subgroup {
        return false;
    }
    match kind {
        CollectiveKind::Reduce(op) => matches!(
            op,
            ReduceOp::Sum
                | ReduceOp::Prod
                | ReduceOp::Max
                | ReduceOp::Min
                | ReduceOp::And
                | ReduceOp::Or
                | ReduceOp::Xor
        ),
        CollectiveKind::Shuffle { .. }
        | CollectiveKind::Vote(_)
        | CollectiveKind::Broadcast { .. } => true,
        _ => false,
    }
}

/// The by-name refusal of the first declined statement in `kernel` (at any depth), or `None`.
///
/// `backend` is the name used in the text (`glsl`, `wgsl`). Every `Collective` is declined; see [`declined_by_name_with`] for a
/// target that lowers the ruled subgroup collectives.
#[must_use]
pub fn declined_by_name(backend: &str, kernel: &KernelFn) -> Option<LowerError> {
    declined_by_name_with(false, backend, kernel)
}

/// As [`declined_by_name`], for a target that **lowers the ruled subgroup collectives** (`lift_subgroup`: see
/// [`collective_is_ruled`]); the other collectives, and every other declined statement, are still refused by name.
#[must_use]
pub fn declined_by_name_with(
    lift_subgroup: bool,
    backend: &str,
    kernel: &KernelFn,
) -> Option<LowerError> {
    let (name, ruling) = find(&kernel.body, lift_subgroup)?;
    Some(LowerError::UnsupportedKernel {
        detail: format!(
            "backend `{backend}` refuses `Stmt::{name}` by name ({ruling}); it is not lowered to a hardware operation on non-NVIDIA hardware (kernel `{}`)",
            kernel.name
        ),
    })
}
