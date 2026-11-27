// SPDX-License-Identifier: MIT OR Apache-2.0
//! Where every [`Stmt`] variant stands in this repository's emitters (ROADMAP row 9f).

use unpopped::kernel::Stmt;

/// What the emitters of this repository do with a statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Lowered by the shared walker and by the SPIR-V emitter.
    Lowered,
    /// Not lowered yet, and a ROADMAP row says what it waits for (the text).
    Row(&'static str),
    /// Refused by name for possible patent exposure (board 434 C, and its extension): see `declined_by_name`.
    Patent,
    /// An NVIDIA-specific operation with no spelling in the target languages of this repository; the reason.
    Nvidia(&'static str),
}

/// The coverage of `stmt`. **This match has no wildcard arm on purpose**: it names every [`Stmt`] variant, so a new
/// statement in `unpopped` fails to compile here until someone decides which of the four it is.
#[must_use]
pub fn coverage(stmt: &Stmt) -> Coverage {
    match stmt {
        Stmt::Let { .. }
        | Stmt::Load { .. }
        | Stmt::Assign { .. }
        | Stmt::Store { .. }
        | Stmt::MaskedLoad { .. }
        | Stmt::MaskedStore { .. }
        | Stmt::If { .. }
        | Stmt::For { .. }
        | Stmt::While { .. }
        | Stmt::Break
        | Stmt::Continue
        | Stmt::Return
        | Stmt::Barrier { .. }
        | Stmt::Fence { .. }
        | Stmt::AtomicRmw { .. }
        | Stmt::AtomicCas { .. } => Coverage::Lowered,
        Stmt::Call { .. } | Stmt::CallTuple { .. } | Stmt::ExternCall { .. } => {
            Coverage::Row("ROADMAP row 8g: device function calls are not lowered yet")
        }
        Stmt::Mma { .. } | Stmt::MatrixLoad { .. } | Stmt::MatrixStore { .. } => Coverage::Row(
            "ROADMAP row 8j: matrix multiply-accumulate waits on a device with the extension (SPIR-V, GLSL) and on Unpopped's tile statements (Metal)",
        ),
        Stmt::TileLoad { .. }
        | Stmt::TileMma { .. }
        | Stmt::TileFill { .. }
        | Stmt::TileStore { .. } => Coverage::Row(
            "ROADMAP row 8j: the tile statements (unpopped 0.62) are declined by name until a device with the cooperative-matrix extension (SPIR-V, GLSL) or simdgroup matrices (Metal) can run them; no saturating variant and no transpose option (US12405770B2 hold)",
        ),
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
        | Stmt::CopyUnpack { .. } => Coverage::Patent,
        Stmt::Wgmma { .. } | Stmt::WgmmaCommit => Coverage::Nvidia(
            "a warpgroup-level asynchronous matrix multiply-accumulate (PTX `wgmma`, Hopper); none of SPIR-V, GLSL, HLSL, WGSL or MSL has a construct with its asynchronous warpgroup semantics",
        ),
        Stmt::Tcgen05Mma { .. }
        | Stmt::Tcgen05Ld { .. }
        | Stmt::Tcgen05Alloc { .. }
        | Stmt::Tcgen05Dealloc { .. }
        | Stmt::Tcgen05RelinquishPermit => Coverage::Nvidia(
            "a fifth-generation tensor-core operation on tensor memory (PTX `tcgen05`, Blackwell); no target language of this repository has tensor memory",
        ),
        Stmt::ProxyFence { .. } => Coverage::Nvidia(
            "an async-proxy fence (PTX `fence.proxy.async`, as unpopped documents `ProxyFence`): it orders generic-proxy writes against the asynchronous proxy that `CopyBulk` and `wgmma` use; no target language of this repository has memory proxies",
        ),
        Stmt::GridDepWait | Stmt::GridDepLaunchDependents => Coverage::Nvidia(
            "CUDA programmatic dependent launch (a kernel waiting on, or releasing, the next kernel of a stream); a Vulkan dispatch has no in-kernel dependency on another dispatch",
        ),
        Stmt::MulticastLoadReduce { .. } => Coverage::Nvidia(
            "an NVLink multicast load-reduce (PTX `multimem`) over several devices' memory; no target language of this repository addresses another device's memory",
        ),
    }
}

/// The sentence an emitter adds to its "not lowered" refusal of `stmt`: the row it waits for, the by-name ruling or
/// the NVIDIA-specific reason, or `None` for a statement the emitters lower (a refusal of one is about its operands).
#[must_use]
pub fn not_lowered_note(stmt: &Stmt) -> Option<String> {
    match coverage(stmt) {
        Coverage::Lowered => None,
        Coverage::Row(why) => Some(why.to_string()),
        Coverage::Patent => Some(
            "refused by name for possible patent exposure (board 434 C), see `declined_by_name`"
                .to_string(),
        ),
        Coverage::Nvidia(why) => Some(format!("NVIDIA-specific: {why}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use unpopped::exec::Scope;
    use unpopped::kernel::{BarrierKind, Stmt};

    #[test]
    fn the_cheap_to_build_variants_are_classified_as_documented() {
        assert_eq!(coverage(&Stmt::Break), Coverage::Lowered);
        assert_eq!(
            coverage(&Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full
            }),
            Coverage::Lowered
        );
        assert_eq!(coverage(&Stmt::AsyncCommit), Coverage::Patent);
        for s in [
            Stmt::WgmmaCommit,
            Stmt::GridDepWait,
            Stmt::GridDepLaunchDependents,
        ] {
            assert!(
                matches!(coverage(&s), Coverage::Nvidia(why) if !why.is_empty()),
                "{s:?}"
            );
        }
    }

    #[test]
    fn the_tile_statements_wait_on_row_8j_and_say_the_patent_hold() {
        use unpopped::ir::ScalarExpr;
        let fill = Stmt::TileFill {
            d: 0,
            value: ScalarExpr::ConstInt(1, unpopped_vocab::ElementKind::U32),
        };
        assert!(
            matches!(coverage(&fill), Coverage::Row(why) if why.contains("row 8j")),
            "{fill:?}"
        );
        let n = not_lowered_note(&fill).expect("a note");
        assert!(
            n.contains("US12405770B2") && n.contains("no transpose option"),
            "{n}"
        );
    }

    #[test]
    fn a_lowered_statement_has_no_note_and_the_others_say_why() {
        assert_eq!(not_lowered_note(&Stmt::Return), None);
        let n = not_lowered_note(&Stmt::GridDepWait).expect("a note");
        assert!(
            n.contains("NVIDIA-specific") && n.contains("dependent launch"),
            "{n}"
        );
        let p = not_lowered_note(&Stmt::AsyncCommit).expect("a note");
        assert!(p.contains("board 434 C"), "{p}");
    }
}
