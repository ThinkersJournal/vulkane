// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 9f: a statement with no lowering says why (the NVIDIA-specific ones carry their reason), instead of the
//! generic "not lowered" text.

use unpopped::backend::{Backend, LowerError};
use vulkane_msl_emit::Msl;
use vulkane_test_support::stmt_kernel;

#[test]
fn an_nvidia_specific_statement_is_refused_by_name_with_its_reason() {
    for (name, stmt, phrase) in stmt_kernel::nvidia_unit_statements() {
        // The one-statement kernel passes the kernel checker, so the refusal below is the emitter's own classification.
        let k = stmt_kernel::kernel_with(stmt.clone());
        assert!(
            unpopped::kernel::check_kernel(&k).is_ok(),
            "{name}: {:?}",
            unpopped::kernel::check_kernel(&k)
        );
        match Msl.lower_kernel(&k) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(
                    !detail.contains("check_kernel refused")
                        && detail.contains(&format!("Stmt::{name}"))
                        && detail.contains("NVIDIA-specific")
                        && detail.contains(phrase),
                    "{name}: {detail}"
                );
            }
            other => panic!("{name}: wanted an UnsupportedKernel, got {other:?}"),
        }
    }
}

#[test]
fn a_byte_addressed_place_is_refused_by_name_not_lowered_to_a_wrong_address() {
    for (name, k) in vulkane_test_support::stmt_kernel::raw_place_kernels() {
        unpopped::kernel::check_kernel(&k)
            .unwrap_or_else(|e| panic!("{name}: the probe kernel must be checker-valid: {e:?}"));
        match Msl.lower_kernel(&k) {
            Err(
                unpopped::backend::LowerError::UnsupportedKernel { detail }
                | unpopped::backend::LowerError::UnsupportedOp { detail }
                | unpopped::backend::LowerError::UnsupportedDtype { detail, .. },
            ) => {
                assert!(
                    detail.contains(name),
                    "{name}: the refusal must name the place: {detail}"
                );
                assert!(
                    !detail.contains("check_kernel refused"),
                    "{name}: the refusal must be the emitter's own: {detail}"
                );
            }
            other => panic!("{name}: wanted a typed refusal, got {other:?}"),
        }
    }
}

/// The tile statements and the `Tile` space of unpopped 0.62: a checker-valid kernel with a tile allocation is refused BY NAME by the
/// emitter's own gate (the allocation arm), not by the checker and not by the generic "not a shared array" text.
#[test]
fn the_tile_statements_and_a_tile_space_are_refused_by_name() {
    for name in ["TileLoad", "TileMma", "TileFill", "TileStore"] {
        let unpopped::fixtures::Fixture::Kernel(k) =
            unpopped::fixtures::fixture(name).expect("a fixture of unpopped 0.62")
        else {
            panic!("{name}: a kernel fixture");
        };
        unpopped::kernel::check_kernel(&k)
            .unwrap_or_else(|e| panic!("{name}: the fixture must be checker-valid: {e:?}"));
        match Msl.lower_kernel(&k) {
            Err(
                unpopped::backend::LowerError::UnsupportedKernel { detail }
                | unpopped::backend::LowerError::UnsupportedOp { detail }
                | unpopped::backend::LowerError::UnsupportedDtype { detail, .. },
            ) => {
                assert!(
                    detail.contains("Tile")
                        && !detail.contains("check_kernel refused")
                        && !detail.contains("checker refuses"),
                    "{name}: the emitter's own refusal must name the Tile space: {detail}"
                );
            }
            other => panic!("{name}: wanted a typed refusal, got {other:?}"),
        }
    }
}
