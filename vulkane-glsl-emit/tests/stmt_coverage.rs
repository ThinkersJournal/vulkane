// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 9f: a statement with no lowering says why (the NVIDIA-specific ones carry their reason), instead of the
//! generic "not lowered" text.

use unpopped::backend::{Backend, LowerError};
use vulkane_glsl_emit::Glsl;
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
        match Glsl.lower_kernel(&k) {
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
    for (name, k) in stmt_kernel::raw_place_kernels() {
        unpopped::kernel::check_kernel(&k)
            .unwrap_or_else(|e| panic!("{name}: the probe kernel must be checker-valid: {e:?}"));
        match Glsl.lower_kernel(&k) {
            Err(
                LowerError::UnsupportedKernel { detail }
                | LowerError::UnsupportedOp { detail }
                | LowerError::UnsupportedDtype { detail, .. },
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
