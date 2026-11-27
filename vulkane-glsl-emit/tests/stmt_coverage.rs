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
