// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 9f: a statement with no lowering says why (the NVIDIA-specific ones carry their reason), instead of the
//! generic "not lowered" text.

use unpopped::backend::{Backend, LowerError};
use vulkane_msl_emit::Msl;
use vulkane_test_support::stmt_kernel;

#[test]
fn an_nvidia_specific_statement_is_refused_by_name_with_its_reason() {
    for (name, stmt, phrase) in stmt_kernel::nvidia_unit_statements() {
        match Msl.lower_kernel(&stmt_kernel::kernel_with(stmt)) {
            Err(LowerError::UnsupportedKernel { detail }) => {
                assert!(
                    detail.contains(&format!("Stmt::{name}"))
                        && detail.contains("NVIDIA-specific")
                        && detail.contains(phrase),
                    "{name}: {detail}"
                );
            }
            other => panic!("{name}: wanted an UnsupportedKernel, got {other:?}"),
        }
    }
}
