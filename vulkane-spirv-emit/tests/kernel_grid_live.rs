// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 8f on a live device: a 3-D launch at a non-cubic block (4x3x2) and grid (3x2x5)
//! shape, every thread writing its `ThreadIdx`/`BlockIdx` (packed) to the cell its `BlockDim` and
//! `GridDim` place it at (`tests/support/grid_kernel.rs`). Every cell is checked against the host
//! value; a cell still holding the sentinel is a thread that wrote elsewhere or never ran.
//!
//! Without a device the test follows `VULKANE_REQUIRE_DEVICE` like every other live test.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/gpu.rs"]
mod gpu;
#[path = "support/grid_kernel.rs"]
mod grid_kernel;

use grid_kernel::{BLOCK, GRID};
use unpopped::backend::Backend;
use vulkane_spirv_emit::SpirV;

#[test]
fn a_non_cubic_3d_launch_writes_every_cell_its_own_ids_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("kernel_grid") else {
        return; // skip already declared
    };
    let g = SpirV
        .lower_kernel(&grid_kernel::grid_kernel())
        .expect("the SPIR-V backend lowers the 3-D kernel");
    let words = g.words().expect("words").to_vec();
    let got = gpu.run_grid(&g.name, &words, grid_kernel::cells(), GRID);
    let (nx, ny) = ((BLOCK[0] * GRID[0]) as usize, (BLOCK[1] * GRID[1]) as usize);
    assert_eq!(got.len(), nx * ny * (BLOCK[2] * GRID[2]) as usize);
    for (i, &w) in got.iter().enumerate() {
        let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
        assert_eq!(
            w,
            grid_kernel::expected(x as u32, y as u32, z as u32),
            "cell ({x},{y},{z})"
        );
    }
}
