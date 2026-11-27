// SPDX-License-Identifier: MIT OR Apache-2.0
// Hand-written probe, not emitted: does Apple's compiler accept a 64-bit atomic at all, and under which flags? The first
// run (45751376095, -std=metal3.2) rejected every atomic_ulong operation; the CI job records what other flags give.
#include <metal_stdlib>
using namespace metal;

kernel void au(device atomic_ulong* a [[buffer(0)]], uint t [[thread_position_in_grid]]) {
    atomic_fetch_max_explicit(&a[0], (ulong)t, memory_order_relaxed);
}
