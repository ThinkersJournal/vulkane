// SPDX-License-Identifier: MIT OR Apache-2.0
// Hand-written probe, not emitted: does Metal accept a threadgroup array whose extent is a
// function constant? (the MSL 4.1 spec claim the emitter's "scalar params" blocker rests on; the CI
// job records whether this compiles, it does not gate on it.)
#include <metal_stdlib>
using namespace metal;

constant uint EXTENT [[function_constant(0)]];

kernel void fc_extent(device uint* out [[buffer(0)]],
                      uint t [[thread_position_in_threadgroup]]) {
    threadgroup uint tile[EXTENT];
    tile[t % EXTENT] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[t] = tile[(t + 1) % EXTENT];
}
