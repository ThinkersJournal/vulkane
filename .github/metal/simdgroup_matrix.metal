// SPDX-License-Identifier: MIT OR Apache-2.0
// Compile-only probe (never run): does Apple's compiler accept the SIMD-group matrix forms of MSL 4.1 section 2.4 and 6.8?
// Variants are chosen by -D macros from the workflow step. Source: the spec's own example (section 6.8.2) and the
// negative controls it implies (a single T for d, a, b and c; Cols and Rows are 8). Lifted by CireSnave 2026-11-26 (board 519 B)
// as his risk call; not a clearance; no attorney; the Intel/NVIDIA tile-instruction method and medium claims remain NEEDS MORE as read.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#if defined(T_HALF)
  typedef simdgroup_half8x8 M;
  typedef half E;
#elif defined(T_BFLOAT)
  typedef simdgroup_bfloat8x8 M;
  typedef bfloat E;
#else
  typedef simdgroup_float8x8 M;
  typedef float E;
#endif

#if defined(MIXED)
// half a, b with a float c and r: the spec's signature has ONE T, so this should be refused.
kernel void probe(device half *pA [[buffer(0)]], device half *pB [[buffer(1)]],
                  device float *pC [[buffer(2)]], device float *pR [[buffer(3)]]) {
  simdgroup_half8x8 a;
  simdgroup_half8x8 b;
  simdgroup_float8x8 c;
  simdgroup_float8x8 r;
  simdgroup_load(a, pA);
  simdgroup_load(b, pB);
  simdgroup_load(c, pC);
  simdgroup_multiply_accumulate(r, a, b, c);
  simdgroup_store(r, pR);
}
#elif defined(BIG16)
// 16x16: the spec says Cols and Rows are 8, so this should be refused.
kernel void probe(device float *pA [[buffer(0)]]) {
  simdgroup_matrix<float, 16, 16> m;
  simdgroup_load(m, pA);
}
#else
kernel void probe(device E *pA [[buffer(0)]], device E *pB [[buffer(1)]],
                  device E *pC [[buffer(2)]], device E *pR [[buffer(3)]]) {
  M a;
  M b;
  M c;
  M r;
  simdgroup_load(a, pA);
  simdgroup_load(b, pB);
  simdgroup_load(c, pC);
  simdgroup_multiply_accumulate(r, a, b, c);
  simdgroup_store(r, pR);
}
#endif
