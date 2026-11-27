# `UnaryOp::{Popc, Clz, Brev}` (unpopped 0.61.0): what each target says (facts, not a lowering)

**Status: facts only.** Nothing in this file is lowered. unpopped 0.61.0 is BREAKING (`UnaryOp` gains `Popc`, `Clz`, `Brev`, so every exhaustive
`UnaryOp` match in this workspace breaks); vulkane cannot take it until `fishstringer-slang-emit` publishes on `unpopped ^0.61` (0.10.1 requires
`^0.60.1`: two copies of `unpopped-vocab` otherwise). This file is read before that take PR so the decline rows or the lowerings quote text, not memory.

**Sources and method.** Every quote below is verbatim from a document downloaded on 2026-11-27 into a scratch directory and read with Python
(`-I`); a prefix of each file's sha256 is given. A cell marked **UNMEASURED** states what the text does not say or what no device has shown.

| document | URL | bytes | sha256 (first 16) |
|---|---|---|---|
| unpopped 0.61.0 crate | https://static.crates.io/crates/unpopped/unpopped-0.61.0.crate | read as source | (crates.io checksum not re-checked here) |
| WGSL (gpuweb `wgsl/index.bs`, main) | https://raw.githubusercontent.com/gpuweb/gpuweb/main/wgsl/index.bs | 928,881 | 7d5d7a0e197847ec |
| GLSL 4.60 | https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html | 923,931 | c5cb5528d71d1d3f |
| SPIR-V unified1 | https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html | 2,489,980 | 494da5daf1e0e6c1 |
| GLSL.std.450 | https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html | 184,172 | d728f559d1d8f7e4 |
| Metal Shading Language 4.1 | https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf | text-extracted 2026-11-26 | (see `docs/design/msl-simdgroup-matrix.md` for the method) |

## 1. The IR rule (the oracle), from unpopped 0.61.0

`src/ir.rs`: *"Population count: the number of set bits of the operand's `w`-bit two's-complement pattern (`w` is the operand type's width, so `-1` at `I8` is 8, not 64). Integer operands only; the result has the **operand's type** ... Source: PTX ISA `popc.b32`/`popc.b64`; CUDA `__popc`/`__popcll`. At 8 and 16 bits the pattern is the type's own width (not the promoted `int`), which is the IR's definition; no source this lane read states it."* / *"Count of leading zero bits of the operand's `w`-bit pattern; `0` gives `w`. ... A signed operand counts its pattern, so a negative value gives 0."* / *"Bit reversal of the operand's `w`-bit pattern: bit `i` moves to bit `w - 1 - i`. ... (a signed operand is read and returned as its pattern)."* `src/oracle/kernel_eval.rs` `int_bit_op`: `Popc => pat.count_ones()`, `Clz => pat.leading_zeros() - (128 - w)`, `Brev => (pat.reverse_bits() >> (128 - w))`, the result wrapped back into the type. The checker refuses them by name on a float or in a plan body (`IntegerOnlyOperand`).

So the three rules to match are: **`Popc` counts set bits of the `w`-bit pattern; `Clz` gives `w` for zero and 0 for a negative signed value (pattern based); `Brev` reverses within `w` bits; each returns the operand's own type.**

## 2. Per target

**MSL** (6.2 integer functions, Table 6.2). `T clz(T x)`: *"Returns the number of leading 0-bits in x, starting at the most significant bit position. If x is 0, returns the size in bits of the type of x or component type of x, if x is a vector"*. `T popcount(T x)`: *"Returns the number of nonzero bits in x."* `T reverse_bits(T x)` (*"All OS: Metal 2.1 and later."*): *"Returns the reversal of the bits of x. The bit numbered n of the result is taken from bit (bits – 1) – n of x, where bits is the total number of bits used to represent x."* `T` is *"one of the scalar or vector integer types"*. Matches the IR rule at every width the type has, including `clz(0) = bits`. **UNMEASURED:** a signed negative operand through `clz` (the text says "x", not a pattern), and the 8- and 16-bit and 64-bit forms on the device (hosted paravirtual probe, not Apple hardware).

**WGSL** (`countOneBits`, `countLeadingZeros`, `reverseBits`; `T` is `[INTEGRAL]`). `countOneBits`: *"The number of 1 bits in the representation of `e`."* `countLeadingZeros`: *"The number of consecutive 0 bits starting from the most significant bit of `e`, when `T` is a scalar type."* `reverseBits`: *"Reverses the bits in `e`: The bit at position `k` of the result equals the bit at position `31 -k` of `e`."* WGSL has `i32` and `u32` only here (the `31 - k` is the 32-bit rule), so the 8, 16 and 64-bit forms are already declined. **`countLeadingZeros(0)` is not stated separately**: by the definition (consecutive 0 bits from the most significant bit) a 32-bit zero gives 32, **UNMEASURED**. A signed negative operand is **UNMEASURED**. naga 30.0.2 may or may not accept these; check before a device test.

**GLSL 4.60.** `genIType bitCount(genIType value)` and `genIType bitCount(genUType value)`: *"Returns the number of one bits in the binary representation of value."* **The result type is `genIType` even for an unsigned argument**, so the IR's operand-typed result needs a cast (`uint(bitCount(x))`). `bitfieldReverse`: *"Reverses the bits of value. The bit numbered n of the result will be taken from bit (bits - 1) - n of value, where bits is the total number of bits used to represent value."* (`genUType bitfieldReverse(genUType)`, same type back.) `findMSB(value)`: *"Returns the bit number of the most significant bit in the binary representation of value. For positive integers, the result will be the bit number of the most significant one bit. For negative integers, the result will be the bit number of the most significant zero bit. For a value of zero or negative one, -1 will be returned."* **There is no GLSL `clz`.** `clz(x) = 31 - findMSB(uint(x))` for a 32-bit pattern: for zero, `31 - (-1) = 32` (equal to the IR's `w` with no select) and a negative signed `x` is read as its pattern through `uint(x)`, which matches the IR (0). **UNMEASURED:** all of it on a device; 8, 16 and 64-bit operands (the text read is the 32-bit built-ins).

**SPIR-V.** `OpBitCount`: *"Count the number of set bits in an object. ... Result Type must be a scalar or vector of integer type. The components must be wide enough to hold the unsigned Width of Base as an unsigned value. ... The result is the unsigned value that is the number of bits in Base that are 1."* `OpBitReverse`: *"Reverse the bits in an object. ... The bit-number n of the result is taken from bit-number Width - 1 - n of Base, where Width is the OpTypeInt operand of the Result Type. Capability: Shader, BitInstructions."* `GLSL.std.450` `FindUMsb` (75): *"Results in the bit number of the most-significant 1-bit in the binary representation of Value. If Value is 0, the result has all bits set (e.g., -1 if interpreted as signed). ... This instruction is currently limited to 32-bit width components."* (`FindSMsb`: *"For a Value of 0 or -1, the result has all bits set"*.) So `clz(x) = 31 - FindUMsb(x)` with 32-bit wrapping arithmetic gives `31 - 0xFFFFFFFF = 32` for zero. **UNMEASURED:** on the Lavapipe and local-device oracles; the width limit means 8, 16 and 64-bit forms are declined, not approximated.

**HLSL.** The kernel dialect reaches HLSL through `fishstringer-slang-emit` (plain HLSL entry pending); `countbits`, `firstbithigh` and `reversebits` are not read here. **Not read.**

## 3. Patent wording (relayed; this lane has not read the source section)

From pm-kernels' task (2026-11-27), to be used if a statement is made: **`Popc` and `Brev`: "outside as read"; `Clz`: "no patent found in N queries: not a clearance; no packed/vector clz"** (the source is named "patents-2 section 8"; this lane did not find that document and does not know N, so a quote of N waits for it). Nothing here is a clearance, there is no attorney, and findings are not advice.

## 4. What the take PR owes (rows, none started)

1. A trial compile at unpopped ~0.61 lists the exhaustive `UnaryOp` matches (13 files name `UnaryOp` today; the enum name is the positive control).
2. For each emitter, per op: **decline by name** with the emitter's own gate asserted (not the checker's) and a mutant that removes the decline, OR lower where the text above supports it, with a device or oracle differential that includes `0`, `-1`, `i32::MIN`, a single set bit at each end, and the signed-negative `Clz` cases.
3. MSL device cases on the hosted probe (paravirtual device, not Apple hardware); GLSL and SPIR-V on the local Vulkan device under `gpu-run`; WGSL through naga to SPIR-V if naga accepts them.
4. A BREAKING fragment (pre-1.0: the minor number); `cargo tree -d` must show one `unpopped-vocab`.
