# Metal Quantized GEMV Kernels

Fused dequant + matrix-vector kernels for OptiQ mixed-quantization (Muse Glimmer q4km-gs128).

## Files

- **`quant_gemv.metal`**: MSL kernels
  - `q4_gemv`: 4-bit weights, group_size 32 (configurable)
  - `q6_gemv`: 6-bit weights, group_size 128 (configurable)
- **`metal_quant.rs`**: Rust bindings via `metal` crate

## Quantization Format

From `core_lm/src/optiq.rs`:

- **Weights**: Packed U32, low bits first, tightly packed
  - 4-bit: 8 values per U32, never crosses word boundary
  - 6-bit: codes may cross U32 word boundaries
- **Scales**: F32 per group, shape `[rows, cols/group_size]`
- **Biases**: F32 per group (zero-point), shape `[rows, cols/group_size]`
- **Out_scale**: F32 per row, shape `[rows]` (optional)

Dequant: `w[i] = (q[i] * scale[i/group_size] + bias[i/group_size]) * out_scale[row]`

## Kernel Design

Each thread computes one output row: `out[row] = dot(dequant(W[row,:]), x[:])`

The dequantization is fused into the GEMV loop — weights are unpacked and dequantized
on-the-fly, never materialized as F32. This saves memory bandwidth (the bottleneck
for decode).

### Thread Mapping
- One thread per output row
- Grid size = `rows`
- Each thread loops over `cols`, accumulating the dot product

### Future Optimizations
- [ ] Use threadgroup memory for `x` vector (broadcast)
- [ ] Vectorize with `float4` for coalesced reads
- [ ] 2D threadgroup tiling for better occupancy on large matrices
- [ ] GEMM variant for prefill (matrix-matrix)

## Usage

```rust
use metal_quant::MetalQuantGemm;

let msl = std::fs::read_to_string("quant_gemv.metal")?;
let gemm = MetalQuantGemm::new(&msl)?;

// 4-bit GEMV
let out = gemm.q4_gemv(
    &packed_weights,  // &[u32]
    &scales_bytes,    // &[u8] (F32 LE)
    &biases_bytes,    // &[u8] (F32 LE)
    Some(&out_scale), // Option<&[f32]>
    &x,               // &[f32]
    rows, cols, 32,   // group_size=32 for 4-bit
)?;

// 6-bit GEMV
let out = gemm.q6_gemv(
    &packed_weights,
    &scales_bytes,
    &biases_bytes,
    Some(&out_scale),
    &x,
    rows, cols, 128,  // group_size=128 for 6-bit
)?;
```

## Verification

**Cannot be tested on Linux** (no Metal). Must be verified on macOS:

1. Compile: `xcrun -sdk macosx metal -c quant_gemv.metal -o quant_gemv.air`
2. Compare against CPU: run same GEMV via `optiq::dequantize` + naive matvec
3. Check bit-exact match (or document FP rounding differences)

## Status

- [x] MSL kernels written (q4_gemv, q6_gemv)
- [x] Rust bindings written
- [ ] Compiled on macOS
- [ ] Verified against CPU reference
- [ ] Integrated into glimmer_forward.rs
- [ ] Benchmarked (expect 10-50x speedup over CPU GEMV)

## Notes

- Scales/biases currently assume F32. F16/BF16 need conversion helpers.
- The `read_f32` helper does unaligned LE byte reads. Ensure buffers are properly aligned
  or use `memcpy` semantics.
- For 4-bit: `values_per_word = 8`, no cross-boundary handling needed.
- For 6-bit: cross-boundary handling via `if (shift + bits > 32)`.
