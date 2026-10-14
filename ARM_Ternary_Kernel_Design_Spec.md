# ARM Ternary Kernel Design Spec

## Mapping FairyFuse (x86 AVX-512) to Apple Silicon

**Date:** 2026-10-13  
**Source:** FairyFuse (Zuo et al., arXiv 2604.20913v1) — multiplication-free LLM inference via fused ternary kernels  
**Target:** Apple Silicon M5 Pro (ARM, 48GB unified memory)

---

## 1. What FairyFuse Does (x86)

- Ternary weights in {-1, 0, +1} replace FP multiplications with conditional add/subtract/no-op
- Fuses 8 real-valued sub-GEMVs of widely-linear layers into a single AVX-512 loop
- Uses AVX-512 masked additions/subtractions — zero floating-point multiplications
- 16x weight compression (vs FP16) shifts roofline from memory-bound toward compute-bound on bandwidth-limited CPUs
- Result: 29.6x kernel speedup, 32.4 tok/s on Intel Xeon 8558P
- Near-lossless: WikiText-2 ppl 5.52 vs 5.47 FP16

**Key constraint:** AVX-512 is 512-bit wide with native masking. This does not exist on ARM.

---

## 2. Apple Silicon Compute Targets

### 2a. ARM Neon (CPU path)
- 128-bit SIMD (4x narrower than AVX-512)
- No native masking — use `vbslq` (bitwise select) for conditional add/subtract
- Available on all Apple Silicon, no special setup
- Best for: CPU-fallback path, small batch inference

### 2b. AMX (Apple Matrix Coprocessor)
- Undocumented but accessible via Accelerate framework (BNNS, vDSP)
- Designed for matrix multiply — not obviously suited to ternary add/sub
- Risk: ternary doesn't map cleanly to AMX's multiply-accumulate datapath
- Verdict: probably not the right target; AMX wants dense FP, not sparse ternary

### 2c. Metal GPU (recommended primary path)
- M5 Pro: 20-core GPU, unified memory (no PCIe transfer bottleneck)
- Metal Shading Language supports subgroup operations, SIMD-group matrix functions
- Ternary GEMV maps to: load packed ternary weights → expand to sign/mask → conditional add/sub in FP16/FP32 accumulators
- Unified memory means weight streaming bandwidth is shared CPU/GPU — ~400 GB/s class
- Best for: primary inference path on M5

---

## 3. Ternary Weight Packing

### Layout
Each weight is {-1, 0, +1}. Pack as 2 bits per weight:
- `00` = 0 (skip)
- `01` = +1 (add)
- `10` = -1 (subtract)
- `11` = unused (or second zero encoding)

4 weights per byte. 16 weights per 32-bit word. 64 weights per 128-bit Neon register.

### Memory math (30B model)
- FP16: 60 GB
- 4-bit: ~16 GB
- Ternary (2-bit): ~7.5 GB
- Effective: less, because zeros are skippable (structured sparsity)

### Block structure
Group weights into blocks of 64 (one Neon register worth). Store as:
- 16 bytes of packed ternary data (64 weights × 2 bits)
- Per-block FP16 scale factor (2 bytes)
- Optional: sparsity bitmap for fast zero-skip (8 bytes per block of 64)

---

## 4. Neon Kernel Design (CPU Path)

### Core loop (pseudo-Neon intrinsics)
```c
// Process 64 weights per iteration
// w_packed: 128-bit register with 64 ternary weights (2 bits each)
// x: input vector (FP16, 64 elements)
// acc: FP32 accumulator

// Expand ternary to sign and mask
uint8x16_t w0 = vget_low_u8(w_packed);   // first 32 weights
uint8x16_t w1 = vget_high_u8(w_packed);  // next 32 weights

// For each byte (4 weights):
//   sign = (bits == 0b10) ? -1 : +1
//   mask = (bits != 0b00)  (non-zero = active)

// Conditional add/subtract using vbsl:
//   acc = vbslq_f32(mask, acc +/- x, acc)
```

### Key differences from AVX-512
| AVX-512 (FairyFuse) | ARM Neon (this spec) |
|---|---|
| 512-bit (32 FP32 per op) | 128-bit (4 FP32 per op) |
| Native `vmaskmov` / masked add | `vbslq` bitwise select |
| Single loop covers 8 sub-GEMVs | Need 4x more iterations for same throughput |
| Fused across sub-GEMVs | Same fusion, narrower |

### Expected throughput
Neon is 4x narrower, but M5 has multiple Neon-capable cores. With 8 performance cores each running Neon:
- Theoretical: ~1/4 the per-core throughput of AVX-512
- Practical: M5 efficiency cores also contribute; unified memory reduces stalls
- Estimate: 8-12 tok/s on CPU-only path (vs FairyFuse 32.4 tok.s on Xeon)

---

## 5. Metal Kernel Design (GPU Path — Recommended)

### Why Metal over Neon
- M5 GPU has ~20 cores vs 512-bit AVX-512's single-thread advantage
- Unified memory: no weight copy to GPU, zero transfer overhead
- Metal subgroup ops allow efficient parallel reduction
- Ternary decode is trivially parallelizable

### Kernel sketch (Metal Shading Language)
```metal
// Each threadgroup processes one output row
// Threads cooperatively decode ternary weights and accumulate

kernel void ternary_gemv(
    device const uint* packed_weights [[buffer(0)]],  // 2-bit packed
    device const half* input          [[buffer(1)]],  // FP16 activations
    device half* output               [[buffer(2)]],
    device const half* scales          [[buffer(3)]],  // per-block scale
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Each thread handles 64 weights (32 bytes packed)
    // Decode: (bits == 1) → +input, (bits == 2) → -input, else 0
    // Accumulate in FP32, apply scale, write FP16 output
}
```

### Sparsity exploitation
- Precompute sparsity bitmap per block
- Threads skip zero-blocks entirely (no memory fetch for all-zero blocks)
- Typical ternary LLM: 30-50% zeros → proportional bandwidth savings

### Expected throughput
- M5 Pro GPU memory bandwidth: ~400 GB/s (unified)
- Ternary 30B: 7.5GB weights → full weight sweep = ~19ms
- Decode-bound at ~50-80 tok/s estimated (vs 26 tok/s for 4-bit MLX baseline)
- This is the path that beats MLX 4-bit

---

## 6. Roofline Analysis (M5 Pro)

### FairyFuse x86 finding
16x compression shifts memory-bound GEMV toward compute-bound on bandwidth-limited CPUs.

### M5 Pro adaptation
- Unified memory bandwidth (~400 GB/s) is shared between CPU and GPU
- 4-bit 30B: 16GB → 40ms per full sweep at peak bandwidth
- Ternary 30B: 7.5GB → 19ms per full sweep at peak bandwidth
- **2.1x bandwidth reduction** → proportional speedup in memory-bound decode
- Compute overhead of ternary decode (bit manipulation) is negligible vs memory savings

### Bottleneck
Decode is memory-bound, not compute-bound, on M5. Ternary wins by halving memory traffic vs 4-bit.

---

## 7. Implementation Phases

### Phase 1: Weight converter
- Input: FP16 or 4-bit MLX safetensors
- Output: packed ternary format (2-bit + scales + sparsity bitmap)
- Ternary quantization: threshold-based (calibrate on WikiText-2 or similar)
- Validate: perplexity within 2% of FP16 baseline

### Phase 2: Metal GEMV kernel
- Single kernel: ternary_gemv (described above)
- Benchmark vs MLX 4-bit baseline on M5 Pro
- Target: 1.5x speedup on decode-bound workloads

### Phase 3: Neon fallback kernel
- CPU-only path for systems without Metal (or when GPU is busy)
- Lower priority — Metal is the primary target on Apple Silicon

### Phase 4: Integration
- Wrap in MLX-compatible interface
- Fused across attention/MLP sub-GEMVs (matching FairyFuse's fusion strategy)
- End-to-end benchmark: WikiText-2 ppl + tok/s

---

## 8. Risks and Open Questions

1. **Ternary quantization quality on 30B+ models** — FairyFuse shows near-lossless on their test model; needs validation on Muse Glimmer / Qwen architectures
2. **Metal subgroup reduction efficiency** — ternary decode adds instruction overhead; net win depends on memory savings outweighing decode cost
3. **Sparsity variance** — if the model has fewer zeros than expected, bandwidth savings shrink
4. **AMX path** — deliberately deprioritized; revisit only if Metal underperforms
5. **No existing ARM ternary kernels** — this is greenfield; expect iteration

---

## 9. Decision Matrix

| Path | Effort | Expected Speedup | Risk |
|---|---|---|---|
| Metal ternary GEMV | Medium | 1.5-2x vs 4-bit MLX | Low-Medium |
| Neon ternary GEMV | Medium | 0.5x vs Metal (CPU fallback) | Low |
| AMX ternary | High | Unknown (poor fit) | High |
| Stay on 4-bit MLX | Zero | Baseline | None |

**Recommendation:** Phase 1 + Phase 2 (converter + Metal kernel). If Metal hits 1.5x, proceed to integration. If not, the decode overhead ate the bandwidth savings and 4-bit remains optimal.

---

*Spec mapped from FairyFuse (Zuo et al., 2026) to Apple Silicon ARM/Metal. All M5 Pro performance figures are estimates — validate on hardware.*
