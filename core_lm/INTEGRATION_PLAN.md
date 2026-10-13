# INTEGRATION_PLAN.md — Geometric Memory Forward Pass

## Objective
Load Qwen3.6-35B-A3B-OptiQ-4bit weights and run a working forward pass in `core_lm`
using GeometricMemoryMap **instead of** traditional KV cache in the full-attention layers.

**Bar:** If it generates non-garbage text, the KV cache is dead.

---

## Architecture Reality (from config.json)

- Architecture: Qwen3_5MoeForConditionalGeneration
- Hidden size: 2048, Layers: 40
- Layer pattern: 3x linear_attention + 1x full_attention (repeating)
- Full-attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 (10 total)
- Linear-attention layers: 30 (Mamba-style SSM, recurrent state, no KV cache)
- Attention heads: 16, KV heads (GQA): 2, Head dim: 256
- Vocab size: 248320
- MoE experts: 256, top-8 per token. MoE intermediate: 512
- RoPE: mrope, partial_rotary_factor=0.25, theta=10M
- Quantization: OptiQ 4-bit affine (gs=64), some layers 8-bit

**Critical:** Only the 10 full_attention layers have KV cache.
The GeometricMemoryMap replaces KV cache in those 10 layers only.

---

## OptiQ Quantization Format

Each quantized tensor has 3 safetensors entries:
- `{name}.weight` — packed quantized integers
- `{name}.scales` — per-group scale (64-element groups)
- `{name}.biases` — per-group zero-point

Dequant: `w[i] = q[i] * scale[g] + bias[g]` where `g = i / 64`

---

## Phase 1: Infrastructure

### 1a. Safetensors parser (src/weights/safetensors.rs)
- Pure Rust: read 8-byte header length, parse JSON header, mmap tensor data
- Return: name -> (dtype, shape, data_offset)
- Test: parse index, verify tensor count matches

### 1b. OptiQ dequantizer (src/weights/optiq.rs)
- Unpack 4-bit (verify nibble order against reference) and 8-bit
- Apply per-group scale+bias -> f32
- Cache dequantized weights (22GB fits in 48GB unified)
- Test: dequantize known tensor, verify values

### 1c. Config loader (src/weights/config.rs)
- Parse config.json with serde_json AT LOAD TIME ONLY
- Extract dims, layer types, rope params, MoE config
- Runtime uses binary struct, never touches JSON
- Test: verify all 40 layer types parsed correctly

### 1d. Tokenizer (src/tokenizer/mod.rs)
- Parse tokenizer.json BPE merges and vocab (build-time)
- BPE encode/decode
- Test: round-trip "Hello world"

---

## Phase 2: Core Operations

### 2a. Tensor ops (src/ops/mod.rs)
- matvec (f32), rmsnorm, rope (mrope variant), softmax, silu
- All return Result, no panics
- Test: each op against known values

### 2b. Fixed-point bridge (src/geometry/bridge.rs)
- f32 key vectors -> LorentzVector (Q32_31)
- Document precision loss at conversion boundary
- Test: round-trip within tolerance

---

## Phase 3: Attention with Geometric Memory

### 3a. Full attention layer (src/model/attention.rs)
- Q/K/V projections, GQA (16Q/2KV heads), RoPE on Q/K
- GEOMETRIC KV: K -> LorentzVector, store in GeometricMemoryMap with V as payload
- Per query: retrieve top-k nearest K, attention over retrieved set only
- O projection
- Test: single-layer forward produces finite output

### 3b. Linear attention layer (src/model/linear_attn.rs)
- Mamba-style: in_proj_qkv/z/a/b, conv1d, SSM, out_proj
- Recurrent state (NOT geometric map — inherent recurrence)
- Test: state advances correctly

### 3c. MoE MLP (src/model/moe.rs)
- Router: linear -> softmax -> top-8
- Experts: gate_proj -> silu -> up_proj -> down_proj
- Shared expert always active, weighted sum
- Test: router selects 8, output finite

---

## Phase 4: Model Assembly

### 4a. Decoder layer (src/model/layer.rs)
- input_layernorm -> attention -> residual
- post_attention_layernorm -> MoE -> residual
- Dispatch on layer_types[i]

### 4b. Full model (src/model/mod.rs)
- embed_tokens -> 40 layers -> final norm -> lm_head -> logits
- Test: single token -> vocab_size logits

---

## Phase 5: Generation

### 5a. Generate loop (src/bin/generate.rs)
- cargo run --bin generate -- "prompt"
- Encode -> prefill (populate geometric maps) -> autoregressive decode
- Sampling: temperature, top-p. Stop on EOS (248044, 248046)

### 5b. Non-garbage validation (tests/generation_quality.rs)
- Prompt "The capital of France is" -> assert "Paris" in output
- Assert no token repeated >5x consecutively
- Assert output length > 10 tokens

---

## File Layout

```
src/
  weights/     — safetensors.rs, optiq.rs, config.rs, mod.rs
  tokenizer/     — mod.rs (BPE)
  ops/           — mod.rs (matvec, rmsnorm, rope, softmax, silu)
  geometry/      — bridge.rs + existing files
  cache/         — geometric.rs (CERTIFIED, DO NOT MODIFY)
  model/         — mod.rs, layer.rs, attention.rs, linear_attn.rs, moe.rs
  bin/           — generate.rs
```

## Dependencies to Add

```toml
memmap2 = "0.9"
byteorder = "1.5"
```

---

## Constraints Compliance

- No Python in runtime: pure Rust safetensors parser
- No JSON in runtime: serde_json only in config loader
- Q32.31 for geometric path: bridge.rs converts at map boundary
- No panics in hot path: all ops return Result
- Every claim tested: each phase has tests

---

## Risks

1. OptiQ nibble order must be verified against reference. Wrong = garbage.
2. Linear attention SSM discretization details need care.
3. MoE expert indexing (256 experts, fused switch_mlp weights).
4. Memory: 22GB weights + activations fits in 48GB unified.
5. Speed: pure Rust f32 matvec ~1-5 tok/s estimated. Proves correctness, not speed.

---

## Success Criteria

- [ ] "The capital of France is" -> output contains "Paris"
- [ ] No k_cache/v_cache tensors in codebase
- [ ] GeometricMemoryMap is only cross-token state in full_attention layers
- [ ] All tests pass: cargo test
- [ ] No panics on 3 different prompts
