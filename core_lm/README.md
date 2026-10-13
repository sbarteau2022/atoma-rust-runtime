# core_lm: Native Execution Plane

The `core_lm` crate is the numerical authority for the Monster Hybrid Server. It implements a high-throughput, "weightless" LLM runtime that replaces the traditional KV cache with a **Geometric Memory Map**.

## Core Architecture: Geometric Memory Map (H³/Γ)

Instead of storing Key-Value pairs in linear VRAM buffers, `core_lm` utilizes a hyperbolic 3-space modulo the Picard group ($H^3/\Gamma$, where $\Gamma = PSL(2, \mathbb{Z}[i])$).

### Design Principles:
- **Geometry as Index**: Memory addresses are coordinates in hyperbolic space. Retrieval is performed via Lorentz boosts to the origin and distance ranking.
- **Deterministic Address Path**: All geometric calculations use $Q32.31$ fixed-point arithmetic to ensure total determinism across hardware.
- **Weightless Execution**: By leveraging mmap'd virtual tensors and a pipelined ring-buffer for layer weights, the server reduces the unified VRAM footprint.
- **Payload Storage**: The geometry stores addresses; payloads are handled via the P5 lossless codec.

### Technical Specification:
- **Space**: $H^3/\Gamma$ (Hyperbolic 3-space / Picard Group).
- **Coordinates**: Lorentz model for arithmetic; Poincaré ball for visualization only.
- **Precision**: $Q32.31$ fixed-point throughout.
- **Retrieval**: Lorentz Boost $\to$ Origin $\to$ Distance Ranking.
- **Tokenizer merges**: Qwen `tokenizer.json` merge entries are decoded as ordered two-string arrays, preserving the source rank and avoiding lossy space-splitting.
- **Weight files**: MLX/Qwen safetensors file metadata is treated as file-level metadata, never as a tensor descriptor.
- **Muse Glimmer weights**: 4-bit tensors store eight low-nibble-first values per packed `u32`. Shape validation and dequantization expand packed columns by eight before matrix operations.
- **Sharded quantization**: Packed weights, scales, and biases are resolved independently through the safetensors index, so quantization companions may live in different shards.
- **Geometric bridge**: Transformer keys are projected into H³ from their first three coordinates only, so the Lorentz norm uses the same dimensions as the stored spatial vector.

## Implementation Status
- [x] Q32.31 Fixed-Point Arithmetic.
- [x] Lorentz Model Vector implementation.
- [x] Geometric Memory Map (Atom Store & Retrieval).
- [x] Picard Group Fundamental Domain Mapping.
- [ ] Native Metal Forward Pass integration.
- [ ] P5 Codec integration.
