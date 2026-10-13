# Monster Hybrid Server

Deterministic Rust inference server research. The goal: replace the transformer's KV cache with a **geometric memory map** — atoms addressed by position on a hyperbolic manifold — inside a Q32.31 fixed-point deterministic shell.

## Current status (2026-10-13)

- `core_lm/` holds a full Muse Glimmer forward pass (52 layers) with the KV cache replaced by per-layer, per-KV-head `GeometricMemoryMap`s. RoPE intact, dot-product rescoring over geometric candidates.
- `core_lm/src/lock.rs` + `kuramoto_tile.metal`: Q32.31 CORDIC trig and Kuramoto phase-locking for the topological `(n,m,ψ*)` addressing layer. Trig verified to 19.5 LSBs max error in VM testing.
- **Honest state of proof:** the map's storage contracts (write / retrieve / evict / audit / determinism) hold under test. Isolated validation showed pure geometric nearest-neighbor cannot approximate 128D attention — so the map currently serves as a candidate pre-filter with exact dot-product rescoring. Forward-pass output coherence is the open test.

## Layout

- `core_lm/`: Rust execution plane — Glimmer forward pass, geometric memory, fixed-point geometry, Metal kernels.
- `design/`: architecture docs and [qualification gates](design/QUALIFICATION_GATES.md).
- `harness/`: qualification tooling.
- `orchestrator/`: reserved for the control plane.
- `immune_system/`: reserved for audit and adversarial tooling.
- `scripts/`: operator scripts.

## Build

```sh
cd core_lm && cargo build
cargo run --bin generate_glimmer -- "The capital of France is" 50
```

Requires the Muse Glimmer q4 checkpoint at `~/models/Muse-Glimmer-q4-MLX` (path is hardcoded in `src/bin/generate_glimmer.rs`).
