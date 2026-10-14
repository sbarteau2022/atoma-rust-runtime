# atoma-rust-runtime

Deterministic Rust inference runtime: hyperbolic geometric memory replaces the KV cache, Q32.31 fixed-point determinism, Metal kernels.

Part of the ATOMA system — ScoutAi, powered by ATOMA.

## Status

Research-stage. What's true right now:

- `core_lm/` runs a full Muse Glimmer forward pass (52 layers) with the KV cache replaced by per-layer, per-KV-head geometric memory maps.
- Map storage contracts (write / retrieve / evict / audit / determinism) hold under test.
- Q32.31 CORDIC trig verified to 19.5 LSBs max error over 200k inputs.
- **Open:** forward-pass output coherence — the congruence test against the q4 MLX baseline hasn't passed yet. See `docs/SESSION_NOTES.md` for the full proven-vs-open ledger. Every claim in this repo is marked verified / reported / stated; nothing is claimed on faith.

## Quickstart

```sh
cd core_lm && cargo build
cargo run --bin generate_glimmer -- "The capital of France is" 50
```

Requires the Muse Glimmer checkpoint at `~/models/Muse-Glimmer-q4km-gs128-MLX` (path is hardcoded in `src/bin/generate_glimmer.rs`).

## Architecture

- **Geometric memory** (`core_lm/src/cache/geometric.rs`) — atoms addressed on a hyperbolic manifold (Lorentz model, Q32.31), replacing the KV cache. Retrieval is a candidate pre-filter; final attention scoring is exact dot-products.
- **Fixed-point shell** (`core_lm/src/geometry/`) — Q32.31 arithmetic, Lorentz boosts, f32↔hyperboloid bridge.
- **Phase locking** (`core_lm/src/lock.rs`, `kuramoto_tile.metal`) — CORDIC trig + Kuramoto ODE for topological addressing, with a CPU fallback for testing.
- **Forward pass** (`core_lm/src/glimmer_forward.rs`) — Glimmer safetensors loading, RoPE, GQA grouped attention over geometric candidates.

## Layout

- `core_lm/` — Rust execution plane
- `design/` — architecture docs and qualification gates
- `docs/` — session notes and handoff docs
- `harness/` — qualification tooling
- `orchestrator/` — reserved for the control plane
- `immune_system/` — reserved for audit and adversarial tooling
- `scripts/` — operator scripts

## License

Apache 2.0 — see [LICENSE](LICENSE).
