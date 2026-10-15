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
- `orchestrator/` — Rust control plane, authenticated HTTP/TLS edge, metrics, health checks, snapshots, and worker-cache generation gates
- `core_lm/src/attention_backend.rs` and `orchestrator/src/glimmer_map.rs` — opt-in boundary that routes real Glimmer layer attention through the universal server map; enable with the orchestrator `glimmer` feature
- `orchestrator/src/bin/qualify_map_glimmer.rs` — warmed, paired Glimmer comparison for baseline softmax, P2 pair angles, map-T² weighting, and direct value recall
- The qualifier captures a selected layer's real Q/K/V and baseline attention output, then evaluates a gain grid on an early/held-out split; typed map atoms carry the key-derived T² phase rather than a placeholder phase
- The same qualifier has an offline-only `--fit-map-adapter true` experiment: repeated `--prompt` values split by prompt into fit/held-out sets, learns a ridge-linear query+direct-map-value adapter, and reports attention-output error; it is not wired into inference and only writes an adapter file when `--adapter-output` names a new path
- `core_lm/src/bin/bench_glimmer_lm_head.rs` isolates the final vocabulary projection. Muse Glimmer's 5-bit/group-128 `lm_head` now dispatches to a specialized Metal kernel instead of silently falling back to CPU; warmed head-only median was 14.06 ms vs 125.86 ms CPU (8.9×), with relative logit L2 8.7e-7 against the CPU control. See `docs/LM_HEAD_Q5_METAL.md`.
- `orchestrator/src/toroidal_attention.rs` — map-backed operator and fit proof; typed token K/V is required
- `immune_system/` — reserved for audit and adversarial tooling
- `scripts/` — operator scripts

## License

Apache 2.0 — see [LICENSE](LICENSE).
