# Orchestrator — Layer 2 Governor & Control Plane (`atoma_orchestrator`)

## Purpose

Supervises the 1 + 12 carved role processes using an Erlang/OTP-style supervision tree and routes all inter-role boundary handoffs through Layer 3 Immune System compliance gates and lock-free atomic ring buffers (`[verified]`).

## Inputs/Outputs

Consumes `TeacherTapRecord`s, `HandoffRequest`s, and authenticated HTTP JSON query frames; produces `DistillRefractCarveReceipt`s, 96-byte-header `FlatHandoffPacket` frames over `AtomicRingBuffer`, `DeliveredHandoff` receipts, Prometheus metrics, and SHA-256-authenticated map snapshots (`[verified]`).

## Dependencies

`core_lm` (`../core_lm`), `atoma_immune` (`../immune_system/implementation`), `rustls`, `serde_json`, and Rust `std` (`[verified]`).

## Production API

Run `cargo run --manifest-path orchestrator/Cargo.toml --bin atoma_api_server` with `ATOMA_API_KEYS` set to one or more comma-separated API keys. `POST /v1/query` and administrative backups require `X-API-Key`; `GET /health/live` is intentionally unauthenticated for liveness probes, while readiness and metrics require a key. Bind defaults to `127.0.0.1:8443`. A non-loopback bind is rejected unless `ATOMA_TLS_CERT` and `ATOMA_TLS_KEY` identify a PEM certificate/key pair. `ATOMA_VAULT`, `ATOMA_BACKUP_DIR`, `ATOMA_BACKUP_RETENTION`, and `ATOMA_MAX_REQUESTS` configure persistence and operations. `POST /v1/admin/backup` writes and atomically publishes retained `.atoma_map` snapshots.

Every successful central-map mutation advances a generation. Before a governed forward step, stale worker-local atom maps are discarded and rehydrated under the existing Layer 3 adapter gate.

## Map-backed toroidal attention proof

[`MonsterHybridServer::ingest_attention_kv`](src/toroidal_attention.rs) writes typed K/V records with layer, K/V head, request-sequence ID, and token position through the existing distill/refract/carve path into the centralized map. [`evaluate_map_attention_layer`](src/toroidal_attention.rs) reads only a role's carved silo, filters those identities and causal token positions, takes the nearest `top_k` eligible atoms, and computes both dot-product/softmax and toroidal attention over the same candidates. Callers must assign a distinct sequence ID to each independent request and pass the correct GQA K/V head. A configured `target_layer` selects toroidal output for exactly one layer; other layer calls select the control. It reports both outputs, weights, and relative L2 error. The P2 arm uses consecutive post-RoPE Q/K pair angles, matching Muse Glimmer's traditional RoPE layout; a separately labeled `MapT2` arm uses the central map's stored T² phases. `gain` is supplied explicitly (`1.0` is the unfitted control), while `baseline_dot_scale` must match the model (`qk_scale_factor / head_dim` for Glimmer).

The central carved silo now maintains a typed K/V position index and a bounded 16×16 T² phase-cell index, updated on insertion and rebuilt when loading a vault. Typed attention writes preserve the key-derived T² phase (not the generic distillation tap phase). Existing distilled atoms are not treated as token K/V. The feature-gated `core_lm` backend wires real Glimmer Q/K/V into this map at every layer; unselected layers keep dot-product/softmax, while selected layers can use P2, T² weighting, or direct value recall. Direct recall skips Q·K and softmax, but still computes projections, MLPs, output projection, and map writes. The qualifier captures a selected layer's attention tensors/output and reports gain-grid fit on first-half positions with second-half held-out error. This is still experimental, not a quality-qualified replacement. Run `cargo test --manifest-path orchestrator/Cargo.toml --features glimmer` for map and forward integration gates.

Build the paired checkpoint qualifier with `cargo build --release --manifest-path orchestrator/Cargo.toml --features glimmer --bin qualify_map_glimmer`. Example: `orchestrator/target/release/qualify_map_glimmer --model /path/to/Muse-Glimmer --vault /path/to/global_swarm_map.atoma_map --prompt 'short prompt' --tokens 16 --layer all --mode direct`. It warms the model, compares against map-backed dot-product/softmax on the same teacher-forced trace, and reports per-token logit error, argmax agreement, TTFT, decode p50/p95, and map operation counts. Model/vault setup is excluded from TTFT; each run is a single sample and does not establish a stable speed claim. See [the implementation record](../docs/MAP_TOROIDAL_ATTENTION.md).

An experimental offline ridge-linear map-value fit can be run with `--fit-map-adapter true` and at least two repeated `--prompt` arguments. Prompts are split 75/25 in supplied order (last prompt(s) held out); it fits query-head + direct phase-recalled value features to the baseline attention output and reports train/held-out mean relative L2. Example: `qualify_map_glimmer --model /path/to/Muse-Glimmer --vault /path/to/global_swarm_map.atoma_map --fit-map-adapter true --tokens 2 --prompt 'The capital of France is' --prompt 'Water freezes at zero degrees Celsius and boils at' --prompt 'A prime number after seven is' --prompt 'The largest ocean on Earth is'`. This is offline research only: it does not alter model serving, and a saved adapter path must not already exist. No production use is justified until held-out attention and final-logit congruence gates pass.

## Qualification

Verified by [tests/full_chain_qualification.rs](tests/full_chain_qualification.rs) (`cargo test --manifest-path Cargo.toml`) and executable demonstration [src/bin/monster_hybrid_server.rs](src/bin/monster_hybrid_server.rs) (`[verified]`).
