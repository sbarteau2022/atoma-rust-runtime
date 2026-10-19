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

## Unified L0 runtime controller

`runtime_model::MainModelCoordinatorRuntime` loads the trajectory, static, and operator vaults into the existing central map through authenticated, deduplicating merges. Its boundary adapter transduces UTF-8 to MLIR carrier IDs and back using the registered tokenizer table (byte-reversible before a tokenizer is installed). A `SessionHandle` retains the real `GeodesicCursor` (Lorentz point, phase, Wigner frame, and chain ID) rather than a dense KV cache. Tier 1 and Tier 2 use the established active-inference path. Tier 3 supports a qualified-teacher callback, with a checkpoint-backed Glimmer implementation available under the `glimmer` feature. The kill runner accepts exactly 24 measured held-out results and delegates pass/fail to the existing `0x94ce028317e0ecac` Layer-3 gate; it does not manufacture metrics.

With the `glimmer` feature, `runtime_model::GlimmerTeacher::load(model_dir)` wires Tier 3 to the checkpoint tokenizer and `GlimmerModel::forward`. `execute_human_turn_with_glimmer` performs prompt prefill plus greedy autoregressive decoding, uses Metal when the model weight manager reports it available, and detokenizes generated IDs with the checkpoint tokenizer. Its geometric attention maps and positions persist per `session_id`; subsequent calls tokenize only the new user turn and insert a turn separator. One loaded model serves multiple isolated session maps. Call `teacher.reset_session(session_id)` to clear one conversation. Example build: `cargo build --release --manifest-path orchestrator/Cargo.toml --features glimmer --bin generate_map_native`. Include the needed chat role markers in each new user-turn string, such as `User: ...\nAssistant:`; the model continues attending to earlier turns held in that session's geometric maps.

## Map-backed toroidal attention proof

[`MonsterHybridServer::ingest_attention_kv`](src/toroidal_attention.rs) writes typed K/V records with layer, K/V head, request-sequence ID, and token position through the existing distill/refract/carve path into the centralized map. [`evaluate_map_attention_layer`](src/toroidal_attention.rs) reads only a role's carved silo, filters those identities and causal token positions, takes the nearest `top_k` eligible atoms, and computes both dot-product/softmax and toroidal attention over the same candidates. Callers must assign a distinct sequence ID to each independent request and pass the correct GQA K/V head. A configured `target_layer` selects toroidal output for exactly one layer; other layer calls select the control. It reports both outputs, weights, and relative L2 error. The P2 arm uses consecutive post-RoPE Q/K pair angles, matching Muse Glimmer's traditional RoPE layout; a separately labeled `MapT2` arm uses the central map's stored T² phases. `gain` is supplied explicitly (`1.0` is the unfitted control), while `baseline_dot_scale` must match the model (`qk_scale_factor / head_dim` for Glimmer).

The central carved silo now maintains a typed K/V position index and a bounded 16×16 T² phase-cell index, updated on insertion and rebuilt when loading a vault. Typed attention writes preserve the key-derived T² phase (not the generic distillation tap phase). Existing distilled atoms are not treated as token K/V. The feature-gated `core_lm` backend wires real Glimmer Q/K/V into this map at every layer; unselected layers keep dot-product/softmax, while selected layers can use P2, T² weighting, or direct value recall. Direct recall skips Q·K and softmax, but still computes projections, MLPs, output projection, and map writes. The qualifier captures a selected layer's attention tensors/output and reports gain-grid fit on first-half positions with second-half held-out error. This is still experimental, not a quality-qualified replacement. Run `cargo test --manifest-path orchestrator/Cargo.toml --features glimmer` for map and forward integration gates.

Build the paired checkpoint qualifier with `cargo build --release --manifest-path orchestrator/Cargo.toml --features glimmer --bin qualify_map_glimmer`. Example: `orchestrator/target/release/qualify_map_glimmer --model /path/to/Muse-Glimmer --vault /path/to/global_swarm_map.atoma_map --prompt 'short prompt' --tokens 16 --layer all --mode direct`. It warms the model, compares against map-backed dot-product/softmax on the same teacher-forced trace, and reports per-token logit error, argmax agreement, TTFT, decode p50/p95, and map operation counts. Model/vault setup is excluded from TTFT; each run is a single sample and does not establish a stable speed claim. See [the implementation record](../docs/MAP_TOROIDAL_ATTENTION.md).

An experimental offline ridge-linear map-value fit can be run with `--fit-map-adapter true` and at least two repeated `--prompt` arguments. Prompts are split 75/25 in supplied order (last prompt(s) held out); it fits query-head + direct phase-recalled value features to the baseline attention output and reports train/held-out mean relative L2. Example: `qualify_map_glimmer --model /path/to/Muse-Glimmer --vault /path/to/global_swarm_map.atoma_map --fit-map-adapter true --tokens 2 --prompt 'The capital of France is' --prompt 'Water freezes at zero degrees Celsius and boils at' --prompt 'A prime number after seven is' --prompt 'The largest ocean on Earth is'`. This is offline research only: it does not alter model serving, and a saved adapter path must not already exist. No production use is justified until held-out attention and final-logit congruence gates pass.

## Native toroidal prompt readout and qualification

The native path converts raw UTF-8 byte IDs into six H³×T² address coordinates and processes
them in session-isolated recurrent fields. It uses no checkpoint tokenizer. The standalone
stack distributes its 0.18 residual mixing budget across its depth; the map-backed layer
substitution path retains its existing per-layer mixing behavior. Inputs are checked before
any byte advances the recurrence. `MainModelCoordinatorRuntime::reset_session` clears both
the legacy cursor and the native fields/hidden caches.

`native_output.rs` classifies prompt-position hidden states to byte addresses using periodic
phase distance. It is untrained and predicts neither the next byte nor EOS. The strict reader
rejects unknown IDs, non-byte expansions, and invalid UTF-8. `/api/turn` labels these diagnostic
fields as `native_output_kind: "prompt_readout"` and `native_output_is_continuation: false`.
Diagnostic hidden-state capture is O(prompt bytes), even though the recurrent field size is fixed.

`qualify_native_toroidal` consumes fixed completion fixtures and emits a JSON evidence report
including raw outputs, structural controls, an untrained self-feedback probe, and hashes of
the executable and its compile-time source. Its dependencies are the native components and
Rust `std`/`serde_json`; it never constructs the legacy server or loads a teacher/map/model.
Build without `glimmer`. The current independent run matched **0/24** completion references.
See [the contract](../docs/NATIVE_TOROIDAL_QUALIFICATION.md) and
[the measured report](../benchmarks/NATIVE_TOROIDAL_REPORT.md).

```sh
cargo run --release --no-default-features --manifest-path orchestrator/Cargo.toml \
  --bin qualify_native_toroidal -- --stress-bytes 100000 \
  --output benchmarks/native_toroidal_quality.json --require-quality
```

Run this command from the repository root. A recorded quality failure exits with status 2.

## Learned native next-atom producer

**Purpose:** Implement the Phase B next-carrier producer in
[`NATIVE_NEXT_ATOM_DESIGN.md`](../docs/NATIVE_NEXT_ATOM_DESIGN.md) with a trained readout
over the native toroidal recurrence. This is a separate generation path from prompt readout.

**Inputs/outputs:** `native_predictor::NativeToroidalPredictor::fit` consumes explicit
`NativeTextRecord { label, prompt, completion }` records. Whole documents have empty prompts.
Training supervises the next UTF-8 byte and final EOS, with no prompt loss. The 257-class
softmax reads 39 causal toroidal features plus a current-byte/BOS branch and bias (76,329
parameters). `generate(prompt, max_new)` starts fresh 52-field state by default and returns
byte IDs, strict text/error, stop reason, EOS ID, and timings. Every predicted byte is fed
through the recurrence. Output is not masked, repaired, or replaced by a fallback. Field
storage is fixed; generated output is O(byte budget), and offline feature caching is O(corpus
positions). Only the head is trained. The phase encoder and rotation-driven radii remain live.

**Dependencies:** Native byte/address geometry, `ToroidalIsotropicRuntime`, strict boundary
egress, serde/serde_json, existing SHA-256, and Rust std. No model, teacher, network call,
tokenizer lexicon, or map is loaded. The standalone `native_toroidal` binary rejects `glimmer`
builds. Saved binary envelopes include JSON parameters, a SHA-256 corruption check, native
schema/configuration, source fingerprint, corpus provenance, and training checkpoint selection.
Loading validates bounds, shapes, finite values and compatibility. Atomic publication refuses
to overwrite files. The primary server cascade is not promoted by creating this artifact.

**Qualification:** Eleven mechanism/edge/artifact unit tests and
[`tests/native_toroidal_predictor.rs`](tests/native_toroidal_predictor.rs) cover training,
saved-model generation, EOS, Unicode, data isolation, corruption, and CLI failure behavior.
Training-only checkpoint selection and regularization around the empirical byte-statistics
prior prevent the saved checkpoint from regressing in training NLL. See the
[measured report](../benchmarks/NATIVE_NEXT_ATOM_REPORT.md) for independent byte-statistics
controls and actual language/termination failures. Synthetic mechanism fixtures in tests are
not the training corpus used for that report.

Run from the repository root (choose new output filenames):

```sh
cargo build --release --no-default-features --manifest-path orchestrator/Cargo.toml \
  --bin native_toroidal
orchestrator/target/release/native_toroidal train \
  --train-text path/to/train.md --eval-text path/to/held-out.md \
  --artifact new-model.atoma_next --report new-training-report.json
orchestrator/target/release/native_toroidal generate \
  --artifact benchmarks/native_toroidal_repo_prose_v2.atoma_next \
  --prompt "The native runtime " --max-new 128
```

Repeat `--train-text`/`--eval-text` for multiple documents. `--train-jsonl`/`--eval-jsonl`
accept strict `{"prompt":"...","completion":"..."}` records. Labels never enter features.
`evaluate --artifact FILE --eval-jsonl FILE --report NEW_FILE --require-exact` writes raw
evidence and exits 2 when exact completion plus EOS fails; I/O/configuration errors exit 1.
Standalone evaluation rejects exact hashes of training documents; full document-containment
checks require both corpora and run before fitting. Held-out labels score teacher-forced
metrics but are never fed into greedy generation. A tiny technical-document corpus cannot
establish general language quality or matched transformer replacement.

## Qualification

Verified by [tests/full_chain_qualification.rs](tests/full_chain_qualification.rs) (`cargo test --manifest-path Cargo.toml`) and executable demonstration [src/bin/monster_hybrid_server.rs](src/bin/monster_hybrid_server.rs) (`[verified]`).
