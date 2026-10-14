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

## Qualification

Verified by [tests/full_chain_qualification.rs](tests/full_chain_qualification.rs) (`cargo test --manifest-path Cargo.toml`) and executable demonstration [src/bin/monster_hybrid_server.rs](src/bin/monster_hybrid_server.rs) (`[verified]`).
