# Orchestrator — Layer 2 Governor & Control Plane (`atoma_orchestrator`)

## Purpose

Supervises the 1 + 12 carved role processes using an Erlang/OTP-style supervision tree and routes all inter-role boundary handoffs through Layer 3 Immune System compliance gates and lock-free atomic ring buffers (`[verified]`).

## Inputs/Outputs

Consumes `TeacherTapRecord`s, `HandoffRequest`s, and role heartbeat/fault signals; produces `DistillRefractCarveReceipt`s, 96-byte-header `FlatHandoffPacket` frames over `AtomicRingBuffer`, and `DeliveredHandoff` receipts authorized by `CompliancePermit`s (`[verified]`).

## Dependencies

`core_lm` (`../core_lm`), `atoma_immune` (`../immune_system/implementation`), and Rust `std` (`[verified]`).

## Qualification

Verified by [tests/full_chain_qualification.rs](tests/full_chain_qualification.rs) (`cargo test --manifest-path Cargo.toml`) and executable demonstration [src/bin/monster_hybrid_server.rs](src/bin/monster_hybrid_server.rs) (`[verified]`).
