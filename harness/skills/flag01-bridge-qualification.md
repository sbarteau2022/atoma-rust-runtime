---
name: flag01-bridge-qualification
description: Run and verify the standalone FLAG-01 empirical key-energy measurement and replacement bridge qualification harness (`harness/bridge`) without modifying `core_lm/`.
---

# Skill: `flag01-bridge-qualification`

## Purpose

Executes the standalone `atoma_bridge_harness` qualification suite (`harness/bridge/`) to measure retained L2 energy, RoPE 2D pair integrity, hyperboloid radial jitter, top-$k$ retrieval recall inside `core_lm::cache::geometric::GeometricMemoryMap`, and downstream attention output cosine similarity across 128D (Muse Glimmer) and 256D (Qwen3.5-MoE) key-head regimes (`[verified]`).

## Inputs/Outputs

- **Inputs**:
  - `harness/bridge/src/lib.rs` and `harness/bridge/src/bin/ion_flag01.rs`
  - Read-only mount of `core_lm/src/geometry/`, `core_lm/src/cache/geometric.rs`, and `core_lm/src/lock.rs` via `atoma_immune`
- **Outputs**:
  - Pass/fail verification of the 3 unit test gates and 20 regime-bridge benchmark configurations (`immune_system/audit/FLAG_01_BRIDGE_STUDY.md`)

## Dependencies

- Rust toolchain (`cargo`, `rustc`)
- `immune_system/implementation` (`atoma_immune`)
- Zero external crates (`--offline`)

## Qualification

1. Execute the unit gates and CLI benchmark runner in `harness/bridge/`:
   ```bash
   cd /Users/stewartbarteau/Desktop/monster-hybrid-server/harness/bridge
   cargo test --offline
   cargo run --offline --bin ion_flag01
   ```
2. Verify that `core_lm/` remains 100% untouched (`git status core_lm/` shows zero modified or untracked files) (`Qualification #2`).
