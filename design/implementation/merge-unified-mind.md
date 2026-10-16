---
title: "Unified One-Mind Vault Merger"
document_type: "implementation"
version: "1.0.0"
date: "2026-10-16T03:09:28Z"
implementation_date: "2026-10-16T03:09:28Z"
author: "Codex / Stewart Barteau"
reviewers: []
status: "draft"
related_files:
  - path: "immune_system/implementation/src/bin/merge_unified_mind.rs"
    lines: "1-94"
    role: "primary"
  - path: "immune_system/implementation/README.md"
    lines: "19-23"
    role: "supporting"
related_issues: []
related_documents: []
tags: ["vault", "merge", "rust", "atoma-map"]
---

## Executive Summary

Added a command-line tool that combines the static, operator, Glimmer, and Qwen vaults into one separate output vault. It uses the repository’s existing deduplicating merge logic and avoids replacing existing data. This makes the combined map available for the existing native reader and server.

## Problem Statement

There was no dedicated merger producing the `unified_one_mind.atoma_map` named in the supplied runbook. The prior compiler targets a different output and includes extra validation/benchmark behavior.

## Solution Overview

The binary accepts defaults matching the runbook, supports path overrides, requires all four inputs, compacts trajectory payloads through `merge_vault_from`, and refuses output collisions. Output is written to a temporary sibling and renamed into place after serialization.

## Implementation Details

### Files Modified

| File | Lines | Change | Description |
| --- | --- | --- | --- |
| `immune_system/implementation/src/bin/merge_unified_mind.rs` | 1-94 | new | CLI, preflight checks, vault merge and save |
| `immune_system/implementation/README.md` | 19-23 | modified | Build and usage instructions |

### Language and Next Step

Rust matches the existing vault API and binary tooling. A future extension could add post-merge compliance and benchmark gates without coupling them to basic merger behavior.

### Literature Grounding

This implementation was not directly grounded in published literature. It applies existing repository vault-merging and safe-output patterns.

### Configuration Changes

No configuration files or environment variables changed. Defaults are the four `design/*.atoma_map` inputs and `design/unified_one_mind.atoma_map`; all can be overridden with CLI flags.

## Testing Evidence

### Commands and Results

| Command | Result |
| --- | --- |
| `cargo check --offline --manifest-path immune_system/implementation/Cargo.toml --bin merge_unified_mind` | Passed |
| `cargo build --offline --release --manifest-path immune_system/implementation/Cargo.toml --bin merge_unified_mind` | Passed |
| `cargo test --offline --manifest-path immune_system/implementation/Cargo.toml` | Passed; 50 library tests |
| `./immune_system/implementation/target/release/merge_unified_mind` | Passed; merged 212,603 atoms and 190,156 arcs; output 127,554,065 bytes |
| `./immune_system/implementation/target/release/merge_unified_mind --out design/unified_one_mind.atoma_map` | Correctly rejected existing output; artifact preserved |
| `cargo run --offline --release --manifest-path immune_system/implementation/Cargo.toml --bin read_mlir_native -- --vault design/unified_one_mind.atoma_map --mlir-ids 151644,872,198 --base-vocab 248320 --max-new 24 --active-inference` | Passed; 24 steps, 16 superposition interventions |
| Same reader command with `--base-vocab 202048 --max-new 6` | Passed; 6 steps, 6 superposition interventions |

These are smoke tests of loading and replay, not evidence of text quality, cross-model semantic equivalence, or server readiness. The implementation was not directly proven necessary by a comparative benchmark.

### Edge Cases

Missing inputs, unknown flags, malformed flag/value pairs, output/input aliasing, pre-existing output, and stale temporary output are rejected in preflight. The successful run exercised distinct inputs and a previously absent output. Automated tests for these error branches remain a follow-up.

## Deployment / Rollout

Run from the repository root: `cargo run --offline --release --manifest-path immune_system/implementation/Cargo.toml --bin merge_unified_mind`. The generated vault is a separate artifact; this implementation does not launch a server. To rebuild after a failed write, inspect/remove the exact partial output only after verifying its path. No input vault is modified.

## Next Stage

The merger provides a combined artifact but does not prove its quality or start the persistent coordinator. Next: add automated CLI collision/missing-input tests, then separately validate quality and server integration. It sets up that goal by producing the requested artifact using native map semantics.

## What We Learned

The repository already provides a merge API that remaps IDs and deduplicates arcs; duplicating those internals would have increased risk. The existing CLI reader can replay the merged artifact for both configured base-vocabulary sizes, but replay alone does not establish useful language output.
