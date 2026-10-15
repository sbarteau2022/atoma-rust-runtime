---
title: "Central map toroidal attention proof"
document_type: "implementation"
version: "1.3.0"
date: "2026-10-14T22:33:33Z"
implementation_date: "2026-10-14T22:33:33Z"
author: "Codex"
reviewers: []
status: "draft"
related_files:
  - path: "core_lm/src/attention_backend.rs"
    lines: "1-46"
    role: "primary"
  - path: "core_lm/src/glimmer_forward.rs"
    lines: "27-28, 229-390, 454-580"
    role: "primary"
  - path: "core_lm/src/lib.rs"
    lines: "7-7"
    role: "supporting"
  - path: "immune_system/implementation/src/silo.rs"
    lines: "211-360, 570-600, 760-790, 860-925"
    role: "supporting"
  - path: "orchestrator/src/glimmer_map.rs"
    lines: "1-371"
    role: "primary"
  - path: "orchestrator/src/toroidal_attention.rs"
    lines: "330-357, 1-582"
    role: "primary"
  - path: "orchestrator/src/bin/qualify_map_glimmer.rs"
    lines: "1-586"
    role: "supporting"
  - path: "orchestrator/src/lib.rs"
    lines: "10-16, 133-177"
    role: "supporting"
  - path: "orchestrator/Cargo.toml"
    lines: "1-21"
    role: "config"
  - path: "orchestrator/Cargo.lock"
    lines: "1-607"
    role: "config"
  - path: "orchestrator/tests/glimmer_map_backend.rs"
    lines: "152-278"
    role: "test"
  - path: "orchestrator/tests/map_toroidal_attention.rs"
    lines: "1-414"
    role: "test"
  - path: "orchestrator/README.md"
    lines: "21-33"
    role: "supporting"
  - path: "README.md"
    lines: "36-49"
    role: "supporting"
related_issues: []
related_documents: ["design/CARVED_SWARM_SPEC.md", "design/DESIGN.md"]
tags: ["rust", "central-map", "toroidal-attention", "experiment"]
---

# Central map toroidal attention proof

## Executive Summary

Real Muse Glimmer Q/K/V now reaches an opt-in attention backend backed by the universal Monster server's carved silo. An offline ridge-linear prototype can fit query plus direct map-value features to baseline attention outputs and score held-out prompts. It improved over direct recall but still had 0.684 mean relative L2 on two held-out prompts, so it is not sufficiently faithful or enabled for serving.

## Problem Statement

The prior path stored local geometric maps inside the model. It did not query the central universal map. Central atoms also lacked typed layer, K/V-head, request-sequence, and token-position metadata, so treating map write order as a causal mask would be incorrect. Even with a live map-backed path, replacing learned attention requires preserving model output quality, which remains an open gate.

## Solution Overview

The server stores typed key/value (K/V) payloads through its existing distill, refract, and carve path. Each payload includes layer, K/V-head, sequence ID, token position, and model epoch. A silo-owned position index and 16×16 phase-cell index are updated on insertion and rebuilt on vault reload. Typed K/V writes explicitly preserve the key's projected T² phase; an initial dummy tap gave every key the same phase and was caught by a flat gain sweep. The live backend writes current post-RoPE K/V before recall; baseline/P2 read all causal positions, while MapT2/Direct retrieve bounded phase-neighbor candidates. Every path checks request identity, window, model epoch, and payload shape. Output gates, projections, norms, and MLPs remain model-owned.

The P2 arm uses angles from consecutive two-dimensional pairs of the post-RoPE query and key heads. Muse Glimmer's traditional RoPE layout has 64 such pairs in a 128-dimensional head. The separate `MapT2` arm compares the supplied query phase with each atom's stored T² phase; it is a new experiment and must not be labeled as the pinned P2 operator. P2 uses the central map for candidate authority but not for its phase score.

## Implementation Details

### Files Modified

| File | Lines | Change |
| --- | --- | --- |
| `core_lm/src/attention_backend.rs` | 1–46 | Validated model/backend boundary |
| `core_lm/src/glimmer_forward.rs` | 229–390, 454–580 | Real forward hook; preserves the existing local path |
| `immune_system/implementation/src/silo.rs` | 211–360, 570–600, 760–790, 860–925 | Typed causal and phase indexes; K/V non-coalescing |
| `orchestrator/src/glimmer_map.rs` | 1–371 | Central-map adapter and captured layer traces |
| `orchestrator/src/toroidal_attention.rs` | 330–357 | Typed write uses key-derived map phase |
| `orchestrator/src/bin/qualify_map_glimmer.rs` | 1–510 | Real-checkpoint paired test, gain/holdout scoring, and offline ridge adapter fit |
| `orchestrator/tests/glimmer_map_backend.rs` | 152–278 | Real-forward adapter and trace tests |
| `orchestrator/tests/map_toroidal_attention.rs` | 1–414 | Map phase, masks, vault, and operator gates |
| `README.md`, `orchestrator/README.md` | 36–48, 21–31 | Usage, scope, and status |

### Language Choice

Rust keeps this operation in the existing server and preserves the same role and compliance gates used by the rest of the control plane.

### Literature Grounding

This implementation was not directly grounded in published literature. It implements the operator described in the user-supplied `MASTER_ARCHITECT.md` v0.8.9 decision and extends it to Muse Glimmer's consecutive-pair RoPE layout. The referenced architecture file was not present in this checkout; the model outputs show the implementation is not qualified as a replacement.

### Configuration Changes

Enable the integration with orchestrator feature `glimmer`. `RecallConfig` selects intervention layers, `Baseline`/`PairAngles`/`MapT2`/`Direct`, positive finite sharpening gain (currently explicitly set to 1.0 in the qualifier), and bounded candidate count (1–512). Baseline scaling is taken from model configuration (`qk_scale_factor / head_dim`). No gain is promoted as fitted. The qualifier needs the real model directory and existing universal map vault. Startup is excluded from TTFT; this is stated in its JSON output.

## Testing Evidence

### Test Files and Locations

| Test file | Lines | Coverage |
| --- | --- | --- |
| `orchestrator/src/toroidal_attention.rs` | 534–580 | P2 geometry, convexity, gain behavior, malformed envelopes |
| `orchestrator/tests/map_toroidal_attention.rs` | 1–414 | Central map, causal/layer/head/sequence/role masks, rewritten slot, vault round trip, cross-epoch adapter, key-phase and invalid-input tests |
| `orchestrator/tests/full_chain_qualification.rs` | 1–1090 | Existing governor, immune, map, and forward chain |
| `orchestrator/tests/glimmer_map_backend.rs` | 1–278 | Actual forward hook, local-map bypass, GQA/window, bounded recall, captured trace, invalid inputs |

### What the Tests Showed

The additional offline adapter experiment used the real Muse Glimmer checkpoint, layer 0, four fit prompts and two distinct held-out prompts (2 token positions per prompt). It fit 960 query-head samples. Mean attention-output relative L2 was 0.471 on the fit set and 0.684 on held-out prompts; direct recall on those same held-out prompts was 1.320. This is a measurable improvement over nearest-value recall, not a replacement-quality result. No final-logit agreement was measured for the learned adapter because it has not been connected to the forward pass. The adapter is not saved unless an explicit new output path is provided and is never loaded by serving.

`cargo test --features glimmer` passed 28 orchestrator tests (6 unit, 13 full-chain, 5 Glimmer backend, 4 central-map operator tests); `core_lm cargo test --lib` passed 26 tests; Clippy passed with existing repository lint allowances. Tests cover malformed/non-finite inputs, request and layer isolation, GQA, sliding windows, overwritten positions, vault re-indexing, key-derived phases, and zero Q·K operations in Direct/MapT2.

On the real Muse Glimmer checkpoint, a one-layer (layer 0), 7-position teacher-forced trial compared exact central-map softmax with MapT2 and P2 (candidate limit 64). Offline gain fitting used early positions and scored remaining positions. For MapT2, gain 0.125 gave mean attention-output relative L2 0.688 fit / 0.791 held out; P2's best training gain (2.0) gave 0.669 / 0.773 held out. At runtime gain 1, nontrivial per-position attention errors ranged 0.83–0.92 for MapT2 and 0.67–0.82 for P2 (the first-token single-value case is zero). Both final-logit comparisons matched argmax on 5/7 positions but had a large divergence at a later token (relative logit L2 about 1.7). Corrected Direct recall performed worse: attention error ranged 1.13–1.44 and its held-out mean was 1.30. These small runs reject the current operators; they are not a benchmark or stable latency evidence. Paired TTFT/decode numbers were noisy and did not show a repeatable decode improvement.

### Test Commands

```bash
cargo test --manifest-path orchestrator/Cargo.toml --features glimmer
cargo clippy --manifest-path orchestrator/Cargo.toml --features glimmer --all-targets -- -D warnings -A clippy::too_many_arguments -A clippy::needless_range_loop -A clippy::manual_checked_ops -A clippy::collapsible_if -A clippy::chunks_exact_to_as_chunks -A clippy::manual_find
cargo run --release --manifest-path orchestrator/Cargo.toml --features glimmer --bin qualify_map_glimmer -- --model /path/to/Muse-Glimmer --vault design/global_swarm_map.atoma_map --tokens 4 --layer all --mode direct --candidates 64
cargo run --release --manifest-path orchestrator/Cargo.toml --features glimmer --bin qualify_map_glimmer -- --model /path/to/Muse-Glimmer --vault design/global_swarm_map.atoma_map --fit-map-adapter true --tokens 2 --layer 0 --prompt 'prompt one' --prompt 'prompt two' --prompt 'held out prompt'
cargo test --manifest-path immune_system/implementation/Cargo.toml
```

### Was It Proven Necessary?

Yes for a central-map experiment: the prior path had no layer or token-position field on map atoms, so it could not make a causal attention comparison from those atoms. The test instead rejects the present direct-value replacement on fidelity; it does not show that the original learned attention can yet be removed.

## Languages Chosen and Why

| Language | Component | Rationale |
| --- | --- | --- |
| Rust | model hook, map index, backend, qualification CLI | Keeps attention tensors and the central server map in the existing native execution/control planes with explicit validation. |

Next evolution: a trained map-to-attention/value representation that passes held-out logit congruence before any production serving or latency claim.

## Deployment / Rollout

| Aspect | Detail |
| --- | --- |
| Deployment date | Not deployed |
| Rollout strategy | Feature-gated local qualification only (`glimmer`) |
| Rollback plan | Omit backend; ordinary attention remains the default |
| Monitoring | Qualifier JSON reports latency, errors, and operation counts; not a production monitor |
| Post-deploy validation | Not applicable; production promotion is blocked by logit divergence |

## Next Stage of Implementation

The ridge adapter remains an offline analysis tool, and the goal is not met: its held-out attention error is 0.684 and it has no end-to-end logit result. Next, improve the map representation beyond this weak linear adapter, then run the candidate inside the real forward pass and gate on held-out final-logit congruence. Do not integrate it into serving or claim latency benefits until both attention and end-to-end fidelity pass. This work does not replace the model's attention layers.

## What We Learned

The map's T² coordinates are not the P2 pair angles. Distilled map bytes cannot be assumed to be trained token K/V. A flat gain sweep exposed that the first typed-write implementation stored a placeholder phase; key-derived phase is now explicit and regression-tested. Even corrected, neither phase operator reproduced the learned attention output. Skipping Q·K alone does not guarantee lower decode latency: projections, MLPs and map writes remain. The frozen base used for this qualifier is session-scoped, not a cryptographic fingerprint of checkpoint tensors; that must be corrected before production qualification.

| Category | Learning |
| --- | --- |
| Technical | Removing score calculations or changing score geometry does not replace learned value aggregation; all tested outputs remain far from baseline. |
| Process | A flat gain sweep found a bad phase source. Corrected one-layer sweeps still failed held-out fidelity; broader data and a learned representation are needed. |
| Tooling | A map-owned typed index is needed to avoid scanning the full silo on every token and to survive vault reload. |
| Estimation | The backend integration is achievable; learning a useful representation and proving fidelity remains the substantive unknown. |
| Collaboration | No agents or external coordination were used. |

## Changelog

| Version | Date | Author | Change |
| --- | --- | --- | --- |
| 1.0.0 | 2026-10-14 | Codex | Initial draft for the map-backed one-layer proof |
| 1.1.0 | 2026-10-14 | Codex | Connected actual Glimmer layers to central map; documented measured fidelity failure and non-promotion |
| 1.2.0 | 2026-10-14 | Codex | Added layer trace and fit/holdout gain sweep; fixed K/V phase writes; recorded one-layer rejection results |
| 1.3.0 | 2026-10-14 | Codex | Added offline ridge map-value adapter fit; measured two-prompt holdout improvement over direct recall; kept it unintegrated due to remaining large error |
