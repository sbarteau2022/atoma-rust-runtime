---
title: "Specialized Metal GEMV for the Glimmer 5-bit output head"
document_type: "implementation"
version: "1.0.0"
date: "2026-10-15T00:10:23Z"
implementation_date: "2026-10-15T00:10:23Z"
author: "Codex"
reviewers: []
status: "draft"
related_files:
  - path: "core_lm/src/quant_gemv.metal"
    lines: "25-105"
    role: "primary"
  - path: "core_lm/src/metal_quant.rs"
    lines: "43-109, 246-304, 507-572"
    role: "primary"
  - path: "core_lm/src/glimmer_model.rs"
    lines: "431-520"
    role: "supporting"
  - path: "core_lm/src/bin/bench_glimmer_lm_head.rs"
    lines: "1-120"
    role: "test"
  - path: "README.md"
    lines: "40-44"
    role: "supporting"
related_issues: []
related_documents: ["MAP_TOROIDAL_ATTENTION.md"]
tags: ["rust", "metal", "gemv", "lm-head", "performance"]
---

# Specialized Metal GEMV for the Glimmer 5-bit output head

## Executive Summary

Muse Glimmer's final vocabulary projection was taking the generic CPU path even though Metal was enabled. Its 5-bit, group-128 matrix now has a dedicated Metal kernel that can consume packed weights without requiring the memory-mapped host pointer to be aligned. On a warmed head-only checkpoint run, median time fell from 125.86 ms to 14.06 ms, about 8.9×, with near-identical logits to the CPU reference.

## Problem Statement

The checkpoint has `lm_head.weight` logical shape 202,048 × 6,656 at 5 bits with groups of 128. The Metal dispatcher only accepted aligned `U32` pointers for generic kernels; this tensor did not meet that host-alignment condition and silently used the CPU GEMV. The CPU path consumed nearly the entire reported decode step.

## Solution Overview

Added a specialized group-128 Q5 kernel and a byte-slice Metal entry point. It computes fixed-width 5-bit offsets, hoists affine scale/bias once per 128-value group, and lets Metal create/cache an aligned device buffer from the original bytes. The dispatcher uses this route only for 5-bit group-128 tensors; unsupported shapes retain the existing fallback.

## Implementation Details

| File | Lines | Change Type | Description |
| --- | --- | --- | --- |
| `core_lm/src/quant_gemv.metal` | 25–105 | modified | Q5 unpack helper and SIMD-group kernel |
| `core_lm/src/metal_quant.rs` | 43–109, 246–304, 507–572 | modified | Q5 pipeline, byte-backed buffer path, validation, GPU-vs-CPU test |
| `core_lm/src/glimmer_model.rs` | 431–520 | modified | Dispatch group-128 Q5 through Metal without host-pointer cast |
| `core_lm/src/bin/bench_glimmer_lm_head.rs` | 1–120 | new | Warmed head-only benchmark and CPU logit control |
| `README.md` | 40–44 | modified | Benchmark result and command pointer |

Rust and Metal Shading Language are used to keep quantized decoding in the existing native model runtime and execute the head on Apple GPU hardware.

This implementation was not directly grounded in published literature. It applies the existing affine quantization identity and SIMD-group reduction pattern to the checkpoint's measured Q5/group-128 layout.

No model config or inference default changed. Q5/128 dispatch is selected from the checkpoint's quantization metadata; other configurations continue through existing routes.

## Testing Evidence

The Metal Q4/Q6/Q5 kernel test passed. The Q5 kernel's output matched the CPU reference within 1e-2 per row on randomized synthetic tensors; the test also verifies an unsupported group size is rejected. `cargo test --manifest-path core_lm/Cargo.toml --lib` passed all 26 tests.

`cargo test --manifest-path core_lm/Cargo.toml --all-targets` and the Glimmer-enabled orchestrator suite passed. Strict Clippy remains red on 51 existing lint findings in other code; the filtered diagnostics contained no findings in the new Q5 implementation or benchmark. Repository-wide `cargo fmt --check` also reports existing unrelated formatting diffs, while the new benchmark and Metal binding files are formatted.

The real checkpoint benchmark used two warmups and nine timed repeats over a deterministic synthetic hidden vector:

| Path | Median | Min–max |
| --- | ---: | ---: |
| CPU generic Q5 control | 125.86 ms | 125.34–127.47 ms |
| Specialized Metal Q5 | 14.06 ms | 13.92–15.17 ms |

That is 8.95× head-only speedup. Metal timed calls made zero CPU fallback GEMVs. Against the CPU output for the same hidden vector, relative logit L2 was 8.71e-7, max absolute error 1.51e-5, and top-1 token agreed. This is a head microbenchmark, not a full end-to-end generation result or a multi-prompt quality qualification.

Reproduce:

```bash
cargo run --release --manifest-path core_lm/Cargo.toml --bin bench_glimmer_lm_head -- --model /path/to/Muse-Glimmer --warmup 2 --repeat 9
cargo run --release --manifest-path core_lm/Cargo.toml --bin bench_glimmer_lm_head -- --model /path/to/Muse-Glimmer --warmup 2 --repeat 7 --cpu-only true
cargo test --manifest-path core_lm/Cargo.toml --lib
```

The benchmark establishes that this optimization is necessary for the reported head bottleneck: Metal was available but not being used for the Q5 head before this change.

## Deployment / Rollout

| Aspect | Detail |
| --- | --- |
| Deployment date | Not deployed |
| Rollout strategy | Kernel is selected only for matching Q5/group-128 tensors |
| Rollback plan | Disable Metal (`WeightManager::use_metal = false`) to use the CPU reference path |
| Monitoring | Benchmark emits dispatch telemetry and CPU-logit comparison |
| Post-deploy validation | Full prompt/token generation benchmark remains required |

## Next Stage of Implementation

This raises head throughput enough to meet the requested 5–8× range in the isolated head. If the head is 98% of full-step time as reported, Amdahl's law predicts roughly 7.7× end-to-end gain; that estimate is not measured yet. Next, benchmark real decode tokens with representative prompts, cache state, and sustained warm execution, then profile remaining time. This sets up the latency goal, but does not establish full generation speed.

## What We Learned

| Category | Learning |
| --- | --- |
| Technical | Metal availability did not mean every projection used Metal; pointer-alignment constraints silently routed this tensor to CPU. |
| Process | A tensor-specific benchmark exposed the fallback quickly and separated it from attention-map work. |
| Tooling | Compare full logits and dispatch counters alongside wall time to avoid promoting a fast but incorrect kernel. |
| Estimation | Head-only speedup is 8.95×; end-to-end gain depends on the unverified 98% profile. |
| Collaboration | No delegation or external coordination was used. |
