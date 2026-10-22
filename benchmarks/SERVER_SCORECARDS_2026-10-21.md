# Server Scorecards — 2026-10-21

Execution records for `/Users/stewartbarteau/models/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP` on the local Apple host. Results are workload- and configuration-specific.

## Winner: native Dspark

Selected profile: `mode=dspark`, `max-draft=3`, `kv-bits=8`, single request lane, 131072 context, 512-token prefill, prefix cache enabled.

| cap | observed decode tok/s | acceptance/roofline | disposition |
|---:|---:|---:|---|
| 2 | 135.0 | 87.3% | pass |
| 3 | **143.5** | **92.9%** | **selected** |
| 4 | 133.8 on independent Rust probe | 86.5% | rejected for quality target |
| 5–6 | no stable quality-constrained advantage established | — | rejected |

### Reproduction command

```bash
export QWEN_PREFIX_CACHE_MAX_BYTES=8589934592
export QWEN_SINGLE_CONVERSATION_SNAPSHOT=1
export QWEN_INTERIOR_CHECKPOINT_MAX_TOKENS=32768
export QWEN_LONG_CONTEXT=1
export QWEN_FP16_ACTIVATIONS=1
export QWEN_PREFILL_CHUNK_TOKENS=512
export QWEN_ACTIVE_KV_MAX_BYTES=8589934592
/Users/stewartbarteau/.local/share/uv/tools/mlx-dspark/bin/python -u \
  /Users/stewartbarteau/.local/share/qwen38-performance/run_server.py serve \
  --model /Users/stewartbarteau/models/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP \
  --mode dspark --max-draft 3 --host 127.0.0.1 --port 8001 \
  --api-key dummy --max-batch 1 --context-window 131072 --kv-bits 8 \
  --default-max-tokens 8192 --max-tokens-cap 8192 \
  --default-temperature 0 --default-top-p 1 --default-top-k 1 \
  --prefix-cache-slots 1 --prefix-cache-rungs 8192
```

Smoke quality: valid Python/Rust code. Dspark emits exact acceptance and throughput in `x_mlx_dspark`.

## Rapid-MLX

Qualified to boot and serve the same model. The best recorded profile used TurboQuant K8/V4, 512-token prefill, pinned system prompt, 0.85 GPU allocation, and 41 GB resident limit. Rapid classified this hybrid checkpoint as `supports_spec_decode=false`; no acceptance score is claimed. CODE passed; a 256-token PLAN probe truncated.

### Reproduction command

```bash
/Users/stewartbarteau/.local/bin/rapid-mlx serve \
  /Users/stewartbarteau/models/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP \
  --served-model-name Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP \
  --host 127.0.0.1 --port 8001 --api-key dummy --no-mllm \
  --disable-model-downloads --yes --log-level INFO --resident-memory-limit-gb 41 \
  --gpu-memory-utilization 0.85 --context-length 131072 --max-prompt-tokens 122880 \
  --max-tokens 8192 --max-num-seqs 1 --max-concurrent-requests 4 \
  --stream-interval 1 --prefill-batch-size 1 --completion-batch-size 1 \
  --prefill-step-size 512 --scheduling-policy shortest_validated_tail \
  --scheduling-max-deferrals 8 --shared-prefix-wait-tokens 1024 \
  --enable-prefix-cache --prefix-cache-index radix --cache-memory-mb 1024 \
  --prefix-cache-size 4 --hybrid-cache-entries 2 --pin-system-prompt \
  --idle-cache-clear-seconds 900 --response-cache-entries 0 \
  --kv-cache-turboquant k8v4 --default-temperature 0 --default-top-p 1 \
  --default-top-k 1 --default-min-p 0 --default-reasoning-effort medium \
  --tool-call-parser qwen3_coder_xml --enable-auto-tool-choice \
  --no-spec-decode --timeout 1800
```

## oMLX

Qualified to discover, load, and serve the same MoE through oMLX 0.7.0. The model is now attached to the multi-model server on `127.0.0.1:8000`.

Observed non-streaming CODE probe: TTFT `0.13 s`, generation `87.74 tok/s`, prompt processing `288.6 tok/s`, valid Python output, `85` completion tokens.

### Reproduction command

```bash
/Users/stewartbarteau/.omlx/runtime/run-pinned-omlx.sh \
  --model-dir /Users/stewartbarteau/models --host 127.0.0.1 --port 8000 \
  --max-concurrent-requests 1 --memory-guard aggressive --memory-guard-gb 41 \
  --paged-ssd-cache-dir /Users/stewartbarteau/.omlx/cache/paged_ssd \
  --paged-ssd-cache-max-size 100GB --hot-cache-max-size 8GB \
  --initial-cache-blocks 256 --api-key dummy
```

The API exposes all discovered models; selecting this exact model ID on the first request attaches/warms it. No oMLX speculation or acceptance score is claimed yet.

## SGLang

The MLX backend is present and passed its runtime gate: Torch `2.13.0`, MLX `0.32.4`, MPS and Metal available. The same MoE did not qualify: normal launch fails because the text-only checkpoint lacks `preprocessor_config.json`; `--language-only` reaches processor initialization and `--language-model-only` rejects `Qwen3_5MoeForConditionalGeneration` as unsupported. No throughput score is reported.

### Reproduction command and failure

```bash
SGLANG_USE_MLX=1 SGLANG_MLX_CACHE_LIMIT_GB=8 SGLANG_MLX_CLEAR_CACHE_STEPS=256 \
  /Users/stewartbarteau/venvs/sglang/bin/python -m sglang.launch_server \
  --model-path /Users/stewartbarteau/models/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP \
  --host 127.0.0.1 --port 8018 --context-length 131072 \
  --disable-cuda-graph --disable-overlap-schedule --tp-size 1 --language-model-only
```

Failure: `--language-model-only does not support Qwen3_5MoeForConditionalGeneration`.

### SGLang MLX flag audit

The MLX-enabled `sglang.launch_server --help` output was captured from the installed checkout and exited successfully. It enumerates 1000+ help lines. The only relevant language/vision controls present are:

```text
--enable-multimodal
--encoder-only
--language-only
--language-model-only
--encoder-urls
--encoder-bootstrap-port
--encoder-register-urls
--image-processor-backend
--disable-fast-image-processor
--mm-processor-worker-num
--mm-process-config
--mm-feature-transport
```

There is no `--no-llm`, `--disable-llm`, `--no-mllm`, `--disable-mllm`, `--no-vision`, `--disable-vision`, `--disable-visual-tower`, `--no-visual-tower`, `--disable-image-encoder`, `--no-image-encoder`, or `--disable-multimodal`. Each candidate was passed with a model path; argparse returned `unrecognized arguments`. `--enable-multimodal` is enable-only. `--language-only` and `--language-model-only` are specialized VLM/encoder modes, not a generic visual-tower-off flag.

## Measurement boundary

## DLite smoke

DLite was fetched at commit `e0aa3d5` and its CLI help/import surface passed. It is a PyTorch speculative-decoding package, not an MLX server integration; upstream currently documents Qwen3 and text-only Qwen3.5 targets, with released drafts for Qwen3.5-4B only. The attached target resolves as `Qwen3_5MoeForConditionalGeneration` (Qwen3.5 MoE), and no compatible DLite draft checkpoint is present locally. Therefore no model-generation score is claimed.

```bash
PYTHONPATH=/Users/stewartbarteau/src/DLite \
  /Users/stewartbarteau/venvs/sglang/bin/python -m dlite.cli --help
```

Comparisons require identical prompts, output cap, sampling, model revision, cache state, and quality/termination checks. Acceptance and cache metrics are reported only when the server exposes them.
