# Atoma Hyperbolic Active Mind — Reproducible Silicon Benchmark Report (v5.0)

- **Teacher Status**: `OFF` (Zero external `:8001` or cloud model calls; real `.safetensors` weights & standalone operators)
- **Manifold**: `(H^3 x T^2) / Gamma_12` (`Q32.31` deterministic fixed-point arithmetic)
- **CLI Invocation**: `orchestrator/target/release/atoma_eval_harness --suite all`

## 1. `ATOMA_OP_V3` Local Jacobian Rank Ablation on Real `.safetensors` Layer (`Muse-Glimmer-q4km-gs128-MLX (Layer 20 .safetensors)`)

- **Weight Tensor Digest (`FNV-1a 64-bit`)**: `0xd72c622fceea2d80` (`RMSNorm + Attention Q/O + SwiGLU MLP`)

| Chart Configuration | Active Rank | In-Chart Attn Cos | In-Chart MLP Cos | In-Chart Rel L2 Error | Spine Preservation Cos |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ATOMA_OP_V2` (Point Dyad) | `Rank-1` | `0.998682` | `0.999950` | `0.052766` | `1.000000` |
| `ATOMA_OP_V3` (Partial MGS) | `Rank-4` | `0.999275` | `0.999954` | `0.042699` | `1.000000` |
| `ATOMA_OP_V3` (Full 7-Ray) | `Rank-8` | `0.999994` | `0.999999` | `0.022903` | `1.000000` |
| `ATOMA_OP_V3` (Unspanned 2048D Isotropic) | `Rank-8` | `0.998719` (`delta`) | `x_out Cos=0.999929` | — | `1.000000` |

## 2. Cross-Model Specialist Subspace Accretion (Real `.safetensors` Checkpoints)

| Teacher Checkpoint (`.safetensors`) | Source Dim | Weight Digest | Target Atom | Rank Transition | Orthogonal Residual Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Qwen3-VL-Embedding-2B-6bit (vision_tower.blocks.20)` | `1024d` | `0x71f4f6ad90f79961` | `AtomId(0)` | `1 -> 2` | `0.2762` |
| `Qwen3.6-35B-A3B-OptiQ-4bit (layers.0)` | `2048d` | `0x9e4ebe238c91c2ce` | `AtomId(0)` | `2 -> 3` | `0.2555` |
| `Qwen3-Embedding-8B-4bit-DWQ (layers.20)` | `4096d` | `0x629fc0ee1be78e74` | `AtomId(0)` | `3 -> 4` | `0.2701` |
| `Muse-Glimmer-q4km-gs128-MLX (layers.20)` | `6656d` | `0x1b3c24ca83ff7845` | `AtomId(0)` | `4 -> 5` | `0.2760` |

## 3. Toroidal Isotropic Runtime `O(1)` Harmonic Field vs. Standard `O(N*D)` KV Cache (True Sequential Streaming)

- **Mathematical Note**: Standard Transformer KV cache storage scales as $O(N \cdot D)$ per layer (with $O(N^2 \cdot D)$ prefill attention compute). The Toroidal Harmonic Field maintains a constant $O(D)$ state (`56.01 KB`) implementing the positive circular-harmonic linear attention kernel $K(\theta_q,\theta_k) = (1 + \cos(\theta_q - \theta_k))^2$.

| Streamed Tokens (`N`) | Cumulative Prefill (`ms`) | Marginal Step (`us`) | Toroidal State (`KB`) | Standard FP16 KV Cache `O(N*D)` (`KB`) | Compression Ratio | Softmax Kernel Cos |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4096` | `33.2` | `12.79` | `56.01` | `16384.0` | **`292.5x`** | `0.7856` |
| `8192` | `70.2` | `13.53` | `56.01` | `32768.0` | **`585.1x`** | `0.7855` |
| `16384` | `153.9` | `12.46` | `56.01` | `65536.0` | **`1170.1x`** | `0.7855` |
| `32768` | `303.8` | `8.32` | `56.01` | `131072.0` | **`2340.2x`** | `0.7855` |
| `65536` | `613.9` | `12.42` | `56.01` | `262144.0` | **`4680.5x`** | `0.7855` |
| `131072` | `1215.5` | `13.80` | `56.01` | `524288.0` | **`9361.0x`** | `0.7855` |

## 4. GSM8K Multi-Step Mathematical Reasoning (`benchmarks/datasets/gsm8k_test.jsonl`)

- **Problems Evaluated**: `1319` (offset `0`)
- **Compiled via `ToroidalWorkingMemory -> MlirSsaBlock`**: `1226/1319` (`92.9%` coverage)
- **Verified `MlirSsaBlock` Precision**: `53/1226` (**`4.3%`**)
- **Overall Standalone Exact-Match Accuracy (Teacher: OFF)**: `53/1319` (**`4.0%`**)
- **Total `Q32.31` SSA Instructions Executed**: `8137` | **Mean Cold Latency**: `12.5 us` | **Mean Tier-1 Geodesic Replay**: `9119.1 us`

## 5. Multi-Length Needle-In-A-Haystack (`NIAH`) Matrix (`L5 + L11` Episodic Indexer)

| Context Length (Tokens) | `5%` | `25%` | `50%` | `75%` | `95%` | Row Accuracy | Mean Episodes Indexed | Mean Latency (`us`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4096` | `PASS` | `PASS` | `PASS` | `PASS` | `PASS` | **`5/5`** | `187` | `91.0` |
| `8192` | `PASS` | `PASS` | `PASS` | `PASS` | `PASS` | **`5/5`** | `373` | `174.8` |
| `16384` | `PASS` | `PASS` | `PASS` | `PASS` | `PASS` | **`5/5`** | `745` | `346.2` |
| `32768` | `PASS` | `PASS` | `PASS` | `PASS` | `PASS` | **`5/5`** | `1490` | `694.4` |
| `65536` | `PASS` | `PASS` | `PASS` | `PASS` | `PASS` | **`5/5`** | `2979` | `1393.8` |
| `131072` | `PASS` | `PASS` | `PASS` | `PASS` | `PASS` | **`5/5`** | `5958` | `2772.4` |

- **Overall Multi-Length NIAH Accuracy**: `30/30` (**`100.0%`**)

## 6. Multi-Length `RULER` Long-Context Reasoning Matrix

| RULER Task Category | `4096` | `8192` | `16384` | `32768` | `65536` | `131072` | Task Total | Mean Latency (`us`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S-NIAH-1 (Passkey)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `914.4` |
| **S-NIAH-2 (Number)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `849.5` |
| **S-NIAH-3 (UUID)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `856.4` |
| **MK-NIAH (Multi-Key)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `850.1` |
| **MV-NIAH (Multi-Value)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `884.9` |
| **MQ-NIAH (Multi-Query)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `879.9` |
| **VT-5Hop (Variable Chain)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `241.5` |
| **VT-Set (Multi-Var Bindings)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `248.3` |
| **CWE (Common Word Extraction)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `128.6` |
| **FWE (Frequent Coded Words)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `151.4` |
| **QA-Long (Document Synthesis)** | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | `3/3` | **`18/18`** | `1067.3` |

- **Overall Multi-Length RULER Accuracy**: `198/198` (**`100.0%`** across `198` total long-context evaluations)

## 7. HumanEval Zero-Leakage Prompt-Only Code Synthesis & Live `python3` Verification (`benchmarks/datasets/humaneval_test.jsonl`)

- **Answer Leakage Audit**: `canonical_solution` is **never read**; completions are synthesized strictly from `prompt` + `entry_point`
- **Problems Evaluated**: `164`
- **Cold Prompt-Only `python3` Unit-Test Pass@1**: `94/164` (**`57.3%`**)
- **Mean Cold Synthesis Latency**: `31.3 us` | **Post-Verification Tier-1 Replay Latency**: `19601.2 us` | **Mean `python3` Execution**: `20.1 ms`

## 8. LongBench-v2 Multi-Document & Long-Context Evaluation (`benchmarks/datasets/longbench_v2.jsonl`)

- **Items Evaluated**: `503` (`446.09 MB` raw text, `136785` episodic windows indexed in `2580 ms`)
- **Standalone Teacher-Off Episodic Accuracy (`L5 + L11 + L8`)**: `148/503` (**`29.42%`**)
- **Tier-2 Superposition Bridge -> Tier-1 Replay Speedup**: `16682 us -> 15178 us` (**`1.1x` speedup**)

| Breakdown Category | Correct / Total | Accuracy |
| :--- | :---: | :---: |
| Difficulty: `easy` | `57/192` | `29.7%` |
| Difficulty: `hard` | `91/311` | `29.3%` |
| Length: `long` | `25/108` | `23.1%` |
| Length: `medium` | `69/215` | `32.1%` |
| Length: `short` | `54/180` | `30.0%` |
| Domain: `Code Repository Understanding` | `11/50` | `22.0%` |
| Domain: `Long In-context Learning` | `28/81` | `34.6%` |
| Domain: `Long Structured Data Understanding` | `12/33` | `36.4%` |
| Domain: `Long-dialogue History Understanding` | `20/39` | `51.3%` |
| Domain: `Multi-Document QA` | `28/125` | `22.4%` |
| Domain: `Single-Document QA` | `49/175` | `28.0%` |

## 9. Autonomous Dream State Consolidation & Production Vault Telemetry

- **Dream Cycle Latency**: `14712 us` (`1` charts upgraded `Rank-1 -> Rank-8`, `+7` basis slots, `4` empty-space bridges minted, `1-cell` arcs `71 -> 165`, Gauss-Bonnet verified: `true`)
- **Live Production Vault**: `16945` `0-cells`, `91395` `1-cell` arcs, `310440` bytes (`0.30 MB`) `.wal`

---
*Generated deterministically on Apple Silicon in `12.31s` by `atoma_eval_harness` (v5.0, Teacher: OFF).*
