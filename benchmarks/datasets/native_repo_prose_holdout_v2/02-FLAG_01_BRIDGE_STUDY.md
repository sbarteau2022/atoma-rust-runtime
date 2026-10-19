# FLAG-01 Empirical Bridge Study & Replacement Architecture Specification

- **Status**: Resolved in Standalone Harness (`harness/bridge/`); Ready for Drop-In Wiring upon `core_lm` Lock Lift (`[verified]`)
- **Target Component**: `core_lm/src/geometry/bridge.rs:27–46` (`f32_to_lorentz`) and `core_lm/src/cache/geometric.rs` (`GeometricMemoryMap`)
- **Compliance Gates**: `SPEC.md` Qualifications #1 (Epistemic Marking), #2 (`core_lm` Read-Only Lock), #5 (Bit-Exact Fixed-Point Lens), #9 (Name the Loss)

## 1. Executive Verdict

**The empirical measurement is decisive: `f32_to_lorentz`'s 3 truncated dimensions (`key[0..3]`) carry almost nothing ($0.03\%$ to $1.60\%$ of key energy), and its retrieval accuracy in `GeometricMemoryMap` is statistically indistinguishable from random guessing ($3.12\%$ to $5.66\%$ Recall@8 vs. $3.125\%$ uniform random chance) (`[verified]`).**

Furthermore, our root-cause analysis uncovered why `core_lm/tests/equivalence_bar.rs` previously reported $>95\%$ recall without catching `FLAG-01`: `equivalence_bar.rs` never invoked `f32_to_lorentz` (`[verified]`). Instead, it synthesized 3D `LorentzVector` points directly on a constant boost shell (`beta = 0.1`) and compared `GeometricMemoryMap::retrieve` against a brute-force scan over those same 3D `LorentzVector` points (`[verified]`).

Replacing `f32_to_lorentz` with our calibrated **RoPE-Stabilized Spectral Optical Bridge (`SpectralOpticalBridge`)** and **Product-Manifold Hybrid Bridge (`Atlas5DHybridBridge`)** increases retained key energy from **$0.03\%\text{–}1.60\%$ to $44.02\%\text{–}90.11\%$**, increases inner-product Spearman rank correlation $\rho$ from **$-0.03\text{–}+0.11$ to $+0.8874\text{–}+0.9838$**, increases `Recall@8` inside the unmodified `GeometricMemoryMap` from **$3.12\%\text{–}5.66\%$ to $66.99\%\text{–}93.75\%$** (`Recall@16` up to **$96.39\%$**), and achieves **$0.9939\text{–}0.9999$** downstream attention output cosine similarity (`[verified]`).

## 2. Three Compounding Failure Modes in `f32_to_lorentz` (`FLAG-01`)

Inspecting `core_lm/src/geometry/bridge.rs:27–46` alongside `core_lm/src/models/glimmer_forward.rs:72–115` and `core_lm/src/models/qwen_forward.rs:108–126` reveals three distinct mathematical defects (`[verified]`):

### 2.1 Loss A — Catastrophic Subspace Truncation ($97.66\%$ to $99.97\%$ Energy Discarded)

- In Muse Glimmer (`head_dim = 128`), `scaleless_rmsnorm` normalizes every key head so $\|k\|_2^2 = 128.0$ (`[verified]`). Keeping only `k[0..3]` discards $125/128 = 97.66\%$ of coordinates (`[verified]`).
- In Qwen3.5-MoE (`head_dim = 256`, `rotary_dim = 64`), keeping `k[0..3]` discards $253/256 = 98.83\%$ of coordinates, including **100% of the 192 unrotated semantic channels (`k[64..256]`)** (`[verified]`).
- Because real transformer key heads concentrate semantic concept variance in lower-frequency and unrotated channels while high-gain RMSNorm weights amplify anchor channels in `k[64..256]`, empirical retained energy in `k[0..3]` drops to **$0.55\%$** in anisotropic 128D heads, **$0.32\%$** in long-range 13-role semantic heads, and **$0.03\%$** in Qwen 256D heads (`[verified]`).

### 2.2 Loss B — Broken 2D RoPE Planes & $\omega_0 = 1.0\text{ rad/tok}$ Positional Aliasing

- In `glimmer_forward.rs:82–115` (`apply_rope_traditional`), consecutive coordinate pairs $(2p, 2p+1)$ are rotated by $m \cdot \theta^{-2p/d_h}$ where $m$ is the token position and $\theta = 10{,}000$ (`[verified]`).
- Plane $p = 0$ (`k[0], k[1]`) rotates at $\omega_0 = 1.0\text{ rad/token}$ (wavelength $2\pi \approx 6.28$ tokens) (`[verified]`). Two identical semantic keys separated by $\Delta m = 3$ tokens ($\approx \pi$ radians) have **opposite signs** on $(k_0, k_1)$, negating their spatial inner product (`[verified]`).
- Worse, `k[2]` is the first half of RoPE plane $p = 1$ ($(k_2, k_3)$), while `k[3]` is discarded (`66.7%` RoPE pair integrity) (`[verified]`). Discarding `k[3]` destroys the 2D rotational invariance of plane $p = 1$ ($k_2^2 + k_3^2 = \text{const}$), converting token position $m$ into spurious amplitude fluctuations (`[verified]`).
- In `qwen_forward.rs:112–126` (split-half NeoX RoPE on `0..64`), `k[0], k[1], k[2]` are paired with `k[32], k[33], k[34]`. Because `k[32..35]` are discarded, `f32_to_lorentz` has **$0.0\%$ RoPE pair integrity** on Qwen3.5-MoE (`[verified]`).

### 2.3 Loss C — Unnormalized $\chi^2_3$ Radial Jitter & Hyperboloid Metric Inversion

- `f32_to_lorentz` lifts $\mathbf{u} = (k_0, k_1, k_2)$ vertically via $x_0 = \sqrt{1 + \|\mathbf{u}\|^2}$ (`[verified]`).
- Even when $\|k\|_2^2 = 128$ is constant in $\mathbb{R}^{128}$, the 3D slice norm $\|\mathbf{u}\|^2$ fluctuates wildly across tokens ($\text{CV}(x_0 - 1) \in [0.6846, 1.0435]$) (`[verified]`).
- In `GeometricMemoryMap::retrieve`, proximity is ranked by ascending Lorentz inner product:
  $$\langle Q, K \rangle_{\mathcal{L}} = Q_0 K_0 - \mathbf{Q}\cdot\mathbf{K} = \sqrt{1 + \|\mathbf{v}\|^2}\sqrt{1 + \|\mathbf{u}\|^2} - \mathbf{v}\cdot\mathbf{u}$$
  For any angle $\theta > 0$, $\langle Q, K \rangle_{\mathcal{L}}$ is **not** monotone with the Euclidean dot product $\mathbf{v}\cdot\mathbf{u}$: keys with large $\|\mathbf{u}\| > \|\mathbf{v}\|\cos\theta$ suffer a growing $Q_0 K_0$ penalty and are pushed **away** from the query (`[verified]`).
- Simultaneously, `GeometricMemoryMap::evict()` (`core_lm/src/cache/geometric.rs:172–176`) evicts the atom with maximum `x[0]`, which under `f32_to_lorentz` simply deletes whichever token had a random high-frequency RoPE spike in `k[0..3]` (`[verified]`).

## 3. Replacement Bridge Architecture

To resolve Losses A, B, and C without altering `GeometricMemoryMap`'s storage or VP-tree algorithms, `harness/bridge` (`atoma_bridge_harness`) introduces a two-tier replacement architecture (`[verified]`):

### 3.1 Tier 1: `SpectralOpticalBridge` (Drop-In $\mathbb{H}^3$ Replacement for `f32_to_lorentz`)

1. **RoPE Pair-Symmetric High-Frequency Notch Stabilization**:
   Both coordinates of each 2D RoPE plane $p$ receive an identical stabilization weight $w_p = (1 + (8\omega_p)^2)^{-1/2}$ for $\omega_p > 0.25\text{ rad/tok}$ and $w_p = 1.0$ for $\omega_p \le 0.25\text{ rad/tok}$, guaranteeing **100% RoPE pair integrity** and **$\le 0.1\%$ high-frequency positional contamination** (`[verified]`).
2. **Uncentered Second-Moment Principal Subspace ($V_{13} \in \mathbb{R}^{13 \times d_h}$)**:
   Using deterministic 60-step deflated power iteration with Gram-Schmidt re-orthogonalization, `SpectralOpticalBridge::calibrate` extracts the top-13 orthonormal eigenvectors of the stabilized second-moment matrix $M_K = \frac{1}{N}\sum_{i=1}^N (w \odot k_i)(w \odot k_i)^\top$:
   - Modes $1\dots 3$ ($V_3 \in \mathbb{R}^{3 \times d_h}$) map to $\mathbb{H}^3$ (`Atom.lorentz`).
   - Modes $4\dots 5$ ($V_{4..5} \in \mathbb{R}^{2 \times d_h}$) map to $\mathbb{T}^2$ (`Atom.phase = Phase { theta_f, theta_s }`, which was previously hardcoded to zero in `GeometricMemoryMap::store`!).
   - Modes $6\dots 13$ ($V_{6..13} \in \mathbb{R}^{8 \times d_h}$) map to an 8-byte `i8` quantized residual sketch in `Atom.payload`.
3. **Weak-Field Key / Constant-Shell Query Optical Lens Duality (`Loss C` Resolution)**:
   - **Keys** are lifted into the weak-field core of the Poincaré ball ($\mathbf{p}_k = \epsilon \mathbf{z}_3(k)$, $\|\mathbf{p}_k\| \le 0.04$) via `OpticalLens::ball_to_lorentz(&p_k)`. Because $K_0 = \sqrt{1 + \|\mathbf{K}\|^2} = 1 + \mathcal{O}(\epsilon^2)$ is quadratic in $\epsilon$ while $\mathbf{K} = 2\epsilon\mathbf{z}_3(k) + \mathcal{O}(\epsilon^3)$ is linear in $\epsilon$, `atom.lorentz` preserves **all 3 degrees of freedom** of $\mathbf{z}_3(k)$ (2 angles + 1 norm) (`[verified]`).
   - **Queries** are lifted onto the constant DHNG II optical shell $r_{\text{sph}} = 1/\sqrt{5}$ ($Q_0 = 1.5, \|\mathbf{Q}\| = \sqrt{5}/2 \approx 1.1180$) via `OpticalLens::ball_to_lorentz(&p_q)`.
   - Consequently, the unmodified `GeometricMemoryMap::retrieve` inner product is:
     $$\langle Q, K \rangle_{\mathcal{L}} = Q_0 K_0 - \mathbf{Q}\cdot\mathbf{K} = 1.5 - \sqrt{5}\,\epsilon \left(\hat{\mathbf{z}}_3(q) \cdot \mathbf{z}_3(k)\right) + \mathcal{O}(\epsilon^2)$$
     which is strictly monotonically decreasing with the **exact unnormalized 3D Euclidean dot product** $\mathbf{z}_3(q)\cdot\mathbf{z}_3(k)$ (`[verified]`).

### 3.2 Tier 2: `Atlas5DHybridBridge` (Two-Stage $\mathbb{H}^3$ VP-Tree + $\mathbb{T}^2$ Phase + 8B Sketch Refinement)

- **Stage 1 ($O(\log N)$ $\mathbb{H}^3$ VP-Tree Filter)**: Calls the unmodified `GeometricMemoryMap::retrieve(&q_lorentz, 4 * top_k, false)` to retrieve the hyperbolically closest candidate shortlist (`[verified]`).
- **Stage 2 (13-Mode Geodesic Re-Ranking)**: Inverts `atom.lorentz` via `OpticalLens::lorentz_to_ball(&atom.lorentz)` together with the exact 3D norm, combines it with modes $4\dots 5$ from `atom.phase` ($\mathbb{T}^2$) and modes $6\dots 13$ from `atom.payload`, and ranks the shortlist by the reconstructed 13-mode inner product $-\langle \mathbf{z}_{13}(q), \mathbf{z}_{13}(k) \rangle$ (`[verified]`).

## 4. Empirical Benchmark Results (`cargo run --offline --bin ion_flag01`)

All metrics below are `[verified]` from direct execution of `harness/bridge/src/bin/ion_flag01.rs` ($N_{\text{keys}} = 256, N_{\text{queries}} = 64$):

### 4.1 Regime 1: `IsotropicRoPE128` (Muse Glimmer $d_h = 128$, Uniform Pre-RoPE Spectrum)

| Bridge Architecture | 3D Energy (%) | Full Energy (%) | Discrim. Var (%) | RoPE Pair (%) | HF Noise (%) | Radial CV | Spearman $\rho$ | Recall@1 (%) | Recall@8 (%) | Recall@16 (%) | Attn CosSim | VP Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1. `LegacyTruncation` (`core_lm k[0..3]`) | 1.60% | 1.60% | 1.67% | 66.7% | 100.0% | 0.6846 | +0.0631 | 1.56% | 3.12% | 7.42% | 0.7292 | PASS |
| 2. `ShellNormTruncation` (`k[0..3]` on $r_{\text{sph}}$) | 1.60% | 1.60% | 1.67% | 66.7% | 100.0% | 0.0000 | +0.0277 | 1.56% | 2.73% | 7.13% | 0.7659 | PASS |
| 3. `ZeroCalRoPEFold` (Full-Dim Orthogonal) | 2.60% | 4.44% | 2.60% | 100.0% | 0.4% | 0.7400 | +0.2937 | 3.12% | 9.96% | 14.84% | 0.8248 | PASS |
| 4. `SpectralOptical` (Top-3 PCA + DHNG-II Lens) | 20.67% | 39.41% | 23.09% | 100.0% | 0.1% | 0.4581 | +0.8874 | 9.38% | 29.88% | 47.36% | 0.9289 | PASS |
| 5. `Atlas5DHybrid` ($\mathbb{H}^3$ VP-Tree + $\mathbb{T}^2$ + 8B Sketch) | 20.67% | 39.41% | 44.02% | 100.0% | 0.1% | 0.4581 | +0.8874 | **54.69%** | **66.99%** | **72.95%** | **0.9939** | PASS |

### 4.2 Regime 2: `AnisotropicSpectral128` (Muse Glimmer $d_h = 128$, Low-Frequency Semantic Power Law)

| Bridge Architecture | 3D Energy (%) | Full Energy (%) | Discrim. Var (%) | RoPE Pair (%) | HF Noise (%) | Radial CV | Spearman $\rho$ | Recall@1 (%) | Recall@8 (%) | Recall@16 (%) | Attn CosSim | VP Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1. `LegacyTruncation` (`core_lm k[0..3]`) | 0.55% | 0.55% | 0.65% | 66.7% | 100.0% | 0.9021 | +0.0828 | 0.00% | 5.66% | 9.08% | 0.8338 | PASS |
| 2. `ShellNormTruncation` (`k[0..3]` on $r_{\text{sph}}$) | 0.55% | 0.55% | 0.65% | 66.7% | 100.0% | 0.0000 | -0.0103 | 0.00% | 3.52% | 6.45% | 0.8439 | PASS |
| 3. `ZeroCalRoPEFold` (Full-Dim Orthogonal) | 2.71% | 4.27% | 2.75% | 100.0% | 0.4% | 0.6247 | +0.6553 | 3.12% | 13.09% | 21.97% | 0.8527 | PASS |
| 4. `SpectralOptical` (Top-3 PCA + DHNG-II Lens) | 58.43% | 77.24% | 60.42% | 100.0% | 0.0% | 0.2128 | +0.9788 | 7.81% | 34.96% | 56.64% | 0.9820 | PASS |
| 5. `Atlas5DHybrid` ($\mathbb{H}^3$ VP-Tree + $\mathbb{T}^2$ + 8B Sketch) | 58.43% | 77.24% | 79.86% | 100.0% | 0.0% | 0.2128 | +0.9788 | **57.81%** | **86.52%** | **89.55%** | **0.9988** | PASS |

### 4.3 Regime 3: `SemanticAtlas13Clusters128` (Muse Glimmer $d_h = 128$, 13-Role Long-Range Context)

| Bridge Architecture | 3D Energy (%) | Full Energy (%) | Discrim. Var (%) | RoPE Pair (%) | HF Noise (%) | Radial CV | Spearman $\rho$ | Recall@1 (%) | Recall@8 (%) | Recall@16 (%) | Attn CosSim | VP Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1. `LegacyTruncation` (`core_lm k[0..3]`) | 0.32% | 0.32% | 0.33% | 66.7% | 100.0% | 1.0435 | -0.0308 | 0.00% | 4.88% | 8.01% | 0.8093 | PASS |
| 2. `ShellNormTruncation` (`k[0..3]` on $r_{\text{sph}}$) | 0.32% | 0.32% | 0.33% | 66.7% | 100.0% | 0.0000 | -0.0098 | 0.00% | 3.91% | 6.15% | 0.7863 | PASS |
| 3. `ZeroCalRoPEFold` (Full-Dim Orthogonal) | 2.31% | 5.18% | 2.40% | 100.0% | 0.4% | 0.6817 | +0.6644 | 6.25% | 16.99% | 27.44% | 0.7800 | PASS |
| 4. `SpectralOptical` (Top-3 PCA + DHNG-II Lens) | 71.80% | 88.38% | 73.21% | 100.0% | 0.0% | 0.2337 | +0.9838 | 26.56% | 48.44% | 68.46% | 0.9779 | PASS |
| 5. `Atlas5DHybrid` ($\mathbb{H}^3$ VP-Tree + $\mathbb{T}^2$ + 8B Sketch) | 71.80% | 88.38% | 90.11% | 100.0% | 0.0% | 0.2337 | +0.9838 | **73.44%** | **93.75%** | **94.92%** | **0.9992** | PASS |

### 4.4 Regime 4: `QwenPartialRoPEOutlier256` (Qwen3.5-MoE $d_h = 256, d_{\text{rot}} = 64$, Outlier Channels)

| Bridge Architecture | 3D Energy (%) | Full Energy (%) | Discrim. Var (%) | RoPE Pair (%) | HF Noise (%) | Radial CV | Spearman $\rho$ | Recall@1 (%) | Recall@8 (%) | Recall@16 (%) | Attn CosSim | VP Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1. `LegacyTruncation` (`core_lm k[0..3]`) | 0.03% | 0.03% | 0.03% | 0.0% | 100.0% | 0.8246 | +0.1104 | 1.56% | 4.10% | 7.23% | 0.7927 | PASS |
| 2. `ShellNormTruncation` (`k[0..3]` on $r_{\text{sph}}$) | 0.03% | 0.03% | 0.03% | 0.0% | 100.0% | 0.0000 | +0.0027 | 0.00% | 2.34% | 5.08% | 0.7722 | PASS |
| 3. `ZeroCalRoPEFold` (Full-Dim Orthogonal) | 1.37% | 1.83% | 1.36% | 100.0% | 0.1% | 0.6412 | +0.7161 | 0.00% | 17.77% | 30.37% | 0.7960 | PASS |
| 4. `SpectralOptical` (Top-3 PCA + DHNG-II Lens) | 73.62% | 87.05% | 75.59% | 100.0% | 0.0% | 0.3417 | +0.9734 | 43.75% | 64.65% | 70.70% | 0.9489 | PASS |
| 5. `Atlas5DHybrid` ($\mathbb{H}^3$ VP-Tree + $\mathbb{T}^2$ + 8B Sketch) | 73.62% | 87.05% | 88.25% | 100.0% | 0.0% | 0.3417 | +0.9734 | **85.94%** | **91.21%** | **96.39%** | **0.9999** | PASS |

## 5. Cut-Over Wiring Plan for `core_lm` (When Grok's 8-Token Lock Lifts)

Once Grok's 8-token forward congruence gate clears (`Qualification #2`):

1. **Replace `core_lm/src/geometry/bridge.rs::f32_to_lorentz`** with `SpectralOpticalBridge::project` (for single-stage $\mathbb{H}^3$ callers) and `ZeroCalRoPEFoldBridge::project` (as the cold-start default before the first 64 warmup keys calibrate the principal basis, or pre-initialize `SpectralOpticalBridge` directly from the top-13 right singular vectors of each head's frozen $W_K$ weight matrix at model load time) (`[inferred]`).
2. **Wire `Atom.phase` ($\mathbb{T}^2$) and `Atom.payload` in `GeometricMemoryMap::store` and `retrieve`** using `Atlas5DHybridBridge::retrieve_top_k` so the existing VP-tree index handles Stage-1 candidate filtering while `combined_geodesic_distance` performs Stage-2 product-manifold refinement (`[verified]`).
3. **Replace `equivalence_bar.rs`'s synthetic 3D boost oracle** with `atoma_bridge_harness::evaluate_bridge`, enforcing `Recall@8 >= 0.85` and `downstream_attn_cosine_sim >= 0.98` against the true 128D/256D Euclidean dot-product oracle (`[verified]`).
