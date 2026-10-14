# `atoma_bridge_harness` — FLAG-01 Empirical Energy Measurement & Replacement Bridge Harness

## Purpose

Resolves **FLAG-01** (`core_lm/src/geometry/bridge.rs:27–46`), the primary open architectural defect identified in `immune_system/audit/ADVERSARIAL_AUDIT.md`. Specifically:

1. **Measures actual retained energy and retrieval fidelity** when real post-RMSNorm, post-RoPE transformer key vectors (`head_dim = 128` for Muse Glimmer `glimmer_forward.rs:72–115` and `head_dim = 256, rotary_dim = 64` for Qwen3.5-MoE `qwen_forward.rs:108–126`) pass through `core_lm::geometry::bridge::f32_to_lorentz` (`[verified]`).
2. **Isolates the three compounding failure modes of `f32_to_lorentz`** (`[verified]`):
   - **Loss A (Subspace Truncation)**: Discards 125/128 ($97.66\%$) to 253/256 ($98.83\%$) of dimensions, retaining only $0.03\%$ to $1.60\%$ of key energy across realistic spectral regimes.
   - **Loss B (Broken RoPE Plane & High-Frequency Positional Aliasing)**: `key[0..3]` lies $100\%$ inside the fastest-oscillating RoPE planes ($\omega_0 = 1.0\text{ rad/tok}$, wavelength $6.28$ tokens), splits Glimmer plane $p=1$ in half ($k_2$ kept, $k_3$ dropped $\to 66.7\%$ pair integrity), and splits three NeoX pairs in Qwen3.5-MoE ($k_0, k_1, k_2$ kept without $k_{32}, k_{33}, k_{34} \to 0.0\%$ pair integrity).
   - **Loss C ($\chi^2_3$ Hyperboloid Radial Jitter)**: Unnormalized vertical lifting $x_0 = \sqrt{1 + \|k_{0..3}\|^2}$ injects a radial coefficient of variation $\text{CV}(x_0 - 1) \in [0.68, 1.04]$ that dominates the Lorentz inner product $\langle Q, K \rangle_{\mathcal{L}} = Q_0 K_0 - \mathbf{Q}\cdot\mathbf{K}$ and corrupts `GeometricMemoryMap::evict()`.
3. **Proves the replacement bridge architecture (`SpectralOpticalBridge` and `Atlas5DHybridBridge`)** against the unmodified `core_lm::cache::geometric::GeometricMemoryMap` (both linear scan and VP-tree index) with zero bytes changed in `core_lm/` (`Qualification #2`, `[verified]`).

## Inputs/Outputs

- **Inputs**:
  - Post-RMSNorm, post-RoPE key, query, and value head tensors across four representative regimes (`IsotropicRoPE128`, `AnisotropicSpectral128`, `SemanticAtlas13Clusters128`, `QwenPartialRoPEOutlier256`).
- **Outputs**:
  - `BridgeProjection`: Certified `LorentzVector` ($\mathbb{H}^3$), `Phase` ($\mathbb{T}^2$), retained 3D/5D/13D energy accounting, and 16-byte compact payload (`z3_norm: f32`, `residual_sketch: [i8; 8]`, `token_index: u32`).
  - `BridgeBenchmarkReport`: Verified Markdown diagnostic tables emitted by `src/bin/ion_flag01.rs`.

## Dependencies

- `atoma_immune` (`../../immune_system/implementation`), which mounts `core_lm/src/geometry/`, `core_lm/src/cache/geometric.rs`, and `core_lm/src/lock.rs` read-only (`[verified]`).
- Zero external crate dependencies (`--offline` build and execution).

## Qualification

Run the unit gates and full 20-configuration empirical study offline:

```bash
cargo test --offline
cargo run --offline --bin ion_flag01
```

All 3 test suites and 20 regime-bridge evaluations must pass with `vptree_index_exact_parity == true` and `max_lorentz_invariant_err < 1e-6` (`[verified]`).
