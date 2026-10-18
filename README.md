# atoma-rust-runtime

Deterministic Rust inference and product-manifold episodic graph memory runtime on $(\mathbb{H}^3 \times \mathbb{T}^2)/\Gamma_{12}$, combining `Q32.31` fixed-point hyperbolic geometry, 12-cluster Sparse Mixture-of-Experts (MoE) subgraph routing, Rank-8 Local Jacobian function approximation (`ATOMA_OP_V3`), Recurrent Linear Attention (Katharopoulos et al., ICML 2020), and crash-safe `.atoma_map` + `.atoma_map.wal` persistence.

Part of the ATOMA system — ScoutAi, powered by ATOMA.

---

## Prior Work & Verified Industry-Standard ML Terminology

All components in this repository use standard machine learning and numerical linear algebra terminology and explicitly credit foundational prior work:

- **Recurrent Linear Attention (`ToroidalHarmonicFieldState` in [`orchestrator/src/toroidal_attention.rs`](file:///Users/stewartbarteau/Desktop/monster-hybrid-server/orchestrator/src/toroidal_attention.rs)):**
  - **Prior Work:** **Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret (ICML 2020), *"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"*** (arXiv:2006.16236).
  - Expresses self-attention similarity as a linear dot product of kernel feature maps $k(q, k) = \langle \phi(q), \phi(k) \rangle$ and applies matrix product associativity $(\phi(Q)\phi(K)^\top)V = \phi(Q)(\phi(K)^\top V)$ to reduce sequence complexity from $O(N^2)$ to $O(N)$ with an $O(1)$ recurrent state $(S_t, Z_t)$.
  - Instantiates $\phi(\theta)$ as a 5-dimensional 2nd-order trigonometric feature map over 2D polar pairs from **Rotary Position Embeddings (RoPE; Su et al., 2024)** with exponential decay $\gamma \in [0.90, 1.0]$ (Peng et al., 2023).
- **Product-Manifold Graph Embedding & Sparse MoE Routing (`CarvedSwarmMap` in [`immune_system/implementation/src/silo.rs`](file:///Users/stewartbarteau/Desktop/monster-hybrid-server/immune_system/implementation/src/silo.rs)):**
  - **Prior Work:** **Gu, Sala, Gunel, & Ré (ICLR 2019), *"Learning Mixed-Curvature Representations in Product Spaces"***; **Nickel & Kiela (NeurIPS 2018)**; **Shazeer et al. (ICLR 2017)**.
  - Embeds graph nodes (`0-cells`) in $\mathbb{H}^3 \times \mathbb{T}^2$ and partitions them across 12 specialist clusters (`L1..L12`).
- **Rank-8 Local Jacobian Function Approximation (`CoupledOperatorAtomPayload` / `ATOMA_OP_V3` in [`immune_system/implementation/src/map_native.rs`](file:///Users/stewartbarteau/Desktop/monster-hybrid-server/immune_system/implementation/src/map_native.rs)):**
  - **Prior Work:** **Two-Pass Modified Gram-Schmidt (MGS)** orthogonalization (Björck, 1996) and local tangent-space chart approximation (Rifai et al., ICML 2011).
  - Stores a locked Rank-1 primary activation plus up to 7 orthonormal tangent basis vectors (`r = 2..=8`) and directional derivatives per operator node.

---

## Status & Proven-vs-Open Ledger

- **Verified (`atoma_immune` — 51 / 51 library tests passing):**
  - `Q32.31` fixed-point arithmetic, CORDIC trigonometry, Lorentz hyperboloid boosts, and Poincaré ball $\leftrightarrow$ hyperboloid isometry (`4.4e-9` agreement across 900 pairs).
  - `Atlas13` 12-cluster topology, $\mathrm{SO}(3)$ Wigner frame parallel transport, and discrete loop consistency verification.
  - `ATOMA_OP_V3` Rank-8 Two-Pass Modified Gram-Schmidt subspace accretion and 7-ray tangent perturbation sweeps (`immune_system/implementation/src/map_native.rs`).
  - `GeodesicActiveTransformer` (`immune_system/implementation/src/continuous_active_mind.rs`): reversible `BoundaryLexicalTransducer` (`ATOMA_LX`), $O(1)$ `GeodesicCursor`, strict `chain_id` trajectory verification, egress coherence filtering (`is_coherent_egress_text`), immutable base trajectory locking (`locked_arc_count`), atomic `.atoma_map` snapshots, and crash-safe `.atoma_map.wal` recovery.
- **Verified (`atoma_orchestrator` — 14 / 14 library tests + 3 / 3 end-to-end qualification audit tests passing):**
  - Loopback/TLS HTTP edge (`orchestrator/src/http_api.rs`), `ToroidalHarmonicFieldState` + `ToroidalIsotropicRuntime` (`orchestrator/src/toroidal_attention.rs`), and `MainModelCoordinatorRuntime` (`orchestrator/src/runtime_model.rs`).
  - End-to-end serving-path qualification (`orchestrator/tests/qualification_audit.rs`):
    1. Honest `/api/map-state` provenance (`qwen_shard_connected: false`, `teacher_reachable: false`, `teacher_mode: "standalone_disabled"`).
    2. Multi-turn session hygiene (`Turn 1 -> Turn 2 -> Turn 3` without cross-turn prefix pollution) and full-length Tier-1 trajectory replay (`Tier1TrajectoryWalk`).
    3. Map learning/replay ablation (`set_map_learning_and_replay(false)` vs `true`): proves `0/6 -> 6/6` recall gain on unseen paraphrases and multi-concept bridges when map replay is enabled, alongside `10/10` pass rate on narrative/numeric distractors, contradictory premises (`#### CONTRADICTION`), and unanswerable premises (`#### UNANSWERABLE`).
    4. Disposable-vault persistence & isolation: atomic survival across interrupted `.tmp` writes, fail-closed SHA-256 rejection of truncated vaults, automatic repair/truncation of torn `.atoma_map.wal` tails across restarts, 8-thread concurrent turn execution, and cross-session coreference isolation.

---

## Live Server Architecture & Deployment Provenance

| Attribute | Standalone Product-Manifold Server (`:8088`) | Local MLX Model Server (`:8001` / `:8000`) |
| :--- | :--- | :--- |
| **Role** | Standalone Graph Memory, Chat, Code & `/api/dream` Server (`External Teacher OFF`) | Local `Qwen3.8-27B-Uncensored-OrcaRouter-MLX-4bit` MLX Server for Zed / Rapid-MLX / oMLX |
| **Binary / Process** | `/Users/stewartbarteau/Desktop/argon/server/orchestrator/target/release/monster_hybrid_server` (synced with `Desktop/monster-hybrid-server`) | `Rapid-MLX` (`127.0.0.1:8001/v1`) & `oMLX` (`127.0.0.1:8000`) |
| **Active Vault** | `design/global_active_mind.atoma_map` (`17,843` `0-cells`, `9,197` `ATOMA_OP_V3` charts, `7,304` full `Rank-8` charts, `91,422` locked verified `1-cell` arcs) | `/Users/stewartbarteau/models/Qwen3.8-27B-Uncensored-OrcaRouter-MLX-4bit` |
| **Trajectory Locking** | `locked_arc_count = 91,422` (immutable base); new live chat turns append to `.atoma_map.wal` | Stateless MLX weights |

---

## Quickstart & Verification Commands

```sh
# 1. Run all 51 library tests in atoma_immune
cargo test --manifest-path immune_system/implementation/Cargo.toml --lib

# 2. Run all 14 library tests in atoma_orchestrator
cargo test --manifest-path orchestrator/Cargo.toml --lib

# 3. Run the 3 end-to-end qualification audit suites
cargo test --manifest-path orchestrator/Cargo.toml --test qualification_audit -- --nocapture

# 4. Rebuild the clean Rank-8 vault from all 3 local model checkpoints (optional; already built)
./orchestrator/target/release/monster_hybrid_server --rebuild-clean-rank8-vault

# 5. Start the standalone server on :8088 with live trajectory learning enabled
./orchestrator/target/release/monster_hybrid_server --serve 127.0.0.1:8088 --learn-query
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
