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

Server ablation scorecards and reproducible full command lines are recorded in [benchmarks/SERVER_SCORECARDS_2026-10-21.md](benchmarks/SERVER_SCORECARDS_2026-10-21.md). The current quality-constrained winner is native Dspark with `max-draft 3`; Rapid-MLX and oMLX both serve the same MoE, while SGLang’s MLX backend currently rejects this Qwen3.5 MoE architecture in language-only mode.

- **Verified (`atoma_immune` — 56 / 56 library tests passing):**
  - `Q32.31` fixed-point arithmetic, CORDIC trigonometry, Lorentz hyperboloid boosts, and Poincaré ball $\leftrightarrow$ hyperboloid isometry (`4.4e-9` agreement across 900 pairs).
  - `Atlas13` 12-cluster topology, $\mathrm{SO}(3)$ Wigner frame parallel transport, and discrete loop consistency verification.
  - `ATOMA_OP_V3` Rank-8 Two-Pass Modified Gram-Schmidt subspace accretion and 7-ray tangent perturbation sweeps (`immune_system/implementation/src/map_native.rs`).
  - `GeodesicActiveTransformer` (`immune_system/implementation/src/continuous_active_mind.rs`): reversible `BoundaryLexicalTransducer` (`ATOMA_LX`), $O(1)$ `GeodesicCursor`, strict `chain_id` trajectory verification, egress coherence filtering (`is_coherent_egress_text`), immutable base trajectory locking (`locked_arc_count`), atomic `.atoma_map` snapshots, and crash-safe `.atoma_map.wal` recovery.
- **Verified (`atoma_orchestrator` — 35 / 35 library, 13 / 13 full-chain, 4 / 4 map-toroidal, 3 / 3 native-predictor integration, and 3 / 3 qualification-audit tests passing):**
  - Loopback/TLS HTTP edge (`orchestrator/src/http_api.rs`), `ToroidalHarmonicFieldState` + `ToroidalIsotropicRuntime` (`orchestrator/src/toroidal_attention.rs`), and `MainModelCoordinatorRuntime` (`orchestrator/src/runtime_model.rs`).
  - End-to-end serving-path qualification (`orchestrator/tests/qualification_audit.rs`):
    1. `/api/map-state` reports shard availability separately from a reachable teacher endpoint; `qwen_shard_connected` is true only when both are present.
    2. Multi-turn session hygiene (`Turn 1 -> Turn 2 -> Turn 3` without cross-turn prefix pollution) and full-length Tier-1 trajectory replay (`Tier1TrajectoryWalk`).
    3. Map learning/replay ablation (`set_map_learning_and_replay(false)` vs `true`): verifies map replay improves recall and reaches `6/6` unseen paraphrase and multi-concept bridge cases, alongside `10/10` pass rate on narrative/numeric distractors, contradictory premises (`#### CONTRADICTION`), and unanswerable premises (`#### UNANSWERABLE`). The replay-disabled baseline may answer some items through non-map paths.
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

Live upstream exchanges are recorded in the configured `.atoma_map.wal` and snapshot before the
runtime returns a successful ingestion receipt. WAL and snapshot I/O failures are returned as
errors; the HTTP proxy logs an ingestion failure because it may already have streamed the upstream
response to the client. A successful upstream answer therefore does not imply that map learning
was persisted; check the server log for `[live exchange persistence failed]`.

WAL replay failures now stop task and subagent execution before it proceeds with potentially stale
map state. Active-inference bridge updates also surface snapshot failures instead of reporting a
successful step. The `design/DESIGN.md` Erlang/OTP, FlatBuffers, ANE, and SSD-paged KV items remain
target architecture requirements; they are not claims about the current Rust/HTTP implementation.

Typed central-map K/V ingestion compensates for refraction's amplitude phase offset so the stored
atom phase matches the encoded key phase. The toroidal attention qualification tests cover this
phase invariant alongside causal, role, layer, head, and sequence filtering.

`ToroidalIsotropicRuntime::full_stack` now allocates an independent recurrent toroidal state for
every declared layer (at least 52) and the server uses that dense layer coverage. This is a layer
coverage change, not by itself a qualified standalone language-model forward pass. The legacy
map decoder still has a Qwen-named sketch path and captured map layer effectors; do not interpret
full-stack allocation as proof of teacher-free language quality.

The native boundary exposes `transduce_utf8_to_mlir` / `transduce_mlir_to_utf8`: direct UTF-8 byte
carriers folded through Atomic IR, with no BPE/model tokenizer. `input_embeddings.rs` maps each
carrier through `NativeReader` into six H³×T² coordinates (four Lorentz coordinates plus two
toroidal phases). `execute_live_operational_turn` feeds that byte sequence through a per-session
full-depth toroidal recurrence and retains its final and per-position states for diagnostic readout. This is a
native geometric representation, not a learned semantic embedding. The output head now reads each
torus hidden state to its nearest canonical byte-carrier address and passes those IDs through the
strict UTF-8 reader, with periodic T² phase distance. Unknown atom IDs are rejected rather than dropped.
`/api/turn` exposes `native_output_candidate` or `native_output_error` and labels the operation
`native_output_kind: "prompt_readout"`, `native_output_is_continuation: false`. It remains an
untrained diagnostic and does not replace the primary response. The legacy vault has only 219
distinct checkpoint-token layer-0 sketches; they are not substituted for native byte embeddings.
The teacher-off response-quality gate remains open, so the primary returned text still uses the
legacy map/teacher paths.

The historical untrained [native qualification report](benchmarks/NATIVE_TOROIDAL_REPORT.md) measures
**0/24 completion references matched** with no teacher, checkpoint, vault, or tokenizer lexicon
loaded. That diagnostic readout has no trained next-carrier prediction or EOS. Regression tests establish
component behavior, not language quality. The 52-field standalone path now distributes its
residual mix over the stack to prevent first-carrier collapse; this preserves more incoming
information but does not create a language model. `/reset` clears all native session caches.
Recurrent fields have fixed size; the per-position diagnostic capture uses O(prompt length)
storage per session. The [qualification contract](docs/NATIVE_TOROIDAL_QUALIFICATION.md) records
the boundaries and reproducer.

The separate [`native_toroidal` producer](orchestrator/src/bin/native_toroidal.rs) now trains,
saves, loads, evaluates, and generates with a learned **next-byte/EOS** head over the native
toroidal state. It uses direct UTF-8 → ML-IR conversion, 52 fixed recurrent fields, and a
76,329-parameter readout. Targets are shifted, prompt loss is masked, and EOS is learned.
Predicted bytes feed back through the same fields; EOS or an explicit byte budget stops
generation. Corrupt/incompatible artifacts, overlapping training/evaluation documents, and
invalid UTF-8 are reported explicitly. No teacher, checkpoint, vault, tokenizer lexicon,
response template, grammar mask, or repetition correction participates in this path.
Only the head is trained. Its [design](docs/NATIVE_NEXT_ATOM_DESIGN.md) and
[measured report](benchmarks/NATIVE_NEXT_ATOM_REPORT.md) distinguish training/generation
mechanics from language quality. The trained head still matches **0/24** completion references;
reliable natural-text EOS stopping remains open. The primary HTTP/server response still uses
the legacy cascade; the native producer is available through its standalone CLI and Rust API.

The native `core_lm` crate uses Metal 0.33. Its transitive `block` 0.1.6 dependency is locally
patched to use an inhabited opaque C-layout type and explicit C ABIs, removing its future
incompatibility and ABI warnings without changing the block layout.

```sh
# 1. Run all 56 library tests in atoma_immune
cargo test --manifest-path immune_system/implementation/Cargo.toml --lib

# 2. Run all 35 library tests in atoma_orchestrator
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
