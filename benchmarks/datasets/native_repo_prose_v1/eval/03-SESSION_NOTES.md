# Session Notes — 2026-10-13

Handoff notes for agents working this repo. The geometry has been fought over — every claim below is marked. Respect the marks.

- **[verified]** — read directly from source on this machine, 2026-10-13
- **[reported]** — from chat/memory, not re-verified today
- **[stewart]** — Stewart's own stated position

## 1. What this repo is

**[verified]** Deterministic Rust inference server research. `core_lm/` holds a Muse Glimmer forward pass (52 layers **[reported]**) with the KV cache replaced by per-layer, per-KV-head geometric memory maps. Q32.31 fixed-point geometry shell. No Python, no JSON in the server path.

**[verified]** Git history starts 2026-10-13 (commit `b39a905`). The prior month of work was local-only and unversioned before that.

## 2. The geometric memory map — as implemented in code

**[verified**, from `core_lm/src/cache/geometric.rs`]

- `GeometricMemoryMap`: `HashMap<u64, Atom>`, `max_atoms` (1,000,000 per map in the forward pass), monotonic `AtomId`, `LogicalClock`, `audit_log`, `use_picard` flag.
- `Atom`: `id`, `clock` (write order; position derived as `clock - 1`), `phase` (T²), `lorentz` (the address), `payload_ref` (raw K/V bytes, 2×128 f32s), `pinned`.
- `write`: validates the address on the hyperboloid, optionally reduces to the Picard fundamental domain (**OFF** in the forward pass: `use_picard = false`), evicts at capacity, stamps id+clock, audit-logs.
- `retrieve`: linear scan, scores by Lorentz inner product **ascending** (smaller = nearer in H³ — this ordering was a real bug, corrected 2026-10-13; the inverted sort failed the retrieval contracts), `AtomId` tie-break, top-k.
- `evict`: victim = max `x[0]` among unpinned atoms, audit-logged.
- `Phase { theta_f, theta_s }`: currently **ZERO placeholders**. `glimmer_forward.rs`: "T² phases are ZERO placeholders until lock.rs supplies (n,m,ψ*)".

**[verified**, from `core_lm/src/geometry/`]

- Address space **in the code**: H³ in the Lorentz model, 4 coords, x₀²−x₁²−x₂²−x₃²=1, all Q32.31. `validate()` tolerance 0.1 for fixed-point noise.
- Bridge (`f32_to_lorentz`): keeps **only the first 3 components** of the 128-dim K head as spatial coords; x₀=√(1+|spatial|²). This projection is lossy by construction — 125 of 128 dims are discarded at the address.

**[verified**, from `core_lm/src/glimmer_forward.rs`]

- One map per layer per KV head. `use_picard = false`. RoPE applied **before** the bridge (`apply_rope_neox`, gated by `config.use_rope(layer)`). `TorPhaseEncoder` is **not** used in the forward path.
- Write path: K head → Lorentz address; K+V bytes → payload.
- Read path: Q head → Lorentz query; `retrieve(top_k = min(512, n_atoms))`; sliding-window mask by clock-derived position; then **exact dot-product + softmax over the retrieved candidates**. The map proposes candidates; dot products score them. Geometric scores do not weight the output.
- Shape handling for packed q4 weights: `q_cols * 8` must equal `hidden_size`; `q_fused` auto-detected from `q_proj` row count.

## 3. lock.rs / CORDIC

**[reported**, from main-chat work 2026-10-13] Three defects were found and fixed in the pasted `cordic_sin`: radian-vs-pi unit mismatch in `ATAN_TABLE`, a sign bug in the fold branch, a wrong gain constant. Repaired version verified in VM: max error 9.776e-9 (19.5 LSBs) over 200k random inputs. Fixes are in `core_lm/src/lock.rs` and `core_lm/src/kuramoto_tile.metal`.

## 4. Baselines (Rapid + Glimmer q4, target-only)

**[reported** — Stewart's runs, 2026-10-13. These are the **incumbent** numbers to beat, **not** this repo's output.]

- N=2 same-domain: TTFT mean 0.20s, per-stream decode 16.9 tok/s, aggregate 33.3 tok/s.
- N=2 mixed-domain: TTFT mean 0.63s, per-stream decode 16.5 tok/s, aggregate 31.5 tok/s.
- Independent agent run: N=1 17.7 tok/s, 116ms TTFT; N=2 32.0 tok/s aggregate.
- DFlash experiment: **disqualified** — crashed in the processor path before emitting a token. 0% acceptance is not a performance result.

## 5. Proven vs open — read before touching the geometry

- **[verified]** Map storage contracts (write / retrieve / evict / audit / determinism) hold under test. This certifies the **data structure**, not attention equivalence.
- **[reported]** Isolated VM validation: 3D/5D geometric nearest-neighbor recall 9%/6% vs 128D attention. **Scope**: that instrument only (first-3-components projection, static frame). It does **not** cover Stewart's map form (§6).
- **[reported]** Last forward-pass output was incoherent. Cause unknown — bridge, filter, or upstream.
- **[open]** `generate_glimmer` coherence. Preregistered order: (1) greedy output congruence vs the q4 MLX baseline on a fixed prompt set, (2) memory footprint at 32K/96K context, (3) throughput. Congruence first.

## 6. Stewart's map form vs the code — DO NOT CONFLATE

**[stewart]** His Manifold Map: H³/Γ₁₂ (index-12 torsion-free Whitehead link subgroup), memories in the **active token's moving frame** (one Q32.31 boost per token), addressing by (n,m,ψ*) pending his `lock.rs`. Visualized as a punctured Poincaré ball.

**[verified]** The code does **not** implement this yet: `use_picard = false` (no quotient), no moving-frame boost in the write path, phases zeroed. The code is the static instrument, not the Manifold Map. Verdicts about the code's geometry apply to the code **only** — never to his map form.

## 7. Build & run

```sh
cd ~/Desktop/monster-hybrid-server/core_lm
cargo build
cargo run --bin generate_glimmer -- "The capital of France is" 50
```

Model path is hardcoded: `~/models/Muse-Glimmer-q4-MLX`.

## 8. Open questions for whoever works this next

1. Does `generate_glimmer` produce coherent output? (congruence test vs baseline)
2. If incoherent — bridge, filter, or forward pass?
3. Measured memory footprint vs a standard KV cache at 32K/96K?
4. (n,m,ψ*) addressing awaits Stewart's `lock.rs` — do not invent it.
