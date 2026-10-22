# Elle-Worker → Rust Model Transfer Matrix

**Directive (Stewart, 2026-10-19):** the Rust model is the automatons' true home;
components built in the elle-worker for it get extracted. **Caution (Stewart,
same night):** the parent folder holds substantial other work we do NOT want to
lose. Nothing in this matrix authorizes any deletion. Every DROP is a proposal
for explicit review, never an action.

**Out of scope entirely (sibling work in the parent folder — do not touch):**
ATLAS_MODEL, Atlas Analytics Rebrand, Dynanic-Hyperbolic-Neural-Graph,
TorPhPoEnc, apps/ (incl. elle-law), harmonizer-edge,
hybrid-HNPA-active-inference-LLM, tit.py, rope.py, the formal-gates and
lemniscate test files.

**Legend:** PORT = code moves (numeric core, substrate-free). REIMPLEMENT =
the design crosses, the code is rebuilt Rust-native. KEEP = stays in the
worker; it is an application, not a model component. DROP = retire proposal —
requires Stewart's explicit sign-off per item.

---

## Already transferred (2026-10-19)

| Source | Destination | Status |
|---|---|---|
| `predict.ts` (ProductPredictor 𝒫) | orchestrator `manifold.rs` + `predict_gate.rs` + `TrajectoryMatchGate` | PORTED. Live in turn path. Note: TS-only helper `surpriseToUnit` was not in the Rust harness original; not ported. |
| `calc.ts` (deterministic arithmetic) | orchestrator `math_eval.rs` + number-word front end | PORTED. Live; takes precedence over stored trajectories. |
| `regulator.ts` (free-energy regulator) | orchestrator `free_energy.rs` + per-session ledgers in `execute_live_operational_turn` | PORTED. Turn coherence measured from native evidence (structural=tier, relational=gate verdict, harmonic=holonomy_verified); receipts carry computed F + dissonance. Replaces hardcoded 0.12/0.96. |
| `oracle.ts` (bet ledger + scars) | orchestrator `outcome_ledger.rs` | PORTED. Append-only binary log beside the vault (`<vault>.outcomes`). Every served turn files a bet; `mark wrong` / `mark right` resolve the latest; `calibration` prints the curve. An incorrect resolution is a SCAR: that prompt can never walk Tier1 again (scar check in the exact-chain gate). |
| `falcon-ground.ts` refusal discipline | `LiveOperationalTurnReceipt.refusal_reason` | PORTED (partial). Gate rejections and empty syntheses now name their cause diagnosably. |
| `pami.ts` core (index/distance/resonance) | orchestrator `pami.rs` | PORTED (module only). The worker's store/tool layers NOT ported (worker substrate). Wiring into the gate as the second relevance channel is deferred until arc provenance exists at the gate point. |

## PORT candidates (pure cores, verified substrate-free)

| Source | Lines | Rust-model problem it solves | Notes |
|---|---|---|---|
| `regulator.ts` (free-energy regulator) | 346 | **Receipts lie.** `runtime_model.rs` hardcodes `total_free_energy_q32 = 0.12` and `min_omega_readiness_q32 = 0.96`. The regulator computes F = U − T·S for real: φ-weighted invariant costs (structural/relational/harmonic coherence), conserved F+W ledger, anisotropy suppression, dissonance residual. | Highest-value unported core. Feeds real values into fields that already exist on the receipt. |
| `harmonic-coherence.ts` (four verdicts) | 168 | Gate verdicts conflate consistency with grounding. Port the taxonomy: incoherent / consistent_only / ungrounded_consistent / grounded, with `grounded` unreachable without a second channel. | Rides on the existing TrajectoryMatchGate receipts. |
| `observer-falsifier.ts` (κ falsifier) | 151 | Gate thresholds are asserted, not earned. Port as a qualification harness: pre-registered Spearman vs permutation null, power floor, PASS/NULL/UNDERPOWERED. | Methodology port; pairs with `kappa-memory/validate.ts` (AUC/Mann–Whitney, KILL/BUILD/INSUFFICIENT, gate flips stay human commits). |
| `coherence-layer.ts` (small-world measurement) | 197 | No measurement of how the recognition layer shortens retrieval paths in the swarm map. Pure BFS. | Diagnostic, not turn-path. |
| `fixed-math.ts` (fixed-point CORDIC) | 194 | Cross-check candidate for the Q32_31 cordic (known erratum: gain bias ~29.8k LSBs). TS reference implementation to diff against the Rust core. | Verify before trusting either side. |

## REIMPLEMENT candidates (design crosses)

| Source | Design | Notes |
|---|---|---|
| `oracle.ts` (bet ledger + calibration) | Outcome records for served answers: filed confidence (tier), resolved later, calibration curve, misses written back as scars. | Monster home: `.atoma_map.wal` outcome frames. Fixes "wrong trajectory walks at full confidence." |
| `pami.ts` + `pami-bridge.ts` (resonance indexing) | Retrieval by structural resonance (wavelet phase fingerprint + multifractal signature) as a second, independent retriever. | Largest item (~775 lines). The relevance "second channel" the gate needs. Coordinate with Argon's predictor work before starting. |
| `seam-wire.ts` Readout A | Per-turn disagreement ż from predictor factor z's — a candidate gate signal with pre-registered kill tests. | Depends on PAMI bridge for its input encoding; after PAMI. |
| `scars.ts` | Recorded injuries as first-class memory (the inverse of skills). | Small design; folds into the oracle outcome-ledger work. |
| `falcon-ground.ts` refusal discipline | Ungrounded → refuse with a *named, diagnosable* error. | The monster already fails closed (teacher removal); port the named-refusal pattern into gate receipts. |

## KEEP (worker applications — not model components)

trading.ts / trading-ground.ts, falcon.ts, flock*.ts, grant-intelligence.ts,
tax suite (tax.ts, tax-calc.ts — note: tax-calc is the same deterministic-
automaton pattern as calc.ts, but its domain lives in the worker),
law.ts, ideas.ts, journal.ts, libre.ts, rapid.ts, atlas*.ts (business apps),
router.ts / index.ts (worker shell), llm.ts (model router), mcp*.ts, lane-*
(PQC comms), pqc-*.ts, connect-sandbox.ts, session-bus.ts, kv-*.ts,
ingest-*.ts, uploads/auth/onboarding plumbing.

## DROP proposals (none without explicit sign-off)

None proposed yet. Duplicates to resolve during review:
- `hyperbolic-sync.ts` / `hyperbolic-sync-fixed.ts` vs the monster's own
  geodesic machinery — likely superseded, verify first.
- `product.ts` / `hyper.ts` / `torus.ts` (worker TS geometry) vs the ported
  Rust `manifold.rs` — the Rust side is canonical now.

## Open Rust-side problems no worker module solves

- Q32_31 overflow panic in the `query` path (`fixed.rs` Sub/Add `.expect()`).
  Fix is saturating/widened arithmetic in core geometry. Worker analogue is
  only the NaN policy in predict.ts (design, already honored in the gate).
- Arc provenance for the relevance check (which arcs produced a candidate) —
  a monster data-path job, probably the builder's.

---

*Prepared by Muse, 2026-10-19. Nothing in this document has been deleted,
moved, or scheduled for deletion.*
