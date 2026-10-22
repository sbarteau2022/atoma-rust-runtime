# MODEL AUDIT — 2026-10-19 (Stewart + Muse)

Method: live REPL turns on the production vault
(`design/global_active_mind.atoma_map`, sha 3ae05923166b8012), code
inspection of the serving paths, transducer round-trip smoke (exact,
all forms), full-set eval numbers. Every claim below is labeled:
**verified** (ran tonight), **code** (read in source), **reported**
(from a component owner, not re-run), **stale/contradicted**.

## 1. Live state (verified)

Vault: 17,982 0-cells / 105,183 1-cell arcs (104,754 locked) /
2,292 ingested tensors. 12 silos uneven (L2 3,094, L7 3,292 vs
L12 803). Banner claims base weights `Qwen3.8-27B+Muse-Glimmer-UnifiedMind
layers=60` — note: the toroidal forward pass steps 52, matching neither
the banner's 60 nor any derived requirement (see BREATHING_DEPTH).

## 2. Seven live turns (verified, tonight)

| Prompt | Tier | Served | Verdict |
|---|---|---|---|
| Capital of Italy | Tier1 walk | "Rome. It is the largest city in Italy and" | CORRECT, 10.9ms |
| 17 x 6 | operator | "17 * 6 = 102." | CORRECT, 17.7ms |
| Natalia/GSM8K | Tier2 (operator text) | "48/2=24. Total=72" | CORRECT (template-1 shape) |
| Python reverse-string | Tier2 synthesis | set-theory prose fragment | **wrong domain, fluent, served** |
| Ocean tides | Tier2 bridge | "a tide is the rise and." | truncated fragment |
| Insulin | Tier2 synthesis | heat-conduction misconception text | **wrong domain, fluent, served** |
| nonsense words | Tier2 synthesis | stove prose + HTML | **wrong domain, fluent, served** |

The serving gate ACCEPTED all of them (verified: no `[no grounded]`
refusal fired). Consistency gate works as measured; relevance is
instrument-only.

## 3. Component findings

**Transducer (verified, clean).** Round-trip ingress→egress is exact on
all five tested forms (JSON 451 chars, English 280 chars, verbose math,
short prompt): 0 char diffs. The codec is a perfect pipe. It is NOT the
4% and NOT the garble. Suspect closed.

**Reader (verified, broken).** Every turn's native-output head is
labeled by the server itself `[untrained]`: `"The capTtal of Italy iQ"`,
or ML-IR egress errors (invalid UTF-8 sequences). The readout is
prompt-echo garble. Nothing served has ever come from it.

**Forward pass (code + verified).** `step_native_toroidal_prompt`
runs the 52-layer stack every turn and banks hidden states. No served
text has ever come from those states. Current honest serving depth: 0.

**Predictor (reported + wired).** Argon: NLL 3.73→1.92, top1
0.18→0.48 on clean vault arcs, z-surprise instrumented, quarantined.
Promotion slot wired tonight: last-chance candidate, gate-scored,
serves only on Accept when the turn would otherwise refuse. Did not
fire in any audit turn (Tier2 synthesis preempts refusals).

**Tier2 synthesis (verified, unreliable).** Fluent off-domain prose
for anything the walk/operators don't cover. This is the map's
answer when it doesn't know: it confabulates.

## 4. FRESH FINDING: indiscriminate crystallization (verified)

Every Tier2 turn crystallized arcs into the vault — including the
wrong-domain ones: the insulin misconception minted +33 arcs, the
nonsense-stove turn +33, the set-theory turn +33. The gate does not
gate crystallization; the falsifier does not feed back into it.
**The vault is learning the system's own fluent garbage as
trajectories.** Over time this poisons retrieval: future walks will
serve today's confabulations. No outcome resolution currently blocks
or retracts crystallization.

## 5. Claims vs reality

- "3-Tier Live Operational Turn" — real (Tiers 1/2 + predictor slot).
- Kill Criteria Gate 24/24 — **self-check**: the system checking
  itself against itself, not capability. State as such.
- Whitepaper GSM8K/HumanEval — **contradicted by full sets**:
  53/1319 (4.0%), 94/164 (57.3%).
- Whitepaper vault 219,439/216,814 vs live 17,982/105,183 —
  **contradicted** (different/larger historical state).
- README 17,843/91,422 — **stale**.
- GSM8K 92.9% "compile coverage" — **test-gaming artifact**
  (phrase templates matched to the benchmark's habits).

## 6. What the model actually is tonight (verified summary)

A fast exact-retrieval engine (walk) + exact deterministic operators +
an unreliable confabulating synthesizer, wrapped in honest
instrumentation (gate, regulator, ledger, falsifier, provenance) —
none of which currently changes what gets served. The learned
components (reader, predictor-as-primary, breathing depth) are built
but unpromoted or untrained. The vault is growing, partly on garbage.

Recommended hard stops before any paper or release claim:
(1) crystallization gated on gate verdict / falsifier NULL;
(2) synthesis confabulation addressed or disclaimed in-product;
(3) predictor promoted to primary candidate with falsified signals;
(4) whitepaper numbers replaced with full-set numbers.
