# BREATHING DEPTH — Toroidal Isotropic Transformer Model Change

Status: DESIGN SPEC (2026-10-19, Stewart + Muse). Not implemented.
Owner lanes: depth-conditional predictor = Argon; gate/ledger/instruments = Muse.

## 1. The defect this corrects

The forward pass runs a fixed 52-layer stack on every turn. That depth is
**inherited from the teacher** (Glimmer's skeleton, recorded by distillation),
not derived from the computation. Consequences:

- Easy turns burn full depth for nothing (the walk/operators serve most
  answers with zero forward-pass layers already).
- The map shows **no variance in depth** — every trajectory lives at the
  same effective depth, because nothing in the machinery *creates* depth.
  (Stewart's standing complaint, 2026-10-15: "THERE DOESNT SEEM TO BE
  DEPTH. AND WE NEED THAT.")
- The toroidal isotropic transformer was supposed to BE the model. Running
  a teacher-shaped stack inside it is teacher-shaped thinking.

## 2. The principle

**Construct, don't truncate.** Depth is not a constant to be sliced down;
it is built up from a minimal scaffold, per turn, per prompt — and only as
far as the computation honestly needs. For lack of a better term: the model
**breathes**. This is active inference at the architectural core:
precision-weighted recruitment of compute, minimizing expected free energy.
Only the attention necessary gets constructed.

## 3. The mechanism — two keys

### 3.1 PAMI complex index (the forecast)

PAMI's multifractal spectrum D(q) measures the **geometric complexity** of
the carrier trajectory's phase signal. Hypothesis (pre-registered, §6):
spectral spread predicts how much construction a prompt needs — a smooth,
low-complexity phase resolves shallow; a rough, multifractal-rich one needs
depth to be represented faithfully. PAMI is the index that decides how
much machine to build. (pami.rs is ported and waiting for exactly this job.)

### 3.2 Marginal free-energy rule (the brake)

The free-energy regulator (free_energy.rs) already keeps a conserved
ledger F + W = F0 per session, where W is work spent. Construction rule:

> Build the next layer while **dF/d(layer) < −cost_per_layer**.

Each layer must pay for itself in free-energy reduction. PAMI forecasts;
the regulator confirms. Either key alone is insufficient (see §5).

## 4. What breathes

- **Depth**: layers constructed from a minimal scaffold (one representational
  step — zero is the limit concept, not a state) until §3.2 stops paying.
- **Rank**: Jacobian accretion already grows sub-ranks 1..8 per atom; rank
  joins depth as a constructed quantity, gated by the same marginal rule.
- **Attention**: only the heads/sectors the phase index lights up.

Depth, rank, and attention used are **outputs of the computation**,
recorded per turn — not hyperparameters.

## 5. Honest failure modes (named in advance)

- **Roughness ≠ difficulty.** Noise is multifractal-rich. Building depth to
  represent noise is the catastrophic version of this idea. Mitigation:
  the marginal rule only pays for layers that reduce F; noise doesn't
  reduce. And §6 falsifies the PAMI index directly.
- **Depth-varying signals break comparisons.** Once depth varies, gate
  z-surprises from a 10-layer turn and a 40-layer turn are different
  instruments. Depth (and rank) must ride the ledger frame as covariates
  (extend kind-2/kind-3 frames) so the falsifier compares like with like.
- **The predictor is depth-trained.** Argon's carrier model was trained on
  fixed 52-layer traces; changing depth changes its input distribution.
  The model change therefore includes: **depth-conditional training**
  (train across depths; depth as an explicit input; per-depth readouts).
  This is the core of Argon's lane in this redesign.

## 6. Falsification (pre-registered before building)

Fixed prompt ladder, easy → hard, frozen before the run. Breathing vs
fixed-52, same prompts, same everything else:

1. **Answer agreement**: breathing matches fixed-52 answers on ≥ the
   pre-registered fraction (threshold set before data, not after).
2. **Economy**: mean constructed depth on the easy half < 52, strictly.
3. **The index hypothesis**: PAMI spectral spread correlates with the
   *marginal gain from added layers* per prompt (Spearman vs permutation
   null, the existing falsifier, alpha 0.01).
   - NULL on (3) kills PAMI-as-depth-index. Breathing survives on the
     marginal rule alone; the index gets replaced, not defended.
4. **Depth variance**: the map's radial (Poincaré) depth distribution
   gains real spread under breathing. If depth still doesn't vary, the
   construction isn't doing what the spec claims.

## 7. Non-goals

- NOT early-exit heuristics bolted onto a fixed stack.
- NOT truncation to a smaller constant (16 is as dishonest as 52 if it
  isn't earned per prompt).
- NOT serving the teacher's geometry and calling it the model.
- The JSON promoter sidecar is a dev cache; the shipped machine interface
  stays binary (Stewart's rule: no JSON machine-interface parsing).

## 8. Current state it builds on (verified 2026-10-19)

- Trajectory gate, free-energy regulator, outcome ledger + scars,
  falsifier (4 signals, UNDERPOWERED, accruing), gate-v2 provenance —
  all live.
- Predictor promotion slot wired in `execute_live_operational_turn`:
  last-chance candidate, gate-scored, serves only on Accept when the turn
  would otherwise refuse. (Trigger is rare while Tier2 synthesis serves
  almost everything — promotion to a primary/shadow-competing slot is a
  staging decision for after signals qualify.)
- Full-set eval baselines (teacher OFF): GSM8K 53/1319 (4.0%),
  HumanEval 94/164 (57.3%). The 10/10 and 8/8 subset numbers did not
  survive the full sets and must not be cited.
