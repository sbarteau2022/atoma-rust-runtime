# Native Toroidal Next-Carrier Transition & Gate Interface Contract

**Status:** Active Interface Contract (2026-10-19)  
**Predictor Module (Argon):** [`orchestrator/src/native_predictor.rs`](../orchestrator/src/native_predictor.rs), [`orchestrator/src/bin/native_toroidal.rs`](../orchestrator/src/bin/native_toroidal.rs)  
**Turn & Single-Gate Chassis (Peer Agent):** [`orchestrator/src/runtime_model.rs`](../orchestrator/src/runtime_model.rs), [`orchestrator/src/predict_gate.rs`](../orchestrator/src/predict_gate.rs), [`orchestrator/src/manifold.rs`](../orchestrator/src/manifold.rs), [`orchestrator/src/input_embeddings.rs`](../orchestrator/src/input_embeddings.rs)  

---

## 1. File Ownership & Parallel-Edit Boundary

To prevent concurrent edits from corrupting work in progress:

1. **Predictor & Training Lane (`Argon`):**
   - Owns [`orchestrator/src/native_predictor.rs`](../orchestrator/src/native_predictor.rs), [`orchestrator/src/bin/native_toroidal.rs`](../orchestrator/src/bin/native_toroidal.rs), and this contract doc.
   - **Will NOT edit** [`orchestrator/src/runtime_model.rs`](../orchestrator/src/runtime_model.rs), [`orchestrator/src/predict_gate.rs`](../orchestrator/src/predict_gate.rs), [`orchestrator/src/manifold.rs`](../orchestrator/src/manifold.rs), [`orchestrator/src/input_embeddings.rs`](../orchestrator/src/input_embeddings.rs), or [`orchestrator/src/lib.rs`](../orchestrator/src/lib.rs).
2. **Gate & Turn Wiring Lane (`Peer Agent`):**
   - Owns [`orchestrator/src/runtime_model.rs`](../orchestrator/src/runtime_model.rs), [`orchestrator/src/predict_gate.rs`](../orchestrator/src/predict_gate.rs), [`orchestrator/src/manifold.rs`](../orchestrator/src/manifold.rs), and [`orchestrator/src/input_embeddings.rs`](../orchestrator/src/input_embeddings.rs).
   - **No second gate:** When the native predictor promotes into the turn path, it feeds the exact same [`TrajectoryMatchGate`](../orchestrator/src/predict_gate.rs) chassis — candidate in, verdict out, generator-agnostic (`candidate.score_with_gate(&mut self.predict_gate)`).

---

## 2. Qualification Discipline & Per-Step `z-surprise` Reporting

1. **Generator Remains 100% Hands-Off During Qualification:**
   - Neither `NativeToroidalPredictor::generate` nor `generate_toroidal_carrier_continuation_with_model` uses the gate to steer, prune, mask, or resample tokens during rollout.
2. **Per-Step Operator `z-surprise` (`predict_gate.rs`) Reported Next to NLL:**
   - `PredictionScore` reports `z_surprise_steps`, `mean_z_surprise`, `max_z_surprise`, `mean_z_hyper`, and `mean_z_torus` alongside `mean_nll_nats` and `bits_per_position`.
   - `NativeGeneration` reports `mean_z_surprise`, `max_z_surprise`, `step_z_surprises: Vec<f64>`, and `gate_verdict: GateVerdict` from `TrajectoryMatchGate::default()`.
   - `CarrierTransitionStep` records per-step `z_surprise`, `z_hyper`, and `z_torus` next to `log_prob` and `margin`.
   - `CarrierTransitionTrainingReceipt` reports `initial_mean_z_surprise`, `final_mean_z_surprise`, and `final_max_z_surprise` next to `initial_branch_nll_nats` and `final_branch_nll_nats`.

---

## 3. Teacher-Contaminated Vault & WAL Quarantine

Inspection of `design/global_active_mind.atoma_map` (`104,538` arcs) and `design/global_active_mind.atoma_map.wal` (`10` binary `ATMWAL02` frames, `194` steps) confirmed teacher-crystallized contamination:
- **WAL Frame 3 (`2+2` proof):** `prompt = "What is two plus two"` -> `emitted = " a system of recording financial transactions in which each transaction affects at least two accounts..."` with `origin_fingerprint == a30a100326adb6d24d3b7dd51e5cb8b92d3b076f71302f6b9439f0d01bc54e61` (`Sha256Digest::compute(b"geodesic-active-transformer-v1")`).
- **Vault Contamination:** Out of `104,511` carrier arcs in `design/global_active_mind.atoma_map`, `104,231` are clean pre-teacher shard-distilled arcs (`7ffa62a0...`), while `280` arcs (indices `104242..=104521`) carry `origin_model_fingerprint == a30a100326adb6d24d3b7dd51e5cb8b92d3b076f71302f6b9439f0d01bc54e61`.

To guarantee the predictor never trains on or evaluates against teacher-crystallized wrong answers:
1. `require_disjoint_corpora` and `NativeToroidalPredictor::evaluate` call `require_disjoint_from_wal(..., &default_workspace_wal_trajectories())`, rejecting any training or evaluation record whose document, completion, or 32-byte window overlaps any `ATMWAL02` WAL frame.
2. `extract_vault_continuation_dataset` and `NativeCarrierTransitionModel::fit_from_swarm_map` quarantine all arcs with `is_teacher_contaminated_origin(&arc.origin_model_fingerprint)` (`a30a1003...` and `0a29d4cf...`) and any WAL `chain_id`s, training exclusively on `clean_chains` (`104,231` clean continuation arcs).

---

## 4. Exact Public Rust Interface in `atoma_orchestrator::native_predictor`

```rust
use atoma_immune::{
    BoundaryLexicalTransducer, CarvedSwarmMap, MlirAtomCatalog, MlirOpId,
};
use crate::input_embeddings::NativeMlirEmbedding;
use crate::predict_gate::{GateScore, TrajectoryMatchGate};
use crate::toroidal_attention::ToroidalIsotropicRuntime;

/// Single-step next-carrier transition decision driven from the toroidal hidden state.
#[derive(Debug, Clone)]
pub struct CarrierTransitionStep {
    pub step_index: usize,
    pub from_carrier: MlirOpId,
    pub predicted_carrier: MlirOpId,
    pub log_prob: f64,
    pub margin: f64,
    pub toroidal_hidden: [f32; 6],
    pub predicted_target_coord: [f32; 6],
    pub matched_vault_arc: bool,
    pub z_surprise: f64,
    pub z_hyper: f64,
    pub z_torus: f64,
}

/// Candidate continuation produced by the toroidal next-carrier transition predictor.
#[derive(Debug, Clone)]
pub struct NativeContinuationCandidate {
    pub prompt_embeddings: Vec<NativeMlirEmbedding>,
    pub emitted_embeddings: Vec<NativeMlirEmbedding>,
    pub emitted_carriers: Vec<MlirOpId>,
    pub decoded_text: Option<String>,
    pub decode_error: Option<String>,
    pub stop: NativeGenerationStop,
    pub mean_log_prob: f64,
    pub min_margin: f64,
    pub vault_arc_coverage: f64,
    pub mean_z_surprise: f64,
    pub max_z_surprise: f64,
    pub gate_score: GateScore,
    pub steps: Vec<CarrierTransitionStep>,
    pub is_continuation: bool,
}

impl NativeContinuationCandidate {
    /// Feed this candidate directly into the single `TrajectoryMatchGate` chassis (`predict_gate.rs`)
    /// — candidate in, verdict out, generator-agnostic.
    pub fn score_with_gate(&self, gate: &mut TrajectoryMatchGate) -> GateScore {
        gate.score_response(&self.prompt_embeddings, &self.emitted_embeddings)
    }
}
```

---

## 5. Drop-In Single-Gate Call Pattern for `runtime_model.rs`

When promoting the predictor into `MainModelCoordinatorRuntime::execute_live_operational_turn` ([`orchestrator/src/runtime_model.rs`](../orchestrator/src/runtime_model.rs)):

```rust
// 1. Predictor owns hands-off candidate generation from the session's toroidal hidden state:
let candidate = crate::native_predictor::generate_toroidal_carrier_continuation(
    toroidal_rt,
    &self.server.swarm_map,
    &self.server.mlir_catalog,
    &self.server.active_transformer.transducer,
    prompt,
    Some(&resolved_check),
    max_new,
)?;

// 2. Single Gate (`self.predict_gate: TrajectoryMatchGate`) runs AFTER predictor:
//    candidate in, verdict out, generator-agnostic.
let gate_score = candidate.score_with_gate(&mut self.predict_gate);
```
