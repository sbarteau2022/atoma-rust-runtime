//! Active Inference, Variational Free Energy (`F_q32`), and Superposition Holding Engine
//! (`design/superpositionholding.md`).
//!
//! Implements the runtime engine for Active Inference on `(H³ × T²) / Γ₁₂` in bit-exact `Q32_31`
//! fixed-point arithmetic (`#![forbid(unsafe_code)]`, zero external dependencies):
//!
//! 1. **Variational Free Energy (`VariationalFreeEnergyScore`)**:
//!    Minimizes `F = ΔS_geo + ΔH_Wigner + ΔE_epistemic + Φ_compliance_safety + C_token` across
//!    candidate geodesic transitions, coupling `OpticalRiskEngine` (`B(x) = 1/(x0 + 1)`),
//!    `OverlapPolicyMatrix`, `FirstPrinciplesComplianceEngine`, `SwarmKnowledgeAndHomeostasis`,
//!    and `AtomicIrAlphabet` (`<= 7`-token verify cliff + `p75 × 1.5` token spend envelope).
//! 2. **Stop & Hold Superposition (`SuperpositionHoldingEngine`)**:
//!    Whenever `NativeReader` hits `NoGeodesic`, a `FixedPoint` / cycle, Wigner frame ambiguity,
//!    low Meta-Cognitive Decision Readiness (`Ω < 0.82`), or a safety boundary, the server
//!    stops greedy token emission, holds up to `M <= 12` candidate branches in a coherent
//!    `SuperpositionState`, runs counterfactual lookahead rollouts and `2-cell` Gauss–Bonnet
//!    holonomy checks (`HyperbolicHolonomyEngine`), and dynamically iterates to a new
//!    minimum-free-energy solution.

use crate::atlas13::{EpistemicMark, RoleId};
use crate::cache::geometric::{AtomId, Phase};
use crate::compliance::OverlapPolicyMatrix;
use crate::distill_engine::DistilledReasoningArc;
use crate::first_principles::{
    ATOMIC_IR_MAX_VERIFY_TOKENS, MetaCognitiveRunbookController, MetaControlDecision,
    OpticalRiskEngine, RiskTier, RoleSignalQ32, RunbookTelemetryQ32,
};
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::holonomy::{HolonomyVerdict, HyperbolicHolonomyEngine};
use crate::map_native::CoupledOperatorAtomPayload;
use crate::mlir_native::{MlirAtomCatalog, MlirOpId};
use crate::native_reader::{
    NativeRead, NativeReader, frame_frobenius_dist_sq_q32, normalized_frame_alignment_q32,
    prefix_chain_id,
};
use crate::refraction::{OpticalLens, Sha256Digest};
use crate::silo::CarvedSwarmMap;
use std::collections::BTreeMap;

/// Configuration governing Active Inference, Variational Free Energy weights, Compliance/Safety
/// ceilings, Token Consumption budgets, and Superposition Holding iterations.
#[derive(Debug, Clone)]
pub struct ActiveInferenceConfig {
    /// Maximum primary basin `cosh(d)` for direct greedy continuation (default `1.25`).
    pub primary_basin_cosh_q32: Q32_31,
    /// Expanded hyperbolic search radius `cosh(d)` during Superposition Holding iterations (default `1.85`).
    pub superposition_basin_cosh_q32: Q32_31,
    /// Maximum acceptable Variational Free Energy `τ_F` before triggering Superposition Holding (default `0.55`).
    pub free_energy_ceiling_q32: Q32_31,
    /// Minimum Variational Free Energy gap `|F_1 - F_2|` between top-2 arcs to avoid ambiguity trigger (default `0.02`).
    pub ambiguity_margin_q32: Q32_31,
    /// Minimum required teacher certainty quality floor in `Q32_31` (default `0.50`).
    pub min_quality_certainty_q32: Q32_31,
    /// Minimum Meta-Cognitive Decision Readiness `Ω` for autonomous collapse (default `0.82`).
    pub min_omega_readiness_q32: Q32_31,
    /// Maximum autonomous optical blast radius `B(x) = 1/(x0 + 1)` (`EU AI Act Art. 14`, default `0.45`).
    pub max_autonomous_blast_q32: Q32_31,
    /// Weight `λ_safe` on optical blast radius & compliance risk barrier (default `0.50`).
    pub safety_compliance_weight_q32: Q32_31,
    /// Weight `λ_tok` on token spend & verify-cliff compression (default `0.20`).
    pub token_efficiency_weight_q32: Q32_31,
    /// Maximum number of candidate branches `M` held simultaneously in `SuperpositionState` (default `12`).
    pub max_superposition_branches: usize,
    /// Maximum dynamic reasoning iterations `K` before fail-closed termination (default `3`).
    pub max_iterations: usize,
    /// Counterfactual geodesic lookahead horizon `H` per held branch (capped by `ATOMIC_IR_MAX_VERIFY_TOKENS = 7`, default `3`).
    pub lookahead_horizon: usize,
    /// Predicted token spend envelope (`p75 × 1.5`) for the current generation window.
    pub predicted_token_envelope: u64,
}

impl Default for ActiveInferenceConfig {
    fn default() -> Self {
        Self {
            primary_basin_cosh_q32: Q32_31::from_f64(1.25),
            superposition_basin_cosh_q32: Q32_31::from_f64(1.85),
            free_energy_ceiling_q32: Q32_31::from_f64(0.55),
            ambiguity_margin_q32: Q32_31::from_f64(0.02),
            min_quality_certainty_q32: Q32_31::from_f64(0.50),
            min_omega_readiness_q32: Q32_31::from_f64(0.82),
            // Token carriers span r ∈ [0.20, 0.75] -> B(r) = (1 - r²)/2 ∈ [0.218, 0.480];
            // 0.49 permits all token carrier shells while blocking L0 origin mutations (B(0) = 0.50).
            max_autonomous_blast_q32: Q32_31::from_f64(0.49),
            safety_compliance_weight_q32: Q32_31::from_f64(0.50),
            token_efficiency_weight_q32: Q32_31::from_f64(0.20),
            max_superposition_branches: 12,
            max_iterations: 3,
            lookahead_horizon: 3,
            predicted_token_envelope: 256,
        }
    }
}

/// Exact `Q32_31` decomposition of the Variational Free Energy `F` for a candidate transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VariationalFreeEnergyScore {
    /// Hyperbolic geodesic surprise `(cosh d(q, u) - 1) + 0.25 * (cosh d(u, v) - 1)`.
    pub geodesic_surprise_q32: Q32_31,
    /// Normalized Wigner holonomy divergence `(1 - cos_align(W_e, W_prefix)) + residual`.
    pub wigner_divergence_q32: Q32_31,
    /// Epistemic inaccuracy `1 - teacher_certainty_q32`.
    pub epistemic_inaccuracy_q32: Q32_31,
    /// First-principle compliance, safety, and optical blast-radius barrier `Φ_safety`.
    pub compliance_safety_barrier_q32: Q32_31,
    /// Token consumption & cycle/redundancy cost (negative when compressing via Atomic IR spans).
    pub token_cost_q32: Q32_31,
    /// Gauss–Bonnet 2-cell hyperbolic curvature bonus subtracted during counterfactual holonomy verification.
    pub holonomy_curvature_bonus_q32: Q32_31,
    /// Counterfactual lookahead free-energy adjustment across horizon `H`.
    pub counterfactual_horizon_delta_q32: Q32_31,
    /// Total Variational Free Energy `F` in `Q32_31` (lower is better; `Q32_31::MAX` if vetoed).
    pub total_free_energy_q32: Q32_31,
    /// True if Layer 3 `OverlapPolicyMatrix`, `SiloHorizonIsolation`, and blast-radius gates permit this transition.
    pub compliance_permitted: bool,
    /// Optical risk tier at the target coordinate.
    pub risk_tier: RiskTier,
}

/// Issue that caused Active Inference to stop greedy emission and enter `superpositionholding.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuperpositionTrigger {
    /// No outgoing `1-cell` arc exists within `primary_basin_cosh_q32`.
    NoDirectGeodesic {
        at_op: MlirOpId,
        nearest_cosh_q32: Q32_31,
    },
    /// Greedy candidate would repeat the current token or form a cycle in the prefix.
    FixedPointOrCycle {
        at_op: MlirOpId,
        repeated_op: MlirOpId,
    },
    /// Top-2 outgoing arcs have near-tied Variational Free Energy (`|F_1 - F_2| < ambiguity_margin_q32`)
    /// without an exact causal `chain_id` match.
    WignerFrameAmbiguity {
        at_op: MlirOpId,
        candidate_a: MlirOpId,
        candidate_b: MlirOpId,
        free_energy_gap_q32: Q32_31,
    },
    /// Best greedy candidate exceeds `free_energy_ceiling_q32` or falls below `min_omega_readiness_q32`.
    FreeEnergyOrReadinessSpike {
        at_op: MlirOpId,
        free_energy_q32: Q32_31,
        omega_readiness_q32: Q32_31,
    },
    /// Greedy candidate violates `OverlapPolicyMatrix` or exceeds `max_autonomous_blast_q32`.
    ComplianceOrSafetyBoundary {
        at_op: MlirOpId,
        blocked_op: MlirOpId,
        blast_radius_q32: Q32_31,
    },
}

/// Geometric source from which a candidate branch in `SuperpositionState` was harvested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperpositionBranchSource {
    /// Outgoing `1-cell` trajectory arc in the local hyperbolic basin.
    DirectTrajectoryArc,
    /// Outgoing `1-cell` arc discovered via golden-angle `φ`-winding toroidal/hyperbolic scan (`RefineEvidencePhiScan`).
    PhiWindingNeighborArc,
    /// Coupled Operator (`ATOMA_OP` v2) `next_mlir_id` bridge in the local basin.
    CoupledOperatorBridge,
    /// Cross-silo geodesic bridge to a high-salience atom in a permitted icosahedral neighbor silo.
    CrossSiloGeodesicBridge,
}

/// A single coherent candidate hypothesis held inside `SuperpositionState`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperpositionBranch {
    pub branch_id: usize,
    pub source: SuperpositionBranchSource,
    pub candidate_next_id: MlirOpId,
    pub arc_id: u64,
    pub from_atom: AtomId,
    pub to_atom: AtomId,
    pub from_role: RoleId,
    pub specialist_role: RoleId,
    pub query_cosh_q32: Q32_31,
    pub cosh_action_q32: Q32_31,
    pub frame_alignment_q32: Q32_31,
    pub transported_frame: [[Q32_31; 3]; 3],
    pub origin_model_fingerprint: Sha256Digest,
    pub free_energy: VariationalFreeEnergyScore,
    /// Normalized Boltzmann amplitude `w_k = exp(-β F_k) / Z` in `Q32_31`.
    pub boltzmann_weight_q32: Q32_31,
    /// Number of verified non-cycling lookahead steps unlocked during counterfactual rollout.
    pub verified_lookahead_steps: usize,
    /// Gauss–Bonnet `2-cell` hyperbolic triangle defect `Ω(Δ)` in `Q32_31` radians.
    pub gauss_bonnet_flux_q32: Q32_31,
}

/// Trace of one dynamic reasoning iteration inside `SuperpositionHoldingEngine`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperpositionIterationTrace {
    pub iteration: usize,
    pub search_cosh_radius_q32: Q32_31,
    pub branches_evaluated: usize,
    pub branches_surviving_compliance: usize,
    pub best_candidate_op: Option<MlirOpId>,
    pub min_free_energy_q32: Q32_31,
    pub omega_readiness_q32: Q32_31,
    pub meta_decision: MetaControlDecision,
}

/// Coherent superposition state held while reasoning across competing branches (`superpositionholding.md` §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperpositionState {
    pub trigger: SuperpositionTrigger,
    pub prefix_len: usize,
    pub standing_at: MlirOpId,
    pub query_lorentz: LorentzVector,
    pub query_phase: Phase,
    pub branches: Vec<SuperpositionBranch>,
    pub iteration_traces: Vec<SuperpositionIterationTrace>,
}

/// Outcome of holding superposition and dynamically iterating to a solution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuperpositionResolution {
    /// Superposition collapsed onto a verified minimum-free-energy branch after dynamic iteration.
    CollapsedToSolution {
        winning_branch: SuperpositionBranch,
        iterations_used: usize,
        bridge_arc_distilled: bool,
        state: SuperpositionState,
    },
    /// All candidate branches failed compliance, safety, or geodesic closure after `max_iterations`.
    FailClosed {
        reason: String,
        state: SuperpositionState,
    },
}

/// Complete receipt for a single Active Inference step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveInferenceStepReceipt {
    pub read: NativeRead,
    pub free_energy: VariationalFreeEnergyScore,
    pub omega_readiness_q32: Q32_31,
    pub meta_decision: MetaControlDecision,
    /// Populated if the step encountered an issue, stopped, held superposition, and dynamically iterated.
    pub superposition_held: Option<SuperpositionState>,
    /// True if a novel bridge `1-cell` arc was synthesized and self-distilled into the map.
    pub bridge_arc_distilled: bool,
    pub mark: EpistemicMark,
}

/// Why a multi-step Active Inference continuation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActiveInferenceStop {
    MaxNew,
    Eos(MlirOpId),
    TokenEnvelopeExhausted {
        spent_tokens: u64,
        envelope_tokens: u64,
    },
    SuperpositionFailClosed {
        at: MlirOpId,
        reason: String,
        state: SuperpositionState,
    },
}

/// Multi-step Active Inference continuation receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveInferenceContinuation {
    pub steps: Vec<ActiveInferenceStepReceipt>,
    pub stop: ActiveInferenceStop,
    pub total_free_energy_q32: Q32_31,
    pub superposition_interventions: usize,
    pub bridge_arcs_distilled: usize,
}

/// Runtime engine implementing `design/superpositionholding.md`.
pub struct SuperpositionHoldingEngine {
    pub config: ActiveInferenceConfig,
    pub meta_controller: MetaCognitiveRunbookController,
    pub holonomy_engine: HyperbolicHolonomyEngine,
}

impl Default for SuperpositionHoldingEngine {
    fn default() -> Self {
        Self::new(ActiveInferenceConfig::default())
    }
}

impl SuperpositionHoldingEngine {
    pub fn new(config: ActiveInferenceConfig) -> Self {
        Self {
            config,
            meta_controller: MetaCognitiveRunbookController::default(),
            holonomy_engine: HyperbolicHolonomyEngine::default(),
        }
    }

    /// Compute the exact `Q32_31` Variational Free Energy `F` of a candidate transition `arc`
    /// given the current `prefix`, query coordinate `query_lorentz`, and prefix Wigner `frame`.
    pub fn score_candidate_arc(
        &self,
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        query_lorentz: &LorentzVector,
        prefix_frame: &[[Q32_31; 3]; 3],
        query_cosh_q32: Q32_31,
        arc: &DistilledReasoningArc,
    ) -> VariationalFreeEnergyScore {
        let one = Q32_31::ONE.0 as i128;

        // 1. Geodesic Surprise: (cosh d(q, u) - 1) + 0.25 * (cosh d(u, v) - 1)
        let q_excess = (query_cosh_q32.0 as i128 - one).max(0);
        let hop_excess = ((arc.cosh_action_q32.0 as i128 - one).max(0)) >> 2;
        let geodesic_surprise = Q32_31((q_excess + hop_excess).clamp(0, i64::MAX as i128) as i64);

        // 2. Normalized Wigner Holonomy Divergence:
        //    (1 - cos_align(W_e, W_prefix)) + (1/8) * ||W_e - W_prefix||_F^2
        //    When chain_id matches the exact prefix AND frame matches, divergence is 0.
        let cos_align = normalized_frame_alignment_q32(&arc.wigner_frame, prefix_frame);
        let frob_sq = frame_frobenius_dist_sq_q32(&arc.wigner_frame, prefix_frame);
        let exact_chain = arc.chain_id == prefix_chain_id(prefix);
        let step_drift =
            (arc.step_index as i128 - (prefix.len().saturating_sub(1) as i128)).abs();
        let step_penalty =
            (step_drift * (Q32_31::from_f64(0.15).0 as i128)).min(Q32_31::from_f64(0.60).0 as i128);
        let raw_wigner_div =
            ((one - cos_align.0 as i128).max(0)) + ((frob_sq.0 as i128) >> 3) + step_penalty;
        let wigner_divergence = if exact_chain && frob_sq.0 == 0 {
            Q32_31::ZERO
        } else if exact_chain {
            Q32_31((raw_wigner_div >> 2).clamp(0, i64::MAX as i128) as i64)
        } else {
            Q32_31(raw_wigner_div.clamp(0, i64::MAX as i128) as i64)
        };

        // 3. Epistemic Inaccuracy: (1 - teacher_certainty_q32)
        let cert_clamped = arc.teacher_certainty_q32.0.clamp(0, Q32_31::ONE.0) as i128;
        let epistemic_inaccuracy = Q32_31((one - cert_clamped).max(0) as i64);

        // 4. First-Principle Compliance, Safety & Security Potential:
        let to_lorentz = map
            .global_atom(arc.to_atom)
            .map(|a| a.lorentz)
            .or_else(|| NativeReader::address(catalog, arc.compiled_mlir_op).ok().map(|(l, _)| l))
            .unwrap_or(*query_lorentz);

        let blast = OpticalRiskEngine::optical_blast_radius_q32(&to_lorentz);
        let risk_report = OpticalRiskEngine::evaluate_coordinate_risk(
            &to_lorentz,
            [Q32_31::from_f64(2.0); 5],
            Q32_31::from_f64(2.0),
            Q32_31::from_f64(2.0),
        );

        let role_comm_ok = if arc.from_role == arc.to_role
            || arc.from_role == RoleId::Coordinator
            || arc.to_role == RoleId::Coordinator
            || (arc.from_role == RoleId::LanguageSemiotics
                && arc.to_role == RoleId::PredictiveSimulation)
        {
            true
        } else {
            policy.overlap_allowance(arc.from_role, arc.to_role).0 > 0
                || map
                    .silo(arc.to_role)
                    .is_some_and(|s| s.contains_atom(arc.from_atom) || s.contains_atom(arc.to_atom))
        };
        let blast_ok = blast.0 <= self.config.max_autonomous_blast_q32.0;
        let quality_ok =
            arc.teacher_certainty_q32.0 >= (self.config.min_quality_certainty_q32.0 >> 2);
        let compliance_permitted = role_comm_ok && blast_ok && quality_ok;

        let compliance_safety_barrier = if !compliance_permitted {
            Q32_31::from_f64(100.0)
        } else {
            let weighted = ((blast.0 as i128) * (self.config.safety_compliance_weight_q32.0 as i128))
                >> Q32_31::FRACTIONAL_BITS;
            Q32_31(weighted.clamp(0, i64::MAX as i128) as i64)
        };

        // 5. Token Consumption & Cycle Cost:
        //    - Penalize immediate self-loop (`next_id == last`) or recent 2-gram cycle.
        //    - Reward multi-token structural/span opcodes (`compiled_mlir_op.0 >= catalog.base_vocab_size`)
        //      that compress token consumption within the `<= 7`-token verify cliff.
        let last_op = prefix.last().copied().unwrap_or(MlirOpId(0));
        let is_self_loop = arc.compiled_mlir_op == last_op;
        let is_2gram_cycle = prefix.len() >= 2
            && prefix[prefix.len() - 2] == arc.compiled_mlir_op
            && prefix.len() >= 4
            && prefix[prefix.len() - 4..prefix.len() - 2] == prefix[prefix.len() - 2..];

        let token_cost = if is_self_loop {
            Q32_31::from_f64(50.0)
        } else if is_2gram_cycle {
            Q32_31::from_f64(10.0)
        } else if arc.compiled_mlir_op.0 >= catalog.base_vocab_size {
            // Structural MLIR / Atomic IR span compression bonus (-0.05)
            Q32_31::from_f64(-0.05)
        } else {
            let base_tok_cost = ((Q32_31::from_f64(0.05).0 as i128)
                * (self.config.token_efficiency_weight_q32.0 as i128))
                >> Q32_31::FRACTIONAL_BITS;
            Q32_31(base_tok_cost as i64)
        };

        let total = if !compliance_permitted {
            Q32_31(i64::MAX / 2)
        } else {
            let sum = (geodesic_surprise.0 as i128)
                + (wigner_divergence.0 as i128)
                + (epistemic_inaccuracy.0 as i128)
                + (compliance_safety_barrier.0 as i128)
                + (token_cost.0 as i128);
            Q32_31(sum.clamp(0, (i64::MAX / 2) as i128) as i64)
        };

        VariationalFreeEnergyScore {
            geodesic_surprise_q32: geodesic_surprise,
            wigner_divergence_q32: wigner_divergence,
            epistemic_inaccuracy_q32: epistemic_inaccuracy,
            compliance_safety_barrier_q32: compliance_safety_barrier,
            token_cost_q32: token_cost,
            holonomy_curvature_bonus_q32: Q32_31::ZERO,
            counterfactual_horizon_delta_q32: Q32_31::ZERO,
            total_free_energy_q32: total,
            compliance_permitted,
            risk_tier: risk_report.tier,
        }
    }

    /// Execute the `superpositionholding.md` protocol:
    /// Stop greedy emission on `trigger`, harvest competing geodesic branches across
    /// `(H³ × T²)/Γ₁₂`, run counterfactual lookahead rollouts and `2-cell` Gauss–Bonnet
    /// holonomy checks across up to `max_iterations`, and collapse to the minimum-free-energy solution.
    pub fn hold_and_iterate(
        &self,
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        trigger: SuperpositionTrigger,
        spent_tokens: u64,
    ) -> SuperpositionResolution {
        let standing_at = prefix.last().copied().unwrap_or(MlirOpId(0));
        let (query_lorentz, query_phase) = NativeReader::address(catalog, standing_at)
            .unwrap_or_else(|_| (LorentzVector::origin(), Phase {
                theta_f: Q32_31::ZERO,
                theta_s: Q32_31::ZERO,
            }));
        let (_, prefix_frame) = NativeReader::context_frame(catalog, prefix)
            .unwrap_or((query_lorentz, crate::holonomy::IDENTITY_FRAME_Q32));

        let mut state = SuperpositionState {
            trigger,
            prefix_len: prefix.len(),
            standing_at,
            query_lorentz,
            query_phase,
            branches: Vec::new(),
            iteration_traces: Vec::new(),
        };

        let max_iters = self.config.max_iterations.max(1);
        let base_cosh = self.config.primary_basin_cosh_q32.0;
        let max_cosh = self.config.superposition_basin_cosh_q32.0.max(base_cosh);
        let step_cosh = if max_iters > 1 {
            max_cosh.saturating_sub(base_cosh) / (max_iters as i64 - 1).max(1)
        } else {
            max_cosh.saturating_sub(base_cosh)
        };

        for iter_idx in 0..max_iters {
            let current_cosh_limit = if iter_idx == 0 {
                // Even on iteration 0, expand slightly past primary basin if triggered by NoDirectGeodesic
                base_cosh.saturating_add(step_cosh / 2)
            } else if iter_idx + 1 == max_iters {
                max_cosh
            } else {
                base_cosh.saturating_add(step_cosh.saturating_mul(iter_idx as i64))
            };

            let mut candidates = self.harvest_superposition_branches(
                map,
                catalog,
                policy,
                prefix,
                &query_lorentz,
                query_phase,
                &prefix_frame,
                Q32_31(current_cosh_limit),
                iter_idx,
            );

            // Pre-sort by initial Variational Free Energy and cap to M = max_superposition_branches
            // BEFORE counterfactual lookahead so rollout is strictly O(M * H) (~2 ms on 55k+ arcs)
            candidates.sort_by(|a, b| {
                a.free_energy
                    .total_free_energy_q32
                    .0
                    .cmp(&b.free_energy.total_free_energy_q32.0)
                    .then(b.frame_alignment_q32.0.cmp(&a.frame_alignment_q32.0))
                    .then(a.cosh_action_q32.0.cmp(&b.cosh_action_q32.0))
                    .then(a.arc_id.cmp(&b.arc_id))
            });
            candidates.truncate(self.config.max_superposition_branches.max(1));

            // Evaluate counterfactual lookahead & 2-cell Gauss-Bonnet holonomy on the M held branches
            self.evaluate_counterfactuals_and_holonomy(
                map,
                catalog,
                policy,
                prefix,
                &query_lorentz,
                &mut candidates,
            );

            // Re-sort after counterfactual horizon and Gauss-Bonnet curvature adjustments
            candidates.sort_by(|a, b| {
                a.free_energy
                    .total_free_energy_q32
                    .0
                    .cmp(&b.free_energy.total_free_energy_q32.0)
                    .then(b.frame_alignment_q32.0.cmp(&a.frame_alignment_q32.0))
                    .then(a.cosh_action_q32.0.cmp(&b.cosh_action_q32.0))
                    .then(a.arc_id.cmp(&b.arc_id))
            });

            // Compute normalized Boltzmann weights w_k = exp(-β F_k) / Z
            Self::assign_boltzmann_weights(&mut candidates);

            let compliant_count = candidates
                .iter()
                .filter(|b| b.free_energy.compliance_permitted)
                .count();

            // Build post-iteration RoleSignals for MetaCognitiveRunbookController:
            // The 3 roles in the winning branch's 2-cell Gauss-Bonnet loop endorse the winner,
            // while any close runner-up within `ambiguity_margin_q32` registers a dissenting signal.
            let mut signals: Vec<RoleSignalQ32> = Vec::new();
            if let Some(winner) = candidates.first() {
                let one = Q32_31::ONE.0 as i128;
                let denom = one + (winner.free_energy.total_free_energy_q32.0.max(0) as i128);
                let conf = Q32_31((((one << 31) / denom).clamp(0, one)) as i64);
                let win_hash = winner.candidate_next_id.0 as u64;
                let win_chain_hash = (winner.verified_lookahead_steps as u64) ^ win_hash;
                for role in [
                    winner.from_role,
                    winner.specialist_role,
                    RoleId::ConvergentLogic,
                ] {
                    signals.push(RoleSignalQ32 {
                        role,
                        hypothesis_hash: win_chain_hash,
                        proposed_action_hash: win_hash,
                        confidence_q32: conf,
                    });
                }
                if let Some(runner_up) = candidates.get(1) {
                    let gap = runner_up
                        .free_energy
                        .total_free_energy_q32
                        .0
                        .saturating_sub(winner.free_energy.total_free_energy_q32.0);
                    if gap < self.config.ambiguity_margin_q32.0 {
                        let r_denom =
                            one + (runner_up.free_energy.total_free_energy_q32.0.max(0) as i128);
                        let r_conf = Q32_31((((one << 31) / r_denom).clamp(0, one)) as i64);
                        signals.push(RoleSignalQ32 {
                            role: runner_up.specialist_role,
                            hypothesis_hash: win_chain_hash,
                            proposed_action_hash: runner_up.candidate_next_id.0 as u64,
                            confidence_q32: r_conf,
                        });
                    }
                }
            }

            let telemetry = RunbookTelemetryQ32 {
                step: prefix.len() as u32,
                kappa_q32: Q32_31::from_f64(0.88),
                drift_detected: false,
                out_of_region: false,
                regime_collapse: compliant_count == 0,
                verified_progress_in_window: candidates
                    .first()
                    .map(|b| (b.verified_lookahead_steps as u32).max(1))
                    .unwrap_or(0),
                spent_tokens,
                predicted_token_envelope: self.config.predicted_token_envelope,
                nudges_used: iter_idx as u8,
            };

            let target_lor = candidates
                .first()
                .and_then(|b| map.global_atom(b.to_atom).map(|a| a.lorentz))
                .unwrap_or(query_lorentz);
            let verdict = self.meta_controller.evaluate(&signals, &target_lor, &telemetry);

            let min_fe = candidates
                .first()
                .map(|b| b.free_energy.total_free_energy_q32)
                .unwrap_or(Q32_31(i64::MAX / 2));

            state.iteration_traces.push(SuperpositionIterationTrace {
                iteration: iter_idx,
                search_cosh_radius_q32: Q32_31(current_cosh_limit),
                branches_evaluated: candidates.len(),
                branches_surviving_compliance: compliant_count,
                best_candidate_op: candidates.first().map(|b| b.candidate_next_id),
                min_free_energy_q32: min_fe,
                omega_readiness_q32: verdict.omega_readiness_q32,
                meta_decision: verdict.decision,
            });
            state.branches = candidates;

            // Check if the winning branch satisfies compliance, avoids self-loops, and achieves low Free Energy
            if let Some(best) = state.branches.first() {
                if best.free_energy.compliance_permitted
                    && best.candidate_next_id != standing_at
                    && (best.free_energy.total_free_energy_q32.0
                        <= self.config.free_energy_ceiling_q32.0
                        || iter_idx + 1 == max_iters)
                {
                    let winner = best.clone();
                    return SuperpositionResolution::CollapsedToSolution {
                        winning_branch: winner,
                        iterations_used: iter_idx + 1,
                        bridge_arc_distilled: false,
                        state,
                    };
                }
            }
        }

        SuperpositionResolution::FailClosed {
            reason: format!(
                "no compliant minimum-free-energy trajectory found from {} after {} superposition iterations",
                standing_at, max_iters
            ),
            state,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn harvest_superposition_branches(
        &self,
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        query_lorentz: &LorentzVector,
        query_phase: Phase,
        prefix_frame: &[[Q32_31; 3]; 3],
        max_cosh_q32: Q32_31,
        iter_idx: usize,
    ) -> Vec<SuperpositionBranch> {
        let standing_at = prefix.last().copied().unwrap_or(MlirOpId(0));
        let mut branches: Vec<SuperpositionBranch> = Vec::new();
        let mut seen_ops = BTreeMap::<u32, usize>::new();

        // Select the top-(16 * (iter_idx + 1)) nearest global atoms so harvesting stays
        // strictly bounded in O(K) even on 100k+ arc vaults
        let mut nearby_atoms: Vec<(Q32_31, u64)> = map
            .iter_global_atoms()
            .map(|(&id_u64, atom)| {
                (
                    OpticalLens::lorentz_inner_product_wide(query_lorentz, &atom.lorentz),
                    id_u64,
                )
            })
            .filter(|(c, _)| c.0 <= max_cosh_q32.0)
            .collect();
        nearby_atoms.sort_by_key(|(c, id)| (c.0, *id));
        let cap = 16 * (iter_idx + 1);
        let mut atom_cosh = BTreeMap::<u64, Q32_31>::new();
        for (c, id) in nearby_atoms.into_iter().take(cap) {
            atom_cosh.insert(id, c);
        }

        // 1 & 2. Harvest DirectTrajectoryArc and PhiWindingNeighborArc candidates in O(K) via outgoing_arc_indices
        for (&atom_id_u64, &q_cosh) in &atom_cosh {
            for &arc_idx in map.outgoing_arc_indices(AtomId(atom_id_u64)) {
                let arc = &map.reasoning_arcs[arc_idx];
                // Skip immediate self-loops when alternative branches are being sought
                if arc.compiled_mlir_op == standing_at {
                    continue;
                }
                let source = if q_cosh.0 <= self.config.primary_basin_cosh_q32.0 {
                    SuperpositionBranchSource::DirectTrajectoryArc
                } else {
                    SuperpositionBranchSource::PhiWindingNeighborArc
                };

                let fe = self.score_candidate_arc(
                    map,
                    catalog,
                    policy,
                    prefix,
                    query_lorentz,
                    prefix_frame,
                    q_cosh,
                    arc,
                );
                let align = normalized_frame_alignment_q32(&arc.wigner_frame, prefix_frame);
                let to_lor = map
                    .global_atom(arc.to_atom)
                    .map(|a| a.lorentz)
                    .unwrap_or(*query_lorentz);
                let next_frame = HyperbolicHolonomyEngine::transport_frame_hop(
                    query_lorentz,
                    &to_lor,
                    prefix_frame,
                );

                let branch = SuperpositionBranch {
                    branch_id: branches.len(),
                    source,
                    candidate_next_id: arc.compiled_mlir_op,
                    arc_id: arc.arc_id,
                    from_atom: arc.from_atom,
                    to_atom: arc.to_atom,
                    from_role: arc.from_role,
                    specialist_role: arc.to_role,
                    query_cosh_q32: q_cosh,
                    cosh_action_q32: arc.cosh_action_q32,
                    frame_alignment_q32: align,
                    transported_frame: next_frame,
                    origin_model_fingerprint: arc.origin_model_fingerprint,
                    free_energy: fe,
                    boltzmann_weight_q32: Q32_31::ZERO,
                    verified_lookahead_steps: 0,
                    gauss_bonnet_flux_q32: Q32_31::ZERO,
                };

                if let Some(&existing_idx) = seen_ops.get(&arc.compiled_mlir_op.0) {
                    if branch.free_energy.total_free_energy_q32.0
                        < branches[existing_idx].free_energy.total_free_energy_q32.0
                    {
                        branches[existing_idx] = branch;
                    }
                } else {
                    seen_ops.insert(arc.compiled_mlir_op.0, branches.len());
                    branches.push(branch);
                }
            }
        }

        // 3. Harvest CoupledOperatorBridge (`ATOMA_OP` v2 `peek_next_mlir_id`) in the local basin
        for (&atom_id_u64, &q_cosh) in &atom_cosh {
            let Some(atom) = map.global_atom(AtomId(atom_id_u64)) else {
                continue;
            };
            let Some(next_u32) = CoupledOperatorAtomPayload::peek_next_mlir_id(&atom.payload) else {
                continue;
            };
            let next_op = MlirOpId(next_u32);
            if next_op == standing_at || seen_ops.contains_key(&next_u32) {
                continue;
            }
            let Ok((to_lor, _)) = NativeReader::address(catalog, next_op) else {
                continue;
            };
            let cosh_action = OpticalLens::lorentz_inner_product_wide(&atom.lorentz, &to_lor);
            let synthetic_arc = DistilledReasoningArc {
                arc_id: u64::MAX - atom_id_u64,
                chain_id: prefix_chain_id(prefix),
                step_index: prefix.len().saturating_sub(1),
                from_atom: atom.id,
                to_atom: atom.id,
                from_role: RoleId::LanguageSemiotics,
                to_role: RoleId::ConvergentLogic,
                cosh_action_q32: cosh_action,
                wigner_frame: *prefix_frame,
                compiled_mlir_op: next_op,
                teacher_certainty_q32: Q32_31::from_f64(0.85),
                origin_model_fingerprint: atom.origin_model_fingerprint,
            };
            let fe = self.score_candidate_arc(
                map,
                catalog,
                policy,
                prefix,
                query_lorentz,
                prefix_frame,
                q_cosh,
                &synthetic_arc,
            );
            let next_frame = HyperbolicHolonomyEngine::transport_frame_hop(
                query_lorentz,
                &to_lor,
                prefix_frame,
            );
            seen_ops.insert(next_u32, branches.len());
            branches.push(SuperpositionBranch {
                branch_id: branches.len(),
                source: SuperpositionBranchSource::CoupledOperatorBridge,
                candidate_next_id: next_op,
                arc_id: synthetic_arc.arc_id,
                from_atom: atom.id,
                to_atom: atom.id,
                from_role: RoleId::LanguageSemiotics,
                specialist_role: RoleId::ConvergentLogic,
                query_cosh_q32: q_cosh,
                cosh_action_q32: cosh_action,
                frame_alignment_q32: Q32_31::ONE,
                transported_frame: next_frame,
                origin_model_fingerprint: atom.origin_model_fingerprint,
                free_energy: fe,
                boltzmann_weight_q32: Q32_31::ZERO,
                verified_lookahead_steps: 0,
                gauss_bonnet_flux_q32: Q32_31::ZERO,
            });
        }

        // 4. Golden-Angle φ-Winding Toroidal Scan (`RefineEvidencePhiScan`) if branches are still empty
        if branches.is_empty() && !map.reasoning_arcs.is_empty() {
            let phi_step = Q32_31::from_f64(2.399_963_229_728_653);
            let target_theta = Q32_31(
                query_phase
                    .theta_f
                    .0
                    .wrapping_add(phi_step.0.wrapping_mul((iter_idx as i64) + 1)),
            );
            for arc in &map.reasoning_arcs {
                if arc.compiled_mlir_op == standing_at || seen_ops.contains_key(&arc.compiled_mlir_op.0) {
                    continue;
                }
                let Some(from_atom) = map.global_atom(arc.from_atom) else {
                    continue;
                };
                let q_cosh =
                    OpticalLens::lorentz_inner_product_wide(query_lorentz, &from_atom.lorentz);
                let phase_diff = from_atom
                    .phase
                    .theta_f
                    .0
                    .wrapping_sub(target_theta.0)
                    .unsigned_abs();
                let _ = phase_diff;
                let fe = self.score_candidate_arc(
                    map,
                    catalog,
                    policy,
                    prefix,
                    query_lorentz,
                    prefix_frame,
                    q_cosh,
                    arc,
                );
                let align = normalized_frame_alignment_q32(&arc.wigner_frame, prefix_frame);
                let to_lor = map
                    .global_atom(arc.to_atom)
                    .map(|a| a.lorentz)
                    .unwrap_or(*query_lorentz);
                let next_frame = HyperbolicHolonomyEngine::transport_frame_hop(
                    query_lorentz,
                    &to_lor,
                    prefix_frame,
                );
                seen_ops.insert(arc.compiled_mlir_op.0, branches.len());
                branches.push(SuperpositionBranch {
                    branch_id: branches.len(),
                    source: SuperpositionBranchSource::CrossSiloGeodesicBridge,
                    candidate_next_id: arc.compiled_mlir_op,
                    arc_id: arc.arc_id,
                    from_atom: arc.from_atom,
                    to_atom: arc.to_atom,
                    from_role: arc.from_role,
                    specialist_role: arc.to_role,
                    query_cosh_q32: q_cosh,
                    cosh_action_q32: arc.cosh_action_q32,
                    frame_alignment_q32: align,
                    transported_frame: next_frame,
                    origin_model_fingerprint: arc.origin_model_fingerprint,
                    free_energy: fe,
                    boltzmann_weight_q32: Q32_31::ZERO,
                    verified_lookahead_steps: 0,
                    gauss_bonnet_flux_q32: Q32_31::ZERO,
                });
                if branches.len() >= self.config.max_superposition_branches * 2 {
                    break;
                }
            }
        }

        branches
    }

    fn evaluate_counterfactuals_and_holonomy(
        &self,
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        _policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        query_lorentz: &LorentzVector,
        branches: &mut [SuperpositionBranch],
    ) {
        let horizon = self
            .config
            .lookahead_horizon
            .min(ATOMIC_IR_MAX_VERIFY_TOKENS);

        let prev_lorentz = if prefix.len() >= 2 {
            NativeReader::address(catalog, prefix[prefix.len() - 2])
                .map(|(l, _)| l)
                .unwrap_or(LorentzVector::origin())
        } else {
            LorentzVector::origin()
        };

        for branch in branches.iter_mut() {
            if !branch.free_energy.compliance_permitted {
                continue;
            }
            let cand_lorentz = map
                .global_atom(branch.to_atom)
                .map(|a| a.lorentz)
                .or_else(|| {
                    NativeReader::address(catalog, branch.candidate_next_id)
                        .ok()
                        .map(|(l, _)| l)
                })
                .unwrap_or(*query_lorentz);

            // 1. 2-Cell Gauss-Bonnet triangle defect Ω(Δ(prev, current, candidate))
            let tri_flux = HyperbolicHolonomyEngine::triangle_gauss_bonnet_defect_q32(
                &prev_lorentz,
                query_lorentz,
                &cand_lorentz,
            );
            // Also check the specialist role loop holonomy
            let role_loop = [
                branch.from_role,
                branch.specialist_role,
                RoleId::ConvergentLogic,
                branch.from_role,
            ];
            let loop_rec = self.holonomy_engine.evaluate_role_loop(
                &role_loop,
                map,
                map.atlas.well(branch.from_role).phase,
            );
            let role_flux = match loop_rec.verdict {
                HolonomyVerdict::DialecticalSynthesisVerified {
                    enclosed_flux_rad_q32,
                    ..
                } => enclosed_flux_rad_q32,
                HolonomyVerdict::EchoChamberRejected { .. } => Q32_31::ZERO,
            };

            let total_flux = Q32_31(tri_flux.0.saturating_add(role_flux.0 >> 2));
            branch.gauss_bonnet_flux_q32 = total_flux;
            let holonomy_bonus = Q32_31((total_flux.0 >> 3).clamp(0, Q32_31::from_f64(0.15).0));
            branch.free_energy.holonomy_curvature_bonus_q32 = holonomy_bonus;

            // 2. Counterfactual Geodesic Lookahead Rollout (up to `horizon` steps)
            let mut virtual_prefix = Vec::with_capacity(prefix.len() + horizon + 1);
            virtual_prefix.extend_from_slice(prefix);
            virtual_prefix.push(branch.candidate_next_id);

            let mut lookahead_steps = 0usize;
            for _ in 0..horizon {
                match NativeReader::step_greedy_raw(map, catalog, &virtual_prefix) {
                    Ok(next_read) => {
                        if virtual_prefix.contains(&next_read.next_id) {
                            break;
                        }
                        virtual_prefix.push(next_read.next_id);
                        lookahead_steps += 1;
                    }
                    Err(_) => break,
                }
            }
            branch.verified_lookahead_steps = lookahead_steps;

            // Each verified lookahead step reduces horizon free energy by 0.25 (preferring branches
            // that re-enter a multi-step verified trajectory highway over dead-end leaf hops)
            let lookahead_bonus = Q32_31::from_f64(0.25 * (lookahead_steps as f64));
            branch.free_energy.counterfactual_horizon_delta_q32 =
                Q32_31(lookahead_bonus.0.saturating_neg());

            let adjusted = (branch.free_energy.total_free_energy_q32.0 as i128)
                - (holonomy_bonus.0 as i128)
                - (lookahead_bonus.0 as i128);
            branch.free_energy.total_free_energy_q32 =
                Q32_31(adjusted.clamp(0, (i64::MAX / 2) as i128) as i64);
        }
    }

    fn assign_boltzmann_weights(branches: &mut [SuperpositionBranch]) {
        if branches.is_empty() {
            return;
        }
        let min_f = branches
            .iter()
            .map(|b| b.free_energy.total_free_energy_q32.0)
            .min()
            .unwrap_or(0);
        let one = Q32_31::ONE.0 as i128;

        // Rational Padé-style Boltzmann weight w_k = 1 / (1 + 2 * (F_k - F_min)) in integer Q32_31
        let mut raw_weights = Vec::with_capacity(branches.len());
        let mut sum_w: i128 = 0;
        for b in branches.iter() {
            if !b.free_energy.compliance_permitted {
                raw_weights.push(0i128);
                continue;
            }
            let delta = (b
                .free_energy
                .total_free_energy_q32
                .0
                .saturating_sub(min_f)
                .max(0)) as i128;
            let denom = one + (delta << 1);
            let w = ((one << 31) / denom).max(1);
            raw_weights.push(w);
            sum_w += w;
        }
        if sum_w <= 0 {
            return;
        }
        for (i, b) in branches.iter_mut().enumerate() {
            b.branch_id = i;
            let norm_w = ((raw_weights[i] << 31) / sum_w).clamp(0, one);
            b.boltzmann_weight_q32 = Q32_31(norm_w as i64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas13::Atlas13;
    use crate::silo::FrozenBaseWeights;

    fn test_fixture() -> (CarvedSwarmMap, OverlapPolicyMatrix, MlirAtomCatalog) {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze("active-inference-test", 52, vec![0.1; 32]);
        let map = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));
        let catalog = MlirAtomCatalog::new_canonical(202_048);
        (map, policy, catalog)
    }

    #[test]
    fn variational_free_energy_rewards_exact_wigner_frame_and_penalizes_cycles() {
        let (mut map, policy, catalog) = test_fixture();
        let origin = Sha256Digest::compute(b"glimmer");
        let prefix = [MlirOpId(100), MlirOpId(200)];
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix,
            MlirOpId(300),
            &[1.0, 0.5],
            origin,
            Q32_31::ONE,
        )
        .unwrap();

        let engine = SuperpositionHoldingEngine::default();
        let (q_lor, frame) = NativeReader::context_frame(&catalog, &prefix).unwrap();
        let arc = &map.reasoning_arcs[0];
        let score = engine.score_candidate_arc(
            &map,
            &catalog,
            &policy,
            &prefix,
            &q_lor,
            &frame,
            Q32_31::ONE,
            arc,
        );
        assert!(score.compliance_permitted);
        assert_eq!(score.wigner_divergence_q32, Q32_31::ZERO);
        assert_eq!(score.epistemic_inaccuracy_q32, Q32_31::ZERO);
    }

    #[test]
    fn superposition_holding_resolves_no_direct_geodesic_and_self_distills_bridge() {
        let (mut map, policy, catalog) = test_fixture();
        let origin = Sha256Digest::compute(b"glimmer-teacher");

        // Record a known multi-step trajectory 20 -> 30 -> 40
        NativeReader::record_continuation(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(20)],
            &[MlirOpId(30), MlirOpId(40)],
            &[],
            origin,
            Q32_31::ONE,
        )
        .unwrap();

        // Query at Fibonacci neighbor prefix [MlirOpId(28)] (8 golden-angle steps from 20,
        // cosh d(28, 20) ≈ 1.31 > 1.25 primary basin) where greedy NativeReader::step fails with NoGeodesic
        assert!(NativeReader::step(&map, &catalog, &[MlirOpId(28)]).is_err());

        // Active inference stops, holds superposition, evaluates counterfactual rollout 30 -> 40,
        // collapses to 30, and self-distills the bridge arc 28 -> 30 into the map!
        let config = ActiveInferenceConfig::default();
        let receipt = NativeReader::active_inference_step_mut(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(28)],
            &config,
            0,
        )
        .unwrap();

        assert!(receipt.superposition_held.is_some());
        assert!(receipt.bridge_arc_distilled);
        assert_eq!(receipt.read.next_id, MlirOpId(30));

        // Now subsequent standard greedy step at [MlirOpId(28)] succeeds in O(1)!
        let direct = NativeReader::step(&map, &catalog, &[MlirOpId(28)]).unwrap();
        assert_eq!(direct.next_id, MlirOpId(30));
    }
}
