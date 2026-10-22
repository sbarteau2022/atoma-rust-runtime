//! Native reader: one generation step whose only model is `CarvedSwarmMap`,
//! upgraded with Active Inference, Variational Free Energy (`F_q32`) minimization,
//! and Superposition Holding (`design/superpositionholding.md`).
//!
//! An ML-IR ID is a point on `(H³ × T²)`. The prefix is a path of those points.
//! The reader stands at the last point, queries the global atoms (every origin
//! fingerprint), and follows the 1-cell whose stored gyroscope matches that path.
//! The arc's `compiled_mlir_op` is the next ML-IR ID.
//!
//! A token trajectory plants the from-atom on that same point and stores the
//! prefix gyroscope on the arc. The hidden state rides in the atom payload.
//!
//! Layer count, hidden width, expert count, and vocabulary size are not inputs.

use crate::atlas13::{EpistemicMark, RoleId};
use crate::cache::geometric::{AtomId, Phase};
use crate::compliance::OverlapPolicyMatrix;
use crate::distill_engine::DistilledReasoningArc;
use crate::first_principles::{MetaControlDecision, OpticalRiskEngine, RoleSignalQ32, RunbookTelemetryQ32};
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::holonomy::{HyperbolicHolonomyEngine, IDENTITY_FRAME_Q32};
use crate::lock::extract_invariant;
use crate::mlir_native::{MlirAtomCatalog, MlirOpId};
use crate::refraction::{
    OpticalLens, ProjectionLossReport, RefractedPoint, RefractionLossReport, Sha256Digest,
    rad_q32_to_pi_unit,
};
use crate::silo::CarvedSwarmMap;
use crate::superposition::{
    ActiveInferenceConfig, ActiveInferenceContinuation, ActiveInferenceStepReceipt,
    ActiveInferenceStop, SuperpositionHoldingEngine, SuperpositionResolution, SuperpositionState,
    SuperpositionTrigger,
};

/// `cosh` of the widest query the reader will treat as standing on the map.
/// Farther than this, the nearest arc is a different place and is not a continuation.
fn query_basin_cosh() -> i64 {
    Q32_31::from_f64(1.25).0
}

/// Atoms this close to the nearest arc-bearing atom (in `cosh`) share its basin.
fn neighborhood_slack() -> i64 {
    Q32_31::from_f64(0.05).0
}

/// One geometric continuation read from the global map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRead {
    pub next_id: MlirOpId,
    pub arc_id: u64,
    pub from_atom: AtomId,
    pub to_atom: AtomId,
    pub query_cosh_q32: Q32_31,
    pub cosh_action_q32: Q32_31,
    /// Normalized Frobenius cosine alignment of the prefix gyroscope and the arc's stored Wigner frame.
    pub frame_alignment_q32: Q32_31,
    pub origin_model_fingerprint: Sha256Digest,
}

/// Why a multi-step read stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeStop {
    MaxNew,
    Eos(MlirOpId),
    FixedPoint(MlirOpId),
    NoGeodesic { at: MlirOpId, detail: String },
}

/// IDs read from the map, and why the walk stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeContinuation {
    pub reads: Vec<NativeRead>,
    pub stop: NativeStop,
}

/// One token step written into the map, and the read that step now produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectoryWrite {
    pub read: NativeRead,
    pub arc_was_novel: bool,
}

/// The play path. Distillation fills the map. This reads it.
pub struct NativeReader;

impl NativeReader {
    /// Place an ML-IR ID on `(H³ × T²)`.
    ///
    /// A registered catalog opcode keeps the well the catalog pinned.
    /// Every other ID, including a token carrier from any vocabulary, is a
    /// pure function of the integer. There is no embedding matrix.
    pub fn address(
        catalog: &MlirAtomCatalog,
        id: MlirOpId,
    ) -> Result<(LorentzVector, Phase), String> {
        if let Some(desc) = catalog.get_op(id) {
            return Ok((desc.lorentz_well, desc.phase_well));
        }
        token_carrier_address(id.0)
    }

    /// Compute the arrival Lorentz point and parallel-transported Wigner frame for `prefix`.
    pub fn context_frame(
        catalog: &MlirAtomCatalog,
        prefix: &[MlirOpId],
    ) -> Result<(LorentzVector, [[Q32_31; 3]; 3]), String> {
        context(catalog, prefix)
    }

    /// Read the next ML-IR ID from `map` given the IDs so far.
    pub fn step(
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        prefix: &[MlirOpId],
    ) -> Result<NativeRead, String> {
        Self::step_greedy_raw(map, catalog, prefix)
    }

    /// Direct single-step geodesic read using normalized Wigner frame alignment,
    /// Frobenius residual distance, and causal `chain_id` matching.
    pub fn step_greedy_raw(
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        prefix: &[MlirOpId],
    ) -> Result<NativeRead, String> {
        let Some(&last) = prefix.last() else {
            return Err("native reader needs at least one ML-IR ID".to_string());
        };
        let (query, frame) = context(catalog, prefix)?;
        let (_, query_phase) = Self::address(catalog, last)?;
        let expected_chain = prefix_chain_id(prefix);

        if map.reasoning_arcs.is_empty() {
            return Err("map has no reasoning arcs".to_string());
        }

        let Some((best_atom, best_cosh, candidates)) =
            map.candidate_arc_atoms_near(&query, query_phase, neighborhood_slack())
        else {
            return Err(format!("no arc-bearing atom on the map for {last}"));
        };
        if best_cosh > query_basin_cosh() {
            return Err(format!(
                "no geodesic within cosh {:.4} of {last} (nearest arc leaves atom {best_atom} at cosh {:.4})",
                Q32_31(query_basin_cosh()).to_f64(),
                Q32_31(best_cosh).to_f64(),
            ));
        }

        let (winner, query_cosh) = candidates
            .iter()
            .flat_map(|&(aid, c)| {
                map.outgoing_arc_indices(AtomId(aid))
                    .iter()
                    .map(move |&idx| (&map.reasoning_arcs[idx], c))
            })
            .min_by(|(a, _), (b, _)| {
                // 1. Exact causal prefix chain + exact Wigner frame match wins unconditionally
                let exact_a = a.chain_id == expected_chain && a.wigner_frame == frame;
                let exact_b = b.chain_id == expected_chain && b.wigner_frame == frame;
                // 2. Normalized cosine alignment in Q32_31 (eliminates Frobenius norm inflation bias)
                let align_a = normalized_frame_alignment_q32(&a.wigner_frame, &frame).0;
                let align_b = normalized_frame_alignment_q32(&b.wigner_frame, &frame).0;
                // 3. Exact Frobenius squared residual distance ||A - B||_F^2 (0 iff A == B)
                let dist_a = frame_frobenius_dist_sq_q32(&a.wigner_frame, &frame).0;
                let dist_b = frame_frobenius_dist_sq_q32(&b.wigner_frame, &frame).0;

                exact_b
                    .cmp(&exact_a)
                    .then(align_b.cmp(&align_a))
                    .then(dist_a.cmp(&dist_b))
                    .then(a.cosh_action_q32.0.cmp(&b.cosh_action_q32.0))
                    .then(b.teacher_certainty_q32.0.cmp(&a.teacher_certainty_q32.0))
                    .then(a.arc_id.cmp(&b.arc_id))
            })
            .ok_or_else(|| format!("basin of atom {best_atom} has no outgoing arc"))?;

        Ok(NativeRead {
            next_id: winner.compiled_mlir_op,
            arc_id: winner.arc_id,
            from_atom: winner.from_atom,
            to_atom: winner.to_atom,
            query_cosh_q32: Q32_31(query_cosh),
            cosh_action_q32: winner.cosh_action_q32,
            frame_alignment_q32: normalized_frame_alignment_q32(&winner.wigner_frame, &frame),
            origin_model_fingerprint: winner.origin_model_fingerprint,
        })
    }

    /// Execute one **Active Inference** step on `(H³ × T²)/Γ₁₂`:
    /// - Scores all candidate transitions by **Variational Free Energy (`F_q32`)** (combining
    ///   geodesic surprise, normalized Wigner holonomy divergence, teacher inaccuracy,
    ///   Layer 3 compliance/safety/blast-radius potential, and token-efficiency cost).
    /// - If an issue is detected (`NoDirectGeodesic`, `FixedPointOrCycle`, `WignerFrameAmbiguity`,
    ///   `FreeEnergyOrReadinessSpike`, or `ComplianceOrSafetyBoundary`), stops greedy emission,
    ///   invokes `SuperpositionHoldingEngine::hold_and_iterate` (`design/superpositionholding.md`),
    ///   and dynamically iterates across counterfactual lookaheads and `2-cell` Gauss–Bonnet
    ///   holonomy loops to resolve a compliant minimum-free-energy solution.
    pub fn active_inference_step(
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        config: &ActiveInferenceConfig,
        spent_tokens: u64,
    ) -> Result<ActiveInferenceStepReceipt, SuperpositionState> {
        let engine = SuperpositionHoldingEngine::new(config.clone());
        let last = prefix.last().copied().unwrap_or(MlirOpId(0));
        let Ok((query_lorentz, prefix_frame)) = context(catalog, prefix) else {
            let res = engine.hold_and_iterate(
                map,
                catalog,
                policy,
                prefix,
                SuperpositionTrigger::NoDirectGeodesic {
                    at_op: last,
                    nearest_cosh_q32: Q32_31(i64::MAX / 2),
                },
                spent_tokens,
            );
            return Self::finalize_superposition_resolution(res);
        };
        let query_phase = Self::address(catalog, last)
            .map(|(_, p)| p)
            .unwrap_or(Phase {
                theta_f: Q32_31::ZERO,
                theta_s: Q32_31::ZERO,
            });

        let expected_chain = prefix_chain_id(prefix);

        let Some((_best_atom, best_cosh, candidates)) =
            map.candidate_arc_atoms_near(&query_lorentz, query_phase, neighborhood_slack())
        else {
            let res = engine.hold_and_iterate(
                map,
                catalog,
                policy,
                prefix,
                SuperpositionTrigger::NoDirectGeodesic {
                    at_op: last,
                    nearest_cosh_q32: Q32_31(i64::MAX / 2),
                },
                spent_tokens,
            );
            return Self::finalize_superposition_resolution(res);
        };

        // Trigger 1: No Direct Geodesic within primary basin
        if best_cosh > config.primary_basin_cosh_q32.0 {
            let res = engine.hold_and_iterate(
                map,
                catalog,
                policy,
                prefix,
                SuperpositionTrigger::NoDirectGeodesic {
                    at_op: last,
                    nearest_cosh_q32: Q32_31(best_cosh),
                },
                spent_tokens,
            );
            return Self::finalize_superposition_resolution(res);
        }

        // Score all outgoing arcs in the primary basin by Variational Free Energy via O(1) 27-cell spatial lookup
        let mut scored_arcs: Vec<(
            &DistilledReasoningArc,
            Q32_31,
            crate::superposition::VariationalFreeEnergyScore,
            bool,
        )> = candidates
            .iter()
            .flat_map(|&(aid, c)| {
                let q_cosh = Q32_31(c);
                map.outgoing_arc_indices(AtomId(aid))
                    .iter()
                    .map(move |&idx| (&map.reasoning_arcs[idx], q_cosh))
            })
            .map(|(arc, q_cosh)| {
                let fe = engine.score_candidate_arc(
                    map,
                    catalog,
                    policy,
                    prefix,
                    &query_lorentz,
                    &prefix_frame,
                    q_cosh,
                    arc,
                );
                let exact = arc.chain_id == expected_chain && arc.wigner_frame == prefix_frame;
                (arc, q_cosh, fe, exact)
            })
            .collect();

        scored_arcs.sort_by(|a, b| {
            b.3.cmp(&a.3)
                .then(
                    a.2.total_free_energy_q32
                        .0
                        .cmp(&b.2.total_free_energy_q32.0),
                )
                .then(a.0.cosh_action_q32.0.cmp(&b.0.cosh_action_q32.0))
                .then(a.0.arc_id.cmp(&b.0.arc_id))
        });

        let Some(&(best_arc, best_q_cosh, best_fe, best_exact)) = scored_arcs.first() else {
            let res = engine.hold_and_iterate(
                map,
                catalog,
                policy,
                prefix,
                SuperpositionTrigger::NoDirectGeodesic {
                    at_op: last,
                    nearest_cosh_q32: Q32_31(best_cosh),
                },
                spent_tokens,
            );
            return Self::finalize_superposition_resolution(res);
        };

        // Trigger 2: Compliance or Safety Boundary Violation
        if !best_fe.compliance_permitted {
            let to_lor = map
                .global_atom(best_arc.to_atom)
                .map(|a| a.lorentz)
                .unwrap_or(query_lorentz);
            let blast = OpticalRiskEngine::optical_blast_radius_q32(&to_lor);
            let res = engine.hold_and_iterate(
                map,
                catalog,
                policy,
                prefix,
                SuperpositionTrigger::ComplianceOrSafetyBoundary {
                    at_op: last,
                    blocked_op: best_arc.compiled_mlir_op,
                    blast_radius_q32: blast,
                },
                spent_tokens,
            );
            return Self::finalize_superposition_resolution(res);
        }

        // Trigger 3: FixedPoint / Self-Loop or Cycle
        if best_arc.compiled_mlir_op == last {
            let res = engine.hold_and_iterate(
                map,
                catalog,
                policy,
                prefix,
                SuperpositionTrigger::FixedPointOrCycle {
                    at_op: last,
                    repeated_op: best_arc.compiled_mlir_op,
                },
                spent_tokens,
            );
            return Self::finalize_superposition_resolution(res);
        }

        // Trigger 4: Wigner Frame / Free-Energy Ambiguity between distinct next_ids (when not an exact chain match)
        if !best_exact {
            if let Some(&(second_arc, _, second_fe, _)) = scored_arcs
                .iter()
                .find(|(arc, _, _, _)| arc.compiled_mlir_op != best_arc.compiled_mlir_op)
            {
                let gap = Q32_31(
                    second_fe
                        .total_free_energy_q32
                        .0
                        .saturating_sub(best_fe.total_free_energy_q32.0)
                        .max(0),
                );
                if gap.0 < config.ambiguity_margin_q32.0 {
                    let res = engine.hold_and_iterate(
                        map,
                        catalog,
                        policy,
                        prefix,
                        SuperpositionTrigger::WignerFrameAmbiguity {
                            at_op: last,
                            candidate_a: best_arc.compiled_mlir_op,
                            candidate_b: second_arc.compiled_mlir_op,
                            free_energy_gap_q32: gap,
                        },
                        spent_tokens,
                    );
                    return Self::finalize_superposition_resolution(res);
                }
            }
        }

        // Evaluate Meta-Cognitive Decision Readiness Ω
        let one = Q32_31::ONE.0 as i128;
        let denom = one + (best_fe.total_free_energy_q32.0.max(0) as i128);
        let conf = Q32_31((((one << 31) / denom).clamp(0, one)) as i64);
        let signal = RoleSignalQ32 {
            role: best_arc.to_role,
            hypothesis_hash: best_arc.compiled_mlir_op.0 as u64,
            proposed_action_hash: best_arc.compiled_mlir_op.0 as u64,
            confidence_q32: conf,
        };
        let to_lor = map
            .global_atom(best_arc.to_atom)
            .map(|a| a.lorentz)
            .unwrap_or(query_lorentz);
        let telemetry = RunbookTelemetryQ32 {
            step: prefix.len() as u32,
            kappa_q32: Q32_31::from_f64(0.92),
            drift_detected: false,
            out_of_region: false,
            regime_collapse: false,
            verified_progress_in_window: 1,
            spent_tokens,
            predicted_token_envelope: config.predicted_token_envelope,
            nudges_used: 0,
        };
        let verdict = engine
            .meta_controller
            .evaluate(&[signal], &to_lor, &telemetry);

        // Trigger 5: Free Energy Exceeds Ceiling or Readiness Ω Below Floor
        if !best_exact
            && (best_fe.total_free_energy_q32.0 > config.free_energy_ceiling_q32.0
                || verdict.omega_readiness_q32.0 < config.min_omega_readiness_q32.0)
        {
            let res = engine.hold_and_iterate(
                map,
                catalog,
                policy,
                prefix,
                SuperpositionTrigger::FreeEnergyOrReadinessSpike {
                    at_op: last,
                    free_energy_q32: best_fe.total_free_energy_q32,
                    omega_readiness_q32: verdict.omega_readiness_q32,
                },
                spent_tokens,
            );
            return Self::finalize_superposition_resolution(res);
        }

        Ok(ActiveInferenceStepReceipt {
            read: NativeRead {
                next_id: best_arc.compiled_mlir_op,
                arc_id: best_arc.arc_id,
                from_atom: best_arc.from_atom,
                to_atom: best_arc.to_atom,
                query_cosh_q32: best_q_cosh,
                cosh_action_q32: best_arc.cosh_action_q32,
                frame_alignment_q32: normalized_frame_alignment_q32(
                    &best_arc.wigner_frame,
                    &prefix_frame,
                ),
                origin_model_fingerprint: best_arc.origin_model_fingerprint,
            },
            free_energy: best_fe,
            omega_readiness_q32: verdict.omega_readiness_q32,
            meta_decision: verdict.decision,
            superposition_held: None,
            bridge_arc_distilled: false,
            mark: EpistemicMark::Verified,
        })
    }

    /// Mutable Active Inference step that **self-distills** any newly resolved superposition
    /// bridge trajectory back into `map` so subsequent traversals resolve in `O(1)` without
    /// re-entering superposition (`design/superpositionholding.md` §3 Phase 4).
    pub fn active_inference_step_mut(
        map: &mut CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        config: &ActiveInferenceConfig,
        spent_tokens: u64,
    ) -> Result<ActiveInferenceStepReceipt, SuperpositionState> {
        let mut receipt =
            Self::active_inference_step(map, catalog, policy, prefix, config, spent_tokens)?;
        if receipt.superposition_held.is_some() {
            // Self-distill the resolved bridge transition into the map
            if let Ok(wrote) = Self::record_step(
                map,
                catalog,
                policy,
                prefix,
                receipt.read.next_id,
                &[],
                receipt.read.origin_model_fingerprint,
                Q32_31::from_f64(0.90),
            ) {
                receipt.read = wrote.read;
                receipt.bridge_arc_distilled = wrote.arc_was_novel;
            }
        }
        Ok(receipt)
    }

    fn finalize_superposition_resolution(
        res: SuperpositionResolution,
    ) -> Result<ActiveInferenceStepReceipt, SuperpositionState> {
        match res {
            SuperpositionResolution::CollapsedToSolution {
                winning_branch,
                iterations_used: _,
                bridge_arc_distilled,
                state,
            } => {
                let omega = state
                    .iteration_traces
                    .last()
                    .map(|t| t.omega_readiness_q32)
                    .unwrap_or(Q32_31::from_f64(0.85));
                let decision = state
                    .iteration_traces
                    .last()
                    .map(|t| t.meta_decision)
                    .unwrap_or(MetaControlDecision::ContinueAutonomous);
                Ok(ActiveInferenceStepReceipt {
                    read: NativeRead {
                        next_id: winning_branch.candidate_next_id,
                        arc_id: winning_branch.arc_id,
                        from_atom: winning_branch.from_atom,
                        to_atom: winning_branch.to_atom,
                        query_cosh_q32: winning_branch.query_cosh_q32,
                        cosh_action_q32: winning_branch.cosh_action_q32,
                        frame_alignment_q32: winning_branch.frame_alignment_q32,
                        origin_model_fingerprint: winning_branch.origin_model_fingerprint,
                    },
                    free_energy: winning_branch.free_energy,
                    omega_readiness_q32: omega,
                    meta_decision: decision,
                    superposition_held: Some(state),
                    bridge_arc_distilled,
                    mark: EpistemicMark::Verified,
                })
            }
            SuperpositionResolution::FailClosed { reason: _, state } => Err(state),
        }
    }

    /// Multi-step **Active Inference** continuation (read-only map).
    pub fn continue_ids_active(
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        max_new: usize,
        eos: &[MlirOpId],
        config: &ActiveInferenceConfig,
    ) -> Result<ActiveInferenceContinuation, String> {
        if prefix.is_empty() {
            return Err("native reader needs at least one ML-IR ID".to_string());
        }
        let mut prefix_buf = prefix.to_vec();
        let mut steps = Vec::new();
        let mut total_fe: i128 = 0;
        let mut interventions = 0usize;

        if max_new == 0 {
            return Ok(ActiveInferenceContinuation {
                steps,
                stop: ActiveInferenceStop::MaxNew,
                total_free_energy_q32: Q32_31::ZERO,
                superposition_interventions: 0,
                bridge_arcs_distilled: 0,
            });
        }

        for step_idx in 0..max_new {
            let spent = step_idx as u64;
            if config.predicted_token_envelope > 0 && spent >= config.predicted_token_envelope {
                return Ok(ActiveInferenceContinuation {
                    steps,
                    stop: ActiveInferenceStop::TokenEnvelopeExhausted {
                        spent_tokens: spent,
                        envelope_tokens: config.predicted_token_envelope,
                    },
                    total_free_energy_q32: Q32_31(
                        total_fe.clamp(0, i64::MAX as i128) as i64,
                    ),
                    superposition_interventions: interventions,
                    bridge_arcs_distilled: 0,
                });
            }

            match Self::active_inference_step(map, catalog, policy, &prefix_buf, config, spent) {
                Ok(receipt) => {
                    let next = receipt.read.next_id;
                    total_fe += receipt.free_energy.total_free_energy_q32.0 as i128;
                    if receipt.superposition_held.is_some() {
                        interventions += 1;
                    }
                    steps.push(receipt);
                    if eos.contains(&next) {
                        return Ok(ActiveInferenceContinuation {
                            steps,
                            stop: ActiveInferenceStop::Eos(next),
                            total_free_energy_q32: Q32_31(
                                total_fe.clamp(0, i64::MAX as i128) as i64,
                            ),
                            superposition_interventions: interventions,
                            bridge_arcs_distilled: 0,
                        });
                    }
                    prefix_buf.push(next);
                }
                Err(state) => {
                    let at = prefix_buf[prefix_buf.len() - 1];
                    return Ok(ActiveInferenceContinuation {
                        steps,
                        stop: ActiveInferenceStop::SuperpositionFailClosed {
                            at,
                            reason: format!(
                                "superposition fail-closed at {} after {} iterations",
                                at,
                                state.iteration_traces.len()
                            ),
                            state,
                        },
                        total_free_energy_q32: Q32_31(
                            total_fe.clamp(0, i64::MAX as i128) as i64,
                        ),
                        superposition_interventions: interventions + 1,
                        bridge_arcs_distilled: 0,
                    });
                }
            }
        }

        Ok(ActiveInferenceContinuation {
            steps,
            stop: ActiveInferenceStop::MaxNew,
            total_free_energy_q32: Q32_31(total_fe.clamp(0, i64::MAX as i128) as i64),
            superposition_interventions: interventions,
            bridge_arcs_distilled: 0,
        })
    }

    /// Multi-step **Active Inference** continuation with online bridge-arc self-distillation.
    pub fn continue_ids_active_mut(
        map: &mut CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        max_new: usize,
        eos: &[MlirOpId],
        config: &ActiveInferenceConfig,
    ) -> Result<ActiveInferenceContinuation, String> {
        if prefix.is_empty() {
            return Err("native reader needs at least one ML-IR ID".to_string());
        }
        let mut prefix_buf = prefix.to_vec();
        let mut steps = Vec::new();
        let mut total_fe: i128 = 0;
        let mut interventions = 0usize;
        let mut distilled_bridges = 0usize;

        for step_idx in 0..max_new {
            let spent = step_idx as u64;
            if config.predicted_token_envelope > 0 && spent >= config.predicted_token_envelope {
                return Ok(ActiveInferenceContinuation {
                    steps,
                    stop: ActiveInferenceStop::TokenEnvelopeExhausted {
                        spent_tokens: spent,
                        envelope_tokens: config.predicted_token_envelope,
                    },
                    total_free_energy_q32: Q32_31(
                        total_fe.clamp(0, i64::MAX as i128) as i64,
                    ),
                    superposition_interventions: interventions,
                    bridge_arcs_distilled: distilled_bridges,
                });
            }

            match Self::active_inference_step_mut(map, catalog, policy, &prefix_buf, config, spent)
            {
                Ok(receipt) => {
                    let next = receipt.read.next_id;
                    total_fe += receipt.free_energy.total_free_energy_q32.0 as i128;
                    if receipt.superposition_held.is_some() {
                        interventions += 1;
                    }
                    if receipt.bridge_arc_distilled {
                        distilled_bridges += 1;
                    }
                    steps.push(receipt);
                    if eos.contains(&next) {
                        return Ok(ActiveInferenceContinuation {
                            steps,
                            stop: ActiveInferenceStop::Eos(next),
                            total_free_energy_q32: Q32_31(
                                total_fe.clamp(0, i64::MAX as i128) as i64,
                            ),
                            superposition_interventions: interventions,
                            bridge_arcs_distilled: distilled_bridges,
                        });
                    }
                    prefix_buf.push(next);
                }
                Err(state) => {
                    let at = prefix_buf[prefix_buf.len() - 1];
                    return Ok(ActiveInferenceContinuation {
                        steps,
                        stop: ActiveInferenceStop::SuperpositionFailClosed {
                            at,
                            reason: format!(
                                "superposition fail-closed at {} after {} iterations",
                                at,
                                state.iteration_traces.len()
                            ),
                            state,
                        },
                        total_free_energy_q32: Q32_31(
                            total_fe.clamp(0, i64::MAX as i128) as i64,
                        ),
                        superposition_interventions: interventions + 1,
                        bridge_arcs_distilled: distilled_bridges,
                    });
                }
            }
        }

        Ok(ActiveInferenceContinuation {
            steps,
            stop: ActiveInferenceStop::MaxNew,
            total_free_energy_q32: Q32_31(total_fe.clamp(0, i64::MAX as i128) as i64),
            superposition_interventions: interventions,
            bridge_arcs_distilled: distilled_bridges,
        })
    }

    /// Walk up to `max_new` continuations, appending each ID to the prefix.
    pub fn continue_ids(
        map: &CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        prefix: &[MlirOpId],
        max_new: usize,
        eos: &[MlirOpId],
    ) -> Result<NativeContinuation, String> {
        if prefix.is_empty() {
            return Err("native reader needs at least one ML-IR ID".to_string());
        }
        let mut prefix_buf = prefix.to_vec();
        let mut reads = Vec::new();
        if max_new == 0 {
            return Ok(NativeContinuation {
                reads,
                stop: NativeStop::MaxNew,
            });
        }
        for _ in 0..max_new {
            match Self::step(map, catalog, &prefix_buf) {
                Ok(read) => {
                    let next = read.next_id;
                    let at = prefix_buf[prefix_buf.len() - 1];
                    reads.push(read);
                    if eos.contains(&next) {
                        return Ok(NativeContinuation {
                            reads,
                            stop: NativeStop::Eos(next),
                        });
                    }
                    if next == at {
                        return Ok(NativeContinuation {
                            reads,
                            stop: NativeStop::FixedPoint(next),
                        });
                    }
                    prefix_buf.push(next);
                }
                Err(detail) => {
                    let at = prefix_buf[prefix_buf.len() - 1];
                    return Ok(NativeContinuation {
                        reads,
                        stop: NativeStop::NoGeodesic { at, detail },
                    });
                }
            }
        }
        Ok(NativeContinuation {
            reads,
            stop: NativeStop::MaxNew,
        })
    }

    /// Write one teacher step onto the wells the reader queries.
    ///
    /// `prefix` is the ML-IR IDs so far. `next_id` is the teacher's next ID.
    /// `hidden_state` is the teacher state at the last prefix ID.
    pub fn record_step(
        map: &mut CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        next_id: MlirOpId,
        hidden_state: &[f32],
        origin: Sha256Digest,
        certainty: Q32_31,
    ) -> Result<TrajectoryWrite, String> {
        if prefix.is_empty() {
            return Err("native reader needs at least one ML-IR ID".to_string());
        }
        let (_query, frame) = context(catalog, prefix)?;
        let last = prefix[prefix.len() - 1];
        let from_atom = plant_carrier(map, policy, catalog, last, hidden_state, origin)?;
        let to_atom = plant_carrier(map, policy, catalog, next_id, &[], origin)?;
        let from_lor = map
            .global_atom(from_atom)
            .map(|atom| atom.lorentz)
            .ok_or_else(|| format!("from atom {} missing after plant", from_atom.0))?;
        let to_lor = map
            .global_atom(to_atom)
            .map(|atom| atom.lorentz)
            .ok_or_else(|| format!("to atom {} missing after plant", to_atom.0))?;
        let cosh_action = OpticalLens::lorentz_inner_product_wide(&from_lor, &to_lor);
        let (_arc_id, arc_was_novel) = map.record_trajectory_arc(DistilledReasoningArc {
            arc_id: 0,
            chain_id: prefix_chain_id(prefix),
            step_index: prefix.len() - 1,
            from_atom,
            to_atom,
            from_role: RoleId::LanguageSemiotics,
            to_role: RoleId::PredictiveSimulation,
            cosh_action_q32: cosh_action,
            wigner_frame: frame,
            compiled_mlir_op: next_id,
            teacher_certainty_q32: certainty,
            origin_model_fingerprint: origin,
        });
        let read = Self::step(map, catalog, prefix)?;
        if read.next_id != next_id && map.locked_arc_count == 0 {
            return Err(format!(
                "recorded mlir_op#{} and the reader emitted {} for this prefix",
                next_id.0, read.next_id.0
            ));
        }
        Ok(TrajectoryWrite {
            read,
            arc_was_novel,
        })
    }

    /// Write each generated ID as the next step of `prefix`, extending the prefix as it goes.
    ///
    /// An empty `hidden_at_step` stores no teacher state. A non-empty slice has one state per
    /// generated ID, the state at the prefix the reader is standing on when that ID is emitted.
    pub fn record_continuation(
        map: &mut CarvedSwarmMap,
        catalog: &MlirAtomCatalog,
        policy: &OverlapPolicyMatrix,
        prefix: &[MlirOpId],
        generated: &[MlirOpId],
        hidden_at_step: &[Vec<f32>],
        origin: Sha256Digest,
        certainty: Q32_31,
    ) -> Result<Vec<TrajectoryWrite>, String> {
        if !hidden_at_step.is_empty() && hidden_at_step.len() != generated.len() {
            return Err("hidden states must cover every generated ID".to_string());
        }
        let mut prefix_buf = prefix.to_vec();
        let mut wrote = Vec::with_capacity(generated.len());
        for (i, next) in generated.iter().copied().enumerate() {
            let hidden = hidden_at_step.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let step = Self::record_step(
                map,
                catalog,
                policy,
                &prefix_buf,
                next,
                hidden,
                origin,
                certainty,
            )?;
            prefix_buf.push(next);
            wrote.push(step);
        }
        Ok(wrote)
    }
}

/// Arrival point and gyroscope after walking the prefix from the hyperboloid origin.
fn context(
    catalog: &MlirAtomCatalog,
    prefix: &[MlirOpId],
) -> Result<(LorentzVector, [[Q32_31; 3]; 3]), String> {
    let (mut lor, _) = NativeReader::address(catalog, prefix[0])?;
    let mut frame = HyperbolicHolonomyEngine::transport_frame_hop(
        &LorentzVector::origin(),
        &lor,
        &IDENTITY_FRAME_Q32,
    );
    for id in &prefix[1..] {
        let (next, _) = NativeReader::address(catalog, *id)?;
        frame = HyperbolicHolonomyEngine::transport_frame_hop(&lor, &next, &frame);
        lor = next;
    }
    Ok((lor, frame))
}

/// Carrier address of an unregistered ID with phyllotaxis hyperbolic placement
/// and `TorPhaseEncoder` amplitude-coupled phase state.
fn token_carrier_address(id: u32) -> Result<(LorentzVector, Phase), String> {
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
    let enc = crate::geometry::torus::TorPhaseEncoder::new();
    let (df, ds) = enc.project_increments(&id.to_le_bytes());
    let state = enc.evolve(crate::geometry::torus::PhaseState::ZERO, df, ds);

    let n = (id as f64) + 1.0;
    let radius = 0.20 + 0.55 * (n * 0.618_033_988_749_895).fract();
    let theta = n * GOLDEN_ANGLE;
    let z = 0.12 * ((((id.wrapping_mul(0x9E37_79B9)) >> 16) as f64) / 65536.0 - 0.5);
    let x = radius * theta.cos();
    let y = radius * theta.sin();
    let ball = [
        Q32_31::from_f64(x),
        Q32_31::from_f64(y),
        Q32_31::from_f64(z),
    ];
    let lorentz = OpticalLens::ball_to_lorentz(&ball).map_err(|e| e.to_string())?;
    let theta_f = ((id % 2048) as f64) * (std::f64::consts::TAU / 2048.0) - std::f64::consts::PI;
    let theta_s = ((id / 2048) as f64).rem_euclid(2048.0) * (std::f64::consts::TAU / 2048.0)
        - std::f64::consts::PI;
    let phase = Phase::with_amplitude(
        Q32_31::from_f64(theta_f),
        Q32_31::from_f64(theta_s),
        Q32_31::from_f64(state.amplitude.to_f64() * 0.02),
    );
    Ok((lorentz, phase))
}

/// Plants or coalesces a token carrier atom without allocating a `26.6 KB` payload buffer
/// when the carrier atom already exists on the map with a non-empty payload.
fn plant_carrier(
    map: &mut CarvedSwarmMap,
    policy: &OverlapPolicyMatrix,
    catalog: &MlirAtomCatalog,
    id: MlirOpId,
    hidden_state: &[f32],
    origin: Sha256Digest,
) -> Result<AtomId, String> {
    let point = refract_id(catalog, id)?;
    let (atom, _novel) = map.keep_and_carve_novel_only(point, Vec::new(), origin, policy);
    if !hidden_state.is_empty()
        && map
            .global_atom(atom)
            .is_some_and(|existing| existing.payload.is_empty())
    {
        map.store_payload_if_empty(atom, hidden_payload(hidden_state));
    }
    Ok(atom)
}

/// Lift an ML-IR ID to the refracted point `address` returns.
fn refract_id(catalog: &MlirAtomCatalog, id: MlirOpId) -> Result<RefractedPoint, String> {
    let (lorentz, phase) = NativeReader::address(catalog, id)?;
    Ok(RefractedPoint {
        poincare_ball: OpticalLens::lorentz_to_ball(&lorentz),
        lorentz,
        phase,
        invariant: extract_invariant(
            rad_q32_to_pi_unit(phase.theta_f),
            rad_q32_to_pi_unit(phase.theta_s),
        ),
        projection_loss: ProjectionLossReport {
            input_state_bits: 32,
            output_tangent_bits: 256,
            discarded_dimensions: 0,
            retained_energy_ratio_q32: Q32_31::ONE,
            discarded_energy_ratio_q32: Q32_31::ZERO,
            teacher_entropy_bits_q32: Q32_31::ZERO,
        },
        refraction_loss: RefractionLossReport {
            quantization_ulp_bound: 1,
            hyperboloid_residual_q32: Q32_31::ZERO,
            refractive_index_q32: Q32_31::ONE,
        },
    })
}

fn hidden_payload(hidden: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(hidden.len() * 4);
    for &v in hidden {
        let canonical = if v == 0.0 { 0.0 } else { v };
        buf.extend_from_slice(&canonical.to_le_bytes());
    }
    buf
}

/// Compute the deterministic 64-bit FNV-style chain identifier for a prefix sequence.
pub fn prefix_chain_id(prefix: &[MlirOpId]) -> u64 {
    let mut acc = 0x9E37_79B9_7F4A_7C15u64;
    for id in prefix {
        acc = acc
            .wrapping_mul(0x0000_0100_0000_01B3)
            .wrapping_add(u64::from(id.0));
    }
    acc
}

/// Normalized Frobenius cosine alignment `<A, B>_F / (||A||_F * ||B||_F)` in `Q32_31` (`[-1, 1]`).
/// Eliminates the self-norm inflation bias of unnormalized Frobenius dot products across long
/// Lorentz parallel-transport prefixes.
pub fn normalized_frame_alignment_q32(a: &[[Q32_31; 3]; 3], b: &[[Q32_31; 3]; 3]) -> Q32_31 {
    if a == b {
        return Q32_31::ONE;
    }
    let mut dot: i128 = 0;
    let mut norm_a_sq: i128 = 0;
    let mut norm_b_sq: i128 = 0;
    for r in 0..3 {
        for c in 0..3 {
            let av = a[r][c].0 as i128;
            let bv = b[r][c].0 as i128;
            dot += (av * bv) >> 31;
            norm_a_sq += (av * av) >> 31;
            norm_b_sq += (bv * bv) >> 31;
        }
    }
    if norm_a_sq <= 0 || norm_b_sq <= 0 {
        return Q32_31::ZERO;
    }
    let na = OpticalLens::sqrt_q32(Q32_31(norm_a_sq.clamp(0, i64::MAX as i128) as i64)).0 as i128;
    let nb = OpticalLens::sqrt_q32(Q32_31(norm_b_sq.clamp(0, i64::MAX as i128) as i64)).0 as i128;
    let denom = (na * nb) >> 31;
    if denom <= 0 {
        return Q32_31::ZERO;
    }
    let cos = (dot << 31) / denom;
    let one = Q32_31::ONE.0 as i128;
    Q32_31(cos.clamp(-one, one) as i64)
}

/// Exact Frobenius squared residual `||A - B||_F^2 = sum_{r,c} (A_{rc} - B_{rc})^2` in `Q32_31`.
/// Guaranteed to be `0` if and only if `A == B` bit-for-bit.
pub fn frame_frobenius_dist_sq_q32(a: &[[Q32_31; 3]; 3], b: &[[Q32_31; 3]; 3]) -> Q32_31 {
    let mut acc: i128 = 0;
    for r in 0..3 {
        for c in 0..3 {
            let diff = (a[r][c].0 as i128) - (b[r][c].0 as i128);
            acc += (diff * diff) >> 31;
        }
    }
    Q32_31(acc.clamp(0, i64::MAX as i128) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas13::{Atlas13, RoleId};
    use crate::compliance::OverlapPolicyMatrix;
    use crate::distill_engine::DistilledReasoningArc;
    use crate::geometry::fixed::Q32_31;
    use crate::holonomy::IDENTITY_FRAME_Q32;
    use crate::map_native::CoupledOperatorAtomPayload;
    use crate::refraction::Sha256Digest;
    use crate::silo::{CarvedSwarmMap, FrozenBaseWeights};
    use std::sync::Arc;

    fn fixture(
        name: &str,
        layers: usize,
    ) -> (CarvedSwarmMap, OverlapPolicyMatrix, MlirAtomCatalog) {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze(name, layers, vec![0.1; 32]);
        let map = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));
        let catalog = MlirAtomCatalog::new_canonical(248_320);
        (map, policy, catalog)
    }

    fn place(
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
        catalog: &MlirAtomCatalog,
        id: u32,
        payload: Vec<u8>,
        origin: Sha256Digest,
    ) -> AtomId {
        let refracted = refract_id(catalog, MlirOpId(id)).unwrap();
        let (atom, novel) = map.keep_and_carve_novel_only(refracted, payload, origin, policy);
        assert!(novel, "ID {id} coalesced into an existing atom");
        atom
    }

    fn link(
        map: &mut CarvedSwarmMap,
        from: AtomId,
        to: AtomId,
        op: u32,
        frame: [[Q32_31; 3]; 3],
        action: f64,
        certainty: f64,
        origin: Sha256Digest,
    ) {
        map.record_novel_reasoning_arc(DistilledReasoningArc {
            arc_id: 0,
            chain_id: 1,
            step_index: 0,
            from_atom: from,
            to_atom: to,
            from_role: RoleId::LanguageSemiotics,
            to_role: RoleId::ConvergentLogic,
            cosh_action_q32: Q32_31::from_f64(action),
            wigner_frame: frame,
            compiled_mlir_op: MlirOpId(op),
            teacher_certainty_q32: Q32_31::from_f64(certainty),
            origin_model_fingerprint: origin,
        });
    }

    fn coupled_lie(next_id: u32, h_dim: u32, layer: u16) -> Vec<u8> {
        let mut buf = Vec::with_capacity(36);
        buf.extend_from_slice(b"ATOMA_OP");
        buf.push(0x04);
        buf.push(0);
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&h_dim.to_le_bytes());
        buf.extend_from_slice(&next_id.to_le_bytes());
        buf
    }

    #[test]
    fn address_uses_the_catalog_well_and_ignores_vocab_for_other_ids() {
        let wide = MlirAtomCatalog::new_canonical(248_320);
        let narrow = MlirAtomCatalog::new_canonical(202_048);
        let op = MlirOpId(wide.base_vocab_size);
        let (lor, phase) = NativeReader::address(&wide, op).unwrap();
        let desc = wide.get_op(op).unwrap();
        assert_eq!(lor.x, desc.lorentz_well.x);
        assert_eq!(phase, desc.phase_well);

        let (a, _) = NativeReader::address(&wide, MlirOpId(10)).unwrap();
        let (b, _) = NativeReader::address(&narrow, MlirOpId(10)).unwrap();
        assert_eq!(a.x, b.x);
        let self_cosh = OpticalLens::lorentz_inner_product_wide(&a, &a);
        assert!(self_cosh.0 <= Q32_31::from_f64(1.01).0);
    }

    #[test]
    fn gyroscope_match_beats_a_shorter_hop() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let a = place(&mut map, &policy, &catalog, 10, b"a".to_vec(), origin);
        let b = place(&mut map, &policy, &catalog, 11, b"b".to_vec(), origin);
        let c = place(&mut map, &policy, &catalog, 99, b"c".to_vec(), origin);
        let (_pos, frame) = context(&catalog, &[MlirOpId(10)]).unwrap();
        let flipped = [
            [Q32_31::from_f64(-1.0), Q32_31::ZERO, Q32_31::ZERO],
            [Q32_31::ZERO, Q32_31::from_f64(-1.0), Q32_31::ZERO],
            [Q32_31::ZERO, Q32_31::ZERO, Q32_31::from_f64(-1.0)],
        ];
        link(&mut map, a, c, 99, flipped, 1.01, 0.99, origin);
        link(&mut map, a, b, 11, frame, 2.0, 0.10, origin);

        let read = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(11));
        assert_eq!(read.to_atom, b);
    }

    #[test]
    fn equal_frames_follow_the_shorter_geodesic() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let a = place(&mut map, &policy, &catalog, 10, b"a".to_vec(), origin);
        let b = place(&mut map, &policy, &catalog, 11, b"b".to_vec(), origin);
        let c = place(&mut map, &policy, &catalog, 12, b"c".to_vec(), origin);
        let (_pos, frame) = context(&catalog, &[MlirOpId(10)]).unwrap();
        link(&mut map, a, c, 12, frame, 1.80, 0.99, origin);
        link(&mut map, a, b, 11, frame, 1.20, 0.10, origin);

        let read = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(11));
    }

    #[test]
    fn continuation_walks_the_chain_and_stops_when_the_chart_ends() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let a = place(&mut map, &policy, &catalog, 10, b"a".to_vec(), origin);
        let b = place(&mut map, &policy, &catalog, 11, b"b".to_vec(), origin);
        let c = place(&mut map, &policy, &catalog, 12, b"c".to_vec(), origin);
        let f10 = context(&catalog, &[MlirOpId(10)]).unwrap().1;
        let f11 = context(&catalog, &[MlirOpId(10), MlirOpId(11)]).unwrap().1;
        link(&mut map, a, b, 11, f10, 1.1, 0.5, origin);
        link(&mut map, b, c, 12, f11, 1.1, 0.5, origin);

        let walk = NativeReader::continue_ids(&map, &catalog, &[MlirOpId(10)], 5, &[]).unwrap();
        let ids: Vec<u32> = walk.reads.iter().map(|r| r.next_id.0).collect();
        assert_eq!(ids, vec![11, 12]);
        assert!(matches!(walk.stop, NativeStop::NoGeodesic { at, .. } if at == MlirOpId(12)));
    }

    #[test]
    fn a_distant_atom_is_not_a_continuation() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let far = place(
            &mut map,
            &policy,
            &catalog,
            50_000,
            vec![0u8; 8_000],
            origin,
        );
        let sink = place(&mut map, &policy, &catalog, 50_001, vec![0u8; 64], origin);
        link(
            &mut map,
            far,
            sink,
            77,
            IDENTITY_FRAME_Q32,
            1.0,
            1.0,
            origin,
        );

        let err = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap_err();
        assert!(err.contains("no geodesic"), "{err}");
    }

    #[test]
    fn two_origins_and_a_backend_swap_do_not_change_the_next_id() {
        let (mut map, policy, catalog) = fixture("dense-wide", 52);
        let wide = map.base_weights.fingerprint;
        let narrow = Sha256Digest::compute(b"moe-narrow");
        let a = place(&mut map, &policy, &catalog, 10, vec![1u8; 6_656], wide);
        let b = place(&mut map, &policy, &catalog, 11, vec![2u8; 2_048], narrow);
        let frame = context(&catalog, &[MlirOpId(10)]).unwrap().1;
        link(&mut map, a, b, 11, frame, 1.2, 0.8, narrow);

        let swapped = FrozenBaseWeights::freeze("swapped-backend", 40, vec![0.2; 16]);
        let receipt = map.hot_swap_model_backend(Arc::clone(&swapped));
        assert_eq!(map.base_weights.num_layers, 40);
        assert!(receipt.preserved_global_atoms >= 2);

        let read = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(11));
        assert_eq!(read.origin_model_fingerprint, narrow);

        let other_catalog = MlirAtomCatalog::new_canonical(202_048);
        let again = NativeReader::step(&map, &other_catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(again.next_id, MlirOpId(11));
    }

    #[test]
    fn coupled_next_id_and_hidden_width_are_not_the_read() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let lie = coupled_lie(99_999, 6_656, 51);
        assert_eq!(
            CoupledOperatorAtomPayload::peek_next_mlir_id(&lie),
            Some(99_999)
        );
        let a = place(&mut map, &policy, &catalog, 10, lie, origin);
        let b = place(&mut map, &policy, &catalog, 11, vec![0u8; 2_048], origin);
        let frame = context(&catalog, &[MlirOpId(10)]).unwrap().1;
        link(&mut map, a, b, 11, frame, 1.3, 0.4, origin);

        let read = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(11));
    }

    #[test]
    fn empty_prefix_is_rejected() {
        let (map, _policy, catalog) = fixture("reader", 1);
        let err = NativeReader::step(&map, &catalog, &[]).unwrap_err();
        assert!(err.contains("at least one"));
    }

    fn bare_arc(from: AtomId, to: AtomId, op: u32, action: f64) -> DistilledReasoningArc {
        DistilledReasoningArc {
            arc_id: 0,
            chain_id: 1,
            step_index: 0,
            from_atom: from,
            to_atom: to,
            from_role: RoleId::LanguageSemiotics,
            to_role: RoleId::PredictiveSimulation,
            cosh_action_q32: Q32_31::from_f64(action),
            wigner_frame: IDENTITY_FRAME_Q32,
            compiled_mlir_op: MlirOpId(op),
            teacher_certainty_q32: Q32_31::from_f64(0.5),
            origin_model_fingerprint: Sha256Digest::compute(b"dedup"),
        }
    }

    #[test]
    fn trajectory_record_keeps_a_second_well_with_the_same_opcode() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = map.base_weights.fingerprint;
        let a = place(&mut map, &policy, &catalog, 10, b"a".to_vec(), origin);
        let b = place(&mut map, &policy, &catalog, 11, b"b".to_vec(), origin);
        let sink = place(&mut map, &policy, &catalog, 30, b"n".to_vec(), origin);
        let first = bare_arc(a, sink, 30, 1.2);
        let second = bare_arc(b, sink, 30, 1.2);
        let (id1, novel1) = map.record_novel_reasoning_arc(first);
        let (id2, novel2) = map.record_novel_reasoning_arc(second.clone());
        assert!(novel1);
        assert!(!novel2);
        assert_eq!(id1, id2);
        let (id3, novel3) = map.record_trajectory_arc(second);
        assert!(novel3);
        assert_ne!(id3, id1);
        assert_eq!(map.reasoning_arcs.len(), 2);
    }

    #[test]
    fn recorded_token_step_is_what_the_reader_returns() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = Sha256Digest::compute(b"glimmer-trajectory");
        let hidden = [1.25f32, -0.5, 0.0];
        let prefix = [MlirOpId(10), MlirOpId(20)];
        let wrote = NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix,
            MlirOpId(30),
            &hidden,
            origin,
            Q32_31::from_f64(0.8),
        )
        .unwrap();
        assert!(wrote.arc_was_novel);
        assert_eq!(wrote.read.next_id, MlirOpId(30));
        assert_eq!(wrote.read.origin_model_fingerprint, origin);
        let from = map.global_atom(wrote.read.from_atom).unwrap();
        let (well, _) = NativeReader::address(&catalog, MlirOpId(20)).unwrap();
        assert_eq!(from.lorentz.x, well.x);
        assert_eq!(from.payload.as_ref(), hidden_payload(&hidden).as_slice());

        let again = NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix,
            MlirOpId(30),
            &hidden,
            origin,
            Q32_31::from_f64(0.9),
        )
        .unwrap();
        assert!(!again.arc_was_novel);
        assert_eq!(map.reasoning_arcs.len(), 1);
        assert_eq!(
            map.reasoning_arcs[0].teacher_certainty_q32,
            Q32_31::from_f64(0.9)
        );
    }

    #[test]
    fn two_wells_that_share_a_next_id_both_play() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = Sha256Digest::compute(b"teacher");
        let cert = Q32_31::from_f64(0.7);
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(10)],
            MlirOpId(30),
            &[1.0],
            origin,
            cert,
        )
        .unwrap();
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(11)],
            MlirOpId(30),
            &[2.0],
            origin,
            cert,
        )
        .unwrap();
        assert_eq!(map.reasoning_arcs.len(), 2);
        let a = NativeReader::step(&map, &catalog, &[MlirOpId(10)]).unwrap();
        let b = NativeReader::step(&map, &catalog, &[MlirOpId(11)]).unwrap();
        assert_eq!(a.next_id, MlirOpId(30));
        assert_eq!(b.next_id, MlirOpId(30));
        assert_ne!(a.from_atom, b.from_atom);
    }

    #[test]
    fn two_histories_of_one_token_keep_both_next_ids() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = Sha256Digest::compute(b"teacher");
        let cert = Q32_31::ONE;
        let prefix_a = [MlirOpId(7), MlirOpId(10)];
        let prefix_b = [MlirOpId(8), MlirOpId(10)];
        let frame_a = context(&catalog, &prefix_a).unwrap().1;
        let frame_b = context(&catalog, &prefix_b).unwrap().1;
        assert_ne!(frame_a, frame_b);
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix_a,
            MlirOpId(11),
            &[0.1],
            origin,
            cert,
        )
        .unwrap();
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &prefix_b,
            MlirOpId(12),
            &[0.2],
            origin,
            cert,
        )
        .unwrap();
        assert_eq!(
            NativeReader::step(&map, &catalog, &prefix_a)
                .unwrap()
                .next_id,
            MlirOpId(11)
        );
        assert_eq!(
            NativeReader::step(&map, &catalog, &prefix_b)
                .unwrap()
                .next_id,
            MlirOpId(12)
        );
    }

    #[test]
    fn a_recorded_continuation_replays() {
        let (mut map, policy, catalog) = fixture("reader", 1);
        let origin = Sha256Digest::compute(b"teacher");
        let hidden = vec![vec![1.0f32], vec![3.0], vec![4.0]];
        NativeReader::record_continuation(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(10)],
            &[MlirOpId(11), MlirOpId(12), MlirOpId(13)],
            &hidden,
            origin,
            Q32_31::ONE,
        )
        .unwrap();
        let walk = NativeReader::continue_ids(&map, &catalog, &[MlirOpId(10)], 5, &[]).unwrap();
        let ids: Vec<u32> = walk.reads.iter().map(|r| r.next_id.0).collect();
        assert_eq!(ids, vec![11, 12, 13]);
        let standing_on_11 = map.global_atom(walk.reads[0].to_atom).unwrap();
        assert_eq!(
            standing_on_11.payload.as_ref(),
            hidden_payload(&[3.0]).as_slice()
        );
        assert!(matches!(
            walk.stop,
            NativeStop::NoGeodesic { at, .. } if at == MlirOpId(13)
        ));
    }

    #[test]
    fn a_recorded_step_survives_the_vault() {
        let (mut map, policy, catalog) = fixture("traj", 1);
        let origin = Sha256Digest::compute(b"teacher");
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(10), MlirOpId(20)],
            MlirOpId(30),
            &[0.25, -1.5],
            origin,
            Q32_31::from_f64(0.8),
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!("traj-{}-30.atoma_map", std::process::id()));
        map.save_to_binary_vault(&path).unwrap();
        let (mut restored, policy2, catalog2) = fixture("restored", 1);
        restored.load_from_binary_vault(&path, &policy2).unwrap();
        let _ = std::fs::remove_file(&path);
        let read = NativeReader::step(&restored, &catalog2, &[MlirOpId(10), MlirOpId(20)]).unwrap();
        assert_eq!(read.next_id, MlirOpId(30));
        assert_eq!(read.origin_model_fingerprint, origin);
    }

    #[test]
    fn query_path_and_active_inference_step_do_not_panic_on_extreme_q32_31() {
        let (mut map, policy, catalog) = fixture("extreme-q32", 1);
        let origin = Sha256Digest::compute(b"teacher");
        NativeReader::record_step(
            &mut map,
            &catalog,
            &policy,
            &[MlirOpId(10)],
            MlirOpId(20),
            &[1e6, -1e6],
            origin,
            Q32_31::MAX,
        )
        .unwrap();

        if let Some(arc) = map.reasoning_arcs.first_mut() {
            arc.cosh_action_q32 = Q32_31::MAX;
        }

        let query_phase = Phase {
            theta_f: Q32_31::MAX,
            theta_s: Q32_31::MIN,
        };
        let high_lor = LorentzVector::from_f64([50_000.0, 49_999.999_99, 0.0, 0.0]);
        let near = map.candidate_arc_atoms_near(&high_lor, query_phase, i64::MAX);
        assert!(near.is_some());

        let config = ActiveInferenceConfig::default();
        let _ = NativeReader::active_inference_step(
            &map,
            &catalog,
            &policy,
            &[MlirOpId(10)],
            &config,
            0,
        );
    }
}
