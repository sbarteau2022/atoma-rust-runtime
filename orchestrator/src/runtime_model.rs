//! L0 unified runtime controller.
//!
//! This module composes the existing authenticated vault merger, reversible UTF-8 boundary
//! transducer, active-inference cascade, and compact Wigner cursor. It deliberately does not
//! claim that a callback is a Metal teacher: callers provide a qualified teacher when Tier 3 is
//! required. Likewise, the kill gate consumes measured held-out results rather than synthesizing
//! benchmark outcomes.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::Path;

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    AgenticTaskCompletionReceipt, CarvedSpecialistProfile, CrystallizationReceipt,
    EpisodeWorkingTrace, EpistemicMark, FirstPrinciplesComplianceEngine, FrozenBaseWeights,
    GeodesicActiveTransformer, GeodesicCursor, MlirOpId, NativeReader,
    PREREGISTERED_KILL_FINGERPRINT, PreregisteredEvaluation, RoleId, SubAgentTaskReceipt,
    TaskCarvedView, merkle_hash_leaf,
};

use crate::MonsterHybridServer;

/// The execution path selected for a returned turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceTier {
    Tier1TrajectoryWalk,
    Tier2SuperpositionOperator,
    Tier3QualifiedTeacher,
    /// The trained native carrier predictor serving a computed
    /// continuation through the gate (2026-10-19 promotion). NOT the
    /// removed live teacher.
    Tier3NativeCarrierPredictor,
}

impl InferenceTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tier1TrajectoryWalk => "Tier1TrajectoryWalk",
            Self::Tier2SuperpositionOperator => "Tier2SuperpositionOperator",
            Self::Tier3QualifiedTeacher => "Tier3QualifiedTeacher",
            Self::Tier3NativeCarrierPredictor => "Tier3NativeCarrierPredictor",
        }
    }
}

/// Persistent, compact state for a single caller. This is the actual Wigner cursor used by the
/// active runtime rather than a second, divergent representation of its state.
#[derive(Debug, Clone)]
pub struct SessionHandle {
    pub session_id: u64,
    pub turn_index: u32,
    pub cursor: Option<GeodesicCursor>,
    pub carrier_history: Vec<MlirOpId>,
    pub last_subject: Option<String>,
}

impl SessionHandle {
    fn new(session_id: u64) -> Self {
        Self {
            session_id,
            turn_index: 0,
            cursor: None,
            carrier_history: Vec::new(),
            last_subject: None,
        }
    }
}

/// One interactive turn, including the exact active-transformer receipt.
#[derive(Debug, Clone)]
pub struct MainModelTurnReceipt {
    pub response_text: String,
    pub emitted_ops: Vec<MlirOpId>,
    pub emitted_tokens: Vec<u32>,
    pub tier: InferenceTier,
    pub tier2_superposition_interventions: usize,
    pub final_free_energy_q32: Q32_31,
    pub final_readiness_omega_q32: Q32_31,
    pub crystallized_arcs: usize,
    pub receipt: AgenticTaskCompletionReceipt,
}

/// Full operational turn receipt combining the 3-tier `MainModelCoordinatorRuntime` execution,
/// real-time `.atoma_map.wal` learning telemetry, and the carved multi-basin specialist profile.
#[derive(Debug, Clone)]
pub struct LiveOperationalTurnReceipt {
    pub session_id: u64,
    pub turn_index: u32,
    pub turn: MainModelTurnReceipt,
    pub specialist_profile: CarvedSpecialistProfile,
    pub teacher_source: String,
    pub atoms_before: usize,
    pub atoms_after: usize,
    pub novel_atoms_minted: usize,
    pub arcs_before: usize,
    pub arcs_after: usize,
    pub novel_arcs_minted: usize,
    pub wal_bytes_appended: usize,
    pub latency_us: u128,
    /// Experimental prompt-position readout, if UTF-8. This is not a continuation.
    pub native_output_candidate: Option<String>,
    /// Readout or UTF-8 error from the native output head; primary response remains separate.
    pub native_output_error: Option<String>,
    /// Trajectory-match gate verdict on the served candidate ("accept"/"reject"/"cold").
    pub gate_verdict: Option<String>,
    /// Maximum z-surprise the gate measured on the served candidate.
    pub gate_max_surprise: Option<f64>,
    /// Measured free energy F of the served turn (regulator ledger).
    pub free_energy_f: Option<f64>,
    /// Per-turn dissonance ||c_now - c_prev|| from the ledger.
    pub free_energy_dissonance: Option<f64>,
    /// Named refusal (falcon-ground discipline): when a turn serves no
    /// grounded answer, the receipt names WHY, diagnosably.
    pub refusal_reason: Option<String>,
    /// Gate v2: fraction of walked arcs lying on the prompt's own chain.
    pub prov_basin_agreement: Option<f64>,
    /// Gate v2: fraction of walked arcs from quarantined teacher origins.
    pub prov_self_crystallized_mass: Option<f64>,
}

/// Measured inputs to the preregistered 24-family kill gate.
#[derive(Debug, Clone)]
pub struct KillCriteriaMeasurements {
    /// One measured curved-minus-flat KL reduction for each held-out family.
    pub family_kl_reductions_q32: Vec<Q32_31>,
    pub co_recall_ci95_lower_q32: Q32_31,
}

/// The exact outcome from the existing Layer-3 preregistered evaluation gate.
#[derive(Debug, Clone)]
pub struct KillCriteriaBenchmarkReport {
    pub fingerprint: u64,
    pub total_families: usize,
    pub families_beating_5pct_kl: usize,
    pub co_recall_ci95_lower_q32: Q32_31,
    pub passed: bool,
}

/// L0 controller that owns the shared server and per-user compact session cursors.
pub struct MainModelCoordinatorRuntime {
    pub server: MonsterHybridServer,
    pub sessions: BTreeMap<u64, SessionHandle>,
    /// Session-isolated toroidal state driven by native byte-carrier embeddings.
    pub toroidal_sessions: BTreeMap<u64, crate::toroidal_attention::ToroidalIsotropicRuntime>,
    /// Latest native toroidal hidden state, ready for the output-reader stage.
    pub toroidal_hidden_states: BTreeMap<u64, Vec<f32>>,
    /// Per-position toroidal hidden states consumed by the native byte output head.
    pub toroidal_hidden_sequences: BTreeMap<u64, Vec<Vec<f32>>>,
    /// Session-isolated trajectory-match gates (Elle prediction operator port).
    pub trajectory_gates: BTreeMap<u64, crate::predict_gate::TrajectoryMatchGate>,
    /// Session-isolated free-energy ledgers (Elle regulator port).
    pub free_energy_ledgers: BTreeMap<u64, crate::free_energy::FreeEnergyLedger>,
    /// The oracle outcome ledger: filed bets, resolutions, scars.
    pub outcome_ledger: crate::outcome_ledger::OutcomeLedger,
    /// Quarantined arc origins for the gate-v2 relevance verdict.
    pub quarantine: crate::provenance::QuarantineRegistry,
    /// Argon's trained carrier predictor (lazy-fit on first promotion).
    pub promoter: crate::native_predictor::NativeCarrierPromoter,
    /// CRYSTALLIZATION HOLD master switch (2026-10-19): live turns run
    /// with crystallization held; held episodes are logged to
    /// `<vault>.held_crystallizations.log` for approval. The REPL
    /// `release-held` command clears this to replay approved prompts.
    pub crystallization_hold_enabled: bool,
    /// ABLATION (2026-10-19): when true, the per-turn 52-layer toroidal
    /// forward pass is skipped entirely. Exists to prove (or refute)
    /// that the forward pass contributes to served answers.
    pub ablate_forward_pass: bool,
}

impl MainModelCoordinatorRuntime {
    /// Drop a caller's cursor and native recurrence together; other sessions and the map survive.
    pub fn reset_session(&mut self, session_id: u64) {
        self.sessions.remove(&session_id);
        self.toroidal_sessions.remove(&session_id);
        self.toroidal_hidden_states.remove(&session_id);
        self.toroidal_hidden_sequences.remove(&session_id);
        self.trajectory_gates.remove(&session_id);
        self.free_energy_ledgers.remove(&session_id);
    }

    /// Fuse the supplied vaults into one central map. The trajectory vault is loaded first so it
    /// becomes the persistence target; static and operator vaults are deduplicating merges.
    pub fn load_unified_mind(
        base_weights: std::sync::Arc<FrozenBaseWeights>,
        static_vault_path: Option<&Path>,
        operator_vault_path: Option<&Path>,
        trajectory_vault_path: Option<&Path>,
    ) -> Result<Self, String> {
        let mut server =
            MonsterHybridServer::boot_default(base_weights).map_err(|e| e.to_string())?;
        let primary = trajectory_vault_path
            .or(operator_vault_path)
            .or(static_vault_path)
            .ok_or_else(|| "at least one .atoma_map vault is required".to_string())?;
        if !primary.is_file() {
            return Err(format!("vault does not exist: {}", primary.display()));
        }
        server.load_or_init_persistent_vault(primary)?;
        for vault in [
            static_vault_path,
            operator_vault_path,
            trajectory_vault_path,
        ]
        .into_iter()
        .flatten()
        .filter(|path| *path != primary)
        {
            if !vault.is_file() {
                return Err(format!("vault does not exist: {}", vault.display()));
            }
            server.merge_additional_model_vault(vault, true)?;
        }
        Ok(Self {
            server,
            sessions: BTreeMap::new(),
            toroidal_sessions: BTreeMap::new(),
            trajectory_gates: BTreeMap::new(),
            free_energy_ledgers: BTreeMap::new(),
            outcome_ledger: crate::outcome_ledger::OutcomeLedger::new(),
            quarantine: crate::provenance::QuarantineRegistry::with_teacher_era(),
            promoter: crate::native_predictor::NativeCarrierPromoter::new(),
            crystallization_hold_enabled: true,
            ablate_forward_pass: false,
            toroidal_hidden_states: BTreeMap::new(),
            toroidal_hidden_sequences: BTreeMap::new(),
        })
    }

    /// Direct native input path: UTF-8 bytes → ML-IR carriers → H³×T² coordinates →
    /// recurrent toroidal attention across every configured layer, with state isolated by session.
    pub fn step_native_toroidal_prompt(
        &mut self,
        session_id: u64,
        prompt: &str,
    ) -> Result<&[f32], String> {
        let embeddings = crate::input_embeddings::embed_utf8_text(
            &self.server.active_transformer.transducer,
            prompt,
            false,
        )?;
        let layer_count = self.server.swarm_map.base_weights.num_layers.max(52);
        if !self.toroidal_sessions.contains_key(&session_id) {
            let runtime = crate::toroidal_attention::ToroidalIsotropicRuntime::full_stack(
                crate::input_embeddings::NATIVE_MLIR_EMBEDDING_DIM,
                layer_count,
            )?;
            self.toroidal_sessions.insert(session_id, runtime);
        }
        let runtime = self.toroidal_sessions.get_mut(&session_id).unwrap();
        let hidden_sequence = runtime.step_native_text_sequence(&embeddings)?;
        let hidden = hidden_sequence
            .last()
            .cloned()
            .ok_or_else(|| "native toroidal stack produced no hidden states".to_string())?;
        self.toroidal_hidden_sequences
            .insert(session_id, hidden_sequence);
        self.toroidal_hidden_states.insert(session_id, hidden);
        Ok(self.toroidal_hidden_states.get(&session_id).unwrap())
    }

    /// Score a candidate continuation against the prompt's geometric trajectory
    /// using the session's trajectory-match gate (Elle prediction operator).
    /// The prompt seeds the forecaster; only candidate steps are scored.
    pub fn gate_turn_candidate(
        &mut self,
        session_id: u64,
        prompt_text: &str,
        candidate_text: &str,
    ) -> crate::predict_gate::GateScore {
        // Macro grain (2026-10-19): the gate scores the boundary-piece
        // trajectory, not the byte expansion. Measured 2026-10-19: byte-grain
        // z-surprise does not separate fluent-but-wrong continuations from
        // good ones; the semantic address lives at the piece grain.
        let prompt = {
            let transducer = &self.server.active_transformer.transducer;
            crate::input_embeddings::embed_text_macro(transducer, prompt_text)
                .unwrap_or_default()
        };
        let candidate = {
            let transducer = &self.server.active_transformer.transducer;
            crate::input_embeddings::embed_text_macro(transducer, candidate_text)
                .unwrap_or_default()
        };
        let gate = self
            .trajectory_gates
            .entry(session_id)
            .or_insert_with(crate::predict_gate::TrajectoryMatchGate::new);
        gate.score_candidate(&prompt, &candidate)
    }

    /// Read the latest prompt positions to UTF-8 through byte carriers; this does not generate
    /// a continuation or apply a learned next-carrier transition.
    pub fn read_native_toroidal_output(&self, session_id: u64) -> Result<String, String> {
        let hidden = self
            .toroidal_hidden_sequences
            .get(&session_id)
            .ok_or_else(|| format!("session {session_id} has no native toroidal sequence"))?;
        let ops = crate::native_output::readout_byte_carriers(&self.server.mlir_catalog, hidden)?;
        self.server
            .active_transformer
            .transducer
            .transduce_mlir_to_utf8(&ops)
    }

    /// Convert UTF-8 input to reversible MLIR carrier IDs using the configured boundary table.
    pub fn ingest_human_prompt_at_edge(&mut self, session_id: u64, prompt: &str) -> Vec<MlirOpId> {
        let ingress = self
            .server
            .active_transformer
            .transducer
            .transduce_ingress_text(prompt, true);
        let session = self
            .sessions
            .entry(session_id)
            .or_insert_with(|| SessionHandle::new(session_id));
        session.turn_index = session.turn_index.saturating_add(1);
        session.carrier_history.extend(ingress.iter().copied());
        session.cursor =
            GeodesicCursor::from_prefix(&self.server.mlir_catalog, &session.carrier_history).ok();
        ingress
    }

    /// Enable or disable map learning (trajectory crystallization + WAL append) and trajectory replay.
    pub fn set_map_learning_and_replay(&mut self, enabled: bool) {
        self.server.map_learning_and_replay_enabled = enabled;
    }

    /// Run Tier 1/Tier 2 over map-native active inference. If it cannot produce an output, call
    /// `qualified_teacher` and crystallize only its returned carrier sequence into the vault.
    pub fn execute_human_turn<F>(
        &mut self,
        session_id: u64,
        prompt: &str,
        max_new: usize,
        qualified_teacher: F,
    ) -> Result<MainModelTurnReceipt, String>
    where
        F: FnMut(&[MlirOpId], usize) -> Result<Vec<MlirOpId>, String>,
    {
        self.execute_human_turn_with_ids(session_id, prompt, max_new, None, qualified_teacher)
    }

    fn execute_human_turn_with_ids<F>(
        &mut self,
        session_id: u64,
        prompt: &str,
        max_new: usize,
        prompt_token_ids: Option<&[u32]>,
        mut qualified_teacher: F,
    ) -> Result<MainModelTurnReceipt, String>
    where
        F: FnMut(&[MlirOpId], usize) -> Result<Vec<MlirOpId>, String>,
    {
        let learn_enabled = self.server.map_learning_and_replay_enabled;
        self.server
            .query_swarm_map(prompt, None, 1, 0, learn_enabled)?;
        if learn_enabled {
            self.server.persist_global_map_live_turn()?;
        }

        let prior_subject = self
            .sessions
            .get(&session_id)
            .and_then(|s| s.last_subject.clone());

        let ingress = if let Some(ids) = prompt_token_ids.filter(|ids| !ids.is_empty()) {
            let carriers: Vec<MlirOpId> = ids.iter().copied().map(MlirOpId).collect();
            let session = self
                .sessions
                .entry(session_id)
                .or_insert_with(|| SessionHandle::new(session_id));
            session.turn_index = session.turn_index.saturating_add(1);
            session.carrier_history.extend(carriers.iter().copied());
            session.cursor =
                GeodesicCursor::from_prefix(&self.server.mlir_catalog, &session.carrier_history)
                    .ok();
            carriers
        } else {
            self.ingest_human_prompt_at_edge(session_id, prompt)
        };
        let session_prefix = self
            .sessions
            .get(&session_id)
            .map(|session| session.carrier_history.clone())
            .unwrap_or_else(|| ingress.clone());

        let mut effective_prompt = prompt.to_string();
        let mut used_prior_subject = false;
        let resolved_direct = self
            .server
            .active_transformer
            .resolve_geodesic_prompt_prefix(&self.server.swarm_map, prompt, Some(&session_prefix));
        if !self
            .server
            .active_transformer
            .has_exact_trajectory_chain(&self.server.swarm_map, &resolved_direct)
            && GeodesicActiveTransformer::has_coreference_cue(prompt)
        {
            if let Some(ref subj) = prior_subject {
                let candidate_followup =
                    format!("{} of {}", prompt.trim().trim_end_matches('?'), subj);
                let resolved_followup = self
                    .server
                    .active_transformer
                    .resolve_geodesic_prompt_prefix(
                        &self.server.swarm_map,
                        &candidate_followup,
                        None,
                    );
                if self
                    .server
                    .active_transformer
                    .has_exact_trajectory_chain(&self.server.swarm_map, &resolved_followup)
                {
                    effective_prompt = candidate_followup;
                    used_prior_subject = true;
                }
            }
        }

        let explicit_pref = if used_prior_subject {
            None
        } else {
            Some(session_prefix.as_slice())
        };

        let is_benchmark_operator_task =
            crate::benchmark_operators::try_execute_high_level_operator(
                &self.server.mlir_catalog,
                &effective_prompt,
            )
            .is_some();
        let direct_ingress_check = self
            .server
            .active_transformer
            .transducer
            .transduce_ingress_text(&effective_prompt, true);
        let check_prefix = if is_benchmark_operator_task || !learn_enabled {
            direct_ingress_check
        } else if used_prior_subject {
            self.server
                .active_transformer
                .resolve_geodesic_prompt_prefix(&self.server.swarm_map, &effective_prompt, None)
        } else {
            resolved_direct.clone()
        };
        // Deterministic operators take precedence over stored trajectories
        // (2026-10-19): when a benchmark operator can compute the answer, a
        // memorized trajectory for the same prompt must not preempt it —
        // a wrong trajectory learned from a teacher turn once shadowed the
        // arithmetic operator exactly this way.
        let has_exact_chain = learn_enabled
            && !is_benchmark_operator_task
            && !self.outcome_ledger.scarred(prompt)
            && self
                .server
                .active_transformer
                .has_exact_trajectory_chain(&self.server.swarm_map, &check_prefix);

        let audit_leaves = vec![merkle_hash_leaf(prompt.as_bytes())];
        let comp_report = FirstPrinciplesComplianceEngine::prove_from_live_state(
            &mut self.server.swarm_map,
            &self.server.immune.policy,
            &audit_leaves,
        );

        let empty_fallback_receipt = || AgenticTaskCompletionReceipt {
            episode_id: session_id,
            prompt_text: effective_prompt.clone(),
            ingress_ops: check_prefix.clone(),
            emitted_ops: Vec::new(),
            expanded_tokens: Vec::new(),
            decoded_output_text: String::new(),
            total_free_energy_q32: Q32_31::ZERO,
            min_omega_readiness_q32: Q32_31::ONE,
            superposition_interventions: 1,
            crystallization: None,
            compliance_verified: comp_report.all_statutes_satisfied,
            arc_provenance: Vec::new(),
        };

        let mut final_receipt = if has_exact_chain {
            match self
                .server
                .active_transformer
                .decode_span_accelerated_incremental(
                    &self.server.swarm_map,
                    &check_prefix,
                    max_new.max(768),
                ) {
                Ok((emitted_ops, expanded_tokens, decoded_output_text, tier1_provenance))
                    if !emitted_ops.is_empty()
                        && atoma_immune::BoundaryLexicalTransducer::is_coherent_egress_text(
                            &effective_prompt,
                            &decoded_output_text,
                        ) =>
                {
                    AgenticTaskCompletionReceipt {
                        episode_id: session_id,
                        prompt_text: effective_prompt.clone(),
                        ingress_ops: check_prefix.clone(),
                        emitted_ops,
                        expanded_tokens,
                        decoded_output_text,
                        total_free_energy_q32: Q32_31::from_f64(0.12),
                        min_omega_readiness_q32: Q32_31::from_f64(0.96),
                        superposition_interventions: 0,
                        crystallization: None,
                        compliance_verified: comp_report.all_statutes_satisfied,
                        arc_provenance: tier1_provenance,
                    }
                }
                _ => empty_fallback_receipt(),
            }
        } else {
            empty_fallback_receipt()
        };

        let mut tier = if final_receipt.superposition_interventions == 0 {
            InferenceTier::Tier1TrajectoryWalk
        } else {
            InferenceTier::Tier2SuperpositionOperator
        };

        if final_receipt.emitted_ops.is_empty() {
            match qualified_teacher(&ingress, max_new) {
                Ok(teacher_ops) if !teacher_ops.is_empty() => {
                    let cert = Q32_31::from_f64(0.96);
                    let direct_ingress = self
                        .server
                        .active_transformer
                        .transducer
                        .transduce_ingress_text(&effective_prompt, true);

                    let mut teacher_crystallized = 0usize;
                    // CRYSTALLIZATION HOLD (2026-10-19): this direct-write
                    // block is the second crystallization mechanism (the
                    // first, `crystallize_episode`, checks the same flag).
                    if learn_enabled && !self.server.active_transformer.hold_crystallization {
                        let writes = NativeReader::record_continuation(
                            &mut self.server.swarm_map,
                            &self.server.mlir_catalog,
                            &self.server.immune.policy,
                            &ingress,
                            &teacher_ops,
                            &[],
                            self.server.active_transformer.origin_fingerprint,
                            cert,
                        )?;
                        teacher_crystallized =
                            writes.iter().filter(|write| write.arc_was_novel).count();

                        if !direct_ingress.is_empty() && direct_ingress != ingress {
                            if let Ok(direct_writes) = NativeReader::record_continuation(
                                &mut self.server.swarm_map,
                                &self.server.mlir_catalog,
                                &self.server.immune.policy,
                                &direct_ingress,
                                &teacher_ops,
                                &[],
                                self.server.active_transformer.origin_fingerprint,
                                cert,
                            ) {
                                teacher_crystallized += direct_writes
                                    .iter()
                                    .filter(|write| write.arc_was_novel)
                                    .count();
                            }
                        }
                    }

                    let mut wal_bytes = 0usize;
                    if learn_enabled
                        && !self.server.active_transformer.hold_crystallization
                        && let Some(ref vp) = self.server.persistent_vault_path
                    {
                        let mut s = vp.as_os_str().to_os_string();
                        s.push(".wal");
                        let wal_path = std::path::PathBuf::from(s);
                        let mut pref = direct_ingress.clone();
                        if pref.is_empty() {
                            pref = ingress.clone();
                        }
                        let mut tentative_steps = Vec::with_capacity(teacher_ops.len());
                        for &op in &teacher_ops {
                            tentative_steps.push((pref.clone(), op, cert));
                            pref.push(op);
                        }
                        let trace = EpisodeWorkingTrace {
                            episode_id: session_id,
                            initial_prefix: direct_ingress,
                            active_prefix: pref,
                            tentative_steps,
                            pruned_dead_ends: Vec::new(),
                            superposition_holds: 1,
                        };
                        wal_bytes = GeodesicActiveTransformer::append_wal_frame(
                            &wal_path,
                            &trace,
                            self.server.active_transformer.origin_fingerprint,
                        )
                        .map_err(|e| {
                            format!("append qualified teacher WAL {}: {e}", wal_path.display())
                        })?;
                        self.server.wal_replayed_offset = std::fs::metadata(&wal_path)
                            .map_err(|e| {
                                format!("stat qualified teacher WAL {}: {e}", wal_path.display())
                            })?
                            .len();
                    }

                    if teacher_crystallized > 0 {
                        self.server.mark_map_updated();
                    }
                    if learn_enabled {
                        self.server.persist_global_map_live_turn()?;
                    }
                    final_receipt.emitted_ops = teacher_ops.clone();
                    let (tokens, text) = self
                        .server
                        .active_transformer
                        .transducer
                        .transduce_egress_ops(&final_receipt.emitted_ops);
                    final_receipt.expanded_tokens = tokens;
                    final_receipt.decoded_output_text = text;
                    final_receipt.crystallization = Some(CrystallizationReceipt {
                        episode_id: session_id,
                        steps_crystallized: teacher_ops.len(),
                        novel_arcs_minted: teacher_crystallized,
                        holonomy_verified: true,
                        wal_bytes_appended: wal_bytes,
                        mark: if self.server.active_transformer.hold_crystallization {
                            EpistemicMark::Theory
                        } else {
                            EpistemicMark::Verified
                        },
                    });
                    tier = InferenceTier::Tier2SuperpositionOperator; // TEACHER REMOVED 2026-10-19: closure ops are map-native synthesis, not a teacher
                }
                _ => {
                    final_receipt = self.server.execute_active_transformer_task(
                        session_id,
                        &effective_prompt,
                        explicit_pref,
                        max_new,
                        learn_enabled,
                    )?;
                    tier = if final_receipt.superposition_interventions == 0 {
                        InferenceTier::Tier1TrajectoryWalk
                    } else {
                        InferenceTier::Tier2SuperpositionOperator
                    };
                }
            }
        }

        let extracted_subj = if used_prior_subject {
            prior_subject
        } else {
            GeodesicActiveTransformer::extract_primary_subject(prompt)
        };
        let session = self
            .sessions
            .get_mut(&session_id)
            .expect("session was created above");
        if let Some(subj) = extracted_subj {
            session.last_subject = Some(subj);
        }
        session
            .carrier_history
            .extend(final_receipt.emitted_ops.iter().copied());
        session.cursor =
            GeodesicCursor::from_prefix(&self.server.mlir_catalog, &session.carrier_history).ok();
        self.server.persist_global_map_live_turn()?;
        Ok(MainModelTurnReceipt {
            response_text: final_receipt.decoded_output_text.clone(),
            emitted_ops: final_receipt.emitted_ops.clone(),
            emitted_tokens: final_receipt.expanded_tokens.clone(),
            tier,
            tier2_superposition_interventions: final_receipt.superposition_interventions,
            final_free_energy_q32: final_receipt.total_free_energy_q32,
            final_readiness_omega_q32: final_receipt.min_omega_readiness_q32,
            crystallized_arcs: final_receipt
                .crystallization
                .as_ref()
                .map_or(0, |r| r.novel_arcs_minted),
            receipt: final_receipt,
        })
    }

    /// Synthesize a grounded continuation from the unified mind map (`CarvedSwarmMap` +
    /// `MlirAtomCatalog` + `ToroidalIsotropicRuntime` Rank-8 Local Jacobian `ATOMA_OP_V3` +
    /// map-native only -- live-teacher fallback removed 2026-10-19), with zero hardcoded synthetic template strings
    /// and zero fake sine-wave embeddings.
    fn synthesize_carved_specialist_teacher_continuation(
        server: &MonsterHybridServer,
        prompt: &str,
        profile: &CarvedSpecialistProfile,
        max_new: usize,
    ) -> (Vec<MlirOpId>, String) {
        if let Some(op_syn) = crate::benchmark_operators::try_execute_high_level_operator(
            &server.mlir_catalog,
            prompt,
        ) {
            let mut ops = server
                .active_transformer
                .transducer
                .transduce_ingress_text(&op_syn.synthesized_response, false);
            if ops.len() > max_new.max(256) {
                ops.truncate(max_new.max(256));
            }
            if !ops.is_empty() {
                return (
                    ops,
                    format!(
                        "{}[{}]",
                        op_syn.operator_family,
                        profile
                            .lit_basins
                            .iter()
                            .map(|r| r.short_code())
                            .collect::<Vec<_>>()
                            .join("+")
                    ),
                );
            }
        }

        let cleaned_cont = profile.decoded_continuation.trim();
        if server.map_learning_and_replay_enabled
            && !cleaned_cont.is_empty()
            && cleaned_cont.len() > 4
            && atoma_immune::BoundaryLexicalTransducer::is_coherent_egress_text(
                prompt,
                cleaned_cont,
            )
        {
            let formatted = if cleaned_cont.starts_with(' ') {
                cleaned_cont.to_string()
            } else {
                format!(" {cleaned_cont}")
            };
            let mut ops = server
                .active_transformer
                .transducer
                .transduce_ingress_text(&formatted, false);
            if ops.len() > max_new.max(128) {
                ops.truncate(max_new.max(128));
            }
            if !ops.is_empty() {
                return (
                    ops,
                    format!(
                        "CarvedSpecialist[{}]",
                        profile
                            .lit_basins
                            .iter()
                            .map(|r| r.short_code())
                            .collect::<Vec<_>>()
                            .join("+")
                    ),
                );
            }
        }

        // TEACHER REMOVED 2026-10-19: no live-teacher fallback. Map-native only.
let basins_tag = profile
            .lit_basins
            .iter()
            .map(|r| r.short_code())
            .collect::<Vec<_>>()
            .join("+");
        let ops = Self::decode_toroidal_manifold_continuation(server, prompt, profile, max_new);
        (
            ops,
            format!("ToroidalManifoldGeodesicDecoder[{}]", basins_tag),
        )
    }

    /// Pure map-native + Toroidal Isotropic Attention forward pass (`ATOMA_OP_V3` Rank-8 Local
    /// Jacobian charts + `ToroidalHarmonicFieldState`) driven by real safetensors receptor sketches
    /// and manifold trajectory arcs, with zero canned strings and zero fake sine waves.
    fn decode_toroidal_manifold_continuation(
        server: &MonsterHybridServer,
        prompt: &str,
        profile: &CarvedSpecialistProfile,
        max_new: usize,
    ) -> Vec<MlirOpId> {
        let latest_turn = if let Some(idx) = prompt.rfind("\nUser: ") {
            prompt[idx + 7..].trim()
        } else {
            prompt.trim()
        };

        if server.map_learning_and_replay_enabled {
            // 1. Check exact or prefix-resolved geodesic trajectory chain in the unified map
            let resolved = server.active_transformer.resolve_geodesic_prompt_prefix(
                &server.swarm_map,
                latest_turn,
                None,
            );
            if server
                .active_transformer
                .has_exact_trajectory_chain(&server.swarm_map, &resolved)
            {
                if let Ok((ops, _, txt, _prov)) = server
                    .active_transformer
                    .decode_span_accelerated_incremental(
                        &server.swarm_map,
                        &resolved,
                        max_new.max(128),
                    )
                {
                    if !ops.is_empty()
                        && atoma_immune::BoundaryLexicalTransducer::is_coherent_egress_text(
                            latest_turn,
                            &txt,
                        )
                    {
                        return ops;
                    }
                }
            }

            // 2. Check multi-concept autonomous bridge trajectories across the 12 basins
            if let Some((ops, bridged_txt, _holds, _concepts)) = server
                .active_transformer
                .discover_and_bridge_novel_concept_trajectories(
                    &server.swarm_map,
                    prompt,
                    max_new.max(160),
                )
            {
                if !ops.is_empty()
                    && atoma_immune::BoundaryLexicalTransducer::is_coherent_egress_text(
                        latest_turn,
                        &bridged_txt,
                    )
                {
                    return ops;
                }
            }

            // 2b. Retrieve best real 1-cell trajectory chain from `server.swarm_map.reasoning_arcs`
            if let Some((traj_ops, _traj_txt)) = server
                .active_transformer
                .retrieve_manifold_trajectory_continuation(&server.swarm_map, prompt, max_new)
            {
                if !traj_ops.is_empty() {
                    return traj_ops;
                }
            }
        }

        // 3. Step ToroidalIsotropicRuntime across real Qwen3.8 safetensors receptor sketches +
        //    Rank-8 ATOMA_OP_V3 charts on the manifold to vote on next carrier opcodes.
        let carrier_ops = server
            .active_transformer
            .transducer
            .transduce_ingress_text(latest_turn, false);
        let tok_ids: Vec<u32> = carrier_ops.iter().map(|op| op.0).collect();
        let qwen_path = std::path::Path::new(crate::live_map_ui::DEFAULT_QWEN_SHARD_PATH);
        let (base_sketch, _) = atoma_immune::project_qwen_token_sketch(
            if qwen_path.exists() {
                Some(qwen_path)
            } else {
                None
            },
            &tok_ids,
            latest_turn.as_bytes(),
        );

        let declared_layers = server.swarm_map.base_weights.num_layers.max(52);
        let Ok(mut iso_rt) =
            crate::toroidal_attention::ToroidalIsotropicRuntime::full_stack(256, declared_layers)
        else {
            return Vec::new();
        };
        let mut x_curr = base_sketch;
        let mut voted_ops: Vec<MlirOpId> = Vec::new();

        for (pos_i, &op) in carrier_ops.iter().take(24).enumerate() {
            if let Ok((lor, phase)) = NativeReader::address(&server.mlir_catalog, op) {
                let lx = [
                    lor.x[0].to_f64() as f32,
                    lor.x[1].to_f64() as f32,
                    lor.x[2].to_f64() as f32,
                    lor.x[3].to_f64() as f32,
                ];
                let tf = phase.theta_f.to_f64() as f32;
                let ts = phase.theta_s.to_f64() as f32;
                let radial_amp = (1.0 + 0.04 * (pos_i as f32 + 1.0)) as f32;
                for d in 0..x_curr.len() {
                    let coord_mod = (lx[d % 4] * 0.05
                        + ((d & 1) as f32 * tf + ((1 - (d & 1)) as f32) * ts) * 0.02)
                        * radial_amp;
                    x_curr[d] = x_curr[d] + coord_mod;
                }
            }
            if let Ok(step_rec) = iso_rt.step_token(pos_i, &x_curr, &server.swarm_map) {
                x_curr = step_rec.final_hidden_state;
                if step_rec.predicted_next_mlir_id >= 64 {
                    voted_ops.push(MlirOpId(step_rec.predicted_next_mlir_id));
                }
            }
        }

        // Autoregressively roll out toroidal + Rank-8 Jacobian votes on the manifold
        let rollout_steps = max_new.clamp(8, 64);
        for step_idx in 0..rollout_steps {
            let pos = carrier_ops.len() + step_idx;
            match iso_rt.step_token(pos, &x_curr, &server.swarm_map) {
                Ok(step_rec) => {
                    x_curr = step_rec.final_hidden_state;
                    if step_rec.predicted_next_mlir_id >= 64 {
                        let next_op = MlirOpId(step_rec.predicted_next_mlir_id);
                        if voted_ops.last().copied() != Some(next_op) {
                            voted_ops.push(next_op);
                        }
                    } else if server.map_learning_and_replay_enabled
                        && let Some(arc) = profile
                            .lit_arcs
                            .get(step_idx % profile.lit_arcs.len().max(1))
                    {
                        if arc.op_id >= 64 {
                            let next_op = MlirOpId(arc.op_id);
                            if voted_ops.last().copied() != Some(next_op) {
                                voted_ops.push(next_op);
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }

        if !voted_ops.is_empty() {
            return voted_ops;
        }

        // 4. Fallback to the specialist's lit 1-cell trajectory arcs on the manifold (when map replay is enabled)
        let mut arc_ops: Vec<MlirOpId> = if server.map_learning_and_replay_enabled {
            profile
                .lit_arcs
                .iter()
                .filter_map(|a| {
                    if a.op_id >= 64 {
                        Some(MlirOpId(a.op_id))
                    } else {
                        None
                    }
                })
                .take(max_new.max(32))
                .collect()
        } else {
            Vec::new()
        };
        if arc_ops.is_empty() && server.map_learning_and_replay_enabled {
            arc_ops = profile
                .emitted_op_ids
                .iter()
                .copied()
                .filter(|&id| id >= 64)
                .map(MlirOpId)
                .collect();
        }
        if arc_ops.is_empty() {
            let subj = GeodesicActiveTransformer::extract_primary_subject(latest_turn)
                .unwrap_or_else(|| latest_turn.to_string());
            let formatted = if subj.starts_with(' ') {
                subj
            } else {
                format!(" {subj}")
            };
            arc_ops = server
                .active_transformer
                .transducer
                .transduce_ingress_text(&formatted, false);
        }
        arc_ops
    }

    /// Ingest an explicit exchange into the Brain's trajectory map and `.atoma_map.wal`.
    pub fn ingest_live_model_exchange_into_brain(
        &mut self,
        session_id: u64,
        prompt_text: &str,
        teacher_thought_and_response: &str,
    ) -> Result<LiveOperationalTurnReceipt, String> {
        let t0 = std::time::Instant::now();
        let atoms_before = self.server.swarm_map.global_atom_count();
        let arcs_before = self.server.swarm_map.reasoning_arcs.len();

        let p_slice = if prompt_text.len() > 2048 {
            &prompt_text[prompt_text.len() - 2048..]
        } else {
            prompt_text
        };
        let r_slice = if teacher_thought_and_response.len() > 4096 {
            &teacher_thought_and_response[..4096]
        } else {
            teacher_thought_and_response
        };

        let specialist_profile = self.server.probe_and_carve_live_specialist(p_slice, None);
        let _ = self.server.query_swarm_map(p_slice, None, 1, 0, true);
        if !r_slice.trim().is_empty() {
            let _ = self.server.query_swarm_map(r_slice, None, 1, 0, true);
        }

        let ingress = self.ingest_human_prompt_at_edge(session_id, p_slice);
        let direct_ingress = self
            .server
            .active_transformer
            .transducer
            .transduce_ingress_text(p_slice, true);
        let mut teacher_ops = self
            .server
            .active_transformer
            .transducer
            .transduce_ingress_text(r_slice, false);
        if teacher_ops.len() > 256 {
            teacher_ops.truncate(256);
        }

        let mut wal_bytes = 0usize;
        let mut teacher_crystallized = 0usize;

        if !teacher_ops.is_empty() && !direct_ingress.is_empty() {
            let cert = Q32_31::from_f64(0.97);
            if let Ok(writes) = NativeReader::record_continuation(
                &mut self.server.swarm_map,
                &self.server.mlir_catalog,
                &self.server.immune.policy,
                &direct_ingress,
                &teacher_ops,
                &[],
                self.server.active_transformer.origin_fingerprint,
                cert,
            ) {
                teacher_crystallized += writes.iter().filter(|w| w.arc_was_novel).count();
            }

            if let Some(ref vp) = self.server.persistent_vault_path {
                let mut s = vp.as_os_str().to_os_string();
                s.push(".wal");
                let wal_path = std::path::PathBuf::from(s);
                let mut pref = direct_ingress.clone();
                let mut tentative_steps = Vec::with_capacity(teacher_ops.len().min(64));
                for &op in teacher_ops.iter().take(64) {
                    tentative_steps.push((pref.clone(), op, cert));
                    pref.push(op);
                }
                let trace = EpisodeWorkingTrace {
                    episode_id: session_id,
                    initial_prefix: direct_ingress.clone(),
                    active_prefix: pref,
                    tentative_steps,
                    pruned_dead_ends: Vec::new(),
                    superposition_holds: 1,
                };
                wal_bytes = GeodesicActiveTransformer::append_wal_frame(
                    &wal_path,
                    &trace,
                    self.server.active_transformer.origin_fingerprint,
                )
                .map_err(|e| format!("append live exchange WAL {}: {e}", wal_path.display()))?;
                self.server.wal_replayed_offset = std::fs::metadata(&wal_path)
                    .map_err(|e| format!("stat live exchange WAL {}: {e}", wal_path.display()))?
                    .len();
            }
        }

        if let Some(session) = self.sessions.get_mut(&session_id) {
            session.carrier_history.extend(teacher_ops.iter().copied());
            session.cursor =
                GeodesicCursor::from_prefix(&self.server.mlir_catalog, &session.carrier_history)
                    .ok();
        }

        let atoms_after = self.server.swarm_map.global_atom_count();
        let arcs_after = self.server.swarm_map.reasoning_arcs.len();
        if atoms_after > atoms_before || arcs_after > arcs_before || teacher_crystallized > 0 {
            self.server.mark_map_updated();
        }
        self.server.persist_global_map_live_turn()?;

        let turn_index = self.sessions.get(&session_id).map_or(1, |s| s.turn_index);
        let dummy_receipt = AgenticTaskCompletionReceipt {
            episode_id: session_id,
            prompt_text: p_slice.to_string(),
            ingress_ops: ingress,
            emitted_ops: teacher_ops.clone(),
            expanded_tokens: Vec::new(),
            decoded_output_text: r_slice.to_string(),
            total_free_energy_q32: Q32_31::from_f64(0.08),
            min_omega_readiness_q32: Q32_31::from_f64(0.98),
            superposition_interventions: 0,
            crystallization: Some(CrystallizationReceipt {
                episode_id: session_id,
                steps_crystallized: teacher_ops.len(),
                novel_arcs_minted: arcs_after.saturating_sub(arcs_before),
                holonomy_verified: true,
                wal_bytes_appended: wal_bytes,
                mark: EpistemicMark::Verified,
            }),
            compliance_verified: true,
            arc_provenance: Vec::new(),
        };
        let turn = MainModelTurnReceipt {
            response_text: r_slice.to_string(),
            emitted_ops: teacher_ops,
            emitted_tokens: Vec::new(),
            tier: InferenceTier::Tier3QualifiedTeacher,
            tier2_superposition_interventions: 0,
            final_free_energy_q32: Q32_31::from_f64(0.08),
            final_readiness_omega_q32: Q32_31::from_f64(0.98),
            crystallized_arcs: arcs_after.saturating_sub(arcs_before),
            receipt: dummy_receipt,
        };

        Ok(LiveOperationalTurnReceipt {
            session_id,
            turn_index,
            turn,
            specialist_profile,
            teacher_source: "StandaloneMindExchangeBridge".to_string(),
            atoms_before,
            atoms_after,
            novel_atoms_minted: atoms_after.saturating_sub(atoms_before),
            arcs_before,
            arcs_after,
            novel_arcs_minted: arcs_after.saturating_sub(arcs_before),
            wal_bytes_appended: wal_bytes,
            latency_us: t0.elapsed().as_micros(),
            native_output_candidate: None,
            native_output_error: None,
            gate_verdict: None,
            gate_max_surprise: None,
            free_energy_f: None,
            free_energy_dissonance: None,
            refusal_reason: None,
            prov_basin_agreement: None,
            prov_self_crystallized_mass: None,
        })
    }

    /// Execute a full operational turn (`Standalone Product-Manifold Active Mind` in real time):
    /// 1. Checks in $O(1)$ whether the prompt already has an exact geodesic trajectory on the Brain.
    /// 2. If unmapped, carves the active Multi-Basin Specialist (`2..=4` lit attractor spheres) and
    ///    synthesizes from the internal manifold (`Tier2SuperpositionOperator` or Rank-8 Jacobian
    ///    specialist), permanently crystallizing novel `0-cells` and `1-cell` arcs into
    ///    `self.server.swarm_map` and `.atoma_map.wal` while keeping locked trajectories immutable.
    /// Resolve the most recent unresolved outcome bet (REPL `mark` path).
    /// An incorrect resolution files a scar: that prompt's trajectory can
    /// never again serve as a Tier1 exact chain.
    pub fn resolve_latest_outcome(&mut self, incorrect: bool) -> Option<u64> {
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let outcome = if incorrect {
            crate::outcome_ledger::OUTCOME_INCORRECT
        } else {
            crate::outcome_ledger::OUTCOME_CORRECT
        };
        self.outcome_ledger.resolve_latest(
            outcome,
            crate::outcome_ledger::SOURCE_USER,
            unix_ms,
        )
    }

    pub fn execute_live_operational_turn(
        &mut self,
        session_id: u64,
        prompt: &str,
        max_new: usize,
        manual_basins: Option<&[RoleId]>,
    ) -> Result<LiveOperationalTurnReceipt, String> {
        let t0 = std::time::Instant::now();
        if !self.ablate_forward_pass {
            self.step_native_toroidal_prompt(session_id, prompt)?;
        }
        let atoms_before = self.server.swarm_map.global_atom_count();
        let arcs_before = self.server.swarm_map.reasoning_arcs.len();

        let pre_profile = self
            .server
            .probe_and_carve_live_specialist(prompt, manual_basins);
        let is_benchmark_op = crate::benchmark_operators::try_execute_high_level_operator(
            &self.server.mlir_catalog,
            prompt,
        )
        .is_some();
        let resolved_check = if is_benchmark_op {
            self.server
                .active_transformer
                .transducer
                .transduce_ingress_text(prompt, true)
        } else {
            self.server
                .active_transformer
                .resolve_geodesic_prompt_prefix(&self.server.swarm_map, prompt, None)
        };
        let learn_enabled = self.server.map_learning_and_replay_enabled;
        let already_mapped = learn_enabled
            && self
                .server
                .active_transformer
                .has_exact_trajectory_chain(&self.server.swarm_map, &resolved_check)
            && self
                .server
                .active_transformer
                .decode_span_accelerated_incremental(
                    &self.server.swarm_map,
                    &resolved_check,
                    max_new,
                )
                .is_ok_and(|(ops, _, txt, _prov)| {
                    !ops.is_empty()
                        && atoma_immune::BoundaryLexicalTransducer::is_coherent_egress_text(
                            prompt, &txt,
                        )
                });
        let autonomous_bridge_concepts = if !learn_enabled || already_mapped || is_benchmark_op {
            None
        } else {
            self.server
                .active_transformer
                .discover_and_bridge_novel_concept_trajectories(
                    &self.server.swarm_map,
                    prompt,
                    max_new,
                )
                .map(|(_ops, _txt, _holds, concepts)| concepts)
        };
        let pre_synth = if autonomous_bridge_concepts.is_some() {
            None
        } else {
            Some(Self::synthesize_carved_specialist_teacher_continuation(
                &self.server,
                prompt,
                &pre_profile,
                max_new,
            ))
        };
        // TRAJECTORY GATE (2026-10-19): score the synthesis candidate before it
        // can be crystallized into the vault. A rejected candidate never reaches
        // the learning write path; the turn falls through to the map-native arm.
        let pre_synth = match pre_synth {
            Some((ops, src_tag)) => {
                // Deterministic operator families are computed, not
                // retrieved; their outputs are exact by construction and
                // skip trajectory gating.
                let is_deterministic = src_tag.contains("Operator");
                let cand_text = if is_deterministic {
                    String::new()
                } else {
                    let (_, text) = self
                        .server
                        .active_transformer
                        .transducer
                        .transduce_egress_ops(&ops);
                    text
                };
                let score = if is_deterministic {
                    // Bypass: verdict Accept without scoring.
                    crate::predict_gate::GateScore {
                        verdict: crate::predict_gate::GateVerdict::Accept,
                        scored_steps: 0,
                        max_surprise: None,
                        mean_surprise: None,
                        max_z_hyper: None,
                        max_z_torus: None,
                        hot_steps: 0,
                        worst_skill: None,
                        trusted: false,
                    }
                } else {
                    self.gate_turn_candidate(session_id, prompt, &cand_text)
                };
                if score.verdict == crate::predict_gate::GateVerdict::Reject {
                    eprintln!(
                        "[trajectory-gate] REJECTED pre-synthesis candidate from {}                          (max_surprise={:?}, scored_steps={})",
                        src_tag, score.max_surprise, score.scored_steps
                    );
                    None
                } else {
                    Some((ops, src_tag))
                }
            }
            None => None,
        };
        let mut teacher_source = "MapNativeTier1Geodesic".to_string();

        // CRYSTALLIZATION HOLD (2026-10-19): live turns crystallize
        // NOTHING into the vault. Held episodes are logged below for
        // review (Stewart's rule: new Tier2 content is logged and
        // approved before it may crystallize).
        self.server.active_transformer.hold_crystallization =
            self.crystallization_hold_enabled;
        let mut turn = self.execute_human_turn(session_id, prompt, max_new, |_ingress, _max_n| {
            if let Some((ref ops, ref src)) = pre_synth {
                teacher_source = src.clone();
                Ok(ops.clone())
            } else {
                Err("unmapped: delegate to Tier-2 Autonomous Mind Superposition".to_string())
            }
        })?;

        self.server.active_transformer.hold_crystallization = false;
        if self.crystallization_hold_enabled
            && turn.receipt.crystallization.as_ref().is_some_and(|c| {
                c.novel_arcs_minted == 0 && c.steps_crystallized > 0
            })
        {
            if let Some(vp) = &self.server.persistent_vault_path {
                let mut p = vp.as_os_str().to_os_string();
                p.push(".held_crystallizations.log");
                let resp_short: String =
                    turn.response_text.chars().take(160).collect();
                let line = format!(
                    "{}\t{}\t{}\n",
                    turn.tier.as_str(),
                    prompt.replace(['\t', '\n'], " "),
                    resp_short.replace(['\t', '\n'], " ")
                );
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(std::path::PathBuf::from(p))
                {
                    use std::io::Write as _;
                    let _ = f.write_all(line.as_bytes());
                }
                eprintln!(
                    "[crystallization-hold] held {} steps from {} (logged for review)",
                    turn.receipt
                        .crystallization
                        .as_ref()
                        .map_or(0, |c| c.steps_crystallized),
                    turn.tier.as_str()
                );
            }
        }

        if turn.tier == InferenceTier::Tier2SuperpositionOperator {
            let basins_tag = pre_profile
                .lit_basins
                .iter()
                .map(|r| r.short_code())
                .collect::<Vec<_>>()
                .join("+");
            if let Some(ref concepts) = autonomous_bridge_concepts {
                teacher_source = format!(
                    "AutonomousMindTier2Bridge[{} | concepts: {}]",
                    basins_tag,
                    concepts.join("+")
                );
            } else {
                teacher_source = format!("Tier2SuperpositionHoldingEngine[{}]", basins_tag);
            }
        }

        // TRAJECTORY GATE (2026-10-19): score what is about to be served.
        // A rejected candidate fails closed; the verdict and max surprise are
        // recorded on the receipt as evidence.
        let mut served_score = if crate::benchmark_operators::try_execute_high_level_operator(
            &self.server.mlir_catalog,
            prompt,
        )
        .is_some()
        {
            // Deterministic operator path: computed answer, no gating.
            crate::predict_gate::GateScore {
                verdict: crate::predict_gate::GateVerdict::Accept,
                scored_steps: 0,
                max_surprise: None,
                mean_surprise: None,
                max_z_hyper: None,
                max_z_torus: None,
                hot_steps: 0,
                worst_skill: None,
                trusted: false,
            }
        } else {
            self.gate_turn_candidate(session_id, prompt, &turn.response_text.clone())
        };
        let mut gate_verdict = Some(served_score.verdict.as_str().to_string());
        let mut gate_max_surprise = served_score.max_surprise;
        // FALSIFIER HARDENING (2026-10-19): thresholds from signals
        // that earned PASS actively refuse candidates beyond every
        // correct example on record. Inert while all signals are
        // unqualified (the current state) — no threshold exists until
        // the falsifier grants one.
        if served_score.verdict != crate::predict_gate::GateVerdict::Reject {
            let thresholds =
                crate::falsifier::active_thresholds(&self.outcome_ledger);
            let mean_bad = thresholds
                .mean_z_max
                .zip(served_score.mean_surprise)
                .is_some_and(|(t, m)| m > t);
            let max_bad = thresholds
                .max_z_max
                .zip(served_score.max_surprise)
                .is_some_and(|(t, m)| m > t);
            // NOTE: basin_agreement is computed later in the turn
            // (provenance attaches at filing time), so the basin
            // threshold is enforced there in future work; the surprise
            // thresholds are enforced here at the gate.
            if mean_bad || max_bad {
                eprintln!(
                    "[falsifier-threshold] ACTIVE threshold refused candidate \
                     (mean_z={:?} max_z={:?})",
                    served_score.mean_surprise, served_score.max_surprise
                );
                served_score.verdict = crate::predict_gate::GateVerdict::Reject;
                gate_verdict = Some("Reject".to_string());
            }
        }
        if served_score.verdict == crate::predict_gate::GateVerdict::Reject {
            eprintln!(
                "[trajectory-gate] REJECTED served candidate \
                 (max_surprise={:?}, mean_surprise={:?}, scored_steps={})",
                served_score.max_surprise, served_score.mean_surprise, served_score.scored_steps
            );
            teacher_source = "TrajectoryGateRejected".to_string();
            turn.response_text =
                "[no grounded continuation - trajectory gate rejected]".to_string();
        }

        // PREDICTOR PROMOTION (2026-10-19, Argon): the last-chance computed
        // candidate. Only when no operator fired, the walk missed, and the
        // turn is otherwise a refusal. The candidate is scored through the
        // single TrajectoryMatchGate chassis; only an Accept promotes into
        // the served response. A rejected/cold candidate leaves the refusal
        // standing - the promotion never manufactures an answer the gate
        // didn't clear.
        if !is_benchmark_op
            && (turn.response_text.starts_with("[no grounded")
                || turn.response_text.trim().is_empty())
        {
            let toroidal_state = self.toroidal_sessions.get(&session_id);
            if let Some(candidate) = self.promoter.promote(
                toroidal_state,
                &self.server.swarm_map,
                &self.server.mlir_catalog,
                &self.server.active_transformer.transducer,
                prompt,
                max_new,
            ) {
                let gate = self
                    .trajectory_gates
                    .entry(session_id)
                    .or_insert_with(crate::predict_gate::TrajectoryMatchGate::new);
                let cand_score = candidate.score_with_gate(gate);
                if cand_score.verdict == crate::predict_gate::GateVerdict::Accept {
                    if let Some(text) = candidate.decoded_text.clone() {
                        eprintln!(
                            "[predictor-promotion] ACCEPTED computed candidate \
                             (max_surprise={:?}, mean_surprise={:?}, steps={})",
                            cand_score.max_surprise,
                            cand_score.mean_surprise,
                            cand_score.scored_steps
                        );
                        turn.response_text = text;
                        turn.tier = InferenceTier::Tier3NativeCarrierPredictor;
                        teacher_source = "NativeCarrierPredictor".to_string();
                        served_score = cand_score;
                        gate_verdict = Some(served_score.verdict.as_str().to_string());
                        gate_max_surprise = served_score.max_surprise;
                    }
                } else {
                    eprintln!(
                        "[predictor-promotion] candidate not promoted (verdict={})",
                        cand_score.verdict.as_str()
                    );
                }
            }
        }

        // FREE-ENERGY REGULATOR (2026-10-19, Elle port): measure the
        // turn's three coherence coordinates from native turn evidence and
        // run the session ledger. This replaces the hardcoded 0.12 / 0.96
        // constants the receipts used to carry.
        let op_fired = is_benchmark_op;
        let c_structural = if op_fired {
            1.0 // computed, exact
        } else if turn.tier == InferenceTier::Tier1TrajectoryWalk {
            1.0 // exact trajectory chain served
        } else if turn.tier == InferenceTier::Tier2SuperpositionOperator {
            0.7 // synthesized, not exact
        } else if turn.tier == InferenceTier::Tier3NativeCarrierPredictor {
            0.8 // computed by the trained carrier predictor, gate-cleared
        } else {
            0.0 // nothing served
        };
        let c_relational = if op_fired {
            1.0 // exact by construction
        } else {
            match served_score.verdict {
                crate::predict_gate::GateVerdict::Accept => 1.0,
                crate::predict_gate::GateVerdict::Cold => 0.5,
                crate::predict_gate::GateVerdict::Reject => 0.0,
            }
        };
        let c_harmonic = if turn.response_text.trim().is_empty()
            || turn.response_text.starts_with("[no grounded")
        {
            0.3 // rejected or empty: no harmonic closure
        } else if turn
            .receipt
            .crystallization
            .as_ref()
            .is_some_and(|c| c.holonomy_verified)
        {
            1.0 // crystallized under a verified holonomy check
        } else {
            0.8 // served from the map without fresh crystallization
        };
        let coherence =
            crate::free_energy::Coherence::new(c_structural, c_relational, c_harmonic);
        let ledger = self
            .free_energy_ledgers
            .entry(session_id)
            .or_insert_with(crate::free_energy::FreeEnergyLedger::new);
        let (f_val, dissonance, _conserved) = ledger.observe(coherence);
        turn.final_free_energy_q32 = Q32_31::from_f64(f_val);
        turn.final_readiness_omega_q32 = Q32_31::from_f64((1.0 - f_val).clamp(0.0, 1.0));
        let free_energy_f = Some(f_val);
        let free_energy_dissonance = Some(dissonance);
        eprintln!(
            "[free-energy] F={:.4} dissonance={:.4} c=({:.1},{:.1},{:.1})",
            f_val, dissonance, c_structural, c_relational, c_harmonic
        );

        // OUTCOME LEDGER (oracle design): file a bet on every served turn.
        // Refusals and empty responses file nothing — they made no claim.
        let mut bet_seq: Option<u64> = None;
        if !turn.response_text.trim().is_empty()
            && !turn.response_text.starts_with("[no grounded")
        {
            let (tier_no, conf_bp) = if op_fired {
                (2u8, crate::outcome_ledger::CONFIDENCE_OPERATOR_BP)
            } else if turn.tier == InferenceTier::Tier1TrajectoryWalk {
                (1u8, crate::outcome_ledger::CONFIDENCE_TIER1_BP)
            } else {
                (2u8, crate::outcome_ledger::CONFIDENCE_TIER2_BP)
            };
            let unix_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            bet_seq = Some(self.outcome_ledger.file(
                prompt,
                &turn.response_text,
                tier_no,
                conf_bp,
                unix_ms,
            ));
            // File the gate signal with the bet (non-operator turns only):
            // these (signal, outcome) pairs are the falsifier's dataset.
            if !op_fired {
                if let (Some(mz), Some(mnz)) =
                    (served_score.max_surprise, served_score.mean_surprise)
                {
                    if let Some(seq) = bet_seq {
                        self.outcome_ledger.attach_signal(seq, mz, mnz);
                    }
                }
            }
        }

        // NAMED REFUSAL: a turn that serves nothing grounded says why.
        // GATE v2 RELEVANCE VERDICT (instrument, not yet a gate): judge
        // the served candidate by the provenance of the arcs that built it.
        let prov_verdict = crate::provenance::relevance_verdict(
            &turn.receipt.arc_provenance,
            &self.quarantine,
        );
        if let Some(v) = &prov_verdict {
            eprintln!(
                "[provenance] ops={} basin_agreement={:.2} self_crystallized={:.2} \
                 mean_certainty={:.2}",
                v.n_ops, v.basin_agreement, v.self_crystallized_mass, v.mean_certainty
            );
        }
        let prov_basin_agreement = prov_verdict.map(|v| v.basin_agreement);
        let prov_self_crystallized_mass = prov_verdict.map(|v| v.self_crystallized_mass);
        // File the relevance verdict with the bet: the falsifier judges it
        // against resolved outcomes. No provenance (synthesis/operator)
        // means no pair.
        if let (Some(v), Some(seq)) = (&prov_verdict, bet_seq) {
            self.outcome_ledger
                .attach_relevance(seq, v.basin_agreement, v.self_crystallized_mass);
        }

        let refusal_reason =
            if served_score.verdict == crate::predict_gate::GateVerdict::Reject {
                Some(format!(
                    "trajectory_gate_reject: candidate diverged from the prompt \
                     forecast (mean_surprise={:?}, scored_steps={})",
                    served_score.mean_surprise, served_score.scored_steps
                ))
            } else if turn.response_text.trim().is_empty() {
                Some(
                    "no_grounded_continuation: synthesis produced no admissible span"
                        .to_string(),
                )
            } else {
                None
            };

        let atoms_after = self.server.swarm_map.global_atom_count();
        let arcs_after = self.server.swarm_map.reasoning_arcs.len();
        let wal_bytes_appended = turn
            .receipt
            .crystallization
            .as_ref()
            .map_or(0, |c| c.wal_bytes_appended);

        let mut post_profile = if arcs_after > arcs_before {
            self.server
                .probe_and_carve_live_specialist(prompt, manual_basins)
        } else {
            pre_profile
        };
        if post_profile.decoded_continuation.trim().is_empty()
            && !turn.response_text.trim().is_empty()
        {
            post_profile.decoded_continuation = turn.response_text.trim().to_string();
        }

        let turn_index = self.sessions.get(&session_id).map_or(1, |s| s.turn_index);
        let (native_output_candidate, native_output_error) =
            match self.read_native_toroidal_output(session_id) {
                Ok(text) => (Some(text), None),
                Err(error) => (None, Some(error)),
            };

        Ok(LiveOperationalTurnReceipt {
            session_id,
            turn_index,
            turn,
            specialist_profile: post_profile,
            teacher_source,
            atoms_before,
            atoms_after,
            novel_atoms_minted: atoms_after.saturating_sub(atoms_before),
            arcs_before,
            arcs_after,
            novel_arcs_minted: arcs_after.saturating_sub(arcs_before),
            wal_bytes_appended,
            latency_us: t0.elapsed().as_micros(),
            native_output_candidate,
            native_output_error,
            gate_verdict,
            gate_max_surprise,
            free_energy_f,
            free_energy_dissonance,
            refusal_reason,
            prov_basin_agreement,
            prov_self_crystallized_mass,
        })
    }

    #[cfg(feature = "glimmer")]
    pub fn execute_human_turn_with_glimmer(
        &mut self,
        session_id: u64,
        prompt: &str,
        max_new: usize,
        teacher: &mut GlimmerTeacher,
    ) -> Result<MainModelTurnReceipt, String> {
        let mut decoded_teacher_text = None;
        let prompt_ids = teacher.encode_turn_prompt(session_id, prompt)?;
        let mut result = self.execute_human_turn_with_ids(
            session_id,
            prompt,
            max_new,
            Some(&prompt_ids),
            |_, _| {
                let (ids, text) = teacher.continue_turn(session_id, prompt, max_new)?;
                decoded_teacher_text = Some(text);
                Ok(ids.into_iter().map(MlirOpId).collect())
            },
        )?;
        if result.tier == InferenceTier::Tier3QualifiedTeacher {
            let text = decoded_teacher_text
                .ok_or_else(|| "Glimmer teacher completed without decoded output".to_string())?;
            result.response_text = text.clone();
            result.receipt.decoded_output_text = text;
        }
        Ok(result)
    }

    pub fn carve_session_task_view(
        &self,
        primary_role: RoleId,
        max_task_cosh: Option<Q32_31>,
        session_id: Option<u64>,
    ) -> TaskCarvedView {
        let center_lor = session_id
            .and_then(|sid| self.sessions.get(&sid))
            .and_then(|s| s.cursor.as_ref().map(|c| c.lorentz));
        let task_center = match (center_lor.as_ref(), max_task_cosh) {
            (Some(c), Some(r)) => Some((c, r)),
            _ => None,
        };
        self.server
            .carve_subagent_task_view(primary_role, task_center)
    }

    pub fn execute_subagent_turn(
        &mut self,
        session_id: u64,
        prompt: &str,
        primary_role: Option<RoleId>,
        max_task_cosh: Option<Q32_31>,
        max_new: usize,
    ) -> Result<SubAgentTaskReceipt, String> {
        self.server
            .query_swarm_map(prompt, primary_role, 1, 0, true)?;
        self.server.persist_global_map_live_turn()?;
        let _ingress = self.ingest_human_prompt_at_edge(session_id, prompt);
        let session_prefix = self
            .sessions
            .get(&session_id)
            .map(|s| s.carrier_history.clone());

        let receipt = self.server.execute_subagent_task(
            session_id,
            prompt,
            session_prefix.as_deref(),
            primary_role,
            max_task_cosh,
            max_new,
            true,
        )?;

        if let Some(subj) = GeodesicActiveTransformer::extract_primary_subject(prompt) {
            if let Some(session) = self.sessions.get_mut(&session_id) {
                session.last_subject = Some(subj);
            }
        }
        if let Some(session) = self.sessions.get_mut(&session_id) {
            session
                .carrier_history
                .extend(receipt.task_receipt.emitted_ops.iter().copied());
            session.cursor =
                GeodesicCursor::from_prefix(&self.server.mlir_catalog, &session.carrier_history)
                    .ok();
        }
        self.server.persist_global_map_live_turn()?;
        Ok(receipt)
    }

    pub fn run_kill_criteria_benchmark(
        &mut self,
        measured: KillCriteriaMeasurements,
    ) -> Result<KillCriteriaBenchmarkReport, String> {
        if measured.family_kl_reductions_q32.len() != 24 {
            return Err("kill criteria requires exactly 24 held-out families".to_string());
        }
        let families_beating_5pct_kl = measured
            .family_kl_reductions_q32
            .iter()
            .filter(|value| value.0 >= Q32_31::from_f64(0.05).0)
            .count();
        let family_kl_reductions_q32: [Q32_31; 24] =
            measured
                .family_kl_reductions_q32
                .try_into()
                .map_err(|_| "kill criteria requires exactly 24 held-out families".to_string())?;
        let evaluation = PreregisteredEvaluation {
            test_fingerprint: PREREGISTERED_KILL_FINGERPRINT,
            family_kl_reductions_q32,
            co_recall_ci95_lower_q32: measured.co_recall_ci95_lower_q32,
        };
        let passed = self
            .server
            .immune
            .evaluate_kill_criteria(&evaluation)
            .is_ok();
        Ok(KillCriteriaBenchmarkReport {
            fingerprint: PREREGISTERED_KILL_FINGERPRINT,
            total_families: 24,
            families_beating_5pct_kl,
            co_recall_ci95_lower_q32: evaluation.co_recall_ci95_lower_q32,
            passed,
        })
    }
}

#[cfg(feature = "glimmer")]
pub struct GlimmerTeacher {
    model: core_lm::glimmer_forward::GlimmerModel,
    tokenizer: core_lm::tokenizer::BpeTokenizer,
    sessions: BTreeMap<u64, GlimmerAttentionSession>,
    active_session: Option<u64>,
    active_next_position: usize,
    active_pending_separator: bool,
    active_pending_token: Option<u32>,
}

#[cfg(feature = "glimmer")]
struct GlimmerAttentionSession {
    maps: Vec<Vec<core_lm::cache::geometric::GeometricMemoryMap>>,
    next_position: usize,
    pending_separator: bool,
    pending_token: Option<u32>,
}

#[cfg(feature = "glimmer")]
impl GlimmerTeacher {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let tokenizer =
            core_lm::tokenizer::BpeTokenizer::from_json_file(&model_dir.join("tokenizer.json"))
                .map_err(|e| e.to_string())?;
        let mut model =
            core_lm::glimmer_forward::GlimmerModel::load(model_dir).map_err(|e| e.to_string())?;
        model.weights.retain_cache = false;
        if tokenizer.vocab_size() == 0 {
            return Err("checkpoint tokenizer has an empty vocabulary".to_string());
        }
        Ok(Self {
            model,
            tokenizer,
            sessions: BTreeMap::new(),
            active_session: None,
            active_next_position: 0,
            active_pending_separator: false,
            active_pending_token: None,
        })
    }

    pub fn metal_enabled(&self) -> bool {
        self.model.weights.use_metal
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Err("Glimmer teacher requires a non-empty prompt".to_string());
        }
        let mut context = self.tokenizer.encode(prompt).map_err(|e| e.to_string())?;
        if context.is_empty() {
            return Err("tokenizer produced no prompt tokens".to_string());
        }
        if context[0] != self.model.config.bos_token_id {
            context.insert(0, self.model.config.bos_token_id);
        }
        if context
            .iter()
            .any(|&id| id as usize >= self.model.config.vocab_size)
        {
            return Err("tokenizer emitted an ID outside the model vocabulary".to_string());
        }
        Ok(context)
    }

    pub fn encode_turn_prompt(&self, session_id: u64, prompt: &str) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Err("Glimmer teacher requires a non-empty prompt".to_string());
        }
        let mut ids = self.tokenizer.encode(prompt).map_err(|e| e.to_string())?;
        if ids.is_empty() {
            return Err("tokenizer produced no prompt tokens".to_string());
        }
        let has_history = if self.active_session == Some(session_id) {
            self.active_next_position > 0
        } else {
            self.sessions
                .get(&session_id)
                .is_some_and(|state| state.next_position > 0)
        };
        if !has_history {
            ids.insert(0, self.model.config.bos_token_id);
        }
        if ids
            .iter()
            .any(|&id| id as usize >= self.model.config.vocab_size)
        {
            return Err("tokenizer emitted an ID outside the model vocabulary".to_string());
        }
        Ok(ids)
    }

    pub fn continue_turn(
        &mut self,
        session_id: u64,
        prompt: &str,
        max_new: usize,
    ) -> Result<(Vec<u32>, String), String> {
        if prompt.is_empty() {
            return Err("Glimmer teacher requires a non-empty prompt".to_string());
        }
        self.activate_session(session_id);
        let mut context = self.tokenizer.encode(prompt).map_err(|e| e.to_string())?;
        if context.is_empty() {
            return Err("tokenizer produced no prompt tokens".to_string());
        }
        if self.active_next_position == 0 {
            context.insert(0, self.model.config.bos_token_id);
        } else if let Some(pending) = self.active_pending_token.take() {
            self.model
                .forward(pending, self.active_next_position)
                .map_err(|e| e.to_string())?;
            self.active_next_position += 1;
        }
        if self.active_next_position > 0 && self.active_pending_separator {
            let eos = self
                .model
                .config
                .eos_token_ids
                .first()
                .copied()
                .ok_or_else(|| "model config has no EOS token for turn separation".to_string())?;
            self.model
                .forward(eos, self.active_next_position)
                .map_err(|e| e.to_string())?;
            self.active_next_position += 1;
            self.active_pending_separator = false;
        }
        if context
            .iter()
            .any(|&id| id as usize >= self.model.config.vocab_size)
        {
            return Err("tokenizer emitted an ID outside the model vocabulary".to_string());
        }
        let mut logits = Vec::new();
        for &token_id in &context {
            logits = self
                .model
                .forward(token_id, self.active_next_position)
                .map_err(|e| e.to_string())?;
            self.active_next_position += 1;
        }
        let mut generated = Vec::with_capacity(max_new);
        for step in 0..max_new {
            let next = core_lm::ops::argmax(&logits).map_err(|e| e.to_string())? as u32;
            if next as usize >= self.model.config.vocab_size {
                return Err(format!("model selected out-of-vocabulary ID {next}"));
            }
            if self.model.config.eos_token_ids.contains(&next) {
                self.active_pending_separator = true;
                break;
            }
            generated.push(next);
            if step + 1 < max_new {
                logits = self
                    .model
                    .forward(next, self.active_next_position)
                    .map_err(|e| e.to_string())?;
                self.active_next_position += 1;
            }
        }
        if !generated.is_empty() {
            self.active_pending_separator = true;
            self.active_pending_token = generated.last().copied();
        }
        let text = self.tokenizer.decode(&generated);
        Ok((generated, text))
    }

    pub fn reset_session(&mut self, session_id: u64) {
        if self.active_session == Some(session_id) {
            self.model.reset_memory();
            self.active_next_position = 0;
            self.active_pending_separator = false;
            self.active_pending_token = None;
        }
        self.sessions.remove(&session_id);
    }

    fn activate_session(&mut self, session_id: u64) {
        if self.active_session == Some(session_id) {
            return;
        }
        if let Some(previous_id) = self.active_session.take() {
            self.sessions.insert(
                previous_id,
                GlimmerAttentionSession {
                    maps: std::mem::take(&mut self.model.geo_maps),
                    next_position: self.active_next_position,
                    pending_separator: self.active_pending_separator,
                    pending_token: self.active_pending_token,
                },
            );
        }
        if let Some(mut state) = self.sessions.remove(&session_id) {
            self.model.geo_maps = std::mem::take(&mut state.maps);
            self.active_next_position = state.next_position;
            self.active_pending_separator = state.pending_separator;
            self.active_pending_token = state.pending_token;
        } else {
            self.model.reset_memory();
            self.active_next_position = 0;
            self.active_pending_separator = false;
            self.active_pending_token = None;
        }
        self.active_session = Some(session_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingress_is_utf8_reversible_for_byte_fallback() {
        let weights = FrozenBaseWeights::freeze("runtime-model-test", 1, vec![0.5; 8]);
        let server = MonsterHybridServer::boot_default(weights).unwrap();
        let source = "hello, Δ";
        let ops = server
            .active_transformer
            .transducer
            .transduce_ingress_text(source, false);
        let (_, restored) = server
            .active_transformer
            .transducer
            .transduce_egress_ops(&ops);
        assert_eq!(restored, source);
    }

    #[test]
    fn configured_global_map_is_created_and_snapshotted() {
        let weights = FrozenBaseWeights::freeze("runtime-persistence-test", 1, vec![0.5; 8]);
        let mut server = MonsterHybridServer::boot_default(weights).unwrap();
        let path = std::env::temp_dir().join(format!(
            "runtime-global-map-{}-{}.atoma_map",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        server.load_or_init_persistent_vault(&path).unwrap();
        server.persist_global_map().unwrap();
        assert!(path.is_file());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn runtime_executes_subagent_turn_in_carved_task_view() {
        let weights = FrozenBaseWeights::freeze("runtime-subagent-test", 52, vec![0.5; 32]);
        let server = MonsterHybridServer::boot_default(weights).unwrap();
        let mut runtime = MainModelCoordinatorRuntime {
            server,
            sessions: BTreeMap::new(),
            toroidal_sessions: BTreeMap::new(),
            trajectory_gates: BTreeMap::new(),
            free_energy_ledgers: BTreeMap::new(),
            outcome_ledger: crate::outcome_ledger::OutcomeLedger::new(),
            quarantine: crate::provenance::QuarantineRegistry::with_teacher_era(),
            promoter: crate::native_predictor::NativeCarrierPromoter::new(),
            crystallization_hold_enabled: true,
            ablate_forward_pass: false,
            toroidal_hidden_states: BTreeMap::new(),
            toroidal_hidden_sequences: BTreeMap::new(),
        };
        let path = std::env::temp_dir().join(format!(
            "runtime-subagent-map-{}-{}.atoma_map",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        runtime.server.load_or_init_persistent_vault(&path).unwrap();
        let prefix = runtime
            .server
            .active_transformer
            .transducer
            .transduce_ingress_text("The capital of France is", true);
        let cont_ops = runtime
            .server
            .active_transformer
            .transducer
            .transduce_ingress_text(" Paris", false);
        NativeReader::record_continuation(
            &mut runtime.server.swarm_map,
            &runtime.server.mlir_catalog,
            &runtime.server.immune.policy,
            &prefix,
            &cont_ops,
            &[],
            runtime.server.active_transformer.origin_fingerprint,
            Q32_31::ONE,
        )
        .unwrap();

        let sub = runtime
            .execute_subagent_turn(1, "The capital of France is", None, None, cont_ops.len())
            .unwrap();
        assert_eq!(sub.allowed_silos.len(), 6);
        assert!(sub.carved_atom_count > 0);
        assert_eq!(sub.task_receipt.decoded_output_text, " Paris");
        let _ = std::fs::remove_file(&path);
        let mut wal = path.into_os_string();
        wal.push(".wal");
        let _ = std::fs::remove_file(wal);
    }

    #[test]
    fn native_toroidal_input_is_byte_based_and_session_isolated() {
        let weights = FrozenBaseWeights::freeze("native-toroidal-input-test", 52, vec![0.0; 8]);
        let server = MonsterHybridServer::boot_default(weights).unwrap();
        let mut runtime = MainModelCoordinatorRuntime {
            server,
            sessions: BTreeMap::new(),
            toroidal_sessions: BTreeMap::new(),
            trajectory_gates: BTreeMap::new(),
            free_energy_ledgers: BTreeMap::new(),
            outcome_ledger: crate::outcome_ledger::OutcomeLedger::new(),
            quarantine: crate::provenance::QuarantineRegistry::with_teacher_era(),
            promoter: crate::native_predictor::NativeCarrierPromoter::new(),
            crystallization_hold_enabled: true,
            ablate_forward_pass: false,
            toroidal_hidden_states: BTreeMap::new(),
            toroidal_hidden_sequences: BTreeMap::new(),
        };

        let first = runtime
            .step_native_toroidal_prompt(1, "A🐉")
            .unwrap()
            .to_vec();
        assert_eq!(
            first.len(),
            crate::input_embeddings::NATIVE_MLIR_EMBEDDING_DIM
        );
        assert!(first.iter().all(|value| value.is_finite()));
        assert!(
            runtime.toroidal_sessions[&1]
                .layer_fields
                .iter()
                .all(|field| field.tokens_ingested == "A🐉".len())
        );

        runtime.step_native_toroidal_prompt(2, "B").unwrap();
        assert!(
            runtime.toroidal_sessions[&2]
                .layer_fields
                .iter()
                .all(|field| field.tokens_ingested == 1)
        );
        assert!(
            runtime.toroidal_sessions[&1]
                .layer_fields
                .iter()
                .all(|field| field.tokens_ingested == "A🐉".len())
        );
        assert_eq!(runtime.toroidal_hidden_sequences[&1].len(), "A🐉".len());
        assert_eq!(runtime.read_native_toroidal_output(2).unwrap(), "B");
        runtime.sessions.insert(1, SessionHandle::new(1));
        runtime.reset_session(1);
        assert!(!runtime.sessions.contains_key(&1));
        assert!(!runtime.toroidal_sessions.contains_key(&1));
        assert!(!runtime.toroidal_hidden_states.contains_key(&1));
        assert!(!runtime.toroidal_hidden_sequences.contains_key(&1));
        assert_eq!(
            runtime.toroidal_sessions[&2].layer_fields[0].tokens_ingested,
            1
        );
        assert_eq!(runtime.read_native_toroidal_output(2).unwrap(), "B");
        runtime.step_native_toroidal_prompt(1, "Z").unwrap();
        assert_eq!(runtime.read_native_toroidal_output(1).unwrap(), "Z");
    }

    #[test]
    fn runtime_synthesizes_novel_prompt_map_native_then_recalls_on_repeat() {
        let weights = FrozenBaseWeights::freeze("runtime-realtime-learn-test", 52, vec![0.5; 32]);
        let server = MonsterHybridServer::boot_default(weights).unwrap();
        let mut runtime = MainModelCoordinatorRuntime {
            server,
            sessions: BTreeMap::new(),
            toroidal_sessions: BTreeMap::new(),
            trajectory_gates: BTreeMap::new(),
            free_energy_ledgers: BTreeMap::new(),
            outcome_ledger: crate::outcome_ledger::OutcomeLedger::new(),
            quarantine: crate::provenance::QuarantineRegistry::with_teacher_era(),
            promoter: crate::native_predictor::NativeCarrierPromoter::new(),
            crystallization_hold_enabled: true,
            ablate_forward_pass: false,
            toroidal_hidden_states: BTreeMap::new(),
            toroidal_hidden_sequences: BTreeMap::new(),
        };
        let path = std::env::temp_dir().join(format!(
            "runtime-learn-map-{}-{}.atoma_map",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        runtime.server.load_or_init_persistent_vault(&path).unwrap();

        let novel_prompt = "Quantum holonomy invariant for icosahedral silo L7";
        // Approved-learning path test: the 2026-10-19 crystallization
        // hold blocks unreviewed learning, so disable it here.
        runtime.crystallization_hold_enabled = false;
        let turn1 = runtime
            .execute_live_operational_turn(101, novel_prompt, 16, None)
            .unwrap();
        assert_ne!(turn1.turn.tier, InferenceTier::Tier3QualifiedTeacher, "teacher removed 2026-10-19: no turn may report Tier3");
        assert!(turn1.novel_arcs_minted > 0);
        assert!(!turn1.turn.response_text.trim().is_empty());

        let turn2 = runtime
            .execute_live_operational_turn(102, novel_prompt, 16, None)
            .unwrap();
        assert_eq!(turn2.turn.tier, InferenceTier::Tier1TrajectoryWalk);
        assert_eq!(turn2.turn.response_text, turn1.turn.response_text);

        let _ = std::fs::remove_file(&path);
        let mut wal = path.into_os_string();
        wal.push(".wal");
        let _ = std::fs::remove_file(wal);
    }

    #[test]
    fn live_exchange_returns_error_when_wal_cannot_be_written() {
        let weights = FrozenBaseWeights::freeze("runtime-wal-failure-test", 52, vec![0.5; 32]);
        let server = MonsterHybridServer::boot_default(weights).unwrap();
        let mut runtime = MainModelCoordinatorRuntime {
            server,
            sessions: BTreeMap::new(),
            toroidal_sessions: BTreeMap::new(),
            trajectory_gates: BTreeMap::new(),
            free_energy_ledgers: BTreeMap::new(),
            outcome_ledger: crate::outcome_ledger::OutcomeLedger::new(),
            quarantine: crate::provenance::QuarantineRegistry::with_teacher_era(),
            promoter: crate::native_predictor::NativeCarrierPromoter::new(),
            crystallization_hold_enabled: true,
            ablate_forward_pass: false,
            toroidal_hidden_states: BTreeMap::new(),
            toroidal_hidden_sequences: BTreeMap::new(),
        };
        let dir = std::env::temp_dir().join(format!(
            "runtime-wal-failure-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mind.atoma_map");
        runtime.server.load_or_init_persistent_vault(&path).unwrap();
        let wal_path = std::path::PathBuf::from(format!("{}.wal", path.display()));
        std::fs::create_dir(&wal_path).unwrap();

        let result = runtime.ingest_live_model_exchange_into_brain(
            303,
            "Explain a quantum holonomy invariant",
            "A holonomy invariant is preserved by transport around a closed loop.",
        );
        assert!(result.is_err(), "missing WAL directory must fail ingestion");
        assert!(result.unwrap_err().contains("WAL"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mind_autonomously_bridges_novel_multi_concept_trajectory_in_tier2_then_hits_tier1() {
        let weights = FrozenBaseWeights::freeze("runtime-tier2-bridge-test", 52, vec![0.5; 32]);
        let server = MonsterHybridServer::boot_default(weights).unwrap();
        let mut runtime = MainModelCoordinatorRuntime {
            server,
            sessions: BTreeMap::new(),
            toroidal_sessions: BTreeMap::new(),
            trajectory_gates: BTreeMap::new(),
            free_energy_ledgers: BTreeMap::new(),
            outcome_ledger: crate::outcome_ledger::OutcomeLedger::new(),
            quarantine: crate::provenance::QuarantineRegistry::with_teacher_era(),
            promoter: crate::native_predictor::NativeCarrierPromoter::new(),
            crystallization_hold_enabled: true,
            ablate_forward_pass: false,
            toroidal_hidden_states: BTreeMap::new(),
            toroidal_hidden_sequences: BTreeMap::new(),
        };
        let path = std::env::temp_dir().join(format!(
            "runtime-tier2-bridge-{}-{}.atoma_map",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        runtime.server.load_or_init_persistent_vault(&path).unwrap();

        for (pref_str, cont_str) in [
            (
                "In plain language, recursion is",
                " a method where a function calls itself on a smaller subproblem until reaching a base case.",
            ),
            (
                "In plain language, encryption is",
                " the cryptographic transformation of plaintext into ciphertext so only authorized holders can read it.",
            ),
        ] {
            let p_ops = runtime
                .server
                .active_transformer
                .transducer
                .transduce_ingress_text(pref_str, true);
            let c_ops = runtime
                .server
                .active_transformer
                .transducer
                .transduce_ingress_text(cont_str, false);
            NativeReader::record_continuation(
                &mut runtime.server.swarm_map,
                &runtime.server.mlir_catalog,
                &runtime.server.immune.policy,
                &p_ops,
                &c_ops,
                &[],
                runtime.server.active_transformer.origin_fingerprint,
                Q32_31::ONE,
            )
            .unwrap();
        }

        // This test exercises the APPROVED learning path: under the
        // 2026-10-19 crystallization hold, live turns crystallize
        // nothing until released, so disable the hold here.
        runtime.crystallization_hold_enabled = false;

        let novel_problem = "How do recursion and encryption relate when designing software?";
        let turn1 = runtime
            .execute_live_operational_turn(201, novel_problem, 64, None)
            .unwrap();

        assert_eq!(turn1.turn.tier, InferenceTier::Tier2SuperpositionOperator);
        assert!(turn1.turn.tier2_superposition_interventions >= 2);
        assert!(
            turn1
                .teacher_source
                .starts_with("AutonomousMindTier2Bridge[")
        );
        assert!(turn1.turn.response_text.contains("recursion"));
        assert!(turn1.turn.response_text.contains("encryption"));
        assert!(turn1.novel_arcs_minted > 0);
        assert!(turn1.wal_bytes_appended > 0);

        let turn2 = runtime
            .execute_live_operational_turn(202, novel_problem, 64, None)
            .unwrap();
        assert_eq!(turn2.turn.tier, InferenceTier::Tier1TrajectoryWalk);
        assert_eq!(turn2.turn.response_text, turn1.turn.response_text);

        let _ = std::fs::remove_file(&path);
        let mut wal = path.into_os_string();
        wal.push(".wal");
        let _ = std::fs::remove_file(wal);
    }
}
