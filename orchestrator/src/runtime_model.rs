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
    AgenticTaskCompletionReceipt, FrozenBaseWeights, GeodesicActiveTransformer, GeodesicCursor,
    MlirOpId, NativeReader, PREREGISTERED_KILL_FINGERPRINT, PreregisteredEvaluation, RoleId,
    SubAgentTaskReceipt, TaskCarvedView,
};

use crate::MonsterHybridServer;

/// The execution path selected for a returned turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceTier {
    Tier1TrajectoryWalk,
    Tier2SuperpositionOperator,
    Tier3QualifiedTeacher,
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
}

impl MainModelCoordinatorRuntime {
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
        })
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
        self.execute_human_turn_with_ids(
            session_id,
            prompt,
            max_new,
            None,
            qualified_teacher,
        )
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
        // Intake is a live map operation, not just session bookkeeping: query_swarm_map encodes
        // the prompt into a deterministic map-native state and novelty-only distills it.
        self.server.query_swarm_map(prompt, None, 1, 0, true)?;
        self.server.persist_global_map_live_turn()?;

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

        // Resolve multi-turn follow-up coreference if the current turn does not directly match
        // a trajectory chain on its own and the session holds a prior turn subject.
        let mut effective_prompt = prompt.to_string();
        let mut used_prior_subject = false;
        let resolved_direct = self
            .server
            .active_transformer
            .resolve_geodesic_prompt_prefix(
                &self.server.swarm_map,
                prompt,
                Some(&session_prefix),
            );
        if !self
            .server
            .active_transformer
            .has_exact_trajectory_chain(&self.server.swarm_map, &resolved_direct)
            && GeodesicActiveTransformer::has_coreference_cue(prompt)
        {
            if let Some(ref subj) = prior_subject {
                let candidate_followup = format!(
                    "{} of {}",
                    prompt.trim().trim_end_matches('?'),
                    subj
                );
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
        let receipt = self.server.execute_active_transformer_task(
            session_id,
            &effective_prompt,
            explicit_pref,
            max_new,
            true,
        )?;
        let mut final_receipt = receipt;
        let mut tier = if final_receipt.superposition_interventions == 0 {
            InferenceTier::Tier1TrajectoryWalk
        } else {
            InferenceTier::Tier2SuperpositionOperator
        };
        let mut teacher_crystallized = 0usize;

        if final_receipt.emitted_ops.is_empty() {
            let teacher_ops = qualified_teacher(&ingress, max_new)?;
            let writes = NativeReader::record_continuation(
                &mut self.server.swarm_map,
                &self.server.mlir_catalog,
                &self.server.immune.policy,
                &ingress,
                &teacher_ops,
                &[],
                self.server.active_transformer.origin_fingerprint,
                Q32_31::from_f64(0.95),
            )?;
            teacher_crystallized = writes.iter().filter(|write| write.arc_was_novel).count();
            if teacher_crystallized > 0 {
                self.server.mark_map_updated();
            }
            self.server.persist_global_map_live_turn()?;
            final_receipt.emitted_ops = teacher_ops;
            let (tokens, text) = self
                .server
                .active_transformer
                .transducer
                .transduce_egress_ops(&final_receipt.emitted_ops);
            final_receipt.expanded_tokens = tokens;
            final_receipt.decoded_output_text = text;
            tier = InferenceTier::Tier3QualifiedTeacher;
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
                .map_or(0, |r| r.novel_arcs_minted)
                + teacher_crystallized,
            receipt: final_receipt,
        })
    }

    /// Execute a human turn with the checkpoint-backed Glimmer teacher. The teacher runs its
    /// complete model forward (including the checkpoint's attention, MLP, norm, and vocabulary
    /// head) for prompt prefill and each generated token. Its weight manager uses Metal on
    /// supported Macs and its existing CPU/Accelerate fallback elsewhere.
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

    /// Carve a read-only [`TaskCarvedView`] for a session's specialist role and optional
    /// hyperbolic horizon (`Qualification #4`).
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
        self.server.carve_subagent_task_view(primary_role, task_center)
    }

    /// Execute an ephemeral sub-agent turn inside a structurally carved [`TaskCarvedView`]
    /// (`Qualification #4`: specialization by subtraction) while maintaining the caller's
    /// session history and coreference state.
    pub fn execute_subagent_turn(
        &mut self,
        session_id: u64,
        prompt: &str,
        primary_role: Option<RoleId>,
        max_task_cosh: Option<Q32_31>,
        max_new: usize,
    ) -> Result<SubAgentTaskReceipt, String> {
        self.server.query_swarm_map(prompt, primary_role, 1, 0, true)?;
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
            session.cursor = GeodesicCursor::from_prefix(
                &self.server.mlir_catalog,
                &session.carrier_history,
            )
            .ok();
        }
        self.server.persist_global_map_live_turn()?;
        Ok(receipt)
    }

    /// Evaluate supplied, measured held-out results. This never fabricates a pass from generation.
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

/// Loaded checkpoint and tokenizer used for the real Tier 3 forward pass.
///
/// Construct one teacher per runtime and reuse it. Geometric attention maps and positions are
/// retained per session; only a session's explicit reset clears its attention history.
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

    /// Whether the loaded checkpoint has an active Metal quantized GEMV backend.
    pub fn metal_enabled(&self) -> bool {
        self.model.weights.use_metal
    }

    /// Encode a prompt with this checkpoint tokenizer and prepend the configured BOS token.
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

    /// Encode the next user turn with checkpoint token IDs. BOS is included only for a new or
    /// cleared session so the runtime's trajectory prefix matches model prefill.
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

    /// Greedy continuation using the session's retained geometric attention maps. The first call
    /// adds BOS; later calls process only the new user text, preceded by an EOS turn separator.
    /// Returned IDs are checkpoint token IDs and can be recorded as native trajectory carriers.
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
        } else {
            if let Some(pending) = self.active_pending_token.take() {
                self.model
                    .forward(pending, self.active_next_position)
                    .map_err(|e| e.to_string())?;
                self.active_next_position += 1;
            }
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

    /// Start a clean attention history for one session. Other sessions' maps remain cached.
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
}
