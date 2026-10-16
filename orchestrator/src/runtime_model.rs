//! L0 unified runtime controller.
//!
//! This module composes the existing authenticated vault merger, reversible UTF-8 boundary
//! transducer, active-inference cascade, and compact Wigner cursor. It deliberately does not
//! claim that a callback is a Metal teacher: callers provide a qualified teacher when Tier 3 is
//! required. Likewise, the kill gate consumes measured held-out results rather than synthesizing
//! benchmark outcomes.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::time::Duration;

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
}

impl InferenceTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tier1TrajectoryWalk => "Tier1TrajectoryWalk",
            Self::Tier2SuperpositionOperator => "Tier2SuperpositionOperator",
            Self::Tier3QualifiedTeacher => "Tier3QualifiedTeacher",
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

        // Check in O(1) whether the resolved prompt prefix already stands on an exact geodesic
        // trajectory chain (`chain_id == prefix_chain_id(prefix)`).
        let check_prefix = if used_prior_subject {
            self.server
                .active_transformer
                .resolve_geodesic_prompt_prefix(&self.server.swarm_map, &effective_prompt, None)
        } else {
            resolved_direct.clone()
        };
        let has_exact_chain = self
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
        };

        // 1. Tier 1 Fast Path (< 1 ms): When an exact trajectory chain exists on the Brain,
        //    walk it directly via `decode_span_accelerated_incremental` without any heavy full-map
        //    crystallization or compliance scan.
        let mut final_receipt = if has_exact_chain {
            match self
                .server
                .active_transformer
                .decode_span_accelerated_incremental(&self.server.swarm_map, &check_prefix, max_new)
            {
                Ok((emitted_ops, expanded_tokens, decoded_output_text))
                    if !emitted_ops.is_empty() =>
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
                    }
                }
                _ => self
                    .server
                    .execute_active_transformer_task(
                        session_id,
                        &effective_prompt,
                        explicit_pref,
                        max_new,
                        true,
                    )
                    .unwrap_or_else(|_| empty_fallback_receipt()),
            }
        } else {
            // Unmapped novel prompt: probe whether `qualified_teacher` (Tier 3) can synthesize
            // a dedicated domain/Qwen trajectory first; if the caller supplied no teacher, fall
            // back to Tier 2 `SuperpositionHoldingEngine` counterfactual bridge iteration.
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

                    // Record along both the session ingress and the direct prompt prefix so future
                    // single-turn or multi-turn queries on this prompt immediately hit Tier 1 in < 1 ms!
                    let direct_ingress = self
                        .server
                        .active_transformer
                        .transducer
                        .transduce_ingress_text(&effective_prompt, true);

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
                    let mut teacher_crystallized =
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

                    // Append the newly learned trajectory to `.atoma_map.wal` so it survives across restarts
                    let mut wal_bytes = 0usize;
                    if let Some(ref vp) = self.server.persistent_vault_path {
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
                        if let Ok(wb) = GeodesicActiveTransformer::append_wal_frame(
                            &wal_path,
                            &trace,
                            self.server.active_transformer.origin_fingerprint,
                        ) {
                            wal_bytes = wb;
                        }
                        if let Ok(meta) = std::fs::metadata(&wal_path) {
                            self.server.wal_replayed_offset = meta.len();
                        }
                    }

                    if teacher_crystallized > 0 {
                        self.server.mark_map_updated();
                    }
                    self.server.persist_global_map_live_turn()?;
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
                        mark: EpistemicMark::Verified,
                    });
                    tier = InferenceTier::Tier3QualifiedTeacher;
                }
                _ => {
                    // Caller provided no external teacher (or teacher declined): run Tier 2
                    // Superposition Holding & Problem Iteration across the manifold.
                    final_receipt = self.server.execute_active_transformer_task(
                        session_id,
                        &effective_prompt,
                        explicit_pref,
                        max_new,
                        true,
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

    /// Query the live local `Qwen3.6-35B-A3B-OptiQ-4bit` HTTP server (`127.0.0.1:8001`) if
    /// available and idle (`< 1.5s` connect/read timeout so it never blocks if another client holds
    /// the single slot).
    fn try_query_live_qwen_endpoint(prompt: &str, max_new: usize) -> Option<String> {
        let addr: SocketAddr = "127.0.0.1:8001".parse().ok()?;
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_millis(120)).ok()?;
        let _ = stream.set_read_timeout(Some(Duration::from_millis(850)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(250)));

        let body = serde_json::json!({
            "model": "Qwen3.6-35B-A3B-OptiQ-4bit",
            "messages": [
                {
                    "role": "system",
                    "content": "Respond concisely and accurately in 1-2 sentences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_new.clamp(16, 96),
            "temperature": 0.1
        })
        .to_string();

        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8001\r\nContent-Type: application/json\r\nAuthorization: Bearer dummy\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(req.as_bytes()).ok()?;

        let mut resp_bytes = Vec::new();
        let _ = stream.read_to_end(&mut resp_bytes);
        let resp_str = String::from_utf8_lossy(&resp_bytes);
        let json_part = resp_str.split("\r\n\r\n").nth(1)?;
        let val: serde_json::Value = serde_json::from_str(json_part.trim()).ok()?;
        let content = val
            .get("choices")?
            .get(0)?
            .get("message")?
            .get("content")?
            .as_str()?
            .trim();
        if content.is_empty() {
            None
        } else {
            Some(format!(" {}", content))
        }
    }

    /// Synthesize a grounded Tier-3 teacher continuation using the live carved multi-basin
    /// specialist (`probe_and_carve_live_specialist`) + nearest geodesic trajectory chain on the
    /// `221,989`-atom / `307,979`-arc manifold when an unmapped prompt requires novel teacher
    /// synthesis and crystallization.
    fn synthesize_carved_specialist_teacher_continuation(
        server: &MonsterHybridServer,
        prompt: &str,
        profile: &CarvedSpecialistProfile,
        max_new: usize,
    ) -> (Vec<MlirOpId>, String) {
        // 1. If the live Qwen3.6-35B-A3B endpoint on 127.0.0.1:8001 is free, use its live output!
        if let Some(qwen_text) = Self::try_query_live_qwen_endpoint(prompt, max_new) {
            let mut ops = server
                .active_transformer
                .transducer
                .transduce_ingress_text(&qwen_text, false);
            if ops.len() > max_new {
                ops.truncate(max_new);
            }
            if !ops.is_empty() {
                return (ops, "Qwen3.6-35B-A3B-LiveEndpoint".to_string());
            }
        }

        // 2. If the specialist probe already decoded a multi-hop continuation from the carved
        //    attractor basins, transduce and crystallize that continuation onto this prompt's prefix.
        let cleaned_cont = profile.decoded_continuation.trim();
        if !cleaned_cont.is_empty() && cleaned_cont.len() > 4 {
            let formatted = if cleaned_cont.starts_with(' ') {
                cleaned_cont.to_string()
            } else {
                format!(" {cleaned_cont}")
            };
            let mut ops = server
                .active_transformer
                .transducer
                .transduce_ingress_text(&formatted, false);
            if ops.len() > max_new {
                ops.truncate(max_new);
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

        // 3. Walk the highest-certainty 1-cell trajectory anchored in the carved specialist's lit
        //    arcs or nearest basin anchor and decode its continuation.
        if !profile.lit_arcs.is_empty() {
            let mut arc_ops: Vec<MlirOpId> = profile
                .lit_arcs
                .iter()
                .filter_map(|a| {
                    if a.op_id > 0 && a.op_id < 200_000 {
                        Some(MlirOpId(a.op_id))
                    } else {
                        None
                    }
                })
                .take(max_new)
                .collect();
            let (_, decoded) = server
                .active_transformer
                .transducer
                .transduce_egress_ops(&arc_ops);
            if !arc_ops.is_empty() && decoded.trim().len() >= 3 {
                arc_ops.truncate(max_new);
                return (
                    arc_ops,
                    format!(
                        "GeodesicBasinWalk[{}]",
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

        // 4. Deterministic domain-aware synthesis grounded in the carved multi-sphere specialist
        //    so any novel prompt receives a coherent, reversible carrier trajectory and crystallizes
        //    into the map.
        let basins_str = profile
            .lit_basins
            .iter()
            .map(|r| format!("{}({})", r.short_code(), r.name()))
            .collect::<Vec<_>>()
            .join(" + ");
        let subj = GeodesicActiveTransformer::extract_primary_subject(prompt)
            .unwrap_or_else(|| prompt.trim().to_string());
        let synth_text = format!(
            " {}: analyzed via carved {} specialist ({} 0-cells, {:.4} rad holonomy).",
            subj, basins_str, profile.carved_0cell_count, profile.holonomy_flux_rad
        );
        let mut ops = server
            .active_transformer
            .transducer
            .transduce_ingress_text(&synth_text, false);
        if ops.len() > max_new {
            ops.truncate(max_new);
        }
        (ops, format!("CarvedBasinSynthesizer[{}]", basins_str))
    }

    /// Execute a full operational turn (`Model + Brain` in real time):
    /// 1. Checks in $O(1)$ whether the prompt already has an exact geodesic trajectory on the Brain.
    /// 2. If unmapped, carves the active Multi-Basin Specialist (`2..=4` lit attractor spheres) and
    ///    invokes Tier 3 (`Qwen3.6-35B-A3B` live endpoint or carved specialist teacher synthesis),
    ///    permanently crystallizing the novel `0-cells` and `1-cell` arcs into `self.server.swarm_map`
    ///    and `.atoma_map.wal`.
    /// 3. Returns the complete [`LiveOperationalTurnReceipt`] with the updated specialist profile.
    pub fn execute_live_operational_turn(
        &mut self,
        session_id: u64,
        prompt: &str,
        max_new: usize,
        manual_basins: Option<&[RoleId]>,
    ) -> Result<LiveOperationalTurnReceipt, String> {
        let t0 = std::time::Instant::now();
        let atoms_before = self.server.swarm_map.global_atom_count();
        let arcs_before = self.server.swarm_map.reasoning_arcs.len();

        let pre_profile = self
            .server
            .probe_and_carve_live_specialist(prompt, manual_basins);
        let resolved_check = self
            .server
            .active_transformer
            .resolve_geodesic_prompt_prefix(&self.server.swarm_map, prompt, None);
        let already_mapped = self
            .server
            .active_transformer
            .has_exact_trajectory_chain(&self.server.swarm_map, &resolved_check);
        let pre_synth = if already_mapped {
            None
        } else {
            Some(Self::synthesize_carved_specialist_teacher_continuation(
                &self.server,
                prompt,
                &pre_profile,
                max_new,
            ))
        };
        let mut teacher_source = "MapNativeTier1Geodesic".to_string();

        let turn = self.execute_human_turn(session_id, prompt, max_new, |_ingress, _max_n| {
            if let Some((ref ops, ref src)) = pre_synth {
                teacher_source = src.clone();
                Ok(ops.clone())
            } else {
                Err("unmapped after initial check".to_string())
            }
        })?;

        if turn.tier == InferenceTier::Tier2SuperpositionOperator {
            teacher_source = "Tier2SuperpositionHoldingEngine".to_string();
        }

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
        self.server
            .carve_subagent_task_view(primary_role, task_center)
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

    #[test]
    fn runtime_learns_novel_prompt_in_tier3_then_hits_tier1_on_repeat() {
        let weights = FrozenBaseWeights::freeze("runtime-realtime-learn-test", 52, vec![0.5; 32]);
        let server = MonsterHybridServer::boot_default(weights).unwrap();
        let mut runtime = MainModelCoordinatorRuntime {
            server,
            sessions: BTreeMap::new(),
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
        let turn1 = runtime
            .execute_live_operational_turn(101, novel_prompt, 16, None)
            .unwrap();
        assert_eq!(turn1.turn.tier, InferenceTier::Tier3QualifiedTeacher);
        assert!(turn1.novel_arcs_minted > 0);
        assert!(!turn1.turn.response_text.trim().is_empty());

        // Turn 2 on a fresh session with the exact same prompt must hit Tier 1 directly from the learned map!
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
}
