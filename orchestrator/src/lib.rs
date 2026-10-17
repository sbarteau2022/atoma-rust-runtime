//! `atoma_orchestrator`: The Unified Three-Layer Monster Hybrid Server.
//!
//! Ties together the complete chain defined in `SPEC.md` §4:
//! `Distill → Refract → Carve → Silo → Agent → Dissolve & Deposit → Govern → Immune-check`
//! with a **Model-Agnostic Centralized `(H³ × T²) / Γ` Memory Architecture** that survives
//! across pluggable model hot-swaps without re-distillation, a **Geodesic Native MLIR /
//! Tokenless Inter-Process Execution Engine**, and an **Open-Weights Knowledge & Reasoning
//! Distillation Engine** (`OpenWeightsDistillationEngine`).

pub mod benchmark_operators;
pub mod bridge;
#[cfg(feature = "glimmer")]
pub mod glimmer_map;
pub mod http_api;
pub mod live_map_ui;
pub mod routing;
pub mod runtime_model;
pub mod swarm;
pub mod toroidal_attention;

use atoma_immune::cache::geometric::{AtomId, Phase};
use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    ActiveInferenceConfig, ActiveInferenceContinuation, ActiveInferenceStepReceipt,
    AgenticTaskCompletionReceipt, Atlas13, AtomicIrAlphabet, CarveLossReport, CarvedSwarmMap,
    CompliancePermit, ComplianceViolation, DepositCompilationReceipt, EphemeralMapDeposit,
    FirstPrinciplesComplianceEngine, FirstPrinciplesComplianceReport, FrozenBaseWeights,
    FrozenShard, FullModelDistillationReceipt, GeodesicActiveTransformer, GeodesicDecodeReceipt,
    GeodesicMlirDecoder, HolonomyLoopReceipt, HyperbolicHolonomyEngine, IDENTITY_FRAME_Q32,
    ImmuneSystem, LiveNoveltyDistillationReceipt, MapNativeInferenceEngine,
    MetaCognitiveRunbookController, MlirAtomCatalog, MlirOpId, MlirSsaBlock,
    MlirSsaValidationReceipt, ModelHotSwapReceipt, NativeReader, OpenWeightsDistillationEngine,
    OpticalLens, OverlapPolicyMatrix, ProjectedTangent, QuantizedDistillationReceipt,
    QuantizedLayerTapDescriptor, ReasoningChainDistillationReceipt, RefractedPoint, RoleId,
    SpanFoldingReceipt, SubAgentTaskReceipt, SuperpositionState, SurgicalErasureReceipt,
    SwarmKnowledgeAndHomeostasis, TangentProjector, TaskCarvedView, TeacherTapRecord,
    WeightDistillationReceipt, fold_high_certainty_trajectory_spans,
};
use routing::{DeliveredHandoff, HandoffRequest};
use std::sync::Arc;
use swarm::{
    CrossEpochHydrationReceipt, GovernedForwardReceipt, Governor, RestartIntensity, RestartStrategy,
};

/// Complete record of a single pass through steps 1–5 (`Distill → Refract → Carve → Silo`).
#[derive(Debug, Clone)]
pub struct DistillRefractCarveReceipt {
    pub atom_id: AtomId,
    pub target_role: RoleId,
    pub shard: FrozenShard,
    pub projected: ProjectedTangent,
    pub refracted: RefractedPoint,
    pub carve_loss: CarveLossReport,
    pub loss_permit: CompliancePermit,
}

/// Receipt emitted when a typed [`MlirSsaBlock`] is validated by Layer 3 (`MlirNativeSsaGate`),
/// transmitted over the binary ring buffer (zero JSON/UTF-8), and deterministically executed
/// by the receiving specialist role.
#[derive(Debug, Clone)]
pub struct NativeMlirDispatchReceipt {
    pub handoff: DeliveredHandoff,
    pub validation: MlirSsaValidationReceipt,
    pub ssa_permit: CompliancePermit,
    pub result_q32: Q32_31,
    pub expanded_model_tokens: Vec<u32>,
}

/// The central Monster Hybrid Server ("Three Layers, One Server — There is one door", `SPEC.md` §1).
pub struct MonsterHybridServer {
    /// Layer 3: Immune System (governs the Governor).
    pub immune: ImmuneSystem,
    /// Layer 2: Governor (Erlang-style supervision tree + router).
    pub governor: Governor,
    /// Layer 1: Model-Agnostic Centralized `(H³ × T²) / Γ` Memory & 12 Carved Silo Views.
    pub swarm_map: CarvedSwarmMap,
    /// Stage 3 deterministic projector with loss measurement.
    pub projector: TangentProjector,
    /// Q32.31 Meta-Cognitive Readiness (Ω) & Runbook Controller (`arXiv:2602.11897v3` + Elle Runbook).
    pub meta_controller: MetaCognitiveRunbookController,
    /// Active Inference & Superposition Holding configuration (`design/superpositionholding.md`).
    pub active_inference_config: ActiveInferenceConfig,
    /// Pinned Atomic IR lossless alphabet (`SPEC_atomic_ir.md`).
    pub atomic_ir: AtomicIrAlphabet,
    /// Pinned Geodesic MLIR Opcode Catalog on `(H³ × T²)/Γ₁₂` (`.atoma_ir` / `mlir_atom_catalog.sql`).
    pub mlir_catalog: MlirAtomCatalog,
    /// 12-Sphere Superpowers ontology (CVE/CWE/ATT&CK) & 8-discipline homeostatic mapping.
    pub knowledge_homeostasis: SwarmKnowledgeAndHomeostasis,
    /// Non-Abelian Levi-Civita Hyperbolic Holonomy & Gauss–Bonnet Berry Phase Engine.
    pub holonomy_engine: HyperbolicHolonomyEngine,
    /// Autonomous Open-Weights Knowledge (`0-cells`) & Reasoning (`1-cells` / `2-cells`) Distillation Engine.
    pub distillation_engine: OpenWeightsDistillationEngine,
    /// Map-Native Coupled Operator-Atom Inference Engine (`Phase 1 -> Phase 2 -> Phase 3 Weightless`).
    pub map_native_engine: MapNativeInferenceEngine,
    /// Continuous Geodesic Active-Inference Transformer (`O(1)` `GeodesicCursor`, Boundary
    /// Transducer, Mid-Task Crystallization, and Binary WAL Journal).
    pub active_transformer: GeodesicActiveTransformer,
    /// Last ingested live inference atom & role for continuous `1-cell` reasoning-arc synthesis.
    pub last_live_atom: Option<(AtomId, RoleId)>,
    /// Optional path to the persistent `.atoma_map` binary vault on disk.
    pub persistent_vault_path: Option<std::path::PathBuf>,
    /// Byte offset up to which `.atoma_map.wal` has already been replayed into `self.swarm_map`.
    pub wal_replayed_offset: u64,
    /// Monotonic centralized-map generation used to invalidate worker-local atom caches.
    pub map_generation: u64,
}

impl MonsterHybridServer {
    /// Boot the Three-Layer Server with the approved icosahedral `Atlas13`, initial pluggable
    /// model weights, and minimal icosahedral-neighbor overlap policy (`Qualification #6, #7, #8`).
    pub fn boot_default(base_weights: Arc<FrozenBaseWeights>) -> Result<Self, ComplianceViolation> {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let weight_fp = base_weights.fingerprint;

        let swarm_map = CarvedSwarmMap::new(atlas, base_weights, Q32_31::from_f64(1.22));
        let mut immune = ImmuneSystem::new(policy, weight_fp);
        let mut governor = Governor::new(
            RestartStrategy::OneForOne,
            RestartIntensity {
                max_restarts: 5,
                within_ticks: 100,
            },
            10,
        );

        governor.spawn_swarm(&swarm_map, &mut immune)?;
        let mut active_transformer = GeodesicActiveTransformer::new(202_048, None);
        let _ = active_transformer.transducer.load_or_bootstrap_default();

        Ok(Self {
            immune,
            governor,
            swarm_map,
            projector: TangentProjector::default(),
            meta_controller: MetaCognitiveRunbookController::default(),
            active_inference_config: ActiveInferenceConfig::default(),
            atomic_ir: AtomicIrAlphabet::new_canonical(202_048),
            mlir_catalog: MlirAtomCatalog::new_canonical(202_048),
            knowledge_homeostasis: SwarmKnowledgeAndHomeostasis::new(),
            holonomy_engine: HyperbolicHolonomyEngine::default(),
            distillation_engine: OpenWeightsDistillationEngine::default(),
            map_native_engine: MapNativeInferenceEngine::default(),
            active_transformer,
            last_live_atom: None,
            persistent_vault_path: None,
            wal_replayed_offset: 0,
            map_generation: 0,
        })
    }

    /// Execute an autonomous task in motion through the server's `GeodesicActiveTransformer`,
    /// transducing raw UTF-8 text at the outer boundary, navigating via $O(1)$ `GeodesicCursor`
    /// + `SuperpositionHoldingEngine`, and crystallizing verified trajectories into `self.swarm_map`
    /// and its write-ahead delta journal (`.atoma_map.wal`).
    pub fn execute_active_transformer_task(
        &mut self,
        episode_id: u64,
        prompt_text: &str,
        explicit_prefix: Option<&[MlirOpId]>,
        max_new_steps: usize,
        auto_crystallize_on_success: bool,
    ) -> Result<AgenticTaskCompletionReceipt, String> {
        let _ = self.sync_live_wal_if_grown();
        self.active_transformer.wal_path = self.persistent_vault_path.as_ref().map(|p| {
            let mut s = p.as_os_str().to_os_string();
            s.push(".wal");
            std::path::PathBuf::from(s)
        });
        let receipt = self.active_transformer.execute_task_in_motion(
            episode_id,
            prompt_text,
            explicit_prefix,
            max_new_steps,
            auto_crystallize_on_success,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        if let Some(ref wal_path) = self.active_transformer.wal_path {
            if let Ok(meta) = std::fs::metadata(wal_path) {
                self.wal_replayed_offset = meta.len();
            }
        }
        if receipt.superposition_interventions > 0 || receipt.crystallization.is_some() {
            self.mark_map_updated();
            self.persist_global_map_live_turn()?;
        }
        Ok(receipt)
    }

    /// Carve a read-only [`TaskCarvedView`] for an ephemeral sub-agent anchored on `primary_role`
    /// and its policy-permitted icosahedral neighbors (`OverlapPolicyMatrix`) plus optional
    /// hyperbolic task horizon `task_center` (`Qualification #4`).
    pub fn carve_subagent_task_view(
        &self,
        primary_role: RoleId,
        task_center: Option<(&atoma_immune::geometry::lorentz::LorentzVector, Q32_31)>,
    ) -> TaskCarvedView {
        self.swarm_map
            .carve_task_view_for_role(primary_role, &self.immune.policy, task_center)
    }

    /// Execute an ephemeral sub-agent task inside a structurally carved [`TaskCarvedView`]
    /// (`Qualification #4`: specialization by subtraction) and optionally crystallize verified
    /// trajectories back into `self.swarm_map` and `.atoma_map.wal`.
    pub fn execute_subagent_task(
        &mut self,
        episode_id: u64,
        prompt_text: &str,
        explicit_prefix: Option<&[MlirOpId]>,
        primary_role: Option<RoleId>,
        max_task_cosh: Option<Q32_31>,
        max_new_steps: usize,
        crystallize_to_parent: bool,
    ) -> Result<SubAgentTaskReceipt, String> {
        let _ = self.sync_live_wal_if_grown();
        self.active_transformer.wal_path = self.persistent_vault_path.as_ref().map(|p| {
            let mut s = p.as_os_str().to_os_string();
            s.push(".wal");
            std::path::PathBuf::from(s)
        });
        let receipt = self.active_transformer.execute_subagent_in_task_view(
            episode_id,
            prompt_text,
            explicit_prefix,
            max_new_steps,
            primary_role,
            max_task_cosh,
            crystallize_to_parent,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        if let Some(ref wal_path) = self.active_transformer.wal_path {
            if let Ok(meta) = std::fs::metadata(wal_path) {
                self.wal_replayed_offset = meta.len();
            }
        }
        if receipt.compiled_back_arcs > 0 || receipt.task_receipt.superposition_interventions > 0 {
            self.mark_map_updated();
            self.persist_global_map_live_turn()?;
        }
        Ok(receipt)
    }

    /// Fold high-certainty unbranched trajectory chains (`3..=6` tokens inside the `<= 7`-token
    /// verify cliff) into single-hop `AtomicIrAlphabet` span opcodes (`MlirOpId >= base_vocab_size`).
    pub fn fold_map_trajectory_spans(
        &mut self,
        max_spans_to_mint: usize,
    ) -> Result<SpanFoldingReceipt, String> {
        let receipt = fold_high_certainty_trajectory_spans(
            &mut self.swarm_map,
            &self.mlir_catalog,
            &self.immune.policy,
            &mut self.active_transformer.transducer.atomic_ir,
            max_spans_to_mint,
        )?;
        if receipt.shortcut_span_arcs_minted > 0 || receipt.new_span_atoms_pinned > 0 {
            self.mark_map_updated();
        }
        Ok(receipt)
    }

    /// Merge another `.atoma_map` binary vault (e.g. a newly distilled model's static modes or
    /// token trajectories) directly into `self.swarm_map` with 3-level deduplication.
    pub fn merge_additional_model_vault(
        &mut self,
        other_vault_path: &std::path::Path,
        recarve: bool,
    ) -> Result<(usize, usize), String> {
        let res =
            self.swarm_map
                .merge_vault_from(other_vault_path, &self.immune.policy, recarve)?;
        self.mark_map_updated();
        if let Some(ref vp) = self.persistent_vault_path {
            self.swarm_map
                .save_to_binary_vault(vp)
                .map_err(|e| format!("persist merged global map {}: {e}", vp.display()))?;
        }
        Ok(res)
    }

    /// Handle a zero-JSON `ATOMA_WR` binary wire frame (`NativeWireRequest` -> `NativeWireResponse`).
    pub fn handle_native_wire_frame(&mut self, request_bytes: &[u8]) -> Result<Vec<u8>, String> {
        self.active_transformer.wal_path = self.persistent_vault_path.as_ref().map(|p| {
            let mut s = p.as_os_str().to_os_string();
            s.push(".wal");
            std::path::PathBuf::from(s)
        });
        let res = self.active_transformer.handle_binary_wire_frame(
            request_bytes,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        self.mark_map_updated();
        Ok(res)
    }

    /// Execute `Distill (Tap + Hash) → Refract (Project + ExpMap) → Carve (×12) → Silo → Immune-check`
    /// anchored around `target_role`'s well direction on `H³ × T²`.
    pub fn ingest_teacher_tap(
        &mut self,
        tap: &TeacherTapRecord,
        target_role: RoleId,
        radial_offset_tangent: [Q32_31; 3],
    ) -> Result<DistillRefractCarveReceipt, String> {
        self.ingest_teacher_tap_at_phase(tap, target_role, radial_offset_tangent, None)
    }

    pub(crate) fn ingest_teacher_tap_at_phase(
        &mut self,
        tap: &TeacherTapRecord,
        target_role: RoleId,
        radial_offset_tangent: [Q32_31; 3],
        target_phase: Option<Phase>,
    ) -> Result<DistillRefractCarveReceipt, String> {
        let shard = FrozenShard::from_tap(tap).map_err(|e| e.to_string())?;
        let mut projected = self.projector.project(tap).map_err(|e| e.to_string())?;

        let well = self.swarm_map.atlas.well(target_role);
        if let Some(phase) = target_phase {
            projected.phase_drive_t2 = [
                Q32_31(phase.theta_f.0.wrapping_sub(well.phase.theta_f.0)),
                Q32_31(phase.theta_s.0.wrapping_sub(well.phase.theta_s.0)),
            ];
        }
        let d_sph = Q32_31::from_f64(0.804_735_5);
        projected.tangent_h3 = [
            (well.unit_dir[0] * d_sph) + radial_offset_tangent[0],
            (well.unit_dir[1] * d_sph) + radial_offset_tangent[1],
            (well.unit_dir[2] * d_sph) + radial_offset_tangent[2],
        ];

        let refracted = OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
        let atom_id =
            self.swarm_map
                .keep_and_carve(refracted, tap.payload.clone(), &self.immune.policy);
        let carve_loss = self.swarm_map.last_carve_loss.clone();
        self.map_generation = self.map_generation.wrapping_add(1);

        let loss_permit = self
            .immune
            .verify_loss_accounting(
                &projected.loss_report,
                &refracted.refraction_loss,
                &carve_loss,
            )
            .map_err(|e| e.to_string())?;

        Ok(DistillRefractCarveReceipt {
            atom_id,
            target_role,
            shard,
            projected,
            refracted,
            carve_loss,
            loss_permit,
        })
    }

    /// Distill static **Knowledge (`0-cells`)** directly from an open-source teacher checkpoint's
    /// frozen weight buffer (`teacher_weights`), auto-routing each principal mode across the 12
    /// icosahedral specialist wells (`L1..L12`).
    pub fn distill_open_source_weights(
        &mut self,
        teacher_weights: &FrozenBaseWeights,
        mode_width: usize,
        max_modes: usize,
    ) -> Result<WeightDistillationReceipt, String> {
        let receipt = self.distillation_engine.distill_static_weight_modes(
            teacher_weights,
            mode_width,
            max_modes,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        self.mark_map_updated();
        Ok(receipt)
    }

    /// Distill **Knowledge (`0-cells`)** directly from a **Quantized Layer Tensor** (`OptiQ` / `MLX`
    /// `4-bit`, `5-bit`, `6-bit`, `8-bit` packed `U32`/`U8` + `BF16`/`F16`/`F32` scales & biases)
    /// without ever dequantizing the full matrix (`128×` faster scale/bias spectral sieve +
    /// top-$K$ single-row `dequantize_row_zero_copy`).
    pub fn distill_quantized_layer(
        &mut self,
        desc: &QuantizedLayerTapDescriptor<'_>,
        top_k_rows: usize,
        teacher_fingerprint: atoma_immune::Sha256Digest,
    ) -> Result<QuantizedDistillationReceipt, String> {
        let receipt = self.distillation_engine.distill_quantized_layer_zero_copy(
            desc,
            top_k_rows,
            teacher_fingerprint,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        self.mark_map_updated();
        Ok(receipt)
    }

    /// Automatically discover and distill all `OptiQ` / `MLX` quantized linear layers (`<prefix>.weight`,
    /// `<prefix>.scales`, `<prefix>.biases`, optional `<prefix>.out_scale`) inside a `.safetensors`
    /// shard buffer into the centralized `CarvedSwarmMap`.
    pub fn distill_safetensors_shard(
        &mut self,
        file_bytes: &[u8],
        top_k_rows_per_layer: usize,
        total_layers_hint: usize,
    ) -> Result<Vec<QuantizedDistillationReceipt>, String> {
        let receipts = self.distillation_engine.distill_safetensors_shard_bytes(
            file_bytes,
            top_k_rows_per_layer,
            total_layers_hint,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        self.mark_map_updated();
        Ok(receipts)
    }

    /// Stream and distill quantized layers (`2D` linear projections and `3D` MoE `switch_mlp` experts)
    /// directly from a `.safetensors` file on disk using selective `File::seek` (`0` full-file read,
    /// `0` full-matrix dequantization), resolving exact per-layer bit-widths from `optiq_metadata.json`
    /// or `config.json`.
    pub fn distill_safetensors_file(
        &mut self,
        safetensors_path: &std::path::Path,
        top_k_rows_per_layer: usize,
        max_layers: usize,
    ) -> Result<Vec<QuantizedDistillationReceipt>, String> {
        let receipts = self
            .distillation_engine
            .distill_safetensors_file_streaming(
                safetensors_path,
                top_k_rows_per_layer,
                max_layers,
                &mut self.swarm_map,
                &self.immune.policy,
            )?;
        self.mark_map_updated();
        Ok(receipts)
    }

    /// Full-model or shard distillation with **3-level novelty-only deduplication** (`0` duplicate noise)
    /// and automatic `.atoma_map` binary vault persistence.
    pub fn distill_full_checkpoint_novel_only(
        &mut self,
        target_path: &std::path::Path,
        top_k_rows_per_layer: usize,
        max_layers_per_shard: usize,
        vault_path: Option<&std::path::Path>,
    ) -> Result<FullModelDistillationReceipt, String> {
        if let Some(vp) = vault_path {
            self.persistent_vault_path = Some(vp.to_path_buf());
            if vp.exists() && self.swarm_map.ingested_tensor_digests.is_empty() {
                let _ = self
                    .swarm_map
                    .load_from_binary_vault(vp, &self.immune.policy)?;
            }
        }

        let mut shard_paths = Vec::new();
        if target_path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(target_path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if !name.starts_with("._")
                            && ((name.starts_with("model-") && name.ends_with(".safetensors"))
                                || name == "model.safetensors")
                        {
                            shard_paths.push(p);
                        }
                    }
                }
            }
            // Fallback / supplement: scan `.cache/huggingface/download/` for partial or cached `.incomplete` / `.safetensors` shards
            let hf_dl_dir = target_path.join(".cache").join("huggingface").join("download");
            if hf_dl_dir.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&hf_dl_dir) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        let len = p.metadata().map(|m| m.len()).unwrap_or(0);
                        if len < 65_536 {
                            continue;
                        }
                        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                            if !name.starts_with("._")
                                && (name.ends_with(".incomplete") || name.ends_with(".safetensors"))
                            {
                                shard_paths.push(p);
                            }
                        }
                    }
                }
            }
            shard_paths.sort();
        } else {
            shard_paths.push(target_path.to_path_buf());
        }

        let mut total_summary = FullModelDistillationReceipt {
            model_path: target_path.display().to_string(),
            shards_processed: shard_paths.len(),
            tensors_distilled: 0,
            tensors_skipped_already_ingested: 0,
            novel_atoms_minted: 0,
            deduplicated_atoms_coalesced: 0,
            novel_reasoning_arcs_minted: 0,
            deduplicated_reasoning_arcs: 0,
            f32_heap_bytes_saved: 0,
            role_distribution: [0usize; 12],
            vault_sha256: None,
        };

        for sp in &shard_paths {
            let (_recs, sub) = self
                .distillation_engine
                .distill_safetensors_file_novel_only(
                    sp,
                    top_k_rows_per_layer,
                    max_layers_per_shard,
                    &self.mlir_catalog,
                    &mut self.swarm_map,
                    &self.immune.policy,
                )?;
            total_summary.tensors_distilled += sub.tensors_distilled;
            total_summary.tensors_skipped_already_ingested += sub.tensors_skipped_already_ingested;
            total_summary.novel_atoms_minted += sub.novel_atoms_minted;
            total_summary.deduplicated_atoms_coalesced += sub.deduplicated_atoms_coalesced;
            total_summary.novel_reasoning_arcs_minted += sub.novel_reasoning_arcs_minted;
            total_summary.deduplicated_reasoning_arcs += sub.deduplicated_reasoning_arcs;
            total_summary.f32_heap_bytes_saved += sub.f32_heap_bytes_saved;
            for i in 0..12 {
                total_summary.role_distribution[i] += sub.role_distribution[i];
            }
        }

        if let Some(vp) = vault_path {
            let (_atoms, _arcs, sha) = self.swarm_map.save_to_binary_vault(vp)?;
            total_summary.vault_sha256 = Some(sha);
        }

        self.mark_map_updated();
        Ok(total_summary)
    }

    /// Continuous Live Novelty-Only Distillation Gate when a model is hooked up inside the server:
    /// - Drops high-entropy guesses (`H(p) > max_entropy_bits_q32`).
    /// - Deduplicates already-known `0-cell` information and `1-cell` reasoning transitions in `O(1)`.
    /// - Mints and persists only genuinely new knowledge and reasoning transitions to `persistent_vault_path`.
    pub fn observe_and_distill_live_step(
        &mut self,
        tap: &TeacherTapRecord,
    ) -> Result<LiveNoveltyDistillationReceipt, String> {
        let model_fp = self.swarm_map.base_weights.fingerprint;
        let total_layers = self.swarm_map.base_weights.num_layers.max(1);
        let receipt = self.distillation_engine.distill_live_step_novel_only(
            tap,
            total_layers,
            model_fp,
            self.last_live_atom,
            &self.mlir_catalog,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        if let (Some(aid), Some(r)) = (receipt.atom_id, receipt.role) {
            self.last_live_atom = Some((aid, r));
        }
        if (receipt.atom_was_novel || receipt.arc_was_novel) && self.persistent_vault_path.is_some()
        {
            self.persist_global_map_live_turn()?;
        }
        if receipt.atom_was_novel || receipt.arc_was_novel {
            self.mark_map_updated();
        }
        Ok(receipt)
    }

    /// Distill a multi-step teacher **Reasoning Trajectory (`1-cells` + `2-cell` Holonomy + Native MLIR)**
    /// into the centralized `CarvedSwarmMap`.
    pub fn distill_reasoning_trajectory(
        &mut self,
        chain_id: u64,
        teacher_steps: &[TeacherTapRecord],
        total_teacher_layers: usize,
        teacher_fingerprint: atoma_immune::Sha256Digest,
    ) -> Result<ReasoningChainDistillationReceipt, String> {
        let receipt = self.distillation_engine.distill_reasoning_chain(
            chain_id,
            teacher_steps,
            total_teacher_layers,
            teacher_fingerprint,
            &self.mlir_catalog,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        self.mark_map_updated();
        Ok(receipt)
    }

    /// Execute `Agent → Govern → Immune-check` inter-role handoff through the single server door,
    /// parallel-transporting the `3×3` Levi-Civita Wigner gyroscope frame on the wire.
    pub fn handoff_between_roles(
        &mut self,
        from_role: RoleId,
        to_role: RoleId,
        shared_atom_id: AtomId,
        message_bytes: Vec<u8>,
    ) -> Result<DeliveredHandoff, ComplianceViolation> {
        let req = HandoffRequest {
            from_role,
            to_role,
            shared_atom_id,
            incoming_wigner_frame: IDENTITY_FRAME_Q32,
            message_bytes,
        };
        self.governor
            .dispatch_handoff(req, &self.swarm_map, &mut self.immune)
    }

    /// Execute a governed `core_lm` forward pass on `role`'s supervised `CoreLmInferenceWorker`
    /// (`core_lm -> immune_system`, `core_lm -> orchestrator`, `orchestrator -> immune_system`).
    pub fn execute_governed_forward(
        &mut self,
        role: RoleId,
        token_id: u32,
        pos: usize,
        q_head: &[f32],
        k_head: &[f32],
        v_head: &[f32],
        o_proj_matrix: &[f32],
    ) -> Result<GovernedForwardReceipt, ComplianceViolation> {
        self.governor.synchronize_worker_memory(
            role,
            self.map_generation,
            &self.swarm_map,
            &mut self.immune,
        )?;
        self.governor.execute_governed_forward(
            role,
            token_id,
            pos,
            q_head,
            k_head,
            v_head,
            o_proj_matrix,
            &self.swarm_map,
            &mut self.immune,
        )
    }

    /// Dissolve `role`'s ephemeral agent upon task completion, surrender its local map back to
    /// the `Orchestrator` as an [`EphemeralMapDeposit`], and compile that deposit into the
    /// centralized `(H³ × T²) / Γ` memory architecture (`authorize_deposit_compilation`).
    pub fn dissolve_agent_and_compile_deposit(
        &mut self,
        role: RoleId,
    ) -> Result<
        (
            EphemeralMapDeposit,
            DepositCompilationReceipt,
            CompliancePermit,
        ),
        ComplianceViolation,
    > {
        let receipt = self.governor.dissolve_and_compile_deposit(
            role,
            &mut self.swarm_map,
            &mut self.immune,
        )?;
        self.mark_map_updated();
        Ok(receipt)
    }

    /// Hot-swap a new pluggable model system into the server while preserving 100% of the
    /// centralized `(H³ × T²) / Γ` memory map without re-distillation (`authorize_model_hot_swap`).
    pub fn hot_swap_model_backend(
        &mut self,
        new_weights: Arc<FrozenBaseWeights>,
    ) -> Result<(ModelHotSwapReceipt, CompliancePermit), ComplianceViolation> {
        let preserved = self.swarm_map.global_atom_count();
        let permit = self
            .immune
            .authorize_model_hot_swap(&new_weights, preserved)?;
        let receipt = self.swarm_map.hot_swap_model_backend(new_weights);
        self.mark_map_updated();
        Ok((receipt, permit))
    }

    /// Hydrate `role`'s specialist worker at `target_head_dim` from the centralized `CarvedSwarmMap`,
    /// automatically projecting any KV payloads deposited by older model epochs through
    /// `ModelAdapterLens` under Layer 3 (`authorize_cross_epoch_adaptation`).
    pub fn hydrate_role_from_central_vault(
        &mut self,
        role: RoleId,
        target_head_dim: usize,
    ) -> Result<CrossEpochHydrationReceipt, ComplianceViolation> {
        self.governor.hydrate_worker_from_silo_with_adapter(
            role,
            target_head_dim,
            &self.swarm_map,
            &mut self.immune,
        )
    }

    /// Synthesize a typed [`MlirSsaBlock`] directly from continuous model `hidden_states` via
    /// [`GeodesicMlirDecoder`] (bypassing `BpeTokenizer` and `token_id`s completely), verify it
    /// through Layer 3 (`MlirNativeSsaGate`), and dispatch + execute it across `from_role -> to_role`.
    pub fn synthesize_and_dispatch_geodesic_mlir(
        &mut self,
        from_role: RoleId,
        to_role: RoleId,
        shared_atom_id: AtomId,
        hidden_states: &[Vec<f32>],
    ) -> Result<(GeodesicDecodeReceipt, NativeMlirDispatchReceipt), ComplianceViolation> {
        let decode_receipt = GeodesicMlirDecoder::decode_from_hidden_states(
            &self.mlir_catalog,
            from_role,
            hidden_states,
        )
        .map_err(|e| ComplianceViolation::InvalidMlirSsaBlock {
            reason: e.to_string(),
        })?;
        let dispatch_receipt = self.dispatch_native_mlir_block(
            from_role,
            to_role,
            shared_atom_id,
            &decode_receipt.block,
        )?;
        Ok((decode_receipt, dispatch_receipt))
    }

    /// Dispatch and execute a native typed [`MlirSsaBlock`] between `from_role` and `to_role`
    /// over the binary handoff ring buffer (`0` JSON, `0` UTF-8, `0` Python glue), gated by
    /// Layer 3 (`ImmuneSystem::authorize_mlir_ssa_block` + `authorize_handoff`).
    pub fn dispatch_native_mlir_block(
        &mut self,
        from_role: RoleId,
        to_role: RoleId,
        shared_atom_id: AtomId,
        block: &MlirSsaBlock,
    ) -> Result<NativeMlirDispatchReceipt, ComplianceViolation> {
        let (validation, ssa_permit) = self.immune.authorize_mlir_ssa_block(
            from_role,
            to_role,
            block,
            &self.mlir_catalog,
            &self.swarm_map,
        )?;
        let wire_bytes = block.to_wire_bytes();
        let handoff = self.handoff_between_roles(from_role, to_role, shared_atom_id, wire_bytes)?;

        // Receiver decodes the binary MLIR_SSA wire payload directly and executes deterministically
        let decoded_block =
            MlirSsaBlock::from_wire_bytes(&handoff.wire_packet.payload).map_err(|e| {
                ComplianceViolation::InvalidMlirSsaBlock {
                    reason: e.to_string(),
                }
            })?;
        let result_q32 = decoded_block
            .execute_deterministic(&self.mlir_catalog, from_role)
            .map_err(|e| ComplianceViolation::InvalidMlirSsaBlock {
                reason: e.to_string(),
            })?;
        let expanded_model_tokens = decoded_block.expand_to_model_tokens(&self.mlir_catalog);

        Ok(NativeMlirDispatchReceipt {
            handoff,
            validation,
            ssa_permit,
            result_q32,
            expanded_model_tokens,
        })
    }

    /// Evaluate the Non-Abelian Levi-Civita Hyperbolic Holonomy and Gauss–Bonnet Berry Phase
    /// around a closed reasoning loop of specialist roles (`roles`).
    pub fn evaluate_reasoning_holonomy(&self, roles: &[RoleId]) -> HolonomyLoopReceipt {
        let init_phase = roles
            .first()
            .map(|&r| self.swarm_map.atlas.well(r).phase)
            .unwrap_or(atoma_immune::cache::geometric::Phase {
                theta_f: Q32_31::ZERO,
                theta_s: Q32_31::ZERO,
            });
        self.holonomy_engine
            .evaluate_role_loop(roles, &self.swarm_map, init_phase)
    }

    /// Execute First-Principle Surgical Right-to-Erasure (`GDPR Art. 17`) across the 1 + 12 Carved Swarm.
    pub fn surgical_erase(
        &mut self,
        atom_id: AtomId,
    ) -> Result<SurgicalErasureReceipt, &'static str> {
        let receipt = self
            .swarm_map
            .surgical_erase_atom(atom_id, &self.immune.policy)?;
        self.mark_map_updated();
        Ok(receipt)
    }

    /// Prove all 8 first-principle invariants and all 5 statutory frameworks directly from live server state.
    pub fn prove_first_principles_compliance(&mut self) -> FirstPrinciplesComplianceReport {
        let digests: Vec<_> = self
            .immune
            .audit_log()
            .iter()
            .map(|e| e.entry_digest)
            .collect();
        FirstPrinciplesComplianceEngine::prove_from_live_state(
            &mut self.swarm_map,
            &self.immune.policy,
            &digests,
        )
    }

    /// Execute one **Active Inference** step on `(H³ × T²)/Γ₁₂` (`design/superpositionholding.md`),
    /// minimizing Variational Free Energy (`F_q32`), enforcing Layer 3 compliance/safety/blast-radius
    /// ceilings and token consumption bounds, holding superposition when an issue occurs, and
    /// self-distilling any newly resolved bridge arc into `self.swarm_map`.
    pub fn active_inference_step(
        &mut self,
        prefix: &[MlirOpId],
        spent_tokens: u64,
    ) -> Result<ActiveInferenceStepReceipt, SuperpositionState> {
        let receipt = NativeReader::active_inference_step_mut(
            &mut self.swarm_map,
            &self.mlir_catalog,
            &self.immune.policy,
            prefix,
            &self.active_inference_config,
            spent_tokens,
        )?;
        if receipt.bridge_arc_distilled {
            self.mark_map_updated();
            let _ = self.persist_global_map_live_turn();
        }
        Ok(receipt)
    }

    /// Execute a multi-step **Active Inference** continuation on `(H³ × T²)/Γ₁₂`
    /// (`design/superpositionholding.md`), automatically stopping, holding superposition,
    /// reasoning across counterfactual `1-cell`/`2-cell` holonomy loops, and iterating to a
    /// new solution whenever an issue is encountered.
    pub fn active_inference_decode(
        &mut self,
        prefix: &[MlirOpId],
        max_new: usize,
        eos: &[MlirOpId],
    ) -> Result<ActiveInferenceContinuation, String> {
        let cont = NativeReader::continue_ids_active_mut(
            &mut self.swarm_map,
            &self.mlir_catalog,
            &self.immune.policy,
            prefix,
            max_new,
            eos,
            &self.active_inference_config,
        )?;
        if cont.bridge_arcs_distilled > 0 {
            self.mark_map_updated();
            self.persist_global_map_live_turn()?;
        }
        Ok(cont)
    }

    /// Load an existing `.atoma_map` binary vault into `self.swarm_map` (if it exists on disk)
    /// and bind `self.persistent_vault_path = Some(vault_path.to_path_buf())`.
    pub fn load_or_init_persistent_vault(
        &mut self,
        vault_path: &std::path::Path,
    ) -> Result<(usize, usize, Option<atoma_immune::Sha256Digest>), String> {
        self.persistent_vault_path = Some(vault_path.to_path_buf());
        if vault_path.exists() {
            let (_atoms, _arcs, sha) = self
                .swarm_map
                .load_from_binary_vault(vault_path, &self.immune.policy)?;
            let wal_path = Self::wal_path_for(vault_path);
            if wal_path.exists() {
                let (_steps, new_offset) = self.active_transformer.replay_wal_from_offset(
                    &wal_path,
                    0,
                    &mut self.swarm_map,
                    &self.immune.policy,
                )?;
                self.wal_replayed_offset = new_offset;
                self.persist_global_map_live_turn()?;
            } else {
                self.wal_replayed_offset = 0;
            }
            self.map_generation = self.map_generation.wrapping_add(1);
            Ok((
                self.swarm_map.global_atom_count(),
                self.swarm_map.reasoning_arcs.len(),
                Some(sha),
            ))
        } else {
            self.wal_replayed_offset = 0;
            Ok((
                self.swarm_map.global_atom_count(),
                self.swarm_map.reasoning_arcs.len(),
                None,
            ))
        }
    }

    /// Incrementally tail any new `.atoma_map.wal` frames appended by a background teacher
    /// harvester (`collect_qwen_trajectories.py`, `live_teacher_harvester.py`, etc.) into the
    /// in-memory `self.swarm_map` in `< 1 ms` without restarting the server.
    pub fn sync_live_wal_if_grown(&mut self) -> Result<usize, String> {
        let Some(ref vault_path) = self.persistent_vault_path else {
            return Ok(0);
        };
        let wal_path = Self::wal_path_for(vault_path);
        if !wal_path.exists() {
            return Ok(0);
        }
        let (replayed, new_offset) = self.active_transformer.replay_wal_from_offset(
            &wal_path,
            self.wal_replayed_offset,
            &mut self.swarm_map,
            &self.immune.policy,
        )?;
        self.wal_replayed_offset = new_offset;
        if replayed > 0 {
            self.mark_map_updated();
        }
        Ok(replayed)
    }

    /// Marks centralized atoms as changed after an external mutation path.
    pub fn mark_map_updated(&mut self) {
        self.map_generation = self.map_generation.wrapping_add(1);
    }

    /// Durably snapshot the shared map after live intake or learning. Errors are surfaced so a
    /// caller cannot mistake an in-memory-only update for persistent storage.
    pub fn persist_global_map(&self) -> Result<(), String> {
        let path = self.persistent_vault_path.as_ref().ok_or_else(|| {
            "global map persistence is not configured; load a persistent vault first".to_string()
        })?;
        self.swarm_map
            .save_to_binary_vault(path)
            .map(|_| ())
            .map_err(|e| format!("persist global map {}: {e}", path.display()))
    }

    /// WAL-first live-turn persistence: enforces that a persistent vault is configured, snapshots
    /// immediately when the vault file does not yet exist or is a small test map (`< 10_000` atoms),
    /// and relies on the `< 50 µs` `.atoma_map.wal` delta journal during live turns on the full
    /// active mind (`> 200k` atoms) until explicit snapshot/shutdown.
    pub fn persist_global_map_live_turn(&self) -> Result<(), String> {
        let path = self.persistent_vault_path.as_ref().ok_or_else(|| {
            "global map persistence is not configured; load a persistent vault first".to_string()
        })?;
        if !path.exists() || self.swarm_map.global_atom_count() < 10_000 {
            self.swarm_map
                .save_to_binary_vault(path)
                .map(|_| ())
                .map_err(|e| format!("persist global map {}: {e}", path.display()))?;
        }
        Ok(())
    }

    fn wal_path_for(vault_path: &std::path::Path) -> std::path::PathBuf {
        let mut name = vault_path.as_os_str().to_os_string();
        name.push(".wal");
        std::path::PathBuf::from(name)
    }

    /// Execute a full Geodesic Query against the in-memory `CarvedSwarmMap`:
    /// 1. Deterministic 6-axis icosahedral routing (`classify_icosahedral_well` or explicit `role_override`)
    ///    and refraction of `query_text` into `(H³ × T²)/Γ₁₂`.
    /// 2. Silo-isolated `top_k` `0-cell` retrieval (`SiloView::retrieve_in_silo`) with cross-epoch
    ///    `ModelAdapterLens` KV energy verification.
    /// 3. Multi-hop `1-cell` geodesic reasoning chain traversal (`DistilledReasoningArc`s) starting
    ///    from the nearest reasoning anchor in the query's hyperbolic neighborhood.
    /// 4. Tokenless `GeodesicMlirDecoder` SSA synthesis & deterministic execution (`result_q32`)
    ///    plus `2-cell` Gauss–Bonnet holonomy verification.
    /// 5. Optional live novelty-only ingestion (`learn_novel_query = true`) that persists genuinely
    ///    new query states back to `persistent_vault_path`.
    pub fn query_swarm_map(
        &mut self,
        query_text: &str,
        role_override: Option<RoleId>,
        top_k: usize,
        max_reasoning_hops: usize,
        learn_novel_query: bool,
    ) -> Result<SwarmMapQueryResult, String> {
        let k = top_k.max(1);
        // Embed `query_text` into a deterministic 256d continuous query vector
        let q_bytes = query_text.as_bytes();
        let q_digest = atoma_immune::Sha256Digest::compute(q_bytes);
        let mut hidden_state = vec![0.0f32; 256];
        for i in 0..256 {
            let b0 = q_bytes.get(i % q_bytes.len().max(1)).copied().unwrap_or(1) as f32;
            let b1 = q_digest.0[i % 32] as f32 / 255.0 - 0.5;
            let phase = (i as f32) * 0.137 + (b0 * 0.029);
            hidden_state[i] = (phase.sin() * 0.65 + b1 * 0.35).clamp(-1.0, 1.0);
        }

        // Derive low-entropy query distribution from 4 spectral sub-bands
        let mut band_e = [1e-6f32; 4];
        for b in 0..4 {
            for &v in &hidden_state[b * 64..(b + 1) * 64] {
                band_e[b] += v * v;
            }
        }
        // Sharpen query focus so well classification is high-certainty
        let max_idx = band_e
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        band_e[max_idx] *= 6.0;
        let sum_e: f32 = band_e.iter().sum();
        let next_op_probs: Vec<f32> = band_e.iter().map(|&e| e / sum_e).collect();

        // Infer depth band hint from query keywords (sensory/syntax -> early, logic/math/causal -> mid, executive/audit/action -> late)
        let lower = query_text.to_ascii_lowercase();
        let total_layers = self.swarm_map.base_weights.num_layers.max(52);
        let layer_hint = if lower.contains("exec")
            || lower.contains("govern")
            || lower.contains("audit")
            || lower.contains("motor")
            || lower.contains("cwe")
            || lower.contains("security")
        {
            (total_layers * 4) / 5
        } else if lower.contains("sensor")
            || lower.contains("parse")
            || lower.contains("token")
            || lower.contains("spatial")
        {
            total_layers / 6
        } else {
            total_layers / 2
        };

        let mut payload = vec![0u8; 256];
        for (i, &f) in hidden_state.iter().take(64).enumerate() {
            payload[i * 4..(i + 1) * 4].copy_from_slice(&f.to_le_bytes());
        }

        let tap = TeacherTapRecord {
            shard_id: u64::from_le_bytes(q_digest.0[0..8].try_into().unwrap()),
            layer_index: layer_hint,
            hidden_state: hidden_state.clone(),
            next_op_probs,
            payload,
        };

        let mut projected = self.projector.project(&tap).map_err(|e| e.to_string())?;
        let (classified_role, offset, _cert) =
            OpenWeightsDistillationEngine::classify_icosahedral_well(
                &self.swarm_map.atlas,
                &tap.hidden_state,
                layer_hint,
                total_layers,
                projected.loss_report.teacher_entropy_bits_q32,
            );
        let target_role = role_override.unwrap_or(classified_role);

        let well = self.swarm_map.atlas.well(target_role);
        let d_sph = Q32_31::from_f64(0.804_735_5);
        projected.tangent_h3 = [
            (well.unit_dir[0] * d_sph) + offset[0],
            (well.unit_dir[1] * d_sph) + offset[1],
            (well.unit_dir[2] * d_sph) + offset[2],
        ];
        let refracted = OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
        let query_lorentz = refracted.lorentz;

        // 1. Silo-isolated 0-cell retrieval (`SiloView::retrieve_in_silo`)
        let silo = self
            .swarm_map
            .silo(target_role)
            .ok_or_else(|| "Target role has no carved silo".to_string())?;
        let silo_atom_count = silo.atom_count();
        let raw_hits = silo.retrieve_in_silo(&query_lorentz, k);
        let target_fp = self.swarm_map.base_weights.fingerprint;

        let mut hits = Vec::with_capacity(raw_hits.len());
        for sa in raw_hits {
            let cosh_dist =
                OpticalLens::lorentz_inner_product_wide(&query_lorentz, &sa.global_lorentz);
            let adapted_energy = if sa.payload.len() >= 16 && sa.payload.len() % 8 == 0 {
                atoma_immune::ModelAdapterLens::adapt_kv_payload(
                    sa.id,
                    &sa.payload,
                    sa.origin_model_fingerprint,
                    target_fp,
                    sa.phase,
                    64,
                )
                .map(|rec| rec.loss_report.retained_k_energy_ratio_q32)
                .unwrap_or(Q32_31::ONE)
            } else {
                Q32_31::ONE
            };
            hits.push(RetrievedAtomHit {
                atom_id: sa.id,
                cosh_distance_q32: cosh_dist,
                well_cosh_dist_q32: sa.well_cosh_dist,
                in_boundary_overlap: sa.in_boundary_overlap,
                salience_count: sa.salience_count,
                origin_model_fingerprint: sa.origin_model_fingerprint,
                adapted_kv_retention_q32: adapted_energy,
                payload_bytes: sa.payload.len(),
            });
        }

        // 2. Find the nearest 1-cell reasoning-chain anchor atom to `query_lorentz` and traverse `max_reasoning_hops`
        let start_anchor = if max_reasoning_hops == 0 {
            None
        } else {
            hits.iter()
                .find(|h| !self.swarm_map.outgoing_arc_indices(h.atom_id).is_empty())
                .map(|h| h.atom_id)
                .or_else(|| {
                    self.swarm_map
                        .arc_bearing_atom_ids()
                        .filter_map(|raw_id| {
                            let id = AtomId(raw_id);
                            self.swarm_map.global_atom(id).map(|ga| {
                                let ip = OpticalLens::lorentz_inner_product_wide(
                                    &query_lorentz,
                                    &ga.lorentz,
                                );
                                (ip, id)
                            })
                        })
                        .min_by_key(|(ip, _)| ip.0)
                        .map(|(_, id)| id)
                })
        };

        let mut reasoning_hops = Vec::new();
        let mut visited_roles = vec![target_role];
        let mut hop_states = vec![hidden_state];

        if let Some(anchor_id) = start_anchor {
            let arcs = self
                .swarm_map
                .query_reasoning_trajectory(anchor_id, max_reasoning_hops);
            for arc in arcs {
                let mnemonic = self
                    .mlir_catalog
                    .get_op(arc.compiled_mlir_op)
                    .map(|d| d.mnemonic.clone())
                    .unwrap_or_else(|| "atoma.unknown".to_string());
                if visited_roles.last() != Some(&arc.to_role) {
                    visited_roles.push(arc.to_role);
                }
                if let Some(to_ga) = self.swarm_map.global_atom(arc.to_atom) {
                    if to_ga.payload.len() >= 64 {
                        let floats: Vec<f32> = to_ga
                            .payload
                            .chunks_exact(4)
                            .take(64)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        hop_states.push(floats);
                    }
                }
                reasoning_hops.push(ReasoningHopHit {
                    arc_id: arc.arc_id,
                    from_atom: arc.from_atom,
                    to_atom: arc.to_atom,
                    from_role: arc.from_role,
                    to_role: arc.to_role,
                    cosh_action_q32: arc.cosh_action_q32,
                    teacher_certainty_q32: arc.teacher_certainty_q32,
                    mlir_op_id: arc.compiled_mlir_op.0,
                    mlir_mnemonic: mnemonic,
                });
            }
        }

        // 3. Synthesize and deterministically execute the native MLIR SSA block for the query + reasoning trajectory
        let decode_rec = GeodesicMlirDecoder::decode_from_hidden_states(
            &self.mlir_catalog,
            target_role,
            &hop_states,
        )
        .map_err(|e| e.to_string())?;
        let ssa_result_q32 = decode_rec
            .block
            .execute_deterministic(&self.mlir_catalog, target_role)
            .map_err(|e| e.to_string())?;
        let ssa_mnemonics: Vec<String> = decode_rec
            .block
            .instructions
            .iter()
            .filter_map(|inst| {
                self.mlir_catalog
                    .get_op(inst.op_id)
                    .map(|d| d.mnemonic.clone())
            })
            .collect();

        // 4. Evaluate 2-cell Gauss-Bonnet holonomy across the visited role trajectory
        let mut closed_roles = visited_roles.clone();
        if closed_roles.len() >= 2 && closed_roles.first() != closed_roles.last() {
            closed_roles.push(closed_roles[0]);
        }
        let holonomy = self.evaluate_reasoning_holonomy(&closed_roles);

        // 5. Optional live novelty-only learning
        let live_ingest = if learn_novel_query {
            Some(self.observe_and_distill_live_step(&tap)?)
        } else {
            None
        };

        Ok(SwarmMapQueryResult {
            query_text: query_text.to_string(),
            routed_role: target_role,
            query_lorentz_x: query_lorentz.x,
            query_sector: refracted.invariant.n,
            silo_atom_count,
            global_atom_count: self.swarm_map.global_atom_count(),
            global_reasoning_arc_count: self.swarm_map.reasoning_arcs.len(),
            hits,
            reasoning_hops,
            synthesized_mlir_program: ssa_mnemonics,
            mlir_execution_result_q32: ssa_result_q32,
            holonomy_flux_rad_q32: holonomy.enclosed_curvature_flux_q32,
            live_ingest,
        })
    }
}

/// A single `0-cell` atom retrieved from a specialist's carved `SiloView` during a query.
#[derive(Debug, Clone)]
pub struct RetrievedAtomHit {
    pub atom_id: AtomId,
    pub cosh_distance_q32: Q32_31,
    pub well_cosh_dist_q32: Q32_31,
    pub in_boundary_overlap: bool,
    pub salience_count: u32,
    pub origin_model_fingerprint: atoma_immune::Sha256Digest,
    pub adapted_kv_retention_q32: Q32_31,
    pub payload_bytes: usize,
}

/// A single `1-cell` geodesic reasoning hop traversed during a query.
#[derive(Debug, Clone)]
pub struct ReasoningHopHit {
    pub arc_id: u64,
    pub from_atom: AtomId,
    pub to_atom: AtomId,
    pub from_role: RoleId,
    pub to_role: RoleId,
    pub cosh_action_q32: Q32_31,
    pub teacher_certainty_q32: Q32_31,
    pub mlir_op_id: u32,
    pub mlir_mnemonic: String,
}

/// Complete response returned by [`MonsterHybridServer::query_swarm_map`].
#[derive(Debug, Clone)]
pub struct SwarmMapQueryResult {
    pub query_text: String,
    pub routed_role: RoleId,
    pub query_lorentz_x: [Q32_31; 4],
    pub query_sector: u8,
    pub silo_atom_count: usize,
    pub global_atom_count: usize,
    pub global_reasoning_arc_count: usize,
    pub hits: Vec<RetrievedAtomHit>,
    pub reasoning_hops: Vec<ReasoningHopHit>,
    pub synthesized_mlir_program: Vec<String>,
    pub mlir_execution_result_q32: Q32_31,
    pub holonomy_flux_rad_q32: Q32_31,
    pub live_ingest: Option<LiveNoveltyDistillationReceipt>,
}
