//! `atoma_orchestrator`: The Unified Three-Layer Monster Hybrid Server.
//!
//! Ties together the complete chain defined in `SPEC.md` §4:
//! `Distill → Refract → Carve → Silo → Agent → Dissolve & Deposit → Govern → Immune-check`
//! with a **Model-Agnostic Centralized `(H³ × T²) / Γ` Memory Architecture** that survives
//! across pluggable model hot-swaps without re-distillation.

pub mod bridge;
pub mod routing;
pub mod swarm;

use atoma_immune::cache::geometric::AtomId;
use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    AtomicIrAlphabet, Atlas13, CarveLossReport, CarvedSwarmMap, CompliancePermit,
    ComplianceViolation, DepositCompilationReceipt, EphemeralMapDeposit,
    FirstPrinciplesComplianceEngine, FirstPrinciplesComplianceReport, FrozenBaseWeights,
    FrozenShard, HolonomyLoopReceipt, HyperbolicHolonomyEngine, IDENTITY_FRAME_Q32, ImmuneSystem,
    MetaCognitiveRunbookController, ModelHotSwapReceipt, OpticalLens, OverlapPolicyMatrix,
    ProjectedTangent, RefractedPoint, RoleId, SurgicalErasureReceipt, SwarmKnowledgeAndHomeostasis,
    TangentProjector, TeacherTapRecord,
};
use routing::{DeliveredHandoff, HandoffRequest};
use std::sync::Arc;
use swarm::{GovernedForwardReceipt, Governor, RestartIntensity, RestartStrategy};

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
    /// Pinned Atomic IR lossless alphabet (`SPEC_atomic_ir.md`).
    pub atomic_ir: AtomicIrAlphabet,
    /// 12-Sphere Superpowers ontology (CVE/CWE/ATT&CK) & 8-discipline homeostatic mapping.
    pub knowledge_homeostasis: SwarmKnowledgeAndHomeostasis,
    /// Non-Abelian Levi-Civita Hyperbolic Holonomy & Gauss–Bonnet Berry Phase Engine.
    pub holonomy_engine: HyperbolicHolonomyEngine,
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

        Ok(Self {
            immune,
            governor,
            swarm_map,
            projector: TangentProjector::default(),
            meta_controller: MetaCognitiveRunbookController::default(),
            atomic_ir: AtomicIrAlphabet::new_canonical(248_320),
            knowledge_homeostasis: SwarmKnowledgeAndHomeostasis::new(),
            holonomy_engine: HyperbolicHolonomyEngine::default(),
        })
    }

    /// Execute `Distill (Tap + Hash) → Refract (Project + ExpMap) → Carve (×12) → Silo → Immune-check`
    /// anchored around `target_role`'s well direction on `H³ × T²`.
    pub fn ingest_teacher_tap(
        &mut self,
        tap: &TeacherTapRecord,
        target_role: RoleId,
        radial_offset_tangent: [Q32_31; 3],
    ) -> Result<DistillRefractCarveReceipt, String> {
        let shard = FrozenShard::from_tap(tap).map_err(|e| e.to_string())?;
        let mut projected = self.projector.project(tap).map_err(|e| e.to_string())?;

        let well = self.swarm_map.atlas.well(target_role);
        let d_sph = Q32_31::from_f64(0.804_735_5);
        projected.tangent_h3 = [
            (well.unit_dir[0] * d_sph) + radial_offset_tangent[0],
            (well.unit_dir[1] * d_sph) + radial_offset_tangent[1],
            (well.unit_dir[2] * d_sph) + radial_offset_tangent[2],
        ];

        let refracted = OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
        let atom_id = self.swarm_map.keep_and_carve(
            refracted,
            tap.payload.clone(),
            &self.immune.policy,
        );
        let carve_loss = self.swarm_map.last_carve_loss.clone();

        let loss_permit = self
            .immune
            .verify_loss_accounting(&projected.loss_report, &refracted.refraction_loss, &carve_loss)
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
    ) -> Result<(EphemeralMapDeposit, DepositCompilationReceipt, CompliancePermit), ComplianceViolation> {
        self.governor
            .dissolve_and_compile_deposit(role, &mut self.swarm_map, &mut self.immune)
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
        Ok((receipt, permit))
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
    pub fn surgical_erase(&mut self, atom_id: AtomId) -> Result<SurgicalErasureReceipt, &'static str> {
        self.swarm_map
            .surgical_erase_atom(atom_id, &self.immune.policy)
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
}
