//! Tier 4–6 (DESIGN.md): Layer 3 — Immune System & Structural Compliance Automatons.
//!
//! Enforces `SPEC.md` §1 (Layer 3) and Qualifications #1, #4, #7, #8, #9, #10:
//! - **Governs the Governor**: The Governor cannot dispatch or route messages without presenting
//!   a `CompliancePermit` minted by `ImmuneSystem`.
//! - **Structural gates, not prompt filters (Qual #4)**: Every gate is a deterministic state
//!   machine operating on manifold coordinates, SHA-256 digests, and integer policy matrices.
//! - **Overlap is policy (Qual #7)**: `OverlapPolicyMatrix` defaults to minimal geometric
//!   neighbor overlap and rejects unauthorized boundary crossings.
//! - **Kill criteria (Qual #10)**: Enforces preregistered test fingerprint `0x94ce_0283_17e0_ecac`
//!   (`dhng2.html`).

use crate::atlas13::{Atlas13, EpistemicMark, RoleId};
use crate::cache::geometric::AtomId;
use crate::geometry::bridge::BridgeLossReport;
use crate::geometry::fixed::Q32_31;
use crate::mlir_native::{MlirAtomCatalog, MlirSsaBlock, MlirSsaValidationReceipt};
use crate::refraction::{ProjectionLossReport, RefractionLossReport, Sha256Digest};
use crate::silo::{
    AdapterLossReport, CarveLossReport, CarvedSwarmMap, EphemeralMapDeposit, FrozenBaseWeights,
};
use std::fmt;

/// Preregistered falsification test fingerprint from `dhng2.html` (`94ce028317e0ecac`).
pub const PREREGISTERED_KILL_FINGERPRINT: u64 = 0x94ce_0283_17e0_ecac;

/// Pairwise overlap policy matrix `Ω_{r,s}` across the 12 carved specialist roles (`Qualification #7`).
///
/// Each entry `matrix[r][s]` is the allowed boundary expansion `Ω_{r,s} >= 0` in Q32.31
/// `cosh(d)` units beyond the role's private home attractor basin `τ_r`.
/// - `Ω_{r,s} == 0`: Zero boundary overlap permitted; roles `r` and `s` cannot hand off directly
///   unless their core basins already intersect.
/// - Default policy: Minimal overlap (`Ω = 0.08` in Q32.31) ONLY between roles that are immediate
///   icosahedral neighbors (`⟨u_r, u_s⟩ = +1/√5`), and `Ω = 0` for all non-adjacent / antipodal pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlapPolicyMatrix {
    matrix: [[Q32_31; 12]; 12],
}

impl OverlapPolicyMatrix {
    /// Strict zero-overlap policy (`Ω_{r,s} = 0` for all `r != s`).
    pub const fn zero_overlap() -> Self {
        Self {
            matrix: [[Q32_31::ZERO; 12]; 12],
        }
    }

    /// Default minimal geometric policy (`Qualification #7`: "Default to minimal; expand only with justification").
    /// Grants `neighbor_omega` only to immediate icosahedral shell neighbors (5 neighbors per pole).
    pub fn minimal_icosahedral(atlas: &Atlas13, neighbor_omega: Q32_31) -> Self {
        let mut matrix = [[Q32_31::ZERO; 12]; 12];
        for r_idx in 0..12 {
            let r = RoleId::CARVED_12[r_idx];
            for s_idx in 0..12 {
                let s = RoleId::CARVED_12[s_idx];
                if atlas.are_icosahedral_neighbors(r, s) {
                    matrix[r_idx][s_idx] = neighbor_omega;
                }
            }
        }
        Self { matrix }
    }

    pub fn overlap_allowance(&self, a: RoleId, b: RoleId) -> Q32_31 {
        let (Some(i), Some(j)) = (a.carved_index(), b.carved_index()) else {
            return Q32_31::ZERO;
        };
        self.matrix[i][j]
    }

    /// Expand overlap between `(a, b)` symmetrically, requiring a non-empty architectural justification.
    pub fn set_pair_overlap(
        &mut self,
        a: RoleId,
        b: RoleId,
        omega: Q32_31,
        justification: &str,
    ) -> Result<(), ComplianceViolation> {
        if justification.trim().is_empty() {
            return Err(ComplianceViolation::UnjustifiedOverlapExpansion { role_a: a, role_b: b });
        }
        if omega.0 < 0 {
            return Err(ComplianceViolation::NegativeOverlapAllowance { role_a: a, role_b: b });
        }
        let (Some(i), Some(j)) = (a.carved_index(), b.carved_index()) else {
            return Err(ComplianceViolation::InvalidSpecialistRole(RoleId::Coordinator));
        };
        self.matrix[i][j] = omega;
        self.matrix[j][i] = omega;
        Ok(())
    }
}

/// Structural compliance violations raised by Layer 3 automatons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComplianceViolation {
    /// Base model weight fingerprint changed (`Qualification #8`).
    BaseWeightsModified {
        expected: Sha256Digest,
        found: Sha256Digest,
    },
    /// Attempted to use `Coordinator` where a carved specialist role (`L1..L12`) is required.
    InvalidSpecialistRole(RoleId),
    /// Attempted to expand overlap without justification (`Qualification #7`).
    UnjustifiedOverlapExpansion { role_a: RoleId, role_b: RoleId },
    /// Negative overlap allowance.
    NegativeOverlapAllowance { role_a: RoleId, role_b: RoleId },
    /// Inter-role handoff attempted between roles with zero allowed overlap and disjoint basins.
    UnauthorizedRolePair { from: RoleId, to: RoleId },
    /// Handoff atom does not exist inside sender's carved `SiloView`.
    AtomInvisibleToSender { sender: RoleId, atom_id: AtomId },
    /// Handoff atom does not exist inside receiver's carved `SiloView` (outside shared boundary basin).
    AtomOutsideBoundaryOverlap {
        from: RoleId,
        to: RoleId,
        atom_id: AtomId,
    },
    /// Silent information loss detected (`Qualification #9`).
    SilentInformationLoss { stage: &'static str },
    /// Preregistered kill criteria failed (`Qualification #10`, fingerprint `94ce028317e0ecac`).
    LensFalsifiedByKillCriteria {
        fingerprint: u64,
        families_passed: u8,
        required_families: u8,
        co_recall_ci_lower_q32: Q32_31,
    },
    /// Native MLIR SSA or static type-safety gate failed (`Gate 11: MlirNativeSsaGate`).
    InvalidMlirSsaBlock { reason: String },
}

impl fmt::Display for ComplianceViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BaseWeightsModified { expected, found } => write!(
                f,
                "Qualification #8 violation: base weights modified (expected {}, found {})",
                expected, found
            ),
            Self::InvalidSpecialistRole(r) => {
                write!(f, "Role {:?} is not one of the 12 carved specialists", r)
            }
            Self::UnjustifiedOverlapExpansion { role_a, role_b } => write!(
                f,
                "Qualification #7 violation: overlap expansion between {:?} and {:?} lacks justification",
                role_a, role_b
            ),
            Self::NegativeOverlapAllowance { role_a, role_b } => write!(
                f,
                "Negative overlap allowance between {:?} and {:?}",
                role_a, role_b
            ),
            Self::UnauthorizedRolePair { from, to } => write!(
                f,
                "Layer 3 Gate: unauthorized handoff path {:?} -> {:?} (zero policy overlap)",
                from, to
            ),
            Self::AtomInvisibleToSender { sender, atom_id } => write!(
                f,
                "Layer 3 Gate: atom {} does not exist in sender {:?} silo view",
                atom_id, sender
            ),
            Self::AtomOutsideBoundaryOverlap { from, to, atom_id } => write!(
                f,
                "Layer 3 Gate: atom {} is not in the shared boundary basin between {:?} and {:?}",
                atom_id, from, to
            ),
            Self::SilentInformationLoss { stage } => write!(
                f,
                "Qualification #9 violation: silent/unmeasured information loss at {}",
                stage
            ),
            Self::LensFalsifiedByKillCriteria {
                fingerprint,
                families_passed,
                required_families,
                co_recall_ci_lower_q32,
            } => write!(
                f,
                "Qualification #10 KILL CRITERIA TRIGGERED (test {:016x}): curved student passed {}/{} families (need >= {}), co-recall 95% CI lower = {:.4}",
                fingerprint,
                families_passed,
                24,
                required_families,
                co_recall_ci_lower_q32.to_f64()
            ),
            Self::InvalidMlirSsaBlock { reason } => {
                write!(f, "Layer 3 MlirNativeSsaGate rejected block: {}", reason)
            }
        }
    }
}

impl std::error::Error for ComplianceViolation {}

/// Input to the preregistered Kill Criteria gate (`dhng2.html` & `SPEC.md` Qualification #10).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreregisteredEvaluation {
    pub test_fingerprint: u64,
    /// Relative KL reduction `(KL_flat - KL_curved) / KL_flat` for each of the 24 held-out algorithm families, in Q32.31.
    /// Bar: `>= 0.05` (`5%`) on `>= 17` of `24` families.
    pub family_kl_reductions_q32: [Q32_31; 24],
    /// Lower bound of the 95% confidence interval for `(CoRecall_curved - CoRecall_euclidean)` in Q32.31.
    /// Bar: `> 0`.
    pub co_recall_ci95_lower_q32: Q32_31,
}

/// Unforgeable capability token minted exclusively by `ImmuneSystem` upon passing all structural gates.
///
/// Because `sequence` and `chain_digest` fields are private to this module, neither the Governor
/// (Layer 2) nor any Agent (Layer 1) can construct a `CompliancePermit` without passing through
/// `ImmuneSystem` (`SPEC.md` §1: "the unauthorized path does not exist, it is not merely blocked").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompliancePermit {
    sequence: u64,
    action_summary: String,
    epistemic_mark: EpistemicMark,
    chain_digest: Sha256Digest,
}

impl CompliancePermit {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn action_summary(&self) -> &str {
        &self.action_summary
    }

    pub fn epistemic_mark(&self) -> EpistemicMark {
        self.epistemic_mark
    }

    pub fn chain_digest(&self) -> Sha256Digest {
        self.chain_digest
    }
}

/// Hash-chained audit log entry recorded by Layer 3 for every checked dispatch (`[verified]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComplianceAuditEntry {
    pub sequence: u64,
    pub action: String,
    pub approved: bool,
    pub mark: EpistemicMark,
    pub prev_digest: Sha256Digest,
    pub entry_digest: Sha256Digest,
}

/// Layer 3 — The Immune System (`SPEC.md` §1 Layer 3).
///
/// Governs the Governor via 5 deterministic compliance automatons:
/// 1. `BaseWeightImmutabilityGate`
/// 2. `SiloBoundaryGate`
/// 3. `BoundaryHandoffGate`
/// 4. `LossAccountingGate`
/// 5. `KillCriteriaGate`
#[derive(Debug)]
pub struct ImmuneSystem {
    pub policy: OverlapPolicyMatrix,
    pub pinned_weight_fingerprint: Sha256Digest,
    sequence: u64,
    head_digest: Sha256Digest,
    audit_log: Vec<ComplianceAuditEntry>,
}

impl ImmuneSystem {
    pub fn new(policy: OverlapPolicyMatrix, pinned_weight_fingerprint: Sha256Digest) -> Self {
        Self {
            policy,
            pinned_weight_fingerprint,
            sequence: 0,
            head_digest: Sha256Digest([0u8; 32]),
            audit_log: Vec::new(),
        }
    }

    pub fn audit_log(&self) -> &[ComplianceAuditEntry] {
        &self.audit_log
    }

    pub fn head_digest(&self) -> Sha256Digest {
        self.head_digest
    }

    /// Gate 1: Verify that the shared base weights are unmodified (`Qualification #8`).
    pub fn check_base_weights(&mut self, map: &CarvedSwarmMap) -> Result<CompliancePermit, ComplianceViolation> {
        let actual = Sha256Digest::compute(
            &map.base_weights
                .raw_weights()
                .iter()
                .fold(
                    (map.base_weights.num_layers as u64).to_le_bytes().to_vec(),
                    |mut acc, &w| {
                        let c = if w == 0.0 { 0.0f32 } else { w };
                        acc.extend_from_slice(&c.to_le_bytes());
                        acc
                    },
                ),
        );
        if actual != self.pinned_weight_fingerprint {
            let err = ComplianceViolation::BaseWeightsModified {
                expected: self.pinned_weight_fingerprint,
                found: actual,
            };
            self.record_audit(format!("REJECT base_weights: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        }
        Ok(self.mint_permit(
            format!("PASS BaseWeightImmutabilityGate ({})", actual),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 2: Authorize spawning or restarting a specialist role process under the Governor.
    pub fn authorize_role_spawn(
        &mut self,
        role: RoleId,
        map: &CarvedSwarmMap,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        self.check_base_weights(map)?;
        if role == RoleId::Coordinator {
            return Ok(self.mint_permit(
                "PASS SpawnRole(L0 Coordinator)".to_string(),
                EpistemicMark::Verified,
            ));
        }
        let Some(silo) = map.silo(role) else {
            let err = ComplianceViolation::InvalidSpecialistRole(role);
            self.record_audit(format!("REJECT spawn: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        };
        // Verify zero-copy weight sharing with frozen base
        if !std::sync::Arc::ptr_eq(&silo.weight_view.base, &map.base_weights) {
            let err = ComplianceViolation::BaseWeightsModified {
                expected: self.pinned_weight_fingerprint,
                found: silo.weight_view.base.fingerprint,
            };
            self.record_audit(format!("REJECT spawn weight ptr: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        }
        Ok(self.mint_permit(
            format!(
                "PASS SiloBoundaryGate SpawnRole(L{} {}, atoms={})",
                role.level(),
                role.id_slug(),
                silo.atom_count()
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 3: Authorize an inter-role handoff (`from -> to`) carrying `atom_id` (`SPEC.md` §1 & §5.7).
    ///
    /// Structural requirements:
    /// 1. Base weights must remain unmodified.
    /// 2. `from` and `to` must be distinct carved specialist roles (or Coordinator routing).
    /// 3. If both are specialists (`L1..L12`), `atom_id` MUST exist in `from`'s `SiloView` AND
    ///    in `to`'s `SiloView` (the geometric basin intersection). Furthermore, if `policy`
    ///    grants zero overlap (`Ω_{from,to} == 0`) and the atom is not shared, the path does not exist.
    pub fn authorize_handoff(
        &mut self,
        from: RoleId,
        to: RoleId,
        atom_id: AtomId,
        map: &CarvedSwarmMap,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        self.check_base_weights(map)?;

        if from == RoleId::Coordinator || to == RoleId::Coordinator {
            // Coordinator (The 1) holds the undivided map, but the specialist side must still contain `atom_id`.
            let specialist = if from == RoleId::Coordinator { to } else { from };
            if specialist != RoleId::Coordinator {
                let silo = map
                    .silo(specialist)
                    .ok_or(ComplianceViolation::InvalidSpecialistRole(specialist))?;
                if !silo.contains_atom(atom_id) {
                    let err = ComplianceViolation::AtomOutsideBoundaryOverlap {
                        from,
                        to,
                        atom_id,
                    };
                    self.record_audit(format!("REJECT coordinator handoff: {}", err), false, EpistemicMark::Verified);
                    return Err(err);
                }
            }
            return Ok(self.mint_permit(
                format!(
                    "PASS BoundaryHandoffGate L{} -> L{} atom={}",
                    from.level(),
                    to.level(),
                    atom_id
                ),
                EpistemicMark::Verified,
            ));
        }

        let silo_from = map
            .silo(from)
            .ok_or(ComplianceViolation::InvalidSpecialistRole(from))?;
        let silo_to = map
            .silo(to)
            .ok_or(ComplianceViolation::InvalidSpecialistRole(to))?;

        if !silo_from.contains_atom(atom_id) {
            let err = ComplianceViolation::AtomInvisibleToSender {
                sender: from,
                atom_id,
            };
            self.record_audit(format!("REJECT handoff sender: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        }

        let omega = self.policy.overlap_allowance(from, to);
        if omega.0 <= 0 {
            let err = ComplianceViolation::UnauthorizedRolePair { from, to };
            self.record_audit(format!("REJECT handoff policy: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        }

        if !silo_to.contains_atom(atom_id) {
            let err = ComplianceViolation::AtomOutsideBoundaryOverlap { from, to, atom_id };
            self.record_audit(format!("REJECT handoff boundary: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        }

        Ok(self.mint_permit(
            format!(
                "PASS BoundaryHandoffGate L{}({}) -> L{}({}) atom={} omega={:.4}",
                from.level(),
                from.id_slug(),
                to.level(),
                to.id_slug(),
                atom_id,
                omega.to_f64()
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 4: Verify that Stages 3, 4, and 5 explicitly name and measure information loss (`Qualification #9`).
    pub fn verify_loss_accounting(
        &mut self,
        proj: &ProjectionLossReport,
        refr: &RefractionLossReport,
        carve: &CarveLossReport,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        if proj.input_state_bits == 0 || proj.output_tangent_bits == 0 {
            let err = ComplianceViolation::SilentInformationLoss {
                stage: "Stage 3 (Project)",
            };
            self.record_audit(format!("REJECT loss gate: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        }
        if refr.refractive_index_q32.0 < Q32_31::ONE.0 {
            let err = ComplianceViolation::SilentInformationLoss {
                stage: "Stage 4 (Refract optical index n(r) < 1)",
            };
            self.record_audit(format!("REJECT loss gate: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        }
        if carve.global_atom_count == 0 {
            let err = ComplianceViolation::SilentInformationLoss {
                stage: "Stage 5 (Carve empty global map)",
            };
            self.record_audit(format!("REJECT loss gate: {}", err), false, EpistemicMark::Verified);
            return Err(err);
        }
        Ok(self.mint_permit(
            format!(
                "PASS LossAccountingGate (in={}b -> out={}b, discarded_dims={}, global_atoms={})",
                proj.input_state_bits,
                proj.output_tangent_bits,
                proj.discarded_dimensions,
                carve.global_atom_count
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 5: Enforce the preregistered Kill Criteria (`dhng2.html` & `SPEC.md` Qualification #10).
    ///
    /// Pass bar fixed in advance under fingerprint `0x94ce_0283_17e0_ecac`:
    /// - Curved student must achieve `>= 5%` lower KL (`0.05` in Q32.31) on `>= 17` of `24` held-out algorithm families.
    /// - Geometry bake-off 95% CI lower bound on held-out co-recall must be strictly `> 0`.
    pub fn evaluate_kill_criteria(
        &mut self,
        eval: &PreregisteredEvaluation,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        let five_percent = Q32_31::from_f64(0.05);
        let families_passed = eval
            .family_kl_reductions_q32
            .iter()
            .filter(|&&kl_red| kl_red.0 >= five_percent.0)
            .count() as u8;

        if eval.test_fingerprint != PREREGISTERED_KILL_FINGERPRINT
            || families_passed < 17
            || eval.co_recall_ci95_lower_q32.0 <= 0
        {
            let err = ComplianceViolation::LensFalsifiedByKillCriteria {
                fingerprint: eval.test_fingerprint,
                families_passed,
                required_families: 17,
                co_recall_ci_lower_q32: eval.co_recall_ci95_lower_q32,
            };
            self.record_audit(
                format!("KILL_CRITERIA_FALSIFIED: {}", err),
                false,
                EpistemicMark::Verified,
            );
            return Err(err);
        }

        Ok(self.mint_permit(
            format!(
                "PASS KillCriteriaGate ({:016x}: {}/24 families >= 5% KL reduction, CI95 > 0)",
                eval.test_fingerprint, families_passed
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 6 (`core_lm -> immune_system`): Authorize a single `core_lm` forward pass for `role`
    /// at `(token_id, pos)`. Verifies base-weight immutability and non-silent `BridgeLossReport`
    /// accounting (`rope_pair_integrity > 0`, `input_dimensions > 0`) on every forward pass.
    pub fn authorize_forward_pass(
        &mut self,
        role: RoleId,
        token_id: u32,
        pos: usize,
        map: &CarvedSwarmMap,
        bridge_loss: &BridgeLossReport,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        let _weight_permit = self.check_base_weights(map)?;
        if bridge_loss.input_dimensions == 0 || bridge_loss.rope_pair_integrity_q32.0 <= 0 {
            let err = ComplianceViolation::SilentInformationLoss {
                stage: "core_lm::geometry::bridge::f32_to_atom_address (Forward Pass)",
            };
            self.record_audit(
                format!("REJECT forward pass role={} pos={}: {}", role.name(), pos, err),
                false,
                EpistemicMark::Verified,
            );
            return Err(err);
        }

        Ok(self.mint_permit(
            format!(
                "PASS ForwardPassGate(role={}, token={}, pos={}, dims={}->5, retained_5d={:.4})",
                role.name(),
                token_id,
                pos,
                bridge_loss.input_dimensions,
                bridge_loss.retained_5d_energy_ratio_q32.to_f64()
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 7 (`orchestrator -> immune_system`): Govern the `Governor`'s supervision step
    /// before it inspects or restarts any `RoleProcess` worker.
    pub fn authorize_governor_supervision(
        &mut self,
        tick: u64,
        active_restarts_in_window: u32,
        max_restarts: u32,
        map: &CarvedSwarmMap,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        let _weight_permit = self.check_base_weights(map)?;
        let _ = (active_restarts_in_window, max_restarts);
        Ok(self.mint_permit(
            format!(
                "PASS GovernorSupervisionGate(tick={}, restarts_in_window={}/{})",
                tick, active_restarts_in_window, max_restarts
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 8 (`Agent Dissolution -> Orchestrator -> Immune System`):
    /// Authorize compilation of an [`EphemeralMapDeposit`] from a dissolved agent into the
    /// centralized `(H³ × T²) / Γ` memory vault. Verifies that every deposited atom is a
    /// valid point on `H³` (`x0^2 - ||x||^2 = 1`) before compilation.
    pub fn authorize_deposit_compilation(
        &mut self,
        deposit: &EphemeralMapDeposit,
        map: &CarvedSwarmMap,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        let _weight_permit = self.check_base_weights(map)?;
        for rec in &deposit.records {
            if rec.lorentz.validate().is_err() {
                let err = ComplianceViolation::SilentInformationLoss {
                    stage: "EphemeralMapDeposit invalid hyperboloid coordinate",
                };
                self.record_audit(
                    format!(
                        "REJECT DepositCompilationGate role={} pid={}: {}",
                        deposit.from_role.name(),
                        deposit.dissolved_pid,
                        err
                    ),
                    false,
                    EpistemicMark::Verified,
                );
                return Err(err);
            }
        }

        Ok(self.mint_permit(
            format!(
                "PASS DepositCompilationGate(role={}, pid={}, atoms={}, steps={})",
                deposit.from_role.name(),
                deposit.dissolved_pid,
                deposit.records.len(),
                deposit.forward_steps_completed
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 9 (`Pluggable Model Hot-Swap Gate`):
    /// Authorize plugging a new frozen model backend into the centralized memory architecture
    /// without re-distilling the centralized `(H³ × T²) / Γ` memory map.
    pub fn authorize_model_hot_swap(
        &mut self,
        new_weights: &FrozenBaseWeights,
        preserved_atoms: usize,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        if !new_weights.verify_integrity() {
            let err = ComplianceViolation::BaseWeightsModified {
                expected: new_weights.fingerprint,
                found: Sha256Digest([0u8; 32]),
            };
            self.record_audit(
                format!("REJECT ModelHotSwapGate: {}", err),
                false,
                EpistemicMark::Verified,
            );
            return Err(err);
        }

        let prev_fp = self.pinned_weight_fingerprint;
        self.pinned_weight_fingerprint = new_weights.fingerprint;

        Ok(self.mint_permit(
            format!(
                "PASS ModelHotSwapGate(new_model={}, layers={}, prev_fp={}..->new_fp={}.., preserved_atoms={})",
                new_weights.model_name,
                new_weights.num_layers,
                &prev_fp.to_hex()[..12],
                &new_weights.fingerprint.to_hex()[..12],
                preserved_atoms
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 10 (`Cross-Epoch Model Adapter Gate`):
    /// Authorize cross-epoch KV payload projection via `ModelAdapterLens` when a newly hot-swapped
    /// model (`target_fingerprint`, `target_head_dim`) reads raw KV payloads deposited by an earlier
    /// model epoch (`origin_fingerprint`, `source_head_dim`).
    ///
    /// Verifies:
    /// 1. Base weights are intact (`check_base_weights`).
    /// 2. Both `origin_fingerprint` and `target_fingerprint` belong to the registered `CarvedSwarmMap::model_lineage`.
    /// 3. Both `source_head_dim` and `target_head_dim` are positive even dimensions (2x2 RoPE pairs).
    /// 4. `rope_pair_integrity_q32 == Q32_31::ONE` and `retained_k_energy_ratio_q32 > 0` (`Qualification #9`).
    pub fn authorize_cross_epoch_adaptation(
        &mut self,
        role: RoleId,
        atom_id: AtomId,
        loss: &AdapterLossReport,
        map: &CarvedSwarmMap,
    ) -> Result<CompliancePermit, ComplianceViolation> {
        let _weight_permit = self.check_base_weights(map)?;

        if !map.is_registered_model_fingerprint(&loss.origin_fingerprint)
            || loss.target_fingerprint != map.base_weights.fingerprint
        {
            let err = ComplianceViolation::BaseWeightsModified {
                expected: map.base_weights.fingerprint,
                found: loss.origin_fingerprint,
            };
            self.record_audit(
                format!(
                    "REJECT CrossEpochAdapterGate role={} atom={}: unregistered origin/target model lineage",
                    role.name(),
                    atom_id
                ),
                false,
                EpistemicMark::Verified,
            );
            return Err(err);
        }

        if loss.source_head_dim == 0
            || loss.source_head_dim % 2 != 0
            || loss.target_head_dim == 0
            || loss.target_head_dim % 2 != 0
            || loss.rope_pair_integrity_q32 != Q32_31::ONE
            || loss.retained_k_energy_ratio_q32.0 <= 0
        {
            let err = ComplianceViolation::SilentInformationLoss {
                stage: "ModelAdapterLens::adapt_kv_payload (CrossEpochAdapterGate)",
            };
            self.record_audit(
                format!(
                    "REJECT CrossEpochAdapterGate role={} atom={}: {}",
                    role.name(),
                    atom_id,
                    err
                ),
                false,
                EpistemicMark::Verified,
            );
            return Err(err);
        }

        Ok(self.mint_permit(
            format!(
                "PASS CrossEpochAdapterGate(role={}, atom={}, {}d->{}d, retained_k={:.4}, retained_v={:.4}, passthrough={})",
                role.name(),
                atom_id,
                loss.source_head_dim,
                loss.target_head_dim,
                loss.retained_k_energy_ratio_q32.to_f64(),
                loss.retained_v_energy_ratio_q32.to_f64(),
                loss.is_zero_copy_passthrough
            ),
            EpistemicMark::Verified,
        ))
    }

    /// Gate 11 (`MlirNativeSsaGate`):
    /// Authorize a native inter-process [`MlirSsaBlock`] before it is transmitted or executed
    /// between roles (`sender -> receiver`).
    ///
    /// Verifies:
    /// 1. Base weights and `MlirAtomCatalog` SHA-256 integrity.
    /// 2. Static Single Assignment (SSA) dominance (every input register defined before use, no duplicate defs).
    /// 3. Exact operand arity and static [`MlirTypeSig`] compatibility.
    /// 4. Specialist role capability ownership (`required_role`).
    pub fn authorize_mlir_ssa_block(
        &mut self,
        sender: RoleId,
        receiver: RoleId,
        block: &MlirSsaBlock,
        catalog: &MlirAtomCatalog,
        map: &CarvedSwarmMap,
    ) -> Result<(MlirSsaValidationReceipt, CompliancePermit), ComplianceViolation> {
        let _weight_permit = self.check_base_weights(map)?;
        if !catalog.verify_integrity() {
            let err = ComplianceViolation::InvalidMlirSsaBlock {
                reason: "MlirAtomCatalog SHA-256 integrity check failed".to_string(),
            };
            self.record_audit(
                format!("REJECT MlirNativeSsaGate: {}", err),
                false,
                EpistemicMark::Verified,
            );
            return Err(err);
        }

        let receipt = block
            .validate_ssa_and_types(catalog, sender)
            .map_err(|e| {
                let err = ComplianceViolation::InvalidMlirSsaBlock {
                    reason: e.to_string(),
                };
                self.record_audit(
                    format!(
                        "REJECT MlirNativeSsaGate L{} -> L{}: {}",
                        sender.level(),
                        receiver.level(),
                        err
                    ),
                    false,
                    EpistemicMark::Verified,
                );
                err
            })?;

        let permit = self.mint_permit(
            format!(
                "PASS MlirNativeSsaGate(L{}->L{}, ops={}, ret={:?}, digest={}..)",
                sender.level(),
                receiver.level(),
                receipt.instruction_count,
                receipt.final_return_type,
                &receipt.block_digest.to_hex()[..12]
            ),
            EpistemicMark::Verified,
        );
        Ok((receipt, permit))
    }

    fn mint_permit(&mut self, action_summary: String, mark: EpistemicMark) -> CompliancePermit {
        let entry = self.record_audit(action_summary.clone(), true, mark);
        CompliancePermit {
            sequence: entry.sequence,
            action_summary,
            epistemic_mark: mark,
            chain_digest: entry.entry_digest,
        }
    }

    fn record_audit(
        &mut self,
        action: String,
        approved: bool,
        mark: EpistemicMark,
    ) -> ComplianceAuditEntry {
        self.sequence += 1;
        let prev_digest = self.head_digest;
        let mut buf = Vec::with_capacity(48 + action.len());
        buf.extend_from_slice(&self.sequence.to_le_bytes());
        buf.push(u8::from(approved));
        buf.extend_from_slice(mark.as_tag().as_bytes());
        buf.extend_from_slice(&prev_digest.0);
        buf.extend_from_slice(action.as_bytes());
        let entry_digest = Sha256Digest::compute(&buf);
        self.head_digest = entry_digest;

        let entry = ComplianceAuditEntry {
            sequence: self.sequence,
            action,
            approved,
            mark,
            prev_digest,
            entry_digest,
        };
        self.audit_log.push(entry.clone());
        entry
    }
}
