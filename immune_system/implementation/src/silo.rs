//! Tier 4–6 (DESIGN.md): Stage 5 (`Keep` & `Carve` × 12), Structural Role Silos,
//! Ephemeral Agent Deposit Compilation, and Model-Agnostic Centralized Memory.
//!
//! Enforces Qualifications #4, #6, #7, #8, and #9 from `SPEC.md`, plus:
//! - **Decoupled Centralized Memory vs. Pluggable Model Backend**:
//!   `CarvedSwarmMap` is a model-agnostic canonical memory vault on `(H³ × T²) / Γ`.
//!   Hot-swapping the underlying model (`hot_swap_model_backend`) re-carves the 12 layer views
//!   onto the new `FrozenBaseWeights` while preserving **100% of existing distilled and compiled
//!   memory atoms** without re-distillation (`[verified]`).
//! - **Ephemeral Agent Dissolution & Deposit Compilation**:
//!   When a specialist agent finishes its task and dissolves, it hands its local map back as an
//!   `EphemeralMapDeposit`. `CarvedSwarmMap::compile_deposit` coalesces duplicate geodesic
//!   coordinates and compiles new canonical atoms into the global vault and 12 `SiloView`s (`[verified]`).

use crate::atlas13::{Atlas13, R_SPH_Q32, RoleId};
use crate::cache::geometric::{AtomId, LogicalClock, Phase};
use crate::compliance::OverlapPolicyMatrix;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::lock::{TopologicalInvariant, extract_invariant};
use crate::refraction::{OpticalLens, RefractedPoint, Sha256Digest, rad_q32_to_pi_unit};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Frozen, shared base model weights (`Qualification #8`: "Specialists are views onto one set
/// of weights. Nothing is duplicated. The base is never modified.").
#[derive(Debug)]
pub struct FrozenBaseWeights {
    pub model_name: String,
    pub num_layers: usize,
    /// Immutable flat weight buffer (simulated or mmap-backed).
    weights: Vec<f32>,
    /// Cryptographic SHA-256 fingerprint computed at freeze time.
    pub fingerprint: Sha256Digest,
}

impl FrozenBaseWeights {
    pub fn freeze(model_name: impl Into<String>, num_layers: usize, weights: Vec<f32>) -> Arc<Self> {
        let fingerprint = Self::hash_f32_slice(num_layers, &weights);
        Arc::new(Self {
            model_name: model_name.into(),
            num_layers,
            weights,
            fingerprint,
        })
    }

    fn hash_f32_slice(num_layers: usize, weights: &[f32]) -> Sha256Digest {
        let mut bytes = Vec::with_capacity(8 + weights.len() * 4);
        bytes.extend_from_slice(&(num_layers as u64).to_le_bytes());
        for &w in weights {
            let canonical = if w == 0.0 { 0.0f32 } else { w };
            bytes.extend_from_slice(&canonical.to_le_bytes());
        }
        Sha256Digest::compute(&bytes)
    }

    /// Re-verify that the in-memory weights match the original freeze fingerprint (`Qualification #8`).
    pub fn verify_integrity(&self) -> bool {
        Self::hash_f32_slice(self.num_layers, &self.weights) == self.fingerprint
    }

    pub fn raw_weights(&self) -> &[f32] {
        &self.weights
    }

    /// Carve a zero-copy layer-subset view onto the frozen weights for `role` (`dhng3.html`:
    /// "12 middle layers removed with no loss; specialists cost the memory of one").
    pub fn carve_layer_view(self: &Arc<Self>, role: RoleId) -> CarvedWeightView {
        let total = self.num_layers.max(1);
        let active_layers: Vec<usize> = match role {
            RoleId::Coordinator => (0..total).collect(),
            specialist => {
                let idx = specialist.carved_index().unwrap_or(0);
                (0..total)
                    .filter(|&l| l < 4 || l + 4 >= total || (l + idx) % 4 != 0)
                    .collect()
            }
        };
        CarvedWeightView {
            role,
            base: Arc::clone(self),
            active_layers,
        }
    }
}

/// Zero-copy layer-carved view onto `FrozenBaseWeights` (`dhng3.html` Role #2: "A carved view").
#[derive(Debug, Clone)]
pub struct CarvedWeightView {
    pub role: RoleId,
    pub base: Arc<FrozenBaseWeights>,
    pub active_layers: Vec<usize>,
}

impl CarvedWeightView {
    /// Returns true if `other` shares the exact same underlying `FrozenBaseWeights` allocation.
    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.base, &other.base)
    }
}

/// A single atom surrendered by a dissolving agent inside an [`EphemeralMapDeposit`].
#[derive(Debug, Clone)]
pub struct DepositedAtomRecord {
    pub local_id: AtomId,
    pub lorentz: LorentzVector,
    pub phase: Phase,
    pub payload: Vec<u8>,
}

/// Immutable map deposit surrendered to the `Orchestrator` when an ephemeral `RoleProcess`
/// finishes its task and dissolves.
#[derive(Debug, Clone)]
pub struct EphemeralMapDeposit {
    pub from_role: RoleId,
    pub dissolved_pid: u64,
    pub origin_model_fingerprint: Sha256Digest,
    pub forward_steps_completed: u64,
    pub final_wigner_frame: [[Q32_31; 3]; 3],
    pub records: Vec<DepositedAtomRecord>,
}

/// Receipt produced when the `Orchestrator` compiles an [`EphemeralMapDeposit`] into the
/// centralized model-agnostic `CarvedSwarmMap`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepositCompilationReceipt {
    pub from_role: RoleId,
    pub dissolved_pid: u64,
    pub submitted_atoms: usize,
    pub newly_compiled_atoms: usize,
    pub coalesced_existing_atoms: usize,
    pub compiled_global_ids: Vec<AtomId>,
    pub total_global_atoms_after: usize,
    pub model_epoch: u64,
}

/// Receipt produced when hot-swapping a new pluggable model backend into the centralized memory map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelHotSwapReceipt {
    pub previous_model_name: String,
    pub previous_fingerprint: Sha256Digest,
    pub new_model_name: String,
    pub new_fingerprint: Sha256Digest,
    pub new_model_epoch: u64,
    /// Number of canonical memory atoms preserved intact across the model swap (zero re-distillation).
    pub preserved_global_atoms: usize,
}

/// An atom stored in the undivided global map (`L0 Coordinator`).
///
/// Notice that `GlobalAtom` lives purely in the canonical `(H³ × T²) / Γ` coordinate space
/// and is completely independent of any specific model's hidden dimension or layer count.
#[derive(Debug, Clone)]
pub struct GlobalAtom {
    pub id: AtomId,
    pub clock: LogicalClock,
    pub poincare_ball: [Q32_31; 3],
    pub lorentz: LorentzVector,
    pub phase: Phase,
    pub invariant: TopologicalInvariant,
    pub payload: Arc<[u8]>,
    /// Fingerprint of the model backend that originally distilled or deposited this atom.
    pub origin_model_fingerprint: Sha256Digest,
    /// Number of times ephemeral agent deposits have reinforced/coalesced into this coordinate.
    pub salience_count: u32,
}

/// An atom projected into a specific role's carved `SiloView`.
#[derive(Debug, Clone)]
pub struct SiloAtom {
    pub id: AtomId,
    pub clock: LogicalClock,
    /// Global hyperboloid coordinate (used for boundary handoff verification).
    pub global_lorentz: LorentzVector,
    /// Coordinate Lorentz-boosted into the role's local moving frame centered at its well `W_r`.
    pub local_lorentz: LorentzVector,
    /// Toroidal phase on `T²`.
    pub phase: Phase,
    /// Hyperbolic inner product `cosh(d(x, W_r))` to this role's well center in Q32.31.
    pub well_cosh_dist: Q32_31,
    /// True if this atom sits in the boundary overlap region with at least one peer role.
    pub in_boundary_overlap: bool,
    /// Zero-copy shared payload (`Qualification #8`).
    pub payload: Arc<[u8]>,
    /// Fingerprint of the originating model backend.
    pub origin_model_fingerprint: Sha256Digest,
    /// Evidence / salience reinforcement count.
    pub salience_count: u32,
}

/// Mandatory Stage 5 loss report (`Qualification #9`: Name the Loss).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarveLossReport {
    pub global_atom_count: usize,
    pub silo_atom_counts: [usize; 12],
    pub excluded_atom_counts: [usize; 12],
    /// Fraction of global atoms visible in each silo, in Q32.31 (`[0, 1]`).
    pub retention_ratio_q32: [Q32_31; 12],
    /// Pairwise count of atoms shared in the boundary overlap `|V_r ∩ V_s|`.
    pub shared_boundary_atoms: [[usize; 12]; 12],
}

/// A role's entire readable coordinate world (`SPEC.md` §1 Layer 1 & Qualification #4).
#[derive(Debug, Clone)]
pub struct SiloView {
    pub role: RoleId,
    pub well_lorentz: LorentzVector,
    pub well_phase: Phase,
    pub base_basin_cosh_radius: Q32_31,
    pub weight_view: CarvedWeightView,
    atoms: BTreeMap<u64, SiloAtom>,
}

impl SiloView {
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    pub fn contains_atom(&self, id: AtomId) -> bool {
        self.atoms.contains_key(&id.0)
    }

    pub fn get_atom(&self, id: AtomId) -> Option<&SiloAtom> {
        self.atoms.get(&id.0)
    }

    pub fn atoms(&self) -> impl Iterator<Item = &SiloAtom> {
        self.atoms.values()
    }

    /// Retrieve the `top_k` nearest atoms **within this role's carved silo** to `query_global`.
    pub fn retrieve_in_silo(&self, query_global: &LorentzVector, top_k: usize) -> Vec<&SiloAtom> {
        if top_k == 0 || self.atoms.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(Q32_31, u64, &SiloAtom)> = self
            .atoms
            .values()
            .map(|atom| {
                let ip = OpticalLens::lorentz_inner_product_wide(query_global, &atom.global_lorentz);
                (ip, atom.id.0, atom)
            })
            .collect();
        scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().take(top_k).map(|(_, _, a)| a).collect()
    }
}

/// The Centralized Model-Agnostic 1 + 12 Carved Swarm Memory Architecture (`SPEC.md` §1–§4).
///
/// Decouples the persistent `(H³ × T²) / Γ` geometric memory (`global_atoms` + 12 `silos`)
/// from the currently plugged-in model (`base_weights` at `model_epoch`).
#[derive(Debug)]
pub struct CarvedSwarmMap {
    pub atlas: Atlas13,
    pub base_weights: Arc<FrozenBaseWeights>,
    /// Monotonic generation counter incremented whenever a new model backend is plugged in.
    pub model_epoch: u64,
    /// History of all model backends that have been plugged into this centralized memory vault.
    pub model_lineage: Vec<(u64, String, Sha256Digest)>,
    /// Default single-well attractor basin threshold `τ_r = cosh(d_max)` in Q32.31.
    pub default_basin_cosh: Q32_31,
    next_id: u64,
    clock: LogicalClock,
    /// Undivided global map ("The 1" — Coordinator L0).
    global_atoms: BTreeMap<u64, GlobalAtom>,
    /// The 12 carved specialist silos (`L1`..`L12`).
    silos: [SiloView; 12],
    /// Most recent Stage 5 loss report.
    pub last_carve_loss: CarveLossReport,
}

impl CarvedSwarmMap {
    pub fn new(
        atlas: Atlas13,
        base_weights: Arc<FrozenBaseWeights>,
        default_basin_cosh: Q32_31,
    ) -> Self {
        let silos = RoleId::CARVED_12.map(|role| {
            let well = atlas.well(role);
            SiloView {
                role,
                well_lorentz: well.lorentz_well,
                well_phase: well.phase,
                base_basin_cosh_radius: default_basin_cosh,
                weight_view: base_weights.carve_layer_view(role),
                atoms: BTreeMap::new(),
            }
        });
        let lineage = vec![(1, base_weights.model_name.clone(), base_weights.fingerprint)];

        Self {
            atlas,
            base_weights,
            model_epoch: 1,
            model_lineage: lineage,
            default_basin_cosh,
            next_id: 0,
            clock: LogicalClock(0),
            global_atoms: BTreeMap::new(),
            silos,
            last_carve_loss: CarveLossReport {
                global_atom_count: 0,
                silo_atom_counts: [0; 12],
                excluded_atom_counts: [0; 12],
                retention_ratio_q32: [Q32_31::ZERO; 12],
                shared_boundary_atoms: [[0; 12]; 12],
            },
        }
    }

    pub fn global_atom_count(&self) -> usize {
        self.global_atoms.len()
    }

    pub fn global_atom(&self, id: AtomId) -> Option<&GlobalAtom> {
        self.global_atoms.get(&id.0)
    }

    pub(crate) fn remove_global_atom_internal(&mut self, id: AtomId) -> bool {
        self.global_atoms.remove(&id.0).is_some()
    }

    pub(crate) fn collect_merkle_leaf_bytes(&self) -> Vec<Vec<u8>> {
        self.global_atoms
            .values()
            .map(|a| {
                let mut buf = Vec::with_capacity(64 + a.payload.len());
                buf.extend_from_slice(&a.id.0.to_le_bytes());
                buf.extend_from_slice(&a.clock.0.to_le_bytes());
                for c in &a.lorentz.x {
                    buf.extend_from_slice(&c.0.to_le_bytes());
                }
                buf.extend_from_slice(&a.phase.theta_f.0.to_le_bytes());
                buf.extend_from_slice(&a.phase.theta_s.0.to_le_bytes());
                buf.extend_from_slice(&a.payload);
                buf
            })
            .collect()
    }

    pub fn silo(&self, role: RoleId) -> Option<&SiloView> {
        let idx = role.carved_index()?;
        Some(&self.silos[idx])
    }

    /// Hot-swap a new pluggable model backend into the centralized memory architecture
    /// **without re-distilling or losing a single atom of the centralized `(H³ × T²) / Γ` memory**!
    pub fn hot_swap_model_backend(
        &mut self,
        new_weights: Arc<FrozenBaseWeights>,
    ) -> ModelHotSwapReceipt {
        let prev_name = self.base_weights.model_name.clone();
        let prev_fp = self.base_weights.fingerprint;
        let new_name = new_weights.model_name.clone();
        let new_fp = new_weights.fingerprint;

        self.model_epoch += 1;
        self.model_lineage
            .push((self.model_epoch, new_name.clone(), new_fp));

        for r_idx in 0..12 {
            let role = RoleId::CARVED_12[r_idx];
            self.silos[r_idx].weight_view = new_weights.carve_layer_view(role);
        }
        self.base_weights = new_weights;

        ModelHotSwapReceipt {
            previous_model_name: prev_name,
            previous_fingerprint: prev_fp,
            new_model_name: new_name,
            new_fingerprint: new_fp,
            new_model_epoch: self.model_epoch,
            preserved_global_atoms: self.global_atoms.len(),
        }
    }

    /// Insert a refracted point (`Stage 4` output) into the undivided global map (`L0`)
    /// and carve it into whichever of the 12 `SiloView`s are permitted by `policy` (`Stage 5 × 12`).
    pub fn keep_and_carve(
        &mut self,
        refracted: RefractedPoint,
        payload: Vec<u8>,
        policy: &OverlapPolicyMatrix,
    ) -> AtomId {
        let id = AtomId(self.next_id);
        self.next_id += 1;
        self.clock.0 += 1;

        let shared_payload: Arc<[u8]> = Arc::from(payload.into_boxed_slice());
        let atom = GlobalAtom {
            id,
            clock: self.clock,
            poincare_ball: refracted.poincare_ball,
            lorentz: refracted.lorentz,
            phase: refracted.phase,
            invariant: refracted.invariant,
            payload: shared_payload,
            origin_model_fingerprint: self.base_weights.fingerprint,
            salience_count: 1,
        };

        self.global_atoms.insert(id.0, atom);
        self.recarve_all_twelve(policy);
        id
    }

    /// Compile an [`EphemeralMapDeposit`] surrendered by a dissolved agent into the centralized
    /// `(H³ × T²) / Γ` memory architecture:
    /// 1. Steers each local worker coordinate into the dissolving role's well basin on `H³`
    ///    so specialist discoveries are anchored in the role's canonical manifold region.
    /// 2. Performs **Geodesic Coalescence**: if an existing canonical `GlobalAtom` lies within
    ///    hyperbolic inner product `cosh(d) <= 1.0005` and matching phase sector, reinforces
    ///    its `salience_count` instead of duplicating memory.
    /// 3. Compiles novel coordinates as new `GlobalAtom`s and runs `recarve_all_twelve(policy)`.
    pub fn compile_deposit(
        &mut self,
        deposit: &EphemeralMapDeposit,
        policy: &OverlapPolicyMatrix,
    ) -> DepositCompilationReceipt {
        let well = self.atlas.well(deposit.from_role);
        let coalesce_cosh_threshold = Q32_31::from_f64(1.0005);

        let mut newly_compiled = 0usize;
        let mut coalesced = 0usize;
        let mut compiled_ids = Vec::with_capacity(deposit.records.len());

        for rec in &deposit.records {
            // Map the worker's local H³ coordinate into the role's canonical well neighborhood
            // via inverse well boost so it lands inside `from_role`'s basin on the global map.
            let global_lorentz = if deposit.from_role == RoleId::Coordinator {
                rec.lorentz
            } else {
                // Blend local perturbation around well center inside the Poincare ball
                let local_ball = OpticalLens::lorentz_to_ball(&rec.lorentz);
                let well_ball = well.poincare_ball;
                let blended_ball = [
                    well_ball[0] + Q32_31(local_ball[0].0 >> 3),
                    well_ball[1] + Q32_31(local_ball[1].0 >> 3),
                    well_ball[2] + Q32_31(local_ball[2].0 >> 3),
                ];
                OpticalLens::ball_to_lorentz(&blended_ball).unwrap_or(well.lorentz_well)
            };

            let poincare_ball = OpticalLens::lorentz_to_ball(&global_lorentz);
            let tf_pi = rad_q32_to_pi_unit(rec.phase.theta_f);
            let ts_pi = rad_q32_to_pi_unit(rec.phase.theta_s);
            let invariant = extract_invariant(tf_pi, ts_pi);

            // Check for geodesic coalescence with an existing GlobalAtom
            let mut matched_existing_id: Option<AtomId> = None;
            for existing in self.global_atoms.values_mut() {
                let ip = OpticalLens::lorentz_inner_product_wide(&global_lorentz, &existing.lorentz);
                if ip <= coalesce_cosh_threshold && existing.invariant.n == invariant.n {
                    existing.salience_count = existing.salience_count.saturating_add(1);
                    matched_existing_id = Some(existing.id);
                    break;
                }
            }

            if let Some(existing_id) = matched_existing_id {
                coalesced += 1;
                compiled_ids.push(existing_id);
            } else {
                let id = AtomId(self.next_id);
                self.next_id += 1;
                self.clock.0 += 1;
                let shared_payload: Arc<[u8]> =
                    Arc::from(rec.payload.clone().into_boxed_slice());
                let atom = GlobalAtom {
                    id,
                    clock: self.clock,
                    poincare_ball,
                    lorentz: global_lorentz,
                    phase: rec.phase,
                    invariant,
                    payload: shared_payload,
                    origin_model_fingerprint: deposit.origin_model_fingerprint,
                    salience_count: 1,
                };
                self.global_atoms.insert(id.0, atom);
                newly_compiled += 1;
                compiled_ids.push(id);
            }
        }

        if newly_compiled > 0 || coalesced > 0 {
            self.recarve_all_twelve(policy);
        }

        DepositCompilationReceipt {
            from_role: deposit.from_role,
            dissolved_pid: deposit.dissolved_pid,
            submitted_atoms: deposit.records.len(),
            newly_compiled_atoms: newly_compiled,
            coalesced_existing_atoms: coalesced,
            compiled_global_ids: compiled_ids,
            total_global_atoms_after: self.global_atoms.len(),
            model_epoch: self.model_epoch,
        }
    }

    /// Run Stage 5 (`Carve`) twelve times — one refracted view per specialist phase (`SPEC.md` §3),
    /// governed strictly by the Layer 3 `OverlapPolicyMatrix`, and update `last_carve_loss`.
    pub fn recarve_all_twelve(&mut self, policy: &OverlapPolicyMatrix) -> CarveLossReport {
        for silo in &mut self.silos {
            silo.atoms.clear();
        }

        for atom in self.global_atoms.values() {
            let mut well_ips = [Q32_31::ZERO; 12];
            for r_idx in 0..12 {
                let role = RoleId::CARVED_12[r_idx];
                let well = &self.atlas.well(role).lorentz_well;
                well_ips[r_idx] = OpticalLens::lorentz_inner_product_wide(&atom.lorentz, well);
            }

            for r_idx in 0..12 {
                let role_r = RoleId::CARVED_12[r_idx];
                let in_home_basin = well_ips[r_idx] <= self.default_basin_cosh;

                let mut in_overlap = false;
                for s_idx in 0..12 {
                    if r_idx == s_idx {
                        continue;
                    }
                    let role_s = RoleId::CARVED_12[s_idx];
                    let omega = policy.overlap_allowance(role_r, role_s);
                    if omega.0 > 0 {
                        let limit_r = Q32_31(self.default_basin_cosh.0.saturating_add(omega.0));
                        let limit_s = Q32_31(self.default_basin_cosh.0.saturating_add(omega.0));
                        if well_ips[r_idx] <= limit_r && well_ips[s_idx] <= limit_s {
                            in_overlap = true;
                            break;
                        }
                    }
                }

                if in_home_basin || in_overlap {
                    let local_lorentz = self.boost_to_well_frame(role_r, &atom.lorentz);
                    let silo_atom = SiloAtom {
                        id: atom.id,
                        clock: atom.clock,
                        global_lorentz: atom.lorentz,
                        local_lorentz,
                        phase: atom.phase,
                        well_cosh_dist: well_ips[r_idx],
                        in_boundary_overlap: in_overlap,
                        payload: Arc::clone(&atom.payload),
                        origin_model_fingerprint: atom.origin_model_fingerprint,
                        salience_count: atom.salience_count,
                    };
                    self.silos[r_idx].atoms.insert(atom.id.0, silo_atom);
                }
            }
        }

        let total = self.global_atoms.len();
        let mut silo_atom_counts = [0usize; 12];
        let mut excluded_atom_counts = [0usize; 12];
        let mut retention_ratio_q32 = [Q32_31::ZERO; 12];
        let mut shared_boundary_atoms = [[0usize; 12]; 12];

        for r_idx in 0..12 {
            let count = self.silos[r_idx].atoms.len();
            silo_atom_counts[r_idx] = count;
            excluded_atom_counts[r_idx] = total.saturating_sub(count);
            retention_ratio_q32[r_idx] = if total == 0 {
                Q32_31::ZERO
            } else {
                let num = (count as i128) << Q32_31::FRACTIONAL_BITS;
                Q32_31((num / (total as i128)) as i64)
            };
        }

        for r_idx in 0..12 {
            for s_idx in 0..12 {
                if r_idx == s_idx {
                    shared_boundary_atoms[r_idx][s_idx] = silo_atom_counts[r_idx];
                } else {
                    let shared = self.silos[r_idx]
                        .atoms
                        .keys()
                        .filter(|k| self.silos[s_idx].atoms.contains_key(k))
                        .count();
                    shared_boundary_atoms[r_idx][s_idx] = shared;
                }
            }
        }

        let report = CarveLossReport {
            global_atom_count: total,
            silo_atom_counts,
            excluded_atom_counts,
            retention_ratio_q32,
            shared_boundary_atoms,
        };
        self.last_carve_loss = report.clone();
        report
    }

    fn boost_to_well_frame(&self, role: RoleId, global: &LorentzVector) -> LorentzVector {
        let well = self.atlas.well(role);
        let neg_dir = [
            Q32_31(-well.unit_dir[0].0),
            Q32_31(-well.unit_dir[1].0),
            Q32_31(-well.unit_dir[2].0),
        ];
        let r = R_SPH_Q32.0 as i128;
        let one = 1i128 << Q32_31::FRACTIONAL_BITS;
        let r_sq = (r * r) >> Q32_31::FRACTIONAL_BITS;
        let beta = (((2 * r) << Q32_31::FRACTIONAL_BITS) / (one + r_sq)) as i64;
        LorentzVector::boost(global, &neg_dir, Q32_31(beta))
    }
}
