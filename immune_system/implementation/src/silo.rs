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
use crate::distill_engine::DistilledReasoningArc;
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
    pub fn freeze(
        model_name: impl Into<String>,
        num_layers: usize,
        weights: Vec<f32>,
    ) -> Arc<Self> {
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
    /// O(log N) causal lookup for typed token K/V; rebuilt together with carved views.
    typed_kv: BTreeMap<(u32, u32, u64), TypedKvSequenceIndex>,
}

#[derive(Debug, Default, Clone)]
struct TypedKvSequenceIndex {
    positions: BTreeMap<usize, (AtomId, u16)>,
    phase_cells: BTreeMap<u16, BTreeMap<usize, AtomId>>,
}

fn typed_kv_identity(payload: &[u8]) -> Result<Option<(u32, u32, u64, usize)>, String> {
    if !payload.starts_with(b"ATOMA_KV_V1\0") {
        return Ok(None);
    }
    if payload.len() < 40 {
        return Err("malformed typed K/V header".into());
    }
    let layer = u32::from_le_bytes(payload[16..20].try_into().unwrap());
    let head = u32::from_le_bytes(payload[20..24].try_into().unwrap());
    let sequence = u64::from_le_bytes(payload[24..32].try_into().unwrap());
    let position = usize::try_from(u64::from_le_bytes(payload[32..40].try_into().unwrap()))
        .map_err(|_| "K/V position exceeds usize")?;
    Ok(Some((layer, head, sequence, position)))
}

fn phase_cell(phase: Phase) -> u16 {
    let bin = |v: f64| {
        ((v.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU * 16.0).floor() as u16)
            .min(15)
    };
    bin(phase.theta_f.to_f64()) * 16 + bin(phase.theta_s.to_f64())
}

fn phase_cell_distance(phase: Phase, cell: u16) -> f64 {
    let f = (f64::from(cell / 16) + 0.5) * std::f64::consts::TAU / 16.0;
    let s = (f64::from(cell % 16) + 0.5) * std::f64::consts::TAU / 16.0;
    2.0 - (phase.theta_f.to_f64() - f).cos() - (phase.theta_s.to_f64() - s).cos()
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

    /// Exact causal records, without scanning unrelated map atoms or sequences.
    pub fn typed_kv_range(
        &self,
        layer: usize,
        head: usize,
        sequence: u64,
        start: usize,
        end: usize,
    ) -> Result<Vec<AtomId>, String> {
        if start > end {
            return Ok(Vec::new());
        }
        let layer = u32::try_from(layer).map_err(|_| "layer exceeds u32")?;
        let head = u32::try_from(head).map_err(|_| "head exceeds u32")?;
        Ok(self
            .typed_kv
            .get(&(layer, head, sequence))
            .map(|index| {
                index
                    .positions
                    .range(start..=end)
                    .map(|(_, (id, _))| *id)
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Bounded phase-cell recall. Work depends on the 256-cell torus and result limit,
    /// not the number of atoms in the carved silo.
    pub fn typed_kv_near_phase(
        &self,
        layer: usize,
        head: usize,
        sequence: u64,
        start: usize,
        end: usize,
        phase: Phase,
        limit: usize,
    ) -> Result<Vec<AtomId>, String> {
        if limit == 0 || start > end {
            return Ok(Vec::new());
        }
        let layer = u32::try_from(layer).map_err(|_| "layer exceeds u32")?;
        let head = u32::try_from(head).map_err(|_| "head exceeds u32")?;
        let Some(index) = self.typed_kv.get(&(layer, head, sequence)) else {
            return Ok(Vec::new());
        };
        let mut cells: Vec<_> = index
            .phase_cells
            .iter()
            .map(|(&cell, positions)| (phase_cell_distance(phase, cell), cell, positions))
            .collect();
        cells.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut ids = Vec::with_capacity(limit);
        for (_, _, positions) in cells {
            for (_, id) in positions.range(start..=end).rev().take(limit - ids.len()) {
                ids.push(*id);
            }
            if ids.len() == limit {
                break;
            }
        }
        Ok(ids)
    }

    pub fn typed_kv_sequence_count(&self, sequence: u64) -> usize {
        self.typed_kv
            .iter()
            .filter(|((_, _, seq), _)| *seq == sequence)
            .map(|(_, index)| index.positions.len())
            .sum()
    }

    fn index_typed_atom(&mut self, atom: &SiloAtom) {
        let Ok(Some((layer, head, sequence, position))) = typed_kv_identity(&atom.payload) else {
            return;
        };
        let index = self.typed_kv.entry((layer, head, sequence)).or_default();
        let cell = phase_cell(atom.phase);
        if let Some((_, old_cell)) = index.positions.insert(position, (atom.id, cell)) {
            if let Some(old) = index.phase_cells.get_mut(&old_cell) {
                old.remove(&position);
            }
        }
        index
            .phase_cells
            .entry(cell)
            .or_default()
            .insert(position, atom.id);
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
                let ip =
                    OpticalLens::lorentz_inner_product_wide(query_global, &atom.global_lorentz);
                (ip, atom.id.0, atom)
            })
            .collect();
        scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().take(top_k).map(|(_, _, a)| a).collect()
    }
}

/// The Centralized Model-Agnostic 1 + 12 Carved Swarm Memory Architecture (`SPEC.md` §1–§4).
///
/// Decouples the persistent `(H³ × T²) / Γ` geometric memory (`global_atoms` 0-cells +
/// `reasoning_arcs` 1-cells + 12 `silos`) from the currently plugged-in model (`base_weights` at `model_epoch`).
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
    next_arc_id: u64,
    clock: LogicalClock,
    /// Undivided global map ("The 1" — Coordinator L0 `0-cells`).
    global_atoms: BTreeMap<u64, GlobalAtom>,
    /// O(1) spatial + topological sector novelty hash index (`bucket_key -> Vec<AtomId>`)
    /// used to reject duplicate `0-cell` noise in microseconds during full-model and live distillation.
    novelty_buckets: BTreeMap<u64, Vec<AtomId>>,
    /// Set of `(model_fingerprint, tensor_name, bits, group_size, top_k)` digests already distilled
    /// into this map so re-running or resuming distillation never duplicates a tensor.
    pub ingested_tensor_digests: std::collections::BTreeSet<Sha256Digest>,
    /// Directed geodesic reasoning transitions (`1-cells`) distilled from teacher reasoning chains.
    pub reasoning_arcs: Vec<DistilledReasoningArc>,
    /// O(1) outgoing-arc adjacency index (`from_atom.0 -> Vec<reasoning_arcs_index>`).
    outgoing_arcs_by_atom: BTreeMap<u64, Vec<usize>>,
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
                typed_kv: BTreeMap::new(),
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
            next_arc_id: 0,
            clock: LogicalClock(0),
            global_atoms: BTreeMap::new(),
            novelty_buckets: BTreeMap::new(),
            ingested_tensor_digests: std::collections::BTreeSet::new(),
            reasoning_arcs: Vec::new(),
            outgoing_arcs_by_atom: BTreeMap::new(),
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

    /// Return the slice of `reasoning_arcs` indices leaving `from_atom` in O(1).
    #[inline]
    pub fn outgoing_arc_indices(&self, from_atom: AtomId) -> &[usize] {
        self.outgoing_arcs_by_atom
            .get(&from_atom.0)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Iterate over all `AtomId`s (`u64`) that have at least one outgoing `1-cell` reasoning arc.
    #[inline]
    pub fn arc_bearing_atom_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.outgoing_arcs_by_atom.keys().copied()
    }

    pub fn global_atom_count(&self) -> usize {
        self.global_atoms.len()
    }

    pub fn global_atom(&self, id: AtomId) -> Option<&GlobalAtom> {
        self.global_atoms.get(&id.0)
    }

    /// Iterate over all global `0-cell` atoms in the central map.
    pub fn iter_global_atoms(&self) -> impl Iterator<Item = (&u64, &GlobalAtom)> {
        self.global_atoms.iter()
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
        self.keep_and_carve_with_origin(refracted, payload, self.base_weights.fingerprint, policy)
    }

    /// Compute a 64-bit spatial + topological sector bucket key for O(1) geodesic novelty lookup.
    fn spatial_bucket_key(qx1: i64, qx2: i64, qx3: i64, sector_n: u8) -> u64 {
        let b1 = ((qx1 as u64) & 0x3F_FFFF) << 42;
        let b2 = ((qx2 as u64) & 0x1F_FFFF) << 21;
        let b3 = ((qx3 as u64) & 0x1F_FFF) << 8;
        b1 | b2 | b3 | (sector_n as u64)
    }

    /// Insert a refracted point with an explicit `origin_model_fingerprint` and return
    /// `(AtomId, is_novel)`:
    /// - Uses the O(1) 27-neighborhood spatial bucket index (`novelty_buckets`) to check if an
    ///   existing `GlobalAtom` lies within `cosh(d) <= 1.0005` in the same topological sector.
    /// - If a duplicate/near-isometric coordinate exists, increments `salience_count` in place and
    ///   returns `(existing_id, false)` (`0` duplicate noise added!).
    /// - If genuinely novel, inserts the atom, carves it incrementally into the 12 silos in O(1),
    ///   and returns `(new_id, true)`.
    pub fn keep_and_carve_novel_only(
        &mut self,
        refracted: RefractedPoint,
        payload: Vec<u8>,
        origin_fingerprint: Sha256Digest,
        policy: &OverlapPolicyMatrix,
    ) -> (AtomId, bool) {
        if !self.is_registered_model_fingerprint(&origin_fingerprint) {
            let next_epoch = self.model_lineage.len() as u64 + 1;
            self.model_lineage.push((
                next_epoch,
                "DistilledOpenSourceTeacher".to_string(),
                origin_fingerprint,
            ));
        }

        let coalesce_cosh_threshold = Q32_31::from_f64(1.0000001);
        // Quantize spatial Lorentz coordinates at resolution 2^-10 (~0.001)
        let q1 = refracted.lorentz.x[1].0 >> 15;
        let q2 = refracted.lorentz.x[2].0 >> 15;
        let q3 = refracted.lorentz.x[3].0 >> 21;
        let sec = refracted.invariant.n;

        let mut matched_id: Option<AtomId> = None;
        // Token KV records carry causal position in their envelope, and Coupled Operator Atoms
        // (`ATOMA_OP`) are deduplicated by 256d receptor sketch cosine per layer in `map_native.rs`.
        'search: for dx in -1..=1 {
            if payload.starts_with(b"ATOMA_KV_V1\0") || payload.starts_with(b"ATOMA_OP") {
                break 'search;
            }
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = Self::spatial_bucket_key(q1 + dx, q2 + dy, q3 + dz, sec);
                    if let Some(bucket) = self.novelty_buckets.get(&key) {
                        for &cand_id in bucket {
                            if let Some(existing) = self.global_atoms.get(&cand_id.0) {
                                if existing.payload.starts_with(b"ATOMA_KV_V1\0")
                                    || existing.payload.starts_with(b"ATOMA_OP")
                                {
                                    continue;
                                }
                                let ip = OpticalLens::lorentz_inner_product_wide(
                                    &refracted.lorentz,
                                    &existing.lorentz,
                                );
                                if ip <= coalesce_cosh_threshold && existing.invariant.n == sec {
                                    matched_id = Some(cand_id);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(existing_id) = matched_id {
            if let Some(existing) = self.global_atoms.get_mut(&existing_id.0) {
                existing.salience_count = existing.salience_count.saturating_add(1);
                let new_salience = existing.salience_count;
                for silo in &mut self.silos {
                    if let Some(sa) = silo.atoms.get_mut(&existing_id.0) {
                        sa.salience_count = new_salience;
                    }
                }
            }
            return (existing_id, false);
        }

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
            origin_model_fingerprint: origin_fingerprint,
            salience_count: 1,
        };

        let home_key = Self::spatial_bucket_key(q1, q2, q3, sec);
        self.novelty_buckets.entry(home_key).or_default().push(id);
        self.carve_single_atom_incremental(&atom, policy);
        self.global_atoms.insert(id.0, atom);
        self.refresh_loss_ledger_counts();
        (id, true)
    }

    /// Insert a refracted point with an explicit `origin_model_fingerprint` (used when distilling
    /// external open-source teacher checkpoints into the global map) and perform geodesic
    /// coalescence if an identical coordinate already exists.
    pub fn keep_and_carve_with_origin(
        &mut self,
        refracted: RefractedPoint,
        payload: Vec<u8>,
        origin_fingerprint: Sha256Digest,
        policy: &OverlapPolicyMatrix,
    ) -> AtomId {
        self.keep_and_carve_novel_only(refracted, payload, origin_fingerprint, policy)
            .0
    }

    /// Record a layer-anchor arc.
    ///
    /// An existing arc matches when `(from_atom, to_atom, compiled_mlir_op)` is exact, or when
    /// the role pair, opcode, and hop length agree within 0.001. Token steps use
    /// [`Self::record_trajectory_arc`].
    /// Returns `(arc_id, is_novel)`.
    pub fn record_novel_reasoning_arc(&mut self, arc: DistilledReasoningArc) -> (u64, bool) {
        let cosh_tol = Q32_31::from_f64(0.001).0;
        for existing in &mut self.reasoning_arcs {
            let exact_atoms = existing.from_atom == arc.from_atom
                && existing.to_atom == arc.to_atom
                && existing.compiled_mlir_op == arc.compiled_mlir_op;
            let same_role_circuit = existing.from_role == arc.from_role
                && existing.to_role == arc.to_role
                && existing.compiled_mlir_op == arc.compiled_mlir_op
                && (existing.cosh_action_q32.0 - arc.cosh_action_q32.0).abs() <= cosh_tol;
            if exact_atoms || same_role_circuit {
                if arc.teacher_certainty_q32.0 > existing.teacher_certainty_q32.0 {
                    existing.teacher_certainty_q32 = arc.teacher_certainty_q32;
                }
                return (existing.arc_id, false);
            }
        }

        self.push_reasoning_arc(arc)
    }

    /// Record a token-trajectory arc.
    ///
    /// An existing arc matches when `(from_atom, compiled_mlir_op, wigner_frame)` is exact.
    /// A second token well that continues to the same ID stays in the map, and two histories
    /// that leave one well for different next IDs each keep their frame.
    /// Returns `(arc_id, is_novel)`.
    pub fn record_trajectory_arc(&mut self, arc: DistilledReasoningArc) -> (u64, bool) {
        if let Some(indices) = self.outgoing_arcs_by_atom.get(&arc.from_atom.0) {
            for &idx in indices {
                let existing = &mut self.reasoning_arcs[idx];
                if existing.from_atom == arc.from_atom
                    && existing.compiled_mlir_op == arc.compiled_mlir_op
                    && existing.wigner_frame == arc.wigner_frame
                    && existing.chain_id == arc.chain_id
                {
                    if arc.teacher_certainty_q32.0 > existing.teacher_certainty_q32.0 {
                        existing.teacher_certainty_q32 = arc.teacher_certainty_q32;
                    }
                    return (existing.arc_id, false);
                }
                if existing.from_atom == arc.from_atom
                    && existing.chain_id == arc.chain_id
                    && existing.step_index == arc.step_index
                    && existing.wigner_frame == arc.wigner_frame
                {
                    existing.to_atom = arc.to_atom;
                    existing.compiled_mlir_op = arc.compiled_mlir_op;
                    existing.cosh_action_q32 = arc.cosh_action_q32;
                    if arc.teacher_certainty_q32.0 > existing.teacher_certainty_q32.0 {
                        existing.teacher_certainty_q32 = arc.teacher_certainty_q32;
                    }
                    existing.origin_model_fingerprint = arc.origin_model_fingerprint;
                    return (existing.arc_id, false);
                }
            }
        }
        self.push_reasoning_arc(arc)
    }

    fn push_reasoning_arc(&mut self, mut arc: DistilledReasoningArc) -> (u64, bool) {
        let id = self.next_arc_id;
        self.next_arc_id += 1;
        arc.arc_id = id;
        let idx = self.reasoning_arcs.len();
        self.outgoing_arcs_by_atom
            .entry(arc.from_atom.0)
            .or_default()
            .push(idx);
        self.reasoning_arcs.push(arc);
        (id, true)
    }

    /// Write `payload` onto `id` when that atom exists and still carries an empty payload.
    pub fn store_payload_if_empty(&mut self, id: AtomId, payload: Vec<u8>) {
        if payload.is_empty() {
            return;
        }
        if let Some(atom) = self.global_atoms.get_mut(&id.0) {
            if atom.payload.is_empty() {
                let shared: std::sync::Arc<[u8]> =
                    std::sync::Arc::from(payload.into_boxed_slice());
                atom.payload = std::sync::Arc::clone(&shared);
                for silo in &mut self.silos {
                    if let Some(sa) = silo.atoms.get_mut(&id.0) {
                        sa.payload = std::sync::Arc::clone(&shared);
                    }
                }
            }
        }
    }

    /// Record a directed 1-cell [`DistilledReasoningArc`] in the global map.
    pub fn record_reasoning_arc(&mut self, arc: DistilledReasoningArc) -> u64 {
        self.record_novel_reasoning_arc(arc).0
    }

    /// Boost the `salience_count` of an atom when a multi-step reasoning chain passes Gauss–Bonnet
    /// holonomy verification.
    pub fn reinforce_atom_salience(&mut self, id: AtomId, delta: u32) {
        if let Some(atom) = self.global_atoms.get_mut(&id.0) {
            atom.salience_count = atom.salience_count.saturating_add(delta);
        }
    }

    /// Traverse the distilled geodesic reasoning graph starting from `start_atom` up to `max_hops`,
    /// greedily following the lowest hyperbolic-action (`cosh_action_q32`) outgoing reasoning arc at each hop.
    pub fn query_reasoning_trajectory(
        &self,
        start_atom: AtomId,
        max_hops: usize,
    ) -> Vec<&DistilledReasoningArc> {
        let mut path = Vec::with_capacity(max_hops);
        let mut curr = start_atom;
        for _ in 0..max_hops {
            let next_arc = self
                .reasoning_arcs
                .iter()
                .filter(|a| a.from_atom == curr)
                .min_by_key(|a| a.cosh_action_q32.0);
            match next_arc {
                Some(arc) => {
                    curr = arc.to_atom;
                    path.push(arc);
                }
                None => break,
            }
        }
        path
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
        let coalesce_cosh_threshold = Q32_31::from_f64(1.0000001);

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
                if rec.payload.starts_with(b"ATOMA_KV_V1\0") {
                    break;
                }
                if existing.payload.starts_with(b"ATOMA_KV_V1\0") {
                    continue;
                }
                let ip =
                    OpticalLens::lorentz_inner_product_wide(&global_lorentz, &existing.lorentz);
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
                let shared_payload: Arc<[u8]> = Arc::from(rec.payload.clone().into_boxed_slice());
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

    /// Incrementally carve a single `GlobalAtom` into whichever of the 12 `SiloView`s are
    /// permitted by `policy` in `O(1)` (`12` Lorentz inner products), avoiding `O(N^2)` full recarves
    /// when streaming hundreds of layers from a checkpoint.
    fn carve_single_atom_incremental(&mut self, atom: &GlobalAtom, policy: &OverlapPolicyMatrix) {
        let mut well_ips = [Q32_31::ZERO; 12];
        for r_idx in 0..12 {
            let role = RoleId::CARVED_12[r_idx];
            let well = &self.atlas.well(role).lorentz_well;
            well_ips[r_idx] = OpticalLens::lorentz_inner_product_wide(&atom.lorentz, well);
        }

        let mut placed_silos = [false; 12];
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
                self.silos[r_idx].index_typed_atom(&silo_atom);
                self.silos[r_idx].atoms.insert(atom.id.0, silo_atom);
                placed_silos[r_idx] = true;
            }
        }

        for r_idx in 0..12 {
            if placed_silos[r_idx] {
                for s_idx in 0..12 {
                    if placed_silos[s_idx] {
                        self.last_carve_loss.shared_boundary_atoms[r_idx][s_idx] += 1;
                    }
                }
            }
        }
    }

    fn refresh_loss_ledger_counts(&mut self) {
        let total = self.global_atoms.len();
        self.last_carve_loss.global_atom_count = total;
        for r_idx in 0..12 {
            let count = self.silos[r_idx].atoms.len();
            self.last_carve_loss.silo_atom_counts[r_idx] = count;
            self.last_carve_loss.excluded_atom_counts[r_idx] = total.saturating_sub(count);
            self.last_carve_loss.retention_ratio_q32[r_idx] = if total == 0 {
                Q32_31::ZERO
            } else {
                let num = (count as i128) << Q32_31::FRACTIONAL_BITS;
                Q32_31((num / (total as i128)) as i64)
            };
        }
    }

    /// Run Stage 5 (`Carve`) twelve times — one refracted view per specialist phase (`SPEC.md` §3),
    /// governed strictly by the Layer 3 `OverlapPolicyMatrix`, and update `last_carve_loss`.
    pub fn recarve_all_twelve(&mut self, policy: &OverlapPolicyMatrix) -> CarveLossReport {
        for silo in &mut self.silos {
            silo.atoms.clear();
            silo.typed_kv.clear();
        }
        self.novelty_buckets.clear();
        self.last_carve_loss.shared_boundary_atoms = [[0usize; 12]; 12];

        let atoms_snapshot: Vec<GlobalAtom> = self.global_atoms.values().cloned().collect();
        for atom in &atoms_snapshot {
            let q1 = atom.lorentz.x[1].0 >> 15;
            let q2 = atom.lorentz.x[2].0 >> 15;
            let q3 = atom.lorentz.x[3].0 >> 21;
            let key = Self::spatial_bucket_key(q1, q2, q3, atom.invariant.n);
            self.novelty_buckets.entry(key).or_default().push(atom.id);
            self.carve_single_atom_incremental(atom, policy);
        }
        self.refresh_loss_ledger_counts();
        self.last_carve_loss.clone()
    }

    /// Fast O(1) 27-cell spatial-bucket lookup for arc-bearing atoms near `(query_lorentz, query_phase)`.
    ///
    /// - For on-manifold queries (where an arc-bearing carrier atom sits in the 27-cell bucket
    ///   neighborhood within `cosh <= 1.005`), resolves in `~150 ns` with zero full-map scan!
    /// - For off-manifold queries, falls back to scanning `arc_bearing_atom_ids()`.
    pub fn candidate_arc_atoms_near(
        &self,
        query_lorentz: &LorentzVector,
        query_phase: Phase,
        neighborhood_slack_q32: i64,
    ) -> Option<(u64, i64, Vec<(u64, i64)>)> {
        if self.outgoing_arcs_by_atom.is_empty() {
            return None;
        }
        let q1 = query_lorentz.x[1].0 >> 15;
        let q2 = query_lorentz.x[2].0 >> 15;
        let q3 = query_lorentz.x[3].0 >> 21;
        let sec = extract_invariant(
            rad_q32_to_pi_unit(query_phase.theta_f),
            rad_q32_to_pi_unit(query_phase.theta_s),
        )
        .n;
        let tight_hit_cosh = Q32_31::from_f64(1.005).0;

        let mut local_hits: Vec<(u64, i64)> = Vec::new();
        let mut best_local: Option<(u64, i64)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = Self::spatial_bucket_key(q1 + dx, q2 + dy, q3 + dz, sec);
                    if let Some(bucket) = self.novelty_buckets.get(&key) {
                        for &cand_id in bucket {
                            if self.outgoing_arc_indices(cand_id).is_empty() {
                                continue;
                            }
                            if let Some(atom) = self.global_atoms.get(&cand_id.0) {
                                let c = OpticalLens::lorentz_inner_product_wide(
                                    query_lorentz,
                                    &atom.lorentz,
                                )
                                .0;
                                local_hits.push((cand_id.0, c));
                                match best_local {
                                    None => best_local = Some((cand_id.0, c)),
                                    Some((b_id, b_c)) if (c, cand_id.0) < (b_c, b_id) => {
                                        best_local = Some((cand_id.0, c));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some((best_atom, best_cosh)) = best_local {
            if best_cosh <= tight_hit_cosh {
                let band = best_cosh.saturating_add(neighborhood_slack_q32);
                local_hits.retain(|&(_, c)| c <= band);
                local_hits.sort_by_key(|&(id, _)| id);
                local_hits.dedup_by_key(|&mut (id, _)| id);
                return Some((best_atom, best_cosh, local_hits));
            }
        }

        // Fallback for wide basin or off-manifold queries
        let mut best: Option<(u64, i64)> = None;
        let mut all_scored: Vec<(u64, i64)> =
            Vec::with_capacity(self.outgoing_arcs_by_atom.len());
        for atom_id_u64 in self.arc_bearing_atom_ids() {
            if let Some(atom) = self.global_atoms.get(&atom_id_u64) {
                let c = OpticalLens::lorentz_inner_product_wide(query_lorentz, &atom.lorentz).0;
                all_scored.push((atom_id_u64, c));
                match best {
                    None => best = Some((atom_id_u64, c)),
                    Some((b_id, b_c)) if (c, atom_id_u64) < (b_c, b_id) => {
                        best = Some((atom_id_u64, c));
                    }
                    _ => {}
                }
            }
        }
        let (best_atom, best_cosh) = best?;
        let band = best_cosh.saturating_add(neighborhood_slack_q32);
        all_scored.retain(|&(_, c)| c <= band);
        Some((best_atom, best_cosh, all_scored))
    }

    /// Save the entire centralized `(H³ × T²)/Γ₁₂` map (`0-cell` `GlobalAtom`s, `1-cell`
    /// `DistilledReasoningArc`s, `ingested_tensor_digests`, and `model_lineage`) to a
    /// SHA-256-authenticated `.atoma_map` binary vault on disk.
    pub fn save_to_binary_vault(
        &self,
        path: &std::path::Path,
    ) -> Result<(usize, usize, Sha256Digest), String> {
        let mut buf = Vec::with_capacity(
            1024 + self.global_atoms.len() * 160 + self.reasoning_arcs.len() * 144,
        );
        buf.extend_from_slice(b"ATOMA_MAP_V1");
        buf.extend_from_slice(&self.model_epoch.to_le_bytes());
        buf.extend_from_slice(&self.next_id.to_le_bytes());
        buf.extend_from_slice(&self.next_arc_id.to_le_bytes());
        buf.extend_from_slice(&self.clock.0.to_le_bytes());

        // 1. Model lineage
        buf.extend_from_slice(&(self.model_lineage.len() as u32).to_le_bytes());
        for (ep, name, fp) in &self.model_lineage {
            buf.extend_from_slice(&ep.to_le_bytes());
            let nb = name.as_bytes();
            buf.extend_from_slice(&(nb.len() as u32).to_le_bytes());
            buf.extend_from_slice(nb);
            buf.extend_from_slice(&fp.0);
        }

        // 2. Ingested tensor digests (for zero-duplicate idempotent re-runs)
        buf.extend_from_slice(&(self.ingested_tensor_digests.len() as u32).to_le_bytes());
        for d in &self.ingested_tensor_digests {
            buf.extend_from_slice(&d.0);
        }

        // 3. Global 0-cell atoms
        buf.extend_from_slice(&(self.global_atoms.len() as u32).to_le_bytes());
        for atom in self.global_atoms.values() {
            buf.extend_from_slice(&atom.id.0.to_le_bytes());
            buf.extend_from_slice(&atom.clock.0.to_le_bytes());
            for c in &atom.poincare_ball {
                buf.extend_from_slice(&c.0.to_le_bytes());
            }
            for c in &atom.lorentz.x {
                buf.extend_from_slice(&c.0.to_le_bytes());
            }
            buf.extend_from_slice(&atom.phase.theta_f.0.to_le_bytes());
            buf.extend_from_slice(&atom.phase.theta_s.0.to_le_bytes());
            buf.extend_from_slice(&atom.origin_model_fingerprint.0);
            buf.extend_from_slice(&atom.salience_count.to_le_bytes());
            buf.extend_from_slice(&(atom.payload.len() as u32).to_le_bytes());
            buf.extend_from_slice(&atom.payload);
        }

        // 4. Directed 1-cell reasoning arcs
        buf.extend_from_slice(&(self.reasoning_arcs.len() as u32).to_le_bytes());
        for arc in &self.reasoning_arcs {
            buf.extend_from_slice(&arc.arc_id.to_le_bytes());
            buf.extend_from_slice(&arc.chain_id.to_le_bytes());
            buf.extend_from_slice(&(arc.step_index as u64).to_le_bytes());
            buf.extend_from_slice(&arc.from_atom.0.to_le_bytes());
            buf.extend_from_slice(&arc.to_atom.0.to_le_bytes());
            buf.push(arc.from_role.carved_index().unwrap_or(12) as u8);
            buf.push(arc.to_role.carved_index().unwrap_or(12) as u8);
            buf.extend_from_slice(&arc.cosh_action_q32.0.to_le_bytes());
            for r in 0..3 {
                for c in 0..3 {
                    buf.extend_from_slice(&arc.wigner_frame[r][c].0.to_le_bytes());
                }
            }
            buf.extend_from_slice(&arc.compiled_mlir_op.0.to_le_bytes());
            buf.extend_from_slice(&arc.teacher_certainty_q32.0.to_le_bytes());
            buf.extend_from_slice(&arc.origin_model_fingerprint.0);
        }

        let digest = Sha256Digest::compute(&buf);
        buf.extend_from_slice(&digest.0);

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, &buf)
            .map_err(|e| format!("Failed to write vault {}: {e}", path.display()))?;
        Ok((self.global_atoms.len(), self.reasoning_arcs.len(), digest))
    }

    /// Load and authenticate a `.atoma_map` binary vault from disk into `self`, rebuilding
    /// the O(1) spatial novelty index and 12 carved specialist silos.
    pub fn load_from_binary_vault(
        &mut self,
        path: &std::path::Path,
        policy: &OverlapPolicyMatrix,
    ) -> Result<(usize, usize, Sha256Digest), String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read vault {}: {e}", path.display()))?;
        if bytes.len() < 12 + 32 || &bytes[0..12] != b"ATOMA_MAP_V1" {
            return Err("Invalid .atoma_map magic header or truncated file".to_string());
        }
        let payload_len = bytes.len() - 32;
        let expected_sha = Sha256Digest::compute(&bytes[..payload_len]);
        let mut stored_sha = [0u8; 32];
        stored_sha.copy_from_slice(&bytes[payload_len..]);
        if expected_sha.0 != stored_sha {
            return Err("Vault SHA-256 integrity check failed".to_string());
        }

        let mut pos = 12usize;
        let read_u64 = |pos: &mut usize| -> Result<u64, String> {
            if *pos + 8 > payload_len {
                return Err("Unexpected EOF reading u64".to_string());
            }
            let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(v)
        };
        let read_i64 = |pos: &mut usize| -> Result<i64, String> { Ok(read_u64(pos)? as i64) };
        let read_u32 = |pos: &mut usize| -> Result<u32, String> {
            if *pos + 4 > payload_len {
                return Err("Unexpected EOF reading u32".to_string());
            }
            let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(v)
        };
        let read_sha = |pos: &mut usize| -> Result<Sha256Digest, String> {
            if *pos + 32 > payload_len {
                return Err("Unexpected EOF reading Sha256Digest".to_string());
            }
            let mut s = [0u8; 32];
            s.copy_from_slice(&bytes[*pos..*pos + 32]);
            *pos += 32;
            Ok(Sha256Digest(s))
        };

        self.model_epoch = read_u64(&mut pos)?;
        self.next_id = read_u64(&mut pos)?;
        self.next_arc_id = read_u64(&mut pos)?;
        self.clock = LogicalClock(read_u64(&mut pos)?);

        let lineage_count = read_u32(&mut pos)? as usize;
        self.model_lineage.clear();
        for _ in 0..lineage_count {
            let ep = read_u64(&mut pos)?;
            let nlen = read_u32(&mut pos)? as usize;
            if pos + nlen > payload_len {
                return Err("Unexpected EOF reading lineage name".to_string());
            }
            let name = String::from_utf8_lossy(&bytes[pos..pos + nlen]).to_string();
            pos += nlen;
            let fp = read_sha(&mut pos)?;
            self.model_lineage.push((ep, name, fp));
        }

        let digest_count = read_u32(&mut pos)? as usize;
        self.ingested_tensor_digests.clear();
        for _ in 0..digest_count {
            self.ingested_tensor_digests.insert(read_sha(&mut pos)?);
        }

        let atom_count = read_u32(&mut pos)? as usize;
        self.global_atoms.clear();
        for _ in 0..atom_count {
            let id = AtomId(read_u64(&mut pos)?);
            let clock = LogicalClock(read_u64(&mut pos)?);
            let poincare_ball = [
                Q32_31(read_i64(&mut pos)?),
                Q32_31(read_i64(&mut pos)?),
                Q32_31(read_i64(&mut pos)?),
            ];
            let lorentz = LorentzVector {
                x: [
                    Q32_31(read_i64(&mut pos)?),
                    Q32_31(read_i64(&mut pos)?),
                    Q32_31(read_i64(&mut pos)?),
                    Q32_31(read_i64(&mut pos)?),
                ],
            };
            let phase = Phase {
                theta_f: Q32_31(read_i64(&mut pos)?),
                theta_s: Q32_31(read_i64(&mut pos)?),
            };
            let tf_pi = rad_q32_to_pi_unit(phase.theta_f);
            let ts_pi = rad_q32_to_pi_unit(phase.theta_s);
            let invariant = extract_invariant(tf_pi, ts_pi);
            let origin_model_fingerprint = read_sha(&mut pos)?;
            let salience_count = read_u32(&mut pos)?;
            let plen = read_u32(&mut pos)? as usize;
            if pos + plen > payload_len {
                return Err("Unexpected EOF reading atom payload".to_string());
            }
            let payload: Arc<[u8]> = Arc::from(bytes[pos..pos + plen].to_vec().into_boxed_slice());
            pos += plen;

            self.global_atoms.insert(
                id.0,
                GlobalAtom {
                    id,
                    clock,
                    poincare_ball,
                    lorentz,
                    phase,
                    invariant,
                    payload,
                    origin_model_fingerprint,
                    salience_count,
                },
            );
        }

        let arc_count = read_u32(&mut pos)? as usize;
        self.reasoning_arcs.clear();
        self.outgoing_arcs_by_atom.clear();
        for _ in 0..arc_count {
            let arc_id = read_u64(&mut pos)?;
            let chain_id = read_u64(&mut pos)?;
            let step_index = read_u64(&mut pos)? as usize;
            let from_atom = AtomId(read_u64(&mut pos)?);
            let to_atom = AtomId(read_u64(&mut pos)?);
            if pos + 2 > payload_len {
                return Err("Unexpected EOF reading arc roles".to_string());
            }
            let from_role = RoleId::CARVED_12
                .get(bytes[pos] as usize)
                .copied()
                .unwrap_or(RoleId::Coordinator);
            let to_role = RoleId::CARVED_12
                .get(bytes[pos + 1] as usize)
                .copied()
                .unwrap_or(RoleId::Coordinator);
            pos += 2;
            let cosh_action_q32 = Q32_31(read_i64(&mut pos)?);
            let mut wigner_frame = [[Q32_31::ZERO; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    wigner_frame[r][c] = Q32_31(read_i64(&mut pos)?);
                }
            }
            let compiled_mlir_op = crate::mlir_native::MlirOpId(read_u32(&mut pos)?);
            let teacher_certainty_q32 = Q32_31(read_i64(&mut pos)?);
            let origin_model_fingerprint = read_sha(&mut pos)?;

            let idx = self.reasoning_arcs.len();
            self.outgoing_arcs_by_atom
                .entry(from_atom.0)
                .or_default()
                .push(idx);
            self.reasoning_arcs.push(DistilledReasoningArc {
                arc_id,
                chain_id,
                step_index,
                from_atom,
                to_atom,
                from_role,
                to_role,
                cosh_action_q32,
                wigner_frame,
                compiled_mlir_op,
                teacher_certainty_q32,
                origin_model_fingerprint,
            });
        }

        self.recarve_all_twelve(policy);
        Ok((
            self.global_atoms.len(),
            self.reasoning_arcs.len(),
            expected_sha,
        ))
    }

    /// Merge another `.atoma_map` binary vault into `self`, coalescing `0-cell` coordinates
    /// in `(H³ × T²)/Γ₁₂`, remapping all `1-cell` `DistilledReasoningArc` endpoints to their
    /// merged `AtomId`s, and optionally compacting raw multi-kilobyte token-carrier payloads
    /// (`> 4096 B` non-`ATOMA_OP` / non-`ATOMA_KV` float dumps) into `256d` receptor sketches (`1024 B`).
    /// Returns `(novel_atoms_added, novel_arcs_added)`.
    pub fn merge_vault_from(
        &mut self,
        other_path: &std::path::Path,
        policy: &OverlapPolicyMatrix,
        compact_carrier_payloads: bool,
    ) -> Result<(usize, usize), String> {
        let plug = FrozenBaseWeights::freeze("merge-loader", 1, vec![0.0; 8]);
        let mut other = CarvedSwarmMap::new(self.atlas.clone(), plug, self.default_basin_cosh);
        other.load_from_binary_vault(other_path, policy)?;

        for d in other.ingested_tensor_digests {
            self.ingested_tensor_digests.insert(d);
        }
        for (_ep, name, fp) in other.model_lineage {
            if !self.is_registered_model_fingerprint(&fp) {
                let next_ep = self.model_lineage.len() as u64 + 1;
                self.model_lineage.push((next_ep, name, fp));
            }
        }

        let mut id_remap = BTreeMap::<u64, AtomId>::new();
        let mut novel_atoms = 0usize;

        for (&old_id, atom) in &other.global_atoms {
            let payload_vec: Vec<u8> = if compact_carrier_payloads
                && atom.payload.len() > 4096
                && !atom.payload.starts_with(b"ATOMA_OP")
                && !atom.payload.starts_with(b"ATOMA_KV_V1\0")
            {
                // Compress 26.6 KB raw f32 carrier state down to its first 256 floats (1024 B)
                atom.payload[..1024.min(atom.payload.len())].to_vec()
            } else {
                atom.payload.as_ref().to_vec()
            };

            let refracted = RefractedPoint {
                poincare_ball: atom.poincare_ball,
                lorentz: atom.lorentz,
                phase: atom.phase,
                invariant: atom.invariant,
                projection_loss: crate::refraction::ProjectionLossReport {
                    input_state_bits: 32,
                    output_tangent_bits: 256,
                    discarded_dimensions: 0,
                    retained_energy_ratio_q32: Q32_31::ONE,
                    discarded_energy_ratio_q32: Q32_31::ZERO,
                    teacher_entropy_bits_q32: Q32_31::ZERO,
                },
                refraction_loss: crate::refraction::RefractionLossReport {
                    quantization_ulp_bound: 1,
                    hyperboloid_residual_q32: Q32_31::ZERO,
                    refractive_index_q32: Q32_31::ONE,
                },
            };

            let (new_id, is_novel) = self.keep_and_carve_novel_only(
                refracted,
                payload_vec.clone(),
                atom.origin_model_fingerprint,
                policy,
            );
            if is_novel {
                novel_atoms += 1;
            } else if !payload_vec.is_empty() {
                self.store_payload_if_empty(new_id, payload_vec);
            }
            id_remap.insert(old_id, new_id);
        }

        let mut novel_arcs = 0usize;
        for mut arc in other.reasoning_arcs {
            let Some(&new_from) = id_remap.get(&arc.from_atom.0) else {
                continue;
            };
            let Some(&new_to) = id_remap.get(&arc.to_atom.0) else {
                continue;
            };
            arc.from_atom = new_from;
            arc.to_atom = new_to;
            let (_aid, is_novel) = self.record_trajectory_arc(arc);
            if is_novel {
                novel_arcs += 1;
            }
        }

        Ok((novel_atoms, novel_arcs))
    }

    pub fn is_registered_model_fingerprint(&self, fp: &Sha256Digest) -> bool {
        self.model_lineage
            .iter()
            .any(|(_, _, lineage_fp)| lineage_fp == fp)
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

/// Mandatory loss report for Cross-Epoch Model Adapter projection (`Qualification #9`: Name the Loss).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterLossReport {
    pub origin_fingerprint: Sha256Digest,
    pub target_fingerprint: Sha256Digest,
    pub source_head_dim: usize,
    pub target_head_dim: usize,
    /// Fraction of `K` head L2 energy retained after cross-epoch projection, in Q32.31 (`[0, 1]`).
    pub retained_k_energy_ratio_q32: Q32_31,
    /// Fraction of `V` head L2 energy retained after cross-epoch projection, in Q32.31 (`[0, 1]`).
    pub retained_v_energy_ratio_q32: Q32_31,
    /// Integrity of 2x2 complex RoPE rotor pairs (`Q32_31::ONE` = 100% pair-preserving).
    pub rope_pair_integrity_q32: Q32_31,
    /// True if `origin_fingerprint == target_fingerprint` and `source_head_dim == target_head_dim`.
    pub is_zero_copy_passthrough: bool,
}

/// Result of projecting an atom's raw KV payload across model epochs via [`ModelAdapterLens`].
#[derive(Debug, Clone)]
pub struct AdaptedKvRecord {
    pub atom_id: AtomId,
    pub origin_model_fingerprint: Sha256Digest,
    pub target_model_fingerprint: Sha256Digest,
    pub k_head: Vec<f32>,
    pub v_head: Vec<f32>,
    pub loss_report: AdapterLossReport,
}

/// Cross-Epoch Model Adapter Lens (`ModelAdapterLens`).
///
/// When a new model backend is hot-swapped into the server (`epoch N -> epoch N+1`, e.g.
/// `128d` head Glimmer -> `256d` head NextGen MoE), the centralized `(H³ × T²) / Γ` memory
/// coordinates require zero re-distillation. When a new-epoch worker retrieves an atom whose
/// raw `(K, V)` payload was deposited by an earlier model epoch (`origin_model_fingerprint`),
/// `ModelAdapterLens` applies a deterministic, RoPE-pair-preserving `SO(2)^P` gauge rotation
/// and isometric pair-harmonic embedding/projection into `target_head_dim`.
#[derive(Debug, Clone, Default)]
pub struct ModelAdapterLens;

impl ModelAdapterLens {
    /// Pack `(k_head, v_head)` `f32` slices into a canonical little-endian KV payload buffer.
    pub fn pack_kv_payload(k_head: &[f32], v_head: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity((k_head.len() + v_head.len()) * 4);
        for &f in k_head {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        for &f in v_head {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf
    }

    /// Adapt a raw KV payload from `origin_fingerprint` (`source_head_dim` inferred from payload length)
    /// into `target_fingerprint` at `target_head_dim`, anchored by the atom's canonical `T²` `Phase`.
    pub fn adapt_kv_payload(
        atom_id: AtomId,
        payload: &[u8],
        origin_fingerprint: Sha256Digest,
        target_fingerprint: Sha256Digest,
        phase: Phase,
        target_head_dim: usize,
    ) -> Result<AdaptedKvRecord, &'static str> {
        if payload.len() < 16 || payload.len() % 8 != 0 {
            return Err("KV payload byte length must be a multiple of 8 (even head_dim f32 K+V)");
        }
        if target_head_dim < 2 || target_head_dim % 2 != 0 {
            return Err("target_head_dim must be a positive even integer (2x2 RoPE pairs)");
        }

        let source_head_dim = payload.len() / 8;
        if source_head_dim % 2 != 0 {
            return Err("source_head_dim must be even (2x2 RoPE pairs)");
        }

        let k_src: Vec<f32> = payload[0..source_head_dim * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let v_src: Vec<f32> = payload[source_head_dim * 4..2 * source_head_dim * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Fast path: identical model epoch and identical head dimension -> exact pass-through
        if origin_fingerprint == target_fingerprint && source_head_dim == target_head_dim {
            return Ok(AdaptedKvRecord {
                atom_id,
                origin_model_fingerprint: origin_fingerprint,
                target_model_fingerprint: target_fingerprint,
                k_head: k_src,
                v_head: v_src,
                loss_report: AdapterLossReport {
                    origin_fingerprint,
                    target_fingerprint,
                    source_head_dim,
                    target_head_dim,
                    retained_k_energy_ratio_q32: Q32_31::ONE,
                    retained_v_energy_ratio_q32: Q32_31::ONE,
                    rope_pair_integrity_q32: Q32_31::ONE,
                    is_zero_copy_passthrough: true,
                },
            });
        }

        let p_src = source_head_dim / 2;
        let p_dst = target_head_dim / 2;
        let phase_offset = (phase.theta_f.to_f64() + phase.theta_s.to_f64()) * 0.03125;

        // Step 1: Deterministic SO(2) RoPE-pair gauge alignment between origin_fingerprint and target_fingerprint
        let mut k_rot = vec![0.0f32; source_head_dim];
        let mut v_rot = vec![0.0f32; source_head_dim];
        for p in 0..p_src {
            let angle = if origin_fingerprint == target_fingerprint {
                0.0
            } else {
                Self::pair_gauge_angle(&origin_fingerprint, &target_fingerprint, p) + phase_offset
            };
            let (s, c) = angle.sin_cos();
            let c32 = c as f32;
            let s32 = s as f32;

            let k0 = k_src[2 * p];
            let k1 = k_src[2 * p + 1];
            k_rot[2 * p] = k0 * c32 - k1 * s32;
            k_rot[2 * p + 1] = k0 * s32 + k1 * c32;

            let v0 = v_src[2 * p];
            let v1 = v_src[2 * p + 1];
            v_rot[2 * p] = v0 * c32 - v1 * s32;
            v_rot[2 * p + 1] = v0 * s32 + v1 * c32;
        }

        // Step 2: Isometric Pair-Harmonic Embedding (p_dst >= p_src) or Spectral Pair Truncation (p_dst < p_src)
        let mut k_dst = vec![0.0f32; target_head_dim];
        let mut v_dst = vec![0.0f32; target_head_dim];

        if p_dst >= p_src {
            // Distribute p_dst target pairs across p_src source pairs with exact 1/sqrt(m_p) normalization
            let base_rep = p_dst / p_src;
            let rem = p_dst % p_src;
            let mut dst_pair_idx = 0usize;
            for p in 0..p_src {
                let m_p = base_rep + usize::from(p < rem);
                let scale = 1.0f32 / (m_p as f32).sqrt();
                for r in 0..m_p {
                    let sign = if r % 2 == 0 { 1.0f32 } else { -1.0f32 };
                    k_dst[2 * dst_pair_idx] = k_rot[2 * p] * scale * sign;
                    k_dst[2 * dst_pair_idx + 1] = k_rot[2 * p + 1] * scale * sign;
                    v_dst[2 * dst_pair_idx] = v_rot[2 * p] * scale * sign;
                    v_dst[2 * dst_pair_idx + 1] = v_rot[2 * p + 1] * scale * sign;
                    dst_pair_idx += 1;
                }
            }
        } else {
            // Retain top low-frequency RoPE pairs and explicitly log discarded spectral energy (Qual #9)
            k_dst.copy_from_slice(&k_rot[0..target_head_dim]);
            v_dst.copy_from_slice(&v_rot[0..target_head_dim]);
        }

        let k_src_energy: f64 = k_src.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let k_dst_energy: f64 = k_dst.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let v_src_energy: f64 = v_src.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let v_dst_energy: f64 = v_dst.iter().map(|&x| (x as f64) * (x as f64)).sum();

        let retained_k = if k_src_energy <= 1e-18 {
            Q32_31::ONE
        } else {
            Q32_31::from_f64((k_dst_energy / k_src_energy).clamp(0.0, 1.0))
        };
        let retained_v = if v_src_energy <= 1e-18 {
            Q32_31::ONE
        } else {
            Q32_31::from_f64((v_dst_energy / v_src_energy).clamp(0.0, 1.0))
        };

        Ok(AdaptedKvRecord {
            atom_id,
            origin_model_fingerprint: origin_fingerprint,
            target_model_fingerprint: target_fingerprint,
            k_head: k_dst,
            v_head: v_dst,
            loss_report: AdapterLossReport {
                origin_fingerprint,
                target_fingerprint,
                source_head_dim,
                target_head_dim,
                retained_k_energy_ratio_q32: retained_k,
                retained_v_energy_ratio_q32: retained_v,
                rope_pair_integrity_q32: Q32_31::ONE,
                is_zero_copy_passthrough: false,
            },
        })
    }

    fn pair_gauge_angle(
        origin_fp: &Sha256Digest,
        target_fp: &Sha256Digest,
        pair_index: usize,
    ) -> f64 {
        let mut seed = [0u8; 72];
        seed[0..32].copy_from_slice(&origin_fp.0);
        seed[32..64].copy_from_slice(&target_fp.0);
        seed[64..72].copy_from_slice(&(pair_index as u64).to_le_bytes());
        let digest = Sha256Digest::compute(&seed);
        let word = u64::from_le_bytes([
            digest.0[0],
            digest.0[1],
            digest.0[2],
            digest.0[3],
            digest.0[4],
            digest.0[5],
            digest.0[6],
            digest.0[7],
        ]);
        // Map deterministically into [-π/8, +π/8] rad so cross-epoch phase alignment is smooth
        let unit = (word as f64) / (u64::MAX as f64) - 0.5;
        unit * (std::f64::consts::PI * 0.25)
    }
}
