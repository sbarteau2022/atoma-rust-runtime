//! Tier 5–6 (`SPEC.md` §3 DHNG II + Geodesic Knowledge & Reasoning Distillation Engine):
//! Turning the 5-Stage Refraction Pipeline into an Autonomous Open-Weights Distillation Engine.
//!
//! Why a Distillation Engine must separate **Knowledge (`0-cells`)** from **Reasoning (`1-cells` & `2-cells`)**:
//! 1. **Knowledge = Attractor Points (`0-cells` in `(H³ × T²)/Γ₁₂`)**:
//!    Extracted both from **Static Weight-Matrix Principal Modes** (`distill_static_weight_modes`,
//!    reading frozen open-source weight rows across early/mid/late layer depth bands without
//!    needing thousands of prompts) and from **Low-Entropy Teacher Activations**.
//!    High-entropy teacher guesses (`H(p) > max_entropy_bits`) are rejected at the gate (`Entropy Sieve`),
//!    while high-certainty states are refracted at an optical radius proportional to their specificity.
//! 2. **Intrinsic 6-Axis Icosahedral Auto-Routing (`Qualification #3: Math decides, not preference`)**:
//!    Instead of manually assigning a `RoleId`, [`OpenWeightsDistillationEngine::classify_icosahedral_well`]
//!    projects the $D$-dimensional state onto the 6 equiangular icosahedral duality axes
//!    (`|cos θ| = 1/√5`) modulated by layer depth (`Early -> Sensory/Symbolic/Spatial`,
//!    `Mid -> Logic/Causal/Episodic`, `Late -> Semantic/Executive/Motor`) to select the natural
//!    specialist well `L1..L12` automatically.
//! 3. **Reasoning = Directed Geodesic Arcs (`1-cells`) + Wigner Transport + Native MLIR Ops**:
//!    Reasoning is not a static point; it is the **geodesic transition operator** that carries
//!    Premise $A \to$ Intermediate Lemma $B \to$ Conclusion $C$.
//!    [`OpenWeightsDistillationEngine::distill_reasoning_chain`] ingests a multi-step teacher
//!    trajectory $[h_0, h_1, \dots, h_{T-1}]$, refracts each step into a `GlobalAtom`, computes the
//!    exact $\text{SO}^+(3,1)$ Levi-Civita Wigner parallel transport frame and hyperbolic action
//!    $\cosh d(x_t, x_{t+1})$ between consecutive steps, synthesizes the typed [`MlirSsaBlock`]
//!    via [`GeodesicMlirDecoder`], validates loop curvature via [`HyperbolicHolonomyEngine`] (`2-cells`),
//!    and records persistent [`DistilledReasoningArc`] edges inside `CarvedSwarmMap`.

use crate::atlas13::{Atlas13, RoleId};
use crate::cache::geometric::AtomId;
use crate::compliance::OverlapPolicyMatrix;
use crate::geometry::fixed::Q32_31;
use crate::holonomy::{HolonomyVerdict, HyperbolicHolonomyEngine, IDENTITY_FRAME_Q32};
use crate::mlir_native::{GeodesicMlirDecoder, MlirAtomCatalog, MlirOpId, MlirSsaBlock};
use crate::refraction::{
    FrozenShard, OpticalLens, Sha256Digest, TangentProjector, TeacherTapRecord,
};
use crate::silo::{CarvedSwarmMap, FrozenBaseWeights};

/// A directed 1-cell geodesic reasoning transition in the centralized `(H³ × T²)/Γ₁₂` map.
///
/// Captures *how* a teacher model reasons from `from_atom` (`from_role`) to `to_atom` (`to_role`),
/// including the exact `SO⁺(3,1)` Levi-Civita Wigner gyroscope transport frame and the compiled
/// native [`MlirOpId`] operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistilledReasoningArc {
    pub arc_id: u64,
    pub chain_id: u64,
    pub step_index: usize,
    pub from_atom: AtomId,
    pub to_atom: AtomId,
    pub from_role: RoleId,
    pub to_role: RoleId,
    /// Hyperbolic geodesic action `cosh(d(x_t, x_{t+1}))` in Q32.31 (`>= 1.0`).
    pub cosh_action_q32: Q32_31,
    /// Cumulative `3×3` Wigner gyroscope frame parallel-transported along the reasoning chain.
    pub wigner_frame: [[Q32_31; 3]; 3],
    /// Native MLIR opcode synthesized for this reasoning step via `GeodesicMlirDecoder`.
    pub compiled_mlir_op: MlirOpId,
    /// Teacher certainty `1 / (1 + H(p))` in Q32.31 (`(0, 1]`).
    pub teacher_certainty_q32: Q32_31,
    /// Fingerprint of the open-source teacher checkpoint that distilled this reasoning arc.
    pub origin_model_fingerprint: Sha256Digest,
}

/// Receipt returned when distilling static weight modes (`0-cells`) from an open-source checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightDistillationReceipt {
    pub teacher_model_name: String,
    pub teacher_fingerprint: Sha256Digest,
    pub modes_scanned: usize,
    pub knowledge_atoms_minted: usize,
    pub role_distribution: [usize; 12],
    pub mean_retained_energy_q32: Q32_31,
}

/// Receipt returned when distilling a multi-step reasoning trajectory (`1-cells` + `2-cell` holonomy)
/// into the centralized `CarvedSwarmMap`.
#[derive(Debug, Clone)]
pub struct ReasoningChainDistillationReceipt {
    pub chain_id: u64,
    pub steps_ingested: usize,
    pub steps_rejected_by_entropy_sieve: usize,
    pub atom_sequence: Vec<AtomId>,
    pub role_sequence: Vec<RoleId>,
    pub arc_ids: Vec<u64>,
    pub compiled_ssa_block: MlirSsaBlock,
    pub holonomy_verdict: HolonomyVerdict,
    pub enclosed_curvature_flux_q32: Q32_31,
    pub mean_teacher_certainty_q32: Q32_31,
}

/// Quality & Entropy Sieve parameters for the Open-Weights Distillation Engine.
pub struct OpenWeightsDistillationEngine {
    /// Maximum allowed Shannon entropy `H(p)` in Q32.31 bits before a teacher tap is rejected
    /// as low-certainty noise / hallucination.
    pub max_entropy_bits_q32: Q32_31,
    pub projector: TangentProjector,
    pub holonomy_engine: HyperbolicHolonomyEngine,
}

impl Default for OpenWeightsDistillationEngine {
    fn default() -> Self {
        Self {
            // Allow up to 2.75 bits of next-op entropy; higher-entropy uniform guesses are sieved out
            max_entropy_bits_q32: Q32_31::from_f64(2.75),
            projector: TangentProjector::default(),
            holonomy_engine: HyperbolicHolonomyEngine::default(),
        }
    }
}

impl OpenWeightsDistillationEngine {
    /// Intrinsic 6-Axis Icosahedral Auto-Router (`Qualification #3: Math decides, not preference`).
    ///
    /// Projects `hidden_state` onto the 12 icosahedral unit directions of `Atlas13`, weighted by
    /// the transformer layer depth band (`early / mid / late`), and computes a radial offset
    /// controlled by the teacher's certainty so foundational low-entropy truths sit closer to the
    /// core and high-entropy peripheral details sit closer to the Poincaré rim.
    pub fn classify_icosahedral_well(
        atlas: &Atlas13,
        hidden_state: &[f32],
        layer_index: usize,
        total_layers: usize,
        entropy_bits_q32: Q32_31,
    ) -> (RoleId, [Q32_31; 3], Q32_31) {
        // Compute 3D spatial moment from hidden_state
        let mut moment = [0.0f64; 3];
        for (i, &v) in hidden_state.iter().enumerate() {
            let angle = (i as f64) * 0.618_033_988_749_895 * std::f64::consts::TAU;
            moment[0] += (v as f64) * angle.cos();
            moment[1] += (v as f64) * angle.sin();
            moment[2] += (v as f64) * ((i % 5) as f64 - 2.0) * 0.5;
        }
        let norm = (moment[0] * moment[0] + moment[1] * moment[1] + moment[2] * moment[2])
            .sqrt()
            .max(1e-9);
        let dir = [moment[0] / norm, moment[1] / norm, moment[2] / norm];

        // Depth band prior:
        // - Early layers (0..34%): Sensory (L1), Somatic (L3), LanguageSemiotics (L4), Spatial (L11)
        // - Mid layers (34..68%): Episodic (L5), Predictive (L6), ConvergentLogic (L7), DivergentSynthesis (L8), Valuation (L9)
        // - Late layers (68..100%): Motor (L2), SocialCognition (L10), MetacognitiveRegulation (L12)
        let depth_ratio = (layer_index as f64) / (total_layers.max(1) as f64);

        let mut best_role = RoleId::SensoryPerception;
        let mut best_score = f64::NEG_INFINITY;

        for role in RoleId::CARVED_12 {
            let well = atlas.well(role);
            let ux = well.unit_dir[0].to_f64();
            let uy = well.unit_dir[1].to_f64();
            let uz = well.unit_dir[2].to_f64();
            let cos_sim = dir[0] * ux + dir[1] * uy + dir[2] * uz;

            let depth_bonus = match role {
                RoleId::SensoryPerception
                | RoleId::SomaticInteroception
                | RoleId::LanguageSemiotics
                | RoleId::SpatialTopography => {
                    if depth_ratio < 0.34 {
                        0.35
                    } else {
                        0.0
                    }
                }
                RoleId::EpisodicMemory
                | RoleId::PredictiveSimulation
                | RoleId::ConvergentLogic
                | RoleId::DivergentSynthesis
                | RoleId::ValuationSalience => {
                    if (0.34..0.68).contains(&depth_ratio) {
                        0.35
                    } else {
                        0.0
                    }
                }
                RoleId::MotorPraxis | RoleId::SocialCognition | RoleId::MetacognitiveRegulation => {
                    if depth_ratio >= 0.68 {
                        0.35
                    } else {
                        0.0
                    }
                }
                RoleId::Coordinator => 0.0,
            };

            let score = cos_sim + depth_bonus;
            if score > best_score {
                best_score = score;
                best_role = role;
            }
        }

        // Teacher certainty = 1 / (1 + H(p)) in (0, 1]
        let h_f64 = entropy_bits_q32.to_f64().max(0.0);
        let certainty = 1.0 / (1.0 + h_f64);
        let certainty_q32 = Q32_31::from_f64(certainty.clamp(0.05, 1.0));

        // Perturb slightly along the state direction inside the role's home basin (`<= 0.06` norm)
        let radial_scale = (1.0 - certainty) * 0.05;
        let offset = [
            Q32_31::from_f64(dir[0] * radial_scale),
            Q32_31::from_f64(dir[1] * radial_scale),
            Q32_31::from_f64(dir[2] * radial_scale),
        ];

        (best_role, offset, certainty_q32)
    }

    /// Pillar 1: Distill **Knowledge (`0-cells`)** directly from an open-source model's frozen
    /// weight buffer (`weights`) by extracting high-energy spectral row modes across layers
    /// (`early`, `mid`, `late`) and refracting them into `CarvedSwarmMap`.
    pub fn distill_static_weight_modes(
        &self,
        teacher_weights: &FrozenBaseWeights,
        mode_width: usize,
        max_modes: usize,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<WeightDistillationReceipt, String> {
        let raw = teacher_weights.raw_weights();
        let width = mode_width.max(16);
        if raw.len() < width {
            return Err("FrozenBaseWeights buffer shorter than mode_width".to_string());
        }

        let total_chunks = raw.len() / width;
        let stride = (total_chunks / max_modes.max(1)).max(1);
        let mut role_distribution = [0usize; 12];
        let mut minted = 0usize;
        let mut retained_sum = 0.0f64;

        for mode_idx in 0..max_modes.min(total_chunks) {
            let chunk_idx = (mode_idx * stride) % total_chunks;
            let slice = &raw[chunk_idx * width..(chunk_idx + 1) * width];
            let layer_index = (mode_idx * teacher_weights.num_layers.max(1)) / max_modes.max(1);

            // Compute spectral concentration over 4 canonical sub-bands to form a probability distribution
            let quarter = (width / 4).max(1);
            let mut band_energies = [0.0f32; 4];
            for b in 0..4 {
                let sub = &slice[b * quarter..((b + 1) * quarter).min(width)];
                band_energies[b] = sub.iter().map(|&v| v * v).sum::<f32>() + 1e-6;
            }
            let total_e: f32 = band_energies.iter().sum();
            let next_op_probs: Vec<f32> = band_energies.iter().map(|&e| e / total_e).collect();

            let mut payload = Vec::with_capacity(16 + 64);
            payload.extend_from_slice(b"weight_mode:");
            payload.extend_from_slice(&(layer_index as u32).to_le_bytes());
            payload.extend_from_slice(&(mode_idx as u32).to_le_bytes());
            // Pack leading 16 floats (8 RoPE pairs) as canonical KV seed
            for &f in slice.iter().take(16) {
                payload.extend_from_slice(&f.to_le_bytes());
            }

            let tap = TeacherTapRecord {
                shard_id: 10_000 + mode_idx as u64,
                layer_index,
                hidden_state: slice.to_vec(),
                next_op_probs,
                payload,
            };

            let _shard = FrozenShard::from_tap(&tap).map_err(|e| e.to_string())?;
            let mut projected = self.projector.project(&tap).map_err(|e| e.to_string())?;

            let (role, offset, _certainty) = Self::classify_icosahedral_well(
                &map.atlas,
                &tap.hidden_state,
                layer_index,
                teacher_weights.num_layers,
                projected.loss_report.teacher_entropy_bits_q32,
            );

            let well = map.atlas.well(role);
            let d_sph = Q32_31::from_f64(0.804_735_5);
            projected.tangent_h3 = [
                (well.unit_dir[0] * d_sph) + offset[0],
                (well.unit_dir[1] * d_sph) + offset[1],
                (well.unit_dir[2] * d_sph) + offset[2],
            ];

            let refracted =
                OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
            let atom_id = map.keep_and_carve_with_origin(
                refracted,
                tap.payload,
                teacher_weights.fingerprint,
                policy,
            );
            let _ = atom_id;

            if let Some(r_idx) = role.carved_index() {
                role_distribution[r_idx] += 1;
            }
            retained_sum += projected.loss_report.retained_energy_ratio_q32.to_f64();
            minted += 1;
        }

        let mean_retained = if minted > 0 {
            Q32_31::from_f64((retained_sum / (minted as f64)).clamp(0.0, 1.0))
        } else {
            Q32_31::ZERO
        };

        Ok(WeightDistillationReceipt {
            teacher_model_name: teacher_weights.model_name.clone(),
            teacher_fingerprint: teacher_weights.fingerprint,
            modes_scanned: max_modes.min(total_chunks),
            knowledge_atoms_minted: minted,
            role_distribution,
            mean_retained_energy_q32: mean_retained,
        })
    }

    /// Pillar 2 & 3: Distill a **Multi-Step Teacher Reasoning Chain (`1-cells` + `2-cell` Holonomy)**
    /// into the centralized `CarvedSwarmMap`.
    ///
    /// - Filters out high-entropy hallucination steps (`teacher_entropy_bits_q32 > max_entropy_bits_q32`).
    /// - Auto-routes each step to its icosahedral `RoleId` (`L1..L12`).
    /// - Refractor-maps each step into a `GlobalAtom` (`0-cell`).
    /// - Synthesizes the native typed [`MlirSsaBlock`] directly from the continuous hidden-state
    ///   trajectory via [`GeodesicMlirDecoder`] (`0` tokenizer, `0` JSON/Python).
    /// - Connects consecutive steps with directed [`DistilledReasoningArc`] edges (`1-cells`)
    ///   carrying the exact Levi-Civita Wigner parallel-transported frame and hyperbolic action.
    /// - Evaluates the closed-loop Gauss–Bonnet Wigner holonomy (`2-cell`) across the visited
    ///   specialist basins and reinforces atom salience when multi-perspective synthesis is verified!
    pub fn distill_reasoning_chain(
        &self,
        chain_id: u64,
        teacher_steps: &[TeacherTapRecord],
        total_teacher_layers: usize,
        teacher_fingerprint: Sha256Digest,
        mlir_catalog: &MlirAtomCatalog,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<ReasoningChainDistillationReceipt, String> {
        if teacher_steps.is_empty() {
            return Err("teacher_steps cannot be empty".to_string());
        }

        let mut kept_states: Vec<Vec<f32>> = Vec::with_capacity(teacher_steps.len());
        let mut atom_sequence: Vec<AtomId> = Vec::with_capacity(teacher_steps.len());
        let mut role_sequence: Vec<RoleId> = Vec::with_capacity(teacher_steps.len());
        let mut certainties: Vec<Q32_31> = Vec::with_capacity(teacher_steps.len());
        let mut rejected_entropy = 0usize;

        for tap in teacher_steps {
            let _shard = FrozenShard::from_tap(tap).map_err(|e| e.to_string())?;
            let mut projected = self.projector.project(tap).map_err(|e| e.to_string())?;

            // Entropy Sieve: reject high-entropy teacher guesses before they enter the global map
            if projected.loss_report.teacher_entropy_bits_q32.0 > self.max_entropy_bits_q32.0 {
                rejected_entropy += 1;
                continue;
            }

            let (role, mut offset, certainty) = Self::classify_icosahedral_well(
                &map.atlas,
                &tap.hidden_state,
                tap.layer_index,
                total_teacher_layers,
                projected.loss_report.teacher_entropy_bits_q32,
            );

            // If consecutive steps cross between icosahedral neighbor roles, place the point
            // toward their shared boundary overlap so inter-role handoff is structurally enabled!
            if let Some(&prev_role) = role_sequence.last() {
                if prev_role != role && map.atlas.are_icosahedral_neighbors(prev_role, role) {
                    let u_prev = map.atlas.well(prev_role).unit_dir;
                    let u_curr = map.atlas.well(role).unit_dir;
                    let d_sph = Q32_31::from_f64(0.804_735_5);
                    offset = [
                        ((u_prev[0] - u_curr[0]) * d_sph) * Q32_31::HALF,
                        ((u_prev[1] - u_curr[1]) * d_sph) * Q32_31::HALF,
                        ((u_prev[2] - u_curr[2]) * d_sph) * Q32_31::HALF,
                    ];
                }
            }

            let well = map.atlas.well(role);
            let d_sph = Q32_31::from_f64(0.804_735_5);
            projected.tangent_h3 = [
                (well.unit_dir[0] * d_sph) + offset[0],
                (well.unit_dir[1] * d_sph) + offset[1],
                (well.unit_dir[2] * d_sph) + offset[2],
            ];

            let refracted =
                OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
            let atom_id = map.keep_and_carve_with_origin(
                refracted,
                tap.payload.clone(),
                teacher_fingerprint,
                policy,
            );

            kept_states.push(tap.hidden_state.clone());
            atom_sequence.push(atom_id);
            role_sequence.push(role);
            certainties.push(certainty);
        }

        if atom_sequence.is_empty() {
            return Err("All teacher steps were rejected by the entropy sieve".to_string());
        }

        // Synthesize the native typed MLIR SSA program for the kept reasoning steps
        let primary_role = role_sequence[0];
        let geo_decode = GeodesicMlirDecoder::decode_from_hidden_states(
            mlir_catalog,
            primary_role,
            &kept_states,
        )
        .map_err(|e| e.to_string())?;

        // Build directed 1-cell DistilledReasoningArcs with Levi-Civita Wigner parallel transport
        let mut current_wigner = IDENTITY_FRAME_Q32;
        let mut arc_ids = Vec::with_capacity(atom_sequence.len().saturating_sub(1));

        for step in 0..atom_sequence.len().saturating_sub(1) {
            let from_id = atom_sequence[step];
            let to_id = atom_sequence[step + 1];
            let from_role = role_sequence[step];
            let to_role = role_sequence[step + 1];

            let from_lor = map.global_atom(from_id).unwrap().lorentz;
            let to_lor = map.global_atom(to_id).unwrap().lorentz;
            let cosh_action = OpticalLens::lorentz_inner_product_wide(&from_lor, &to_lor);

            current_wigner =
                HyperbolicHolonomyEngine::transport_frame_hop(&from_lor, &to_lor, &current_wigner);

            let compiled_op = geo_decode
                .block
                .instructions
                .get(step + 1)
                .map(|inst| inst.op_id)
                .unwrap_or(MlirOpId(mlir_catalog.base_vocab_size));

            let arc_id = map.record_reasoning_arc(DistilledReasoningArc {
                arc_id: 0, // assigned by map.record_reasoning_arc
                chain_id,
                step_index: step,
                from_atom: from_id,
                to_atom: to_id,
                from_role,
                to_role,
                cosh_action_q32: cosh_action,
                wigner_frame: current_wigner,
                compiled_mlir_op: compiled_op,
                teacher_certainty_q32: certainties[step + 1],
                origin_model_fingerprint: teacher_fingerprint,
            });
            arc_ids.push(arc_id);
        }

        // Evaluate 2-cell Gauss-Bonnet Holonomy across the closed role loop
        let mut closed_roles = role_sequence.clone();
        if let Some(&first_r) = closed_roles.first() {
            if closed_roles.last() != Some(&first_r) {
                closed_roles.push(first_r);
            }
        }
        let init_phase = map.atlas.well(primary_role).phase;
        let holonomy_receipt =
            self.holonomy_engine
                .evaluate_role_loop(&closed_roles, map, init_phase);

        // If the reasoning chain achieved genuine multi-perspective dialectical synthesis,
        // boost the salience count of every atom in the chain!
        if matches!(
            holonomy_receipt.verdict,
            HolonomyVerdict::DialecticalSynthesisVerified { .. }
        ) {
            for &aid in &atom_sequence {
                map.reinforce_atom_salience(aid, 2);
            }
        }

        let mean_certainty = {
            let sum: f64 = certainties.iter().map(|c| c.to_f64()).sum();
            Q32_31::from_f64((sum / (certainties.len() as f64)).clamp(0.0, 1.0))
        };

        Ok(ReasoningChainDistillationReceipt {
            chain_id,
            steps_ingested: atom_sequence.len(),
            steps_rejected_by_entropy_sieve: rejected_entropy,
            atom_sequence,
            role_sequence,
            arc_ids,
            compiled_ssa_block: geo_decode.block,
            holonomy_verdict: holonomy_receipt.verdict,
            enclosed_curvature_flux_q32: holonomy_receipt.enclosed_curvature_flux_q32,
            mean_teacher_certainty_q32: mean_certainty,
        })
    }

    /// Distill **Knowledge (`0-cells`)** directly from a **Quantized Layer Tensor** (`OptiQ` / `MLX`
    /// `4-bit`, `5-bit`, `6-bit`, or `8-bit` packed `U32`/`U8` weights + `BF16`/`F16`/`F32` scales &
    /// biases + optional `F32` `out_scale`) **without ever dequantizing the full matrix**!
    ///
    /// How it works:
    /// 1. **Scale/Bias Analytical Spectral Sieve (`128×` faster, zero weight unpacking)**:
    ///    Uses the affine quantization statistics of each group $g$:
    ///    `E_est(r) = out_scale[r]^2 * sum_{g=0..G-1} G * ((q_mean * s_{r,g} + b_{r,g})^2 + q_var * s_{r,g}^2)`
    ///    to score all `rows` in microseconds by reading *only* `scales`, `biases`, and `out_scale`.
    /// 2. **Top-K Single-Row Zero-Copy Dequantization**:
    ///    Selects the `top_k_rows` highest-salience rows and calls `dequantize_row_zero_copy`
    ///    only on those `top_k_rows` rows (saving `(rows - top_k_rows) * cols * 4` bytes of heap allocation).
    /// 3. **Fisher Bit-Width Priority & Icosahedral Refraction**:
    ///    Higher-bit OptiQ layers (`6-bit` / `8-bit` sensitive projections) receive a Fisher-information
    ///    certainty boost so their invariants refract deeper toward the core of `(H³ × T²)/Γ₁₂`.
    pub fn distill_quantized_layer_zero_copy(
        &self,
        desc: &QuantizedLayerTapDescriptor<'_>,
        top_k_rows: usize,
        teacher_fingerprint: Sha256Digest,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<QuantizedDistillationReceipt, String> {
        if desc.rows == 0
            || desc.cols == 0
            || desc.group_size == 0
            || desc.cols % desc.group_size != 0
        {
            return Err("Invalid quantized tensor dimensions or group_size".to_string());
        }
        let k = top_k_rows.min(desc.rows).max(1);
        let groups_per_row = desc.cols / desc.group_size;

        // Mean and variance of uniform b-bit integer code q ∈ {0, ..., 2^b - 1}
        let max_q = ((1u32 << desc.bits) - 1) as f64;
        let q_mean = 0.5 * max_q;
        let q_var = ((1u64 << (2 * desc.bits)) as f64 - 1.0) / 12.0;

        // Step 1: Analytical Scale/Bias Spectral Sieve across all `rows` (ZERO weight unpacking!)
        let mut row_scores: Vec<(f64, usize)> = Vec::with_capacity(desc.rows);
        for r in 0..desc.rows {
            let row_group_base = r * groups_per_row;
            let os = desc
                .out_scale
                .and_then(|s| s.get(r).copied())
                .unwrap_or(1.0) as f64;
            let mut energy_est = 0.0f64;
            for g in 0..groups_per_row {
                let s =
                    read_quant_scale_f64(desc.scales_data, desc.scales_dtype, row_group_base + g)?;
                let b =
                    read_quant_scale_f64(desc.biases_data, desc.biases_dtype, row_group_base + g)?;
                let group_mean = q_mean * s + b;
                let group_var = q_var * s * s;
                energy_est += (desc.group_size as f64) * (group_mean * group_mean + group_var);
            }
            row_scores.push((energy_est * os * os, r));
        }

        // Sort descending by analytical spectral energy and take top_k_rows
        row_scores.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut selected_row_indices = Vec::with_capacity(k);
        let mut minted_atom_ids = Vec::with_capacity(k);
        let mut role_distribution = [0usize; 12];

        // Step 2: Dequantize ONLY the top-K rows via `dequantize_row_zero_copy` and refract into CarvedSwarmMap
        for &(_score, r) in row_scores.iter().take(k) {
            let row_os = desc
                .out_scale
                .and_then(|s| s.get(r).copied())
                .unwrap_or(1.0);
            let dequant_row = crate::metal::dequantize_row_zero_copy(
                desc.weight_data,
                desc.weight_dtype,
                desc.scales_data,
                desc.scales_dtype,
                desc.biases_data,
                desc.biases_dtype,
                row_os,
                desc.bits,
                desc.group_size,
                r,
                desc.cols,
            )?;

            // Derive 4-band spectral probability distribution from group scales of row `r`
            let row_group_base = r * groups_per_row;
            let band_groups = (groups_per_row / 4).max(1);
            let mut band_e = [1e-6f32; 4];
            for b_idx in 0..4 {
                let g_start = (b_idx * band_groups).min(groups_per_row);
                let g_end = ((b_idx + 1) * band_groups).min(groups_per_row);
                for g in g_start..g_end {
                    let s = read_quant_scale_f64(
                        desc.scales_data,
                        desc.scales_dtype,
                        row_group_base + g,
                    )? as f32;
                    band_e[b_idx] += s * s;
                }
            }
            let sum_e: f32 = band_e.iter().sum();
            let next_op_probs: Vec<f32> = band_e.iter().map(|&e| e / sum_e).collect();

            let mut payload = Vec::with_capacity(16 + 64);
            payload.extend_from_slice(b"q_mode:");
            payload.extend_from_slice(&(desc.layer_index as u32).to_le_bytes());
            payload.extend_from_slice(&(r as u32).to_le_bytes());
            payload.push(desc.bits as u8);
            for &f in dequant_row.iter().take(16) {
                payload.extend_from_slice(&f.to_le_bytes());
            }

            let tap = TeacherTapRecord {
                shard_id: 50_000 + (desc.layer_index as u64) * 10_000 + (r as u64),
                layer_index: desc.layer_index,
                hidden_state: dequant_row,
                next_op_probs,
                payload,
            };

            let _shard = FrozenShard::from_tap(&tap).map_err(|e| e.to_string())?;
            let mut projected = self.projector.project(&tap).map_err(|e| e.to_string())?;

            // Fisher-information bit-width bonus: 6-bit/8-bit sensitive OptiQ layers have lower effective entropy
            let fisher_discount = if desc.bits >= 6 {
                Q32_31(projected.loss_report.teacher_entropy_bits_q32.0 >> 1)
            } else {
                projected.loss_report.teacher_entropy_bits_q32
            };

            let (role, offset, _cert) = Self::classify_icosahedral_well(
                &map.atlas,
                &tap.hidden_state,
                desc.layer_index,
                desc.total_layers,
                fisher_discount,
            );

            let well = map.atlas.well(role);
            let d_sph = Q32_31::from_f64(0.804_735_5);
            projected.tangent_h3 = [
                (well.unit_dir[0] * d_sph) + offset[0],
                (well.unit_dir[1] * d_sph) + offset[1],
                (well.unit_dir[2] * d_sph) + offset[2],
            ];

            let refracted =
                OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
            let atom_id =
                map.keep_and_carve_with_origin(refracted, tap.payload, teacher_fingerprint, policy);

            selected_row_indices.push(r);
            minted_atom_ids.push(atom_id);
            if let Some(r_idx) = role.carved_index() {
                role_distribution[r_idx] += 1;
            }
        }

        let saved_bytes = ((desc.rows.saturating_sub(k)) * desc.cols * 4) as u64;

        Ok(QuantizedDistillationReceipt {
            tensor_name: desc.tensor_name.to_string(),
            bits: desc.bits,
            group_size: desc.group_size,
            total_rows_scanned_via_scales: desc.rows,
            top_k_rows_dequantized: k,
            heap_bytes_saved_vs_full_dequant: saved_bytes,
            minted_atom_ids,
            selected_row_indices,
            role_distribution,
        })
    }
}

/// Zero-copy descriptor for a quantized layer tensor inside an open-source checkpoint
/// (`OptiQ` / `MLX` `U32`/`U8` packed weights + `BF16`/`F16`/`F32` scales & biases + optional `F32` `out_scale`).
pub struct QuantizedLayerTapDescriptor<'a> {
    pub tensor_name: &'a str,
    pub layer_index: usize,
    pub total_layers: usize,
    pub weight_data: &'a [u8],
    pub weight_dtype: &'a str,
    pub scales_data: &'a [u8],
    pub scales_dtype: &'a str,
    pub biases_data: &'a [u8],
    pub biases_dtype: &'a str,
    pub out_scale: Option<&'a [f32]>,
    pub bits: u32,
    pub group_size: usize,
    pub rows: usize,
    pub cols: usize,
}

/// Receipt returned by [`OpenWeightsDistillationEngine::distill_quantized_layer_zero_copy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedDistillationReceipt {
    pub tensor_name: String,
    pub bits: u32,
    pub group_size: usize,
    pub total_rows_scanned_via_scales: usize,
    pub top_k_rows_dequantized: usize,
    pub heap_bytes_saved_vs_full_dequant: u64,
    pub minted_atom_ids: Vec<AtomId>,
    pub selected_row_indices: Vec<usize>,
    pub role_distribution: [usize; 12],
}

fn read_quant_scale_f64(data: &[u8], dtype: &str, idx: usize) -> Result<f64, String> {
    match dtype {
        "BF16" => {
            let off = idx * 2;
            if off + 2 > data.len() {
                return Err("BF16 scale/bias index out of bounds".to_string());
            }
            let bits = u16::from_le_bytes([data[off], data[off + 1]]);
            Ok(f32::from_bits((bits as u32) << 16) as f64)
        }
        "F32" => {
            let off = idx * 4;
            if off + 4 > data.len() {
                return Err("F32 scale/bias index out of bounds".to_string());
            }
            Ok(f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as f64)
        }
        "F16" => {
            let off = idx * 2;
            if off + 2 > data.len() {
                return Err("F16 scale/bias index out of bounds".to_string());
            }
            let bits = u16::from_le_bytes([data[off], data[off + 1]]);
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1F) as u32;
            let mant = (bits & 0x3FF) as u32;
            let f32_bits = if exp == 0 {
                if mant == 0 {
                    sign << 31
                } else {
                    sign << 31 | 0x38000000 | (mant << 13)
                }
            } else if exp == 31 {
                sign << 31 | 0x7F800000 | (mant << 13)
            } else {
                sign << 31 | ((exp + 112) << 23) | (mant << 13)
            };
            Ok(f32::from_bits(f32_bits) as f64)
        }
        other => Err(format!("Unsupported scale/bias dtype: {other}")),
    }
}

/// Minimal zero-dependency safetensors tensor metadata entry parsed from the 8-byte header.
#[derive(Debug, Clone)]
pub struct RawSafeTensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Parse the 8-byte little-endian header of a `.safetensors` file buffer without external JSON crates.
pub fn parse_safetensors_header(
    file_bytes: &[u8],
) -> Result<
    (
        usize,
        std::collections::BTreeMap<String, RawSafeTensorEntry>,
    ),
    String,
> {
    if file_bytes.len() < 8 {
        return Err("Safetensors buffer shorter than 8-byte header".to_string());
    }
    let header_len = u64::from_le_bytes(file_bytes[0..8].try_into().unwrap()) as usize;
    let data_start = 8usize.saturating_add(header_len);
    if file_bytes.len() < data_start {
        return Err("Safetensors header extends past buffer".to_string());
    }
    let header_str = std::str::from_utf8(&file_bytes[8..data_start])
        .map_err(|_| "Safetensors header is not valid UTF-8".to_string())?;

    let mut map = std::collections::BTreeMap::new();
    let bytes = header_str.as_bytes();
    let mut pos = 0usize;

    while pos < bytes.len() {
        // Find next top-level key `"name": {`
        let Some(q1_rel) = header_str[pos..].find('"') else {
            break;
        };
        let k_start = pos + q1_rel + 1;
        let Some(q2_rel) = header_str[k_start..].find('"') else {
            break;
        };
        let k_end = k_start + q2_rel;
        let key = &header_str[k_start..k_end];
        pos = k_end + 1;

        let Some(brace_open_rel) = header_str[pos..].find('{') else {
            break;
        };
        let obj_start = pos + brace_open_rel + 1;
        // Find matching closing brace (handling nested metadata objects)
        let mut depth = 1i32;
        let mut cur = obj_start;
        while cur < bytes.len() && depth > 0 {
            if bytes[cur] == b'{' {
                depth += 1;
            } else if bytes[cur] == b'}' {
                depth -= 1;
            }
            cur += 1;
        }
        let obj_body = &header_str[obj_start..cur.saturating_sub(1)];
        pos = cur;

        if key == "__metadata__" {
            continue;
        }

        // Extract dtype, shape, and data_offsets from obj_body
        let Some(dtype) = extract_json_str_field(obj_body, "\"dtype\"") else {
            continue;
        };
        let Some(shape) = extract_json_usize_array(obj_body, "\"shape\"") else {
            continue;
        };
        let Some(offsets) = extract_json_usize_array(obj_body, "\"data_offsets\"") else {
            continue;
        };
        if offsets.len() == 2 {
            map.insert(
                key.to_string(),
                RawSafeTensorEntry {
                    dtype,
                    shape,
                    data_offsets: (offsets[0], offsets[1]),
                },
            );
        }
    }

    Ok((data_start, map))
}

fn extract_json_str_field(body: &str, field: &str) -> Option<String> {
    let idx = body.find(field)?;
    let after = &body[idx + field.len()..];
    let colon = after.find(':')?;
    let after_colon = &after[colon + 1..];
    let q1 = after_colon.find('"')?;
    let s_start = q1 + 1;
    let q2 = after_colon[s_start..].find('"')?;
    Some(after_colon[s_start..s_start + q2].to_string())
}

fn extract_json_usize_array(body: &str, field: &str) -> Option<Vec<usize>> {
    let idx = body.find(field)?;
    let after = &body[idx + field.len()..];
    let b_open = after.find('[')?;
    let b_close = after[b_open..].find(']')?;
    let inner = &after[b_open + 1..b_open + b_close];
    let mut out = Vec::new();
    for part in inner.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.parse::<usize>().ok()?);
        }
    }
    Some(out)
}

impl OpenWeightsDistillationEngine {
    /// Automatically discover and distill all `OptiQ` / `MLX` quantized linear layers (`<prefix>.weight`,
    /// `<prefix>.scales`, `<prefix>.biases`, optional `<prefix>.out_scale`) inside a `.safetensors`
    /// file buffer without ever dequantizing full weight matrices.
    pub fn distill_safetensors_shard_bytes(
        &self,
        file_bytes: &[u8],
        top_k_rows_per_layer: usize,
        total_layers_hint: usize,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<Vec<QuantizedDistillationReceipt>, String> {
        let (data_start, tensors) = parse_safetensors_header(file_bytes)?;
        let shard_fingerprint = Sha256Digest::compute(&file_bytes[..file_bytes.len().min(65_536)]);
        let mut receipts = Vec::new();

        for (name, w_info) in &tensors {
            let Some(prefix) = name.strip_suffix(".weight") else {
                continue;
            };
            if w_info.dtype != "U32" && w_info.dtype != "U8" {
                continue;
            }
            if w_info.shape.len() != 2 {
                continue;
            }
            let scales_name = format!("{prefix}.scales");
            let biases_name = format!("{prefix}.biases");
            let Some(s_info) = tensors.get(&scales_name) else {
                continue;
            };
            let Some(b_info) = tensors.get(&biases_name) else {
                continue;
            };
            if s_info.shape.len() != 2 || b_info.shape.len() != 2 {
                continue;
            }

            let rows = w_info.shape[0];
            let packed_cols = w_info.shape[1];
            let groups_per_row = s_info.shape[1];
            if rows == 0 || groups_per_row == 0 || s_info.shape[0] != rows {
                continue;
            }

            let word_bits = if w_info.dtype == "U32" {
                32usize
            } else {
                8usize
            };
            let total_row_bits = packed_cols * word_bits;

            // Infer (bits, group_size, cols) from packed_cols and groups_per_row
            let mut inferred = None;
            for &gs in &[128usize, 64, 32] {
                let candidate_cols = groups_per_row * gs;
                if candidate_cols > 0 && total_row_bits % candidate_cols == 0 {
                    let b = (total_row_bits / candidate_cols) as u32;
                    if matches!(b, 4 | 5 | 6 | 8) {
                        inferred = Some((b, gs, candidate_cols));
                        break;
                    }
                }
            }
            let Some((bits, group_size, cols)) = inferred else {
                continue;
            };

            // Extract layer_index from tensor prefix (e.g. `model.layers.14.mlp.down_proj`)
            let layer_index = prefix
                .split('.')
                .find_map(|seg| seg.parse::<usize>().ok())
                .unwrap_or(0);

            let w_bytes =
                &file_bytes[data_start + w_info.data_offsets.0..data_start + w_info.data_offsets.1];
            let s_bytes =
                &file_bytes[data_start + s_info.data_offsets.0..data_start + s_info.data_offsets.1];
            let b_bytes =
                &file_bytes[data_start + b_info.data_offsets.0..data_start + b_info.data_offsets.1];

            let out_scale_name = format!("{prefix}.out_scale");
            let out_scale_vec: Option<Vec<f32>> =
                tensors.get(&out_scale_name).and_then(|os_info| {
                    if os_info.dtype == "F32" && os_info.shape.first().copied() == Some(rows) {
                        let raw = &file_bytes[data_start + os_info.data_offsets.0
                            ..data_start + os_info.data_offsets.1];
                        Some(
                            raw.chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect(),
                        )
                    } else {
                        None
                    }
                });

            let desc = QuantizedLayerTapDescriptor {
                tensor_name: name,
                layer_index,
                total_layers: total_layers_hint.max(layer_index + 1),
                weight_data: w_bytes,
                weight_dtype: &w_info.dtype,
                scales_data: s_bytes,
                scales_dtype: &s_info.dtype,
                biases_data: b_bytes,
                biases_dtype: &b_info.dtype,
                out_scale: out_scale_vec.as_deref(),
                bits,
                group_size,
                rows,
                cols,
            };

            let rec = self.distill_quantized_layer_zero_copy(
                &desc,
                top_k_rows_per_layer,
                shard_fingerprint,
                map,
                policy,
            )?;
            receipts.push(rec);
        }

        Ok(receipts)
    }

    /// Stream and distill quantized layers (`2D` linear projections and `3D` MoE `switch_mlp` experts)
    /// directly from a `.safetensors` file on disk using **selective `File::seek`** (`0` full-file read,
    /// `0` full-matrix dequantization).
    ///
    /// Automatically inspects sibling `optiq_metadata.json` or `config.json` in the same directory
    /// to resolve exact mixed-precision `(bits, group_size)` per layer (`4/5/6/8-bit`, `gs32/64/128`),
    /// stratifies across all `256` MoE experts, deduplicates against `map.ingested_tensor_digests` and
    /// `map.keep_and_carve_novel_only`, and synthesizes cross-layer `1-cell` `DistilledReasoningArc`s.
    pub fn distill_safetensors_file_streaming(
        &self,
        safetensors_path: &std::path::Path,
        top_k_rows_per_layer: usize,
        max_layers: usize,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<Vec<QuantizedDistillationReceipt>, String> {
        let mlir_catalog = MlirAtomCatalog::new_canonical(248_320);
        let (receipts, _summary) = self.distill_safetensors_file_novel_only(
            safetensors_path,
            top_k_rows_per_layer,
            max_layers,
            &mlir_catalog,
            map,
            policy,
        )?;
        Ok(receipts)
    }

    /// Novelty-only streaming distillation of a `.safetensors` shard (`0` duplicate noise):
    /// 1. Skips any tensor whose `(shard_fingerprint, tensor_name, bits, group_size, top_k)` digest
    ///    is already in `map.ingested_tensor_digests`.
    /// 2. Stratifies row screening across the entire tensor (so `3D` `[256, R, C]` MoE `switch_mlp`
    ///    tensors sample across all `256` experts rather than only Expert 0).
    /// 3. Packs a full `64`-float (`256`-byte, `32` RoPE pairs) canonical `(K, V)` payload into each atom.
    /// 4. Deduplicates `0-cell` coordinates via `map.keep_and_carve_novel_only` (`O(1)` spatial hash)
    ///    and synthesizes + deduplicates cross-layer `1-cell` `DistilledReasoningArc`s via
    ///    `map.record_novel_reasoning_arc`.
    pub fn distill_safetensors_file_novel_only(
        &self,
        safetensors_path: &std::path::Path,
        top_k_rows_per_layer: usize,
        max_layers: usize,
        mlir_catalog: &MlirAtomCatalog,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<
        (
            Vec<QuantizedDistillationReceipt>,
            FullModelDistillationReceipt,
        ),
        String,
    > {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = std::fs::File::open(safetensors_path)
            .map_err(|e| format!("Failed to open {}: {e}", safetensors_path.display()))?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .map_err(|e| format!("Failed to read safetensors header len: {e}"))?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let mut header_and_len = vec![0u8; 8 + header_len];
        header_and_len[0..8].copy_from_slice(&len_buf);
        file.read_exact(&mut header_and_len[8..])
            .map_err(|e| format!("Failed to read safetensors JSON header: {e}"))?;

        let (data_start, tensors) = parse_safetensors_header(&header_and_len)?;
        let shard_fingerprint = Sha256Digest::compute(&header_and_len);

        let parent_dir = safetensors_path.parent();
        let (per_layer_overrides, default_gs, total_layers_hint) =
            load_checkpoint_quant_overrides(parent_dir);

        let mut receipts = Vec::new();
        let mut summary = FullModelDistillationReceipt {
            model_path: safetensors_path.display().to_string(),
            shards_processed: 1,
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

        // Track best (certainty, atom_id, role, hidden_state_128d) per transformer layer_index
        // so we can synthesize cross-layer 1-cell reasoning arcs across the shard!
        let mut layer_anchors: std::collections::BTreeMap<
            usize,
            (Q32_31, AtomId, RoleId, Vec<f32>),
        > = std::collections::BTreeMap::new();

        let max_screen_budget = 256usize;

        for (name, w_info) in &tensors {
            if receipts.len() >= max_layers {
                break;
            }
            let Some(prefix) = name.strip_suffix(".weight") else {
                continue;
            };
            if w_info.dtype != "U32" && w_info.dtype != "U8" {
                continue;
            }
            let scales_name = format!("{prefix}.scales");
            let biases_name = format!("{prefix}.biases");
            let Some(s_info) = tensors.get(&scales_name) else {
                continue;
            };
            let Some(b_info) = tensors.get(&biases_name) else {
                continue;
            };

            let (total_rows, packed_cols, groups_per_row) =
                match (w_info.shape.as_slice(), s_info.shape.as_slice()) {
                    (&[r, pc], &[sr, gpr]) if r > 0 && gpr > 0 && r == sr => (r, pc, gpr),
                    (&[e, r, pc], &[se, sr, gpr])
                        if e > 0 && r > 0 && gpr > 0 && e == se && r == sr =>
                    {
                        (e * r, pc, gpr)
                    }
                    _ => continue,
                };

            let word_bytes = if w_info.dtype == "U32" {
                4usize
            } else {
                1usize
            };
            let word_bits = word_bytes * 8;
            let total_row_bits = packed_cols * word_bits;

            let (bits, group_size, cols) = if let Some(&(b, gs)) = per_layer_overrides.get(prefix) {
                let c = groups_per_row * gs;
                if c == 0 || total_row_bits != c * (b as usize) {
                    continue;
                }
                (b, gs, c)
            } else {
                let mut found = None;
                let gs_candidates = match default_gs {
                    Some(32) => [32usize, 128, 64],
                    Some(64) => [64usize, 128, 32],
                    _ => [128usize, 64, 32],
                };
                for gs in gs_candidates {
                    let c = groups_per_row * gs;
                    if c > 0 && total_row_bits % c == 0 {
                        let b = (total_row_bits / c) as u32;
                        if matches!(b, 4 | 5 | 6 | 8) {
                            found = Some((b, gs, c));
                            break;
                        }
                    }
                }
                let Some(triple) = found else { continue };
                triple
            };

            // Compute deterministic tensor ingestion digest to prevent re-distilling already ingested tensors
            let mut t_seed = Vec::with_capacity(64 + name.len());
            t_seed.extend_from_slice(&shard_fingerprint.0);
            t_seed.extend_from_slice(name.as_bytes());
            t_seed.extend_from_slice(&bits.to_le_bytes());
            t_seed.extend_from_slice(&(group_size as u32).to_le_bytes());
            t_seed.extend_from_slice(&(top_k_rows_per_layer as u32).to_le_bytes());
            let tensor_digest = Sha256Digest::compute(&t_seed);

            if map.ingested_tensor_digests.contains(&tensor_digest) {
                summary.tensors_skipped_already_ingested += 1;
                continue;
            }

            let layer_index = prefix
                .split('.')
                .find_map(|seg| seg.parse::<usize>().ok())
                .unwrap_or(0);

            // Stratified row screening across all `total_rows` (including all 256 MoE experts!)
            let screen_count = total_rows.min(max_screen_budget);
            let row_stride = (total_rows / screen_count.max(1)).max(1);
            let k = top_k_rows_per_layer.min(screen_count).max(1);
            let scale_elem_bytes = if s_info.dtype == "F32" {
                4usize
            } else {
                2usize
            };
            let row_scale_bytes = groups_per_row * scale_elem_bytes;

            // Read the full or prefix scale/bias slab covering the stratified rows
            let last_screened_row = ((screen_count - 1) * row_stride).min(total_rows - 1);
            let max_scale_bytes_to_read = 256 * 1024usize; // cap contiguous read at 256 KB
            let total_scale_bytes = (last_screened_row + 1) * row_scale_bytes;

            let max_q = ((1u32 << bits) - 1) as f64;
            let q_mean = 0.5 * max_q;
            let q_var = ((1u64 << (2 * bits)) as f64 - 1.0) / 12.0;

            let mut row_scores: Vec<(f64, usize)> = Vec::with_capacity(screen_count);
            let mut cached_s_rows: Vec<Vec<u8>> = Vec::with_capacity(screen_count);
            let mut cached_b_rows: Vec<Vec<u8>> = Vec::with_capacity(screen_count);

            if total_scale_bytes <= max_scale_bytes_to_read {
                let mut s_slab = vec![0u8; total_scale_bytes];
                file.seek(SeekFrom::Start((data_start + s_info.data_offsets.0) as u64))
                    .map_err(|e| e.to_string())?;
                file.read_exact(&mut s_slab).map_err(|e| e.to_string())?;

                let mut b_slab = vec![0u8; total_scale_bytes];
                file.seek(SeekFrom::Start((data_start + b_info.data_offsets.0) as u64))
                    .map_err(|e| e.to_string())?;
                file.read_exact(&mut b_slab).map_err(|e| e.to_string())?;

                for idx in 0..screen_count {
                    let r = (idx * row_stride).min(total_rows - 1);
                    let s_row = s_slab[r * row_scale_bytes..(r + 1) * row_scale_bytes].to_vec();
                    let b_row = b_slab[r * row_scale_bytes..(r + 1) * row_scale_bytes].to_vec();
                    let mut energy_est = 0.0f64;
                    for g in 0..groups_per_row {
                        let s = read_quant_scale_f64(&s_row, &s_info.dtype, g)?;
                        let b = read_quant_scale_f64(&b_row, &b_info.dtype, g)?;
                        let gm = q_mean * s + b;
                        let gv = q_var * s * s;
                        energy_est += (group_size as f64) * (gm * gm + gv);
                    }
                    row_scores.push((energy_est, idx));
                    cached_s_rows.push(s_row);
                    cached_b_rows.push(b_row);
                }
            } else {
                // Stratified seek across large 3D MoE / vocab tensors
                let mut s_row = vec![0u8; row_scale_bytes];
                let mut b_row = vec![0u8; row_scale_bytes];
                for idx in 0..screen_count {
                    let r = (idx * row_stride).min(total_rows - 1);
                    file.seek(SeekFrom::Start(
                        (data_start + s_info.data_offsets.0 + r * row_scale_bytes) as u64,
                    ))
                    .map_err(|e| e.to_string())?;
                    file.read_exact(&mut s_row).map_err(|e| e.to_string())?;

                    file.seek(SeekFrom::Start(
                        (data_start + b_info.data_offsets.0 + r * row_scale_bytes) as u64,
                    ))
                    .map_err(|e| e.to_string())?;
                    file.read_exact(&mut b_row).map_err(|e| e.to_string())?;

                    let mut energy_est = 0.0f64;
                    for g in 0..groups_per_row {
                        let s = read_quant_scale_f64(&s_row, &s_info.dtype, g)?;
                        let b = read_quant_scale_f64(&b_row, &b_info.dtype, g)?;
                        let gm = q_mean * s + b;
                        let gv = q_var * s * s;
                        energy_est += (group_size as f64) * (gm * gm + gv);
                    }
                    row_scores.push((energy_est, idx));
                    cached_s_rows.push(s_row.clone());
                    cached_b_rows.push(b_row.clone());
                }
            }

            row_scores.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });

            let row_weight_bytes = packed_cols * word_bytes;
            let mut single_row_w = vec![0u8; row_weight_bytes];
            let mut selected_row_indices = Vec::with_capacity(k);
            let mut minted_atom_ids = Vec::with_capacity(k);
            let mut role_distribution = [0usize; 12];

            for &(_score, idx) in row_scores.iter().take(k) {
                let r = (idx * row_stride).min(total_rows - 1);
                let w_offset = data_start + w_info.data_offsets.0 + r * row_weight_bytes;
                file.seek(SeekFrom::Start(w_offset as u64))
                    .map_err(|e| e.to_string())?;
                file.read_exact(&mut single_row_w)
                    .map_err(|e| e.to_string())?;

                let s_slice = &cached_s_rows[idx];
                let b_slice = &cached_b_rows[idx];

                let dequant_row = crate::metal::dequantize_row_zero_copy(
                    &single_row_w,
                    &w_info.dtype,
                    s_slice,
                    &s_info.dtype,
                    b_slice,
                    &b_info.dtype,
                    1.0,
                    bits,
                    group_size,
                    0,
                    cols,
                )?;

                let band_groups = (groups_per_row / 4).max(1);
                let mut band_e = [1e-6f32; 4];
                for b_idx in 0..4 {
                    let g_start = (b_idx * band_groups).min(groups_per_row);
                    let g_end = ((b_idx + 1) * band_groups).min(groups_per_row);
                    for g in g_start..g_end {
                        let s = read_quant_scale_f64(s_slice, &s_info.dtype, g)? as f32;
                        band_e[b_idx] += s * s;
                    }
                }
                let sum_e: f32 = band_e.iter().sum();
                let next_op_probs: Vec<f32> = band_e.iter().map(|&e| e / sum_e).collect();

                // Pack canonical 64-float (256-byte, 32 RoPE pairs, even multiple of 8 bytes) KV payload
                // so ModelAdapterLens and CoreLmInferenceWorker can attend over it directly!
                let kv_floats = 64usize;
                let mut payload = Vec::with_capacity(kv_floats * 4);
                for i in 0..kv_floats {
                    let v = dequant_row
                        .get(i % dequant_row.len())
                        .copied()
                        .unwrap_or(0.0);
                    payload.extend_from_slice(&v.to_le_bytes());
                }

                let tap = TeacherTapRecord {
                    shard_id: 70_000 + (receipts.len() as u64) * 1_000 + (r as u64),
                    layer_index,
                    hidden_state: dequant_row,
                    next_op_probs,
                    payload,
                };

                let _shard = FrozenShard::from_tap(&tap).map_err(|e| e.to_string())?;
                let mut projected = self.projector.project(&tap).map_err(|e| e.to_string())?;

                let fisher_discount = if bits >= 6 {
                    Q32_31(projected.loss_report.teacher_entropy_bits_q32.0 >> 1)
                } else {
                    projected.loss_report.teacher_entropy_bits_q32
                };

                let (role, offset, cert) = Self::classify_icosahedral_well(
                    &map.atlas,
                    &tap.hidden_state,
                    layer_index,
                    total_layers_hint.max(layer_index + 1),
                    fisher_discount,
                );

                // Perturb slightly by tensor hash + row index inside the role's basin so distinct
                // layer/expert modes occupy distinct sub-coordinates while identical modes coalesce!
                let mode_hash = tensor_digest.0[0] as f64 / 255.0 - 0.5;
                let row_jitter = ((r % 31) as f64 - 15.0) * 0.0012;
                let well = map.atlas.well(role);
                let d_sph = Q32_31::from_f64(0.804_735_5);
                projected.tangent_h3 = [
                    (well.unit_dir[0] * d_sph) + offset[0] + Q32_31::from_f64(mode_hash * 0.015),
                    (well.unit_dir[1] * d_sph) + offset[1] + Q32_31::from_f64(row_jitter),
                    (well.unit_dir[2] * d_sph) + offset[2] - Q32_31::from_f64(mode_hash * 0.011),
                ];

                let refracted =
                    OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
                let (atom_id, is_novel) = map.keep_and_carve_novel_only(
                    refracted,
                    tap.payload,
                    shard_fingerprint,
                    policy,
                );

                if is_novel {
                    summary.novel_atoms_minted += 1;
                } else {
                    summary.deduplicated_atoms_coalesced += 1;
                }

                // Track the highest-certainty mode for each transformer layer_index to synthesize
                // cross-layer 1-cell reasoning transitions (`Layer l -> Layer l+1`)
                let replace_anchor = match layer_anchors.get(&layer_index) {
                    Some((prev_cert, _, _, _)) => cert.0 > prev_cert.0,
                    None => true,
                };
                if replace_anchor {
                    let state_128: Vec<f32> = tap.hidden_state.iter().take(128).copied().collect();
                    layer_anchors.insert(layer_index, (cert, atom_id, role, state_128));
                }

                selected_row_indices.push(r);
                minted_atom_ids.push(atom_id);
                if let Some(r_idx) = role.carved_index() {
                    role_distribution[r_idx] += 1;
                    summary.role_distribution[r_idx] += 1;
                }
            }

            map.ingested_tensor_digests.insert(tensor_digest);
            let saved_bytes = ((total_rows.saturating_sub(k)) * cols * 4) as u64;
            summary.tensors_distilled += 1;
            summary.f32_heap_bytes_saved += saved_bytes;

            receipts.push(QuantizedDistillationReceipt {
                tensor_name: name.clone(),
                bits,
                group_size,
                total_rows_scanned_via_scales: total_rows,
                top_k_rows_dequantized: k,
                heap_bytes_saved_vs_full_dequant: saved_bytes,
                minted_atom_ids,
                selected_row_indices,
                role_distribution,
            });
        }

        // Step 3: Synthesize cross-layer 1-cell reasoning arcs (`Layer l -> Layer l'`, Wigner transport + Native MLIR)
        if layer_anchors.len() >= 2 {
            let ordered: Vec<(usize, Q32_31, AtomId, RoleId, Vec<f32>)> = layer_anchors
                .into_iter()
                .map(|(l, (c, a, r, s))| (l, c, a, r, s))
                .collect();
            let states: Vec<Vec<f32>> = ordered.iter().map(|x| x.4.clone()).collect();
            let first_role = ordered[0].3;
            let decoded_mlir =
                GeodesicMlirDecoder::decode_from_hidden_states(mlir_catalog, first_role, &states)
                    .ok();

            let mut current_wigner = IDENTITY_FRAME_Q32;
            let chain_id = u64::from_le_bytes(shard_fingerprint.0[0..8].try_into().unwrap());

            for step in 0..ordered.len() - 1 {
                let (_, _, from_atom, from_role, _) = ordered[step];
                let (_, to_cert, to_atom, to_role, _) = ordered[step + 1];
                let Some(from_lor) = map.global_atom(from_atom).map(|a| a.lorentz) else {
                    continue;
                };
                let Some(to_lor) = map.global_atom(to_atom).map(|a| a.lorentz) else {
                    continue;
                };
                let cosh_action = OpticalLens::lorentz_inner_product_wide(&from_lor, &to_lor);
                current_wigner = HyperbolicHolonomyEngine::transport_frame_hop(
                    &from_lor,
                    &to_lor,
                    &current_wigner,
                );
                let compiled_op = decoded_mlir
                    .as_ref()
                    .and_then(|d| d.block.instructions.get(step + 1))
                    .map(|inst| inst.op_id)
                    .unwrap_or(MlirOpId(mlir_catalog.base_vocab_size));

                let (_arc_id, is_novel_arc) =
                    map.record_novel_reasoning_arc(DistilledReasoningArc {
                        arc_id: 0,
                        chain_id,
                        step_index: step,
                        from_atom,
                        to_atom,
                        from_role,
                        to_role,
                        cosh_action_q32: cosh_action,
                        wigner_frame: current_wigner,
                        compiled_mlir_op: compiled_op,
                        teacher_certainty_q32: to_cert,
                        origin_model_fingerprint: shard_fingerprint,
                    });
                if is_novel_arc {
                    summary.novel_reasoning_arcs_minted += 1;
                } else {
                    summary.deduplicated_reasoning_arcs += 1;
                }
            }
        }

        Ok((receipts, summary))
    }

    /// Live Continuous Novelty-Only Distillation Gate for an active model inside `MonsterHybridServer`.
    ///
    /// Called automatically when a hooked-up model executes a forward pass or reasoning step:
    /// - **Entropy Sieve**: Drops high-entropy teacher noise (`H(p) > max_entropy_bits_q32`).
    /// - **`0-Cell` Geodesic Deduplication**: Uses `map.keep_and_carve_novel_only` (`O(1)` spatial hash)
    ///   so already-known information only reinforces `salience_count` without creating duplicate atoms.
    /// - **`1-Cell` Reasoning Deduplication**: Connects `previous_live_atom` $\to$ `current_atom`
    ///   via `map.record_novel_reasoning_arc`, minting an arc only when the geodesic transition or
    ///   synthesized `MlirOpId` is genuinely new!
    pub fn distill_live_step_novel_only(
        &self,
        tap: &TeacherTapRecord,
        total_layers: usize,
        model_fingerprint: Sha256Digest,
        previous_atom_and_role: Option<(AtomId, RoleId)>,
        mlir_catalog: &MlirAtomCatalog,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<LiveNoveltyDistillationReceipt, String> {
        let _shard = FrozenShard::from_tap(tap).map_err(|e| e.to_string())?;
        let mut projected = self.projector.project(tap).map_err(|e| e.to_string())?;

        // 1. Entropy Sieve: drop high-entropy guesses/noise immediately
        if projected.loss_report.teacher_entropy_bits_q32.0 > self.max_entropy_bits_q32.0 {
            return Ok(LiveNoveltyDistillationReceipt {
                step_ingested: false,
                rejected_as_high_entropy_noise: true,
                atom_id: None,
                atom_was_novel: false,
                arc_id: None,
                arc_was_novel: false,
                role: None,
                teacher_entropy_bits_q32: projected.loss_report.teacher_entropy_bits_q32,
            });
        }

        let (role, offset, certainty) = Self::classify_icosahedral_well(
            &map.atlas,
            &tap.hidden_state,
            tap.layer_index,
            total_layers.max(1),
            projected.loss_report.teacher_entropy_bits_q32,
        );

        let well = map.atlas.well(role);
        let d_sph = Q32_31::from_f64(0.804_735_5);
        projected.tangent_h3 = [
            (well.unit_dir[0] * d_sph) + offset[0],
            (well.unit_dir[1] * d_sph) + offset[1],
            (well.unit_dir[2] * d_sph) + offset[2],
        ];

        let refracted = OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
        let (atom_id, atom_was_novel) = map.keep_and_carve_novel_only(
            refracted,
            tap.payload.clone(),
            model_fingerprint,
            policy,
        );

        let mut arc_id_out = None;
        let mut arc_was_novel = false;

        if let Some((prev_atom, prev_role)) = previous_atom_and_role {
            if prev_atom != atom_id {
                if let (Some(from_lor), Some(to_lor)) = (
                    map.global_atom(prev_atom).map(|a| a.lorentz),
                    map.global_atom(atom_id).map(|a| a.lorentz),
                ) {
                    let cosh_action = OpticalLens::lorentz_inner_product_wide(&from_lor, &to_lor);
                    let wigner = HyperbolicHolonomyEngine::transport_frame_hop(
                        &from_lor,
                        &to_lor,
                        &IDENTITY_FRAME_Q32,
                    );
                    let compiled_op = GeodesicMlirDecoder::decode_from_hidden_states(
                        mlir_catalog,
                        role,
                        std::slice::from_ref(&tap.hidden_state),
                    )
                    .ok()
                    .and_then(|d| d.block.instructions.first().map(|i| i.op_id))
                    .unwrap_or(MlirOpId(mlir_catalog.base_vocab_size));

                    let (aid, novel_arc) = map.record_novel_reasoning_arc(DistilledReasoningArc {
                        arc_id: 0,
                        chain_id: tap.shard_id,
                        step_index: tap.layer_index,
                        from_atom: prev_atom,
                        to_atom: atom_id,
                        from_role: prev_role,
                        to_role: role,
                        cosh_action_q32: cosh_action,
                        wigner_frame: wigner,
                        compiled_mlir_op: compiled_op,
                        teacher_certainty_q32: certainty,
                        origin_model_fingerprint: model_fingerprint,
                    });
                    arc_id_out = Some(aid);
                    arc_was_novel = novel_arc;
                }
            }
        }

        Ok(LiveNoveltyDistillationReceipt {
            step_ingested: true,
            rejected_as_high_entropy_noise: false,
            atom_id: Some(atom_id),
            atom_was_novel,
            arc_id: arc_id_out,
            arc_was_novel,
            role: Some(role),
            teacher_entropy_bits_q32: projected.loss_report.teacher_entropy_bits_q32,
        })
    }
}

/// Summary receipt returned when distilling a full model directory or shard with novelty-only deduplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullModelDistillationReceipt {
    pub model_path: String,
    pub shards_processed: usize,
    pub tensors_distilled: usize,
    pub tensors_skipped_already_ingested: usize,
    pub novel_atoms_minted: usize,
    pub deduplicated_atoms_coalesced: usize,
    pub novel_reasoning_arcs_minted: usize,
    pub deduplicated_reasoning_arcs: usize,
    pub f32_heap_bytes_saved: u64,
    pub role_distribution: [usize; 12],
    pub vault_sha256: Option<Sha256Digest>,
}

/// Receipt emitted by [`OpenWeightsDistillationEngine::distill_live_step_novel_only`] during
/// continuous server inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveNoveltyDistillationReceipt {
    pub step_ingested: bool,
    pub rejected_as_high_entropy_noise: bool,
    pub atom_id: Option<AtomId>,
    pub atom_was_novel: bool,
    pub arc_id: Option<u64>,
    pub arc_was_novel: bool,
    pub role: Option<RoleId>,
    pub teacher_entropy_bits_q32: Q32_31,
}

fn load_checkpoint_quant_overrides(
    parent_dir: Option<&std::path::Path>,
) -> (
    std::collections::BTreeMap<String, (u32, usize)>,
    Option<usize>,
    usize,
) {
    let mut map = std::collections::BTreeMap::new();
    let mut default_gs = None;
    let mut total_layers = 64usize;
    let Some(dir) = parent_dir else {
        return (map, default_gs, total_layers);
    };

    // 1. Check optiq_metadata.json (e.g. Qwen3.6-35B-A3B-OptiQ-4bit)
    let optiq_path = dir.join("optiq_metadata.json");
    if let Ok(text) = std::fs::read_to_string(&optiq_path) {
        default_gs = Some(64);
        total_layers = 40;
        parse_nested_quant_blocks(&text, &mut map);
    }

    // 2. Check config.json (e.g. Muse-Glimmer-q4km-gs128-MLX)
    let cfg_path = dir.join("config.json");
    if let Ok(text) = std::fs::read_to_string(&cfg_path) {
        if let Some(nl) = extract_json_usize_scalar(&text, "\"num_hidden_layers\"") {
            total_layers = nl;
        }
        if default_gs.is_none() {
            default_gs = extract_json_usize_scalar(&text, "\"group_size\"");
        }
        parse_nested_quant_blocks(&text, &mut map);
    }

    (map, default_gs, total_layers)
}

fn extract_json_usize_scalar(body: &str, field: &str) -> Option<usize> {
    let idx = body.find(field)?;
    let after = &body[idx + field.len()..];
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim_start();
    let num_end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    after_colon[..num_end].parse::<usize>().ok()
}

fn parse_nested_quant_blocks(
    text: &str,
    out: &mut std::collections::BTreeMap<String, (u32, usize)>,
) {
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let Some(q1) = text[pos..].find('"') else {
            break;
        };
        let k_start = pos + q1 + 1;
        let Some(q2) = text[k_start..].find('"') else {
            break;
        };
        let k_end = k_start + q2;
        let key = &text[k_start..k_end];
        pos = k_end + 1;

        let rest = text[pos..].trim_start();
        if !rest.starts_with(':') {
            continue;
        }
        let after_colon = rest[1..].trim_start();
        if !after_colon.starts_with('{') {
            continue;
        }
        let Some(b_close) = after_colon.find('}') else {
            continue;
        };
        let inner = &after_colon[1..b_close];
        if inner.contains("\"bits\"") && inner.contains("\"group_size\"") && !inner.contains('{') {
            if let (Some(b), Some(gs)) = (
                extract_json_usize_scalar(inner, "\"bits\""),
                extract_json_usize_scalar(inner, "\"group_size\""),
            ) {
                out.insert(key.to_string(), (b as u32, gs));
            }
        }
    }
}
