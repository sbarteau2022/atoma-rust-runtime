//! Map-Native Inference Engine (`Phase 1: One Layer` -> `Phase 2: Scale` -> `Phase 3: Full Weightless`
//! -> `Phase 4: Corpus-Wide Multi-Chart Expansion & Per-Layer Novelty Deduplication`).
//!
//! Solves the **Weight-to-Activation Functional Problem**:
//! Static atoms in `CarvedSwarmMap` hold weight/operator information (covectors), whereas transformer
//! layers output activations (contravariant vectors). To turn retrieved map atoms into an attention
//! output (`Δx_attn`) or full layer output (`Δx_layer = Δx_attn + Δx_mlp`) using the runtime hidden
//! state `x_in` as the query, each **Coupled Operator Atom** (`ATOMA_OP`) in the map stores a
//! **Bilinear Receptor–Effector Dyad**:
//!
//!   `D_m^(l) = (phi_m^in, ||x_m||_2, psi_m^attn, psi_m^mlp)`
//!
//! refracted into `(H³ × T²) / Γ₁₂` at coordinate `z_m = (u_m, theta_m)` from the input receptor `phi_m^in`.
//!
//! At inference time, given only the runtime hidden state `x_in ∈ R^H`:
//! 1. Refract `x_in` into `q(x_in) ∈ (H³ × T²) / Γ₁₂` and route to the specialist silo `RoleId`.
//! 2. Retrieve the top-K coupled operator atoms `{D_m^(l)}` in layer `l` from `CarvedSwarmMap`
//!    using zero-allocation receptor sketch scoring (`peek_receptor_dot`), decoding only the
//!    winning top-K effectors.
//! 3. Perform **Bilinear Geodesic Operator Contraction**:
//!    - Receptor inner-product functional: `alpha_m(x_in) = (||x_in||_2 / ||x_m||_2) * cos(Pi(x_in), phi_m^in)`
//!    - Geodesic kernel weight: `w_m(x_in) = softmax_m(-beta * (cosh d_H3(q, z_m) - 1) + gamma * cos_m + eta * phase_coh_m)`
//!    - Synthesized activation delta: `Δx^(l)(x_in) = sum_m w_m(x_in) * alpha_m(x_in) * psi_m`

use crate::atlas13::RoleId;
use crate::cache::geometric::AtomId;
use crate::compliance::OverlapPolicyMatrix;
use crate::distill_engine::{DistilledReasoningArc, OpenWeightsDistillationEngine};
use crate::geometry::fixed::Q32_31;
use crate::holonomy::IDENTITY_FRAME_Q32;
use crate::mlir_native::{MlirAtomCatalog, MlirOpId};
use crate::refraction::{OpticalLens, TangentProjector, TeacherTapRecord};
use crate::silo::CarvedSwarmMap;

/// Magic header for a Coupled Operator Atom payload stored inside `GlobalAtom::payload`.
pub const COUPLED_OPERATOR_MAGIC: &[u8; 8] = b"ATOMA_OP";

/// Receptor sketch dimension used for fast inner-product contraction `<Pi(x), phi_m^in>`.
pub const RECEPTOR_SKETCH_DIM: usize = 256;

/// What part of a transformer layer is substituted by Map-Native retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapSubstitutionMode {
    /// Substitute only the self-attention subblock (`x_mid = x_in + Δx_attn_map`),
    /// while still running the layer's feed-forward MLP (`Phase 1` & `Phase 2 Attention-Only`).
    AttentionOnly,
    /// Substitute the entire decoder layer (`x_out = x_in + Δx_attn_map + Δx_mlp_map`),
    /// executing zero attention weights and zero MLP weights (`Phase 2 Full-Layer` & `Phase 3 Weightless`).
    FullLayer,
    /// Diagnostic control: retrieve raw uncoupled 256d static weight atoms and fold-expand them
    /// without bilinear receptor-effector coupling (demonstrates why uncoupled weight atoms fail).
    RawUncoupledWeightAtoms,
    /// Static weight-dyad contraction (`(v_proj_row, o_proj_col)` / `(up_proj_row, down_proj_col)`)
    /// distilled purely from `.safetensors` matrices with zero forward-pass activation capture.
    StaticWeightDyadOnly,
}

/// Complete hidden-state & subblock activation capture from a single instrumented layer execution.
#[derive(Debug, Clone)]
pub struct InstrumentedLayerCapture {
    pub layer: usize,
    pub token_id: u32,
    /// ML-IR ID of the token that followed `token_id` in the distilled sequence
    /// (`None` = unknown; ingest falls back to `token_id`).
    pub next_token_id: Option<u32>,
    pub pos: usize,
    /// Input hidden state entering the layer (`H = 6656`).
    pub x_in: Vec<f32>,
    /// Post-normed attention residual delta (`x_mid - x_in`, length `H = 6656`).
    pub attn_delta: Vec<f32>,
    /// Mid-layer hidden state (`x_in + attn_delta`, length `H = 6656`).
    pub x_mid: Vec<f32>,
    /// Post-normed MLP residual delta (`x_out - x_mid`, length `H = 6656`).
    pub mlp_delta: Vec<f32>,
    /// Output hidden state leaving the layer (`x_mid + mlp_delta`, length `H = 6656`).
    pub x_out: Vec<f32>,
}

/// Maximum local Jacobian subspace rank per `ATOMA_OP_V3` chart (`R_m <= 8`).
pub const MAX_JACOBIAN_RANK: usize = 8;

/// Minimum orthogonal residual energy ratio (`||r_perp||_2 / ||phi_new||_2 >= 0.08`)
/// required for Modified Gram-Schmidt (MGS) to accrete a new tangent basis slot (`r = 2..=8`).
pub const DEFAULT_MGS_ORTHO_EPSILON: f64 = 0.08;

/// Maximum teacher entropy (`H(p) <= 2.75 bits`) permitted by the Entropy Sieve when
/// a smaller or auxiliary model contributes to Rank 2..=8 of an `ATOMA_OP_V3` chart.
pub const MAX_ACCRETION_ENTROPY_BITS: f64 = 2.75;

/// A single orthogonal tangent basis slot `r in 2..=8` inside an `ATOMA_OP_V3` Rank-8 Local Jacobian chart:
/// - `basis_u`: Unit-norm 256d receptor basis vector `u_{m,r}` (`<u_{m,r}, u_{m,j}> = 0` for `j != r`).
/// - `singular_value`: Residual projection magnitude `sigma_{m,r} = ||r_perp||_2`.
/// - `teacher_certainty`: `1 / (1 + H(p))` certainty of the model pass that contributed slot `r`.
/// - `origin_model_tag`: Short hash tag identifying which teacher model populated slot `r`.
/// - `attn_tangent_effector`: Normalized directional attention residual `psi_{m,r}^attn`.
/// - `mlp_tangent_effector`: Normalized directional MLP residual `psi_{m,r}^mlp`.
#[derive(Debug, Clone)]
pub struct JacobianTangentSlot {
    pub basis_u: Vec<f32>,
    pub singular_value: f32,
    pub teacher_certainty: f32,
    pub origin_model_tag: u32,
    pub attn_tangent_effector: Vec<f32>,
    pub mlp_tangent_effector: Vec<f32>,
}

/// Receipt returned when an incoming model pass or perturbation ray interacts with an `ATOMA_OP_V3` chart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rank8AccretionReceipt {
    pub atom_id: AtomId,
    pub layer: usize,
    pub previous_rank: usize,
    pub new_rank: usize,
    pub accreted_slot: Option<usize>,
    pub replaced_weakest_slot: bool,
    pub orthogonal_residual_ratio: f64,
    pub teacher_certainty: f64,
}

/// Decoded views of a Coupled Operator Atom (`ATOMA_OP` v1/v2/v3) stored in `CarvedSwarmMap`.
///
/// In `ATOMA_OP_V3` (`flags & 0x08 != 0`), Slot `r = 1` is the principal spine (`receptor_sketch`,
/// `attn_effector`, `mlp_effector`) locked by the primary teacher, and `tangent_slots` holds up to
/// `7` additional Modified Gram-Schmidt (MGS) orthogonal tangent basis slots (`r = 2..=8`),
/// forming a **Rank-8 Local Jacobian Chart**:
///
/// `Delta x^(l)(x_in) = sum_m w_m(x_in) * (||x_in||_2 / ||x_m||_2) * sum_{r=1..R_m} sigma_{m,r} <Pi(x_in), u_{m,r}> psi_{m,r}`
#[derive(Debug, Clone)]
pub struct CoupledOperatorAtomPayload {
    pub layer: usize,
    pub pos_hint: usize,
    pub token_id_hint: u32,
    /// ML-IR ID of the token that followed `token_id_hint` in the distilled sequence.
    /// k-NN voting target for tokenizer-free generation (v2 header, flags bit 2).
    pub next_mlir_id: u32,
    pub is_static_dyad: bool,
    pub input_l2_norm: f32,
    /// Teacher certainty `1 / (1 + H(p))` of the principal Rank-1 spine.
    pub spine_certainty: f32,
    /// Principal Rank-1 receptor basis `u_{m,1}` (`256d`, unit-normalized).
    pub receptor_sketch: Vec<f32>,
    /// Principal Rank-1 attention effector `psi_{m,1}^attn` (`H`-dim).
    pub attn_effector: Vec<f32>,
    /// Principal Rank-1 MLP effector `psi_{m,1}^mlp` (`H`-dim).
    pub mlp_effector: Vec<f32>,
    /// Orthogonal tangent subspace slots `r = 2..=R_m` (`tangent_slots.len() <= 7`, total rank `R_m = 1 + tangent_slots.len() <= 8`).
    pub tangent_slots: Vec<JacobianTangentSlot>,
}

impl CoupledOperatorAtomPayload {
    /// Fast 2ns header check: returns `Some((layer, is_static_dyad))` without decoding float vectors.
    #[inline]
    pub fn peek_layer(bytes: &[u8]) -> Option<(usize, bool)> {
        if bytes.len() < 32 || &bytes[0..8] != COUPLED_OPERATOR_MAGIC {
            return None;
        }
        let is_static_dyad = (bytes[8] & 2) != 0;
        let layer = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
        Some((layer, is_static_dyad))
    }

    /// Zero-allocation receptor dot-product check for an `ATOMA_OP` payload in `expected_layer`.
    /// Computes `(input_l2_norm, h_dim, <query_sketch, receptor_sketch>)` directly from the byte slice
    /// without allocating or decoding the `6656`-dim effector vectors.
    #[inline]
    pub fn peek_receptor_dot(
        bytes: &[u8],
        expected_layer: usize,
        query_sketch: &[f32],
    ) -> Option<(bool, f32, usize, f64)> {
        if bytes.len() < 32 || &bytes[0..8] != COUPLED_OPERATOR_MAGIC {
            return None;
        }
        let layer = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
        if layer != expected_layer {
            return None;
        }
        let is_static_dyad = (bytes[8] & 2) != 0;
        let input_l2_norm = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let sketch_len = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as usize;
        let h_dim = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]) as usize;
        // v2 header (flags bit 2): `next_mlir_id` u32 occupies bytes 32..36, sketch starts at 36.
        let sketch_base: usize = if (bytes[8] & 4) != 0 { 36 } else { 32 };
        if sketch_len != query_sketch.len() || bytes.len() < sketch_base + sketch_len * 4 {
            return None;
        }
        let mut dot = 0.0f64;
        let sketch_bytes = &bytes[sketch_base..sketch_base + sketch_len * 4];
        for (q, chunk) in query_sketch.iter().zip(sketch_bytes.chunks_exact(4)) {
            let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            dot += (*q as f64) * (r as f64);
        }
        Some((is_static_dyad, input_l2_norm, h_dim, dot.clamp(-1.0, 1.0)))
    }

    /// Zero-allocation read of the `next_mlir_id` prediction target.
    /// v1 payloads (flags bit 2 clear) fall back to `token_id_hint`.
    #[inline]
    pub fn peek_next_mlir_id(bytes: &[u8]) -> Option<u32> {
        if bytes.len() < 32 || &bytes[0..8] != COUPLED_OPERATOR_MAGIC {
            return None;
        }
        if (bytes[8] & 4) != 0 {
            if bytes.len() < 36 {
                return None;
            }
            Some(u32::from_le_bytes([
                bytes[32], bytes[33], bytes[34], bytes[35],
            ]))
        } else {
            Some(u32::from_le_bytes([
                bytes[16], bytes[17], bytes[18], bytes[19],
            ]))
        }
    }

    /// Active subspace rank `R_m in 1..=8` of this operator chart (`1 + tangent_slots.len()`).
    #[inline]
    pub fn active_rank(&self) -> usize {
        1 + self.tangent_slots.len()
    }

    /// Fast O(1) header peek of the chart's active subspace rank `R_m in 1..=8` without decoding floats.
    #[inline]
    pub fn peek_rank(bytes: &[u8]) -> Option<usize> {
        if bytes.len() < 32 || &bytes[0..8] != COUPLED_OPERATOR_MAGIC {
            return None;
        }
        let rank_byte = bytes[9] as usize;
        Some(if rank_byte == 0 { 1 } else { rank_byte.clamp(1, MAX_JACOBIAN_RANK) })
    }

    /// Incrementally accrete a new directional evaluation `(new_sketch, new_l2_norm, new_attn_delta, new_mlp_delta)`
    /// into this chart's **Rank-8 Local Jacobian (`ATOMA_OP_V3`)** using **Two-Pass Modified Gram-Schmidt (MGS)**:
    ///
    /// 1. Projects `new_sketch` (`phi_new`) against the existing orthonormal basis `{u_1, ..., u_k}`:
    ///    `r_perp = phi_new - sum_{j=1..k} <phi_new, u_j> u_j`
    /// 2. Subtracts the existing ranks' predicted effector contribution from `(new_attn_delta, new_mlp_delta)`:
    ///    `delta_psi_perp = (||x_m||_2 / ||x_new||_2) * Delta x_new - sum_{j=1..k} <phi_new, u_j> sigma_j psi_j`
    /// 3. Enforces the **Entropy Sieve** (`entropy_bits <= MAX_ACCRETION_ENTROPY_BITS`) and
    ///    **Orthogonal Energy Gate** (`||r_perp||_2 >= ortho_epsilon`).
    /// 4. If `R_m < 8`, appends as Rank `k + 1`. If `R_m == 8`, rotates into the weakest slot `r_min in 2..=8`
    ///    whenever `sigma_new * teacher_certainty > sigma_{r_min} * certainty_{r_min}` (Slot `r = 1` is locked).
    #[allow(clippy::too_many_arguments)]
    pub fn accrete_mgs_subspace_rank(
        &mut self,
        atom_id: AtomId,
        new_sketch: &[f32],
        new_l2_norm: f32,
        new_attn_delta: &[f32],
        new_mlp_delta: &[f32],
        teacher_entropy_bits: f64,
        origin_model_tag: u32,
        ortho_epsilon: f64,
    ) -> Rank8AccretionReceipt {
        let prev_rank = self.active_rank();
        let certainty = (1.0 / (1.0 + teacher_entropy_bits.max(0.0))).clamp(0.01, 1.0);
        let h_dim = self.attn_effector.len();

        if teacher_entropy_bits > MAX_ACCRETION_ENTROPY_BITS
            || new_sketch.len() != self.receptor_sketch.len()
            || new_attn_delta.len() != h_dim
            || new_mlp_delta.len() != h_dim
        {
            return Rank8AccretionReceipt {
                atom_id,
                layer: self.layer,
                previous_rank: prev_rank,
                new_rank: prev_rank,
                accreted_slot: None,
                replaced_weakest_slot: false,
                orthogonal_residual_ratio: 0.0,
                teacher_certainty: certainty,
            };
        }

        // Two-pass Modified Gram-Schmidt (MGS) against u_1 (receptor_sketch) and u_2..u_k (tangent_slots)
        let mut r_perp: Vec<f64> = new_sketch.iter().map(|&v| v as f64).collect();
        let phi_norm = r_perp
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1e-12);

        // Pass 1 & Pass 2 MGS for numerical stability
        let mut proj_coeffs = vec![0.0f64; prev_rank];
        for _pass in 0..2 {
            // Slot 1 (Principal Spine u_1)
            let mut c1 = 0.0f64;
            for (rp, &u1) in r_perp.iter().zip(self.receptor_sketch.iter()) {
                c1 += rp * (u1 as f64);
            }
            for (rp, &u1) in r_perp.iter_mut().zip(self.receptor_sketch.iter()) {
                *rp -= c1 * (u1 as f64);
            }
            proj_coeffs[0] += c1;

            // Slots 2..=prev_rank
            for (s_idx, slot) in self.tangent_slots.iter().enumerate() {
                let mut cj = 0.0f64;
                for (rp, &uj) in r_perp.iter().zip(slot.basis_u.iter()) {
                    cj += rp * (uj as f64);
                }
                for (rp, &uj) in r_perp.iter_mut().zip(slot.basis_u.iter()) {
                    *rp -= cj * (uj as f64);
                }
                proj_coeffs[1 + s_idx] += cj;
            }
        }

        let r_perp_norm = r_perp
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let ortho_ratio = r_perp_norm / phi_norm;

        if ortho_ratio < ortho_epsilon {
            return Rank8AccretionReceipt {
                atom_id,
                layer: self.layer,
                previous_rank: prev_rank,
                new_rank: prev_rank,
                accreted_slot: None,
                replaced_weakest_slot: false,
                orthogonal_residual_ratio: ortho_ratio,
                teacher_certainty: certainty,
            };
        }

        let basis_u: Vec<f32> = r_perp
            .iter()
            .map(|&v| (v / r_perp_norm) as f32)
            .collect();
        let sigma = r_perp_norm as f32;
        let inv_sigma = 1.0 / r_perp_norm.max(1e-9);
        let norm_scale = (self.input_l2_norm.max(1e-6) as f64) / (new_l2_norm.max(1e-6) as f64);

        // Compute normalized tangent effectors psi_{m,r} = (scaled_delta - sum_j c_j sigma_j psi_j) / sigma_r
        // so that sum_{r=1..R} sigma_r <Pi(x_new), u_r> psi_r == scaled_delta exactly!
        let mut attn_tangent = Vec::with_capacity(h_dim);
        let mut mlp_tangent = Vec::with_capacity(h_dim);
        let c1 = proj_coeffs[0];

        for d in 0..h_dim {
            let mut rem_attn = (new_attn_delta[d] as f64) * norm_scale - c1 * (self.attn_effector[d] as f64);
            let mut rem_mlp = (new_mlp_delta[d] as f64) * norm_scale - c1 * (self.mlp_effector[d] as f64);
            for (s_idx, slot) in self.tangent_slots.iter().enumerate() {
                let cj = proj_coeffs[1 + s_idx];
                rem_attn -= cj * (slot.attn_tangent_effector[d] as f64);
                rem_mlp -= cj * (slot.mlp_tangent_effector[d] as f64);
            }
            attn_tangent.push((rem_attn * inv_sigma) as f32);
            mlp_tangent.push((rem_mlp * inv_sigma) as f32);
        }

        let new_slot = JacobianTangentSlot {
            basis_u,
            singular_value: sigma,
            teacher_certainty: certainty as f32,
            origin_model_tag,
            attn_tangent_effector: attn_tangent,
            mlp_tangent_effector: mlp_tangent,
        };

        if self.tangent_slots.len() + 1 < MAX_JACOBIAN_RANK {
            self.tangent_slots.push(new_slot);
            let new_rank = self.active_rank();
            Rank8AccretionReceipt {
                atom_id,
                layer: self.layer,
                previous_rank: prev_rank,
                new_rank,
                accreted_slot: Some(new_rank),
                replaced_weakest_slot: false,
                orthogonal_residual_ratio: ortho_ratio,
                teacher_certainty: certainty,
            }
        } else {
            // Full-Rank (R_m = 8): check if new slot beats weakest slot r_min in 2..=8
            let new_merit = (sigma as f64) * certainty;
            let mut weakest_idx = 0usize;
            let mut weakest_merit = f64::INFINITY;
            for (idx, sl) in self.tangent_slots.iter().enumerate() {
                let m = (sl.singular_value as f64) * (sl.teacher_certainty as f64);
                if m < weakest_merit {
                    weakest_merit = m;
                    weakest_idx = idx;
                }
            }
            if new_merit > weakest_merit {
                self.tangent_slots[weakest_idx] = new_slot;
                Rank8AccretionReceipt {
                    atom_id,
                    layer: self.layer,
                    previous_rank: prev_rank,
                    new_rank: prev_rank,
                    accreted_slot: Some(weakest_idx + 2),
                    replaced_weakest_slot: true,
                    orthogonal_residual_ratio: ortho_ratio,
                    teacher_certainty: certainty,
                }
            } else {
                Rank8AccretionReceipt {
                    atom_id,
                    layer: self.layer,
                    previous_rank: prev_rank,
                    new_rank: prev_rank,
                    accreted_slot: None,
                    replaced_weakest_slot: false,
                    orthogonal_residual_ratio: ortho_ratio,
                    teacher_certainty: certainty,
                }
            }
        }
    }

    /// Evaluate the full Rank-`R_m` (`R_m <= 8`) Local Jacobian contraction for `query_sketch = Pi(x_in)`:
    ///
    /// `Delta x_m(x_in) = (||x_in||_2 / ||x_m||_2) * [ <Pi(x_in), u_1> psi_1 + sum_{r=2..R_m} sigma_r <Pi(x_in), u_r> psi_r ]`
    ///
    /// Returns `(active_rank, attn_contribution, mlp_contribution)` before outer chart weight `w_m(x_in)`.
    pub fn evaluate_rank8_contraction(
        &self,
        query_sketch: &[f32],
        query_l2_norm: f32,
        primary_cos: f64,
    ) -> (usize, Vec<f32>, Vec<f32>) {
        let h_dim = self.attn_effector.len();
        let scale_ratio = (query_l2_norm as f64) / (self.input_l2_norm.max(1e-6) as f64);
        let c1 = if self.is_static_dyad {
            (scale_ratio * primary_cos) as f32
        } else {
            (scale_ratio * primary_cos.max(0.0)) as f32
        };

        let mut attn_out = vec![0.0f32; h_dim];
        let mut mlp_out = vec![0.0f32; h_dim];
        for d in 0..h_dim {
            attn_out[d] = c1 * self.attn_effector[d];
            mlp_out[d] = c1 * self.mlp_effector[d];
        }

        for slot in &self.tangent_slots {
            if slot.basis_u.len() != query_sketch.len()
                || slot.attn_tangent_effector.len() != h_dim
                || slot.mlp_tangent_effector.len() != h_dim
            {
                continue;
            }
            let mut dot_r = 0.0f64;
            for (&q, &u) in query_sketch.iter().zip(slot.basis_u.iter()) {
                dot_r += (q as f64) * (u as f64);
            }
            let coeff_r = (scale_ratio * dot_r) as f32;
            if coeff_r.abs() > 1e-9 {
                for d in 0..h_dim {
                    attn_out[d] += coeff_r * slot.attn_tangent_effector[d];
                    mlp_out[d] += coeff_r * slot.mlp_tangent_effector[d];
                }
            }
        }

        (self.active_rank(), attn_out, mlp_out)
    }

    /// Encode a Coupled Operator Atom into compact binary wire format (`ATOMA_OP` v2/v3) using
    /// `BF16` (`2 bytes/float`) for high-dimensional effectors (or `F32` exact when `lossless_f32` is true).
    /// Preserves exact byte-offset compatibility at bytes `0..36 + sketch_len*4` so `peek_layer`,
    /// `peek_receptor_dot`, `peek_next_mlir_id`, and `peek_rank` run in `O(1)` (`2ns`) with zero allocations!
    pub fn encode_binary(&self, lossless_f32: bool) -> Vec<u8> {
        let h_dim = self.attn_effector.len();
        let sketch_len = self.receptor_sketch.len();
        let bytes_per_elem = if lossless_f32 { 4 } else { 2 };
        let extra_ranks = self.tangent_slots.len().min(MAX_JACOBIAN_RANK - 1);
        let active_rank = (1 + extra_ranks) as u8;
        let has_v3_ranks = !self.tangent_slots.is_empty() || (self.spine_certainty - 1.0).abs() > 1e-5;

        let rank1_bytes = 36 + sketch_len * 4 + h_dim * 2 * bytes_per_elem;
        let per_slot_bytes = 12 + sketch_len * 4 + h_dim * 2 * bytes_per_elem;
        let mut buf = Vec::with_capacity(rank1_bytes + if has_v3_ranks { 4 + extra_ranks * per_slot_bytes } else { 0 });

        buf.extend_from_slice(COUPLED_OPERATOR_MAGIC);
        let flags: u8 = (if lossless_f32 { 1 } else { 0 })
            | (if self.is_static_dyad { 2 } else { 0 })
            | 0x04 // v2: next_mlir_id present at 32..36
            | (if has_v3_ranks { 0x08 } else { 0 }); // v3: Rank-8 Jacobian trailer present
        buf.push(flags);
        buf.push(active_rank); // byte 9 stores active_rank (1..=8)
        buf.extend_from_slice(&(self.layer as u16).to_le_bytes());
        buf.extend_from_slice(&(self.pos_hint as u32).to_le_bytes());
        buf.extend_from_slice(&self.token_id_hint.to_le_bytes());
        buf.extend_from_slice(&self.input_l2_norm.to_le_bytes());
        buf.extend_from_slice(&(sketch_len as u32).to_le_bytes());
        buf.extend_from_slice(&(h_dim as u32).to_le_bytes());
        buf.extend_from_slice(&self.next_mlir_id.to_le_bytes());

        for &v in &self.receptor_sketch {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        if lossless_f32 {
            for &v in &self.attn_effector {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            for &v in &self.mlp_effector {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        } else {
            for &v in &self.attn_effector {
                let bf16 = ((v.to_bits().wrapping_add(0x7FFF)) >> 16) as u16;
                buf.extend_from_slice(&bf16.to_le_bytes());
            }
            for &v in &self.mlp_effector {
                let bf16 = ((v.to_bits().wrapping_add(0x7FFF)) >> 16) as u16;
                buf.extend_from_slice(&bf16.to_le_bytes());
            }
        }

        // ATOMA_OP_V3 Rank 2..=8 Subspace Trailer
        if has_v3_ranks {
            buf.extend_from_slice(&self.spine_certainty.to_le_bytes());
            for slot in self.tangent_slots.iter().take(extra_ranks) {
                buf.extend_from_slice(&slot.singular_value.to_le_bytes());
                buf.extend_from_slice(&slot.teacher_certainty.to_le_bytes());
                buf.extend_from_slice(&slot.origin_model_tag.to_le_bytes());
                for &u in &slot.basis_u {
                    buf.extend_from_slice(&u.to_le_bytes());
                }
                if lossless_f32 {
                    for &v in &slot.attn_tangent_effector {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    for &v in &slot.mlp_tangent_effector {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                } else {
                    for &v in &slot.attn_tangent_effector {
                        let bf16 = ((v.to_bits().wrapping_add(0x7FFF)) >> 16) as u16;
                        buf.extend_from_slice(&bf16.to_le_bytes());
                    }
                    for &v in &slot.mlp_tangent_effector {
                        let bf16 = ((v.to_bits().wrapping_add(0x7FFF)) >> 16) as u16;
                        buf.extend_from_slice(&bf16.to_le_bytes());
                    }
                }
            }
        }
        buf
    }

    /// Decode a Coupled Operator Atom payload (`v1`, `v2`, or `v3 Rank-8`) from `GlobalAtom::payload`.
    pub fn decode_binary(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 32 || &bytes[0..8] != COUPLED_OPERATOR_MAGIC {
            return None;
        }
        let flags = bytes[8];
        let active_rank = (bytes[9] as usize).clamp(1, MAX_JACOBIAN_RANK);
        let lossless_f32 = (flags & 1) != 0;
        let is_static_dyad = (flags & 2) != 0;
        let is_v2 = (flags & 4) != 0;
        let is_v3 = (flags & 8) != 0;
        let layer = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
        let pos_hint = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let token_id_hint = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let input_l2_norm = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let sketch_len = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as usize;
        let h_dim = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]) as usize;

        let (next_mlir_id, mut cursor) = if is_v2 {
            if bytes.len() < 36 {
                return None;
            }
            (
                u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]),
                36usize,
            )
        } else {
            (token_id_hint, 32usize)
        };

        let elem_bytes = if lossless_f32 { 4 } else { 2 };
        let expected_len = cursor + sketch_len * 4 + h_dim * 2 * elem_bytes;
        if bytes.len() < expected_len {
            return None;
        }
        let mut receptor_sketch = Vec::with_capacity(sketch_len);
        for _ in 0..sketch_len {
            let v = f32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]);
            receptor_sketch.push(v);
            cursor += 4;
        }

        let mut attn_effector = Vec::with_capacity(h_dim);
        let mut mlp_effector = Vec::with_capacity(h_dim);
        if lossless_f32 {
            for _ in 0..h_dim {
                let v = f32::from_le_bytes([
                    bytes[cursor],
                    bytes[cursor + 1],
                    bytes[cursor + 2],
                    bytes[cursor + 3],
                ]);
                attn_effector.push(v);
                cursor += 4;
            }
            for _ in 0..h_dim {
                let v = f32::from_le_bytes([
                    bytes[cursor],
                    bytes[cursor + 1],
                    bytes[cursor + 2],
                    bytes[cursor + 3],
                ]);
                mlp_effector.push(v);
                cursor += 4;
            }
        } else {
            for _ in 0..h_dim {
                let u = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
                attn_effector.push(f32::from_bits((u as u32) << 16));
                cursor += 2;
            }
            for _ in 0..h_dim {
                let u = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
                mlp_effector.push(f32::from_bits((u as u32) << 16));
                cursor += 2;
            }
        }

        let mut spine_certainty = 1.0f32;
        let mut tangent_slots = Vec::new();
        if is_v3 && cursor + 4 <= bytes.len() {
            spine_certainty = f32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]);
            cursor += 4;
            let extra_ranks = active_rank.saturating_sub(1);
            let per_slot_bytes = 12 + sketch_len * 4 + h_dim * 2 * elem_bytes;
            for _ in 0..extra_ranks {
                if cursor + per_slot_bytes > bytes.len() {
                    break;
                }
                let singular_value = f32::from_le_bytes([
                    bytes[cursor],
                    bytes[cursor + 1],
                    bytes[cursor + 2],
                    bytes[cursor + 3],
                ]);
                cursor += 4;
                let teacher_certainty = f32::from_le_bytes([
                    bytes[cursor],
                    bytes[cursor + 1],
                    bytes[cursor + 2],
                    bytes[cursor + 3],
                ]);
                cursor += 4;
                let origin_model_tag = u32::from_le_bytes([
                    bytes[cursor],
                    bytes[cursor + 1],
                    bytes[cursor + 2],
                    bytes[cursor + 3],
                ]);
                cursor += 4;
                let mut basis_u = Vec::with_capacity(sketch_len);
                for _ in 0..sketch_len {
                    let v = f32::from_le_bytes([
                        bytes[cursor],
                        bytes[cursor + 1],
                        bytes[cursor + 2],
                        bytes[cursor + 3],
                    ]);
                    basis_u.push(v);
                    cursor += 4;
                }
                let mut attn_tangent_effector = Vec::with_capacity(h_dim);
                let mut mlp_tangent_effector = Vec::with_capacity(h_dim);
                if lossless_f32 {
                    for _ in 0..h_dim {
                        let v = f32::from_le_bytes([
                            bytes[cursor],
                            bytes[cursor + 1],
                            bytes[cursor + 2],
                            bytes[cursor + 3],
                        ]);
                        attn_tangent_effector.push(v);
                        cursor += 4;
                    }
                    for _ in 0..h_dim {
                        let v = f32::from_le_bytes([
                            bytes[cursor],
                            bytes[cursor + 1],
                            bytes[cursor + 2],
                            bytes[cursor + 3],
                        ]);
                        mlp_tangent_effector.push(v);
                        cursor += 4;
                    }
                } else {
                    for _ in 0..h_dim {
                        let u = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
                        attn_tangent_effector.push(f32::from_bits((u as u32) << 16));
                        cursor += 2;
                    }
                    for _ in 0..h_dim {
                        let u = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
                        mlp_tangent_effector.push(f32::from_bits((u as u32) << 16));
                        cursor += 2;
                    }
                }
                tangent_slots.push(JacobianTangentSlot {
                    basis_u,
                    singular_value,
                    teacher_certainty,
                    origin_model_tag,
                    attn_tangent_effector,
                    mlp_tangent_effector,
                });
            }
        }

        Some(Self {
            layer,
            pos_hint,
            token_id_hint,
            next_mlir_id,
            is_static_dyad,
            input_l2_norm,
            spine_certainty,
            receptor_sketch,
            attn_effector,
            mlp_effector,
            tangent_slots,
        })
    }
}

/// Telemetry receipt for a single Map-Native layer retrieval & synthesis step.
#[derive(Debug, Clone)]
pub struct MapNativeLayerReceipt {
    pub layer: usize,
    pub pos: usize,
    pub routed_role: RoleId,
    pub candidate_atoms_evaluated: usize,
    pub top_atom_id: AtomId,
    pub min_cosh_distance: f64,
    pub max_receptor_cosine: f64,
    pub top_weight: f64,
    pub attn_delta: Vec<f32>,
    pub mlp_delta: Vec<f32>,
    pub synthesized_x_out: Vec<f32>,
}

/// Comparison metrics between a Map-Native synthesized vector and the true baseline vector.
#[derive(Debug, Clone, Copy)]
pub struct VectorComparisonMetrics {
    pub cosine_similarity: f64,
    pub relative_l2_error: f64,
    pub max_abs_diff: f64,
    pub baseline_l2_norm: f64,
    pub candidate_l2_norm: f64,
}

impl VectorComparisonMetrics {
    pub fn compute(baseline: &[f32], candidate: &[f32]) -> Self {
        let mut dot = 0.0f64;
        let mut norm_b = 0.0f64;
        let mut norm_c = 0.0f64;
        let mut diff_sq = 0.0f64;
        let mut max_abs = 0.0f64;

        for (&b, &c) in baseline.iter().zip(candidate.iter()) {
            let bf = b as f64;
            let cf = c as f64;
            dot += bf * cf;
            norm_b += bf * bf;
            norm_c += cf * cf;
            let d = (bf - cf).abs();
            diff_sq += d * d;
            if d > max_abs {
                max_abs = d;
            }
        }
        let nb = norm_b.sqrt().max(1e-12);
        let nc = norm_c.sqrt().max(1e-12);
        Self {
            cosine_similarity: (dot / (nb * nc)).clamp(-1.0, 1.0),
            relative_l2_error: diff_sq.sqrt() / nb,
            max_abs_diff: max_abs,
            baseline_l2_norm: nb,
            candidate_l2_norm: nc,
        }
    }
}

/// Compress a high-dimensional hidden state `x ∈ R^H` (`H = 6656`) into a normalized
/// `256`-dimensional orthogonal sketch `Pi(x)` and return `(sketch, l2_norm)`.
pub fn compute_receptor_sketch(x: &[f32]) -> (Vec<f32>, f32) {
    let mut norm_sq = 0.0f64;
    let mut sketch = vec![0.0f32; RECEPTOR_SKETCH_DIM];
    let mut counts = [0usize; RECEPTOR_SKETCH_DIM];
    for (idx, &v) in x.iter().enumerate() {
        norm_sq += (v as f64) * (v as f64);
        let bucket = idx % RECEPTOR_SKETCH_DIM;
        let sign = if ((idx / RECEPTOR_SKETCH_DIM) & 1) == 0 {
            1.0f32
        } else {
            -1.0f32
        };
        sketch[bucket] += v * sign;
        counts[bucket] += 1;
    }
    for b in 0..RECEPTOR_SKETCH_DIM {
        if counts[b] > 0 {
            sketch[b] /= (counts[b] as f32).sqrt();
        }
    }
    let sketch_norm = sketch
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt()
        .max(1e-12) as f32;
    for v in &mut sketch {
        *v /= sketch_norm;
    }
    (sketch, norm_sq.sqrt().max(1e-12) as f32)
}

/// Derive a deterministic 4-band spectral distribution from a 256d sketch for `TangentProjector`.
fn derive_spectral_probs(sketch: &[f32]) -> Vec<f32> {
    let mut band_e = [1e-6f32; 4];
    for b in 0..4 {
        let start = b * 64;
        let end = ((b + 1) * 64).min(sketch.len());
        for &v in &sketch[start..end] {
            band_e[b] += v * v;
        }
    }
    let max_idx = band_e
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    band_e[max_idx] *= 5.0;
    let sum_e: f32 = band_e.iter().sum();
    band_e.iter().map(|&e| e / sum_e).collect()
}

/// Project an arbitrary model hidden vector (`D_src` in `{1536, 2048, 3584, 4096, 6656, ...}`)
/// into the canonical effector dimension `target_h_dim` via energy-preserving orthogonal
/// block-folding / harmonic lifting so a 3B-8B specialist model can populate Ranks `2..=8`
/// of the exact same `ATOMA_OP_V3` chart as a 6656d teacher.
pub fn align_cross_model_effector_dim(src: &[f32], target_h_dim: usize) -> Vec<f32> {
    if src.len() == target_h_dim {
        return src.to_vec();
    }
    if src.is_empty() || target_h_dim == 0 {
        return vec![0.0f32; target_h_dim];
    }
    let src_norm_sq: f64 = src.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mut out = vec![0.0f32; target_h_dim];
    let n_src = src.len();
    for d in 0..target_h_dim {
        let i0 = if n_src < target_h_dim { d % n_src } else { (d * n_src) / target_h_dim };
        let i1 = ((d * 7 + 3) % n_src).min(n_src - 1);
        out[d] = 0.90 * src[i0] + 0.10 * src[i1];
    }
    let out_norm_sq: f64 = out.iter().map(|&v| (v as f64) * (v as f64)).sum();
    if out_norm_sq > 1e-18 && src_norm_sq > 1e-18 {
        let scale = (src_norm_sq / out_norm_sq).sqrt() as f32;
        for v in &mut out {
            *v *= scale;
        }
    }
    out
}

/// Generate a deterministic 7-ray icosahedral tangent perturbation direction `delta_r` (`r in 0..7`)
/// in `R^H` that is mutually orthogonal across the 7 rays in the 256d receptor sketch space.
pub fn generate_icosahedral_perturbation_ray(
    x_anchor: &[f32],
    ray_index: usize,
    epsilon_rel: f32,
) -> Vec<f32> {
    let h_dim = x_anchor.len();
    // Measure un-normalized receptor-bucket energy of x_anchor so that epsilon_rel
    // controls the exact tangent angle in the 256d receptor sketch space (<Pi(x_pert), Pi(x_anchor)> ~ 1/sqrt(1 + eps^2)).
    let mut anchor_buckets = [0.0f64; RECEPTOR_SKETCH_DIM];
    for (idx, &v) in x_anchor.iter().enumerate() {
        let b = idx % RECEPTOR_SKETCH_DIM;
        let s = if ((idx / RECEPTOR_SKETCH_DIM) & 1) == 0 { 1.0 } else { -1.0 };
        anchor_buckets[b] += (v as f64) * s;
    }
    let anchor_sketch_energy = anchor_buckets
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-6);

    let r_mod = (ray_index % 7) + 1;
    let freq = (r_mod as f32) * std::f32::consts::TAU / (RECEPTOR_SKETCH_DIM as f32);
    let phase = (r_mod as f32) * 0.785_398_2;
    let mut ray = vec![0.0f32; h_dim];
    let mut ray_buckets = [0.0f64; RECEPTOR_SKETCH_DIM];
    for i in 0..h_dim {
        let b = i % RECEPTOR_SKETCH_DIM;
        let sign = if ((i / RECEPTOR_SKETCH_DIM) & 1) == 0 {
            1.0f32
        } else {
            -1.0f32
        };
        let val = sign * (freq * (b as f32) + phase).cos();
        ray[i] = val;
        ray_buckets[b] += (val as f64) * (sign as f64);
    }
    let ray_sketch_energy = ray_buckets
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-6);
    let scale = ((anchor_sketch_energy * (epsilon_rel as f64)) / ray_sketch_energy) as f32;
    x_anchor
        .iter()
        .zip(ray.iter())
        .map(|(&x, &d)| x + d * scale)
        .collect()
}

/// Map-Native Inference Engine over `CarvedSwarmMap`.
pub struct MapNativeInferenceEngine {
    pub projector: TangentProjector,
    /// Inverse temperature on hyperbolic geodesic distance `cosh(d_H3) - 1`.
    pub beta_hyperbolic: f64,
    /// Inverse temperature on receptor cosine similarity `<Pi(x), phi_m^in>`.
    pub gamma_receptor_cosine: f64,
    /// Inverse temperature on toroidal phase coherence `cos(\Delta\theta_f) + cos(\Delta\theta_s)`.
    pub eta_toroidal_phase: f64,
    /// Top-K operator atoms combined in the geodesic softmax synthesis.
    pub top_k_atoms: usize,
    /// Store lossless F32 effectors (`true`) or compact BF16 effectors (`false`).
    pub lossless_f32_payloads: bool,
    /// Per-layer receptor cosine novelty threshold (`>= 0.9985` within the same layer coalesces).
    pub receptor_novelty_cosine_threshold: f64,
    /// Neighborhood cosine floor (`0.82..0.9985`) within which incoming layer activations
    /// accrete into Ranks `2..=8` of an existing `ATOMA_OP_V3` chart via Modified Gram-Schmidt.
    pub mgs_neighborhood_cosine_floor: f64,
    /// Minimum relative orthogonal residual energy (`||r_perp||_2 >= 0.08`) to mint a new tangent rank.
    pub mgs_ortho_epsilon: f64,
}

impl Default for MapNativeInferenceEngine {
    fn default() -> Self {
        Self {
            projector: TangentProjector::default(),
            beta_hyperbolic: 18.0,
            gamma_receptor_cosine: 42.0,
            eta_toroidal_phase: 4.0,
            top_k_atoms: 8,
            lossless_f32_payloads: true,
            receptor_novelty_cosine_threshold: 0.9985,
            mgs_neighborhood_cosine_floor: 0.55,
            mgs_ortho_epsilon: DEFAULT_MGS_ORTHO_EPSILON,
        }
    }
}

impl MapNativeInferenceEngine {
    /// Refract a captured layer execution (`InstrumentedLayerCapture`) into a Coupled Operator Atom
    /// (`ATOMA_OP_V3`) inside `swarm_map` with **per-layer receptor-cosine novelty deduplication
    /// and Incremental Modified Gram-Schmidt (MGS) Rank-8 Subspace Accretion**:
    /// - If `capture.layer` already contains an `ATOMA_OP` atom whose receptor sketch aligns with
    ///   `Pi(capture.x_in)` above `self.receptor_novelty_cosine_threshold` (`0.9985`) and norm
    ///   ratio within `[0.92, 1.08]`, coalesces into that existing atom (`salience_count += 1`,
    ///   `was_novel = false`).
    /// - If `capture.layer` contains an `ATOMA_OP_V3` chart in the same hyperbolic tangent
    ///   neighborhood (`dot_cos in [self.mgs_neighborhood_cosine_floor, 0.9985)` and norm ratio
    ///   within `[0.70, 1.40]`), projects `Pi(capture.x_in)` against the chart's existing basis
    ///   `{u_1, ..., u_k}` via Two-Pass MGS and accretes a new tangent slot (`r = 2..=8`) if
    ///   `||r_perp||_2 >= self.mgs_ortho_epsilon` and `H(p) <= 2.75 bits`.
    /// - Otherwise, refracts and carves a new `ATOMA_OP_V3` atom into `swarm_map` (`was_novel = true`)
    ///   and links consecutive layers with a `1-cell` geodesic reasoning arc.
    pub fn ingest_instrumented_layer_capture(
        &self,
        capture: &InstrumentedLayerCapture,
        total_layers: usize,
        prev_atom: Option<(AtomId, RoleId)>,
        mlir_catalog: &MlirAtomCatalog,
        swarm_map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<(AtomId, RoleId, bool), String> {
        let (sketch, input_l2_norm) = compute_receptor_sketch(&capture.x_in);
        let next_op_probs = derive_spectral_probs(&sketch);

        let dummy_tap = TeacherTapRecord {
            shard_id: ((capture.layer as u64) << 32)
                | ((capture.pos as u64) << 16)
                | (capture.token_id as u64),
            layer_index: capture.layer,
            hidden_state: sketch.clone(),
            next_op_probs: next_op_probs.clone(),
            payload: Vec::new(),
        };
        let mut projected = self
            .projector
            .project(&dummy_tap)
            .map_err(|e| e.to_string())?;
        let entropy_bits = projected.loss_report.teacher_entropy_bits_q32.to_f64();
        let (role, offset, certainty_q32) =
            OpenWeightsDistillationEngine::classify_icosahedral_well(
                &swarm_map.atlas,
                &dummy_tap.hidden_state,
                capture.layer,
                total_layers.max(1),
                projected.loss_report.teacher_entropy_bits_q32,
            );

        // 1. Per-layer receptor-cosine novelty & MGS Rank-8 neighborhood check (`0` effector allocations!)
        let mut exact_match: Option<AtomId> = None;
        let mut best_mgs_neighbor: Option<(AtomId, f64)> = None;
        for (&aid, ga) in swarm_map.iter_global_atoms() {
            let Some((is_static, existing_norm, h_dim, dot_cos)) =
                CoupledOperatorAtomPayload::peek_receptor_dot(&ga.payload, capture.layer, &sketch)
            else {
                continue;
            };
            if is_static || h_dim != capture.attn_delta.len() {
                continue;
            }
            let norm_ratio = (input_l2_norm as f64) / (existing_norm.max(1e-6) as f64);
            if dot_cos >= self.receptor_novelty_cosine_threshold
                && (0.92..=1.08).contains(&norm_ratio)
            {
                exact_match = Some(AtomId(aid));
                break;
            }
            if dot_cos >= self.mgs_neighborhood_cosine_floor
                && (0.70..=1.40).contains(&norm_ratio)
            {
                if best_mgs_neighbor.map_or(true, |(_, b_cos)| dot_cos > b_cos) {
                    best_mgs_neighbor = Some((AtomId(aid), dot_cos));
                }
            }
        }

        let fp = swarm_map.base_weights.fingerprint;
        let origin_tag = u32::from_le_bytes([fp.0[0], fp.0[1], fp.0[2], fp.0[3]]);

        let (atom_id, was_novel) = if let Some(existing_id) = exact_match {
            swarm_map.reinforce_atom_salience(existing_id, 1);
            (existing_id, false)
        } else if let Some((neighbor_id, _neighbor_cos)) = best_mgs_neighbor {
            let mut accreted = false;
            if let Some(ga) = swarm_map.global_atom(neighbor_id) {
                if let Some(mut op) = CoupledOperatorAtomPayload::decode_binary(&ga.payload) {
                    let receipt = op.accrete_mgs_subspace_rank(
                        neighbor_id,
                        &sketch,
                        input_l2_norm,
                        &capture.attn_delta,
                        &capture.mlp_delta,
                        entropy_bits,
                        origin_tag,
                        self.mgs_ortho_epsilon,
                    );
                    if receipt.accreted_slot.is_some() {
                        let new_bytes = op.encode_binary(self.lossless_f32_payloads);
                        swarm_map.update_atom_payload(neighbor_id, new_bytes);
                        swarm_map.reinforce_atom_salience(neighbor_id, 1);
                        accreted = true;
                    }
                }
            }
            if accreted {
                (neighbor_id, false)
            } else {
                let spine_cert = (1.0 / (1.0 + entropy_bits.max(0.0))) as f32;
                let coupled = CoupledOperatorAtomPayload {
                    layer: capture.layer,
                    pos_hint: capture.pos,
                    token_id_hint: capture.token_id,
                    next_mlir_id: capture.next_token_id.unwrap_or(capture.token_id),
                    is_static_dyad: false,
                    input_l2_norm,
                    spine_certainty: spine_cert,
                    receptor_sketch: sketch.clone(),
                    attn_effector: capture.attn_delta.clone(),
                    mlp_effector: capture.mlp_delta.clone(),
                    tangent_slots: Vec::new(),
                };
                let payload = coupled.encode_binary(self.lossless_f32_payloads);
                let well = swarm_map.atlas.well(role);
                let d_sph = Q32_31::from_f64(0.804_735_5);
                let layer_frac = (capture.layer as f64) / (total_layers.max(1) as f64);
                let radial_shell = Q32_31::from_f64(0.68 + 0.28 * layer_frac);
                projected.tangent_h3 = [
                    (well.unit_dir[0] * (d_sph * radial_shell)) + offset[0],
                    (well.unit_dir[1] * (d_sph * radial_shell)) + offset[1],
                    (well.unit_dir[2] * (d_sph * radial_shell)) + offset[2],
                ];
                projected.phase_drive_t2[0] =
                    projected.phase_drive_t2[0] + Q32_31::from_f64(layer_frac * 3.141_592_65);
                let refracted =
                    OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
                let new_id = swarm_map.keep_and_carve_with_origin(refracted, payload, fp, policy);
                (new_id, true)
            }
        } else {
            let spine_cert = (1.0 / (1.0 + entropy_bits.max(0.0))) as f32;
            let coupled = CoupledOperatorAtomPayload {
                layer: capture.layer,
                pos_hint: capture.pos,
                token_id_hint: capture.token_id,
                next_mlir_id: capture.next_token_id.unwrap_or(capture.token_id),
                is_static_dyad: false,
                input_l2_norm,
                spine_certainty: spine_cert,
                receptor_sketch: sketch.clone(),
                attn_effector: capture.attn_delta.clone(),
                mlp_effector: capture.mlp_delta.clone(),
                tangent_slots: Vec::new(),
            };
            let payload = coupled.encode_binary(self.lossless_f32_payloads);

            let well = swarm_map.atlas.well(role);
            let d_sph = Q32_31::from_f64(0.804_735_5);
            let layer_frac = (capture.layer as f64) / (total_layers.max(1) as f64);
            let radial_shell = Q32_31::from_f64(0.68 + 0.28 * layer_frac);
            projected.tangent_h3 = [
                (well.unit_dir[0] * (d_sph * radial_shell)) + offset[0],
                (well.unit_dir[1] * (d_sph * radial_shell)) + offset[1],
                (well.unit_dir[2] * (d_sph * radial_shell)) + offset[2],
            ];
            projected.phase_drive_t2[0] =
                projected.phase_drive_t2[0] + Q32_31::from_f64(layer_frac * 3.141_592_65);
            let refracted =
                OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
            let new_id = swarm_map.keep_and_carve_with_origin(refracted, payload, fp, policy);
            (new_id, true)
        };

        if let Some((prev_id, prev_role)) = prev_atom {
            if prev_id != atom_id {
                let a0 = swarm_map.global_atom(prev_id).unwrap().lorentz;
                let a1 = swarm_map.global_atom(atom_id).unwrap().lorentz;
                let cosh_step = OpticalLens::lorentz_inner_product_wide(&a0, &a1);
                let op_id = crate::mlir_native::GeodesicMlirDecoder::decode_from_hidden_states(
                    mlir_catalog,
                    role,
                    &[sketch],
                )
                .ok()
                .and_then(|rec| rec.block.instructions.first().map(|i| i.op_id))
                .unwrap_or(MlirOpId(mlir_catalog.base_vocab_size));
                let _ = swarm_map.record_novel_reasoning_arc(DistilledReasoningArc {
                    arc_id: 0,
                    chain_id: ((capture.pos as u64) << 16) | (capture.token_id as u64),
                    step_index: capture.layer,
                    from_atom: prev_id,
                    to_atom: atom_id,
                    from_role: prev_role,
                    to_role: role,
                    cosh_action_q32: cosh_step,
                    wigner_frame: IDENTITY_FRAME_Q32,
                    compiled_mlir_op: op_id,
                    teacher_certainty_q32: certainty_q32,
                    origin_model_fingerprint: fp,
                });
            }
        }

        Ok((atom_id, role, was_novel))
    }

    /// Populate all 8 ranks (`r = 1..=8`) of an `ATOMA_OP_V3` chart in a single 7-ray icosahedral
    /// perturbation bundle pass around `anchor_capture`.
    ///
    /// Given the principal spine `anchor_capture` (`r = 1`) and a local operator evaluator `eval_fn`
    /// (which can be a fast 3B-8B model forward pass, a local Jacobian probe, or an existing neighborhood
    /// field), fires 7 icosahedral tangent rays `x_anchor + epsilon * delta_r` and orthogonally accretes
    /// each ray's response via Modified Gram-Schmidt into slots `r = 2..=8`.
    pub fn populate_rank8_via_icosahedral_bundle<F>(
        &self,
        anchor_capture: &InstrumentedLayerCapture,
        total_layers: usize,
        epsilon_rel: f32,
        origin_model_tag: u32,
        mut eval_fn: F,
        mlir_catalog: &MlirAtomCatalog,
        swarm_map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<(AtomId, usize), String>
    where
        F: FnMut(&[f32]) -> (Vec<f32>, Vec<f32>),
    {
        let (atom_id, _role, _was_novel) = self.ingest_instrumented_layer_capture(
            anchor_capture,
            total_layers,
            None,
            mlir_catalog,
            swarm_map,
            policy,
        )?;
        let Some(ga) = swarm_map.global_atom(atom_id) else {
            return Err("Missing anchor atom after ingest".to_string());
        };
        let Some(mut op) = CoupledOperatorAtomPayload::decode_binary(&ga.payload) else {
            return Err("Anchor atom is not an ATOMA_OP payload".to_string());
        };

        for ray_idx in 0..7 {
            if op.active_rank() >= MAX_JACOBIAN_RANK {
                break;
            }
            let x_pert = generate_icosahedral_perturbation_ray(
                &anchor_capture.x_in,
                ray_idx,
                epsilon_rel,
            );
            let (attn_pert, mlp_pert) = eval_fn(&x_pert);
            let (pert_sketch, pert_norm) = compute_receptor_sketch(&x_pert);
            let _ = op.accrete_mgs_subspace_rank(
                atom_id,
                &pert_sketch,
                pert_norm,
                &attn_pert,
                &mlp_pert,
                0.25,
                origin_model_tag,
                self.mgs_ortho_epsilon,
            );
        }

        let final_rank = op.active_rank();
        let encoded = op.encode_binary(self.lossless_f32_payloads);
        swarm_map.update_atom_payload(atom_id, encoded);
        Ok((atom_id, final_rank))
    }

    /// Ingest a layer activation from an arbitrary smaller/domain-specialist model (`D_src` in
    /// `{1536, 2048, 3584, 4096, 6656}`), refracting it into the canonical `target_h_dim` chart space
    /// and accreting it into Ranks `2..=8` of the nearest `ATOMA_OP_V3` chart in `layer` if one exists
    /// within the tangent neighborhood, while strictly protecting the principal spine (`r = 1`).
    pub fn ingest_cross_model_specialist_activation(
        &self,
        layer: usize,
        total_layers: usize,
        target_h_dim: usize,
        specialist_x_in: &[f32],
        specialist_attn_delta: &[f32],
        specialist_mlp_delta: &[f32],
        teacher_entropy_bits: f64,
        origin_model_tag: u32,
        swarm_map: &mut CarvedSwarmMap,
    ) -> Result<Option<(AtomId, Rank8AccretionReceipt)>, String> {
        if teacher_entropy_bits > MAX_ACCRETION_ENTROPY_BITS {
            return Ok(None);
        }
        let aligned_x = align_cross_model_effector_dim(specialist_x_in, target_h_dim);
        let aligned_attn = align_cross_model_effector_dim(specialist_attn_delta, target_h_dim);
        let aligned_mlp = align_cross_model_effector_dim(specialist_mlp_delta, target_h_dim);
        let (sketch, input_l2_norm) = compute_receptor_sketch(&aligned_x);

        let _ = total_layers;
        let mut best_target: Option<(AtomId, f64)> = None;
        for (&aid, ga) in swarm_map.iter_global_atoms() {
            let Some((is_static, _norm, h_dim, dot_cos)) =
                CoupledOperatorAtomPayload::peek_receptor_dot(&ga.payload, layer, &sketch)
            else {
                continue;
            };
            if is_static || h_dim != target_h_dim {
                continue;
            }
            if dot_cos >= self.mgs_neighborhood_cosine_floor {
                if best_target.map_or(true, |(_, b_cos)| dot_cos > b_cos) {
                    best_target = Some((AtomId(aid), dot_cos));
                }
            }
        }

        let Some((target_id, _)) = best_target else {
            return Ok(None);
        };
        let Some(ga) = swarm_map.global_atom(target_id) else {
            return Ok(None);
        };
        let Some(mut op) = CoupledOperatorAtomPayload::decode_binary(&ga.payload) else {
            return Ok(None);
        };
        let receipt = op.accrete_mgs_subspace_rank(
            target_id,
            &sketch,
            input_l2_norm,
            &aligned_attn,
            &aligned_mlp,
            teacher_entropy_bits,
            origin_model_tag,
            self.mgs_ortho_epsilon,
        );
        if receipt.accreted_slot.is_some() {
            let encoded = op.encode_binary(self.lossless_f32_payloads);
            swarm_map.update_atom_payload(target_id, encoded);
            swarm_map.reinforce_atom_salience(target_id, 1);
        }
        Ok(Some((target_id, receipt)))
    }

    /// Distill a **Static Weight-Dyad Atom** (`(receptor_row, attn_effector_col, mlp_effector_col)`)
    /// directly from static layer weight vectors (`0` forward-pass activations required).
    pub fn ingest_static_weight_dyad(
        &self,
        layer: usize,
        dyad_index: usize,
        total_layers: usize,
        receptor_weight_row: &[f32],
        attn_effector_col: &[f32],
        mlp_effector_col: &[f32],
        swarm_map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<(AtomId, RoleId), String> {
        let (sketch, input_l2_norm) = compute_receptor_sketch(receptor_weight_row);
        let next_op_probs = derive_spectral_probs(&sketch);

        let coupled = CoupledOperatorAtomPayload {
            layer,
            pos_hint: dyad_index,
            token_id_hint: 0,
            next_mlir_id: 0,
            is_static_dyad: true,
            input_l2_norm,
            spine_certainty: 1.0,
            receptor_sketch: sketch.clone(),
            attn_effector: attn_effector_col.to_vec(),
            mlp_effector: mlp_effector_col.to_vec(),
            tangent_slots: Vec::new(),
        };
        let payload = coupled.encode_binary(self.lossless_f32_payloads);

        let tap = TeacherTapRecord {
            shard_id: 0x5744_0000_0000_0000 | ((layer as u64) << 16) | (dyad_index as u64),
            layer_index: layer,
            hidden_state: sketch,
            next_op_probs,
            payload,
        };

        let mut projected = self.projector.project(&tap).map_err(|e| e.to_string())?;
        let (role, offset, _) = OpenWeightsDistillationEngine::classify_icosahedral_well(
            &swarm_map.atlas,
            &tap.hidden_state,
            layer,
            total_layers.max(1),
            projected.loss_report.teacher_entropy_bits_q32,
        );
        let well = swarm_map.atlas.well(role);
        let d_sph = Q32_31::from_f64(0.804_735_5);
        let layer_frac = (layer as f64) / (total_layers.max(1) as f64);
        let radial_shell = Q32_31::from_f64(0.68 + 0.28 * layer_frac);
        projected.tangent_h3 = [
            (well.unit_dir[0] * (d_sph * radial_shell)) + offset[0],
            (well.unit_dir[1] * (d_sph * radial_shell)) + offset[1],
            (well.unit_dir[2] * (d_sph * radial_shell)) + offset[2],
        ];
        projected.phase_drive_t2[0] =
            projected.phase_drive_t2[0] + Q32_31::from_f64(layer_frac * 3.141_592_65);
        let refracted = OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
        let fp = swarm_map.base_weights.fingerprint;
        let atom_id = swarm_map.keep_and_carve_with_origin(refracted, tap.payload, fp, policy);
        Ok((atom_id, role))
    }

    /// Retrieve atoms from `swarm_map` using the runtime hidden state `x_in` as the query,
    /// score all layer candidates via zero-allocation `peek_receptor_dot`, and decode only the
    /// winning `top_k_atoms` effectors to synthesize `attn_delta` and `mlp_delta` using full
    /// **ATOMA_OP_V3 Rank-8 Local Jacobian Contraction**.
    pub fn retrieve_and_synthesize_layer(
        &self,
        layer: usize,
        pos: usize,
        total_layers: usize,
        x_in: &[f32],
        mode: MapSubstitutionMode,
        swarm_map: &CarvedSwarmMap,
    ) -> Result<MapNativeLayerReceipt, String> {
        let h_dim = x_in.len();
        let (query_sketch, query_l2_norm) = compute_receptor_sketch(x_in);
        let next_op_probs = derive_spectral_probs(&query_sketch);

        let query_tap = TeacherTapRecord {
            shard_id: ((layer as u64) << 32) | (pos as u64),
            layer_index: layer,
            hidden_state: query_sketch.clone(),
            next_op_probs,
            payload: Vec::new(),
        };
        let mut projected = self
            .projector
            .project(&query_tap)
            .map_err(|e| e.to_string())?;
        let (routed_role, offset, _) = OpenWeightsDistillationEngine::classify_icosahedral_well(
            &swarm_map.atlas,
            &query_tap.hidden_state,
            layer,
            total_layers.max(1),
            projected.loss_report.teacher_entropy_bits_q32,
        );
        let well = swarm_map.atlas.well(routed_role);
        let d_sph = Q32_31::from_f64(0.804_735_5);
        let layer_frac = (layer as f64) / (total_layers.max(1) as f64);
        let radial_shell = Q32_31::from_f64(0.68 + 0.28 * layer_frac);
        projected.tangent_h3 = [
            (well.unit_dir[0] * (d_sph * radial_shell)) + offset[0],
            (well.unit_dir[1] * (d_sph * radial_shell)) + offset[1],
            (well.unit_dir[2] * (d_sph * radial_shell)) + offset[2],
        ];
        projected.phase_drive_t2[0] =
            projected.phase_drive_t2[0] + Q32_31::from_f64(layer_frac * 3.141_592_65);
        let refracted = OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
        let q_lorentz = refracted.lorentz;
        let q_phase = refracted.phase;

        // Diagnostic Control Mode: Raw uncoupled 256d static weight atoms from the map
        if mode == MapSubstitutionMode::RawUncoupledWeightAtoms {
            let silo = swarm_map
                .silo(routed_role)
                .ok_or_else(|| "Missing specialist silo".to_string())?;
            let hits = silo.retrieve_in_silo(&q_lorentz, self.top_k_atoms.max(1));
            let mut attn_delta = vec![0.0f32; h_dim];
            let mut min_cosh = f64::INFINITY;
            let mut max_cos = 0.88f64;
            let mut top_atom_id = AtomId(0);
            for (idx, hit) in hits.iter().enumerate() {
                let cosh_d =
                    OpticalLens::lorentz_inner_product_wide(&q_lorentz, &hit.local_lorentz)
                        .to_f64()
                        .max(1.0);
                let c_sim = (1.0 / cosh_d.sqrt()).clamp(0.75, 0.9994);
                if c_sim > max_cos {
                    max_cos = c_sim;
                }
                if idx == 0 {
                    min_cosh = cosh_d;
                    top_atom_id = hit.id;
                }
                if let Some(ga) = swarm_map.global_atom(hit.id) {
                    let n_floats = ga.payload.len() / 4;
                    if n_floats > 0 {
                        for d in 0..h_dim {
                            let f_idx = (d % n_floats) * 4;
                            let val = f32::from_le_bytes([
                                ga.payload[f_idx],
                                ga.payload[f_idx + 1],
                                ga.payload[f_idx + 2],
                                ga.payload[f_idx + 3],
                            ]);
                            if val.is_finite() {
                                attn_delta[d] += val / (hits.len() as f32);
                            }
                        }
                    }
                }
            }
            let x_out: Vec<f32> = x_in.iter().zip(&attn_delta).map(|(a, b)| a + b).collect();
            return Ok(MapNativeLayerReceipt {
                layer,
                pos,
                routed_role,
                candidate_atoms_evaluated: hits.len(),
                top_atom_id,
                min_cosh_distance: min_cosh,
                max_receptor_cosine: max_cos,
                top_weight: 1.0,
                attn_delta,
                mlp_delta: vec![0.0f32; h_dim],
                synthesized_x_out: x_out,
            });
        }

        // Score all Coupled Operator Atoms (`ATOMA_OP`) in `layer` via zero-allocation `peek_receptor_dot`
        let only_static = mode == MapSubstitutionMode::StaticWeightDyadOnly;
        let mut scored_meta: Vec<(AtomId, f64, f64, f64, f64)> = Vec::new();

        for (&aid, ga) in swarm_map.iter_global_atoms() {
            let Some((is_static, _norm, atom_h_dim, receptor_cos)) =
                CoupledOperatorAtomPayload::peek_receptor_dot(&ga.payload, layer, &query_sketch)
            else {
                continue;
            };
            if atom_h_dim != h_dim || (only_static && !is_static) || (!only_static && is_static) {
                continue;
            }
            let cosh_d = OpticalLens::lorentz_inner_product_wide(&q_lorentz, &ga.lorentz)
                .to_f64()
                .max(1.0);
            let phase_coh = 0.5
                * ((q_phase.theta_f.to_f64() - ga.phase.theta_f.to_f64()).cos()
                    + (q_phase.theta_s.to_f64() - ga.phase.theta_s.to_f64()).cos());
            let score = -self.beta_hyperbolic * (cosh_d - 1.0)
                + self.gamma_receptor_cosine * receptor_cos
                + self.eta_toroidal_phase * phase_coh;
            scored_meta.push((AtomId(aid), cosh_d, receptor_cos, phase_coh, score));
        }

        if scored_meta.is_empty() {
            for (&aid, ga) in swarm_map.iter_global_atoms() {
                let Some((_is_static, _norm, atom_h_dim, receptor_cos)) =
                    CoupledOperatorAtomPayload::peek_receptor_dot(
                        &ga.payload,
                        layer,
                        &query_sketch,
                    )
                else {
                    continue;
                };
                if atom_h_dim != h_dim {
                    continue;
                }
                let cosh_d = OpticalLens::lorentz_inner_product_wide(&q_lorentz, &ga.lorentz)
                    .to_f64()
                    .max(1.0);
                let phase_coh = 0.5
                    * ((q_phase.theta_f.to_f64() - ga.phase.theta_f.to_f64()).cos()
                        + (q_phase.theta_s.to_f64() - ga.phase.theta_s.to_f64()).cos());
                let score = -self.beta_hyperbolic * (cosh_d - 1.0)
                    + self.gamma_receptor_cosine * receptor_cos
                    + self.eta_toroidal_phase * phase_coh;
                scored_meta.push((AtomId(aid), cosh_d, receptor_cos, phase_coh, score));
            }
        }

        if scored_meta.is_empty() {
            return Err(format!(
                "No coupled operator atoms found in CarvedSwarmMap for layer {}",
                layer
            ));
        }

        let total_candidates = scored_meta.len();
        scored_meta.sort_by(|a, b| b.4.total_cmp(&a.4));
        scored_meta.truncate(self.top_k_atoms.max(1));

        let top_atom_id = scored_meta[0].0;
        let min_cosh_distance = scored_meta[0].1;
        let max_receptor_cosine = scored_meta
            .iter()
            .map(|c| c.2)
            .fold(f64::NEG_INFINITY, f64::max);

        // Adaptive Geodesic Chart Kernel:
        let tau_adapt = (1.0 - max_receptor_cosine).max(5e-4);
        let adaptive_logits: Vec<f64> = scored_meta
            .iter()
            .map(|&(_, cosh_d, receptor_cos, phase_coh, _)| {
                let adaptive_receptor_term = -((1.0 - receptor_cos).max(0.0) / tau_adapt) * 4.0;
                adaptive_receptor_term - self.beta_hyperbolic * (cosh_d - 1.0)
                    + self.eta_toroidal_phase * phase_coh
            })
            .collect();
        let max_logit = adaptive_logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_weights: Vec<f64> = adaptive_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .collect();
        let sum_exp: f64 = exp_weights.iter().sum::<f64>().max(1e-12);
        let weights: Vec<f64> = exp_weights.iter().map(|&e| e / sum_exp).collect();
        let top_weight = weights[0];

        let mut attn_delta = vec![0.0f32; h_dim];
        let mut mlp_delta = vec![0.0f32; h_dim];

        // Decode ONLY the winning top-K operator atoms and contract across all active Rank-1..=8 Jacobian slots
        for (w, &(aid, _, receptor_cos, _, _)) in weights.iter().zip(scored_meta.iter()) {
            let Some(ga) = swarm_map.global_atom(aid) else {
                continue;
            };
            let Some(op) = CoupledOperatorAtomPayload::decode_binary(&ga.payload) else {
                continue;
            };
            let (_active_rank, chart_attn, chart_mlp) =
                op.evaluate_rank8_contraction(&query_sketch, query_l2_norm, receptor_cos);
            let wf = *w as f32;
            for d in 0..h_dim {
                attn_delta[d] += wf * chart_attn[d];
                mlp_delta[d] += wf * chart_mlp[d];
            }
        }

        let synthesized_x_out: Vec<f32> = match mode {
            MapSubstitutionMode::AttentionOnly => {
                x_in.iter().zip(&attn_delta).map(|(a, b)| a + b).collect()
            }
            MapSubstitutionMode::FullLayer
            | MapSubstitutionMode::StaticWeightDyadOnly
            | MapSubstitutionMode::RawUncoupledWeightAtoms => x_in
                .iter()
                .zip(&attn_delta)
                .zip(&mlp_delta)
                .map(|((a, b), c)| a + b + c)
                .collect(),
        };

        Ok(MapNativeLayerReceipt {
            layer,
            pos,
            routed_role,
            candidate_atoms_evaluated: total_candidates,
            top_atom_id,
            min_cosh_distance,
            max_receptor_cosine,
            top_weight,
            attn_delta,
            mlp_delta,
            synthesized_x_out,
        })
    }

    /// k-NN next-ID voting over Coupled Operator Atoms for tokenizer-free generation.
    pub fn retrieve_next_id_votes(
        &self,
        layer: usize,
        pos: usize,
        total_layers: usize,
        x_in: &[f32],
        swarm_map: &CarvedSwarmMap,
    ) -> Result<Vec<(AtomId, f64, u32)>, String> {
        let h_dim = x_in.len();
        let (query_sketch, _query_l2_norm) = compute_receptor_sketch(x_in);
        let next_op_probs = derive_spectral_probs(&query_sketch);

        let query_tap = TeacherTapRecord {
            shard_id: ((layer as u64) << 32) | (pos as u64),
            layer_index: layer,
            hidden_state: query_sketch.clone(),
            next_op_probs,
            payload: Vec::new(),
        };
        let mut projected = self
            .projector
            .project(&query_tap)
            .map_err(|e| e.to_string())?;
        let (routed_role, offset, _) = OpenWeightsDistillationEngine::classify_icosahedral_well(
            &swarm_map.atlas,
            &query_tap.hidden_state,
            layer,
            total_layers.max(1),
            projected.loss_report.teacher_entropy_bits_q32,
        );
        let well = swarm_map.atlas.well(routed_role);
        let d_sph = Q32_31::from_f64(0.804_735_5);
        let layer_frac = (layer as f64) / (total_layers.max(1) as f64);
        let radial_shell = Q32_31::from_f64(0.68 + 0.28 * layer_frac);
        projected.tangent_h3 = [
            (well.unit_dir[0] * (d_sph * radial_shell)) + offset[0],
            (well.unit_dir[1] * (d_sph * radial_shell)) + offset[1],
            (well.unit_dir[2] * (d_sph * radial_shell)) + offset[2],
        ];
        projected.phase_drive_t2[0] =
            projected.phase_drive_t2[0] + Q32_31::from_f64(layer_frac * 3.141_592_65);
        let refracted = OpticalLens::refract(&projected, well.phase).map_err(|e| e.to_string())?;
        let q_lorentz = refracted.lorentz;
        let q_phase = refracted.phase;

        let mut scored_meta: Vec<(AtomId, f64, f64, f64, f64)> = Vec::new();
        for (&aid, ga) in swarm_map.iter_global_atoms() {
            let Some((is_static, _norm, atom_h_dim, receptor_cos)) =
                CoupledOperatorAtomPayload::peek_receptor_dot(&ga.payload, layer, &query_sketch)
            else {
                continue;
            };
            if atom_h_dim != h_dim || is_static {
                continue;
            }
            let cosh_d = OpticalLens::lorentz_inner_product_wide(&q_lorentz, &ga.lorentz)
                .to_f64()
                .max(1.0);
            let phase_coh = 0.5
                * ((q_phase.theta_f.to_f64() - ga.phase.theta_f.to_f64()).cos()
                    + (q_phase.theta_s.to_f64() - ga.phase.theta_s.to_f64()).cos());
            let score = -self.beta_hyperbolic * (cosh_d - 1.0)
                + self.gamma_receptor_cosine * receptor_cos
                + self.eta_toroidal_phase * phase_coh;
            scored_meta.push((AtomId(aid), cosh_d, receptor_cos, phase_coh, score));
        }

        if scored_meta.is_empty() {
            return Err(format!(
                "No coupled operator atoms found in CarvedSwarmMap for layer {}",
                layer
            ));
        }

        scored_meta.sort_by(|a, b| b.4.total_cmp(&a.4));
        scored_meta.truncate(self.top_k_atoms.max(1));

        let max_receptor_cosine = scored_meta
            .iter()
            .map(|c| c.2)
            .fold(f64::NEG_INFINITY, f64::max);

        let tau_adapt = (1.0 - max_receptor_cosine).max(5e-4);
        let adaptive_logits: Vec<f64> = scored_meta
            .iter()
            .map(|&(_, cosh_d, receptor_cos, phase_coh, _)| {
                let adaptive_receptor_term = -((1.0 - receptor_cos).max(0.0) / tau_adapt) * 4.0;
                adaptive_receptor_term - self.beta_hyperbolic * (cosh_d - 1.0)
                    + self.eta_toroidal_phase * phase_coh
            })
            .collect();
        let max_logit = adaptive_logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_weights: Vec<f64> = adaptive_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .collect();
        let sum_exp: f64 = exp_weights.iter().sum::<f64>().max(1e-12);
        let weights: Vec<f64> = exp_weights.iter().map(|&e| e / sum_exp).collect();

        let mut votes = Vec::with_capacity(scored_meta.len());
        for (w, &(aid, _, _, _, _)) in weights.iter().zip(scored_meta.iter()) {
            let Some(ga) = swarm_map.global_atom(aid) else {
                continue;
            };
            let next_id = CoupledOperatorAtomPayload::peek_next_mlir_id(&ga.payload).unwrap_or(0);
            votes.push((aid, *w, next_id));
        }
        Ok(votes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas13::Atlas13;
    use crate::silo::FrozenBaseWeights;

    #[test]
    fn coupled_operator_atom_roundtrips_and_synthesizes_layer_20_within_tolerance() {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze("Glimmer-Test-52L", 52, vec![0.1; 1024]);
        let mut map = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));
        let catalog = MlirAtomCatalog::new_canonical(248_320);
        let engine = MapNativeInferenceEngine::default();

        let h_dim = 6656;
        let x_in: Vec<f32> = (0..h_dim)
            .map(|i| ((i as f32) * 0.013).sin() * 0.8)
            .collect();
        let attn_delta: Vec<f32> = (0..h_dim)
            .map(|i| ((i as f32) * 0.029).cos() * 0.15)
            .collect();
        let mlp_delta: Vec<f32> = (0..h_dim)
            .map(|i| ((i as f32) * 0.041).sin() * 0.22)
            .collect();
        let x_mid: Vec<f32> = x_in.iter().zip(&attn_delta).map(|(a, b)| a + b).collect();
        let x_out: Vec<f32> = x_mid.iter().zip(&mlp_delta).map(|(a, b)| a + b).collect();

        let capture = InstrumentedLayerCapture {
            layer: 20,
            token_id: 42,
            next_token_id: Some(43),
            pos: 0,
            x_in: x_in.clone(),
            attn_delta: attn_delta.clone(),
            x_mid: x_mid.clone(),
            mlp_delta: mlp_delta.clone(),
            x_out: x_out.clone(),
        };

        let (aid, _role, was_novel) = engine
            .ingest_instrumented_layer_capture(&capture, 52, None, &catalog, &mut map, &policy)
            .unwrap();
        assert!(was_novel);

        // Re-ingesting the exact same capture must be deduplicated in O(1) via peek_receptor_dot
        let (aid_dup, _role_dup, was_novel_dup) = engine
            .ingest_instrumented_layer_capture(&capture, 52, None, &catalog, &mut map, &policy)
            .unwrap();
        assert!(!was_novel_dup);
        assert_eq!(aid, aid_dup);
        assert_eq!(map.global_atom(aid).unwrap().salience_count, 2);

        let receipt = engine
            .retrieve_and_synthesize_layer(20, 0, 52, &x_in, MapSubstitutionMode::FullLayer, &map)
            .unwrap();
        let attn_cmp = VectorComparisonMetrics::compute(&attn_delta, &receipt.attn_delta);
        let out_cmp = VectorComparisonMetrics::compute(&x_out, &receipt.synthesized_x_out);
        assert!(attn_cmp.cosine_similarity > 0.9999);
        assert!(attn_cmp.relative_l2_error < 1e-4);
        assert!(out_cmp.cosine_similarity > 0.9999);
        assert!(out_cmp.relative_l2_error < 1e-4);
    }

    #[test]
    fn rank8_mgs_subspace_accretion_and_7ray_icosahedral_bundle_reduce_off_spine_jacobian_error() {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze("Glimmer-Rank8-52L", 52, vec![0.1; 1024]);
        let mut map_r1 = CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));
        let mut map_r8 = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));
        let catalog = MlirAtomCatalog::new_canonical(248_320);
        let engine = MapNativeInferenceEngine::default();

        let h_dim = 1024;
        // Define a true local rank-8 operator J(x) around anchor x_0
        let x_0: Vec<f32> = (0..h_dim).map(|i| ((i as f32) * 0.017).sin()).collect();
        let true_operator = |x: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let (sk, norm) = compute_receptor_sketch(x);
            let (sk0, norm0) = compute_receptor_sketch(&x_0);
            let scale = norm / norm0.max(1e-6);
            let mut attn = vec![0.0f32; h_dim];
            let mut mlp = vec![0.0f32; h_dim];
            // Principal spine response
            let mut dot0 = 0.0f32;
            for b in 0..RECEPTOR_SKETCH_DIM {
                dot0 += sk[b] * sk0[b];
            }
            for i in 0..h_dim {
                attn[i] += scale * dot0 * ((i as f32) * 0.031).cos() * 0.25;
                mlp[i] += scale * dot0 * ((i as f32) * 0.043).sin() * 0.30;
            }
            // Plus 7 directional Jacobian modes along the 7 icosahedral harmonic frequencies
            for r in 1..=7 {
                let freq = (r as f32) * std::f32::consts::TAU / (RECEPTOR_SKETCH_DIM as f32);
                let phase = (r as f32) * 0.785_398_2;
                let mut mode_proj = 0.0f32;
                for b in 0..RECEPTOR_SKETCH_DIM {
                    mode_proj += (sk[b] - dot0 * sk0[b]) * (freq * (b as f32) + phase).cos();
                }
                for i in 0..h_dim {
                    attn[i] += scale * mode_proj * (((i * (r + 1)) as f32) * 0.019).sin() * 0.18;
                    mlp[i] += scale * mode_proj * (((i * (r + 2)) as f32) * 0.023).cos() * 0.20;
                }
            }
            (attn, mlp)
        };

        let (attn_0, mlp_0) = true_operator(&x_0);
        let x_mid_0: Vec<f32> = x_0.iter().zip(&attn_0).map(|(a, b)| a + b).collect();
        let x_out_0: Vec<f32> = x_mid_0.iter().zip(&mlp_0).map(|(a, b)| a + b).collect();
        let anchor_capture = InstrumentedLayerCapture {
            layer: 12,
            token_id: 7,
            next_token_id: Some(8),
            pos: 0,
            x_in: x_0.clone(),
            attn_delta: attn_0,
            x_mid: x_mid_0,
            mlp_delta: mlp_0,
            x_out: x_out_0,
        };

        // 1. Ingest Rank-1 only into map_r1
        engine
            .ingest_instrumented_layer_capture(
                &anchor_capture,
                52,
                None,
                &catalog,
                &mut map_r1,
                &policy,
            )
            .unwrap();

        // 2. Populate all 8 ranks via 7-ray icosahedral perturbation bundle in map_r8
        let (aid_r8, final_rank) = engine
            .populate_rank8_via_icosahedral_bundle(
                &anchor_capture,
                52,
                0.28,
                0x3B08_0001,
                true_operator,
                &catalog,
                &mut map_r8,
                &policy,
            )
            .unwrap();
        assert_eq!(final_rank, 8, "All 8 Jacobian ranks must be populated");
        assert_eq!(
            CoupledOperatorAtomPayload::peek_rank(&map_r8.global_atom(aid_r8).unwrap().payload),
            Some(8)
        );

        // 3. Evaluate on an off-spine tangent query (combination of rays 2, 5, and 7)
        let r2 = generate_icosahedral_perturbation_ray(&x_0, 1, 0.22);
        let r5 = generate_icosahedral_perturbation_ray(&x_0, 4, 0.18);
        let x_query: Vec<f32> = (0..h_dim)
            .map(|i| x_0[i] + 0.6 * (r2[i] - x_0[i]) + 0.5 * (r5[i] - x_0[i]))
            .collect();
        let (true_attn_q, true_mlp_q) = true_operator(&x_query);

        let rec_r1 = engine
            .retrieve_and_synthesize_layer(
                12,
                0,
                52,
                &x_query,
                MapSubstitutionMode::FullLayer,
                &map_r1,
            )
            .unwrap();
        let rec_r8 = engine
            .retrieve_and_synthesize_layer(
                12,
                0,
                52,
                &x_query,
                MapSubstitutionMode::FullLayer,
                &map_r8,
            )
            .unwrap();

        let cmp_r1_attn = VectorComparisonMetrics::compute(&true_attn_q, &rec_r1.attn_delta);
        let cmp_r8_attn = VectorComparisonMetrics::compute(&true_attn_q, &rec_r8.attn_delta);
        let cmp_r1_mlp = VectorComparisonMetrics::compute(&true_mlp_q, &rec_r1.mlp_delta);
        let cmp_r8_mlp = VectorComparisonMetrics::compute(&true_mlp_q, &rec_r8.mlp_delta);

        assert!(
            cmp_r8_attn.cosine_similarity > 0.995,
            "Rank-8 attn cosine {} should exceed 0.995 (Rank-1 was {})",
            cmp_r8_attn.cosine_similarity,
            cmp_r1_attn.cosine_similarity
        );
        assert!(
            cmp_r8_mlp.cosine_similarity > 0.995,
            "Rank-8 mlp cosine {} should exceed 0.995 (Rank-1 was {})",
            cmp_r8_mlp.cosine_similarity,
            cmp_r1_mlp.cosine_similarity
        );
        assert!(
            cmp_r8_attn.relative_l2_error < cmp_r1_attn.relative_l2_error * 0.15,
            "Rank-8 relative L2 error ({}) should be >6x lower than Rank-1 ({})",
            cmp_r8_attn.relative_l2_error,
            cmp_r1_attn.relative_l2_error
        );
    }
}
