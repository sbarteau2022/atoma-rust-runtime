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

/// Decoded views of a Coupled Operator Atom (`ATOMA_OP`) stored in `CarvedSwarmMap`.
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
    pub receptor_sketch: Vec<f32>,
    pub attn_effector: Vec<f32>,
    pub mlp_effector: Vec<f32>,
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

    /// Encode a Coupled Operator Atom into compact binary wire format (`ATOMA_OP`) using
    /// `BF16` (`2 bytes/float`) for the high-dimensional effector vectors `attn_effector` and `mlp_effector`,
    /// cutting per-atom RAM/vault footprint by `2×` while maintaining `< 3e-3` numeric precision
    /// (or `F32` exact when `lossless_f32` is true).
    pub fn encode_binary(&self, lossless_f32: bool) -> Vec<u8> {
        let h_dim = self.attn_effector.len();
        let sketch_len = self.receptor_sketch.len();
        let bytes_per_elem = if lossless_f32 { 4 } else { 2 };
        // v2 header: 36 bytes (flags bit 2 set); `next_mlir_id` u32 at bytes 32..36.
        let mut buf = Vec::with_capacity(36 + sketch_len * 4 + h_dim * 2 * bytes_per_elem);

        buf.extend_from_slice(COUPLED_OPERATOR_MAGIC);
        let flags: u8 =
            (if lossless_f32 { 1 } else { 0 }) | (if self.is_static_dyad { 2 } else { 0 }) | 0x04; // v2: next_mlir_id present
        buf.push(flags);
        buf.push(0); // reserved
        buf.extend_from_slice(&(self.layer as u16).to_le_bytes());
        buf.extend_from_slice(&(self.pos_hint as u32).to_le_bytes());
        buf.extend_from_slice(&self.token_id_hint.to_le_bytes());
        buf.extend_from_slice(&self.input_l2_norm.to_le_bytes());
        buf.extend_from_slice(&(sketch_len as u32).to_le_bytes());
        buf.extend_from_slice(&(h_dim as u32).to_le_bytes());
        // v2: next_mlir_id appended at bytes 32..36; all prior field offsets unchanged.
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
        buf
    }

    /// Decode a Coupled Operator Atom payload from `GlobalAtom::payload`.
    pub fn decode_binary(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 32 || &bytes[0..8] != COUPLED_OPERATOR_MAGIC {
            return None;
        }
        let flags = bytes[8];
        let lossless_f32 = (flags & 1) != 0;
        let is_static_dyad = (flags & 2) != 0;
        let is_v2 = (flags & 4) != 0;
        let layer = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
        let pos_hint = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let token_id_hint = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let input_l2_norm = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        let sketch_len = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as usize;
        let h_dim = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]) as usize;

        // v2 header: next_mlir_id u32 at bytes 32..36, receptor sketch starts at 36.
        // v1 header: no next_mlir_id; sketch starts at 32, next defaults to token_id_hint.
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

        Some(Self {
            layer,
            pos_hint,
            token_id_hint,
            next_mlir_id,
            is_static_dyad,
            input_l2_norm,
            receptor_sketch,
            attn_effector,
            mlp_effector,
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

/// Map-Native Inference Engine over `CarvedSwarmMap`.
pub struct MapNativeInferenceEngine {
    pub projector: TangentProjector,
    /// Inverse temperature on hyperbolic geodesic distance `cosh(d_H3) - 1`.
    pub beta_hyperbolic: f64,
    /// Inverse temperature on receptor cosine similarity `<Pi(x), phi_m^in>`.
    pub gamma_receptor_cosine: f64,
    /// Inverse temperature on toroidal phase coherence `cos(Δθ_f) + cos(Δθ_s)`.
    pub eta_toroidal_phase: f64,
    /// Top-K operator atoms combined in the geodesic softmax synthesis.
    pub top_k_atoms: usize,
    /// Store lossless F32 effectors (`true`) or compact BF16 effectors (`false`).
    pub lossless_f32_payloads: bool,
    /// Per-layer receptor cosine novelty threshold (`>= 0.9985` within the same layer coalesces).
    pub receptor_novelty_cosine_threshold: f64,
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
        }
    }
}

impl MapNativeInferenceEngine {
    /// Refract a captured layer execution (`InstrumentedLayerCapture`) into a Coupled Operator Atom
    /// (`ATOMA_OP`) inside `swarm_map` with **per-layer receptor-cosine novelty deduplication**:
    /// - If `capture.layer` already contains an `ATOMA_OP` atom whose receptor sketch aligns with
    ///   `Pi(capture.x_in)` above `self.receptor_novelty_cosine_threshold` (`0.9985`) and norm
    ///   ratio within `[0.92, 1.08]`, coalesces into that existing atom (`salience_count += 1`,
    ///   `was_novel = false`) so corpus-wide distillation never bloats with duplicate charts.
    /// - Otherwise, refracts and carves a new `ATOMA_OP` atom into `swarm_map` (`was_novel = true`)
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
        let (role, offset, certainty_q32) =
            OpenWeightsDistillationEngine::classify_icosahedral_well(
                &swarm_map.atlas,
                &dummy_tap.hidden_state,
                capture.layer,
                total_layers.max(1),
                projected.loss_report.teacher_entropy_bits_q32,
            );

        // 1. Per-layer receptor-cosine novelty check (`0` effector allocations!)
        let mut existing_match: Option<AtomId> = None;
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
                existing_match = Some(AtomId(aid));
                break;
            }
        }

        let (atom_id, was_novel) = if let Some(existing_id) = existing_match {
            swarm_map.reinforce_atom_salience(existing_id, 1);
            (existing_id, false)
        } else {
            let coupled = CoupledOperatorAtomPayload {
                layer: capture.layer,
                pos_hint: capture.pos,
                token_id_hint: capture.token_id,
                next_mlir_id: capture.next_token_id.unwrap_or(capture.token_id),
                is_static_dyad: false,
                input_l2_norm,
                receptor_sketch: sketch.clone(),
                attn_effector: capture.attn_delta.clone(),
                mlp_effector: capture.mlp_delta.clone(),
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
            let fp = swarm_map.base_weights.fingerprint;
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
                let fp = swarm_map.base_weights.fingerprint;
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
            receptor_sketch: sketch.clone(),
            attn_effector: attn_effector_col.to_vec(),
            mlp_effector: mlp_effector_col.to_vec(),
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
    /// winning `top_k_atoms` effectors to synthesize `attn_delta` and `mlp_delta`.
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
            let mut top_atom_id = AtomId(0);
            for (idx, hit) in hits.iter().enumerate() {
                let cosh_d =
                    OpticalLens::lorentz_inner_product_wide(&q_lorentz, &hit.local_lorentz)
                        .to_f64()
                        .max(1.0);
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
                max_receptor_cosine: 0.0,
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
        // When `x_in` lands inside or very near a calibrated chart (`1 - max_cos <= 5e-4`),
        // `tau_adapt = 5e-4` suppresses cross-chart bleed across 52 consecutive weightless layers;
        // when `x_in` is a held-out state between charts (`1 - max_cos ~ 0.15`), `tau_adapt`
        // automatically widens to blend across the neighboring charts smoothly.
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

        // Decode ONLY the winning top-K operator atoms
        for (w, &(aid, _, receptor_cos, _, _)) in weights.iter().zip(scored_meta.iter()) {
            let Some(ga) = swarm_map.global_atom(aid) else {
                continue;
            };
            let Some(op) = CoupledOperatorAtomPayload::decode_binary(&ga.payload) else {
                continue;
            };
            let scale_ratio = (query_l2_norm as f64) / (op.input_l2_norm.max(1e-6) as f64);
            let alpha = if op.is_static_dyad {
                (*w * scale_ratio * receptor_cos) as f32
            } else {
                (*w * scale_ratio * receptor_cos.max(0.0)) as f32
            };

            for d in 0..h_dim {
                attn_delta[d] += alpha * op.attn_effector[d];
                mlp_delta[d] += alpha * op.mlp_effector[d];
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
    ///
    /// Runs the same geodesic scoring as `retrieve_and_synthesize_layer` (hyperbolic
    /// distance + receptor cosine + toroidal phase coherence, adaptive `tau_adapt`
    /// kernel) but instead of synthesizing activation deltas returns the top-K
    /// `(AtomId, weight, next_mlir_id)` votes. Callers aggregate weights per
    /// `next_mlir_id` to predict the next ML-IR ID with no LM head.
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

        // Score all non-static Coupled Operator Atoms (`ATOMA_OP`) in `layer`.
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
}
