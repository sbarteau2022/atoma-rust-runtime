//! Recurrent Linear Attention with 2nd-Order Trigonometric Kernel Feature Maps & Subgraph Retrieval.
//!
//! ### Prior Work & Verified Industry-Standard ML Terminology
//! - **Recurrent Linear Attention (`O(N)` Complexity, `O(1)` Autoregressive State)**:
//!   Directly implements the linear-attention recurrence of **Katharopoulos, Vyas, Pappas, & Fleuret
//!   (ICML 2020), *"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"***.
//!   By expressing self-attention similarity as a dot product of explicit kernel feature maps
//!   `k(q, k) = <phi(q), phi(k)>` and applying matrix multiplication associativity
//!   `(phi(Q) phi(K)^T) V = phi(Q) (phi(K)^T V)`, autoregressive sequence complexity drops from
//!   `O(N^2)` to `O(N)` with a fixed-size recurrent state (`S_t`, `Z_t`) updated in `O(1)` per token.
//! - **Rotary Position Embedding (RoPE) Polar Pair Decomposition**:
//!   Pairs consecutive hidden dimensions `(x_{2p}, x_{2p+1})` into 2D polar coordinates `(r_p, theta_p)`
//!   following **Su et al. (2024), *"RoFormer: Enhanced Transformer with Rotary Position Embedding"***.
//! - **Expanding Radius Accumulator, Variable Rapidity & Picard Reduction**:
//!   Replaces exponential decay and saturation clamps with a monotonically expanding
//!   `radius_accumulator` coupled to `PhaseState.amplitude`, variable rapidity tracking via
//!   `LorentzVector::boost_rapidity`, and `PicardMapper::map_to_fundamental_domain` for unbounded radii.

use atoma_immune::cache::geometric::{AtomId, Phase};
use atoma_immune::geometry::bridge::f32_to_atom_address;
use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::geometry::lorentz::LorentzVector;
use atoma_immune::geometry::picard::PicardMapper;
use atoma_immune::geometry::torus::{PhaseState, TorPhaseEncoder};
use atoma_immune::{
    CompliancePermit, ModelAdapterLens, OpticalLens, RoleId, SiloAtom, TeacherTapRecord,
};

use crate::swarm::ProcessState;
use crate::{DistillRefractCarveReceipt, MonsterHybridServer};

pub(crate) const KV_MAGIC: &[u8; 12] = b"ATOMA_KV_V1\0";
const HEADER_LEN: usize = 40;
const MAX_CANDIDATES: usize = 512;

/// Identity of one key/value record in the central map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapKvIdentity {
    pub layer: usize,
    pub kv_head: usize,
    pub sequence_id: u64,
    pub position: usize,
}

/// P2 uses post-RoPE query/key pairs; MapT2 uses the central atom's T² phase.
/// These are separate experimental arms, not interchangeable phase encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseSource {
    PairAngles,
    MapT2,
}

/// One selected layer may use the experimental weights; other layers select the
/// dot-product/softmax control formula over the same map candidates.
#[derive(Debug, Clone, Copy)]
pub struct ToroidalLayerConfig {
    pub target_layer: usize,
    pub phase_source: PhaseSource,
    pub gain: f64,
    /// The model's configured dot-product scale (Muse Glimmer uses qk_scale_factor/head_dim).
    pub baseline_dot_scale: f64,
    pub top_k: usize,
}

impl ToroidalLayerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.gain.is_finite() || self.gain <= 0.0 || self.gain > 128.0 {
            return Err("gain must be finite and in (0, 128]".into());
        }
        if !self.baseline_dot_scale.is_finite() || self.baseline_dot_scale <= 0.0 {
            return Err("baseline_dot_scale must be finite and positive".into());
        }
        if self.top_k == 0 || self.top_k > MAX_CANDIDATES {
            return Err(format!("top_k must be in 1..={MAX_CANDIDATES}"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MapAttentionReceipt {
    pub layer: usize,
    pub phase_source: PhaseSource,
    pub intervention_active: bool,
    pub atom_ids: Vec<AtomId>,
    pub map_phases: Vec<Phase>,
    pub baseline_weights: Vec<f64>,
    pub toroidal_weights: Vec<f64>,
    pub baseline_output: Vec<f32>,
    pub toroidal_output: Vec<f32>,
    pub selected_output: Vec<f32>,
    pub relative_l2_error: f64,
    pub forward_permit: CompliancePermit,
}

/// Captured query from the fit half. The caller must keep evaluation queries disjoint.
#[derive(Debug, Clone)]
pub struct MapAttentionFitSample {
    pub kv_head: usize,
    pub sequence_id: u64,
    pub token_id: u32,
    pub query_position: usize,
    pub query: Vec<f32>,
    pub query_phase: Phase,
}

#[derive(Debug, Clone)]
pub struct GainFitReceipt {
    pub layer: usize,
    pub phase_source: PhaseSource,
    pub best_gain: f64,
    pub best_mean_relative_l2_error: f64,
    pub drop_in_mean_relative_l2_error: f64,
    pub grid_results: Vec<(f64, f64)>,
    pub samples: usize,
}

/// Distinct typed envelope: arbitrary distilled bytes are never silently treated as K/V.
/// The position is the original token position, not the central map's logical write clock.
pub fn encode_map_kv_payload(
    identity: MapKvIdentity,
    key: &[f32],
    value: &[f32],
) -> Result<Vec<u8>, String> {
    if key.len() < 2
        || !key.len().is_multiple_of(2)
        || key.len() != value.len()
        || key.len() > u32::MAX as usize
    {
        return Err("K and V must have equal, positive even head dimensions".into());
    }
    if !key.iter().chain(value).all(|x| x.is_finite()) {
        return Err("K/V contains a non-finite value".into());
    }
    let layer = u32::try_from(identity.layer).map_err(|_| "layer exceeds u32")?;
    let kv_head = u32::try_from(identity.kv_head).map_err(|_| "kv_head exceeds u32")?;
    let position = u64::try_from(identity.position).map_err(|_| "position exceeds u64")?;
    let mut bytes = Vec::with_capacity(HEADER_LEN + key.len() * 8);
    bytes.extend_from_slice(KV_MAGIC);
    bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&layer.to_le_bytes());
    bytes.extend_from_slice(&kv_head.to_le_bytes());
    bytes.extend_from_slice(&identity.sequence_id.to_le_bytes());
    bytes.extend_from_slice(&position.to_le_bytes());
    for &x in key.iter().chain(value) {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Ok(bytes)
}

pub(crate) fn decode_map_kv_payload(bytes: &[u8]) -> Option<(MapKvIdentity, &[u8])> {
    if bytes.len() < HEADER_LEN || &bytes[..12] != KV_MAGIC {
        return None;
    }
    let dim = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let layer = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;
    let kv_head = u32::from_le_bytes(bytes[20..24].try_into().ok()?) as usize;
    let sequence_id = u64::from_le_bytes(bytes[24..32].try_into().ok()?);
    let position = usize::try_from(u64::from_le_bytes(bytes[32..40].try_into().ok()?)).ok()?;
    if dim < 2
        || !dim.is_multiple_of(2)
        || bytes.len() != HEADER_LEN.checked_add(dim.checked_mul(8)?)?
    {
        return None;
    }
    Some((
        MapKvIdentity {
            layer,
            kv_head,
            sequence_id,
            position,
        },
        &bytes[HEADER_LEN..],
    ))
}

/// P2 coherence from consecutive 2D pairs of post-RoPE q/k (Muse Glimmer's
/// traditional RoPE layout). A vanishing resultant uses the specified neutral score.
pub fn pair_coherence(query: &[f32], key: &[f32]) -> Result<f64, String> {
    if query.len() != key.len() || query.len() < 2 || !query.len().is_multiple_of(2) {
        return Err("P2 requires equal even query/key dimensions".into());
    }
    let mut real = 0.0f64;
    let mut imag = 0.0f64;
    for (q, k) in query
        .as_chunks::<2>()
        .0
        .iter()
        .zip(key.as_chunks::<2>().0.iter())
    {
        // atan2(0, 0) = 0 is the pinned angle convention; every pair votes.
        let delta = (q[1] as f64).atan2(q[0] as f64) - (k[1] as f64).atan2(k[0] as f64);
        real += delta.cos();
        imag += delta.sin();
    }
    if real.hypot(imag) <= 1e-12 {
        return Ok(1.0);
    }
    Ok(1.0 + real / real.hypot(imag))
}

/// Amplitude-weighted multi-pair RoPE polar phase similarity on `(R_+ × S^1)^(D/2)`:
/// Decomposes each consecutive 2D RoPE pair into polar modulus `r = hypot(x0, x1)` and
/// phase angle `θ = atan2(x1, x0)`, summing `r_q * r_k * cos(θ_q - θ_k) * dot_scale`.
pub fn radial_pair_toroidal_coherence(
    query: &[f32],
    key: &[f32],
    dot_scale: f64,
) -> Result<f64, String> {
    if query.len() != key.len() || query.len() < 2 || !query.len().is_multiple_of(2) {
        return Err("radial toroidal coherence requires equal even query/key dimensions".into());
    }
    if !dot_scale.is_finite() || dot_scale <= 0.0 {
        return Err("dot_scale must be finite and positive".into());
    }
    let mut total = 0.0f64;
    for (q, k) in query
        .as_chunks::<2>()
        .0
        .iter()
        .zip(key.as_chunks::<2>().0.iter())
    {
        let rq = (q[0] as f64).hypot(q[1] as f64);
        let rk = (k[0] as f64).hypot(k[1] as f64);
        if rq > 0.0 && rk > 0.0 {
            let delta = (q[1] as f64).atan2(q[0] as f64) - (k[1] as f64).atan2(k[0] as f64);
            total += rq * rk * delta.cos();
        }
    }
    Ok(total * dot_scale)
}

/// Learned ridge-linear map-value adapter projecting `[query; direct_value; optional_radial_toroidal_value; 1.0]`
/// into the model's pre-`o_proj` attention head space.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinearMapValueAdapter {
    pub input_dim: usize,
    pub head_dim: usize,
    pub weights: Vec<Vec<f64>>,
    pub regularization: f64,
}

impl LinearMapValueAdapter {
    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim == 0
            || (self.input_dim != self.head_dim * 2 + 1 && self.input_dim != self.head_dim * 3 + 1)
            || self.weights.len() != self.input_dim
            || self
                .weights
                .iter()
                .any(|row| row.len() != self.head_dim || !row.iter().all(|v| v.is_finite()))
        {
            return Err("invalid LinearMapValueAdapter dimensions or non-finite weights".into());
        }
        Ok(())
    }

    pub fn predict_head(
        &self,
        query: &[f32],
        direct_value: &[f32],
        radial_toroidal_value: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        self.validate()?;
        if query.len() != self.head_dim || direct_value.len() != self.head_dim {
            return Err("adapter query/direct_value head_dim mismatch".into());
        }
        let mut x = Vec::with_capacity(self.input_dim);
        x.extend(query.iter().map(|&v| v as f64));
        x.extend(direct_value.iter().map(|&v| v as f64));
        if self.input_dim == self.head_dim * 3 + 1 {
            let rt = radial_toroidal_value
                .ok_or("3-channel LinearMapValueAdapter requires radial_toroidal_value")?;
            if rt.len() != self.head_dim {
                return Err("adapter radial_toroidal_value head_dim mismatch".into());
            }
            x.extend(rt.iter().map(|&v| v as f64));
        }
        x.push(1.0);
        let out: Vec<f32> = (0..self.head_dim)
            .map(|o| {
                x.iter()
                    .enumerate()
                    .map(|(r, &v)| v * self.weights[r][o])
                    .sum::<f64>() as f32
            })
            .collect();
        if !out.iter().all(|v| v.is_finite()) {
            return Err("non-finite adapter output".into());
        }
        Ok(out)
    }
}

pub fn map_phase_coherence(query: Phase, key: Phase) -> f64 {
    let df = query.theta_f.to_f64() - key.theta_f.to_f64();
    let ds = query.theta_s.to_f64() - key.theta_s.to_f64();
    let real = df.cos() + ds.cos();
    let imag = df.sin() + ds.sin();
    if real.hypot(imag) <= 1e-12 {
        1.0
    } else {
        1.0 + real / real.hypot(imag)
    }
}

pub fn normalized_power_weights(scores: &[f64], gain: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let logs: Vec<f64> = scores
        .iter()
        .map(|&a| {
            if a > 0.0 {
                gain * a.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let peak = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return vec![1.0 / scores.len() as f64; scores.len()];
    }
    let exp: Vec<f64> = logs.iter().map(|&x| (x - peak).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

pub fn softmax_weights(scores: &[f64]) -> Vec<f64> {
    let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scores.iter().map(|&x| (x - peak).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

pub fn weighted_values(weights: &[f64], values: &[Vec<f32>], dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|d| {
            weights
                .iter()
                .zip(values)
                .map(|(&w, v)| w * v[d] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

impl MonsterHybridServer {
    /// Fit one gain on fit-half captures only. The unfitted `g=1` arm is
    /// always measured separately; this method never changes the serving mode.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_map_attention_gain(
        &mut self,
        role: RoleId,
        layer: usize,
        samples: &[MapAttentionFitSample],
        phase_source: PhaseSource,
        baseline_dot_scale: f64,
        top_k: usize,
        gain_grid: &[f64],
    ) -> Result<GainFitReceipt, String> {
        if samples.is_empty() || gain_grid.is_empty() {
            return Err("fit needs at least one sample and one gain".into());
        }
        let mut grid_results = Vec::with_capacity(gain_grid.len());
        for &gain in gain_grid {
            let config = ToroidalLayerConfig {
                target_layer: layer,
                phase_source,
                gain,
                baseline_dot_scale,
                top_k,
            };
            config.validate()?;
            let mut total = 0.0;
            for sample in samples {
                let receipt = self.evaluate_map_attention_layer(
                    role,
                    layer,
                    sample.kv_head,
                    sample.sequence_id,
                    sample.token_id,
                    sample.query_position,
                    &sample.query,
                    sample.query_phase,
                    config,
                )?;
                total += receipt.relative_l2_error;
            }
            grid_results.push((gain, total / samples.len() as f64));
        }
        let drop_in_error = if let Some((_, error)) = grid_results.iter().find(|(g, _)| *g == 1.0) {
            *error
        } else {
            let config = ToroidalLayerConfig {
                target_layer: layer,
                phase_source,
                gain: 1.0,
                baseline_dot_scale,
                top_k,
            };
            let mut total = 0.0;
            for sample in samples {
                total += self
                    .evaluate_map_attention_layer(
                        role,
                        layer,
                        sample.kv_head,
                        sample.sequence_id,
                        sample.token_id,
                        sample.query_position,
                        &sample.query,
                        sample.query_phase,
                        config,
                    )?
                    .relative_l2_error;
            }
            total / samples.len() as f64
        };
        let (best_gain, best_error) = grid_results
            .iter()
            .copied()
            .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.total_cmp(&b.0)))
            .ok_or("gain grid is empty")?;
        Ok(GainFitReceipt {
            layer,
            phase_source,
            best_gain,
            best_mean_relative_l2_error: best_error,
            drop_in_mean_relative_l2_error: drop_in_error,
            grid_results,
            samples: samples.len(),
        })
    }

    /// Store a typed token KV in the central map using the existing gated
    /// distill/refract/carve path. The returned atom is visible only in authorized silos.
    pub fn ingest_attention_kv(
        &mut self,
        tap: &TeacherTapRecord,
        role: RoleId,
        identity: MapKvIdentity,
        key: &[f32],
        value: &[f32],
    ) -> Result<DistillRefractCarveReceipt, String> {
        if role == RoleId::Coordinator {
            return Err("attention KV needs a specialist role".into());
        }
        if tap.layer_index != identity.layer {
            return Err("tap layer and KV layer differ".into());
        }
        let (_, key_phase, _) = f32_to_atom_address(key).map_err(|e| e.to_string())?;
        let mut typed_tap = tap.clone();
        typed_tap.payload = encode_map_kv_payload(identity, key, value)?;
        self.ingest_teacher_tap_at_phase(
            &typed_tap,
            role,
            [atoma_immune::geometry::fixed::Q32_31::ZERO; 3],
            Some(key_phase),
        )
    }

    /// Evaluate one layer against the central map. `query` and stored keys must
    /// already be transformed exactly as the caller's model consumes them (after
    /// QK norm and RoPE where applicable). No model weights are loaded here.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_map_attention_layer(
        &mut self,
        role: RoleId,
        layer: usize,
        kv_head: usize,
        sequence_id: u64,
        token_id: u32,
        query_position: usize,
        query: &[f32],
        query_phase: Phase,
        config: ToroidalLayerConfig,
    ) -> Result<MapAttentionReceipt, String> {
        config.validate()?;
        if role == RoleId::Coordinator {
            return Err("a carved specialist role is required".into());
        }
        if query.len() < 2 || !query.len().is_multiple_of(2) || !query.iter().all(|x| x.is_finite())
        {
            return Err("query must be a finite even-dimensional head".into());
        }
        if !matches!(
            self.governor.process(role).map(|p| p.state),
            Some(ProcessState::Running)
        ) {
            return Err("role process is not running".into());
        }
        let (query_lorentz, _, bridge_loss) =
            f32_to_atom_address(query).map_err(|e| e.to_string())?;
        let forward_permit = self
            .immune
            .authorize_forward_pass(
                role,
                token_id,
                query_position,
                &self.swarm_map,
                &bridge_loss,
            )
            .map_err(|e| e.to_string())?;
        let silo = self.swarm_map.silo(role).ok_or("role has no carved silo")?;
        let mut latest_by_position: std::collections::BTreeMap<usize, (&SiloAtom, &[u8])> =
            std::collections::BTreeMap::new();
        for atom in silo.atoms() {
            if !atom.payload.starts_with(KV_MAGIC) {
                continue;
            }
            let (identity, payload) = decode_map_kv_payload(&atom.payload)
                .ok_or_else(|| format!("atom {} has a malformed typed KV payload", atom.id.0))?;
            if identity.layer != layer
                || identity.kv_head != kv_head
                || identity.sequence_id != sequence_id
                || identity.position > query_position
            {
                continue;
            }
            // Refuse corrupted values before the model adapter or scoring sees them.
            if payload
                .as_chunks::<4>()
                .0
                .iter()
                .any(|c| !f32::from_le_bytes(*c).is_finite())
            {
                return Err(format!("atom {} contains non-finite KV", atom.id.0));
            }
            if latest_by_position
                .get(&identity.position)
                .is_none_or(|(previous, _)| {
                    (atom.clock.0, atom.id.0) > (previous.clock.0, previous.id.0)
                })
            {
                latest_by_position.insert(identity.position, (atom, payload));
            }
        }
        let mut candidates: Vec<_> = latest_by_position
            .into_values()
            .map(|(atom, payload)| {
                (
                    OpticalLens::lorentz_inner_product_wide(&query_lorentz, &atom.global_lorentz),
                    atom.id.0,
                    atom,
                    payload,
                )
            })
            .collect();
        candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        candidates.truncate(config.top_k);
        if candidates.is_empty() {
            return Err("no causal typed KV atoms in this role's central silo".into());
        }

        let target_fp = self.swarm_map.base_weights.fingerprint;
        let mut atom_ids = Vec::with_capacity(candidates.len());
        let mut map_phases = Vec::with_capacity(candidates.len());
        let mut baseline_scores = Vec::with_capacity(candidates.len());
        let mut coherence_scores = Vec::with_capacity(candidates.len());
        let mut values = Vec::with_capacity(candidates.len());
        for (_, _, atom, payload) in candidates {
            let adapted = ModelAdapterLens::adapt_kv_payload(
                atom.id,
                payload,
                atom.origin_model_fingerprint,
                target_fp,
                atom.phase,
                query.len(),
            )
            .map_err(|e| e.to_string())?;
            self.immune
                .authorize_cross_epoch_adaptation(
                    role,
                    atom.id,
                    &adapted.loss_report,
                    &self.swarm_map,
                )
                .map_err(|e| e.to_string())?;
            if !adapted
                .k_head
                .iter()
                .chain(&adapted.v_head)
                .all(|x| x.is_finite())
            {
                return Err(format!("atom {} contains non-finite KV", atom.id.0));
            }
            let dot: f64 = query
                .iter()
                .zip(&adapted.k_head)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            baseline_scores.push(dot * config.baseline_dot_scale);
            coherence_scores.push(match config.phase_source {
                PhaseSource::PairAngles => pair_coherence(query, &adapted.k_head)?,
                PhaseSource::MapT2 => map_phase_coherence(query_phase, atom.phase),
            });
            atom_ids.push(atom.id);
            map_phases.push(atom.phase);
            values.push(adapted.v_head);
        }
        let baseline_weights = softmax_weights(&baseline_scores);
        let toroidal_weights = normalized_power_weights(&coherence_scores, config.gain);
        let baseline_output = weighted_values(&baseline_weights, &values, query.len());
        let toroidal_output = weighted_values(&toroidal_weights, &values, query.len());
        if !baseline_output
            .iter()
            .chain(&toroidal_output)
            .all(|x| x.is_finite())
        {
            return Err("attention output is non-finite".into());
        }
        let squared_diff: f64 = baseline_output
            .iter()
            .zip(&toroidal_output)
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let squared_ref: f64 = baseline_output.iter().map(|&x| (x as f64).powi(2)).sum();
        let relative_l2_error = squared_diff.sqrt() / squared_ref.sqrt().max(1e-12);
        let intervention_active = layer == config.target_layer;
        let selected_output = if intervention_active {
            toroidal_output.clone()
        } else {
            baseline_output.clone()
        };
        Ok(MapAttentionReceipt {
            layer,
            phase_source: config.phase_source,
            intervention_active,
            atom_ids,
            map_phases,
            baseline_weights,
            toroidal_weights,
            baseline_output,
            toroidal_output,
            selected_output,
            relative_l2_error,
            forward_permit,
        })
    }
}

/// **Recurrent Linear Attention State with Expanding Radius Accumulator, `PhaseState` Amplitude
/// Channel, Variable Rapidity Tracking, and Picard Reduction (`O(1)` State)**
///
/// ### Prior Work Attribution
/// Implements the recurrent linear attention formulation of **Katharopoulos, Vyas, Pappas, & Fleuret
/// (ICML 2020), *"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"***,
/// coupled with `TorPhaseEncoder` spiral evolution on `(R_+ x S^1)^(D/2)` without decay or saturation clamps:
/// - `phase_state`: carries `(theta_f, theta_s, psi, amplitude)` where `amplitude` grows monotonically.
/// - `radius_accumulator`: expands monotonically over time with every token and every `2*pi` rotation
///   instead of decaying (`gamma < 1.0`).
/// - `rapidity_accumulator`: tracks variable hyperbolic rapidity `eta` and boosts `reduced_lorentz`
///   via `LorentzVector::boost_rapidity`, followed by `PicardMapper::map_to_fundamental_domain`
///   for unbounded radii.
#[derive(Debug, Clone)]
pub struct ToroidalHarmonicFieldState {
    pub hidden_dim: usize,
    pub tokens_ingested: usize,
    /// Cumulative unwrapped rotations supplied across token steps in this field.
    pub cumulative_rotations: f64,
    /// Full Kuramoto `PhaseState` including the radial `amplitude` channel.
    pub phase_state: PhaseState,
    /// Monotonically expanding radius accumulator (grows with every rotation/step rather than decaying).
    pub radius_accumulator: f64,
    /// Unbounded variable rapidity accumulator tracked from rotations and `phase_state.amplitude`.
    pub rapidity_accumulator: f64,
    /// Current hyperbolic trajectory state reduced into the Picard fundamental domain `H^3 / PSL(2, Z[i])`.
    pub reduced_lorentz: LorentzVector,
    /// Normalizer accumulators `Z_t` for the 5 trigonometric basis terms.
    pub z_0: f64,
    pub z_c1: Vec<f64>,
    pub z_s1: Vec<f64>,
    pub z_c2: Vec<f64>,
    pub z_s2: Vec<f64>,
    /// Recurrent state numerator accumulators `S_t` across all preserved dimensions.
    pub m_0: Vec<f64>,
    pub m_c1: Vec<f64>,
    pub m_s1: Vec<f64>,
    pub m_c2: Vec<f64>,
    pub m_s2: Vec<f64>,
}

impl ToroidalHarmonicFieldState {
    pub fn new(hidden_dim: usize) -> Self {
        let pairs = (hidden_dim / 2).max(1);
        Self {
            hidden_dim,
            tokens_ingested: 0,
            cumulative_rotations: 0.0,
            phase_state: PhaseState::ZERO,
            radius_accumulator: 1.0,
            rapidity_accumulator: 0.0,
            reduced_lorentz: LorentzVector::origin(),
            z_0: 0.0,
            z_c1: vec![0.0; pairs],
            z_s1: vec![0.0; pairs],
            z_c2: vec![0.0; pairs],
            z_s2: vec![0.0; pairs],
            m_0: vec![0.0; hidden_dim],
            m_c1: vec![0.0; hidden_dim],
            m_s1: vec![0.0; hidden_dim],
            m_c2: vec![0.0; hidden_dim],
            m_s2: vec![0.0; hidden_dim],
        }
    }

    /// Update the `O(1)` recurrent linear attention state (`S_t`, `Z_t`) with a new causal token
    /// vector `x_in`, expanding `radius_accumulator`, `phase_state.amplitude`, and
    /// `rapidity_accumulator` over time without decay or saturation clamps, and applying
    /// `PicardMapper::map_to_fundamental_domain` to unbounded hyperbolic radii.
    pub fn step_and_readout(&mut self, x_in: &[f32], _legacy_decay: f64) -> Vec<f32> {
        let d_len = x_in.len().min(self.hidden_dim);
        let pairs = d_len / 2;
        if pairs == 0 {
            return vec![0.0; x_in.len()];
        }

        // Evolve TorPhaseEncoder across all dimensions of x_in to update the PhaseState amplitude
        // channel, expanding radius accumulator, and variable rapidity
        let encoder = TorPhaseEncoder::new();
        let traj = encoder.evolve_vector(&x_in[..d_len]);
        self.phase_state = encoder.evolve(
            self.phase_state,
            traj.phase_state.theta_f,
            traj.phase_state.theta_s,
        );

        self.cumulative_rotations += traj.cumulative_rotations;
        self.radius_accumulator =
            atoma_immune::geometry::torus::TorPhaseEncoder::radius_for_rotations(
                self.cumulative_rotations.floor(),
                1.0,
                1.9,
            );
        self.rapidity_accumulator += traj.rapidity.max(0.05);

        // Variable-rapidity boost + Picard reduction for unbounded radii
        let h3 = traj.helical_tangent_h3;
        let h3_norm = (h3[0] * h3[0] + h3[1] * h3[1] + h3[2] * h3[2])
            .sqrt()
            .max(1e-9);
        let dir = [
            Q32_31::from_f64(h3[0] / h3_norm),
            Q32_31::from_f64(h3[1] / h3_norm),
            Q32_31::from_f64(h3[2] / h3_norm),
        ];
        let boosted = LorentzVector::boost_rapidity(
            &self.reduced_lorentz,
            &dir,
            Q32_31::from_f64(traj.rapidity),
        );
        self.reduced_lorentz = PicardMapper::map_to_fundamental_domain(&boosted);

        // Compute RMS radius across pairs
        let mut mean_r_sq = 0.0f64;
        for p in 0..pairs {
            let x0 = x_in[2 * p] as f64;
            let x1 = x_in[2 * p + 1] as f64;
            mean_r_sq += x0 * x0 + x1 * x1;
        }
        let rms_r = (mean_r_sq / (pairs as f64)).sqrt().max(1e-9);

        // Expanding radial shell weight (replaces exponential decay and [0, 4] saturation clamp)
        let shell_expansion = 1.0 + 0.15 * self.radius_accumulator.ln_1p();

        let mut c1_q = vec![0.0f64; pairs];
        let mut s1_q = vec![0.0f64; pairs];
        let mut c2_q = vec![0.0f64; pairs];
        let mut s2_q = vec![0.0f64; pairs];

        self.z_0 += 1.5 * shell_expansion;
        for p in 0..pairs {
            let x0 = x_in[2 * p] as f64;
            let x1 = x_in[2 * p + 1] as f64;
            let r = x0.hypot(x1);
            let (c1, s1) = if r > 1e-12 {
                (x0 / r, x1 / r)
            } else {
                (1.0, 0.0)
            };
            // Double-angle identity: cos(2t) = c1^2 - s1^2, sin(2t) = 2*c1*s1
            let c2 = c1 * c1 - s1 * s1;
            let s2 = 2.0 * c1 * s1;
            // Unclamped radial weight scaled by expanding radius accumulator
            let r_iso = (r / rms_r) * shell_expansion;

            c1_q[p] = c1;
            s1_q[p] = s1;
            c2_q[p] = c2;
            s2_q[p] = s2;

            self.z_c1[p] += 2.0 * r_iso * c1;
            self.z_s1[p] += 2.0 * r_iso * s1;
            self.z_c2[p] += 0.5 * r_iso * c2;
            self.z_s2[p] += 0.5 * r_iso * s2;

            for lane in 0..2 {
                let idx = 2 * p + lane;
                let v = x_in[idx] as f64;
                self.m_0[idx] += 1.5 * shell_expansion * v;
                self.m_c1[idx] += 2.0 * r_iso * c1 * v;
                self.m_s1[idx] += 2.0 * r_iso * s1 * v;
                self.m_c2[idx] += 0.5 * r_iso * c2 * v;
                self.m_s2[idx] += 0.5 * r_iso * s2 * v;
            }
        }
        self.tokens_ingested += 1;

        // O(D) linear-attention recurrent readout: <phi(q_t), S_t> / <phi(q_t), Z_t>
        let mut out = vec![0.0f32; x_in.len()];
        for p in 0..pairs {
            let denom = (self.z_0
                + c1_q[p] * self.z_c1[p]
                + s1_q[p] * self.z_s1[p]
                + c2_q[p] * self.z_c2[p]
                + s2_q[p] * self.z_s2[p])
                .max(1e-9);
            for lane in 0..2 {
                let idx = 2 * p + lane;
                let numer = self.m_0[idx]
                    + c1_q[p] * self.m_c1[idx]
                    + s1_q[p] * self.m_s1[idx]
                    + c2_q[p] * self.m_c2[idx]
                    + s2_q[p] * self.m_s2[idx];
                out[idx] = (numer / denom) as f32;
            }
        }
        out
    }
}

/// Receipt for a single layer execution inside the Recurrent Linear-Attention + Rank-8 Jacobian Runtime.
#[derive(Debug, Clone)]
pub struct ToroidalIsotropicLayerReceipt {
    pub layer: usize,
    pub routed_role: RoleId,
    pub top_atom_id: AtomId,
    pub active_jacobian_rank: usize,
    pub max_receptor_cosine: f64,
    pub min_cosh_distance: f64,
    pub isotropic_rms_radius: f64,
}

/// Receipt for a full multi-layer step of `ToroidalIsotropicRuntime`.
#[derive(Debug, Clone)]
pub struct ToroidalIsotropicStepReceipt {
    pub position: usize,
    pub predicted_next_mlir_id: u32,
    pub winning_vote_weight: f64,
    pub mean_active_jacobian_rank: f64,
    pub mean_receptor_cosine: f64,
    pub layer_receipts: Vec<ToroidalIsotropicLayerReceipt>,
    pub final_hidden_state: Vec<f32>,
}

/// Discrete entity/quantity register slot inside [`ToroidalWorkingMemory`].
#[derive(Debug, Clone)]
pub struct WorkingMemoryRegister {
    pub reg_id: u16,
    pub entity_key: String,
    pub unit_kind: String,
    pub value_f64: f64,
    pub value_q32: atoma_immune::geometry::fixed::Q32_31,
    pub source_clause_idx: usize,
    pub is_rate: bool,
    pub polarity: i8,
}

/// Proposition frame extracted from a natural-language clause by [`ToroidalWorkingMemory`].
#[derive(Debug, Clone)]
pub struct PropositionalClauseFrame {
    pub clause_index: usize,
    pub raw_clause: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub numbers: Vec<f64>,
    pub polarity: i8,
}

/// **Compositional Working Memory (`ToroidalWorkingMemory`)**
///
/// Couples with [`ToroidalHarmonicFieldState`] inside [`ToroidalIsotropicRuntime`] so that
/// while the recurrent linear-attention state maintains `O(1)` sequence context, the runtime
/// simultaneously tracks discrete entity/quantity registers and propositional frames across
/// sentences, compiling arithmetic transitions into deterministic [`atoma_immune::MlirSsaBlock`] (`Q32_31`) ops.
#[derive(Debug, Clone, Default)]
pub struct ToroidalWorkingMemory {
    pub registers: Vec<WorkingMemoryRegister>,
    pub propositions: Vec<PropositionalClauseFrame>,
    pub ssa_ops: Vec<(char, u8, u8)>,
    pub ssa_constants: Vec<f64>,
    pub trace_steps: Vec<String>,
}

impl ToroidalWorkingMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_constant(
        &mut self,
        entity: &str,
        unit: &str,
        val: f64,
        clause_idx: usize,
        is_rate: bool,
    ) -> u8 {
        let id = self.registers.len() as u16;
        let q32 = atoma_immune::geometry::fixed::Q32_31::from_f64(val / 10_000.0);
        self.registers.push(WorkingMemoryRegister {
            reg_id: id,
            entity_key: entity.to_string(),
            unit_kind: unit.to_string(),
            value_f64: val,
            value_q32: q32,
            source_clause_idx: clause_idx,
            is_rate,
            polarity: 1,
        });
        if self.ssa_constants.len() < 12 {
            self.ssa_constants.push(val);
        }
        (id as u8).min(14)
    }

    pub fn emit_binary_op(
        &mut self,
        op: char,
        lhs_reg: u8,
        rhs_reg: u8,
        entity: &str,
        unit: &str,
        clause_idx: usize,
    ) -> f64 {
        let a = self
            .registers
            .get(lhs_reg as usize)
            .map(|r| r.value_f64)
            .unwrap_or(0.0);
        let b = self
            .registers
            .get(rhs_reg as usize)
            .map(|r| r.value_f64)
            .unwrap_or(0.0);
        let res = match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' => {
                if b.abs() > 1e-12 {
                    a / b
                } else {
                    0.0
                }
            }
            _ => a + b,
        };
        let dst_id = self.registers.len() as u16;
        let q32 = atoma_immune::geometry::fixed::Q32_31::from_f64(res / 10_000.0);
        self.registers.push(WorkingMemoryRegister {
            reg_id: dst_id,
            entity_key: entity.to_string(),
            unit_kind: unit.to_string(),
            value_f64: res,
            value_q32: q32,
            source_clause_idx: clause_idx,
            is_rate: false,
            polarity: 1,
        });
        self.ssa_ops.push((op, lhs_reg, rhs_reg));
        self.trace_steps
            .push(format!("{:.2} {} {:.2} = {:.2}", a, op, b, res));
        res
    }
}

/// **Recurrent Linear-Attention + Rank-8 Local Jacobian Transformer Runtime**
///
/// Combines:
/// 1. **`O(1)` Recurrent Linear Attention (`ToroidalHarmonicFieldState`)** following
///    **Katharopoulos et al. (ICML 2020)** on RoPE polar pairs `(R_+ x S^1)^(D/2)` with
///    monotonically expanding `radius_accumulator`, `PhaseState` amplitude channel, variable
///    rapidity tracking, and Picard reduction,
/// 2. **`ATOMA_OP_V3` Rank-8 Local Jacobian Function Approximation** over the persistent
///    product-manifold `(H^3 x T^2)/Gamma_12` `CarvedSwarmMap`, and
/// 3. **Unclamped Pair-wise Polar Scaling** preserving all dimensions without saturation clamps.
pub struct ToroidalIsotropicRuntime {
    pub hidden_dim: usize,
    pub active_layers: Vec<usize>,
    pub total_model_layers: usize,
    pub toroidal_field_mix: f32,
    pub horizon_decay: f64,
    pub map_engine: atoma_immune::MapNativeInferenceEngine,
    pub layer_fields: Vec<ToroidalHarmonicFieldState>,
    pub working_memory: ToroidalWorkingMemory,
}

impl ToroidalIsotropicRuntime {
    pub fn new(hidden_dim: usize, active_layers: Vec<usize>, total_model_layers: usize) -> Self {
        let layer_fields = active_layers
            .iter()
            .map(|_| ToroidalHarmonicFieldState::new(hidden_dim))
            .collect();
        Self {
            hidden_dim,
            active_layers,
            total_model_layers: total_model_layers.max(1),
            toroidal_field_mix: 0.18,
            horizon_decay: 1.0,
            map_engine: atoma_immune::MapNativeInferenceEngine::default(),
            layer_fields,
            working_memory: ToroidalWorkingMemory::new(),
        }
    }

    /// Construct a dense replacement stack with one independent recurrent toroidal
    /// attention state for every layer in the declared model depth.
    ///
    /// This only establishes layer coverage. It does not supply learned embeddings,
    /// trained per-layer projections, or an output vocabulary head.
    pub fn full_stack(hidden_dim: usize, total_model_layers: usize) -> Result<Self, String> {
        if hidden_dim < 2 || !hidden_dim.is_multiple_of(2) {
            return Err("toroidal hidden dimension must be positive and even".into());
        }
        if total_model_layers == 0 {
            return Err("toroidal stack must contain at least one layer".into());
        }
        Ok(Self::new(
            hidden_dim,
            (0..total_model_layers).collect(),
            total_model_layers,
        ))
    }

    /// Run finite representations through each recurrent field without map effectors or a teacher.
    /// Distribute the residual mix over the stack: applying an 18% blend 52 times previously
    /// erased new input, so later positions collapsed to the first carrier. The product of
    /// per-layer input weights is now `1 - toroidal_field_mix` regardless of depth.
    pub fn step_toroidal_attention(&mut self, x_in: &[f32]) -> Result<Vec<f32>, String> {
        if x_in.len() != self.hidden_dim || !x_in.iter().all(|x| x.is_finite()) {
            return Err("toroidal input must match hidden dimension and be finite".into());
        }
        if self.layer_fields.is_empty()
            || !self.toroidal_field_mix.is_finite()
            || !(0.0..=1.0).contains(&self.toroidal_field_mix)
        {
            return Err(
                "toroidal stack requires fields and a finite residual mix in [0, 1]".into(),
            );
        }
        let u = f64::from(-self.toroidal_field_mix).ln_1p()
            / (self.layer_fields.len() as f64);
        let layer_mix = (-u.exp_m1()).clamp(0.0001, 1.0) as f32;
        let mut x = x_in.to_vec();
        for (layer, field) in self.layer_fields.iter_mut().enumerate() {
            let attended = field.step_and_readout(&x, self.horizon_decay);
            for (value, context) in x.iter_mut().zip(attended) {
                *value += layer_mix * (context - *value);
            }
            if x.len() >= 6 {
                let x1 = f64::from(x[1]);
                let x2 = f64::from(x[2]);
                let x3 = f64::from(x[3]);
                x[0] = (1.0 + x1 * x1 + x2 * x2 + x3 * x3).sqrt() as f32;
                x[4] = f64::from(x[4]).rem_euclid(std::f64::consts::TAU) as f32;
                x[5] = f64::from(x[5]).rem_euclid(std::f64::consts::TAU) as f32;
            }
            if !x.iter().all(|value| value.is_finite()) {
                return Err(format!("toroidal layer {layer} produced non-finite output"));
            }
        }
        Ok(x)
    }

    /// Process native UTF-8→ML-IR address embeddings in order through every toroidal layer.
    /// The caller owns this runtime per session so recurrent state remains isolated.
    pub fn step_native_text(
        &mut self,
        embeddings: &[crate::input_embeddings::NativeMlirEmbedding],
    ) -> Result<Vec<f32>, String> {
        self.step_native_text_sequence(embeddings)?
            .pop()
            .ok_or_else(|| "native text contains no carrier embeddings".into())
    }

    /// Return one toroidal hidden state per byte position for the native output head.
    pub fn step_native_text_sequence(
        &mut self,
        embeddings: &[crate::input_embeddings::NativeMlirEmbedding],
    ) -> Result<Vec<Vec<f32>>, String> {
        if embeddings.is_empty() {
            return Err("native text contains no carrier embeddings".into());
        }
        if self.hidden_dim != crate::input_embeddings::NATIVE_MLIR_EMBEDDING_DIM {
            return Err(format!(
                "native ML-IR embeddings are {}D but toroidal stack is {}D",
                crate::input_embeddings::NATIVE_MLIR_EMBEDDING_DIM,
                self.hidden_dim
            ));
        }
        // Validate the complete input before changing any recurrent state.
        if embeddings
            .iter()
            .any(|entry| entry.coordinates.iter().any(|value| !value.is_finite()))
        {
            return Err("native text contains non-finite carrier coordinates".into());
        }
        let mut outputs = Vec::with_capacity(embeddings.len());
        for token in embeddings {
            outputs.push(self.step_toroidal_attention(&token.coordinates)?);
        }
        Ok(outputs)
    }

    /// Apply pair-wise RMS scaling on `(R_+ x S^1)^(D/2)` without `max_r = 3.0 * mean_pair_r`
    /// saturation clamps, preserving all pair relative amplitudes and `target_l2_norm`.
    pub fn isotropic_toroidal_rms_normalize(x: &mut [f32], target_l2_norm: f32) {
        if x.is_empty() {
            return;
        }
        let norm_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let cur_norm = norm_sq.sqrt().max(1e-9) as f32;
        let scale = target_l2_norm / cur_norm;
        for v in x.iter_mut() {
            *v *= scale;
        }
    }

    /// Execute one token step through all `active_layers` using `O(1)` Recurrent Linear Attention
    /// (`ToroidalHarmonicFieldState`, Katharopoulos et al. 2020) + `ATOMA_OP_V3` Rank-8 Local
    /// Jacobian charts from `swarm_map`.
    pub fn step_token(
        &mut self,
        position: usize,
        x_in: &[f32],
        swarm_map: &atoma_immune::CarvedSwarmMap,
    ) -> Result<ToroidalIsotropicStepReceipt, String> {
        let mut x_curr = x_in.to_vec();
        let base_norm = x_curr
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt()
            .max(1e-6) as f32;

        let mut layer_receipts = Vec::with_capacity(self.active_layers.len());
        let mut rank_sum = 0usize;
        let mut cos_sum = 0.0f64;

        for (idx, &layer) in self.active_layers.iter().enumerate() {
            let toroidal_ctx = self.layer_fields[idx].step_and_readout(&x_curr, self.horizon_decay);
            let map_rec = self.map_engine.retrieve_and_synthesize_layer(
                layer,
                position,
                self.total_model_layers,
                &x_curr,
                atoma_immune::MapSubstitutionMode::FullLayer,
                swarm_map,
            )?;

            let active_rank = swarm_map
                .global_atom(map_rec.top_atom_id)
                .and_then(|ga| atoma_immune::CoupledOperatorAtomPayload::peek_rank(&ga.payload))
                .unwrap_or(1);

            let mix = self.toroidal_field_mix;
            let inv_mix = 1.0 - mix;
            for d in 0..x_curr.len() {
                let map_step = map_rec.attn_delta[d] + map_rec.mlp_delta[d];
                let torus_delta = toroidal_ctx[d] - x_curr[d];
                x_curr[d] += inv_mix * map_step + mix * torus_delta;
            }
            let target_norm = map_rec
                .synthesized_x_out
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                .sqrt()
                .max(base_norm as f64) as f32;
            Self::isotropic_toroidal_rms_normalize(&mut x_curr, target_norm);

            let iso_rms = self.layer_fields[idx].radius_accumulator;

            rank_sum += active_rank;
            cos_sum += map_rec.max_receptor_cosine;
            layer_receipts.push(ToroidalIsotropicLayerReceipt {
                layer,
                routed_role: map_rec.routed_role,
                top_atom_id: map_rec.top_atom_id,
                active_jacobian_rank: active_rank,
                max_receptor_cosine: map_rec.max_receptor_cosine,
                min_cosh_distance: map_rec.min_cosh_distance,
                isotropic_rms_radius: iso_rms,
            });
        }

        let last_layer = *self.active_layers.last().unwrap_or(&0);
        let votes = self
            .map_engine
            .retrieve_next_id_votes(
                last_layer,
                position,
                self.total_model_layers,
                &x_curr,
                swarm_map,
            )
            .unwrap_or_default();
        let mut tally = std::collections::BTreeMap::<u32, f64>::new();
        for (_aid, w, next_id) in votes {
            *tally.entry(next_id).or_insert(0.0) += w;
        }
        let (predicted_next_mlir_id, winning_vote_weight) = tally
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, 0.0));

        let n_layers = layer_receipts.len().max(1) as f64;
        Ok(ToroidalIsotropicStepReceipt {
            position,
            predicted_next_mlir_id,
            winning_vote_weight,
            mean_active_jacobian_rank: (rank_sum as f64) / n_layers,
            mean_receptor_cosine: cos_sum / n_layers,
            layer_receipts,
            final_hidden_state: x_curr,
        })
    }
}

/// Configuration for **Breathing Depth** (`docs/BREATHING_DEPTH_DESIGN.md`):
/// - Construct, don't truncate: builds layers from a minimal 1-layer scaffold (`min_depth = 1`).
/// - PAMI multifractal spectral spread (`crate::pami::pami_index`) forecasts the depth cap.
/// - Marginal free-energy rule (`dF/d(layer) < -cost_per_layer`) acts as the construction brake.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BreathingDepthConfig {
    /// Minimal scaffold depth (1 representational step; 0 is the limit concept, not a state).
    pub min_depth: usize,
    /// Maximum layer capacity ceiling (52 in the reference stack).
    pub max_depth: usize,
    /// Marginal free-energy reduction (`-dF/dlayer`) required for a layer to pay for itself.
    pub cost_per_layer: f64,
    /// Marginal free-energy reduction (`-dF/drank`) required to recruit an additional sub-rank `1..=8`.
    pub cost_per_rank: f64,
    /// Temperature `T` for the free-energy functional `F = U + T*A`.
    pub regulator_temperature: f64,
    /// Whether PAMI spectral spread forecasts the construction ceiling. When `false`
    /// (e.g. NULL-on-index fallback under §6.3), construction is governed purely by the
    /// marginal free-energy rule up to `max_depth`.
    pub use_pami_forecast: bool,
}

impl Default for BreathingDepthConfig {
    fn default() -> Self {
        Self {
            min_depth: 1,
            max_depth: 52,
            cost_per_layer: 0.0032,
            cost_per_rank: 0.0045,
            regulator_temperature: 0.5,
            use_pami_forecast: true,
        }
    }
}

/// Per-layer construction telemetry recorded as the stack breathes upward from Layer 1.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BreathingLayerStep {
    /// 1-based layer depth (`1..=constructed_depth`).
    pub depth: usize,
    pub structural_coherence: f64,
    pub relational_coherence: f64,
    pub harmonic_coherence: f64,
    /// Free energy `F(depth)` at this constructed layer.
    pub free_energy_f: f64,
    /// Marginal change `dF/dlayer = F(depth) - F(depth - 1)` (`0.0` at Layer 1).
    pub delta_f: f64,
    /// Whether `delta_f < -cost_per_layer` (always `true` for the minimal Layer-1 scaffold).
    pub paid_for_itself: bool,
    /// Radial position in the Poincaré ball `r in [0, 1)` at this layer.
    pub poincare_radius: f64,
}

/// Recorded outputs of a breathing forward pass (`docs/BREATHING_DEPTH_DESIGN.md` §4):
/// Depth, rank, and attention sectors recruited are outputs of the computation, not hyperparameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BreathingConstructionReceipt {
    /// Number of toroidal layers constructed and committed (`1..=max_depth`).
    pub constructed_depth: usize,
    /// Active Jacobian/harmonic sub-rank constructed by the marginal rule (`1..=8`).
    pub constructed_rank: usize,
    /// Number of phase attention sectors lit by PAMI (`1..=8`).
    pub active_attention_sectors: usize,
    /// 8-bit mask of active phase sectors from `PamiIndex::phases`.
    pub sector_mask: u8,
    /// PAMI multifractal spectral spread `max(D(q)) - min(D(q))` across `q in -6..=6`.
    pub pami_spectral_spread: Option<f64>,
    /// Depth ceiling forecasted by PAMI before the marginal free-energy brake.
    pub pami_forecast_depth: usize,
    /// Free energy `F(1)` at the minimal 1-layer scaffold.
    pub initial_free_energy: f64,
    /// Free energy `F(L)` at the final constructed depth `L`.
    pub final_free_energy: f64,
    /// Total free-energy reduction earned by adding layers beyond Layer 1: `max(0, F(1) - F(L))`.
    pub marginal_free_energy_gain: f64,
    /// Coherence dissonance `||c(L) - c(1)||` across construction.
    pub construction_dissonance: f64,
    /// Effective radial coordinate in the Poincaré ball `r = tanh(rapidity / 2) in [0, 1)`.
    pub poincare_radius: f64,
    /// Step-by-step trace of constructed layers.
    pub layer_trace: Vec<BreathingLayerStep>,
    /// Per-position 6D toroidal hidden states produced by the constructed stack.
    pub hidden_sequence: Vec<Vec<f32>>,
    /// Final 6D toroidal hidden state at the end of the prompt.
    pub final_hidden_state: Vec<f32>,
}

fn wrap_phase_delta(a: f64, b: f64) -> f64 {
    let d = a - b;
    d.sin().atan2(d.cos())
}

/// Synthesize a 256-sample continuous phase-curvature wave packet from a carrier embedding
/// sequence so `crate::pami::pami_index` resolves all 8 within-scale phase slots (`k in 1..=8`)
/// and 13 multifractal dimensions `D(q)` (`q in -6..=6`), and measure the trajectory's
/// structural autocorrelation and multi-scale tension.
pub fn analyze_carrier_trajectory_complexity(
    embeddings: &[crate::input_embeddings::NativeMlirEmbedding],
) -> (Option<crate::pami::PamiIndex>, Option<f64>, u8, f64, f64) {
    if embeddings.is_empty() {
        return (None, None, 1, 0.0, 0.0);
    }
    let m = embeddings.len();
    let mut rapidities = Vec::with_capacity(m);
    let mut d_theta_f = Vec::with_capacity(m);
    let mut d_theta_s = Vec::with_capacity(m);
    let mut chords = Vec::with_capacity(m);

    for (i, emb) in embeddings.iter().enumerate() {
        let c = emb.coordinates;
        let x0 = f64::from(c[0]).max(1.0);
        rapidities.push(x0.acosh());
        if i > 0 {
            let prev = embeddings[i - 1].coordinates;
            let df = wrap_phase_delta(f64::from(c[4]), f64::from(prev[4]));
            let ds = wrap_phase_delta(f64::from(c[5]), f64::from(prev[5]));
            let ch = ((f64::from(c[1]) - f64::from(prev[1])).powi(2)
                + (f64::from(c[2]) - f64::from(prev[2])).powi(2)
                + (f64::from(c[3]) - f64::from(prev[3])).powi(2))
            .sqrt();
            d_theta_f.push(df);
            d_theta_s.push(ds);
            chords.push(ch);
        } else {
            d_theta_f.push(0.0);
            d_theta_s.push(0.0);
            chords.push(0.0);
        }
    }

    // Measure 2nd-order phase/chord intermittency and multi-step structural autocorrelation.
    // Structured multi-clause language has both multi-scale variation AND positive directional
    // coherence across steps, whereas uncorrelated noise has near-zero directional coherence.
    let mut phase_jerk_sum = 0.0f64;
    let mut chord_var_sum = 0.0f64;
    let mean_chord = chords.iter().sum::<f64>() / (m.max(1) as f64);
    let mut autocorr_num = 0.0f64;
    let mut autocorr_den = 0.0f64;
    let mut unique_turn_scales = 0usize;

    for i in 1..m {
        let jf = if i >= 2 {
            (d_theta_f[i] - d_theta_f[i - 1]).abs()
        } else {
            d_theta_f[i].abs()
        };
        let js = if i >= 2 {
            (d_theta_s[i] - d_theta_s[i - 1]).abs()
        } else {
            d_theta_s[i].abs()
        };
        phase_jerk_sum += jf + 0.7 * js;
        chord_var_sum += (chords[i] - mean_chord).powi(2);
        if jf > 0.35 || chords[i] > 0.45 {
            unique_turn_scales += 1;
        }
        if i >= 2 {
            // Cosine similarity between consecutive 3D spatial chords + phase increments
            let c_curr = embeddings[i].coordinates;
            let c_prev = embeddings[i - 1].coordinates;
            let c_prev2 = embeddings[i - 2].coordinates;
            let v1 = [
                f64::from(c_curr[1] - c_prev[1]),
                f64::from(c_curr[2] - c_prev[2]),
                f64::from(c_curr[3] - c_prev[3]),
                d_theta_f[i],
                d_theta_s[i],
            ];
            let v0 = [
                f64::from(c_prev[1] - c_prev2[1]),
                f64::from(c_prev[2] - c_prev2[2]),
                f64::from(c_prev[3] - c_prev2[3]),
                d_theta_f[i - 1],
                d_theta_s[i - 1],
            ];
            let dot: f64 = v1.iter().zip(&v0).map(|(a, b)| a * b).sum();
            let n1: f64 = v1.iter().map(|a| a * a).sum::<f64>().sqrt();
            let n0: f64 = v0.iter().map(|a| a * a).sum::<f64>().sqrt();
            if n1 > 1e-9 && n0 > 1e-9 {
                // Smooth or structured grammatical transitions maintain non-alternating energy
                let cos_step = (dot / (n1 * n0)).clamp(-1.0, 1.0);
                autocorr_num += (cos_step + 0.65).clamp(0.0, 1.0);
                autocorr_den += 1.0;
            }
        }
    }

    // Structural coherence in [0.05, 1.0]: penalizes pure sign-alternating/uncorrelated noise
    let structural_coherence = if autocorr_den > 0.0 {
        (autocorr_num / autocorr_den).clamp(0.08, 1.0)
    } else {
        0.55
    };

    // Length and multi-clause complexity factor
    let length_factor = ((m as f64) / 28.0).clamp(0.08, 1.8);
    let mean_jerk = phase_jerk_sum / (m.max(1) as f64);
    let chord_std = (chord_var_sum / (m.max(1) as f64)).sqrt();
    let clause_density = ((unique_turn_scales as f64) / 14.0).clamp(0.05, 1.5);
    let raw_tension = (0.28 * length_factor
        + 0.24 * (mean_jerk / 1.8).clamp(0.0, 1.5)
        + 0.24 * (chord_std / 0.8).clamp(0.0, 1.5)
        + 0.24 * clause_density)
        .clamp(0.06, 1.45);

    // Build 320-sample continuous wave packet across the carrier stream for PAMI (`k in -3..=6`)
    const WAVE_LEN: usize = 320;
    let mut wave = Vec::with_capacity(WAVE_LEN);
    for s_idx in 0..WAVE_LEN {
        let u = (s_idx as f64) / (WAVE_LEN as f64);
        let pos_f = u * ((m - 1).max(1) as f64);
        let idx0 = (pos_f.floor() as usize).min(m - 1);
        let idx1 = (idx0 + 1).min(m - 1);
        let frac = pos_f - (idx0 as f64);
        let c0 = embeddings[idx0].coordinates;
        let c1 = embeddings[idx1].coordinates;
        let tf = f64::from(c0[4]) + frac * wrap_phase_delta(f64::from(c1[4]), f64::from(c0[4]));
        let ts = f64::from(c0[5]) + frac * wrap_phase_delta(f64::from(c1[5]), f64::from(c0[5]));
        let rap = rapidities[idx0] * (1.0 - frac) + rapidities[idx1] * frac;
        let ch = chords[idx0] * (1.0 - frac) + chords[idx1] * frac;
        // Multi-scale carrier modulation packet: smooth prompts stay mono-scale; multi-clause
        // prompts with varied `ch` and `d_theta` excite higher wavelet leader moments.
        let local_burst = (d_theta_f[idx0].abs() + ch) * (u * (m as f64) * std::f64::consts::PI).sin();
        let sample = (1.0 + 0.5 * rap) * tf.cos()
            + (0.5 + ch) * (2.0 * ts).sin()
            + raw_tension * local_burst * (frac * std::f64::consts::TAU * (1.0 + (idx0 % 5) as f64)).cos();
        wave.push(sample);
    }

    let pami_opt = crate::pami::pami_index(&wave, crate::pami::SPEC_CONFIG);
    let (spread_opt, sector_mask) = if let Some(ref idx) = pami_opt {
        let min_d = idx
            .dims
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_d = idx
            .dims
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let raw_spread = if max_d.is_finite() && min_d.is_finite() {
            (max_d - min_d).max(0.0)
        } else {
            0.0
        };
        // Combine PAMI Legendre dimension spread with carrier phase-scale spread
        let active_slots = idx.phases.iter().filter(|&&p| p != 0.0).count();
        let combined_spread =
            (0.45 * raw_spread + 0.40 * raw_tension + 0.04 * (active_slots as f64)).clamp(0.05, 2.0);
        let mut mask = 0u8;
        for (k, &p) in idx.phases.iter().take(8).enumerate() {
            if p != 0.0 {
                mask |= 1u8 << k;
            }
        }
        if mask == 0 {
            mask = 0b0000_0001;
        }
        (Some(combined_spread), mask)
    } else {
        (Some(raw_tension), 0b0000_0011)
    };

    (
        pami_opt,
        spread_opt,
        sector_mask,
        raw_tension,
        structural_coherence,
    )
}

impl ToroidalIsotropicRuntime {
    /// Construct a minimal 1-layer breathing scaffold (`Construct, don't truncate`,
    /// `docs/BREATHING_DEPTH_DESIGN.md` §2 & §4). Additional layers are constructed per turn
    /// only while `dF/d(layer) < -cost_per_layer`.
    pub fn breathing_scaffold(hidden_dim: usize, max_model_layers: usize) -> Result<Self, String> {
        if hidden_dim < 2 || !hidden_dim.is_multiple_of(2) {
            return Err("toroidal hidden dimension must be positive and even".into());
        }
        if max_model_layers == 0 {
            return Err("max_model_layers must be at least 1".into());
        }
        Ok(Self::new(hidden_dim, vec![0], max_model_layers))
    }

    /// Current constructed depth (`1..=total_model_layers`).
    pub fn constructed_depth(&self) -> usize {
        self.layer_fields.len().max(1)
    }

    /// Construct layers upward from a 1-layer minimal scaffold for `embeddings` using:
    /// 1. **PAMI complex index** (`crate::pami::pami_index`) to forecast the depth ceiling and
    ///    select active phase attention sectors, and
    /// 2. **Marginal free-energy rule** (`dF/d(layer) < -cfg.cost_per_layer` via
    ///    `crate::free_energy::free_energy`) as the construction brake.
    ///
    /// Mutates `self.layer_fields` and `self.active_layers` to hold the exact constructed stack
    /// of depth `constructed_depth` (`1..=cfg.max_depth`) and returns [`BreathingConstructionReceipt`].
    pub fn step_breathing_text_sequence(
        &mut self,
        embeddings: &[crate::input_embeddings::NativeMlirEmbedding],
        cfg: &BreathingDepthConfig,
    ) -> Result<BreathingConstructionReceipt, String> {
        self.step_breathing_text_sequence_inner(embeddings, cfg, None)
    }

    /// Breathing construction whose DEPTH DECISION reads only the
    /// relational/numerical backbone (2026-10-20, Stewart): PAMI
    /// complexity is computed over `backbone` when supplied, so
    /// decorative narrative cannot inflate constructed depth. The
    /// full `embeddings` sequence still propagates through the
    /// constructed stack.
    pub fn step_breathing_text_sequence_backbone(
        &mut self,
        embeddings: &[crate::input_embeddings::NativeMlirEmbedding],
        cfg: &BreathingDepthConfig,
        backbone: Option<&[crate::input_embeddings::NativeMlirEmbedding]>,
    ) -> Result<BreathingConstructionReceipt, String> {
        self.step_breathing_text_sequence_inner(embeddings, cfg, backbone)
    }

    fn step_breathing_text_sequence_inner(
        &mut self,
        embeddings: &[crate::input_embeddings::NativeMlirEmbedding],
        cfg: &BreathingDepthConfig,
        backbone: Option<&[crate::input_embeddings::NativeMlirEmbedding]>,
    ) -> Result<BreathingConstructionReceipt, String> {
        if embeddings.is_empty() {
            return Err("native text contains no carrier embeddings".into());
        }
        if self.hidden_dim != crate::input_embeddings::NATIVE_MLIR_EMBEDDING_DIM {
            return Err(format!(
                "native ML-IR embeddings are {}D but toroidal stack is {}D",
                crate::input_embeddings::NATIVE_MLIR_EMBEDDING_DIM,
                self.hidden_dim
            ));
        }
        if embeddings
            .iter()
            .any(|entry| entry.coordinates.iter().any(|value| !value.is_finite()))
        {
            return Err("native text contains non-finite carrier coordinates".into());
        }

        let min_depth = cfg.min_depth.max(1);
        let max_depth = cfg.max_depth.max(min_depth);
        let (
            _pami_idx,
            pami_spectral_spread,
            sector_mask,
            raw_tension,
            structural_autocorr,
        ) = analyze_carrier_trajectory_complexity(
            backbone
                .filter(|b| b.len() >= 2)
                .unwrap_or(embeddings),
        );

        let active_attention_sectors = (sector_mask.count_ones() as usize).max(1);
        let pami_forecast_depth = if cfg.use_pami_forecast {
            let spread = pami_spectral_spread.unwrap_or(raw_tension);
            let forecast = (min_depth as f64 + (spread * 34.0).round()) as usize;
            forecast.clamp(min_depth, max_depth)
        } else {
            max_depth
        };

        // Helper: propagate a 6D sequence through one newly constructed `ToroidalHarmonicFieldState`
        // gated by `sector_mask`.
        let step_one_layer = |field: &mut ToroidalHarmonicFieldState,
                              seq_in: &[Vec<f32>],
                              depth_idx: usize,
                              mix_total: f32,
                              decay: f64,
                              mask: u8|
         -> Result<Vec<Vec<f32>>, String> {
            let active_frac = (mask.count_ones().max(1) as f32) / 8.0;
            // Residual blend per constructed layer decays gently with depth so new layers refine
            // without erasing the carrier identity.
            let layer_mix = (mix_total * active_frac / ((depth_idx as f32).sqrt().max(1.0)))
                .clamp(0.015, 0.25);
            let mut seq_out = Vec::with_capacity(seq_in.len());
            for x_in in seq_in {
                let attended = field.step_and_readout(x_in, decay);
                let mut x_next = x_in.clone();
                for (d, (val, ctx)) in x_next.iter_mut().zip(attended).enumerate() {
                    // Gate pair channels by active PAMI sectors
                    let sector_bit = 1u8 << (d % 8);
                    let sector_gain = if (mask & sector_bit) != 0 { 1.0 } else { 0.35 };
                    *val += layer_mix * sector_gain * (ctx - *val);
                }
                // Maintain exact hyperboloid sheet x0 = sqrt(1 + x1^2 + x2^2 + x3^2) and wrapped torus angles
                let x1 = f64::from(x_next[1]);
                let x2 = f64::from(x_next[2]);
                let x3 = f64::from(x_next[3]);
                x_next[0] = (1.0 + x1 * x1 + x2 * x2 + x3 * x3).sqrt() as f32;
                x_next[4] = f64::from(x_next[4]).rem_euclid(std::f64::consts::TAU) as f32;
                x_next[5] = f64::from(x_next[5]).rem_euclid(std::f64::consts::TAU) as f32;
                if !x_next.iter().all(|v| v.is_finite()) {
                    return Err(format!(
                        "breathing toroidal layer {depth_idx} produced non-finite state"
                    ));
                }
                seq_out.push(x_next);
            }
            Ok(seq_out)
        };

        // Helper: measure the 3 coherence coordinates `(structural, relational, harmonic)` at
        // constructed depth `d` from the field state and current sequence.
        // - `effective_tension` is the structured complexity (`raw_tension * structural_autocorr`).
        //   When input is uncorrelated noise (`structural_autocorr` small), deeper layers cannot
        //   extract real multi-layer coherence reduction, so the marginal brake stops early (§5).
        let effective_tension = (raw_tension * structural_autocorr).clamp(0.04, 1.35);
        let decay_rate = (0.11 / effective_tension.powf(0.65)).clamp(0.065, 0.48);

        let measure_layer_coherence =
            |depth: usize,
             field: &ToroidalHarmonicFieldState,
             seq: &[Vec<f32>]|
             -> (crate::free_energy::Coherence, f64) {
                let pairs = (self.hidden_dim / 2).max(1);
                let z0 = field.z_0.max(1.0);
                let mut r1_sum = 0.0f64;
                let mut r2_sum = 0.0f64;
                for p in 0..pairs {
                    r1_sum += (field.z_c1[p].hypot(field.z_s1[p]) / z0).clamp(0.0, 1.0);
                    r2_sum += (field.z_c2[p].hypot(field.z_s2[p]) / z0).clamp(0.0, 1.0);
                }
                let r1 = (r1_sum / (pairs as f64)).clamp(0.0, 1.0);
                let r2 = (r2_sum / (pairs as f64)).clamp(0.0, 1.0);

                // Remaining unresolved multi-layer tension at depth `depth`:
                // At depth=1, `unresolved = effective_tension * exp(-decay_rate * 0.5)`.
                // Each added layer resolves `exp(-decay_rate)` of the remaining structured tension,
                // modulated by the field's phase resultant `0.5 + 0.5 * r1`.
                let layer_progress = ((depth - 1) as f64) * decay_rate * (0.65 + 0.35 * r1);
                let unresolved = effective_tension * (-layer_progress).exp();

                let c_structural = (0.985 - 0.52 * unresolved).clamp(0.15, 0.99);
                let c_relational =
                    (0.975 - 0.58 * unresolved + 0.015 * r1).clamp(0.12, 0.99);
                let c_harmonic =
                    (0.965 - 0.62 * unresolved + 0.015 * r2).clamp(0.10, 0.99);

                // Compute Poincaré ball radius r = tanh(eta_eff / 2) in (0, 1)
                let last_coord = seq.last().map(Vec::as_slice).unwrap_or(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                let x0 = f64::from(last_coord[0]).max(1.0);
                let base_rapidity = x0.acosh();
                let stack_rapidity =
                    base_rapidity + 0.25 * ((depth as f64).ln_1p()) * (1.0 + 0.5 * effective_tension);
                let poincare_r = (stack_rapidity * 0.5).tanh().clamp(0.0, 0.999_999);

                (
                    crate::free_energy::Coherence::new(c_structural, c_relational, c_harmonic),
                    poincare_r,
                )
            };

        let weights = crate::free_energy::phi_weights();
        let initial_seq: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| e.coordinates.to_vec())
            .collect();

        // 1. Construct minimal scaffold (Layer 1..=min_depth)
        self.active_layers.clear();
        self.layer_fields.clear();
        let mut current_seq = initial_seq;
        let mut layer_trace = Vec::with_capacity(pami_forecast_depth);
        let mut initial_coherence = crate::free_energy::Coherence::new(0.5, 0.5, 0.5);
        let mut prev_coherence = initial_coherence;
        let mut initial_f = 0.0f64;
        let mut prev_f = 0.0f64;
        let mut current_poincare_r = 0.0f64;

        for depth in 1..=pami_forecast_depth {
            let mut field = ToroidalHarmonicFieldState::new(self.hidden_dim);
            let next_seq = step_one_layer(
                &mut field,
                &current_seq,
                depth,
                self.toroidal_field_mix,
                self.horizon_decay,
                sector_mask,
            )?;
            let (coh, poincare_r) = measure_layer_coherence(depth, &field, &next_seq);
            let fe = crate::free_energy::free_energy(coh, cfg.regulator_temperature, weights);
            let delta_f = if depth == 1 { 0.0 } else { fe.f - prev_f };
            let paid = depth <= min_depth || (delta_f < -cfg.cost_per_layer);

            if paid {
                self.active_layers.push(depth - 1);
                self.layer_fields.push(field);
                current_seq = next_seq;
                if depth == 1 {
                    initial_coherence = coh;
                    initial_f = fe.f;
                }
                prev_coherence = coh;
                prev_f = fe.f;
                current_poincare_r = poincare_r;
                layer_trace.push(BreathingLayerStep {
                    depth,
                    structural_coherence: coh.structural,
                    relational_coherence: coh.relational,
                    harmonic_coherence: coh.harmonic,
                    free_energy_f: fe.f,
                    delta_f,
                    paid_for_itself: true,
                    poincare_radius: poincare_r,
                });
            } else {
                // Marginal free-energy brake fired: `dF/dlayer >= -cost_per_layer`.
                // Discard the unpaying candidate layer and stop construction here.
                break;
            }
        }

        let constructed_depth = self.layer_fields.len().max(1);
        self.total_model_layers = constructed_depth;

        // 2. Construct active Jacobian/harmonic sub-rank (1..=8) by the same marginal rule:
        //    `dF/drank < -cfg.cost_per_rank`.
        let mut constructed_rank = 1usize;
        let mut prev_rank_f = crate::free_energy::free_energy(
            crate::free_energy::Coherence::new(
                (0.96 - 0.38 * effective_tension).clamp(0.15, 0.99),
                (0.94 - 0.42 * effective_tension).clamp(0.15, 0.99),
                (0.92 - 0.45 * effective_tension).clamp(0.15, 0.99),
            ),
            cfg.regulator_temperature,
            weights,
        )
        .f;
        for cand_rank in 2..=8 {
            let rank_unresolved =
                effective_tension * (-0.34 * ((cand_rank - 1) as f64) / effective_tension.sqrt().max(0.45)).exp();
            let rank_coh = crate::free_energy::Coherence::new(
                (0.96 - 0.38 * rank_unresolved).clamp(0.15, 0.99),
                (0.94 - 0.42 * rank_unresolved).clamp(0.15, 0.99),
                (0.92 - 0.45 * rank_unresolved).clamp(0.15, 0.99),
            );
            let rank_f =
                crate::free_energy::free_energy(rank_coh, cfg.regulator_temperature, weights).f;
            if rank_f - prev_rank_f < -cfg.cost_per_rank {
                constructed_rank = cand_rank;
                prev_rank_f = rank_f;
            } else {
                break;
            }
        }

        let dissonance = {
            let a = prev_coherence.as_array();
            let b = initial_coherence.as_array();
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        let final_hidden_state = current_seq
            .last()
            .cloned()
            .unwrap_or_else(|| vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let marginal_gain = (initial_f - prev_f).max(0.0);

        Ok(BreathingConstructionReceipt {
            constructed_depth,
            constructed_rank,
            active_attention_sectors,
            sector_mask,
            pami_spectral_spread,
            pami_forecast_depth,
            initial_free_energy: initial_f,
            final_free_energy: prev_f,
            marginal_free_energy_gain: marginal_gain,
            construction_dissonance: dissonance,
            poincare_radius: current_poincare_r,
            layer_trace,
            hidden_sequence: current_seq,
            final_hidden_state,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_stack_allocates_one_state_per_declared_layer() {
        let mut runtime = ToroidalIsotropicRuntime::full_stack(256, 52).unwrap();
        assert_eq!(runtime.total_model_layers, 52);
        assert_eq!(runtime.active_layers, (0..52).collect::<Vec<_>>());
        assert_eq!(runtime.layer_fields.len(), 52);
        assert!(ToroidalIsotropicRuntime::full_stack(255, 52).is_err());
        assert!(ToroidalIsotropicRuntime::full_stack(256, 0).is_err());

        let hidden = vec![0.25; 256];
        let output = runtime.step_toroidal_attention(&hidden).unwrap();
        assert_eq!(output.len(), hidden.len());
        assert!(output.iter().all(|x| x.is_finite()));
        assert!(
            runtime
                .layer_fields
                .iter()
                .all(|field| field.tokens_ingested == 1)
        );

        let transducer =
            atoma_immune::BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        let embeddings =
            crate::input_embeddings::embed_utf8_text(&transducer, "go", false).unwrap();
        let mut native_runtime = ToroidalIsotropicRuntime::full_stack(6, 52).unwrap();
        let native_output = native_runtime.step_native_text(&embeddings).unwrap();
        assert_eq!(native_output.len(), 6);
        assert!(
            native_runtime
                .layer_fields
                .iter()
                .all(|field| field.tokens_ingested == 2)
        );
    }

    #[test]
    fn p2_aligned_opposed_and_degenerate() {
        assert!(
            (pair_coherence(&[1.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0, 1.0]).unwrap() - 2.0).abs()
                < 1e-12
        );
        assert!(pair_coherence(&[1.0, 0.0], &[-1.0, 0.0]).unwrap() < 1e-12);
        assert!((pair_coherence(&[0.0, 0.0], &[1.0, 0.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!(
            (pair_coherence(&[1.0, 0.0, 1.0, 0.0], &[1.0, 0.0, -1.0, 0.0]).unwrap() - 1.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn convexity_and_zero_score_fallback() {
        for scores in [&[2.0, 1.0, 0.0][..], &[0.0, 0.0, 0.0][..]] {
            let weights = normalized_power_weights(scores, 20.0);
            assert!(weights.iter().all(|w| *w >= 0.0 && *w <= 1.0));
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let g1 = normalized_power_weights(&[1.8, 1.2], 1.0);
        let g4 = normalized_power_weights(&[1.8, 1.2], 4.0);
        assert!(
            g4[0] > g1[0],
            "higher gain must sharpen the stronger candidate"
        );
    }

    #[test]
    fn typed_envelope_rejects_malformed_and_nonfinite() {
        let identity = MapKvIdentity {
            layer: 3,
            kv_head: 1,
            sequence_id: 17,
            position: 7,
        };
        assert!(encode_map_kv_payload(identity, &[f32::NAN, 0.0], &[1.0, 2.0]).is_err());
        let mut bytes = encode_map_kv_payload(identity, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(decode_map_kv_payload(&bytes).unwrap().0, identity);
        bytes.pop();
        assert!(decode_map_kv_payload(&bytes).is_none());
    }

    #[test]
    fn orchestrator_radius_accumulator_expands_over_time_with_amplitude_and_picard_reduction() {
        let mut state = ToroidalHarmonicFieldState::new(128);
        let mut prev_radius = state.radius_accumulator;
        let mut prev_completed_rotations = 0.0f64;
        let mut prev_amp = state.phase_state.amplitude.to_f64();
        let mut prev_rapidity = state.rapidity_accumulator;

        for step in 0..16 {
            let x_in: Vec<f32> = (0..128)
                .map(|i| (((i + step * 7) as f32) * 0.19).cos() * 1.2)
                .collect();
            let out = state.step_and_readout(&x_in, 0.995);
            assert_eq!(out.len(), 128);
            assert!(out.iter().all(|v| v.is_finite()));
            let completed_rotations = state.cumulative_rotations.floor();
            assert!(state.radius_accumulator >= prev_radius);
            if completed_rotations > prev_completed_rotations {
                assert!(state.radius_accumulator > prev_radius);
            } else {
                assert_eq!(state.radius_accumulator, prev_radius);
            }
            let expected_radius =
                TorPhaseEncoder::radius_for_rotations(state.cumulative_rotations.floor(), 1.0, 1.9);
            assert!((state.radius_accumulator - expected_radius).abs() < 1e-12);
            assert!(
                state.phase_state.amplitude.to_f64() > prev_amp,
                "PhaseState amplitude must expand over time: step {step} {} <= {prev_amp}",
                state.phase_state.amplitude.to_f64()
            );
            assert!(
                state.rapidity_accumulator > prev_rapidity,
                "rapidity_accumulator must expand over time: step {step} {} <= {prev_rapidity}",
                state.rapidity_accumulator
            );
            assert!(
                state.reduced_lorentz.validate().is_ok(),
                "Picard-reduced hyperbolic state must remain on H^3 at step {step}"
            );
            prev_radius = state.radius_accumulator;
            prev_completed_rotations = completed_rotations;
            prev_amp = state.phase_state.amplitude.to_f64();
            prev_rapidity = state.rapidity_accumulator;
        }
        assert!(
            state.radius_accumulator > 1.0,
            "completed turns must expand radius"
        );
    }
    #[test]
    fn standalone_depth_preserves_current_input_instead_of_locking_to_first_carrier() {
        let transducer =
            atoma_immune::BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        let embedded = crate::input_embeddings::embed_utf8_text(&transducer, "AB", false).unwrap();
        for depth in [1, 52] {
            let mut runtime = ToroidalIsotropicRuntime::full_stack(6, depth).unwrap();
            let output = runtime.step_native_text_sequence(&embedded).unwrap();
            let distance = |target: &[f32]| {
                output[1]
                    .iter()
                    .zip(target)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum::<f64>()
            };
            assert!(
                distance(&embedded[1].coordinates) < distance(&embedded[0].coordinates),
                "depth {depth} erased the new carrier"
            );
        }
    }

    #[test]
    fn invalid_later_carrier_is_rejected_before_recurrent_state_changes() {
        let transducer =
            atoma_immune::BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        let mut embedded =
            crate::input_embeddings::embed_utf8_text(&transducer, "AB", false).unwrap();
        embedded[1].coordinates[0] = f32::NAN;
        let mut runtime = ToroidalIsotropicRuntime::full_stack(6, 52).unwrap();
        assert!(runtime.step_native_text_sequence(&embedded).is_err());
        assert!(
            runtime
                .layer_fields
                .iter()
                .all(|field| field.tokens_ingested == 0)
        );
        let valid = &embedded[..1];
        let mut fresh = ToroidalIsotropicRuntime::full_stack(6, 52).unwrap();
        assert_eq!(
            runtime.step_native_text_sequence(valid).unwrap(),
            fresh.step_native_text_sequence(valid).unwrap()
        );
    }

    #[test]
    fn breathing_construction_builds_layers_upward_by_marginal_free_energy_and_pami_index() {
        let transducer =
            atoma_immune::BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        let easy_embs =
            crate::input_embeddings::embed_utf8_text(&transducer, "2 + 2 =", false).unwrap();
        let hard_embs = crate::input_embeddings::embed_utf8_text(
            &transducer,
            "On the product manifold H^3 x T^2, when a hyperbolic geodesic boosts across curvature shells while toroidal phases wind at golden-ratio frequencies, the marginal variational free energy",
            false,
        )
        .unwrap();

        let cfg = BreathingDepthConfig::default();
        let mut easy_rt = ToroidalIsotropicRuntime::breathing_scaffold(6, 52).unwrap();
        assert_eq!(easy_rt.constructed_depth(), 1);
        let easy_receipt = easy_rt
            .step_breathing_text_sequence(&easy_embs, &cfg)
            .unwrap();
        assert_eq!(easy_receipt.hidden_sequence.len(), easy_embs.len());
        assert_eq!(easy_rt.constructed_depth(), easy_receipt.constructed_depth);
        assert!(
            (1..52).contains(&easy_receipt.constructed_depth),
            "easy prompt should halt before 52 layers: got {}",
            easy_receipt.constructed_depth
        );
        assert!((1..=8).contains(&easy_receipt.constructed_rank));
        assert!((1..=8).contains(&easy_receipt.active_attention_sectors));

        let mut hard_rt = ToroidalIsotropicRuntime::breathing_scaffold(6, 52).unwrap();
        let hard_receipt = hard_rt
            .step_breathing_text_sequence(&hard_embs, &cfg)
            .unwrap();
        assert_eq!(hard_receipt.hidden_sequence.len(), hard_embs.len());
        assert_eq!(hard_rt.constructed_depth(), hard_receipt.constructed_depth);
        assert!(
            hard_receipt.constructed_depth > easy_receipt.constructed_depth,
            "hard prompt depth ({}) must exceed easy prompt depth ({})",
            hard_receipt.constructed_depth,
            easy_receipt.constructed_depth
        );
        assert!(
            hard_receipt.pami_spectral_spread > easy_receipt.pami_spectral_spread,
            "hard prompt PAMI spread ({:?}) must exceed easy prompt PAMI spread ({:?})",
            hard_receipt.pami_spectral_spread,
            easy_receipt.pami_spectral_spread
        );
        assert!(
            hard_receipt.marginal_free_energy_gain > easy_receipt.marginal_free_energy_gain,
            "hard prompt marginal free-energy gain ({:.4}) must exceed easy prompt ({:.4})",
            hard_receipt.marginal_free_energy_gain,
            easy_receipt.marginal_free_energy_gain
        );
    }
}
