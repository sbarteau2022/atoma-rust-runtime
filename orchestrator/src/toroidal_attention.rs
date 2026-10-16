//! Tier 4–6 map attention experiment (CARVED_SWARM_SPEC.md §1, MASTER_ARCHITECT.md B1).
//!
//! The central `CarvedSwarmMap` is the only candidate authority. Its carved silo controls
//! visibility, its T² coordinates supply the map-phase arm, and typed KV payloads carry
//! explicit layer, head, sequence, and token positions so the causal mask can be applied before normalization.

use atoma_immune::cache::geometric::{AtomId, Phase};
use atoma_immune::geometry::bridge::f32_to_atom_address;
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
    Ok((1.0 + real / real.hypot(imag)).clamp(0.0, 2.0))
}

/// Amplitude-weighted multi-pair toroidal phase coherence on `(R_+ × S^1)^(D/2)`:
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
            || (self.input_dim != self.head_dim * 2 + 1
                && self.input_dim != self.head_dim * 3 + 1)
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
        (1.0 + real / real.hypot(imag)).clamp(0.0, 2.0)
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
    #[allow(clippy::too_many_arguments)] // Captured sample provenance and the fit grid are explicit.
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
    #[allow(clippy::too_many_arguments)] // Layer, token, and phase provenance must be explicit at the map boundary.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
