//! Real Glimmer attention boundary backed by the universal server's carved map.
//! The index stores atom IDs only; all values are read from the current central silo.

use crate::toroidal_attention::{
    LinearMapValueAdapter, MapKvIdentity, decode_map_kv_payload, map_phase_coherence,
    normalized_power_weights, pair_coherence, radial_pair_toroidal_coherence, softmax_weights,
    weighted_values,
};
use crate::{MonsterHybridServer, swarm::ProcessState};
use atoma_immune::geometry::bridge::f32_to_atom_address;
use atoma_immune::{RoleId, TeacherTapRecord};
use core_lm::attention_backend::{AttentionBackend, AttentionInput};
use serde::Serialize;
use std::collections::BTreeSet;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RecallMode {
    Baseline,
    PairAngles,
    RadialToroidal,
    MapT2,
    Direct,
    AdaptedMapValue,
}

#[derive(Debug, Clone)]
pub struct RecallConfig {
    pub mode: RecallMode,
    pub layers: BTreeSet<usize>,
    pub gain: f64,
    /// Bound on payloads considered by MapT2/Direct. Baseline/P2/RadialToroidal/AdaptedMapValue use causal keys.
    pub candidate_limit: usize,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct RecallMetrics {
    pub layer_calls: u64,
    pub written_atoms: u64,
    pub heads: u64,
    pub fetched_values: u64,
    pub dot_products: u64,
    pub phase_scores: u64,
    pub direct_reads: u64,
    pub write_ns: u128,
    pub recall_ns: u128,
}

#[derive(Debug, Clone)]
pub struct AttentionSnapshot {
    pub layer: usize,
    pub position: usize,
    pub window_start: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub dot_scale: f64,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub key_phases: Vec<atoma_immune::cache::geometric::Phase>,
    pub baseline_output: Vec<f32>,
}

fn phase_distance(
    a: atoma_immune::cache::geometric::Phase,
    b: atoma_immune::cache::geometric::Phase,
) -> f64 {
    2.0 - (a.theta_f.to_f64() - b.theta_f.to_f64()).cos()
        - (a.theta_s.to_f64() - b.theta_s.to_f64()).cos()
}

pub struct CentralMapAttention<'a> {
    server: &'a mut MonsterHybridServer,
    role: RoleId,
    sequence_id: u64,
    config: RecallConfig,
    adapter: Option<LinearMapValueAdapter>,
    pub metrics: RecallMetrics,
    trace_layer: Option<usize>,
    pub attention_trace: Vec<AttentionSnapshot>,
}

impl<'a> CentralMapAttention<'a> {
    pub fn new(
        server: &'a mut MonsterHybridServer,
        role: RoleId,
        sequence_id: u64,
        config: RecallConfig,
    ) -> Result<Self, String> {
        if role == RoleId::Coordinator
            || !config.gain.is_finite()
            || config.gain <= 0.0
            || config.gain > 128.0
            || !(1..=512).contains(&config.candidate_limit)
            || config
                .layers
                .iter()
                .any(|l| *l >= server.swarm_map.base_weights.num_layers)
        {
            return Err("invalid recall role, layer, gain, or candidate limit".into());
        }
        if server.swarm_map.silo(role).is_none() {
            return Err("role has no silo".into());
        }
        Ok(Self {
            server,
            role,
            sequence_id,
            config,
            adapter: None,
            metrics: RecallMetrics::default(),
            trace_layer: None,
            attention_trace: Vec::new(),
        })
    }

    /// Attach a validated `LinearMapValueAdapter` for `RecallMode::AdaptedMapValue`.
    pub fn set_adapter(&mut self, adapter: LinearMapValueAdapter) -> Result<(), String> {
        adapter.validate()?;
        self.adapter = Some(adapter);
        Ok(())
    }

    /// Record the pre-output-projection attention vector for one layer per token.
    pub fn capture_layer(&mut self, layer: usize) -> Result<(), String> {
        if layer >= self.server.swarm_map.base_weights.num_layers {
            return Err("capture layer outside loaded model".into());
        }
        self.trace_layer = Some(layer);
        self.attention_trace.clear();
        Ok(())
    }

    pub fn indexed_slots(&self) -> usize {
        self.server
            .swarm_map
            .silo(self.role)
            .map(|s| s.typed_kv_sequence_count(self.sequence_id))
            .unwrap_or_default()
    }

    fn read_head(
        &mut self,
        input: &AttentionInput<'_>,
        qh: usize,
        mode: RecallMode,
    ) -> Result<Vec<f32>, String> {
        let dim = input.head_dim;
        let query = &input.query[qh * dim..(qh + 1) * dim];
        let kvh = qh / (input.query_heads / input.kv_heads);
        let (_, query_phase, _) = f32_to_atom_address(query).map_err(|e| e.to_string())?;
        let ids = {
            let silo = self
                .server
                .swarm_map
                .silo(self.role)
                .ok_or("missing silo")?;
            if matches!(mode, RecallMode::MapT2 | RecallMode::Direct) {
                silo.typed_kv_near_phase(
                    input.layer,
                    kvh,
                    self.sequence_id,
                    input.window_start,
                    input.position,
                    query_phase,
                    self.config.candidate_limit,
                )?
            } else {
                silo.typed_kv_range(
                    input.layer,
                    kvh,
                    self.sequence_id,
                    input.window_start,
                    input.position,
                )?
            }
        };
        if ids.is_empty() {
            return Err("no causal map values".into());
        }
        let silo = self
            .server
            .swarm_map
            .silo(self.role)
            .ok_or("missing silo")?;
        let mut values = Vec::new();
        let mut scores = Vec::new();
        let mut direct: Option<(f64, u64, Vec<f32>)> = None;
        for id in ids {
            let atom = silo
                .get_atom(id)
                .ok_or("indexed atom no longer visible")?
                .clone();
            let (identity, raw) = decode_map_kv_payload(&atom.payload).ok_or("malformed KV")?;
            if identity.layer != input.layer
                || identity.kv_head != kvh
                || identity.sequence_id != self.sequence_id
                || !(input.window_start..=input.position).contains(&identity.position)
                || raw.len() != dim * 8
                || atom.origin_model_fingerprint != self.server.swarm_map.base_weights.fingerprint
            {
                return Err("KV identity, dimension, or model epoch mismatch".into());
            }
            let value: Vec<f32> = raw[dim * 4..]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if !value.iter().all(|v| v.is_finite()) {
                return Err("non-finite map value".into());
            }
            self.metrics.fetched_values += 1;
            let score = match mode {
                RecallMode::Baseline
                | RecallMode::PairAngles
                | RecallMode::RadialToroidal
                | RecallMode::AdaptedMapValue => {
                    let key: Vec<f32> = raw[..dim * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    if !key.iter().all(|v| v.is_finite()) {
                        return Err("non-finite key".into());
                    }
                    if mode == RecallMode::AdaptedMapValue {
                        let d = phase_distance(query_phase, atom.phase);
                        if direct.as_ref().is_none_or(|(old, old_id, _)| {
                            d < *old || (d == *old && id.0 < *old_id)
                        }) {
                            direct = Some((d, id.0, value.clone()));
                        }
                    }
                    if mode == RecallMode::Baseline {
                        self.metrics.dot_products += 1;
                        query
                            .iter()
                            .zip(&key)
                            .map(|(&q, &k)| q as f64 * k as f64)
                            .sum::<f64>()
                            * input.dot_scale
                    } else if mode == RecallMode::PairAngles {
                        self.metrics.phase_scores += 1;
                        pair_coherence(query, &key)?
                    } else {
                        self.metrics.phase_scores += 1;
                        radial_pair_toroidal_coherence(query, &key, input.dot_scale)?
                    }
                }
                RecallMode::MapT2 => {
                    self.metrics.phase_scores += 1;
                    map_phase_coherence(query_phase, atom.phase)
                }
                RecallMode::Direct => {
                    self.metrics.phase_scores += 1;
                    let d = phase_distance(query_phase, atom.phase);
                    if direct
                        .as_ref()
                        .is_none_or(|(old, old_id, _)| d < *old || (d == *old && id.0 < *old_id))
                    {
                        direct = Some((d, id.0, value));
                    }
                    continue;
                }
            };
            if !score.is_finite() {
                return Err("non-finite recall score".into());
            }
            scores.push(score);
            values.push(value);
        }
        self.metrics.heads += 1;
        if mode == RecallMode::Direct {
            self.metrics.direct_reads += 1;
            return direct
                .map(|(_, _, v)| v)
                .ok_or("direct recall missed".into());
        }
        if mode == RecallMode::AdaptedMapValue {
            self.metrics.direct_reads += 1;
            let direct_val = direct
                .map(|(_, _, v)| v)
                .ok_or("adapted recall missed direct candidate")?;
            let rt_weights = softmax_weights(&scores);
            let rt_val = weighted_values(&rt_weights, &values, dim);
            let adapter = self
                .adapter
                .as_ref()
                .ok_or("AdaptedMapValue requires a loaded LinearMapValueAdapter")?;
            return adapter.predict_head(query, &direct_val, Some(&rt_val));
        }
        let weights = if matches!(mode, RecallMode::Baseline | RecallMode::RadialToroidal) {
            softmax_weights(&scores)
        } else {
            normalized_power_weights(&scores, self.config.gain)
        };
        let output = weighted_values(&weights, &values, dim);
        if !output.iter().all(|v| v.is_finite()) {
            return Err("non-finite recall output".into());
        }
        Ok(output)
    }
}

impl AttentionBackend for CentralMapAttention<'_> {
    fn attend(&mut self, input: AttentionInput<'_>) -> Result<Vec<f32>, String> {
        input.validate()?;
        if input.layer >= self.server.swarm_map.base_weights.num_layers {
            return Err("layer outside loaded model".into());
        }
        if !matches!(
            self.server.governor.process(self.role).map(|p| p.state),
            Some(ProcessState::Running)
        ) {
            return Err("role is not running".into());
        }
        let (_, _, loss) =
            f32_to_atom_address(&input.key[..input.head_dim]).map_err(|e| e.to_string())?;
        self.server
            .immune
            .authorize_forward_pass(
                self.role,
                input.token_id,
                input.position,
                &self.server.swarm_map,
                &loss,
            )
            .map_err(|e| e.to_string())?;
        let started = Instant::now();
        let mut key_phases = Vec::with_capacity(input.kv_heads);
        for head in 0..input.kv_heads {
            let key = &input.key[head * input.head_dim..(head + 1) * input.head_dim];
            let value = &input.value[head * input.head_dim..(head + 1) * input.head_dim];
            let tap = TeacherTapRecord {
                shard_id: input.position as u64,
                layer_index: input.layer,
                hidden_state: key.to_vec(),
                next_op_probs: vec![1.0],
                payload: Vec::new(),
            };
            let identity = MapKvIdentity {
                layer: input.layer,
                kv_head: head,
                sequence_id: self.sequence_id,
                position: input.position,
            };
            let receipt = self
                .server
                .ingest_attention_kv(&tap, self.role, identity, key, value)?;
            if !self
                .server
                .swarm_map
                .silo(self.role)
                .ok_or("missing silo")?
                .typed_kv_range(
                    input.layer,
                    head,
                    self.sequence_id,
                    input.position,
                    input.position,
                )?
                .contains(&receipt.atom_id)
            {
                return Err("written KV is outside the central silo index".into());
            }
            key_phases.push(
                self.server
                    .swarm_map
                    .silo(self.role)
                    .and_then(|s| s.get_atom(receipt.atom_id))
                    .ok_or("written KV atom disappeared")?
                    .phase,
            );
            self.metrics.written_atoms += 1;
        }
        self.metrics.write_ns += started.elapsed().as_nanos();
        let started = Instant::now();
        let mode = if self.config.layers.contains(&input.layer) {
            self.config.mode
        } else {
            RecallMode::Baseline
        };
        let mut output = Vec::with_capacity(input.query.len());
        for qh in 0..input.query_heads {
            output.extend(self.read_head(&input, qh, mode)?);
        }
        if self.trace_layer == Some(input.layer) {
            self.attention_trace.push(AttentionSnapshot {
                layer: input.layer,
                position: input.position,
                window_start: input.window_start,
                query_heads: input.query_heads,
                kv_heads: input.kv_heads,
                head_dim: input.head_dim,
                dot_scale: input.dot_scale,
                query: input.query.to_vec(),
                key: input.key.to_vec(),
                value: input.value.to_vec(),
                key_phases,
                baseline_output: output.clone(),
            });
        }
        self.metrics.recall_ns += started.elapsed().as_nanos();
        self.metrics.layer_calls += 1;
        Ok(output)
    }
}
