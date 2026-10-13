//! Muse Glimmer forward pass with GeometricMemoryMap replacing KV cache.
//!
//! Architecture (from muse_glimmer_mlx.py reference):
//! - 52 layers, pure transformer (39 sliding-window-2048 + 13 full attention)
//! - Sandwich norms: x = x + post_norm(branch(pre_norm(x)))
//! - QK-norm (scaleless RMS) BEFORE RoPE
//! - Folded attention scale: qk_scale_factor / head_dim
//! - q_proj fused with output gate ([q_head; gate_head] per-head interleaved)
//! - NeoX RoPE (rope_is_neox_style=true), NoPE on full_attention layers
//! - SwiGLU MLP
//! - Logit soft cap: cap * tanh(logits * mult / cap)
//!
//! Geometric KV: NO k_cache/v_cache tensors exist. K/V data lives in
//! per-layer, per-KV-head GeometricMemoryMap atoms (5D: H³×T²).
//! Standard RoPE for positions. Retrieval by H³ geometry;
//! scoring by exact q·k dot products. GQA mapping preserved:
//! each Q head attends only to its assigned KV head's atoms.
//! T² phases are ZERO placeholders until lock.rs supplies (n,m,ψ*).

use crate::cache::geometric::{GeometricMemoryMap, MemoryError, Phase};
use crate::geometry::bridge::{BridgeError, f32_to_lorentz};
use crate::glimmer_model::{GlimmerConfig, GlimmerError, WeightManager};
use crate::ops::{OpsError, matvec, rmsnorm, silu, softmax};
use std::path::Path;

#[derive(Debug)]
pub enum ForwardError {
    Model(GlimmerError),
    Ops(OpsError),
    Memory(MemoryError),
    Bridge(BridgeError),
    Shape(String),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardError::Model(e) => write!(f, "Model: {}", e),
            ForwardError::Ops(e) => write!(f, "Ops: {}", e),
            ForwardError::Memory(e) => write!(f, "Memory: {}", e),
            ForwardError::Bridge(e) => write!(f, "Bridge: {}", e),
            ForwardError::Shape(s) => write!(f, "Shape: {}", s),
        }
    }
}
impl std::error::Error for ForwardError {}
impl From<GlimmerError> for ForwardError {
    fn from(e: GlimmerError) -> Self {
        ForwardError::Model(e)
    }
}
impl From<OpsError> for ForwardError {
    fn from(e: OpsError) -> Self {
        ForwardError::Ops(e)
    }
}
impl From<MemoryError> for ForwardError {
    fn from(e: MemoryError) -> Self {
        ForwardError::Memory(e)
    }
}
impl From<BridgeError> for ForwardError {
    fn from(e: BridgeError) -> Self {
        ForwardError::Bridge(e)
    }
}

fn scaleless_rmsnorm(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

fn apply_rope_neox(
    x: &mut [f32],
    head_dim: usize,
    pos: usize,
    theta: f32,
) -> Result<(), ForwardError> {
    if x.len() % head_dim != 0 {
        return Err(ForwardError::Shape(format!(
            "x len {} not divisible by head_dim {}",
            x.len(),
            head_dim
        )));
    }
    if head_dim % 2 != 0 {
        return Err(ForwardError::Shape(
            "head_dim must be even for NeoX RoPE".to_string(),
        ));
    }
    let num_heads = x.len() / head_dim;
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * (i as f32) / (head_dim as f32));
            let angle = (pos as f32) * freq;
            let (sin, cos) = angle.sin_cos();
            let x0 = x[base + i];
            let x1 = x[base + i + half];
            x[base + i] = x0 * cos - x1 * sin;
            x[base + i + half] = x0 * sin + x1 * cos;
        }
    }
    Ok(())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct GlimmerModel {
    pub config: GlimmerConfig,
    pub weights: WeightManager,
    pub geo_maps: Vec<Vec<GeometricMemoryMap>>,
    pub q_fused: bool,
}

impl GlimmerModel {
    pub fn load(model_dir: &Path) -> Result<Self, ForwardError> {
        let config = GlimmerConfig::from_json_file(&model_dir.join("config.json"))?;
        let mut weights =
            WeightManager::load(model_dir, &model_dir.join("model.safetensors.index.json"))?;
        let h = config.hidden_size;
        let q_rows_expected_fused = 2 * config.num_attention_heads * config.head_dim;
        let q_rows_expected_plain = config.num_attention_heads * config.head_dim;
        let (q_rows, q_cols) = weights.weight_shape("model.layers.0.self_attn.q_proj.weight")?;
        if q_cols * 8 != h {
            return Err(ForwardError::Shape(format!(
                "q_proj packed cols {} (logical {}) != hidden_size {}",
                q_cols,
                q_cols * 8,
                h
            )));
        }
        let q_fused = if q_rows == q_rows_expected_fused {
            true
        } else if q_rows == q_rows_expected_plain {
            false
        } else {
            return Err(ForwardError::Shape(format!(
                "q_proj rows {} matches neither fused ({}) nor plain ({})",
                q_rows, q_rows_expected_fused, q_rows_expected_plain
            )));
        };
        let mut geo_maps: Vec<Vec<GeometricMemoryMap>> = Vec::new();
        for _ in 0..config.num_hidden_layers {
            let mut layer_maps = Vec::new();
            for _ in 0..config.num_key_value_heads {
                let mut m = GeometricMemoryMap::new(1_000_000);
                m.use_picard = false;
                layer_maps.push(m);
            }
            geo_maps.push(layer_maps);
        }
        Ok(Self {
            config,
            weights,
            geo_maps,
            q_fused,
        })
    }
    fn layer_prefix(&self, layer: usize) -> String {
        format!("model.layers.{}", layer)
    }
    fn embed(&mut self, token_id: u32) -> Result<Vec<f32>, ForwardError> {
        let w = self.weights.get_weight("model.embed_tokens.weight")?;
        let h = self.config.hidden_size;
        let start = (token_id as usize) * h;
        if start + h > w.len() {
            return Err(ForwardError::Shape(format!(
                "token {} out of range",
                token_id
            )));
        }
        let emb = w[start..start + h].to_vec();
        Ok(scaleless_rmsnorm(&emb, self.config.rms_norm_eps))
    }
    fn split_q_gate(&self, q_full: &[f32]) -> Result<(Vec<f32>, Option<Vec<f32>>), ForwardError> {
        let nh = self.config.num_attention_heads;
        let hd = self.config.head_dim;
        if self.q_fused {
            if q_full.len() != 2 * nh * hd {
                return Err(ForwardError::Shape(format!(
                    "fused q len {} != {}",
                    q_full.len(),
                    2 * nh * hd
                )));
            }
            let mut q = Vec::with_capacity(nh * hd);
            let mut gate = Vec::with_capacity(nh * hd);
            for h in 0..nh {
                let base = h * 2 * hd;
                q.extend_from_slice(&q_full[base..base + hd]);
                gate.extend_from_slice(&q_full[base + hd..base + 2 * hd]);
            }
            Ok((q, Some(gate)))
        } else {
            if q_full.len() != nh * hd {
                return Err(ForwardError::Shape(format!(
                    "q len {} != {}",
                    q_full.len(),
                    nh * hd
                )));
            }
            Ok((q_full.to_vec(), None))
        }
    }
    fn attention(&mut self, layer: usize, x: &[f32], pos: usize) -> Result<Vec<f32>, ForwardError> {
        let h = self.config.hidden_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let eps = self.config.rms_norm_eps;
        let theta = self.config.rope_theta;
        let scale = self.config.attn_scale;
        let p = self.layer_prefix(layer);
        let xn = rmsnorm(
            x,
            &self
                .weights
                .get_weight(&format!("{}.input_layernorm.weight", p))?,
            eps,
        )?;
        let q_full = matvec(
            &self
                .weights
                .get_weight(&format!("{}.self_attn.q_proj.weight", p))?,
            if self.q_fused { 2 * nh * hd } else { nh * hd },
            h,
            &xn,
        )?;
        let (mut q, gate) = self.split_q_gate(&q_full)?;
        let mut k = matvec(
            &self
                .weights
                .get_weight(&format!("{}.self_attn.k_proj.weight", p))?,
            nkv * hd,
            h,
            &xn,
        )?;
        let v = matvec(
            &self
                .weights
                .get_weight(&format!("{}.self_attn.v_proj.weight", p))?,
            nkv * hd,
            h,
            &xn,
        )?;
        for head in 0..nh {
            let base = head * hd;
            let normed = scaleless_rmsnorm(&q[base..base + hd], eps);
            q[base..base + hd].copy_from_slice(&normed);
        }
        for head in 0..nkv {
            let base = head * hd;
            let normed = scaleless_rmsnorm(&k[base..base + hd], eps);
            k[base..base + hd].copy_from_slice(&normed);
        }
        if self.config.use_rope(layer) {
            apply_rope_neox(&mut q, hd, pos, theta)?;
            apply_rope_neox(&mut k, hd, pos, theta)?;
        }
        let phase = Phase {
            theta_f: crate::geometry::fixed::Q32_31::ZERO,
            theta_s: crate::geometry::fixed::Q32_31::ZERO,
        };
        for kv_h in 0..nkv {
            let geo = &mut self.geo_maps[layer][kv_h];
            let k_head = &k[kv_h * hd..(kv_h + 1) * hd];
            let v_head = &v[kv_h * hd..(kv_h + 1) * hd];
            let lv = f32_to_lorentz(k_head)?;
            let mut payload = Vec::with_capacity(2 * hd * 4);
            for f in k_head.iter() {
                payload.extend_from_slice(&f.to_le_bytes());
            }
            for f in v_head.iter() {
                payload.extend_from_slice(&f.to_le_bytes());
            }
            geo.write(lv, phase, payload)?;
        }
        let is_sliding = self.config.is_sliding(layer);
        let window_start = pos.saturating_sub(self.config.sliding_window - 1) as u64;
        let mut attn_out = vec![0.0f32; nh * hd];
        for qh in 0..nh {
            // GQA: each Q head attends only to its assigned KV head's atoms.
            let kv_h = qh / (nh / nkv);
            let geo = &self.geo_maps[layer][kv_h];
            let q_head = &q[qh * hd..(qh + 1) * hd];
            let q_lv = f32_to_lorentz(q_head)?;
            let n_atoms = geo.atoms.len();
            let top_k = 512.min(n_atoms).max(1);
            let neighbors = geo.retrieve(&q_lv, top_k, false)?;
            if neighbors.is_empty() {
                continue;
            }
            let mut scored: Vec<(f32, Vec<f32>)> = Vec::new();
            for atom in neighbors {
                // One atom per position in this KV head's map: no nkv division.
                let atom_pos = atom.clock.0.saturating_sub(1);
                if is_sliding && atom_pos < window_start {
                    continue;
                }
                let payload = &atom.payload;
                if payload.len() < 2 * hd * 4 {
                    continue;
                }
                let k_head: Vec<f32> = payload[0..hd * 4]
                    .chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let v_head: Vec<f32> = payload[hd * 4..2 * hd * 4]
                    .chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                if k_head.len() != hd || v_head.len() != hd {
                    continue;
                }
                let dot: f32 = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
                scored.push((dot * scale, v_head));
            }
            if scored.is_empty() {
                continue;
            }
            let scores: Vec<f32> = scored.iter().map(|(s, _)| *s).collect();
            let probs = softmax(&scores)?;
            for d in 0..hd {
                let mut acc = 0.0f32;
                for (pi, (_, vh)) in probs.iter().zip(scored.iter()) {
                    acc += pi * vh[d];
                }
                attn_out[qh * hd + d] = acc;
            }
        }
        if let Some(g) = gate {
            for i in 0..attn_out.len() {
                attn_out[i] *= sigmoid(g[i]);
            }
        }
        let o = matvec(
            &self
                .weights
                .get_weight(&format!("{}.self_attn.o_proj.weight", p))?,
            h,
            nh * hd,
            &attn_out,
        )?;
        Ok(o)
    }
    fn mlp(&mut self, layer: usize, x: &[f32]) -> Result<Vec<f32>, ForwardError> {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let p = self.layer_prefix(layer);
        let gate = matvec(
            &self
                .weights
                .get_weight(&format!("{}.mlp.gate_proj.weight", p))?,
            inter,
            h,
            x,
        )?;
        let up = matvec(
            &self
                .weights
                .get_weight(&format!("{}.mlp.up_proj.weight", p))?,
            inter,
            h,
            x,
        )?;
        let gated = silu(&gate);
        let hidden: Vec<f32> = gated.iter().zip(up.iter()).map(|(g, u)| g * u).collect();
        let down = matvec(
            &self
                .weights
                .get_weight(&format!("{}.mlp.down_proj.weight", p))?,
            h,
            inter,
            &hidden,
        )?;
        Ok(down)
    }
    fn decoder_layer(
        &mut self,
        layer: usize,
        x: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, ForwardError> {
        let eps = self.config.rms_norm_eps;
        let post_eps = self.config.post_norm_eps;
        let p = self.layer_prefix(layer);
        let attn_branch = self.attention(layer, x, pos)?;
        let attn_normed = rmsnorm(
            &attn_branch,
            &self
                .weights
                .get_weight(&format!("{}.post_attn_norm.weight", p))?,
            post_eps,
        )?;
        let x2: Vec<f32> = x
            .iter()
            .zip(attn_normed.iter())
            .map(|(a, b)| a + b)
            .collect();
        let mlp_in = rmsnorm(
            &x2,
            &self
                .weights
                .get_weight(&format!("{}.post_attention_layernorm.weight", p))?,
            eps,
        )?;
        let mlp_branch = self.mlp(layer, &mlp_in)?;
        let mlp_normed = rmsnorm(
            &mlp_branch,
            &self
                .weights
                .get_weight(&format!("{}.post_ffn_norm.weight", p))?,
            post_eps,
        )?;
        let x3: Vec<f32> = x2
            .iter()
            .zip(mlp_normed.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(x3)
    }
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>, ForwardError> {
        let mut x = self.embed(token_id)?;
        for layer in 0..self.config.num_hidden_layers {
            x = self.decoder_layer(layer, &x, pos)?;
        }
        let xn = rmsnorm(
            &x,
            &self.weights.get_weight("model.norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        let mut logits = matvec(
            &self.weights.get_weight("lm_head.weight")?,
            self.config.vocab_size,
            self.config.hidden_size,
            &xn,
        )?;
        let cap = self.config.output_soft_cap_temp;
        let mult = self.config.output_multiplier;
        if cap > 0.0 {
            for v in logits.iter_mut() {
                *v = cap * (*v * mult / cap).tanh();
            }
        } else {
            for v in logits.iter_mut() {
                *v *= mult;
            }
        }
        Ok(logits)
    }
    pub fn reset_memory(&mut self) {
        for layer_maps in self.geo_maps.iter_mut() {
            for geo in layer_maps.iter_mut() {
                let mut fresh = GeometricMemoryMap::new(1_000_000);
                fresh.use_picard = false;
                *geo = fresh;
            }
        }
    }
}
