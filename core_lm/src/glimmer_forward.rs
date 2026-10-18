//! Muse Glimmer forward pass with GeometricMemoryMap replacing KV cache,
//! Layer 3 `ImmuneForwardGate` (`CompliancePermit` check on every forward pass),
//! and Zero-Copy Metal/Accelerate Fused Quantized `matvec_weight` execution.
//!
//! Architecture (from muse_glimmer_mlx.py reference):
//! - 52 layers, pure transformer (39 sliding-window-2048 + 13 full attention)
//! - Sandwich norms: x = x + post_norm(branch(pre_norm(x)))
//! - QK-norm (scaleless RMS) BEFORE RoPE
//! - Folded attention scale: qk_scale_factor / head_dim
//! - q_proj fused with output gate ([q_head; gate_head] per-head interleaved)
//! - Interleaved RoPE (nn.RoPE traditional=true). This packaged config leaves
//!   rope_is_neox_style unset, so the checkpoint model uses consecutive pairs.
//!   NoPE on full_attention layers.
//! - SwiGLU MLP
//! - Logit soft cap: cap * tanh(logits * mult / cap)
//!
//! Three-Layer & Hardware Integration:
//! - `core_lm -> immune_system`: Every `forward` call requires a verified
//!   `ForwardCompliancePermit` from `self.immune_gate` and records the layer's
//!   `BridgeLossReport` (`Qualification #9`).
//! - Hardware Acceleration: All 7 per-layer linear projections (`q_proj`, `k_proj`,
//!   `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) and `lm_head` execute via
//!   `WeightManager::matvec_weight` (`qmatvec_fused_zero_copy` + `Accelerate::cblas_sgemv`),
//!   eliminating ~108 GB/token of temporary `Vec<f32>` heap allocations.

use crate::cache::geometric::{GeometricMemoryMap, MemoryError};
use crate::attention_backend::{AttentionBackend, AttentionInput};
use crate::geometry::bridge::{BridgeError, BridgeLossReport, f32_to_atom_address, f32_to_lorentz};
use crate::glimmer_model::{GlimmerConfig, GlimmerError, WeightManager};
use crate::ops::{OpsError, rmsnorm, silu, softmax};
use crate::{ForwardCompliancePermit, ImmuneForwardGate};
use std::path::Path;

#[derive(Debug)]
pub enum ForwardError {
    Model(GlimmerError),
    Ops(OpsError),
    Memory(MemoryError),
    Bridge(BridgeError),
    Compliance(String),
    Shape(String),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardError::Model(e) => write!(f, "Model: {}", e),
            ForwardError::Ops(e) => write!(f, "Ops: {}", e),
            ForwardError::Memory(e) => write!(f, "Memory: {}", e),
            ForwardError::Bridge(e) => write!(f, "Bridge: {}", e),
            ForwardError::Compliance(s) => write!(f, "ComplianceRejected: {}", s),
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

/// Consecutive-pair RoPE, matching `mx.fast.rope(..., traditional=true)`.
///
/// Pair `i` is elements `(2i, 2i+1)` with `freq = theta^(-2i/head_dim)`.
fn apply_rope_traditional(
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
            "head_dim must be even for RoPE".to_string(),
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
            let i0 = base + 2 * i;
            let x0 = x[i0];
            let x1 = x[i0 + 1];
            x[i0] = x0 * cos - x1 * sin;
            x[i0 + 1] = x0 * sin + x1 * cos;
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
    pub immune_gate: ImmuneForwardGate,
}

impl GlimmerModel {
    pub fn load(model_dir: &Path) -> Result<Self, ForwardError> {
        let config = GlimmerConfig::from_json_file(&model_dir.join("config.json"))?;
        let weights =
            WeightManager::load(model_dir, &model_dir.join("model.safetensors.index.json"))?;
        let h = config.hidden_size;
        let q_rows_expected_fused = 2 * config.num_attention_heads * config.head_dim;
        let q_rows_expected_plain = config.num_attention_heads * config.head_dim;
        let (q_rows, q_cols) = weights.weight_shape("model.layers.0.self_attn.q_proj.weight")?;
        let q_bits = weights.quant_bits("model.layers.0.self_attn.q_proj.weight");
        let packed_bits = q_cols.saturating_mul(32);
        let logical_cols = packed_bits / q_bits.max(1) as usize;
        if q_bits == 0 || packed_bits % q_bits as usize != 0 || logical_cols != h {
            return Err(ForwardError::Shape(format!(
                "q_proj packed cols {} at {} bits (logical {}) != hidden_size {}",
                q_cols, q_bits, logical_cols, h
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
                m.use_picard = true;
                layer_maps.push(m);
            }
            geo_maps.push(layer_maps);
        }
        let fp = weights.structural_fingerprint();
        Ok(Self {
            config,
            weights,
            geo_maps,
            q_fused,
            immune_gate: ImmuneForwardGate::new(fp),
        })
    }

    fn layer_prefix(&self, layer: usize) -> String {
        format!("model.layers.{}", layer)
    }

    fn embed(&mut self, token_id: u32) -> Result<Vec<f32>, ForwardError> {
        let h = self.config.hidden_size;
        let emb = self
            .weights
            .get_weight_row("model.embed_tokens.weight", token_id as usize, h)?;
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

    fn attention(
        &mut self,
        layer: usize,
        x: &[f32],
        pos: usize,
        token_id: u32,
        backend: Option<&mut dyn AttentionBackend>,
    ) -> Result<(Vec<f32>, Option<BridgeLossReport>), ForwardError> {
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
        let q_full = self.weights.matvec_weight(
            &format!("{}.self_attn.q_proj.weight", p),
            if self.q_fused { 2 * nh * hd } else { nh * hd },
            h,
            &xn,
        )?;
        let (mut q, gate) = self.split_q_gate(&q_full)?;
        let mut k = self.weights.matvec_weight(
            &format!("{}.self_attn.k_proj.weight", p),
            nkv * hd,
            h,
            &xn,
        )?;
        let v = self.weights.matvec_weight(
            &format!("{}.self_attn.v_proj.weight", p),
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
            apply_rope_traditional(&mut q, hd, pos, theta)?;
            apply_rope_traditional(&mut k, hd, pos, theta)?;
        }

        let (mut attn_out, last_loss) = if let Some(backend) = backend {
            let input = AttentionInput {
                layer, token_id, position: pos,
                window_start: if self.config.is_sliding(layer) {
                    pos.saturating_sub(self.config.sliding_window.saturating_sub(1))
                } else { 0 },
                query_heads: nh, kv_heads: nkv, head_dim: hd, dot_scale: scale as f64,
                query: &q, key: &k, value: &v,
            };
            input.validate().map_err(ForwardError::Shape)?;
            let output = backend.attend(input).map_err(ForwardError::Compliance)?;
            if output.len() != nh * hd || !output.iter().all(|x| x.is_finite()) {
                return Err(ForwardError::Shape("attention backend returned invalid output".into()));
            }
            let (_, _, loss) = f32_to_atom_address(&k[..hd])?;
            (output, Some(loss))
        } else {
        let mut last_loss: Option<BridgeLossReport> = None;
        for kv_h in 0..nkv {
            let geo = &mut self.geo_maps[layer][kv_h];
            let k_head = &k[kv_h * hd..(kv_h + 1) * hd];
            let v_head = &v[kv_h * hd..(kv_h + 1) * hd];
            let (lv, phase, loss) = f32_to_atom_address(k_head)?;
            last_loss = Some(loss);
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
        (attn_out, last_loss)
        };
        if let Some(g) = gate {
            for i in 0..attn_out.len() {
                attn_out[i] *= sigmoid(g[i]);
            }
        }
        let o = self.weights.matvec_weight(
            &format!("{}.self_attn.o_proj.weight", p),
            h,
            nh * hd,
            &attn_out,
        )?;
        Ok((o, last_loss))
    }

    fn mlp(&mut self, layer: usize, x: &[f32]) -> Result<Vec<f32>, ForwardError> {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let p = self.layer_prefix(layer);
        let gate = self
            .weights
            .matvec_weight(&format!("{}.mlp.gate_proj.weight", p), inter, h, x)?;
        let up = self
            .weights
            .matvec_weight(&format!("{}.mlp.up_proj.weight", p), inter, h, x)?;
        let gated = silu(&gate);
        let hidden: Vec<f32> = gated.iter().zip(up.iter()).map(|(g, u)| g * u).collect();
        let down = self.weights.matvec_weight(
            &format!("{}.mlp.down_proj.weight", p),
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
        token_id: u32,
        backend: Option<&mut dyn AttentionBackend>,
    ) -> Result<(Vec<f32>, Option<BridgeLossReport>), ForwardError> {
        let eps = self.config.rms_norm_eps;
        let post_eps = self.config.post_norm_eps;
        let p = self.layer_prefix(layer);
        let (attn_branch, loss_rep) = self.attention(layer, x, pos, token_id, backend)?;
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
        Ok((x3, loss_rep))
    }

    /// Execute a single token forward pass gated by an explicit Layer 3 `ForwardCompliancePermit`
    /// (`core_lm -> immune_system`).
    pub fn forward_with_permit(
        &mut self,
        token_id: u32,
        pos: usize,
        permit: &ForwardCompliancePermit,
    ) -> Result<Vec<f32>, ForwardError> {
        self.forward_internal(token_id, pos, permit, None)
    }

    fn forward_internal(
        &mut self,
        token_id: u32,
        pos: usize,
        permit: &ForwardCompliancePermit,
        mut backend: Option<&mut dyn AttentionBackend>,
    ) -> Result<Vec<f32>, ForwardError> {
        self.immune_gate
            .verify_permit_and_loss(permit, None)
            .map_err(ForwardError::Compliance)?;

        let mut x = self.embed(token_id)?;
        let mut last_loss: Option<BridgeLossReport> = None;
        // Debug: dump hidden states if ATOMA_DUMP_HIDDEN is set
        let dump_hidden = std::env::var("ATOMA_DUMP_HIDDEN").is_ok();
        for layer in 0..self.config.num_hidden_layers {
            let (next_x, layer_loss) = match backend.as_deref_mut() {
                Some(b) => self.decoder_layer(layer, &x, pos, token_id, Some(b))?,
                None => self.decoder_layer(layer, &x, pos, token_id, None)?,
            };
            x = next_x;
            if dump_hidden && layer == 20 && pos == 0 {
                let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write("/tmp/hidden_l20.bin", &bytes).ok();
                eprintln!("[debug] Dumped hidden state layer 20: {} floats", x.len());
            }
            if layer_loss.is_some() {
                last_loss = layer_loss;
            }
        }

        // Verify mandatory loss accounting after layer stack execution
        self.immune_gate
            .verify_permit_and_loss(permit, last_loss)
            .map_err(ForwardError::Compliance)?;

        let xn = rmsnorm(
            &x,
            &self.weights.get_weight("model.norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        let mut logits = self.weights.matvec_weight(
            "lm_head.weight",
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

    /// Standard entry point: automatically requests a Layer 3 `ForwardCompliancePermit`
    /// from `self.immune_gate` on every forward pass (`core_lm -> immune_system`).
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>, ForwardError> {
        let current_fp = self.weights.structural_fingerprint();
        let permit = self
            .immune_gate
            .authorize_forward_pass(token_id, pos, current_fp)
            .map_err(ForwardError::Compliance)?;
        self.forward_with_permit(token_id, pos, &permit)
    }

    pub fn reset_memory(&mut self) {
        for layer_maps in self.geo_maps.iter_mut() {
            for geo in layer_maps.iter_mut() {
                let mut fresh = GeometricMemoryMap::new(1_000_000);
                fresh.use_picard = true;
                *geo = fresh;
            }
        }
    }

    /// Use the universal server's memory backend for every layer. The backend selects
    /// intervention layers; output gates, projections, norms, and MLPs remain model-owned.
    pub fn forward_with_backend(
        &mut self, token_id: u32, pos: usize, backend: &mut dyn AttentionBackend,
    ) -> Result<Vec<f32>, ForwardError> {
        let permit = self.immune_gate.authorize_forward_pass(
            token_id, pos, self.weights.structural_fingerprint(),
        ).map_err(ForwardError::Compliance)?;
        self.forward_internal(token_id, pos, &permit, Some(backend))
    }

    /// Execute a single decoder layer while capturing all intermediate hidden states:
    /// `(x_in, attn_delta, x_mid, mlp_delta, x_out)`.
    pub fn decoder_layer_instrumented(
        &mut self,
        layer: usize,
        x: &[f32],
        pos: usize,
        token_id: u32,
    ) -> Result<(LayerActivationRecord, Option<BridgeLossReport>), ForwardError> {
        let eps = self.config.rms_norm_eps;
        let post_eps = self.config.post_norm_eps;
        let p = self.layer_prefix(layer);
        let (attn_branch, loss_rep) = self.attention(layer, x, pos, token_id, None)?;
        let attn_delta = rmsnorm(
            &attn_branch,
            &self
                .weights
                .get_weight(&format!("{}.post_attn_norm.weight", p))?,
            post_eps,
        )?;
        let x_mid: Vec<f32> = x
            .iter()
            .zip(attn_delta.iter())
            .map(|(a, b)| a + b)
            .collect();
        let mlp_in = rmsnorm(
            &x_mid,
            &self
                .weights
                .get_weight(&format!("{}.post_attention_layernorm.weight", p))?,
            eps,
        )?;
        let mlp_branch = self.mlp(layer, &mlp_in)?;
        let mlp_delta = rmsnorm(
            &mlp_branch,
            &self
                .weights
                .get_weight(&format!("{}.post_ffn_norm.weight", p))?,
            post_eps,
        )?;
        let x_out: Vec<f32> = x_mid
            .iter()
            .zip(mlp_delta.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok((
            LayerActivationRecord {
                layer,
                token_id,
                next_token_id: None,
                pos,
                x_in: x.to_vec(),
                attn_delta,
                x_mid,
                mlp_delta,
                x_out,
            },
            loss_rep,
        ))
    }

    /// Execute only the MLP half of `layer` given `x_mid = x_in + attn_delta_from_map`
    /// (`Phase 1` & `Phase 2` Attention-Only Map Substitution).
    pub fn mlp_subblock_from_mid(
        &mut self,
        layer: usize,
        x_mid: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), ForwardError> {
        let eps = self.config.rms_norm_eps;
        let post_eps = self.config.post_norm_eps;
        let p = self.layer_prefix(layer);
        let mlp_in = rmsnorm(
            x_mid,
            &self
                .weights
                .get_weight(&format!("{}.post_attention_layernorm.weight", p))?,
            eps,
        )?;
        let mlp_branch = self.mlp(layer, &mlp_in)?;
        let mlp_delta = rmsnorm(
            &mlp_branch,
            &self
                .weights
                .get_weight(&format!("{}.post_ffn_norm.weight", p))?,
            post_eps,
        )?;
        let x_out: Vec<f32> = x_mid
            .iter()
            .zip(mlp_delta.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok((mlp_delta, x_out))
    }

    /// Public wrapper around `embed(token_id)` (`embed_tokens` + `scaleless_rmsnorm`) for
    /// Map-Native Phase 1/2/3 instrumentation and Phase 3 Weightless mode.
    pub fn embed_token(&mut self, token_id: u32) -> Result<Vec<f32>, ForwardError> {
        self.embed(token_id)
    }

    /// Public wrapper around final `model.norm` + `lm_head.weight` + soft-cap `tanh`
    /// (`Phase 3 Full Weightless` where only the decoder head runs after 52 map layers).
    pub fn decode_lm_head(&mut self, final_hidden: &[f32]) -> Result<Vec<f32>, ForwardError> {
        let xn = rmsnorm(
            final_hidden,
            &self.weights.get_weight("model.norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        let mut logits = self.weights.matvec_weight(
            "lm_head.weight",
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

    /// Execute a single non-replaced decoder layer (`attention` + `mlp`) during hybrid
    /// Phase 1 / Phase 2 Map-Native scaling experiments.
    pub fn execute_standard_layer(
        &mut self,
        layer: usize,
        x: &[f32],
        pos: usize,
        token_id: u32,
    ) -> Result<(Vec<f32>, Option<BridgeLossReport>), ForwardError> {
        self.decoder_layer(layer, x, pos, token_id, None)
    }

    /// Instrumented forward pass that captures `(x_in, attn_delta, x_mid, mlp_delta, x_out)`
    /// for every layer in `capture_layers` (`Phase 1` Layer 20 or all 52 layers).
    pub fn forward_instrumented(
        &mut self,
        token_id: u32,
        pos: usize,
        capture_layers: &std::collections::BTreeSet<usize>,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<LayerActivationRecord>), ForwardError> {
        let current_fp = self.weights.structural_fingerprint();
        let permit = self
            .immune_gate
            .authorize_forward_pass(token_id, pos, current_fp)
            .map_err(ForwardError::Compliance)?;
        self.immune_gate
            .verify_permit_and_loss(&permit, None)
            .map_err(ForwardError::Compliance)?;

        let mut x = self.embed(token_id)?;
        let mut last_loss: Option<BridgeLossReport> = None;
        let mut captures = Vec::with_capacity(capture_layers.len());

        for layer in 0..self.config.num_hidden_layers {
            if capture_layers.contains(&layer) {
                let (rec, layer_loss) = self.decoder_layer_instrumented(layer, &x, pos, token_id)?;
                x = rec.x_out.clone();
                captures.push(rec);
                if layer_loss.is_some() {
                    last_loss = layer_loss;
                }
            } else {
                let (next_x, layer_loss) = self.decoder_layer(layer, &x, pos, token_id, None)?;
                x = next_x;
                if layer_loss.is_some() {
                    last_loss = layer_loss;
                }
            }
        }

        self.immune_gate
            .verify_permit_and_loss(&permit, last_loss)
            .map_err(ForwardError::Compliance)?;
        let logits = self.decode_lm_head(&x)?;
        let next_token_id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32);
        for cap in captures.iter_mut() {
            cap.next_token_id = next_token_id;
        }
        Ok((logits, x, captures))
    }
}

/// Complete activation record captured at a single Glimmer transformer layer.
#[derive(Debug, Clone)]
pub struct LayerActivationRecord {
    pub layer: usize,
    pub token_id: u32,
    /// Next token ID (argmax of the step logits); stamped by `forward_instrumented`.
    pub next_token_id: Option<u32>,
    pub pos: usize,
    pub x_in: Vec<f32>,
    pub attn_delta: Vec<f32>,
    pub x_mid: Vec<f32>,
    pub mlp_delta: Vec<f32>,
    pub x_out: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::apply_rope_traditional;

    #[test]
    fn traditional_rope_matches_mlx_fast_rope() {
        // mx.fast.rope(arange(8), dims=8, traditional=True, base=10000, offset=3)
        let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        apply_rope_traditional(&mut x, 8, 3, 10_000.0).unwrap();
        let expected = [
            -0.14112002_f32,
            -0.9899925,
            1.0241122,
            3.4570496,
            3.8482227,
            5.1177325,
            5.9789734,
            7.0179687,
        ];
        for (got, want) in x.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "rope {got} != {want}");
        }
    }
}
