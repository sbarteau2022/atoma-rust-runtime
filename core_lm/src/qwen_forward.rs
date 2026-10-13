//! Qwen3.5-MoE forward pass with geometric memory.
//!
//! Each full_attention layer owns a GeometricMemoryMap.
//! Keys are projected to LorentzVector; values stored as payload.
//! There are NO k_cache / v_cache tensors in this module.

use crate::cache::geometric::{GeometricMemoryMap, Phase};
use crate::geometry::bridge::f32_to_lorentz;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::model_config::{LayerType, ModelConfig};
use crate::ops::{apply_rope, argmax, matvec, rmsnorm, silu, softmax, top_k_indices};
use crate::qwen_model::{ModelError, WeightManager};
use std::path::Path;

pub struct QwenModel {
    pub config: ModelConfig,
    pub weights: WeightManager,
    pub geo_maps: Vec<GeometricMemoryMap>,
}

impl QwenModel {
    pub fn load(model_dir: &Path) -> Result<Self, ModelError> {
        let config = ModelConfig::from_json_file(&model_dir.join("config.json"))?;
        let weights =
            WeightManager::load(model_dir, &model_dir.join("model.safetensors.index.json"))?;
        let num_full = config.full_attention_layers().len();
        let mut geo_maps = Vec::new();
        for _ in 0..num_full {
            geo_maps.push(GeometricMemoryMap::new(1_000_000));
        }
        Ok(Self {
            config,
            weights,
            geo_maps,
        })
    }

    fn p(&self, l: usize) -> String {
        format!("language_model.model.layers.{}", l)
    }

    fn embed(&mut self, token_id: u32) -> Result<Vec<f32>, ModelError> {
        let w = self
            .weights
            .get_weight("language_model.model.embed_tokens.weight")?;
        let h = self.config.hidden_size;
        let start = (token_id as usize) * h;
        if start + h > w.len() {
            return Err(ModelError::MissingWeight(format!(
                "token {} out of range",
                token_id
            )));
        }
        Ok(w[start..start + h].to_vec())
    }

    fn full_attention(
        &mut self,
        layer: usize,
        geo_idx: usize,
        x: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let h = self.config.hidden_size;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let eps = self.config.rms_norm_eps;
        let theta = self.config.rope_theta;
        let prf = self.config.partial_rotary_factor;
        let p = self.p(layer);

        let xn = rmsnorm(
            x,
            &self
                .weights
                .get_f32(&format!("{}.input_layernorm.weight", p))?,
            eps,
        )?;

        let q = matvec(
            &self
                .weights
                .get_weight(&format!("{}.self_attn.q_proj.weight", p))?,
            nh * hd,
            h,
            &xn,
        )?;
        let k = matvec(
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

        let mut q = q;
        let mut k = k;
        let rotary_dim = (hd as f32 * prf) as usize;
        apply_rope(&mut q, hd, rotary_dim, pos, theta)?;
        apply_rope(&mut k, hd, rotary_dim, pos, theta)?;

        let phase = Phase {
            theta_f: Q32_31::ZERO,
            theta_s: Q32_31::ZERO,
        };
        let geo = &mut self.geo_maps[geo_idx];

        for kv_h in 0..nkv {
            let k_head = &k[kv_h * hd..(kv_h + 1) * hd];
            let v_head = &v[kv_h * hd..(kv_h + 1) * hd];
            let lv = f32_to_lorentz(k_head)?;
            let v_bytes: Vec<u8> = v_head.iter().flat_map(|f| f.to_le_bytes()).collect();
            geo.write(lv, phase, v_bytes)?;
        }

        let repeat = nh / nkv;
        let mut attn_out = vec![0.0f32; nh * hd];
        let top_k = 64;

        for qh in 0..nh {
            let q_head = &q[qh * hd..(qh + 1) * hd];
            let q_lv = f32_to_lorentz(q_head)?;

            let neighbors = geo.retrieve(&q_lv, top_k, false)?;
            if neighbors.is_empty() {
                continue;
            }

            let mut scores = Vec::new();
            let mut v_list = Vec::new();
            for atom in neighbors {
                let k_lv = atom.lorentz;
                let ip = LorentzVector::inner_product(&q_lv, &k_lv).to_f64() as f32;
                let v_bytes = &atom.payload;
                let vh: Vec<f32> = v_bytes
                    .chunks(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                scores.push(-ip / (hd as f32).sqrt());
                v_list.push(vh);
            }

            let probs = softmax(&scores)?;
            for d in 0..hd {
                let mut acc = 0.0f32;
                for (pi, vh) in probs.iter().zip(v_list.iter()) {
                    if d < vh.len() {
                        acc += pi * vh[d];
                    }
                }
                attn_out[qh * hd + d] = acc;
            }
            let _ = repeat;
        }

        let o = matvec(
            &self
                .weights
                .get_weight(&format!("{}.self_attn.o_proj.weight", p))?,
            h,
            nh * hd,
            &attn_out,
        )?;

        Ok(x.iter().zip(o.iter()).map(|(a, b)| a + b).collect())
    }

    fn moe_mlp(&mut self, layer: usize, x: &[f32]) -> Result<Vec<f32>, ModelError> {
        let h = self.config.hidden_size;
        let inter = self.config.moe_intermediate_size;
        let shared_inter = self.config.shared_expert_intermediate_size;
        let num_experts = self.config.num_experts;
        let top_k = self.config.num_experts_per_tok;
        let eps = self.config.rms_norm_eps;
        let p = self.p(layer);

        let xn = rmsnorm(
            x,
            &self
                .weights
                .get_f32(&format!("{}.post_attention_layernorm.weight", p))?,
            eps,
        )?;

        let gate_logits = matvec(
            &self.weights.get_weight(&format!("{}.mlp.gate.weight", p))?,
            num_experts,
            h,
            &xn,
        )?;
        let gate_probs = softmax(&gate_logits)?;
        let top_experts = top_k_indices(&gate_probs, top_k);

        let switch_gate = self
            .weights
            .get_weight(&format!("{}.mlp.switch_mlp.gate_proj.weight", p))?;
        let switch_up = self
            .weights
            .get_weight(&format!("{}.mlp.switch_mlp.up_proj.weight", p))?;
        let switch_down = self
            .weights
            .get_weight(&format!("{}.mlp.switch_mlp.down_proj.weight", p))?;

        let mut out = vec![0.0f32; h];

        for &ei in &top_experts {
            let prob = gate_probs[ei];
            let g_base = ei * inter * h;
            let d_base = ei * h * inter;

            if g_base + inter * h > switch_gate.len() {
                continue;
            }
            if d_base + h * inter > switch_down.len() {
                continue;
            }

            let gate_w = &switch_gate[g_base..g_base + inter * h];
            let up_w = &switch_up[g_base..g_base + inter * h];
            let down_w = &switch_down[d_base..d_base + h * inter];

            let gate = matvec(gate_w, inter, h, &xn)?;
            let up = matvec(up_w, inter, h, &xn)?;
            let gated = silu(&gate);
            let hidden: Vec<f32> = gated.iter().zip(up.iter()).map(|(g, u)| g * u).collect();
            let down = matvec(down_w, h, inter, &hidden)?;
            for i in 0..h {
                out[i] += prob * down[i];
            }
        }

        let sp = format!("{}.mlp.shared_expert", p);
        let s_gate = matvec(
            &self
                .weights
                .get_weight(&format!("{}.gate_proj.weight", sp))?,
            shared_inter,
            h,
            &xn,
        )?;
        let s_up = matvec(
            &self.weights.get_weight(&format!("{}.up_proj.weight", sp))?,
            shared_inter,
            h,
            &xn,
        )?;
        let s_gated = silu(&s_gate);
        let s_hidden: Vec<f32> = s_gated
            .iter()
            .zip(s_up.iter())
            .map(|(g, u)| g * u)
            .collect();
        let s_down = matvec(
            &self
                .weights
                .get_weight(&format!("{}.down_proj.weight", sp))?,
            h,
            shared_inter,
            &s_hidden,
        )?;
        for i in 0..h {
            out[i] += s_down[i];
        }

        Ok(x.iter().zip(out.iter()).map(|(a, b)| a + b).collect())
    }

    fn linear_attention(&mut self, layer: usize, x: &[f32]) -> Result<Vec<f32>, ModelError> {
        let h = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let p = self.p(layer);

        let xn = rmsnorm(
            x,
            &self
                .weights
                .get_f32(&format!("{}.input_layernorm.weight", p))?,
            eps,
        )?;

        let qkv = matvec(
            &self
                .weights
                .get_weight(&format!("{}.linear_attn.in_proj_qkv.weight", p))?,
            h * 3,
            h,
            &xn,
        )?;
        let z = matvec(
            &self
                .weights
                .get_weight(&format!("{}.linear_attn.in_proj_z.weight", p))?,
            h,
            h,
            &xn,
        )?;
        let o = matvec(
            &self
                .weights
                .get_weight(&format!("{}.linear_attn.out_proj.weight", p))?,
            h,
            h,
            &qkv[..h],
        )?;

        let gated: Vec<f32> = o.iter().zip(silu(&z).iter()).map(|(a, b)| a * b).collect();
        Ok(x.iter().zip(gated.iter()).map(|(a, b)| a + b).collect())
    }

    pub fn forward(&mut self, token_id: u32, pos: usize) -> Result<Vec<f32>, ModelError> {
        let mut x = self.embed(token_id)?;
        let mut geo_idx = 0;

        for layer in 0..self.config.num_hidden_layers {
            let is_full = matches!(self.config.layer_types[layer], LayerType::FullAttention);
            if is_full {
                x = self.full_attention(layer, geo_idx, &x, pos)?;
                geo_idx += 1;
            } else {
                x = self.linear_attention(layer, &x)?;
            }
            x = self.moe_mlp(layer, &x)?;
        }

        let norm_w = self.weights.get_f32("language_model.model.norm.weight")?;
        let xn = rmsnorm(&x, &norm_w, self.config.rms_norm_eps)?;
        let logits = matvec(
            &self.weights.get_weight("language_model.lm_head.weight")?,
            self.config.vocab_size,
            self.config.hidden_size,
            &xn,
        )?;
        Ok(logits)
    }

    pub fn reset_memory(&mut self) {
        for geo in self.geo_maps.iter_mut() {
            *geo = GeometricMemoryMap::new(1_000_000);
        }
    }
}
