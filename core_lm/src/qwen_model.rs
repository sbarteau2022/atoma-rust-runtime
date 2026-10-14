//! Qwen3.5-MoE weight manager with Zero-Copy Metal/Accelerate Fused Quantized GEMV
//! and Single-Row Embedding Dequantization (`get_weight_row`).
//!
//! Weight layout (OptiQ):
//!   - Quantized: `{name}.weight` + `{name}.scales` + `{name}.biases`
//!   - Per-tensor bits from `optiq_metadata.json` (`4`, `6`, or `8`)
//!   - Unquantized: norms, biases loaded directly

use crate::cache::geometric::MemoryError;
use crate::geometry::bridge::BridgeError;
use crate::metal::{accelerate_matvec_f32, dequantize_row_zero_copy, qmatvec_fused_zero_copy};
use crate::metal_quant::MetalQuantGemm;
use crate::model_config::ConfigError;
use crate::ops::OpsError;
use crate::optiq::{DequantError, dequantize, load_f32_tensor};
use crate::safetensors::{SafeTensorError, SafeTensorFile};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub enum ModelError {
    SafeTensor(SafeTensorError),
    Dequant(DequantError),
    Config(ConfigError),
    Ops(OpsError),
    Memory(MemoryError),
    Bridge(BridgeError),
    MissingWeight(String),
    NotLoaded,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::SafeTensor(e) => write!(f, "Safetensor: {}", e),
            ModelError::Dequant(e) => write!(f, "Dequant: {}", e),
            ModelError::Config(e) => write!(f, "Config: {}", e),
            ModelError::Ops(e) => write!(f, "Ops: {}", e),
            ModelError::Memory(e) => write!(f, "Memory: {}", e),
            ModelError::Bridge(e) => write!(f, "Bridge: {}", e),
            ModelError::MissingWeight(s) => write!(f, "Missing weight: {}", s),
            ModelError::NotLoaded => write!(f, "Model not loaded"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<SafeTensorError> for ModelError {
    fn from(e: SafeTensorError) -> Self {
        ModelError::SafeTensor(e)
    }
}
impl From<DequantError> for ModelError {
    fn from(e: DequantError) -> Self {
        ModelError::Dequant(e)
    }
}
impl From<ConfigError> for ModelError {
    fn from(e: ConfigError) -> Self {
        ModelError::Config(e)
    }
}
impl From<OpsError> for ModelError {
    fn from(e: OpsError) -> Self {
        ModelError::Ops(e)
    }
}
impl From<MemoryError> for ModelError {
    fn from(e: MemoryError) -> Self {
        ModelError::Memory(e)
    }
}
impl From<BridgeError> for ModelError {
    fn from(e: BridgeError) -> Self {
        ModelError::Bridge(e)
    }
}

pub struct WeightManager {
    shards: Vec<SafeTensorFile>,
    tensor_to_shard: HashMap<String, usize>,
    bits_map: HashMap<String, u32>,
    group_map: HashMap<String, usize>,
    default_group: usize,
    cache: HashMap<String, Vec<f32>>,
    pub retain_cache: bool,
    metal_gemm: Option<MetalQuantGemm>,
    pub use_metal: bool,
}

impl WeightManager {
    pub fn load(model_dir: &Path, index_path: &Path) -> Result<Self, ModelError> {
        let index_data = std::fs::read_to_string(index_path)
            .map_err(|e| ModelError::Config(ConfigError::Io(e.to_string())))?;
        let index: serde_json::Value = serde_json::from_str(&index_data)
            .map_err(|e| ModelError::Config(ConfigError::Parse(e.to_string())))?;

        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| ModelError::Config(ConfigError::Parse("no weight_map".to_string())))?;

        let mut shard_files: HashMap<String, usize> = HashMap::new();
        let mut tensor_to_shard: HashMap<String, usize> = HashMap::new();
        let mut shards: Vec<SafeTensorFile> = Vec::new();

        for (tensor_name, shard_name) in weight_map {
            let shard_name = shard_name.as_str().unwrap_or("");
            let idx = if let Some(&i) = shard_files.get(shard_name) {
                i
            } else {
                let path = model_dir.join(shard_name);
                let sf = SafeTensorFile::open(&path)?;
                shards.push(sf);
                let i = shards.len() - 1;
                shard_files.insert(shard_name.to_string(), i);
                i
            };
            tensor_to_shard.insert(tensor_name.clone(), idx);
        }

        let mut bits_map: HashMap<String, u32> = HashMap::new();
        let mut group_map: HashMap<String, usize> = HashMap::new();
        let mut default_group = 64usize;

        let meta_path = model_dir.join("optiq_metadata.json");
        if meta_path.exists() {
            let meta_data = std::fs::read_to_string(&meta_path).unwrap_or_default();
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_data) {
                if let Some(gs) = meta["group_size"].as_u64() {
                    default_group = gs as usize;
                }
                if let Some(per_layer) = meta["per_layer"].as_object() {
                    for (name, info) in per_layer {
                        if let Some(b) = info["bits"].as_u64() {
                            bits_map.insert(name.clone(), b as u32);
                        }
                        if let Some(g) = info["group_size"].as_u64() {
                            group_map.insert(name.clone(), g as usize);
                        }
                    }
                }
            }
        }

        const MSL_SOURCE: &str = include_str!("quant_gemv.metal");
        let metal_gemm = MetalQuantGemm::new(MSL_SOURCE).ok();
        let use_metal = metal_gemm.is_some();

        Ok(Self {
            shards,
            tensor_to_shard,
            bits_map,
            group_map,
            default_group,
            cache: HashMap::new(),
            retain_cache: false,
            metal_gemm,
            use_metal,
        })
    }

    fn shard_for(&self, tensor: &str) -> Result<&SafeTensorFile, ModelError> {
        let idx = self
            .tensor_to_shard
            .get(tensor)
            .ok_or_else(|| ModelError::MissingWeight(tensor.to_string()))?;
        Ok(&self.shards[*idx])
    }

    fn quant_for(&self, base: &str, name: &str) -> (u32, usize) {
        let bits = self
            .bits_map
            .get(base)
            .copied()
            .or_else(|| self.bits_map.get(name).copied())
            .unwrap_or(4);
        let group = self
            .group_map
            .get(base)
            .copied()
            .or_else(|| self.group_map.get(name).copied())
            .unwrap_or(self.default_group);
        (bits, group)
    }

    pub fn get_weight(&mut self, name: &str) -> Result<Vec<f32>, ModelError> {
        if let Some(v) = self.cache.get(name) {
            return Ok(v.clone());
        }
        let base = name.trim_end_matches(".weight");
        let (bits, group_size) = self.quant_for(base, name);

        let scales_name = format!("{}.scales", base);
        let biases_name = format!("{}.biases", base);
        let file = self.shard_for(name)?;
        let has_scales = self.tensor_to_shard.contains_key(&scales_name);

        let data = if has_scales {
            let scales_file = self.shard_for(&scales_name)?;
            let biases_file = self.shard_for(&biases_name)?;
            dequantize(
                file.tensor_data(name)?,
                file.tensor_info(name)?,
                scales_file.tensor_data(&scales_name)?,
                scales_file.tensor_info(&scales_name)?,
                biases_file.tensor_data(&biases_name)?,
                biases_file.tensor_info(&biases_name)?,
                bits,
                group_size,
            )?
        } else {
            load_f32_tensor(file, name)?
        };
        if self.retain_cache || data.len() <= 65_536 {
            self.cache.insert(name.to_string(), data.clone());
        }
        Ok(data)
    }

    /// Single-row zero-copy dequantization for `QwenModel::embed(token_id)`.
    pub fn get_weight_row(
        &mut self,
        name: &str,
        row: usize,
        cols: usize,
    ) -> Result<Vec<f32>, ModelError> {
        if let Some(cached) = self.cache.get(name) {
            let start = row.saturating_mul(cols);
            if start + cols > cached.len() {
                return Err(ModelError::MissingWeight(format!(
                    "row {row} out of range for cached {name}"
                )));
            }
            return Ok(cached[start..start + cols].to_vec());
        }

        let base = name.trim_end_matches(".weight");
        let scales_name = format!("{}.scales", base);
        let biases_name = format!("{}.biases", base);
        let has_scales = self.tensor_to_shard.contains_key(&scales_name);

        if has_scales && !self.retain_cache {
            let (bits, group_size) = self.quant_for(base, name);
            let w_file = self.shard_for(name)?;
            let w_info = w_file.tensor_info(name)?;
            let w_data = w_file.tensor_data(name)?;
            let s_file = self.shard_for(&scales_name)?;
            let s_info = s_file.tensor_info(&scales_name)?;
            let s_data = s_file.tensor_data(&scales_name)?;
            let b_file = self.shard_for(&biases_name)?;
            let b_info = b_file.tensor_info(&biases_name)?;
            let b_data = b_file.tensor_data(&biases_name)?;

            return dequantize_row_zero_copy(
                w_data,
                &w_info.dtype,
                s_data,
                &s_info.dtype,
                b_data,
                &b_info.dtype,
                1.0,
                bits,
                group_size,
                row,
                cols,
            )
            .map_err(|e| ModelError::Dequant(DequantError::ShapeMismatch(e)));
        }

        let full = self.get_weight(name)?;
        let start = row.saturating_mul(cols);
        if start + cols > full.len() {
            return Err(ModelError::MissingWeight(format!(
                "row {row} out of range for {name}"
            )));
        }
        Ok(full[start..start + cols].to_vec())
    }

    /// Zero-copy Metal / Multi-Core Fused Quantized GEMV (`y = W * x`).
    pub fn matvec_weight(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, ModelError> {
        if let Some(cached) = self.cache.get(name) {
            return Ok(accelerate_matvec_f32(cached, rows, cols, x));
        }

        let base = name.trim_end_matches(".weight");
        let scales_name = format!("{}.scales", base);
        let biases_name = format!("{}.biases", base);
        let has_scales = self.tensor_to_shard.contains_key(&scales_name);

        if has_scales && !self.retain_cache {
            let (bits, group_size) = self.quant_for(base, name);
            let w_file = self.shard_for(name)?;
            let w_info = w_file.tensor_info(name)?;
            let w_data = w_file.tensor_data(name)?;
            let s_file = self.shard_for(&scales_name)?;
            let s_info = s_file.tensor_info(&scales_name)?;
            let s_data = s_file.tensor_data(&scales_name)?;
            let b_file = self.shard_for(&biases_name)?;
            let b_info = b_file.tensor_info(&biases_name)?;
            let b_data = b_file.tensor_data(&biases_name)?;

            if self.use_metal && w_info.dtype == "U32" && w_data.len() % 4 == 0 && (w_data.as_ptr() as usize) % 4 == 0 {
                if let Some(gemm) = self.metal_gemm.as_ref() {
                    let packed: &[u32] = unsafe {
                        std::slice::from_raw_parts(w_data.as_ptr() as *const u32, w_data.len() / 4)
                    };
                    let res = if bits == 4 {
                        gemm.q4_gemv(
                            packed,
                            s_data,
                            b_data,
                            None,
                            x,
                            rows as u32,
                            cols as u32,
                            group_size as u32,
                            &s_info.dtype,
                            &b_info.dtype,
                        )
                    } else if bits == 6 {
                        gemm.q6_gemv(
                            packed,
                            s_data,
                            b_data,
                            None,
                            x,
                            rows as u32,
                            cols as u32,
                            group_size as u32,
                            &s_info.dtype,
                            &b_info.dtype,
                        )
                    } else {
                        gemm.q_generic_gemv(
                            packed,
                            s_data,
                            b_data,
                            None,
                            x,
                            rows as u32,
                            cols as u32,
                            group_size as u32,
                            bits,
                            &s_info.dtype,
                            &b_info.dtype,
                        )
                    };
                    if let Ok(out) = res {
                        return Ok(out);
                    }
                }
            }

            return qmatvec_fused_zero_copy(
                w_data,
                &w_info.dtype,
                s_data,
                &s_info.dtype,
                b_data,
                &b_info.dtype,
                None,
                bits,
                group_size,
                rows,
                cols,
                x,
            )
            .map_err(|e| ModelError::Dequant(DequantError::ShapeMismatch(e)));
        }

        let w = self.get_weight(name)?;
        Ok(accelerate_matvec_f32(&w, rows, cols, x))
    }

    pub fn get_f32(&mut self, name: &str) -> Result<Vec<f32>, ModelError> {
        if let Some(v) = self.cache.get(name) {
            return Ok(v.clone());
        }
        let file = self.shard_for(name)?;
        let data = load_f32_tensor(file, name)?;
        self.cache.insert(name.to_string(), data.clone());
        Ok(data)
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}
