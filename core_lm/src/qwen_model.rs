//! Qwen3.5-MoE forward pass with GeometricMemoryMap replacing KV cache.
//!
//! Weight layout (OptiQ):
//!   - Quantized: {name}.weight + {name}.scales + {name}.biases
//!   - Per-tensor bits from optiq_metadata.json (4 or 8)
//!   - Unquantized: norms, biases loaded directly
//!
//! Geometric KV: in full_attention layers, K vectors are projected to
//! LorentzVector and stored in GeometricMemoryMap with V as payload.
//! NO k_cache / v_cache tensors exist anywhere in this file.

use crate::cache::geometric::{GeometricMemoryMap, MemoryError, Phase};
use crate::geometry::bridge::{BridgeError, f32_to_lorentz};
use crate::geometry::fixed::Q32_31;
use crate::model_config::{ConfigError, LayerType, ModelConfig};
use crate::ops::{OpsError, apply_rope, argmax, matvec, rmsnorm, silu, softmax, top_k_indices};
use crate::optiq::{DequantError, dequantize, load_f32_tensor};
use crate::safetensors::{SafeTensorError, SafeTensorFile};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    cache: HashMap<String, Vec<f32>>,
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
        let meta_path = model_dir.join("optiq_metadata.json");
        if meta_path.exists() {
            let meta_data = std::fs::read_to_string(&meta_path).unwrap_or_default();
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_data) {
                if let Some(per_layer) = meta["per_layer"].as_object() {
                    for (name, info) in per_layer {
                        if let Some(b) = info["bits"].as_u64() {
                            bits_map.insert(name.clone(), b as u32);
                        }
                    }
                }
            }
        }

        Ok(Self {
            shards,
            tensor_to_shard,
            bits_map,
            cache: HashMap::new(),
        })
    }

    fn shard_for(&self, tensor: &str) -> Result<&SafeTensorFile, ModelError> {
        let idx = self
            .tensor_to_shard
            .get(tensor)
            .ok_or_else(|| ModelError::MissingWeight(tensor.to_string()))?;
        Ok(&self.shards[*idx])
    }

    pub fn get_weight(&mut self, name: &str) -> Result<Vec<f32>, ModelError> {
        if let Some(v) = self.cache.get(name) {
            return Ok(v.clone());
        }
        let base = name.trim_end_matches(".weight");
        let bits = self
            .bits_map
            .get(base)
            .copied()
            .or_else(|| self.bits_map.get(name).copied())
            .unwrap_or(4);

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
                bits,
            )?
        } else {
            load_f32_tensor(file, name)?
        };
        self.cache.insert(name.to_string(), data.clone());
        Ok(data)
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
