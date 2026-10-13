//! Muse Glimmer weight manager and config.
//!
//! Pure transformer (52 layers, all standard attention).
//! Tensor names: `model.layers.{n}.self_attn.{q,k,v,o}_proj.{weight,scales,biases}`
//! No `language_model.` prefix (unlike Qwen).

use crate::optiq::{DequantError, dequantize, load_f32_tensor};
use crate::safetensors::{SafeTensorError, SafeTensorFile};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub enum GlimmerError {
    SafeTensor(SafeTensorError),
    Dequant(DequantError),
    Io(String),
    Parse(String),
    MissingWeight(String),
}

impl std::fmt::Display for GlimmerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlimmerError::SafeTensor(e) => write!(f, "Safetensor: {}", e),
            GlimmerError::Dequant(e) => write!(f, "Dequant: {}", e),
            GlimmerError::Io(s) => write!(f, "IO: {}", s),
            GlimmerError::Parse(s) => write!(f, "Parse: {}", s),
            GlimmerError::MissingWeight(s) => write!(f, "Missing weight: {}", s),
        }
    }
}

impl std::error::Error for GlimmerError {}

impl From<SafeTensorError> for GlimmerError {
    fn from(e: SafeTensorError) -> Self {
        GlimmerError::SafeTensor(e)
    }
}
impl From<DequantError> for GlimmerError {
    fn from(e: DequantError) -> Self {
        GlimmerError::Dequant(e)
    }
}

#[derive(Debug, Clone)]
pub struct GlimmerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub post_norm_eps: f32,
    pub rope_theta: f32,
    pub qk_scale_factor: f32,
    pub output_multiplier: f32,
    pub output_soft_cap_temp: f32,
    pub sliding_window: usize,
    pub no_rope_layers: Vec<i64>,
    pub layer_types: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    pub attn_scale: f32,
}

impl GlimmerConfig {
    pub fn from_json_file(path: &Path) -> Result<Self, GlimmerError> {
        let data = std::fs::read_to_string(path).map_err(|e| GlimmerError::Io(e.to_string()))?;
        let v: serde_json::Value =
            serde_json::from_str(&data).map_err(|e| GlimmerError::Parse(e.to_string()))?;
        let get_usize = |k: &str| -> Result<usize, GlimmerError> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| GlimmerError::Parse(format!("missing {}", k)))
        };
        let get_f32 = |k: &str| -> Result<f32, GlimmerError> {
            v.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| GlimmerError::Parse(format!("missing {}", k)))
        };
        let head_dim = get_usize("head_dim")?;
        let qk_scale_factor = get_f32("qk_scale_factor")?;
        let no_rope_layers: Vec<i64> = v
            .get("no_rope_layers")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
            .ok_or_else(|| GlimmerError::Parse("missing no_rope_layers".to_string()))?;
        let layer_types: Vec<String> = v
            .get("layer_types")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .ok_or_else(|| GlimmerError::Parse("missing layer_types".to_string()))?;
        let bos_token_id = v
            .get("bos_token_id")
            .and_then(|x| x.as_u64())
            .map(|x| x as u32)
            .unwrap_or(200000);
        let eos_token_ids: Vec<u32> = v
            .get("eos_token_id")
            .and_then(|x| {
                if let Some(arr) = x.as_array() {
                    Some(
                        arr.iter()
                            .filter_map(|e| e.as_u64().map(|n| n as u32))
                            .collect(),
                    )
                } else if let Some(n) = x.as_u64() {
                    Some(vec![n as u32])
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![200001]);
        Ok(Self {
            hidden_size: get_usize("hidden_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            num_key_value_heads: get_usize("num_key_value_heads")?,
            head_dim,
            intermediate_size: get_usize("intermediate_size")?,
            vocab_size: get_usize("vocab_size")?,
            rms_norm_eps: get_f32("rms_norm_eps")?,
            post_norm_eps: v
                .get("post_norm_eps")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-8),
            rope_theta: get_f32("rope_theta")?,
            qk_scale_factor,
            output_multiplier: get_f32("output_multiplier")?,
            output_soft_cap_temp: v
                .get("output_soft_cap_temp")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(20.0),
            sliding_window: get_usize("sliding_window")?,
            no_rope_layers,
            layer_types,
            bos_token_id,
            eos_token_ids,
            attn_scale: qk_scale_factor / (head_dim as f32),
        })
    }
    pub fn use_rope(&self, layer: usize) -> bool {
        self.no_rope_layers.get(layer).copied().unwrap_or(1) == 1
    }
    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .map(|s| s == "sliding_attention")
            .unwrap_or(false)
    }
}

pub struct WeightManager {
    shards: Vec<SafeTensorFile>,
    tensor_to_shard: HashMap<String, usize>,
    cache: HashMap<String, Vec<f32>>,
}

impl WeightManager {
    pub fn load(model_dir: &Path, index_path: &Path) -> Result<Self, GlimmerError> {
        let index_data =
            std::fs::read_to_string(index_path).map_err(|e| GlimmerError::Io(e.to_string()))?;
        let index: serde_json::Value =
            serde_json::from_str(&index_data).map_err(|e| GlimmerError::Parse(e.to_string()))?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| GlimmerError::Parse("no weight_map".to_string()))?;
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
        Ok(Self {
            shards,
            tensor_to_shard,
            cache: HashMap::new(),
        })
    }
    fn shard_for(&self, tensor: &str) -> Result<&SafeTensorFile, GlimmerError> {
        let idx = self
            .tensor_to_shard
            .get(tensor)
            .ok_or_else(|| GlimmerError::MissingWeight(tensor.to_string()))?;
        Ok(&self.shards[*idx])
    }
    pub fn get_weight(&mut self, name: &str) -> Result<Vec<f32>, GlimmerError> {
        if let Some(v) = self.cache.get(name) {
            return Ok(v.clone());
        }
        let base = name.trim_end_matches(".weight");
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
                4,
            )?
        } else {
            load_f32_tensor(file, name)?
        };
        self.cache.insert(name.to_string(), data.clone());
        Ok(data)
    }
    pub fn weight_shape(&self, name: &str) -> Result<(usize, usize), GlimmerError> {
        let file = self.shard_for(name)?;
        let info = file.tensor_info(name)?;
        if info.shape.len() != 2 {
            return Err(GlimmerError::Parse(format!(
                "expected 2D shape for {}, got {:?}",
                name, info.shape
            )));
        }
        Ok((info.shape[0], info.shape[1]))
    }
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::WeightManager;
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_tensor(path: &Path, name: &str, dtype: &str, payload: &[u8]) {
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[1,1],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(payload);
        fs::write(path, file).unwrap();
    }

    #[test]
    fn resolves_quantization_companions_from_separate_shards() {
        let root = std::env::temp_dir().join(format!(
            "core_lm_glimmer_split_shards_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir(&root).unwrap();
        write_tensor(
            &root.join("weight.safetensors"),
            "layer.weight",
            "U32",
            &0x7654_3210u32.to_le_bytes(),
        );
        write_tensor(
            &root.join("scales.safetensors"),
            "layer.scales",
            "BF16",
            &0x3f80u16.to_le_bytes(),
        );
        write_tensor(
            &root.join("biases.safetensors"),
            "layer.biases",
            "BF16",
            &0u16.to_le_bytes(),
        );
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"weight_map":{"layer.weight":"weight.safetensors","layer.scales":"scales.safetensors","layer.biases":"biases.safetensors"}}"#,
        )
        .unwrap();

        let mut manager =
            WeightManager::load(&root, &root.join("model.safetensors.index.json")).unwrap();
        let values = manager.get_weight("layer.weight").unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(values, (0..8).map(|value| value as f32).collect::<Vec<_>>());
    }
}
