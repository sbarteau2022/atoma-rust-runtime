//! Muse Glimmer weight manager and config.
//!
//! Pure transformer (52 layers, all standard attention).
//! Tensor names: `model.layers.{n}.self_attn.{q,k,v,o}_proj.{weight,scales,biases}`
//! No `language_model.` prefix (unlike Qwen).

use crate::metal::{accelerate_matvec_f32, qmatvec_fused_zero_copy};
use crate::optiq::{DequantError, dequantize, load_f32_tensor};
use crate::metal_quant::MetalQuantGemm;
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
    pub retain_cache: bool,
    default_bits: u32,
    default_group: usize,
    quant_overrides: HashMap<String, (u32, usize)>,
    metal_gemm: Option<MetalQuantGemm>,
    pub use_metal: bool,
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
        let (default_bits, default_group, quant_overrides) = read_quant_config(model_dir)?;
        // Try to initialize Metal for GPU-accelerated quantized GEMV.
        // Falls back to CPU if Metal is unavailable.
        let metal_gemm = Self::init_metal();
        let use_metal = metal_gemm.is_some();
        if use_metal {
            eprintln!("[WeightManager] Metal GPU acceleration enabled");
        } else {
            eprintln!("[WeightManager] Metal not available, using CPU");
        }
        Ok(Self {
            shards,
            tensor_to_shard,
            cache: HashMap::new(),
            retain_cache: false,
            default_bits,
            default_group,
            quant_overrides,
            metal_gemm,
            use_metal,
        })
    }
    /// Initialize Metal GPU for quantized GEMV. Returns None if unavailable.
    fn init_metal() -> Option<MetalQuantGemm> {
        // MSL source is embedded at compile time
        const MSL_SOURCE: &str = include_str!("quant_gemv.metal");
        match MetalQuantGemm::new(MSL_SOURCE) {
            Ok(gemm) => Some(gemm),
            Err(e) => {
                eprintln!("[WeightManager] Metal init failed: {}, using CPU", e);
                None
            }
        }
    }

    pub fn quant_defaults(&self) -> (u32, usize) {
        (self.default_bits, self.default_group)
    }
    pub fn quant_override_count(&self) -> usize {
        self.quant_overrides.len()
    }
    fn quant_for(&self, base: &str) -> (u32, usize) {
        self.quant_overrides
            .get(base)
            .copied()
            .unwrap_or((self.default_bits, self.default_group))
    }
    pub fn quant_bits(&self, name: &str) -> u32 {
        self.quant_for(name.trim_end_matches(".weight")).0
    }
    fn shard_for(&self, tensor: &str) -> Result<&SafeTensorFile, GlimmerError> {
        let idx = self
            .tensor_to_shard
            .get(tensor)
            .ok_or_else(|| GlimmerError::MissingWeight(tensor.to_string()))?;
        Ok(&self.shards[*idx])
    }
    pub fn get_weight(&mut self, name: &str) -> Result<Vec<f32>, GlimmerError> {
        if self.retain_cache {
            if let Some(v) = self.cache.get(name) {
                return Ok(v.clone());
            }
        }
        let base = name.trim_end_matches(".weight");
        let scales_name = format!("{}.scales", base);
        let biases_name = format!("{}.biases", base);
        let has_scales = self.tensor_to_shard.contains_key(&scales_name);
        let (rows, mut data) = if has_scales {
            let (bits, group_size) = self.quant_for(base);
            let file = self.shard_for(name)?;
            let info = file.tensor_info(name)?;
            let rows = *info.shape.first().unwrap_or(&1);
            let scales_file = self.shard_for(&scales_name)?;
            let biases_file = self.shard_for(&biases_name)?;
            let values = dequantize(
                file.tensor_data(name)?,
                info,
                scales_file.tensor_data(&scales_name)?,
                scales_file.tensor_info(&scales_name)?,
                biases_file.tensor_data(&biases_name)?,
                biases_file.tensor_info(&biases_name)?,
                bits,
                group_size,
            )?;
            (rows, values)
        } else {
            let file = self.shard_for(name)?;
            (1, load_f32_tensor(file, name)?)
        };
        if has_scales {
            self.apply_out_scale(base, rows, &mut data)?;
        }
        if self.retain_cache {
            self.cache.insert(name.to_string(), data.clone());
        }
        Ok(data)
    }
    fn apply_out_scale(&self, base: &str, rows: usize, data: &mut [f32]) -> Result<(), GlimmerError> {
        let name = format!("{base}.out_scale");
        if !self.tensor_to_shard.contains_key(&name) {
            return Ok(());
        }
        let file = self.shard_for(&name)?;
        let scale = load_f32_tensor(file, &name)?;
        if rows == 0 || scale.len() != rows || data.len() % rows != 0 {
            return Err(GlimmerError::Parse(format!(
                "{name} length {} does not match {} rows of {}",
                scale.len(),
                rows,
                data.len()
            )));
        }
        let cols = data.len() / rows;
        for row in 0..rows {
            let factor = scale[row];
            if factor == 1.0 {
                continue;
            }
            for value in &mut data[row * cols..(row + 1) * cols] {
                *value *= factor;
            }
        }
        Ok(())
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

    /// Zero-copy fused matrix-vector multiply (`y = W * x`) directly over memory-mapped
    /// `SafeTensorFile` shards (`core_lm::metal::qmatvec_fused_zero_copy`).
    ///
    /// Why this eliminates the 87-sec/token CPU bottleneck:
    /// When `retain_cache == false` (required for 27B/35B models whose full f32 weights exceed RAM),
    /// calling `get_weight()` allocated ~108 GB of temporary `Vec<f32>` buffers per token.
    /// `matvec_weight` computes `W * x` in-place directly from the mmap'd `U32`/`BF16` slices with
    /// zero weight-matrix heap allocations and precomputed group sums `X_group[g]`.
    /// Try Metal GPU quantized GEMV. Returns None if Metal unavailable or scales not F32.
    /// Falls back to CPU on any error.
    #[allow(clippy::too_many_arguments)]
    fn try_metal_gemv(
        &self,
        w_data: &[u8],
        w_dtype: &str,
        s_data: &[u8],
        s_dtype: &str,
        b_data: &[u8],
        b_dtype: &str,
        out_scale: Option<&[f32]>,
        bits: u32,
        group_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> Option<Vec<f32>> {
        let gemm = self.metal_gemm.as_ref()?;
        
        // Only F32 scales/biases supported in Metal kernel currently
        if s_dtype != "F32" || b_dtype != "F32" {
            return None;
        }
        if w_dtype != "U32" {
            return None;
        }
        
        // Convert w_data bytes to u32 slice
        if w_data.len() % 4 != 0 {
            return None;
        }
        let packed: &[u32] = unsafe {
            std::slice::from_raw_parts(
                w_data.as_ptr() as *const u32,
                w_data.len() / 4,
            )
        };
        
        let result = if bits == 4 {
            gemm.q4_gemv(
                packed, s_data, b_data, out_scale, x,
                rows as u32, cols as u32, group_size as u32,
            )
        } else if bits == 6 {
            gemm.q6_gemv(
                packed, s_data, b_data, out_scale, x,
                rows as u32, cols as u32, group_size as u32,
            )
        } else {
            return None; // Unsupported bits for Metal
        };
        
        match result {
            Ok(out) => Some(out),
            Err(e) => {
                eprintln!("[WeightManager] Metal GEMV failed: {}, falling back to CPU", e);
                None
            }
        }
    }

    pub fn matvec_weight(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, GlimmerError> {
        if x.len() != cols {
            return Err(GlimmerError::Parse(format!(
                "matvec_weight({name}): x len {} != cols {cols}",
                x.len()
            )));
        }
        if let Some(cached) = self.cache.get(name) {
            if cached.len() != rows * cols {
                return Err(GlimmerError::Parse(format!(
                    "cached weight {name} len {} != {rows}x{cols}",
                    cached.len()
                )));
            }
            return Ok(accelerate_matvec_f32(cached, rows, cols, x));
        }

        let base = name.trim_end_matches(".weight");
        let scales_name = format!("{}.scales", base);
        let biases_name = format!("{}.biases", base);
        let has_scales = self.tensor_to_shard.contains_key(&scales_name);

        if has_scales && !self.retain_cache {
            let (bits, group_size) = self.quant_for(base);
            let out_scale_name = format!("{base}.out_scale");
            let out_scale_vec = if self.tensor_to_shard.contains_key(&out_scale_name) {
                let os_file = self.shard_for(&out_scale_name)?;
                Some(load_f32_tensor(os_file, &out_scale_name)?)
            } else {
                None
            };
            let w_file = self.shard_for(name)?;
            let w_info = w_file.tensor_info(name)?;
            let w_data = w_file.tensor_data(name)?;
            let s_file = self.shard_for(&scales_name)?;
            let s_info = s_file.tensor_info(&scales_name)?;
            let s_data = s_file.tensor_data(&scales_name)?;
            let b_file = self.shard_for(&biases_name)?;
            let b_info = b_file.tensor_info(&biases_name)?;
            let b_data = b_file.tensor_data(&biases_name)?;

            // Try Metal GPU first (if available and scales are F32)
            if self.use_metal {
                if let Some(metal_out) = self.try_metal_gemv(
                    w_data, &w_info.dtype,
                    s_data, &s_info.dtype,
                    b_data, &b_info.dtype,
                    out_scale_vec.as_deref(),
                    bits, group_size, rows, cols, x,
                ) {
                    return Ok(metal_out);
                }
                // Fall through to CPU on Metal failure
            }

            return Ok(qmatvec_fused_zero_copy(
                w_data,
                &w_info.dtype,
                s_data,
                &s_info.dtype,
                b_data,
                &b_info.dtype,
                out_scale_vec.as_deref(),
                bits,
                group_size,
                rows,
                cols,
                x,
            )
            .map_err(DequantError::ShapeMismatch)?);
        }

        let w = self.get_weight(name)?;
        if w.len() != rows * cols {
            return Err(GlimmerError::Parse(format!(
                "weight {name} len {} != {rows}x{cols}",
                w.len()
            )));
        }
        Ok(accelerate_matvec_f32(&w, rows, cols, x))
    }

    /// Computes a deterministic 64-bit structural fingerprint of the loaded shard layout
    /// so `ImmuneForwardGate` can verify base-weight immutability on every forward pass.
    pub fn structural_fingerprint(&self) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        h ^= self.shards.len() as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
        h ^= self.tensor_to_shard.len() as u64;
        h = h.wrapping_mul(0x0100_0000_01B3);
        h ^= (self.default_bits as u64) << 32 | (self.default_group as u64);
        h.wrapping_mul(0x0100_0000_01B3)
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

fn read_quant_config(
    model_dir: &Path,
) -> Result<(u32, usize, HashMap<String, (u32, usize)>), GlimmerError> {
    let mut default_bits = 4u32;
    let mut default_group = 64usize;
    let mut overrides = HashMap::new();
    let path = model_dir.join("config.json");
    if !path.exists() {
        return Ok((default_bits, default_group, overrides));
    }
    let data = std::fs::read_to_string(&path).map_err(|e| GlimmerError::Io(e.to_string()))?;
    let v: serde_json::Value =
        serde_json::from_str(&data).map_err(|e| GlimmerError::Parse(e.to_string()))?;
    let Some(quant) = v.get("quantization").and_then(|value| value.as_object()) else {
        return Ok((default_bits, default_group, overrides));
    };
    if let Some(bits) = quant.get("bits").and_then(|value| value.as_u64()) {
        default_bits = bits as u32;
    }
    if let Some(group) = quant.get("group_size").and_then(|value| value.as_u64()) {
        default_group = group as usize;
    }
    if let Some(mode) = quant.get("mode").and_then(|value| value.as_str()) {
        if mode != "affine" {
            return Err(GlimmerError::Parse(format!(
                "quantization mode {mode} is not affine"
            )));
        }
    }
    for (key, value) in quant {
        if key == "bits" || key == "group_size" || key == "mode" {
            continue;
        }
        let spec = value.as_object().ok_or_else(|| {
            GlimmerError::Parse(format!("quantization entry {key} is not an object"))
        })?;
        let bits = spec
            .get("bits")
            .and_then(|item| item.as_u64())
            .ok_or_else(|| GlimmerError::Parse(format!("quantization entry {key} has no bits")))?
            as u32;
        let group = spec
            .get("group_size")
            .and_then(|item| item.as_u64())
            .ok_or_else(|| {
                GlimmerError::Parse(format!("quantization entry {key} has no group_size"))
            })? as usize;
        if let Some(mode) = spec.get("mode").and_then(|item| item.as_str()) {
            if mode != "affine" {
                return Err(GlimmerError::Parse(format!(
                    "quantization entry {key} mode {mode} is not affine"
                )));
            }
        }
        if bits == 0 || group == 0 {
            return Err(GlimmerError::Parse(format!(
                "quantization entry {key} has bits {bits} group {group}"
            )));
        }
        overrides.insert(key.clone(), (bits, group));
    }
    if default_bits == 0 || default_group == 0 {
        return Err(GlimmerError::Parse(format!(
            "quantization default bits {default_bits} group {default_group}"
        )));
    }
    Ok((default_bits, default_group, overrides))
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

    #[test]
    fn applies_config_group_size_and_out_scale() {
        let root = std::env::temp_dir().join(format!(
            "core_lm_glimmer_quant_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir(&root).unwrap();
        let header = r#"{"layer.weight":{"dtype":"U32","shape":[1,1],"data_offsets":[0,4]},"layer.scales":{"dtype":"BF16","shape":[1,2],"data_offsets":[4,8]},"layer.biases":{"dtype":"BF16","shape":[1,2],"data_offsets":[8,12]},"layer.out_scale":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}}"#;
        let mut packed = (header.len() as u64).to_le_bytes().to_vec();
        packed.extend_from_slice(header.as_bytes());
        packed.extend_from_slice(&0x7654_3210u32.to_le_bytes());
        packed.extend_from_slice(&0x4000_3f80u32.to_le_bytes());
        packed.extend_from_slice(&0u32.to_le_bytes());
        packed.extend_from_slice(&0.5f32.to_le_bytes());
        fs::write(root.join("model.safetensors"), packed).unwrap();
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"weight_map":{"layer.weight":"model.safetensors","layer.scales":"model.safetensors","layer.biases":"model.safetensors","layer.out_scale":"model.safetensors"}}"#,
        )
        .unwrap();
        fs::write(
            root.join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64,"mode":"affine","layer":{"bits":4,"group_size":4,"mode":"affine"}}}"#,
        )
        .unwrap();

        let mut manager =
            WeightManager::load(&root, &root.join("model.safetensors.index.json")).unwrap();
        assert_eq!(manager.quant_defaults(), (4, 64));
        assert_eq!(manager.quant_override_count(), 1);
        assert_eq!(manager.quant_bits("layer.weight"), 4);
        let values = manager.get_weight("layer.weight").unwrap();
        fs::remove_dir_all(&root).unwrap();

        assert_eq!(values, vec![0.0, 0.5, 1.0, 1.5, 4.0, 5.0, 6.0, 7.0]);
    }
}
