//! Model configuration. Parsed from config.json AT LOAD TIME ONLY.
//! The runtime path uses the binary ModelConfig struct, never touches JSON.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub layer_types: Vec<LayerType>,
    pub eos_token_ids: Vec<u32>,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

#[derive(Debug, Deserialize)]
struct RawTextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    rms_norm_eps: f32,
    layer_types: Vec<String>,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    #[serde(default)]
    eos_token_id: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct RawRopeParams {
    partial_rotary_factor: f32,
    rope_theta: f32,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    text_config: RawTextConfig,
}

impl ModelConfig {
    pub fn from_json_file(path: &Path) -> Result<Self, ConfigError> {
        let data = std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;
        let raw: RawConfig =
            serde_json::from_str(&data).map_err(|e| ConfigError::Parse(e.to_string()))?;
        let tc = raw.text_config;

        let layer_types = tc
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "linear_attention" => LayerType::LinearAttention,
                "full_attention" => LayerType::FullAttention,
                _ => LayerType::LinearAttention,
            })
            .collect();

        let eos_token_ids = match &tc.eos_token_id {
            serde_json::Value::Number(n) => vec![n.as_u64().unwrap_or(0) as u32],
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            _ => vec![],
        };

        Ok(Self {
            hidden_size: tc.hidden_size,
            num_hidden_layers: tc.num_hidden_layers,
            num_attention_heads: tc.num_attention_heads,
            num_key_value_heads: tc.num_key_value_heads,
            head_dim: tc.head_dim,
            vocab_size: tc.vocab_size,
            num_experts: tc.num_experts,
            num_experts_per_tok: tc.num_experts_per_tok,
            moe_intermediate_size: tc.moe_intermediate_size,
            shared_expert_intermediate_size: tc.shared_expert_intermediate_size,
            rms_norm_eps: tc.rms_norm_eps,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            layer_types,
            eos_token_ids,
            linear_key_head_dim: tc.linear_key_head_dim,
            linear_num_key_heads: tc.linear_num_key_heads,
            linear_num_value_heads: tc.linear_num_value_heads,
            linear_value_head_dim: tc.linear_value_head_dim,
            linear_conv_kernel_dim: tc.linear_conv_kernel_dim,
        })
    }

    pub fn full_attention_layers(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t, LayerType::FullAttention))
            .map(|(i, _)| i)
            .collect()
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io(String),
    Parse(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(s) => write!(f, "IO: {}", s),
            ConfigError::Parse(s) => write!(f, "Parse: {}", s),
        }
    }
}

impl std::error::Error for ConfigError {}
