//! Infrastructure tests: ops, safetensors parsing, config loading.

use core_lm::model_config::ModelConfig;
use core_lm::ops::{argmax, matvec, rmsnorm, silu, softmax};
use std::path::PathBuf;

#[test]
fn test_matvec() {
    let m = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![1.0, 1.0];
    let out = matvec(&m, 2, 2, &v).unwrap();
    assert_eq!(out, vec![3.0, 7.0]);
}

#[test]
fn test_matvec_shape_mismatch() {
    let m = vec![1.0, 2.0];
    let v = vec![1.0, 1.0, 1.0];
    assert!(matvec(&m, 1, 2, &v).is_err());
}

#[test]
fn test_rmsnorm() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let w = vec![1.0, 1.0, 1.0, 1.0];
    let out = rmsnorm(&x, &w, 1e-6).unwrap();
    assert_eq!(out.len(), 4);
    for v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_softmax_sums_to_one() {
    let x = vec![1.0, 2.0, 3.0];
    let out = softmax(&x).unwrap();
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax must sum to 1, got {}",
        sum
    );
}

#[test]
fn test_silu() {
    let out = silu(&[0.0]);
    assert!((out[0] - 0.0).abs() < 1e-6, "silu(0) should be 0");
}

#[test]
fn test_argmax() {
    let x = vec![1.0, 5.0, 3.0];
    assert_eq!(argmax(&x).unwrap(), 1);
}

#[test]
fn test_config_loads() {
    let path = PathBuf::from("/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit/config.json");
    if !path.exists() {
        println!("SKIP: config not found");
        return;
    }
    let cfg = ModelConfig::from_json_file(&path).unwrap();
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.layer_types.len(), 40);
    let full = cfg.full_attention_layers();
    assert_eq!(full.len(), 10, "should have 10 full attention layers");
    assert_eq!(full[0], 3);
    assert_eq!(full[9], 39);
    println!("Config OK: 40 layers, 10 full attention");
}
