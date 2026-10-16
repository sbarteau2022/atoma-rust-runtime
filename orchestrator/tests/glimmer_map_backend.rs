#![cfg(feature = "glimmer")]
use atoma_immune::{FrozenBaseWeights, RoleId};
use atoma_orchestrator::{
    MonsterHybridServer,
    glimmer_map::{CentralMapAttention, RecallConfig, RecallMode},
};
use core_lm::attention_backend::{AttentionBackend, AttentionInput};

fn server() -> MonsterHybridServer {
    MonsterHybridServer::boot_default(FrozenBaseWeights::freeze("fixture", 2, vec![0.5; 8]))
        .unwrap()
}
fn config(mode: RecallMode) -> RecallConfig {
    RecallConfig {
        mode,
        layers: [0].into(),
        gain: 1.0,
        candidate_limit: 2,
    }
}
fn input<'a>(
    pos: usize,
    start: usize,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
) -> AttentionInput<'a> {
    AttentionInput {
        layer: 0,
        token_id: 1,
        position: pos,
        window_start: start,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 8,
        dot_scale: 0.125,
        query: q,
        key: k,
        value: v,
    }
}

#[test]
fn real_backend_gqa_sliding_window_and_overwrite() {
    let mut s = server();
    let mut backend = CentralMapAttention::new(
        &mut s,
        RoleId::SensoryPerception,
        7,
        config(RecallMode::Baseline),
    )
    .unwrap();
    let q = vec![1.0; 32];
    let k = vec![1.0; 16];
    let v: Vec<_> = [vec![2.0; 8], vec![4.0; 8]].concat();
    let first = backend.attend(input(0, 0, &q, &k, &v)).unwrap();
    assert_eq!(first, [vec![2.0; 16], vec![4.0; 16]].concat());
    let next: Vec<_> = [vec![6.0; 8], vec![8.0; 8]].concat();
    let second = backend.attend(input(1, 0, &q, &k, &next)).unwrap();
    assert_eq!(second, [vec![4.0; 16], vec![6.0; 16]].concat());
    let window = backend.attend(input(2, 2, &q, &k, &next)).unwrap();
    assert_eq!(window, [vec![6.0; 16], vec![8.0; 16]].concat());
    let rewrite = backend.attend(input(2, 2, &q, &k, &v)).unwrap();
    assert_eq!(rewrite, first);
    assert_eq!(backend.indexed_slots(), 6);
}

#[test]
fn bounded_t2_and_direct_have_no_dot_products() {
    for mode in [RecallMode::MapT2, RecallMode::Direct] {
        let mut s = server();
        let mut backend =
            CentralMapAttention::new(&mut s, RoleId::SensoryPerception, 7, config(mode)).unwrap();
        let q = vec![1.0; 32];
        let k = vec![1.0; 16];
        let v = vec![3.0; 16];
        for pos in 0..12 {
            assert_eq!(
                backend.attend(input(pos, 0, &q, &k, &v)).unwrap(),
                vec![3.0; 32]
            );
        }
        assert_eq!(backend.metrics.dot_products, 0);
        assert!(backend.metrics.fetched_values <= 12 * 4 * 2);
        assert_eq!(
            backend.metrics.direct_reads,
            if mode == RecallMode::Direct { 48 } else { 0 }
        );
    }
}

#[test]
fn request_isolation_reconstruction_and_invalid_inputs() {
    let mut s = server();
    let q = vec![1.0; 32];
    let k = vec![1.0; 16];
    {
        let mut a = CentralMapAttention::new(
            &mut s,
            RoleId::SensoryPerception,
            10,
            config(RecallMode::Direct),
        )
        .unwrap();
        a.attend(input(0, 0, &q, &k, &[99.0; 16])).unwrap();
    }
    {
        let mut a = CentralMapAttention::new(
            &mut s,
            RoleId::SensoryPerception,
            11,
            config(RecallMode::Direct),
        )
        .unwrap();
        assert_eq!(a.indexed_slots(), 0);
        assert_eq!(
            a.attend(input(0, 0, &q, &k, &[2.0; 16])).unwrap(),
            vec![2.0; 32]
        );
        assert!(a.attend(input(1, 2, &q, &k, &[2.0; 16])).is_err());
        assert!(a.attend(input(1, 0, &q, &k, &[f32::NAN; 16])).is_err());
        assert_eq!(a.indexed_slots(), 2);
    }
    let a = CentralMapAttention::new(
        &mut s,
        RoleId::SensoryPerception,
        10,
        config(RecallMode::Direct),
    )
    .unwrap();
    assert_eq!(a.indexed_slots(), 2);
}

#[test]
fn unselected_layer_uses_dense_control() {
    let mut s = server();
    let mut cfg = config(RecallMode::Direct);
    cfg.layers.clear();
    let mut a = CentralMapAttention::new(&mut s, RoleId::SensoryPerception, 10, cfg).unwrap();
    let q = vec![1.0; 32];
    let k = vec![1.0; 16];
    a.attend(input(0, 0, &q, &k, &[1.0; 16])).unwrap();
    assert_eq!(
        a.attend(input(1, 0, &q, &k, &[3.0; 16])).unwrap(),
        vec![2.0; 32]
    );
    assert_eq!(a.metrics.direct_reads, 0);
    assert_eq!(a.metrics.dot_products, 12);
}

#[test]
fn glimmer_forward_routes_real_qkv_through_server_and_keeps_local_maps_empty() {
    use core_lm::glimmer_forward::GlimmerModel;
    use serde_json::json;
    let root = std::env::temp_dir().join(format!(
        "glimmer-map-forward-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let mut header = serde_json::Map::new();
    let mut weight_map = serde_json::Map::new();
    let mut data = Vec::new();
    let mut tensor = |name: String, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>| {
        let start = data.len();
        data.extend(bytes);
        header.insert(
            name.clone(),
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
        );
        weight_map.insert(name, json!("model.safetensors"));
    };
    for (name, rows, cols) in [
        ("model.embed_tokens.weight", 4, 8),
        ("lm_head.weight", 4, 8),
    ] {
        let values: Vec<u8> = (0..rows * cols)
            .flat_map(|i| ((i as f32 + 1.0) * 0.01).to_le_bytes())
            .collect();
        tensor(name.into(), "F32", vec![rows, cols], values);
    }
    tensor(
        "model.norm.weight".into(),
        "F32",
        vec![8],
        [1.0f32.to_le_bytes(); 8].concat(),
    );
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        for name in [
            "input_layernorm",
            "post_attn_norm",
            "post_attention_layernorm",
            "post_ffn_norm",
        ] {
            tensor(
                format!("{p}.{name}.weight"),
                "F32",
                vec![8],
                [1.0f32.to_le_bytes(); 8].concat(),
            );
        }
        tensor(
            format!("{p}.self_attn.q_proj.weight"),
            "U32",
            vec![8, 1],
            [0x12345678u32.to_le_bytes(); 8].concat(),
        );
        tensor(
            format!("{p}.self_attn.q_proj.scales"),
            "F32",
            vec![8, 1],
            [0.01f32.to_le_bytes(); 8].concat(),
        );
        tensor(
            format!("{p}.self_attn.q_proj.biases"),
            "F32",
            vec![8, 1],
            [0.0f32.to_le_bytes(); 8].concat(),
        );
        for (name, rows, cols) in [
            ("self_attn.k_proj", 4, 8),
            ("self_attn.v_proj", 4, 8),
            ("self_attn.o_proj", 8, 8),
            ("mlp.gate_proj", 8, 8),
            ("mlp.up_proj", 8, 8),
            ("mlp.down_proj", 8, 8),
        ] {
            let bytes: Vec<u8> = (0..rows * cols)
                .flat_map(|i| ((i % 7) as f32 * 0.01).to_le_bytes())
                .collect();
            tensor(format!("{p}.{name}.weight"), "F32", vec![rows, cols], bytes);
        }
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut packed = (header.len() as u64).to_le_bytes().to_vec();
    packed.extend(header);
    packed.extend(data);
    std::fs::write(root.join("model.safetensors"), packed).unwrap();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        json!({"weight_map":weight_map}).to_string(),
    )
    .unwrap();
    std::fs::write(root.join("config.json"),json!({"hidden_size":8,"num_hidden_layers":2,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"intermediate_size":8,
        "vocab_size":4,"rms_norm_eps":0.00001,"post_norm_eps":0.00001,"rope_theta":10000.0,
        "qk_scale_factor":2.0,"output_multiplier":1.0,"output_soft_cap_temp":0.0,
        "sliding_window":2,"no_rope_layers":[1],"layer_types":["sliding_attention","full_attention"],
        "bos_token_id":0,"eos_token_id":[3],"quantization":{"bits":4,"group_size":8}}).to_string()).unwrap();
    let mut model = GlimmerModel::load(&root).unwrap();
    let mut s = server();
    let cfg = RecallConfig {
        layers: [0, 1].into(),
        ..config(RecallMode::Direct)
    };
    let mut backend = CentralMapAttention::new(&mut s, RoleId::SensoryPerception, 1, cfg).unwrap();
    backend.capture_layer(0).unwrap();
    for pos in 0..3 {
        let logits = model.forward_with_backend(1, pos, &mut backend).unwrap();
        assert_eq!(logits.len(), 4);
        assert!(logits.iter().all(|x| x.is_finite()));
    }
    assert_eq!(backend.metrics.layer_calls, 6);
    assert_eq!(backend.metrics.direct_reads, 12);
    assert_eq!(backend.metrics.dot_products, 0);
    assert_eq!(backend.attention_trace.len(), 3);
    assert!(backend
        .attention_trace
        .iter()
        .all(|row| row.baseline_output.len() == 8 && row.query.len() == 8));
    assert!(model.geo_maps.iter().flatten().all(|m| m.atoms.is_empty()));
    drop(model);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn radial_toroidal_and_live_adapted_map_value_match_baseline_attention() {
    use atoma_orchestrator::toroidal_attention::LinearMapValueAdapter;

    let mut s = server();
    let q = vec![
        0.6f32, 0.8, -0.5, 0.5, 0.2, -0.9, 1.1, 0.3, // head 0
        -0.4, 0.7, 0.9, -0.2, 0.5, 0.5, -0.8, 0.1, // head 1
        0.3, -0.6, 0.4, 0.9, -0.7, 0.2, 0.5, -0.5, // head 2
        1.0, 0.0, 0.0, 1.0, -0.5, 0.5, 0.2, 0.8, // head 3
    ];
    let k0 = vec![
        0.5f32, 0.7, -0.4, 0.6, 0.1, -0.8, 0.9, 0.2, // kv 0
        -0.3, 0.6, 0.8, -0.1, 0.4, 0.4, -0.7, 0.2, // kv 1
    ];
    let v0 = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // kv 0
        2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0, // kv 1
    ];
    let k1 = vec![
        -0.5f32, 0.3, 0.7, -0.2, 0.4, 0.5, -0.3, 0.9, // kv 0
        0.6, -0.5, -0.2, 0.8, -0.4, 0.3, 0.9, -0.1, // kv 1
    ];
    let v1 = vec![
        8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, // kv 0
        7.0, 5.0, 3.0, 1.0, 8.0, 6.0, 4.0, 2.0, // kv 1
    ];

    let baseline_out = {
        let mut b = CentralMapAttention::new(
            &mut s,
            RoleId::SensoryPerception,
            90,
            config(RecallMode::Baseline),
        )
        .unwrap();
        b.attend(input(0, 0, &q, &k0, &v0)).unwrap();
        b.attend(input(1, 0, &q, &k1, &v1)).unwrap()
    };

    let radial_out = {
        let mut rt = CentralMapAttention::new(
            &mut s,
            RoleId::SensoryPerception,
            91,
            config(RecallMode::RadialToroidal),
        )
        .unwrap();
        rt.attend(input(0, 0, &q, &k0, &v0)).unwrap();
        let out = rt.attend(input(1, 0, &q, &k1, &v1)).unwrap();
        assert_eq!(rt.metrics.dot_products, 0);
        assert!(rt.metrics.phase_scores > 0);
        out
    };

    for (a, b) in baseline_out.iter().zip(&radial_out) {
        assert!((a - b).abs() < 1e-5, "radial toroidal {b} != baseline {a}");
    }

    // Build an identity projection on the radial-toroidal channel (3rd block of 8 dims)
    let head_dim = 8usize;
    let input_dim = head_dim * 3 + 1;
    let mut weights = vec![vec![0.0f64; head_dim]; input_dim];
    for d in 0..head_dim {
        weights[head_dim * 2 + d][d] = 1.0;
    }
    let adapter = LinearMapValueAdapter {
        input_dim,
        head_dim,
        weights,
        regularization: 1e-6,
    };

    let adapted_out = {
        let mut ad = CentralMapAttention::new(
            &mut s,
            RoleId::SensoryPerception,
            92,
            config(RecallMode::AdaptedMapValue),
        )
        .unwrap();
        ad.set_adapter(adapter).unwrap();
        ad.attend(input(0, 0, &q, &k0, &v0)).unwrap();
        let out = ad.attend(input(1, 0, &q, &k1, &v1)).unwrap();
        assert_eq!(ad.metrics.dot_products, 0);
        out
    };

    for (a, b) in baseline_out.iter().zip(&adapted_out) {
        assert!((a - b).abs() < 1e-5, "adapted output {b} != baseline {a}");
    }
}

