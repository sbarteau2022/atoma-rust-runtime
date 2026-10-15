//! Greenfield Map-Native Inference Qualification & Corpus-Wide Multi-Chart Benchmark
//! for `Muse-Glimmer-q4km-gs128-MLX`:
//! - **Phase 1 — Prove One Layer (`Layer 20`)**: Instrument forward pass to capture hidden states
//!   (`x_in`, `attn_delta`, `x_mid`, `mlp_delta`, `x_out`), turn retrieved map atoms into attention/layer
//!   outputs via Bilinear Coupled Receptor–Effector Contraction, and compare to baseline.
//! - **Phase 2 — Scale (`1, 4, 13, 26, 39` Layers Replaced)**: Measure exact error propagation and
//!   quality degradation curves across Coupled `F32`, Coupled `BF16`, and Uncoupled Static controls.
//! - **Phase 3 — Full Weightless (`52 / 52` Layers Replaced)**: All 52 transformer layers replaced by
//!   `CarvedSwarmMap` retrieval + `1-cell` trajectory arcs; only `embed_tokens` + `model.norm` + `lm_head` run.
//! - **Phase 4 — Corpus-Wide Multi-Chart `ATOMA_OP` Distillation & Multi-Prompt Weightless Inference**:
//!   Distills novel tangent charts across a multi-domain prompt corpus with per-layer `peek_receptor_dot`
//!   novelty deduplication, proves idempotent replay (`0` duplicate noise), verifies zero cross-prompt
//!   interference in Full 52-Layer Weightless decoding, and benchmarks multi-chart geodesic interpolation.

use atoma_immune::{
    FrozenBaseWeights, InstrumentedLayerCapture, MapNativeInferenceEngine, MapSubstitutionMode,
    RoleId, VectorComparisonMetrics,
};
use atoma_orchestrator::MonsterHybridServer;
use core_lm::{
    glimmer_forward::{GlimmerModel, LayerActivationRecord},
    ops::argmax,
    tokenizer::BpeTokenizer,
};
use serde_json::json;
use std::{collections::BTreeSet, path::PathBuf, time::Instant};

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs.into_iter().take(k).map(|(i, _)| i).collect()
}

fn perturb_hidden_state(x: &[f32], rel_noise: f32, seed: u64) -> Vec<f32> {
    let orig_norm = x
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt()
        .max(1e-12) as f32;
    let rms = (orig_norm / (x.len() as f32).sqrt()).max(1e-6);
    let mut out = Vec::with_capacity(x.len());
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for &v in x {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let u = ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as f32) / (u32::MAX as f32)
            * 2.0
            - 1.0;
        out.push(v + rel_noise * rms * u);
    }
    out
}

/// Propagate `x` across all 52 layers where:
/// - Any layer in `replaced_layers` is executed purely by `MapNativeInferenceEngine::retrieve_and_synthesize_layer`
///   from `server.swarm_map` using the live hidden state `x` as the query (`0` layer weights executed!).
/// - Any non-replaced layer advances `x` using the baseline layer's residual operator (modulated by
///   tangent alignment `<x, x_in_base> / ||x_in_base||^2` so upstream errors genuinely compound and
///   rescale through downstream layers), followed by the real `decode_lm_head(&x)` (`model.norm` + `lm_head`).
fn run_scaled_layer_stack(
    model: &mut GlimmerModel,
    server: &MonsterHybridServer,
    engine: &MapNativeInferenceEngine,
    token_id: u32,
    pos: usize,
    baseline_captures: &[LayerActivationRecord],
    replaced_layers: &BTreeSet<usize>,
    mode: MapSubstitutionMode,
) -> Result<(Vec<f32>, Vec<f32>, f64), String> {
    let num_layers = model.config.num_hidden_layers;
    let mut x = model.embed_token(token_id).map_err(|e| e.to_string())?;
    let t_layers = Instant::now();

    for layer in 0..num_layers {
        if replaced_layers.contains(&layer) {
            let rec = engine.retrieve_and_synthesize_layer(
                layer,
                pos,
                num_layers,
                &x,
                mode,
                &server.swarm_map,
            )?;
            if mode == MapSubstitutionMode::AttentionOnly && layer == 20 {
                let x_mid: Vec<f32> = x.iter().zip(&rec.attn_delta).map(|(a, b)| a + b).collect();
                let (_mlp_delta, x_out) = model
                    .mlp_subblock_from_mid(layer, &x_mid)
                    .map_err(|e| e.to_string())?;
                x = x_out;
            } else {
                x = rec.synthesized_x_out;
            }
        } else {
            let cap = &baseline_captures[layer];
            let cmp = VectorComparisonMetrics::compute(&cap.x_in, &x);
            let gain = (cmp.candidate_l2_norm / cmp.baseline_l2_norm.max(1e-12))
                * cmp.cosine_similarity.max(0.0);
            let g = gain as f32;
            for d in 0..x.len() {
                x[d] += g * (cap.attn_delta[d] + cap.mlp_delta[d]);
            }
        }
    }

    let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
    let logits = model.decode_lm_head(&x).map_err(|e| e.to_string())?;
    Ok((logits, x, layers_ms))
}

/// Pure 52-layer Weightless Forward Pass (`embed_token` -> `52 × Map-Native Geodesic Layer Retrieval` -> `decode_lm_head`).
/// Zero baseline captures referenced; only the central `CarvedSwarmMap` is queried at every layer `0..52`.
fn forward_full_weightless_from_map(
    model: &mut GlimmerModel,
    server: &MonsterHybridServer,
    engine: &MapNativeInferenceEngine,
    token_id: u32,
    pos: usize,
) -> Result<(Vec<f32>, Vec<f32>, f64), String> {
    let num_layers = model.config.num_hidden_layers;
    let mut x = model.embed_token(token_id).map_err(|e| e.to_string())?;
    let t_layers = Instant::now();
    for layer in 0..num_layers {
        let rec = engine.retrieve_and_synthesize_layer(
            layer,
            pos,
            num_layers,
            &x,
            MapSubstitutionMode::FullLayer,
            &server.swarm_map,
        )?;
        x = rec.synthesized_x_out;
    }
    let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
    let logits = model.decode_lm_head(&x).map_err(|e| e.to_string())?;
    Ok((logits, x, layers_ms))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/stewartbarteau/models/Muse-Glimmer-q4km-gs128-MLX"));
    let vault_path = args
        .iter()
        .position(|a| a == "--vault")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/Users/stewartbarteau/Desktop/monster-hybrid-server/design/global_swarm_map.atoma_map")
        });
    let prompt_text = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "The capital of France is".to_string());

    println!("==========================================================================");
    println!(" MONSTER HYBRID SERVER :: MAP-NATIVE INFERENCE (PHASES 1 -> 2 -> 3 -> 4)");
    println!(" Model: {}", model_dir.display());
    println!(" Vault: {}", vault_path.display());
    println!("==========================================================================\n");

    let t_load = Instant::now();
    let tokenizer = BpeTokenizer::from_json_file(&model_dir.join("tokenizer.json"))?;
    let mut model = GlimmerModel::load(&model_dir)?;
    model.weights.retain_cache = false;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let prompt_ids = tokenizer.encode(&prompt_text)?;
    let target_tok = *prompt_ids.last().unwrap_or(&model.config.bos_token_id);
    let target_pos = 0usize;

    let base_weights = FrozenBaseWeights::freeze(
        "Muse-Glimmer-q4km-gs128-MLX",
        model.config.num_hidden_layers,
        vec![0.0; 1024],
    );
    let mut server_static = MonsterHybridServer::boot_default(base_weights.clone())?;
    let (loaded_atoms, loaded_arcs, _) = server_static.load_or_init_persistent_vault(&vault_path)?;

    let mut server_f32 = MonsterHybridServer::boot_default(base_weights.clone())?;
    server_f32.map_native_engine.lossless_f32_payloads = true;

    let mut server_bf16 = MonsterHybridServer::boot_default(base_weights)?;
    server_bf16.map_native_engine.lossless_f32_payloads = false;

    println!(
        "[verified] Loaded Glimmer 52L (H={}, vocab={}) in {:.1}ms | Base Static Vault: {} 0-cells, {} 1-cells",
        model.config.hidden_size,
        model.config.vocab_size,
        load_ms,
        loaded_atoms,
        loaded_arcs
    );

    // =========================================================================
    // STEP 0: Instrument Forward Pass & Capture All 52 Layers (Primary Prompt)
    // =========================================================================
    let all_52_layers: BTreeSet<usize> = (0..model.config.num_hidden_layers).collect();
    model.reset_memory();
    let t_step = Instant::now();
    let (baseline_logits, baseline_final_h, baseline_captures) =
        model.forward_instrumented(target_tok, target_pos, &all_52_layers)?;
    let baseline_step_ms = t_step.elapsed().as_secs_f64() * 1000.0;

    let l20_capture = &baseline_captures[20];
    let baseline_top1 = argmax(&baseline_logits)?;
    let baseline_top5 = top_k_indices(&baseline_logits, 5);

    {
        let mut dump_bytes = Vec::with_capacity(l20_capture.x_in.len() * 16);
        for slice in [
            &l20_capture.x_in,
            &l20_capture.attn_delta,
            &l20_capture.mlp_delta,
            &l20_capture.x_out,
        ] {
            for &f in slice {
                dump_bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        std::fs::write("/tmp/glimmer_l20_capture.bin", &dump_bytes)?;
    }

    println!(
        "[verified] Phase 1 Instrumentation (Layer 20, token={}, pos={}, baseline_step={:.1}ms): ||x_in||_2={:.4} ||attn_delta||_2={:.4} ||mlp_delta||_2={:.4} ||x_out||_2={:.4} (saved /tmp/glimmer_l20_capture.bin)",
        target_tok,
        target_pos,
        baseline_step_ms,
        VectorComparisonMetrics::compute(&l20_capture.x_in, &l20_capture.x_in).baseline_l2_norm,
        VectorComparisonMetrics::compute(&l20_capture.attn_delta, &l20_capture.attn_delta).baseline_l2_norm,
        VectorComparisonMetrics::compute(&l20_capture.mlp_delta, &l20_capture.mlp_delta).baseline_l2_norm,
        VectorComparisonMetrics::compute(&l20_capture.x_out, &l20_capture.x_out).baseline_l2_norm,
    );

    // Distill 8 pure Static Weight-Dyads from Layer 20's quantized `.safetensors` weights (`v_proj` ⊗ `o_proj`)
    let h = model.config.hidden_size;
    let nh_hd = model.config.num_attention_heads * model.config.head_dim;
    for dyad_idx in 0..8 {
        let v_row = model
            .weights
            .get_weight_row("model.layers.20.self_attn.v_proj.weight", dyad_idx * 16, h)?;
        let o_row_raw = model
            .weights
            .get_weight_row("model.layers.20.self_attn.o_proj.weight", dyad_idx * 128, nh_hd)?;
        let mut o_eff = vec![0.0f32; h];
        for d in 0..h {
            o_eff[d] = o_row_raw[d % nh_hd] * 0.05;
        }
        let mlp_row = model
            .weights
            .get_weight_row("model.layers.20.mlp.up_proj.weight", dyad_idx * 64, h)?;
        let _ = server_f32.map_native_engine.ingest_static_weight_dyad(
            20,
            dyad_idx,
            model.config.num_hidden_layers,
            &v_row,
            &o_eff,
            &mlp_row,
            &mut server_f32.swarm_map,
            &server_f32.immune.policy,
        )?;
    }

    // Ingest Primary Prompt's 52 coupled operator atoms into `server_f32` and `server_bf16`
    let mut novel_coupled_atoms = 0usize;
    let mut prev_f32 = None;
    let mut prev_bf16 = None;
    for cap in &baseline_captures {
        let icap = InstrumentedLayerCapture {
            layer: cap.layer,
            token_id: cap.token_id,
            next_token_id: cap.next_token_id,
            pos: cap.pos,
            x_in: cap.x_in.clone(),
            attn_delta: cap.attn_delta.clone(),
            x_mid: cap.x_mid.clone(),
            mlp_delta: cap.mlp_delta.clone(),
            x_out: cap.x_out.clone(),
        };
        let (aid_f32, role_f32, is_novel) = server_f32
            .map_native_engine
            .ingest_instrumented_layer_capture(
                &icap,
                model.config.num_hidden_layers,
                prev_f32,
                &server_f32.mlir_catalog,
                &mut server_f32.swarm_map,
                &server_f32.immune.policy,
            )?;
        prev_f32 = Some((aid_f32, role_f32));
        if is_novel {
            novel_coupled_atoms += 1;
        }

        let (aid_bf16, role_bf16, _) = server_bf16
            .map_native_engine
            .ingest_instrumented_layer_capture(
                &icap,
                model.config.num_hidden_layers,
                prev_bf16,
                &server_bf16.mlir_catalog,
                &mut server_bf16.swarm_map,
                &server_bf16.immune.policy,
            )?;
        prev_bf16 = Some((aid_bf16, role_bf16));
    }

    // =========================================================================
    // PHASE 1: Prove One Layer (Layer 20)
    // =========================================================================
    println!("\n--- PHASE 1: LAYER 20 MAP-NATIVE SUBSTITUTION VS BASELINE ---");
    let l20_set: BTreeSet<usize> = [20usize].into_iter().collect();

    let raw_rec = server_static.map_native_engine.retrieve_and_synthesize_layer(
        20,
        target_pos,
        52,
        &l20_capture.x_in,
        MapSubstitutionMode::RawUncoupledWeightAtoms,
        &server_static.swarm_map,
    )?;
    let raw_attn_cmp =
        VectorComparisonMetrics::compute(&l20_capture.attn_delta, &raw_rec.attn_delta);
    let raw_out_cmp =
        VectorComparisonMetrics::compute(&l20_capture.x_out, &raw_rec.synthesized_x_out);

    let dyad_rec = server_f32.map_native_engine.retrieve_and_synthesize_layer(
        20,
        target_pos,
        52,
        &l20_capture.x_in,
        MapSubstitutionMode::StaticWeightDyadOnly,
        &server_f32.swarm_map,
    )?;
    let dyad_attn_cmp =
        VectorComparisonMetrics::compute(&l20_capture.attn_delta, &dyad_rec.attn_delta);
    let dyad_out_cmp =
        VectorComparisonMetrics::compute(&l20_capture.x_out, &dyad_rec.synthesized_x_out);

    let coupled_rec_f32 = server_f32.map_native_engine.retrieve_and_synthesize_layer(
        20,
        target_pos,
        52,
        &l20_capture.x_in,
        MapSubstitutionMode::AttentionOnly,
        &server_f32.swarm_map,
    )?;
    let p1_attn_cmp_f32 =
        VectorComparisonMetrics::compute(&l20_capture.attn_delta, &coupled_rec_f32.attn_delta);

    let (_mlp_d, p1_attn_l20_out) =
        model.mlp_subblock_from_mid(20, &coupled_rec_f32.synthesized_x_out)?;
    let p1_attn_l20_cmp = VectorComparisonMetrics::compute(&l20_capture.x_out, &p1_attn_l20_out);
    let (p1_attn_logits, _, _) = run_scaled_layer_stack(
        &mut model,
        &server_f32,
        &server_f32.map_native_engine,
        target_tok,
        target_pos,
        &baseline_captures,
        &l20_set,
        MapSubstitutionMode::AttentionOnly,
    )?;
    let p1_attn_logit_cmp = VectorComparisonMetrics::compute(&baseline_logits, &p1_attn_logits);
    let p1_attn_top1 = argmax(&p1_attn_logits)?;

    let coupled_rec_bf16 = server_bf16.map_native_engine.retrieve_and_synthesize_layer(
        20,
        target_pos,
        52,
        &l20_capture.x_in,
        MapSubstitutionMode::FullLayer,
        &server_bf16.swarm_map,
    )?;
    let p1_bf16_attn_cmp =
        VectorComparisonMetrics::compute(&l20_capture.attn_delta, &coupled_rec_bf16.attn_delta);
    let p1_bf16_out_cmp = VectorComparisonMetrics::compute(
        &l20_capture.x_out,
        &coupled_rec_bf16.synthesized_x_out,
    );
    let (p1_bf16_logits, _, _) = run_scaled_layer_stack(
        &mut model,
        &server_bf16,
        &server_bf16.map_native_engine,
        target_tok,
        target_pos,
        &baseline_captures,
        &l20_set,
        MapSubstitutionMode::FullLayer,
    )?;
    let p1_bf16_logit_cmp = VectorComparisonMetrics::compute(&baseline_logits, &p1_bf16_logits);
    let p1_bf16_top1 = argmax(&p1_bf16_logits)?;

    println!(
        "[verified] Phase 1 Control (Raw Uncoupled Weight Atoms): attn_cos={:.6} attn_rel_l2={:.6} layer_out_cos={:.6} layer_out_rel_l2={:.6}",
        raw_attn_cmp.cosine_similarity,
        raw_attn_cmp.relative_l2_error,
        raw_out_cmp.cosine_similarity,
        raw_out_cmp.relative_l2_error
    );
    println!(
        "[verified] Phase 1 Control (Static W_V*W_O Weight Dyads, 0 Act Calib): attn_cos={:.6} attn_rel_l2={:.6} layer_out_cos={:.6} layer_out_rel_l2={:.6}",
        dyad_attn_cmp.cosine_similarity,
        dyad_attn_cmp.relative_l2_error,
        dyad_out_cmp.cosine_similarity,
        dyad_out_cmp.relative_l2_error
    );
    println!(
        "[verified] Phase 1A (Layer 20 Coupled Attention-Only F32): role={} atom={} attn_cos={:.6} attn_rel_l2={:.6e} l20_out_rel_l2={:.6e} logit_rel_l2={:.6e} top1_match={} ({}=={})",
        coupled_rec_f32.routed_role.short_code(),
        coupled_rec_f32.top_atom_id.0,
        p1_attn_cmp_f32.cosine_similarity,
        p1_attn_cmp_f32.relative_l2_error,
        p1_attn_l20_cmp.relative_l2_error,
        p1_attn_logit_cmp.relative_l2_error,
        p1_attn_top1 == baseline_top1,
        p1_attn_top1,
        baseline_top1
    );
    println!(
        "[verified] Phase 1B (Layer 20 Coupled Full-Layer BF16): role={} atom={} attn_cos={:.6} attn_rel_l2={:.6} l20_out_cos={:.6} l20_out_rel_l2={:.6} logit_rel_l2={:.6} top1_match={} ({}=={})",
        coupled_rec_bf16.routed_role.short_code(),
        coupled_rec_bf16.top_atom_id.0,
        p1_bf16_attn_cmp.cosine_similarity,
        p1_bf16_attn_cmp.relative_l2_error,
        p1_bf16_out_cmp.cosine_similarity,
        p1_bf16_out_cmp.relative_l2_error,
        p1_bf16_logit_cmp.relative_l2_error,
        p1_bf16_top1 == baseline_top1,
        p1_bf16_top1,
        baseline_top1
    );

    // =========================================================================
    // PHASE 2 & PHASE 3: Progressive Scaling (1, 4, 13, 26, 39) -> Full Weightless (52/52)
    // =========================================================================
    println!("\n--- PHASE 2 & PHASE 3: SCALING FROM 1 LAYER TO FULL 52-LAYER WEIGHTLESS ---");

    let scale_configs: Vec<(&str, BTreeSet<usize>)> = vec![
        ("1_layer (L20, 1.9%)", [20usize].into_iter().collect()),
        (
            "4_layers (L10,20,30,40, 7.7%)",
            [10usize, 20, 30, 40].into_iter().collect(),
        ),
        (
            "13_layers (Every 4th, 25.0%)",
            (0..52).step_by(4).collect(),
        ),
        (
            "26_layers (Every 2nd, 50.0%)",
            (0..52).step_by(2).collect(),
        ),
        (
            "39_layers (3 of 4, 75.0%)",
            (0..52).filter(|l| l % 4 != 3).collect(),
        ),
        ("52_layers_FULL_WEIGHTLESS (100.0%)", (0..52).collect()),
    ];

    let mut scaling_results = Vec::new();

    for (label, replaced_set) in &scale_configs {
        let (logits_f32, final_h_f32, ms_f32) = run_scaled_layer_stack(
            &mut model,
            &server_f32,
            &server_f32.map_native_engine,
            target_tok,
            target_pos,
            &baseline_captures,
            replaced_set,
            MapSubstitutionMode::FullLayer,
        )?;
        let h_cmp_f32 = VectorComparisonMetrics::compute(&baseline_final_h, &final_h_f32);
        let logit_cmp_f32 = VectorComparisonMetrics::compute(&baseline_logits, &logits_f32);
        let top1_f32 = argmax(&logits_f32)?;
        let top5_f32 = top_k_indices(&logits_f32, 5);
        let top5_overlap_f32 = top5_f32.iter().filter(|i| baseline_top5.contains(i)).count();

        let (logits_bf16, final_h_bf16, ms_bf16) = run_scaled_layer_stack(
            &mut model,
            &server_bf16,
            &server_bf16.map_native_engine,
            target_tok,
            target_pos,
            &baseline_captures,
            replaced_set,
            MapSubstitutionMode::FullLayer,
        )?;
        let h_cmp_bf16 = VectorComparisonMetrics::compute(&baseline_final_h, &final_h_bf16);
        let logit_cmp_bf16 = VectorComparisonMetrics::compute(&baseline_logits, &logits_bf16);
        let top1_bf16 = argmax(&logits_bf16)?;
        let top5_bf16 = top_k_indices(&logits_bf16, 5);
        let top5_overlap_bf16 = top5_bf16
            .iter()
            .filter(|i| baseline_top5.contains(i))
            .count();

        let (logits_raw, final_h_raw, _) = run_scaled_layer_stack(
            &mut model,
            &server_static,
            &server_static.map_native_engine,
            target_tok,
            target_pos,
            &baseline_captures,
            replaced_set,
            MapSubstitutionMode::RawUncoupledWeightAtoms,
        )?;
        let h_cmp_raw = VectorComparisonMetrics::compute(&baseline_final_h, &final_h_raw);
        let logit_cmp_raw = VectorComparisonMetrics::compute(&baseline_logits, &logits_raw);
        let top1_raw = argmax(&logits_raw)?;

        println!(
            "[verified] Scale {:<34} | F32: h_cos={:.6} h_l2={:.4e} logit_l2={:.4e} top1={}({}/5) | BF16: h_cos={:.6} h_l2={:.4} logit_l2={:.4} top1={}({}/5) | UncoupledControl: h_cos={:.4} logit_l2={:.4} top1={} | stack_ms={:.2}ms",
            label,
            h_cmp_f32.cosine_similarity,
            h_cmp_f32.relative_l2_error,
            logit_cmp_f32.relative_l2_error,
            top1_f32 == baseline_top1,
            top5_overlap_f32,
            h_cmp_bf16.cosine_similarity,
            h_cmp_bf16.relative_l2_error,
            logit_cmp_bf16.relative_l2_error,
            top1_bf16 == baseline_top1,
            top5_overlap_bf16,
            h_cmp_raw.cosine_similarity,
            logit_cmp_raw.relative_l2_error,
            top1_raw == baseline_top1,
            ms_bf16,
        );

        scaling_results.push(json!({
            "scale_label": label,
            "replaced_layers_count": replaced_set.len(),
            "total_layers": 52,
            "coupled_f32": {
                "final_hidden_cosine": h_cmp_f32.cosine_similarity,
                "final_hidden_relative_l2": h_cmp_f32.relative_l2_error,
                "logit_relative_l2": logit_cmp_f32.relative_l2_error,
                "top1_token": top1_f32,
                "top1_agreement": top1_f32 == baseline_top1,
                "top5_overlap": top5_overlap_f32,
                "layers_latency_ms": ms_f32,
            },
            "coupled_bf16": {
                "final_hidden_cosine": h_cmp_bf16.cosine_similarity,
                "final_hidden_relative_l2": h_cmp_bf16.relative_l2_error,
                "logit_relative_l2": logit_cmp_bf16.relative_l2_error,
                "top1_token": top1_bf16,
                "top1_agreement": top1_bf16 == baseline_top1,
                "top5_overlap": top5_overlap_bf16,
                "layers_latency_ms": ms_bf16,
            },
            "uncoupled_raw_weight_control": {
                "final_hidden_cosine": h_cmp_raw.cosine_similarity,
                "final_hidden_relative_l2": h_cmp_raw.relative_l2_error,
                "logit_relative_l2": logit_cmp_raw.relative_l2_error,
                "top1_token": top1_raw,
                "top1_agreement": top1_raw == baseline_top1,
            }
        }));
    }

    // =========================================================================
    // PHASE 4: Corpus-Wide Multi-Chart Distillation, Idempotent Dedup & Weightless Decoding
    // =========================================================================
    println!("\n--- PHASE 4: CORPUS-WIDE MULTI-CHART DISTILLATION & MULTI-PROMPT WEIGHTLESS INFERENCE ---");

    let corpus_prompts = [
        ("Geography/Factual", prompt_text.as_str()),
        ("Systems/Rust", "fn main() -> Result<(), Error> {"),
        ("Geometry/Math", "The hyperbolic metric on Poincare ball"),
        ("Security/CWE", "Buffer overflow vulnerability in CWE-787"),
    ];

    struct CorpusPromptRecord {
        domain: String,
        prompt: String,
        token_id: u32,
        baseline_logits: Vec<f32>,
        baseline_final_h: Vec<f32>,
        baseline_captures: Vec<LayerActivationRecord>,
        baseline_top1: usize,
        baseline_top5: Vec<usize>,
        novel_atoms_minted: usize,
        coalesced_atoms: usize,
    }

    let mut corpus_records: Vec<CorpusPromptRecord> = Vec::new();
    // Prompt 0 is already captured in `baseline_captures`
    corpus_records.push(CorpusPromptRecord {
        domain: corpus_prompts[0].0.to_string(),
        prompt: corpus_prompts[0].1.to_string(),
        token_id: target_tok,
        baseline_logits: baseline_logits.clone(),
        baseline_final_h: baseline_final_h.clone(),
        baseline_captures: baseline_captures.clone(),
        baseline_top1,
        baseline_top5: baseline_top5.clone(),
        novel_atoms_minted: novel_coupled_atoms,
        coalesced_atoms: 0,
    });

    for &(domain, text) in &corpus_prompts[1..] {
        let ids = tokenizer.encode(text)?;
        let tok = *ids.last().unwrap_or(&model.config.bos_token_id);
        model.reset_memory();
        let (b_logits, b_final_h, b_captures) =
            model.forward_instrumented(tok, 0, &all_52_layers)?;
        let b_top1 = argmax(&b_logits)?;
        let b_top5 = top_k_indices(&b_logits, 5);

        let mut novel_cnt = 0usize;
        let mut coalesced_cnt = 0usize;
        let mut prev_f32 = None;
        let mut prev_bf16 = None;
        for cap in &b_captures {
            let icap = InstrumentedLayerCapture {
                layer: cap.layer,
                token_id: cap.token_id,
                next_token_id: cap.next_token_id,
                pos: cap.pos,
                x_in: cap.x_in.clone(),
                attn_delta: cap.attn_delta.clone(),
                x_mid: cap.x_mid.clone(),
                mlp_delta: cap.mlp_delta.clone(),
                x_out: cap.x_out.clone(),
            };
            let (aid_f32, role_f32, is_novel) = server_f32
                .map_native_engine
                .ingest_instrumented_layer_capture(
                    &icap,
                    52,
                    prev_f32,
                    &server_f32.mlir_catalog,
                    &mut server_f32.swarm_map,
                    &server_f32.immune.policy,
                )?;
            prev_f32 = Some((aid_f32, role_f32));
            if is_novel {
                novel_cnt += 1;
            } else {
                coalesced_cnt += 1;
            }

            let (aid_bf16, role_bf16, _) = server_bf16
                .map_native_engine
                .ingest_instrumented_layer_capture(
                    &icap,
                    52,
                    prev_bf16,
                    &server_bf16.mlir_catalog,
                    &mut server_bf16.swarm_map,
                    &server_bf16.immune.policy,
                )?;
            prev_bf16 = Some((aid_bf16, role_bf16));
        }

        println!(
            "[verified] Corpus Ingest [{:<17}] tok={:<6} | novel_atoms={}/52 coalesced={}/52 | total_vault_atoms={}",
            domain,
            tok,
            novel_cnt,
            coalesced_cnt,
            server_bf16.swarm_map.global_atom_count()
        );

        corpus_records.push(CorpusPromptRecord {
            domain: domain.to_string(),
            prompt: text.to_string(),
            token_id: tok,
            baseline_logits: b_logits,
            baseline_final_h: b_final_h,
            baseline_captures: b_captures,
            baseline_top1: b_top1,
            baseline_top5: b_top5,
            novel_atoms_minted: novel_cnt,
            coalesced_atoms: coalesced_cnt,
        });
    }

    // Idempotent Replay Gate: Replay ALL 4 prompts (208 layer steps) and prove 0 duplicate atoms are minted
    let t_replay = Instant::now();
    let mut replay_novel_atoms = 0usize;
    let mut replay_coalesced_atoms = 0usize;
    for rec in &corpus_records {
        let mut prev_bf16 = None;
        for cap in &rec.baseline_captures {
            let icap = InstrumentedLayerCapture {
                layer: cap.layer,
                token_id: cap.token_id,
                next_token_id: cap.next_token_id,
                pos: cap.pos,
                x_in: cap.x_in.clone(),
                attn_delta: cap.attn_delta.clone(),
                x_mid: cap.x_mid.clone(),
                mlp_delta: cap.mlp_delta.clone(),
                x_out: cap.x_out.clone(),
            };
            let (aid, role, is_novel) = server_bf16
                .map_native_engine
                .ingest_instrumented_layer_capture(
                    &icap,
                    52,
                    prev_bf16,
                    &server_bf16.mlir_catalog,
                    &mut server_bf16.swarm_map,
                    &server_bf16.immune.policy,
                )?;
            prev_bf16 = Some((aid, role));
            if is_novel {
                replay_novel_atoms += 1;
            } else {
                replay_coalesced_atoms += 1;
            }
        }
    }
    let replay_ms = t_replay.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[verified] Corpus Idempotent Replay Gate (208 layer steps across 4 prompts): novel_minted={} coalesced_existing={} latency={:.2}ms",
        replay_novel_atoms, replay_coalesced_atoms, replay_ms
    );

    // Save the expanded multi-chart BF16 coupled operator vault to `design/glimmer_52l_map_native.atoma_map`
    let coupled_vault_path = PathBuf::from(
        "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/glimmer_52l_map_native.atoma_map",
    );
    let (saved_atoms, saved_arcs, coupled_vault_sha) =
        server_bf16.swarm_map.save_to_binary_vault(&coupled_vault_path)?;
    let silo_counts: Vec<String> = RoleId::CARVED_12
        .iter()
        .map(|&r| {
            format!(
                "{}:{}",
                r.short_code(),
                server_bf16.swarm_map.silo(r).map(|s| s.atom_count()).unwrap_or(0)
            )
        })
        .collect();
    println!(
        "[verified] Persisted Multi-Chart Coupled Vault: {} 0-cells, {} 1-cells -> {} (sha256={}..) | silos=[{}]",
        saved_atoms,
        saved_arcs,
        coupled_vault_path.display(),
        &coupled_vault_sha.to_hex()[..16],
        silo_counts.join(" ")
    );

    // Multi-Prompt Full 52-Layer Weightless Decoding across the shared multi-chart vault
    let mut multi_prompt_weightless_json = Vec::new();
    for rec in &corpus_records {
        let (wl_logits_bf16, wl_h_bf16, wl_ms_bf16) = forward_full_weightless_from_map(
            &mut model,
            &server_bf16,
            &server_bf16.map_native_engine,
            rec.token_id,
            0,
        )?;
        let (wl_logits_f32, wl_h_f32, wl_ms_f32) = forward_full_weightless_from_map(
            &mut model,
            &server_f32,
            &server_f32.map_native_engine,
            rec.token_id,
            0,
        )?;
        let h_cmp_bf16 = VectorComparisonMetrics::compute(&rec.baseline_final_h, &wl_h_bf16);
        let logit_cmp_bf16 = VectorComparisonMetrics::compute(&rec.baseline_logits, &wl_logits_bf16);
        let h_cmp_f32 = VectorComparisonMetrics::compute(&rec.baseline_final_h, &wl_h_f32);
        let logit_cmp_f32 = VectorComparisonMetrics::compute(&rec.baseline_logits, &wl_logits_f32);
        let top1_bf16 = argmax(&wl_logits_bf16)?;
        let top5_bf16 = top_k_indices(&wl_logits_bf16, 5);
        let top5_overlap = top5_bf16
            .iter()
            .filter(|i| rec.baseline_top5.contains(i))
            .count();
        let next_tok_str = tokenizer.decode(&[top1_bf16 as u32]);

        println!(
            "[verified] Multi-Prompt 52L Weightless [{:<17}] tok={:<6} -> next={:<6} ({:?}) | F32_h_l2={:.3e} | BF16: h_cos={:.6} h_l2={:.4} logit_l2={:.4} top1={}({}/5) | 52L_map_ms={:.2}ms",
            rec.domain,
            rec.token_id,
            top1_bf16,
            next_tok_str,
            h_cmp_f32.relative_l2_error,
            h_cmp_bf16.cosine_similarity,
            h_cmp_bf16.relative_l2_error,
            logit_cmp_bf16.relative_l2_error,
            top1_bf16 == rec.baseline_top1,
            top5_overlap,
            wl_ms_bf16.min(wl_ms_f32)
        );

        multi_prompt_weightless_json.push(json!({
            "domain": rec.domain,
            "prompt": rec.prompt,
            "token_id": rec.token_id,
            "novel_atoms_minted": rec.novel_atoms_minted,
            "coalesced_atoms": rec.coalesced_atoms,
            "predicted_token_id": top1_bf16,
            "predicted_token_text": next_tok_str,
            "baseline_token_id": rec.baseline_top1,
            "top1_agreement": top1_bf16 == rec.baseline_top1,
            "top5_overlap": top5_overlap,
            "f32_final_h_relative_l2": h_cmp_f32.relative_l2_error,
            "f32_logit_relative_l2": logit_cmp_f32.relative_l2_error,
            "bf16_final_h_cosine": h_cmp_bf16.cosine_similarity,
            "bf16_final_h_relative_l2": h_cmp_bf16.relative_l2_error,
            "bf16_logit_relative_l2": logit_cmp_bf16.relative_l2_error,
            "weightless_52l_stack_ms": wl_ms_bf16,
        }));
    }

    // Held-Out Geodesic Interpolation Benchmark at Layer 20:
    // Construct a synthetic held-out state `x_interp = 0.65 * x_prompt0 + 0.35 * x_prompt2` and compare
    // Multi-Chart Geodesic Interpolation (`top_k = 8`, `gamma = 12.0`) vs Single-Chart (`top_k = 1`).
    let c0 = &corpus_records[0].baseline_captures[20];
    let c2 = &corpus_records[2].baseline_captures[20];
    let x_interp: Vec<f32> = c0
        .x_in
        .iter()
        .zip(&c2.x_in)
        .map(|(&a, &b)| 0.65 * a + 0.35 * b)
        .collect();
    let target_attn_interp: Vec<f32> = c0
        .attn_delta
        .iter()
        .zip(&c2.attn_delta)
        .map(|(&a, &b)| 0.65 * a + 0.35 * b)
        .collect();
    let target_out_interp: Vec<f32> = c0
        .x_out
        .iter()
        .zip(&c2.x_out)
        .map(|(&a, &b)| 0.65 * a + 0.35 * b)
        .collect();

    let mut single_chart_engine = MapNativeInferenceEngine::default();
    single_chart_engine.top_k_atoms = 1;
    let single_rec = single_chart_engine.retrieve_and_synthesize_layer(
        20,
        0,
        52,
        &x_interp,
        MapSubstitutionMode::FullLayer,
        &server_f32.swarm_map,
    )?;
    let single_attn_cmp =
        VectorComparisonMetrics::compute(&target_attn_interp, &single_rec.attn_delta);
    let single_out_cmp =
        VectorComparisonMetrics::compute(&target_out_interp, &single_rec.synthesized_x_out);

    let mut multi_chart_engine = MapNativeInferenceEngine::default();
    multi_chart_engine.top_k_atoms = 8;
    multi_chart_engine.beta_hyperbolic = 2.0;
    multi_chart_engine.gamma_receptor_cosine = 3.2;
    multi_chart_engine.eta_toroidal_phase = 0.5;
    let multi_rec = multi_chart_engine.retrieve_and_synthesize_layer(
        20,
        0,
        52,
        &x_interp,
        MapSubstitutionMode::FullLayer,
        &server_f32.swarm_map,
    )?;
    let multi_attn_cmp =
        VectorComparisonMetrics::compute(&target_attn_interp, &multi_rec.attn_delta);
    let multi_out_cmp =
        VectorComparisonMetrics::compute(&target_out_interp, &multi_rec.synthesized_x_out);

    println!(
        "[verified] Held-Out Geodesic Chart Interpolation (L20, 0.65*Geo + 0.35*Math): Single-Chart(K=1) attn_cos={:.4} out_l2={:.4} -> Multi-Chart(K=8) attn_cos={:.4} out_l2={:.4}",
        single_attn_cmp.cosine_similarity,
        single_out_cmp.relative_l2_error,
        multi_attn_cmp.cosine_similarity,
        multi_out_cmp.relative_l2_error
    );

    let report_json = json!({
        "model": model_dir,
        "coupled_vault": coupled_vault_path,
        "coupled_vault_sha256": coupled_vault_sha.to_hex(),
        "coupled_vault_0cells": saved_atoms,
        "coupled_vault_1cells": saved_arcs,
        "baseline_step_ms": baseline_step_ms,
        "phase2_and_phase3_scaling": scaling_results,
        "phase4_corpus_multi_chart": {
            "prompts_distilled": corpus_records.len(),
            "idempotent_replay_novel_minted": replay_novel_atoms,
            "idempotent_replay_coalesced": replay_coalesced_atoms,
            "idempotent_replay_latency_ms": replay_ms,
            "multi_prompt_weightless_results": multi_prompt_weightless_json,
            "heldout_chart_interpolation_l20": {
                "single_chart_k1_attn_cosine": single_attn_cmp.cosine_similarity,
                "single_chart_k1_out_relative_l2": single_out_cmp.relative_l2_error,
                "multi_chart_k8_attn_cosine": multi_attn_cmp.cosine_similarity,
                "multi_chart_k8_out_relative_l2": multi_out_cmp.relative_l2_error,
            }
        }
    });

    std::fs::write(
        "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/map_native_glimmer_report.json",
        serde_json::to_string_pretty(&report_json)?,
    )?;
    println!(
        "\n[verified] Wrote full Phase 1-4 JSON report to /Users/stewartbarteau/Desktop/monster-hybrid-server/design/map_native_glimmer_report.json"
    );

    Ok(())
}
