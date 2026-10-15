//! Map-native token generator for Muse-Glimmer-q4km-gs128-MLX.
//!
//! Full 52-layer weightless inference in an autoregressive decode loop:
//! `embed_token` -> 52 x `MapNativeInferenceEngine::retrieve_and_synthesize_layer`
//! (`MapSubstitutionMode::FullLayer`) -> `decode_lm_head` -> greedy argmax.
//!
//! Zero transformer attention/MLP weights execute during decode. Only the prompt
//! prefill runs the baseline instrumented forward pass (to distill prompt-specific
//! coupled-operator charts into the map); pass `--no-ingest` to skip.
//!
//! Usage:
//!   generate_map_native "<prompt>" [max_tokens] [--vault PATH] [--model DIR]
//!                        [--no-ingest] [--baseline-ids FILE]
//!
//! The optional `--baseline-ids` file (whitespace-separated u32 token ids, e.g. the
//! `GEN_IDS` output of `generate_glimmer`) is compared position-by-position against
//! the map-native output to measure parity / compounding divergence.

use atoma_immune::{FrozenBaseWeights, InstrumentedLayerCapture, MapSubstitutionMode};
use atoma_orchestrator::MonsterHybridServer;
use core_lm::glimmer_forward::GlimmerModel;
use core_lm::ops::argmax;
use core_lm::tokenizer::BpeTokenizer;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Instant;

/// Pure 52-layer weightless forward pass for one decode step.
/// Returns `(logits, layers_ms)`.
fn forward_full_weightless(
    model: &mut GlimmerModel,
    server: &MonsterHybridServer,
    token_id: u32,
    pos: usize,
) -> Result<(Vec<f32>, f64), String> {
    let num_layers = model.config.num_hidden_layers;
    let mut x = model.embed_token(token_id).map_err(|e| e.to_string())?;
    let t_layers = Instant::now();
    for layer in 0..num_layers {
        let rec = server.map_native_engine.retrieve_and_synthesize_layer(
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
    Ok((logits, layers_ms))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("The capital of France is");
    let max_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
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
            PathBuf::from(
                "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/glimmer_52l_map_native.atoma_map",
            )
        });
    let no_ingest = args.iter().any(|a| a == "--no-ingest");
    let baseline_ids_path = args
        .iter()
        .position(|a| a == "--baseline-ids")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);

    println!("==========================================================================");
    println!(" MONSTER HYBRID SERVER :: MAP-NATIVE TOKEN GENERATOR (52-LAYER WEIGHTLESS)");
    println!(" Model: {}", model_dir.display());
    println!(" Vault: {}", vault_path.display());
    println!(" Prompt: {:?}", prompt);
    println!(" Max tokens: {} | ingest-prompt: {}", max_tokens, !no_ingest);
    println!("==========================================================================\n");

    // ---- Load tokenizer + model ----
    let t_load = Instant::now();
    let tok = BpeTokenizer::from_json_file(&model_dir.join("tokenizer.json"))?;
    let config = core_lm::glimmer_model::GlimmerConfig::from_json_file(&model_dir.join("config.json"))?;
    let mut model = GlimmerModel::load(&model_dir)?;
    model.weights.retain_cache = false;
    let num_layers = model.config.num_hidden_layers;
    println!(
        "[init] Model loaded in {:.1}s (layers={}, H={})",
        t_load.elapsed().as_secs_f64(),
        num_layers,
        model.config.hidden_size
    );

    // ---- Boot server + load coupled-operator vault ----
    let t_vault = Instant::now();
    let base_weights =
        FrozenBaseWeights::freeze("Muse-Glimmer-q4km-gs128-MLX", num_layers, vec![0.0; 1024]);
    let mut server = MonsterHybridServer::boot_default(base_weights)?;
    // Vault was distilled with BF16 effectors; payloads are self-describing, but keep
    // the engine flag consistent for any prompt-specific atoms ingested below.
    server.map_native_engine.lossless_f32_payloads = false;
    let (loaded_atoms, loaded_arcs, _) = server.load_or_init_persistent_vault(&vault_path)?;
    println!(
        "[init] Vault loaded in {:.1}s ({} 0-cells, {} 1-cells)",
        t_vault.elapsed().as_secs_f64(),
        loaded_atoms,
        loaded_arcs
    );

    // ---- Tokenize prompt (BOS-prefixed, matching generate_glimmer) ----
    let mut tokens: Vec<u32> = tok.encode(prompt)?;
    tokens.insert(0, config.bos_token_id);
    println!("[init] PROMPT_IDS {}: {:?}", tokens.len(), tokens);

    // ---- Baseline IDs for parity (optional) ----
    let baseline_ids: Option<Vec<u32>> = baseline_ids_path
        .as_ref()
        .map(|p| -> Result<Vec<u32>, Box<dyn std::error::Error>> {
            let text = std::fs::read_to_string(p)?;
            Ok(text
                .split_whitespace()
                .filter_map(|s| s.trim_matches(|c| c == '[' || c == ']' || c == ',').parse().ok())
                .collect())
        })
        .transpose()?;
    if let Some(ref ids) = baseline_ids {
        println!("[init] Baseline parity file: {} ids", ids.len());
    }

    // ---- Prefill: baseline instrumented forward per prompt token + ingest ----
    let mut last_logits: Vec<f32> = Vec::new();
    let mut novel_atoms = 0usize;
    let t_prefill = Instant::now();
    if !no_ingest {
        model.reset_memory();
        let all_layers: BTreeSet<usize> = (0..num_layers).collect();
        for (pos, &tok_id) in tokens.iter().enumerate() {
            let (logits, _, captures) = model
                .forward_instrumented(tok_id, pos, &all_layers)
                .map_err(|e| e.to_string())?;
            let mut prev = None;
            for cap in &captures {
                let icap = InstrumentedLayerCapture {
                    layer: cap.layer,
                    token_id: cap.token_id,
                    pos: cap.pos,
                    x_in: cap.x_in.clone(),
                    attn_delta: cap.attn_delta.clone(),
                    x_mid: cap.x_mid.clone(),
                    mlp_delta: cap.mlp_delta.clone(),
                    x_out: cap.x_out.clone(),
                };
                let (aid, role, is_novel) =
                    server.map_native_engine.ingest_instrumented_layer_capture(
                        &icap,
                        num_layers,
                        prev,
                        &server.mlir_catalog,
                        &mut server.swarm_map,
                        &server.immune.policy,
                    )?;
                prev = Some((aid, role));
                if is_novel {
                    novel_atoms += 1;
                }
            }
            last_logits = logits;
            let choice = argmax(&last_logits).map_err(|e| format!("argmax: {}", e))? as u32;
            println!(
                "[prefill] pos={} tok={} argmax={} piece={:?}",
                pos,
                tok_id,
                choice,
                tok.decode(&[choice]).trim()
            );
        }
        println!(
            "[prefill] done in {:.2}s (novel coupled atoms minted: {})",
            t_prefill.elapsed().as_secs_f64(),
            novel_atoms
        );
    } else {
        // No ingest: still need prefill logits for the boundary token; run plain baseline.
        model.reset_memory();
        for (pos, &tok_id) in tokens.iter().enumerate() {
            last_logits = model.forward(tok_id, pos).map_err(|e| e.to_string())?;
        }
        println!(
            "[prefill] baseline-only done in {:.2}s",
            t_prefill.elapsed().as_secs_f64()
        );
    }

    let boundary = argmax(&last_logits).map_err(|e| format!("argmax: {}", e))? as u32;
    println!(
        "[prefill] boundary argmax={} piece={:?}",
        boundary,
        tok.decode(&[boundary]).trim()
    );

    // ---- Decode: pure map-native loop ----
    model.reset_memory();
    let eos = model.config.eos_token_ids.clone();
    let mut generated: Vec<u32> = Vec::new();
    let mut next = boundary;
    let mut first_diverge: Option<usize> = None;
    let mut layers_ms_total = 0.0f64;
    let t_decode = Instant::now();
    for i in 0..max_tokens {
        let pos = tokens.len();
        let t1 = Instant::now();
        let (logits, layers_ms) = forward_full_weightless(&mut model, &server, next, pos)?;
        let step_ms = t1.elapsed().as_secs_f64() * 1000.0;
        layers_ms_total += layers_ms;

        let choice = argmax(&logits).map_err(|e| format!("argmax: {}", e))? as u32;
        let piece = tok.decode(&[next]);
        print!(
            "GEN {} token={} piece={:?} step_ms={:.1} layers_ms={:.2}",
            i,
            next,
            piece.trim(),
            step_ms,
            layers_ms
        );
        if let Some(ref ids) = baseline_ids {
            if let Some(&expected) = ids.get(i) {
                if expected == next {
                    print!(" parity=OK");
                } else {
                    print!(" parity=MISMATCH(expected {})", expected);
                    if first_diverge.is_none() {
                        first_diverge = Some(i);
                    }
                }
            }
        }
        println!();
        generated.push(next);
        if eos.contains(&next) {
            println!("EOS reached.");
            break;
        }
        if i + 1 == max_tokens {
            break;
        }
        tokens.push(next);
        next = choice;
    }
    let decode_s = t_decode.elapsed().as_secs_f64();
    let n_gen = generated.len().max(1) as f64;
    println!("\n=== MAP-NATIVE DECODE SUMMARY ===");
    println!("GEN_IDS {:?}", generated);
    println!(
        "decode_tokens={} decode_time={:.2}s tok/s={:.2} avg_layers_ms={:.2}",
        generated.len(),
        decode_s,
        n_gen / decode_s,
        layers_ms_total / n_gen
    );
    match first_diverge {
        Some(i) => println!("parity: FIRST DIVERGENCE at generated index {}", i),
        None if baseline_ids.is_some() => println!(
            "parity: all {} compared positions match baseline",
            baseline_ids.as_ref().unwrap().len().min(generated.len())
        ),
        None => println!("parity: no baseline file provided"),
    }
    let output = tok.decode(&generated);
    println!("\n=== OUTPUT ===\n{}{}", prompt, output);
    println!("=== END ===");
    Ok(())
}
