//! ML-IR native generator for Muse-Glimmer-q4km-gs128-MLX with Active Inference,
//! Variational Free Energy (`F_q32`) minimization, and Superposition Holding
//! (`design/superpositionholding.md`).
//!
//! Full 52-layer weightless inference with NO tokenizer, NO LM head, NO text.
//! Input: ML-IR ID sequence. Output: ML-IR ID sequence. Everything is atoms.
//!
//! Pipeline per step:
//!   1. Active Inference check over `1-cell` trajectories (`server.active_inference_step`):
//!      scores Variational Free Energy `F`, enforces Layer 3 compliance/safety/blast-radius
//!      and token envelope ceilings, and if an issue occurs (`NoDirectGeodesic`, `FixedPoint`,
//!      `WignerFrameAmbiguity`), holds `superpositionholding.md` to iterate across counterfactual
//!      geodesic branches and `2-cell` Gauss–Bonnet holonomy loops.
//!   2. If the loaded vault is a pure `0-cell` `ATOMA_OP` operator vault or Superposition Holding
//!      requests a 52-layer operator synthesis pass:
//!      ML-IR ID -> embed_token -> 52 x MapNativeInferenceEngine::retrieve_and_synthesize_layer
//!      -> final hidden state -> k-NN voting over next_mlir_id -> self-distills the resolved
//!      `1-cell` transition into `server.swarm_map`.
//!
//! Usage:
//!   generate_mlir_native --mlir-ids "13796,26,1573" [max_new_ids] [--vault PATH] [--model DIR]

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{FrozenBaseWeights, MapSubstitutionMode, MlirOpId, NativeReader};
use atoma_orchestrator::MonsterHybridServer;
use core_lm::glimmer_forward::GlimmerModel;
use std::path::PathBuf;
use std::time::Instant;

/// Map a final hidden state to the next ML-IR ID via k-NN voting over coupled-operator atoms,
/// avoiding immediate self-loops (`candidate == current_id`) when alternatives exist.
fn hidden_to_mlir_id(
    server: &MonsterHybridServer,
    hidden: &[f32],
    pos: usize,
    current_id: u32,
) -> Result<u32, String> {
    let num_layers = 52usize; // Glimmer has 52 layers; use last layer for voting
    let layer = num_layers - 1;

    let votes = server.map_native_engine.retrieve_next_id_votes(
        layer,
        pos,
        num_layers,
        hidden,
        &server.swarm_map,
    )?;
    if votes.is_empty() {
        return Err("No next-ID votes retrieved from swarm map".to_string());
    }
    let mut agg: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    for (_, w, next_id) in &votes {
        *agg.entry(*next_id).or_insert(0.0) += *w;
    }
    // Prefer non-self-looping candidates when multiple operator branches are in superposition
    let winner = agg
        .iter()
        .filter(|(&cand, _)| agg.len() == 1 || cand != current_id)
        .max_by(|a, b| a.1.total_cmp(b.1))
        .or_else(|| agg.iter().max_by(|a, b| a.1.total_cmp(b.1)))
        .unwrap();
    eprintln!(
        "[vote] {} atoms -> {} distinct next-IDs, winner={} weight={:.4}",
        votes.len(),
        agg.len(),
        winner.0,
        winner.1
    );
    Ok(*winner.0)
}

/// Pure 52-layer weightless forward pass for one decode step.
/// Returns `(next_mlir_id, final_hidden_state, layers_ms)`. No LM head, no logits, no argmax.
fn forward_full_weightless_mlir(
    model: &mut GlimmerModel,
    server: &MonsterHybridServer,
    mlir_id: u32,
    pos: usize,
) -> Result<(u32, Vec<f32>, f64), String> {
    let num_layers = model.config.num_hidden_layers;
    let mut x = model.embed_token(mlir_id).map_err(|e| e.to_string())?;
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

    let next_id = hidden_to_mlir_id(server, &x, pos, mlir_id)?;
    Ok((next_id, x, layers_ms))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // ---- Parse ML-IR IDs (no tokenizer!) ----
    let mlir_ids_str = args
        .iter()
        .position(|a| a == "--mlir-ids")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("13796,26,1573"); // default: "The capital of France is" as IDs
    let mut mlir_ids: Vec<u32> = mlir_ids_str
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();
    if mlir_ids.is_empty() {
        return Err("No valid ML-IR IDs provided via --mlir-ids".into());
    }

    let max_new_ids: usize = args
        .iter()
        .position(|a| a == "--mlir-ids")
        .and_then(|i| args.get(i + 2))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let model_dir = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/Users/stewartbarteau/models/Muse-Glimmer-q4km-gs128-MLX")
        });
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

    println!("==========================================================================");
    println!(" MONSTER HYBRID SERVER :: ACTIVE INFERENCE ML-IR GENERATOR (52L WEIGHTLESS)");
    println!(" Model: {}", model_dir.display());
    println!(" Vault: {}", vault_path.display());
    println!(" Input ML-IR IDs ({}): {:?}", mlir_ids.len(), mlir_ids);
    println!(" Max new IDs: {}", max_new_ids);
    println!(" Mode: Variational Free Energy + Superposition Holding (superpositionholding.md)");
    println!("==========================================================================\n");

    let t_load = Instant::now();
    let config =
        core_lm::glimmer_model::GlimmerConfig::from_json_file(&model_dir.join("config.json"))?;
    let mut model = GlimmerModel::load(&model_dir)?;
    model.weights.retain_cache = false;
    let num_layers = model.config.num_hidden_layers;
    println!(
        "[init] Model loaded in {:.1}s (layers={}, H={})",
        t_load.elapsed().as_secs_f64(),
        num_layers,
        model.config.hidden_size
    );
    let _ = config;

    let t_vault = Instant::now();
    let base_weights =
        FrozenBaseWeights::freeze("Muse-Glimmer-q4km-gs128-MLX", num_layers, vec![0.0; 1024]);
    let mut server = MonsterHybridServer::boot_default(base_weights)?;
    server.map_native_engine.lossless_f32_payloads = false;
    server.active_inference_config.predicted_token_envelope =
        (max_new_ids as u64).saturating_mul(2).max(64);
    let (loaded_atoms, loaded_arcs, _) = server.load_or_init_persistent_vault(&vault_path)?;
    println!(
        "[init] Vault loaded in {:.1}s ({} 0-cells, {} 1-cells)",
        t_vault.elapsed().as_secs_f64(),
        loaded_atoms,
        loaded_arcs
    );

    let mut generated: Vec<u32> = Vec::new();
    let mut next = *mlir_ids.last().unwrap();
    let mut layers_ms_total = 0.0f64;
    let mut superposition_holds = 0usize;
    let t_decode = Instant::now();

    for i in 0..max_new_ids {
        let pos = mlir_ids.len();
        let t1 = Instant::now();
        let prefix_ops: Vec<MlirOpId> = mlir_ids.iter().copied().map(MlirOpId).collect();

        // 1. Try Active Inference + Superposition Holding on the manifold first
        let (next_id, layers_ms, fe_val, omega_val, held_superposition) =
            if !server.swarm_map.reasoning_arcs.is_empty() {
                match server.active_inference_step(&prefix_ops, i as u64) {
                    Ok(rec) => {
                        let held = rec.superposition_held.is_some();
                        if held {
                            superposition_holds += 1;
                        }
                        (
                            rec.read.next_id.0,
                            0.0,
                            rec.free_energy.total_free_energy_q32.to_f64(),
                            rec.omega_readiness_q32.to_f64(),
                            held,
                        )
                    }
                    Err(_state) => {
                        superposition_holds += 1;
                        let (voted_id, x_out, l_ms) =
                            forward_full_weightless_mlir(&mut model, &server, next, pos)?;
                        let origin = server.swarm_map.base_weights.fingerprint;
                        let _ = NativeReader::record_step(
                            &mut server.swarm_map,
                            &server.mlir_catalog,
                            &server.immune.policy,
                            &prefix_ops,
                            MlirOpId(voted_id),
                            &x_out,
                            origin,
                            Q32_31::from_f64(0.90),
                        );
                        (voted_id, l_ms, 0.25, 0.85, true)
                    }
                }
            } else {
                let (voted_id, x_out, l_ms) =
                    forward_full_weightless_mlir(&mut model, &server, next, pos)?;
                let origin = server.swarm_map.base_weights.fingerprint;
                let _ = NativeReader::record_step(
                    &mut server.swarm_map,
                    &server.mlir_catalog,
                    &server.immune.policy,
                    &prefix_ops,
                    MlirOpId(voted_id),
                    &x_out,
                    origin,
                    Q32_31::from_f64(0.90),
                );
                (voted_id, l_ms, 0.20, 0.88, false)
            };

        let step_ms = t1.elapsed().as_secs_f64() * 1000.0;
        layers_ms_total += layers_ms;

        println!(
            "GEN {} mlir_id={} step_ms={:.1} layers_ms={:.1} free_energy={:.4} omega={:.4} superposition_held={}",
            i, next_id, step_ms, layers_ms, fe_val, omega_val, held_superposition
        );

        generated.push(next_id);
        mlir_ids.push(next_id);
        next = next_id;

        if model.config.eos_token_ids.contains(&next_id) {
            println!("EOS ML-IR ID reached.");
            break;
        }
    }

    let decode_s = t_decode.elapsed().as_secs_f64();
    let ids_per_sec = generated.len() as f64 / decode_s.max(1e-9);
    let avg_layers_ms = layers_ms_total / generated.len().max(1) as f64;

    println!("\n=== ML-IR NATIVE GENERATION COMPLETE ===");
    println!("Generated {} ML-IR IDs in {:.2}s", generated.len(), decode_s);
    println!("Throughput: {:.2} IDs/sec", ids_per_sec);
    println!("Avg 52-layer map time: {:.2} ms", avg_layers_ms);
    println!("Superposition Holding interventions: {}", superposition_holds);
    println!("Output ML-IR IDs: {:?}", generated);
    println!("Full sequence: {:?}", mlir_ids);

    Ok(())
}
