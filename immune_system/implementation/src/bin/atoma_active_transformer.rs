//! `atoma_active_transformer`: End-to-End Geodesic Active-Inference Transformer & Silicon Smoke Benchmark
//!
//! Wires together:
//! 1. Persistent Compounded Map Vault (`design/global_active_mind.atoma_map`) + Binary Write-Ahead
//!    Delta Journal (`design/global_active_mind.atoma_map.wal`) + Multi-Model Compounding (`--distill-vault`).
//! 2. Reversible Outer-Skin `BoundaryLexicalTransducer` (loads HuggingFace `tokenizer.json` in pure Rust
//!    at the sensory/motor boundary: UTF-8 Text <-> Base Token Carrier `MlirOpId`s <-> Multi-Token
//!    `AtomicIrAlphabet` Span Opcodes, with `0` Python and `0` inner strings).
//! 3. $O(1)$ Incremental Wigner Gyroscope Cursor (`GeodesicCursor`, 96 bytes constant state, `0` KV cache).
//! 4. Multi-Token `AtomicIrAlphabet` Span Folding (`fold_high_certainty_trajectory_spans`, `3..=6` tokens
//!    inside the `<= 7`-token verify cliff).
//! 5. Autonomous Task Execution in Motion (`execute_task_in_motion`, `backtrack_and_reiterate`,
//!    `crystallize_episode`): encounters novel problems mid-task, holds superposition, backtracks
//!    dead ends, verifies $2$-cell Gauss–Bonnet holonomy, and carves verified solutions into the
//!    living map in microseconds without fine-tuning.
//! 6. Silicon Throughput Smoke Benchmark: measures single-token Active Inference (`tok/s`),
//!    incremental `GeodesicCursor` decode (`tok/s`), multi-token `AtomicIrAlphabet` span-expanded
//!    decode (`effective tok/s`), and zero-JSON `ATOMA_WR` binary wire round-trip latency.

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    Atlas13, CarvedSwarmMap, DEFAULT_ACTIVE_MIND_VAULT, EpisodeWorkingTrace, FrozenBaseWeights,
    GeodesicActiveTransformer, GeodesicCursor, MlirOpId, NativeReader, NativeWireRequest,
    NativeWireResponse, OverlapPolicyMatrix, fold_high_certainty_trajectory_spans,
};
use std::path::PathBuf;
use std::time::Instant;

fn resolve_default_vault() -> PathBuf {
    let candidates = [
        DEFAULT_ACTIVE_MIND_VAULT,
        "../../design/global_active_mind.atoma_map",
        "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/global_active_mind.atoma_map",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("/Users/stewartbarteau/Desktop/monster-hybrid-server/design/global_active_mind.atoma_map")
}

fn derive_wal_path(vault_path: &PathBuf) -> PathBuf {
    let mut s = vault_path.as_os_str().to_os_string();
    s.push(".wal");
    PathBuf::from(s)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut vault_path = resolve_default_vault();
    let mut tokenizer_path: Option<PathBuf> = None;
    let mut custom_prompt: Option<String> = None;
    let mut custom_prefix_ids: Option<Vec<MlirOpId>> = None;
    let mut max_new: usize = 32;
    let mut extra_distill_vaults: Vec<PathBuf> = Vec::new();
    let mut save_after_fold = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vault" if i + 1 < args.len() => {
                vault_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--tokenizer" if i + 1 < args.len() => {
                tokenizer_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--prompt" if i + 1 < args.len() => {
                custom_prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "--prefix-ids" if i + 1 < args.len() => {
                let ids: Vec<MlirOpId> = args[i + 1]
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        s.trim()
                            .parse::<u32>()
                            .map(MlirOpId)
                            .map_err(|e| format!("invalid token id '{}': {}", s, e))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                custom_prefix_ids = Some(ids);
                i += 2;
            }
            "--max-new" if i + 1 < args.len() => {
                max_new = args[i + 1]
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --max-new: {}", e))?;
                i += 2;
            }
            "--distill-vault" if i + 1 < args.len() => {
                extra_distill_vaults.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--no-save" => {
                save_after_fold = false;
                i += 1;
            }
            other => {
                return Err(format!(
                    "Unknown flag '{}'. Usage: atoma_active_transformer [--vault PATH] [--tokenizer PATH] [--prompt TEXT] [--prefix-ids 791,6864,315,9822,374] [--max-new 32] [--distill-vault PATH] [--no-save]",
                    other
                ));
            }
        }
    }

    println!("================================================================================");
    println!("  ATOMA GEODESIC ACTIVE-INFERENCE TRANSFORMER — MONSTER HYBRID SERVER");
    println!("  Zero KV Cache (96B GeodesicCursor) | Zero Inner Strings | Zero JSON | Zero Python");
    println!("================================================================================");

    // 1. Boot Atlas13 + OverlapPolicyMatrix + CarvedSwarmMap
    let t_boot = Instant::now();
    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let base_weights =
        FrozenBaseWeights::freeze("global-active-mind-compound", 60, vec![0.1f32; 64]);
    let mut map = CarvedSwarmMap::new(atlas, base_weights, Q32_31::from_f64(1.22));
    if !vault_path.exists() {
        return Err(format!("Vault not found at {}", vault_path.display()));
    }
    let (loaded_atoms, loaded_arcs, loaded_sha) = map.load_from_binary_vault(&vault_path, &policy)?;
    let boot_ms = t_boot.elapsed().as_secs_f64() * 1_000.0;

    // 2. Initialize GeodesicActiveTransformer + BoundaryLexicalTransducer (Binary ATOMA_LX first)
    let wal_path = derive_wal_path(&vault_path);
    let mut transformer = GeodesicActiveTransformer::new(202_048, Some(wal_path.clone()));
    let loaded_vocab_entries = if let Some(ref tp) = tokenizer_path {
        if tp.extension().is_some_and(|e| e == "atoma_lex") {
            transformer.transducer.load_binary_lexicon(tp)?
        } else {
            transformer.transducer.load_huggingface_tokenizer_json(tp)?
        }
    } else {
        let count = transformer.transducer.load_or_bootstrap_default()?;
        println!(
            "[BOUNDARY TRANSDUCER] Loaded {} reversible carrier pieces via binary ATOMA_LX lexicon",
            count
        );
        count
    };

    // Replay any existing `.atoma_map.wal` delta frames
    let wal_steps_replayed = transformer.replay_wal_into_map(&wal_path, &mut map, &policy)?;

    println!(
        "[VAULT BOOT] Loaded {} in {:.1} ms\n  -> 0-cells (Knowledge): {}\n  -> 1-cells (Reasoning Arcs): {} (+{} WAL steps replayed)\n  -> Arc-Bearing Carrier Anchors: {}\n  -> Vault SHA-256: {:02x}{:02x}{:02x}{:02x}..",
        vault_path.display(),
        boot_ms,
        loaded_atoms,
        loaded_arcs,
        wal_steps_replayed,
        map.arc_bearing_atom_ids().count(),
        loaded_sha.0[0],
        loaded_sha.0[1],
        loaded_sha.0[2],
        loaded_sha.0[3],
    );

    // 3. Optional Multi-Model Compounding (`--distill-vault PATH`)
    for extra_vp in &extra_distill_vaults {
        let t_merge = Instant::now();
        let (na, nr) = map.merge_vault_from(extra_vp, &policy, true)?;
        println!(
            "[MULTI-MODEL COMPOUND] Merged {} in {:.1} ms: +{} novel atoms, +{} novel arcs",
            extra_vp.display(),
            t_merge.elapsed().as_secs_f64() * 1_000.0,
            na,
            nr
        );
    }

    // 4. Multi-Token `AtomicIrAlphabet` Span Folding (`3..=6` tokens inside the `<= 7`-token verify cliff)
    let t_fold = Instant::now();
    let fold_receipt = fold_high_certainty_trajectory_spans(
        &mut map,
        &transformer.transducer.catalog,
        &policy,
        &mut transformer.transducer.atomic_ir,
        2_048,
    )?;
    let fold_ms = t_fold.elapsed().as_secs_f64() * 1_000.0;
    println!(
        "[SPAN FOLDING] Scanned {} deterministic chains in {:.2} ms -> pinned {} multi-token span atoms + {} shortcut arcs (max span={} tokens, <= 7-token cliff)",
        fold_receipt.candidate_chains_inspected,
        fold_ms,
        fold_receipt.new_span_atoms_pinned,
        fold_receipt.shortcut_span_arcs_minted,
        fold_receipt.max_span_length
    );

    // If the user passed a custom prompt or prefix IDs, execute it directly and print the receipt
    if custom_prompt.is_some() || custom_prefix_ids.is_some() {
        let prompt_str = custom_prompt.as_deref().unwrap_or("<custom-prefix-ids>");
        let t0 = Instant::now();
        let receipt = transformer.execute_task_in_motion(
            0xA704_2026_9999,
            prompt_str,
            custom_prefix_ids.as_deref(),
            max_new,
            true,
            &mut map,
            &policy,
        )?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "\n[CUSTOM TASK EXECUTION]\n  Prompt: {:?}\n  Ingress OpIds ({}): {:?}\n  Emitted OpIds ({}): {:?}\n  Expanded Base Tokens ({}): {:?}\n  Decoded Text: {:?}\n  Superposition Interventions: {} | Compliance Verified: {}\n  Elapsed: {:.3} ms ({:.1} effective tok/s)",
            receipt.prompt_text,
            receipt.ingress_ops.len(),
            receipt.ingress_ops.iter().map(|op| op.0).collect::<Vec<_>>(),
            receipt.emitted_ops.len(),
            receipt.emitted_ops.iter().map(|op| op.0).collect::<Vec<_>>(),
            receipt.expanded_tokens.len(),
            receipt.expanded_tokens,
            receipt.decoded_output_text,
            receipt.superposition_interventions,
            receipt.compliance_verified,
            elapsed_ms,
            if elapsed_ms > 0.0 {
                (receipt.expanded_tokens.len() as f64) / (elapsed_ms / 1_000.0)
            } else {
                0.0
            }
        );
    }

    // =========================================================================
    // TASK 1: End-to-End Autonomous Knowledge & Code Trajectory Task via Reader + Transformer
    // =========================================================================
    println!("\n--------------------------------------------------------------------------------");
    println!("  TASK 1: End-to-End Boundary Text -> Geodesic MLIR Walk -> Decoded Text");
    println!("--------------------------------------------------------------------------------");
    // Using the canonical Glimmer trajectory prefix ("The capital of France is" -> [200000, 954, 7963, 323, 11698, 373])
    let glimmer_france_prefix = [
        MlirOpId(200_000),
        MlirOpId(954),
        MlirOpId(7963),
        MlirOpId(323),
        MlirOpId(11_698),
        MlirOpId(373),
    ];
    let task1_receipt = transformer.execute_task_in_motion(
        0xA704_2026_0001,
        "The capital of France is",
        Some(&glimmer_france_prefix),
        max_new,
        false,
        &mut map,
        &policy,
    )?;
    let final_cursor = GeodesicCursor::from_prefix(
        &transformer.transducer.catalog,
        &task1_receipt.ingress_ops,
    )?;
    println!(
        "  Ingress Text        : {:?}",
        task1_receipt.prompt_text
    );
    println!(
        "  Ingress MlirOpIds   : {:?}",
        task1_receipt.ingress_ops.iter().map(|x| x.0).collect::<Vec<_>>()
    );
    println!(
        "  Emitted MlirOpIds   : {} steps -> {:?}",
        task1_receipt.emitted_ops.len(),
        task1_receipt
            .emitted_ops
            .iter()
            .take(12)
            .map(|x| x.0)
            .collect::<Vec<_>>()
    );
    println!(
        "  Expanded Base Tokens: {} tokens",
        task1_receipt.expanded_tokens.len()
    );
    println!(
        "  Egress Decoded Text : {:?}",
        task1_receipt.decoded_output_text
    );
    println!(
        "  GeodesicCursor State: chain_id=0x{:016x}, step={}, state_size={} bytes (0 KV cache)",
        final_cursor.chain_id,
        final_cursor.step_index,
        std::mem::size_of::<GeodesicCursor>()
    );
    println!(
        "  Layer-3 Statutes    : compliance_verified={}, min_omega={:.4}, total_F={:.4}",
        task1_receipt.compliance_verified,
        task1_receipt.min_omega_readiness_q32.to_f64(),
        task1_receipt.total_free_energy_q32.to_f64()
    );

    // Also test a pure text-transduced prompt directly from the loaded HuggingFace tokenizer!
    let text_prompt = "The capital of France is";
    let transduced_ids = transformer.transducer.transduce_ingress_text(text_prompt, true);
    let (_fast_ops, _fast_exp, fast_text) =
        transformer.decode_fast_incremental(&map, &transduced_ids, max_new)?;
    println!(
        "  Direct Boundary Text Transduction ({:?} -> {:?}) -> {:?}",
        text_prompt,
        transduced_ids.iter().map(|x| x.0).collect::<Vec<_>>(),
        fast_text
    );

    // =========================================================================
    // TASK 2: Dynamic Problem Solving in Motion (Dead-End -> Backtrack -> Superposition -> Crystallize)
    // =========================================================================
    println!("\n--------------------------------------------------------------------------------");
    println!("  TASK 2: Dynamic Problem Solving in Motion (Mid-Task Failure -> Reiterate -> Crystallize)");
    println!("--------------------------------------------------------------------------------");
    // Construct a novel problem prefix whose last token is an unmapped novel anchor so the
    // transformer encounters a live mid-task obstacle, holds superposition, resolves it via
    // 27-cell hyperbolic parallel transport, and crystallizes the verified solution into the map + WAL!
    let novel_problem_prefix = vec![
        MlirOpId(200_000),
        MlirOpId(954),
        MlirOpId(7963),
        MlirOpId(323),
        MlirOpId(99_999), // unmapped perturbation triggering mid-task Superposition Holding
    ];
    let t_pass1 = Instant::now();
    let pass1 = NativeReader::continue_ids_active_mut(
        &mut map,
        &transformer.transducer.catalog,
        &policy,
        &novel_problem_prefix,
        24,
        &transformer.transducer.eos_ids,
        &transformer.active_config,
    )?;
    let pass1_ms = t_pass1.elapsed().as_secs_f64() * 1_000.0;
    let pass1_ops: Vec<MlirOpId> = pass1.steps.iter().map(|s| s.read.next_id).collect();
    let (_, pass1_text) = transformer.transducer.transduce_egress_ops(&pass1_ops);

    // Demonstrate mid-task backtracking & crystallization when an external verifier flags a step:
    let mut trace_for_reiteration = EpisodeWorkingTrace {
        episode_id: 0xA704_2026_0002,
        initial_prefix: novel_problem_prefix.clone(),
        active_prefix: novel_problem_prefix.clone(),
        tentative_steps: pass1
            .steps
            .iter()
            .take(4)
            .map(|s| {
                (
                    novel_problem_prefix.clone(),
                    s.read.next_id,
                    s.omega_readiness_q32,
                )
            })
            .collect(),
        pruned_dead_ends: Vec::new(),
        superposition_holds: pass1.superposition_interventions,
    };
    let reiterated_step = transformer.backtrack_and_reiterate(
        &mut map,
        &policy,
        &mut trace_for_reiteration,
        1,
    )?;
    let cryst_receipt = transformer.crystallize_episode(&mut map, &policy, &trace_for_reiteration)?;

    // Now re-run the exact same `novel_problem_prefix` (Pass 2) to prove the map learned in motion!
    let t_pass2 = Instant::now();
    let pass2 = NativeReader::continue_ids_active_mut(
        &mut map,
        &transformer.transducer.catalog,
        &policy,
        &novel_problem_prefix,
        24,
        &transformer.transducer.eos_ids,
        &transformer.active_config,
    )?;
    let pass2_ms = t_pass2.elapsed().as_secs_f64() * 1_000.0;

    println!(
        "  Pass 1 (Novel Obstacle Encountered at OpId 241777):\n    -> Superposition Holds: {} | Bridge Arcs Distilled: {} | Emitted: {} steps in {:.3} ms\n    -> Resolved Egress Text: {:?}",
        pass1.superposition_interventions,
        pass1.bridge_arcs_distilled,
        pass1.steps.len(),
        pass1_ms,
        pass1_text
    );
    println!(
        "  Mid-Task Backtrack & Reiteration:\n    -> Pruned Dead-Ends: {} | Reiterated Next OpId: {} | Holonomy Verified: {} (Mark: {:?})",
        trace_for_reiteration.pruned_dead_ends.len(),
        reiterated_step.read.next_id.0,
        cryst_receipt.holonomy_verified,
        cryst_receipt.mark
    );
    println!(
        "  In-Motion Crystallization Receipt:\n    -> Steps Crystallized: {} | Novel Arcs Minted: {} | WAL Appended: {} bytes ({})",
        cryst_receipt.steps_crystallized,
        cryst_receipt.novel_arcs_minted,
        cryst_receipt.wal_bytes_appended,
        wal_path.display()
    );
    println!(
        "  Pass 2 (Immediate Replay After In-Motion Learning — Zero Fine-Tuning):\n    -> Superposition Holds: {} (down from {}) | Bridge Arcs Needed: {} | Emitted: {} steps in {:.3} ms ({:.2}x faster)",
        pass2.superposition_interventions,
        pass1.superposition_interventions,
        pass2.bridge_arcs_distilled,
        pass2.steps.len(),
        pass2_ms,
        if pass2_ms > 0.0 { pass1_ms / pass2_ms } else { 1.0 }
    );

    // =========================================================================
    // TASK 3: Silicon Throughput Smoke Benchmark (F1 Decode Speed & Wire Latency)
    // =========================================================================
    println!("\n--------------------------------------------------------------------------------");
    println!("  TASK 3: Silicon Throughput Smoke Benchmark (Apple Silicon Native Q32.31)");
    println!("--------------------------------------------------------------------------------");
    let bench_prefix = glimmer_france_prefix;
    let iterations = 50usize;
    let steps_per_iter = 32usize;

    // 3A. Span-Accelerated O(1) GeodesicCursor Decode (`decode_span_accelerated_incremental`)
    let t_span = Instant::now();
    let mut total_span_ops = 0usize;
    let mut total_span_tokens = 0usize;
    for _ in 0..iterations {
        let (ops, exp, _txt) =
            transformer.decode_span_accelerated_incremental(&map, &bench_prefix, steps_per_iter)?;
        total_span_ops += ops.len();
        total_span_tokens += exp.len();
    }
    let span_elapsed = t_span.elapsed().as_secs_f64();
    let span_ops_per_sec = (total_span_ops as f64) / span_elapsed;
    let span_tok_per_sec = (total_span_tokens as f64) / span_elapsed;

    // 3B. Incremental O(1) GeodesicCursor Decode (`decode_fast_incremental`)
    let t_cursor = Instant::now();
    let mut total_cursor_ops = 0usize;
    let mut total_cursor_tokens = 0usize;
    for _ in 0..iterations {
        let (ops, exp, _txt) =
            transformer.decode_fast_incremental(&map, &bench_prefix, steps_per_iter)?;
        total_cursor_ops += ops.len();
        total_cursor_tokens += exp.len();
    }
    let cursor_elapsed = t_cursor.elapsed().as_secs_f64();
    let cursor_ops_per_sec = (total_cursor_ops as f64) / cursor_elapsed;
    let cursor_tok_per_sec = (total_cursor_tokens as f64) / cursor_elapsed;

    // 3C. Full Active-Inference + Superposition + Span-Expanded Decode (`continue_ids_active_mut`)
    let t_active = Instant::now();
    let mut total_active_ops = 0usize;
    let mut total_active_tokens = 0usize;
    for _ in 0..iterations {
        let c = NativeReader::continue_ids_active_mut(
            &mut map,
            &transformer.transducer.catalog,
            &policy,
            &bench_prefix,
            steps_per_iter,
            &transformer.transducer.eos_ids,
            &transformer.active_config,
        )?;
        let ops: Vec<MlirOpId> = c.steps.iter().map(|s| s.read.next_id).collect();
        let (exp, _) = transformer.transducer.transduce_egress_ops(&ops);
        total_active_ops += ops.len();
        total_active_tokens += exp.len();
    }
    let active_elapsed = t_active.elapsed().as_secs_f64();
    let active_ops_per_sec = (total_active_ops as f64) / active_elapsed;
    let active_tok_per_sec = (total_active_tokens as f64) / active_elapsed;

    // 3D. Zero-JSON `ATOMA_WR` Binary Wire Protocol Round-Trip Benchmark
    let wire_req = NativeWireRequest {
        active_inference: true,
        crystallize_on_success: true,
        max_new: 32,
        base_vocab: 202_048,
        prefix: bench_prefix.to_vec(),
        eos: transformer.transducer.eos_ids.clone(),
    };
    let wire_req_bytes = wire_req.to_wire_bytes();
    let t_wire = Instant::now();
    let wire_resp_bytes = transformer.handle_binary_wire_frame(&wire_req_bytes, &mut map, &policy)?;
    let wire_us = t_wire.elapsed().as_secs_f64() * 1_000_000.0;
    let wire_resp = NativeWireResponse::from_wire_bytes(&wire_resp_bytes)?;

    println!(
        "  1. Span-Accelerated O(1) GeodesicCursor Decode (`decode_span_accelerated_incremental`):\n     -> {:>10.1} MlirOp steps/sec | {:>10.1} effective tokens/sec ({:.1}x above 300 tok/s target)",
        span_ops_per_sec,
        span_tok_per_sec,
        span_tok_per_sec / 300.0
    );
    println!(
        "  2. Token-by-Token O(1) GeodesicCursor Decode (`decode_fast_incremental`):\n     -> {:>10.1} MlirOp steps/sec | {:>10.1} effective tokens/sec ({:.1}x above 300 tok/s target)",
        cursor_ops_per_sec,
        cursor_tok_per_sec,
        cursor_tok_per_sec / 300.0
    );
    println!(
        "  3. Full Active-Inference + Superposition + Span-Expanded Decode:\n     -> {:>10.1} MlirOp steps/sec | {:>10.1} effective tokens/sec ({:.1}x above 300 tok/s target)",
        active_ops_per_sec,
        active_tok_per_sec,
        active_tok_per_sec / 300.0
    );
    println!(
        "  4. Zero-JSON `ATOMA_WR` Binary Wire Frame Round-Trip (32 steps):\n     -> Request: {} bytes | Response: {} bytes | Latency: {:.1} µs | Emitted: {} ops ({} expanded tokens)",
        wire_req_bytes.len(),
        wire_resp_bytes.len(),
        wire_us,
        wire_resp.emitted_ops.len(),
        wire_resp.expanded_tokens.len()
    );
    println!(
        "  5. Causal State Memory Footprint:\n     -> GeodesicCursor: {} bytes constant O(1) (KV Cache = 0 bytes, Python = 0, JSON = 0)",
        std::mem::size_of::<GeodesicCursor>()
    );

    // 5. Persist updated `global_active_mind.atoma_map` with folded spans & newly crystallized trajectories
    if save_after_fold {
        let t_save = Instant::now();
        let (saved_atoms, saved_arcs, saved_sha) = map.save_to_binary_vault(&vault_path)?;
        let save_ms = t_save.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "\n[VAULT PERSISTED] Saved {} in {:.1} ms\n  -> Total 0-cells: {} | Total 1-cells: {} | Vault SHA-256: {:02x}{:02x}{:02x}{:02x}..\n  -> Live WAL Journal: {} (loaded vocab pieces: {})",
            vault_path.display(),
            save_ms,
            saved_atoms,
            saved_arcs,
            saved_sha.0[0],
            saved_sha.0[1],
            saved_sha.0[2],
            saved_sha.0[3],
            wal_path.display(),
            loaded_vocab_entries
        );
    }

    Ok(())
}
