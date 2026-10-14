//! Generate binary for Muse Glimmer with geometric memory.
//!
//! Usage: cargo run --bin generate_glimmer -- "prompt text" [max_tokens]

use core_lm::glimmer_forward::GlimmerModel;
use core_lm::ops::argmax;
use core_lm::tokenizer::BpeTokenizer;
use std::path::PathBuf;
use std::time::Instant;

fn top_n(logits: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(n.min(order.len()));
    order.into_iter().map(|i| (i, logits[i])).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("The capital of France is");
    let max_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let model_dir = PathBuf::from("/Users/stewartbarteau/models/Muse-Glimmer-q4km-gs128-MLX");
    let tokens_only = std::env::var("GLIMMER_TOKENS_ONLY").is_ok();
    let stop_after_prefill = std::env::var("GLIMMER_STOP_AFTER_PREFILL").is_ok();
    println!("Loading tokenizer...");
    let tok = BpeTokenizer::from_json_file(&model_dir.join("tokenizer.json"))?;
    let config = core_lm::glimmer_model::GlimmerConfig::from_json_file(
        &model_dir.join("config.json"),
    )?;
    let mut tokens = tok.encode(prompt)?;
    tokens.insert(0, config.bos_token_id);
    println!("PROMPT_IDS {}: {:?}", tokens.len(), tokens);
    if tokens_only {
        return Ok(());
    }
    println!("Loading Glimmer model (this takes a while)...");
    let t0 = Instant::now();
    let mut model = GlimmerModel::load(&model_dir)?;
    // Full f32 cache of this checkpoint does not fit next to the mapped q4 file.
    model.weights.retain_cache = false;
    let (quant_bits, quant_group) = model.weights.quant_defaults();
    println!(
        "Model loaded in {:.1}s (q_fused={}, layers={}, quant={}bit/g{} overrides={})",
        t0.elapsed().as_secs_f32(),
        model.q_fused,
        model.config.num_hidden_layers,
        quant_bits,
        quant_group,
        model.weights.quant_override_count()
    );
    let eos = model.config.eos_token_ids.clone();
    let t0 = Instant::now();
    let mut last_logits: Vec<f32> = Vec::new();
    for pos in 0..tokens.len() {
        let logits = model.forward(tokens[pos], pos)?;
        let choice = argmax(&logits).map_err(|e| format!("argmax: {}", e))?;
        let piece = tok.decode(&[choice as u32]);
        println!(
            "PREFILL_POS {} in={} argmax={} piece={} logit={:.4}",
            pos,
            tokens[pos],
            choice,
            piece.trim(),
            logits[choice]
        );
        last_logits = logits;
    }
    println!("Prefill done in {:.1}s.", t0.elapsed().as_secs_f32());
    let boundary = argmax(&last_logits).map_err(|e| format!("argmax: {}", e))?;
    println!(
        "PREFILL_ARGMAX {} piece={} logit={:.4}",
        boundary,
        tok.decode(&[boundary as u32]).trim(),
        last_logits[boundary]
    );
    for (id, logit) in top_n(&last_logits, 5) {
        println!(
            "PREFILL_TOP {} piece={} logit={:.4}",
            id,
            tok.decode(&[id as u32]).trim(),
            logit
        );
    }
    if stop_after_prefill {
        return Ok(());
    }
    // First output is the prefill argmax. Each later forward takes the id just emitted,
    // at the next position, which is the same step MLX generate_step uses.
    let mut generated: Vec<u32> = Vec::new();
    let mut next = boundary as u32;
    let mut next_logit = last_logits[boundary];
    for i in 0..max_tokens {
        let piece = tok.decode(&[next]);
        println!(
            "GEN {} token={} piece={} logit={:.4}",
            i,
            next,
            piece.trim(),
            next_logit
        );
        generated.push(next);
        if eos.contains(&next) {
            println!("EOS reached.");
            break;
        }
        if i + 1 == max_tokens {
            break;
        }
        let pos = tokens.len();
        let t1 = Instant::now();
        let logits = model.forward(next, pos)?;
        println!("  forward pos={} {:.2}s", pos, t1.elapsed().as_secs_f32());
        tokens.push(next);
        next = argmax(&logits).map_err(|e| format!("argmax: {}", e))? as u32;
        next_logit = logits[next as usize];
    }
    println!("GEN_IDS {:?}", generated);
    let output = tok.decode(&generated);
    println!("\n=== OUTPUT ===\n{}{}", prompt, output);
    println!("=== END ===");
    Ok(())
}
