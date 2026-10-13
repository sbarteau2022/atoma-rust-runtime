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
    let model_dir = PathBuf::from("/Users/stewartbarteau/models/Muse-Glimmer-q4-MLX");
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
    println!(
        "Model loaded in {:.1}s (q_fused={}, layers={})",
        t0.elapsed().as_secs_f32(),
        model.q_fused,
        model.config.num_hidden_layers
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
    let mut generated: Vec<u32> = Vec::new();
    for i in 0..max_tokens {
        let pos = tokens.len() + i;
        let last = *tokens.last().unwrap();
        let t1 = Instant::now();
        let logits = model.forward(last, pos)?;
        let next = argmax(&logits).map_err(|e| format!("argmax: {}", e))? as u32;
        let dt = t1.elapsed().as_secs_f32();
        let piece = tok.decode(&[next]);
        println!("  token {} ({}): {:.2}s", next, piece.trim(), dt);
        if eos.contains(&next) {
            println!("EOS reached.");
            break;
        }
        tokens.push(next);
        generated.push(next);
    }
    let output = tok.decode(&generated);
    println!("\n=== OUTPUT ===\n{}{}", prompt, output);
    println!("=== END ===");
    Ok(())
}
