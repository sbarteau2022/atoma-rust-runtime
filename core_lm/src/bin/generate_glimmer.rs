//! Generate binary for Muse Glimmer with geometric memory.
//!
//! Usage: cargo run --bin generate_glimmer -- "prompt text" [max_tokens]

use core_lm::glimmer_forward::GlimmerModel;
use core_lm::ops::argmax;
use core_lm::tokenizer::BpeTokenizer;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("The capital of France is");
    let max_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let model_dir = PathBuf::from("/Users/stewartbarteau/models/Muse-Glimmer-q4-MLX");
    println!("Loading tokenizer...");
    let tok = BpeTokenizer::from_json_file(&model_dir.join("tokenizer.json"))?;
    println!("Loading Glimmer model (this takes a while)...");
    let t0 = Instant::now();
    let mut model = GlimmerModel::load(&model_dir)?;
    println!(
        "Model loaded in {:.1}s (q_fused={}, layers={})",
        t0.elapsed().as_secs_f32(),
        model.q_fused,
        model.config.num_hidden_layers
    );
    let mut tokens = tok.encode(prompt)?;
    tokens.insert(0, model.config.bos_token_id);
    println!(
        "Prompt tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(10)]
    );
    let eos = model.config.eos_token_ids.clone();
    let t0 = Instant::now();
    for pos in 0..tokens.len() {
        let logits = model.forward(tokens[pos], pos)?;
        let _ = logits;
        if (pos + 1) % 8 == 0 || pos + 1 == tokens.len() {
            println!("  prefill {}/{} ...", pos + 1, tokens.len());
        }
    }
    println!("Prefill done in {:.1}s.", t0.elapsed().as_secs_f32());
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
