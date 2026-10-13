//! Generate binary: cargo run --bin generate -- "prompt text"
//!
//! Loads Qwen3.6-35B-A3B-OptiQ-4bit and generates with geometric memory.

use core_lm::ops::argmax;
use core_lm::qwen_forward::QwenModel;
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

    let model_dir = PathBuf::from("/Users/stewartbarteau/models/Qwen3.6-35B-A3B-OptiQ-4bit");

    println!("Loading tokenizer...");
    let tok = BpeTokenizer::from_json_file(&model_dir.join("tokenizer.json"))?;

    println!("Loading model (this takes a while)...");
    let t0 = Instant::now();
    let mut model = QwenModel::load(&model_dir)?;
    println!("Model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut tokens = tok.encode(prompt)?;
    println!("Prompt tokens: {:?}", tokens);

    let eos = model.config.eos_token_ids.clone();

    for pos in 0..tokens.len() {
        let logits = model.forward(tokens[pos], pos)?;
        let _ = logits;
    }
    println!("Prefill done.");

    let mut generated: Vec<u32> = Vec::new();
    for i in 0..max_tokens {
        let pos = tokens.len() + i;
        let last = *tokens.last().unwrap();
        let t1 = Instant::now();
        let logits = model.forward(last, pos)?;
        let next = argmax(&logits).map_err(|e| format!("argmax: {}", e))? as u32;
        let dt = t1.elapsed().as_secs_f32();
        println!(
            "  token {} ({}): {:.2}s",
            next,
            tok.decode(&[next]).trim(),
            dt
        );

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
