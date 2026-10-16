//! Paired, teacher-forced Glimmer qualification through the universal map.
use atoma_immune::geometry::bridge::f32_to_atom_address;
use atoma_immune::{FrozenBaseWeights, RoleId, Sha256Digest};
use atoma_orchestrator::{
    MonsterHybridServer,
    glimmer_map::{
        AttentionSnapshot, CentralMapAttention, RecallConfig, RecallMetrics, RecallMode,
    },
    toroidal_attention::{
        LinearMapValueAdapter, map_phase_coherence, normalized_power_weights, pair_coherence,
        radial_pair_toroidal_coherence, softmax_weights, weighted_values,
    },
};
use core_lm::{glimmer_forward::GlimmerModel, ops::argmax, tokenizer::BpeTokenizer};
use serde_json::json;
use std::{
    collections::BTreeSet, fs::OpenOptions, io::Write, path::PathBuf, sync::Arc, time::Instant,
};

struct Run {
    inputs: Vec<u32>,
    logits: Vec<Vec<f32>>,
    ttft_ms: f64,
    decode_ms: Vec<f64>,
    metrics: RecallMetrics,
    attention_trace: Vec<AttentionSnapshot>,
}

#[allow(clippy::too_many_arguments)]
fn run(
    model: &mut GlimmerModel,
    vault: &std::path::Path,
    base: Arc<FrozenBaseWeights>,
    prompt: &[u32],
    tokens: usize,
    teacher: Option<&[u32]>,
    config: RecallConfig,
    adapter: Option<LinearMapValueAdapter>,
    sequence: u64,
    capture_layer: usize,
) -> Result<Run, Box<dyn std::error::Error>> {
    let mut server = MonsterHybridServer::boot_default(base)?;
    server.load_or_init_persistent_vault(vault)?;
    let mut backend =
        CentralMapAttention::new(&mut server, RoleId::SensoryPerception, sequence, config)?;
    if let Some(a) = adapter {
        backend.set_adapter(a)?;
    }
    backend.capture_layer(capture_layer)?;
    if backend.indexed_slots() != 0 {
        return Err("qualification sequence already exists in vault".into());
    }
    let mut inputs = prompt.to_vec();
    let mut logits = Vec::new();
    let started = Instant::now();
    for (pos, &id) in prompt.iter().enumerate() {
        logits.push(model.forward_with_backend(id, pos, &mut backend)?);
    }
    let ttft_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut decode_ms = Vec::new();
    for _ in 1..tokens {
        let pos = inputs.len();
        let id = match teacher {
            Some(ids) => *ids.get(pos).ok_or("teacher trace too short")?,
            None => argmax(logits.last().ok_or("empty prompt")?)? as u32,
        };
        inputs.push(id);
        let started = Instant::now();
        logits.push(model.forward_with_backend(id, pos, &mut backend)?);
        decode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(Run {
        inputs,
        logits,
        ttft_ms,
        decode_ms,
        metrics: backend.metrics.clone(),
        attention_trace: backend.attention_trace.clone(),
    })
}

fn latency(run: &Run) -> serde_json::Value {
    let mut sorted = run.decode_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        sorted
            .get(((sorted.len() as f64 * p).ceil() as usize).saturating_sub(1))
            .copied()
    };
    let throughput = if sorted.is_empty() {
        None
    } else {
        Some(1000.0 * sorted.len() as f64 / sorted.iter().sum::<f64>())
    };
    json!({"ttft_ms":run.ttft_ms,"decode_step_ms":run.decode_ms,
        "decode_p50_ms":percentile(0.5),"decode_p95_ms":percentile(0.95),
        "decode_tokens_per_second":throughput,"map":run.metrics})
}

fn attention_prediction(
    samples: &[AttentionSnapshot],
    index: usize,
    mode: RecallMode,
    gain: f64,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let current = &samples[index];
    let mut output = Vec::with_capacity(current.query.len());
    for qh in 0..current.query_heads {
        let kvh = qh / (current.query_heads / current.kv_heads);
        let query = &current.query[qh * current.head_dim..(qh + 1) * current.head_dim];
        let (_, query_phase, _) = f32_to_atom_address(query)?;
        let history: Vec<_> = samples[..=index]
            .iter()
            .filter(|s| (current.window_start..=current.position).contains(&s.position))
            .collect();
        let mut scores = Vec::with_capacity(history.len());
        let mut values = Vec::with_capacity(history.len());
        let mut nearest: Option<(f64, Vec<f32>)> = None;
        for item in history {
            let key = &item.key[kvh * current.head_dim..(kvh + 1) * current.head_dim];
            let value = item.value[kvh * current.head_dim..(kvh + 1) * current.head_dim].to_vec();
            let phase = item.key_phases[kvh];
            let score = match mode {
                RecallMode::PairAngles => pair_coherence(query, key)?,
                RecallMode::RadialToroidal | RecallMode::AdaptedMapValue => {
                    radial_pair_toroidal_coherence(query, key, current.dot_scale)?
                }
                RecallMode::MapT2 => map_phase_coherence(query_phase, phase),
                RecallMode::Direct => {
                    let d = 2.0
                        - (query_phase.theta_f.to_f64() - phase.theta_f.to_f64()).cos()
                        - (query_phase.theta_s.to_f64() - phase.theta_s.to_f64()).cos();
                    if nearest.as_ref().is_none_or(|(old, _)| d < *old) {
                        nearest = Some((d, value));
                    }
                    continue;
                }
                RecallMode::Baseline => {
                    query
                        .iter()
                        .zip(key)
                        .map(|(&q, &k)| q as f64 * k as f64)
                        .sum::<f64>()
                        * current.dot_scale
                }
            };
            scores.push(score);
            values.push(value);
        }
        let head = if mode == RecallMode::Direct {
            nearest.ok_or("direct snapshot has no values")?.1
        } else {
            let weights = if matches!(
                mode,
                RecallMode::Baseline | RecallMode::RadialToroidal | RecallMode::AdaptedMapValue
            ) {
                softmax_weights(&scores)
            } else {
                normalized_power_weights(&scores, gain)
            };
            weighted_values(&weights, &values, current.head_dim)
        };
        output.extend(head);
    }
    Ok(output)
}

fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let n = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let d = a
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        .max(1e-12);
    n / d
}

fn solve_linear_adapter(
    training: &[Run],
    head_dim: usize,
) -> Result<LinearMapValueAdapter, Box<dyn std::error::Error>> {
    let dim = head_dim * 3 + 1;
    let mut gram = vec![vec![0.0f64; dim]; dim];
    let mut cross = vec![vec![0.0f64; head_dim]; dim];
    let mut samples = 0usize;
    for run in training {
        let trace = &run.attention_trace;
        for (i, snapshot) in trace.iter().enumerate().skip(1) {
            let direct = attention_prediction(trace, i, RecallMode::Direct, 1.0)?;
            let radial_tor = attention_prediction(trace, i, RecallMode::RadialToroidal, 1.0)?;
            for qh in 0..snapshot.query_heads {
                let q = &snapshot.query[qh * head_dim..(qh + 1) * head_dim];
                let d = &direct[qh * head_dim..(qh + 1) * head_dim];
                let rt = &radial_tor[qh * head_dim..(qh + 1) * head_dim];
                let y = &snapshot.baseline_output[qh * head_dim..(qh + 1) * head_dim];
                let mut x: Vec<f64> = q
                    .iter()
                    .chain(d)
                    .chain(rt)
                    .map(|v| *v as f64)
                    .collect();
                x.push(1.0);
                for r in 0..dim {
                    for c in 0..dim {
                        gram[r][c] += x[r] * x[c];
                    }
                    for o in 0..head_dim {
                        cross[r][o] += x[r] * y[o] as f64;
                    }
                }
                samples += 1;
            }
        }
    }
    if samples == 0 {
        return Err("map adapter needs at least one training head sample".into());
    }
    let mean_diag = (0..dim - 1).map(|i| gram[i][i]).sum::<f64>() / (dim - 1) as f64;
    let regularization = (mean_diag / samples as f64).max(1e-12) * 1e-3;
    for i in 0..dim - 1 {
        gram[i][i] += regularization;
    }
    let mut augmented = vec![vec![0.0f64; dim + head_dim]; dim];
    for r in 0..dim {
        augmented[r][..dim].copy_from_slice(&gram[r]);
        augmented[r][dim..].copy_from_slice(&cross[r]);
    }
    for pivot in 0..dim {
        let best = (pivot..dim)
            .max_by(|&a, &b| {
                augmented[a][pivot]
                    .abs()
                    .total_cmp(&augmented[b][pivot].abs())
            })
            .ok_or("singular adapter normal equations")?;
        if augmented[best][pivot].abs() < 1e-14 {
            return Err("singular adapter normal equations".into());
        }
        augmented.swap(pivot, best);
        let scale = augmented[pivot][pivot];
        for c in pivot..dim + head_dim {
            augmented[pivot][c] /= scale;
        }
        for row in 0..dim {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            for c in pivot..dim + head_dim {
                augmented[row][c] -= factor * augmented[pivot][c];
            }
        }
    }
    let weights = (0..dim).map(|r| augmented[r][dim..].to_vec()).collect();
    Ok(LinearMapValueAdapter {
        input_dim: dim,
        head_dim,
        weights,
        regularization,
    })
}

fn evaluate_linear_adapter(
    adapter: &LinearMapValueAdapter,
    runs: &[Run],
) -> Result<f64, Box<dyn std::error::Error>> {
    let mut errors = Vec::new();
    for run in runs {
        let trace = &run.attention_trace;
        for (i, snapshot) in trace.iter().enumerate().skip(1) {
            let direct = attention_prediction(trace, i, RecallMode::Direct, 1.0)?;
            let radial_tor = attention_prediction(trace, i, RecallMode::RadialToroidal, 1.0)?;
            for qh in 0..snapshot.query_heads {
                let q = &snapshot.query[qh * adapter.head_dim..(qh + 1) * adapter.head_dim];
                let d = &direct[qh * adapter.head_dim..(qh + 1) * adapter.head_dim];
                let rt = &radial_tor[qh * adapter.head_dim..(qh + 1) * adapter.head_dim];
                let y =
                    &snapshot.baseline_output[qh * adapter.head_dim..(qh + 1) * adapter.head_dim];
                let prediction = adapter.predict_head(q, d, Some(rt))?;
                errors.push(relative_l2(y, &prediction));
            }
        }
    }
    if errors.is_empty() {
        return Err("no adapter evaluation samples".into());
    }
    Ok(errors.iter().sum::<f64>() / errors.len() as f64)
}

fn evaluate_direct(runs: &[Run]) -> Result<f64, Box<dyn std::error::Error>> {
    let mut errors = Vec::new();
    for run in runs {
        let trace = &run.attention_trace;
        for (i, snapshot) in trace.iter().enumerate().skip(1) {
            let prediction = attention_prediction(trace, i, RecallMode::Direct, 1.0)?;
            errors.push(relative_l2(&snapshot.baseline_output, &prediction));
        }
    }
    if errors.is_empty() {
        return Err("no direct-recall evaluation samples".into());
    }
    Ok(errors.iter().sum::<f64>() / errors.len() as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help") {
        println!(
            "qualify_map_glimmer --model PATH --vault PATH [--prompt TEXT ...] [--tokens 4] [--layer 0|all|0,4] [--mode map-t2|direct|p2|radial-toroidal|adapted] [--fit-map-adapter true --adapter-output NEW_PATH] [--gain 1] [--candidates 64]"
        );
        return Ok(());
    }
    let allowed = [
        "--model",
        "--vault",
        "--prompt",
        "--tokens",
        "--layer",
        "--mode",
        "--gain",
        "--candidates",
        "--fit-map-adapter",
        "--adapter-output",
    ];
    if !args.len().is_multiple_of(2) || args.chunks(2).any(|p| !allowed.contains(&p[0].as_str())) {
        return Err("expected option/value pairs; see --help".into());
    }
    let get = |key: &str| args.chunks(2).find(|p| p[0] == key).map(|p| p[1].as_str());
    let model_dir = PathBuf::from(get("--model").ok_or("--model is required")?);
    let vault = PathBuf::from(get("--vault").ok_or("--vault is required")?);
    if !vault.is_file() {
        return Err("--vault must be the existing universal server map".into());
    }
    let tokens: usize = get("--tokens").unwrap_or("4").parse()?;
    if !(1..=128).contains(&tokens) {
        return Err("tokens must be 1..128".into());
    }
    let tokenizer = BpeTokenizer::from_json_file(&model_dir.join("tokenizer.json"))?;
    let started = Instant::now();
    let mut model = GlimmerModel::load(&model_dir)?;
    model.weights.retain_cache = false;
    let model_load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let prompt_texts: Vec<String> = args
        .chunks(2)
        .filter(|p| p[0] == "--prompt")
        .map(|p| p[1].clone())
        .collect();
    let prompt_texts = if prompt_texts.is_empty() {
        vec!["The capital of France is".to_string()]
    } else {
        prompt_texts
    };
    let fit_adapter = get("--fit-map-adapter")
        .unwrap_or("false")
        .parse::<bool>()?;
    if get("--fit-map-adapter").is_some() && !fit_adapter {
        return Err("--fit-map-adapter must be true when specified".into());
    }
    if fit_adapter && prompt_texts.len() < 2 {
        return Err("map adapter fitting needs repeated --prompt values: fit prompts first, held-out prompts last".into());
    }
    let mut prompt = tokenizer.encode(&prompt_texts[0])?;
    prompt.insert(0, model.config.bos_token_id);
    if prompt.len() + tokens - 1 > 128 {
        return Err("qualification is limited to 128 total input tokens".into());
    }
    let layers: BTreeSet<usize> = match get("--layer").unwrap_or("0") {
        "all" => (0..model.config.num_hidden_layers).collect(),
        s => s.split(',').map(str::parse).collect::<Result<_, _>>()?,
    };
    let mode = match get("--mode").unwrap_or("map-t2") {
        "map-t2" => RecallMode::MapT2,
        "direct" => RecallMode::Direct,
        "p2" => RecallMode::PairAngles,
        "radial-toroidal" => RecallMode::RadialToroidal,
        "adapted" => RecallMode::AdaptedMapValue,
        _ => return Err("invalid mode".into()),
    };
    let config = RecallConfig {
        mode,
        layers,
        gain: get("--gain").unwrap_or("1").parse()?,
        candidate_limit: get("--candidates").unwrap_or("64").parse()?,
    };
    let capture_layer = config.layers.iter().next().copied().unwrap_or(0);
    // Freeze the server base using the real cryptographic SHA-256 structural fingerprint
    // of the loaded Glimmer checkpoint tensors.
    let sequence = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
    )?;
    let checkpoint_fp =
        Sha256Digest::compute(&model.weights.structural_fingerprint().to_le_bytes());
    let base = FrozenBaseWeights::freeze_with_fingerprint(
        "Muse-Glimmer-q4km-gs128-MLX",
        model.config.num_hidden_layers,
        Vec::new(),
        checkpoint_fp,
    );
    if fit_adapter {
        let mut prompt_runs = Vec::with_capacity(prompt_texts.len());
        for (index, text) in prompt_texts.iter().enumerate() {
            let mut ids = tokenizer.encode(text)?;
            ids.insert(0, model.config.bos_token_id);
            if ids.len() + tokens - 1 > 128 {
                return Err(
                    format!("prompt {index} exceeds the 128-token qualification cap").into(),
                );
            }
            prompt_runs.push(run(
                &mut model,
                &vault,
                base.clone(),
                &ids,
                tokens,
                None,
                RecallConfig {
                    mode: RecallMode::Baseline,
                    ..config.clone()
                },
                None,
                sequence.wrapping_add(index as u64),
                capture_layer,
            )?);
        }
        let train_count = (prompt_runs.len() * 3 / 4).clamp(1, prompt_runs.len() - 1);
        let adapter = solve_linear_adapter(&prompt_runs[..train_count], model.config.head_dim)?;
        let direct_holdout = evaluate_direct(&prompt_runs[train_count..])?;
        let learned_holdout = evaluate_linear_adapter(&adapter, &prompt_runs[train_count..])?;
        let train_error = evaluate_linear_adapter(&adapter, &prompt_runs[..train_count])?;

        // Run the fitted adapter LIVE inside the real forward pass on held-out prompts
        let mut heldout_logit_l2s = Vec::new();
        let mut heldout_argmax_matches = 0usize;
        let mut heldout_total_steps = 0usize;
        for (h_idx, base_run) in prompt_runs[train_count..].iter().enumerate() {
            let prompt_len = base_run.inputs.len().saturating_sub(tokens.saturating_sub(1));
            let live_run = run(
                &mut model,
                &vault,
                base.clone(),
                &base_run.inputs[..prompt_len],
                tokens,
                Some(&base_run.inputs),
                RecallConfig {
                    mode: RecallMode::AdaptedMapValue,
                    ..config.clone()
                },
                Some(adapter.clone()),
                sequence.wrapping_add(10_000 + h_idx as u64),
                capture_layer,
            )?;
            for (b_log, c_log) in base_run.logits.iter().zip(&live_run.logits) {
                heldout_logit_l2s.push(relative_l2(b_log, c_log));
                if argmax(b_log)? == argmax(c_log)? {
                    heldout_argmax_matches += 1;
                }
                heldout_total_steps += 1;
            }
        }
        let mean_heldout_logit_l2 = if heldout_logit_l2s.is_empty() {
            0.0
        } else {
            heldout_logit_l2s.iter().sum::<f64>() / heldout_logit_l2s.len() as f64
        };
        let heldout_argmax_agreement = if heldout_total_steps == 0 {
            1.0
        } else {
            heldout_argmax_matches as f64 / heldout_total_steps as f64
        };

        let saved_to = if let Some(path) = get("--adapter-output") {
            let path = PathBuf::from(path);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            serde_json::to_writer_pretty(&mut file, &adapter)?;
            file.flush()?;
            Some(path)
        } else {
            None
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "prototype":"multi-channel ridge-linear map-value adapter (live-wired)",
                "checkpoint_structural_fingerprint":checkpoint_fp.to_hex(),
                "layer":capture_layer,
                "training_prompts":train_count,
                "heldout_prompts":prompt_runs.len()-train_count,
                "training_head_samples":prompt_runs[..train_count].iter().map(|r| r.attention_trace.iter().skip(1).map(|s| s.query_heads).sum::<usize>()).sum::<usize>(),
                "feature":"query head concatenated with direct phase-recalled map value + radial-toroidal phase-coherent map value plus bias",
                "fit_mean_attention_relative_l2":train_error,
                "heldout_direct_mean_attention_relative_l2":direct_holdout,
                "heldout_learned_mean_attention_relative_l2":learned_holdout,
                "heldout_live_forward_mean_logit_relative_l2":mean_heldout_logit_l2,
                "heldout_live_forward_argmax_agreement":heldout_argmax_agreement,
                "regularization":adapter.regularization,
                "adapter_output":saved_to,
                "status":"wired into CentralMapAttention (RecallMode::AdaptedMapValue) and verified in live forward pass"
            }))?
        );
        return Ok(());
    }
    eprintln!(
        "Warming checkpoint, then running baseline and candidate on the same token trace; load {model_load_ms:.1} ms"
    );
    let warmup = run(
        &mut model,
        &vault,
        base.clone(),
        &prompt[..1],
        1,
        None,
        RecallConfig {
            mode: RecallMode::Baseline,
            ..config.clone()
        },
        None,
        sequence,
        capture_layer,
    )?;
    let baseline = run(
        &mut model,
        &vault,
        base.clone(),
        &prompt,
        tokens,
        None,
        RecallConfig {
            mode: RecallMode::Baseline,
            ..config.clone()
        },
        None,
        sequence,
        capture_layer,
    )?;
    let candidate = run(
        &mut model,
        &vault,
        base,
        &prompt,
        tokens,
        Some(&baseline.inputs),
        config.clone(),
        None,
        sequence,
        capture_layer,
    )?;
    let errors: Vec<f64> = baseline
        .logits
        .iter()
        .zip(&candidate.logits)
        .map(|(a, b)| {
            let numerator = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let denominator = a
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            numerator / denominator
        })
        .collect();
    let baseline_choices: Vec<_> = baseline
        .logits
        .iter()
        .map(|x| argmax(x))
        .collect::<Result<_, _>>()?;
    let candidate_choices: Vec<_> = candidate
        .logits
        .iter()
        .map(|x| argmax(x))
        .collect::<Result<_, _>>()?;
    let agreements = baseline_choices
        .iter()
        .zip(&candidate_choices)
        .filter(|(a, b)| a == b)
        .count();
    let layer_errors: Vec<f64> = baseline
        .attention_trace
        .iter()
        .zip(&candidate.attention_trace)
        .map(|(a, b)| relative_l2(&a.baseline_output, &b.baseline_output))
        .collect();
    let samples = &baseline.attention_trace;
    let split = (samples.len() / 2).max(1);
    let gains = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let mut gain_fit = Vec::new();
    for gain in gains {
        let train: Vec<f64> = (1..split)
            .map(|i| {
                attention_prediction(samples, i, mode, gain)
                    .map(|pred| relative_l2(&samples[i].baseline_output, &pred))
            })
            .collect::<Result<_, _>>()?;
        let heldout: Vec<f64> = (split..samples.len())
            .map(|i| {
                attention_prediction(samples, i, mode, gain)
                    .map(|pred| relative_l2(&samples[i].baseline_output, &pred))
            })
            .collect::<Result<_, _>>()?;
        gain_fit.push(json!({"gain":gain,
            "train_mean_relative_l2":if train.is_empty(){None}else{Some(train.iter().sum::<f64>()/train.len() as f64)},
            "heldout_mean_relative_l2":if heldout.is_empty(){None}else{Some(heldout.iter().sum::<f64>()/heldout.len() as f64)}}));
    }
    let best_gain_fit = gain_fit
        .iter()
        .filter_map(|row| Some((row["train_mean_relative_l2"].as_f64()?, row["gain"].clone())))
        .min_by(|a, b| a.0.total_cmp(&b.0));
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model":model_dir,"vault":vault,"mode":mode,"layers":config.layers,"gain":config.gain,
            "checkpoint_structural_fingerprint":checkpoint_fp.to_hex(),
            "candidate_limit":config.candidate_limit,"model_load_ms":model_load_ms,"warmup_ms":warmup.ttft_ms,
            "measurement":"single warmed baseline-first paired teacher-forced run; model/vault loading excluded from TTFT; no promotion",
            "input_ids":baseline.inputs,"baseline_argmax":baseline_choices,"candidate_argmax":candidate_choices,
            "argmax_agreement":agreements as f64 / errors.len() as f64,"relative_logit_l2":errors,
            "captured_layer":capture_layer,"attention_output_relative_l2":layer_errors,
            "offline_attention_gain_grid_train_first_half_holdout_second_half":gain_fit,
            "best_train_gain":best_gain_fit.map(|(_, gain)| gain),
            "baseline":latency(&baseline),"candidate":latency(&candidate),
            "ttft_reduction_fraction":1.0-candidate.ttft_ms/baseline.ttft_ms
        }))?
    );
    Ok(())
}
