use core_lm::{glimmer_forward::GlimmerModel, metal::kernel_telemetry};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() % 2 != 0 {
        return Err("expected --model PATH [--warmup N] [--repeat N] [--cpu-only true]".into());
    }
    let get = |key: &str| args.chunks(2).find(|p| p[0] == key).map(|p| p[1].as_str());
    let model_dir = PathBuf::from(get("--model").ok_or("--model is required")?);
    let warmup: usize = get("--warmup").unwrap_or("2").parse()?;
    let repeat: usize = get("--repeat").unwrap_or("7").parse()?;
    let cpu_only = get("--cpu-only").unwrap_or("false").parse::<bool>()?;
    if repeat == 0 || repeat > 100 || warmup > 100 {
        return Err("warmup must be 0..100 and repeat must be 1..100".into());
    }

    let mut model = GlimmerModel::load(&model_dir)?;
    model.weights.retain_cache = false;
    if cpu_only {
        model.weights.use_metal = false;
    }
    let (rows, cols) = model.weights.weight_shape("lm_head.weight")?;
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let bits = model.weights.quant_bits("lm_head.weight");
    let group_size = config["quantization"]["lm_head"]["group_size"]
        .as_u64()
        .ok_or("missing lm_head group_size")? as usize;
    let hidden: Vec<f32> = (0..model.config.hidden_size)
        .map(|i| ((i as f32) * 0.017).sin() * 0.2)
        .collect();
    for _ in 0..warmup {
        let _ = model.decode_lm_head(&hidden)?;
    }
    let mut times = Vec::with_capacity(repeat);
    let mut reference: Option<Vec<f32>> = None;
    for _ in 0..repeat {
        let start = Instant::now();
        let logits = model.decode_lm_head(&hidden)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        if let Some(expected) = &reference {
            if logits.len() != expected.len() {
                return Err("lm_head returned inconsistent logit length".into());
            }
            let max_abs_error = logits
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if max_abs_error > 1e-5 {
                return Err(format!(
                    "repeated lm_head output changed; max abs error {max_abs_error}"
                )
                .into());
            }
        } else {
            reference = Some(logits);
        }
    }
    times.sort_by(f64::total_cmp);
    let logits = reference.ok_or("no timed logits")?;
    let metal_enabled_during_measurement = model.weights.use_metal;
    let measurement_telemetry = kernel_telemetry();
    let top1 = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i);
    model.weights.use_metal = false;
    let cpu_logits = model.decode_lm_head(&hidden)?;
    let max_abs_error = logits
        .iter()
        .zip(&cpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rel_l2 = logits
        .iter()
        .zip(&cpu_logits)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt()
        / logits
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
    let cpu_top1 = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i);
    let logical_cols = model.config.hidden_size;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model":model_dir,"tensor":"lm_head.weight","shape":[rows,logical_cols],
            "packed_shape":[rows,cols],
            "bits":bits,"group_size":group_size,"metal_enabled":metal_enabled_during_measurement,
            "warmup":warmup,"repeat":repeat,"median_ms":times[times.len()/2],
            "min_ms":times[0],"max_ms":times[times.len()-1],"samples_ms":times,
            "top1_token":top1,"cpu_control_top1_token":cpu_top1,
            "cpu_control_top1_agreement":top1 == cpu_top1,
            "cpu_control_max_abs_logit_error":max_abs_error,
            "cpu_control_relative_logit_l2":rel_l2,
            "kernel_telemetry":{
                "zero_copy_qmatvec_calls":measurement_telemetry.zero_copy_qmatvec_calls,
                "accelerate_sgemv_calls":measurement_telemetry.accelerate_sgemv_calls,
                "bytes_saved_from_heap_alloc":measurement_telemetry.bytes_saved_from_heap_alloc
            },
            "measurement":"head-only decode_lm_head including final RMSNorm, warmed; deterministic synthetic hidden state",
            "cpu_only":cpu_only
        }))?
    );
    Ok(())
}
