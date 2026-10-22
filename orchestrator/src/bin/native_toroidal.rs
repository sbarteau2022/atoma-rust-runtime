//! Standalone Phase B training/evaluation/generation. See docs/NATIVE_NEXT_ATOM_DESIGN.md.
//! No server, checkpoint, vault, tokenizer lexicon, socket, or teacher is constructed.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use atoma_immune::refraction::Sha256Digest;
use atoma_orchestrator::native_predictor::{
    MAX_EXAMPLES, MAX_RECORD_BYTES, NativePredictorConfig, NativeTextRecord,
    NativeToroidalPredictor, NativeTrainingOptions, require_disjoint_corpora, write_new_atomic,
};
use serde::Deserialize;
use serde_json::{Value, json};

const HELP: &str = "native_toroidal train --train-text FILE [--train-text FILE ...] --eval-text FILE --artifact NEW_FILE --report NEW_FILE\n\
native_toroidal evaluate --artifact FILE --eval-jsonl FILE --report NEW_FILE [--require-exact]\n\
native_toroidal generate --artifact FILE --prompt TEXT [--max-new 128] [--report NEW_FILE]\n\
Datasets: --train-text/--eval-text are whole UTF-8 documents.\n\
--train-jsonl/--eval-jsonl contain one {\"prompt\":\"...\",\"completion\":\"...\"} per line.\n\
Training: --epochs 8 --batch-size 128 --learning-rate 0.003 --l2 0.0001 --seed 20261018 [--without-ordered-history].\n\
Reports/artifacts must not exist. The model always uses 52 native toroidal fields.";

#[derive(Default)]
struct Args {
    command: String,
    artifact: PathBuf,
    report: Option<PathBuf>,
    train: Vec<(PathBuf, bool)>,
    evaluation: Vec<(PathBuf, bool)>,
    prompt: Option<String>,
    options: NativeTrainingOptions,
    max_new: usize,
    require_exact: bool,
    without_ordered_history: bool,
}

fn parse() -> Result<Option<Args>, String> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        return Ok(None);
    };
    if command == "--help" || command == "-h" {
        return Ok(None);
    }
    if !["train", "evaluate", "generate"].contains(&command.as_str()) {
        return Err(format!("unknown command {command}"));
    }
    let mut output = Args {
        command,
        max_new: 128,
        ..Args::default()
    };
    let mut single = BTreeMap::new();
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        if flag == "--require-exact" {
            if output.require_exact {
                return Err("duplicate --require-exact".into());
            }
            output.require_exact = true;
            continue;
        }
        if flag == "--without-ordered-history" {
            if output.without_ordered_history || output.command != "train" {
                return Err(
                    "--without-ordered-history is train-only and cannot be repeated".into(),
                );
            }
            output.without_ordered_history = true;
            continue;
        }
        let repeated = [
            "--train-text",
            "--train-jsonl",
            "--eval-text",
            "--eval-jsonl",
        ]
        .contains(&flag.as_str());
        if !repeated && single.insert(flag.clone(), true).is_some() {
            return Err(format!("duplicate {flag}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--artifact" => output.artifact = value.into(),
            "--report" => output.report = Some(value.into()),
            "--train-text" | "--train-jsonl" => {
                output.train.push((value.into(), flag.ends_with("jsonl")))
            }
            "--eval-text" | "--eval-jsonl" => output
                .evaluation
                .push((value.into(), flag.ends_with("jsonl"))),
            "--prompt" => output.prompt = Some(value),
            "--max-new" => output.max_new = value.parse().map_err(|_| "invalid --max-new")?,
            "--epochs" => output.options.epochs = value.parse().map_err(|_| "invalid --epochs")?,
            "--batch-size" => {
                output.options.batch_size = value.parse().map_err(|_| "invalid --batch-size")?
            }
            "--learning-rate" => {
                output.options.learning_rate =
                    value.parse().map_err(|_| "invalid --learning-rate")?
            }
            "--l2" => output.options.l2 = value.parse().map_err(|_| "invalid --l2")?,
            "--seed" => output.options.seed = value.parse().map_err(|_| "invalid --seed")?,
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if output.artifact.as_os_str().is_empty() {
        return Err("--artifact is required".into());
    }
    if !(1..=4096).contains(&output.max_new) {
        return Err("--max-new must be in 1..=4096 bytes".into());
    }
    if output.command != "train"
        && (!output.train.is_empty()
            || single.keys().any(|flag| {
                [
                    "--epochs",
                    "--batch-size",
                    "--learning-rate",
                    "--l2",
                    "--seed",
                ]
                .contains(&flag.as_str())
            }))
    {
        return Err("training inputs/options require the train command".into());
    }
    match output.command.as_str() {
        "train" | "evaluate"
            if output.evaluation.is_empty()
                || output.report.is_none()
                || output.prompt.is_some() =>
        {
            return Err(
                "train/evaluate require evaluation data and --report, with no --prompt".into(),
            );
        }
        "train" if output.train.is_empty() => return Err("train requires training data".into()),
        "generate"
            if output.prompt.is_none() || !output.evaluation.is_empty() || output.require_exact =>
        {
            return Err(
                "generate requires --prompt and accepts no evaluation inputs or --require-exact"
                    .into(),
            );
        }
        _ => (),
    }
    if output.command == "train" {
        require_new(&output.artifact)?;
        if output
            .report
            .as_ref()
            .map(|path| Ok::<_, String>(destination(path)? == destination(&output.artifact)?))
            .transpose()?
            .unwrap_or(false)
        {
            return Err("artifact and report paths must differ".into());
        }
    }
    if let Some(path) = &output.report {
        require_new(path)?;
    }
    Ok(Some(output))
}

fn require_new(path: &Path) -> Result<(), String> {
    if path.file_name().is_none() {
        return Err("output must have a file name".into());
    }
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(format!("output already exists: {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            if !parent.is_dir() {
                return Err(format!(
                    "output directory does not exist: {}",
                    parent.display()
                ));
            }
            Ok(())
        }
        Err(error) => Err(error.to_string()),
    }
}

fn destination(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent
        .canonicalize()
        .map_err(|e| e.to_string())?
        .join(path.file_name().ok_or("output must have a file name")?))
}

fn bounded_read(path: &Path) -> Result<Vec<u8>, String> {
    const LIMIT: u64 = 2 * 1024 * 1024;
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .take(LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > LIMIT {
        return Err(format!("dataset file exceeds 2 MiB: {}", path.display()));
    }
    Ok(bytes)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    prompt: String,
    completion: String,
}

fn corpus(files: &[(PathBuf, bool)]) -> Result<(Vec<NativeTextRecord>, Vec<Value>), String> {
    let mut records = Vec::new();
    let mut sources = Vec::new();
    let mut total_bytes = 0usize;
    let mut positions = 0usize;
    let mut add = |record: NativeTextRecord| -> Result<(), String> {
        let bytes = record.prompt.len() + record.completion.len();
        total_bytes += bytes;
        positions += record.completion.len() + 1;
        if bytes > MAX_RECORD_BYTES || total_bytes > MAX_EXAMPLES || positions > MAX_EXAMPLES {
            return Err("native corpus exceeds record or corpus limits".into());
        }
        records.push(record);
        Ok(())
    };
    for (path, jsonl) in files {
        let bytes = bounded_read(path)?;
        sources.push(json!({"path": path, "sha256": Sha256Digest::compute(&bytes).to_hex(), "bytes": bytes.len(), "kind": if *jsonl { "prompt_completion_jsonl" } else { "utf8_document" }}));
        let text = String::from_utf8(bytes)
            .map_err(|e| format!("{} is not strict UTF-8: {e}", path.display()))?;
        let label = path
            .canonicalize()
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .to_string();
        if *jsonl {
            for (line, text) in text.lines().enumerate() {
                if text.trim().is_empty() {
                    continue;
                }
                let pair: Pair = serde_json::from_str(text)
                    .map_err(|e| format!("{}:{}: {e}", path.display(), line + 1))?;
                add(NativeTextRecord {
                    label: format!("{label}:{}", line + 1),
                    prompt: pair.prompt,
                    completion: pair.completion,
                })?;
            }
        } else {
            add(NativeTextRecord::document(label, text))?;
        }
    }
    Ok((records, sources))
}

fn exact_quality(evaluation: &atoma_orchestrator::native_predictor::EvaluationReceipt) -> Value {
    let cases = &evaluation.cases;
    json!({
        "cases": cases.len(),
        "exact_completion_and_eos": cases.iter().filter(|c| c.exact_completion_and_eos).count(),
        "reference_prefix_matches": cases.iter().filter(|c| c.matches_reference_prefix).count(),
        "eos_stops": cases.iter().filter(|c| c.generated.eos_id.is_some()).count(),
        "valid_utf8": cases.iter().filter(|c| c.generated.text.is_some()).count(),
        "all_exact_and_eos": !cases.is_empty() && cases.iter().all(|c| c.exact_completion_and_eos),
        "held_out_nll_better_than_bigram": evaluation.teacher_forced.learned.mean_nll_nats < evaluation.teacher_forced.bigram.mean_nll_nats,
        "toroidal_feature_nll_gain_nats": evaluation.teacher_forced.without_toroidal_features.mean_nll_nats - evaluation.teacher_forced.learned.mean_nll_nats,
        "ordered_history_nll_gain_nats": evaluation.teacher_forced.without_ordered_history.mean_nll_nats - evaluation.teacher_forced.learned.mean_nll_nats,
        "held_out_mean_z_surprise": evaluation.teacher_forced.learned.mean_z_surprise,
        "held_out_max_z_surprise": evaluation.teacher_forced.learned.max_z_surprise,
        "general_language_promotion_evaluated": false,
    })
}

fn run(args: Args) -> Result<i32, String> {
    if cfg!(feature = "glimmer") {
        return Err("native_toroidal requires a build without glimmer".into());
    }
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let executable_hash =
        Sha256Digest::compute(&std::fs::read(&executable).map_err(|e| e.to_string())?).to_hex();
    let (model, training, train_sources, eval_records, eval_sources) = if args.command == "train" {
        let (train, train_sources) = corpus(&args.train)?;
        let (evaluation, eval_sources) = corpus(&args.evaluation)?;
        require_disjoint_corpora(&train, &evaluation)?;
        let config = NativePredictorConfig {
            ordered_history: !args.without_ordered_history,
            ..NativePredictorConfig::default()
        };
        let (model, receipt) = NativeToroidalPredictor::fit(config, args.options, &train)?;
        model.save_new(&args.artifact)?;
        (
            model,
            Some(receipt),
            train_sources,
            evaluation,
            eval_sources,
        )
    } else {
        let model = NativeToroidalPredictor::load(&args.artifact)?;
        let (records, sources) = if args.command == "evaluate" {
            corpus(&args.evaluation)?
        } else {
            (Vec::new(), Vec::new())
        };
        (model, None, Vec::new(), records, sources)
    };
    let mut report = json!({
        "schema": "native-next-atom-report-v2", "command": args.command,
        "argv": std::env::args().collect::<Vec<_>>(),
        "created_unix_seconds": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_secs(),
        "os": std::env::consts::OS, "architecture": std::env::consts::ARCH,
        "executable_sha256": executable_hash,
        "source_sha256": model.source_sha256(),
        "artifact_path": args.artifact,
        "artifact_sha256": Sha256Digest::compute(&std::fs::read(&args.artifact).map_err(|e| e.to_string())?).to_hex(),
        "model_config": model.config(), "learned_parameters": model.parameter_count(),
        "training_provenance": model.training(), "training_sources": train_sources,
        "evaluation_sources": eval_sources, "training": training,
        "max_new_bytes": args.max_new,
        "teacher_attached": false, "checkpoint_loaded": false, "tokenizer_lexicon_loaded": false,
        "vault_loaded": false, "grammar_or_repetition_mask": false,
        "recurrence_trained": false, "readout_trained": true,
    });
    let mut failed = false;
    if args.command == "generate" {
        let prompt = args.prompt.as_deref().ok_or("missing prompt")?;
        report["prompt"] = json!(prompt);
        report["generation"] = serde_json::to_value(model.generate(prompt, args.max_new)?)
            .map_err(|e| e.to_string())?;
    } else {
        let evaluation = model.evaluate(&eval_records, args.max_new)?;
        let quality = exact_quality(&evaluation);
        failed = args.require_exact && quality["all_exact_and_eos"] != true;
        report["quality"] = quality;
        report["evaluation"] = serde_json::to_value(evaluation).map_err(|e| e.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    if let Some(path) = args.report {
        write_new_atomic(&path, &bytes)?;
        println!(
            "{}",
            json!({"report": path, "artifact": args.artifact, "quality": report.get("quality")})
        );
    } else {
        println!("{}", String::from_utf8(bytes).map_err(|e| e.to_string())?);
    }
    Ok(if failed { 2 } else { 0 })
}

fn main() {
    let result = match parse() {
        Ok(None) => {
            println!("{HELP}");
            return;
        }
        Ok(Some(args)) => run(args),
        Err(error) => Err(error),
    };
    match result {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("native_toroidal: {error}");
            std::process::exit(1);
        }
    }
}
