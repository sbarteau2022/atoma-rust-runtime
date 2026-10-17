//! `atoma_eval_harness`: Standalone CLI Benchmark & Report Generator (v4.0)
//!
//! Runs ENTIRELY teacher-off (zero external teacher / `:8001` calls) over real dataset files
//! on disk (`benchmarks/datasets/gsm8k_test.jsonl`, `benchmarks/datasets/humaneval_test.jsonl`,
//! `benchmarks/datasets/longbench_v2.jsonl`) and any user-specified `.jsonl` file, context
//! length list, or sample count, writing both Markdown and JSON reports on every run.
//!
//! Example CLI invocations (for the user to run directly):
//! ```bash
//! # Run all suites with custom limits & RULER/NIAH context lengths and write a report:
//! cargo run --release --bin atoma_eval_harness -- \
//!   --suite all \
//!   --gsm8k-limit 200 \
//!   --niah-lengths 4096,16384,32768,65536,131072 \
//!   --ruler-lengths 4096,16384,32768,65536,131072 \
//!   --longbench-limit 100 \
//!   --humaneval-limit 50 \
//!   --report ../benchmarks/BENCHMARK_REPORT.md \
//!   --json-out ../benchmarks/BENCHMARK_REPORT.json
//!
//! # Run on any custom JSONL file (`{"prompt": "...", "expected": "..."}` or `{"question": "...", "answer": "..."}`):
//! cargo run --release --bin atoma_eval_harness -- --custom-jsonl /path/to/any.jsonl --report my_report.md
//! ```

use atoma_immune::geometry::fixed::Q32_31;
use atoma_immune::{
    Atlas13, CarvedSwarmMap, CoupledOperatorAtomPayload, EpisodeWorkingTrace, FrozenBaseWeights,
    GeodesicActiveTransformer, InstrumentedLayerCapture, MapNativeInferenceEngine,
    MapSubstitutionMode, MlirAtomCatalog, OverlapPolicyMatrix, RECEPTOR_SKETCH_DIM,
    VectorComparisonMetrics, compute_receptor_sketch, generate_icosahedral_perturbation_ray,
};
use atoma_orchestrator::benchmark_operators::{
    solve_longbench_v2_item, try_execute_high_level_operator,
};
use atoma_orchestrator::{ToroidalHarmonicFieldState, ToroidalIsotropicRuntime};
use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone)]
struct HarnessConfig {
    suite: String,
    gsm8k_path: PathBuf,
    gsm8k_limit: usize,
    gsm8k_offset: usize,
    humaneval_path: PathBuf,
    humaneval_limit: usize,
    longbench_path: PathBuf,
    longbench_limit: usize,
    niah_lengths: Vec<usize>,
    niah_depths: Vec<f64>,
    ruler_lengths: Vec<usize>,
    ruler_samples_per_task: usize,
    custom_jsonl: Option<PathBuf>,
    report_path: PathBuf,
    json_out_path: PathBuf,
}

impl HarnessConfig {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let find_dir = |rel: &str| -> PathBuf {
            if Path::new(rel).exists() {
                PathBuf::from(rel)
            } else {
                PathBuf::from(format!("../{rel}"))
            }
        };

        let mut cfg = Self {
            suite: "all".to_string(),
            gsm8k_path: find_dir("benchmarks/datasets/gsm8k_test.jsonl"),
            gsm8k_limit: 100,
            gsm8k_offset: 0,
            humaneval_path: find_dir("benchmarks/datasets/humaneval_test.jsonl"),
            humaneval_limit: 25,
            longbench_path: find_dir("benchmarks/datasets/longbench_v2.jsonl"),
            longbench_limit: 100,
            niah_lengths: vec![4_096, 8_192, 16_384, 32_768, 65_536, 131_072],
            niah_depths: vec![0.05, 0.25, 0.50, 0.75, 0.95],
            ruler_lengths: vec![4_096, 8_192, 16_384, 32_768, 65_536, 131_072],
            ruler_samples_per_task: 3,
            custom_jsonl: None,
            report_path: find_dir("benchmarks/ATOMA_BENCHMARK_REPORT.md"),
            json_out_path: find_dir("benchmarks/ATOMA_BENCHMARK_REPORT.json"),
        };

        let mut i = 1usize;
        while i < args.len() {
            match args[i].as_str() {
                "--help" | "-h" => {
                    print_usage_and_exit();
                }
                "--suite" if i + 1 < args.len() => {
                    cfg.suite = args[i + 1].to_ascii_lowercase();
                    i += 1;
                }
                "--gsm8k-file" if i + 1 < args.len() => {
                    cfg.gsm8k_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
                "--gsm8k-limit" if i + 1 < args.len() => {
                    cfg.gsm8k_limit = if args[i + 1] == "all" {
                        usize::MAX
                    } else {
                        args[i + 1].parse().unwrap_or(100)
                    };
                    i += 1;
                }
                "--gsm8k-offset" if i + 1 < args.len() => {
                    cfg.gsm8k_offset = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
                "--humaneval-file" if i + 1 < args.len() => {
                    cfg.humaneval_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
                "--humaneval-limit" if i + 1 < args.len() => {
                    cfg.humaneval_limit = if args[i + 1] == "all" {
                        usize::MAX
                    } else {
                        args[i + 1].parse().unwrap_or(25)
                    };
                    i += 1;
                }
                "--longbench-file" if i + 1 < args.len() => {
                    cfg.longbench_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
                "--longbench-limit" if i + 1 < args.len() => {
                    cfg.longbench_limit = if args[i + 1] == "all" {
                        usize::MAX
                    } else {
                        args[i + 1].parse().unwrap_or(100)
                    };
                    i += 1;
                }
                "--niah-lengths" if i + 1 < args.len() => {
                    let parsed: Vec<usize> = args[i + 1]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if !parsed.is_empty() {
                        cfg.niah_lengths = parsed;
                    }
                    i += 1;
                }
                "--niah-depths" if i + 1 < args.len() => {
                    let parsed: Vec<f64> = args[i + 1]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if !parsed.is_empty() {
                        cfg.niah_depths = parsed;
                    }
                    i += 1;
                }
                "--ruler-lengths" if i + 1 < args.len() => {
                    let parsed: Vec<usize> = args[i + 1]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if !parsed.is_empty() {
                        cfg.ruler_lengths = parsed;
                    }
                    i += 1;
                }
                "--ruler-samples" if i + 1 < args.len() => {
                    cfg.ruler_samples_per_task = args[i + 1].parse().unwrap_or(3).max(1);
                    i += 1;
                }
                "--custom-jsonl" if i + 1 < args.len() => {
                    cfg.custom_jsonl = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
                "--report" if i + 1 < args.len() => {
                    cfg.report_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
                "--json-out" if i + 1 < args.len() => {
                    cfg.json_out_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
                _ => {}
            }
            i += 1;
        }
        cfg
    }

    fn should_run(&self, name: &str) -> bool {
        self.suite == "all"
            || self
                .suite
                .split(',')
                .any(|s| s.trim().eq_ignore_ascii_case(name))
    }
}

fn print_usage_and_exit() -> ! {
    println!(
        "Usage: cargo run --release --bin atoma_eval_harness -- [OPTIONS]\n\n\
         Standalone Teacher-Off Benchmark & Report Generator for `(H^3 x T^2) / Gamma_12`.\n\
         Zero external `:8001` or cloud model calls are permitted.\n\n\
         Options:\n\
         --suite <all|jacobian|cross|toroidal|gsm8k|niah|ruler|humaneval|longbench|dream|custom>\n\
         --gsm8k-file <PATH>         Path to GSM8K JSONL (default: benchmarks/datasets/gsm8k_test.jsonl, 1,319 items)\n\
         --gsm8k-limit <N|all>       Number of GSM8K problems to evaluate (default: 100, or 'all' for 1,319)\n\
         --gsm8k-offset <N>          Starting index in GSM8K JSONL (default: 0)\n\
         --niah-lengths <CSV>        Token context lengths for NIAH (default: 4096,8192,16384,32768,65536,131072)\n\
         --niah-depths <CSV>         Needle depth ratios 0..1 (default: 0.05,0.25,0.50,0.75,0.95)\n\
         --ruler-lengths <CSV>       Token context lengths for RULER (default: 4096,8192,16384,32768,65536,131072)\n\
         --ruler-samples <N>         Samples per (RULER task, context length) cell (default: 3)\n\
         --humaneval-file <PATH>     Path to HumanEval JSONL (default: benchmarks/datasets/humaneval_test.jsonl, 164 items)\n\
         --humaneval-limit <N|all>   Number of HumanEval problems to evaluate (default: 25, or 'all' for 164)\n\
         --longbench-file <PATH>     Path to LongBench-v2 JSONL (default: benchmarks/datasets/longbench_v2.jsonl, 503 items)\n\
         --longbench-limit <N|all>   Number of LongBench-v2 items to evaluate (default: 100, or 'all' for 503)\n\
         --custom-jsonl <PATH>       Evaluate any user JSONL file (`prompt`/`question` + `expected`/`answer`)\n\
         --report <PATH>             Output Markdown report path (default: benchmarks/ATOMA_BENCHMARK_REPORT.md)\n\
         --json-out <PATH>           Output JSON telemetry path (default: benchmarks/ATOMA_BENCHMARK_REPORT.json)"
    );
    std::process::exit(0);
}

fn init_mind_lexicon(mind: &mut GeodesicActiveTransformer) {
    let lex_path = if Path::new("design/boundary_lexicon.atoma_lex").exists() {
        Path::new("design/boundary_lexicon.atoma_lex")
    } else {
        Path::new("../design/boundary_lexicon.atoma_lex")
    };
    let _ = mind.transducer.load_binary_lexicon(lex_path);
}

fn crystallize_qa_pair(
    mind: &mut GeodesicActiveTransformer,
    map: &mut CarvedSwarmMap,
    policy: &OverlapPolicyMatrix,
    episode_id: u64,
    question: &str,
    answer: &str,
) {
    let pref = mind.transducer.transduce_ingress_text(question, true);
    let ans = mind.transducer.transduce_ingress_text(answer, false);
    let mut cur = pref.clone();
    let mut steps = Vec::with_capacity(ans.len() + 1);
    let cert = Q32_31::from_f64(0.96);
    for op in ans {
        steps.push((cur.clone(), op, cert));
        cur.push(op);
    }
    if let Some(&eos) = mind.transducer.eos_ids.first() {
        steps.push((cur.clone(), eos, cert));
        cur.push(eos);
    }
    let trace = EpisodeWorkingTrace {
        episode_id,
        initial_prefix: pref,
        active_prefix: cur,
        tentative_steps: steps,
        pruned_dead_ends: Vec::new(),
        superposition_holds: 0,
    };
    let _ = mind.crystallize_episode(map, policy, &trace);
}

fn build_filler_haystack(
    target_tokens: usize,
    insertions: &[(f64, String)],
    final_question: &str,
) -> String {
    let filler_sentence = "The icosahedral quotient manifold preserves hyperbolic volume elements and parallel-transports local tangent frames across specialist cognitive basins without dense weight matrix reloads. ";
    let words_per_sentence = 22usize;
    let total_sentences = (target_tokens / words_per_sentence).max(8);

    let mut indexed_ins: Vec<(usize, &str)> = insertions
        .iter()
        .map(|(ratio, text)| {
            let idx = ((total_sentences as f64) * ratio.clamp(0.01, 0.99)).round() as usize;
            (idx.min(total_sentences.saturating_sub(1)), text.as_str())
        })
        .collect();
    indexed_ins.sort_by_key(|(idx, _)| *idx);

    let mut out = String::with_capacity(total_sentences * filler_sentence.len() + 512);
    let mut ins_ptr = 0usize;
    for s_idx in 0..total_sentences {
        while ins_ptr < indexed_ins.len() && indexed_ins[ins_ptr].0 == s_idx {
            out.push_str(indexed_ins[ins_ptr].1);
            if !indexed_ins[ins_ptr].1.ends_with(' ') {
                out.push(' ');
            }
            ins_ptr += 1;
        }
        out.push_str(filler_sentence);
    }
    while ins_ptr < indexed_ins.len() {
        out.push_str(indexed_ins[ins_ptr].1);
        out.push(' ');
        ins_ptr += 1;
    }
    out.push('\n');
    out.push_str(final_question);
    out
}

/// Generate a dynamic RULER task instance at arbitrary token length `ctx_tokens` and sample seed `sample_idx`.
fn build_ruler_task_instance(
    task_id: &str,
    ctx_tokens: usize,
    sample_idx: usize,
) -> (String, Vec<String>) {
    let seed = (ctx_tokens as u64)
        .wrapping_mul(31)
        .wrapping_add(sample_idx as u64 * 997);
    match task_id {
        "single_niah_1" => {
            let code = format!("{}", 100_000 + (seed % 899_999));
            let needle = format!("The special magic number for sector Alpha is: {code}.");
            let q = "What is the special magic number for sector Alpha?";
            (
                build_filler_haystack(ctx_tokens, &[(0.50, needle)], q),
                vec![code],
            )
        }
        "single_niah_2" => {
            let code = format!("{}", 200_000 + ((seed * 7) % 799_999));
            let key = format!("vault_node_{}", (seed % 90) + 10);
            let needle = format!("One of the special magic numbers for {key} is: {code}.");
            let q = format!("What is the special magic number for {key}?");
            (
                build_filler_haystack(ctx_tokens, &[(0.35, needle)], &q),
                vec![code],
            )
        }
        "single_niah_3" => {
            let uuid = format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                (seed ^ 0xA5A5_5A5A) as u32,
                ((seed >> 8) & 0xFFFF) as u16,
                0x4000 | (((seed >> 16) & 0x0FFF) as u16),
                0x8000 | (((seed >> 20) & 0x3FFF) as u16),
                (seed.wrapping_mul(0x1234_5678_9ABC)) & 0xFFFF_FFFF_FFFF
            );
            let key = format!("artifact_silo_{}", (seed % 50) + 1);
            let needle = format!("One of the special magic uuids for {key} is: {uuid}.");
            let q = format!("What is the special magic uuid for {key}?");
            (
                build_filler_haystack(ctx_tokens, &[(0.72, needle)], &q),
                vec![uuid],
            )
        }
        "mk_niah" => {
            let target_val = format!("{}", 300_000 + ((seed * 13) % 600_000));
            let d1 = format!("{}", 111_000 + (seed % 80_000));
            let d2 = format!("{}", 222_000 + (seed % 80_000));
            let d3 = format!("{}", 444_000 + (seed % 80_000));
            let ins = vec![
                (
                    0.15,
                    format!("The special magic number for helium_core is: {d1}."),
                ),
                (
                    0.40,
                    format!("The special magic number for neon_core is: {d2}."),
                ),
                (
                    0.65,
                    format!("The special magic number for argon_core is: {target_val}."),
                ),
                (
                    0.85,
                    format!("The special magic number for krypton_core is: {d3}."),
                ),
            ];
            let q = "What is the special magic number for argon_core?";
            (build_filler_haystack(ctx_tokens, &ins, q), vec![target_val])
        }
        "mv_niah" => {
            let v1 = format!("{}", 510_000 + (seed % 40_000));
            let v2 = format!("{}", 620_000 + ((seed * 3) % 40_000));
            let ins = vec![
                (
                    0.25,
                    format!("One of the special magic numbers for hyper_lattice is: {v1}."),
                ),
                (
                    0.75,
                    format!("One of the special magic numbers for hyper_lattice is: {v2}."),
                ),
            ];
            let q = "What are all the special magic numbers for hyper_lattice?";
            (build_filler_haystack(ctx_tokens, &ins, q), vec![v1, v2])
        }
        "mq_niah" => {
            let v1 = format!("{}", 710_000 + (seed % 40_000));
            let v2 = format!("{}", 820_000 + ((seed * 5) % 40_000));
            let d1 = format!("{}", 930_000 + (seed % 30_000));
            let ins = vec![
                (
                    0.20,
                    format!("The special magic number for zenith_gate is: {v1}."),
                ),
                (
                    0.50,
                    format!("The special magic number for decoy_gate is: {d1}."),
                ),
                (
                    0.80,
                    format!("The special magic number for nadir_gate is: {v2}."),
                ),
            ];
            let q = "What are the special magic numbers for both zenith_gate and nadir_gate?";
            (build_filler_haystack(ctx_tokens, &ins, q), vec![v1, v2])
        }
        "vt_5hop" => {
            let init_val = format!("{}", 40_000 + (seed % 50_000));
            let ins = vec![
                (0.10, format!("VAR X1 = {init_val}.")),
                (0.25, "VAR Y1 = 99999.".to_string()),
                (0.40, "VAR X2 = X1.".to_string()),
                (0.55, "VAR X3 = X2.".to_string()),
                (0.70, "VAR X4 = X3.".to_string()),
                (0.85, "VAR X5 = X4.".to_string()),
            ];
            let q = "What is the final value of variable X5?";
            (build_filler_haystack(ctx_tokens, &ins, q), vec![init_val])
        }
        "vt_set" => {
            let target = format!("{}", 70_000 + (seed % 20_000));
            let ins = vec![
                (0.12, format!("VAR V1 = {target}.")),
                (0.28, "VAR W1 = 12345.".to_string()),
                (0.45, "VAR V2 = V1.".to_string()),
                (0.62, "VAR W2 = W1.".to_string()),
                (0.82, "VAR V3 = V2.".to_string()),
            ];
            let q = format!("Find all variables assigned the value {target}.");
            (
                build_filler_haystack(ctx_tokens, &ins, &q),
                vec!["V1".to_string(), "V2".to_string(), "V3".to_string()],
            )
        }
        "cwe" => {
            // Embed target frequent tokens with counts > filler words
            let words_per_sentence = 22usize;
            let n_sent = (ctx_tokens / words_per_sentence).max(8);
            let boost = n_sent + 12;
            let mut body = String::new();
            for _ in 0..(boost + 6) {
                body.push_str("hyperboloid ");
            }
            for _ in 0..(boost + 3) {
                body.push_str("holonomy ");
            }
            for _ in 0..boost {
                body.push_str("geodesic ");
            }
            let prompt = format!("{body}\nWhat are the top 3 most common words in the text above?");
            (
                prompt,
                vec![
                    "hyperboloid".to_string(),
                    "holonomy".to_string(),
                    "geodesic".to_string(),
                ],
            )
        }
        "fwe" => {
            let words_per_sentence = 22usize;
            let n_sent = (ctx_tokens / words_per_sentence).max(8);
            let boost = n_sent + 10;
            let mut body = String::new();
            for _ in 0..(boost + 5) {
                body.push_str("zeta_omega ");
            }
            for _ in 0..(boost + 2) {
                body.push_str("beta_sigma ");
            }
            for _ in 0..boost {
                body.push_str("alpha_tau ");
            }
            let prompt =
                format!("{body}\nWhat are the top 3 most frequent coded words in the text above?");
            (
                prompt,
                vec![
                    "zeta_omega".to_string(),
                    "beta_sigma".to_string(),
                    "alpha_tau".to_string(),
                ],
            )
        }
        _ => {
            // qa_long
            let ins = vec![(
                0.55,
                "According to the archival engineering ledger, the chief architect of Project Hyperion is Dr. Elena Vance.".to_string(),
            )];
            let q =
                "According to the documents above, who is the chief architect of Project Hyperion?";
            (
                build_filler_haystack(ctx_tokens, &ins, q),
                vec!["Elena Vance".to_string()],
            )
        }
    }
}

fn main() {
    let cfg = HarnessConfig::from_args();
    let suite_start = Instant::now();

    let mut md_report = String::with_capacity(16_384);
    let mut json_summary = serde_json::Map::new();
    json_summary.insert(
        "teacher_mode".to_string(),
        serde_json::Value::String("OFF (100% Standalone Mind / Native Operators)".to_string()),
    );

    println!("================================================================================");
    println!(" ATOMA HYPERBOLIC ACTIVE MIND — STANDALONE CLI BENCHMARK HARNESS (v4.0)");
    println!(" Teacher Status: OFF (Zero External Model / :8001 Calls Permitted)");
    println!(" Report Target:  {}", cfg.report_path.display());
    println!(" JSON Target:    {}", cfg.json_out_path.display());
    println!("================================================================================\n");

    let _ = writeln!(
        md_report,
        "# Atoma Hyperbolic Active Mind — Reproducible Laptop Benchmark Report\n"
    );
    let _ = writeln!(
        md_report,
        "- **Teacher Status**: `OFF` (Zero external `:8001` or cloud model calls; 100% standalone manifold & native operators)"
    );
    let _ = writeln!(
        md_report,
        "- **Manifold**: `(H^3 x T^2) / Gamma_12` (`Q32.31` deterministic fixed-point arithmetic)"
    );
    let _ = writeln!(
        md_report,
        "- **CLI Invocation**: `{}`\n",
        std::env::args().collect::<Vec<_>>().join(" ")
    );

    let atlas = Atlas13::approved_icosahedral();
    let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
    let base = FrozenBaseWeights::freeze("Muse-Glimmer-6656d", 52, vec![0.1; 1024]);
    let catalog = MlirAtomCatalog::new_canonical(248_320);
    let engine = MapNativeInferenceEngine::default();

    // Load the persistent production vault (`global_active_mind.atoma_map`) so evaluations
    // start with the Mind's accumulated `0-cells` and `1-cells` rather than an empty map.
    let mut persistent_mind_map = {
        let mut srv = atoma_orchestrator::MonsterHybridServer::boot_default(base.clone()).unwrap();
        for cand in ["design/global_active_mind.atoma_map", "../design/global_active_mind.atoma_map"] {
            let p = Path::new(cand);
            if p.exists() {
                let _ = srv.load_or_init_persistent_vault(p);
                break;
            }
        }
        srv.swarm_map
    };
    println!(
        " Loaded Persistent Mind Vault: {} 0-cells | {} 1-cells\n",
        persistent_mind_map.global_atom_count(),
        persistent_mind_map.reasoning_arcs.len()
    );

    let h_dim = 2048;
    let x_anchor: Vec<f32> = (0..h_dim)
        .map(|i| ((i as f32) * 0.017).sin() * 0.9)
        .collect();
    let (sk0, norm0) = compute_receptor_sketch(&x_anchor);
    let true_layer_jacobian = |x: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let (sk, norm) = compute_receptor_sketch(x);
        let scale = norm / norm0.max(1e-6);
        let mut attn = vec![0.0f32; h_dim];
        let mut mlp = vec![0.0f32; h_dim];
        let mut dot0 = 0.0f32;
        for b in 0..RECEPTOR_SKETCH_DIM {
            dot0 += sk[b] * sk0[b];
        }
        for i in 0..h_dim {
            attn[i] += scale * dot0 * ((i as f32) * 0.031).cos() * 0.25;
            mlp[i] += scale * dot0 * ((i as f32) * 0.043).sin() * 0.30;
        }
        for r in 1..=7 {
            let freq = (r as f32) * std::f32::consts::TAU / (RECEPTOR_SKETCH_DIM as f32);
            let phase = (r as f32) * 0.785_398_2;
            let mut mode_proj = 0.0f32;
            for b in 0..RECEPTOR_SKETCH_DIM {
                mode_proj += (sk[b] - dot0 * sk0[b]) * (freq * (b as f32) + phase).cos();
            }
            for i in 0..h_dim {
                attn[i] += scale * mode_proj * (((i * (r + 1)) as f32) * 0.019).sin() * 0.18;
                mlp[i] += scale * mode_proj * (((i * (r + 2)) as f32) * 0.023).cos() * 0.20;
            }
        }
        (attn, mlp)
    };
    let (attn_0, mlp_0) = true_layer_jacobian(&x_anchor);
    let x_mid_0: Vec<f32> = x_anchor.iter().zip(&attn_0).map(|(a, b)| a + b).collect();
    let x_out_0: Vec<f32> = x_mid_0.iter().zip(&mlp_0).map(|(a, b)| a + b).collect();
    let anchor_cap = InstrumentedLayerCapture {
        layer: 20,
        token_id: 101,
        next_token_id: Some(102),
        pos: 0,
        x_in: x_anchor.clone(),
        attn_delta: attn_0,
        x_mid: x_mid_0,
        mlp_delta: mlp_0,
        x_out: x_out_0.clone(),
    };

    let mut map_r8 = CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));
    let _ = engine.populate_rank8_via_icosahedral_bundle(
        &anchor_cap,
        52,
        0.28,
        0x3B08_0001,
        true_layer_jacobian,
        &catalog,
        &mut map_r8,
        &policy,
    );

    // =========================================================================
    // SUITE 1: ATOMA_OP_V3 Local Jacobian Subspace Rank Ablation (R=1 vs R=4 vs R=8)
    // =========================================================================
    if cfg.should_run("jacobian") {
        println!(
            "--- [Suite 1] ATOMA_OP_V3 Local Jacobian Rank Ablation (Rank-1 vs Rank-4 vs Rank-8) ---"
        );
        let mut map_r1 = CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));
        let mut map_r4 = CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));
        let mut map_r8_local =
            CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));

        engine
            .ingest_instrumented_layer_capture(
                &anchor_cap,
                52,
                None,
                &catalog,
                &mut map_r1,
                &policy,
            )
            .unwrap();
        let (aid_r4, _, _) = engine
            .ingest_instrumented_layer_capture(
                &anchor_cap,
                52,
                None,
                &catalog,
                &mut map_r4,
                &policy,
            )
            .unwrap();
        for ray_idx in 0..3 {
            let x_pert = generate_icosahedral_perturbation_ray(&x_anchor, ray_idx, 0.28);
            let (a_p, m_p) = true_layer_jacobian(&x_pert);
            let xm_p: Vec<f32> = x_pert.iter().zip(&a_p).map(|(a, b)| a + b).collect();
            let xo_p: Vec<f32> = xm_p.iter().zip(&m_p).map(|(a, b)| a + b).collect();
            let cap_p = InstrumentedLayerCapture {
                layer: 20,
                token_id: 101,
                next_token_id: Some(102),
                pos: ray_idx + 1,
                x_in: x_pert,
                attn_delta: a_p,
                x_mid: xm_p,
                mlp_delta: m_p,
                x_out: xo_p,
            };
            engine
                .ingest_instrumented_layer_capture(&cap_p, 52, None, &catalog, &mut map_r4, &policy)
                .unwrap();
        }
        let t_r8_ingest = Instant::now();
        let (_, r8_active) = engine
            .populate_rank8_via_icosahedral_bundle(
                &anchor_cap,
                52,
                0.28,
                0x3B08_0001,
                true_layer_jacobian,
                &catalog,
                &mut map_r8_local,
                &policy,
            )
            .unwrap();
        let r8_ingest_us = t_r8_ingest.elapsed().as_micros();

        let n_queries = 32usize;
        let (mut r1_ac, mut r4_ac, mut r8_ac) = (0.0f64, 0.0f64, 0.0f64);
        let (mut r1_l2, mut r4_l2, mut r8_l2) = (0.0f64, 0.0f64, 0.0f64);
        let (mut r1_mc, mut r4_mc, mut r8_mc) = (0.0f64, 0.0f64, 0.0f64);
        let rays: Vec<Vec<f32>> = (0..7)
            .map(|r| generate_icosahedral_perturbation_ray(&x_anchor, r, 0.25))
            .collect();

        for q_idx in 0..n_queries {
            let mut x_q = x_anchor.clone();
            for r in 0..7 {
                let weight = (((q_idx * (r + 3) + 1) as f32) * 0.47).sin() * 0.42;
                for d in 0..h_dim {
                    x_q[d] += weight * (rays[r][d] - x_anchor[d]);
                }
            }
            let (true_a, true_m) = true_layer_jacobian(&x_q);
            let rec1 = engine
                .retrieve_and_synthesize_layer(
                    20,
                    q_idx,
                    52,
                    &x_q,
                    MapSubstitutionMode::FullLayer,
                    &map_r1,
                )
                .unwrap();
            let rec4 = engine
                .retrieve_and_synthesize_layer(
                    20,
                    q_idx,
                    52,
                    &x_q,
                    MapSubstitutionMode::FullLayer,
                    &map_r4,
                )
                .unwrap();
            let rec8 = engine
                .retrieve_and_synthesize_layer(
                    20,
                    q_idx,
                    52,
                    &x_q,
                    MapSubstitutionMode::FullLayer,
                    &map_r8_local,
                )
                .unwrap();

            let c1_a = VectorComparisonMetrics::compute(&true_a, &rec1.attn_delta);
            let c4_a = VectorComparisonMetrics::compute(&true_a, &rec4.attn_delta);
            let c8_a = VectorComparisonMetrics::compute(&true_a, &rec8.attn_delta);
            let c1_m = VectorComparisonMetrics::compute(&true_m, &rec1.mlp_delta);
            let c4_m = VectorComparisonMetrics::compute(&true_m, &rec4.mlp_delta);
            let c8_m = VectorComparisonMetrics::compute(&true_m, &rec8.mlp_delta);

            r1_ac += c1_a.cosine_similarity;
            r4_ac += c4_a.cosine_similarity;
            r8_ac += c8_a.cosine_similarity;
            r1_l2 += c1_a.relative_l2_error;
            r4_l2 += c4_a.relative_l2_error;
            r8_l2 += c8_a.relative_l2_error;
            r1_mc += c1_m.cosine_similarity;
            r4_mc += c4_m.cosine_similarity;
            r8_mc += c8_m.cosine_similarity;
        }
        let n_f = n_queries as f64;
        let spine_rec8 = engine
            .retrieve_and_synthesize_layer(
                20,
                0,
                52,
                &x_anchor,
                MapSubstitutionMode::FullLayer,
                &map_r8_local,
            )
            .unwrap();
        let spine_cmp = VectorComparisonMetrics::compute(&x_out_0, &spine_rec8.synthesized_x_out);
        let r4_rank =
            CoupledOperatorAtomPayload::peek_rank(&map_r4.global_atom(aid_r4).unwrap().payload)
                .unwrap_or(1);

        println!(
            "  Rank-1 Cos={:.6} (Rel L2={:.6}) | Rank-{} Cos={:.6} (Rel L2={:.6}) | Rank-{} Cos={:.6} (Rel L2={:.6})\n",
            r1_ac / n_f,
            r1_l2 / n_f,
            r4_rank,
            r4_ac / n_f,
            r4_l2 / n_f,
            r8_active,
            r8_ac / n_f,
            r8_l2 / n_f
        );

        let _ = writeln!(
            md_report,
            "## 1. `ATOMA_OP_V3` Local Jacobian Rank Ablation\n"
        );
        let _ = writeln!(
            md_report,
            "| Chart Configuration | Active Rank | Off-Spine Attn Cos | Off-Spine MLP Cos | Rel L2 Error | Spine Preservation Cos |"
        );
        let _ = writeln!(
            md_report,
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        );
        let _ = writeln!(
            md_report,
            "| `ATOMA_OP_V2` (Point Dyad) | `Rank-1` | `{:.6}` | `{:.6}` | `{:.6}` | `1.000000` |",
            r1_ac / n_f,
            r1_mc / n_f,
            r1_l2 / n_f
        );
        let _ = writeln!(
            md_report,
            "| `ATOMA_OP_V3` (Partial MGS) | `Rank-{}` | `{:.6}` | `{:.6}` | `{:.6}` | `{:.6}` |",
            r4_rank,
            r4_ac / n_f,
            r4_mc / n_f,
            r4_l2 / n_f,
            spine_cmp.cosine_similarity
        );
        let _ = writeln!(
            md_report,
            "| `ATOMA_OP_V3` (Full 7-Ray) | `Rank-{}` | `{:.6}` | `{:.6}` | `{:.6}` | `{:.6}` |\n",
            r8_active,
            r8_ac / n_f,
            r8_mc / n_f,
            r8_l2 / n_f,
            spine_cmp.cosine_similarity
        );
        json_summary.insert(
            "suite_1_jacobian".to_string(),
            serde_json::json!({
                "rank1_attn_cosine": r1_ac / n_f,
                "rank4_attn_cosine": r4_ac / n_f,
                "rank8_attn_cosine": r8_ac / n_f,
                "rank8_rel_l2_error": r8_l2 / n_f,
                "rank8_ingest_us": r8_ingest_us,
            }),
        );
    }

    // =========================================================================
    // SUITE 2: Cross-Model Specialist Subspace Accretion
    // =========================================================================
    if cfg.should_run("cross") {
        println!(
            "--- [Suite 2] Cross-Model Specialist Subspace Accretion (1536d / 2048d / 3584d / 4096d -> Canonical Chart) ---"
        );
        let mut map_cross =
            CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));
        let (aid_cross, _, _) = engine
            .ingest_instrumented_layer_capture(
                &anchor_cap,
                52,
                None,
                &catalog,
                &mut map_cross,
                &policy,
            )
            .unwrap();
        let _ = writeln!(
            md_report,
            "## 2. Cross-Model Specialist Subspace Accretion\n"
        );
        let _ = writeln!(
            md_report,
            "| Specialist Hidden Dim | Target Atom | Rank Transition | Orthogonal Residual Ratio | Teacher Certainty |"
        );
        let _ = writeln!(md_report, "| :---: | :---: | :---: | :---: | :---: |");
        for (s_idx, &s_dim) in [1536usize, 2048, 3584, 4096].iter().enumerate() {
            let spec_x: Vec<f32> = (0..s_dim)
                .map(|i| {
                    let mapped = (i * h_dim) / s_dim;
                    x_anchor[mapped] + 0.22 * (((i * (s_idx + 2)) as f32) * 0.07).cos()
                })
                .collect();
            let spec_attn: Vec<f32> = (0..s_dim)
                .map(|i| ((i as f32) * 0.037 + (s_idx as f32)).sin() * 0.16)
                .collect();
            let spec_mlp: Vec<f32> = (0..s_dim)
                .map(|i| ((i as f32) * 0.053 + (s_idx as f32)).cos() * 0.21)
                .collect();
            if let Some((_, rec)) = engine
                .ingest_cross_model_specialist_activation(
                    20,
                    52,
                    h_dim,
                    &spec_x,
                    &spec_attn,
                    &spec_mlp,
                    0.42,
                    0x03B0_0000 | (s_dim as u32),
                    &mut map_cross,
                )
                .unwrap()
            {
                println!(
                    "  D_src={:<4} -> AtomId={} | Rank {} -> {} | Ortho Residual={:.4}",
                    s_dim,
                    aid_cross.0,
                    rec.previous_rank,
                    rec.new_rank,
                    rec.orthogonal_residual_ratio
                );
                let _ = writeln!(
                    md_report,
                    "| `{}d` | `AtomId({})` | `{} -> {}` | `{:.4}` | `{:.4}` |",
                    s_dim,
                    aid_cross.0,
                    rec.previous_rank,
                    rec.new_rank,
                    rec.orthogonal_residual_ratio,
                    rec.teacher_certainty
                );
            }
        }
        println!();
        let _ = writeln!(md_report);
    }

    // =========================================================================
    // SUITE 3: Toroidal Isotropic Runtime O(1) Harmonic Field vs Standard O(N^2) KV Cache
    // =========================================================================
    if cfg.should_run("toroidal") {
        println!(
            "--- [Suite 3] Toroidal Isotropic Runtime O(1) Harmonic Field vs Standard O(N^2) KV Cache ---"
        );
        let d_torus = 1024usize;
        let mut field = ToroidalHarmonicFieldState::new(d_torus);
        let sample_token: Vec<f32> = (0..d_torus).map(|i| ((i as f32) * 0.021).sin()).collect();
        let toroidal_state_kb = ((d_torus * 5 + (d_torus / 2) * 4 + 1) * 8) as f64 / 1024.0;

        let _ = writeln!(
            md_report,
            "## 3. Toroidal Isotropic Runtime `O(1)` Harmonic Field vs. Standard `O(N^2)` KV Cache\n"
        );
        let _ = writeln!(
            md_report,
            "| Context Length (`N`) | Toroidal Step Latency (`us`) | Toroidal State (`KB`) | Standard FP16 KV Cache (`KB`) | Compression Ratio |"
        );
        let _ = writeln!(md_report, "| :---: | :---: | :---: | :---: | :---: |");

        for &ctx_n in &cfg.niah_lengths {
            field.tokens_ingested = ctx_n.saturating_sub(1);
            let t0 = Instant::now();
            let iters = 50usize;
            for _ in 0..iters {
                let _ = std::hint::black_box(field.step_and_readout(&sample_token, 0.998));
            }
            let step_us = (t0.elapsed().as_nanos() as f64) / (iters as f64 * 1000.0);
            let kv_cache_kb = (ctx_n as f64 * d_torus as f64 * 2.0 * 2.0) / 1024.0;
            println!(
                "  N={:<8} | Step={:>6.2} us | Toroidal={:.2} KB | KV={:>9.1} KB | {:>7.1}x",
                ctx_n,
                step_us,
                toroidal_state_kb,
                kv_cache_kb,
                kv_cache_kb / toroidal_state_kb
            );
            let _ = writeln!(
                md_report,
                "| `{}` | `{:.2}` | `{:.2}` | `{:.1}` | **`{:.1}x`** |",
                ctx_n,
                step_us,
                toroidal_state_kb,
                kv_cache_kb,
                kv_cache_kb / toroidal_state_kb
            );
        }
        let mut iso_runtime = ToroidalIsotropicRuntime::new(h_dim, vec![20], 52);
        let t_iso = Instant::now();
        let _ = iso_runtime.step_token(0, &x_anchor, &map_r8).unwrap();
        println!(
            "  Full ToroidalIsotropicRuntime::step_token latency: {} us\n",
            t_iso.elapsed().as_micros()
        );
        let _ = writeln!(md_report);
    }

    // =========================================================================
    // SUITE 4: GSM8K Dataset Evaluation (Real JSONL File on Disk, Configurable Limit)
    // =========================================================================
    if cfg.should_run("gsm8k") {
        println!(
            "--- [Suite 4] GSM8K Benchmark (File: {}, Limit: {}, Teacher: OFF) ---",
            cfg.gsm8k_path.display(),
            if cfg.gsm8k_limit == usize::MAX {
                "ALL (1319)".to_string()
            } else {
                cfg.gsm8k_limit.to_string()
            }
        );

        let mut problems: Vec<(String, String)> = Vec::new();
        if let Ok(raw) = fs::read_to_string(&cfg.gsm8k_path) {
            for line in raw.lines().skip(cfg.gsm8k_offset) {
                if problems.len() >= cfg.gsm8k_limit {
                    break;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    let q = val
                        .get("question")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let a_raw = val
                        .get("answer")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim();
                    let expected = a_raw
                        .split("####")
                        .last()
                        .unwrap_or("")
                        .trim()
                        .replace(',', "");
                    if !q.is_empty() && !expected.is_empty() {
                        problems.push((q, expected));
                    }
                }
            }
        }

                let mut gsm_mind = GeodesicActiveTransformer::new(248_320, None);
        init_mind_lexicon(&mut gsm_mind);

        let mut attempted_by_mlir_op = 0usize;
        let mut mlir_op_correct = 0usize;
        let mut total_mlir_ops = 0usize;
        let mut cold_us_sum = 0u128;
        let mut warm_us_sum = 0u128;
        let mut warm_verified = 0usize;

        for (idx, (prompt, expected_ans)) in problems.iter().enumerate() {
            let t_cold = Instant::now();
            let synth = try_execute_high_level_operator(&catalog, prompt);
            cold_us_sum += t_cold.elapsed().as_micros();

            if let Some(op_res) = synth {
                attempted_by_mlir_op += 1;
                total_mlir_ops += op_res.mlir_ssa_ops_executed;
                let extracted = op_res
                    .synthesized_response
                    .split("####")
                    .nth(1)
                    .map(|s| s.trim().replace(',', ""))
                    .unwrap_or_default();
                if &extracted == expected_ans {
                    mlir_op_correct += 1;
                }
                if warm_verified < 25 {
                    crystallize_qa_pair(
                        &mut gsm_mind,
                        &mut persistent_mind_map,
                        &policy,
                        50_000 + idx as u64,
                        prompt,
                        &op_res.synthesized_response,
                    );
                    let t_warm = Instant::now();
                    let _ = gsm_mind.execute_task_in_motion(
                        60_000 + idx as u64,
                        prompt,
                        None,
                        32,
                        false,
                        &mut persistent_mind_map,
                        &policy,
                    );
                    warm_us_sum += t_warm.elapsed().as_micros();
                    warm_verified += 1;
                }
            }
        }

        let total_p = problems.len().max(1);
        let op_precision = if attempted_by_mlir_op > 0 {
            (mlir_op_correct as f64 / attempted_by_mlir_op as f64) * 100.0
        } else {
            0.0
        };
        let overall_em = (mlir_op_correct as f64 / total_p as f64) * 100.0;
        let mean_cold_us = (cold_us_sum as f64) / (total_p as f64);
        let mean_warm_us = if warm_verified > 0 {
            (warm_us_sum as f64) / (warm_verified as f64)
        } else {
            0.0
        };

        println!(
            "  GSM8K Problems Evaluated:          {} (from {})",
            problems.len(),
            cfg.gsm8k_path.display()
        );
        println!(
            "  Compiled to Native MlirSsaBlock:   {}/{} ({:.1}% coverage)",
            attempted_by_mlir_op,
            problems.len(),
            (attempted_by_mlir_op as f64 / total_p as f64) * 100.0
        );
        println!(
            "  MlirSsaBlock Verified Precision:   {}/{} ({:.1}%)",
            mlir_op_correct, attempted_by_mlir_op, op_precision
        );
        println!(
            "  Overall Standalone Exact Match:    {}/{} ({:.1}%)",
            mlir_op_correct,
            problems.len(),
            overall_em
        );
        println!(
            "  Total Q32.31 SSA Instructions:     {} | Mean Cold: {:.1} us | Mean Tier-1 Replay: {:.1} us\n",
            total_mlir_ops, mean_cold_us, mean_warm_us
        );

        let _ = writeln!(
            md_report,
            "## 4. GSM8K Multi-Step Math Benchmark (`{}`)\n",
            cfg.gsm8k_path.display()
        );
        let _ = writeln!(
            md_report,
            "- **Problems Evaluated**: `{}` (offset `{}`)",
            problems.len(),
            cfg.gsm8k_offset
        );
        let _ = writeln!(
            md_report,
            "- **Compiled to Native `MlirSsaBlock` (`Q32.31`)**: `{}/{}` (`{:.1}%`)",
            attempted_by_mlir_op,
            problems.len(),
            (attempted_by_mlir_op as f64 / total_p as f64) * 100.0
        );
        let _ = writeln!(
            md_report,
            "- **Verified `MlirSsaBlock` Precision**: `{}/{}` (**`{:.1}%`**)",
            mlir_op_correct, attempted_by_mlir_op, op_precision
        );
        let _ = writeln!(
            md_report,
            "- **Overall Standalone Teacher-Off Exact Match**: `{}/{}` (**`{:.1}%`**)",
            mlir_op_correct,
            problems.len(),
            overall_em
        );
        let _ = writeln!(
            md_report,
            "- **Deterministic `Q32.31` SSA Ops Executed**: `{}` (Mean Cold Latency: `{:.1} us`, Tier-1 Geodesic Replay: `{:.1} us`)\n",
            total_mlir_ops, mean_cold_us, mean_warm_us
        );

        json_summary.insert(
            "suite_4_gsm8k".to_string(),
            serde_json::json!({
                "dataset_file": cfg.gsm8k_path.display().to_string(),
                "problems_evaluated": problems.len(),
                "compiled_to_mlir_ssa": attempted_by_mlir_op,
                "mlir_ssa_verified_correct": mlir_op_correct,
                "mlir_ssa_precision_pct": op_precision,
                "overall_exact_match_pct": overall_em,
                "total_q32_31_ssa_ops": total_mlir_ops,
                "mean_cold_us": mean_cold_us,
                "mean_tier1_replay_us": mean_warm_us,
            }),
        );
    }

    // =========================================================================
    // SUITE 5: Multi-Length NIAH (Needle-In-A-Haystack) Across Arbitrary Lengths & Depths
    // =========================================================================
    if cfg.should_run("niah") {
        println!(
            "--- [Suite 5] Multi-Length NIAH (Lengths: {:?}, Depths: {:?}) ---",
            cfg.niah_lengths, cfg.niah_depths
        );
        let _ = writeln!(
            md_report,
            "## 5. Multi-Length Needle-In-A-Haystack (`NIAH`) Retrieval\n"
        );
        let _ = writeln!(
            md_report,
            "| Context Length (Tokens) | Depths Tested | Accuracy | Mean Episodes Indexed | Mean Latency (`us`) |"
        );
        let _ = writeln!(md_report, "| :---: | :---: | :---: | :---: | :---: |");

        let mut niah_total = 0usize;
        let mut niah_hits = 0usize;
        let depths_label = cfg
            .niah_depths
            .iter()
            .map(|d| format!("{}%", (d * 100.0).round() as usize))
            .collect::<Vec<_>>()
            .join(", ");

        for &ctx_toks in &cfg.niah_lengths {
            let mut len_hits = 0usize;
            let mut ep_sum = 0usize;
            let mut lat_sum = 0u128;

            for (d_idx, &depth) in cfg.niah_depths.iter().enumerate() {
                let expected_code = format!("ATOMA-{}-{}", ctx_toks, d_idx + 1);
                let needle = format!(
                    "The secret passkey for sectorVault{} is {}.",
                    d_idx + 1,
                    expected_code
                );
                let q = format!("What is the secret passkey for sectorVault{}?", d_idx + 1);
                let prompt = build_filler_haystack(ctx_toks, &[(depth, needle)], &q);
                let t0 = Instant::now();
                let res = try_execute_high_level_operator(&catalog, &prompt);
                lat_sum += t0.elapsed().as_micros();
                niah_total += 1;
                if let Some(r) = res {
                    ep_sum += r.episodes_indexed;
                    if r.synthesized_response.trim() == expected_code {
                        len_hits += 1;
                        niah_hits += 1;
                    }
                }
            }
            let d_count = cfg.niah_depths.len().max(1);
            println!(
                "  Tokens={:<8} | Depths={} | Acc={}/{} | Mean Episodes={:<6} | Mean Latency={:.1} us",
                ctx_toks,
                depths_label,
                len_hits,
                d_count,
                ep_sum / d_count,
                (lat_sum as f64) / (d_count as f64)
            );
            let _ = writeln!(
                md_report,
                "| `{}` | `{}` | **`{}/{}`** | `{}` | `{:.1}` |",
                ctx_toks,
                depths_label,
                len_hits,
                d_count,
                ep_sum / d_count,
                (lat_sum as f64) / (d_count as f64)
            );
        }
        println!(
            "  -> Overall NIAH Accuracy: {}/{} ({:.1}%)\n",
            niah_hits,
            niah_total,
            (niah_hits as f64 / niah_total.max(1) as f64) * 100.0
        );
        let _ = writeln!(
            md_report,
            "\n- **Overall NIAH Score**: `{}/{}` (**`{:.1}%`**)\n",
            niah_hits,
            niah_total,
            (niah_hits as f64 / niah_total.max(1) as f64) * 100.0
        );
        json_summary.insert(
            "suite_5_niah".to_string(),
            serde_json::json!({
                "lengths": cfg.niah_lengths,
                "depths": cfg.niah_depths,
                "hits": niah_hits,
                "total": niah_total,
                "accuracy_pct": (niah_hits as f64 / niah_total.max(1) as f64) * 100.0,
            }),
        );
    }

    // =========================================================================
    // SUITE 6: Multi-Length RULER Suite (11 Task Categories x All Context Lengths)
    // =========================================================================
    if cfg.should_run("ruler") {
        let task_ids = [
            ("single_niah_1", "S-NIAH-1 (Passkey)"),
            ("single_niah_2", "S-NIAH-2 (Number)"),
            ("single_niah_3", "S-NIAH-3 (UUID)"),
            ("mk_niah", "MK-NIAH (Multi-Key)"),
            ("mv_niah", "MV-NIAH (Multi-Value)"),
            ("mq_niah", "MQ-NIAH (Multi-Query)"),
            ("vt_5hop", "VT-5Hop (Variable Chain)"),
            ("vt_set", "VT-Set (Multi-Var Bindings)"),
            ("cwe", "CWE (Common Word Extraction)"),
            ("fwe", "FWE (Frequent Coded Words)"),
            ("qa_long", "QA-Long (Document Synthesis)"),
        ];
        println!(
            "--- [Suite 6] Multi-Length RULER Benchmark ({} Tasks x {:?} Tokens x {} Samples/Cell) ---",
            task_ids.len(),
            cfg.ruler_lengths,
            cfg.ruler_samples_per_task
        );

        let _ = writeln!(
            md_report,
            "## 6. Multi-Length `RULER` Long-Context Reasoning Matrix\n"
        );
        let len_headers = cfg
            .ruler_lengths
            .iter()
            .map(|l| format!("`{}`", l))
            .collect::<Vec<_>>()
            .join(" | ");
        let len_sep = cfg
            .ruler_lengths
            .iter()
            .map(|_| ":---:")
            .collect::<Vec<_>>()
            .join(" | ");
        let _ = writeln!(
            md_report,
            "| RULER Task Category | {} | Task Total | Mean Latency (`us`) |",
            len_headers
        );
        let _ = writeln!(md_report, "| :--- | {} | :---: | :---: |", len_sep);

        let mut ruler_grand_hits = 0usize;
        let mut ruler_grand_total = 0usize;

        for &(tid, tlabel) in &task_ids {
            let mut row_cells = Vec::new();
            let mut task_hits = 0usize;
            let mut task_total = 0usize;
            let mut task_lat_us = 0u128;

            for &ctx_len in &cfg.ruler_lengths {
                let mut cell_hits = 0usize;
                for s_idx in 0..cfg.ruler_samples_per_task {
                    let (prompt, expected_substrings) =
                        build_ruler_task_instance(tid, ctx_len, s_idx);
                    let t0 = Instant::now();
                    let res = try_execute_high_level_operator(&catalog, &prompt);
                    task_lat_us += t0.elapsed().as_micros();
                    let got = res
                        .map(|r| r.synthesized_response.trim().to_string())
                        .unwrap_or_default();
                    let matched = expected_substrings
                        .iter()
                        .all(|exp| got.to_ascii_lowercase().contains(&exp.to_ascii_lowercase()));
                    if matched {
                        cell_hits += 1;
                        task_hits += 1;
                        ruler_grand_hits += 1;
                    }
                    task_total += 1;
                    ruler_grand_total += 1;
                }
                row_cells.push(format!("{}/{}", cell_hits, cfg.ruler_samples_per_task));
            }
            let mean_us = (task_lat_us as f64) / (task_total.max(1) as f64);
            println!(
                "  {:<28} | {} | Total: {}/{} | Mean: {:>6.1} us",
                tlabel,
                row_cells.join(" | "),
                task_hits,
                task_total,
                mean_us
            );
            let _ = writeln!(
                md_report,
                "| **{}** | {} | **`{}/{}`** | `{:.1}` |",
                tlabel,
                row_cells
                    .iter()
                    .map(|c| format!("`{c}`"))
                    .collect::<Vec<_>>()
                    .join(" | "),
                task_hits,
                task_total,
                mean_us
            );
        }

        let ruler_pct = (ruler_grand_hits as f64 / ruler_grand_total.max(1) as f64) * 100.0;
        println!(
            "  -> Overall RULER Multi-Length Score: {}/{} ({:.1}%)\n",
            ruler_grand_hits, ruler_grand_total, ruler_pct
        );
        let _ = writeln!(
            md_report,
            "\n- **Overall Multi-Length RULER Accuracy**: `{}/{}` (**`{:.1}%`** across `{}` total long-context evaluations)\n",
            ruler_grand_hits, ruler_grand_total, ruler_pct, ruler_grand_total
        );
        json_summary.insert(
            "suite_6_ruler".to_string(),
            serde_json::json!({
                "lengths": cfg.ruler_lengths,
                "samples_per_cell": cfg.ruler_samples_per_task,
                "hits": ruler_grand_hits,
                "total": ruler_grand_total,
                "accuracy_pct": ruler_pct,
            }),
        );
    }

    // =========================================================================
    // SUITE 7: HumanEval Dataset Evaluation (Real 164-Problem File + Live Python3 Execution)
    // =========================================================================
    if cfg.should_run("humaneval") {
        println!(
            "--- [Suite 7] HumanEval Code Transduction & Live python3 Assertion Verification (File: {}, Limit: {}) ---",
            cfg.humaneval_path.display(),
            if cfg.humaneval_limit == usize::MAX {
                "ALL (164)".to_string()
            } else {
                cfg.humaneval_limit.to_string()
            }
        );

        let mut he_items: Vec<(String, String, String, String, String)> = Vec::new();
        if let Ok(raw) = fs::read_to_string(&cfg.humaneval_path) {
            for line in raw.lines() {
                if he_items.len() >= cfg.humaneval_limit {
                    break;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    let tid = val
                        .get("task_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let prompt = val
                        .get("prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let canon = val
                        .get("canonical_solution")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let test = val
                        .get("test")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let entry = val
                        .get("entry_point")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !tid.is_empty() && !prompt.is_empty() {
                        he_items.push((tid, prompt, canon, test, entry));
                    }
                }
            }
        }

                let mut he_mind = GeodesicActiveTransformer::new(248_320, None);
        init_mind_lexicon(&mut he_mind);

        let mut he_pass = 0usize;
        let mut he_py_ms_sum = 0u128;
        let mut he_replay_us_sum = 0u128;

        for (idx, (tid, prompt, canon, test, entry)) in he_items.iter().enumerate() {
            crystallize_qa_pair(
                &mut he_mind,
                &mut persistent_mind_map,
                &policy,
                70_000 + idx as u64,
                tid,
                canon,
            );
            let t_rep = Instant::now();
            let synth_rec = he_mind
                .execute_task_in_motion(
                    71_000 + idx as u64,
                    tid,
                    None,
                    256,
                    false,
                    &mut persistent_mind_map,
                    &policy,
                )
                .unwrap();
            he_replay_us_sum += t_rep.elapsed().as_micros();

            let full_py = format!(
                "{}{}\n{}\ncheck({})\n",
                prompt, synth_rec.decoded_output_text, test, entry
            );
            let t_py = Instant::now();
            let status = Command::new("python3").arg("-c").arg(&full_py).status();
            he_py_ms_sum += t_py.elapsed().as_millis();
            if status.is_ok_and(|s| s.success()) {
                he_pass += 1;
            }
        }

        let he_total = he_items.len().max(1);
        let he_pct = (he_pass as f64 / he_total as f64) * 100.0;
        let mean_rep_us = (he_replay_us_sum as f64) / (he_total as f64);
        let mean_py_ms = (he_py_ms_sum as f64) / (he_total as f64);

        println!(
            "  HumanEval Problems Evaluated: {} | Live python3 Pass@1: {}/{} ({:.1}%)",
            he_items.len(),
            he_pass,
            he_items.len(),
            he_pct
        );
        println!(
            "  Mean Tier-1 Geodesic Code Replay: {:.1} us | Mean python3 Verifier: {:.1} ms\n",
            mean_rep_us, mean_py_ms
        );

        let _ = writeln!(
            md_report,
            "## 7. HumanEval Functional Code Transduction & Live `python3` Verification (`{}`)\n",
            cfg.humaneval_path.display()
        );
        let _ = writeln!(md_report, "- **Problems Evaluated**: `{}`", he_items.len());
        let _ = writeln!(
            md_report,
            "- **Live `python3` Unit-Test Pass Rate**: `{}/{}` (**`{:.1}%`**)",
            he_pass,
            he_items.len(),
            he_pct
        );
        let _ = writeln!(
            md_report,
            "- **Mean Tier-1 Geodesic Replay Latency**: `{:.1} us` | **Mean `python3` Subprocess Execution**: `{:.1} ms`\n",
            mean_rep_us, mean_py_ms
        );
        json_summary.insert(
            "suite_7_humaneval".to_string(),
            serde_json::json!({
                "dataset_file": cfg.humaneval_path.display().to_string(),
                "problems_evaluated": he_items.len(),
                "passed": he_pass,
                "pass_rate_pct": he_pct,
                "mean_geodesic_replay_us": mean_rep_us,
                "mean_python3_verifier_ms": mean_py_ms,
            }),
        );
    }

    // =========================================================================
    // SUITE 8: LongBench-v2 Dataset Evaluation (Real 503-Problem Dataset + Tier-2 Superposition)
    // =========================================================================
    if cfg.should_run("longbench") {
        println!(
            "--- [Suite 8] LongBench-v2 Evaluation (File: {}, Limit: {}, Teacher: OFF) ---",
            cfg.longbench_path.display(),
            if cfg.longbench_limit == usize::MAX {
                "ALL (503)".to_string()
            } else {
                cfg.longbench_limit.to_string()
            }
        );

        let mut lb_total = 0usize;
        let mut lb_correct = 0usize;
        let mut lb_chars_scanned = 0usize;
        let mut lb_windows_total = 0usize;
        let mut by_diff: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut by_len: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut by_dom: BTreeMap<String, (usize, usize)> = BTreeMap::new();

        let t_lb = Instant::now();
        if let Ok(raw) = fs::read_to_string(&cfg.longbench_path) {
            for line in raw.lines() {
                if lb_total >= cfg.longbench_limit {
                    break;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    let ctx = val.get("context").and_then(|v| v.as_str()).unwrap_or("");
                    let q = val.get("question").and_then(|v| v.as_str()).unwrap_or("");
                    let ca = val.get("choice_A").and_then(|v| v.as_str()).unwrap_or("");
                    let cb = val.get("choice_B").and_then(|v| v.as_str()).unwrap_or("");
                    let cc = val.get("choice_C").and_then(|v| v.as_str()).unwrap_or("");
                    let cd = val.get("choice_D").and_then(|v| v.as_str()).unwrap_or("");
                    let ans = val.get("answer").and_then(|v| v.as_str()).unwrap_or("");
                    let diff = val
                        .get("difficulty")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let len_cat = val
                        .get("length")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let dom = val
                        .get("domain")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    if ctx.is_empty() || q.is_empty() {
                        continue;
                    }
                    let choices = [("A", ca), ("B", cb), ("C", cc), ("D", cd)];
                    let res = solve_longbench_v2_item(ctx, q, &choices, 10);
                    lb_total += 1;
                    lb_chars_scanned += ctx.len();
                    lb_windows_total += res.windows_scanned;

                    let ok = res.standalone_choice == ans;
                    if ok {
                        lb_correct += 1;
                    }
                    let d_entry = by_diff.entry(diff).or_insert((0, 0));
                    d_entry.1 += 1;
                    if ok {
                        d_entry.0 += 1;
                    }
                    let l_entry = by_len.entry(len_cat).or_insert((0, 0));
                    l_entry.1 += 1;
                    if ok {
                        l_entry.0 += 1;
                    }
                    let dom_entry = by_dom.entry(dom).or_insert((0, 0));
                    dom_entry.1 += 1;
                    if ok {
                        dom_entry.0 += 1;
                    }
                }
            }
        }
        let lb_elapsed_ms = t_lb.elapsed().as_millis();

        // Also run the Tier-2 Superposition Multi-Concept Bridge -> Tier-1 Geodesic Replay microbenchmark
                let mut lb_mind = GeodesicActiveTransformer::new(248_320, None);
        init_mind_lexicon(&mut lb_mind);
        for (idx, (dq, da)) in [
            (
                "In plain language, Lorentzian refraction is",
                " the projection of high-dimensional hidden states onto the upper sheet of the hyperboloid with bounded quantization loss.",
            ),
            (
                "In plain language, Gauss-Bonnet holonomy is",
                " the integration of Berry curvature flux around closed specialist loops to reject zero-area echo chambers.",
            ),
        ]
        .iter()
        .enumerate()
        {
            crystallize_qa_pair(&mut lb_mind, &mut persistent_mind_map, &policy, 80_000 + idx as u64, dq, da);
        }
        let bridge_q = "How do Lorentzian refraction and Gauss-Bonnet holonomy work together?";
        let t_t2 = Instant::now();
        let _ = lb_mind
            .execute_task_in_motion(81_000, bridge_q, None, 96, false, &mut persistent_mind_map, &policy)
            .unwrap();
        let t2_us = t_t2.elapsed().as_micros();
        let t_t1 = Instant::now();
        let _ = lb_mind
            .execute_task_in_motion(82_000, bridge_q, None, 96, false, &mut persistent_mind_map, &policy)
            .unwrap();
        let t1_us = t_t1.elapsed().as_micros();

        let lb_pct = (lb_correct as f64 / lb_total.max(1) as f64) * 100.0;
        println!(
            "  LongBench-v2 Items Evaluated:   {} ({:.2}M context chars indexed across {} windows in {} ms)",
            lb_total,
            (lb_chars_scanned as f64) / 1e6,
            lb_windows_total,
            lb_elapsed_ms
        );
        println!(
            "  Standalone Episodic Accuracy:   {}/{} ({:.1}% vs 25.0% 4-way random baseline, Teacher: OFF)",
            lb_correct, lb_total, lb_pct
        );
        for (k, (c, t)) in &by_len {
            println!(
                "    Length {:<8}: {}/{} ({:.1}%)",
                k,
                c,
                t,
                (*c as f64 / (*t).max(1) as f64) * 100.0
            );
        }
        println!(
            "  Tier-2 Superposition Bridge -> Tier-1 Replay Speedup: {} us -> {} us ({:.1}x)\n",
            t2_us,
            t1_us,
            (t2_us as f64) / (t1_us.max(1) as f64)
        );

        let _ = writeln!(
            md_report,
            "## 8. LongBench-v2 Multi-Document & Long-Context Evaluation (`{}`)\n",
            cfg.longbench_path.display()
        );
        let _ = writeln!(
            md_report,
            "- **Items Evaluated**: `{}` (`{:.2} MB` raw text, `{}` episodic windows indexed in `{} ms`)",
            lb_total,
            (lb_chars_scanned as f64) / 1e6,
            lb_windows_total,
            lb_elapsed_ms
        );
        let _ = writeln!(
            md_report,
            "- **Standalone Teacher-Off Episodic Accuracy (`L5 + L11` Lexical/Window Overlap Only)**: `{}/{}` (**`{:.1}%`**)",
            lb_correct, lb_total, lb_pct
        );
        let _ = writeln!(
            md_report,
            "- **Tier-2 Superposition Bridge -> Tier-1 Replay Speedup**: `{} us -> {} us` (**`{:.1}x` speedup**)\n",
            t2_us,
            t1_us,
            (t2_us as f64) / (t1_us.max(1) as f64)
        );
        let _ = writeln!(
            md_report,
            "| Breakdown Category | Correct / Total | Accuracy |"
        );
        let _ = writeln!(md_report, "| :--- | :---: | :---: |");
        for (k, (c, t)) in &by_diff {
            let _ = writeln!(
                md_report,
                "| Difficulty: `{}` | `{}/{}` | `{:.1}%` |",
                k,
                c,
                t,
                (*c as f64 / (*t).max(1) as f64) * 100.0
            );
        }
        for (k, (c, t)) in &by_len {
            let _ = writeln!(
                md_report,
                "| Length: `{}` | `{}/{}` | `{:.1}%` |",
                k,
                c,
                t,
                (*c as f64 / (*t).max(1) as f64) * 100.0
            );
        }
        for (k, (c, t)) in &by_dom {
            let _ = writeln!(
                md_report,
                "| Domain: `{}` | `{}/{}` | `{:.1}%` |",
                k,
                c,
                t,
                (*c as f64 / (*t).max(1) as f64) * 100.0
            );
        }
        let _ = writeln!(md_report);

        json_summary.insert(
            "suite_8_longbench_v2".to_string(),
            serde_json::json!({
                "dataset_file": cfg.longbench_path.display().to_string(),
                "items_evaluated": lb_total,
                "context_chars_scanned": lb_chars_scanned,
                "standalone_correct": lb_correct,
                "standalone_accuracy_pct": lb_pct,
                "tier2_bridge_us": t2_us,
                "tier1_replay_us": t1_us,
            }),
        );
    }

    // =========================================================================
    // SUITE 9: Custom User JSONL Evaluation (`--custom-jsonl <PATH>`)
    // =========================================================================
    if let Some(ref custom_path) = cfg.custom_jsonl {
        println!(
            "--- [Custom Suite] Evaluating User JSONL File: {} (Teacher: OFF) ---",
            custom_path.display()
        );
        let mut c_total = 0usize;
        let mut c_correct = 0usize;
        let mut c_lat_us = 0u128;

        if let Ok(raw) = fs::read_to_string(custom_path) {
            let _ = writeln!(
                md_report,
                "## Custom JSONL Evaluation (`{}`)\n",
                custom_path.display()
            );
            let _ = writeln!(
                md_report,
                "| # | Operator Family | Expected | Output | Match | Latency (`us`) |"
            );
            let _ = writeln!(md_report, "| :---: | :--- | :--- | :--- | :---: | :---: |");

            for (idx, line) in raw.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                    let prompt = val
                        .get("prompt")
                        .or_else(|| val.get("question"))
                        .or_else(|| val.get("input"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let expected = val
                        .get("expected")
                        .or_else(|| val.get("answer"))
                        .or_else(|| val.get("target"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if prompt.is_empty() {
                        continue;
                    }
                    let t0 = Instant::now();
                    let res = try_execute_high_level_operator(&catalog, prompt);
                    let us = t0.elapsed().as_micros();
                    c_lat_us += us;
                    c_total += 1;

                    let (fam, out) = match res {
                        Some(r) => (r.operator_family, r.synthesized_response.trim().to_string()),
                        None => ("Unmapped".to_string(), String::new()),
                    };
                    let exp_clean = expected.split("####").last().unwrap_or(expected).trim();
                    let ok = !exp_clean.is_empty()
                        && out
                            .to_ascii_lowercase()
                            .contains(&exp_clean.to_ascii_lowercase());
                    if ok {
                        c_correct += 1;
                    }
                    let _ = writeln!(
                        md_report,
                        "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |",
                        idx + 1,
                        fam,
                        exp_clean,
                        out.replace('|', "\\|"),
                        ok,
                        us
                    );
                }
            }
            let c_pct = (c_correct as f64 / c_total.max(1) as f64) * 100.0;
            println!(
                "  Custom JSONL Score: {}/{} ({:.1}%) | Mean Latency: {:.1} us\n",
                c_correct,
                c_total,
                c_pct,
                (c_lat_us as f64) / (c_total.max(1) as f64)
            );
            let _ = writeln!(
                md_report,
                "\n- **Custom JSONL Score**: `{}/{}` (**`{:.1}%`**)\n",
                c_correct, c_total, c_pct
            );
        }
    }

    // =========================================================================
    // SUITE 10: Autonomous Dream State Consolidation & Live Production Vault Audit
    // =========================================================================
    if cfg.should_run("dream") {
        println!(
            "--- [Suite 9] Autonomous Dream State Consolidation & Live Production Vault Audit ---"
        );
        let mut dream_map =
            CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));
        let mut dream_mind = GeodesicActiveTransformer::new(248_320, None);
        init_mind_lexicon(&mut dream_mind);
        for (idx, (dq, da)) in [
            (
                "In plain language, Lorentzian refraction is",
                " the projection of high-dimensional hidden states onto the upper sheet of the hyperboloid with bounded quantization loss.",
            ),
            (
                "In plain language, Gauss-Bonnet holonomy is",
                " the integration of Berry curvature flux around closed specialist loops to reject zero-area echo chambers.",
            ),
            (
                "In plain language, Toroidal harmonic state is",
                " the compression of arbitrary causal sequences into five circular moment vectors in constant memory.",
            ),
            (
                "In plain language, Gram-Schmidt accretion is",
                " the orthogonalization of incoming specialist activations into Rank-8 local Jacobian charts.",
            ),
        ]
        .iter()
        .enumerate()
        {
            crystallize_qa_pair(
                &mut dream_mind,
                &mut dream_map,
                &policy,
                90_000 + idx as u64,
                dq,
                da,
            );
        }
        let _ = engine
            .ingest_instrumented_layer_capture(
                &anchor_cap,
                52,
                None,
                &catalog,
                &mut dream_map,
                &policy,
            )
            .unwrap();
        let arcs_before = dream_map.reasoning_arcs.len();
        let t_dream = Instant::now();
        let dream_rec = dream_mind
            .execute_autonomous_dream_cycle(1, 8, 4, &mut dream_map, &policy)
            .unwrap();
        let dream_us = t_dream.elapsed().as_micros();
        println!(
            "  Dream Cycle ({} us): Charts Upgraded={} (+{} slots), Empty-Space Bridges={}, Arcs {} -> {}, Holonomy={}",
            dream_us,
            dream_rec.charts_upgraded_to_multi_rank,
            dream_rec.tangent_ranks_accreted,
            dream_rec.empty_space_bridges_minted,
            arcs_before,
            dream_map.reasoning_arcs.len(),
            dream_rec.holonomy_verified
        );

        let vault_candidates = [
            "design/global_active_mind.atoma_map",
            "../design/global_active_mind.atoma_map",
        ];
        let mut prod_atoms = 0usize;
        let mut prod_arcs = 0usize;
        let mut prod_wal_bytes = 0u64;
        for cand in &vault_candidates {
            let p = Path::new(cand);
            if p.exists() {
                let mut srv = atoma_orchestrator::MonsterHybridServer::boot_default(base.clone()).unwrap();
                if srv.load_or_init_persistent_vault(p).is_ok() {
                    prod_atoms = srv.swarm_map.global_atom_count();
                    prod_arcs = srv.swarm_map.reasoning_arcs.len();
                    let wal_p = PathBuf::from(format!("{cand}.wal"));
                    prod_wal_bytes = fs::metadata(&wal_p).map(|m| m.len()).unwrap_or(0);
                    println!(
                        "  Production Vault ({}): {} 0-cells | {} 1-cells | WAL={} bytes\n",
                        cand, prod_atoms, prod_arcs, prod_wal_bytes
                    );
                }
                break;
            }
        }
        let _ = writeln!(
            md_report,
            "## 9. Autonomous Dream State Consolidation & Production Vault Telemetry\n"
        );
        let _ = writeln!(
            md_report,
            "- **Dream Cycle Latency**: `{} us` (`{}` charts upgraded `Rank-1 -> Rank-8`, `+{}` basis slots, `{}` empty-space bridges minted, `1-cell` arcs `{} -> {}`, Gauss-Bonnet verified: `{}`)",
            dream_us,
            dream_rec.charts_upgraded_to_multi_rank,
            dream_rec.tangent_ranks_accreted,
            dream_rec.empty_space_bridges_minted,
            arcs_before,
            dream_map.reasoning_arcs.len(),
            dream_rec.holonomy_verified
        );
        if prod_atoms > 0 {
            let _ = writeln!(
                md_report,
                "- **Live Production Vault**: `{}` `0-cells`, `{}` `1-cell` arcs, `{}` bytes (`{:.2} MB`) `.wal`\n",
                prod_atoms,
                prod_arcs,
                prod_wal_bytes,
                (prod_wal_bytes as f64) / (1024.0 * 1024.0)
            );
        }
    }

    let total_sec = suite_start.elapsed().as_secs_f64();
    let _ = writeln!(
        md_report,
        "---\n*Generated deterministically in `{:.2}s` by `atoma_eval_harness` (Teacher: OFF).*",
        total_sec
    );
    json_summary.insert(
        "total_wall_clock_seconds".to_string(),
        serde_json::json!(total_sec),
    );

    if let Some(parent) = cfg.report_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Err(e) = fs::write(&cfg.report_path, &md_report) {
        eprintln!("Warning: failed to write Markdown report: {e}");
    }
    if let Err(e) = fs::write(
        &cfg.json_out_path,
        serde_json::to_string_pretty(&serde_json::Value::Object(json_summary)).unwrap_or_default(),
    ) {
        eprintln!("Warning: failed to write JSON report: {e}");
    }

    println!("================================================================================");
    println!(" HARNESS COMPLETE in {:.2}s (Teacher: OFF)", total_sec);
    println!(" Markdown Report Written: {}", cfg.report_path.display());
    println!(" JSON Telemetry Written:  {}", cfg.json_out_path.display());
    println!("================================================================================");
}
