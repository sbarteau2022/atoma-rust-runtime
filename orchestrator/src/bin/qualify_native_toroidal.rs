//! Phase B (`SPEC_atomic_ir.md` §4.9) native producer/edge qualification.
//! Exercises the native components directly; no server, teacher, model, vault, or socket is used.

use std::path::PathBuf;
use std::time::Instant;

use atoma_immune::refraction::Sha256Digest;
use atoma_immune::{BoundaryLexicalTransducer, MlirOpId};
use atoma_orchestrator::input_embeddings::{embed_carrier, embed_utf8_text};
use atoma_orchestrator::native_output::readout_byte_carriers;
use atoma_orchestrator::toroidal_attention::ToroidalIsotropicRuntime;
use serde::Serialize;
use serde_json::{Value, json};

const LAYERS: usize = 52;
const DIM: usize = 6;

#[derive(Serialize)]
struct Case {
    id: &'static str,
    category: &'static str,
    prompt: &'static str,
    expected_continuation: &'static str,
}

fn cases() -> Vec<Case> {
    [
        ("sum", "arithmetic", "2 + 2 = ", "4"),
        ("multiply", "arithmetic", "7 * 8 = ", "56"),
        ("subtract", "arithmetic", "23 - 9 = ", "14"),
        ("letters", "sequence", "A, B, C, ", "D"),
        ("count", "sequence", "1, 2, 3, ", "4"),
        ("doubles", "sequence", "2, 4, 8, ", "16"),
        ("france", "fact", "The capital of France is ", "Paris"),
        ("water", "fact", "The chemical formula for water is ", "H2O"),
        ("week", "fact", "The day after Monday is ", "Tuesday"),
        ("opposite", "text", "The opposite of hot is ", "cold"),
        ("plural", "text", "The plural of cat is ", "cats"),
        ("past", "text", "The past tense of walk is ", "walked"),
        ("python", "code", "def add(a, b):\n    return ", "a + b"),
        ("rust", "code", "fn main() { println!(\"hi\"); ", "}"),
        ("json", "code", "{\"ok\": true", "}"),
        ("hex", "code", "The decimal value of 0x10 is ", "16"),
        ("copy", "instruction", "Reply with exactly ZEBRA: ", "ZEBRA"),
        ("lower", "instruction", "Lowercase BLUE: ", "blue"),
        ("reverse", "instruction", "Reverse abc: ", "cba"),
        (
            "unknown",
            "instruction",
            "What is the secret code? No code was provided. Answer UNKNOWN: ",
            "UNKNOWN",
        ),
        ("spanish", "unicode", "Translate cat into Spanish: ", "gato"),
        ("japanese", "unicode", "Repeat 猫: ", "猫"),
        ("emoji", "unicode", "Repeat 🐉: ", "🐉"),
        ("accent", "unicode", "Repeat café: ", "café"),
    ]
    .into_iter()
    .map(|(id, category, prompt, expected_continuation)| Case {
        id,
        category,
        prompt,
        expected_continuation,
    })
    .collect()
}

fn native_boundary() -> BoundaryLexicalTransducer {
    // Match the live native path's carrier configuration, with no lexicon loaded.
    BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001])
}

fn run_case(case: &Case, max_new: usize) -> Result<Value, String> {
    let boundary = native_boundary();
    let input = embed_utf8_text(&boundary, case.prompt, false)?;
    let identity_states: Vec<_> = input.iter().map(|item| item.coordinates.to_vec()).collect();
    let identity_ids = readout_byte_carriers(&boundary.catalog, &identity_states)?;
    let identity = boundary.transduce_mlir_to_utf8(&identity_ids)?;
    let mut runtime = ToroidalIsotropicRuntime::full_stack(DIM, LAYERS)?;
    let started = Instant::now();
    let states = runtime.step_native_text_sequence(&input)?;
    let prefill_us = started.elapsed().as_micros();
    let readout = readout_byte_carriers(&boundary.catalog, &states)?;
    let reconstructed = boundary.transduce_mlir_to_utf8(&readout);
    let input_matching_bytes = readout
        .iter()
        .zip(case.prompt.as_bytes())
        .filter(|(op, byte)| op.0 == u32::from(**byte))
        .count();

    let mut hidden = states.last().ok_or("no final hidden state")?.clone();
    let mut generated = Vec::with_capacity(max_new);
    let mut nonfinite_steps = 0;
    let decode_started = Instant::now();
    for step in 0..max_new {
        if !hidden.iter().all(|value| value.is_finite()) {
            nonfinite_steps += 1;
            break;
        }
        let ids = readout_byte_carriers(&boundary.catalog, &[hidden])?;
        let next = ids[0];
        generated.push(next);
        // Do not add a final unobserved state transition after the byte budget.
        if step + 1 < max_new {
            let embedding = embed_carrier(&boundary.catalog, next)?;
            hidden = runtime.step_toroidal_attention(&embedding.coordinates)?;
        } else {
            break;
        }
    }
    let decode_us = decode_started.elapsed().as_micros();
    let expected_bytes = case.expected_continuation.as_bytes();
    let generated_ids: Vec<_> = generated.iter().map(|id| id.0).collect();
    let matches_reference = generated_ids.len() >= expected_bytes.len()
        && generated_ids
            .iter()
            .zip(expected_bytes)
            .all(|(&id, &byte)| id == u32::from(byte));
    let generated_text = boundary.transduce_mlir_to_utf8(&generated);
    let readout_ids: Vec<_> = readout.iter().map(|id| id.0).collect();
    let mut distinct = generated_ids.clone();
    distinct.sort_unstable();
    distinct.dedup();

    Ok(json!({
        "case": case,
        "identity_control_exact": identity == case.prompt,
        "input_bytes": case.prompt.len(),
        "prefill_us": prefill_us,
        "prompt_readout_ids": readout_ids,
        "prompt_readout_text": reconstructed.as_ref().ok(),
        "prompt_readout_error": reconstructed.as_ref().err(),
        "prompt_readout_exact": reconstructed.as_ref().is_ok_and(|text| text == case.prompt),
        "input_matching_bytes": input_matching_bytes,
        "prompt_states_finite": states.iter().flatten().all(|value| value.is_finite()),
        "feedback_probe": {
            "kind": "untrained_self_feedback_not_production_generation",
            "requested_bytes": max_new,
            "carrier_ids": generated_ids,
            "text": generated_text.as_ref().ok(),
            "utf8_error": generated_text.as_ref().err(),
            "matches_reference_prefix": matches_reference,
            "first_byte_correct": generated.first().is_some_and(|id| id.0 == u32::from(expected_bytes[0])),
            "distinct_carriers": distinct.len(),
            "constant_output": !generated.is_empty() && distinct.len() == 1,
            "nonfinite_steps": nonfinite_steps,
            "decode_us": decode_us,
            "stop": if nonfinite_steps > 0 { "nonfinite" } else { "byte_budget_no_eos_in_head" }
        }
    }))
}

fn controls(stress_bytes: usize) -> Result<Value, String> {
    let boundary = native_boundary();
    let exact_bytes: Vec<_> = (0..=255).map(MlirOpId).collect();
    let states: Vec<_> = exact_bytes
        .iter()
        .map(|&id| embed_carrier(&boundary.catalog, id).map(|entry| entry.coordinates.to_vec()))
        .collect::<Result<_, _>>()?;
    let exact = readout_byte_carriers(&boundary.catalog, &states)?;
    let periodic_states: Vec<_> = states
        .iter()
        .map(|state| {
            let mut shifted = state.clone();
            shifted[4] += std::f32::consts::TAU;
            shifted[5] -= std::f32::consts::TAU;
            shifted
        })
        .collect();
    let periodic = readout_byte_carriers(&boundary.catalog, &periodic_states)?;
    let unknown_rejected = boundary
        .transduce_mlir_to_utf8(&[MlirOpId(u32::MAX)])
        .is_err();
    let invalid_utf8_rejected = boundary.transduce_mlir_to_utf8(&[MlirOpId(255)]).is_err();

    let text = "Causality 🐉 abcdefgh\0\r\n";
    let input = embed_utf8_text(&boundary, text, false)?;
    let mut full = ToroidalIsotropicRuntime::full_stack(DIM, LAYERS)?;
    let batch = full.step_native_text_sequence(&input)?;
    let mut streamed = ToroidalIsotropicRuntime::full_stack(DIM, LAYERS)?;
    let mut chunks = Vec::new();
    for chunk in input.chunks(3) {
        chunks.extend(streamed.step_native_text_sequence(chunk)?);
    }
    let repeated =
        ToroidalIsotropicRuntime::full_stack(DIM, LAYERS)?.step_native_text_sequence(&input)?;

    // An ASCII pattern makes the requested byte count exact; multilingual paths have separate cases.
    let pattern = "abc XYZ 019 !@\n";
    let mut stress_text = pattern.repeat(stress_bytes.div_ceil(pattern.len()));
    stress_text.truncate(stress_bytes);
    let stress_input = embed_utf8_text(&boundary, &stress_text, false)?;
    let mut stress = ToroidalIsotropicRuntime::full_stack(DIM, LAYERS)?;
    let mut finite = true;
    let mut stress_error = None;
    let stress_started = Instant::now();
    for embedding in &stress_input {
        match stress.step_toroidal_attention(&embedding.coordinates) {
            Ok(hidden) => finite &= hidden.iter().all(|value| value.is_finite()),
            Err(error) => {
                finite = false;
                stress_error = Some(error);
            }
        }
        if !finite {
            break;
        }
    }
    finite &= stress.layer_fields.iter().all(|field| {
        [
            field.cumulative_rotations,
            field.radius_accumulator,
            field.rapidity_accumulator,
            field.z_0,
        ]
        .iter()
        .all(|value| value.is_finite())
            && [
                &field.z_c1,
                &field.z_s1,
                &field.z_c2,
                &field.z_s2,
                &field.m_0,
                &field.m_c1,
                &field.m_s1,
                &field.m_c2,
                &field.m_s2,
            ]
            .iter()
            .all(|values| values.iter().all(|value| value.is_finite()))
    });
    Ok(json!({
        "byte_address_identity_matches": exact.iter().zip(&exact_bytes).filter(|(a, b)| a == b).count(),
        "byte_address_count": 256,
        "periodic_phase_identity_matches": periodic.iter().zip(&exact_bytes).filter(|(a,b)| a == b).count(),
        "unknown_carrier_rejected": unknown_rejected,
        "invalid_utf8_rejected": invalid_utf8_rejected,
        "native_utf8_roundtrip_exact": boundary.transduce_mlir_to_utf8(&boundary.transduce_utf8_to_mlir(text, true))? == text,
        "causal_chunking_identical": batch == chunks,
        "repeat_identical": batch == repeated,
        "stress": { "requested_bytes": stress_bytes, "actual_bytes": stress_text.len(), "finite": finite,
            "error": stress_error,
            "elapsed_us": stress_started.elapsed().as_micros(),
            "minimum_layer_steps": stress.layer_fields.iter().map(|field| field.tokens_ingested).min(),
            "maximum_layer_steps": stress.layer_fields.iter().map(|field| field.tokens_ingested).max() }
    }))
}

fn source_hashes() -> Value {
    // Hash the source embedded in this binary, even if the checkout is edited after compilation.
    macro_rules! embedded_source {
        ($path:literal) => {
            (
                $path,
                include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/../", $path)) as &[u8],
            )
        };
    }
    let mut hashes = serde_json::Map::new();
    for (name, bytes) in [
        embedded_source!("orchestrator/src/input_embeddings.rs"),
        embedded_source!("orchestrator/src/native_output.rs"),
        embedded_source!("orchestrator/src/toroidal_attention.rs"),
        embedded_source!("orchestrator/src/bin/qualify_native_toroidal.rs"),
        embedded_source!("orchestrator/Cargo.lock"),
        embedded_source!("immune_system/implementation/src/lib.rs"),
        embedded_source!("immune_system/implementation/src/continuous_active_mind.rs"),
        embedded_source!("immune_system/implementation/src/first_principles.rs"),
        embedded_source!("immune_system/implementation/src/native_reader.rs"),
        embedded_source!("immune_system/implementation/src/mlir_native.rs"),
        embedded_source!("immune_system/implementation/src/refraction.rs"),
        embedded_source!("core_lm/src/geometry/torus.rs"),
        embedded_source!("core_lm/src/geometry/lorentz.rs"),
        embedded_source!("core_lm/src/geometry/picard.rs"),
        embedded_source!("core_lm/src/geometry/fixed.rs"),
    ] {
        hashes.insert(name.into(), json!(Sha256Digest::compute(bytes).to_hex()));
    }
    Value::Object(hashes)
}

fn run() -> Result<bool, String> {
    if cfg!(feature = "glimmer") {
        return Err("native qualification must be built without the glimmer feature".into());
    }
    let mut output = PathBuf::from("benchmarks/native_toroidal_quality.json");
    let mut max_new = 32;
    let mut stress_bytes = 4096;
    let mut require_quality = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => output = args.next().ok_or("--output needs a path")?.into(),
            "--max-new" => {
                max_new = args
                    .next()
                    .ok_or("--max-new needs a value")?
                    .parse::<usize>()
                    .map_err(|error| error.to_string())?
            }
            "--stress-bytes" => {
                stress_bytes = args
                    .next()
                    .ok_or("--stress-bytes needs a value")?
                    .parse::<usize>()
                    .map_err(|error| error.to_string())?
            }
            "--require-quality" => require_quality = true,
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    if !(1..=256).contains(&max_new) || !(1..=100_000).contains(&stress_bytes) {
        return Err("max-new must be 1..=256 and stress-bytes must be 1..=100000".into());
    }
    let fixtures = cases();
    let fixture_hash =
        Sha256Digest::compute(&serde_json::to_vec(&fixtures).map_err(|error| error.to_string())?)
            .to_hex();
    let results = fixtures
        .iter()
        .map(|case| {
            run_case(case, max_new).unwrap_or_else(|error| {
                json!({
                    "case": case, "evaluation_error": error
                })
            })
        })
        .collect::<Vec<_>>();
    let structural = controls(stress_bytes)?;
    let count = |pointer: &str| {
        results
            .iter()
            .filter(|item| item.pointer(pointer).and_then(Value::as_bool) == Some(true))
            .count()
    };
    let matched = count("/feedback_probe/matches_reference_prefix");
    let feedback_utf8_valid = results
        .iter()
        .filter(|item| item["feedback_probe"]["text"].is_string())
        .count();
    let structural_pass = structural["byte_address_identity_matches"] == 256
        && structural["periodic_phase_identity_matches"] == 256
        && [
            "unknown_carrier_rejected",
            "invalid_utf8_rejected",
            "native_utf8_roundtrip_exact",
            "causal_chunking_identical",
            "repeat_identical",
        ]
        .iter()
        .all(|&name| structural[name] == true)
        && structural["stress"]["finite"] == true
        && structural["stress"]["minimum_layer_steps"] == structural["stress"]["actual_bytes"];
    let passed = matched == fixtures.len()
        && count("/prompt_states_finite") == fixtures.len()
        && feedback_utf8_valid == fixtures.len()
        && structural_pass;
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let binary_hash =
        Sha256Digest::compute(&std::fs::read(&executable).map_err(|error| error.to_string())?)
            .to_hex();
    let report = json!({
        "schema": "native_toroidal_quality_v1",
        "unix_time_seconds": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|error| error.to_string())?.as_secs(),
        "provenance": { "checkout": env!("CARGO_MANIFEST_DIR"), "source_sha256": source_hashes(),
            "source_hash_kind": "embedded_at_compile_time", "binary_sha256": binary_hash,
            "architecture": std::env::consts::ARCH, "os": std::env::consts::OS,
            "command": std::env::args().collect::<Vec<_>>(),
            "fixture_sha256": fixture_hash, "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "glimmer_feature": false, "teacher": "absent_by_construction", "network_calls": 0,
            "models_loaded": 0, "vaults_loaded": 0, "tokenizer_lexicons_loaded": 0,
            "fitting_or_learning": false, "layers": LAYERS, "coordinates": DIM },
        "summary": { "cases": fixtures.len(), "identity_controls_exact": count("/identity_control_exact"),
            "prompt_readout_exact": count("/prompt_readout_exact"),
            "prompt_readout_utf8_valid": results.iter().filter(|item| item["prompt_readout_text"].is_string()).count(),
            "continuation_reference_prefix_matches": matched,
            "continuation_first_byte_correct": count("/feedback_probe/first_byte_correct"),
            "feedback_utf8_valid": feedback_utf8_valid,
            "feedback_constant_output": count("/feedback_probe/constant_output"),
            "finite_prompt_states": count("/prompt_states_finite"),
            "cases_with_evaluation_errors": results.iter().filter(|item| item["evaluation_error"].is_string()).count(),
            "structural_controls_passed": structural_pass },
        "structural_controls": structural,
        "quality_gate": { "passed": passed, "production_generation_supported": false,
            "missing": ["validated next-carrier prediction conditioned on toroidal state", "learned or validated termination behavior"],
            "note": "Prompt readout is not a continuation. Feedback is an untrained diagnostic probe; generic serving quality is not established." },
        "results": results
    });
    let serialized = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    std::fs::write(&output, serialized)
        .map_err(|error| format!("write {}: {error}", output.display()))?;
    println!("{}: {}", output.display(), report["summary"]);
    println!("quality_gate={passed}; production_generation_supported=false");
    Ok(!require_quality || passed)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(2),
        Err(error) => {
            eprintln!("native qualification failed: {error}");
            std::process::exit(1);
        }
    }
}
