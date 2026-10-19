//! Native CLI integration, using labeled mechanism fixtures, never language scores.
#![cfg(not(feature = "glimmer"))]

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("atoma-native-cli-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_native_toroidal"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .unwrap()
    }
    fn json(&self, name: &str) -> Value {
        serde_json::from_slice(&fs::read(self.0.join(name)).unwrap()).unwrap()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn saved_training_artifact_drives_standalone_generation_and_held_out_scoring() {
    let scratch = Scratch::new();
    fs::write(
        scratch.0.join("train.jsonl"),
        "{\"prompt\":\"A\",\"completion\":\"B\"}\n",
    )
    .unwrap();
    fs::write(
        scratch.0.join("eval.jsonl"),
        "{\"prompt\":\"A\",\"completion\":\"C\"}\n",
    )
    .unwrap();
    let trained = scratch.run(&[
        "train",
        "--train-jsonl",
        "train.jsonl",
        "--eval-jsonl",
        "eval.jsonl",
        "--artifact",
        "model.atoma_next",
        "--report",
        "train-report.json",
        "--epochs",
        "1",
        "--max-new",
        "8",
    ]);
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );
    let report = scratch.json("train-report.json");
    assert_eq!(report["model_config"]["layers"], 52);
    assert_eq!(report["model_config"]["ordered_history"], true);
    assert_eq!(report["schema"], "native-next-atom-report-v2");
    assert_eq!(report["learned_parameters"], 109225);
    assert_eq!(report["training"]["positions"], 2);
    assert_eq!(report["teacher_attached"], false);
    assert_eq!(report["tokenizer_lexicon_loaded"], false);
    assert_eq!(report["quality"]["all_exact_and_eos"], false);
    let generated = scratch.run(&[
        "generate",
        "--artifact",
        "model.atoma_next",
        "--prompt",
        "A",
        "--max-new",
        "8",
    ]);
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated: Value = serde_json::from_slice(&generated.stdout).unwrap();
    assert_eq!(generated["generation"]["text"], "B");
    assert_eq!(generated["generation"]["stop"], "eos");
    let scored = scratch.run(&[
        "evaluate",
        "--artifact",
        "model.atoma_next",
        "--eval-jsonl",
        "eval.jsonl",
        "--report",
        "held-out.json",
        "--require-exact",
        "--max-new",
        "8",
    ]);
    assert_eq!(scored.status.code(), Some(2));
    assert_eq!(
        scratch.json("held-out.json")["quality"]["exact_completion_and_eos"],
        0
    );
    assert_eq!(
        scratch.json("held-out.json")["evaluation"]["cases"][0]["generated"]["text"],
        "B"
    );
    let original = fs::read(scratch.0.join("model.atoma_next")).unwrap();
    let overwrite = scratch.run(&[
        "train",
        "--train-jsonl",
        "train.jsonl",
        "--eval-jsonl",
        "eval.jsonl",
        "--artifact",
        "model.atoma_next",
        "--report",
        "unused.json",
    ]);
    assert_eq!(overwrite.status.code(), Some(1));
    assert_eq!(
        fs::read(scratch.0.join("model.atoma_next")).unwrap(),
        original
    );
    assert!(!scratch.0.join("unused.json").exists());
    let contaminated = scratch.run(&[
        "evaluate",
        "--artifact",
        "model.atoma_next",
        "--eval-jsonl",
        "train.jsonl",
        "--report",
        "contaminated.json",
    ]);
    assert_eq!(contaminated.status.code(), Some(1));
    assert!(!scratch.0.join("contaminated.json").exists());
}

#[test]
fn cli_rejects_overlap_invalid_utf8_unknown_fields_and_invalid_options_before_writing() {
    let scratch = Scratch::new();
    fs::write(
        scratch.0.join("doc.txt"),
        "The same training and evaluation document.",
    )
    .unwrap();
    fs::write(scratch.0.join("other.txt"), "A separate held-out document.").unwrap();
    fs::write(scratch.0.join("bad.txt"), [0xff, 0xfe]).unwrap();
    fs::write(
        scratch.0.join("bad.jsonl"),
        "{\"prompt\":\"A\",\"completions\":\"B\"}\n",
    )
    .unwrap();
    for (flag, input, eval) in [
        ("--train-text", "doc.txt", "doc.txt"),
        ("--train-text", "bad.txt", "other.txt"),
        ("--train-jsonl", "bad.jsonl", "other.txt"),
    ] {
        let result = scratch.run(&[
            "train",
            flag,
            input,
            "--eval-text",
            eval,
            "--artifact",
            "invalid.atoma_next",
            "--report",
            "invalid.json",
        ]);
        assert_eq!(result.status.code(), Some(1));
        assert!(!scratch.0.join("invalid.atoma_next").exists());
        assert!(!scratch.0.join("invalid.json").exists());
    }
    for args in [
        vec![
            "generate",
            "--artifact",
            "absent",
            "--prompt",
            "A",
            "--without-ordered-history",
        ],
        vec![
            "train",
            "--without-ordered-history",
            "--without-ordered-history",
        ],
        vec![
            "generate",
            "--artifact",
            "absent",
            "--prompt",
            "A",
            "--max-new",
            "0",
        ],
        vec![
            "generate",
            "--artifact",
            "absent",
            "--prompt",
            "A",
            "--epochs",
            "1",
        ],
        vec![
            "generate",
            "--artifact",
            "absent",
            "--prompt",
            "A",
            "--unknown",
            "value",
        ],
        vec![
            "generate",
            "--artifact",
            "absent",
            "--prompt",
            "A",
            "--prompt",
            "B",
        ],
    ] {
        assert_eq!(scratch.run(&args).status.code(), Some(1));
    }
}

#[test]
fn whole_document_input_trains_a_bos_conditioned_first_byte_and_eos_target() {
    let scratch = Scratch::new();
    fs::write(scratch.0.join("train.txt"), "X").unwrap();
    fs::write(scratch.0.join("held-out.txt"), "Y").unwrap();
    let result = scratch.run(&[
        "train",
        "--without-ordered-history",
        "--train-text",
        "train.txt",
        "--eval-text",
        "held-out.txt",
        "--artifact",
        "document.atoma_next",
        "--report",
        "document.json",
        "--epochs",
        "1",
        "--max-new",
        "4",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report = scratch.json("document.json");
    assert_eq!(report["model_config"]["ordered_history"], false);
    assert_eq!(
        report["evaluation"]["teacher_forced"]["learned"],
        report["evaluation"]["teacher_forced"]["without_ordered_history"]
    );
    assert_eq!(report["training"]["positions"], 2);
    assert_eq!(report["evaluation"]["cases"][0]["prompt"], "");
    assert_eq!(report["evaluation"]["cases"][0]["generated"]["text"], "X");
    assert_eq!(report["evaluation"]["cases"][0]["generated"]["stop"], "eos");
}
