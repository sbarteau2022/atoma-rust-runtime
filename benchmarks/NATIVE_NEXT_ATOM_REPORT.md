# Learned native next-atom qualification

## Purpose and disposition

Implement and measure the missing native next-byte/EOS producer under
[`NATIVE_NEXT_ATOM_DESIGN.md`](../docs/NATIVE_NEXT_ATOM_DESIGN.md).
**Training and generation mechanics are implemented. Language/termination
qualification fails: 0/24 completion references matched and 0/24 EOS stops.**
The primary server response remains on its existing cascade. No running service
was restarted or replaced by this work.

## Inputs/outputs and dependencies

The standalone Rust path is UTF-8 bytes → native ML-IR carriers → canonical 6D
H³×T² addresses → 52 fixed toroidal recurrent fields → learned 257-class
next-byte/EOS softmax → strict UTF-8 reader. Predicted bytes feed back through
the same recurrence; learned EOS or an explicit byte budget ends the request.
The live phase encoder and completed-rotation radius expansion run on every
input, including BOS. Only the output head is trained; there are no trained
semantic embeddings or per-layer attention projections in this experiment.

The head has 76,329 parameters: bias, a current-byte/BOS branch, and 39 continuous
features from final hidden coordinates and first/last field statistics. Its
byte-statistics branch is a learned local continuation control, not a tokenizer.
Every request starts fresh state. No checkpoint, teacher, map, tokenizer lexicon,
network call, canned answer, UTF-8 grammar mask, or repetition correction is
used. Dependencies are the native Rust components, serde/serde_json, existing
SHA-256 implementation, and std. Builds with `glimmer` are rejected.

The frozen [training corpus](datasets/native_repo_prose_v1/README.md) contains
five existing technical documents, 53,674 bytes and 53,679 supervised positions,
including five EOS targets. Their text includes historical/proposed architecture;
training on it does not establish the truth of those documents. There are no
generated QA examples or teacher traces. Prompt/completion JSONL is also
supported, with loss applied only to next completion bytes and final EOS.

## Training correction and retained evidence

The first experiment regularized initialized byte-statistics weights toward zero.
Training NLL increased from **2.730674 to 2.818627 nats/position** over eight epochs.
Its [report](native_toroidal_trained_prose_v1.json),
[completion report](native_toroidal_trained_completions_v1.json),
[artifact](native_toroidal_repo_prose_v1.atoma_next), and
[source archive](native_toroidal_v1_sources.zip) are retained.

The correction regularizes deviations from the training-only byte-statistics
initialization, including zero-centered context coefficients. It also retains
the lowest-training-NLL checkpoint, with initialization as epoch 0. Selection
uses no evaluation labels or evaluation metrics. The fixed settings remain:
eight epochs, batch 128, Adam learning rate 0.003, L2 0.0001, seed 20261018.

The corrected [training report](native_toroidal_trained_prose_v2.json) selects
epoch **8**, with training NLL **2.693042**, below initialization's **2.730674**.
The [current artifact](native_toroidal_repo_prose_v2.atoma_next) SHA-256 is
`f7778e18703ebd914fb6faa45e6e7deff4cfd3c8eff63776e1dd7b3a2da44edb`.
Reports bind executable, source, corpus, configuration, and artifact hashes.
The current loader deliberately rejects the earlier artifact's incompatible
source/provenance format; the archive preserves its older implementation.

After this engineering correction, the original three-document prose evaluation
is a **validation comparison**. A separate [two-document holdout](datasets/native_repo_prose_holdout_v2/README.md)
was frozen before scoring the corrected model. No subsequent fitting or
hyperparameter selection used its results. The 24 completion pairs remain the
same fixed regression benchmark as the historical untrained qualifier; they are
not a pristine unseen general-language test.

## Teacher-forced next-position results

“Teacher-forced” here means supplying true **text bytes** preceding each scored
position. There is no teacher model. These metrics do not measure free-running
answer correctness. NLL includes EOS; byte accuracy excludes EOS.

| Corpus / control | Mean NLL, nats ↓ | Next-byte accuracy ↑ |
|---|---:|---:|
| Fresh holdout: unigram | 3.891750 | 12.96% |
| Fresh holdout: independently fitted bigram | 3.064191 | 21.16% |
| Fresh holdout: trained head, toroidal features removed after fitting | 3.075590 | 21.30% |
| Fresh holdout: full trained head | **3.045993** | **22.99%** |
| Earlier validation: independently fitted bigram | **2.966466** | **22.08%** |
| Earlier validation: trained head, toroidal features removed after fitting | 2.971351 | 22.48% |
| Earlier validation: full trained head | 3.001499 | 21.84% |

The fresh holdout has 20,262 byte targets plus two EOS targets; the earlier
validation set has 8,745 byte targets plus three EOS targets. The small fresh
gain reverses on the earlier validation set. Two documents cannot establish a
general or statistically robust language improvement. Removing features after
fitting is a contribution ablation, not a separately retrained equal-capacity
model. Both byte-statistics controls use training text only.

Full evidence: [fresh prose](native_toroidal_heldout_prose_v2.json) and
[earlier validation](native_toroidal_trained_prose_v2.json).

## Free-running generation and termination

The [completion report](native_toroidal_trained_completions_v2.json) records:

- Reference-prefix matches: **0/24**; exact completion plus EOS: **0/24**.
- Learned EOS stops: **0/24**; all requests reached the 32-byte budget.
- Strictly valid UTF-8: **23/24**. The dragon-repeat prompt ended with an
  incomplete UTF-8 sequence; raw IDs and the reader error are preserved.
- Teacher-forced next-byte accuracy: **7/82**; final-EOS accuracy: **0/24**.
  Full-head NLL is **8.947236**, worse than bigram **6.990520** on these positions.

Prose generation from empty prompts also reaches the 128-byte budget and enters
repetitive punctuation. No EOS is predicted at either fresh document endpoint
or the three earlier validation endpoints. Learned EOS works on the explicitly
labeled mechanism fixtures in tests; those results do not establish reliable
natural-text stopping. Five document-ending training examples are extremely
sparse supervision for EOS. Corpus size/domain, the linear head, and the fixed
causal representation remain confounded explanations for poor generation.

This implementation provides a real native optimization/generation path. It
does not qualify replacement of a traditional trained transformer. Richer
ordered native representations and representative real text/turn data with
meaningful endpoints require independent evaluation; increasing the layer count
or fitting a head alone does not establish that quality.

## Qualification and adversarial checks

Default-feature Rust suites pass: immune library **56**, orchestrator library
**35** (including 11 native predictor tests), full-chain **13**, map attention
**4**, native CLI integration **3**, and qualification audit **3**: **114 tests**.
`cargo check --no-default-features --all-targets` is clean. Optional Glimmer
qualification, live server deployment, matched transformer parity, N=2 serving
performance, and broad language quality are not established by this run.

Edge checks include shifted targets, masked prompt loss, future-label isolation,
training-only checkpoint selection, deterministic optimization, BOS/empty EOS,
byte feedback, chunk-equivalent causal features, session isolation, unmodified
invalid UTF-8 decisions, budget truncation, disjoint corpus checks, bounded
inputs, corrupt/truncated/oversized artifacts, valid-digest schema/source/shape
attacks, normalization and count consistency, refusal to overwrite, and eight
concurrent attempts at atomic publication. SHA-256 detects corruption; it is not
an artifact authenticity signature. Synthetic mechanism fixtures stay in tests.

## Reproduce

Run from the repository root. New output filenames are required:

```sh
cargo build --release --no-default-features --manifest-path orchestrator/Cargo.toml \
  --bin native_toroidal
bash -c '
  train_args=(); eval_args=()
  for p in benchmarks/datasets/native_repo_prose_v1/train/*.md; do
    train_args+=(--train-text "$p")
  done
  for p in benchmarks/datasets/native_repo_prose_v1/eval/*.md; do
    eval_args+=(--eval-text "$p")
  done
  orchestrator/target/release/native_toroidal train \
    "${train_args[@]}" "${eval_args[@]}" \
    --artifact benchmarks/native-torus-reproduced.atoma_next \
    --report benchmarks/native-torus-reproduced-training.json \
    --epochs 8 --batch-size 128 --learning-rate 0.003 \
    --l2 0.0001 --seed 20261018 --max-new 128
'
orchestrator/target/release/native_toroidal evaluate \
  --artifact benchmarks/native_toroidal_repo_prose_v2.atoma_next \
  --eval-jsonl benchmarks/datasets/native_completion_eval_v1.jsonl \
  --report benchmarks/native-torus-reproduced-completions.json \
  --max-new 32 --require-exact
```

The final command intentionally exits **2** for the failed exact-completion gate,
after writing evidence. Configuration/I/O errors exit 1. Use `evaluate --eval-text`
for the two fresh holdout files. `generate --artifact FILE --prompt TEXT` returns
raw IDs, strict text/error, EOS/budget reason, and provenance as JSON. Each report
also records its full argv. Historical untrained baseline files are unchanged.
