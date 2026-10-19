# Native next-atom predictor (Phase B)

## Purpose

Implement the missing producer in `SPEC_atomic_ir.md` §4.9: predict the **next**
native carrier from a causal toroidal state, including a learned end-of-sequence
decision. This extends the standalone execution plane in `design/DESIGN.md`.
It does not qualify or promote a replacement for the primary server response.

## Inputs and outputs

- Closed alphabet `native-utf8-v1`: byte IDs 0..255, BOS 200000 (input only),
  EOS 200001 (output class 256). No checkpoint vocabulary or macro IDs are
  prediction classes. BOS and EOS cannot collide with byte carriers.
- Each document/request starts with BOS and fresh 6D, 52-field toroidal state.
  The existing phase encoder and per-completed-rotation radius expansion run
  for every input. Text is converted directly from UTF-8 bytes to ML-IR IDs.
- A regularized softmax readout has 257 output classes. Its inputs are a bias,
  current-carrier identity (257 possibilities including BOS), and 167 continuous
  features from the final hidden state and the first/last recurrent fields.
  These features use Lorentz coordinates, periodic sine/cosine phase, normalized
  causal moments, phase-encoder state, expanding radius, and elapsed positions.
  The original 39 summaries are followed by a fixed eight-position delay bank
  of actual final toroidal outputs (including BOS). Each slot contains four
  bounded Lorentz coordinates, sine/cosine of both phases, and a presence bit
  (72 features). Seven coordinatewise products bind the current slot to each
  earlier slot (56 features). Missing slots and their bindings are zero. Slots
  are newest first and update only after an input has traversed every field.
  This supplies explicit recent order and nonlinear interactions; it is neither
  a lookup of text continuations nor a learned recurrent transition. State size
  is constant, and no exponential forgetting or radius cap is added.
  The carrier branch is an output-head bigram residual, not a tokenizer or a
  semantic embedding. Only the readout is trained; the recurrence stays fixed.
- Training examples are `(prompt, completion)` pairs or whole UTF-8 documents
  (empty prompt). Prefill has no loss. At completion position i, the input
  contains only BOS, prompt, and completion bytes before i. The target is byte i;
  after the final byte the target is EOS. Empty completions train EOS directly.
- Training uses deterministic minibatch Adam with L2 regularization of deviations
  from the training-only smoothed bigram initialization (zero for context
  coefficients). This preserves the learned byte-statistics prior. Normalization
  is fitted on training features only. The epoch budget is fixed; the saved
  checkpoint has the lowest **training** NLL, including initialization as epoch 0.
  Held-out targets do not select weights or hyperparameters. No teacher calls,
  checkpoint weights, maps, or benchmark completion references enter training.
- Greedy generation predicts from prefill, feeds each chosen byte through the
  same recurrence, and stops on learned EOS or an explicit byte budget. EOS is
  not emitted as text. No token/UTF-8 grammar mask, correction, fallback, forced
  stop, or repetition heuristic changes a model decision. Invalid/incomplete
  UTF-8 is returned with raw byte IDs and an error, never silently repaired.

## Dependencies and artifact contract

`native_predictor.rs` uses the native input geometry, toroidal recurrence,
boundary reader, serde, and the existing SHA-256 implementation. The standalone
`native_toroidal` executable provides train/evaluate/generate operations without
constructing a server or accessing the network; it rejects `glimmer` builds.

Artifacts bind the alphabet, feature schema, recurrence configuration, compiled
source fingerprint, corpus hashes, normalization, optimizer settings, and learned
weights and selected training epoch. Loading checks a bounded binary envelope, SHA-256 integrity, version,
source compatibility, dimensions, and finite parameters. Saving is atomic and
refuses to replace an existing file. SHA-256 detects corruption; it is not an
authenticity signature. Do not load artifacts from an untrusted source.

## Qualification

Ordered-history experiment contract (declared before implementation): keep the
frozen training documents, optimizer, epoch budget, and seed unchanged. Compare
the full model with a separately fitted `--without-ordered-history` control and
post-fit removal of the 128 history features. The control has identical parameter
allocation but receives zeros in those channels during both fit and inference.
Reuse of existing prose and completion sets is regression/validation, not a new
unseen test. Report NLL, byte accuracy, free generation, and EOS even if worse.
Mechanism tests must distinguish reordered equal-count prefixes with the same
last byte, verify exact slot ordering, missing-slot handling, chunked streaming,
bounded long-stream state, and shifted targets. Version the schema and preserve
v2 sources/artifacts; incompatible artifacts must continue to fail closed.

- Unit/integration gates: shifted targets, masked prompt, EOS including empty
  completion, deterministic optimization, real loss reduction, byte feedback,
  strict Unicode egress, state isolation, malformed/corrupt/incompatible artifacts,
  input bounds, and rejected train/evaluation document overlap.
- Evaluation reports teacher-forced next-byte/EOS loss and accuracy, EOS recall,
  raw greedy continuations, termination and UTF-8 outcomes. Labels used to score
  are never fed to greedy generation. Teacher-forced metrics are conditional on
  true preceding bytes and must not be described as generated-answer accuracy.
- Controls: training-only unigram and current-byte/bigram models, and removal of
  the head's continuous toroidal contribution at evaluation. This separates
  basic byte statistics from evidence that the toroidal history helps.
- The initial experiment's regularization incorrectly shrank the byte-statistics
  initialization toward zero. Its reports, artifact, and source archive are
  retained. The correction and training-only checkpoint selection were chosen
  from training-loss diagnostics. The original prose evaluation becomes a
  validation comparison after this engineering iteration; a separate frozen
  document set supplies a new held-out check. The fixed 24 completion cases
  remain a regression benchmark, not a pristine unseen language test.
- A small repository-prose corpus can establish an implemented training path,
  not general language ability. Synthetic sequences are permitted only as
  explicitly labeled mechanism tests. Historical 0/24 untrained evidence stays
  intact. Promotion still requires `design/QUALIFICATION_GATES.md` and a new,
  representative held-out language/instruction evaluation.
