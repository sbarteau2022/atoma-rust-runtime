---
title: "Atomic IR — pinned atoms for lossless drafting now, the tokenless edge later"
document_type: "architecture"
version: "1.0.0"
date: "2026-09-27T22:20:00Z"
implementation_date: "not implemented — specification"
author: "Claude (verifier/architect)"
reviewers: ["Stewart Barteau"]
status: "draft — gate values PROPOSED; the commander locks them before any build (R4)"
related_files:
  - path: "Desktop/Model Chat:IDE/Qwen3.8-27B-8bit/draft-head/experiments/custom_mtp_bf16/config.json"
    lines: "1-40"
    role: "supporting"
  - path: "Desktop/Model Chat:IDE/Qwen3.8-27B-8bit/.venv/lib/python3.12/site-packages/mlx_vlm/server/generation.py"
    lines: "1153-1155"
    role: "supporting"
  - path: "Desktop/Model Chat:IDE/Qwen3.8-27B-8bit/chat_template.jinja"
    lines: "62-73"
    role: "supporting"
related_issues: []
related_documents: ["PREREG_native_runtime_op1.md", "STATE_2026-09-26.md", "../DOCUMENTATION_STANDARDS.md", "../PROTOCOL.md"]
tags: ["atomic-ir", "speculative-decoding", "draft-head", "tokenless-edge", "grammar"]
---

# Atomic IR

## 1. Executive Summary
We define a set of "atoms": fixed, numbered units that each stand for an exact run of the model's existing tokens. A draft head that proposes atoms instead of single tokens can guess whole structures at once (a tool-call wrapper, an IR instruction, a common code phrase), and the main model still checks every token, so the output stays identical to the model alone. Because the structure lives in the drafter's alphabet rather than in a mask on the main model, fast speculative decoding stays on even for tool calls. The same atoms become the output units of the tokenless runtime later.

## 2. Problem Statement
- **Measured (2026-09-27):** live decode is 24.2 tok/s with the stock MTP drafter, ~2.5× the 9.68 tok/s no-draft baseline.
- **Measured:** the custom draft head (`custom_mtp_bf16`) accepts 2.2% of drafts. Its config shows the full 248,320-token vocabulary: it is an undertrained full-vocabulary head, with no special alphabet.
- **Measured:** the serving stack refuses structured output and speculation together (`server/generation.py:1153`: "Structured response_format is not supported with speculative decoding"). Guaranteed-valid tool calls therefore cost the ~2.5× speculative speed-up on those turns.
- **Derived:** tool-call scaffolds (`<tool_call>\n<function=`, `<parameter=`, `</parameter>\n`, `</function>\n</tool_call>`) and IR/code syntax span several tokens but are near-deterministic. Drafting them one token at a time spends draft depth on text the target would accept anyway.
- **Measured (2026-09-27; 5 Studio sessions, 20,456 generated tokens, Qwen tokenizer):**
  - 47.4% of the agent's output is inside tool-call blocks (mean 128 tokens per block);
  - 17.5% repeats runs of ≥ 8 tokens already present in the session;
  - at most 35.1% is free text.
  - The sample is small, so treat these as directional.
- **Grounded:** a training-free suffix-tree drafter (SuffixDecoding, arXiv 2411.04975) reports up to 5.3× on agentic workloads, where output repeats context. That makes it the baseline atoms must beat.
- **Measured (2026-09-27, suffix drafting now live):**
  - On a copy-heavy task, decode goes from 21.4 to 50.0 tok/s with a 6-token draft cap, with identical output. Prose is unchanged.
  - **Verify-cost cliff:** an 8-token cap gives 34.4 tok/s and a 16-token cap gives 4.5, even at 98–100% acceptance. On this hybrid model, verify blocks above ~7 tokens cost far more than they return.
  - Any atom drafter must therefore keep its expanded block ≤ 7 tokens, or the verify path must be fixed first (a native-runtime target).
  - This caps the multi-token advantage of atoms until the cliff is removed.
- **Architecture (STATE §A):** the end state has no tokens; text exists only at the edge, and the TIT returns "the next pinned ID or representation". That needs a unit that exists before the TIT does.

## 3. Solution Overview
An atom is a pinned integer ID that expands to a fixed sequence of target token IDs. The alphabet contains every target token (so any text is representable), plus mined multi-token atoms, plus structural atoms for tool calls and IR. The draft head predicts atoms; the verify step expands them to tokens and the target checks the whole block in one forward pass, accepting the longest matching token prefix. The output is exactly what the target alone would produce (lossless).

Structured regions get a grammar over atoms applied to the **drafter**, never a logit mask on the target, so the `generation.py:1153` exclusion never triggers. Phase B keeps the same alphabet and swaps the producer: the TIT emits atom IDs directly, and text is decoded only at the edge.

## 4. Implementation Details (the specification)

### 4.1 Definitions
- **Target vocabulary V**: the Qwen3.8-27B tokenizer's 248,320 token IDs.
- **Atom a**: `{ id: u32, expansion: [tok_id; k ≥ 1], class }`, where class ∈ {token, span, struct}.
- **Alphabet A_v**: the versioned set of atoms, including all of V as single-token atoms (class `token`, ids 0..|V|−1, identity expansion).
- **encode(tokens) → atoms**, **expand(atoms) → tokens**; text ↔ tokens stays the existing tokenizer at the edge.

### 4.2 Invariants (every one is tested; see §5)
- I1 **Coverage:** every token sequence encodes (single-token atoms guarantee it).
- I2 **Exact round trip:** `expand(encode(t)) == t` for all t.
- I3 **Canonical:** encode is deterministic — greedy longest match over a trie of expansions, ties broken by the lowest atom id.
- I4 **Streaming:** encoding a prefix of t yields a prefix of encode(t), except for the last atom, which may still grow (the decoder holds one pending atom).
- I5 **Versioned and pinned:** `ALPHABET_v<n>.json` is immutable once published. Its sha256 is appended to the ledger, and a runtime refuses an alphabet whose hash doesn't match.
- I6 **Lossless speculation (Phase A):** the emitted token stream equals target-alone greedy decoding, bit for bit.

### 4.3 Building the alphabet
- Sources:
  - `struct` atoms: the tool-call envelope from `chat_template.jinja` lines 62–73, the ⟦tool⟧ block syntax, MLIR/LLVM IR syntax (op names, dialect prefixes, type spellings), and Rust/Python/TS keyword clusters.
  - `span` atoms: frequent token n-grams mined from the operator's own traffic (Studio chat-sessions, OPS_BOARD reports, the local code corpus).
- Selection criterion: expected accepted tokens per verify step, i.e. frequency × length × the target's own greedy agreement on that span (measured by teacher forcing on the frozen target). Frequency alone overweights spans the target doesn't produce.
- Size budget: |A \ V| between 8,192 and 32,768 multi-token atoms. For comparison, EAGLE-3 drafts over a reduced vocabulary of 32,000 tokens.
- No private material (PAMI, WORKING_THEORY) is used in mining unless the commander lifts R8 for this order, and mining runs locally only.

### 4.4 Encoding and decoding
Rust crate `atoms` (house standard: integer, deterministic, no floats in the core):
- A double-array trie over expansions; greedy longest match; streaming decoder with one pending atom.
- `encode`, `expand`, `load(alphabet, expected_sha256)`.
- Property tests for I1–I4. A Python binding exists only for the mining and eval harness.

### 4.5 Grammar over atoms (structured regions)
- Tool calls and IR are regular or context-free over `struct` atoms, and the drafter's proposals are constrained by that grammar, which is cheap because the structural alphabet is small.
- The target is never masked. A well-formedness check runs **after** acceptance, and a violation is reported as a model error, not silently fixed.
- This moves guaranteed structure from "grammar mask XOR speculation" to "speculation with a structured drafter".

### 4.6 The drafter (atom head)
- An MTP-shaped head with its output layer over A (|V| plus the multi-token atoms, or a reduced core set as EAGLE-3 does), trained by teacher forcing on the frozen target's hidden states. This is the training path already planned in `draft-head/README.md`; the backbone is never updated.
- This is the proposed job for the custom head: retrain it as an atom head rather than a second full-vocabulary MTP.
- Block and tree drafting (DDTree / TreeFlash) work unchanged over atoms.

### 4.7 Verification (Phase A)
Expand the drafted atoms to a token block, run one target forward over the block, and accept the longest prefix where the target's greedy token matches. The accepted prefix may end mid-atom. Emit the accepted tokens, plus the target's correction token as usual. Acceptance is reported in tokens and in atoms.

### 4.8 Artifacts
- `atoms/ALPHABET_v1.json` (+ sha256 in the ledger)
- `atoms/` Rust crate
- `atoms/mine.py` (local)
- `atoms/eval/` — the frozen corpora and the harness that computes every gate in §5 from one command
- A serving flag: `ELLE_DRAFT_KIND=atom` in `draft-head/env.sh`. Stock MTP stays the default.

### 4.9 Phase B contract (for the TIT; semantics are the commander's)
- The TIT emits atom IDs from a pinned alphabet version, and the edge decodes them to text.
- Whether atoms map onto pinned IDs or sphere-map addresses, and what a non-token atom *means*, is defined by WORKING_THEORY (R8, local only) and is out of scope here.
- This spec guarantees only the carrier: a closed, versioned, losslessly decodable alphabet.

## 5. Testing Evidence
None yet: this is a specification. These gates are **proposed**. The commander fixes the numbers before the first run (R4), and after that they never move (R7).

| Gate | What it proves | Proposed threshold |
|---|---|---|
| G1 round trip | I1–I3 | `expand(encode(t)) == t` for 100% of a frozen corpus (Studio sessions, OPS_BOARD reports, tool calls, IR samples) |
| G2 lossless | I6 | 0 token divergences vs target-alone over the frozen X004 Gate-3 set (≥ 3 prompts × 32 greedy steps) plus 36 tool-call requests |
| G3 acceptance | the point of atoms | mean accepted tokens per verify step ≥ 1.20× **stock MTP + suffix (prompt-lookup) drafting** on the tool-heavy suite; ≥ 0.95× on the prose suite |
| G4 throughput | the speed claim | end-to-end decode tok/s ≥ 1.10× **stock MTP + suffix drafting** on the tool-heavy suite; ≥ 1.00× on prose |
| G4b validity | structure without losing speculation | 0 malformed tool calls with speculation on, vs today's choice of grammar mask (target-only speed) or unconstrained drafting |
| G5 control ⚠ | the gain comes from fit, not size | an equal-size alphabet mined from a mismatched corpus (or random spans of matched length) must show a G3 gain at least 50% smaller; otherwise the claim fails |
| G6 determinism | I3, I5 | encode/expand bit-identical across 3 processes and 2 builds; alphabet hash matches the ledger |

Reproduce: one command, from OPS_BOARD, regenerating every number above: `atoms/eval/run_all.sh` (to be written with the build).

## 6. Deployment / Rollout
Behind `ELLE_DRAFT_KIND=atom` only. Stock MTP remains the serving default until G2, G3 and G4 pass and the verifier re-runs them (R6). Rollback means unsetting the flag. The alphabet version is pinned per deployment.

## 7. Next Stage of Implementation
| Next step | Description | Depends on this? |
|---|---|---|
| Lock gates | Commander fixes the G3–G5 thresholds | Yes |
| Corpora | Freeze the tool-heavy and prose eval suites; hash them | Yes |
| Mining v0 | `struct` atoms + top span candidates, scored by target agreement | Yes |
| `atoms` crate | Trie, encode/expand, property tests for I1–I4 | Yes |
| Atom head | Retrain the custom head on atoms (teacher forcing, frozen backbone) | Yes |
| Verify path | Expand-then-verify inside the speculative loop; partial-atom acceptance | Yes |
| Native runtime | Carry the atom verify path into op1's streaming loop (PREREG_native_runtime_op1) | Parallel |
| TIT (Phase B) | Emit atoms directly; edge decode | Yes — this is the carrier |

This specification **does** set us up for the tokenless edge, because it fixes the carrier (a pinned, lossless alphabet) before the producer (the TIT) exists. It also makes structured output compatible with speculation today.

## 8. What We Learned (from the work that led here)
| Category | Learning |
|---|---|
| Technical | The 2.2% custom-head result was undertraining over the full vocabulary, not a wrong-target measurement: the config proves the output space. |
| Technical | Grammar masking on the target and speculation are exclusive in this stack (`generation.py:1153`); structure belongs in the drafter. |
| Process | Live throughput (6.5 tok/s) differed from the benchmark (19.4) because of swap, not the drafter. Measure in the real serving conditions. |

## Rationale (PROTOCOL §5.2)
- **Assumptions:**
  - A1 (checked): the target's tokenizer is fixed and shared by the drafter.
  - A2 (hypothesis): tool, IR and code traffic contains long near-deterministic spans the target agrees with. G3 tests this, and G5 is its control.
  - A3 (hypothesis): an atom head can be trained from frozen hidden states within the draft-head plan's budget.
- **Decisions and rejected alternatives:**
  - Mask the target with a grammar: rejected, because it disables speculation (§2).
  - A new tokenizer for the target: rejected, because it breaks the pinned weights.
  - A reduced-vocabulary EAGLE-style draft only: rejected as insufficient, because it cannot draft multi-token units. Its vocabulary reduction remains an option for the atom head's output layer.
- **Known weaknesses:**
  - The gain depends on traffic mix. Prose may see no gain, which is why G3 and G4 have separate prose bars.
  - Partial-atom acceptance complicates the drafter's state after a mid-atom rejection.
  - Mining on the operator's own traffic can overfit to it, which is why G5 exists.
- **Grounding** (titles and abstracts read during this session; full texts not yet read):
  - "Speculative Decoding with a Speculative Vocabulary", arXiv 2602.13836
  - DFlash, arXiv 2602.06036
  - TreeFlash, arXiv 2606.03819
  - EAGLE-3's reduced 32k draft vocabulary (vLLM / Red Hat engineering posts)

## Changelog
| Version | Date | Author | Change |
|---|---|---|---|
| 1.0.0 | 2026-09-27 | Claude | Initial specification; gates proposed, awaiting the commander's lock |
| 1.1.0 | 2026-09-27 | Claude | Traffic measured. The baseline is raised from stock MTP to MTP + training-free suffix drafting, because atoms must beat the free method. G4 bar lowered to 1.10× against the stronger baseline. G4b (tool-call validity with speculation on) added. |
