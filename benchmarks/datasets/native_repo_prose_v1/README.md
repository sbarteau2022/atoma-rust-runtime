# Frozen repository prose v1

## Purpose

Provide a small, auditable local corpus for exercising native next-byte/EOS
training without checkpoint weights, teacher traces, generated QA, or tokenizer
lexicons. These documents include architecture proposals and historical claims;
their inclusion as text is not verification of their contents.

## Inputs/outputs

`manifest.json` pins the original repository paths, exact bytes and SHA-256
digests. Five whole documents (53,674 bytes) belong to training; three different
whole documents (8,745 bytes) belong to evaluation. No paragraph from an
evaluation document is intentionally moved into training. Common technical
words and ordinary n-grams naturally overlap. Do not split or reshuffle this
frozen corpus after examining results.

Each file is one raw UTF-8 document: empty prompt, document bytes as the
completion, one BOS input and one final EOS target. No normalization or
checkpoint tokenization occurs. The first-byte and final-EOS positions are
included in metrics. The corpus is tiny and domain-specific; its results do
not qualify a general language or instruction model.

## Dependencies

Existing repository Markdown, Rust `native_toroidal`, and the manifest's SHA-256
checks. No external downloads or private memory files were used.

## Qualification

Use `native_toroidal train --train-text FILE ... --eval-text FILE ...` as
documented in [the learned-head report](../../NATIVE_NEXT_ATOM_REPORT.md).
The CLI rejects exact cross-split document matches and containment of documents
at least 32 bytes long. Standalone artifact evaluation can reject exact document
hashes; it cannot detect arbitrary paraphrases or excerpts of a training document.

`../native_completion_eval_v1.jsonl` preserves the same 24 prompt/reference pairs
as the historical untrained native qualification report. These references are
for scoring only and must never enter fitting. During corpus freezing, every
concatenated prompt/reference was checked against each training document and
none occurred. This check cannot prove absence of all conceptual overlap.

Synthetic `A -> B` and Unicode examples in Rust tests are mechanism fixtures;
they are not training records in this corpus or language-quality evidence.
