# Additional held-out repository prose

## Purpose

Supply a fresh document check after correcting the first trainer's regularization
from its training-loss diagnostics. The earlier three-document prose evaluation
is retained as a validation comparison after that engineering iteration.

## Inputs/outputs

Two existing repository documents, 20,262 UTF-8 bytes in total, copied unchanged
with source paths and hashes in `manifest.json`. They were selected before scoring
the corrected trainer and never used for fitting, normalization, or checkpoint
selection. They were checked for full-document containment against training.
No claim is made that all common words or concepts are absent from training.

## Dependencies

Repository text, the pinned manifest, and the Rust `native_toroidal evaluate`
command. The training split stays frozen in `../native_repo_prose_v1/train/`.

## Qualification

See [the measured report](../../NATIVE_NEXT_ATOM_REPORT.md). This is a small
technical-document check, not a representative general language benchmark. Its
teacher-forced metrics use true preceding bytes; unconstrained generation is
reported separately. Do not tune training against this split.
