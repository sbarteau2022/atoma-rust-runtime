# Native toroidal forward-pass qualification

## Purpose

Evaluate the UTF-8 → ML-IR byte carriers → six-coordinate H³×T² input → 52 recurrent
toroidal fields → byte-carrier readout → strict UTF-8 path independently of checkpoints,
teachers, vaults, benchmark operators, and canned response fallbacks.

This implements an evidence gate for the execution plane described in `design/DESIGN.md`
and the Phase B producer/edge boundary in `SPEC_atomic_ir.md` §4.9. It does not change the
architecture or relax `design/QUALIFICATION_GATES.md` promotion requirements.

## Inputs and outputs

`qualify_native_toroidal` uses a fixed, hashed set of short completion references covering
arithmetic, text, facts, code, Unicode, and instructions. Each case starts with fresh recurrent
state. The references are read only for scoring; no fitting, priming, or map learning occurs.
The JSON report includes the full inputs, expected continuations, readout carrier IDs, strict
UTF-8 results/errors, timing, fixed-point behavior, and source hashes.

Two operations are measured separately:

1. **Prompt readout:** classify each prompt position's hidden state. Its length is tied to the
   input length. This is not a completion. Input-address decoding without recurrence is the
   identity control.
2. **Feedback probe:** read the last prompt state, feed the selected byte's canonical address
   back through the recurrence, and repeat for a fixed byte budget. The current head has no
   learned next-byte objective and cannot emit EOS. This is an experimental failure probe,
   not a production generation API. No reference bytes are supplied during feedback.

The report also checks periodic phase equivalence, exact native UTF-8 round trips, rejection
of unknown/non-byte carrier IDs, deterministic repeated runs, and causal streaming equivalence.
Finite-state behavior is checked on a longer input. A 100k-byte run measures numerical
stability only; long-context answer quality, VRAM, and matched throughput remain open.

## Dependencies

Only the Rust native embedding, toroidal recurrence, output readout, and boundary converter
are called. The executable rejects builds with the `glimmer` feature. It never constructs a
`MonsterHybridServer` or `MainModelCoordinatorRuntime`, loads model files/vaults, or calls a
network endpoint. The optional legacy tokenizer lexicon loader is not invoked.

## Qualification

Run from the repository root:

```sh
cargo run --release --no-default-features --manifest-path orchestrator/Cargo.toml \
  --bin qualify_native_toroidal -- --stress-bytes 100000 \
  --output benchmarks/native_toroidal_quality.json --require-quality
```

The executable always records the evidence. `--require-quality` makes a failed quality gate
exit with status 2. Harness/I/O failures exit with status 1. `--stress-bytes N` controls the
longer input (default 4096); `--max-new N` controls feedback length (default 32, maximum 256).

Promotion requires all easy completion references to match, finite outputs, valid UTF-8, and
all applicable structural controls to pass. Passing that small suite would still be insufficient
for broad language quality. A failed suite must leave the primary serving response unchanged.

This untrained qualifier retains its original diagnostic head and historical evidence. A
separate learned next-byte/EOS producer now implements the training and generation mechanics
described in [NATIVE_NEXT_ATOM_DESIGN.md](NATIVE_NEXT_ATOM_DESIGN.md). Its
[measured qualification](../benchmarks/NATIVE_NEXT_ATOM_REPORT.md) must establish language
and termination quality independently; a reversible carrier converter does not provide them.
