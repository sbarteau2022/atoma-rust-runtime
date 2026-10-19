# Native toroidal forward-pass audit

Measured 2026-10-19 UTC (2026-10-18 local). **Language continuation gate: FAILED, 0/24.**

## Purpose

Determine whether the newly connected native toroidal input/readout path produces useful
continuations without a teacher, checkpoint tokenizer, stored model map, or canned operator.

## Inputs and outputs

The [current JSON evidence](native_toroidal_quality.json) contains 24 fixed completion cases,
their exact references, prompt-position carrier IDs, strict UTF-8 results/errors, and 32-byte
self-feedback traces. Each case uses fresh 52-field, six-coordinate recurrent state. References
are used only for scoring; there is no fitting, priming with answers, or learning.

Fixture SHA-256: `86c5cc14933bee28362313e64cc78b813c4fbc4e82c20aa7772ac72122257394`.
The executable's SHA-256 and its compile-time source hashes are recorded in the JSON. Both
were checked against the executable and checkout after the run.

The [pre-fix baseline](native_toroidal_baseline.json) preserves the earlier source fingerprints
and the same completion fixtures. Its longer-input probe used 4972 bytes; the updated probe
used 100000 bytes. Those timings are not a matched performance comparison.

| Measurement | Before structural fixes | After structural fixes |
| --- | ---: | ---: |
| Canonical byte addresses read back correctly | 256/256 | 256/256 |
| Same addresses after ±one phase turn | 10/256 | 256/256 |
| Unknown atom rejected | No | Yes |
| Prompt bytes read back correctly after recurrence | 30/484 | 385/484 |
| Exact whole-prompt readouts | 0/24 | 1/24 |
| Prompt readouts that are valid UTF-8 | 24/24 | 20/24 |
| Completion reference prefixes matched by feedback probe | 0/24 | 0/24 |

The lower UTF-8 validity in the updated run is a real failure, not repaired or hidden. Earlier
outputs mostly repeated a single ASCII character; that trivially gave valid UTF-8. Retaining
more input does not make an untrained nearest-address head a text generator.

For `2 + 2 = `, the original readout was `22222222`. The updated prompt readout is
`2 8 2 ( `. Self-feedback returns 32 spaces; the expected completion is `4`. All 24 updated
feedback traces produce constant carriers. Their prompts end in a space, and the head
continues choosing that carrier. This is a fixed point, not successful generation.

## Defects fixed

- **Depth collapse:** the standalone recurrence applied an 18% blend at every layer. Its
  direct input path had weight `0.82^52`. Mixing is now distributed over the stack so that
  the product of direct input weights is 0.82. Map-backed layer substitution is unchanged.
- **Phase discontinuity:** nearest-address output distance now wraps each T² phase in radians.
  A complete turn no longer changes the decoded carrier.
- **Silent data loss:** native ML-IR egress uses checked expansion and rejects unknown IDs
  rather than dropping them. Non-byte carriers and invalid UTF-8 also return errors.
- **Stale session state:** `/reset` clears the native recurrence and both hidden-state caches
  along with the legacy cursor, while preserving other sessions and the central map.
- **Partial progress on invalid input:** an entire native input sequence is validated before
  any position updates the recurrent fields.

The API identifies the diagnostic as `native_output_kind: "prompt_readout"` and
`native_output_is_continuation: false`. Console diagnostics escape control characters.

## Dependencies and boundaries

The Rust qualifier calls the input embedding, toroidal recurrence, carrier readout, and byte
boundary converter directly. It never constructs the legacy serving runtime. No teacher,
checkpoint, vault, tokenizer lexicon, or network endpoint is loaded or contacted. The probe
uses the native CPU implementation; it does not establish Metal model throughput.

All 52 fields completed 100000 input-byte steps with finite outputs and accumulators.
Streaming in chunks and processing the same input in one call produced identical states;
fresh repeated runs were identical. This is numerical/component evidence only. Prompt
diagnostic capture still uses O(prompt bytes) storage per active session.

The structural fixes do not supply learned per-layer transforms or a next-carrier prediction
objective. The current output head selects among bytes 0..255 and has no EOS/termination
decision. A validated next-atom transition model and explicit termination are required
before this path can replace the primary language response.

## Qualification

```sh
cargo run --release --no-default-features --manifest-path orchestrator/Cargo.toml \
  --bin qualify_native_toroidal -- --stress-bytes 100000 \
  --output benchmarks/native_toroidal_quality.json --require-quality
```

The report is written and this command exits **2**, because the quality gate fails.
Harness/I/O failures use exit 1.

Regression checks passed: 56 immune library tests; 24 orchestrator library tests; 13 full-chain,
4 map-toroidal, and 3 qualification-audit integration tests. Default-feature all-target
compilation and `git diff --check` passed. The existing map/operator audits exercise separate
legacy capabilities and are not evidence of native continuation quality. Glimmer feature
integration, a deployed-service restart, matched-model parity/quality, and matched performance
were not part of this run. See [the contract](../docs/NATIVE_TOROIDAL_QUALIFICATION.md).
