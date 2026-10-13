# Harness

## Purpose

Reserved home for reproducible qualification runs that decide whether a component may enter the global harness.

## Inputs/Outputs

Consumes versioned backend profiles and test vectors; emits immutable parity, throughput, and adversarial-audit reports.

## Dependencies

`core_lm`, the documented backend profiles, and the selected model server.

## Qualification

Tier 0's unit gate lives in `core_lm`; live qualification is blocked until both backend contracts are captured.
