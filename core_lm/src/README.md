# coreLM source

## Purpose

Contains the Tier 0 scheduler, backend seam, parity gate, and executable health check.

## Inputs/Outputs

The library accepts tokenized `GenerateRequest` values and returns ordered tokenized `GenerateResponse` values; the executable prints one deterministic health-check response.

## Dependencies

The parent `core_lm` crate and Rust standard library.

## Qualification

Run `cargo test --manifest-path ../Cargo.toml` from this directory.
