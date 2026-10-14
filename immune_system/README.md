# Immune System (Layer 3)

## Purpose

Governs the Layer 2 Governor and enforces structural silo boundaries, optical refraction loss accounting, base-weight immutability, and preregistered kill criteria across the 12-around-1 carved swarm (`[verified]`).

## Inputs/Outputs

Consumes frozen teacher tap records (`TeacherTapRecord`), pairwise overlap policies (`OverlapPolicyMatrix`), inter-role handoff requests, and preregistered evaluation vectors (`PreregisteredEvaluation`); produces bit-exact `H³ × T²` `RefractedPoint`s, 12 carved `SiloView`s, unforgeable `CompliancePermit`s, and hash-chained `ComplianceAuditEntry` logs (`[verified]`).

## Dependencies

`core_lm` (read-only link for `Q32_31`, `LorentzVector`, `Phase`, and `lock::extract_invariant`) and the Rust standard library (`[verified]`).

## Qualification

Verified by `immune_system/implementation` unit tests (`cargo test --manifest-path immune_system/implementation/Cargo.toml`) and the end-to-end suite in [orchestrator/tests/full_chain_qualification.rs](../orchestrator/tests/full_chain_qualification.rs) (`[verified]`). See also [audit/BASELINE_CONGRUENCE_AUDIT.md](audit/BASELINE_CONGRUENCE_AUDIT.md).
