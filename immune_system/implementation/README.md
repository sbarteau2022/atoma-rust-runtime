# atoma_immune (`immune_system/implementation`)

## Purpose

Implements the DHNG II 5-stage optical refraction pipeline (`n(r) = 2/(1-r²)`), the DHNG III 12-around-1 icosahedral/toroidal phase atlas (`Atlas13`), the 12 carved coordinate silos (`CarvedSwarmMap`), and the Layer 3 finite-state compliance automatons (`ImmuneSystem`) (`[verified]`).

## Inputs/Outputs

Consumes `TeacherTapRecord`s (frozen layer states + next-op probabilities), `FrozenBaseWeights`, and `OverlapPolicyMatrix`; produces `FrozenShard`s (SHA-256 hashed), `ProjectedTangent`s with `ProjectionLossReport`s, bit-exact `RefractedPoint`s with `RefractionLossReport`s, 12 `SiloView`s with `CarveLossReport`s, and `CompliancePermit`s (`[verified]`).

## Dependencies

`core_lm` (`../../core_lm`) and Rust `std` (`[verified]`).

## Qualification

Run `cargo test --manifest-path Cargo.toml` from this directory, or run the full chain qualification in [../../orchestrator/tests/full_chain_qualification.rs](../../orchestrator/tests/full_chain_qualification.rs) (`[verified]`).
