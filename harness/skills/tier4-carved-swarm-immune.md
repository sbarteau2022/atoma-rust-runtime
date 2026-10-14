# Tier 4–6 Carved Swarm & Immune System Qualification

## Plain Language

Verifies the complete 7-step chain (`Distill → Refract → Carve → Silo → Agent → Govern → Immune-check`) across the Three-Layer Monster Hybrid Server without modifying the frozen base model weights or touching `core_lm/` while the 8-token baseline comparison is running (`[verified]`).

## Workflow

1. Freeze the base model weights into an immutable `FrozenBaseWeights` allocation and record its SHA-256 fingerprint (`[verified]`).
2. Boot `MonsterHybridServer::boot_default`, which initializes the approved icosahedral `Atlas13` (`|cos θ| = 1/√5`), sets the minimal icosahedral-neighbor `OverlapPolicyMatrix`, and spawns all 1 + 12 role processes under the Layer 2 `Governor` (`[verified]`).
3. Ingest teacher taps via `ingest_teacher_tap`, executing Stage 1 (`Tap`), Stage 2 (`Hash`), Stage 3 (`Project` with `ProjectionLossReport`), Stage 4 (`Refract` via bit-exact `Q32_31` optical lens `n(r) = 2/(1-r²)`), and Stage 5 (`Carve × 12` with `CarveLossReport`) (`[verified]`).
4. Dispatch inter-role handoffs via `handoff_between_roles`, verifying that only atoms inside the shared boundary basin `V_from ∩ V_to` under a positive policy allowance `Ω_{from,to} > 0` receive a Layer 3 `CompliancePermit` and traverse the `AtomicRingBuffer` (`[verified]`).
5. Simulate a role crash via `Governor::simulate_crash` and verify `step_supervision` recovers the role in `< 50ms` (`[verified]`).
6. Run `ImmuneSystem::evaluate_kill_criteria` against preregistered fingerprint `0x94ce_0283_17e0_ecac` (`>= 5%` KL reduction on `>= 17/24` held-out families and 95% CI lower bound `> 0`) (`[verified]`).

## Edge Cases

- **Unjustified overlap expansion**: `OverlapPolicyMatrix::set_pair_overlap` with an empty justification string or negative `Ω` is rejected with `ComplianceViolation::UnjustifiedOverlapExpansion` (`[verified]`).
- **Private interior handoff**: Attempting to hand off an atom that sits inside the sender's private core rather than the shared boundary overlap `V_from ∩ V_to` is rejected with `ComplianceViolation::AtomOutsideBoundaryOverlap` (`[verified]`).
- **Antipodal direct handoff**: Attempting to hand off directly between opposite poles (`Ω = 0` by default) is rejected with `ComplianceViolation::UnauthorizedRolePair` (`[verified]`).
- **Poincaré rim overflow**: Points with `r² > MAX_BALL_RADIUS_SQ_Q32` (`r > 0.9990`) are rejected before `x0 * x0` can overflow `Q32_31` (`[verified]`).
