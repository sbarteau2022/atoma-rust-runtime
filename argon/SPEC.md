# Carved Swarm & Immune System — Design Spec

## 1. Architecture: Three Layers, One Server

**The Server** is central Rust infrastructure serving the entire ecosystem. All inference, all agents, all dispatch flow through it. There is one door.

### Layer 3 — Immune System (active monitoring dispatch layer)
- Governs the governor. Not the agents directly — the layer above them.
- Every dispatch passes through compliance automatons. Structural gates, not prompt filters.
- Controls silo boundaries: which role gets what map view, what knowledge crosses between silos, how much overlap each pair is allowed.
- Regulatory compliance as architecture: the unauthorized path does not exist, it is not merely blocked.

### Layer 2 — Governor (active orchestrator)
- Erlang-style supervision tree. Spawns role instances, monitors heartbeats, kills and restarts on failure or timeout.
- All inter-role messages route through the governor. Roles never communicate directly.
- The governor cannot dispatch outside immune system policy. Its action space is the subset the immune system permits.

### Layer 1 — Carved Swarm (12 phases around the 1)
- 12 specialized roles. Each is a refracted silo of the global map — hyperspecialized knowledge only.
- **Not system prompts.** A prompt is a suggestion the model can reason around. A siloed map is a world the agent cannot see past. The planner does not choose to avoid execution knowledge; those coordinates do not exist in its space.
- Silos overlap at basin boundaries. The 12 wells have overlapping attractors by geometry. Overlap is the communication channel — without shared boundary knowledge, roles cannot hand off.
- Overlap amount per pair is immune system policy, not accident.

## 2. Geometry

- **Global map**: H³ × T² — hyperbolic 3-space crossed with the dual-phase torus (θf, θs).
- **The 1**: central core attractor. The undivided map.
- **The 12**: twelve wells/phases around the core (cf. DHNG III, "Twelve Around One"). Each well is a carved role's home basin.
- **The lens**: refractive index n(r) = 2/(1−r²). Under Fermat's principle, light in this medium follows exactly the geodesics of hyperbolic space. Hyperbolic geodesics are light paths — this is exact maths, not metaphor.
- **Ball and hyperboloid are one space**: the Poincaré ball and the two-sheeted Lorentz hyperboloid carry the same distances, reproduced in integer code to 4.4 × 10⁻⁹ across 900 pairs (measured).
- **Silo boundaries are attractor basins**, not hard partitions. Basins overlap by geometry. The carving does not cut; it refracts.

## 3. The Carving Mechanism (DHNG II)

Source: `dhng2.html` — "Knowledge distillation through optical refraction."

Five stages, each read-only toward the model, each hashed:

1. **Tap** — Model stays frozen. Record layer states and next-token probabilities. No weights change.
2. **Hash** — One dump, sharded, every shard hashed. Teacher's floats become fixed bytes.
3. **Project** — Measured, lossy step from full state dimension down to a small tangent vector. This is where information is lost, so it gets measured.
4. **Refract** — The tangent vector enters curved space (H³ × T²) through the exponential map: the lens itself. Integer maths, same bits on every machine.
5. **Keep** — Positions and relations join the persistent map.

For the swarm, stage 5 runs **twelve times** — one refracted view per phase. Each view is a role's entire readable world. Quoting the source: *"Specialists are views onto one set of weights."* Nothing is duplicated. The base is never modified.

What the lens cannot do (from source, non-negotiable):
- It cannot be lossless. One token's state is ~82,000 bits; an H³ × T² point holds 256. Refraction keeps relations, not everything.
- A mirror does not reduce dimension. Reduction happens in stage 3 and costs information.
- The teacher must be open weights we run. Hosted APIs hide internal states.

## 4. The Full Chain

```
Distill → Refract → Carve → Silo → Agent → Govern → Immune-check
```

| Step | Action | Owner |
|------|--------|-------|
| Distill | Read frozen model, hash states | DHNG II stages 1–2 |
| Refract | Project + exp map into H³ × T² | DHNG II stages 3–4 |
| Carve | 12 refracted views, one per phase | DHNG II stage 5 × 12 |
| Silo | Each view is a role's entire map | Immune system policy |
| Agent | Role instance operating within silo | Governor spawns |
| Govern | Supervise agents, route messages | Governor (Layer 2) |
| Immune-check | Validate every dispatch | Immune system (Layer 3) |

## 5. Qualifications for the Implementing Agent

Non-negotiable. Violation of any of these is a stop-work condition.

1. **Proof-bounded claims.** Every statement about the system carries one of four marks: `verified` (measured on this machine), `reported` (someone else measured it), `inferred` (follows from evidence), `theory` (unproven). No mark, no claim.

2. **Read-only until the baseline lands.** Grok's 8-token forward comparison is in progress. Do not modify the loader, the geometric map, or any retrieval wiring until the congruence result is reported. Document and flag; do not change.

3. **Math decides, not preference.** "It's not about what I want, it's about what the math allows as necessary." If a design choice isn't forced by the geometry or the constraints, say so and present the alternatives.

4. **Structural, never prompt-based.** No safety, no role separation, no access control implemented as a system prompt or instruction. If it can be reasoned around, it is not a mechanism. Silos are map-level; gates are code-level.

5. **Integer discipline where the architecture demands it.** The refraction path (exp map, position encoding) runs in fixed-point integer maths, bit-exact across machines. Float is permitted only where the spec explicitly allows it.

6. **The 12 are architectural.** Twelve phases is not a tunable hyperparameter. It comes from the DHNG III twelve-around-one structure. Do not propose 8 or 16 without a geometric argument.

7. **Overlap is policy.** Silo boundaries overlap by geometry. How much overlap each role-pair gets — what knowledge they share at the boundary — is an immune system policy decision. Default to minimal; expand only with justification.

8. **The base is never modified.** Carving produces views. The underlying weights are frozen and shared. Any proposal that duplicates or fine-tunes the base weights is out of scope.

9. **Name the loss.** Every stage that discards information (projection, refraction, siloing) must state what is lost and measure it where possible. Silent information loss is a defect.

10. **Kill criteria.** If the curved student does not beat the flat baseline on the preregistered test, the lens is the wrong shape and that result gets reported just as loudly. Do not optimize the test to save the theory.

## Sources
- `dhng2.html` — DHNG II · Refraction (distillation mechanism)
- `dhng3.html` — DHNG III · Twelve Around One (phase structure)
