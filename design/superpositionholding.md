# Superposition Holding & Active Inference Specification (`superpositionholding.md`)

**Status**: `[verified]` First-Principles Runtime Specification  
**Manifold**: $(\mathbb{H}^3 \times \mathbb{T}^2) / \Gamma_{12}$ (`Q32.31` fixed-point arithmetic, `#![forbid(unsafe_code)]`)  
**Modules**:
- `immune_system/implementation/src/superposition.rs` (`SuperpositionHoldingEngine`, `VariationalFreeEnergyScore`, `SuperpositionState`, `ActiveInferenceReceipt`)
- `immune_system/implementation/src/native_reader.rs` (`NativeReader::active_inference_step`, `NativeReader::continue_ids_active`)
- `immune_system/implementation/src/first_principles.rs` (`OpticalRiskEngine`, `MetaCognitiveRunbookController`, `FirstPrinciplesComplianceEngine`, `AtomicIrAlphabet`)
- `immune_system/implementation/src/holonomy.rs` (`HyperbolicHolonomyEngine`)

## 1. Purpose & First Principles

A static greedy lookup over a trajectory vault (`NativeReader::step`) fails whenever a prompt prefix steps slightly off a recorded trajectory (`NativeStop::NoGeodesic`), enters a self-loop (`NativeStop::FixedPoint`), or reaches a token carrier where multiple candidate trajectories share nearly colinear $SO(3)$ Wigner frames.

Under **Active Inference on $(\mathbb{H}^3 \times \mathbb{T}^2)/\Gamma_{12}$**, the server treats every decoding step as a variational free-energy minimization problem subject to hard structural compliance, safety/security, and token-efficiency constraints. Whenever an anomaly, conflict, or dead end is encountered, the server **must not** greedily emit a degraded token or immediately abort. Instead, it **stops external token emission, holds a coherent superposition of candidate geodesic branches (`SuperpositionState`), reasons across counterfactual rollouts and $2$-cell Gauss–Bonnet holonomy loops, and dynamically iterates to a new minimum-free-energy solution**.

## 2. Variational Free Energy Functional ($\mathcal{F}$ in `Q32.31`)

Given a token/ML-IR prefix $P = (x_1, \dots, x_t)$ arriving at hyperboloid coordinate $q \in \mathbb{H}^3$ with parallel-transported Levi-Civita Wigner gyroscope frame $W_{\text{prefix}} \in SO(3)$, every candidate transition $e = (u \to v)$ emitting operator/token `next_id` is scored by the unified Variational Free Energy functional:

$$\mathcal{F}(e \mid P) = \underbrace{\Delta \mathcal{S}_{\text{geo}}(q, u, v)}_{\text{Geodesic Surprise}} + \underbrace{\Delta \mathcal{H}_{\text{Wigner}}(W_e, W_{\text{prefix}})}_{\text{Normalized Holonomy Divergence}} + \underbrace{\Delta \mathcal{E}_{\text{epistemic}}(e)}_{\text{Teacher Inaccuracy}} + \underbrace{\Phi_{\text{compliance/safety}}(u \to v)}_{\text{Safety, Security & Statutory Barrier}} + \underbrace{\mathcal{C}_{\text{token}}(v \mid P)}_{\text{Token Spend & Redundancy Cost}}$$

### 2.1 Normalized Wigner Frame Alignment & Exact Residual Distance
Unnormalized Frobenius inner products $\langle W_e, W_{\text{prefix}} \rangle_F$ suffer from norm drift across multi-hop Lorentz boosts (`transport_frame_hop`). To guarantee zero false frame collisions:

1. **Normalized Cosine Alignment (`cos_align_q32`)**:
   $$\text{align}_{\text{norm}}(W_e, W_{\text{prefix}}) = \frac{\langle W_e, W_{\text{prefix}} \rangle_F}{\|W_e\|_F \, \|W_{\text{prefix}}\|_F} \in [-1, 1]$$
2. **Exact Frobenius Residual (`frobenius_dist_sq_q32`)**:
   $$d_F^2(W_e, W_{\text{prefix}}) = \sum_{r=0}^{2} \sum_{c=0}^{2} \left(W_{e,rc} - W_{\text{prefix},rc}\right)^2 \ge 0$$
   which achieves $d_F^2 = 0$ if and only if $W_e = W_{\text{prefix}}$ bit-for-bit, paired with `chain_id == prefix_chain_id(prefix)` causal chain matching.
3. **Receptor Sketch Tie-Breaking (`receptor_cosine_q32`)**:
   When two candidate carrier atoms carry hidden-state payloads (`ATOMA_OP` or raw `f32` teacher state), their 256-dimensional receptor sketch cosine (`compute_receptor_sketch`) disambiguates colliding frames.

### 2.2 Compliance, Safety & Security Potential ($\Phi_{\text{compliance/safety}}$)
Every candidate transition is evaluated before emission against Layer 3 first-principle invariants:

1. **Optical Blast Radius (`OpticalRiskEngine`)**:
   $$B(v) = \frac{1}{v_0 + 1} = \frac{1 - r^2}{2} \in (0, 0.5]$$
   Transitions that push state toward the `L0` Coordinator origin above the autonomous ceiling ($B(v) > 0.45$, EU AI Act Art. 14 / GDPR Art. 22) incur a steep barrier penalty or trigger `EscalateToHuman`.
2. **Silo Horizon & Pairwise Overlap Gate (`OverlapPolicyMatrix`)**:
   Any transition $(r_{\text{from}} \to r_{\text{to}})$ where `!policy.can_communicate(r_from, r_to)` is structurally vetoed ($\Phi = +\infty$, `compliance_permitted = false`).
3. **Statutory & Security Ontology Gate (`FirstPrinciplesComplianceEngine` & `SwarmKnowledgeAndHomeostasis`)**:
   Enforces GDPR Art. 5/17/25/32, EU AI Act Art. 9/13/14/15, SOC2 CC6.1/CC7.2, NIST CSF 2.0, ISO/IEC 42001, and CVE/CWE/ATT&CK invariants (CVE-2012-2459, CWE-78, CWE-79, AML.T0051).

### 2.3 Token Consumption Minimization with Absolute Quality Floor ($\mathcal{C}_{\text{token}}$)
Token consumption is minimized without sacrificing output quality:

1. **Atomic IR Verify-Cliff & Span Compression (`AtomicIrAlphabet`)**:
   Candidate transitions that emit pinned structural/span opcodes (`MlirOpId >= base_vocab_size`) or reach a high-certainty geodesic highway within the empirical `<= 7`-token verify cliff (`ATOMIC_IR_MAX_VERIFY_TOKENS = 7`) receive a negative token-cost bonus (lower $\mathcal{F}$).
2. **Spend Envelope Gate (`predict_token_envelope`)**:
   Tracks cumulative tokens against the `p75 × 1.5` envelope.
3. **Absolute Quality Floor**:
   No token-saving shortcut may be selected if its teacher certainty (`teacher_certainty_q32`) or Meta-Cognitive Decision Readiness
   $$\Omega = \frac{(1 - U)(1 - D)}{B(v) + \epsilon}$$
   falls below the quality floor ($\Omega_{\min} = 0.82$ in autonomous mode, or $\mathcal{F} > \tau_{\mathcal{F}}$).

## 3. The Superposition Holding State Machine (`hold_and_iterate`)

### Phase 1: Stop on Issue Detection (`SuperpositionTrigger`)
During active inference, greedy emission immediately **stops** if any of the following triggers fire:
- `NoDirectGeodesic`: No outgoing `1-cell` arc exists in the tight primary basin ($\cosh d > 1.25$).
- `FixedPointOrCycle`: The greedy winner repeats the current token (`next_id == last`) or forms a short $n$-gram cycle in the active prefix.
- `WignerFrameAmbiguity`: The top-2 candidate arcs have Variational Free Energy within `ambiguity_margin_q32` ($\Delta \mathcal{F} < 0.04$) or inter-role disagreement $D \ge 0.35$.
- `HighFreeEnergyOrLowReadiness`: The greedy winner's free energy exceeds $\tau_{\mathcal{F}}$ or Meta-Cognitive Readiness $\Omega < 0.82$.
- `ComplianceOrSafetyBoundary`: The greedy winner violates `OverlapPolicyMatrix` or exceeds `max_autonomous_blast`.

### Phase 2: Hold Coherent Superposition ($\Psi_t$)
Instead of collapsing or failing out, `SuperpositionHoldingEngine` constructs a coherent `SuperpositionState` holding up to $M \le 12$ competing candidate branches:

$$\Psi_t = \left\{ \left(\gamma_k, W_k, R_k, \mathcal{F}_k, w_k = \frac{e^{-\beta \mathcal{F}_k}}{\sum_j e^{-\beta \mathcal{F}_j}}\right) \right\}_{k=1}^{M}$$

Candidate branches are harvested from four geometric sources on $(\mathbb{H}^3 \times \mathbb{T}^2)/\Gamma_{12}$:
1. **`DirectTrajectoryArc`**: All non-cycling outgoing `1-cell` arcs in the local basin.
2. **`PhiWindingNeighborArc`**: Outgoing `1-cell` arcs from neighbor atoms discovered by $\phi$-winding phase scan ($\Delta \theta_f = k \cdot \frac{2\pi}{\phi^2}$) and broadened hyperbolic basin ($\cosh d \le 1.85$).
3. **`CoupledOperatorBridge`**: `ATOMA_OP` v2 coupled operator atoms in the basin whose `peek_next_mlir_id` and 256d receptor sketch cosine provide a weightless operator bridge.
4. **`CrossSiloGeodesicBridge`**: Nearest high-salience `0-cell` / `1-cell` anchors in permitted icosahedral neighbor silos (`OverlapPolicyMatrix`) that bridge across distilled teacher minds (`glimmer` $\leftrightarrow$ `qwen`).

### Phase 3: Dynamic Counterfactual Iteration & $2$-Cell Gauss–Bonnet Reasoning
While holding $\Psi_t$ (with zero external tokens emitted), the engine executes up to $K$ dynamic reasoning iterations:

1. **Counterfactual Lookahead Rollout (Horizon $H \le 4$)**:
   Each branch $\gamma_k$ is virtually rolled forward up to $H$ steps on the map (`[prefix..., v_k, v_{k,2}, ...]`). Branches that unlock a low-free-energy multi-step geodesic highway reduce their effective horizon free energy $\overline{\mathcal{F}}_k$; branches that hit an immediate dead end or cycle are penalized.
2. **$2$-Cell Gauss–Bonnet Holonomy Verification (`HyperbolicHolonomyEngine`)**:
   For each branch, the engine computes the hyperbolic triangle defect $\Omega(\triangle(q_{\text{prev}}, u_k, v_k)) = \pi - (\alpha + \beta + \gamma)$ and evaluates the 3-role dialectical loop.
   - Ping-pong echo chambers ($\Omega_{\triangle} = 0$) are rejected (`HolonomyVerdict::EchoChamberRejected`).
   - Positive hyperbolic curvature flux ($\Omega_{\triangle} \ge \Omega_{\min}$) injects a Berry-phase free-energy reduction (`HolonomyVerdict::DialecticalSynthesisVerified`).
3. **Iterative Pruning & Re-Weighting**:
   At the end of each iteration $i \in \{1, \dots, K\}$, branches with $\mathcal{F}_k > \mathcal{F}_{\min} + \Delta_{\text{prune}}$ are pruned, Boltzmann weights $w_k$ are recomputed, and if the top branch hasn't yet crossed the readiness threshold $\Omega \ge 0.82$, the search radius expands along the golden-angle spiral for iteration $i + 1$.

### Phase 4: Collapse & Bridge Self-Distillation
- **Collapse to Solution**: Once a branch $\gamma^*$ achieves minimum Variational Free Energy $\mathcal{F}^*$ with `compliance_permitted == true` and positive progress, the superposition collapses onto $\gamma^*$.
- **Self-Distillation**: When invoked with mutable map access (`active_inference_step_mut`), the newly synthesized bridge transition $(q \to v^*)$ is recorded into `CarvedSwarmMap` as a distilled `1-cell` reasoning arc so subsequent passes resolve in $O(1)$ without re-entering superposition.
- **Fail-Closed Safety**: If no compliant, cycle-free branch exists after $K$ iterations, the engine halts cleanly with `NativeStop::SuperpositionFailClosed` and emits a cryptographic audit receipt of all held branches.
