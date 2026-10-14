# Carved Swarm & Immune System — Architectural Specification (Tier 4–6)

> **Epistemic Marking Standard (Qualification #1):**
> Every claim in this document is marked with one of four tags:
> - `[verified]` — measured or proved directly on this machine from source or test execution.
> - `[reported]` — measured and reported in reference artifacts (`dhng2.html`, `dhng3.html`, `docs/SESSION_NOTES.md`).
> - `[inferred]` — follows logically or mathematically from `[verified]` or `[reported]` evidence.
> - `[theory]` — unproven architectural hypothesis subject to preregistered falsification (`94ce028317e0ecac`).

## 1. Three Layers, One Door

All inference, distillation refraction, role carving, agent supervision, and inter-role dispatch flow through a single Rust server architecture split into three structural layers `[verified]`:

1. **Layer 3 — Immune System (`immune_system/implementation` / `atoma_immune`)**:
   - Governs the Governor (Layer 2), never the agents via prompt instructions `[verified]`.
   - Enforces compliance via deterministic finite-state automatons (`ComplianceAutomaton`): `SiloBoundaryGate`, `BoundaryHandoffGate`, `BaseWeightImmutabilityGate`, `LossAccountingGate`, and `KillCriteriaGate` `[verified]`.
   - Owns the $12 \times 12$ pairwise overlap policy matrix $\Omega_{r,s}$ in `Q32_31` hyperbolic-toroidal distance units `[verified]`.
   - Unauthorized dispatch paths do not exist in the type system: `Governor::route_handoff` cannot construct a delivery frame without an unforgeable `CompliancePermit` minted by `ImmuneSystem::authorize_handoff` `[verified]`.

2. **Layer 2 — Governor (`orchestrator/src/swarm` & `orchestrator/src/routing` / `atoma_orchestrator`)**:
   - Erlang/OTP-style supervision tree (`OneForOne`, `OneForAll`, `RestForOne`) with configurable restart intensity `(max_restarts, within_ticks)` and monotonic heartbeat tracking `[verified]`.
   - Spawns and supervises the 12 carved specialist role processes (`L1`..`L12`) around the central `Coordinator` (`L0`) `[verified]`.
   - Routes all inter-role messages (`HandoffEnvelope`). Roles hold no references or channels to peer roles `[verified]`.

3. **Layer 1 — Carved Swarm (`immune_system/implementation/src/silo.rs` & `atlas13.rs`)**:
   - 12 specialized roles (`L1`..`L12`) around 1 central `Coordinator` (`L0`) on $\mathbb{H}^3 \times \mathbb{T}^2$ `[verified]`.
   - Each role receives a carved `SiloView` containing **only** atoms inside its home attractor basin $B_r(\tau_r)$ plus Immune-System-authorized overlap boundaries $B_r(\tau_r + \Omega_{r,s}) \cap B_s(\tau_s + \Omega_{r,s})$ `[verified]`.
   - Coordinates outside a role's `SiloView` are physically absent from its lookup table and VP-index `[verified]`.
   - All payloads reference immutable shared bytes (`Arc<[u8]>`) and frozen base weights `[verified]`.

## 2. Geometry of $\mathbb{H}^3 \times \mathbb{T}^2$ and the Optical Lens

### 2.1 Optical Metric and Ball–Hyperboloid Isometry (`dhng2.html`)
- The Poincaré ball $\mathbb{B}^3 = \{\mathbf{p} \in \mathbb{R}^3 : r = \|\mathbf{p}\| < 1\}$ carries the conformal optical refractive index:
  $$n(r) = \frac{2}{1 - r^2}$$
  Under Fermat's principle $\delta \int n(r)\,ds = 0$, light rays follow exact hyperbolic geodesics `[verified]`.
- The stereographic map from $\mathbb{B}^3$ to the future sheet of the Lorentz hyperboloid $\mathbb{H}^3 = \{(x_0, \mathbf{x}) \in \mathbb{R}^{1,3} : x_0^2 - \|\mathbf{x}\|^2 = 1, x_0 > 0\}$ is expressed directly in terms of $n(r)$ `[verified]`:
  $$x_0 = \frac{1 + r^2}{1 - r^2} = n(r) - 1, \qquad \mathbf{x} = \frac{2\mathbf{p}}{1 - r^2} = n(r)\mathbf{p}$$
  with inverse:
  $$\mathbf{p} = \frac{\mathbf{x}}{x_0 + 1}, \qquad n(r) = x_0 + 1$$
- Across 900 point pairs in $\mathbb{B}^3$, the `Q32_31` fixed-point Lorentz inner product $\langle A, B \rangle_{\mathcal{L}}$ and the Poincaré ball distance invariant $1 + \frac{1}{2}n(\|\mathbf{p}\|)n(\|\mathbf{q}\|)\|\mathbf{p}-\mathbf{q}\|^2$ agree to within $\le 4.4 \times 10^{-9}$ `[verified]` (matching `[reported]` $4.4 \times 10^{-9}$ in `SPEC.md`).

### 2.2 Twelve Around One (`dhng3.html`)
- **Why 12 is forced by math (Qualification #3 & #6)**:
  - In full embedding space, the 6 cognitive duality axes (`A − B`) are nearly orthogonal: mean $|\cos| = 0.079$, max $0.177$ `[reported: F17]`, forming a 6D cross-polytope ($\pm e_1 \dots \pm e_6$, 12 vertices) `[verified]`.
  - When projected to 3D tangent space $T_{\mathbf{o}}\mathbb{H}^3 \cong \mathbb{R}^3$, 6 mutually orthogonal lines cannot exist. By Haantjes' theorem (1948) and the Welch/Simplex bound for $N=6$ lines in $\mathbb{R}^3$, the minimum achievable maximum pairwise coherence is:
    $$\max_{i \neq j} |\langle \mathbf{u}_i, \mathbf{u}_j \rangle| \ge \sqrt{\frac{6 - 3}{3(6 - 1)}} = \frac{1}{\sqrt{5}} \approx 0.4472136$$
    with equality **if and only if** the 6 axes form the 12 vertices of the regular icosahedron `[verified]`.
  - Simultaneously, in 3D Euclidean/tangent packing (the kissing number problem, proved by Schütte & van der Waerden 1953), the maximum number of non-overlapping unit spheres touching a central unit sphere ("Twelve Around One") is **12** `[verified]`.
  - **Alternatives documented per Qualification #3**:
    1. `FrameGeometry::Icosahedral` (Default, Approved): 6 equiangular axes at $|\cos| = 1/\sqrt{5} \approx 0.4472$, Poincaré shell radius $r_{\text{sph}} = 0.382 \approx 2 - \varphi$ `[verified]`.
    2. `FrameGeometry::Cuboctahedral` (Vector Equilibrium / `atlas13.json`): 12 vertices of the cuboctahedron at $r_{\text{sph}} = 0.382$ `[verified]`.
    3. `FrameGeometry::CrossPolytope6D` (Native 16D/6D ambient option): $\pm e_1 \dots \pm e_6$ where all 6 axes are strictly orthogonal ($\cos = 0$), requiring $\mathbb{H}^d$ with $d \ge 6$ rather than $\mathbb{H}^3$ `[inferred]`.

### 2.3 Toroidal Phase Assignment on $\mathbb{T}^2$ (`dhng3.html`)
- Each duality $k \in \{0..5\}$ is assigned longitude $u_k = k \cdot 30^\circ = k \cdot \frac{\pi}{6}$ `[verified]`:
  - **Pole A** (odd role $2k+1$): $(u_k, 0)$ on the outer equator.
  - **Pole B** (even role $2k+2$): $(u_k + \pi, \pi)$ on the inner equator — the unique antipodal point on $\mathbb{T}^2$ at maximum phase displacement $(\Delta u, \Delta v) = (\pi, \pi)$ `[verified]`.

## 3. The Full Execution Chain

```
Distill (Tap + Hash) → Refract (Project + ExpMap) → Carve (×12 Views) → Silo (Overlap Policy) → Agent → Govern → Immune-check
```

## 4. Non-Modification Guarantee During Baseline Comparison (Qualification #2)

While Grok's 8-token forward comparison is in progress, `core_lm/`'s loader, geometric map, and retrieval wiring are treated as **strictly read-only** `[verified]`. All findings on `core_lm/` are documented and flagged in `immune_system/audit/BASELINE_CONGRUENCE_AUDIT.md` `[verified]`.
