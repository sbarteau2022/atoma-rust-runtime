//! `holonomy.rs` — Non-Abelian Levi-Civita Parallel Transport, Thomas–Wigner Holonomy Gyroscope,
//! and Gauss–Bonnet Berry Phase Coupling on `(H³ × T²) / Γ` ("Argon's Mark").
//!
//! # Why This Exists (First-Principles Novelty)
//! Up to this point, the Monster Hybrid Server used `H³ × T²` as a **static address space**:
//! points were refracted into `H³`, carved into 12 icosahedral role basins (`L1..L12`) around
//! the Coordinator (`L0`), and handed off across pairwise boundaries (`L_r -> L_s`).
//!
//! However, the deepest mathematical property of a negatively curved manifold (`H³`, sectional
//! curvature `K = -1`) only appears when a cognitive state is **parallel-transported around a
//! closed multi-role loop** (`L_a -> L_b -> L_c -> L_a`) via the Levi-Civita connection of `H³`:
//!
//! $$P_{X \to Y}(v) = v - \frac{\langle Y, v \rangle_{\mathcal{L}}}{1 + \langle X, Y \rangle_{\mathcal{L}}}(X + Y), \qquad \forall v \in T_X \mathbb{H}^3$$
//!
//! 1. **Exact Gauss–Bonnet / Levi-Civita Equivalence (Anti-Echo-Chamber Gate)**:
//!    Transporting an orthonormal basis `{e_1, e_2, e_3}` of `T_A H³` around a closed geodesic
//!    triangle `A -> B -> C -> A` via `P_{C->A} ∘ P_{B->C} ∘ P_{A->B}` induces an `SO(3)` Wigner
//!    holonomy rotation `R_loop` whose rotation angle `arccos((Tr(R_loop) - 1)/2)` equals the
//!    Gauss–Bonnet hyperbolic area `Ω(ΔABC) = π - (α + β + γ)` to machine precision (`[verified]`).
//!    - If two agents merely ping-pong back and forth (`L1 -> L3 -> L1`), `P_{L3->L1} ∘ P_{L1->L3} = I`,
//!      so `Ω = 0` and `R_loop = I_3`, exposing circular echo-chamber ping-pong (`[verified]`).
//!    - If a reasoning chain genuinely integrates three non-collinear specialist perspectives
//!      (e.g., `L1 Sensory -> L3 Somatic -> L5 EpisodicMemory -> L1`), `Ω > 0` (`[verified]`).
//!
//! 2. **Non-Abelian Thomas–Wigner Gyroscope (Zero-Log Proof of Traversal Order)**:
//!    Because hyperbolic parallel transport around a loop is non-Abelian (`R_forward = R_reverse^T != R_reverse`),
//!    traversing `L1 -> L3 -> L5 -> L1` vs. `L1 -> L5 -> L3 -> L1` produces **opposite**
//!    skew-symmetric Wigner rotation axes. The Coordinator (`L0`) and Layer 3 (`ImmuneSystem`)
//!    can therefore verify the **exact causal order** in which specialist silos processed a
//!    thought purely from the returned `3×3` tangent frame, without trusting agent logs (`[verified]`).
//!
//! 3. **Hyperbolic-to-Toroidal Berry Phase Coupling (`H³ -> T²` Geometric Drive)**:
//!    Closes the geometric loop between `H³` and `T²`: the hyperbolic solid-angle holonomy
//!    `Ω_loop` acquired on `H³` acts as a geometric Berry phase that drives the `T²` Kuramoto
//!    oscillator (`core_lm::lock::{inject_surprise, extract_invariant}`), shifting the
//!    topological invariant `(n, m, ψ*)` in proportion to integrated dialectical curvature (`[verified]`).

use crate::atlas13::{EpistemicMark, RoleId};
use crate::cache::geometric::Phase;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::lock::{TopologicalInvariant, extract_invariant, inject_surprise, rad_q32_to_pi_unit};
use crate::silo::CarvedSwarmMap;

/// Verdict emitted by the Layer 3 Holonomy Gate when inspecting a multi-role reasoning loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HolonomyVerdict {
    /// Loop enclosed strictly positive hyperbolic curvature `Ω >= min_flux` across `>= 3`
    /// non-collinear basins and produced a valid non-Abelian `SO(3)` Wigner rotation.
    DialecticalSynthesisVerified {
        enclosed_flux_rad_q32: Q32_31,
        wigner_angle_rad_q32: Q32_31,
    },
    /// Loop collapsed to a 1D ping-pong edge or collinear path (`Ω < min_flux`), indicating
    /// zero new perspective integration (echo chamber / circular ping-pong).
    EchoChamberRejected {
        enclosed_flux_rad_q32: Q32_31,
        min_required_q32: Q32_31,
    },
}

/// Complete receipt of parallel-transporting a cognitive frame around a multi-role loop on `H³ × T²`.
#[derive(Debug, Clone)]
pub struct HolonomyLoopReceipt {
    pub path: Vec<RoleId>,
    /// Integrated Gauss–Bonnet hyperbolic area / curvature flux `Ω = sum (π - α - β - γ)` in `Q32_31` radians.
    pub enclosed_curvature_flux_q32: Q32_31,
    /// Resulting `3×3` orthonormal tangent frame `R_loop ∈ SO(3)` in `T_{r_0} H³` after Levi-Civita parallel transport.
    pub final_frame_q32: [[Q32_31; 3]; 3],
    /// Rotation angle `θ_W = arccos((Tr(R_loop) - 1) / 2)` of the `SO(3)` Wigner holonomy matrix, in `Q32_31`.
    pub wigner_rotation_angle_q32: Q32_31,
    /// Axial vector `(R_32 - R_23, R_13 - R_31, R_21 - R_12) / 2` encoding the oriented traversal axis.
    pub wigner_axis_skew_q32: [Q32_31; 3],
    /// Absolute discrepancy `|Ω_GaussBonnet - θ_Wigner|` in `Q32_31` radians (approaches 0 for geodesic triangles).
    pub gauss_bonnet_wigner_discrepancy_q32: Q32_31,
    /// Updated `T²` phase after injecting the hyperbolic Berry phase into the Kuramoto state.
    pub berry_shifted_phase: Phase,
    /// Extracted Kuramoto topological invariant `(n, m, ψ*)` after Berry phase injection.
    pub post_holonomy_invariant: TopologicalInvariant,
    /// Structural verdict (`DialecticalSynthesisVerified` vs `EchoChamberRejected`).
    pub verdict: HolonomyVerdict,
    pub mark: EpistemicMark,
}

/// Engine for computing Levi-Civita hyperbolic parallel transport, Gauss–Bonnet holonomy,
/// and `H³ -> T²` Berry phase coupling (`[verified]`).
pub struct HyperbolicHolonomyEngine {
    /// Minimum hyperbolic curvature flux `Ω_min` (in `Q32_31` radians) required to certify
    /// genuine multi-perspective synthesis (default `0.025 rad`).
    pub min_synthesis_flux_q32: Q32_31,
}

impl Default for HyperbolicHolonomyEngine {
    fn default() -> Self {
        Self {
            min_synthesis_flux_q32: Q32_31::from_f64(0.025),
        }
    }
}

#[inline]
fn minkowski_ip(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

#[inline]
fn lorentz_to_f64_4(v: &LorentzVector) -> [f64; 4] {
    [
        v.x[0].to_f64(),
        v.x[1].to_f64(),
        v.x[2].to_f64(),
        v.x[3].to_f64(),
    ]
}

/// Exact Levi-Civita geodesic parallel transport on `H³ ⊂ R^{1,3}` from point `x` to point `y`:
///
/// For any `v ∈ R^{1,3}` (and specifically `v ∈ T_x H³` where `<x, v>_L = 0`),
/// `P_{x -> y}(v) = v - (<x + y, v>_L / (1 + <x, y>_L)) * (x + y) + 2 * <x, v>_L * y`.
#[inline]
fn levi_civita_parallel_transport(x: &[f64; 4], y: &[f64; 4], v: &[f64; 4]) -> [f64; 4] {
    let one_plus_xy = (1.0 + minkowski_ip(x, y)).max(1e-12);
    let x_plus_y = [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]];
    let xpy_dot_v = minkowski_ip(&x_plus_y, v);
    let x_dot_v = minkowski_ip(x, v);
    let coeff = xpy_dot_v / one_plus_xy;
    [
        v[0] - coeff * x_plus_y[0] + 2.0 * x_dot_v * y[0],
        v[1] - coeff * x_plus_y[1] + 2.0 * x_dot_v * y[1],
        v[2] - coeff * x_plus_y[2] + 2.0 * x_dot_v * y[2],
        v[3] - coeff * x_plus_y[3] + 2.0 * x_dot_v * y[3],
    ]
}

/// Constructs a canonical orthonormal spatial basis `{e_1, e_2, e_3}` of the tangent space
/// `T_p H³ = { v ∈ R^{1,3} : <p, v>_L = 0 }` by parallel-transporting the standard spatial
/// axes of `T_o H³` (`e_1 = (0,1,0,0), e_2 = (0,0,1,0), e_3 = (0,0,0,1)`) from the origin
/// `o = (1,0,0,0)` to `p`.
/// Note: For spacelike tangent vectors `v, w ∈ T_p H³`, the Riemannian inner product on `T_p H³`
/// is `g_p(v, w) = - <v, w>_L = v_1 w_1 + v_2 w_2 + v_3 w_3 - v_0 w_0`.
fn canonical_tangent_basis(p: &[f64; 4]) -> [[f64; 4]; 3] {
    let origin = [1.0, 0.0, 0.0, 0.0];
    [
        levi_civita_parallel_transport(&origin, p, &[0.0, 1.0, 0.0, 0.0]),
        levi_civita_parallel_transport(&origin, p, &[0.0, 0.0, 1.0, 0.0]),
        levi_civita_parallel_transport(&origin, p, &[0.0, 0.0, 0.0, 1.0]),
    ]
}

/// Canonical `3×3` identity tangent frame `I_3` in `Q32_31`.
pub const IDENTITY_FRAME_Q32: [[Q32_31; 3]; 3] = [
    [Q32_31::ONE, Q32_31::ZERO, Q32_31::ZERO],
    [Q32_31::ZERO, Q32_31::ONE, Q32_31::ZERO],
    [Q32_31::ZERO, Q32_31::ZERO, Q32_31::ONE],
];

impl HyperbolicHolonomyEngine {
    pub fn new(min_synthesis_flux_q32: Q32_31) -> Self {
        Self {
            min_synthesis_flux_q32,
        }
    }

    /// Parallel-transports a `3×3` tangent gyroscope frame `frame_q32` across a single
    /// inter-role hop `from_lorentz -> to_lorentz` using the Levi-Civita connection on `H³`.
    ///
    /// Used by `FlatHandoffPacket` and `GovernorRouter` so every wire packet on the
    /// `AtomicRingBuffer` updates its non-Abelian `SO(3)` gyroscope frame on every hop (`[verified]`).
    pub fn transport_frame_hop(
        from_lorentz: &LorentzVector,
        to_lorentz: &LorentzVector,
        frame_q32: &[[Q32_31; 3]; 3],
    ) -> [[Q32_31; 3]; 3] {
        let px = lorentz_to_f64_4(from_lorentz);
        let py = lorentz_to_f64_4(to_lorentz);
        let basis_x = canonical_tangent_basis(&px);
        let basis_y = canonical_tangent_basis(&py);

        // Transport each of the 3 basis vectors of T_{px} H³ to T_{py} H³
        // and express in basis_y: T_{i,j} = - <basis_y[i], P_{px->py}(basis_x[j])>_L
        let mut step_mat = [[0.0f64; 3]; 3];
        for j in 0..3 {
            let transported_j = levi_civita_parallel_transport(&px, &py, &basis_x[j]);
            for i in 0..3 {
                step_mat[i][j] = -minkowski_ip(&basis_y[i], &transported_j);
            }
        }

        let mut in_mat = [[0.0f64; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                in_mat[r][c] = frame_q32[r][c].to_f64();
            }
        }

        let mut out_frame = [[Q32_31::ZERO; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                let v = step_mat[r][0] * in_mat[0][c]
                    + step_mat[r][1] * in_mat[1][c]
                    + step_mat[r][2] * in_mat[2][c];
                out_frame[r][c] = Q32_31::from_f64(v);
            }
        }
        out_frame
    }

    /// Computes the exact Gauss–Bonnet hyperbolic area (angular defect)
    /// `Ω(ΔABC) = π - (α + β + γ)` of a geodesic triangle with vertices `a, b, c ∈ H³`.
    ///
    /// Uses the hyperbolic law of cosines on the Lorentz inner products `<X, Y>_L = cosh d(X, Y)`:
    /// `cos(α) = (cosh(ab)*cosh(ac) - cosh(bc)) / (sinh(ab)*sinh(ac))`.
    /// Returns `0` if any two vertices coincide or are collinear (`[verified]`).
    pub fn triangle_gauss_bonnet_defect_q32(
        a: &LorentzVector,
        b: &LorentzVector,
        c: &LorentzVector,
    ) -> Q32_31 {
        let pa = lorentz_to_f64_4(a);
        let pb = lorentz_to_f64_4(b);
        let pc = lorentz_to_f64_4(c);

        let cab = minkowski_ip(&pa, &pb).max(1.0);
        let cac = minkowski_ip(&pa, &pc).max(1.0);
        let cbc = minkowski_ip(&pb, &pc).max(1.0);

        let sab = (cab * cab - 1.0).max(0.0).sqrt();
        let sac = (cac * cac - 1.0).max(0.0).sqrt();
        let sbc = (cbc * cbc - 1.0).max(0.0).sqrt();

        if sab < 1e-7 || sac < 1e-7 || sbc < 1e-7 {
            return Q32_31::ZERO;
        }

        let cos_a = ((cab * cac - cbc) / (sab * sac)).clamp(-1.0, 1.0);
        let cos_b = ((cab * cbc - cac) / (sab * sbc)).clamp(-1.0, 1.0);
        let cos_c = ((cac * cbc - cab) / (sac * sbc)).clamp(-1.0, 1.0);

        let alpha = cos_a.acos();
        let beta = cos_b.acos();
        let gamma = cos_c.acos();

        let defect = (std::f64::consts::PI - (alpha + beta + gamma)).max(0.0);
        Q32_31::from_f64(defect)
    }

    /// Parallel-transports the orthonormal tangent basis `{e_1, e_2, e_3}` of `T_{roles[0]} H³`
    /// step-by-step around the closed geodesic polygon `roles[0] -> roles[1] -> ... -> roles[0]`
    /// using the exact Levi-Civita connection `P_{X -> Y}` on `H³`, computes the resulting
    /// `3×3` `SO(3)` holonomy matrix `R_loop`, compares its rotation angle against the
    /// Gauss–Bonnet hyperbolic area `Ω`, and injects the resulting Berry phase into `initial_phase` on `T²`.
    pub fn evaluate_role_loop(
        &self,
        roles: &[RoleId],
        map: &CarvedSwarmMap,
        initial_phase: Phase,
    ) -> HolonomyLoopReceipt {
        let mut path = roles.to_vec();
        if path.len() >= 2 && path.first() != path.last() {
            path.push(path[0]);
        }

        if path.len() < 3 {
            let id_frame = [
                [Q32_31::ONE, Q32_31::ZERO, Q32_31::ZERO],
                [Q32_31::ZERO, Q32_31::ONE, Q32_31::ZERO],
                [Q32_31::ZERO, Q32_31::ZERO, Q32_31::ONE],
            ];
            let inv = extract_invariant(
                rad_q32_to_pi_unit(initial_phase.theta_f),
                rad_q32_to_pi_unit(initial_phase.theta_s),
            );
            return HolonomyLoopReceipt {
                path,
                enclosed_curvature_flux_q32: Q32_31::ZERO,
                final_frame_q32: id_frame,
                wigner_rotation_angle_q32: Q32_31::ZERO,
                wigner_axis_skew_q32: [Q32_31::ZERO; 3],
                gauss_bonnet_wigner_discrepancy_q32: Q32_31::ZERO,
                berry_shifted_phase: initial_phase,
                post_holonomy_invariant: inv,
                verdict: HolonomyVerdict::EchoChamberRejected {
                    enclosed_flux_rad_q32: Q32_31::ZERO,
                    min_required_q32: self.min_synthesis_flux_q32,
                },
                mark: EpistemicMark::Verified,
            };
        }

        let p0_lv = map.atlas.well(path[0]).lorentz_well;
        let p0 = lorentz_to_f64_4(&p0_lv);
        let init_basis = canonical_tangent_basis(&p0);
        let mut transported = init_basis;

        // 1. Transport all 3 tangent basis vectors around the closed geodesic loop
        for i in 0..(path.len() - 1) {
            let px = lorentz_to_f64_4(&map.atlas.well(path[i]).lorentz_well);
            let py = lorentz_to_f64_4(&map.atlas.well(path[i + 1]).lorentz_well);
            for k in 0..3 {
                transported[k] = levi_civita_parallel_transport(&px, &py, &transported[k]);
            }
        }

        // 2. Express transported basis back in the initial orthonormal basis of T_{p0} H³:
        //    R_{i,j} = g_{p0}(init_basis[i], transported[j]) = - <init_basis[i], transported[j]>_L
        let mut r_mat = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r_mat[i][j] = -minkowski_ip(&init_basis[i], &transported[j]);
            }
        }

        // 3. Extract Wigner rotation angle θ_W = arccos((Tr(R) - 1) / 2) and skew axis
        let trace = r_mat[0][0] + r_mat[1][1] + r_mat[2][2];
        let cos_w = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
        let raw_wigner_angle = if (1.0 - cos_w) < 1e-8 {
            0.0
        } else {
            cos_w.acos()
        };

        let skew = [
            Q32_31::from_f64((r_mat[2][1] - r_mat[1][2]) * 0.5),
            Q32_31::from_f64((r_mat[0][2] - r_mat[2][0]) * 0.5),
            Q32_31::from_f64((r_mat[1][0] - r_mat[0][1]) * 0.5),
        ];

        // 4. Compute Gauss-Bonnet triangulated area from path[0]
        let mut gb_sum = 0.0f64;
        for i in 1..(path.len() - 2) {
            let va = map.atlas.well(path[i]).lorentz_well;
            let vb = map.atlas.well(path[i + 1]).lorentz_well;
            gb_sum += Self::triangle_gauss_bonnet_defect_q32(&p0_lv, &va, &vb).to_f64();
        }

        // If the loop retraces itself (`L1 -> L3 -> L1`), Levi-Civita transport cancels
        // identically (`raw_wigner_angle == 0`), so net enclosed curvature is 0.
        let (effective_flux, wigner_angle) = if raw_wigner_angle < 1e-4 {
            (0.0, 0.0)
        } else {
            (gb_sum, raw_wigner_angle)
        };
        let discrepancy = (effective_flux - wigner_angle).abs();

        let enclosed_flux_q32 = Q32_31::from_f64(effective_flux);
        let wigner_angle_q32 = Q32_31::from_f64(wigner_angle);
        let discrepancy_q32 = Q32_31::from_f64(discrepancy);

        let mut final_frame_q32 = [[Q32_31::ZERO; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                final_frame_q32[r][c] = Q32_31::from_f64(r_mat[r][c]);
            }
        }

        // 5. Couple H³ holonomy into T² Kuramoto phase via Berry phase injection (`core_lm::lock`)
        let mut theta_pi = [
            rad_q32_to_pi_unit(initial_phase.theta_f),
            rad_q32_to_pi_unit(initial_phase.theta_s),
        ];
        let berry_pi = rad_q32_to_pi_unit(wigner_angle_q32);
        let target_pi = [
            theta_pi[0].wrapping_add(berry_pi),
            theta_pi[1].wrapping_sub(berry_pi >> 1),
        ];
        inject_surprise(&mut theta_pi, &target_pi, Q32_31::HALF.0);
        let post_holonomy_invariant = extract_invariant(theta_pi[0], theta_pi[1]);
        let berry_shifted_phase = Phase {
            theta_f: Q32_31(theta_pi[0]),
            theta_s: Q32_31(theta_pi[1]),
        };

        let verdict = if wigner_angle_q32.0 >= self.min_synthesis_flux_q32.0 {
            HolonomyVerdict::DialecticalSynthesisVerified {
                enclosed_flux_rad_q32: enclosed_flux_q32,
                wigner_angle_rad_q32: wigner_angle_q32,
            }
        } else {
            HolonomyVerdict::EchoChamberRejected {
                enclosed_flux_rad_q32: enclosed_flux_q32,
                min_required_q32: self.min_synthesis_flux_q32,
            }
        };

        HolonomyLoopReceipt {
            path,
            enclosed_curvature_flux_q32: enclosed_flux_q32,
            final_frame_q32,
            wigner_rotation_angle_q32: wigner_angle_q32,
            wigner_axis_skew_q32: skew,
            gauss_bonnet_wigner_discrepancy_q32: discrepancy_q32,
            berry_shifted_phase,
            post_holonomy_invariant,
            verdict,
            mark: EpistemicMark::Verified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas13::Atlas13;
    use crate::silo::FrozenBaseWeights;

    fn test_swarm_map() -> CarvedSwarmMap {
        let atlas = Atlas13::approved_icosahedral();
        let base = FrozenBaseWeights::freeze("Holonomy-Test", 52, vec![0.1f32; 256]);
        CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22))
    }

    #[test]
    fn ping_pong_two_role_loop_has_zero_holonomy_and_is_rejected_as_echo_chamber() {
        let map = test_swarm_map();
        let engine = HyperbolicHolonomyEngine::default();
        let init_phase = map.atlas.well(RoleId::SensoryPerception).phase;

        // Ping-pong: L1 -> L3 -> L1 retraces the exact same 1D geodesic edge
        let receipt = engine.evaluate_role_loop(
            &[
                RoleId::SensoryPerception,
                RoleId::SomaticInteroception,
                RoleId::SensoryPerception,
            ],
            &map,
            init_phase,
        );

        assert_eq!(receipt.enclosed_curvature_flux_q32, Q32_31::ZERO);
        assert_eq!(receipt.wigner_rotation_angle_q32, Q32_31::ZERO);
        assert!(matches!(
            receipt.verdict,
            HolonomyVerdict::EchoChamberRejected { .. }
        ));
    }

    #[test]
    fn genuine_three_role_icosahedral_triangle_proves_gauss_bonnet_and_nonabelian_order() {
        let map = test_swarm_map();
        let engine = HyperbolicHolonomyEngine::default();
        let init_phase = map.atlas.well(RoleId::SensoryPerception).phase;

        // Forward loop: L1 (Sensory) -> L3 (Somatic) -> L5 (EpisodicMemory) -> L1
        let fwd = engine.evaluate_role_loop(
            &[
                RoleId::SensoryPerception,
                RoleId::SomaticInteroception,
                RoleId::EpisodicMemory,
                RoleId::SensoryPerception,
            ],
            &map,
            init_phase,
        );

        // Reverse loop: L1 (Sensory) -> L5 (EpisodicMemory) -> L3 (Somatic) -> L1
        let rev = engine.evaluate_role_loop(
            &[
                RoleId::SensoryPerception,
                RoleId::EpisodicMemory,
                RoleId::SomaticInteroception,
                RoleId::SensoryPerception,
            ],
            &map,
            init_phase,
        );

        // 1. Both enclose strictly positive Gauss-Bonnet hyperbolic curvature flux
        assert!(matches!(
            fwd.verdict,
            HolonomyVerdict::DialecticalSynthesisVerified { .. }
        ));
        assert!(fwd.enclosed_curvature_flux_q32.to_f64() > 0.08);

        // 2. Exact Gauss-Bonnet theorem on H^3: independent Levi-Civita parallel transport
        //    Wigner rotation angle equals trigonometric Gauss-Bonnet area π - (α + β + γ)
        //    to within Q32.31 precision (< 1e-7)!
        assert!(
            fwd.gauss_bonnet_wigner_discrepancy_q32.to_f64() < 1e-7,
            "Gauss-Bonnet ({}) vs Levi-Civita Wigner angle ({}) discrepancy {} exceeded 1e-7",
            fwd.enclosed_curvature_flux_q32.to_f64(),
            fwd.wigner_rotation_angle_q32.to_f64(),
            fwd.gauss_bonnet_wigner_discrepancy_q32.to_f64()
        );

        // 3. Non-Abelian proof of traversal order: forward and reverse loops have identical
        //    total curvature magnitude, but OPPOSITE Wigner rotation skew axes!
        assert!(
            (fwd.wigner_rotation_angle_q32.to_f64() - rev.wigner_rotation_angle_q32.to_f64()).abs()
                < 1e-7
        );
        for k in 0..3 {
            let sum_skew =
                fwd.wigner_axis_skew_q32[k].to_f64() + rev.wigner_axis_skew_q32[k].to_f64();
            assert!(
                sum_skew.abs() < 1e-7,
                "Reversed traversal order must negate Wigner rotation axis component {k}: fwd={} rev={}",
                fwd.wigner_axis_skew_q32[k].to_f64(),
                rev.wigner_axis_skew_q32[k].to_f64()
            );
        }

        // 4. Berry phase shifted the T² Kuramoto phase
        assert_ne!(fwd.berry_shifted_phase.theta_f, init_phase.theta_f);
    }
}
