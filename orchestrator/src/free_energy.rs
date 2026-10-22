//! Free-energy regulator (ported 2026-10-19).
//!
//! PROVENANCE: port of the Elle worker's `src/regulator.ts` — "THE FREE-ENERGY
//! REGULATOR". Constrains a build to its invariants by making each invariant a
//! cost and driving the system down a free-energy functional:
//!
//! ```text
//! F(c) = U(c) − T·S(c)
//!      = Σ_k a_k (1 − c_k)²   [internal energy: invariant/coherence cost]
//!      + T · Σ_k (c_k − c̄)²   [anisotropy term: entropy highest when homogeneous]
//! ```
//!
//! c = (structural, relational, harmonic) coherence, each in [0,1]; the
//! φ-weights a_k = (1, 1/φ, 1/φ²) shape the descent without moving the fixed
//! point c* = (1,1,1). Free energy is conserved: F(t) + W(t) = F(0) exactly,
//! where work W(t) = Σ (F(t−1) − F(t)). A golden-angle quasiperiodic
//! perturbation (annealed to zero) escapes spurious dissonance wells.
//!
//! In the monster, the per-turn coherence c is MEASURED (not descended):
//! each live turn reports its three coherence coordinates, and the ledger
//! tracks F, work, conservation, and dissonance across the session. This
//! replaces the hardcoded 0.12 / 0.96 constants the receipts used to carry.
//! Deterministic, pure, no allocation beyond small vectors.

pub const PHI: f64 = 1.6180339887498949; // (1 + sqrt(5)) / 2
pub const PHI_INV: f64 = 1.0 / PHI;

/// φ-partition regulator gains (1, 1/φ, 1/φ²), normalized to sum 1.
pub fn phi_weights() -> [f64; 3] {
    let raw = [1.0, PHI_INV, PHI_INV * PHI_INV];
    let sum: f64 = raw.iter().sum();
    [raw[0] / sum, raw[1] / sum, raw[2] / sum]
}

/// The three coherence coordinates the regulator drives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coherence {
    pub structural: f64,
    pub relational: f64,
    pub harmonic: f64,
}

impl Coherence {
    pub fn new(structural: f64, relational: f64, harmonic: f64) -> Self {
        Self {
            structural: clamp01(structural),
            relational: clamp01(relational),
            harmonic: clamp01(harmonic),
        }
    }

    pub fn as_array(&self) -> [f64; 3] {
        [self.structural, self.relational, self.harmonic]
    }
}

/// The free-energy readout for one coherence state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreeEnergy {
    pub f: f64, // total free energy
    pub u: f64, // internal energy — the invariant/coherence cost
    pub s: f64, // homogeneity readout (1 − normalized anisotropy)
    pub a: f64, // anisotropy = Σ (c_k − c̄)² (0 ⇒ isotropic)
}

pub fn clamp01(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// F = U + T·A over the coherence coordinates.
pub fn free_energy(c: Coherence, t: f64, weights: [f64; 3]) -> FreeEnergy {
    let v = c.as_array();
    let mean = (v[0] + v[1] + v[2]) / 3.0;
    let a: f64 = v.iter().map(|x| (x - mean).powi(2)).sum();
    let u: f64 = weights
        .iter()
        .zip(v.iter())
        .map(|(w, x)| w * (1.0 - x).powi(2))
        .sum();
    let f = u + t * a;
    let s = 1.0 - a / 0.6667; // normalized homogeneity readout (A_max ≈ 2/3)
    FreeEnergy {
        f: round6(f),
        u: round6(u),
        s: round6(s),
        a: round6(a),
    }
}

/// ∂F/∂c_j = −2 a_j (1 − c_j) + 2T (c_j − c̄)
fn gradient(v: [f64; 3], t: f64, w: [f64; 3]) -> [f64; 3] {
    let mean = (v[0] + v[1] + v[2]) / 3.0;
    [
        -2.0 * w[0] * (1.0 - v[0]) + 2.0 * t * (v[0] - mean),
        -2.0 * w[1] * (1.0 - v[1]) + 2.0 * t * (v[1] - mean),
        -2.0 * w[2] * (1.0 - v[2]) + 2.0 * t * (v[2] - mean),
    ]
}

fn frac(x: f64) -> f64 {
    x - x.floor()
}

/// Golden-angle quasiperiodic perturbation: phase per coordinate offset so no
/// coordinate is privileged; equidistributed (Weyl) so it never repeats.
fn phi_perturb(step: u64, k: usize, amp: f64) -> f64 {
    const GOLDEN: f64 = 0.6180339887498949; // 1/φ
    let phase = frac((step as f64 + 1.0) * GOLDEN + k as f64 * GOLDEN * GOLDEN) * 2.0
        * std::f64::consts::PI;
    amp * phase.cos()
}

#[derive(Debug, Clone)]
pub struct RegulatorConfig {
    pub t: f64,
    pub steps: usize,
    pub lr: f64,
    pub perturb: f64,
    pub anneal: f64,
    pub tol: f64,
    pub weights: [f64; 3],
}

impl Default for RegulatorConfig {
    fn default() -> Self {
        Self {
            t: 0.5,
            steps: 400,
            lr: 0.12,
            perturb: 0.0,
            anneal: 0.97,
            tol: 1e-5,
            weights: phi_weights(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RegulatorStep {
    pub t: usize,
    pub coherence: Coherence,
    pub f: f64,
    pub u: f64,
    pub a: f64,
    pub work: f64,
    pub conserved: f64,
    pub dissonance: f64,
}

#[derive(Debug, Clone)]
pub struct RegulatorResult {
    pub trace: Vec<RegulatorStep>,
    pub final_step: RegulatorStep,
    pub f0: f64,
    pub converged: bool,
    pub monotone_descent: bool,
    pub conserved_ok: bool,
    pub isotropic: bool,
    pub balanced_superposition: bool,
    pub dissonance_final: f64,
}

/// Drive the coherence state down the free-energy functional with an exact
/// conserved ledger. Pure descent (perturb = 0) is a monotone Lyapunov
/// descent to c* = (1,1,1).
pub fn regulate(init: Coherence, cfg: &RegulatorConfig) -> RegulatorResult {
    let mut v = [
        clamp01(init.structural),
        clamp01(init.relational),
        clamp01(init.harmonic),
    ];
    let f0 = free_energy(
        Coherence::new(v[0], v[1], v[2]),
        cfg.t,
        cfg.weights,
    )
    .f;
    let mut trace: Vec<RegulatorStep> = Vec::new();
    let mut work = 0.0;
    let mut monotone = true;
    let mut prev_f = f0;
    let mut amp = cfg.perturb;

    for step in 0..cfg.steps {
        let g = gradient(v, cfg.t, cfg.weights);
        let mut next = [0.0; 3];
        let mut delta = 0.0;
        for k in 0..3 {
            let p = if amp > 0.0 {
                phi_perturb(step as u64, k, amp)
            } else {
                0.0
            };
            let nk = clamp01(v[k] - cfg.lr * g[k] + p);
            delta += (nk - v[k]).powi(2);
            next[k] = nk;
        }
        v = next;
        amp *= cfg.anneal;

        let fe = free_energy(Coherence::new(v[0], v[1], v[2]), cfg.t, cfg.weights);
        work += prev_f - fe.f;
        if fe.f > prev_f + 1e-12 {
            monotone = false;
        }
        prev_f = fe.f;
        let dissonance = delta.sqrt();

        trace.push(RegulatorStep {
            t: step,
            coherence: Coherence::new(v[0], v[1], v[2]),
            f: fe.f,
            u: fe.u,
            a: fe.a,
            work: round6(work),
            conserved: round6(fe.f + work),
            dissonance: round6(dissonance),
        });

        if amp < 1e-6 && dissonance < cfg.tol {
            break;
        }
    }

    let final_step = trace.last().cloned().expect("regulate ran zero steps");
    let converged = final_step.dissonance < cfg.tol.max(1e-4) && final_step.f < 1e-3;
    let conserved_ok = trace
        .iter()
        .all(|s| (s.conserved - f0).abs() < 1e-5);
    let isotropic = final_step.a < 1e-3;
    let fv = final_step.coherence.as_array();
    let balanced_superposition = (fv[0] - fv[1]).abs() < 1e-2
        && (fv[1] - fv[2]).abs() < 1e-2
        && fv.iter().all(|&x| x > 0.98);

    RegulatorResult {
        trace,
        final_step,
        f0: round6(f0),
        converged,
        monotone_descent: monotone,
        conserved_ok,
        isotropic,
        balanced_superposition,
        dissonance_final: final_step.dissonance,
    }
}

/// Per-session free-energy ledger for measured (not descended) coherence.
/// The monster observes c each turn; the ledger reports F, work converted,
/// conservation, and per-turn dissonance ‖Δc‖.
#[derive(Debug, Clone)]
pub struct FreeEnergyLedger {
    pub f0: Option<f64>,
    pub prev: Option<Coherence>,
    pub prev_f: f64,
    pub work: f64,
    pub turns: u64,
    pub t: f64,
    pub weights: [f64; 3],
}

impl FreeEnergyLedger {
    pub fn new() -> Self {
        Self {
            f0: None,
            prev: None,
            prev_f: 0.0,
            work: 0.0,
            turns: 0,
            t: 0.5,
            weights: phi_weights(),
        }
    }

    /// Observe one turn's coherence; returns (F, dissonance, conserved).
    pub fn observe(&mut self, c: Coherence) -> (f64, f64, f64) {
        let fe = free_energy(c, self.t, self.weights);
        let f = fe.f;
        let dissonance = match self.prev {
            Some(p) => {
                let d: f64 = c
                    .as_array()
                    .iter()
                    .zip(p.as_array().iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                round6(d.sqrt())
            }
            None => 0.0,
        };
        if self.f0.is_none() {
            self.f0 = Some(f);
        } else {
            self.work = round6(self.work + (self.prev_f - f));
        }
        self.prev = Some(c);
        self.prev_f = f;
        self.turns += 1;
        let conserved = round6(f + self.work);
        (f, dissonance, conserved)
    }
}

impl Default for FreeEnergyLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_energy_at_fixed_point_is_zero() {
        let fe = free_energy(Coherence::new(1.0, 1.0, 1.0), 0.5, phi_weights());
        assert_eq!(fe.f, 0.0);
        assert_eq!(fe.u, 0.0);
        assert_eq!(fe.a, 0.0);
    }

    #[test]
    fn phi_weights_normalize_to_one() {
        let w = phi_weights();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(w[0] > w[1] && w[1] > w[2]);
    }

    #[test]
    fn regulate_conserves_free_energy_and_descends_monotonically() {
        let r = regulate(
            Coherence::new(0.3, 0.5, 0.7),
            &RegulatorConfig::default(),
        );
        assert!(r.conserved_ok, "F + work must equal F0 throughout");
        assert!(r.monotone_descent, "pure descent is monotone");
        assert!(r.final_step.f < r.f0);
        assert!(r.isotropic);
        assert!(r.balanced_superposition);
        assert!(r.converged);
    }

    #[test]
    fn regulate_from_worst_case_reaches_fixed_point() {
        let r = regulate(
            Coherence::new(0.0, 0.0, 0.0),
            &RegulatorConfig::default(),
        );
        assert!(r.converged);
        assert!(r.conserved_ok);
    }

    #[test]
    fn ledger_tracks_dissonance_and_conservation() {
        let mut l = FreeEnergyLedger::new();
        let (f1, d1, c1) = l.observe(Coherence::new(0.6, 0.6, 0.6));
        assert_eq!(d1, 0.0);
        assert_eq!(c1, f1); // first turn: conserved == F0
        let (f2, d2, c2) = l.observe(Coherence::new(1.0, 1.0, 1.0));
        assert!(d2 > 0.0);
        assert!(f2 < f1); // higher coherence, lower free energy
        assert!((c2 - f1).abs() < 1e-5); // conserved == F0 still
    }
}
