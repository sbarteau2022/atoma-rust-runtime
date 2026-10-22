//! PROVENANCE: ported 2026-10-19 from the Elle worker's harness
//! (`harness-dhng/src/product.rs`). Pure f64 product-manifold math (Poincare
//! ball + flat torus), unchanged apart from this header and the module name.
//!
//! Mixed-curvature metric space H^n x T^d and geometric primitives.
//!
//! Combines the Poincaré ball H^n (derivation depth & hierarchy) with the flat torus T^d
//! (phase, periodicity & recurrence).
//! Product metric: d^2 = w_h * d_H^2 + w_t * d_T^2.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const BOUNDARY_EPS: f64 = 1e-5;
pub const EPS: f64 = 1e-12;
pub const PHI: f64 = 1.618033988749895; // Golden ratio (1 + sqrt(5)) / 2
pub const TORUS_DIM: usize = 8;
pub const TWO_PI: f64 = std::f64::consts::TAU;
pub const GOLDEN_ANGLE: f64 = TWO_PI * (2.0 - PHI); // ~2.39996 rad

/// Inner product (Euclidean dot product) in R^n.
#[inline]
pub fn dot(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
}

/// Euclidean L2 norm in R^n.
#[inline]
pub fn norm(u: &[f64]) -> f64 {
    dot(u, u).sqrt()
}

/// Project coordinates strictly into the open Poincaré ball ||x|| <= 1 - BOUNDARY_EPS.
pub fn project(x: &[f64]) -> Vec<f64> {
    let n = norm(x);
    let max = 1.0 - BOUNDARY_EPS;
    if n <= max {
        x.to_vec()
    } else {
        let s = max / (n + EPS);
        x.iter().map(|&v| v * s).collect()
    }
}

/// Möbius addition in the Poincaré ball B^n (Ungar gyrovector space):
/// u (+) v = ((1 + 2<u,v> + ||v||^2)u + (1 - ||u||^2)v) / (1 + 2<u,v> + ||u||^2 ||v||^2)
pub fn mobius_add(u: &[f64], v: &[f64]) -> Vec<f64> {
    let uv = dot(u, v);
    let uu = dot(u, u);
    let vv = dot(v, v);
    let den = 1.0 + 2.0 * uv + uu * vv;
    // Sign-preserving guard: near-antipodal boundary points can push `den`
    // to a small *negative* value. Clamping its magnitude to EPS without
    // preserving the sign (e.g. always substituting +EPS) would flip the
    // result across the origin instead of merely bounding its magnitude.
    let safe_den = if den.abs() < EPS {
        if den.is_sign_negative() {
            -EPS
        } else {
            EPS
        }
    } else {
        den
    };
    let a = (1.0 + 2.0 * uv + vv) / safe_den;
    let b = (1.0 - uu) / safe_den;
    let added: Vec<f64> = u.iter().zip(v.iter()).map(|(&ui, &vi)| a * ui + b * vi).collect();
    project(&added)
}

/// Poincaré ball hyperbolic distance:
/// d_H(u, v) = acosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
///
/// A NaN coordinate propagates to a NaN result rather than being silently
/// clamped away: `f64::max` discards NaN (`NaN.max(1.0) == 1.0`), so without
/// this upfront check a corrupted (NaN) prior would report `acosh(1.0) == 0`
/// — "these points are identical" — the wrong direction for a safety signal
/// that is supposed to flag drift.
pub fn poincare_dist(u: &[f64], v: &[f64]) -> f64 {
    if u.iter().any(|x| !x.is_finite()) || v.iter().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    let alpha = (1.0 - dot(u, u)).max(EPS);
    let beta = (1.0 - dot(v, v)).max(EPS);
    let duv: f64 = u.iter().zip(v.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    let gamma = 1.0 + (2.0 * duv) / (alpha * beta);
    gamma.max(1.0).acosh()
}

/// Derivation depth in the Poincaré ball: depth(x) = 2 * atanh(||x||).
pub fn depth(x: &[f64]) -> f64 {
    let n = norm(x).min(1.0 - BOUNDARY_EPS);
    2.0 * n.atanh()
}

/// Exponential map at the origin in the Poincaré ball: exp_0(v) = tanh(||v||) * (v / ||v||).
pub fn exp_map0(v: &[f64]) -> Vec<f64> {
    let n = norm(v);
    if n < EPS {
        return vec![0.0; v.len()];
    }
    let s = n.tanh() / n;
    let res: Vec<f64> = v.iter().map(|&x| x * s).collect();
    project(&res)
}

/// Logarithmic map at the origin in the Poincaré ball: log_0(y) = atanh(||y||) * (y / ||y||).
pub fn log_map0(y: &[f64]) -> Vec<f64> {
    let n = norm(y);
    if n < EPS {
        return vec![0.0; y.len()];
    }
    let s = (n.min(1.0 - BOUNDARY_EPS)).atanh() / n;
    y.iter().map(|&x| x * s).collect()
}

/// Riemannian Euclidean gradient of d_H(u, v) w.r.t. u (Nickel & Kiela 2017).
pub fn dist_grad(u: &[f64], v: &[f64]) -> Vec<f64> {
    let alpha = (1.0 - dot(u, u)).max(EPS);
    let beta = (1.0 - dot(v, v)).max(EPS);
    let duv: f64 = u.iter().zip(v.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    let gamma = 1.0 + (2.0 * duv) / (alpha * beta);
    let denom = beta * (gamma * gamma - 1.0).max(EPS).sqrt();
    let a = (dot(v, v) - 2.0 * dot(u, v) + 1.0) / (alpha * alpha);
    u.iter().zip(v.iter()).map(|(&ui, &vi)| (4.0 / denom) * (a * ui - vi / alpha)).collect()
}

// ── Torus Geometry & Invariants ─────────────────────────────────────────────

/// Normalize any angle to [0, 2*pi).
#[inline]
pub fn norm_2pi(a: f64) -> f64 {
    let x = a % TWO_PI;
    if x < 0.0 {
        x + TWO_PI
    } else {
        x
    }
}

/// Signed wrapped angular difference in (-pi, pi].
#[inline]
pub fn wrap_angle(delta: f64) -> f64 {
    let mut d = delta % TWO_PI;
    if d > std::f64::consts::PI {
        d -= TWO_PI;
    } else if d <= -std::f64::consts::PI {
        d += TWO_PI;
    }
    d
}

/// Torus geodesic distance with optional axis weights:
/// d_T(a, b) = sqrt(sum_i w_i^2 * wrap(a_i - b_i)^2)
pub fn torus_dist(a: &[f64], b: &[f64], weights: Option<&[f64]>) -> f64 {
    let n = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..n {
        let d = wrap_angle(a[i] - b[i]);
        let w = match weights {
            Some(ws) if i < ws.len() => ws[i],
            _ => 1.0,
        };
        sum += (w * d).powi(2);
    }
    sum.sqrt()
}

/// Flat unit torus geodesic distance in [0, 1)^d:
/// d_T,unit(u, v) = sqrt(sum_i min(|u_i - v_i|, 1 - |u_i - v_i|)^2)
pub fn unit_torus_dist(u: &[f64], v: &[f64]) -> f64 {
    let n = u.len().min(v.len());
    let mut sum = 0.0;
    for i in 0..n {
        let diff = (u[i] - v[i]).abs() % 1.0;
        let d = diff.min(1.0 - diff);
        sum += d * d;
    }
    sum.sqrt()
}

/// Golden ratio scaling weights w_i = phi^(-i / 2) for multi-scale toroidal axes.
pub fn phi_scale_weights(dim: usize) -> Vec<f64> {
    (0..dim).map(|i| PHI.powf(-(i as f64) / 2.0)).collect()
}

/// Generalized golden ratio constant solving g^(d+1) = g + 1.
fn generalized_golden(d: usize) -> f64 {
    let mut g: f64 = 2.0;
    for _ in 0..64 {
        g = (1.0 + g).powf(1.0 / ((d + 1) as f64));
    }
    g
}

/// Roberts' R_d low-discrepancy sequence seating bare nodes across T^d.
pub fn golden_sequence(n: usize, dim: usize, seed: f64) -> Vec<Vec<f64>> {
    let g = generalized_golden(dim);
    let alpha: Vec<f64> = (0..dim).map(|j| g.powf(-((j + 1) as f64))).collect();
    let mut out = Vec::with_capacity(n);
    for k in 1..=n {
        let pt: Vec<f64> = alpha
            .iter()
            .map(|&a| norm_2pi(TWO_PI * ((seed + (k as f64) * a) % 1.0)))
            .collect();
        out.push(pt);
    }
    out
}

/// Topological winding numbers in pi_1(T^d) = Z^d and exact cycle turns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindingResult {
    pub winding: Vec<i64>,
    pub turns: Vec<f64>,
}

pub fn winding_numbers(seq: &[Vec<f64>]) -> WindingResult {
    if seq.len() < 2 {
        let dim = seq.first().map(|v| v.len()).unwrap_or(0);
        return WindingResult {
            winding: vec![0; dim],
            turns: vec![0.0; dim],
        };
    }
    let dim = seq[0].len();
    let mut acc = vec![0.0; dim];
    for k in 1..seq.len() {
        for (j, acc_val) in acc.iter_mut().enumerate().take(dim) {
            let prev = seq[k - 1].get(j).copied().unwrap_or(0.0);
            let curr = seq[k].get(j).copied().unwrap_or(0.0);
            *acc_val += wrap_angle(curr - prev);
        }
    }
    let winding = acc.iter().map(|&a| (a / TWO_PI).round() as i64).collect();
    let turns = acc.iter().map(|&a| ((a / TWO_PI) * 10000.0).round() / 10000.0).collect();
    WindingResult { winding, turns }
}

/// Exact topological recognition invariant: integer vector in pi_1(T^d) = Z^d.
pub fn recognition_invariant(phase_seq: &[Vec<f64>]) -> Vec<i64> {
    winding_numbers(phase_seq).winding
}

/// Returns true if two phase trajectories share identical topological winding recurrence identity.
pub fn same_recurrence_class(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    let wa = recognition_invariant(a);
    let wb = recognition_invariant(b);
    wa == wb
}

/// Metric return: closest approach distance to trajectory origin on the torus.
pub fn metric_return(phase_seq: &[Vec<f64>]) -> f64 {
    if phase_seq.len() < 2 {
        return f64::INFINITY;
    }
    let start = &phase_seq[0];
    let mut best = f64::INFINITY;
    for pt in &phase_seq[1..] {
        let d = torus_dist(start, pt, None);
        if d < best {
            best = d;
        }
    }
    best
}

/// Translation alignment score between two phase vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationAlignResult {
    pub shift: f64,
    pub score: f64,
}

pub fn translation_align(a: &[f64], b: &[f64]) -> TranslationAlignResult {
    let n = a.len().min(b.len());
    if n == 0 {
        return TranslationAlignResult { shift: 0.0, score: 0.0 };
    }
    let mut sin_s = 0.0;
    let mut cos_s = 0.0;
    for i in 0..n {
        let d = wrap_angle(a[i] - b[i]);
        sin_s += d.sin();
        cos_s += d.cos();
    }
    let shift = sin_s.atan2(cos_s);
    let mut score = 0.0;
    for i in 0..n {
        score += wrap_angle(a[i] - b[i] - shift).cos();
    }
    TranslationAlignResult {
        shift: ((shift * 100000.0).round()) / 100000.0,
        score: (((score / (n as f64)) * 100000.0).round()) / 100000.0,
    }
}

/// 1D Star discrepancy of angles on circle [0, 2*pi).
pub fn axis_discrepancy(angles: &[f64]) -> f64 {
    let n = angles.len();
    if n == 0 {
        return 1.0;
    }
    let mut pts: Vec<f64> = angles.iter().map(|&a| norm_2pi(a) / TWO_PI).collect();
    // `partial_cmp().unwrap_or(Equal)` makes NaN compare equal to every value
    // while finite values still order normally — not a total order, and the
    // 1.94 standard-library sort detects that non-transitivity and panics.
    // `total_cmp` is a genuine total order over all f64 (NaN included), so
    // sorting never panics regardless of input.
    pts.sort_by(f64::total_cmp);
    let mut d_max = 0.0f64;
    for (i, &p) in pts.iter().enumerate() {
        let d1 = ((i + 1) as f64 / n as f64 - p).abs();
        let d2 = (i as f64 / n as f64 - p).abs();
        d_max = d_max.max(d1).max(d2);
    }
    d_max
}

/// Discrepancy metrics across multi-axis torus atlas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasDiscrepancy {
    pub mean: f64,
    pub max: f64,
    pub per_axis: Vec<f64>,
}

pub fn atlas_discrepancy(points: &[Vec<f64>], dim: usize) -> AtlasDiscrepancy {
    // `dim == 0` must short-circuit here too, not just on an empty `points`:
    // otherwise `mean: sum / (dim as f64)` below divides by zero, and with a
    // non-empty (sum == 0.0) input that is `0.0 / 0.0 == NaN`, not `0.0`.
    if points.is_empty() || dim == 0 {
        return AtlasDiscrepancy {
            mean: 0.0,
            max: 0.0,
            per_axis: vec![0.0; dim],
        };
    }
    let mut per_axis = Vec::with_capacity(dim);
    for j in 0..dim {
        let col: Vec<f64> = points.iter().map(|p| p.get(j).copied().unwrap_or(0.0)).collect();
        per_axis.push(axis_discrepancy(&col));
    }
    let sum: f64 = per_axis.iter().sum();
    let max = per_axis.iter().copied().fold(0.0, f64::max);
    AtlasDiscrepancy {
        mean: sum / (dim as f64),
        max,
        per_axis,
    }
}

/// Nobility metric certifying golden irrational non-resonance:
/// delta_inf(omega) = inf_{n=1..N} n * ||n*omega||_Z
pub fn nobility(omega: f64, n_max: usize) -> f64 {
    let mut w = omega % 1.0;
    if w < 0.0 {
        w += 1.0;
    }
    let mut best = f64::INFINITY;
    for n in 1..=n_max {
        let nw = (n as f64) * w;
        let frac = nw - nw.round();
        let val = (n as f64) * frac.abs();
        if val < best {
            best = val;
        }
        if best < 1e-12 {
            break;
        }
    }
    best
}

/// Maps raw phase vectors to T^d coordinates [0, 2*pi).
pub fn phases_to_torus(phases: &[f64], dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|i| {
            let val = phases.get(i).copied().unwrap_or(0.0);
            if val.is_finite() {
                norm_2pi(val)
            } else {
                0.0
            }
        })
        .collect()
}

// ── Mixed-Curvature Product Space ──────────────────────────────────────────

/// Curvature mix weighting factor for H^n and T^d.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mix {
    pub hyperbolic: f64,
    pub toroidal: f64,
}

impl Default for Mix {
    fn default() -> Self {
        Self {
            hyperbolic: 1.0,
            toroidal: 1.0,
        }
    }
}

/// Combined product distance: d^2 = w_h * d_H^2 + w_t * d_T^2.
pub fn product_dist(
    a_ball: &[f64],
    a_torus: &[f64],
    b_ball: &[f64],
    b_torus: &[f64],
    mix: Option<Mix>,
) -> f64 {
    let m = mix.unwrap_or_default();
    let d_h = poincare_dist(a_ball, b_ball);
    let weights = phi_scale_weights(a_torus.len().min(b_torus.len()));
    let d_t = torus_dist(a_torus, b_torus, Some(&weights));
    (m.hyperbolic * d_h * d_h + m.toroidal * d_t * d_t).sqrt()
}

/// Pairwise chart disagreement entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disagreement {
    pub a: String,
    pub b: String,
    pub ball: f64,
    pub torus: f64,
}

/// Dual-chart disagreement rankings disambiguating lineage and phase.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Disagreements {
    /// Same rhythm (torus-close), different lineage (ball-far): convergent concepts.
    pub same_rhythm_diff_lineage: Vec<Disagreement>,
    /// Same lineage (ball-close), drifted phase (torus-far): derivational drift.
    pub same_lineage_drift_phase: Vec<Disagreement>,
}

/// Dual-chart disagreement ranking options.
#[derive(Debug, Clone)]
pub struct DisagreementOpts {
    pub max_nodes: usize,
    pub top_k: usize,
    pub mix: Option<Mix>,
}

impl Default for DisagreementOpts {
    fn default() -> Self {
        Self {
            max_nodes: 128,
            top_k: 8,
            mix: Some(Mix::default()),
        }
    }
}

/// Over the shared node set, rank the pairs where the two charts most disagree.
pub fn disagreements(
    hyper_points: &HashMap<String, Vec<f64>>,
    torus_points: &HashMap<String, Vec<f64>>,
    opts: Option<DisagreementOpts>,
) -> Disagreements {
    let opt = opts.unwrap_or_default();
    let max_nodes = opt.max_nodes.min(256);
    let top_k = opt.top_k.min(50);
    let mix = opt.mix.unwrap_or_default();

    let mut ids: Vec<String> = hyper_points
        .keys()
        .filter(|k| torus_points.contains_key(*k))
        .cloned()
        .collect();
    ids.sort();
    ids.truncate(max_nodes);

    if ids.len() < 2 {
        return Disagreements::default();
    }

    let torus_dim = torus_points.get(&ids[0]).map(|v| v.len()).unwrap_or(TORUS_DIM);
    let weights = phi_scale_weights(torus_dim);

    struct PairData {
        a: String,
        b: String,
        ball: f64,
        torus: f64,
    }

    let mut pairs = Vec::new();
    let mut ball_max = 1e-9f64;
    let mut torus_max = 1e-9f64;

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let id_a = &ids[i];
            let id_b = &ids[j];
            let ball = poincare_dist(&hyper_points[id_a], &hyper_points[id_b]);
            let torus = torus_dist(&torus_points[id_a], &torus_points[id_b], Some(&weights));
            ball_max = ball_max.max(ball);
            torus_max = torus_max.max(torus);
            pairs.push(PairData {
                a: id_a.clone(),
                b: id_b.clone(),
                ball,
                torus,
            });
        }
    }

    let norm = |p: &PairData| {
        let nb = mix.hyperbolic * (p.ball / ball_max);
        let nt = mix.toroidal * (p.torus / torus_max);
        (nb, nt)
    };

    let mut rhythm_pairs = pairs;
    // total_cmp, not partial_cmp().unwrap_or(Equal): the latter makes NaN
    // compare equal to everything while finite scores still order normally,
    // which is not a total order and the 1.94 stdlib sort panics on it.
    rhythm_pairs.sort_by(|x, y| {
        let (nxb, nxt) = norm(x);
        let (nyb, nyt) = norm(y);
        let score_x = nxb - nxt;
        let score_y = nyb - nyt;
        score_y.total_cmp(&score_x)
    });

    let rhythm: Vec<Disagreement> = rhythm_pairs
        .iter()
        .take(top_k)
        .map(|p| Disagreement {
            a: p.a.clone(),
            b: p.b.clone(),
            ball: ((p.ball * 10000.0).round()) / 10000.0,
            torus: ((p.torus * 10000.0).round()) / 10000.0,
        })
        .collect();

    let mut lineage_pairs = rhythm_pairs;
    lineage_pairs.sort_by(|x, y| {
        let (nxb, nxt) = norm(x);
        let (nyb, nyt) = norm(y);
        let score_x = nxt - nxb;
        let score_y = nyt - nyb;
        score_y.total_cmp(&score_x)
    });

    let lineage: Vec<Disagreement> = lineage_pairs
        .iter()
        .take(top_k)
        .map(|p| Disagreement {
            a: p.a.clone(),
            b: p.b.clone(),
            ball: ((p.ball * 10000.0).round()) / 10000.0,
            torus: ((p.torus * 10000.0).round()) / 10000.0,
        })
        .collect();

    Disagreements {
        same_rhythm_diff_lineage: rhythm,
        same_lineage_drift_phase: lineage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_dist_properties() {
        let origin = vec![0.0, 0.0];
        let p1 = vec![0.3, 0.4]; // norm = 0.5
        let p2 = vec![0.3, 0.4];

        // Identity
        assert!(poincare_dist(&p1, &p2) < 1e-7);

        // Distance from origin: 2 * atanh(||x||)
        let d_origin = poincare_dist(&origin, &p1);
        let expected = 2.0 * (0.5f64).atanh();
        assert!((d_origin - expected).abs() < 1e-6);

        // Symmetry
        let p3 = vec![-0.2, 0.5];
        assert!((poincare_dist(&p1, &p3) - poincare_dist(&p3, &p1)).abs() < 1e-9);

        // Triangle Inequality: d(p1, p3) <= d(p1, origin) + d(origin, p3)
        let d13 = poincare_dist(&p1, &p3);
        let d1o = poincare_dist(&p1, &origin);
        let do3 = poincare_dist(&origin, &p3);
        assert!(d13 <= d1o + do3 + 1e-9);
    }

    #[test]
    fn test_mobius_addition_origin() {
        let origin = vec![0.0, 0.0];
        let v = vec![0.2, -0.3];
        let sum = mobius_add(&origin, &v);
        assert!((sum[0] - v[0]).abs() < 1e-6);
        assert!((sum[1] - v[1]).abs() < 1e-6);
    }

    #[test]
    fn test_torus_dist_and_wrapping() {
        let a = vec![0.1, 0.0];
        let b = vec![TWO_PI - 0.1, 0.0];
        // The wrapped distance on circle should be 0.2, not 2*pi - 0.2
        let d = torus_dist(&a, &b, None);
        assert!((d - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_unit_torus_dist() {
        let u = vec![0.05, 0.95];
        let v = vec![0.95, 0.05];
        // min(|0.05 - 0.95|, 1 - 0.9) = 0.1 for both coordinates
        let d = unit_torus_dist(&u, &v);
        let expected = (0.1f64.powi(2) + 0.1f64.powi(2)).sqrt();
        assert!((d - expected).abs() < 1e-6);
    }

    #[test]
    fn test_topological_winding_numbers() {
        // A single full revolution around axis 0
        let steps = 16;
        let mut loop_traj = Vec::new();
        for k in 0..=steps {
            let theta = (k as f64 / steps as f64) * TWO_PI;
            loop_traj.push(vec![norm_2pi(theta), 0.0]);
        }
        let w = winding_numbers(&loop_traj);
        assert_eq!(w.winding[0], 1);
        assert_eq!(w.winding[1], 0);

        // A jitter in place
        let jitter_traj = vec![vec![0.1, 0.1], vec![0.15, 0.12], vec![0.08, 0.09], vec![0.1, 0.1]];
        let w_jit = winding_numbers(&jitter_traj);
        assert_eq!(w_jit.winding, vec![0, 0]);
    }

    #[test]
    fn test_mixed_curvature_product_dist() {
        let a_ball = vec![0.1, 0.2];
        let a_torus = vec![0.0, 1.0];
        let b_ball = vec![0.3, 0.4];
        let b_torus = vec![0.2, 1.2];

        let d = product_dist(&a_ball, &a_torus, &b_ball, &b_torus, Some(Mix { hyperbolic: 1.0, toroidal: 1.0 }));
        assert!(d > 0.0);
    }

    // ── Fix: NaN handling ──

    #[test]
    fn test_poincare_dist_propagates_nan_instead_of_reporting_identical() {
        // `f64::max` discards NaN (`NaN.max(1.0) == 1.0`), so clamping gamma
        // with `.max(1.0)` alone would turn a NaN (corrupted) coordinate
        // into `acosh(1.0) == 0.0` -- "these points are identical", the
        // wrong direction for a safety/drift signal.
        let u = vec![f64::NAN, 0.2];
        let v = vec![0.1, 0.2];
        let d = poincare_dist(&u, &v);
        assert!(d.is_nan(), "a NaN coordinate must propagate to a NaN distance, got {d}");

        // A healthy input must still behave normally.
        let healthy = poincare_dist(&[0.1, 0.2], &[0.15, 0.22]);
        assert!(healthy.is_finite());
    }

    #[test]
    fn test_mobius_add_sign_preserving_near_antipodal_guard() {
        // Near-antipodal boundary points push the Mobius-addition
        // denominator to a small *negative* value. Replacing it with a bare
        // `+EPS` (ignoring sign) flips the result across the origin instead
        // of merely bounding the magnitude.
        let u = vec![0.999995, 0.0];
        let v = vec![-0.999995, 0.0];
        let result = mobius_add(&u, &v);
        assert!(result.iter().all(|x| x.is_finite()), "mobius_add must not produce NaN/Inf: {result:?}");
        // The result must stay inside the open ball.
        assert!(norm(&result) < 1.0);
    }

    #[test]
    fn test_atlas_discrepancy_zero_dim_with_nonempty_points_is_not_nan() {
        // `dim == 0` must short-circuit even when `points` is non-empty --
        // otherwise `mean: sum / (dim as f64)` is `0.0 / 0.0 == NaN`.
        let points = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let disc = atlas_discrepancy(&points, 0);
        assert!(!disc.mean.is_nan(), "atlas_discrepancy(points, 0) must not be NaN");
        assert_eq!(disc.mean, 0.0);
        assert_eq!(disc.max, 0.0);
        assert!(disc.per_axis.is_empty());
    }

    #[test]
    fn test_axis_discrepancy_sort_does_not_panic_on_nan() {
        // `partial_cmp().unwrap_or(Equal)` makes NaN compare equal to every
        // value while finite values still order normally -- not a total
        // order, and the standard-library sort (Rust 1.94 toolchain here)
        // detects the violated invariant and panics. `total_cmp` fixes it.
        let angles = vec![0.1, f64::NAN, 2.0, -1.0, f64::NAN];
        // The real assertion is that this call returns at all instead of
        // panicking inside sort_by; `f64::max`'s NaN-discarding semantics
        // keep the aggregated result itself finite.
        let d = axis_discrepancy(&angles);
        assert!(d.is_finite(), "expected a finite discrepancy, got {d}");
    }

    #[test]
    fn test_disagreements_sort_does_not_panic_on_nan_points() {
        // A NaN hyperbolic coordinate now propagates through poincare_dist
        // into a NaN pairwise score; the ranking sort must not panic on it.
        let mut hyper_points = HashMap::new();
        hyper_points.insert("a".to_string(), vec![0.1, 0.1]);
        hyper_points.insert("b".to_string(), vec![f64::NAN, 0.2]);
        hyper_points.insert("c".to_string(), vec![0.2, -0.1]);
        hyper_points.insert("d".to_string(), vec![-0.1, 0.3]);

        let mut torus_points = HashMap::new();
        torus_points.insert("a".to_string(), vec![0.0, 1.0]);
        torus_points.insert("b".to_string(), vec![0.5, 1.5]);
        torus_points.insert("c".to_string(), vec![1.0, 0.2]);
        torus_points.insert("d".to_string(), vec![2.0, 0.4]);

        // Must complete without panicking; the exact ranking of NaN-tainted
        // pairs is unspecified, only that sorting terminates cleanly.
        let result = disagreements(&hyper_points, &torus_points, None);
        assert!(result.same_rhythm_diff_lineage.len() <= 8);
        assert!(result.same_lineage_drift_phase.len() <= 8);
    }
}
