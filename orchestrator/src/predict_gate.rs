//! PROVENANCE: ported 2026-10-19 from the Elle worker's prediction operator
//! (`harness-dhng/src/predict.rs`, itself a port of the Rust harness operator).
//! The `ProductPredictor`, Welford z-scoring, and best-of persistence/EMA/AR(2)
//! core are Stewart's design, adapted here: the manifold import points at the
//! local `manifold` module, and a `TrajectoryMatchGate` wrapper (below) turns the
//! per-step surprise signal into an accept/reject verdict on map candidates.
//! Lorentz->Poincare conversion for the monster's 6D carrier addresses
//! (4 Lorentz + 2 toroidal phases) lives in the gate wrapper.
//!
//! Prediction operator P: one-step forecasting with residual-as-surprise.
//!
//! Two predictors over the same classical core (persistence / EMA / AR(2),
//! best-of by running one-step MAE, skill reported against persistence per
//! Hyndman & Koehler 2006):
//!
//! 1. [`EuclidPredictor`] (v1) — treats observations as flat feature vectors;
//!    residual = Euclidean L2 between forecast and observation.
//! 2. [`ProductPredictor`] (v2) — observations are points on H^n x T^d.
//!    Hyperbolic factor: trajectory is mapped to the tangent space at the
//!    origin (log_map0), the Euclidean core forecasts there, exp_map0 maps
//!    back, and the residual is the true Poincare distance d_H. Toroidal
//!    factor: per-axis wrapped phase velocity (Fisher & Lee-style circular
//!    handling — velocities are wrapped into (-pi, pi] before any averaging),
//!    forecast phases re-wrapped, residual = d_T. Combined residual uses the
//!    product metric d^2 = w_h*d_H^2 + w_t*d_T^2 ([`Mix`]).
//!
//! Surprise normalization: each residual stream is z-scored against its own
//! running Welford moments (precision-weighting of prediction errors, after
//! Rao & Ballard 1999 / Friston 2005). The combined surprise for v2 is the
//! max of the per-factor z-scores so hyperbolic and toroidal residuals share
//! one scale despite incommensurate units.
//!
//! NaN policy matches `poincare_dist`: a non-finite observation yields a
//! NaN-residual step and is NOT ingested, so one corrupted frame neither
//! silences the surprise signal nor poisons future forecasts.
//!
//! Base-point choice: the hyperbolic factor predicts in the tangent space at
//! the origin (a single global chart, as in HTGN, KDD 2021) rather than at a
//! moving base point. This avoids parallel transport entirely and keeps the
//! operator deterministic and closed-form; the upgrade path is Exp/Log at the
//! last observation with transported velocities (Pennec 2006; Hauberg et al.
//! 2013 manifold UKF).

use crate::manifold::{
    exp_map0, log_map0, norm_2pi, phi_scale_weights, poincare_dist, torus_dist, wrap_angle, Mix,
};
use serde::{Deserialize, Serialize};

/// Default EMA smoothing factor (Brown/Holt simple exponential smoothing).
pub const PREDICT_EMA_ALPHA: f64 = 0.3;
/// Default sliding-window capacity for AR fitting.
pub const PREDICT_WINDOW: usize = 64;
/// Minimum history before an AR(2) fit is attempted.
pub const PREDICT_MIN_AR_HISTORY: usize = 6;
/// Minimum residual count before z-scores are reported (variance is
/// meaningless below this).
pub const PREDICT_MIN_Z_COUNT: usize = 3;
/// Floor on the residual standard deviation used for z-scoring.
pub const PREDICT_Z_STD_FLOOR: f64 = 1e-9;

/// Forecast method chosen for a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastMethod {
    Persistence,
    Ema,
    Ar2,
}

/// One prediction step: forecast made BEFORE seeing the observation, and the
/// resulting residual/surprise once the observation arrived.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictStep {
    /// One-step-ahead forecast that was compared against the observation.
    /// None during warm-up (no history yet) or on a rejected observation.
    pub predicted: Option<Vec<f64>>,
    /// Residual in the predictor's metric (L2 for v1). NaN when the
    /// observation was non-finite or mis-dimensioned.
    pub residual: Option<f64>,
    /// Running z-score of the residual against its own history (0.0 until
    /// PREDICT_MIN_Z_COUNT residuals have been seen).
    pub z: Option<f64>,
    /// Running MAE(chosen) / MAE(persistence); < 1.0 means the operator is
    /// beating the persistence baseline (MASE-style skill).
    pub skill_vs_persistence: Option<f64>,
    /// Which forecast method produced `predicted`.
    pub method: Option<ForecastMethod>,
}

impl PredictStep {
    fn empty() -> Self {
        Self {
            predicted: None,
            residual: None,
            z: None,
            skill_vs_persistence: None,
            method: None,
        }
    }

    fn rejected() -> Self {
        Self {
            predicted: None,
            residual: Some(f64::NAN),
            z: Some(f64::NAN),
            skill_vs_persistence: None,
            method: None,
        }
    }
}

/// Welford running mean/variance for residual z-scoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Welford {
    pub count: usize,
    pub mean: f64,
    m2: f64,
}

impl Welford {
    /// z-score of `x` against the moments accumulated so far, then ingest it.
    pub fn z_then_update(&mut self, x: f64) -> f64 {
        let z = if self.count >= PREDICT_MIN_Z_COUNT {
            let var = self.m2 / (self.count as f64 - 1.0);
            (x - self.mean) / var.sqrt().max(PREDICT_Z_STD_FLOOR)
        } else {
            0.0
        };
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
        z
    }
}

/// Running mean absolute error for one forecast method.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RunningMae {
    count: usize,
    sum_abs: f64,
}

impl RunningMae {
    fn update(&mut self, abs_err: f64) {
        self.count += 1;
        self.sum_abs += abs_err;
    }

    fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum_abs / self.count as f64)
        }
    }
}

/// Fit AR(2) with intercept by ordinary least squares on `series` and return
/// the one-step-ahead forecast. Returns None when the window is too short or
/// the normal equations are ill-conditioned.
/// AR(1)+intercept OLS fallback for AR(2)-singular series (e.g. pure ramps,
/// where y_t = c + y_{t-1} holds exactly).
fn ar1_forecast(series: &[f64]) -> Option<f64> {
    let n = series.len();
    if n < 3 {
        return None;
    }
    let k = (n - 1) as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for t in 1..n {
        let x = series[t - 1];
        let y = series[t];
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let det = k * sxx - sx * sx;
    if det.abs() < 1e-9 * (1.0 + (k * sxx).abs()) {
        return None;
    }
    let p = (k * sxy - sx * sy) / det;
    let c = (sy - p * sx) / k;
    let forecast = c + p * series[n - 1];
    forecast.is_finite().then_some(forecast)
}

fn ar2_forecast(series: &[f64]) -> Option<f64> {
    let n = series.len();
    if n < PREDICT_MIN_AR_HISTORY {
        return None;
    }
    // Regression rows: y_t = c + p1*y_{t-1} + p2*y_{t-2}, t = 2..n-1.
    let (mut s1, mut s2, mut sy) = (0.0, 0.0, 0.0);
    let (mut s11, mut s12, mut s22) = (0.0, 0.0, 0.0);
    let (mut s1y, mut s2y) = (0.0, 0.0);
    let m = (n - 2) as f64;
    for t in 2..n {
        let x1 = series[t - 1];
        let x2 = series[t - 2];
        let y = series[t];
        s1 += x1;
        s2 += x2;
        sy += y;
        s11 += x1 * x1;
        s12 += x1 * x2;
        s22 += x2 * x2;
        s1y += x1 * y;
        s2y += x2 * y;
    }
    // Normal equations (3x3, symmetric): solve by Cramer's rule.
    // | m    s1   s2  | | c  |   | sy  |
    // | s1   s11  s12 | | p1 | = | s1y |
    // | s2   s12  s22 | | p2 |   | s2y |
    let det = m * (s11 * s22 - s12 * s12) - s1 * (s1 * s22 - s12 * s2)
        + s2 * (s1 * s12 - s11 * s2);
    // Scale-aware singularity guard: a pure linear ramp makes the AR(2)
    // regressors exactly collinear (x_{t-1} = x_{t-2} + const), so fall back
    // to AR(1)+intercept, which fits ramps exactly; a constant series is
    // singular for that too and yields None (persistence covers it).
    if det.abs() < 1e-9 * (1.0 + (m * s11 * s22).abs()) {
        return ar1_forecast(series);
    }
    let det_c = sy * (s11 * s22 - s12 * s12) - s1 * (s1y * s22 - s12 * s2y)
        + s2 * (s1y * s12 - s11 * s2y);
    let det_p1 = m * (s1y * s22 - s12 * s2y) - sy * (s1 * s22 - s12 * s2)
        + s2 * (s1 * s2y - s1y * s2);
    let det_p2 = m * (s11 * s2y - s1y * s12) - s1 * (s1 * s2y - s1y * s2)
        + sy * (s1 * s12 - s11 * s2);
    let c = det_c / det;
    let p1 = det_p1 / det;
    let p2 = det_p2 / det;
    let forecast = c + p1 * series[n - 1] + p2 * series[n - 2];
    if forecast.is_finite() {
        Some(forecast)
    } else {
        None
    }
}

/// Shared classical forecasting core over R^dim: persistence, EMA and
/// per-dimension AR(2), best-of selected by running one-step MAE.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EuclidCore {
    dim: usize,
    alpha: f64,
    window_cap: usize,
    history: Vec<Vec<f64>>,
    ema_level: Option<Vec<f64>>,
    mae_persistence: RunningMae,
    mae_ema: RunningMae,
    mae_ar2: RunningMae,
}

impl EuclidCore {
    fn new(dim: usize, alpha: f64, window_cap: usize) -> Self {
        Self {
            dim,
            alpha,
            window_cap: window_cap.max(PREDICT_MIN_AR_HISTORY),
            history: Vec::new(),
            ema_level: None,
            mae_persistence: RunningMae::default(),
            mae_ema: RunningMae::default(),
            mae_ar2: RunningMae::default(),
        }
    }

    fn forecast_persistence(&self) -> Option<Vec<f64>> {
        self.history.last().cloned()
    }

    fn forecast_ema(&self) -> Option<Vec<f64>> {
        self.ema_level.clone()
    }

    fn forecast_ar2(&self) -> Option<Vec<f64>> {
        if self.history.len() < PREDICT_MIN_AR_HISTORY {
            return None;
        }
        let mut out = Vec::with_capacity(self.dim);
        let mut series = Vec::with_capacity(self.history.len());
        for d in 0..self.dim {
            series.clear();
            series.extend(self.history.iter().map(|v| v[d]));
            out.push(ar2_forecast(&series)?);
        }
        Some(out)
    }

    /// Pick the method with the lowest running MAE so far (persistence when
    /// nothing has been scored yet — the honest cold-start default).
    fn choose(&self) -> ForecastMethod {
        let mut best = ForecastMethod::Persistence;
        let mut best_mae = self.mae_persistence.mean().unwrap_or(f64::INFINITY);
        if let Some(m) = self.mae_ema.mean() {
            if m < best_mae {
                best = ForecastMethod::Ema;
                best_mae = m;
            }
        }
        if let Some(m) = self.mae_ar2.mean() {
            if m < best_mae {
                best = ForecastMethod::Ar2;
            }
        }
        best
    }

    fn l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Forecast (using only past data), score against `obs`, then ingest it.
    /// Returns (forecast, method, residual, skill) — all None during warm-up.
    #[allow(clippy::type_complexity)]
    fn step(
        &mut self,
        obs: &[f64],
    ) -> (Option<Vec<f64>>, Option<ForecastMethod>, Option<f64>, Option<f64>) {
        let method = self.choose();
        let forecast = match method {
            ForecastMethod::Persistence => self.forecast_persistence(),
            ForecastMethod::Ema => self.forecast_ema(),
            ForecastMethod::Ar2 => self.forecast_ar2(),
        };
        let residual = forecast.as_ref().map(|f| Self::l2(f, obs));

        // Score every candidate method on this observation for future choice.
        if let Some(f) = self.forecast_persistence() {
            self.mae_persistence.update(Self::l2(&f, obs));
        }
        if let Some(f) = self.forecast_ema() {
            self.mae_ema.update(Self::l2(&f, obs));
        }
        if let Some(f) = self.forecast_ar2() {
            self.mae_ar2.update(Self::l2(&f, obs));
        }

        let skill = match (
            self.mae_persistence.mean(),
            match method {
                ForecastMethod::Persistence => self.mae_persistence.mean(),
                ForecastMethod::Ema => self.mae_ema.mean(),
                ForecastMethod::Ar2 => self.mae_ar2.mean(),
            },
        ) {
            (Some(p), Some(c)) if p > 0.0 => Some(c / p),
            _ => None,
        };

        // Ingest.
        match &mut self.ema_level {
            Some(level) => {
                for (l, &o) in level.iter_mut().zip(obs.iter()) {
                    *l = self.alpha * o + (1.0 - self.alpha) * *l;
                }
            }
            None => self.ema_level = Some(obs.to_vec()),
        }
        self.history.push(obs.to_vec());
        if self.history.len() > self.window_cap {
            self.history.remove(0);
        }

        let chosen = forecast.is_some().then_some(method);
        (forecast, chosen, residual, skill)
    }
}

/// Options shared by both predictors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictOpts {
    /// EMA smoothing factor in (0, 1].
    pub alpha: f64,
    /// Sliding-window capacity for AR fitting.
    pub window: usize,
}

impl Default for PredictOpts {
    fn default() -> Self {
        Self {
            alpha: PREDICT_EMA_ALPHA,
            window: PREDICT_WINDOW,
        }
    }
}

// ── v1: Euclidean predictor (no DHNG) ───────────────────────────────────────

/// v1 prediction operator: classical one-step forecasting over flat feature
/// vectors, blind to the manifold. Residual = Euclidean L2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EuclidPredictor {
    core: EuclidCore,
    welford: Welford,
}

impl EuclidPredictor {
    pub fn new(dim: usize, opts: Option<PredictOpts>) -> Self {
        let o = opts.unwrap_or_default();
        Self {
            core: EuclidCore::new(dim, o.alpha, o.window),
            welford: Welford::default(),
        }
    }

    /// Feed one observation; returns forecast-vs-observation for this step.
    pub fn step(&mut self, obs: &[f64]) -> PredictStep {
        if obs.len() != self.core.dim || obs.iter().any(|x| !x.is_finite()) {
            return PredictStep::rejected();
        }
        let (forecast, method, residual, skill) = self.core.step(obs);
        let z = residual.map(|r| self.welford.z_then_update(r));
        if residual.is_none() {
            return PredictStep {
                predicted: forecast,
                method,
                ..PredictStep::empty()
            };
        }
        PredictStep {
            predicted: forecast,
            residual,
            z,
            skill_vs_persistence: skill,
            method,
        }
    }
}

// ── v2: product-manifold predictor (DHNG-aware) ─────────────────────────────

/// One v2 prediction step over H^n x T^d, with the residual decomposed into
/// its hyperbolic and toroidal components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductPredictStep {
    /// Forecast point in the Poincare ball (None during warm-up/rejection).
    pub predicted_ball: Option<Vec<f64>>,
    /// Forecast phases in [0, 2*pi) (None during warm-up/rejection).
    pub predicted_phases: Option<Vec<f64>>,
    /// Hyperbolic residual d_H(predicted, observed).
    pub residual_hyper: Option<f64>,
    /// Toroidal residual d_T(predicted, observed) with phi-decayed weights.
    pub residual_torus: Option<f64>,
    /// Product residual sqrt(w_h*d_H^2 + w_t*d_T^2).
    pub residual_product: Option<f64>,
    /// Running z-score of the hyperbolic residual.
    pub z_hyper: Option<f64>,
    /// Running z-score of the toroidal residual.
    pub z_torus: Option<f64>,
    /// Combined surprise: max of the component z-scores (shared scale).
    pub surprise: Option<f64>,
    /// Skill vs persistence of the tangent-space (hyperbolic) core.
    pub skill_hyper: Option<f64>,
    /// Skill vs persistence of the phase-velocity (toroidal) core.
    pub skill_torus: Option<f64>,
}

impl ProductPredictStep {
    fn rejected() -> Self {
        Self {
            predicted_ball: None,
            predicted_phases: None,
            residual_hyper: Some(f64::NAN),
            residual_torus: Some(f64::NAN),
            residual_product: Some(f64::NAN),
            z_hyper: Some(f64::NAN),
            z_torus: Some(f64::NAN),
            surprise: Some(f64::NAN),
            skill_hyper: None,
            skill_torus: None,
        }
    }
}

/// v2 prediction operator on the product manifold H^n x T^d.
///
/// Hyperbolic factor: forecasts in the tangent space at the origin and
/// measures the residual with the intrinsic Poincare distance, so a jump that
/// is metrically large near the boundary registers as large even when its
/// Euclidean coordinate displacement is tiny. Toroidal factor: forecasts
/// per-axis wrapped phase velocity, so a trajectory crossing the 0/2*pi seam
/// produces no spurious residual tear.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductPredictor {
    hyper_dim: usize,
    torus_dim: usize,
    mix: Mix,
    torus_weights: Vec<f64>,
    /// Euclidean core over tangent-space coordinates at the origin.
    tangent_core: EuclidCore,
    /// Euclidean core over per-axis wrapped phase velocities.
    velocity_core: EuclidCore,
    last_phases: Option<Vec<f64>>,
    welford_hyper: Welford,
    welford_torus: Welford,
}

impl ProductPredictor {
    pub fn new(hyper_dim: usize, torus_dim: usize, mix: Option<Mix>, opts: Option<PredictOpts>) -> Self {
        let o = opts.unwrap_or_default();
        Self {
            hyper_dim,
            torus_dim,
            mix: mix.unwrap_or_default(),
            torus_weights: phi_scale_weights(torus_dim),
            tangent_core: EuclidCore::new(hyper_dim, o.alpha, o.window),
            velocity_core: EuclidCore::new(torus_dim, o.alpha, o.window),
            last_phases: None,
            welford_hyper: Welford::default(),
            welford_torus: Welford::default(),
        }
    }

    /// Feed one observed point (ball coordinates, phases in radians).
    pub fn step(&mut self, ball: &[f64], phases: &[f64]) -> ProductPredictStep {
        let bad = ball.len() != self.hyper_dim
            || phases.len() != self.torus_dim
            || ball.iter().any(|x| !x.is_finite())
            || phases.iter().any(|x| !x.is_finite())
            || crate::manifold::norm(ball) >= 1.0;
        if bad {
            return ProductPredictStep::rejected();
        }

        // Hyperbolic factor: log to the origin tangent space, forecast there,
        // exp back, measure with d_H.
        let tangent_obs = log_map0(ball);
        let (tangent_forecast, _method, _tres, skill_hyper) = self.tangent_core.step(&tangent_obs);
        let predicted_ball = tangent_forecast.as_ref().map(|v| exp_map0(v));
        let residual_hyper = predicted_ball.as_ref().map(|p| poincare_dist(p, ball));

        // Toroidal factor: forecast wrapped per-axis velocity, advance, rewrap.
        let wrapped: Vec<f64> = phases.iter().map(|&p| norm_2pi(p)).collect();
        let (predicted_phases, skill_torus) = match &self.last_phases {
            Some(last) => {
                let velocity: Vec<f64> = wrapped
                    .iter()
                    .zip(last.iter())
                    .map(|(&cur, &prev)| wrap_angle(cur - prev))
                    .collect();
                let (vel_forecast, _m, _r, skill) = self.velocity_core.step(&velocity);
                let predicted = vel_forecast.map(|vf| {
                    last.iter()
                        .zip(vf.iter())
                        .map(|(&prev, &v)| norm_2pi(prev + v))
                        .collect::<Vec<f64>>()
                });
                (predicted, skill)
            }
            None => (None, None),
        };
        let residual_torus = predicted_phases
            .as_ref()
            .map(|p| torus_dist(p, &wrapped, Some(&self.torus_weights)));
        self.last_phases = Some(wrapped);

        let residual_product = match (residual_hyper, residual_torus) {
            (Some(dh), Some(dt)) => {
                Some((self.mix.hyperbolic * dh * dh + self.mix.toroidal * dt * dt).sqrt())
            }
            _ => None,
        };
        let z_hyper = residual_hyper.map(|r| self.welford_hyper.z_then_update(r));
        let z_torus = residual_torus.map(|r| self.welford_torus.z_then_update(r));
        let surprise = match (z_hyper, z_torus) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        ProductPredictStep {
            predicted_ball,
            predicted_phases,
            residual_hyper,
            residual_torus,
            residual_product,
            z_hyper,
            z_torus,
            surprise,
            skill_hyper,
            skill_torus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ar2_recovers_linear_ramp() {
        // y_t = 2*y_{t-1} - y_{t-2} reproduces a linear ramp exactly.
        let series: Vec<f64> = (0..12).map(|i| 1.0 + 0.5 * i as f64).collect();
        let f = ar2_forecast(&series).unwrap();
        assert!((f - 7.0).abs() < 1e-9, "expected 7.0, got {f}");
    }

    #[test]
    fn test_ar2_too_short_or_singular() {
        assert!(ar2_forecast(&[1.0, 2.0, 3.0]).is_none());
        // Constant series makes the design matrix singular.
        assert!(ar2_forecast(&[5.0; 20]).is_none() || {
            let f = ar2_forecast(&[5.0; 20]).unwrap();
            (f - 5.0).abs() < 1e-6
        });
    }

    #[test]
    fn test_welford_z_flags_outlier() {
        let mut w = Welford::default();
        for _ in 0..20 {
            w.z_then_update(1.0);
        }
        // Slight jitter so the variance floor is realistic, then an outlier.
        let mut w2 = Welford::default();
        for i in 0..20 {
            w2.z_then_update(1.0 + 0.01 * ((i % 3) as f64 - 1.0));
        }
        let z = w2.z_then_update(5.0);
        assert!(z > 10.0, "outlier z should be large, got {z}");
    }

    #[test]
    fn test_v1_warmup_and_determinism() {
        let mut a = EuclidPredictor::new(2, None);
        let mut b = EuclidPredictor::new(2, None);
        let first = a.step(&[1.0, 2.0]);
        assert!(first.predicted.is_none());
        assert!(first.residual.is_none());
        b.step(&[1.0, 2.0]);
        for i in 1..30 {
            let obs = [1.0 + 0.1 * i as f64, 2.0 - 0.05 * i as f64];
            let ra = a.step(&obs);
            let rb = b.step(&obs);
            assert_eq!(ra.residual, rb.residual);
            assert_eq!(ra.method, rb.method);
        }
    }

    #[test]
    fn test_v1_beats_persistence_on_ramp() {
        let mut p = EuclidPredictor::new(1, None);
        let mut last = PredictStep::empty();
        for i in 0..40 {
            last = p.step(&[0.5 * i as f64]);
        }
        let skill = last.skill_vs_persistence.unwrap();
        assert!(skill < 1.0, "should beat persistence on a ramp, skill={skill}");
        assert!(last.residual.unwrap() < 1e-6, "AR(2) should nail a ramp");
    }

    #[test]
    fn test_v1_rejects_nan_without_poisoning() {
        let mut p = EuclidPredictor::new(1, None);
        for i in 0..10 {
            p.step(&[i as f64]);
        }
        let bad = p.step(&[f64::NAN]);
        assert!(bad.residual.unwrap().is_nan());
        // Clean observation right after: state unpoisoned, residual finite.
        let good = p.step(&[10.0]);
        assert!(good.residual.unwrap().is_finite());
    }

    #[test]
    fn test_v1_dim_mismatch_rejected() {
        let mut p = EuclidPredictor::new(3, None);
        let s = p.step(&[1.0, 2.0]);
        assert!(s.residual.unwrap().is_nan());
    }

    #[test]
    fn test_v2_seam_crossing_stays_calm() {
        // Phase advancing at constant 0.05 rad/step across the 0/2pi seam:
        // wrapped velocity prediction must keep the residual tiny throughout.
        let mut v2 = ProductPredictor::new(2, 1, None, None);
        let mut max_resid: f64 = 0.0;
        let steps = 200; // crosses the seam at ~step 126
        for i in 0..steps {
            let theta = 0.05 * i as f64; // exceeds 2pi, norm_2pi wraps it
            let s = v2.step(&[0.1, 0.1], &[theta]);
            if i > 10 {
                if let Some(r) = s.residual_torus {
                    max_resid = max_resid.max(r);
                }
            }
        }
        assert!(
            max_resid < 0.1,
            "v2 residual must not tear at the seam, max={max_resid}"
        );
    }

    #[test]
    fn test_v2_surprise_spike_on_phase_jump() {
        let mut v2 = ProductPredictor::new(2, 1, None, None);
        let mut last = None;
        for i in 0..60 {
            let theta = 0.05 * i as f64;
            last = Some(v2.step(&[0.1, 0.1], &[theta]));
        }
        // Injected phase surprise: jump by pi/2.
        let s = v2.step(&[0.1, 0.1], &[0.05 * 60.0 + std::f64::consts::FRAC_PI_2]);
        assert!(s.z_torus.unwrap() > 5.0, "z={:?}", s.z_torus);
        assert!(s.surprise.unwrap() > 5.0);
        let _ = last;
    }

    #[test]
    fn test_v2_rejects_bad_points() {
        let mut v2 = ProductPredictor::new(2, 1, None, None);
        let s = v2.step(&[1.5, 0.0], &[0.0]); // outside the open ball
        assert!(s.residual_product.unwrap().is_nan());
        let s2 = v2.step(&[0.1, f64::NAN], &[0.0]);
        assert!(s2.residual_product.unwrap().is_nan());
    }

    #[test]
    fn test_v2_determinism() {
        let mut a = ProductPredictor::new(2, 2, None, None);
        let mut b = ProductPredictor::new(2, 2, None, None);
        for i in 0..50 {
            let ball = [0.3 * (0.02 * i as f64).sin(), 0.3 * (0.02 * i as f64).cos()];
            let phases = [0.07 * i as f64, 0.03 * i as f64];
            let ra = a.step(&ball, &phases);
            let rb = b.step(&ball, &phases);
            assert_eq!(ra.residual_product, rb.residual_product);
            assert_eq!(ra.surprise, rb.surprise);
        }
    }
}

// ===========================================================================
// TrajectoryMatchGate — monster-server addition (2026-10-19).
//
// Turns the Elle prediction operator's per-step surprise signal into an
// accept/reject verdict on a map-produced candidate continuation.
//
// The monster's native 6D carrier address is [x0,x1,x2,x3 Lorentz, theta_f,
// theta_s phases]. The gate maps the Lorentz factor to the Poincare ball
// (b = x_space / (x0 + 1)) and feeds phases directly, so the predictor sees
// the observation as a point on H^3 x T^2. The prompt's carriers seed the
// forecaster (their surprises are NOT scored); each candidate carrier is
// scored against the one-step forecast. A candidate whose trajectory
// diverges from the prompt's geometric heading produces large z-surprises.
//
// Thresholds are sigma-levels on statistically calibrated z-scores (Welford
// running moments), not hand-tuned distances. Cold gate (too few scored
// steps) passes and says so.
// ===========================================================================

use crate::input_embeddings::NativeMlirEmbedding;

/// Mean surprise over the scored steps above this rejects the candidate.
pub const GATE_Z_MEAN: f64 = 2.5;
/// A scored step is "hot" above this z-surprise.
pub const GATE_Z_HOT: f64 = 3.0;
/// If more than this fraction of scored steps are hot, the candidate is
/// diverging in a sustained way and is rejected. A single hot step in a
/// long candidate is statistically unremarkable and never rejects alone.
pub const GATE_HOT_FRACTION: f64 = 0.20;
/// Minimum scored candidate steps before the gate is allowed to reject.
pub const GATE_MIN_SCORED_STEPS: usize = 3;

/// Verdict on a candidate continuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    /// Candidate continues the prompt's geometric trajectory; serve it.
    Accept,
    /// Candidate diverges from the prompt's trajectory; do not serve.
    Reject,
    /// Too few scored steps to judge; passed without evidence.
    Cold,
}

impl GateVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            GateVerdict::Accept => "accept",
            GateVerdict::Reject => "reject",
            GateVerdict::Cold => "cold",
        }
    }
}

/// Full gate evidence for receipts and tests.
#[derive(Debug, Clone)]
pub struct GateScore {
    pub verdict: GateVerdict,
    pub scored_steps: usize,
    pub max_surprise: Option<f64>,
    pub mean_surprise: Option<f64>,
    pub max_z_hyper: Option<f64>,
    pub max_z_torus: Option<f64>,
    /// Number of scored steps whose surprise exceeded GATE_Z_HOT.
    pub hot_steps: usize,
    /// Worst (largest) factor skill vs persistence over the scored steps.
    /// < 1.0 means the forecaster beat persistence on both factors.
    pub worst_skill: Option<f64>,
    /// True when the forecaster has demonstrated skill on this stream and
    /// its surprise signal may be acted on (Elle discipline: an unproven
    /// forecaster does not gate).
    pub trusted: bool,
}

/// Session-scoped gate. One per session, isolated like the toroidal sessions.
pub struct TrajectoryMatchGate {
    predictor: ProductPredictor,
}

impl TrajectoryMatchGate {
    pub fn new() -> Self {
        Self {
            predictor: ProductPredictor::new(3, 2, None, None),
        }
    }

    /// Map one 6D carrier address to a (ball, phases) observation.
    /// Returns None when the address cannot be represented in the ball.
    fn observe(&mut self, emb: &NativeMlirEmbedding) -> Option<ProductPredictStep> {
        let c = emb.coordinates;
        let denom = c[0] as f64 + 1.0;
        if !denom.is_finite() || denom <= 0.0 {
            return None;
        }
        let ball = [
            c[1] as f64 / denom,
            c[2] as f64 / denom,
            c[3] as f64 / denom,
        ];
        let n2 = ball[0] * ball[0] + ball[1] * ball[1] + ball[2] * ball[2];
        if !n2.is_finite() || n2 >= 1.0 {
            return None;
        }
        let phases = [c[4] as f64, c[5] as f64];
        if !phases.iter().all(|p| p.is_finite()) {
            return None;
        }
        Some(self.predictor.step(&ball, &phases))
    }

    /// Seed with the prompt, score the candidate, return the verdict.
    /// Prompt steps train the forecaster but are never scored.
    pub fn score_candidate(
        &mut self,
        prompt: &[NativeMlirEmbedding],
        candidate: &[NativeMlirEmbedding],
    ) -> GateScore {
        for emb in prompt {
            let _ = self.observe(emb);
        }
        let mut surprises: Vec<f64> = Vec::new();
        let mut skills: Vec<f64> = Vec::new();
        let mut max_zh: Option<f64> = None;
        let mut max_zt: Option<f64> = None;
        for emb in candidate {
            if let Some(step) = self.observe(emb) {
                if let Some(s) = step.surprise {
                    if s.is_finite() {
                        surprises.push(s);
                    }
                }
                if let (Some(sh), Some(st)) = (step.skill_hyper, step.skill_torus) {
                    if sh.is_finite() && st.is_finite() {
                        // Both factors must beat persistence; keep the worse one.
                        skills.push(sh.max(st));
                    }
                }
                if let Some(z) = step.z_hyper {
                    if z.is_finite() {
                        max_zh = Some(max_zh.map_or(z, |m: f64| m.max(z)));
                    }
                }
                if let Some(z) = step.z_torus {
                    if z.is_finite() {
                        max_zt = Some(max_zt.map_or(z, |m: f64| m.max(z)));
                    }
                }
            }
        }
        let scored_steps = surprises.len();
        let hot_steps = surprises.iter().filter(|&&s| s > GATE_Z_HOT).count();
        let max_surprise = surprises.iter().cloned().reduce(f64::max);
        let mean_surprise = if scored_steps > 0 {
            Some(surprises.iter().sum::<f64>() / scored_steps as f64)
        } else {
            None
        };
        let worst_skill = if skills.is_empty() {
            None
        } else {
            Some(skills.iter().sum::<f64>() / skills.len() as f64)
        };
        // Elle discipline: the forecaster must be BEATING persistence on this
        // stream (skill < 1.0) before its surprise is evidence. An unproven
        // forecaster reports Cold, never Reject.
        let trusted = worst_skill.map_or(false, |s| s < 1.0);
        let verdict = if scored_steps < GATE_MIN_SCORED_STEPS || !trusted {
            GateVerdict::Cold
        } else {
            let mean_breach = mean_surprise.map_or(false, |m| m > GATE_Z_MEAN);
            let hot_fraction = hot_steps as f64 / scored_steps.max(1) as f64;
            if mean_breach || hot_fraction > GATE_HOT_FRACTION {
                GateVerdict::Reject
            } else {
                GateVerdict::Accept
            }
        };
        GateScore {
            verdict,
            scored_steps,
            max_surprise,
            mean_surprise,
            max_z_hyper: max_zh,
            max_z_torus: max_zt,
            hot_steps,
            worst_skill,
            trusted,
        }
    }
}

impl Default for TrajectoryMatchGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::input_embeddings::{embed_utf8_text, NativeMlirEmbedding};
    use atoma_immune::BoundaryLexicalTransducer;

    fn embed_all(text: &str) -> Vec<NativeMlirEmbedding> {
        let transducer = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        embed_utf8_text(&transducer, text, false).expect("embed")
    }

    #[test]
    fn gate_accepts_smooth_continuation_of_prompt() {
        let mut gate = TrajectoryMatchGate::new();
        let prompt = embed_all("The capital of France is");
        let candidate = embed_all(" Paris.");
        let score = gate.score_candidate(&prompt, &candidate);
        assert_ne!(
            score.verdict,
            GateVerdict::Reject,
            "smooth continuation should not be rejected: {:?}",
            score
        );
    }

    #[test]
    fn gate_is_cold_on_too_few_steps() {
        let mut gate = TrajectoryMatchGate::new();
        let prompt = embed_all("ab");
        let candidate = embed_all("c");
        let score = gate.score_candidate(&prompt, &candidate);
        assert_eq!(score.verdict, GateVerdict::Cold);
    }

    #[test]
    fn gate_verdict_roundtrip_strings() {
        assert_eq!(GateVerdict::Accept.as_str(), "accept");
        assert_eq!(GateVerdict::Reject.as_str(), "reject");
        assert_eq!(GateVerdict::Cold.as_str(), "cold");
    }
}
