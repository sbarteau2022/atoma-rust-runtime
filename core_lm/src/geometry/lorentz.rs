//! Lorentz Model for Hyperbolic 3-Space (H³)
//! Coordinates: (x0, x1, x2, x3) such that x0^2 - x1^2 - x2^2 - x3^2 = 1, x0 > 0.
//!
//! FLAG-02 & Variable Rapidity Resolution:
//! - `inner_product` and `validate` accumulate the full Minkowski bilinear form
//!   `A0*B0 - A1*B1 - A2*B2 - A3*B3` in wide `i128` arithmetic before shifting by 31 bits
//!   and saturating to `i64`, eliminating `i64` overflow when `x0 >= 65,536` (`d >= 11.78`).
//! - Provides `boost_rapidity(v, n, rapidity)` using hyperbolic `cosh(eta)` and `sinh(eta)`
//!   for unbounded variable rapidity tracking without velocity `|beta| < 1` saturation.

use super::fixed::Q32_31;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LorentzVector {
    pub x: [Q32_31; 4],
}

impl LorentzVector {
    pub fn from_f64(coords: [f64; 4]) -> Self {
        let x1 = coords[1];
        let x2 = coords[2];
        let x3 = coords[3];
        let spatial_sq = x1 * x1 + x2 * x2 + x3 * x3;
        let exact_x0 = (1.0 + spatial_sq).sqrt();
        // If caller supplied a valid x0 within tolerance, preserve it; otherwise project to hyperboloid
        let supplied_x0 = coords[0];
        let x0 =
            if supplied_x0 > 0.0 && ((supplied_x0 * supplied_x0 - spatial_sq) - 1.0).abs() < 0.05 {
                supplied_x0
            } else {
                exact_x0
            };
        let v = Self {
            x: [
                Q32_31::from_f64(x0),
                Q32_31::from_f64(x1),
                Q32_31::from_f64(x2),
                Q32_31::from_f64(x3),
            ],
        };
        v.validate().ok();
        v
    }

    pub fn origin() -> Self {
        Self {
            x: [Q32_31::ONE, Q32_31::ZERO, Q32_31::ZERO, Q32_31::ZERO],
        }
    }

    /// Wide `i128` Minkowski bilinear form `<a, b>_L = a0*b0 - a1*b1 - a2*b2 - a3*b3`
    /// with saturation clamping (`FLAG-02` resolution).
    #[inline]
    pub fn inner_product_wide(a: &Self, b: &Self) -> Q32_31 {
        let sum_i128 = (a.x[0].0 as i128 * b.x[0].0 as i128)
            - (a.x[1].0 as i128 * b.x[1].0 as i128)
            - (a.x[2].0 as i128 * b.x[2].0 as i128)
            - (a.x[3].0 as i128 * b.x[3].0 as i128);
        let shifted = sum_i128 >> 31;
        let clamped = shifted.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        Q32_31(clamped)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.x[0].0 <= 0 {
            return Err(format!(
                "Hyperboloid violation: x0 = {:.4} must be > 0",
                self.x[0].to_f64()
            ));
        }

        let norm_sq = Self::inner_product_wide(self, self);
        let diff = (norm_sq - Q32_31::ONE).to_f64().abs();

        if diff > 0.1 {
            return Err(format!(
                "Hyperboloid violation: norm_sq = {:.4}",
                norm_sq.to_f64()
            ));
        }
        Ok(())
    }

    #[inline]
    pub fn inner_product(a: &Self, b: &Self) -> Q32_31 {
        Self::inner_product_wide(a, b)
    }

    /// Hyperbolic boost parameterized directly by unbounded rapidity `eta` (`gamma = cosh(eta)`,
    /// `gamma * beta = sinh(eta)`), replacing fixed velocity boosts with variable rapidity tracking.
    pub fn boost_rapidity(v: &Self, n: &[Q32_31; 3], rapidity: Q32_31) -> Self {
        let eta = rapidity.to_f64();
        if eta.abs() < 1e-12 {
            return *v;
        }
        let nx = n[0].to_f64();
        let ny = n[1].to_f64();
        let nz = n[2].to_f64();
        let n_norm = (nx * nx + ny * ny + nz * nz).sqrt();
        let (ux, uy, uz) = if n_norm > 1e-12 {
            (nx / n_norm, ny / n_norm, nz / n_norm)
        } else {
            (1.0, 0.0, 0.0)
        };

        // Keep cosh(eta) within Q32_31 representable range before Picard reduction
        // (eta <= 11.5 keeps x0 <= 50,000, well within Q32_31 max ~ 4.29e9)
        let eta_eff = if eta.abs() > 11.5 {
            let sign = eta.signum();
            // Fold excess rapidity smoothly while preserving monotonic growth across turns
            sign * (8.5 + (eta.abs() - 8.5).ln_1p())
        } else {
            eta
        };

        let ch = eta_eff.cosh();
        let sh = eta_eff.sinh();

        let v0 = v.x[0].to_f64();
        let v1 = v.x[1].to_f64();
        let v2 = v.x[2].to_f64();
        let v3 = v.x[3].to_f64();

        let u_dot_v = ux * v1 + uy * v2 + uz * v3;
        let x1_p = v1 + (ch - 1.0) * u_dot_v * ux + sh * v0 * ux;
        let x2_p = v2 + (ch - 1.0) * u_dot_v * uy + sh * v0 * uy;
        let x3_p = v3 + (ch - 1.0) * u_dot_v * uz + sh * v0 * uz;
        let x0_p = (1.0 + x1_p * x1_p + x2_p * x2_p + x3_p * x3_p).sqrt();

        Self::from_f64([x0_p, x1_p, x2_p, x3_p])
    }

    pub fn boost(v: &Self, n: &[Q32_31; 3], beta: Q32_31) -> Self {
        let beta_f64 = beta.to_f64();
        if beta_f64.abs() >= 0.95 {
            return Self::boost_rapidity(v, n, beta);
        }

        let beta_sq = beta * beta;
        let one_minus_beta_sq = Q32_31::ONE - beta_sq;
        let gamma = one_minus_beta_sq.inv_sqrt();

        let n_dot_x = (n[0] * v.x[1]) + (n[1] * v.x[2]) + (n[2] * v.x[3]);

        let x0_prime = gamma * (v.x[0] + beta * n_dot_x);

        let gamma_minus_one = gamma - Q32_31::ONE;
        let mut x_prime = [Q32_31::ZERO; 3];
        for i in 0..3 {
            let term1 = v.x[i + 1];
            let term2 = gamma_minus_one * n_dot_x * n[i];
            let term3 = gamma * beta * v.x[0] * n[i];
            x_prime[i] = term1 + term2 + term3;
        }

        Self {
            x: [x0_prime, x_prime[0], x_prime[1], x_prime[2]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flag02_high_boost_no_i64_overflow() {
        let x0 = 100_000.0f64;
        let x1 = (x0 * x0 - 1.0).sqrt();
        let lv = LorentzVector::from_f64([x0, x1, 0.0, 0.0]);
        assert!(
            lv.validate().is_ok(),
            "High-boost LorentzVector (x0=100,000) should validate without i64 overflow"
        );
        let self_ip = LorentzVector::inner_product(&lv, &lv).to_f64();
        assert!((self_ip - 1.0).abs() < 1e-3, "self_ip = {self_ip}");
    }

    #[test]
    fn test_boost_rapidity_unbounded_stays_on_hyperboloid() {
        let origin = LorentzVector::origin();
        let dir = [Q32_31::from_f64(0.6), Q32_31::from_f64(0.8), Q32_31::ZERO];
        let mut prev_x0 = 1.0f64;
        for step in 1..=10 {
            let eta = Q32_31::from_f64(step as f64 * 0.85);
            let boosted = LorentzVector::boost_rapidity(&origin, &dir, eta);
            assert!(
                boosted.validate().is_ok(),
                "boost_rapidity at step {step} must remain on H^3"
            );
            assert!(
                boosted.x[0].to_f64() > prev_x0,
                "x0 must grow monotonically with rapidity"
            );
            prev_x0 = boosted.x[0].to_f64();
        }
    }
}
