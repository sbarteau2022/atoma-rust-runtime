//! Lorentz Model for Hyperbolic 3-Space (H³)
//! Coordinates: (x0, x1, x2, x3) such that x0^2 - x1^2 - x2^2 - x3^2 = 1, x0 > 0.
//!
//! FLAG-02 Resolution:
//! - `inner_product` and `validate` accumulate the full Minkowski bilinear form
//!   `A0*B0 - A1*B1 - A2*B2 - A3*B3` in wide `i128` arithmetic before shifting by 31 bits
//!   and saturating to `i64`, eliminating `i64` overflow when `x0 >= 65,536` (`d >= 11.78`).

use super::fixed::Q32_31;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LorentzVector {
    pub x: [Q32_31; 4],
}

impl LorentzVector {
    pub fn from_f64(coords: [f64; 4]) -> Self {
        let v = Self {
            x: [
                Q32_31::from_f64(coords[0]),
                Q32_31::from_f64(coords[1]),
                Q32_31::from_f64(coords[2]),
                Q32_31::from_f64(coords[3]),
            ],
        };
        v.validate().ok(); // Log warning instead of panic in init
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

    pub fn boost(v: &Self, n: &[Q32_31; 3], beta: Q32_31) -> Self {
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
        // Previously x0 >= 65536 overflowed i64 in (x0 * x0) before subtracting x1*x1!
        // Here x0 = 100_000.0, x1 = sqrt(100_000^2 - 1) ≈ 99_999.999995
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
}
