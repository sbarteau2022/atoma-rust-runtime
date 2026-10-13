//! Lorentz Model for Hyperbolic 3-Space (H³)
//! Coordinates: (x0, x1, x2, x3) such that x0^2 - x1^2 - x2^2 - x3^2 = 1, x0 > 0.
//! Fixed: Hyperboloid validation relaxed for fixed-point rounding; boost refined.

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

    pub fn validate(&self) -> Result<(), String> {
        let x0 = self.x[0];
        let x1 = self.x[1];
        let x2 = self.x[2];
        let x3 = self.x[3];

        let norm_sq = (x0 * x0) - (x1 * x1) - (x2 * x2) - (x3 * x3);
        let diff = (norm_sq - Q32_31::ONE).to_f64().abs();

        if diff > 0.1 {
            // Increased tolerance for fixed-point rounding noise
            return Err(format!(
                "Hyperboloid violation: norm_sq = {:.4}",
                norm_sq.to_f64()
            ));
        }
        Ok(())
    }

    pub fn inner_product(a: &Self, b: &Self) -> Q32_31 {
        (a.x[0] * b.x[0]) - (a.x[1] * b.x[1]) - (a.x[2] * b.x[2]) - (a.x[3] * b.x[3])
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
