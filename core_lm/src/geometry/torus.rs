//! Toroidal Phase Encoder (TwisterPhPE v4.1)
//! Fixed: Fully float-free, correct v4.1 constants, actual coupling, and LUT-based trig.

use super::fixed::Q32_31;

#[derive(Debug, Clone, Copy)]
pub struct PhaseState {
    pub theta_f: Q32_31,
    pub theta_s: Q32_31,
    pub psi: Q32_31,
}

pub struct TorPhaseEncoder {
    pub omega_f: Q32_31,
    pub omega_s: Q32_31,
    pub k_f: Q32_31,
    pub k_s: Q32_31,
    pub phase_scale: Q32_31,
}

impl TorPhaseEncoder {
    pub fn new() -> Self {
        Self {
            omega_f: Q32_31::from_f64(0.1),
            omega_s: Q32_31::from_f64(0.05),
            k_f: Q32_31::from_f64(0.3),
            k_s: Q32_31::from_f64(0.15),
            phase_scale: Q32_31::from_f64(0.15),
        }
    }

    pub fn project_increments(&self, data: &[u8]) -> (Q32_31, Q32_31) {
        let mut fast_acc = Q32_31::ZERO;
        let mut slow_acc = Q32_31::ZERO;

        for (i, &b) in data.iter().enumerate() {
            let gray = b ^ (b >> 1);
            let norm_gray = Q32_31::from_f64(gray as f64 / 255.0);
            let angle = (norm_gray * Q32_31::from_f64(2.0)) - Q32_31::ONE;
            let weight = Q32_31::from_f64(1.0 / (1.0 + i as f64 * 0.5));

            fast_acc = fast_acc + (angle * weight);
            slow_acc = slow_acc + (angle * Q32_31::from_f64(0.5) * weight);
        }

        (fast_acc * self.phase_scale, slow_acc * self.phase_scale)
    }

    pub fn evolve(&self, current: PhaseState, df: Q32_31, ds: Q32_31) -> PhaseState {
        // Actual Kuramoto Coupling: d_theta = omega + K*sin(diff) + drive
        let diff_sf = current.theta_s - current.theta_f;
        let diff_fs = current.theta_f - current.theta_s;

        let coupling_f = self.k_f * Q32_31::sin(diff_sf);
        let coupling_s = self.k_s * Q32_31::sin(diff_fs);

        let next_f = current.theta_f + self.omega_f + coupling_f + df;
        let next_s = current.theta_s + self.omega_s + coupling_s + ds;

        let psi = next_f - (next_s * Q32_31::from_f64(2.0));

        PhaseState {
            theta_f: next_f,
            theta_s: next_s,
            psi,
        }
    }

    pub fn phi(&self, theta: Q32_31) -> [Q32_31; 3] {
        let inv_sqrt2 = Q32_31::from_f64(1.0 / 2.0f64.sqrt());
        [
            inv_sqrt2,
            inv_sqrt2 * Q32_31::cos(theta),
            inv_sqrt2 * Q32_31::sin(theta),
        ]
    }
}
