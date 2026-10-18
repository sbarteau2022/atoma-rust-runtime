//! Toroidal Phase Encoder (TwisterPhPE v4.1) with Full-Dimension Kuramoto Spiral Evolution,
//! Amplitude Channel, Unbounded Variable Rapidity Tracking, and Zero Saturation Clamps.
//!
//! - Steps `TorPhaseEncoder::evolve` across all `D/2` 2D coordinate pairs (`64` pairs for `128D`,
//!   `128` pairs for `256D`, `2560` pairs for `5120D`) with zero discarded dimensions (`0` tossed).
//! - Includes an explicit `amplitude: Q32_31` channel on `PhaseState` that accumulates radial
//!   expansion with every phase rotation rather than decaying.
//! - Tracks unbounded hyperbolic rapidity `rapidity` from cumulative rotations and `amplitude`
//!   without saturation clamps.

use super::fixed::Q32_31;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseState {
    pub theta_f: Q32_31,
    pub theta_s: Q32_31,
    pub psi: Q32_31,
    pub amplitude: Q32_31,
}

impl PhaseState {
    pub const ZERO: Self = Self {
        theta_f: Q32_31::ZERO,
        theta_s: Q32_31::ZERO,
        psi: Q32_31::ZERO,
        amplitude: Q32_31::ZERO,
    };
}

/// Complete trajectory state produced by evolving `TorPhaseEncoder` across all `D` coordinates
/// of an input vector (`0` dimensions discarded), with monotonic radial & rapidity expansion per rotation.
#[derive(Debug, Clone, Copy)]
pub struct ToroidalSpiralTrajectory {
    /// Final fixed-point phase state `(theta_f, theta_s, psi, amplitude)`.
    pub phase_state: PhaseState,
    /// Total unwrapped phase arc length in radians accumulated across all steps.
    pub unwrapped_phase_radians: f64,
    /// Number of full `2*pi` rotations completed (`unwrapped_phase_radians / TAU`).
    pub cumulative_rotations: f64,
    /// Monotonically expanded radius `r(w)` driven by rotation count `w` and amplitude (unclamped).
    pub expanded_ball_radius: f64,
    /// Unbounded hyperbolic rapidity `eta` tracked from cumulative rotations and amplitude.
    pub rapidity: f64,
    /// Number of input dimensions preserved across the trajectory.
    pub preserved_dimensions: usize,
    /// 3D helical tangent vector accumulated from `phi(psi)` and full-spectrum pair harmonics.
    pub helical_tangent_h3: [f64; 3],
    /// 2D toroidal phase drive `[fast, slow]` (unclamped).
    pub phase_drive_pi: [f64; 2],
    /// Input L2 energy `sum_i x_i^2`.
    pub input_norm_sq: f64,
    /// Energy captured in the 3D helical tangent modes.
    pub retained_3d_norm_sq: f64,
    /// Energy captured across the 5D `(H^3, T^2)` spiral trajectory.
    pub retained_5d_norm_sq: f64,
}

pub struct TorPhaseEncoder {
    pub omega_f: Q32_31,
    pub omega_s: Q32_31,
    pub k_f: Q32_31,
    pub k_s: Q32_31,
    pub phase_scale: Q32_31,
}

impl Default for TorPhaseEncoder {
    fn default() -> Self {
        Self::new()
    }
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
        let mut fast_acc = 0.0f64;
        let mut slow_acc = 0.0f64;
        let scale = self.phase_scale.to_f64();

        for (i, &b) in data.iter().enumerate() {
            let gray = b ^ (b >> 1);
            let norm_gray = (gray as f64) / 255.0;
            let angle = norm_gray * 2.0 - 1.0;
            // Expanding radial weight per step rather than 1/(1 + 0.5*i) decay
            let expansion_weight = 1.0 + (i as f64) * 0.05;

            fast_acc += angle * expansion_weight * scale;
            slow_acc += angle * 0.5 * expansion_weight * scale;
        }

        (Q32_31::from_f64(fast_acc), Q32_31::from_f64(slow_acc))
    }

    pub fn evolve(&self, current: PhaseState, df: Q32_31, ds: Q32_31) -> PhaseState {
        // Actual Kuramoto Coupling: d_theta = omega + K*sin(diff) + drive
        let diff_sf = current.theta_s - current.theta_f;
        let diff_fs = current.theta_f - current.theta_s;

        let coupling_f = self.k_f * Q32_31::sin(diff_sf);
        let coupling_s = self.k_s * Q32_31::sin(diff_fs);

        let step_f = self.omega_f + coupling_f + df;
        let step_s = self.omega_s + coupling_s + ds;

        let next_f = current.theta_f + step_f;
        let next_s = current.theta_s + step_s;

        let psi = next_f - (next_s * Q32_31::from_f64(2.0));

        // Monotonically accumulate radial amplitude from both phase rotation arc and input drive
        let sf = step_f.to_f64();
        let ss = step_s.to_f64();
        let df_f = df.to_f64();
        let ds_f = ds.to_f64();
        let arc_step = (sf * sf + ss * ss).sqrt() / std::f64::consts::TAU;
        let drive_mod = (df_f * df_f + ds_f * ds_f).sqrt();
        let next_amp_f64 = current.amplitude.to_f64() + arc_step + drive_mod;

        PhaseState {
            theta_f: next_f,
            theta_s: next_s,
            psi,
            amplitude: Q32_31::from_f64(next_amp_f64),
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

    /// Unbounded rotation-driven radial shell expansion:
    /// Every cumulative `2*pi` rotation `w >= 0` expands the radius outward from `r_min`
    /// (`r(w_2) > r(w_1)` for all `w_2 > w_1`), without saturation clamps.
    #[inline]
    pub fn radius_for_rotations(rotations: f64, r_min: f64, r_scale: f64) -> f64 {
        let w = rotations.max(0.0);
        let span = (r_scale - r_min).max(0.15);
        r_min + span * (0.35 * w + 0.65 * w.ln_1p())
    }

    /// Variable hyperbolic rapidity `eta` tracked from cumulative rotations and `PhaseState.amplitude`.
    #[inline]
    pub fn rapidity_for_state(rotations: f64, amplitude: f64, energy_rms: f64) -> f64 {
        let w = rotations.max(0.0);
        let amp = amplitude.max(0.0);
        0.22 + 0.38 * w + 0.28 * amp + 0.20 * energy_rms.ln_1p()
    }

    /// Evolve `TorPhaseEncoder` across all `D` coordinates of `values` (`D/2` 2D RoPE pairs,
    /// e.g. `64` pairs for `128D`, `128` pairs for `256D`, `2560` pairs for `5120D`).
    ///
    /// - **Zero discarded dimensions (`0` tossed)**: every coordinate `values[i]` drives both
    ///   the Kuramoto phase evolution `self.evolve(state, df, ds)` and the 3D helical frame `self.phi(psi)`.
    /// - **Amplitude & radius expand with every rotation**: no saturation clamps.
    pub fn evolve_vector(&self, values: &[f32]) -> ToroidalSpiralTrajectory {
        let d = values.len();
        if d == 0 {
            return ToroidalSpiralTrajectory {
                phase_state: PhaseState::ZERO,
                unwrapped_phase_radians: 0.0,
                cumulative_rotations: 0.0,
                expanded_ball_radius: 0.18,
                rapidity: 0.18,
                preserved_dimensions: 0,
                helical_tangent_h3: [0.0; 3],
                phase_drive_pi: [0.0; 2],
                input_norm_sq: 0.0,
                retained_3d_norm_sq: 0.0,
                retained_5d_norm_sq: 0.0,
            };
        }

        let mut input_norm_sq = 0.0f64;
        for &v in values {
            let vf = v as f64;
            input_norm_sq += vf * vf;
        }
        let rms = (input_norm_sq / (d as f64)).sqrt().max(1e-6);

        let rope_theta = 10_000.0f64;
        let coherence_span = 16.0f64;
        let num_pairs = d.div_ceil(2);

        let mut state = PhaseState::ZERO;
        let mut unwrapped_f = 0.0f64;
        let mut unwrapped_s = 0.0f64;
        let mut unwrapped_arc = 0.0f64;

        let mut raw_modes = [0.0f64; 5];
        let mut basis_norm_sq = [0.0f64; 5];

        for p in 0..num_pairs {
            let x0 = values[2 * p] as f64;
            let x1 = if 2 * p + 1 < d {
                values[2 * p + 1] as f64
            } else {
                0.0
            };

            // Unclamped pair increments scaled relative to RMS
            let df_norm = x0 / (rms * 3.0);
            let ds_norm = x1 / (rms * 3.0);
            let df_q32 = Q32_31::from_f64(df_norm) * self.phase_scale;
            let ds_q32 = Q32_31::from_f64(ds_norm) * self.phase_scale;

            let prev_f = state.theta_f.to_f64();
            let prev_s = state.theta_s.to_f64();
            state = self.evolve(state, df_q32, ds_q32);

            // Track exact unwrapped phase step
            let step_df = self.omega_f.to_f64()
                + self.k_f.to_f64() * (prev_s - prev_f).sin()
                + df_q32.to_f64();
            let step_ds = self.omega_s.to_f64()
                + self.k_s.to_f64() * (prev_f - prev_s).sin()
                + ds_q32.to_f64();
            unwrapped_f += step_df;
            unwrapped_s += step_ds;
            unwrapped_arc += (step_df * step_df + step_ds * step_ds).sqrt();

            // Current rotation count and amplitude at pair `p` expand the helical shell weight monotonically
            let rot_p = unwrapped_arc / std::f64::consts::TAU;
            let shell_r =
                Self::radius_for_rotations(rot_p, 0.22, 0.90) + 0.05 * state.amplitude.to_f64();

            // Evaluate TorPhaseEncoder::phi(psi) on the evolved Kuramoto phase
            let phi_vec = self.phi(state.psi);
            let phi_f64 = [
                phi_vec[0].to_f64(),
                phi_vec[1].to_f64(),
                phi_vec[2].to_f64(),
            ];

            // RoPE positional coherence weight for pair `p`
            let freq_exp = (2 * p) as f64 / (d.max(2) as f64);
            let omega = rope_theta.powf(-freq_exp);
            let w_rope = 1.0 / (1.0 + (coherence_span * omega).powi(2)).sqrt();
            let w_total = w_rope * (0.92 + 0.08 * (shell_r / (1.0 + shell_r)));

            for (sub, &kv) in [x0, x1].iter().enumerate() {
                if 2 * p + sub >= d {
                    break;
                }
                let parity = if sub == 0 { 1.0 } else { -0.61803398875 };
                for r in 0..5 {
                    let harmonic_angle = ((r + 1) as f64) * ((p + 1) as f64) * 1.61803398875
                        + (r as f64) * 0.78539816339;
                    let carrier = if r < 3 {
                        0.96 * harmonic_angle.cos() + 0.04 * phi_f64[r]
                    } else if r == 3 {
                        0.96 * harmonic_angle.cos() + 0.04 * unwrapped_f.cos()
                    } else {
                        0.96 * harmonic_angle.sin() + 0.04 * unwrapped_s.sin()
                    };
                    let b = w_total * (carrier + 0.5 * parity * (harmonic_angle * 0.5).sin());
                    raw_modes[r] += kv * b;
                    basis_norm_sq[r] += b * b;
                }
            }
        }

        for r in 0..5 {
            let bn = basis_norm_sq[r].sqrt().max(1e-12);
            raw_modes[r] /= bn;
        }

        let cumulative_rotations = unwrapped_arc / std::f64::consts::TAU;
        let amp_f64 = state.amplitude.to_f64();
        let rapidity = Self::rapidity_for_state(cumulative_rotations, amp_f64, rms);
        let expanded_ball_radius =
            Self::radius_for_rotations(cumulative_rotations + amp_f64 * 0.25, 0.22, 0.90);

        let z3 = [raw_modes[0], raw_modes[1], raw_modes[2]];
        let z2 = [raw_modes[3], raw_modes[4]];
        let retained_3d_norm_sq = z3[0] * z3[0] + z3[1] * z3[1] + z3[2] * z3[2];
        let retained_5d_norm_sq = retained_3d_norm_sq + z2[0] * z2[0] + z2[1] * z2[1];

        // Unclamped phase drive
        let drive_f = (z2[0] / rms) * 0.65 + unwrapped_f.sin() * 0.35;
        let drive_s = (z2[1] / rms) * 0.65 + unwrapped_s.cos() * 0.35;

        ToroidalSpiralTrajectory {
            phase_state: state,
            unwrapped_phase_radians: unwrapped_arc,
            cumulative_rotations,
            expanded_ball_radius,
            rapidity,
            preserved_dimensions: d,
            helical_tangent_h3: z3,
            phase_drive_pi: [drive_f, drive_s],
            input_norm_sq,
            retained_3d_norm_sq,
            retained_5d_norm_sq,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tor_phase_encoder_expands_radius_and_amplitude_with_every_rotation_and_uses_all_128_dims() {
        let enc = TorPhaseEncoder::new();
        let mut prev_r = 0.0f64;
        let mut prev_amp = 0.0f64;
        let mut prev_eta = 0.0f64;
        for turns in 1..=8 {
            let len = turns * 32; // 16 pairs per turn chunk
            let vec: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.23).cos() * 0.8).collect();
            let traj = enc.evolve_vector(&vec);
            assert!(
                traj.cumulative_rotations > 0.0,
                "rotations must be positive"
            );
            assert!(
                traj.phase_state.amplitude.to_f64() > prev_amp,
                "amplitude channel must expand with every rotation: turn {} amp={} <= prev_amp={}",
                turns,
                traj.phase_state.amplitude.to_f64(),
                prev_amp
            );
            assert!(
                traj.expanded_ball_radius > prev_r,
                "radius must expand with every rotation: turn {} r={} <= prev_r={}",
                turns,
                traj.expanded_ball_radius,
                prev_r
            );
            assert!(
                traj.rapidity > prev_eta,
                "rapidity must expand with every rotation: turn {} eta={} <= prev_eta={}",
                turns,
                traj.rapidity,
                prev_eta
            );
            prev_r = traj.expanded_ball_radius;
            prev_amp = traj.phase_state.amplitude.to_f64();
            prev_eta = traj.rapidity;
        }

        // Verify 128th dimension (index 127) changes the evolved trajectory (0 dims discarded)
        let v_a = vec![0.25f32; 128];
        let mut v_b = vec![0.25f32; 128];
        v_b[127] = -1.75f32;
        let t_a = enc.evolve_vector(&v_a);
        let t_b = enc.evolve_vector(&v_b);
        assert_ne!(t_a.phase_state, t_b.phase_state);
        assert_ne!(t_a.helical_tangent_h3, t_b.helical_tangent_h3);
    }
}
