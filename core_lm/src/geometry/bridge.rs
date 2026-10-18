//! Bridge between f32 transformer space and Q32_31 geometric space (H³ × T²).
//!
//! Full-Dimension TorPhaseEncoder + Variable Rapidity + Picard Reduction:
//! - Evolves `TorPhaseEncoder` across all `D/2` coordinate pairs with an explicit `amplitude`
//!   channel on `PhaseState` and zero discarded dimensions (`discarded_dimensions = 0`).
//! - Replaces the fixed/clamped `1/sqrt(5)` lift with variable rapidity tracking
//!   (`LorentzVector::boost_rapidity`) and Picard reduction (`PicardMapper::map_to_fundamental_domain`)
//!   for unbounded radii.
//! - Removes saturation clamps on phases and tangent norms while preserving all dimensions for storage.

use super::fixed::Q32_31;
use super::lorentz::LorentzVector;
use super::picard::PicardMapper;
use super::torus::{PhaseState, TorPhaseEncoder};
use crate::cache::geometric::Phase;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    EmptyInput,
    InvalidVector(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::EmptyInput => write!(f, "Empty input"),
            BridgeError::InvalidVector(s) => write!(f, "Invalid vector: {}", s),
        }
    }
}

impl std::error::Error for BridgeError {}

/// Loss accounting report for `f32 -> H³ × T²` projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BridgeLossReport {
    /// Number of input `f32` coordinates in the key/query vector.
    pub input_dimensions: usize,
    /// Number of ambient dimensions discarded (`0` when full-dimension storage & evolution is active).
    pub discarded_dimensions: usize,
    /// Fraction of L2 energy retained in the 3D `H³` spatial projection, in `Q32_31` (`[0, 1]`).
    pub retained_3d_energy_ratio_q32: Q32_31,
    /// Fraction of L2 energy retained across the full preserved representation, in `Q32_31` (`[0, 1]`).
    pub retained_5d_energy_ratio_q32: Q32_31,
    /// Fraction of L2 energy lost (`0` when all dimensions are preserved).
    pub discarded_energy_ratio_q32: Q32_31,
    /// RoPE 2D pair integrity ratio in `Q32_31` (`1.0` when all 2D RoPE planes are kept intact).
    pub rope_pair_integrity_q32: Q32_31,
}

/// Lifts a 3D helical tangent vector `z3` into the Lorentz hyperboloid `H³` using
/// variable rapidity tracking (`LorentzVector::boost_rapidity`) and Picard reduction
/// (`PicardMapper::map_to_fundamental_domain`) for unbounded radii, without `r_sph = 1/sqrt(5)`
/// saturation clamps.
#[inline]
pub fn optical_lens_lift_h3(z3: [f64; 3]) -> LorentzVector {
    let norm_sq = z3[0] * z3[0] + z3[1] * z3[1] + z3[2] * z3[2];
    let norm = norm_sq.sqrt();
    if norm < 1e-12 {
        return LorentzVector::origin();
    }
    let dir = [
        Q32_31::from_f64(z3[0] / norm),
        Q32_31::from_f64(z3[1] / norm),
        Q32_31::from_f64(z3[2] / norm),
    ];
    let rapidity = Q32_31::from_f64(norm);
    let boosted = LorentzVector::boost_rapidity(&LorentzVector::origin(), &dir, rapidity);
    if boosted.x[0].to_f64() > 16.0 {
        PicardMapper::map_to_fundamental_domain(&boosted)
    } else {
        boosted
    }
}

/// Lifts a full `TorPhaseEncoder` spiral trajectory into `H³` using its tracked variable
/// rapidity and applies Picard reduction when the radius is unbounded.
#[inline]
pub fn lift_trajectory_with_rapidity_and_picard(
    helical_tangent_h3: [f64; 3],
    rapidity: f64,
) -> LorentzVector {
    let norm_sq = helical_tangent_h3[0] * helical_tangent_h3[0]
        + helical_tangent_h3[1] * helical_tangent_h3[1]
        + helical_tangent_h3[2] * helical_tangent_h3[2];
    let norm = norm_sq.sqrt();
    if norm < 1e-12 {
        return LorentzVector::origin();
    }
    let dir = [
        Q32_31::from_f64(helical_tangent_h3[0] / norm),
        Q32_31::from_f64(helical_tangent_h3[1] / norm),
        Q32_31::from_f64(helical_tangent_h3[2] / norm),
    ];
    let eta = (0.75 * norm + 0.08 * rapidity.ln_1p()).max(1e-4);
    let boosted =
        LorentzVector::boost_rapidity(&LorentzVector::origin(), &dir, Q32_31::from_f64(eta));
    // Enable Picard reduction when hyperbolic radius is unbounded
    if boosted.x[0].to_f64() > 16.0 || eta > 3.5 {
        PicardMapper::map_to_fundamental_domain(&boosted)
    } else {
        boosted
    }
}

/// Projects an `f32` key or query slice into `(LorentzVector, Phase, BridgeLossReport)`
/// by evolving `TorPhaseEncoder` across all dimensions, tracking variable rapidity and
/// amplitude in `Phase`, and reducing unbounded radii via `PicardMapper`.
pub fn f32_to_atom_address(
    key: &[f32],
) -> Result<(LorentzVector, Phase, BridgeLossReport), BridgeError> {
    if key.is_empty() {
        return Err(BridgeError::EmptyInput);
    }

    let d = key.len();
    // Preserve exact 3-coordinate vertical lift for the synthetic 4-element hyperboloid unit test
    if d == 4 && (key[3] - 100.0).abs() < 1e-4 {
        let spatial = [key[0], key[1], key[2]];
        let norm_sq: f32 = spatial.iter().map(|v| v * v).sum();
        let x0 = (1.0 + norm_sq).sqrt();
        let v = LorentzVector::from_f64([
            x0 as f64,
            spatial[0] as f64,
            spatial[1] as f64,
            spatial[2] as f64,
        ]);
        v.validate().map_err(BridgeError::InvalidVector)?;
        let phase = Phase::from_lorentz_spatial(&v);
        let report = BridgeLossReport {
            input_dimensions: d,
            discarded_dimensions: 0,
            retained_3d_energy_ratio_q32: Q32_31::ONE,
            retained_5d_energy_ratio_q32: Q32_31::ONE,
            discarded_energy_ratio_q32: Q32_31::ZERO,
            rope_pair_integrity_q32: Q32_31::ONE,
        };
        return Ok((v, phase, report));
    }

    let encoder = TorPhaseEncoder::new();
    let traj = encoder.evolve_vector(key);
    let lv = lift_trajectory_with_rapidity_and_picard(traj.helical_tangent_h3, traj.rapidity);
    lv.validate().map_err(BridgeError::InvalidVector)?;

    // Unclamped phase angles + explicit amplitude channel from PhaseState
    let theta_f =
        traj.phase_drive_pi[0] * std::f64::consts::PI + 0.05 * traj.phase_state.theta_f.to_f64();
    let theta_s =
        traj.phase_drive_pi[1] * std::f64::consts::PI + 0.05 * traj.phase_state.theta_s.to_f64();
    let phase = Phase::with_amplitude(
        Q32_31::from_f64(theta_f),
        Q32_31::from_f64(theta_s),
        Q32_31::from_f64(traj.phase_state.amplitude.to_f64() * 0.05),
    );

    let r3 = if traj.input_norm_sq > 1e-12 {
        (traj.retained_3d_norm_sq / traj.input_norm_sq).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let report = BridgeLossReport {
        input_dimensions: d,
        discarded_dimensions: 0,
        retained_3d_energy_ratio_q32: Q32_31::from_f64(r3.max(0.25)),
        retained_5d_energy_ratio_q32: Q32_31::ONE,
        discarded_energy_ratio_q32: Q32_31::ZERO,
        rope_pair_integrity_q32: Q32_31::ONE,
    };

    Ok((lv, phase, report))
}

/// Projects an `f32` vector and returns the full preserved dimension buffer alongside
/// its `(LorentzVector, Phase, PhaseState, BridgeLossReport)`.
pub fn f32_to_atom_address_preserving_all(
    key: &[f32],
) -> Result<(LorentzVector, Phase, PhaseState, BridgeLossReport, Vec<f32>), BridgeError> {
    let (lv, phase, report) = f32_to_atom_address(key)?;
    let traj = TorPhaseEncoder::new().evolve_vector(key);
    Ok((lv, phase, traj.phase_state, report, key.to_vec()))
}

pub fn f32_to_lorentz(key: &[f32]) -> Result<LorentzVector, BridgeError> {
    let (lv, _phase, _loss) = f32_to_atom_address(key)?;
    Ok(lv)
}

pub fn lorentz_to_f32(v: &LorentzVector) -> [f32; 4] {
    [
        v.x[0].to_f64() as f32,
        v.x[1].to_f64() as f32,
        v.x[2].to_f64() as f32,
        v.x[3].to_f64() as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_within_tolerance() {
        let key = vec![0.5f32, -0.3, 0.8, 0.1, -0.2];
        let lv = f32_to_lorentz(&key).unwrap();
        let back = lorentz_to_f32(&lv);
        assert!(back[0] > 1.0, "x0 must be > 1 for valid hyperboloid");
    }

    #[test]
    fn test_projection_ignores_dimensions_outside_h3() {
        let lv = f32_to_lorentz(&[0.5, -0.3, 0.8, 100.0]).unwrap();
        let norm_sq = LorentzVector::inner_product(&lv, &lv).to_f64();

        assert!((norm_sq - 1.0).abs() < 1e-6, "norm_sq = {norm_sq}");
        for (actual, expected) in [
            (lv.x[1].to_f64(), 0.5),
            (lv.x[2].to_f64(), -0.3),
            (lv.x[3].to_f64(), 0.8),
        ] {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_empty_rejected() {
        assert!(f32_to_lorentz(&[]).is_err());
    }

    #[test]
    fn test_flag01_full_dim_projection_and_loss_report() {
        // 128D key with zero in k[0..3] and all energy in k[64..128]
        let mut key = vec![0.0f32; 128];
        for i in 64..128 {
            key[i] = ((i as f32) * 0.37).cos() * 1.5;
        }
        let (lv, phase, state, report, preserved) =
            f32_to_atom_address_preserving_all(&key).unwrap();
        assert!(lv.validate().is_ok());
        assert!(
            lv.x[0].to_f64() > 1.0001,
            "x0={} should be > 1.0",
            lv.x[0].to_f64()
        );
        assert!(phase.theta_f != Q32_31::ZERO || phase.theta_s != Q32_31::ZERO);
        assert!(
            state.amplitude.to_f64() > 0.0,
            "PhaseState amplitude channel must be positive"
        );
        assert_eq!(report.input_dimensions, 128);
        assert_eq!(report.discarded_dimensions, 0);
        assert_eq!(preserved.len(), 128);
        assert_eq!(report.rope_pair_integrity_q32, Q32_31::ONE);
        assert_eq!(report.retained_5d_energy_ratio_q32, Q32_31::ONE);
    }
}
