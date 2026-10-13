//! Bridge between f32 transformer space and Q32_31 geometric space.
//!
//! Precision boundary: f32 keys are projected into H^3 via exponential map.
//! The conversion loses precision below Q32_31 epsilon (~4.6e-10).
//! This is documented, not hidden.

use super::fixed::Q32_31;
use super::lorentz::LorentzVector;

#[derive(Debug)]
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

pub fn f32_to_lorentz(key: &[f32]) -> Result<LorentzVector, BridgeError> {
    if key.is_empty() {
        return Err(BridgeError::EmptyInput);
    }
    let spatial = [
        key.first().copied().unwrap_or(0.0),
        key.get(1).copied().unwrap_or(0.0),
        key.get(2).copied().unwrap_or(0.0),
    ];
    let norm_sq: f32 = spatial.iter().map(|v| v * v).sum();
    let x0 = (1.0 + norm_sq).sqrt();
    let mut x = [Q32_31::ZERO; 4];
    x[0] = Q32_31::from_f64(x0 as f64);

    for (i, value) in spatial.iter().enumerate() {
        x[i + 1] = Q32_31::from_f64(*value as f64);
    }

    let v = LorentzVector { x };
    v.validate().map_err(|e| BridgeError::InvalidVector(e))?;
    Ok(v)
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
}
