//! Picard Group Mapping (Γ = PSL(2, ℤ[i]))
//! Maps Lorentz coordinates (including unbounded radii) into the fundamental domain
//! of the Picard group `H³ / PSL(2, ℤ[i])`.

use super::fixed::Q32_31;
use super::lorentz::LorentzVector;

pub struct PicardMapper;

impl PicardMapper {
    pub fn map_to_fundamental_domain(v: &LorentzVector) -> LorentzVector {
        // For moderate radii (`x0 <= 8.0`) where `Q32_31` division has high UHS resolution,
        // execute the fixed-point reduction first and verify hyperboloid membership.
        if v.x[0].to_f64() <= 8.0 {
            let reduced_fp = Self::map_fixed_point(v);
            if reduced_fp.validate().is_ok() {
                return reduced_fp;
            }
        }

        // For unbounded radii (`x0 > 8.0` or when fixed-point UHS `w = 1 / (x0 + x3)` underflows),
        // perform exact upper-half-space PSL(2, Z[i]) reduction in wide f64 precision
        // using cancellation-free `x0 + x3 = (1 + x1^2 + x2^2) / (x0 - x3)` when `x3 < 0`.
        Self::map_unbounded_f64(v)
    }

    fn map_fixed_point(v: &LorentzVector) -> LorentzVector {
        let mut current = *v;

        for _ in 0..32 {
            let denom = current.x[0] + current.x[3];
            if denom.0 <= 0 {
                return Self::map_unbounded_f64(&current);
            }

            let mut z_re = current.x[1].div(denom);
            let mut z_im = current.x[2].div(denom);
            let mut w = Q32_31::ONE.div(denom);

            let gamma_re = Q32_31::from_f64((z_re.0.saturating_add(1 << 30) >> 31) as f64);
            let gamma_im = Q32_31::from_f64((z_im.0.saturating_add(1 << 30) >> 31) as f64);

            let mut moved = false;
            if gamma_re != Q32_31::ZERO || gamma_im != Q32_31::ZERO {
                z_re = z_re - gamma_re;
                z_im = z_im - gamma_im;
                moved = true;
            }

            let tau_norm_sq = (z_re * z_re) + (z_im * z_im) + (w * w);
            if tau_norm_sq < Q32_31::ONE && tau_norm_sq != Q32_31::ZERO {
                let inv = Q32_31::ONE.div(tau_norm_sq);
                z_re = (Q32_31::ZERO - z_re) * inv;
                z_im = z_im * inv;
                w = w * inv;
                current = Self::uhs_to_lorentz(z_re, z_im, w);
                continue;
            }

            if moved {
                current = Self::uhs_to_lorentz(z_re, z_im, w);
            }
            break;
        }
        current
    }

    fn map_unbounded_f64(v: &LorentzVector) -> LorentzVector {
        let x1 = v.x[1].to_f64();
        let x2 = v.x[2].to_f64();
        let x3 = v.x[3].to_f64();
        let x0 = (1.0 + x1 * x1 + x2 * x2 + x3 * x3)
            .sqrt()
            .max(v.x[0].to_f64().max(1.0));

        // Cancellation-free computation of `x0 + x3` on the future hyperboloid sheet:
        let denom = if x3 >= 0.0 {
            (x0 + x3).max(1e-15)
        } else {
            ((1.0 + x1 * x1 + x2 * x2) / (x0 - x3).max(1e-15)).max(1e-15)
        };

        let mut z_re = x1 / denom;
        let mut z_im = x2 / denom;
        let mut w = (1.0 / denom).max(1e-12);

        for _ in 0..128 {
            let n_re = z_re.round();
            let n_im = z_im.round();
            let mut moved = false;
            if n_re != 0.0 || n_im != 0.0 {
                z_re -= n_re;
                z_im -= n_im;
                moved = true;
            }

            let tau_norm_sq = z_re * z_re + z_im * z_im + w * w;
            if tau_norm_sq < 1.0 - 1e-12 && tau_norm_sq > 1e-24 {
                let inv = 1.0 / tau_norm_sq;
                z_re = -z_re * inv;
                z_im *= inv;
                w *= inv;
                continue;
            }

            if !moved {
                break;
            }
        }

        // Final Gaussian-integer strip reduction so z_re, z_im in [-0.5, 0.5]
        z_re -= z_re.round();
        z_im -= z_im.round();

        // If inversion sent `w` high up the cusp (`w > 64.0`), fold `ln(w)` into the
        // compact horoball collar `[1.0, 64.0]` while preserving the exact cusp ray `(z_re, z_im)`.
        let w_eff = if w > 64.0 {
            1.0 + (w - 1.0).ln_1p()
        } else {
            w.max(0.70710678118)
        };

        let z_sq = z_re * z_re + z_im * z_im;
        let w_sq = w_eff * w_eff;
        let two_w = 2.0 * w_eff;

        let rx1 = z_re / w_eff;
        let rx2 = z_im / w_eff;
        let rx3 = (1.0 - z_sq - w_sq) / two_w;
        let rx0 = (1.0 + rx1 * rx1 + rx2 * rx2 + rx3 * rx3).sqrt();

        LorentzVector::from_f64([rx0, rx1, rx2, rx3])
    }

    fn uhs_to_lorentz(z_re: Q32_31, z_im: Q32_31, w: Q32_31) -> LorentzVector {
        let w_safe = if w.0 <= 0 { Q32_31::from_f64(1e-7) } else { w };

        let z_sq = (z_re * z_re) + (z_im * z_im);
        let w_sq = w_safe * w_safe;
        let two_w = w_safe * Q32_31::from_f64(2.0);

        let common_term = Q32_31::ONE + z_sq + w_sq;

        let x0 = common_term.div(two_w);
        let x1 = z_re.div(w_safe);
        let x2 = z_im.div(w_safe);
        let x3 = (Q32_31::ONE - z_sq - w_sq).div(two_w);

        LorentzVector {
            x: [x0, x1, x2, x3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_is_fixed_and_distant_point_stays_farther() {
        let origin = LorentzVector::origin();
        assert_eq!(PicardMapper::map_to_fundamental_domain(&origin), origin);

        let far = LorentzVector::from_f64([2.0, 1.732, 0.0, 0.0]);
        let mapped = PicardMapper::map_to_fundamental_domain(&far);
        assert!(
            mapped.validate().is_ok(),
            "reduced point left the hyperboloid"
        );
        assert!(
            mapped.x[0] > origin.x[0],
            "reduced x0 {} is not farther than the origin",
            mapped.x[0].to_f64()
        );
        assert_eq!(PicardMapper::map_to_fundamental_domain(&mapped), mapped);
    }

    #[test]
    fn picard_reduction_handles_unbounded_radii_without_leaving_hyperboloid() {
        for &radius in &[25.0f64, 250.0, 5_000.0, 50_000.0] {
            let x1 = radius * 0.7;
            let x2 = -radius * 0.5;
            let x3 = -radius * 0.4;
            let x0 = (1.0 + x1 * x1 + x2 * x2 + x3 * x3).sqrt();
            let unbounded = LorentzVector::from_f64([x0, x1, x2, x3]);
            let reduced = PicardMapper::map_to_fundamental_domain(&unbounded);
            assert!(
                reduced.validate().is_ok(),
                "Picard reduction at radius {radius} failed hyperboloid check: {:?}",
                reduced
            );
            assert!(
                reduced.x[0].to_f64() >= 1.0,
                "Picard-reduced x0 must be >= 1.0"
            );
        }
    }
}
