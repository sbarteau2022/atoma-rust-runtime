//! Picard Group Mapping (Γ = PSL(2, ℤ[i]))
//! Maps Lorentz coordinates into the fundamental domain of the Picard group.

use super::fixed::Q32_31;
use super::lorentz::LorentzVector;

pub struct PicardMapper;

impl PicardMapper {
    pub fn map_to_fundamental_domain(v: &LorentzVector) -> LorentzVector {
        let mut current = *v;

        for _ in 0..32 {
            let denom = current.x[0] + current.x[3];
            // x0 + x3 is positive on the future sheet. Zero is a fixed-point
            // singularity, not a point of H³.
            if denom == Q32_31::ZERO {
                return current;
            }

            let mut z_re = current.x[1].div(denom);
            let mut z_im = current.x[2].div(denom);
            let mut w = Q32_31::ONE.div(denom);

            let gamma_re = Q32_31::from_f64(((z_re.0 + (1 << 30)) >> 31) as f64);
            let gamma_im = Q32_31::from_f64(((z_im.0 + (1 << 30)) >> 31) as f64);

            // round(z) == 0 means z is already inside the fundamental strip.
            // Rewriting that point still jitters the Lorentz coordinates by 1 ulp.
            let mut moved = false;
            if gamma_re != Q32_31::ZERO || gamma_im != Q32_31::ZERO {
                z_re = z_re - gamma_re;
                z_im = z_im - gamma_im;
                moved = true;
            }

            // τ ↦ -1/τ uses |z|² + w² of the *translated* point. Inverting the
            // pre-translation z, or dividing by |z|² alone, leaves the hyperboloid.
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

    fn uhs_to_lorentz(z_re: Q32_31, z_im: Q32_31, w: Q32_31) -> LorentzVector {
        // Underflow protection: w cannot be zero.
        let w_safe = if w == Q32_31::ZERO {
            Q32_31::from_f64(1e-7)
        } else {
            w
        };

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
}
