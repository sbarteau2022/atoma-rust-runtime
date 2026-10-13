//! lock.rs — Kuramoto ODE stepper with zero-slip verification.
//!
//! Bit-exact Q32.31 fixed-point. No floats in the ODE loop.
//! (n,m,ψ*) invariants extracted at lock.

const FRAC_BITS: u32 = 31;
const PI: i64 = 1 << FRAC_BITS;

// Corrected CORDIC constants (pi-unit Q32.31)
const CORDIC_GAIN: i64 = 1304065748; // 0.6072529350088814 * 2^31
const ATAN_TABLE: [i64; 32] = [
    536870912, 316933406, 167458907, 85004756, 42667331,
    21354465, 10679838, 5340245, 2670163, 1335087,
    667544, 333772, 166886, 83443, 41722,
    20861, 10430, 5215, 2608, 1304,
    652, 326, 163, 81, 41,
    20, 10, 5, 3, 1, 1, 0
];

/// Bit-exact Q32.31 CORDIC sine. theta in pi-units (1<<31 = π).
/// FIXED: pi-unit table, corrected gain, sign bug removed.
pub fn cordic_sin(theta: i64) -> i64 {
    let mut z = (theta as i32) as i64; // hardware torus wrap
    if z == 0 {
        return 0;
    }
    let pi_half = 1i64 << 30;
    let pi = 1i64 << 31;
    let sign = 1; // NOTE: sign stays 1, fold preserves sin
    if z > pi_half {
        z = pi - z;
    } else if z < -pi_half {
        z = -pi - z;
        // sign = -1 REMOVED (was bug: sin(-π-z) = sin(z))
    }
    let mut x = CORDIC_GAIN;
    let mut y = 0i64;
    for i in 0..32 {
        let x_shift = x >> i;
        let y_shift = y >> i;
        if z >= 0 {
            x = x.wrapping_sub(y_shift);
            y = y.wrapping_add(x_shift);
            z = z.wrapping_sub(ATAN_TABLE[i]);
        } else {
            x = x.wrapping_add(y_shift);
            y = y.wrapping_sub(x_shift);
            z = z.wrapping_add(ATAN_TABLE[i]);
        }
    }
    y * sign
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologicalInvariant {
    pub n: u8,
    pub m: u8,
    pub psi_star: i64,
}

/// Extract (n,m,ψ*) from locked state.
/// n,m: 12-sector quantization of theta_f, theta_s.
/// psi_star: theta_f - 2*theta_s (Q32.31).
pub fn extract_invariant(theta_f: i64, theta_s: i64) -> TopologicalInvariant {
    let shifted_f = (theta_f as i32 as i64).wrapping_add(PI) as u64;
    let shifted_s = (theta_s as i32 as i64).wrapping_add(PI) as u64;
    let n = ((shifted_f.wrapping_mul(12)) >> (FRAC_BITS + 1)) as u8 % 12;
    let m = ((shifted_s.wrapping_mul(12)) >> (FRAC_BITS + 1)) as u8 % 12;
    let psi_star = (theta_f.wrapping_sub(theta_s.wrapping_mul(2)) as i32) as i64;
    TopologicalInvariant { n, m, psi_star }
}

/// Inject surprise (phase error) to break lock.
/// theta: mutable slice, theta_new: new phases, alpha: Q32.31 learning rate.
pub fn inject_surprise(theta: &mut [i64], theta_new: &[i64], alpha: i64) {
    for (t, tn) in theta.iter_mut().zip(theta_new.iter()) {
        let phase_error = (tn.wrapping_sub(*t) as i32) as i64;
        let perturbation = (alpha.wrapping_mul(phase_error)) >> FRAC_BITS;
        *t = (t.wrapping_add(perturbation) as i32) as i64;
    }
}

/// CPU fallback Kuramoto step (for testing; Metal does the real work).
pub fn kuramoto_step_cpu(
    theta: &mut [i64],
    omega: &[i64],
    dt: i64,
    k_coupling: i64,
) -> i64 {
    let n = theta.len();
    let mut max_slip: i64 = 0;
    let mut new_theta = theta.to_vec();
    for i in 0..n {
        let mut sum = 0i64;
        for j in 0..n {
            if i == j { continue; }
            let diff = (theta[j].wrapping_sub(theta[i]) as i32) as i64;
            sum = sum.wrapping_add(cordic_sin(diff));
        }
        let avg = sum / n as i64;
        let interaction = (k_coupling.wrapping_mul(avg)) >> FRAC_BITS;
        let deriv = omega[i].wrapping_add(interaction);
        let delta = (deriv.wrapping_mul(dt)) >> FRAC_BITS;
        let ad = delta.wrapping_abs();
        if ad > max_slip { max_slip = ad; }
        new_theta[i] = (theta[i].wrapping_add(delta) as i32) as i64;
    }
    theta.copy_from_slice(&new_theta);
    max_slip
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cordic_zero() { assert_eq!(cordic_sin(0), 0); }
    #[test]
    fn sectors_cover_12() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..10000 {
            let t = (((i as f64 / 10000.0) * 2.0 - 1.0) * ((1i64 << 31) as f64)) as i64;
            let inv = extract_invariant(t, 0);
            seen.insert(inv.n);
        }
        assert_eq!(seen.len(), 12);
    }
}
