//! Autonomous Monster-Hybrid Synthesized Module: `geodesic_ring_and_meta_gate.rs`
//! 100% Safe Rust (`#![forbid(unsafe_code)]`) — Zero external dependencies.
//! Combines a deterministic Q32.31 fixed-point ring buffer with the Superpowers
//! Meta-Cognitive Decision Readiness (Omega) & Thalamic Salience Gate.

#![forbid(unsafe_code)]

/// Q32.31 signed fixed-point scalar (`1.0 == 1 << 31`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Q32(pub i64);

impl Q32 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1i64 << 31);

    #[inline]
    pub const fn from_milli(milli: i64) -> Self {
        Self((milli << 31) / 1000)
    }

    #[inline]
    pub const fn mul(self, rhs: Self) -> Self {
        let prod = (self.0 as i128) * (rhs.0 as i128);
        Self((prod >> 31) as i64)
    }

    #[inline]
    pub const fn div_bounded(self, denom: Self) -> Self {
        if denom.0 <= 0 {
            return Self::ZERO;
        }
        let num = (self.0 as i128) << 31;
        let raw = num / (denom.0 as i128);
        let one = Self::ONE.0 as i128;
        let q = if raw < 0 {
            0
        } else if raw > one {
            one
        } else {
            raw
        };
        Self(q as i64)
    }
}

/// Fixed-capacity deterministic geodesic ring buffer (`N` power of two) with Thalamic Salience Gating.
#[derive(Debug, Clone)]
pub struct ThalamicGeodesicRing<const N: usize> {
    slots: [u64; N],
    salience_q32: [Q32; N],
    head: usize,
    tail: usize,
    count: usize,
    pub min_salience_gate: Q32,
    pub suppressed_noise_events: u64,
}

impl<const N: usize> ThalamicGeodesicRing<N> {
    pub const fn new(min_salience_gate: Q32) -> Self {
        Self {
            slots: [0u64; N],
            salience_q32: [Q32::ZERO; N],
            head: 0,
            tail: 0,
            count: 0,
            min_salience_gate,
            suppressed_noise_events: 0,
        }
    }

    #[inline]
    pub fn push_gated(&mut self, atom_id: u64, salience: Q32) -> bool {
        if salience.0 < self.min_salience_gate.0 {
            self.suppressed_noise_events += 1;
            return false;
        }
        if self.count == N {
            self.head = (self.head + 1) % N;
            self.count -= 1;
        }
        self.slots[self.tail] = atom_id;
        self.salience_q32[self.tail] = salience;
        self.tail = (self.tail + 1) % N;
        self.count += 1;
        true
    }

    #[inline]
    pub fn pop(&mut self) -> Option<(u64, Q32)> {
        if self.count == 0 {
            return None;
        }
        let item = (self.slots[self.head], self.salience_q32[self.head]);
        self.head = (self.head + 1) % N;
        self.count -= 1;
        Some(item)
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.count
    }
}

/// Compute Kojukhov-Bovshover Meta-Cognitive Readiness `Omega = ((1 - U) * (1 - D)) / (B + eps)` in Q32.31.
#[inline]
pub const fn compute_omega_readiness_q32(
    uncertainty: Q32,
    disagreement: Q32,
    blast_radius: Q32,
) -> Q32 {
    let eps = Q32::from_milli(80); // 0.08
    let diff_u = Q32::ONE.0 - uncertainty.0;
    let diff_d = Q32::ONE.0 - disagreement.0;
    let one_minus_u = Q32(if diff_u > 0 { diff_u } else { 0 });
    let one_minus_d = Q32(if diff_d > 0 { diff_d } else { 0 });
    let num = one_minus_u.mul(one_minus_d);
    let denom = Q32(blast_radius.0.saturating_add(eps.0));
    num.div_bounded(denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thalamic_geodesic_ring_filters_low_salience_and_preserves_fifo() {
        let mut ring = ThalamicGeodesicRing::<8>::new(Q32::from_milli(500));
        assert!(!ring.push_gated(101, Q32::from_milli(200)));
        assert_eq!(ring.suppressed_noise_events, 1);
        assert!(ring.push_gated(202, Q32::from_milli(850)));
        assert!(ring.push_gated(303, Q32::from_milli(920)));
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.pop().unwrap().0, 202);
        assert_eq!(ring.pop().unwrap().0, 303);
    }

    #[test]
    fn omega_readiness_matches_superpowers_thresholds() {
        let omega_high = compute_omega_readiness_q32(
            Q32::from_milli(80),
            Q32::from_milli(50),
            Q32::from_milli(200),
        );
        assert!(omega_high.0 >= Q32::from_milli(820).0);

        let omega_blast = compute_omega_readiness_q32(
            Q32::from_milli(100),
            Q32::from_milli(100),
            Q32::from_milli(900),
        );
        assert!(omega_blast.0 < omega_high.0);
    }
}
