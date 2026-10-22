//! PAMI — Phase-Augmented Multifractal Indexing (ported 2026-10-19).
//!
//! PROVENANCE: port of the Elle worker's `src/pami.ts` core (the pure index /
//! distance / resonance machinery; the worker's store, tool surface, and
//! adaptive-delta layers are NOT ported — they belong to the worker's
//! persistence substrate).
//!
//! Memory indexed by the geometric residue of surprisal: a signal is
//! fingerprinted by 8 relative phase components (𝕋⁸) + 13 generalized
//! multifractal dimensions D(q), q ∈ {−6..6} (ℝ¹³) = 21 floats, and retrieved
//! by structural resonance, not content lookup:
//!
//! ```text
//! signal → φ-spaced Morlet wavelet transform →
//!   (a) within-scale phase increments at fixed slots k ∈ {1..8}
//!       (energy-gated; the §VI.3 erratum is preserved — see pamiIndex)
//!   (b) wavelet leaders → structure functions S(q,s) → ζ(q) →
//!       D(q) = q·h(q) − ζ(q) + 1  (Legendre relation)
//! distance d = 0.5·d_phase + 0.5·d_dims,  resonance ρ = exp(−d/0.25)
//! ```
//!
//! In the monster, this is the gate's SECOND channel: an index of the
//! prompt's carrier stream and an index of a candidate's carrier stream give
//! a relevance signal independent of trajectory forecasting. Deterministic,
//! pure, no model anywhere in here.

pub const PHI: f64 = 1.6180339887498949;
pub const PHI_INV: f64 = 0.6180339887498949;

const MAX_SIGNAL: usize = 4096;
const ALPHA: f64 = 0.5;
const BETA: f64 = 0.5;
const TAU: f64 = 0.25;
pub const DELTA_DEFAULT: f64 = 0.3;

#[derive(Debug, Clone, Copy)]
pub struct PamiConfig {
    pub phase_count: usize, // 8 in the spec
    pub q_max: i32,         // 6 → q ∈ {−6..6} = 13 dimensions
    pub scale_k_min: i32,   // φ-grid k ∈ {−4..8} = 13 scales
    pub scale_k_max: i32,
    pub base_scale: f64,    // samples at k = 0
    pub scale_base: f64,    // φ by the spec's Hurwitz argument
}

pub const SPEC_CONFIG: PamiConfig = PamiConfig {
    phase_count: 8,
    q_max: 6,
    scale_k_min: -4,
    scale_k_max: 8,
    base_scale: 4.0,
    scale_base: PHI,
};

#[derive(Debug, Clone, PartialEq)]
pub struct PamiIndex {
    pub phases: Vec<f64>, // length = phase_count
    pub dims: Vec<f64>,   // length = 2·q_max + 1
}

pub fn index_length(cfg: PamiConfig) -> usize {
    cfg.phase_count + 2 * cfg.q_max as usize + 1
}

fn round_dp(x: f64, dp: i32) -> f64 {
    let m = 10f64.powi(dp);
    (x * m).round() / m
}

struct ScaleRow {
    k: i32,
    scale: f64,
    re: Vec<f64>,
    im: Vec<f64>,
    energy: f64,
}

/// φ-spaced complex Morlet continuous wavelet transform.
pub fn phi_wavelet_transform(signal: &[f64], cfg: PamiConfig) -> Option<Vec<(i32, f64, Vec<f64>, Vec<f64>, f64)>> {
    let x: Vec<f64> = signal
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .take(MAX_SIGNAL)
        .collect();
    let n = x.len();
    if n < 32 {
        return None;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let w0 = 6.0_f64;
    let mut rows: Vec<ScaleRow> = Vec::new();

    for k in cfg.scale_k_min..=cfg.scale_k_max {
        let scale = cfg.base_scale * cfg.scale_base.powi(k);
        if scale < 0.8 || scale * 3.0 > n as f64 {
            continue;
        }
        let half = ((3.0 * scale).ceil() as i64).max(1);
        let mut kre: Vec<f64> = Vec::new();
        let mut kim: Vec<f64> = Vec::new();
        for t in -half..=half {
            let tf = t as f64;
            let g = (-(tf * tf) / (2.0 * scale * scale)).exp() / scale.sqrt();
            kre.push(g * (w0 * tf / scale).cos());
            kim.push(-g * (w0 * tf / scale).sin());
        }
        let hop = ((scale / 2.0).floor() as i64).max(1);
        let mut re: Vec<f64> = Vec::new();
        let mut im: Vec<f64> = Vec::new();
        let mut energy = 0.0;
        let mut c = half;
        while c + half < n as i64 {
            let mut sr = 0.0;
            let mut si = 0.0;
            for t in -half..=half {
                let v = x[(c + t) as usize] - mean;
                let ki = (t + half) as usize;
                sr += v * kre[ki];
                si += v * kim[ki];
            }
            re.push(sr);
            im.push(si);
            energy += sr * sr + si * si;
            c += hop;
        }
        if re.len() >= 3 {
            let count = re.len() as f64;
            rows.push(ScaleRow {
                k,
                scale,
                re,
                im,
                energy: energy / count,
            });
        }
    }
    if rows.len() < 4 {
        return None;
    }
    Some(
        rows.into_iter()
            .map(|r| (r.k, r.scale, r.re, r.im, r.energy))
            .collect(),
    )
}

fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let xm = xs.iter().sum::<f64>() / n as f64;
    let ym = ys.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        num += (xs[i] - xm) * (ys[i] - ym);
        den += (xs[i] - xm).powi(2);
    }
    if den > 1e-12 {
        num / den
    } else {
        0.0
    }
}

/// Fingerprint a signal. None when the signal is too short to resolve.
pub fn pami_index(signal: &[f64], cfg: PamiConfig) -> Option<PamiIndex> {
    let raw = phi_wavelet_transform(signal, cfg)?;
    let rows: Vec<ScaleRow> = raw
        .into_iter()
        .map(|(k, scale, re, im, energy)| ScaleRow {
            k,
            scale,
            re,
            im,
            energy,
        })
        .collect();

    // Φ̃ — within-scale phase increments at FIXED slots (the §VI.3 erratum:
    // cross-scale relative phases are not stable observables on a φ grid;
    // increments deliver the demanded invariances). A slot whose band carries
    // no real content reads 0 — the energy gate's sentinel.
    let phase_increment = |r: &ScaleRow| -> f64 {
        let mut sr = 0.0;
        let mut si = 0.0;
        for i in 1..r.re.len() {
            let pr = r.re[i] * r.re[i - 1] + r.im[i] * r.im[i - 1];
            let pi = r.im[i] * r.re[i - 1] - r.re[i] * r.im[i - 1];
            sr += pr;
            si += pi;
        }
        if (sr * sr + si * si).sqrt() < 1e-12 {
            0.0
        } else {
            si.atan2(sr)
        }
    };
    let max_energy = rows
        .iter()
        .map(|r| r.energy)
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let mut phases: Vec<f64> = Vec::with_capacity(cfg.phase_count);
    for k in 1..=cfg.phase_count as i32 {
        match rows.iter().find(|r| r.k == k) {
            Some(row) if row.energy > 0.02 * max_energy => {
                phases.push(round_dp(phase_increment(row), 5));
            }
            _ => phases.push(0.0),
        }
    }

    // 𝐃 — wavelet leaders → structure functions → ζ(q) → D(q).
    let mut leaders: Vec<(f64, Vec<f64>)> = Vec::new();
    for j in 0..rows.len() {
        let r = &rows[j];
        let mut l: Vec<f64> = Vec::with_capacity(r.re.len());
        for i in 0..r.re.len() {
            let mut sup = 0.0f64;
            for rr in rows.iter().take(j + 1) {
                let ratio = rr.re.len() as f64 / r.re.len() as f64;
                let c = (i as f64 * ratio).round() as i64;
                for d in -1i64..=1 {
                    let idx = c + d;
                    if idx >= 0 && (idx as usize) < rr.re.len() {
                        let m = (rr.re[idx as usize].powi(2) + rr.im[idx as usize].powi(2)).sqrt();
                        if m > sup {
                            sup = m;
                        }
                    }
                }
            }
            l.push(sup + 1e-12);
        }
        leaders.push((r.scale, l));
    }

    let qs: Vec<i32> = (-cfg.q_max..=cfg.q_max).collect();
    let zeta: Vec<f64> = qs
        .iter()
        .map(|&q| {
            let mut xs: Vec<f64> = Vec::new();
            let mut ys: Vec<f64> = Vec::new();
            for (scale, l) in &leaders {
                let mut s: f64 = l.iter().map(|v| v.powi(q)).sum();
                s /= l.len() as f64;
                if s > 0.0 && s.is_finite() {
                    xs.push(scale.log2());
                    ys.push(s.log2());
                }
            }
            slope(&xs, &ys)
        })
        .collect();
    let dims: Vec<f64> = qs
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let h = if i == 0 {
                zeta[1] - zeta[0]
            } else if i == qs.len() - 1 {
                zeta[i] - zeta[i - 1]
            } else {
                (zeta[i + 1] - zeta[i - 1]) / 2.0
            };
            round_dp(q as f64 * h - zeta[i] + 1.0, 4)
        })
        .collect();

    Some(PamiIndex { phases, dims })
}

/// Structural distance between two indices (§VI.4).
pub fn pami_distance(a: &PamiIndex, b: &PamiIndex) -> f64 {
    // Phase half: sentinel-aware (0 = "no content in this band").
    let np = a.phases.len().min(b.phases.len());
    let mut dphi = 0.0;
    let mut counted = 0usize;
    for i in 0..np {
        let (pa, pb) = (a.phases[i], b.phases[i]);
        let (has_a, has_b) = (pa != 0.0, pb != 0.0);
        if !has_a && !has_b {
            continue;
        }
        counted += 1;
        if has_a != has_b {
            dphi += 0.5;
            continue;
        }
        let mut d = (pa - pb).abs() % (2.0 * std::f64::consts::PI);
        if d > std::f64::consts::PI {
            d = 2.0 * std::f64::consts::PI - d;
        }
        dphi += d / std::f64::consts::PI;
    }
    let dphi = if counted > 0 { dphi / counted as f64 } else { 0.5 };

    // Dimension half: weighted RMS, tails damped (w = 1/(1 + 0.1 q²)).
    let nd = a.dims.len().min(b.dims.len());
    let q_max = (nd as f64 - 1.0) / 2.0;
    let mut dd = 0.0;
    let mut wsum = 0.0;
    for i in 0..nd {
        let q = i as f64 - q_max;
        let w = 1.0 / (1.0 + 0.1 * q * q);
        dd += w * (a.dims[i] - b.dims[i]).powi(2);
        wsum += w;
    }
    let dd = if wsum > 0.0 { (dd / wsum).sqrt() } else { 0.0 };
    ALPHA * dphi + BETA * dd.min(1.0)
}

/// Resonance ρ = exp(−d/τ) ∈ (0, 1].
pub fn resonance(a: &PamiIndex, b: &PamiIndex) -> f64 {
    round_dp((-pami_distance(a, b) / TAU).exp(), 4)
}

/// κ(T,t) cross-modal resonance between two simultaneous residual windows.
pub fn kappa_cross_modal(a: &[f64], b: &[f64], cfg: PamiConfig) -> Option<f64> {
    let ia = pami_index(a, cfg)?;
    let ib = pami_index(b, cfg)?;
    Some(resonance(&ia, &ib))
}

#[cfg(test)]
mod tests {




    use super::*;

    fn xorshift_signal(seed: u64, n: usize, freq: f64) -> Vec<f64> {
        // Deterministic structured signal: carrier + seeded jitter.
        let mut s = seed;
        (0..n)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let jitter = ((s % 1000) as f64 / 1000.0 - 0.5) * 0.2;
                (2.0 * std::f64::consts::PI * freq * i as f64 / 64.0).sin() + jitter
            })
            .collect()
    }

    #[test]
    fn index_shape_is_21() {
        let sig = xorshift_signal(42, 512, 3.0);
        let idx = pami_index(&sig, SPEC_CONFIG).expect("index");
        assert_eq!(idx.phases.len(), 8);
        assert_eq!(idx.dims.len(), 13);
        assert_eq!(index_length(SPEC_CONFIG), 21);
    }

    #[test]
    fn deterministic_same_signal_same_index() {
        let sig = xorshift_signal(7, 512, 2.0);
        let a = pami_index(&sig, SPEC_CONFIG).unwrap();
        let b = pami_index(&sig, SPEC_CONFIG).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn self_distance_zero_resonance_one() {
        let sig = xorshift_signal(99, 512, 4.0);
        let idx = pami_index(&sig, SPEC_CONFIG).unwrap();
        assert!(pami_distance(&idx, &idx).abs() < 1e-12);
        assert_eq!(resonance(&idx, &idx), 1.0);
    }

    #[test]
    fn different_structure_measures_farther_than_same_structure() {
        let a = pami_index(&xorshift_signal(5, 512, 2.0), SPEC_CONFIG).unwrap();
        let b = pami_index(&xorshift_signal(5, 512, 2.0), SPEC_CONFIG).unwrap();
        let c = pami_index(&xorshift_signal(5, 512, 11.0), SPEC_CONFIG).unwrap();
        assert!(pami_distance(&a, &b) < pami_distance(&a, &c));
    }

    #[test]
    fn short_signal_declines() {
        assert!(pami_index(&[1.0; 16], SPEC_CONFIG).is_none());
    }
}
