//! The falsifier harness — gate qualification by pre-registered experiment
//! (2026-10-19, ported methodology from the Elle worker's
//! `src/observer-falsifier.ts` and `src/kappa-memory/validate.ts`).
//!
//! The discipline, pre-registered here before any data is consulted:
//!
//! A candidate gate signal earns the right to gate ONLY if it demonstrably
//! predicts correctness. The test is a Spearman rank correlation between the
//! signal and the outcome (from the outcome ledger's resolved bets, an
//! INDEPENDENT label — a human or operator resolution, never the gate
//! itself), against a PERMUTATION NULL with a seeded, reproducible PRNG.
//!
//! Pre-registered candidate signals (monster gate):
//!   * `mean_surprise` — mean per-step z-surprise of the served candidate.
//!     Hypothesis: NEGATIVELY predicts correctness.
//!   * `max_z_surprise` — worst single-step z. Same hypothesis.
//!   * `basin_agreement` — fraction of walked arcs on the prompt's own
//!     chain (gate v2 provenance). Hypothesis: POSITIVELY predicts
//!     correctness.
//!   * WITHDRAWN 2026-10-19: the fourth signal ("contaminated_mass")
//!     was built on a false premise — the fingerprint it keyed on
//!     (a30a1003…) is the transformer's OWN identity stamp
//!     (SHA-256 of "geodesic-active-transformer-v1"), not a teacher's.
//!     The measured quantity is SELF-CRYSTALLIZED mass: the fraction of
//!     walked arcs the mind itself crystallized on its own turns. The
//!     ledger still records it (kind-3 frames); no directional
//!     hypothesis is registered until the question is re-framed.
//! Pre-registered decision rule:
//!   * PASS         — p < 0.01 (one-sided, expected direction) AND rho has
//!                    the expected sign. The signal may inform thresholds.
//!   * NULL         — otherwise. The signal gates nothing.
//!   * UNDERPOWERED — fewer than 8 resolved pairs. No verdict; never dress
//!                    sampling noise as signal.
//! Deterministic operator turns are excluded from the pairs: a computed
//! answer is not a prediction, and there is nothing to falsify.
//!
//! PURE: no I/O. The ledger supplies the pairs; this module judges them.

pub const POWER_FLOOR: usize = 8;
pub const ALPHA: f64 = 0.01;
pub const N_NULL: usize = 2000;
pub const SEED: u32 = 0x9e3779b9;

/// mulberry32 — the same seeded PRNG family as the worker original, so the
/// null is reproducible and resume-safe.
pub struct Mulberry32 {
    a: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self { a: seed }
    }
    pub fn next_f64(&mut self) -> f64 {
        self.a = self.a.wrapping_add(0x6d2b79f5);
        let mut t = (self.a ^ (self.a >> 15)).wrapping_mul(1 | self.a);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize
    }
}

/// Spearman rank correlation. Ties broken by first-seen order; the
/// permutation null uses the same ranking on both sides, so any bias cancels.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn rank(xs: &[f64]) -> Vec<f64> {
        let mut order: Vec<(f64, usize)> = xs.iter().copied().zip(0..).collect();
        order.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![0.0; xs.len()];
        for (k, (_, i)) in order.iter().enumerate() {
            r[*i] = k as f64;
        }
        r
    }
    let n = a.len();
    if n == 0 || n != b.len() {
        return 0.0;
    }
    let ra = rank(a);
    let rb = rank(b);
    let mean = (n as f64 - 1.0) / 2.0;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let x = ra[i] - mean;
        let y = rb[i] - mean;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    let den = (da * db).sqrt();
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalsifyVerdict {
    Pass,
    Null,
    Underpowered,
}

impl FalsifyVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            FalsifyVerdict::Pass => "PASS",
            FalsifyVerdict::Null => "NULL",
            FalsifyVerdict::Underpowered => "UNDERPOWERED",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FalsifyResult {
    pub verdict: FalsifyVerdict,
    pub rho: Option<f64>,
    pub p: Option<f64>,
    pub n: usize,
    pub alpha: f64,
}

/// Judge (signal, correctness) pairs. Correctness is 1.0 (correct) or 0.0
/// (incorrect). `expected_positive`: true when the hypothesis is "higher
/// signal predicts higher correctness" (coherence-like signals); false for
/// surprise-like signals.
pub fn falsify_pairs(
    pairs: &[(f64, f64)],
    expected_positive: bool,
    seed: u32,
) -> FalsifyResult {
    let n = pairs.len();
    if n < POWER_FLOOR {
        return FalsifyResult {
            verdict: FalsifyVerdict::Underpowered,
            rho: None,
            p: None,
            n,
            alpha: ALPHA,
        };
    }
    let sig: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let out: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let real = spearman(&sig, &out);

    let mut rng = Mulberry32::new(seed);
    let mut shuffled = out.clone();
    let mut ge = 0usize;
    for _ in 0..N_NULL {
        for i in (1..shuffled.len()).rev() {
            let j = rng.below(i + 1);
            shuffled.swap(i, j);
        }
        let r = spearman(&sig, &shuffled);
        // One-sided in the expected direction.
        let extreme = if expected_positive {
            r >= real
        } else {
            r <= real
        };
        if extreme {
            ge += 1;
        }
    }
    let p = (ge as f64 + 1.0) / (N_NULL as f64 + 1.0);
    let sign_ok = if expected_positive { real > 0.0 } else { real < 0.0 };
    let verdict = if p < ALPHA && sign_ok {
        FalsifyVerdict::Pass
    } else {
        FalsifyVerdict::Null
    };
    FalsifyResult {
        verdict,
        rho: Some(real),
        p: Some(p),
        n,
        alpha: ALPHA,
    }
}

/// One signal's qualification report against the ledger's resolved bets.
#[derive(Debug, Clone)]
pub struct SignalQualification {
    pub signal: &'static str,
    pub result: FalsifyResult,
}

/// Qualify the two pre-registered gate signals against an outcome ledger.
/// Bets without a filed signal (operator turns, refusals, pre-signal-era
/// bets) are excluded — the pairs are only ever (filed signal, resolved
/// outcome).
pub fn qualify_gate_signals(
    ledger: &crate::outcome_ledger::OutcomeLedger,
) -> Vec<SignalQualification> {
    let resolved: Vec<&crate::outcome_ledger::FiledBet> = ledger
        .bets
        .iter()
        .filter(|b| b.outcome != crate::outcome_ledger::OUTCOME_UNRESOLVED)
        .collect();
    let correct = |b: &crate::outcome_ledger::FiledBet| -> f64 {
        if b.outcome == crate::outcome_ledger::OUTCOME_CORRECT {
            1.0
        } else {
            0.0
        }
    };
    let mean_pairs: Vec<(f64, f64)> = resolved
        .iter()
        .filter_map(|b| b.mean_z.map(|s| (s, correct(b))))
        .collect();
    let max_pairs: Vec<(f64, f64)> = resolved
        .iter()
        .filter_map(|b| b.max_z.map(|s| (s, correct(b))))
        .collect();
    let basin_pairs: Vec<(f64, f64)> = resolved
        .iter()
        .filter_map(|b| b.basin_agreement.map(|s| (s, correct(b))))
        .collect();
    vec![
        SignalQualification {
            signal: "mean_surprise",
            result: falsify_pairs(&mean_pairs, false, SEED),
        },
        SignalQualification {
            signal: "max_z_surprise",
            result: falsify_pairs(&max_pairs, false, SEED ^ 0x51ab),
        },
        SignalQualification {
            signal: "basin_agreement",
            result: falsify_pairs(&basin_pairs, true, SEED ^ 0x9c41),
        },
    ]
}

/// Active gate thresholds derived from QUALIFIED signals (2026-10-19,
/// falsifier hardening, Stewart's order). A signal that earned PASS
/// becomes a serving threshold set at the most permissive value any
/// CORRECT resolved bet exhibited — no correct turn is ever rejected
/// by construction; anything beyond every correct example is refused.
/// Unqualified signals contribute nothing: the thresholds stay inert
/// until the data earns them.
#[derive(Debug, Clone, Copy, Default)]
pub struct ActiveThresholds {
    pub mean_z_max: Option<f64>,
    pub max_z_max: Option<f64>,
    pub basin_min: Option<f64>,
}

pub fn active_thresholds(ledger: &crate::outcome_ledger::OutcomeLedger) -> ActiveThresholds {
    let resolved: Vec<&crate::outcome_ledger::FiledBet> = ledger
        .bets
        .iter()
        .filter(|b| b.outcome != crate::outcome_ledger::OUTCOME_UNRESOLVED)
        .collect();
    let mut out = ActiveThresholds::default();
    for q in qualify_gate_signals(ledger) {
        if q.result.verdict != FalsifyVerdict::Pass {
            continue;
        }
        match q.signal {
            "mean_surprise" => {
                out.mean_z_max = resolved
                    .iter()
                    .filter(|b| b.outcome == crate::outcome_ledger::OUTCOME_CORRECT)
                    .filter_map(|b| b.mean_z)
                    .fold(None, |acc: Option<f64>, v| {
                        Some(acc.map_or(v, |a| a.max(v)))
                    });
            }
            "max_z_surprise" => {
                out.max_z_max = resolved
                    .iter()
                    .filter(|b| b.outcome == crate::outcome_ledger::OUTCOME_CORRECT)
                    .filter_map(|b| b.max_z)
                    .fold(None, |acc: Option<f64>, v| {
                        Some(acc.map_or(v, |a| a.max(v)))
                    });
            }
            "basin_agreement" => {
                out.basin_min = resolved
                    .iter()
                    .filter(|b| b.outcome == crate::outcome_ledger::OUTCOME_CORRECT)
                    .filter_map(|b| b.basin_agreement)
                    .fold(None, |acc: Option<f64>, v| {
                        Some(acc.map_or(v, |a| a.min(v)))
                    });
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(mode: &str, n: usize, seed: u32) -> Vec<(f64, f64)> {
        let mut rng = Mulberry32::new(seed);
        let mut sig = Vec::new();
        let mut out = Vec::new();
        for _ in 0..n {
            let s = rng.next_f64();
            sig.push(s);
            out.push(match mode {
                "noise" => rng.next_f64(),
                // "coupled": correctness falls as surprise rises
                _ => (1.0 - 0.7 * s - 0.3 * rng.next_f64()).clamp(0.0, 1.0),
            });
        }
        if mode == "shuffled" {
            for i in (1..out.len()).rev() {
                let j = rng.below(i + 1);
                out.swap(i, j);
            }
        }
        sig.into_iter().zip(out).collect()
    }

    #[test]
    fn coupled_signal_passes() {
        let pairs = synthetic("coupled", 40, 7);
        let r = falsify_pairs(&pairs, false, SEED);
        assert_eq!(r.verdict, FalsifyVerdict::Pass);
        assert!(r.rho.unwrap() < 0.0);
    }

    #[test]
    fn noise_signal_is_null() {
        let pairs = synthetic("noise", 40, 7);
        let r = falsify_pairs(&pairs, false, SEED);
        assert_eq!(r.verdict, FalsifyVerdict::Null);
    }

    #[test]
    fn shuffled_pairing_is_null() {
        let pairs = synthetic("shuffled", 40, 7);
        let r = falsify_pairs(&pairs, false, SEED);
        assert_eq!(r.verdict, FalsifyVerdict::Null);
    }

    #[test]
    fn underpowered_below_floor() {
        let pairs = synthetic("coupled", POWER_FLOOR - 1, 7);
        let r = falsify_pairs(&pairs, false, SEED);
        assert_eq!(r.verdict, FalsifyVerdict::Underpowered);
        assert!(r.rho.is_none());
    }

    #[test]
    fn permutation_null_is_deterministic() {
        let pairs = synthetic("noise", 30, 3);
        let a = falsify_pairs(&pairs, false, SEED);
        let b = falsify_pairs(&pairs, false, SEED);
        assert_eq!(a.p, b.p);
        assert_eq!(a.rho, b.rho);
    }

    #[test]
    fn spearman_perfect_monotone() {
        let a: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let b: Vec<f64> = a.iter().map(|x| x * x).collect();
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-9);
        let c: Vec<f64> = a.iter().rev().copied().collect();
        assert!((spearman(&a, &c) + 1.0).abs() < 1e-9);
    }
}
