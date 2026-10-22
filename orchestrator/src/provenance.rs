//! Gate v2 relevance instrumentation (2026-10-19).
//!
//! The trajectory gate (predict_gate.rs) measures trajectory CONSISTENCY —
//! how well a candidate matches the prompt's forecast. Measured result: it
//! does not separate fluent-but-wrong-domain continuations (the bookkeeping
//! answer to "two plus two") from good ones, because the bookkeeping walk is
//! internally consistent. CONSISTENCY IS NOT RELEVANCE.
//!
//! This module computes the relevance evidence from ARC PROVENANCE — where
//! the walked arcs actually came from (plumbed through the span decode):
//!
//!   * `basin_agreement` — fraction of emitted ops whose winning arc lay on
//!     the prompt's own trajectory chain (vs. generic vocabulary fallback
//!     arcs with no chain relation to the prompt).
//!   * `self_crystallized_mass` — fraction of ops whose arc was
//!     quarantined teacher-era origin (the registry seeded from Argon's
//!     WAL/vault quarantine — the 2+2 bookkeeping proof).
//!   * `mean_certainty` — mean teacher certainty of the walked arcs.
//!
//! DISCIPLINE (same as gate v1 at birth and as the falsifier requires):
//! these are INSTRUMENTS, not yet gates. Candidate rejection thresholds
//! (e.g. basin_agreement < 0.5 or self_crystallized_mass > 0.5) are
//! pre-registered here as FUTURE falsifier targets; nothing rejects on
//! provenance until a signal qualifies against resolved outcomes.

use std::collections::HashSet;

/// The transformer's OWN identity fingerprint (hex) — VERIFIED
/// 2026-10-19: this is SHA-256 of the string
/// "geodesic-active-transformer-v1", the default `origin_fingerprint`
/// stamped on every arc the transformer crystallizes on its own turns.
/// It is NOT a teacher fingerprint. Arcs carrying it are the mind's own
/// turn-crystallizations (which INCLUDE the live-teacher turns, since
/// those crystallized through the transformer). Gemini's predictor
/// training quarantines this origin — correctly excluding
/// turn-crystallized arcs, under a corrected label.
pub const TRANSFORMER_IDENTITY_FP_HEX: &str =
    "a30a100326adb6d24d3b7dd51e5cb8b92d3b076f71302f6b9439f0d01bc54e61";

pub fn hex_to_bytes32(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

/// Registry of quarantined arc origins (teacher-era fingerprints).
#[derive(Debug, Clone)]
pub struct QuarantineRegistry {
    origins: HashSet<[u8; 32]>,
}

impl QuarantineRegistry {
    pub fn with_teacher_era() -> Self {
        let mut origins = HashSet::new();
        if let Some(fp) = hex_to_bytes32(TRANSFORMER_IDENTITY_FP_HEX) {
            origins.insert(fp);
        }
        Self { origins }
    }
    pub fn add(&mut self, fp: [u8; 32]) {
        self.origins.insert(fp);
    }
    pub fn contains(&self, fp: &[u8; 32]) -> bool {
        self.origins.contains(fp)
    }
    pub fn len(&self) -> usize {
        self.origins.len()
    }
    pub fn is_empty(&self) -> bool {
        self.origins.is_empty()
    }
}

impl Default for QuarantineRegistry {
    fn default() -> Self {
        Self::with_teacher_era()
    }
}

/// The relevance evidence for one served candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelevanceVerdict {
    pub n_ops: usize,
    pub basin_agreement: f64,
    pub self_crystallized_mass: f64,
    pub mean_certainty: f64,
}

/// Compute the verdict from decode provenance. `None` when no trajectory
/// was walked (no provenance to judge).
pub fn relevance_verdict(
    prov: &[atoma_immune::ArcProvenance],
    registry: &QuarantineRegistry,
) -> Option<RelevanceVerdict> {
    if prov.is_empty() {
        return None;
    }
    let n = prov.len() as f64;
    let on_chain = prov.iter().filter(|p| p.on_prompt_chain).count() as f64;
    let contaminated = prov
        .iter()
        .filter(|p| registry.contains(&p.origin_fp))
        .count() as f64;
    let mean_certainty = prov
        .iter()
        .map(|p| p.certainty_q32 as f64 / 2_147_483_648.0)
        .sum::<f64>()
        / n;
    Some(RelevanceVerdict {
        n_ops: prov.len(),
        basin_agreement: on_chain / n,
        self_crystallized_mass: contaminated / n,
        mean_certainty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use atoma_immune::RoleId;

    fn prov(on_chain: bool, origin: [u8; 32], certainty: i64) -> atoma_immune::ArcProvenance {
        atoma_immune::ArcProvenance {
            arc_id: 1,
            chain_id: 7,
            from_role: RoleId::SensoryPerception,
            to_role: RoleId::MotorPraxis,
            origin_fp: origin,
            certainty_q32: certainty,
            on_prompt_chain: on_chain,
        }
    }

    #[test]
    fn teacher_fp_parses() {
        let fp = hex_to_bytes32(TRANSFORMER_IDENTITY_FP_HEX).unwrap();
        assert_eq!(fp[0], 0xa3);
        assert_eq!(fp[1], 0x0a);
        assert!(hex_to_bytes32("zz").is_none());
    }

    #[test]
    fn verdict_measures_chain_and_contamination() {
        let reg = QuarantineRegistry::with_teacher_era();
        let teacher = hex_to_bytes32(TRANSFORMER_IDENTITY_FP_HEX).unwrap();
        let clean = [0x11u8; 32];
        let chain = [
            prov(true, teacher, 1 << 30),
            prov(true, clean, 1 << 30),
            prov(false, teacher, 1 << 30),
            prov(false, clean, 1 << 30),
        ];
        let v = relevance_verdict(&chain, &reg).unwrap();
        assert_eq!(v.n_ops, 4);
        assert!((v.basin_agreement - 0.5).abs() < 1e-12);
        assert!((v.self_crystallized_mass - 0.5).abs() < 1e-12);
        assert!((v.mean_certainty - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_provenance_declines() {
        let reg = QuarantineRegistry::with_teacher_era();
        assert!(relevance_verdict(&[], &reg).is_none());
    }

    #[test]
    fn clean_chain_scores_zero_contamination() {
        let reg = QuarantineRegistry::with_teacher_era();
        let chain = [prov(true, [0x22u8; 32], 1 << 31)];
        let v = relevance_verdict(&chain, &reg).unwrap();
        assert_eq!(v.self_crystallized_mass, 0.0);
        assert_eq!(v.basin_agreement, 1.0);
    }
}
