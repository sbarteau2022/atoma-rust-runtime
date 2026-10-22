//! Outcome ledger — the oracle's bet ledger, reimplemented Rust-native
//! (2026-10-19, from the Elle worker's `src/oracle.ts` design).
//!
//! Nobody ever tells the map when it was wrong; it only finds out if someone
//! happens to notice. So: every served live turn files a bet — the answer,
//! the tier that produced it, and a stated confidence — and resolutions land
//! later (operator check, gate rejection, or a human mark). The ledger
//! accumulates into calibration: stated confidence vs observed accuracy.
//!
//! A filed bet resolved INCORRECT is a SCAR: the prompt's trajectory must
//! never again walk at Tier1 full confidence — the scar lookup demotes it to
//! synthesis. This is the mechanism that stops a wrong learned trajectory
//! (e.g. a teacher-era bookkeeping answer to an arithmetic prompt) from
//! being served as exact recall forever.
//!
//! Format: append-only binary log, no JSON. Header line, then fixed-size
//! frames: kind 0 = filed bet, kind 1 = resolution referencing a filed seq.
//! A partial tail frame (crash mid-write) is ignored on load.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::PathBuf;

pub const LEDGER_HEADER: &[u8] = b"ATOMA_OUTCOME_LEDGER_V1\n";

pub const OUTCOME_UNRESOLVED: u8 = 0;
pub const OUTCOME_CORRECT: u8 = 1;
pub const OUTCOME_INCORRECT: u8 = 2;

pub const SOURCE_NONE: u8 = 0;
pub const SOURCE_OPERATOR: u8 = 1;
pub const SOURCE_USER: u8 = 2;
pub const SOURCE_GATE: u8 = 3;

/// Stated confidence (basis points) by serving path. Provisional by design:
/// the calibration curve exists to check whether these numbers are honest.
pub const CONFIDENCE_TIER1_BP: u16 = 10_000;
pub const CONFIDENCE_TIER2_BP: u16 = 6_500;
pub const CONFIDENCE_OPERATOR_BP: u16 = 10_000;

/// Two independent 64-bit hashes packed as one 128-bit fingerprint.
/// Deterministic, no external crates.
pub fn fingerprint(parts: &[&[u8]]) -> [u8; 16] {
    let mut fnv: u64 = 0xcbf29ce484222325;
    let mut djb: u64 = 5381;
    for part in parts {
        for &b in part.iter() {
            fnv ^= b as u64;
            fnv = fnv.wrapping_mul(0x100000001b3);
            djb = djb.wrapping_mul(33).wrapping_add(b as u64);
        }
        // part separator so ("ab","c") != ("a","bc")
        fnv ^= 0xff;
        fnv = fnv.wrapping_mul(0x100000001b3);
        djb = djb.wrapping_mul(33).wrapping_add(0xff);
    }
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&fnv.to_le_bytes());
    out[8..].copy_from_slice(&djb.to_le_bytes());
    out
}

#[derive(Debug, Clone)]
pub struct FiledBet {
    pub seq: u64,
    pub prompt_fp: [u8; 16],
    pub answer_fp: [u8; 16],
    pub tier: u8,
    pub confidence_bp: u16,
    pub outcome: u8,
    pub source: u8,
    /// Gate signals filed with the bet (kind-2 frames): the served
    /// candidate's forecast surprise. None for operator turns and for
    /// pre-signal-era bets — the falsifier only reads resolved bets that
    /// carry a signal.
    pub max_z: Option<f64>,
    pub mean_z: Option<f64>,
    /// Relevance signals filed with the bet (kind-3 frames): the served
    /// candidate's provenance verdict. None when no trajectory was walked.
    pub basin_agreement: Option<f64>,
    pub contaminated_mass: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct OutcomeLedger {
    pub bets: Vec<FiledBet>,
    pub path: Option<PathBuf>,
    next_seq: u64,
}

// Frame layout (filed, kind 0): kind u8 | seq u64 | prompt_fp 16 | answer_fp 16
//   | tier u8 | confidence_bp u16 | unix_ms u64            = 52 bytes
// Frame layout (resolution, kind 1): kind u8 | seq u64 | outcome u8 | source u8
//   | unix_ms u64                                          = 19 bytes
const FILED_FRAME_LEN: usize = 52;
const RESOLUTION_FRAME_LEN: usize = 19;
// Frame layout (signal, kind 2): kind u8 | seq u64 | max_z f64 | mean_z f64
//   = 25 bytes
const SIGNAL_FRAME_LEN: usize = 25;
// Frame layout (relevance, kind 3): kind u8 | seq u64 | basin_agreement f64
//   | contaminated_mass f64                              = 25 bytes
const RELEVANCE_FRAME_LEN: usize = 25;

impl OutcomeLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a backing file: load existing frames, create with header if new.
    pub fn attach(&mut self, path: PathBuf) -> std::io::Result<()> {
        if path.exists() {
            let mut bytes = Vec::new();
            std::fs::File::open(&path)?.read_to_end(&mut bytes)?;
            if bytes.starts_with(LEDGER_HEADER) {
                self.parse_frames(&bytes[LEDGER_HEADER.len()..]);
            }
        } else {
            let mut f = std::fs::File::create(&path)?;
            f.write_all(LEDGER_HEADER)?;
            f.flush()?;
        }
        self.path = Some(path);
        Ok(())
    }

    fn parse_frames(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            match bytes[0] {
                0 => {
                    if bytes.len() < FILED_FRAME_LEN {
                        break;
                    }
                    let seq = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
                    let mut prompt_fp = [0u8; 16];
                    prompt_fp.copy_from_slice(&bytes[9..25]);
                    let mut answer_fp = [0u8; 16];
                    answer_fp.copy_from_slice(&bytes[25..41]);
                    let tier = bytes[41];
                    let confidence_bp =
                        u16::from_le_bytes(bytes[42..44].try_into().unwrap());
                    self.bets.push(FiledBet {
                        seq,
                        prompt_fp,
                        answer_fp,
                        tier,
                        confidence_bp,
                        outcome: OUTCOME_UNRESOLVED,
                        source: SOURCE_NONE,
                        max_z: None,
                        mean_z: None,
                        basin_agreement: None,
                        contaminated_mass: None,
                    });
                    self.next_seq = self.next_seq.max(seq + 1);
                    bytes = &bytes[FILED_FRAME_LEN..];
                }
                1 => {
                    if bytes.len() < RESOLUTION_FRAME_LEN {
                        break;
                    }
                    let seq = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
                    let outcome = bytes[9];
                    let source = bytes[10];
                    if let Some(bet) = self.bets.iter_mut().find(|b| b.seq == seq) {
                        bet.outcome = outcome;
                        bet.source = source;
                    }
                    bytes = &bytes[RESOLUTION_FRAME_LEN..];
                }
                2 => {
                    if bytes.len() < SIGNAL_FRAME_LEN {
                        break;
                    }
                    let seq = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
                    let max_z = f64::from_le_bytes(bytes[9..17].try_into().unwrap());
                    let mean_z = f64::from_le_bytes(bytes[17..25].try_into().unwrap());
                    if let Some(bet) = self.bets.iter_mut().find(|b| b.seq == seq) {
                        bet.max_z = Some(max_z);
                        bet.mean_z = Some(mean_z);
                    }
                    bytes = &bytes[SIGNAL_FRAME_LEN..];
                }
                3 => {
                    if bytes.len() < RELEVANCE_FRAME_LEN {
                        break;
                    }
                    let seq = u64::from_le_bytes(bytes[1..9].try_into().unwrap());
                    let basin = f64::from_le_bytes(bytes[9..17].try_into().unwrap());
                    let contaminated =
                        f64::from_le_bytes(bytes[17..25].try_into().unwrap());
                    if let Some(bet) = self.bets.iter_mut().find(|b| b.seq == seq) {
                        bet.basin_agreement = Some(basin);
                        bet.contaminated_mass = Some(contaminated);
                    }
                    bytes = &bytes[RELEVANCE_FRAME_LEN..];
                }
                _ => break, // unknown frame kind: stop, keep what parsed
            }
        }
    }

    fn append(&self, frame: &[u8]) {
        if let Some(path) = &self.path {
            if let Ok(mut f) = OpenOptions::new().append(true).open(path) {
                let _ = f.write_all(frame);
                let _ = f.flush();
            }
        }
    }

    /// File a bet for a served turn. Returns the bet's sequence number.
    pub fn file(
        &mut self,
        prompt: &str,
        answer: &str,
        tier: u8,
        confidence_bp: u16,
        unix_ms: u64,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let prompt_fp = fingerprint(&[prompt.as_bytes()]);
        let answer_fp = fingerprint(&[prompt.as_bytes(), answer.as_bytes()]);
        let mut frame = Vec::with_capacity(FILED_FRAME_LEN);
        frame.push(0u8);
        frame.extend_from_slice(&seq.to_le_bytes());
        frame.extend_from_slice(&prompt_fp);
        frame.extend_from_slice(&answer_fp);
        frame.push(tier);
        frame.extend_from_slice(&confidence_bp.to_le_bytes());
        frame.extend_from_slice(&unix_ms.to_le_bytes());
        self.append(&frame);
        self.bets.push(FiledBet {
            seq,
            prompt_fp,
            answer_fp,
            tier,
            confidence_bp,
            outcome: OUTCOME_UNRESOLVED,
            source: SOURCE_NONE,
            max_z: None,
            mean_z: None,
            basin_agreement: None,
            contaminated_mass: None,
        });
        seq
    }

    /// Resolve a filed bet. Returns false if no such unresolved bet exists.
    pub fn resolve(&mut self, seq: u64, outcome: u8, source: u8, unix_ms: u64) -> bool {
        let Some(bet) = self.bets.iter_mut().find(|b| b.seq == seq) else {
            return false;
        };
        if bet.outcome != OUTCOME_UNRESOLVED {
            return false;
        }
        bet.outcome = outcome;
        bet.source = source;
        let mut frame = Vec::with_capacity(RESOLUTION_FRAME_LEN);
        frame.push(1u8);
        frame.extend_from_slice(&seq.to_le_bytes());
        frame.push(outcome);
        frame.push(source);
        frame.extend_from_slice(&unix_ms.to_le_bytes());
        self.append(&frame);
        true
    }

    /// File the gate signal for a bet (called at turn time, right after
    /// `file`). Operator turns never call this: a computed answer is not a
    /// prediction, and the falsifier must not be fed fake pairs.
    pub fn attach_signal(&mut self, seq: u64, max_z: f64, mean_z: f64) -> bool {
        let Some(bet) = self.bets.iter_mut().find(|b| b.seq == seq) else {
            return false;
        };
        bet.max_z = Some(max_z);
        bet.mean_z = Some(mean_z);
        let mut frame = Vec::with_capacity(SIGNAL_FRAME_LEN);
        frame.push(2u8);
        frame.extend_from_slice(&seq.to_le_bytes());
        frame.extend_from_slice(&max_z.to_le_bytes());
        frame.extend_from_slice(&mean_z.to_le_bytes());
        self.append(&frame);
        true
    }

    /// File the relevance verdict for a bet (gate v2 instrument). Only
    /// called for turns that walked a trajectory (non-empty provenance).
    pub fn attach_relevance(
        &mut self,
        seq: u64,
        basin_agreement: f64,
        contaminated_mass: f64,
    ) -> bool {
        let Some(bet) = self.bets.iter_mut().find(|b| b.seq == seq) else {
            return false;
        };
        bet.basin_agreement = Some(basin_agreement);
        bet.contaminated_mass = Some(contaminated_mass);
        let mut frame = Vec::with_capacity(RELEVANCE_FRAME_LEN);
        frame.push(3u8);
        frame.extend_from_slice(&seq.to_le_bytes());
        frame.extend_from_slice(&basin_agreement.to_le_bytes());
        frame.extend_from_slice(&contaminated_mass.to_le_bytes());
        self.append(&frame);
        true
    }

    /// Resolve the most recent unresolved bet (the REPL `mark` path).
    pub fn resolve_latest(&mut self, outcome: u8, source: u8, unix_ms: u64) -> Option<u64> {
        let seq = self
            .bets
            .iter()
            .rev()
            .find(|b| b.outcome == OUTCOME_UNRESOLVED)
            .map(|b| b.seq)?;
        if self.resolve(seq, outcome, source, unix_ms) {
            Some(seq)
        } else {
            None
        }
    }

    /// THE SCAR CHECK: has this prompt ever been resolved incorrect?
    pub fn scarred(&self, prompt: &str) -> bool {
        let fp = fingerprint(&[prompt.as_bytes()]);
        self.bets
            .iter()
            .any(|b| b.prompt_fp == fp && b.outcome == OUTCOME_INCORRECT)
    }

    /// Calibration buckets: (label, n, stated confidence, observed accuracy).
    pub fn calibration(&self) -> Vec<(String, usize, f64, f64)> {
        let edges = [0u16, 2000, 4000, 6000, 8000, 10001];
        let mut out = Vec::new();
        for w in edges.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            let rows: Vec<&FiledBet> = self
                .bets
                .iter()
                .filter(|b| {
                    b.outcome != OUTCOME_UNRESOLVED
                        && b.confidence_bp >= lo
                        && b.confidence_bp < hi
                })
                .collect();
            if rows.is_empty() {
                continue;
            }
            let n = rows.len();
            let stated =
                rows.iter().map(|b| b.confidence_bp as f64).sum::<f64>() / n as f64 / 10_000.0;
            let observed = rows
                .iter()
                .filter(|b| b.outcome == OUTCOME_CORRECT)
                .count() as f64
                / n as f64;
            out.push((
                format!("{:.0}-{}%", lo as f64 / 100.0, (hi - 1) as f64 / 100.0),
                n,
                stated,
                observed,
            ));
        }
        out
    }

    pub fn unresolved_count(&self) -> usize {
        self.bets
            .iter()
            .filter(|b| b.outcome == OUTCOME_UNRESOLVED)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_resolve_and_scar() {
        let mut l = OutcomeLedger::new();
        let seq = l.file("What is two plus two", "bookkeeping def", 1, CONFIDENCE_TIER1_BP, 1);
        assert!(!l.scarred("What is two plus two"));
        assert!(l.resolve(seq, OUTCOME_INCORRECT, SOURCE_USER, 2));
        assert!(l.scarred("What is two plus two"));
        assert!(!l.scarred("The capital of France is"));
        // double resolution refused
        assert!(!l.resolve(seq, OUTCOME_CORRECT, SOURCE_USER, 3));
    }

    #[test]
    fn resolve_latest_picks_newest_unresolved() {
        let mut l = OutcomeLedger::new();
        l.file("a", "1", 2, CONFIDENCE_TIER2_BP, 1);
        let s2 = l.file("b", "2", 2, CONFIDENCE_TIER2_BP, 2);
        assert_eq!(l.resolve_latest(OUTCOME_CORRECT, SOURCE_USER, 3), Some(s2));
        assert_eq!(l.unresolved_count(), 1);
    }

    #[test]
    fn calibration_reports_stated_vs_observed() {
        let mut l = OutcomeLedger::new();
        let s1 = l.file("p1", "a", 1, CONFIDENCE_TIER1_BP, 1);
        let s2 = l.file("p2", "b", 1, CONFIDENCE_TIER1_BP, 2);
        l.resolve(s1, OUTCOME_CORRECT, SOURCE_USER, 3);
        l.resolve(s2, OUTCOME_INCORRECT, SOURCE_USER, 4);
        let cal = l.calibration();
        assert_eq!(cal.len(), 1);
        let (_, n, stated, observed) = &cal[0];
        assert_eq!(*n, 2);
        assert!((stated - 1.0).abs() < 1e-9);
        assert!((observed - 0.5).abs() < 1e-9);
    }

    #[test]
    fn persistence_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("outcome_ledger_test_{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let mut l = OutcomeLedger::new();
            l.attach(path.clone()).unwrap();
            let s = l.file("q", "wrong", 1, CONFIDENCE_TIER1_BP, 10);
            l.resolve(s, OUTCOME_INCORRECT, SOURCE_USER, 11);
        }
        let mut l2 = OutcomeLedger::new();
        l2.attach(path.clone()).unwrap();
        assert!(l2.scarred("q"));
        assert_eq!(l2.unresolved_count(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fingerprint_distinguishes_part_boundaries() {
        assert_ne!(
            fingerprint(&[b"ab", b"c"]),
            fingerprint(&[b"a", b"bc"])
        );
    }
}
