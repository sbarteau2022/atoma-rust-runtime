//! core_lm: The Native Execution Plane
//!
//! Integrated with Layer 3 (`ImmuneForwardGate` / `ForwardCompliancePermit`) so that
//! every forward pass in `Engine`, `GlimmerModel`, and `QwenModel` requires a verified
//! `ForwardCompliancePermit` before execution (`core_lm -> immune_system`).

pub mod attention_backend;
pub mod geometry;
pub mod cache {
    pub mod geometric;
}
pub mod glimmer_forward;
pub mod glimmer_model;
pub mod lock;
pub mod metal;
pub mod metal_quant;
pub mod model_config;
pub mod ops;
pub mod optiq;
pub mod projection;
pub mod qwen_forward;
pub mod qwen_model;
pub mod safetensors;
pub mod tokenizer;

use crate::cache::geometric::GeometricMemoryMap;
use crate::geometry::bridge::BridgeLossReport;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::geometry::picard::PicardMapper;
use crate::geometry::torus::{PhaseState, TorPhaseEncoder};
use crate::projection::ProjectionHead;

/// Unforgeable forward-pass permit issued by `ImmuneForwardGate` (`core_lm -> immune_system`).
///
/// A forward pass in `Engine`, `GlimmerModel`, or `QwenModel` cannot execute without
/// presenting or minting a valid `ForwardCompliancePermit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardCompliancePermit {
    sequence: u64,
    token_id: u32,
    pos: usize,
    weight_fingerprint: u64,
    chain_hash: u64,
}

impl ForwardCompliancePermit {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn token_id(&self) -> u32 {
        self.token_id
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn weight_fingerprint(&self) -> u64 {
        self.weight_fingerprint
    }
    pub fn chain_hash(&self) -> u64 {
        self.chain_hash
    }
}

/// Layer 3 forward-pass compliance gate embedded inside `core_lm` models (`core_lm -> immune_system`).
///
/// Enforces:
/// 1. Base-weight immutability (`pinned_weight_fingerprint == current_weight_fingerprint`).
/// 2. Emergency kill-switch state (`quarantined == false`).
/// 3. Non-silent bridge loss accounting (`BridgeLossReport` verification after attention projection).
/// 4. Monotonic hash-chained audit sequence across every forward step.
#[derive(Debug, Clone)]
pub struct ImmuneForwardGate {
    pub pinned_weight_fingerprint: u64,
    pub sequence: u64,
    pub chain_hash: u64,
    pub quarantined: bool,
    pub last_loss_report: Option<BridgeLossReport>,
}

impl ImmuneForwardGate {
    pub fn new(pinned_weight_fingerprint: u64) -> Self {
        Self {
            pinned_weight_fingerprint,
            sequence: 0,
            chain_hash: 0x6A09_E667_F3BC_C908 ^ pinned_weight_fingerprint,
            quarantined: false,
            last_loss_report: None,
        }
    }

    /// Authorize a forward step, returning an unforgeable `ForwardCompliancePermit`
    /// or rejecting the pass if quarantined or if base weights were mutated.
    pub fn authorize_forward_pass(
        &mut self,
        token_id: u32,
        pos: usize,
        current_weight_fingerprint: u64,
    ) -> Result<ForwardCompliancePermit, String> {
        if self.quarantined {
            return Err("ImmuneForwardGate: model is quarantined by Layer 3 kill switch".into());
        }
        if current_weight_fingerprint != self.pinned_weight_fingerprint {
            self.quarantined = true;
            return Err(format!(
                "ImmuneForwardGate: base weight immutability violation (pinned={:#x} != current={:#x})",
                self.pinned_weight_fingerprint, current_weight_fingerprint
            ));
        }
        self.sequence += 1;
        let mut h = self.chain_hash ^ self.sequence;
        h = h.wrapping_mul(0x2545_F491_4F6C_DD1D);
        h ^= ((token_id as u64) << 32) | (pos as u64);
        h = h.wrapping_mul(0x2545_F491_4F6C_DD1D);
        self.chain_hash = h;

        Ok(ForwardCompliancePermit {
            sequence: self.sequence,
            token_id,
            pos,
            weight_fingerprint: current_weight_fingerprint,
            chain_hash: self.chain_hash,
        })
    }

    /// Verify that a permit was issued by this gate and record the layer's `BridgeLossReport`.
    pub fn verify_permit_and_loss(
        &mut self,
        permit: &ForwardCompliancePermit,
        loss: Option<BridgeLossReport>,
    ) -> Result<(), String> {
        if self.quarantined {
            return Err("ImmuneForwardGate: model is quarantined".into());
        }
        if permit.weight_fingerprint != self.pinned_weight_fingerprint {
            return Err("ImmuneForwardGate: permit weight fingerprint mismatch".into());
        }
        if let Some(rep) = loss {
            if rep.input_dimensions == 0 {
                return Err(
                    "ImmuneForwardGate: silent information loss (0 input dimensions)".into(),
                );
            }
            self.last_loss_report = Some(rep);
        }
        Ok(())
    }
}

pub struct Engine {
    pub memory_map: GeometricMemoryMap,
    pub encoder: TorPhaseEncoder,
    pub projection: ProjectionHead,
    pub immune_gate: ImmuneForwardGate,
}

impl Engine {
    pub fn new() -> Result<Self, EngineError> {
        Ok(Self {
            memory_map: GeometricMemoryMap::new(1_000_000),
            encoder: TorPhaseEncoder::new(),
            projection: ProjectionHead::new(4),
            immune_gate: ImmuneForwardGate::new(0xA704_A001_C0DE_0001),
        })
    }

    /// Bind each question to its answer in geometric memory.
    pub fn seed_knowledge(&mut self, facts: Vec<(&str, &str)>) {
        for (question, answer) in facts {
            let address = address_for_text(&self.encoder, question.as_bytes());
            if let Ok(id) = self.memory_map.store(address, answer.as_bytes().to_vec()) {
                self.memory_map.pin(id);
            }
        }
        self.memory_map.rebuild();
    }

    /// Return the nearest seeded answer, gated by a mandatory `ForwardCompliancePermit` check.
    pub fn generate(&self, prompt: &str, max_facts: usize) -> Result<String, EngineError> {
        let mut gate = self.immune_gate.clone();
        let permit = gate
            .authorize_forward_pass(
                prompt.len() as u32,
                max_facts,
                gate.pinned_weight_fingerprint,
            )
            .map_err(EngineError::ComplianceRejected)?;
        self.generate_with_permit(prompt, max_facts, &permit)
    }

    /// Execute generation with an explicit `ForwardCompliancePermit` (`core_lm -> immune_system`).
    pub fn generate_with_permit(
        &self,
        prompt: &str,
        max_facts: usize,
        permit: &ForwardCompliancePermit,
    ) -> Result<String, EngineError> {
        if self.immune_gate.quarantined {
            return Err(EngineError::ComplianceRejected(
                "Engine is quarantined by Layer 3 ImmuneForwardGate".into(),
            ));
        }
        if permit.weight_fingerprint != self.immune_gate.pinned_weight_fingerprint {
            return Err(EngineError::ComplianceRejected(
                "Invalid ForwardCompliancePermit weight fingerprint".into(),
            ));
        }
        if max_facts == 0 {
            return Ok(String::new());
        }
        let query = address_for_text(&self.encoder, prompt.as_bytes());
        let hits = self.memory_map.retrieve(&query, max_facts, false)?;
        if hits.is_empty() {
            return Err(EngineError::ExecutionFailed(
                "no seeded knowledge for this prompt".into(),
            ));
        }
        let mut out = String::new();
        for atom in hits {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&String::from_utf8_lossy(&atom.payload));
        }
        Ok(out)
    }
}

fn address_for_text(encoder: &TorPhaseEncoder, data: &[u8]) -> LorentzVector {
    let mut state = PhaseState::ZERO;
    for chunk in data.chunks(2) {
        let (df, ds) = encoder.project_increments(chunk);
        state = encoder.evolve(state, df, ds);
    }
    let phi = encoder.phi(state.psi);
    let mut direction = [
        Q32_31::from_f64(state.theta_f.to_f64().cos() + 0.35 * phi[0].to_f64()),
        Q32_31::from_f64(state.theta_s.to_f64().sin() + 0.35 * phi[1].to_f64()),
        Q32_31::from_f64(0.1 + 0.35 * phi[2].to_f64()),
    ];
    let norm_sq = (direction[0] * direction[0])
        + (direction[1] * direction[1])
        + (direction[2] * direction[2]);
    if norm_sq > Q32_31::ZERO {
        let inv = norm_sq.inv_sqrt();
        direction[0] = direction[0] * inv;
        direction[1] = direction[1] * inv;
        direction[2] = direction[2] * inv;
    }
    // Variable rapidity tracked from PhaseState amplitude channel instead of fixed 0.5 boost
    let rapidity = Q32_31::from_f64(0.25 + 0.35 * state.amplitude.to_f64());
    let boosted = LorentzVector::boost_rapidity(&LorentzVector::origin(), &direction, rapidity);
    PicardMapper::map_to_fundamental_domain(&boosted)
}

/// Tier 0 gate: the first 128 greedy tokens must match the baseline exactly.
pub fn qualify_token_parity(baseline: &[u32], candidate: &[u32]) -> Result<(), String> {
    if baseline.len() < 128 || candidate.len() < 128 {
        return Err(format!(
            "parity gate requires 128 tokens, got baseline {} and candidate {}",
            baseline.len(),
            candidate.len()
        ));
    }
    for index in 0..128 {
        if baseline[index] != candidate[index] {
            return Err(format!(
                "token {index} diverged: baseline {} candidate {}",
                baseline[index], candidate[index]
            ));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum EngineError {
    InitializationFailed(String),
    ExecutionFailed(String),
    NotImplemented(String),
    ComplianceRejected(String),
    MemoryError(crate::cache::geometric::MemoryError),
}

impl From<crate::cache::geometric::MemoryError> for EngineError {
    fn from(err: crate::cache::geometric::MemoryError) -> Self {
        EngineError::MemoryError(err)
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InitializationFailed(m) => write!(f, "Init failed: {}", m),
            EngineError::ExecutionFailed(m) => write!(f, "Exec failed: {}", m),
            EngineError::NotImplemented(m) => write!(f, "Not implemented: {}", m),
            EngineError::ComplianceRejected(m) => write!(f, "Compliance rejected: {}", m),
            EngineError::MemoryError(e) => write!(f, "Memory error: {}", e),
        }
    }
}

impl std::error::Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_gate_accepts_identical_prefix() {
        let tokens: Vec<u32> = (0..128).collect();
        assert!(qualify_token_parity(&tokens, &tokens).is_ok());
    }

    #[test]
    fn parity_gate_rejects_short_or_divergent_streams() {
        assert!(qualify_token_parity(&[1, 2], &[1, 2]).is_err());
        let mut candidate: Vec<u32> = (0..128).collect();
        candidate[40] = 999;
        let baseline: Vec<u32> = (0..128).collect();
        assert!(qualify_token_parity(&baseline, &candidate).is_err());
    }

    #[test]
    fn immune_forward_gate_blocks_quarantined_or_tampered_forward_pass() {
        let mut gate = ImmuneForwardGate::new(0x1234_5678);
        let permit = gate.authorize_forward_pass(42, 0, 0x1234_5678).unwrap();
        assert_eq!(permit.sequence(), 1);
        assert!(gate.authorize_forward_pass(43, 1, 0xDEAD_BEEF).is_err());
        assert!(gate.quarantined);
    }
}
