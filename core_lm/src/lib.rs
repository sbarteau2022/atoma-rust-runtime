//! core_lm: The Native Execution Plane

pub mod geometry;
pub mod cache {
    pub mod geometric;
}
pub mod glimmer_forward;
pub mod glimmer_model;
pub mod lock;
pub mod model_config;
pub mod ops;
pub mod optiq;
pub mod projection;
pub mod qwen_forward;
pub mod qwen_model;
pub mod safetensors;
pub mod tokenizer;

use crate::cache::geometric::GeometricMemoryMap;
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::geometry::picard::PicardMapper;
use crate::geometry::torus::TorPhaseEncoder;
use crate::projection::ProjectionHead;

pub struct Engine {
    pub memory_map: GeometricMemoryMap,
    pub encoder: TorPhaseEncoder,
    pub projection: ProjectionHead,
}

impl Engine {
    pub fn new() -> Result<Self, EngineError> {
        Ok(Self {
            memory_map: GeometricMemoryMap::new(1_000_000),
            encoder: TorPhaseEncoder::new(),
            projection: ProjectionHead::new(4),
        })
    }

    /// Bind each question to its answer in geometric memory.
    pub fn seed_knowledge(&mut self, facts: Vec<(&str, &str)>) {
        for (question, answer) in facts {
            let address = address_for_text(&self.encoder, question.as_bytes());
            if let Ok(id) = self
                .memory_map
                .store(address, answer.as_bytes().to_vec())
            {
                self.memory_map.pin(id);
            }
        }
        self.memory_map.rebuild();
    }

    /// Return the nearest seeded answer. `max_facts` is how many answers to join.
    pub fn generate(&self, prompt: &str, max_facts: usize) -> Result<String, EngineError> {
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
    let (fast, slow) = encoder.project_increments(data);
    let mut direction = [
        Q32_31::from_f64(fast.to_f64().cos()),
        Q32_31::from_f64(slow.to_f64().sin()),
        Q32_31::from_f64(0.1),
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
    let boosted = LorentzVector::boost(
        &LorentzVector::origin(),
        &direction,
        Q32_31::from_f64(0.5),
    );
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
}
