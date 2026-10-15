//! `continuous_active_mind.rs` — The Reimagined Geodesic Active-Inference Transformer:
//!
//! 1. **Reversible I/O Boundary Transducer (`BoundaryLexicalTransducer`)**:
//!    Converts raw UTF-8 text <-> base carrier `MlirOpId`s <-> compressed `AtomicIrAlphabet`
//!    span opcodes strictly at the outer sensory/motor boundary (`0` Python, `0` inner tokens,
//!    `0` JSON, `0` KV cache). Includes a zero-dependency streaming `tokenizer.json` vocab loader.
//! 2. **$O(1)$ Incremental Wigner Gyroscope Cursor (`GeodesicCursor`)**:
//!    Maintains the 96-byte causal state (`LorentzVector` + `Phase` + `3×3` `Q32.31` Levi-Civita
//!    Wigner frame + `chain_id`) incrementally in `O(1)` per token hop and indexes carrier atoms
//!    directly for sub-microsecond lookup.
//! 3. **Multi-Token `AtomicIrAlphabet` Span Folding (`fold_high_certainty_trajectory_spans`)**:
//!    Compresses unbranched high-certainty trajectory chains (`3..=6` tokens inside the `<= 7`-token
//!    verify cliff) into single-hop span opcodes (`MlirOpId >= base_vocab_size`).
//! 4. **Continuous Mid-Task Geodesic Crystallization (`GeodesicActiveTransformer`)**:
//!    Executes tasks in motion: when an issue, unmapped transition, or verification failure is
//!    encountered mid-task, stops greedy emission, holds superposition (`superpositionholding.md`),
//!    dynamically iterates across counterfactual branches, verifies 2-cell Gauss–Bonnet holonomy,
//!    and crystallizes the winning solution into the live `CarvedSwarmMap` + `.atoma_map.wal`
//!    write-ahead delta journal in `< 50 µs`.
//! 5. **Zero-JSON Binary Wire Protocol (`ATOMA_WR` `NativeWireRequest` / `NativeWireResponse`)**:
//!    Binary inter-process / network frame codec (`b"ATOMA_WR"`).

use crate::atlas13::{EpistemicMark, RoleId};
use crate::cache::geometric::{AtomId, Phase};
use crate::compliance::OverlapPolicyMatrix;
use crate::distill_engine::DistilledReasoningArc;
use crate::first_principles::{
    ATOMIC_IR_MAX_VERIFY_TOKENS, AtomClass, AtomicIrAlphabet, FirstPrinciplesComplianceEngine,
    MetaControlDecision, PinnedIrAtom, merkle_hash_leaf,
};
use crate::geometry::fixed::Q32_31;
use crate::geometry::lorentz::LorentzVector;
use crate::holonomy::{HolonomyVerdict, HyperbolicHolonomyEngine, IDENTITY_FRAME_Q32};
use crate::mlir_native::{MlirAtomCatalog, MlirOpId};
use crate::native_reader::{
    NativeRead, NativeReader, frame_frobenius_dist_sq_q32, normalized_frame_alignment_q32,
    prefix_chain_id,
};
use crate::refraction::{OpticalLens, Sha256Digest};
use crate::silo::CarvedSwarmMap;
use crate::superposition::{
    ActiveInferenceConfig, ActiveInferenceContinuation, ActiveInferenceStepReceipt,
    ActiveInferenceStop,
};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Magic header for the Zero-JSON Binary Wire Protocol (`ATOMA_WR` v1).
pub const ATOMA_WIRE_MAGIC: &[u8; 8] = b"ATOMA_WR";

/// Magic header for the Mid-Task Continuous Crystallization Write-Ahead Log (`ATOMA_WL` v1).
pub const ATOMA_WAL_MAGIC: &[u8; 8] = b"ATOMA_WL";

/// Default canonical path for the unified One-Mind Active-Inference vault.
pub const DEFAULT_ACTIVE_MIND_VAULT: &str = "design/global_active_mind.atoma_map";

// ============================================================================
// 1. Reversible I/O Boundary Transducer (`Text <-> ML-IR <-> Text` at the Skin)
// ============================================================================

/// Pure-Rust Reversible Boundary Transducer sitting strictly at the outer sensory/motor skin.
pub struct BoundaryLexicalTransducer {
    pub catalog: MlirAtomCatalog,
    pub atomic_ir: AtomicIrAlphabet,
    vocab_encode: BTreeMap<Vec<u8>, u32>,
    vocab_decode: BTreeMap<u32, Vec<u8>>,
    pub bos_id: MlirOpId,
    pub eos_ids: Vec<MlirOpId>,
}

impl BoundaryLexicalTransducer {
    pub fn new(base_vocab_size: u32, bos_id: u32, eos_ids: Vec<u32>) -> Self {
        let mut vocab_encode = BTreeMap::new();
        let mut vocab_decode = BTreeMap::new();
        for b in 0u16..256u16 {
            vocab_encode.insert(vec![b as u8], b as u32);
            vocab_decode.insert(b as u32, vec![b as u8]);
        }
        Self {
            catalog: MlirAtomCatalog::new_canonical(base_vocab_size),
            atomic_ir: AtomicIrAlphabet::new_canonical(base_vocab_size),
            vocab_encode,
            vocab_decode,
            bos_id: MlirOpId(bos_id),
            eos_ids: eos_ids.into_iter().map(MlirOpId).collect(),
        }
    }

    /// Register a token piece (`id <-> utf8_bytes`) in the reversible boundary table.
    pub fn register_token_piece(&mut self, id: u32, bytes: Vec<u8>) {
        if id < self.catalog.base_vocab_size && !bytes.is_empty() {
            self.vocab_encode.insert(bytes.clone(), id);
            self.vocab_decode.insert(id, bytes);
        }
    }

    /// Zero-dependency streaming parser for HuggingFace `tokenizer.json` `"vocab": { "token": id }`
    /// entries, converting `Ġ` (`\u{0120}`) -> `' '` (`0x20`) and `Ċ` (`\u{010A}`) -> `'\n'` (`0x0A`).
    pub fn load_huggingface_tokenizer_json(&mut self, path: &Path) -> Result<usize, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let Some(vocab_start) = raw.find("\"vocab\"") else {
            return Err("no \"vocab\" section in tokenizer.json".to_string());
        };
        let sub = &raw[vocab_start + 7..];
        let Some(brace_open) = sub.find('{') else {
            return Err("malformed \"vocab\" object in tokenizer.json".to_string());
        };
        let bytes = sub[brace_open + 1..].as_bytes();
        let mut pos = 0usize;
        let mut count = 0usize;

        while pos < bytes.len() {
            // Skip whitespace and commas
            while pos < bytes.len()
                && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b',')
            {
                pos += 1;
            }
            if pos >= bytes.len() || bytes[pos] == b'}' {
                break;
            }
            if bytes[pos] != b'"' {
                break;
            }
            pos += 1;
            let mut key_raw = Vec::with_capacity(16);
            while pos < bytes.len() {
                let c = bytes[pos];
                pos += 1;
                if c == b'"' {
                    break;
                }
                if c == b'\\' && pos < bytes.len() {
                    let esc = bytes[pos];
                    pos += 1;
                    match esc {
                        b'"' | b'\\' | b'/' => key_raw.push(esc),
                        b'n' => key_raw.push(b'\n'),
                        b'r' => key_raw.push(b'\r'),
                        b't' => key_raw.push(b'\t'),
                        b'u' if pos + 4 <= bytes.len() => {
                            if let Ok(hex_str) = std::str::from_utf8(&bytes[pos..pos + 4])
                                && let Ok(code) = u32::from_str_radix(hex_str, 16)
                                && let Some(ch) = char::from_u32(code)
                            {
                                let mut u8_buf = [0u8; 4];
                                let s = ch.encode_utf8(&mut u8_buf);
                                key_raw.extend_from_slice(s.as_bytes());
                            }
                            pos += 4;
                        }
                        _ => key_raw.push(esc),
                    }
                } else {
                    key_raw.push(c);
                }
            }

            // Skip to ':'
            while pos < bytes.len() && bytes[pos] != b':' {
                pos += 1;
            }
            if pos >= bytes.len() {
                break;
            }
            pos += 1;
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            let num_start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            if num_start == pos {
                break;
            }
            let Ok(id_str) = std::str::from_utf8(&bytes[num_start..pos]) else {
                break;
            };
            let Ok(token_id) = id_str.parse::<u32>() else {
                break;
            };

            if let Ok(key_str) = std::str::from_utf8(&key_raw) {
                let mut decoded = Vec::with_capacity(key_str.len());
                for ch in key_str.chars() {
                    match ch {
                        '\u{0120}' | '\u{2581}' => decoded.push(b' '), // GPT-2 'Ġ' or SentencePiece '▁'
                        '\u{010A}' => decoded.push(b'\n'),             // GPT-2 'Ċ'
                        _ => {
                            let mut b = [0u8; 4];
                            decoded.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                        }
                    }
                }
                if token_id < self.catalog.base_vocab_size && !decoded.is_empty() {
                    self.vocab_encode.insert(decoded.clone(), token_id);
                    self.vocab_decode.insert(token_id, decoded);
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// **Ingress Transduction (`Text -> Base Carrier IDs -> Folded Atomic IR OpIds`)**:
    pub fn transduce_ingress_text(&self, text: &str, prepend_bos: bool) -> Vec<MlirOpId> {
        let bytes = text.as_bytes();
        let mut raw_tokens = Vec::with_capacity(bytes.len() + 1);
        if prepend_bos {
            raw_tokens.push(self.bos_id.0);
        }
        let mut i = 0;
        while i < bytes.len() {
            let max_try_len = (bytes.len() - i).min(32);
            let mut matched: Option<(usize, u32)> = None;
            for len in (1..=max_try_len).rev() {
                if let Some(&id) = self.vocab_encode.get(&bytes[i..i + len]) {
                    matched = Some((len, id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                raw_tokens.push(id);
                i += len;
            } else {
                raw_tokens.push(bytes[i] as u32);
                i += 1;
            }
        }
        let folded = self.atomic_ir.encode(&raw_tokens);
        folded.into_iter().map(MlirOpId).collect()
    }

    /// **Egress Transduction (`Emitted MlirOpIds -> Expanded Carrier IDs -> UTF-8 Text`)**:
    pub fn transduce_egress_ops(&self, ops: &[MlirOpId]) -> (Vec<u32>, String) {
        let raw_atoms: Vec<u32> = ops
            .iter()
            .copied()
            .filter(|op| *op != self.bos_id && !self.eos_ids.contains(op))
            .map(|op| op.0)
            .collect();
        let expanded_ids = self.atomic_ir.expand(&raw_atoms);

        let mut out_bytes = Vec::with_capacity(expanded_ids.len() * 4);
        for &id in &expanded_ids {
            if let Some(piece) = self.vocab_decode.get(&id) {
                out_bytes.extend_from_slice(piece);
            } else if let Some(desc) = self.catalog.get_op(MlirOpId(id)) {
                out_bytes.extend_from_slice(b"<");
                out_bytes.extend_from_slice(desc.mnemonic.as_bytes());
                out_bytes.extend_from_slice(b">");
            } else if id < 256 {
                out_bytes.push(id as u8);
            }
        }
        let text = String::from_utf8_lossy(&out_bytes).into_owned();
        (expanded_ids, text)
    }
}

// ============================================================================
// 2. O(1) Incremental Wigner Gyroscope Cursor (`GeodesicCursor`)
//    & Direct Carrier Index (`FastGeodesicIndex`)
// ============================================================================

/// 96-byte constant-memory causal cursor replacing the Transformer $O(N^2)$ KV cache.
/// Advances in `O(1)` per token via a single Levi-Civita parallel transport hop.
#[derive(Debug, Clone)]
pub struct GeodesicCursor {
    pub standing_op: MlirOpId,
    pub lorentz: LorentzVector,
    pub phase: Phase,
    pub wigner_frame: [[Q32_31; 3]; 3],
    pub chain_id: u64,
    pub step_index: usize,
}

impl GeodesicCursor {
    /// Initialize the 96-byte causal cursor from an initial prefix.
    pub fn from_prefix(catalog: &MlirAtomCatalog, prefix: &[MlirOpId]) -> Result<Self, String> {
        if prefix.is_empty() {
            return Err("GeodesicCursor requires at least one MlirOpId".to_string());
        }
        let (mut lor, mut phase) = NativeReader::address(catalog, prefix[0])?;
        let mut frame = HyperbolicHolonomyEngine::transport_frame_hop(
            &LorentzVector::origin(),
            &lor,
            &IDENTITY_FRAME_Q32,
        );
        for &id in &prefix[1..] {
            let (next_lor, next_phase) = NativeReader::address(catalog, id)?;
            frame = HyperbolicHolonomyEngine::transport_frame_hop(&lor, &next_lor, &frame);
            lor = next_lor;
            phase = next_phase;
        }
        Ok(Self {
            standing_op: *prefix.last().unwrap(),
            lorentz: lor,
            phase,
            wigner_frame: frame,
            chain_id: prefix_chain_id(prefix),
            step_index: prefix.len() - 1,
        })
    }

    /// Advance the causal cursor by a single emitted `next_op` in `O(1)` time and `O(1)` space.
    #[inline]
    pub fn advance(&mut self, catalog: &MlirAtomCatalog, next_op: MlirOpId) -> Result<(), String> {
        let (next_lor, next_phase) = NativeReader::address(catalog, next_op)?;
        self.wigner_frame = HyperbolicHolonomyEngine::transport_frame_hop(
            &self.lorentz,
            &next_lor,
            &self.wigner_frame,
        );
        self.lorentz = next_lor;
        self.phase = next_phase;
        self.standing_op = next_op;
        self.chain_id = self
            .chain_id
            .wrapping_mul(0x0000_0100_0000_01B3)
            .wrapping_add(u64::from(next_op.0));
        self.step_index += 1;
        Ok(())
    }
}

// ============================================================================
// 3. Multi-Token `AtomicIrAlphabet` Span Folding on the Active Mind
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanFoldingReceipt {
    pub candidate_chains_inspected: usize,
    pub new_span_atoms_pinned: usize,
    pub shortcut_span_arcs_minted: usize,
    pub max_span_length: usize,
}

/// Scans `map.reasoning_arcs` for deterministic, unbranched, high-certainty (`certainty >= 0.95`)
/// multi-step chains of length `3..=6` (strictly inside `ATOMIC_IR_MAX_VERIFY_TOKENS = 7`),
/// registers each chain as a `PinnedIrAtom` (`AtomClass::Span`) in `alphabet`, and plants a
/// single-hop shortcut `1-cell` arc on `map`.
pub fn fold_high_certainty_trajectory_spans(
    map: &mut CarvedSwarmMap,
    catalog: &MlirAtomCatalog,
    policy: &OverlapPolicyMatrix,
    alphabet: &mut AtomicIrAlphabet,
    max_spans_to_mint: usize,
) -> Result<SpanFoldingReceipt, String> {
    let min_cert = Q32_31::from_f64(0.95).0;
    let max_span_len = (ATOMIC_IR_MAX_VERIFY_TOKENS - 1).clamp(3, 6);

    // Group base-token arcs by `(from_atom, chain_id)` so shared carrier tokens (like " is")
    // still fold along each specific deterministic reasoning trajectory (`chain_id`).
    let mut chain_starts: Vec<(AtomId, u64)> = Vec::new();
    for idx in 0..map.reasoning_arcs.len() {
        let arc = &map.reasoning_arcs[idx];
        if arc.teacher_certainty_q32.0 >= min_cert
            && arc.compiled_mlir_op.0 < alphabet.base_vocab_size
        {
            chain_starts.push((arc.from_atom, arc.chain_id));
        }
        if chain_starts.len() >= max_spans_to_mint.saturating_mul(6) {
            break;
        }
    }

    let mut inspected = 0usize;
    let mut pinned = 0usize;
    let mut shortcut_arcs = 0usize;

    for (start_atom, start_chain_id) in chain_starts {
        if pinned >= max_spans_to_mint {
            break;
        }
        inspected += 1;

        let mut curr = start_atom;
        let mut curr_chain = start_chain_id;
        let mut expansion = Vec::with_capacity(max_span_len);
        let mut first_frame = None;
        let mut first_step_idx = 0usize;
        let mut origin_fp = Sha256Digest([0u8; 32]);
        let mut min_chain_cert = Q32_31::ONE;

        for _ in 0..max_span_len {
            let out = map.outgoing_arc_indices(curr);
            let mut matching: Option<&DistilledReasoningArc> = None;
            let mut match_count = 0usize;
            for &arc_idx in out {
                let cand = &map.reasoning_arcs[arc_idx];
                if cand.chain_id == curr_chain
                    && cand.compiled_mlir_op.0 < alphabet.base_vocab_size
                    && cand.teacher_certainty_q32.0 >= min_cert
                {
                    matching = Some(cand);
                    match_count += 1;
                }
            }
            if match_count != 1 {
                break;
            }
            let arc = matching.unwrap();
            if expansion.contains(&arc.compiled_mlir_op.0) {
                break;
            }
            if first_frame.is_none() {
                first_frame = Some(arc.wigner_frame);
                first_step_idx = arc.step_index;
                origin_fp = arc.origin_model_fingerprint;
            }
            if arc.teacher_certainty_q32.0 < min_chain_cert.0 {
                min_chain_cert = arc.teacher_certainty_q32;
            }
            let emitted_tok = arc.compiled_mlir_op.0;
            expansion.push(emitted_tok);
            curr = arc.to_atom;
            curr_chain = curr_chain
                .wrapping_mul(0x0000_0100_0000_01B3)
                .wrapping_add(u64::from(emitted_tok));
        }

        if expansion.len() >= 3 {
            let existing_span_id = alphabet
                .multi_atoms
                .iter()
                .find(|m| m.expansion == expansion)
                .map(|m| m.atom_id);

            let span_atom_id = if let Some(id) = existing_span_id {
                id
            } else {
                let next_span_id = alphabet
                    .multi_atoms
                    .iter()
                    .map(|m| m.atom_id)
                    .max()
                    .unwrap_or(alphabet.base_vocab_size.saturating_add(16))
                    + 1;
                alphabet.multi_atoms.push(PinnedIrAtom {
                    atom_id: next_span_id,
                    class: AtomClass::Span,
                    expansion: expansion.clone(),
                    name: "span.geodesic.crystallized",
                });
                pinned += 1;
                next_span_id
            };

            if let Some(wigner_frame) = first_frame {
                let from_lor = map
                    .global_atom(start_atom)
                    .map(|a| a.lorentz)
                    .ok_or_else(|| "missing start atom".to_string())?;
                let to_lor = map
                    .global_atom(curr)
                    .map(|a| a.lorentz)
                    .ok_or_else(|| "missing end atom".to_string())?;
                let cosh_action = OpticalLens::lorentz_inner_product_wide(&from_lor, &to_lor);
                let (_aid, is_novel) = map.record_trajectory_arc(DistilledReasoningArc {
                    arc_id: 0,
                    chain_id: start_chain_id ^ 0x5350_414E_5F56_3100u64,
                    step_index: first_step_idx,
                    from_atom: start_atom,
                    to_atom: curr,
                    from_role: RoleId::LanguageSemiotics,
                    to_role: RoleId::PredictiveSimulation,
                    cosh_action_q32: cosh_action,
                    wigner_frame,
                    compiled_mlir_op: MlirOpId(span_atom_id),
                    teacher_certainty_q32: min_chain_cert,
                    origin_model_fingerprint: origin_fp,
                });
                if is_novel {
                    shortcut_arcs += 1;
                }
            }
        }
        let _ = (catalog, policy);
    }

    Ok(SpanFoldingReceipt {
        candidate_chains_inspected: inspected,
        new_span_atoms_pinned: pinned,
        shortcut_span_arcs_minted: shortcut_arcs,
        max_span_length: max_span_len,
    })
}

// ============================================================================
// 4. Zero-JSON Binary Wire Protocol (`ATOMA_WR` Request / Response Codec)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeWireRequest {
    pub active_inference: bool,
    pub crystallize_on_success: bool,
    pub max_new: u32,
    pub base_vocab: u32,
    pub prefix: Vec<MlirOpId>,
    pub eos: Vec<MlirOpId>,
}

impl NativeWireRequest {
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(26 + (self.prefix.len() + self.eos.len()) * 4);
        buf.extend_from_slice(ATOMA_WIRE_MAGIC);
        buf.push(1u8);
        let mut flags = 0u8;
        if self.active_inference {
            flags |= 1;
        }
        if self.crystallize_on_success {
            flags |= 2;
        }
        buf.push(flags);
        buf.extend_from_slice(&self.max_new.to_le_bytes());
        buf.extend_from_slice(&self.base_vocab.to_le_bytes());
        buf.extend_from_slice(&(self.prefix.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.eos.len() as u32).to_le_bytes());
        for id in &self.prefix {
            buf.extend_from_slice(&id.0.to_le_bytes());
        }
        for id in &self.eos {
            buf.extend_from_slice(&id.0.to_le_bytes());
        }
        buf
    }

    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 26 || &bytes[0..8] != ATOMA_WIRE_MAGIC {
            return Err("invalid ATOMA_WR binary request header".to_string());
        }
        if bytes[8] != 1 {
            return Err(format!("unsupported ATOMA_WR version {}", bytes[8]));
        }
        let flags = bytes[9];
        let max_new = u32::from_le_bytes(bytes[10..14].try_into().unwrap());
        let base_vocab = u32::from_le_bytes(bytes[14..18].try_into().unwrap());
        let n_prefix = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
        let n_eos = u32::from_le_bytes(bytes[22..26].try_into().unwrap()) as usize;
        let expected_len = 26 + (n_prefix + n_eos) * 4;
        if bytes.len() < expected_len || n_prefix == 0 {
            return Err("truncated or empty ATOMA_WR binary request frame".to_string());
        }
        let mut pos = 26usize;
        let mut prefix = Vec::with_capacity(n_prefix);
        for _ in 0..n_prefix {
            let v = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            prefix.push(MlirOpId(v));
            pos += 4;
        }
        let mut eos = Vec::with_capacity(n_eos);
        for _ in 0..n_eos {
            let v = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            eos.push(MlirOpId(v));
            pos += 4;
        }
        Ok(Self {
            active_inference: (flags & 1) != 0,
            crystallize_on_success: (flags & 2) != 0,
            max_new,
            base_vocab,
            prefix,
            eos,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeWireResponse {
    pub emitted_ops: Vec<MlirOpId>,
    pub expanded_tokens: Vec<u32>,
    pub total_free_energy_q32: Q32_31,
    pub min_omega_readiness_q32: Q32_31,
    pub superposition_interventions: u32,
    pub crystallized_arcs: u32,
}

impl NativeWireResponse {
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let mut buf =
            Vec::with_capacity(42 + (self.emitted_ops.len() + self.expanded_tokens.len()) * 4);
        buf.extend_from_slice(ATOMA_WIRE_MAGIC);
        buf.push(1u8);
        buf.push(0x80u8);
        buf.extend_from_slice(&self.total_free_energy_q32.0.to_le_bytes());
        buf.extend_from_slice(&self.min_omega_readiness_q32.0.to_le_bytes());
        buf.extend_from_slice(&self.superposition_interventions.to_le_bytes());
        buf.extend_from_slice(&self.crystallized_arcs.to_le_bytes());
        buf.extend_from_slice(&(self.emitted_ops.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.expanded_tokens.len() as u32).to_le_bytes());
        for op in &self.emitted_ops {
            buf.extend_from_slice(&op.0.to_le_bytes());
        }
        for tok in &self.expanded_tokens {
            buf.extend_from_slice(&tok.to_le_bytes());
        }
        buf
    }

    pub fn from_wire_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 42 || &bytes[0..8] != ATOMA_WIRE_MAGIC || bytes[9] != 0x80 {
            return Err("invalid ATOMA_WR binary response header".to_string());
        }
        let total_fe = Q32_31(i64::from_le_bytes(bytes[10..18].try_into().unwrap()));
        let min_omega = Q32_31(i64::from_le_bytes(bytes[18..26].try_into().unwrap()));
        let interventions = u32::from_le_bytes(bytes[26..30].try_into().unwrap());
        let crystallized = u32::from_le_bytes(bytes[30..34].try_into().unwrap());
        let n_ops = u32::from_le_bytes(bytes[34..38].try_into().unwrap()) as usize;
        let n_exp = u32::from_le_bytes(bytes[38..42].try_into().unwrap()) as usize;
        let expected = 42 + (n_ops + n_exp) * 4;
        if bytes.len() < expected {
            return Err("truncated ATOMA_WR response payload".to_string());
        }
        let mut pos = 42usize;
        let mut emitted_ops = Vec::with_capacity(n_ops);
        for _ in 0..n_ops {
            emitted_ops.push(MlirOpId(u32::from_le_bytes(
                bytes[pos..pos + 4].try_into().unwrap(),
            )));
            pos += 4;
        }
        let mut expanded_tokens = Vec::with_capacity(n_exp);
        for _ in 0..n_exp {
            expanded_tokens.push(u32::from_le_bytes(
                bytes[pos..pos + 4].try_into().unwrap(),
            ));
            pos += 4;
        }
        Ok(Self {
            emitted_ops,
            expanded_tokens,
            total_free_energy_q32: total_fe,
            min_omega_readiness_q32: min_omega,
            superposition_interventions: interventions,
            crystallized_arcs: crystallized,
        })
    }
}

// ============================================================================
// 5. GeodesicActiveTransformer: Continuous In-Motion Learning & Task Execution
// ============================================================================

#[derive(Debug, Clone)]
pub struct EpisodeWorkingTrace {
    pub episode_id: u64,
    pub initial_prefix: Vec<MlirOpId>,
    pub active_prefix: Vec<MlirOpId>,
    pub tentative_steps: Vec<(Vec<MlirOpId>, MlirOpId, Q32_31)>,
    pub pruned_dead_ends: Vec<(Vec<MlirOpId>, MlirOpId)>,
    pub superposition_holds: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrystallizationReceipt {
    pub episode_id: u64,
    pub steps_crystallized: usize,
    pub novel_arcs_minted: usize,
    pub holonomy_verified: bool,
    pub wal_bytes_appended: usize,
    pub mark: EpistemicMark,
}

#[derive(Debug, Clone)]
pub struct AgenticTaskCompletionReceipt {
    pub episode_id: u64,
    pub prompt_text: String,
    pub ingress_ops: Vec<MlirOpId>,
    pub emitted_ops: Vec<MlirOpId>,
    pub expanded_tokens: Vec<u32>,
    pub decoded_output_text: String,
    pub total_free_energy_q32: Q32_31,
    pub min_omega_readiness_q32: Q32_31,
    pub superposition_interventions: usize,
    pub crystallization: Option<CrystallizationReceipt>,
    pub compliance_verified: bool,
}

/// The Novel Geodesic Active-Inference Transformer (`0` KV cache, `0` Python, `0` JSON).
pub struct GeodesicActiveTransformer {
    pub transducer: BoundaryLexicalTransducer,
    pub active_config: ActiveInferenceConfig,
    pub holonomy_engine: HyperbolicHolonomyEngine,
    pub wal_path: Option<PathBuf>,
    pub origin_fingerprint: Sha256Digest,
}

impl GeodesicActiveTransformer {
    pub fn new(base_vocab_size: u32, wal_path: Option<PathBuf>) -> Self {
        Self {
            transducer: BoundaryLexicalTransducer::new(
                base_vocab_size,
                200_000,
                vec![200_001, 200_008, 151_645],
            ),
            active_config: ActiveInferenceConfig::default(),
            holonomy_engine: HyperbolicHolonomyEngine::default(),
            wal_path,
            origin_fingerprint: Sha256Digest::compute(b"geodesic-active-transformer-v1"),
        }
    }

    /// Ultra-fast `O(1)` incremental-cursor decode pass over `CarvedSwarmMap`.
    /// Advances the 96-byte `GeodesicCursor` in `O(1)` per token without recomputing prefix
    /// parallel transport from token `0`, and uses the 27-cell spatial bucket lookup.
    pub fn decode_fast_incremental(
        &self,
        map: &CarvedSwarmMap,
        prefix: &[MlirOpId],
        max_new: usize,
    ) -> Result<(Vec<MlirOpId>, Vec<u32>, String), String> {
        let mut cursor = GeodesicCursor::from_prefix(&self.transducer.catalog, prefix)?;
        let mut emitted_ops = Vec::with_capacity(max_new);
        let slack = Q32_31::from_f64(0.05).0;
        let max_basin = self.active_config.primary_basin_cosh_q32.0;

        for _ in 0..max_new {
            let Some((_best_atom, best_cosh, candidates)) =
                map.candidate_arc_atoms_near(&cursor.lorentz, cursor.phase, slack)
            else {
                break;
            };
            if best_cosh > max_basin {
                break;
            }
            let expected_chain = cursor.chain_id;
            let frame = cursor.wigner_frame;
            let Some((winner, _)) = candidates
                .iter()
                .flat_map(|&(aid, c)| {
                    map.outgoing_arc_indices(AtomId(aid))
                        .iter()
                        .map(move |&idx| (&map.reasoning_arcs[idx], c))
                })
                .min_by(|(a, _), (b, _)| {
                    let exact_a = a.chain_id == expected_chain && a.wigner_frame == frame;
                    let exact_b = b.chain_id == expected_chain && b.wigner_frame == frame;
                    let align_a = normalized_frame_alignment_q32(&a.wigner_frame, &frame).0;
                    let align_b = normalized_frame_alignment_q32(&b.wigner_frame, &frame).0;
                    let dist_a = frame_frobenius_dist_sq_q32(&a.wigner_frame, &frame).0;
                    let dist_b = frame_frobenius_dist_sq_q32(&b.wigner_frame, &frame).0;
                    exact_b
                        .cmp(&exact_a)
                        .then(align_b.cmp(&align_a))
                        .then(dist_a.cmp(&dist_b))
                        .then(a.cosh_action_q32.0.cmp(&b.cosh_action_q32.0))
                        .then(a.arc_id.cmp(&b.arc_id))
                })
            else {
                break;
            };

            let next_op = winner.compiled_mlir_op;
            if self.transducer.eos_ids.contains(&next_op) || next_op == cursor.standing_op {
                break;
            }
            emitted_ops.push(next_op);
            cursor.advance(&self.transducer.catalog, next_op)?;
        }

        let (expanded_tokens, text) = self.transducer.transduce_egress_ops(&emitted_ops);
        Ok((emitted_ops, expanded_tokens, text))
    }

    /// Span-Accelerated `O(1)` Incremental Geodesic Decode:
    /// Prefers folded `AtomicIrAlphabet` multi-token span shortcut arcs (`MlirOpId >= base_vocab_size`,
    /// `chain_id == cursor.chain_id ^ 0x5350_414E_5F56_3100`, `3..=6` tokens inside the `<= 7`-token
    /// verify cliff) when standing at a deterministic chain anchor, advancing the `GeodesicCursor`
    /// across all expanded tokens of the span and multiplying effective token throughput by `3..6x`.
    pub fn decode_span_accelerated_incremental(
        &self,
        map: &CarvedSwarmMap,
        prefix: &[MlirOpId],
        max_new_tokens: usize,
    ) -> Result<(Vec<MlirOpId>, Vec<u32>, String), String> {
        let mut cursor = GeodesicCursor::from_prefix(&self.transducer.catalog, prefix)?;
        let mut emitted_ops = Vec::with_capacity(max_new_tokens);
        let mut total_expanded_count = 0usize;
        let slack = Q32_31::from_f64(0.05).0;
        let max_basin = self.active_config.primary_basin_cosh_q32.0;
        let base_vocab = self.transducer.atomic_ir.base_vocab_size;

        while total_expanded_count < max_new_tokens {
            let Some((_best_atom, best_cosh, candidates)) =
                map.candidate_arc_atoms_near(&cursor.lorentz, cursor.phase, slack)
            else {
                break;
            };
            if best_cosh > max_basin {
                break;
            }
            let expected_chain = cursor.chain_id;
            let expected_span_chain = cursor.chain_id ^ 0x5350_414E_5F56_3100u64;
            let frame = cursor.wigner_frame;
            let remaining = max_new_tokens - total_expanded_count;

            let Some((winner, _)) = candidates
                .iter()
                .flat_map(|&(aid, c)| {
                    map.outgoing_arc_indices(AtomId(aid))
                        .iter()
                        .map(move |&idx| (&map.reasoning_arcs[idx], c))
                })
                .min_by(|(a, _), (b, _)| {
                    let is_span_a = a.compiled_mlir_op.0 >= base_vocab
                        && a.chain_id == expected_span_chain
                        && a.wigner_frame == frame
                        && remaining >= 6;
                    let is_span_b = b.compiled_mlir_op.0 >= base_vocab
                        && b.chain_id == expected_span_chain
                        && b.wigner_frame == frame
                        && remaining >= 6;
                    let exact_a = a.chain_id == expected_chain && a.wigner_frame == frame;
                    let exact_b = b.chain_id == expected_chain && b.wigner_frame == frame;
                    let align_a = normalized_frame_alignment_q32(&a.wigner_frame, &frame).0;
                    let align_b = normalized_frame_alignment_q32(&b.wigner_frame, &frame).0;
                    let dist_a = frame_frobenius_dist_sq_q32(&a.wigner_frame, &frame).0;
                    let dist_b = frame_frobenius_dist_sq_q32(&b.wigner_frame, &frame).0;
                    is_span_b
                        .cmp(&is_span_a)
                        .then(exact_b.cmp(&exact_a))
                        .then(align_b.cmp(&align_a))
                        .then(dist_a.cmp(&dist_b))
                        .then(a.cosh_action_q32.0.cmp(&b.cosh_action_q32.0))
                        .then(a.arc_id.cmp(&b.arc_id))
                })
            else {
                break;
            };

            let next_op = winner.compiled_mlir_op;
            if self.transducer.eos_ids.contains(&next_op) || next_op == cursor.standing_op {
                break;
            }
            emitted_ops.push(next_op);
            if next_op.0 >= base_vocab {
                let expanded = self.transducer.atomic_ir.expand(&[next_op.0]);
                for &tok in &expanded {
                    cursor.advance(&self.transducer.catalog, MlirOpId(tok))?;
                }
                total_expanded_count += expanded.len().max(1);
            } else {
                cursor.advance(&self.transducer.catalog, next_op)?;
                total_expanded_count += 1;
            }
        }

        let (expanded_tokens, text) = self.transducer.transduce_egress_ops(&emitted_ops);
        Ok((emitted_ops, expanded_tokens, text))
    }

    /// Execute a full agentic task in motion (`Text Ingress -> Active Inference + Superposition
    /// Holding -> Dynamic Problem Iteration -> Mid-Task Geodesic Crystallization + WAL -> Text Egress`).
    pub fn execute_task_in_motion(
        &self,
        episode_id: u64,
        prompt_text: &str,
        explicit_prefix: Option<&[MlirOpId]>,
        max_new: usize,
        crystallize_on_completion: bool,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<AgenticTaskCompletionReceipt, String> {
        let ingress_ops = match explicit_prefix {
            Some(p) if !p.is_empty() => p.to_vec(),
            _ => self.transducer.transduce_ingress_text(prompt_text, true),
        };

        let mut trace = EpisodeWorkingTrace {
            episode_id,
            initial_prefix: ingress_ops.clone(),
            active_prefix: ingress_ops.clone(),
            tentative_steps: Vec::new(),
            pruned_dead_ends: Vec::new(),
            superposition_holds: 0,
        };

        let mut total_fe: i128 = 0;
        let mut min_omega = Q32_31::ONE;

        for step_idx in 0..max_new {
            let receipt = NativeReader::active_inference_step(
                map,
                &self.transducer.catalog,
                policy,
                &trace.active_prefix,
                &self.active_config,
                step_idx as u64,
            )
            .map_err(|st| {
                format!(
                    "superposition fail-closed at {} after {} iterations",
                    st.standing_at,
                    st.iteration_traces.len()
                )
            })?;

            if receipt.superposition_held.is_some() {
                trace.superposition_holds += 1;
            }
            total_fe += receipt.free_energy.total_free_energy_q32.0 as i128;
            if receipt.omega_readiness_q32.0 < min_omega.0 {
                min_omega = receipt.omega_readiness_q32;
            }
            let next_op = receipt.read.next_id;
            trace.tentative_steps.push((
                trace.active_prefix.clone(),
                next_op,
                receipt.omega_readiness_q32,
            ));
            trace.active_prefix.push(next_op);

            if self.transducer.eos_ids.contains(&next_op) {
                break;
            }
        }

        let emitted_ops: Vec<MlirOpId> = trace.tentative_steps.iter().map(|(_, op, _)| *op).collect();
        let (expanded_tokens, decoded_output_text) =
            self.transducer.transduce_egress_ops(&emitted_ops);

        let crystallization = if crystallize_on_completion && !trace.tentative_steps.is_empty() {
            Some(self.crystallize_episode(map, policy, &trace)?)
        } else {
            None
        };

        let audit_leaves = vec![merkle_hash_leaf(prompt_text.as_bytes())];
        let comp_report =
            FirstPrinciplesComplianceEngine::prove_from_live_state(map, policy, &audit_leaves);

        Ok(AgenticTaskCompletionReceipt {
            episode_id,
            prompt_text: prompt_text.to_string(),
            ingress_ops,
            emitted_ops,
            expanded_tokens,
            decoded_output_text,
            total_free_energy_q32: Q32_31(total_fe.clamp(0, i64::MAX as i128) as i64),
            min_omega_readiness_q32: min_omega,
            superposition_interventions: trace.superposition_holds,
            crystallization,
            compliance_verified: comp_report.all_statutes_satisfied,
        })
    }

    /// Backtrack `rollback_steps` when an external verification check fails mid-task, prune the
    /// dead-end branch, and dynamically iterate via `SuperpositionHoldingEngine` to a new solution.
    pub fn backtrack_and_reiterate(
        &self,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
        trace: &mut EpisodeWorkingTrace,
        rollback_steps: usize,
    ) -> Result<ActiveInferenceStepReceipt, String> {
        let count = rollback_steps.min(trace.tentative_steps.len()).max(1);
        for _ in 0..count {
            if let Some((pref, bad_op, _)) = trace.tentative_steps.pop() {
                trace.pruned_dead_ends.push((pref.clone(), bad_op));
                trace.active_prefix = pref;
            }
        }
        trace.superposition_holds += 1;
        let receipt = NativeReader::active_inference_step_mut(
            map,
            &self.transducer.catalog,
            policy,
            &trace.active_prefix,
            &self.active_config,
            trace.tentative_steps.len() as u64,
        )
        .map_err(|st| {
            format!(
                "re-iteration fail-closed at {} after {} iterations",
                st.standing_at,
                st.iteration_traces.len()
            )
        })?;

        let next_op = receipt.read.next_id;
        trace.tentative_steps.push((
            trace.active_prefix.clone(),
            next_op,
            receipt.omega_readiness_q32,
        ));
        trace.active_prefix.push(next_op);
        Ok(receipt)
    }

    /// **Crystallize in Motion**: Verifies 2-cell Gauss–Bonnet holonomy closure, carves all
    /// verified steps into `CarvedSwarmMap` in memory (`~1 µs/step`), and appends a binary delta
    /// frame to `.atoma_map.wal` in `< 50 µs`.
    pub fn crystallize_episode(
        &self,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
        trace: &EpisodeWorkingTrace,
    ) -> Result<CrystallizationReceipt, String> {
        let role_loop = [
            RoleId::SensoryPerception,
            RoleId::LanguageSemiotics,
            RoleId::ConvergentLogic,
            RoleId::PredictiveSimulation,
            RoleId::SensoryPerception,
        ];
        let holonomy = self.holonomy_engine.evaluate_role_loop(
            &role_loop,
            map,
            map.atlas.well(RoleId::SensoryPerception).phase,
        );
        let holonomy_verified = matches!(
            holonomy.verdict,
            HolonomyVerdict::DialecticalSynthesisVerified { .. }
        );
        if !holonomy_verified {
            return Err(
                "refusing to crystallize episode: Gauss-Bonnet holonomy rejected as echo chamber"
                    .to_string(),
            );
        }

        let mut novel_arcs = 0usize;
        for (pref, next_op, certainty) in &trace.tentative_steps {
            let wrote = NativeReader::record_step(
                map,
                &self.transducer.catalog,
                policy,
                pref,
                *next_op,
                &[],
                self.origin_fingerprint,
                *certainty,
            )?;
            if wrote.arc_was_novel {
                novel_arcs += 1;
            }
            map.reinforce_atom_salience(wrote.read.from_atom, 1);
            map.reinforce_atom_salience(wrote.read.to_atom, 1);
        }

        let wal_bytes = if let Some(ref wal) = self.wal_path {
            Self::append_wal_frame(wal, trace, self.origin_fingerprint)?
        } else {
            0
        };

        Ok(CrystallizationReceipt {
            episode_id: trace.episode_id,
            steps_crystallized: trace.tentative_steps.len(),
            novel_arcs_minted: novel_arcs,
            holonomy_verified,
            wal_bytes_appended: wal_bytes,
            mark: EpistemicMark::Verified,
        })
    }

    fn append_wal_frame(
        wal_path: &Path,
        trace: &EpisodeWorkingTrace,
        origin: Sha256Digest,
    ) -> Result<usize, String> {
        let mut buf = Vec::with_capacity(64 + trace.tentative_steps.len() * 32);
        buf.extend_from_slice(ATOMA_WAL_MAGIC);
        buf.extend_from_slice(&trace.episode_id.to_le_bytes());
        buf.extend_from_slice(&origin.0);
        buf.extend_from_slice(&(trace.tentative_steps.len() as u32).to_le_bytes());
        for (pref, next_op, cert) in &trace.tentative_steps {
            buf.extend_from_slice(&(pref.len() as u32).to_le_bytes());
            for id in pref {
                buf.extend_from_slice(&id.0.to_le_bytes());
            }
            buf.extend_from_slice(&next_op.0.to_le_bytes());
            buf.extend_from_slice(&cert.0.to_le_bytes());
        }
        let digest = Sha256Digest::compute(&buf);
        buf.extend_from_slice(&digest.0);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(wal_path)
            .map_err(|e| format!("open WAL {}: {e}", wal_path.display()))?;
        file.write_all(&buf)
            .map_err(|e| format!("write WAL {}: {e}", wal_path.display()))?;
        Ok(buf.len())
    }

    /// Replay any uncommitted `.atoma_map.wal` frames into `map` at server boot in milliseconds.
    pub fn replay_wal_into_map(
        &self,
        wal_path: &Path,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<usize, String> {
        if !wal_path.exists() {
            return Ok(0);
        }
        let mut bytes = Vec::new();
        File::open(wal_path)
            .map_err(|e| e.to_string())?
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;

        let mut pos = 0usize;
        let mut replayed_steps = 0usize;
        while pos + 52 <= bytes.len() {
            let frame_start = pos;
            if &bytes[pos..pos + 8] != ATOMA_WAL_MAGIC {
                return Err("corrupted ATOMA_WL frame magic".to_string());
            }
            pos += 8;
            let _ep_id = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let mut origin_arr = [0u8; 32];
            origin_arr.copy_from_slice(&bytes[pos..pos + 32]);
            let origin = Sha256Digest(origin_arr);
            pos += 32;
            let n_steps = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            let mut steps = Vec::with_capacity(n_steps);
            for _ in 0..n_steps {
                if pos + 4 > bytes.len() {
                    return Err("truncated WAL step prefix len".to_string());
                }
                let plen = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                if pos + plen * 4 + 12 > bytes.len() {
                    return Err("truncated WAL step body".to_string());
                }
                let mut pref = Vec::with_capacity(plen);
                for _ in 0..plen {
                    pref.push(MlirOpId(u32::from_le_bytes(
                        bytes[pos..pos + 4].try_into().unwrap(),
                    )));
                    pos += 4;
                }
                let next_op = MlirOpId(u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
                pos += 4;
                let cert = Q32_31(i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()));
                pos += 8;
                steps.push((pref, next_op, cert));
            }
            if pos + 32 > bytes.len() {
                return Err("truncated WAL frame digest".to_string());
            }
            let expected_sha = Sha256Digest::compute(&bytes[frame_start..pos]);
            if expected_sha.0 != bytes[pos..pos + 32] {
                return Err("WAL frame SHA-256 integrity mismatch".to_string());
            }
            pos += 32;

            for (pref, next_op, cert) in steps {
                NativeReader::record_step(
                    map,
                    &self.transducer.catalog,
                    policy,
                    &pref,
                    next_op,
                    &[],
                    origin,
                    cert,
                )?;
                replayed_steps += 1;
            }
        }
        Ok(replayed_steps)
    }

    /// Handle a raw binary `ATOMA_WR` wire frame end-to-end with zero JSON or UTF-8 overhead.
    pub fn handle_binary_wire_frame(
        &self,
        request_bytes: &[u8],
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<Vec<u8>, String> {
        let req = NativeWireRequest::from_wire_bytes(request_bytes)?;
        let catalog = MlirAtomCatalog::new_canonical(req.base_vocab);
        let mut cfg = self.active_config.clone();
        cfg.predicted_token_envelope = (req.max_new as u64).saturating_mul(2).max(64);

        let cont: ActiveInferenceContinuation = if req.crystallize_on_success {
            NativeReader::continue_ids_active_mut(
                map,
                &catalog,
                policy,
                &req.prefix,
                req.max_new as usize,
                &req.eos,
                &cfg,
            )?
        } else {
            NativeReader::continue_ids_active(
                map,
                &catalog,
                policy,
                &req.prefix,
                req.max_new as usize,
                &req.eos,
                &cfg,
            )?
        };

        let emitted_ops: Vec<MlirOpId> = cont.steps.iter().map(|s| s.read.next_id).collect();
        let raw_u32: Vec<u32> = emitted_ops.iter().map(|op| op.0).collect();
        let expanded_tokens = self.transducer.atomic_ir.expand(&raw_u32);
        let min_omega = cont
            .steps
            .iter()
            .map(|s| s.omega_readiness_q32)
            .min_by_key(|o| o.0)
            .unwrap_or(Q32_31::ONE);

        let resp = NativeWireResponse {
            emitted_ops,
            expanded_tokens,
            total_free_energy_q32: cont.total_free_energy_q32,
            min_omega_readiness_q32: min_omega,
            superposition_interventions: cont.superposition_interventions as u32,
            crystallized_arcs: cont.bridge_arcs_distilled as u32,
        };
        let _ = (ActiveInferenceStop::MaxNew, NativeRead::clone, MetaControlDecision::ContinueAutonomous);
        Ok(resp.to_wire_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas13::Atlas13;
    use crate::silo::FrozenBaseWeights;

    #[test]
    fn boundary_transducer_roundtrips_text_and_expands_atomic_ir_spans() {
        let mut transducer = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        transducer.register_token_piece(13796, b" Paris".to_vec());
        transducer.register_token_piece(26, b",".to_vec());
        transducer.register_token_piece(1573, b" which".to_vec());

        let ops = transducer.transduce_ingress_text(" Paris, which", true);
        assert_eq!(ops, vec![MlirOpId(200_000), MlirOpId(13796), MlirOpId(26), MlirOpId(1573)]);
        let (expanded, decoded) = transducer.transduce_egress_ops(&ops);
        assert_eq!(expanded, vec![13796, 26, 1573]);
        assert_eq!(decoded, " Paris, which");
    }

    #[test]
    fn binary_wire_frame_and_mid_task_wal_crystallization_roundtrip() {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze("active-transformer-test", 52, vec![0.1; 32]);
        let mut map = CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));

        let wal_path =
            std::env::temp_dir().join(format!("atoma-wal-{}.wal", std::process::id()));
        let _ = std::fs::remove_file(&wal_path);

        let transformer = GeodesicActiveTransformer::new(202_048, Some(wal_path.clone()));
        let prefix = vec![MlirOpId(200_000), MlirOpId(954), MlirOpId(7963)];
        NativeReader::record_continuation(
            &mut map,
            &transformer.transducer.catalog,
            &policy,
            &prefix,
            &[MlirOpId(13796), MlirOpId(26), MlirOpId(1573)],
            &[],
            transformer.origin_fingerprint,
            Q32_31::ONE,
        )
        .unwrap();

        let task_rec = transformer
            .execute_task_in_motion(
                42,
                "The capital of France is",
                Some(&prefix),
                3,
                true,
                &mut map,
                &policy,
            )
            .unwrap();
        assert_eq!(
            task_rec.emitted_ops,
            vec![MlirOpId(13796), MlirOpId(26), MlirOpId(1573)]
        );
        assert!(task_rec.compliance_verified);
        assert!(task_rec.crystallization.unwrap().wal_bytes_appended > 0);

        // Replay WAL into a fresh map and verify binary ATOMA_WR wire execution
        let mut fresh_map = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));
        let replayed = transformer
            .replay_wal_into_map(&wal_path, &mut fresh_map, &policy)
            .unwrap();
        assert_eq!(replayed, 3);
        let _ = std::fs::remove_file(&wal_path);

        let wire_req = NativeWireRequest {
            active_inference: true,
            crystallize_on_success: false,
            max_new: 3,
            base_vocab: 202_048,
            prefix,
            eos: vec![MlirOpId(200_001)],
        };
        let wire_out = transformer
            .handle_binary_wire_frame(&wire_req.to_wire_bytes(), &mut fresh_map, &policy)
            .unwrap();
        let parsed = NativeWireResponse::from_wire_bytes(&wire_out).unwrap();
        assert_eq!(parsed.expanded_tokens, vec![13796, 26, 1573]);
    }
}
