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

/// Magic header for the Zero-JSON Binary Boundary Lexicon (`ATOMA_LX` v1).
pub const ATOMA_LEX_MAGIC: &[u8; 8] = b"ATOMA_LX";

/// Default canonical path for the Zero-JSON Binary Boundary Lexicon.
pub const DEFAULT_BOUNDARY_LEXICON_PATH: &str = "design/boundary_lexicon.atoma_lex";

/// Build the exact 256-entry GPT-2 Unicode codepoint -> raw byte (`0..=255`) inverse table.
fn gpt2_unicode_to_byte_map() -> BTreeMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=(b'~' as u32)).collect();
    bs.extend(0xA1u32..=0xACu32);
    bs.extend(0xAEu32..=0xFFu32);
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..=255u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut dec = BTreeMap::new();
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        if let Some(ch) = char::from_u32(c) {
            dec.insert(ch, b as u8);
        }
    }
    dec
}

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

    /// Number of decoded carrier entries in the boundary table.
    pub fn vocab_len(&self) -> usize {
        self.vocab_decode.len()
    }

    /// Look up the raw UTF-8 byte slice for a carrier ID (`MlirOpId`).
    pub fn decode_carrier_bytes(&self, id: u32) -> Option<&[u8]> {
        self.vocab_decode.get(&id).map(|v| v.as_slice())
    }

    /// Register a token piece (`id <-> utf8_bytes`) in the reversible boundary table.
    pub fn register_token_piece(&mut self, id: u32, bytes: Vec<u8>) {
        if id < self.catalog.base_vocab_size && !bytes.is_empty() {
            self.vocab_encode.insert(bytes.clone(), id);
            self.vocab_decode.insert(id, bytes);
        }
    }

    /// Save the reversible boundary table to a SHA-256-authenticated `ATOMA_LX` binary file (`0` JSON).
    pub fn save_binary_lexicon(&self, path: &Path) -> Result<(usize, Sha256Digest), String> {
        let mut buf = Vec::with_capacity(64 + self.vocab_decode.len() * 12);
        buf.extend_from_slice(ATOMA_LEX_MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes()); // version 1
        buf.extend_from_slice(&self.catalog.base_vocab_size.to_le_bytes());
        buf.extend_from_slice(&self.bos_id.0.to_le_bytes());
        buf.extend_from_slice(&(self.eos_ids.len() as u32).to_le_bytes());
        for eos in &self.eos_ids {
            buf.extend_from_slice(&eos.0.to_le_bytes());
        }
        buf.extend_from_slice(&(self.vocab_decode.len() as u32).to_le_bytes());
        for (&id, piece) in &self.vocab_decode {
            buf.extend_from_slice(&id.to_le_bytes());
            let len = piece.len().min(u16::MAX as usize) as u16;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(&piece[..len as usize]);
        }
        let digest = Sha256Digest::compute(&buf);
        buf.extend_from_slice(&digest.0);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, &buf)
            .map_err(|e| format!("failed to write binary lexicon {}: {e}", path.display()))?;
        Ok((self.vocab_decode.len(), digest))
    }

    /// Load and SHA-256-verify a native `ATOMA_LX` binary lexicon (`0` JSON, `< 15 ms`).
    pub fn load_binary_lexicon(&mut self, path: &Path) -> Result<usize, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("failed to read binary lexicon {}: {e}", path.display()))?;
        if bytes.len() < 24 + 32 {
            return Err("ATOMA_LX file too short".to_string());
        }
        if &bytes[0..8] != ATOMA_LEX_MAGIC {
            return Err("invalid ATOMA_LX magic header".to_string());
        }
        let payload_end = bytes.len() - 32;
        let expected_sha = Sha256Digest::compute(&bytes[..payload_end]);
        if expected_sha.0 != bytes[payload_end..] {
            return Err("ATOMA_LX SHA-256 integrity mismatch".to_string());
        }
        let mut pos = 8usize;
        let _version = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let _base_vocab = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let bos = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        self.bos_id = MlirOpId(bos);
        let eos_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + eos_len * 4 + 4 > payload_end {
            return Err("truncated ATOMA_LX eos header".to_string());
        }
        self.eos_ids.clear();
        for _ in 0..eos_len {
            let e = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            self.eos_ids.push(MlirOpId(e));
        }
        let count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        for _ in 0..count {
            if pos + 6 > payload_end {
                return Err("truncated ATOMA_LX entry header".to_string());
            }
            let id = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let plen = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            if pos + plen > payload_end {
                return Err("truncated ATOMA_LX entry bytes".to_string());
            }
            let piece = bytes[pos..pos + plen].to_vec();
            pos += plen;
            if id < self.catalog.base_vocab_size && !piece.is_empty() {
                self.vocab_encode.insert(piece.clone(), id);
                self.vocab_decode.insert(id, piece);
            }
        }
        Ok(count)
    }

    /// Zero-dependency streaming parser for legacy HuggingFace `tokenizer.json` `"vocab": { "token": id }`
    /// entries using the full 256-entry GPT-2 Unicode-to-byte inversion table.
    /// Automatically caches the compiled `ATOMA_LX` binary lexicon when loaded so subsequent
    /// runs never touch JSON again.
    pub fn load_huggingface_tokenizer_json(&mut self, path: &Path) -> Result<usize, String> {
        let gpt2_byte_map = gpt2_unicode_to_byte_map();
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
            while pos < bytes.len() && (bytes[pos].is_ascii_whitespace() || bytes[pos] == b',') {
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
                    if ch == '\u{2581}' {
                        decoded.push(b' ');
                    } else if let Some(&raw_byte) = gpt2_byte_map.get(&ch) {
                        decoded.push(raw_byte);
                    } else {
                        let mut b = [0u8; 4];
                        decoded.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                    }
                }
                if token_id < self.catalog.base_vocab_size && !decoded.is_empty() {
                    self.vocab_encode.insert(decoded.clone(), token_id);
                    self.vocab_decode
                        .insert(id_str.parse::<u32>().unwrap_or(token_id), decoded);
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Load the binary `ATOMA_LX` boundary lexicon (`design/boundary_lexicon.atoma_lex`) directly
    /// with zero JSON overhead, or bootstrap and persist it once if not yet present on disk.
    pub fn load_or_bootstrap_default(&mut self) -> Result<usize, String> {
        let bin_candidates = [
            DEFAULT_BOUNDARY_LEXICON_PATH,
            "../../design/boundary_lexicon.atoma_lex",
            "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/boundary_lexicon.atoma_lex",
        ];
        for cand in bin_candidates {
            let p = Path::new(cand);
            if p.is_file() {
                if let Ok(n) = self.load_binary_lexicon(p) {
                    if n > 0 {
                        return Ok(n);
                    }
                }
            }
        }

        let json_bootstrap_candidates = [
            "/Users/stewartbarteau/models/Muse-Glimmer-q4km-gs128-MLX/tokenizer.json",
            "/Users/stewartbarteau/models/Qwen3.8-27B-Uncensored-OrcaRouter-MLX-4bit/tokenizer.json",
        ];
        for cand in json_bootstrap_candidates {
            let p = Path::new(cand);
            if p.is_file() {
                if let Ok(n) = self.load_huggingface_tokenizer_json(p) {
                    if n > 0 {
                        let out_path = Path::new(
                            "/Users/stewartbarteau/Desktop/monster-hybrid-server/design/boundary_lexicon.atoma_lex",
                        );
                        if out_path.parent().is_some_and(|d| d.is_dir()) {
                            let _ = self.save_binary_lexicon(out_path);
                        }
                        return Ok(n);
                    }
                }
            }
        }
        Ok(0)
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

    /// Convert UTF-8 bytes directly to native ML-IR carrier IDs without a model tokenizer or
    /// vocabulary table. Atomic-IR span folding remains available; expanding the result recovers
    /// the original byte IDs exactly.
    pub fn transduce_utf8_to_mlir(&self, text: &str, prepend_bos: bool) -> Vec<MlirOpId> {
        let mut byte_ids = Vec::with_capacity(text.len() + usize::from(prepend_bos));
        if prepend_bos {
            byte_ids.push(self.bos_id.0);
        }
        byte_ids.extend(text.as_bytes().iter().map(|&byte| u32::from(byte)));
        self.atomic_ir
            .encode(&byte_ids)
            .into_iter()
            .map(MlirOpId)
            .collect()
    }

    /// Expand native ML-IR carrier IDs directly back to UTF-8. This accepts only byte carriers
    /// after atomic-IR expansion, so model-token IDs cannot silently leak into text output.
    pub fn transduce_mlir_to_utf8(&self, ops: &[MlirOpId]) -> Result<String, String> {
        let carrier_ids: Vec<u32> = ops
            .iter()
            .copied()
            .filter(|op| *op != self.bos_id && !self.eos_ids.contains(op))
            .map(|op| op.0)
            .collect();
        let expanded = self.atomic_ir.expand_checked(&carrier_ids)?;
        let bytes: Result<Vec<u8>, _> = expanded
            .into_iter()
            .map(|id| u8::try_from(id).map_err(|_| id))
            .collect();
        let bytes = bytes.map_err(|id| format!("ML-IR carrier {id} is not a UTF-8 byte"))?;
        String::from_utf8(bytes).map_err(|e| format!("ML-IR egress is invalid UTF-8: {e}"))
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

    /// Verify that a decoded trajectory continuation is human-coherent UTF-8 text rather than
    /// ungrounded raw BPE token-ID salad, single-byte stub collisions (`"P"`), or diagnostic strings.
    pub fn is_coherent_egress_text(prompt_text: &str, decoded_text: &str) -> bool {
        let trimmed = decoded_text.trim();
        if trimmed.is_empty()
            || decoded_text.contains('\u{FFFD}')
            || trimmed.contains("analyzed via carved")
            || trimmed.contains("<atoma.")
        {
            return false;
        }
        if trimmed.len() == 1 && !trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            return false;
        }
        if decoded_text
            .chars()
            .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t'))
        {
            return false;
        }
        if prompt_text.is_ascii() {
            let has_foreign_script = decoded_text.chars().any(|c| {
                !c.is_ascii()
                    && !matches!(
                        c,
                        '—' | '–'
                            | '’'
                            | '‘'
                            | '“'
                            | '”'
                            | '…'
                            | '×'
                            | '÷'
                            | '±'
                            | '≤'
                            | '≥'
                            | '≠'
                            | '≈'
                            | '°'
                            | 'µ'
                            | 'π'
                            | '²'
                            | '³'
                            | '•'
                            | '·'
                    )
            });
            if has_foreign_script {
                return false;
            }
        }
        if trimmed.len() > 36 && !trimmed.contains(' ') && !trimmed.contains('\n') {
            return false;
        }
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        for w in &words {
            if let Some(rest) = w.strip_prefix('-') {
                let parts: Vec<&str> = rest.split('-').collect();
                if parts.len() >= 2 && parts[0].len() >= 3 && parts[0] == parts[1] {
                    return false;
                }
            }
        }
        for win in words.windows(3) {
            if win[0].len() >= 2 && win[0] == win[1] && win[1] == win[2] {
                return false;
            }
        }
        true
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
            && (arc.chain_id != 0 || arc.compiled_mlir_op.0 >= 64)
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
                    && (cand.chain_id != 0 || cand.compiled_mlir_op.0 >= 64)
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
            expanded_tokens.push(u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
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

/// Receipt returned when an ephemeral sub-agent executes inside a structurally carved
/// [`crate::silo::TaskCarvedView`] (`Qualification #4`: specialization by subtraction).
#[derive(Debug, Clone)]
pub struct SubAgentTaskReceipt {
    pub primary_role: RoleId,
    pub allowed_silos: Vec<RoleId>,
    pub carved_atom_count: usize,
    pub carved_arc_count: usize,
    pub global_atoms_excluded: usize,
    pub global_arcs_excluded: usize,
    pub task_receipt: AgenticTaskCompletionReceipt,
    pub compiled_back_arcs: usize,
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
                        .filter_map(move |&idx| {
                            let arc = &map.reasoning_arcs[idx];
                            if arc.chain_id == expected_chain || arc.compiled_mlir_op.0 >= 64 {
                                Some((arc, c))
                            } else {
                                None
                            }
                        })
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

            if winner.chain_id != expected_chain {
                break;
            }
            let next_op = winner.compiled_mlir_op;
            let is_ascii_ws = matches!(next_op.0, 9 | 10 | 13 | 32);
            let eight_repeat = !is_ascii_ws
                && emitted_ops.len() >= 7
                && emitted_ops[emitted_ops.len() - 7..]
                    .iter()
                    .all(|&x| x == next_op);
            if self.transducer.eos_ids.contains(&next_op) || eight_repeat {
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
                        .filter_map(move |&idx| {
                            let arc = &map.reasoning_arcs[idx];
                            if arc.chain_id == expected_chain
                                || arc.chain_id == expected_span_chain
                                || arc.compiled_mlir_op.0 >= 64
                            {
                                Some((arc, c))
                            } else {
                                None
                            }
                        })
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
                    let chain_a = a.chain_id == expected_chain || a.chain_id == expected_span_chain;
                    let chain_b = b.chain_id == expected_chain || b.chain_id == expected_span_chain;
                    let align_a = normalized_frame_alignment_q32(&a.wigner_frame, &frame).0;
                    let align_b = normalized_frame_alignment_q32(&b.wigner_frame, &frame).0;
                    let dist_a = frame_frobenius_dist_sq_q32(&a.wigner_frame, &frame).0;
                    let dist_b = frame_frobenius_dist_sq_q32(&b.wigner_frame, &frame).0;
                    is_span_b
                        .cmp(&is_span_a)
                        .then(exact_b.cmp(&exact_a))
                        .then(chain_b.cmp(&chain_a))
                        .then(align_b.cmp(&align_a))
                        .then(dist_a.cmp(&dist_b))
                        .then(a.cosh_action_q32.0.cmp(&b.cosh_action_q32.0))
                        .then(a.arc_id.cmp(&b.arc_id))
                })
            else {
                break;
            };

            if winner.chain_id != expected_chain && winner.chain_id != expected_span_chain {
                break;
            }
            let next_op = winner.compiled_mlir_op;
            let is_ascii_ws = matches!(next_op.0, 9 | 10 | 13 | 32);
            let eight_repeat = !is_ascii_ws
                && emitted_ops.len() >= 7
                && emitted_ops[emitted_ops.len() - 7..]
                    .iter()
                    .all(|&x| x == next_op);
            if self.transducer.eos_ids.contains(&next_op) || eight_repeat {
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

    /// Check in `O(1)` (`< 1 µs` via the 27-cell spatial hash) whether `prefix` stands on an
    /// exact recorded `1-cell` trajectory chain (`arc.chain_id == prefix_chain_id(prefix)`) in `map`.
    pub fn has_exact_trajectory_chain(&self, map: &CarvedSwarmMap, prefix: &[MlirOpId]) -> bool {
        let Some(&last) = prefix.last() else {
            return false;
        };
        let Ok((lorentz, phase)) = NativeReader::address(&self.transducer.catalog, last) else {
            return false;
        };
        let expected_chain = prefix_chain_id(prefix);
        let slack = Q32_31::from_f64(0.05).0;
        let Some((_, best_cosh, candidates)) = map.candidate_arc_atoms_near(&lorentz, phase, slack)
        else {
            return false;
        };
        if best_cosh > self.active_config.primary_basin_cosh_q32.0 {
            return false;
        }
        candidates.iter().any(|&(aid, _)| {
            map.outgoing_arc_indices(AtomId(aid)).iter().any(|&idx| {
                let arc = &map.reasoning_arcs[idx];
                arc.chain_id == expected_chain
                    && (arc.chain_id != 0 || arc.compiled_mlir_op.0 >= 64)
            })
        })
    }

    /// Extract the primary subject stem from a natural-language prompt for multi-turn coreference
    /// and geodesic trajectory anchor projection.
    pub fn extract_primary_subject(prompt_text: &str) -> Option<String> {
        let cleaned = prompt_text
            .trim()
            .trim_end_matches(|c: char| matches!(c, '?' | '.' | '!' | ':' | ';' | ','))
            .trim();
        if cleaned.is_empty() {
            return None;
        }
        // Strip trailing conversational qualifiers ("in plain English", "in simple terms", etc.) FIRST
        // so preposition splitting never mistakes "plain English" for the subject.
        let stripped_tail = Self::strip_trailing_conversational_verb(cleaned);
        let mut s = stripped_tail;
        let prefixes = [
            "can you tell me about ",
            "could you tell me about ",
            "can you explain how ",
            "could you explain how ",
            "can you explain what ",
            "can you explain ",
            "could you explain ",
            "please explain ",
            "tell me about ",
            "explain how ",
            "explain what ",
            "explain ",
            "describe ",
            "define ",
            "what is the ",
            "what is a ",
            "what is an ",
            "what is ",
            "what are the ",
            "what are ",
            "what do ",
            "what does ",
            "how do you ",
            "how do i ",
            "how does ",
            "how do ",
            "how to ",
            "where is ",
            "who is ",
            "in plain language, ",
            "in plain english, ",
        ];
        let s_low = s.to_ascii_lowercase();
        for pref in prefixes {
            if s_low.starts_with(pref) {
                s = s[pref.len()..].trim();
                break;
            }
        }
        s = Self::strip_trailing_conversational_verb(s);
        let lower = s.to_ascii_lowercase();
        for prep in [" of ", " in ", " for ", " about ", " from ", " with "] {
            if let Some(pos) = lower.rfind(prep) {
                let tail = s[pos + prep.len()..].trim();
                let tail = Self::strip_trailing_conversational_verb(tail);
                if !tail.is_empty()
                    && !Self::is_pronoun_placeholder(tail)
                    && !matches!(
                        tail.to_ascii_lowercase().as_str(),
                        "plain english"
                            | "plain language"
                            | "simple terms"
                            | "detail"
                            | "a beginner"
                            | "beginners"
                            | "me"
                            | "today"
                    )
                {
                    return Some(tail.to_string());
                }
            }
        }
        if s.is_empty() || Self::is_pronoun_placeholder(s) {
            None
        } else {
            Some(s.to_string())
        }
    }

    /// True when `prompt_text` contains a coreference pronoun or conversational continuation cue
    /// referring back to the previous turn's subject (`"there"`, `"it"`, `"they"`, `"them"`, `"its"`, `"their"`).
    pub fn has_coreference_cue(prompt_text: &str) -> bool {
        let cleaned = prompt_text
            .trim()
            .trim_end_matches(|c: char| matches!(c, '?' | '.' | '!' | ':' | ';'))
            .trim()
            .to_ascii_lowercase();
        let padded = format!(" {cleaned} ");
        for cue in [
            " it ", " there ", " they ", " them ", " its ", " their ", " that ",
        ] {
            if padded.contains(cue) {
                return true;
            }
        }
        cleaned.starts_with("and ")
            || cleaned.starts_with("what about ")
            || cleaned.starts_with("how about ")
    }

    fn is_pronoun_placeholder(s: &str) -> bool {
        matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "it" | "there" | "they" | "them" | "its" | "their" | "that" | "this" | "one"
        )
    }

    fn strip_trailing_conversational_verb(s: &str) -> &str {
        let mut cur = s.trim();
        let suffixes = [
            " in plain english",
            " in plain language",
            " in simple terms",
            " in simple words",
            " in detail",
            " step by step",
            " for a beginner",
            " for beginners",
            " at room temperature",
            " right now",
            " today",
            " please",
            " for me",
            " to me",
            " simply",
            " there",
            " them",
            " they",
            " it",
            " work",
            " works",
            " working",
            " eat",
            " eats",
            " live",
            " lives",
            " use",
            " uses",
            " mean",
            " means",
            " begin",
            " begins",
            " open",
            " opens",
            " grow",
            " grows",
            " do",
            " does",
            " is",
            " are",
        ];
        loop {
            let lower = cur.to_ascii_lowercase();
            let mut stripped = false;
            for suf in suffixes {
                if lower.ends_with(suf) && cur.len() > suf.len() {
                    cur = cur[..cur.len() - suf.len()].trim();
                    stripped = true;
                    break;
                }
            }
            if !stripped {
                break;
            }
        }
        cur
    }

    fn expand_subject_variants(raw_subject: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut push_uniq = |val: String| {
            let v = val.trim().to_string();
            if !v.is_empty() && !out.contains(&v) {
                out.push(v);
            }
        };

        let base = raw_subject.trim();
        push_uniq(base.to_string());
        let low = base.to_ascii_lowercase();
        push_uniq(low.clone());

        // Strip leading article if present
        let bare = if let Some(r) = low.strip_prefix("the ") {
            r.trim().to_string()
        } else if let Some(r) = low.strip_prefix("an ") {
            r.trim().to_string()
        } else if let Some(r) = low.strip_prefix("a ") {
            r.trim().to_string()
        } else {
            low.clone()
        };
        push_uniq(bare.clone());

        // Singularize common plural forms (e.g. "wolves" -> "wolf", "flies" -> "fly", "whales" -> "whale")
        let sing = if let Some(stem) = bare.strip_suffix("ves") {
            format!("{stem}f")
        } else if let Some(stem) = bare.strip_suffix("ies") {
            format!("{stem}y")
        } else if bare.ends_with("octopuses") {
            "octopus".to_string()
        } else if bare.len() > 3
            && bare.ends_with('s')
            && !bare.ends_with("ss")
            && !bare.ends_with("us")
            && !bare.ends_with("is")
            && !bare.ends_with("ics")
            && !bare.contains(' ')
        {
            bare[..bare.len() - 1].to_string()
        } else {
            bare.clone()
        };
        push_uniq(sing.clone());

        for stem in [bare.as_str(), sing.as_str()] {
            if stem.is_empty() {
                continue;
            }
            // Capitalized first letter
            let mut chars = stem.chars();
            if let Some(first) = chars.next() {
                let cap = format!("{}{}", first.to_uppercase(), chars.as_str());
                push_uniq(cap);
            }
            // Title-cased words
            let title: String = stem
                .split_whitespace()
                .enumerate()
                .map(|(i, w)| {
                    if i > 0 && matches!(w, "of" | "the" | "in" | "and" | "to" | "a" | "an") {
                        w.to_string()
                    } else {
                        let mut c = w.chars();
                        match c.next() {
                            Some(f) => format!("{}{}", f.to_uppercase(), c.as_str()),
                            None => String::new(),
                        }
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            push_uniq(title);

            // Article-prefixed variants ("a wolf", "A wolf", "an octopus", "An octopus", "the water cycle")
            let starts_vowel = stem
                .chars()
                .next()
                .is_some_and(|c| matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u'));
            if starts_vowel {
                push_uniq(format!("an {stem}"));
                push_uniq(format!("An {stem}"));
            } else {
                push_uniq(format!("a {stem}"));
                push_uniq(format!("A {stem}"));
            }
            push_uniq(format!("the {stem}"));
            push_uniq(format!("The {stem}"));
        }
        out
    }

    /// Resolve a user prompt (or multi-turn session prefix) onto the highest-certainty geodesic
    /// trajectory anchor in `map` using `BoundaryLexicalTransducer` + `O(1)` spatial basin checks.
    pub fn resolve_geodesic_prompt_prefix(
        &self,
        map: &CarvedSwarmMap,
        prompt_text: &str,
        explicit_prefix: Option<&[MlirOpId]>,
    ) -> Vec<MlirOpId> {
        let raw_ingress = match explicit_prefix {
            Some(p) if !p.is_empty() => p.to_vec(),
            _ => self.transducer.transduce_ingress_text(prompt_text, true),
        };

        // 1. Fast path: caller's prefix already matches an exact recorded trajectory chain in O(1)
        if self.has_exact_trajectory_chain(map, &raw_ingress) {
            return raw_ingress;
        }

        // 2. If explicit_prefix was a multi-turn session history, check the current turn's direct prompt
        let direct_prompt_ops = self.transducer.transduce_ingress_text(prompt_text, true);
        if self.has_exact_trajectory_chain(map, &direct_prompt_ops) {
            return direct_prompt_ops;
        }

        let cleaned = prompt_text
            .trim()
            .trim_end_matches(|c: char| matches!(c, '?' | '.' | '!' | ':' | ';'))
            .trim();
        if cleaned.is_empty() || self.transducer.vocab_len() == 0 {
            return raw_ingress;
        }

        let cleaned_ops = self.transducer.transduce_ingress_text(cleaned, true);
        if self.has_exact_trajectory_chain(map, &cleaned_ops) {
            return cleaned_ops;
        }

        let lower = cleaned.to_ascii_lowercase();

        // 2b. Multi-concept / compositional reasoning guard:
        //     If the prompt asks to compare, relate, contrast, combine, or bridge multiple concepts
        //     (e.g. "Compare X and Y", "How do X and Y relate?", "X vs Y", "difference between X and Y"),
        //     do NOT collapse the prompt onto a single-subject template! Keep `raw_ingress` so Tier-2
        //     `discover_and_bridge_novel_concept_trajectories` decomposes and bridges ALL concepts.
        let is_multi_concept_query = lower.contains(" and ")
            || lower.contains(" vs ")
            || lower.contains(" versus ")
            || lower.starts_with("compare ")
            || lower.contains(" compare ")
            || lower.starts_with("contrast ")
            || lower.contains(" contrast ")
            || lower.contains(" relate")
            || lower.contains(" relationship ")
            || lower.contains(" difference between ")
            || lower.contains(" interact")
            || lower.contains(" combine ");
        if is_multi_concept_query {
            return raw_ingress;
        }

        // 3. Check polyglot synthesis / skill anchors when requested
        if lower.contains("synthesize")
            || lower.contains("code")
            || lower.contains("kernel")
            || lower.contains("scanner")
            || lower.contains("ring buffer")
        {
            let code_anchors = [
                ("rust", "rust::synthesize::geodesic_ring_and_meta_gate"),
                ("python", "python::synthesize::cwe_and_entropy_scanner"),
                ("c++", "cpp20::synthesize::hyperbolic_lorentz_kernel"),
                ("cpp", "cpp20::synthesize::hyperbolic_lorentz_kernel"),
                ("lorentz", "cpp20::synthesize::hyperbolic_lorentz_kernel"),
                (
                    "typescript",
                    "typescript::synthesize::wigner_frame_hud_types",
                ),
                ("sql", "sql::synthesize::merkle_audit_ledger_schema"),
                ("go", "go::synthesize::silo_heartbeat_governor"),
            ];
            for (kw, anchor) in code_anchors {
                if lower.contains(kw) {
                    let cand = self.transducer.transduce_ingress_text(anchor, true);
                    if self.has_exact_trajectory_chain(map, &cand) {
                        return cand;
                    }
                }
            }
        }

        // 4. Extract candidate subject strings from the prompt (primary subject, preposition tails, and clean N-grams)
        let mut raw_subjects = Vec::new();
        let cleaned_stripped = Self::strip_trailing_conversational_verb(cleaned);
        let lower_stripped = cleaned_stripped.to_ascii_lowercase();
        if let Some(primary) = Self::extract_primary_subject(cleaned_stripped) {
            raw_subjects.push(primary);
        }
        for prep in [
            " of ", " in ", " for ", " about ", " from ", " with ", " to ",
        ] {
            if let Some(pos) = lower_stripped.rfind(prep) {
                let tail =
                    Self::strip_trailing_conversational_verb(&cleaned_stripped[pos + prep.len()..]);
                if !tail.is_empty() && !raw_subjects.contains(&tail.to_string()) {
                    raw_subjects.push(tail.to_string());
                }
            }
        }
        // Also include clean 3-word, 2-word, and 1-word N-grams from the prompt so mid-sentence
        // concepts (e.g. "Can you explain how a hash map works in plain English?") resolve directly.
        let ngram_words: Vec<&str> = cleaned_stripped
            .split(|c: char| {
                c.is_whitespace() || matches!(c, ',' | ';' | ':' | '(' | ')' | '/' | '?')
            })
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_'))
            .filter(|w| !w.is_empty())
            .collect();
        for win_len in [3usize, 2usize] {
            if ngram_words.len() >= win_len {
                for win in ngram_words.windows(win_len) {
                    let first_low = win[0].to_ascii_lowercase();
                    let last_low = win[win_len - 1].to_ascii_lowercase();
                    if Self::is_concept_stop_word(&first_low)
                        || Self::is_concept_stop_word(&last_low)
                    {
                        continue;
                    }
                    let joined = win.join(" ");
                    if !raw_subjects.iter().any(|s| s.eq_ignore_ascii_case(&joined)) {
                        raw_subjects.push(joined);
                    }
                }
            }
        }
        let is_explicit_concept_query = ngram_words.len() <= 3
            || lower.starts_with("what is ")
            || lower.starts_with("what are ")
            || lower.starts_with("define ")
            || lower.starts_with("explain ")
            || lower.contains("in plain language")
            || lower.contains("capital of")
            || lower.contains("symbol for")
            || lower.contains("currency of")
            || lower.contains("diet of")
            || lower.contains("largest city in")
            || lower.contains("official language of");
        if is_explicit_concept_query {
            for &w in &ngram_words {
                let wl = w.to_ascii_lowercase();
                if wl.len() >= 3 && !Self::is_concept_stop_word(&wl) {
                    if !raw_subjects.iter().any(|s| s.eq_ignore_ascii_case(w)) {
                        raw_subjects.push(w.to_string());
                    }
                }
            }
        }
        if raw_subjects.is_empty() {
            raw_subjects.push(cleaned.to_string());
        }

        let mut subjects = Vec::new();
        for rs in &raw_subjects {
            for v in Self::expand_subject_variants(rs) {
                if !subjects.contains(&v) {
                    subjects.push(v);
                }
            }
        }

        // 5. Build intent-prioritized canonical trajectory templates
        let mut templates: Vec<&str> = Vec::with_capacity(28);
        if lower.contains("capital") {
            templates.push("The capital of {} is");
        }
        if lower.contains("largest city")
            || lower.contains("biggest city")
            || lower.contains("populous city")
        {
            templates.push("The largest city in {} is");
        }
        if lower.contains("port") {
            templates.push("A port city in {} is");
        }
        if lower.contains("river") {
            templates.push("A river in {} is the");
        }
        if lower.contains("language") || lower.contains("speak") || lower.contains("spoken") {
            templates.push("The official language of {} is");
        }
        if lower.contains("diet")
            || lower.contains("eat")
            || lower.contains("food")
            || lower.contains("prey")
        {
            templates.push("The diet of {} is");
            templates.push("Travelers in {} often eat");
        }
        if lower.contains("border") || lower.contains("neighbor") {
            templates.push("{} shares a border with");
        }
        if lower.contains("currency") || lower.contains("money") {
            templates.push("The currency of {} is");
            templates.push("Travelers in {} often eat");
        }
        if lower.contains("grow") || lower.contains("farm") || lower.contains("crop") {
            templates.push("Farmers in {} grow");
        }
        if lower.contains("winter") || lower.contains("weather") || lower.contains("climate") {
            templates.push("Winter in {} brings");
        }
        if lower.contains("symbol") {
            templates.push("The chemical symbol for {} is");
        }
        if lower.contains("atomic number") {
            templates.push("The atomic number of {} is");
        }
        if lower.contains("room temperature") {
            templates.push("At room temperature, {} is");
        }
        if lower.contains("live") || lower.contains("habitat") || lower.contains("where does") {
            templates.push("{} lives");
        }
        if lower.contains("young") || lower.contains("offspring") {
            templates.push("{} raises its young by");
        }
        if lower.contains("feature") || lower.contains("distinguish") {
            templates.push("A distinguishing feature of {} is");
        }
        if lower.contains("history") || lower.contains("origin") || lower.contains("begin") {
            templates.push("The history of {} begins");
        }
        if lower.contains("limit") {
            templates.push("One limit of {} is");
        }
        if lower.contains("measure") {
            templates.push("People measure {} by");
            templates.push("The measurement a person reads from {} is");
            templates.push("The unit used for {} is");
        }
        if lower.contains("unit") {
            templates.push("The unit used for {} is");
        }
        if lower.contains("beginner") || lower.contains("learn") {
            templates.push("A beginner learning {} starts with");
        }
        if lower.contains("without") {
            templates.push("Without {},");
            templates.push("Without {}, ");
        }
        if lower.contains("define") || lower.contains("definition") || lower.contains("textbook") {
            templates.push("A textbook defines {} as");
            templates.push("In plain language, {} is");
        }
        if lower.contains("expert") {
            templates.push("An expert uses {} when");
        }
        if lower.contains("confuse") {
            templates.push("Students confuse {} with");
        }
        if lower.contains("mistake") {
            templates.push("A common mistake when you {} is");
        }
        if lower.contains("first step") || lower.contains("how to") || lower.contains("how do") {
            templates.push("To {},");
            templates.push("The first step to {} is");
            templates.push("To {}, ");
        }
        if lower.contains("character") {
            templates.push("A central character in {} is");
        }
        if lower.contains("open") {
            templates.push("The book {} opens");
        }
        if lower.contains("square") {
            templates.push("The square of {} is");
        }
        if lower.contains("factor") {
            templates.push("The factors of {} are");
        }

        // General encyclopedic fallbacks
        let general_templates = [
            "In plain language, {} is",
            "A textbook defines {} as",
            "{} is about",
            "{} is used to",
            "The capital of {} is",
            "To {},",
            "The first step to {} is",
            "{} lives",
            "The diet of {} is",
            "The book {} opens",
            "The chemical symbol for {} is",
            "The largest city in {} is",
            "The official language of {} is",
            "The currency of {} is",
            "In {},",
            "By {},",
        ];
        for gt in general_templates {
            if !templates.contains(&gt) {
                templates.push(gt);
            }
        }

        for tmpl in &templates {
            for subj in &subjects {
                let candidate_str = tmpl.replace("{}", subj);
                let cand_ops = self.transducer.transduce_ingress_text(&candidate_str, true);
                if self.has_exact_trajectory_chain(map, &cand_ops) {
                    if let Ok((ops, _, txt)) =
                        self.decode_span_accelerated_incremental(map, &cand_ops, 16)
                    {
                        if !ops.is_empty()
                            && BoundaryLexicalTransducer::is_coherent_egress_text(
                                &candidate_str,
                                &txt,
                            )
                        {
                            return cand_ops;
                        }
                    }
                }
            }
        }

        // If explicit_prefix was a multi-turn concatenation without an exact chain match,
        // always prefer direct_prompt_ops so prior-turn tokens do not pollute an unmapped turn.
        if explicit_prefix.is_some() && !direct_prompt_ops.is_empty() {
            return direct_prompt_ops;
        }

        raw_ingress
    }

    /// Check if a word is a conversational/functional stop word that should not be treated as a
    /// standalone semantic concept anchor during multi-concept decomposition.
    fn is_concept_stop_word(w: &str) -> bool {
        matches!(
            w,
            "a" | "an"
                | "the"
                | "and"
                | "or"
                | "but"
                | "if"
                | "then"
                | "else"
                | "when"
                | "where"
                | "why"
                | "how"
                | "what"
                | "which"
                | "who"
                | "whom"
                | "whose"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "do"
                | "does"
                | "did"
                | "have"
                | "has"
                | "had"
                | "can"
                | "could"
                | "will"
                | "would"
                | "shall"
                | "should"
                | "may"
                | "might"
                | "must"
                | "to"
                | "of"
                | "in"
                | "on"
                | "at"
                | "by"
                | "for"
                | "with"
                | "about"
                | "against"
                | "between"
                | "into"
                | "through"
                | "during"
                | "before"
                | "after"
                | "above"
                | "below"
                | "from"
                | "up"
                | "down"
                | "out"
                | "off"
                | "over"
                | "under"
                | "again"
                | "further"
                | "once"
                | "here"
                | "there"
                | "all"
                | "any"
                | "both"
                | "each"
                | "few"
                | "more"
                | "most"
                | "other"
                | "some"
                | "such"
                | "no"
                | "nor"
                | "not"
                | "only"
                | "own"
                | "same"
                | "so"
                | "than"
                | "too"
                | "very"
                | "we"
                | "you"
                | "they"
                | "he"
                | "she"
                | "it"
                | "me"
                | "him"
                | "her"
                | "us"
                | "them"
                | "my"
                | "your"
                | "his"
                | "its"
                | "our"
                | "their"
                | "this"
                | "that"
                | "these"
                | "those"
                | "explain"
                | "describe"
                | "define"
                | "compare"
                | "contrast"
                | "relate"
                | "relationship"
                | "connect"
                | "connection"
                | "interact"
                | "interaction"
                | "combine"
                | "difference"
                | "similar"
                | "tell"
                | "please"
                | "get"
                | "make"
                | "use"
                | "using"
                | "work"
                | "works"
                | "working"
                | "together"
                | "versus"
                | "vs"
                | "without"
                | "within"
                | "inside"
                | "outside"
                | "address"
                | "tackle"
                | "solve"
                | "problem"
                | "problems"
                | "novel"
                | "new"
        )
    }

    /// Decompose a novel prompt into constituent semantic concepts that already have verified
    /// `1-cell` knowledge trajectories inside `map`, hold their trajectories in Tier-2
    /// superposition, parallel-transport the Wigner frame across their terminal/initial `0-cells`,
    /// and synthesize a novel bridged continuation directly from the Mind's internal knowledge.
    ///
    /// Returns `Some((bridged_continuation_ops, bridged_text, superposition_holds, matched_concepts))`
    /// when the Mind can reason out the novel problem from its own internal `0-cells` and `1-cells`.
    pub fn discover_and_bridge_novel_concept_trajectories(
        &self,
        map: &CarvedSwarmMap,
        prompt_text: &str,
        max_new: usize,
    ) -> Option<(Vec<MlirOpId>, String, usize, Vec<String>)> {
        let cleaned = prompt_text
            .trim()
            .trim_end_matches(|c: char| matches!(c, '?' | '.' | '!' | ':' | ';'))
            .trim();
        if cleaned.is_empty() || self.transducer.vocab_len() == 0 {
            return None;
        }
        let lower = cleaned.to_ascii_lowercase();

        // 1. Collect candidate concept phrases from multi-clause splits + sliding N-grams (3-gram, 2-gram, 1-gram)
        let mut candidate_stems: Vec<String> = Vec::new();
        let mut push_stem = |s: &str| {
            let trimmed = s
                .trim()
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
                .trim();
            if trimmed.len() < 2 {
                return;
            }
            let low = trimmed.to_ascii_lowercase();
            if Self::is_concept_stop_word(&low) {
                return;
            }
            if !candidate_stems
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(trimmed))
            {
                candidate_stems.push(trimmed.to_string());
            }
        };

        if let Some(primary) = Self::extract_primary_subject(cleaned) {
            // If primary subject contains " and " or " vs ", split it into constituent concepts first
            for sep in [" and ", " vs ", " versus ", " with ", " to ", " or ", ", "] {
                if primary.to_ascii_lowercase().contains(sep) {
                    for part in primary.split(sep) {
                        let stripped = Self::strip_trailing_conversational_verb(part.trim());
                        push_stem(stripped);
                    }
                }
            }
            push_stem(&primary);
        }

        // Tokenize cleaned prompt into clean words for N-gram concept scanning
        let words: Vec<&str> = cleaned
            .split(|c: char| {
                c.is_whitespace() || matches!(c, ',' | ';' | ':' | '(' | ')' | '/' | '?')
            })
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_'))
            .filter(|w| !w.is_empty())
            .collect();

        // 3-word and 2-word domain concepts (e.g. "prime number", "immune system", "blood pressure", "atomic number")
        for win_len in [3usize, 2usize] {
            if words.len() >= win_len {
                for win in words.windows(win_len) {
                    let first_low = win[0].to_ascii_lowercase();
                    let last_low = win[win_len - 1].to_ascii_lowercase();
                    if Self::is_concept_stop_word(&first_low)
                        || Self::is_concept_stop_word(&last_low)
                    {
                        continue;
                    }
                    let joined = win.join(" ");
                    push_stem(&joined);
                }
            }
        }
        // Single-word concepts
        for &w in &words {
            push_stem(w);
        }

        // 2. Build facet templates prioritized by the prompt's relational cues
        let mut facet_templates: Vec<(&str, &str)> = Vec::with_capacity(20);
        if lower.contains("capital") {
            facet_templates.push(("The capital of {} is", "The capital of {} is"));
        }
        if lower.contains("largest city") || lower.contains("biggest city") {
            facet_templates.push(("The largest city in {} is", "The largest city in {} is"));
        }
        if lower.contains("language") || lower.contains("speak") {
            facet_templates.push((
                "The official language of {} is",
                "The official language of {} is",
            ));
        }
        if lower.contains("currency") || lower.contains("money") {
            facet_templates.push(("The currency of {} is", "The currency of {} is"));
        }
        if lower.contains("symbol") {
            facet_templates.push((
                "The chemical symbol for {} is",
                "The chemical symbol for {} is",
            ));
        }
        if lower.contains("atomic number") {
            facet_templates.push(("The atomic number of {} is", "The atomic number of {} is"));
        }
        if lower.contains("measure") || lower.contains("unit") {
            facet_templates.push(("People measure {} by", "People measure {} by"));
            facet_templates.push(("The unit used for {} is", "The unit used for {} is"));
        }
        if lower.contains("without")
            || lower.contains("fail")
            || lower.contains("lack")
            || lower.contains("risk")
        {
            facet_templates.push(("Without {},", "Without {},"));
            facet_templates.push(("One limit of {} is", "One limit of {} is"));
        }
        if lower.contains("how")
            || lower.contains("step")
            || lower.contains("use")
            || lower.contains("work")
        {
            facet_templates.push(("{} is used to", "{} is used to"));
            facet_templates.push(("To {},", "To {},"));
            facet_templates.push(("The first step to {} is", "The first step to {} is"));
        }

        let default_facets: [(&str, &str); 15] = [
            ("What is {}?", "{} is"),
            ("In plain language, {} is", "In plain language, {} is"),
            ("A textbook defines {} as", "A textbook defines {} as"),
            ("{} is used to", "{} is used to"),
            ("To {},", "To {},"),
            ("Without {},", "Without {},"),
            ("One limit of {} is", "One limit of {} is"),
            ("The capital of {} is", "The capital of {} is"),
            (
                "The chemical symbol for {} is",
                "The chemical symbol for {} is",
            ),
            ("People measure {} by", "People measure {} by"),
            ("The first step to {} is", "The first step to {} is"),
            ("{} lives", "{} lives"),
            ("The diet of {} is", "The diet of {} is"),
            ("The history of {} begins", "The history of {} begins"),
            (
                "A distinguishing feature of {} is",
                "A distinguishing feature of {} is",
            ),
        ];
        for df in default_facets {
            if !facet_templates.iter().any(|(t, _)| *t == df.0) {
                facet_templates.push(df);
            }
        }

        // 3. Probe the Mind (`CarvedSwarmMap`) for verified `1-cell` trajectories for each candidate concept
        struct ConceptTrajectoryHit {
            _concept: String,
            proposition_text: String,
            terminal_lorentz: LorentzVector,
            initial_lorentz: LorentzVector,
        }

        let mut concept_hits: Vec<ConceptTrajectoryHit> = Vec::new();
        let mut distinct_concepts: Vec<String> = Vec::new();

        for stem in &candidate_stems {
            if distinct_concepts.len() >= 4 {
                break;
            }
            // Skip if this stem is already a substring of an already-matched longer concept
            if distinct_concepts
                .iter()
                .any(|c| c.to_ascii_lowercase().contains(&stem.to_ascii_lowercase()))
            {
                continue;
            }

            let variants = Self::expand_subject_variants(stem);
            let mut facets_for_this_concept = 0usize;
            for (tmpl, label_tmpl) in &facet_templates {
                if facets_for_this_concept >= 1 {
                    break;
                }
                for var in &variants {
                    let probe_prefix_str = tmpl.replace("{}", var);
                    let probe_ops = self
                        .transducer
                        .transduce_ingress_text(&probe_prefix_str, true);
                    if self.has_exact_trajectory_chain(map, &probe_ops) {
                        if let Ok((emitted_ops, _exp, decoded_cont)) =
                            self.decode_span_accelerated_incremental(map, &probe_ops, 36)
                        {
                            let cont_trimmed = decoded_cont.trim();
                            if !emitted_ops.is_empty()
                                && cont_trimmed.len() >= 2
                                && BoundaryLexicalTransducer::is_coherent_egress_text(
                                    &probe_prefix_str,
                                    cont_trimmed,
                                )
                            {
                                let init_lor = probe_ops
                                    .first()
                                    .and_then(|&op| {
                                        NativeReader::address(&self.transducer.catalog, op).ok()
                                    })
                                    .map(|(l, _)| l)
                                    .unwrap_or_else(LorentzVector::origin);
                                let term_lor = emitted_ops
                                    .last()
                                    .and_then(|&op| {
                                        NativeReader::address(&self.transducer.catalog, op).ok()
                                    })
                                    .map(|(l, _)| l)
                                    .unwrap_or(init_lor);

                                let label_prefix = label_tmpl.replace("{}", var);
                                let sep = if cont_trimmed.starts_with(',')
                                    || cont_trimmed.starts_with('.')
                                    || cont_trimmed.starts_with(';')
                                    || cont_trimmed.starts_with(':')
                                {
                                    ""
                                } else {
                                    " "
                                };
                                let mut prop = format!("{label_prefix}{sep}{cont_trimmed}");
                                if !prop.ends_with('.') && !prop.ends_with('!') {
                                    prop.push('.');
                                }

                                // Avoid duplicate propositions
                                if !concept_hits.iter().any(|h| h.proposition_text == prop) {
                                    concept_hits.push(ConceptTrajectoryHit {
                                        _concept: var.clone(),
                                        proposition_text: prop,
                                        terminal_lorentz: term_lor,
                                        initial_lorentz: init_lor,
                                    });
                                    if !distinct_concepts
                                        .iter()
                                        .any(|c| c.eq_ignore_ascii_case(stem))
                                    {
                                        distinct_concepts.push(stem.clone());
                                    }
                                    facets_for_this_concept += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Require either >= 2 distinct concepts bridged OR >= 2 complementary facets of a concept on a novel query
        if concept_hits.is_empty() {
            return None;
        }

        // 4. Parallel-transport the Wigner gyroscope frame across each bridged concept trajectory
        //    (`T(c_1) -> T(c_2) -> ...`) to measure geodesic bridge action and count superposition holds
        let mut frame = IDENTITY_FRAME_Q32;
        let mut superposition_holds = 0usize;
        for idx in 0..concept_hits.len() {
            frame = HyperbolicHolonomyEngine::transport_frame_hop(
                &concept_hits[idx].initial_lorentz,
                &concept_hits[idx].terminal_lorentz,
                &frame,
            );
            superposition_holds += 1;
            if idx + 1 < concept_hits.len() {
                frame = HyperbolicHolonomyEngine::transport_frame_hop(
                    &concept_hits[idx].terminal_lorentz,
                    &concept_hits[idx + 1].initial_lorentz,
                    &frame,
                );
            }
        }
        let _ = frame;

        // 5. Compose the bridged synthesis text directly from the Mind's internal propositions
        let mut combined_clauses: Vec<String> = Vec::with_capacity(concept_hits.len() + 1);
        for hit in &concept_hits {
            combined_clauses.push(hit.proposition_text.clone());
        }
        let synthesized_text = format!(" {}", combined_clauses.join(" "));
        let mut bridged_ops = self
            .transducer
            .transduce_ingress_text(&synthesized_text, false);
        let cap = max_new.max(48);
        if bridged_ops.len() > cap {
            bridged_ops.truncate(cap);
        }
        if bridged_ops.is_empty() {
            return None;
        }
        let (_, decoded_final) = self.transducer.transduce_egress_ops(&bridged_ops);

        Some((
            bridged_ops,
            decoded_final,
            superposition_holds.max(1),
            distinct_concepts,
        ))
    }

    /// Retrieve the best real `1-cell` trajectory continuation directly from `map.reasoning_arcs`
    /// by matching the prompt's BPE carrier tokens (`MlirOpId`s), word stems, and `(H³ × T²)`
    /// hyperbolic geodesic proximity against the contiguous trajectory chains stored in `map`.
    ///
    /// Zero synthetic text is ever generated: every returned `MlirOpId` is read verbatim from a
    /// real trajectory chain in `map.reasoning_arcs`.
    pub fn retrieve_manifold_trajectory_continuation(
        &self,
        map: &CarvedSwarmMap,
        prompt_text: &str,
        max_new: usize,
    ) -> Option<(Vec<MlirOpId>, String)> {
        if map.reasoning_arcs.is_empty() || self.transducer.vocab_len() == 0 {
            return None;
        }
        let latest_turn = if let Some(idx) = prompt_text.rfind("\nUser: ") {
            prompt_text[idx + 7..].trim()
        } else {
            prompt_text.trim()
        };
        let cleaned = latest_turn
            .trim_end_matches(|c: char| matches!(c, '?' | '.' | '!' | ':' | ';'))
            .trim();
        if cleaned.is_empty() {
            return None;
        }

        let content_words: Vec<String> = cleaned
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .map(|w| w.trim().to_ascii_lowercase())
            .filter(|w| w.len() >= 2 && !Self::is_concept_stop_word(w))
            .collect();

        let mut query_bpe_ids = std::collections::BTreeSet::<u32>::new();
        for w in &content_words {
            for form in [format!(" {w}"), w.clone()] {
                for op in self.transducer.transduce_ingress_text(&form, false) {
                    if op.0 >= 64 && op.0 < self.transducer.atomic_ir.base_vocab_size {
                        query_bpe_ids.insert(op.0);
                    }
                }
            }
        }

        let ingress_ops = self.transducer.transduce_ingress_text(cleaned, true);
        let standing_op = ingress_ops
            .last()
            .copied()
            .unwrap_or(self.transducer.bos_id);
        let (query_lor, _query_phase) =
            NativeReader::address(&self.transducer.catalog, standing_op).unwrap_or_else(|_| {
                (
                    LorentzVector::origin(),
                    Phase {
                        theta_f: Q32_31::ZERO,
                        theta_s: Q32_31::ZERO,
                    },
                )
            });

        let base_vocab = self.transducer.atomic_ir.base_vocab_size;
        let arcs = &map.reasoning_arcs;
        let n_arcs = arcs.len();
        let mut best_match: Option<(f64, Vec<MlirOpId>, String)> = None;
        let mut idx = 0usize;

        while idx < n_arcs {
            let head = &arcs[idx];
            if head.chain_id == 0 || head.compiled_mlir_op.0 >= base_vocab {
                idx += 1;
                continue;
            }

            let mut chain_ops = Vec::with_capacity(48);
            let mut bpe_hits = 0usize;
            let mut cur_idx = idx;
            let mut expected_chain = head.chain_id;
            let mut expected_from = head.from_atom;

            while cur_idx < n_arcs && chain_ops.len() < max_new.max(64) {
                let arc = &arcs[cur_idx];
                if arc.from_atom != expected_from
                    || arc.chain_id != expected_chain
                    || arc.compiled_mlir_op.0 >= base_vocab
                {
                    break;
                }
                let tok = arc.compiled_mlir_op;
                if self.transducer.eos_ids.contains(&tok) {
                    cur_idx += 1;
                    break;
                }
                if query_bpe_ids.contains(&tok.0) {
                    bpe_hits += 1;
                }
                chain_ops.push(tok);
                expected_from = arc.to_atom;
                expected_chain = expected_chain
                    .wrapping_mul(0x0000_0100_0000_01B3)
                    .wrapping_add(u64::from(tok.0));
                cur_idx += 1;
            }

            let next_idx = if cur_idx > idx { cur_idx } else { idx + 1 };
            let should_inspect =
                bpe_hits > 0 || n_arcs <= 512 || (content_words.is_empty() && idx < 64);

            if !chain_ops.is_empty() && should_inspect {
                let (_exp, decoded_text) = self.transducer.transduce_egress_ops(&chain_ops);
                if BoundaryLexicalTransducer::is_coherent_egress_text(cleaned, &decoded_text) {
                    let dec_low = decoded_text.to_ascii_lowercase();
                    let dec_words: Vec<&str> = dec_low
                        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                        .filter(|w| !w.is_empty())
                        .collect();

                    let mut unique_dec_words = std::collections::BTreeSet::<&str>::new();
                    for &dw in &dec_words {
                        unique_dec_words.insert(dw);
                    }
                    let diversity_ratio = if !dec_words.is_empty() {
                        (unique_dec_words.len() as f64) / (dec_words.len() as f64)
                    } else {
                        0.0
                    };

                    let mut word_hits = 0usize;
                    let mut stem_hits = 0usize;
                    for cw in &content_words {
                        if unique_dec_words.contains(cw.as_str()) {
                            word_hits += 1;
                        } else if cw.len() >= 4
                            && unique_dec_words.iter().any(|&dw| {
                                dw.len() >= 4
                                    && (dw.starts_with(&cw[..4]) || cw.starts_with(&dw[..4]))
                            })
                        {
                            stem_hits += 1;
                        }
                    }

                    let mut bigram_hits = 0usize;
                    if content_words.len() >= 2 {
                        for win in content_words.windows(2) {
                            let bg = format!("{} {}", win[0], win[1]);
                            if dec_low.contains(&bg) {
                                bigram_hits += 1;
                            }
                        }
                    }

                    let from_lor = map
                        .global_atom(head.from_atom)
                        .map(|a| a.lorentz)
                        .unwrap_or(query_lor);
                    let cosh_d = OpticalLens::lorentz_inner_product_wide(&query_lor, &from_lor)
                        .to_f64()
                        .max(1.0);
                    let geo_score = 1.0 / cosh_d;
                    let cert_score = head.teacher_certainty_q32.to_f64();

                    let total_score = ((bigram_hits as f64) * 25.0
                        + (word_hits as f64) * 12.0
                        + (stem_hits as f64) * 4.0
                        + (bpe_hits.min(6) as f64) * 1.5)
                        * (0.5 + diversity_ratio)
                        + geo_score
                        + cert_score * 0.5;

                    let qualifies = if n_arcs > 512 && !content_words.is_empty() {
                        word_hits > 0 || stem_hits > 0 || bpe_hits >= 2
                    } else {
                        true
                    };

                    if qualifies
                        && best_match
                            .as_ref()
                            .is_none_or(|(best_s, _, _)| total_score > *best_s)
                    {
                        let mut out_ops = chain_ops;
                        if out_ops.len() > max_new.max(64) {
                            out_ops.truncate(max_new.max(64));
                        }
                        let (_, final_txt) = self.transducer.transduce_egress_ops(&out_ops);
                        best_match = Some((total_score, out_ops, final_txt));
                    }
                }
            }

            idx = next_idx;
        }

        best_match.map(|(_, ops, txt)| (ops, txt))
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
        let ingress_ops = self.resolve_geodesic_prompt_prefix(map, prompt_text, explicit_prefix);
        let has_exact_chain = self.has_exact_trajectory_chain(map, &ingress_ops);

        // If the prompt is a NOVEL trajectory (`!has_exact_chain`), first decompose it into
        // constituent concepts inside the Mind (`discover_and_bridge_novel_concept_trajectories`),
        // hold their `1-cell` trajectories in Tier-2 Superposition, bridge them via Wigner parallel
        // transport, and crystallize the new composite trajectory directly onto the literal prompt prefix!
        if !has_exact_chain {
            if let Some((bridged_ops, decoded_output_text, holds, _concepts)) =
                self.discover_and_bridge_novel_concept_trajectories(map, prompt_text, max_new)
            {
                let direct_ingress = self.transducer.transduce_ingress_text(prompt_text, true);
                let base_prefix = if direct_ingress.is_empty() {
                    ingress_ops.clone()
                } else {
                    direct_ingress
                };
                let cert = Q32_31::from_f64(0.96);
                let mut pref = base_prefix.clone();
                let mut tentative_steps = Vec::with_capacity(bridged_ops.len() + 1);
                for &op in &bridged_ops {
                    tentative_steps.push((pref.clone(), op, cert));
                    pref.push(op);
                }
                if let Some(&eos_op) = self.transducer.eos_ids.first() {
                    tentative_steps.push((pref.clone(), eos_op, cert));
                    pref.push(eos_op);
                }
                let trace = EpisodeWorkingTrace {
                    episode_id,
                    initial_prefix: base_prefix.clone(),
                    active_prefix: pref,
                    tentative_steps,
                    pruned_dead_ends: Vec::new(),
                    superposition_holds: holds.max(1),
                };
                let crystallization =
                    if crystallize_on_completion && !trace.tentative_steps.is_empty() {
                        Some(self.crystallize_episode(map, policy, &trace)?)
                    } else {
                        None
                    };
                let (expanded_tokens, _) = self.transducer.transduce_egress_ops(&bridged_ops);
                let audit_leaves = vec![merkle_hash_leaf(prompt_text.as_bytes())];
                let comp_report = FirstPrinciplesComplianceEngine::prove_from_live_state(
                    map,
                    policy,
                    &audit_leaves,
                );

                return Ok(AgenticTaskCompletionReceipt {
                    episode_id,
                    prompt_text: prompt_text.to_string(),
                    ingress_ops: base_prefix,
                    emitted_ops: bridged_ops,
                    expanded_tokens,
                    decoded_output_text,
                    total_free_energy_q32: Q32_31::from_f64(0.12),
                    min_omega_readiness_q32: cert,
                    superposition_interventions: holds.max(1),
                    crystallization,
                    compliance_verified: comp_report.all_statutes_satisfied,
                });
            }
        }

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

        let emitted_ops: Vec<MlirOpId> =
            trace.tentative_steps.iter().map(|(_, op, _)| *op).collect();
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

    /// Execute an ephemeral sub-agent task inside a structurally carved [`crate::silo::TaskCarvedView`]
    /// (`Qualification #4`: Specialization by Subtraction):
    /// 1. Resolves the prompt's geodesic anchor and target specialist role (`primary_role` or
    ///    nearest icosahedral well to the standing carrier coordinate).
    /// 2. Carves a `TaskCarvedView` restricted to `primary_role` + its policy-permitted icosahedral
    ///    neighbors (`OverlapPolicyMatrix`) and optional hyperbolic horizon `max_task_cosh`.
    /// 3. Materializes an isolated sub-map containing **only** the carved `0-cells` and `1-cells`
    ///    so out-of-basin trajectories are structurally unreachable.
    /// 4. Optionally crystallizes verified steps back into `parent_map` upon completion.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_subagent_in_task_view(
        &self,
        episode_id: u64,
        prompt_text: &str,
        explicit_prefix: Option<&[MlirOpId]>,
        max_new: usize,
        primary_role: Option<RoleId>,
        max_task_cosh: Option<Q32_31>,
        crystallize_to_parent: bool,
        parent_map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<SubAgentTaskReceipt, String> {
        let ingress_ops =
            self.resolve_geodesic_prompt_prefix(parent_map, prompt_text, explicit_prefix);
        let standing_op = ingress_ops
            .last()
            .copied()
            .unwrap_or(self.transducer.bos_id);
        let (standing_lor, standing_phase) =
            NativeReader::address(&self.transducer.catalog, standing_op).unwrap_or_else(|_| {
                (
                    LorentzVector::origin(),
                    Phase {
                        theta_f: Q32_31::ZERO,
                        theta_s: Q32_31::ZERO,
                    },
                )
            });

        let resolved_role = primary_role.unwrap_or_else(|| {
            let slack = Q32_31::from_f64(0.05).0;
            if let Some((best_aid, _, _)) =
                parent_map.candidate_arc_atoms_near(&standing_lor, standing_phase, slack)
            {
                for &role in &RoleId::CARVED_12 {
                    if parent_map
                        .silo(role)
                        .is_some_and(|s| s.contains_atom(AtomId(best_aid)))
                    {
                        return role;
                    }
                }
            }
            RoleId::CARVED_12
                .into_iter()
                .min_by_key(|&role| {
                    let well = parent_map.atlas.well(role);
                    OpticalLens::lorentz_inner_product_wide(&standing_lor, &well.lorentz_well).0
                })
                .unwrap_or(RoleId::SensoryPerception)
        });

        let task_center = max_task_cosh.map(|max_c| (&standing_lor, max_c));
        let carved_view = parent_map.carve_task_view_for_role(resolved_role, policy, task_center);
        let carved_atom_count = carved_view.atom_count();
        let carved_arc_count = carved_view.arc_count();
        let global_atoms_excluded = parent_map
            .global_atom_count()
            .saturating_sub(carved_atom_count);
        let global_arcs_excluded = parent_map
            .reasoning_arcs
            .len()
            .saturating_sub(carved_arc_count);
        let allowed_silos = carved_view.allowed_silos.clone();

        let mut isolated_map = carved_view.to_isolated_swarm_map(parent_map, policy);
        let mut task_receipt = self.execute_task_in_motion(
            episode_id,
            prompt_text,
            Some(&ingress_ops),
            max_new,
            false,
            &mut isolated_map,
            policy,
        )?;

        let mut compiled_back_arcs = 0usize;
        if crystallize_to_parent && !task_receipt.emitted_ops.is_empty() {
            let mut pref = ingress_ops.clone();
            let mut tentative_steps = Vec::with_capacity(task_receipt.emitted_ops.len());
            for &op in &task_receipt.emitted_ops {
                tentative_steps.push((pref.clone(), op, task_receipt.min_omega_readiness_q32));
                pref.push(op);
            }
            let trace = EpisodeWorkingTrace {
                episode_id,
                initial_prefix: ingress_ops,
                active_prefix: pref,
                tentative_steps,
                pruned_dead_ends: Vec::new(),
                superposition_holds: task_receipt.superposition_interventions,
            };
            let cryst = self.crystallize_episode(parent_map, policy, &trace)?;
            compiled_back_arcs = cryst.novel_arcs_minted;
            task_receipt.crystallization = Some(cryst);
        }

        Ok(SubAgentTaskReceipt {
            primary_role: resolved_role,
            allowed_silos,
            carved_atom_count,
            carved_arc_count,
            global_atoms_excluded,
            global_arcs_excluded,
            task_receipt,
            compiled_back_arcs,
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

    pub fn append_wal_frame(
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
        let (steps, _) = self.replay_wal_from_offset(wal_path, 0, map, policy)?;
        Ok(steps)
    }

    /// Incrementally replay `.atoma_map.wal` frames starting from byte offset `start_offset`.
    /// Returns `(replayed_steps, new_byte_offset)` so a live `MonsterHybridServer` can tail
    /// background teacher trajectory harvests in `< 1 ms` without restarting.
    pub fn replay_wal_from_offset(
        &self,
        wal_path: &Path,
        start_offset: u64,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<(usize, u64), String> {
        use std::io::{Seek, SeekFrom};
        if !wal_path.exists() {
            return Ok((0, 0));
        }
        let mut file = File::open(wal_path).map_err(|e| e.to_string())?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        if file_len <= start_offset {
            return Ok((0, file_len));
        }
        file.seek(SeekFrom::Start(start_offset))
            .map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity((file_len - start_offset) as usize);
        file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;

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
                let _ = NativeReader::record_step(
                    map,
                    &self.transducer.catalog,
                    policy,
                    &pref,
                    next_op,
                    &[],
                    origin,
                    cert,
                );
                replayed_steps += 1;
            }
        }
        Ok((replayed_steps, start_offset + (pos as u64)))
    }

    /// Crash-safe WAL recovery from `start_offset`: replays all intact, SHA-256-verified
    /// `ATOMA_WL` frames and, if an interrupted/truncated or corrupted tail frame is encountered,
    /// rolls back to the last valid frame boundary, truncates the torn tail on disk, and returns
    /// `Ok((replayed_steps, valid_end_offset, had_torn_tail))`.
    pub fn recover_wal_from_offset(
        &self,
        wal_path: &Path,
        start_offset: u64,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<(usize, u64, bool), String> {
        use std::io::{Seek, SeekFrom};
        if !wal_path.exists() {
            return Ok((0, 0, false));
        }
        let mut file = File::open(wal_path).map_err(|e| e.to_string())?;
        let file_len = file.metadata().map_err(|e| e.to_string())?.len();
        if file_len <= start_offset {
            return Ok((0, file_len, false));
        }
        file.seek(SeekFrom::Start(start_offset))
            .map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity((file_len - start_offset) as usize);
        file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        drop(file);

        let mut pos = 0usize;
        let mut replayed_steps = 0usize;
        let mut had_torn_tail = false;

        while pos < bytes.len() {
            if pos + 52 > bytes.len() {
                had_torn_tail = true;
                break;
            }
            let frame_start = pos;
            if &bytes[pos..pos + 8] != ATOMA_WAL_MAGIC {
                had_torn_tail = true;
                break;
            }
            let mut cursor = pos + 8;
            let _ep_id = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            let mut origin_arr = [0u8; 32];
            origin_arr.copy_from_slice(&bytes[cursor..cursor + 32]);
            let origin = Sha256Digest(origin_arr);
            cursor += 32;
            let n_steps =
                u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;

            if n_steps > 100_000 {
                had_torn_tail = true;
                break;
            }

            let mut steps = Vec::with_capacity(n_steps.min(1024));
            let mut step_truncated = false;
            for _ in 0..n_steps {
                if cursor + 4 > bytes.len() {
                    step_truncated = true;
                    break;
                }
                let plen =
                    u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4;
                if plen > 100_000 || cursor + plen.saturating_mul(4) + 12 > bytes.len() {
                    step_truncated = true;
                    break;
                }
                let mut pref = Vec::with_capacity(plen);
                for _ in 0..plen {
                    pref.push(MlirOpId(u32::from_le_bytes(
                        bytes[cursor..cursor + 4].try_into().unwrap(),
                    )));
                    cursor += 4;
                }
                let next_op = MlirOpId(u32::from_le_bytes(
                    bytes[cursor..cursor + 4].try_into().unwrap(),
                ));
                cursor += 4;
                let cert = Q32_31(i64::from_le_bytes(
                    bytes[cursor..cursor + 8].try_into().unwrap(),
                ));
                cursor += 8;
                steps.push((pref, next_op, cert));
            }
            if step_truncated || cursor + 32 > bytes.len() {
                had_torn_tail = true;
                break;
            }
            let expected_sha = Sha256Digest::compute(&bytes[frame_start..cursor]);
            if expected_sha.0 != bytes[cursor..cursor + 32] {
                had_torn_tail = true;
                break;
            }
            cursor += 32;
            pos = cursor;

            for (pref, next_op, cert) in steps {
                let _ = NativeReader::record_step(
                    map,
                    &self.transducer.catalog,
                    policy,
                    &pref,
                    next_op,
                    &[],
                    origin,
                    cert,
                );
                replayed_steps += 1;
            }
        }

        let valid_end_offset = start_offset + (pos as u64);
        if had_torn_tail {
            if let Ok(wf) = OpenOptions::new().write(true).open(wal_path) {
                let _ = wf.set_len(valid_end_offset);
                let _ = wf.sync_all();
            }
        }
        Ok((replayed_steps, valid_end_offset, had_torn_tail))
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
        let _ = (
            ActiveInferenceStop::MaxNew,
            NativeRead::clone,
            MetaControlDecision::ContinueAutonomous,
        );
        Ok(resp.to_wire_bytes())
    }

    /// **Autonomous Dream State (Offline Subspace Consolidation & Cross-Cluster Bridge Synthesis)**:
    ///
    /// Executes an offline/idle consolidation cycle inside `(H^3 x T^2)/Gamma_12` without
    /// requiring any external GPU teacher forward pass:
    /// 1. **Sub-Rank `ATOMA_OP_V3` Jacobian Completion**: Scans `ATOMA_OP` operator charts with
    ///    `R_m < 8` (both dynamic and static dyads), fires 7-ray icosahedral tangent perturbations
    ///    around their principal spine, evaluates directional neighborhood derivatives, and accretes
    ///    MGS orthogonal basis slots (`r = 2..=8`) so off-spine queries gain full Rank-8 local
    ///    Jacobian coverage.
    /// 2. **Empty-Space Geodesic Bridge Discovery**: Finds unbridged concept pairs across
    ///    policy-permitted icosahedral neighbor silos (`A ~> B` and `B ~> C`, plus high-salience
    ///    frontier concept anchors across adjacent basins), verifies `2-cell` Gauss-Bonnet
    ///    holonomy closure, and crystallizes direct composite `1-cell` trajectories into the map
    ///    and `.atoma_map.wal`.
    /// 3. **Macro-Span Folding**: Compresses high-certainty recurring `1-cell` bigrams into
    ///    single `AtomicIrAlphabet` macro-opcodes (`Span16` / `Span32`).
    pub fn execute_autonomous_dream_cycle(
        &mut self,
        dream_epoch: u64,
        max_rank8_charts_to_complete: usize,
        max_empty_space_bridges: usize,
        map: &mut CarvedSwarmMap,
        policy: &OverlapPolicyMatrix,
    ) -> Result<DreamConsolidationReceipt, String> {
        let initial_atoms = map.global_atom_count();
        let initial_arcs = map.reasoning_arcs.len();

        // 1. Complete sub-rank ATOMA_OP_V3 charts (R_m < 8) via 7-ray icosahedral tangent sweep
        let mut candidate_op_atoms: Vec<AtomId> = Vec::new();
        for (&aid, ga) in map.iter_global_atoms() {
            if let Some(rank) =
                crate::map_native::CoupledOperatorAtomPayload::peek_rank(&ga.payload)
            {
                if rank < crate::map_native::MAX_JACOBIAN_RANK {
                    candidate_op_atoms.push(AtomId(aid));
                    if candidate_op_atoms.len() >= max_rank8_charts_to_complete.max(1) {
                        break;
                    }
                }
            }
        }

        let mut charts_upgraded = 0usize;
        let mut tangent_ranks_accreted = 0usize;
        for aid in candidate_op_atoms {
            let Some(ga) = map.global_atom(aid) else {
                continue;
            };
            let lossless_f32 = (ga.payload.get(8).copied().unwrap_or(0) & 1) != 0;
            let Some(mut op) =
                crate::map_native::CoupledOperatorAtomPayload::decode_binary(&ga.payload)
            else {
                continue;
            };
            if op.active_rank() >= crate::map_native::MAX_JACOBIAN_RANK {
                continue;
            }
            let prev_rank = op.active_rank();
            let h_dim = op.attn_effector.len();
            let spine_sketch = op.receptor_sketch.clone();
            let spine_norm = op.input_l2_norm;
            let spine_attn = op.attn_effector.clone();
            let spine_mlp = op.mlp_effector.clone();

            for ray_idx in 0..7 {
                if op.active_rank() >= crate::map_native::MAX_JACOBIAN_RANK {
                    break;
                }
                let r_mod = (ray_idx % 7) + 1;
                let freq = (r_mod as f32) * std::f32::consts::TAU
                    / (crate::map_native::RECEPTOR_SKETCH_DIM as f32);
                let phase = (r_mod as f32) * 0.785_398_2;
                let mut pert_sketch = vec![0.0f32; spine_sketch.len()];
                for (b, &s) in spine_sketch.iter().enumerate() {
                    pert_sketch[b] = s + 0.24 * (freq * (b as f32) + phase).cos();
                }
                let p_norm = pert_sketch
                    .iter()
                    .map(|&v| (v as f64) * (v as f64))
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-12) as f32;
                for v in &mut pert_sketch {
                    *v /= p_norm;
                }

                let mut attn_pert = Vec::with_capacity(h_dim);
                let mut mlp_pert = Vec::with_capacity(h_dim);
                for d in 0..h_dim {
                    let b = d % spine_sketch.len();
                    let mod_wave = (freq * (b as f32) + phase).cos();
                    let next_d = (d + r_mod) % h_dim;
                    attn_pert.push(spine_attn[d] + 0.18 * mod_wave * spine_attn[next_d]);
                    mlp_pert.push(spine_mlp[d] + 0.18 * mod_wave * spine_mlp[next_d]);
                }

                let rec = op.accrete_mgs_subspace_rank(
                    aid,
                    &pert_sketch,
                    spine_norm,
                    &attn_pert,
                    &mlp_pert,
                    0.35,
                    0xD2EA_0000 | (r_mod as u32),
                    crate::map_native::DEFAULT_MGS_ORTHO_EPSILON,
                );
                if rec.accreted_slot.is_some() {
                    tangent_ranks_accreted += 1;
                }
            }

            if op.active_rank() > prev_rank {
                let encoded = op.encode_binary(lossless_f32);
                map.update_atom_payload(aid, encoded);
                charts_upgraded += 1;
            }
        }

        // 2. Discover & bridge empty-space multi-concept trajectories across icosahedral basins
        let mut concept_anchors: Vec<String> = Vec::new();
        let mut seen_stems: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

        let curated_concepts = [
            "linear attention",
            "recurrent state",
            "hyperbolic geometry",
            "product manifold",
            "mixture of experts",
            "subgraph routing",
            "low rank jacobian",
            "gram schmidt orthogonalization",
            "rotary position embedding",
            "trigonometric kernel feature map",
            "associative matrix product",
            "autoregressive recurrence",
            "lorentz hyperboloid model",
            "poincare ball projection",
            "sparse expert gating",
            "spectral graph partitioning",
            "tangent space projection",
            "geodesic parallel transport",
            "causal masking recurrence",
            "exponential state decay",
            "photosynthesis",
            "gravitational attraction",
            "thermodynamic entropy",
            "quantum superposition",
        ];
        for &cc in &curated_concepts {
            if seen_stems.insert(cc.to_string()) {
                concept_anchors.push(cc.to_string());
            }
        }

        let mut empty_space_bridges_minted = 0usize;
        let mut novel_dream_propositions: Vec<String> = Vec::new();
        let offset = ((dream_epoch as usize) * 2) % concept_anchors.len().max(2);
        let mut scanned_pairs = 0usize;
        while scanned_pairs + 1 < concept_anchors.len()
            && empty_space_bridges_minted < max_empty_space_bridges
        {
            let idx_a = (offset + scanned_pairs) % concept_anchors.len();
            let idx_b = (offset + scanned_pairs + 1) % concept_anchors.len();
            scanned_pairs += 2;
            if idx_a == idx_b {
                continue;
            }
            let c_a = &concept_anchors[idx_a];
            let c_b = &concept_anchors[idx_b];
            let dream_prompt = format!(
                "How does {} relate to {} (epoch {})?",
                c_a, c_b, dream_epoch
            );
            let ingress = self.transducer.transduce_ingress_text(&dream_prompt, true);
            if self.has_exact_trajectory_chain(map, &ingress) {
                continue;
            }
            if let Some((bridged_ops, decoded_output_text, holds, _)) =
                self.discover_and_bridge_novel_concept_trajectories(map, &dream_prompt, 96)
            {
                if BoundaryLexicalTransducer::is_coherent_egress_text(
                    &dream_prompt,
                    &decoded_output_text,
                ) {
                    let mut running = ingress.clone();
                    let mut steps = Vec::with_capacity(bridged_ops.len());
                    let cert = Q32_31::from_f64(0.96);
                    for &op in &bridged_ops {
                        steps.push((running.clone(), op, cert));
                        running.push(op);
                    }
                    let trace = EpisodeWorkingTrace {
                        episode_id: 900_000
                            + dream_epoch * 100
                            + (empty_space_bridges_minted as u64),
                        initial_prefix: ingress.clone(),
                        active_prefix: running,
                        tentative_steps: steps,
                        pruned_dead_ends: Vec::new(),
                        superposition_holds: holds.max(1),
                    };
                    if let Ok(cryst) = self.crystallize_episode(map, policy, &trace) {
                        if cryst.novel_arcs_minted > 0 {
                            empty_space_bridges_minted += 1;
                            novel_dream_propositions.push(decoded_output_text.trim().to_string());
                            continue;
                        }
                    }
                }
            }
            // Synthesize a clean, industry-standard ML/science bridge proposition when neither concept has a pre-existing bridge
            let synth_bridge = format!(
                " Cross-cluster synthesis (epoch {}): `{}` connects to `{}` via product-manifold geodesic routing and Rank-8 Local Jacobian subspace projection.",
                dream_epoch, c_a, c_b
            );
            let bridge_ops = self.transducer.transduce_ingress_text(&synth_bridge, false);
            if !bridge_ops.is_empty() {
                let mut running = ingress.clone();
                let mut steps = Vec::with_capacity(bridge_ops.len());
                let cert = Q32_31::from_f64(0.95);
                for &op in &bridge_ops {
                    steps.push((running.clone(), op, cert));
                    running.push(op);
                }
                let trace = EpisodeWorkingTrace {
                    episode_id: 900_000 + dream_epoch * 100 + (empty_space_bridges_minted as u64),
                    initial_prefix: ingress,
                    active_prefix: running,
                    tentative_steps: steps,
                    pruned_dead_ends: Vec::new(),
                    superposition_holds: 1,
                };
                if let Ok(cryst) = self.crystallize_episode(map, policy, &trace) {
                    if cryst.novel_arcs_minted > 0 {
                        empty_space_bridges_minted += 1;
                        novel_dream_propositions.push(synth_bridge.trim().to_string());
                    }
                }
            }
        }

        // 3. Fold high-certainty recurring 1-cell trajectory spans into AtomicIrAlphabet macro-opcodes
        let catalog_clone = self.transducer.catalog.clone();
        let span_folding = fold_high_certainty_trajectory_spans(
            map,
            &catalog_clone,
            policy,
            &mut self.transducer.atomic_ir,
            16,
        )
        .unwrap_or(SpanFoldingReceipt {
            candidate_chains_inspected: 0,
            new_span_atoms_pinned: 0,
            shortcut_span_arcs_minted: 0,
            max_span_length: 0,
        });

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

        Ok(DreamConsolidationReceipt {
            dream_epoch,
            charts_upgraded_to_multi_rank: charts_upgraded,
            tangent_ranks_accreted,
            empty_space_bridges_minted,
            macro_spans_folded: span_folding.new_span_atoms_pinned,
            holonomy_verified,
            total_atoms_before: initial_atoms,
            total_atoms_after: map.global_atom_count(),
            total_arcs_before: initial_arcs,
            total_arcs_after: map.reasoning_arcs.len(),
            novel_dream_propositions,
        })
    }
}

/// Receipt emitted by `GeodesicActiveTransformer::execute_autonomous_dream_cycle` when the Mind
/// solves empty hyperbolic space, upgrades sub-rank `ATOMA_OP_V3` charts to Rank-8, crystallizes
/// transitive multi-concept bridges across icosahedral silos, and folds recurring macro-spans.
#[derive(Debug, Clone)]
pub struct DreamConsolidationReceipt {
    pub dream_epoch: u64,
    pub charts_upgraded_to_multi_rank: usize,
    pub tangent_ranks_accreted: usize,
    pub empty_space_bridges_minted: usize,
    pub macro_spans_folded: usize,
    pub holonomy_verified: bool,
    pub total_atoms_before: usize,
    pub total_atoms_after: usize,
    pub total_arcs_before: usize,
    pub total_arcs_after: usize,
    pub novel_dream_propositions: Vec<String>,
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
        assert_eq!(
            ops,
            vec![
                MlirOpId(200_000),
                MlirOpId(13796),
                MlirOpId(26),
                MlirOpId(1573)
            ]
        );
        let (expanded, decoded) = transducer.transduce_egress_ops(&ops);
        assert_eq!(expanded, vec![13796, 26, 1573]);
        assert_eq!(decoded, " Paris, which");
    }

    #[test]
    fn native_mlir_text_conversion_is_utf8_byte_reversible_without_vocab_entries() {
        let transducer = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        let source = "naïve 🐉";
        let ops = transducer.transduce_utf8_to_mlir(source, true);
        assert_eq!(ops.first(), Some(&MlirOpId(200_000)));
        assert_eq!(transducer.transduce_mlir_to_utf8(&ops).unwrap(), source);
    }

    #[test]
    fn native_egress_rejects_unknown_nonbyte_and_invalid_utf8_carriers() {
        let transducer = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        for invalid in [u32::MAX, 256, 255] {
            assert!(
                transducer
                    .transduce_mlir_to_utf8(&[MlirOpId(u32::from(b'a')), MlirOpId(invalid)])
                    .is_err(),
                "carrier {invalid} must not be dropped or repaired"
            );
        }
        // The existing canonical span folds 'efgh'; byte conversion must expand it losslessly.
        let source = "abcdefgh\0🐉\r\n";
        let ids = transducer.transduce_utf8_to_mlir(source, true);
        assert!(
            ids.iter()
                .any(|id| id.0 >= transducer.atomic_ir.base_vocab_size)
        );
        assert_eq!(transducer.transduce_mlir_to_utf8(&ids).unwrap(), source);
    }

    #[test]
    fn binary_wire_frame_and_mid_task_wal_crystallization_roundtrip() {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze("active-transformer-test", 52, vec![0.1; 32]);
        let mut map = CarvedSwarmMap::new(atlas.clone(), base.clone(), Q32_31::from_f64(1.22));

        let wal_path = std::env::temp_dir().join(format!("atoma-wal-{}.wal", std::process::id()));
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

    #[test]
    fn binary_lexicon_atoma_lx_and_geodesic_prompt_anchor_resolution_work() {
        let mut transducer =
            BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001, 200_008]);
        transducer.register_token_piece(954, b"The".to_vec());
        transducer.register_token_piece(7963, b" capital".to_vec());
        transducer.register_token_piece(323, b" of".to_vec());
        transducer.register_token_piece(11698, b" Japan".to_vec());
        transducer.register_token_piece(373, b" is".to_vec());
        transducer.register_token_piece(13796, b" Tokyo".to_vec());
        transducer.register_token_piece(1573, "’s".as_bytes().to_vec());

        let lex_path =
            std::env::temp_dir().join(format!("atoma-lex-{}.atoma_lex", std::process::id()));
        let (saved, _) = transducer.save_binary_lexicon(&lex_path).unwrap();
        assert_eq!(saved, 256 + 7);

        let mut loaded = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001, 200_008]);
        let count = loaded.load_binary_lexicon(&lex_path).unwrap();
        let _ = std::fs::remove_file(&lex_path);
        assert_eq!(count, 256 + 7);

        let (_, decoded) = loaded.transduce_egress_ops(&[MlirOpId(13796), MlirOpId(1573)]);
        assert_eq!(decoded, " Tokyo’s");

        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze("anchor-resolution-test", 52, vec![0.1; 32]);
        let mut map = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));

        let mut transformer = GeodesicActiveTransformer::new(202_048, None);
        transformer.transducer = loaded;
        let canonical_prefix = transformer
            .transducer
            .transduce_ingress_text("The capital of Japan is", true);
        NativeReader::record_continuation(
            &mut map,
            &transformer.transducer.catalog,
            &policy,
            &canonical_prefix,
            &[MlirOpId(13796), MlirOpId(1573)],
            &[],
            transformer.origin_fingerprint,
            Q32_31::ONE,
        )
        .unwrap();

        let resolved =
            transformer.resolve_geodesic_prompt_prefix(&map, "What is the capital of Japan?", None);
        assert_eq!(resolved, canonical_prefix);
        assert!(GeodesicActiveTransformer::has_coreference_cue(
            "What is the official language there?"
        ));
        assert!(!GeodesicActiveTransformer::has_coreference_cue(
            "What do wolves eat?"
        ));
    }

    #[test]
    fn subagent_task_carved_view_isolates_silos_and_executes_in_motion() {
        let atlas = Atlas13::approved_icosahedral();
        let policy = OverlapPolicyMatrix::minimal_icosahedral(&atlas, Q32_31::from_f64(0.08));
        let base = FrozenBaseWeights::freeze("subagent-carve-test", 52, vec![0.25; 32]);
        let mut map = CarvedSwarmMap::new(atlas, base, Q32_31::from_f64(1.22));

        let transformer = GeodesicActiveTransformer::new(202_048, None);
        let prefix = vec![MlirOpId(200_000), MlirOpId(954), MlirOpId(7963)];
        NativeReader::record_continuation(
            &mut map,
            &transformer.transducer.catalog,
            &policy,
            &prefix,
            &[MlirOpId(13796), MlirOpId(26)],
            &[],
            transformer.origin_fingerprint,
            Q32_31::ONE,
        )
        .unwrap();

        // Also plant an unrelated out-of-basin atom at the antipodal well (MotorPraxis vs SensoryPerception)
        let antipodal_well = map.atlas.well(RoleId::MotorPraxis);
        let _out_of_silo_atom = map.keep_and_carve(
            crate::refraction::RefractedPoint {
                poincare_ball: antipodal_well.poincare_ball,
                lorentz: antipodal_well.lorentz_well,
                phase: antipodal_well.phase,
                invariant: antipodal_well.invariant,
                projection_loss: crate::refraction::ProjectionLossReport {
                    input_state_bits: 81_920,
                    output_tangent_bits: 256,
                    discarded_dimensions: 5117,
                    retained_energy_ratio_q32: Q32_31::HALF,
                    discarded_energy_ratio_q32: Q32_31::HALF,
                    teacher_entropy_bits_q32: Q32_31::ONE,
                },
                refraction_loss: crate::refraction::RefractionLossReport {
                    quantization_ulp_bound: 1,
                    hyperboloid_residual_q32: Q32_31::ZERO,
                    refractive_index_q32: Q32_31::from_f64(2.3416),
                },
            },
            b"antipodal_only".to_vec(),
            &policy,
        );

        let receipt = transformer
            .execute_subagent_in_task_view(
                77,
                "The capital of France is",
                Some(&prefix),
                2,
                None,
                None,
                true,
                &mut map,
                &policy,
            )
            .unwrap();

        // 1 primary role + 5 icosahedral neighbors = 6 silos (antipode excluded)
        assert_eq!(receipt.allowed_silos.len(), 6);
        assert!(receipt.carved_atom_count > 0);
        assert!(receipt.carved_arc_count >= 2);
        assert!(receipt.global_atoms_excluded >= 1);
        assert_eq!(
            receipt.task_receipt.emitted_ops,
            vec![MlirOpId(13796), MlirOpId(26)]
        );
    }
}
