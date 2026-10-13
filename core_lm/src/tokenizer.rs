//! BPE tokenizer. Parsed from tokenizer.json AT LOAD TIME ONLY.
//! Runtime uses the compiled BpeTokenizer struct.

use fancy_regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub enum TokenizerError {
    Io(String),
    Parse(String),
    Encode(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Io(s) => write!(f, "IO: {}", s),
            TokenizerError::Parse(s) => write!(f, "Parse: {}", s),
            TokenizerError::Encode(s) => write!(f, "Encode: {}", s),
        }
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Debug, Deserialize)]
struct RawTokenizer {
    model: RawModel,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Pair(Vec<String>),
    Joined(String),
}

#[derive(Debug, Deserialize)]
struct RawModel {
    vocab: HashMap<String, u32>,
    merges: Vec<MergeEntry>,
}

pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    inv_vocab: HashMap<u32, String>,
    merge_ranks: HashMap<(String, String), u32>,
    byte_encoder: HashMap<u8, String>,
    byte_decoder: HashMap<String, u8>,
    pre_split: Option<Regex>,
}

fn split_regex(pre: &serde_json::Value) -> Option<String> {
    let mut found = None;
    fn walk(v: &serde_json::Value, found: &mut Option<String>) {
        let Some(obj) = v.as_object() else {
            return;
        };
        if obj.get("type").and_then(|t| t.as_str()) == Some("Split") {
            if let Some(pattern) = obj
                .get("pattern")
                .and_then(|p| p.get("Regex"))
                .and_then(|r| r.as_str())
            {
                *found = Some(pattern.to_string());
            }
        }
        if let Some(parts) = obj.get("pretokenizers").and_then(|p| p.as_array()) {
            for part in parts {
                walk(part, found);
            }
        }
    }
    walk(pre, &mut found);
    found
}

fn isolated_pieces<'a>(re: &Regex, text: &'a str) -> Result<Vec<&'a str>, TokenizerError> {
    let mut pieces = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        let m = m.map_err(|e| TokenizerError::Encode(e.to_string()))?;
        if m.start() > last {
            pieces.push(&text[last..m.start()]);
        }
        if !m.as_str().is_empty() {
            pieces.push(m.as_str());
        }
        last = m.end();
    }
    if last < text.len() {
        pieces.push(&text[last..]);
    }
    Ok(pieces)
}

fn bytes_to_unicode() -> (HashMap<u8, String>, HashMap<String, u8>) {
    let mut bs: Vec<u32> = (b'!' as u32..=(b'~' as u32)).collect();
    bs.extend((0xA1..=0xAC).collect::<Vec<u32>>());
    bs.extend((0xAE..=0xFF).collect::<Vec<u32>>());
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..=255u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut enc = HashMap::new();
    let mut dec = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let s = char::from_u32(*c).unwrap().to_string();
        enc.insert(*b as u8, s.clone());
        dec.insert(s, *b as u8);
    }
    (enc, dec)
}

impl BpeTokenizer {
    pub fn from_json_file(path: &Path) -> Result<Self, TokenizerError> {
        let data = std::fs::read_to_string(path).map_err(|e| TokenizerError::Io(e.to_string()))?;
        let raw: RawTokenizer =
            serde_json::from_str(&data).map_err(|e| TokenizerError::Parse(e.to_string()))?;
        let inv_vocab: HashMap<u32, String> = raw
            .model
            .vocab
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        let mut merge_ranks = HashMap::new();
        for (i, m) in raw.model.merges.iter().enumerate() {
            match m {
                MergeEntry::Pair(pair) if pair.len() == 2 => {
                    merge_ranks.insert((pair[0].clone(), pair[1].clone()), i as u32);
                }
                MergeEntry::Joined(s) => {
                    let parts: Vec<&str> = s.splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        merge_ranks.insert((parts[0].to_string(), parts[1].to_string()), i as u32);
                    }
                }
                _ => {}
            }
        }
        let (byte_encoder, byte_decoder) = bytes_to_unicode();
        let pre_split = match raw.pre_tokenizer.as_ref().and_then(split_regex) {
            Some(pattern) => {
                Some(Regex::new(&pattern).map_err(|e| TokenizerError::Parse(e.to_string()))?)
            }
            None => None,
        };
        Ok(Self {
            vocab: raw.model.vocab,
            inv_vocab,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            pre_split,
        })
    }
    fn bpe(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return vec![];
        }
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() == 1 {
            return word;
        }
        loop {
            let mut best: Option<((String, String), u32)> = None;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    match &best {
                        Some((_, best_rank)) if rank >= *best_rank => {}
                        _ => best = Some((pair, rank)),
                    }
                }
            }
            match best {
                None => break,
                Some(((a, b), _)) => {
                    let mut new_word = Vec::new();
                    let mut i = 0;
                    while i < word.len() {
                        if i < word.len() - 1 && word[i] == a && word[i + 1] == b {
                            new_word.push(format!("{}{}", a, b));
                            i += 2;
                        } else {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    }
                    word = new_word;
                    if word.len() == 1 {
                        break;
                    }
                }
            }
        }
        word
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        let owned;
        let pieces: Vec<&str> = if let Some(re) = &self.pre_split {
            isolated_pieces(re, text)?
        } else {
            owned = text
                .split_inclusive(|c: char| c.is_whitespace())
                .collect::<Vec<_>>();
            owned
        };
        for word in pieces {
            let encoded: String = word
                .bytes()
                .map(|b| self.byte_encoder.get(&b).cloned().unwrap_or_default())
                .collect();
            for piece in self.bpe(&encoded) {
                match self.vocab.get(&piece) {
                    Some(&id) => ids.push(id),
                    None => {
                        return Err(TokenizerError::Encode(format!("Unknown piece: {}", piece)));
                    }
                }
            }
        }
        Ok(ids)
    }
    pub fn decode(&self, ids: &[u32]) -> String {
        let text: String = ids
            .iter()
            .filter_map(|id| self.inv_vocab.get(id))
            .cloned()
            .collect();
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c.to_string()).copied())
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::BpeTokenizer;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[test]
    fn loads_qwen_style_pair_merges() {
        let path = std::env::temp_dir().join(format!(
            "core_lm_tokenizer_pairs_{}_{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let json = r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}}"#;
        fs::write(&path, json).unwrap();
        let tokenizer = BpeTokenizer::from_json_file(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(tokenizer.encode("ab").unwrap(), vec![2]);
        assert_eq!(tokenizer.decode(&[2]), "ab");
    }
    #[test]
    fn loads_hf_style_joined_merges() {
        let path = std::env::temp_dir().join(format!(
            "core_lm_tokenizer_joined_{}_{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let json = r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}"#;
        fs::write(&path, json).unwrap();
        let tokenizer = BpeTokenizer::from_json_file(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(tokenizer.encode("ab").unwrap(), vec![2]);
        assert_eq!(tokenizer.decode(&[2]), "ab");
    }
    #[test]
    fn glimmer_prompt_matches_mlx_pretokenizer() {
        let path = std::path::Path::new(
            "/Users/stewartbarteau/models/Muse-Glimmer-q4-MLX/tokenizer.json",
        );
        let tokenizer = BpeTokenizer::from_json_file(path).expect("glimmer tokenizer");
        // Ids from the checkpoint tokenizer as shipped. mlx-lm loads that
        // file. transformers' fix_mistral_regex splits "it's" and "12345"
        // differently; the baseline run did not apply that flag.
        let cases = [
            ("The capital of France is", vec![954, 7963, 323, 11698, 373]),
            (
                "The capital of France is ",
                vec![954, 7963, 323, 11698, 373, 220],
            ),
            ("it's a test", vec![90715, 262, 1600]),
            ("Hello\nworld", vec![19873, 198, 27349]),
            ("a  b", vec![77, 220, 297]),
            ("12345", vec![7235, 2127]),
            ("end.", vec![431, 26]),
        ];
        for (text, ids) in cases {
            assert_eq!(tokenizer.encode(text).unwrap(), ids, "{text:?}");
        }
    }
}
