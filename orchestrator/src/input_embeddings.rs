//! Native input representation for tokenizer-free UTF-8 to ML-IR inference.
//!
//! Each byte carrier is represented by its canonical H³×T² address from `NativeReader`:
//! four Lorentz coordinates and two toroidal phases. No checkpoint tokenizer, learned
//! embedding table, or model-specific vocabulary is involved.

use atoma_immune::{BoundaryLexicalTransducer, MlirAtomCatalog, MlirOpId, NativeReader};

pub const NATIVE_MLIR_EMBEDDING_DIM: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeMlirEmbedding {
    pub carrier_id: MlirOpId,
    pub coordinates: [f32; NATIVE_MLIR_EMBEDDING_DIM],
}

pub fn embed_carrier(
    catalog: &MlirAtomCatalog,
    carrier_id: MlirOpId,
) -> Result<NativeMlirEmbedding, String> {
    let (lorentz, phase) = NativeReader::address(catalog, carrier_id)?;
    let coordinates = [
        lorentz.x[0].to_f64() as f32,
        lorentz.x[1].to_f64() as f32,
        lorentz.x[2].to_f64() as f32,
        lorentz.x[3].to_f64() as f32,
        phase.theta_f.to_f64() as f32,
        phase.theta_s.to_f64() as f32,
    ];
    if !coordinates.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "ML-IR carrier {} has non-finite address",
            carrier_id.0
        ));
    }
    Ok(NativeMlirEmbedding {
        carrier_id,
        coordinates,
    })
}

/// Embed text at the MACRO grain: transduce to boundary-lexicon piece ops
/// WITHOUT expanding them to byte carriers, and take each piece's own atlas
/// address. The observation stream is the semantic-piece trajectory (the
/// grain the trajectory-match gate scores), as opposed to `embed_utf8_text`
/// which expands every macro to its byte carriers.
pub fn embed_text_macro(
    transducer: &BoundaryLexicalTransducer,
    text: &str,
) -> Result<Vec<NativeMlirEmbedding>, String> {
    let ops = transducer.transduce_ingress_text(text, false);
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        if op == transducer.bos_id || transducer.eos_ids.contains(&op) {
            continue;
        }
        out.push(embed_carrier(&transducer.catalog, op)?);
    }
    Ok(out)
}

/// Convert query text directly to positional native address vectors. Atomic-IR macro IDs
/// are expanded before embedding so each UTF-8 byte retains its own carrier position.
pub fn embed_utf8_text(
    transducer: &BoundaryLexicalTransducer,
    text: &str,
    prepend_bos: bool,
) -> Result<Vec<NativeMlirEmbedding>, String> {
    let ops = transducer.transduce_utf8_to_mlir(text, prepend_bos);
    let carriers: Vec<u32> = ops
        .iter()
        .copied()
        .filter(|op| *op != transducer.bos_id && !transducer.eos_ids.contains(op))
        .map(|op| op.0)
        .collect();
    transducer
        .atomic_ir
        .expand_checked(&carriers)?
        .into_iter()
        .map(|id| embed_carrier(&transducer.catalog, MlirOpId(id)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_text_embeddings_preserve_utf8_byte_carriers_without_model_vocab() {
        let transducer = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        let text = "A🐉";
        let embedded = embed_utf8_text(&transducer, text, true).unwrap();
        let expected: Vec<u32> = text.as_bytes().iter().map(|&b| u32::from(b)).collect();
        assert_eq!(
            embedded
                .iter()
                .map(|entry| entry.carrier_id.0)
                .collect::<Vec<_>>(),
            expected
        );
        assert!(
            embedded
                .iter()
                .all(|entry| entry.coordinates.iter().all(|value| value.is_finite()))
        );
    }
}

/// BACKBONE CLASSIFIER (2026-10-20, Stewart): a lexicon piece belongs
/// to the relational/numerical backbone of a problem if it is a number
/// (digits or a number word), an operator word, an implicit-constant
/// word, or a closed-class relational word. Decorative narrative —
/// adjectives and atmospheric setup — is NOT backbone. Breathing depth
/// is decided from the backbone only; the full sequence still
/// propagates through the constructed depth.
pub fn is_backbone_piece(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    if t.is_empty() {
        return false;
    }
    if t.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    if crate::math_eval::number_word_value(&t).is_some() {
        return true;
    }
    matches!(
        t.as_str(),
        "plus" | "minus" | "times" | "multiply" | "multiplied" | "divided"
            | "divide" | "over" | "per" | "each" | "every" | "total"
            | "altogether" | "left" | "remain" | "remains" | "remaining"
            | "more" | "fewer" | "less" | "than" | "half" | "twice"
            | "double" | "triple" | "dozen" | "pair" | "weekly" | "equal"
            | "equals" | "sum" | "difference" | "product" | "quotient"
            | "cost" | "price" | "earn" | "earns" | "spend" | "spends"
            | "buy" | "bought" | "sell" | "sold" | "pay" | "paid" | "how"
            | "many" | "much"
    )
}

/// Filter embeddings to the relational/numerical backbone, decoding
/// each carrier's piece text through the transducer. Falls back to
/// the full set only when the backbone is degenerate (< 2).
pub fn backbone_filter(
    transducer: &atoma_immune::BoundaryLexicalTransducer,
    embeddings: &[NativeMlirEmbedding],
) -> Vec<NativeMlirEmbedding> {
    let kept: Vec<NativeMlirEmbedding> = embeddings
        .iter()
        .filter(|e| {
            let (_, text) = transducer.transduce_egress_ops(&[e.carrier_id]);
            is_backbone_piece(&text)
        })
        .cloned()
        .collect();
    if kept.len() >= 2 {
        kept
    } else {
        embeddings.to_vec()
    }
}
