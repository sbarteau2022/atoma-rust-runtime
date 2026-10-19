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
