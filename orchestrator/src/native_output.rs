//! Tokenizer-free geometric output readout for native toroidal hidden states.
//!
//! This is a deterministic nearest-address byte carrier head with periodic T² phase distance.
//! It is not trained and does not
//! imply language quality; callers must pass its IDs through the strict ML-IR UTF-8 reader.

use atoma_immune::{MlirAtomCatalog, MlirOpId};

use crate::input_embeddings::{NATIVE_MLIR_EMBEDDING_DIM, embed_carrier};

/// Project each six-dimensional hidden state onto its nearest canonical byte-carrier address.
pub fn readout_byte_carriers(
    catalog: &MlirAtomCatalog,
    hidden_states: &[Vec<f32>],
) -> Result<Vec<MlirOpId>, String> {
    if hidden_states.is_empty() {
        return Err("native output head received no hidden states".into());
    }

    let byte_addresses: Vec<_> = (0..=u8::MAX)
        .map(|byte| embed_carrier(catalog, MlirOpId(u32::from(byte))))
        .collect::<Result<_, _>>()?;

    hidden_states
        .iter()
        .map(|hidden| {
            if hidden.len() != NATIVE_MLIR_EMBEDDING_DIM
                || !hidden.iter().all(|value| value.is_finite())
            {
                return Err(format!(
                    "native output state must contain {} finite coordinates",
                    NATIVE_MLIR_EMBEDDING_DIM
                ));
            }
            let nearest = byte_addresses
                .iter()
                .min_by(|a, b| {
                    squared_distance(&a.coordinates, hidden)
                        .total_cmp(&squared_distance(&b.coordinates, hidden))
                        .then(a.carrier_id.cmp(&b.carrier_id))
                })
                .expect("byte address table is non-empty");
            Ok(nearest.carrier_id)
        })
        .collect()
}

fn squared_distance(a: &[f32; NATIVE_MLIR_EMBEDDING_DIM], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(axis, (&left, &right))| {
            let raw_delta = f64::from(left) - f64::from(right);
            // Phases are stored in radians; crossing ±π or adding a turn is the same point.
            let delta = if axis >= 4 {
                (raw_delta + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI
            } else {
                raw_delta
            };
            delta * delta
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input_embeddings::embed_utf8_text;
    use atoma_immune::BoundaryLexicalTransducer;

    #[test]
    fn canonical_byte_addresses_roundtrip_through_native_output_head() {
        let transducer = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        let expected = "Native toroidal output: 🐉";
        let embedded = embed_utf8_text(&transducer, expected, false).unwrap();
        let hidden: Vec<Vec<f32>> = embedded
            .iter()
            .map(|entry| entry.coordinates.to_vec())
            .collect();
        let ids = readout_byte_carriers(&transducer.catalog, &hidden).unwrap();
        assert_eq!(transducer.transduce_mlir_to_utf8(&ids).unwrap(), expected);
    }

    #[test]
    fn native_output_head_rejects_empty_nonfinite_and_wrong_width_states() {
        let transducer = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        assert!(readout_byte_carriers(&transducer.catalog, &[]).is_err());
        assert!(readout_byte_carriers(&transducer.catalog, &[vec![0.0; 5]]).is_err());
        assert!(
            readout_byte_carriers(
                &transducer.catalog,
                &[vec![f32::NAN; NATIVE_MLIR_EMBEDDING_DIM]]
            )
            .is_err()
        );
    }

    #[test]
    fn native_output_head_is_invariant_to_complete_phase_turns() {
        let transducer = BoundaryLexicalTransducer::new(202_048, 200_000, vec![200_001]);
        let hidden: Vec<_> = (0..=255)
            .map(|byte| {
                let mut coordinates = embed_carrier(&transducer.catalog, MlirOpId(byte))
                    .unwrap()
                    .coordinates;
                coordinates[4] += std::f32::consts::TAU;
                coordinates[5] -= std::f32::consts::TAU;
                coordinates.to_vec()
            })
            .collect();
        assert_eq!(
            readout_byte_carriers(&transducer.catalog, &hidden).unwrap(),
            (0..=255).map(MlirOpId).collect::<Vec<_>>()
        );
    }
}
