//! Portable (no-Apple-frameworks) single-row dequantization used when the
//! `metal` feature is disabled.
//!
//! This mirrors the semantics of `core_lm::metal::dequantize_row_zero_copy`
//! (mounted as `atoma_immune::metal` behind `#[cfg(feature = "metal")]`) using
//! only safe, bounds-checked standard-library code so that Linux CI runners
//! can build and test every consumer of row dequantization
//! (`distill_engine.rs`, orchestrator binaries, harness).

/// Decode one scalar element of a little-endian, bit-packed weight array.
#[inline]
fn unpack_weight(weight_data: &[u8], elem_idx: usize, bits: u32) -> Option<u32> {
    let mask = (1u32 << bits) - 1;
    match bits {
        8 => weight_data.get(elem_idx).map(|b| *b as u32),
        16 => {
            let off = elem_idx.checked_mul(2)?;
            let lo = *weight_data.get(off)? as u32;
            let hi = *weight_data.get(off + 1)? as u32;
            Some(lo | (hi << 8))
        }
        32 => {
            let off = elem_idx.checked_mul(4)?;
            let b0 = *weight_data.get(off)? as u32;
            let b1 = *weight_data.get(off + 1)? as u32;
            let b2 = *weight_data.get(off + 2)? as u32;
            let b3 = *weight_data.get(off + 3)? as u32;
            Some(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
        }
        _ => {
            let bitpos = elem_idx.checked_mul(bits as usize)?;
            let byte_idx = bitpos / 8;
            let shift = (bitpos % 8) as u32;
            let needed = ((shift + bits + 7) / 8) as usize;
            if byte_idx + needed > weight_data.len() || shift + bits > 32 {
                return None;
            }
            let mut packed = 0u32;
            for i in 0..needed {
                packed |= (*weight_data.get(byte_idx + i)? as u32) << (i * 8);
            }
            Some((packed >> shift) & mask)
        }
    }
}

/// Convert raw little-endian bytes of a BF16/F16/F32 scalar to `f32`.
#[inline]
fn scalar_to_f32(dtype: &str, data: &[u8], idx: usize) -> Option<f32> {
    match dtype {
        "F32" => {
            let off = idx.checked_mul(4)?;
            let bytes: [u8; 4] = data.get(off..off + 4)?.try_into().ok()?;
            Some(f32::from_le_bytes(bytes))
        }
        "BF16" => {
            let off = idx.checked_mul(2)?;
            let bytes: [u8; 2] = data.get(off..off + 2)?.try_into().ok()?;
            Some(f32::from_bits(u16::from_le_bytes(bytes) as u32))
        }
        "F16" => {
            let off = idx.checked_mul(2)?;
            let bytes: [u8; 2] = data.get(off..off + 2)?.try_into().ok()?;
            Some(f16_bits_to_f32(u16::from_le_bytes(bytes)))
        }
        _ => None,
    }
}

/// IEEE-754 half precision -> f32 (subnormal-safe).
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            sign << 31 | 0x3800_0000 | (mant << 13)
        }
    } else if exp == 31 {
        sign << 31 | 0x7F80_0000 | (mant << 13)
    } else {
        sign << 31 | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Dequantize a single row (`cols` elements) from bit-packed weights with
/// per-group affine scales/biases, plus an optional per-row output scale.
///
/// Signature and numerical contract intentionally match
/// `crate::metal::dequantize_row_zero_copy` so call sites can be swapped via
/// `#[cfg(feature = "metal")]`.
#[allow(clippy::too_many_arguments)]
pub fn dequantize_row_zero_copy(
    weight_data: &[u8],
    weight_dtype: &str,
    scales_data: &[u8],
    scales_dtype: &str,
    biases_data: &[u8],
    biases_dtype: &str,
    row_out_scale: f32,
    bits: u32,
    group_size: usize,
    row: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if bits == 0 || bits >= 32 || group_size == 0 || cols % group_size != 0 {
        return Err(format!(
            "Invalid quant params bits={bits} group_size={group_size} cols={cols}"
        ));
    }
    if !matches!(weight_dtype, "U32" | "U8" | "U16") {
        return Err(format!(
            "Unsupported weight_dtype for row dequant: {weight_dtype}"
        ));
    }
    if !matches!(scales_dtype, "BF16" | "F16" | "F32")
        || !matches!(biases_dtype, "BF16" | "F16" | "F32")
    {
        return Err(format!(
            "Unsupported scale/bias dtype: {} / {}",
            scales_dtype, biases_dtype
        ));
    }

    let groups_per_row = cols / group_size;
    let row_group_base = row * groups_per_row;
    let row_elem_base = row * cols;

    let mut out = vec![0.0f32; cols];
    for g in 0..groups_per_row {
        let s = scalar_to_f32(scales_dtype, scales_data, row_group_base + g)
            .ok_or_else(|| format!("Scales buffer too small for group {}", row_group_base + g))?;
        let b = scalar_to_f32(biases_dtype, biases_data, row_group_base + g)
            .ok_or_else(|| format!("Biases buffer too small for group {}", row_group_base + g))?;
        let c_start = g * group_size;
        for j in 0..group_size {
            let c = c_start + j;
            let q = unpack_weight(weight_data, row_elem_base + c, bits)
                .ok_or_else(|| format!("Weight buffer too small at element {}", row_elem_base + c))?
                as f32;
            out[c] = (q * s + b) * row_out_scale;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(v: f32) -> Vec<u8> {
        (f32::to_bits(v) >> 16)
            .try_into()
            .unwrap()
            .to_le_bytes()
            .to_vec()
    }

    #[test]
    fn unpacks_4bit_nibbles_lsb_first() {
        // Two nibbles per byte: 0x21 => elem0 = 1, elem1 = 2.
        let w = [0x21u8];
        assert_eq!(unpack_weight(&w, 0, 4), Some(1));
        assert_eq!(unpack_weight(&w, 1, 4), Some(2));
    }

    #[test]
    fn unpacks_cross_byte_boundary_5bit() {
        // elem0 occupies bits 0..5, elem1 bits 5..10 (spans two bytes).
        let w = [0b1111_1000u8, 0b0000_0011u8];
        assert_eq!(unpack_weight(&w, 0, 5), Some(0b11000));
        assert_eq!(unpack_weight(&w, 1, 5), Some(0b00001));
    }

    #[test]
    fn row_dequant_matches_affine_contract() {
        // 1 row x 4 cols, 4-bit weights packed into 2 bytes: values [1,2,3,4].
        let weight_data = vec![0x21, 0x43];
        let scales_data = bf16_bytes(0.5);
        let biases_data = bf16_bytes(0.25);
        let row = dequantize_row_zero_copy(
            &weight_data,
            "U8",
            &scales_data,
            "BF16",
            &biases_data,
            "BF16",
            2.0,
            4,
            4,
            0,
            4,
        )
        .unwrap();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|q| (q * 0.5 + 0.25) * 2.0)
            .collect();
        for (a, e) in row.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-6, "got {a}, expected {e}");
        }
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(
            dequantize_row_zero_copy(&[], "U8", &[], "BF16", &[], "BF16", 1.0, 0, 4, 0, 4).is_err()
        );
        assert!(
            dequantize_row_zero_copy(&[], "U8", &[], "BF16", &[], "BF16", 1.0, 4, 3, 0, 4).is_err()
        );
        assert!(
            dequantize_row_zero_copy(&[], "F99", &[], "BF16", &[], "BF16", 1.0, 4, 4, 0, 4).is_err()
        );
    }
}
