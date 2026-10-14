//! Affine dequantizer for packed integer weights.
//!
//! Each quantized tensor has:
//!   {name}.weight — packed ints, low bits first, in U32 or U8
//!   {name}.scales — f32, f16, or bf16 scale per group
//!   {name}.biases — f32, f16, or bf16 zero-point per group
//!
//! A U32 stream is packed tightly, so a code may cross a word boundary
//! when `bits` does not divide 32. Groups run along each row.
//! Dequant: w[i] = q[i] * scale[i/group_size] + bias[i/group_size]

use crate::safetensors::{SafeTensorError, SafeTensorFile, TensorInfo};

#[derive(Debug)]
pub enum DequantError {
    SafeTensor(SafeTensorError),
    ShapeMismatch(String),
    UnsupportedBits(u32),
}

impl std::fmt::Display for DequantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DequantError::SafeTensor(e) => write!(f, "Safetensors error: {}", e),
            DequantError::ShapeMismatch(s) => write!(f, "Shape mismatch: {}", s),
            DequantError::UnsupportedBits(b) => write!(f, "Unsupported bits: {}", b),
        }
    }
}

impl std::error::Error for DequantError {}

impl From<SafeTensorError> for DequantError {
    fn from(e: SafeTensorError) -> Self {
        DequantError::SafeTensor(e)
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn read_f32_le(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn read_bf16_le(data: &[u8], offset: usize) -> f32 {
    let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
    bf16_to_f32(bits)
}

fn read_scale_bias(data: &[u8], dtype: &str, idx: usize) -> Result<f32, DequantError> {
    match dtype {
        "F32" => Ok(read_f32_le(data, idx * 4)),
        "BF16" => Ok(read_bf16_le(data, idx * 2)),
        "F16" => {
            let bits = u16::from_le_bytes([data[idx * 2], data[idx * 2 + 1]]);
            Ok(half_to_f32(bits))
        }
        _ => Err(DequantError::ShapeMismatch(format!(
            "Unsupported scale dtype: {}",
            dtype
        ))),
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            sign << 31 | 0x38000000 | (mant << 13)
        }
    } else if exp == 31 {
        sign << 31 | 0x7F800000 | (mant << 13)
    } else {
        sign << 31 | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

fn read_u32_le(data: &[u8], word: usize) -> Result<u32, DequantError> {
    let start = word.checked_mul(4).ok_or_else(|| {
        DequantError::ShapeMismatch(format!("packed word index {word} overflows"))
    })?;
    let end = start + 4;
    let bytes = data.get(start..end).ok_or_else(|| {
        DequantError::ShapeMismatch(format!("packed weight ends before word {word}"))
    })?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(bytes);
    Ok(u32::from_le_bytes(buf))
}

fn affine_at(
    q: f32,
    index: usize,
    group_size: usize,
    scales_data: &[u8],
    scales_dtype: &str,
    biases_data: &[u8],
    biases_dtype: &str,
) -> Result<f32, DequantError> {
    let g = index / group_size;
    let scale = read_scale_bias(scales_data, scales_dtype, g)?;
    let bias = read_scale_bias(biases_data, biases_dtype, g)?;
    Ok(q * scale + bias)
}

pub fn dequantize(
    weight_data: &[u8],
    weight_info: &TensorInfo,
    scales_data: &[u8],
    scales_info: &TensorInfo,
    biases_data: &[u8],
    biases_info: &TensorInfo,
    bits: u32,
    group_size: usize,
) -> Result<Vec<f32>, DequantError> {
    if bits == 0 || bits >= 32 {
        return Err(DequantError::UnsupportedBits(bits));
    }
    if group_size == 0 {
        return Err(DequantError::ShapeMismatch(
            "group_size must be non-zero".to_string(),
        ));
    }
    let packed_numel: usize = weight_info.shape.iter().product();
    let numel = match weight_info.dtype.as_str() {
        "U32" => {
            let bits_total = packed_numel.checked_mul(32).ok_or_else(|| {
                DequantError::ShapeMismatch("packed weight bit count overflows".to_string())
            })?;
            if bits_total % bits as usize != 0 {
                return Err(DequantError::ShapeMismatch(format!(
                    "packed U32 bits {bits_total} are not divisible by {bits}"
                )));
            }
            bits_total / bits as usize
        }
        "U8" => {
            if 8 % bits != 0 {
                return Err(DequantError::UnsupportedBits(bits));
            }
            packed_numel * (8 / bits as usize)
        }
        dtype => {
            return Err(DequantError::ShapeMismatch(format!(
                "Unsupported packed weight dtype: {}",
                dtype
            )));
        }
    };
    let mut output = vec![0.0f32; numel];
    let mask = (1u32 << bits) - 1;
    match weight_info.dtype.as_str() {
        "U32" => {
            for i in 0..numel {
                let bitpos = i * bits as usize;
                let word = bitpos / 32;
                let shift = (bitpos % 32) as u32;
                let mut packed = read_u32_le(weight_data, word)? >> shift;
                if shift + bits > 32 {
                    packed |= read_u32_le(weight_data, word + 1)? << (32 - shift);
                }
                let q = (packed & mask) as f32;
                output[i] = affine_at(
                    q,
                    i,
                    group_size,
                    scales_data,
                    &scales_info.dtype,
                    biases_data,
                    &biases_info.dtype,
                )?;
            }
        }
        "U8" => {
            let values_per_element = 8 / bits as usize;
            for i in 0..numel {
                let packed_index = i / values_per_element;
                let shift = bits * (i % values_per_element) as u32;
                let packed = *weight_data.get(packed_index).ok_or_else(|| {
                    DequantError::ShapeMismatch(format!(
                        "packed U8 weight ends before byte {packed_index}"
                    ))
                })? as u32;
                let q = ((packed >> shift) & mask) as f32;
                output[i] = affine_at(
                    q,
                    i,
                    group_size,
                    scales_data,
                    &scales_info.dtype,
                    biases_data,
                    &biases_info.dtype,
                )?;
            }
        }
        _ => unreachable!("validated packed weight dtype"),
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::dequantize;
    use crate::safetensors::SafeTensorFile;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn expands_eight_four_bit_values_from_one_u32() {
        let header = r#"{"weight":{"dtype":"U32","shape":[1,1],"data_offsets":[0,4]},"weight.scales":{"dtype":"BF16","shape":[1,1],"data_offsets":[4,6]},"weight.biases":{"dtype":"BF16","shape":[1,1],"data_offsets":[6,8]}}"#;
        let path = std::env::temp_dir().join(format!(
            "core_lm_optiq_u32_{}_{}.safetensors",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(&0x7654_3210u32.to_le_bytes());
        data.extend_from_slice(&0x3f80u16.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        fs::write(&path, data).unwrap();

        let file = SafeTensorFile::open(&path).unwrap();
        let values = dequantize(
            file.tensor_data("weight").unwrap(),
            file.tensor_info("weight").unwrap(),
            file.tensor_data("weight.scales").unwrap(),
            file.tensor_info("weight.scales").unwrap(),
            file.tensor_data("weight.biases").unwrap(),
            file.tensor_info("weight.biases").unwrap(),
            4,
            64,
        )
        .unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(values, (0..8).map(|value| value as f32).collect::<Vec<_>>());
    }

    fn pack_low_first(values: &[u32], bits: u32) -> Vec<u8> {
        let nwords = (values.len() * bits as usize + 31) / 32;
        let mut words = vec![0u32; nwords];
        for (i, &value) in values.iter().enumerate() {
            let bitpos = i * bits as usize;
            let word = bitpos / 32;
            let shift = (bitpos % 32) as u32;
            words[word] |= value << shift;
            if shift + bits > 32 {
                words[word + 1] |= value >> (32 - shift);
            }
        }
        words.into_iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn write_affine_tensor(weight: &[u8], scales: &[u8], biases: &[u8]) -> std::path::PathBuf {
        let header = format!(
            r#"{{"weight":{{"dtype":"U32","shape":[1,{}],"data_offsets":[0,{}]}},"weight.scales":{{"dtype":"BF16","shape":[1,{}],"data_offsets":[{},{}]}},"weight.biases":{{"dtype":"BF16","shape":[1,{}],"data_offsets":[{},{}]}}}}"#,
            weight.len() / 4,
            weight.len(),
            scales.len() / 2,
            weight.len(),
            weight.len() + scales.len(),
            biases.len() / 2,
            weight.len() + scales.len(),
            weight.len() + scales.len() + biases.len(),
        );
        let path = std::env::temp_dir().join(format!(
            "core_lm_optiq_pack_{}_{}.safetensors",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(weight);
        data.extend_from_slice(scales);
        data.extend_from_slice(biases);
        fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn splits_four_bit_groups_at_the_requested_size() {
        let path = write_affine_tensor(
            &0x7654_3210u32.to_le_bytes(),
            &0x4000_3f80u32.to_le_bytes(),
            &0u32.to_le_bytes(),
        );
        let file = SafeTensorFile::open(&path).unwrap();
        let values = dequantize(
            file.tensor_data("weight").unwrap(),
            file.tensor_info("weight").unwrap(),
            file.tensor_data("weight.scales").unwrap(),
            file.tensor_info("weight.scales").unwrap(),
            file.tensor_data("weight.biases").unwrap(),
            file.tensor_info("weight.biases").unwrap(),
            4,
            4,
        )
        .unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 8.0, 10.0, 12.0, 14.0]);
    }

    #[test]
    fn unpacks_codes_that_cross_a_u32_boundary() {
        // 32 five-bit codes fill 5 u32 words exactly, and several codes cross a word.
        let five: Vec<u32> = (0..32).collect();
        let path = write_affine_tensor(
            &pack_low_first(&five, 5),
            &0x4000_3f80u32.to_le_bytes(),
            &0u32.to_le_bytes(),
        );
        let file = SafeTensorFile::open(&path).unwrap();
        let values = dequantize(
            file.tensor_data("weight").unwrap(),
            file.tensor_info("weight").unwrap(),
            file.tensor_data("weight.scales").unwrap(),
            file.tensor_info("weight.scales").unwrap(),
            file.tensor_data("weight.biases").unwrap(),
            file.tensor_info("weight.biases").unwrap(),
            5,
            16,
        )
        .unwrap();
        fs::remove_file(&path).unwrap();
        let mut expected: Vec<f32> = (0..16).map(|value| value as f32).collect();
        expected.extend((16..32).map(|value| value as f32 * 2.0));
        assert_eq!(values, expected);

        // 16 six-bit codes fill 3 u32 words exactly.
        let six: Vec<u32> = (0..16).collect();
        let path = write_affine_tensor(
            &pack_low_first(&six, 6),
            &0x4000_3f80u32.to_le_bytes(),
            &0u32.to_le_bytes(),
        );
        let file = SafeTensorFile::open(&path).unwrap();
        let values = dequantize(
            file.tensor_data("weight").unwrap(),
            file.tensor_info("weight").unwrap(),
            file.tensor_data("weight.scales").unwrap(),
            file.tensor_info("weight.scales").unwrap(),
            file.tensor_data("weight.biases").unwrap(),
            file.tensor_info("weight.biases").unwrap(),
            6,
            8,
        )
        .unwrap();
        fs::remove_file(&path).unwrap();
        let mut expected: Vec<f32> = (0..8).map(|value| value as f32).collect();
        expected.extend((8..16).map(|value| value as f32 * 2.0));
        assert_eq!(values, expected);
    }
}

pub fn load_f32_tensor(file: &SafeTensorFile, name: &str) -> Result<Vec<f32>, DequantError> {
    let data = file.tensor_data(name)?;
    let info = file.tensor_info(name)?;
    let numel: usize = info.shape.iter().product();
    let mut output = vec![0.0f32; numel];

    match info.dtype.as_str() {
        "F32" => {
            for i in 0..numel {
                output[i] = read_f32_le(data, i * 4);
            }
        }
        "BF16" => {
            for i in 0..numel {
                output[i] = read_bf16_le(data, i * 2);
            }
        }
        "F16" => {
            for i in 0..numel {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                output[i] = half_to_f32(bits);
            }
        }
        _ => {
            return Err(DequantError::ShapeMismatch(format!(
                "Unsupported dtype for direct load: {}",
                info.dtype
            )));
        }
    }
    Ok(output)
}
