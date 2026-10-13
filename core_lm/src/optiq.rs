//! OptiQ dequantizer: 4-bit and 8-bit affine quantization with group_size=64.
//!
//! Each quantized tensor has:
//!   {name}.weight — packed ints (4-bit: 2 per byte, low nibble first)
//!   {name}.scales — f32 or bf16 scale per 64-element group
//!   {name}.biases — f32 or bf16 zero-point per 64-element group
//!
//! Dequant: w[i] = q[i] * scale[i/64] + bias[i/64]

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

pub fn dequantize(
    weight_data: &[u8],
    weight_info: &TensorInfo,
    scales_data: &[u8],
    scales_info: &TensorInfo,
    biases_data: &[u8],
    bits: u32,
) -> Result<Vec<f32>, DequantError> {
    let packed_numel: usize = weight_info.shape.iter().product();

    let values_per_element = match weight_info.dtype.as_str() {
        "U32" => 32 / bits as usize,
        "U8" => 8 / bits as usize,
        dtype => {
            return Err(DequantError::ShapeMismatch(format!(
                "Unsupported packed weight dtype: {}",
                dtype
            )));
        }
    };
    let numel = packed_numel * values_per_element;
    let mut output = vec![0.0f32; numel];

    match bits {
        4 => {
            for i in 0..numel {
                let packed_index = i / values_per_element;
                let shift = 4 * (i % values_per_element);
                let packed = match weight_info.dtype.as_str() {
                    "U32" => u32::from_le_bytes(
                        weight_data[packed_index * 4..packed_index * 4 + 4]
                            .try_into()
                            .unwrap(),
                    ),
                    "U8" => weight_data[packed_index] as u32,
                    _ => unreachable!("validated packed weight dtype"),
                };
                let q = ((packed >> shift) & 0x0F) as f32;
                let g = i / 64;
                let scale = read_scale_bias(scales_data, &scales_info.dtype, g)?;
                let bias = read_scale_bias(biases_data, &scales_info.dtype, g)?;
                output[i] = q * scale + bias;
            }
        }
        8 => {
            for i in 0..numel {
                let packed_index = i / values_per_element;
                let shift = 8 * (i % values_per_element);
                let packed = match weight_info.dtype.as_str() {
                    "U32" => u32::from_le_bytes(
                        weight_data[packed_index * 4..packed_index * 4 + 4]
                            .try_into()
                            .unwrap(),
                    ),
                    "U8" => weight_data[packed_index] as u32,
                    _ => unreachable!("validated packed weight dtype"),
                };
                let q = ((packed >> shift) & 0xFF) as f32;
                let g = i / 64;
                let scale = read_scale_bias(scales_data, &scales_info.dtype, g)?;
                let bias = read_scale_bias(biases_data, &scales_info.dtype, g)?;
                output[i] = q * scale + bias;
            }
        }
        _ => return Err(DequantError::UnsupportedBits(bits)),
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
            4,
        )
        .unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(values, (0..8).map(|value| value as f32).collect::<Vec<_>>());
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
