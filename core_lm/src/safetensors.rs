//! Pure Rust safetensors parser. No Python dependency.
//!
//! Format: [8-byte LE header_len][JSON header][tensor data]
//! Header JSON maps tensor name -> {dtype, shape, data_offsets: [start, end]}.

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug)]
pub enum SafeTensorError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidHeader(String),
    TensorNotFound(String),
    DtypeMismatch { expected: String, found: String },
}

impl std::fmt::Display for SafeTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafeTensorError::Io(e) => write!(f, "IO error: {}", e),
            SafeTensorError::Json(e) => write!(f, "JSON error: {}", e),
            SafeTensorError::InvalidHeader(s) => write!(f, "Invalid header: {}", s),
            SafeTensorError::TensorNotFound(s) => write!(f, "Tensor not found: {}", s),
            SafeTensorError::DtypeMismatch { expected, found } => {
                write!(f, "Dtype mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for SafeTensorError {}

impl From<std::io::Error> for SafeTensorError {
    fn from(e: std::io::Error) -> Self {
        SafeTensorError::Io(e)
    }
}

impl From<serde_json::Error> for SafeTensorError {
    fn from(e: serde_json::Error) -> Self {
        SafeTensorError::Json(e)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

pub struct SafeTensorFile {
    mmap: Mmap,
    pub tensors: HashMap<String, TensorInfo>,
    data_start: usize,
}

impl SafeTensorFile {
    pub fn open(path: &Path) -> Result<Self, SafeTensorError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| SafeTensorError::Io(e))? };

        if mmap.len() < 8 {
            return Err(SafeTensorError::InvalidHeader("File too small".to_string()));
        }

        let mut cursor = &mmap[0..8];
        let header_len = cursor.read_u64::<LittleEndian>()? as usize;

        if mmap.len() < 8 + header_len {
            return Err(SafeTensorError::InvalidHeader(
                "Header extends beyond file".to_string(),
            ));
        }

        let header_json = std::str::from_utf8(&mmap[8..8 + header_len])
            .map_err(|_| SafeTensorError::InvalidHeader("Header is not valid UTF-8".to_string()))?;

        // Safetensors reserves `__metadata__` for file-level values, whose
        // schema differs from a tensor descriptor. Decode entries separately
        // so MLX/Qwen shard metadata does not get parsed as a tensor.
        let entries: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)?;
        let mut tensors = HashMap::new();
        for (name, value) in entries {
            if name != "__metadata__" {
                tensors.insert(name, serde_json::from_value(value)?);
            }
        }

        Ok(Self {
            mmap,
            tensors,
            data_start: 8 + header_len,
        })
    }

    pub fn tensor_data(&self, name: &str) -> Result<&[u8], SafeTensorError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| SafeTensorError::TensorNotFound(name.to_string()))?;
        let (start, end) = info.data_offsets;
        let abs_start = self.data_start + start;
        let abs_end = self.data_start + end;
        if abs_end > self.mmap.len() {
            return Err(SafeTensorError::InvalidHeader(format!(
                "Tensor {} extends beyond file",
                name
            )));
        }
        Ok(&self.mmap[abs_start..abs_end])
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo, SafeTensorError> {
        self.tensors
            .get(name)
            .ok_or_else(|| SafeTensorError::TensorNotFound(name.to_string()))
    }

    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }
}

pub fn dtype_size(dtype: &str) -> Result<usize, SafeTensorError> {
    match dtype {
        "F32" => Ok(4),
        "F16" => Ok(2),
        "BF16" => Ok(2),
        "U8" => Ok(1),
        "I8" => Ok(1),
        "I32" => Ok(4),
        "I64" => Ok(8),
        _ => Err(SafeTensorError::InvalidHeader(format!(
            "Unknown dtype: {}",
            dtype
        ))),
    }
}

pub fn num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::SafeTensorFile;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn ignores_file_metadata_when_parsing_tensor_headers() {
        let header = r#"{"__metadata__":{"format":"mlx"},"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let path = std::env::temp_dir().join(format!(
            "core_lm_safetensors_metadata_{}_{}.safetensors",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.push(42);
        fs::write(&path, data).unwrap();

        let file = SafeTensorFile::open(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(file.num_tensors(), 1);
        assert_eq!(file.tensor_data("weight").unwrap(), &[42]);
    }
}
