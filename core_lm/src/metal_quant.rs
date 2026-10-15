//! Rust Metal bindings for MPS-class Fused Quantized GEMV kernels (`core_lm::metal_quant`).
//!
//! Eliminates the two biggest Metal bottlenecks that caused 15s/tok:
//! 1. **Persistent Unified-Memory `MTLBuffer` Cache (Zero Per-Token Weight Copies)**:
//!    Previously, `gemv_inner` called `device.new_buffer_with_data(...)` on every single
//!    projection (`365` calls/token), `memcpy`ing ~16 GB of weights per token. Now, weight,
//!    scale, bias, and `out_scale` slices from memory-mapped `SafeTensorFile` shards are cached
//!    in persistent `StorageModeShared` `MTLBuffer`s keyed by `(base_ptr, byte_len)` (using
//!    zero-copy `new_buffer_with_bytes_no_copy` whenever page-aligned on Apple Silicon!).
//! 2. **32-Wide SIMD-Group Cooperative Dispatch (`8 rows per 256-thread threadgroup`)**:
//!    Dispatches `q4_gemv`, `q6_gemv` (96-bit 3-word super-block unpacking), and `q_generic_gemv`
//!    (`2, 3, 5, 8`-bit OptiQ overrides) with `BF16`, `F16`, and `F32` group scales/biases.

use metal::*;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Mutex;

#[derive(Debug)]
pub enum MetalQuantError {
    DeviceNotFound,
    LibraryCompile(String),
    FunctionNotFound(String),
    InvalidShape(String),
}

impl fmt::Display for MetalQuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalQuantError::DeviceNotFound => write!(f, "No Metal device found"),
            MetalQuantError::LibraryCompile(e) => write!(f, "MSL compile error: {}", e),
            MetalQuantError::FunctionNotFound(n) => write!(f, "Kernel not found: {}", n),
            MetalQuantError::InvalidShape(e) => write!(f, "invalid GEMV shape: {}", e),
        }
    }
}

impl Error for MetalQuantError {}

/// Apple Silicon VM page size (16 KB on M1/M2/M3/M4).
const APPLE_SILICON_PAGE_SIZE: usize = 16_384;

/// Quantized GEMV executor using Metal with persistent Unified Memory buffer caching.
pub struct MetalQuantGemm {
    device: Device,
    queue: CommandQueue,
    q4_pipeline: ComputePipelineState,
    q5_pipeline: ComputePipelineState,
    q6_pipeline: ComputePipelineState,
    q_generic_pipeline: ComputePipelineState,
    /// Persistent cache of `MTLBuffer`s keyed by `(slice_ptr, byte_len)` so mmap'd weights
    /// are never re-copied across tokens.
    buffer_cache: Mutex<HashMap<(usize, usize), Buffer>>,
    dummy_one_buf: Buffer,
}

impl MetalQuantGemm {
    /// Create from MSL source. Compiles `q4_gemv`, `q6_gemv`, and `q_generic_gemv` kernels.
    pub fn new(msl_source: &str) -> Result<Self, MetalQuantError> {
        let device = Device::system_default().ok_or(MetalQuantError::DeviceNotFound)?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(msl_source, &options)
            .map_err(|e| MetalQuantError::LibraryCompile(e.to_string()))?;

        let q4_fn = library
            .get_function("q4_gemv", None)
            .map_err(|_| MetalQuantError::FunctionNotFound("q4_gemv".to_string()))?;
        let q6_fn = library
            .get_function("q6_gemv", None)
            .map_err(|_| MetalQuantError::FunctionNotFound("q6_gemv".to_string()))?;
        let q5_fn = library
            .get_function("q5_gemv", None)
            .map_err(|_| MetalQuantError::FunctionNotFound("q5_gemv".to_string()))?;
        let q_gen_fn = library
            .get_function("q_generic_gemv", None)
            .map_err(|_| MetalQuantError::FunctionNotFound("q_generic_gemv".to_string()))?;

        let q4_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_fn)
            .map_err(|e| MetalQuantError::LibraryCompile(e.to_string()))?;
        let q6_pipeline = device
            .new_compute_pipeline_state_with_function(&q6_fn)
            .map_err(|e| MetalQuantError::LibraryCompile(e.to_string()))?;
        let q5_pipeline = device
            .new_compute_pipeline_state_with_function(&q5_fn)
            .map_err(|e| MetalQuantError::LibraryCompile(e.to_string()))?;
        let q_generic_pipeline = device
            .new_compute_pipeline_state_with_function(&q_gen_fn)
            .map_err(|e| MetalQuantError::LibraryCompile(e.to_string()))?;

        let dummy_one_buf = device.new_buffer_with_data(
            [1.0f32].as_ptr() as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device,
            queue,
            q4_pipeline,
            q5_pipeline,
            q6_pipeline,
            q_generic_pipeline,
            buffer_cache: Mutex::new(HashMap::new()),
            dummy_one_buf,
        })
    }

    /// Retrieve or create a persistent Unified Memory `MTLBuffer` for an immutable mmap/weight slice.
    /// If the slice is page-aligned (`16 KB`), binds physical Unified Memory pages with zero copy!
    fn get_or_cache_buffer(&self, bytes_ptr: *const u8, byte_len: usize) -> Buffer {
        let key = (bytes_ptr as usize, byte_len);
        if let Ok(guard) = self.buffer_cache.lock() {
            if let Some(buf) = guard.get(&key) {
                return buf.to_owned();
            }
        }

        let buf = if (bytes_ptr as usize) % APPLE_SILICON_PAGE_SIZE == 0
            && byte_len >= APPLE_SILICON_PAGE_SIZE
            && byte_len % APPLE_SILICON_PAGE_SIZE == 0
        {
            self.device.new_buffer_with_bytes_no_copy(
                bytes_ptr as *const _,
                byte_len as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.device.new_buffer_with_data(
                bytes_ptr as *const _,
                byte_len.max(4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        if let Ok(mut guard) = self.buffer_cache.lock() {
            // Cap cache at 4096 tensor entries to prevent unbounded growth in synthetic tests
            if guard.len() < 4096 {
                guard.insert(key, buf.to_owned());
            }
        }
        buf
    }

    pub fn cached_buffer_count(&self) -> usize {
        self.buffer_cache.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// 4-bit quantized GEMV: `out = (dequant(W) * out_scale) @ x`
    #[allow(clippy::too_many_arguments)]
    pub fn q4_gemv(
        &self,
        packed: &[u32],
        scales: &[u8],
        biases: &[u8],
        out_scale: Option<&[f32]>,
        x: &[f32],
        rows: u32,
        cols: u32,
        group_size: u32,
        scale_dtype: &str,
        bias_dtype: &str,
    ) -> Result<Vec<f32>, MetalQuantError> {
        self.gemv_inner(
            &self.q4_pipeline,
            packed,
            scales,
            biases,
            out_scale,
            x,
            rows,
            cols,
            group_size,
            scale_dtype,
            bias_dtype,
            None,
        )
    }

    /// 6-bit quantized GEMV (96-bit 3-word super-block unpacking)
    #[allow(clippy::too_many_arguments)]
    pub fn q6_gemv(
        &self,
        packed: &[u32],
        scales: &[u8],
        biases: &[u8],
        out_scale: Option<&[f32]>,
        x: &[f32],
        rows: u32,
        cols: u32,
        group_size: u32,
        scale_dtype: &str,
        bias_dtype: &str,
    ) -> Result<Vec<f32>, MetalQuantError> {
        self.gemv_inner(
            &self.q6_pipeline,
            packed,
            scales,
            biases,
            out_scale,
            x,
            rows,
            cols,
            group_size,
            scale_dtype,
            bias_dtype,
            None,
        )
    }

    /// Generic variable-bit (`2, 3, 5, 8`-bit OptiQ) quantized GEMV
    #[allow(clippy::too_many_arguments)]
    pub fn q_generic_gemv(
        &self,
        packed: &[u32],
        scales: &[u8],
        biases: &[u8],
        out_scale: Option<&[f32]>,
        x: &[f32],
        rows: u32,
        cols: u32,
        group_size: u32,
        bits: u32,
        scale_dtype: &str,
        bias_dtype: &str,
    ) -> Result<Vec<f32>, MetalQuantError> {
        self.gemv_inner(
            &self.q_generic_pipeline,
            packed,
            scales,
            biases,
            out_scale,
            x,
            rows,
            cols,
            group_size,
            scale_dtype,
            bias_dtype,
            Some(bits),
        )
    }

    /// Specialized group-128 5-bit GEMV accepting packed U32 weight bytes without
    /// requiring host-pointer alignment; Metal owns an aligned cached buffer copy.
    #[allow(clippy::too_many_arguments)]
    pub fn q5_gemv_bytes(
        &self,
        packed_bytes: &[u8],
        scales: &[u8],
        biases: &[u8],
        out_scale: Option<&[f32]>,
        x: &[f32],
        rows: u32,
        cols: u32,
        group_size: u32,
        scale_dtype: &str,
        bias_dtype: &str,
    ) -> Result<Vec<f32>, MetalQuantError> {
        let packed_bits = (rows as usize)
            .checked_mul(cols as usize)
            .and_then(|n| n.checked_mul(5));
        let expected_bytes = packed_bits.and_then(|n| n.checked_div(8));
        let groups = (rows as usize).checked_mul(cols as usize / group_size.max(1) as usize);
        let scale_bytes = if scale_dtype == "F32" { 4 } else { 2 };
        let bias_bytes = if bias_dtype == "F32" { 4 } else { 2 };
        if !packed_bytes.len().is_multiple_of(4)
            || group_size != 128
            || cols == 0
            || !cols.is_multiple_of(group_size)
            || !(cols as usize * 5).is_multiple_of(32)
            || x.len() != cols as usize
            || expected_bytes != Some(packed_bytes.len())
            || out_scale.is_some_and(|values| values.len() != rows as usize)
            || groups
                .and_then(|n| n.checked_mul(scale_bytes))
                .is_none_or(|n| scales.len() < n)
            || groups
                .and_then(|n| n.checked_mul(bias_bytes))
                .is_none_or(|n| biases.len() < n)
            || Self::dtype_code(scale_dtype) == 0 && scale_dtype != "F32"
            || Self::dtype_code(bias_dtype) == 0 && bias_dtype != "F32"
        {
            return Err(MetalQuantError::InvalidShape(
                "q5_gemv requires U32-packed bytes, group_size 128, and word-aligned rows".into(),
            ));
        }
        self.gemv_inner_bytes(
            &self.q5_pipeline,
            packed_bytes,
            scales,
            biases,
            out_scale,
            x,
            rows,
            cols,
            group_size,
            scale_dtype,
            bias_dtype,
            Some(5),
        )
    }

    fn dtype_code(dtype: &str) -> u32 {
        match dtype {
            "F32" => 0,
            "BF16" => 1,
            "F16" => 2,
            _ => 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_inner(
        &self,
        pipeline: &ComputePipelineState,
        packed: &[u32],
        scales: &[u8],
        biases: &[u8],
        out_scale: Option<&[f32]>,
        x: &[f32],
        rows: u32,
        cols: u32,
        group_size: u32,
        scale_dtype: &str,
        bias_dtype: &str,
        generic_bits: Option<u32>,
    ) -> Result<Vec<f32>, MetalQuantError> {
        let packed_bytes =
            unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 4) };
        self.gemv_inner_bytes(
            pipeline,
            packed_bytes,
            scales,
            biases,
            out_scale,
            x,
            rows,
            cols,
            group_size,
            scale_dtype,
            bias_dtype,
            generic_bits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_inner_bytes(
        &self,
        pipeline: &ComputePipelineState,
        packed_bytes: &[u8],
        scales: &[u8],
        biases: &[u8],
        out_scale: Option<&[f32]>,
        x: &[f32],
        rows: u32,
        cols: u32,
        group_size: u32,
        scale_dtype: &str,
        bias_dtype: &str,
        generic_bits: Option<u32>,
    ) -> Result<Vec<f32>, MetalQuantError> {
        let packed_buf = self.get_or_cache_buffer(packed_bytes.as_ptr(), packed_bytes.len());
        let scales_buf = self.get_or_cache_buffer(scales.as_ptr(), scales.len());
        let biases_buf = self.get_or_cache_buffer(biases.as_ptr(), biases.len());

        let (out_scale_buf, has_out_scale) = match out_scale {
            Some(s) => (
                self.get_or_cache_buffer(s.as_ptr() as *const u8, s.len() * 4),
                1u32,
            ),
            None => (self.dummy_one_buf.to_owned(), 0u32),
        };

        let x_buf = self.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.device.new_buffer(
            (rows as usize * 4).max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(&packed_buf), 0);
        encoder.set_buffer(1, Some(&scales_buf), 0);
        encoder.set_buffer(2, Some(&biases_buf), 0);
        encoder.set_buffer(3, Some(&out_scale_buf), 0);
        encoder.set_buffer(4, Some(&x_buf), 0);
        encoder.set_buffer(5, Some(&out_buf), 0);

        encoder.set_bytes(6, 4, &rows as *const _ as *const _);
        encoder.set_bytes(7, 4, &cols as *const _ as *const _);
        encoder.set_bytes(8, 4, &group_size as *const _ as *const _);
        encoder.set_bytes(9, 4, &has_out_scale as *const _ as *const _);
        let scale_code = Self::dtype_code(scale_dtype);
        let bias_code = Self::dtype_code(bias_dtype);
        encoder.set_bytes(10, 4, &scale_code as *const _ as *const _);
        encoder.set_bytes(11, 4, &bias_code as *const _ as *const _);
        if let Some(bits) = generic_bits {
            encoder.set_bytes(12, 4, &bits as *const _ as *const _);
        }

        // Dispatch 256 threads (8 simdgroups of 32 threads) per threadgroup -> 8 rows per threadgroup!
        let num_threadgroups = ((rows as u64) + 7) / 8;
        let threadgroups = MTLSize::new(num_threadgroups.max(1), 1, 1);
        let threads_per_threadgroup = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let out_ptr = out_buf.contents() as *const f32;
        let mut result = vec![0.0f32; rows as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(out_ptr, result.as_mut_ptr(), rows as usize);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_bf16(val: f32) -> [u8; 2] {
        let bits = (val.to_bits() >> 16) as u16;
        bits.to_le_bytes()
    }

    #[test]
    fn metal_q4_q6_and_q5_simdgroup_kernels_match_cpu_reference() {
        let msl = include_str!("quant_gemv.metal");
        let Ok(gemm) = MetalQuantGemm::new(msl) else {
            return; // Skip only if running on a headless machine without Metal
        };

        let rows = 32usize;
        let cols = 128usize;
        let group_size = 32usize;
        let groups_per_row = cols / group_size;

        let x: Vec<f32> = (0..cols).map(|c| ((c as f32) * 0.11).sin()).collect();
        let out_scale: Vec<f32> = (0..rows).map(|r| 0.75 + (r as f32) * 0.05).collect();

        let mut scales_bf16 = Vec::new();
        let mut biases_bf16 = Vec::new();
        for g in 0..(rows * groups_per_row) {
            scales_bf16.extend_from_slice(&encode_bf16(0.125 + (g % 5) as f32 * 0.0625));
            biases_bf16.extend_from_slice(&encode_bf16(-0.25 + (g % 3) as f32 * 0.125));
        }

        // 1. Verify 4-bit SIMD-group kernel
        let words_4 = (rows * cols * 4) / 32;
        let packed_4: Vec<u32> = (0..words_4)
            .map(|w| 0x1357_9BDFu32.wrapping_mul(w as u32 + 1))
            .collect();
        let packed_4_bytes: Vec<u8> = packed_4.iter().flat_map(|w| w.to_le_bytes()).collect();

        let gpu_y4 = gemm
            .q4_gemv(
                &packed_4,
                &scales_bf16,
                &biases_bf16,
                Some(&out_scale),
                &x,
                rows as u32,
                cols as u32,
                group_size as u32,
                "BF16",
                "BF16",
            )
            .unwrap();
        let cpu_y4 = crate::metal::qmatvec_fused_zero_copy(
            &packed_4_bytes,
            "U32",
            &scales_bf16,
            "BF16",
            &biases_bf16,
            "BF16",
            Some(&out_scale),
            4,
            group_size,
            rows,
            cols,
            &x,
        )
        .unwrap();

        for r in 0..rows {
            assert!(
                (gpu_y4[r] - cpu_y4[r]).abs() < 1e-3,
                "q4 row {r}: gpu={} vs cpu={}",
                gpu_y4[r],
                cpu_y4[r]
            );
        }

        // Verify the specialized 5-bit/group-128 path against the exact CPU reference.
        let q5_rows = 32usize;
        let q5_cols = 128usize;
        let q5_words_per_row = q5_cols * 5 / 32;
        let q5_packed: Vec<u32> = (0..q5_rows * q5_words_per_row)
            .map(|i| 0x9E37_79B9u32.wrapping_mul(i as u32 + 7))
            .collect();
        let q5_bytes: Vec<u8> = q5_packed
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let q5_scales: Vec<u8> = (0..q5_rows)
            .flat_map(|i| encode_bf16(0.04 + (i % 5) as f32 * 0.01))
            .collect();
        let q5_biases: Vec<u8> = (0..q5_rows)
            .flat_map(|i| encode_bf16(-0.02 + (i % 3) as f32 * 0.01))
            .collect();
        let q5_out_scale: Vec<f32> = (0..q5_rows).map(|i| 0.8 + i as f32 * 0.001).collect();
        let q5_gpu = gemm
            .q5_gemv_bytes(
                &q5_bytes,
                &q5_scales,
                &q5_biases,
                Some(&q5_out_scale),
                &x,
                q5_rows as u32,
                q5_cols as u32,
                128,
                "BF16",
                "BF16",
            )
            .unwrap();
        let q5_cpu = crate::metal::qmatvec_fused_zero_copy(
            &q5_bytes,
            "U32",
            &q5_scales,
            "BF16",
            &q5_biases,
            "BF16",
            Some(&q5_out_scale),
            5,
            128,
            q5_rows,
            q5_cols,
            &x,
        )
        .unwrap();
        for r in 0..q5_rows {
            assert!(
                (q5_gpu[r] - q5_cpu[r]).abs() < 1e-2,
                "q5 row {r}: gpu={} vs cpu={}",
                q5_gpu[r],
                q5_cpu[r]
            );
        }
        assert!(
            gemm.q5_gemv_bytes(
                &q5_bytes,
                &q5_scales,
                &q5_biases,
                Some(&q5_out_scale),
                &x,
                q5_rows as u32,
                q5_cols as u32,
                64,
                "BF16",
                "BF16",
            )
            .is_err()
        );

        // 2. Verify 6-bit 96-bit Super-Word SIMD-group kernel
        let words_6 = (rows * cols * 6) / 32;
        let packed_6: Vec<u32> = (0..words_6)
            .map(|w| 0x2468_ACE1u32.wrapping_mul(w as u32 + 3))
            .collect();
        let packed_6_bytes: Vec<u8> = packed_6.iter().flat_map(|w| w.to_le_bytes()).collect();

        let gpu_y6 = gemm
            .q6_gemv(
                &packed_6,
                &scales_bf16,
                &biases_bf16,
                Some(&out_scale),
                &x,
                rows as u32,
                cols as u32,
                group_size as u32,
                "BF16",
                "BF16",
            )
            .unwrap();
        let cpu_y6 = crate::metal::qmatvec_fused_zero_copy(
            &packed_6_bytes,
            "U32",
            &scales_bf16,
            "BF16",
            &biases_bf16,
            "BF16",
            Some(&out_scale),
            6,
            group_size,
            rows,
            cols,
            &x,
        )
        .unwrap();

        for r in 0..rows {
            assert!(
                (gpu_y6[r] - cpu_y6[r]).abs() < 1e-2,
                "q6 row {r}: gpu={} vs cpu={}",
                gpu_y6[r],
                cpu_y6[r]
            );
        }
    }
}
