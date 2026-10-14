//! Rust Metal bindings for quantized GEMV kernels.
//!
//! Uses the `metal` crate to dispatch MSL kernels for fused dequant + matvec.
//! Supports 4-bit (group 32) and 6-bit (group 128) OptiQ quantization.

use metal::*;
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum MetalQuantError {
    DeviceNotFound,
    LibraryCompile(String),
    FunctionNotFound(String),
}

impl fmt::Display for MetalQuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalQuantError::DeviceNotFound => write!(f, "No Metal device found"),
            MetalQuantError::LibraryCompile(e) => write!(f, "MSL compile error: {}", e),
            MetalQuantError::FunctionNotFound(n) => write!(f, "Kernel not found: {}", n),
        }
    }
}

impl Error for MetalQuantError {}

/// Quantized GEMV executor using Metal.
pub struct MetalQuantGemm {
    device: Device,
    queue: CommandQueue,
    q4_pipeline: ComputePipelineState,
    q6_pipeline: ComputePipelineState,
}

impl MetalQuantGemm {
    /// Create from MSL source. Compiles q4_gemv and q6_gemv kernels.
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
        
        let q4_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_fn)
            .map_err(|e| MetalQuantError::LibraryCompile(e.to_string()))?;
        let q6_pipeline = device
            .new_compute_pipeline_state_with_function(&q6_fn)
            .map_err(|e| MetalQuantError::LibraryCompile(e.to_string()))?;
        
        Ok(Self {
            device,
            queue,
            q4_pipeline,
            q6_pipeline,
        })
    }
    
    /// 4-bit quantized GEMV: out = (dequant(W) * out_scale) @ x
    /// 
    /// # Arguments
    /// * `packed` - U32 packed weights, shape [rows, cols/8]
    /// * `scales` - F32 scales, shape [rows, cols/group_size]
    /// * `biases` - F32 biases, shape [rows, cols/group_size]
    /// * `out_scale` - Optional F32 per-row scale, shape [rows]
    /// * `x` - F32 input vector, shape [cols]
    /// * `rows`, `cols`, `group_size`
    pub fn q4_gemv(
        &self,
        packed: &[u32],
        scales: &[u8],  // F32 bytes
        biases: &[u8],  // F32 bytes
        out_scale: Option<&[f32]>,
        x: &[f32],
        rows: u32,
        cols: u32,
        group_size: u32,
    ) -> Result<Vec<f32>, MetalQuantError> {
        self.gemv_inner(
            &self.q4_pipeline,
            packed, scales, biases, out_scale, x,
            rows, cols, group_size,
        )
    }
    
    /// 6-bit quantized GEMV
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
    ) -> Result<Vec<f32>, MetalQuantError> {
        self.gemv_inner(
            &self.q6_pipeline,
            packed, scales, biases, out_scale, x,
            rows, cols, group_size,
        )
    }
    
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
    ) -> Result<Vec<f32>, MetalQuantError> {
        // Create buffers
        let packed_buf = self.device.new_buffer_with_data(
            packed.as_ptr() as *const _,
            (packed.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scales_buf = self.device.new_buffer_with_data(
            scales.as_ptr() as *const _,
            scales.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let biases_buf = self.device.new_buffer_with_data(
            biases.as_ptr() as *const _,
            biases.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Out scale: use dummy 1.0 if None
        let (out_scale_buf, has_out_scale) = match out_scale {
            Some(s) => (
                self.device.new_buffer_with_data(
                    s.as_ptr() as *const _,
                    (s.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
                1u32,
            ),
            None => (
                self.device.new_buffer_with_data(
                    [1.0f32].as_ptr() as *const _,
                    4,
                    MTLResourceOptions::StorageModeShared,
                ),
                0u32,
            ),
        };
        
        let x_buf = self.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.device.new_buffer(
            (rows as usize * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Encode
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        
        encoder.set_buffer(0, Some(&packed_buf), 0);
        encoder.set_buffer(1, Some(&scales_buf), 0);
        encoder.set_buffer(2, Some(&biases_buf), 0);
        encoder.set_buffer(3, Some(&out_scale_buf), 0);
        encoder.set_buffer(4, Some(&x_buf), 0);
        encoder.set_buffer(5, Some(&out_buf), 0);
        
        // Set constants
        encoder.set_bytes(6, 4, &rows as *const _ as *const _);
        encoder.set_bytes(7, 4, &cols as *const _ as *const _);
        encoder.set_bytes(8, 4, &group_size as *const _ as *const _);
        encoder.set_bytes(9, 4, &has_out_scale as *const _ as *const _);
        
        // Dispatch: one thread per row
        let grid = MTLSize::new(rows as u64, 1, 1);
        let threadgroup = MTLSize::new(
            pipeline.max_total_threads_per_threadgroup().min(rows as u64),
            1, 1,
        );
        encoder.dispatch_threads(grid, threadgroup);
        encoder.end_encoding();
        
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        // Read back
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
    
    // Note: These tests require a Metal device (macOS only).
    // They are ignored on non-Metal platforms.
    
    #[test]
    #[ignore]
    fn q4_gemv_matches_cpu() {
        // TODO: Compare against CPU dequant + GEMV
    }
}
