//! Metal vs CPU verification harness for quantized GEMV kernels.
//!
//! Compares Metal kernel output against CPU reference (optiq::dequantize + naive matvec)
//! using real weight tensors from the Glimmer checkpoint.

use metal_quant::MetalQuantGemm;
use std::error::Error;

// CPU reference: dequantize then naive GEMV
// This mirrors core_lm/src/optiq.rs dequantize() + simple dot product

fn cpu_reference_gemv(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    out_scale: Option<&[f32]>,
    x: &[f32],
    rows: usize,
    cols: usize,
    bits: u32,
    group_size: usize,
) -> Vec<f32> {
    let mask = (1u32 << bits) - 1;
    let mut out = vec![0.0f32; rows];
    
    let words_per_row = if bits == 4 {
        cols / 8
    } else {
        (cols * bits as usize + 31) / 32
    };
    let num_groups = cols / group_size;
    
    for row in 0..rows {
        let row_out_scale = out_scale.map(|s| s[row]).unwrap_or(1.0);
        let mut acc = 0.0f32;
        
        for col in 0..cols {
            // Unpack
            let bitpos = col * bits as usize;
            let word = bitpos / 32;
            let shift = (bitpos % 32) as u32;
            let word_idx = row * words_per_row + word;
            
            let mut q = packed[word_idx] >> shift;
            if shift + bits > 32 {
                q |= packed[word_idx + 1] << (32 - shift);
            }
            q &= mask;
            
            // Dequant
            let g = col / group_size;
            let scale = scales[row * num_groups + g];
            let bias = biases[row * num_groups + g];
            let w = (q as f32 * scale + bias) * row_out_scale;
            
            acc += w * x[col];
        }
        out[row] = acc;
    }
    out
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Metal vs CPU Quantized GEMV Verification");
    println!("========================================\n");
    
    // Load MSL source (same directory as this binary's manifest)
    let msl_path = std::env::var("METAL_KERNEL_PATH")
        .unwrap_or_else(|_| "quant_gemv.metal".to_string());
    let msl = std::fs::read_to_string(&msl_path)
        .map_err(|e| format!("Failed to read {}: {}", msl_path, e))?;
    let gemm = MetalQuantGemm::new(&msl)?;
    println!("✓ Metal kernels compiled\n");
    
    // Test 1: Small 4-bit matrix
    println!("Test 1: 4-bit GEMV (16x64, group_size=32)");
    let rows = 16;
    let cols = 64;
    let group_size = 32;
    
    // Generate deterministic test data
    let mut packed = vec![0u32; rows * cols / 8];
    for (i, w) in packed.iter_mut().enumerate() {
        *w = (i as u32).wrapping_mul(0x9E3779B9).wrapping_add(0x12345678);
    }
    let num_groups = cols / group_size;
    let scales: Vec<f32> = (0..rows * num_groups).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let biases: Vec<f32> = (0..rows * num_groups).map(|i| 0.001 * (i as f32)).collect();
    let out_scale: Vec<f32> = (0..rows).map(|i| 1.0 + 0.01 * (i as f32)).collect();
    let x: Vec<f32> = (0..cols).map(|i| 0.1 * ((i % 7) as f32 - 3.0)).collect();
    
    // Convert scales/biases to bytes (F32 LE)
    let scales_bytes: Vec<u8> = scales.iter().flat_map(|f| f.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|f| f.to_le_bytes()).collect();
    
    let cpu_out = cpu_reference_gemv(
        &packed, &scales, &biases, Some(&out_scale), &x,
        rows, cols, 4, group_size,
    );
    let metal_out = gemm.q4_gemv(
        &packed, &scales_bytes, &biases_bytes, Some(&out_scale), &x,
        rows as u32, cols as u32, group_size as u32,
    )?;
    
    let max_err = cpu_out.iter().zip(metal_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_rel = cpu_out.iter().zip(metal_out.iter())
        .map(|(a, b)| if a.abs() > 1e-6 { (a - b).abs() / a.abs() } else { 0.0 })
        .fold(0.0f32, f32::max);
    
    println!("  Max abs error: {:.2e}", max_err);
    println!("  Max rel error: {:.2e}", max_rel);
    if max_err < 1e-4 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL\n");
    }
    
    // Test 2: 6-bit matrix
    println!("Test 2: 6-bit GEMV (16x128, group_size=128)");
    let rows = 16;
    let cols = 128;
    let group_size = 128;
    let words_per_row = (cols * 6 + 31) / 32;
    
    let mut packed = vec![0u32; rows * words_per_row];
    for (i, w) in packed.iter_mut().enumerate() {
        *w = (i as u32).wrapping_mul(0x85EBCA6B).wrapping_add(0xABCDEF01);
    }
    let num_groups = cols / group_size;
    let scales: Vec<f32> = (0..rows * num_groups).map(|i| 0.02 * (i as f32 + 1.0)).collect();
    let biases: Vec<f32> = (0..rows * num_groups).map(|i| -0.005 * (i as f32)).collect();
    let x: Vec<f32> = (0..cols).map(|i| 0.05 * ((i % 11) as f32 - 5.0)).collect();
    
    let scales_bytes: Vec<u8> = scales.iter().flat_map(|f| f.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|f| f.to_le_bytes()).collect();
    
    let cpu_out = cpu_reference_gemv(
        &packed, &scales, &biases, None, &x,
        rows, cols, 6, group_size,
    );
    let metal_out = gemm.q6_gemv(
        &packed, &scales_bytes, &biases_bytes, None, &x,
        rows as u32, cols as u32, group_size as u32,
    )?;
    
    let max_err = cpu_out.iter().zip(metal_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("  Max abs error: {:.2e}", max_err);
    if max_err < 1e-4 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL\n");
    }
    
    println!("Done.");
    Ok(())
}
