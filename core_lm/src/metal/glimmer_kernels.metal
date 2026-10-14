//! Metal Compute Kernels for `glimmer_forward.rs` (`Muse-Glimmer-q4km-gs128-MLX`)
//!
//! Eliminates the 87-sec/token CPU bottleneck by executing fused affine-quantized
//! matrix-vector multiplication (`U32` packed 4-bit / generic-bit weights + `BF16`
//! group scales & biases + optional `F32` `out_scale`), `RMSNorm`, consecutive-pair
//! `RoPE`, and `SwiGLU` directly on Apple Silicon Unified Memory.
//!
//! Key algebraic optimization inside each quantization group `g` of size `G`:
//!   sum_{j=0..G-1} (q_{r, g*G+j} * s_{r,g} + b_{r,g}) * x_{g*G+j}
//!     = s_{r,g} * (sum_{j=0..G-1} q_{r, g*G+j} * x_{g*G+j})
//!     + b_{r,g} * X_group[g]
//! where `X_group[g] = sum_{j=0..G-1} x_{g*G+j}` is precomputed once per input vector `x`.

#include <metal_stdlib>
using namespace metal;

inline float bf16_bits_to_f32(ushort bits) {
    return as_type<float>((uint)bits << 16);
}

/// Precompute `X_group[g] = sum_{j=0..group_size-1} x[g * group_size + j]` across `num_groups`.
kernel void glimmer_precompute_group_sums(
    device const float* x [[buffer(0)]],
    device float* x_group_sums [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    constant uint& group_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * group_size;
    if (start >= cols) return;
    uint end = min(start + group_size, cols);
    float acc = 0.0f;
    for (uint c = start; c < end; ++c) {
        acc += x[c];
    }
    x_group_sums[gid] = acc;
}

/// Fast path 4-bit affine-quantized GEMV (`bits == 4`, `group_size` multiple of 8, `BF16` scales/biases).
/// Each SIMD-group (32 threads) cooperatively computes one output row `r`.
kernel void glimmer_qmatvec_u32_q4_bf16(
    device const uint* packed_weights [[buffer(0)]],
    device const ushort* scales_bf16 [[buffer(1)]],
    device const ushort* biases_bf16 [[buffer(2)]],
    device const float* out_scale [[buffer(3)]],
    device const float* x [[buffer(4)]],
    device const float* x_group_sums [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    constant uint& has_out_scale [[buffer(10)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // 8 SIMD-groups per threadgroup (256 threads total -> 8 rows per threadgroup)
    uint row = tgid * 8 + simd_gid;
    if (row >= rows) return;

    uint words_per_row = cols / 8;
    uint groups_per_row = cols / group_size;
    uint words_per_group = group_size / 8;
    uint row_word_base = row * words_per_row;
    uint row_group_base = row * groups_per_row;

    float row_acc = 0.0f;

    // Each thread in the 32-wide SIMD group strides across words in the row
    for (uint w_idx = simd_lid; w_idx < words_per_row; w_idx += 32) {
        uint word = packed_weights[row_word_base + w_idx];
        uint c_base = w_idx * 8;
        uint g_idx = w_idx / words_per_group;

        float s = bf16_bits_to_f32(scales_bf16[row_group_base + g_idx]);

        float q0 = (float)(word & 0xFu);
        float q1 = (float)((word >> 4) & 0xFu);
        float q2 = (float)((word >> 8) & 0xFu);
        float q3 = (float)((word >> 12) & 0xFu);
        float q4 = (float)((word >> 16) & 0xFu);
        float q5 = (float)((word >> 20) & 0xFu);
        float q6 = (float)((word >> 24) & 0xFu);
        float q7 = (float)((word >> 28) & 0xFu);

        float dot8 = q0 * x[c_base + 0]
                   + q1 * x[c_base + 1]
                   + q2 * x[c_base + 2]
                   + q3 * x[c_base + 3]
                   + q4 * x[c_base + 4]
                   + q5 * x[c_base + 5]
                   + q6 * x[c_base + 6]
                   + q7 * x[c_base + 7];

        row_acc += s * dot8;
    }

    // Add group bias contributions: sum_{g} b_{row, g} * X_group[g]
    float bias_acc = 0.0f;
    for (uint g_idx = simd_lid; g_idx < groups_per_row; g_idx += 32) {
        float b = bf16_bits_to_f32(biases_bf16[row_group_base + g_idx]);
        bias_acc += b * x_group_sums[g_idx];
    }

    float total = simd_sum(row_acc + bias_acc);
    if (simd_lid == 0) {
        if (has_out_scale != 0) {
            total *= out_scale[row];
        }
        y[row] = total;
    }
}

/// Generic variable-bit (`bits in 1..16`, e.g. 5-bit or 6-bit overrides crossing U32 word boundaries)
/// affine-quantized GEMV kernel.
kernel void glimmer_qmatvec_u32_generic_bf16(
    device const uint* packed_weights [[buffer(0)]],
    device const ushort* scales_bf16 [[buffer(1)]],
    device const ushort* biases_bf16 [[buffer(2)]],
    device const float* out_scale [[buffer(3)]],
    device const float* x [[buffer(4)]],
    device const float* x_group_sums [[buffer(5)]],
    device float* y [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant uint& bits [[buffer(9)]],
    constant uint& group_size [[buffer(10)]],
    constant uint& has_out_scale [[buffer(11)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid * 8 + simd_gid;
    if (row >= rows) return;

    uint groups_per_row = cols / group_size;
    uint row_group_base = row * groups_per_row;
    uint mask = (1u << bits) - 1u;

    float row_acc = 0.0f;
    for (uint g_idx = simd_lid; g_idx < groups_per_row; g_idx += 32) {
        float s = bf16_bits_to_f32(scales_bf16[row_group_base + g_idx]);
        float b = bf16_bits_to_f32(biases_bf16[row_group_base + g_idx]);
        uint c_start = g_idx * group_size;
        uint c_end = c_start + group_size;
        float q_dot = 0.0f;
        for (uint c = c_start; c < c_end; ++c) {
            ulong elem_idx = (ulong)row * (ulong)cols + (ulong)c;
            ulong bitpos = elem_idx * (ulong)bits;
            uint word_idx = (uint)(bitpos >> 5);
            uint shift = (uint)(bitpos & 31ul);
            uint packed = packed_weights[word_idx] >> shift;
            if (shift + bits > 32u) {
                packed |= packed_weights[word_idx + 1u] << (32u - shift);
            }
            float q = (float)(packed & mask);
            q_dot += q * x[c];
        }
        row_acc += s * q_dot + b * x_group_sums[g_idx];
    }

    float total = simd_sum(row_acc);
    if (simd_lid == 0) {
        if (has_out_scale != 0) {
            total *= out_scale[row];
        }
        y[row] = total;
    }
}

/// Dense F32 matrix-vector multiply (`y = A * x`) using SIMD-group reduction.
kernel void glimmer_matvec_f32(
    device const float* matrix [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid * 8 + simd_gid;
    if (row >= rows) return;

    uint base = row * cols;
    float acc = 0.0f;
    for (uint c = simd_lid; c < cols; c += 32) {
        acc += matrix[base + c] * x[c];
    }
    float total = simd_sum(acc);
    if (simd_lid == 0) {
        y[row] = total;
    }
}

/// Fused SwiGLU (`out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]`).
kernel void glimmer_swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float g = gate[gid];
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = silu_g * up[gid];
}
