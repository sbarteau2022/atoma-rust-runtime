#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Adapted MPS-Class Fused Dequant + GEMV Kernels for OptiQ Mixed-Quantization
// (`Muse-Glimmer-q4km-gs128-MLX` & `Qwen3.6-35B-A3B-OptiQ-4bit`)
// =============================================================================
//
// Key Hardware Adaptations over Naive 1-Thread-Per-Row Kernels:
// 1. **32-Wide SIMD-Group Cooperative Row Reduction (`8 simdgroups = 256 threads/TG`)**:
//    Instead of 1 thread looping serially over `cols = 5120..15360`, all 32 threads in a
//    `simdgroup` stride coalesced across the row and reduce in 1 cycle via `simd_sum()`.
// 2. **Group-Hoisted Affine Dequantization**:
//    Uses the exact identity:
//      sum_{j in group g} (q_j * s_g + b_g) * x_j = s_g * (sum_j q_j * x_j) + b_g * (sum_j x_j)
//    so `scale` and `bias` are loaded ONCE per group instead of on every column.
// 3. **96-Bit (`3 x u32 = 16 weights`) Super-Word Unpacking for Tight 6-Bit OptiQ (`q6_gemv`)**:
//    Because `lcm(6, 32) = 96 bits = 3 u32 words = 16 weights`, every 16 consecutive 6-bit
//    codes align bit-exactly on a 3-word boundary. Unpacking 16 weights from `(w0, w1, w2)`
//    eliminates per-element division, modulo, and cross-word branch divergence (`if (shift + 6 > 32)`).
// 4. **Aligned Halfword/Word Reads for `BF16` / `F16` / `F32` Scales & Biases**.

inline float read_scale_or_bias(
    device const uchar* raw,
    uint idx,
    uint dtype_code // 0 = F32, 1 = BF16, 2 = F16
) {
    if (dtype_code == 0u) {
        device const float* f_ptr = reinterpret_cast<device const float*>(raw);
        return f_ptr[idx];
    } else if (dtype_code == 1u) {
        device const ushort* u16_ptr = reinterpret_cast<device const ushort*>(raw);
        uint bits = (uint)u16_ptr[idx];
        return as_type<float>(bits << 16);
    } else {
        device const half* h_ptr = reinterpret_cast<device const half*>(raw);
        return (float)h_ptr[idx];
    }
}

// -----------------------------------------------------------------------------
// 1. 4-Bit Fused SIMD-Group GEMV (`q4_gemv`)
//    8 nibbles per U32 word, group_size divisible by 8 (e.g. 32, 64, 128).
//    Launched with 256 threads per threadgroup (8 simdgroups -> 8 rows/TG).
// -----------------------------------------------------------------------------
kernel void q4_gemv(
    device const uint* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const uchar* biases [[buffer(2)]],
    device const float* out_scale [[buffer(3)]],
    device const float* x [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    constant uint& has_out_scale [[buffer(9)]],
    constant uint& scale_dtype [[buffer(10)]],
    constant uint& bias_dtype [[buffer(11)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid * 8u + simd_gid;
    if (row >= rows) return;

    uint words_per_row = cols >> 3; // cols / 8
    uint num_groups = cols / group_size;
    uint words_per_group = group_size >> 3;
    uint row_word_base = row * words_per_row;
    uint row_group_base = row * num_groups;

    float row_acc = 0.0f;

    // If num_groups >= 32, stride lanes across groups; otherwise stride lanes across words
    if (words_per_group >= 1u && (cols % 8u == 0u)) {
        for (uint w_idx = simd_lid; w_idx < words_per_row; w_idx += 32u) {
            uint word = packed[row_word_base + w_idx];
            uint c_base = w_idx << 3;
            uint g = w_idx / words_per_group;
            uint sb_idx = row_group_base + g;

            float scale = read_scale_or_bias(scales, sb_idx, scale_dtype);
            float bias = read_scale_or_bias(biases, sb_idx, bias_dtype);

            float x0 = x[c_base + 0u];
            float x1 = x[c_base + 1u];
            float x2 = x[c_base + 2u];
            float x3 = x[c_base + 3u];
            float x4 = x[c_base + 4u];
            float x5 = x[c_base + 5u];
            float x6 = x[c_base + 6u];
            float x7 = x[c_base + 7u];

            float q0 = (float)(word & 0xFu);
            float q1 = (float)((word >> 4) & 0xFu);
            float q2 = (float)((word >> 8) & 0xFu);
            float q3 = (float)((word >> 12) & 0xFu);
            float q4 = (float)((word >> 16) & 0xFu);
            float q5 = (float)((word >> 20) & 0xFu);
            float q6 = (float)((word >> 24) & 0xFu);
            float q7 = (float)((word >> 28) & 0xFu);

            float q_dot = (q0 * x0 + q1 * x1) + (q2 * x2 + q3 * x3)
                        + (q4 * x4 + q5 * x5) + (q6 * x6 + q7 * x7);
            float x_sum = (x0 + x1) + (x2 + x3) + (x4 + x5) + (x6 + x7);

            row_acc += scale * q_dot + bias * x_sum;
        }
    } else {
        for (uint col = simd_lid; col < cols; col += 32u) {
            uint bitpos = col * 4u;
            uint word = bitpos >> 5;
            uint shift = bitpos & 31u;
            uint q = (packed[row_word_base + word] >> shift) & 0xFu;
            uint g = col / group_size;
            uint sb_idx = row_group_base + g;
            float scale = read_scale_or_bias(scales, sb_idx, scale_dtype);
            float bias = read_scale_or_bias(biases, sb_idx, bias_dtype);
            row_acc += (float(q) * scale + bias) * x[col];
        }
    }

    float total = simd_sum(row_acc);
    if (simd_lid == 0u) {
        float row_scale = (has_out_scale != 0u) ? out_scale[row] : 1.0f;
        out[row] = total * row_scale;
    }
}

// -----------------------------------------------------------------------------
// 2. 6-Bit Tight-Packed Super-Word SIMD-Group GEMV (`q6_gemv`)
//    Unpacks 16 consecutive 6-bit codes from 3 consecutive U32 words (96 bits)
//    with zero branch divergence when group_size is a multiple of 16 (32, 64, 128).
// -----------------------------------------------------------------------------
kernel void q6_gemv(
    device const uint* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const uchar* biases [[buffer(2)]],
    device const float* out_scale [[buffer(3)]],
    device const float* x [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    constant uint& has_out_scale [[buffer(9)]],
    constant uint& scale_dtype [[buffer(10)]],
    constant uint& bias_dtype [[buffer(11)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid * 8u + simd_gid;
    if (row >= rows) return;

    uint total_bits = cols * 6u;
    uint words_per_row = (total_bits + 31u) >> 5;
    uint row_word_base = row * words_per_row;
    uint num_groups = cols / group_size;
    uint row_group_base = row * num_groups;

    float row_acc = 0.0f;

    // Fast 96-bit super-word path when cols and group_size are multiples of 16
    if ((cols & 15u) == 0u && (group_size & 15u) == 0u) {
        uint num_blocks16 = cols >> 4; // cols / 16
        uint blocks16_per_group = group_size >> 4;

        for (uint blk = simd_lid; blk < num_blocks16; blk += 32u) {
            uint w_base = row_word_base + blk * 3u;
            uint w0 = packed[w_base + 0u];
            uint w1 = packed[w_base + 1u];
            uint w2 = packed[w_base + 2u];

            uint g = blk / blocks16_per_group;
            uint sb_idx = row_group_base + g;
            float scale = read_scale_or_bias(scales, sb_idx, scale_dtype);
            float bias = read_scale_or_bias(biases, sb_idx, bias_dtype);

            uint c = blk << 4;
            // 16 6-bit codes packed across w0 (bits 0..31), w1 (bits 32..63), w2 (bits 64..95)
            float q0  = (float)( w0        & 0x3Fu);
            float q1  = (float)((w0 >>  6) & 0x3Fu);
            float q2  = (float)((w0 >> 12) & 0x3Fu);
            float q3  = (float)((w0 >> 18) & 0x3Fu);
            float q4  = (float)((w0 >> 24) & 0x3Fu);
            float q5  = (float)(((w0 >> 30) | (w1 << 2)) & 0x3Fu);
            float q6  = (float)((w1 >>  4) & 0x3Fu);
            float q7  = (float)((w1 >> 10) & 0x3Fu);
            float q8  = (float)((w1 >> 16) & 0x3Fu);
            float q9  = (float)((w1 >> 22) & 0x3Fu);
            float q10 = (float)(((w1 >> 28) | (w2 << 4)) & 0x3Fu);
            float q11 = (float)((w2 >>  2) & 0x3Fu);
            float q12 = (float)((w2 >>  8) & 0x3Fu);
            float q13 = (float)((w2 >> 14) & 0x3Fu);
            float q14 = (float)((w2 >> 20) & 0x3Fu);
            float q15 = (float)((w2 >> 26) & 0x3Fu);

            float q_dot = q0 * x[c + 0u]  + q1 * x[c + 1u]  + q2 * x[c + 2u]  + q3 * x[c + 3u]
                        + q4 * x[c + 4u]  + q5 * x[c + 5u]  + q6 * x[c + 6u]  + q7 * x[c + 7u]
                        + q8 * x[c + 8u]  + q9 * x[c + 9u]  + q10 * x[c + 10u] + q11 * x[c + 11u]
                        + q12 * x[c + 12u] + q13 * x[c + 13u] + q14 * x[c + 14u] + q15 * x[c + 15u];

            float x_sum = (x[c + 0u] + x[c + 1u] + x[c + 2u] + x[c + 3u])
                        + (x[c + 4u] + x[c + 5u] + x[c + 6u] + x[c + 7u])
                        + (x[c + 8u] + x[c + 9u] + x[c + 10u] + x[c + 11u])
                        + (x[c + 12u] + x[c + 13u] + x[c + 14u] + x[c + 15u]);

            row_acc += scale * q_dot + bias * x_sum;
        }
    } else {
        for (uint col = simd_lid; col < cols; col += 32u) {
            uint bitpos = col * 6u;
            uint word = bitpos >> 5;
            uint shift = bitpos & 31u;
            uint word_idx = row_word_base + word;
            uint q = packed[word_idx] >> shift;
            if (shift + 6u > 32u) {
                q |= packed[word_idx + 1u] << (32u - shift);
            }
            q &= 0x3Fu;
            uint g = col / group_size;
            uint sb_idx = row_group_base + g;
            float scale = read_scale_or_bias(scales, sb_idx, scale_dtype);
            float bias = read_scale_or_bias(biases, sb_idx, bias_dtype);
            row_acc += (float(q) * scale + bias) * x[col];
        }
    }

    float total = simd_sum(row_acc);
    if (simd_lid == 0u) {
        float row_scale = (has_out_scale != 0u) ? out_scale[row] : 1.0f;
        out[row] = total * row_scale;
    }
}

// -----------------------------------------------------------------------------
// 3. Generic Arbitrary-Bit (`bits in 1..16`: 2, 3, 5, 8-bit OptiQ overrides)
//    SIMD-Group Cooperative GEMV (`q_generic_gemv`).
// -----------------------------------------------------------------------------
kernel void q_generic_gemv(
    device const uint* packed [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const uchar* biases [[buffer(2)]],
    device const float* out_scale [[buffer(3)]],
    device const float* x [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant uint& group_size [[buffer(8)]],
    constant uint& has_out_scale [[buffer(9)]],
    constant uint& scale_dtype [[buffer(10)]],
    constant uint& bias_dtype [[buffer(11)]],
    constant uint& bits [[buffer(12)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid * 8u + simd_gid;
    if (row >= rows) return;

    uint mask = (1u << bits) - 1u;
    uint num_groups = cols / group_size;
    uint row_group_base = row * num_groups;
    ulong row_elem_base = (ulong)row * (ulong)cols;

    float row_acc = 0.0f;
    for (uint col = simd_lid; col < cols; col += 32u) {
        ulong elem_idx = row_elem_base + (ulong)col;
        ulong bitpos = elem_idx * (ulong)bits;
        uint word_idx = (uint)(bitpos >> 5);
        uint shift = (uint)(bitpos & 31ul);
        uint q = packed[word_idx] >> shift;
        if (shift + bits > 32u) {
            q |= packed[word_idx + 1u] << (32u - shift);
        }
        q &= mask;

        uint g = col / group_size;
        uint sb_idx = row_group_base + g;
        float scale = read_scale_or_bias(scales, sb_idx, scale_dtype);
        float bias = read_scale_or_bias(biases, sb_idx, bias_dtype);
        row_acc += (float(q) * scale + bias) * x[col];
    }

    float total = simd_sum(row_acc);
    if (simd_lid == 0u) {
        float row_scale = (has_out_scale != 0u) ? out_scale[row] : 1.0f;
        out[row] = total * row_scale;
    }
}
