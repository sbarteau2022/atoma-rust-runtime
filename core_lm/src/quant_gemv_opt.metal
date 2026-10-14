#include <metal_stdlib>
using namespace metal;

// Optimized fused dequant + GEMV for OptiQ mixed-quantization
// MPS-style optimizations:
// - Threadgroup memory for x-vector (loaded once, shared by all threads in group)
// - Coalesced reads via cooperative loading
// - FMA for accumulation

// Dtype helpers
inline float read_f32(device const uchar *data, uint idx) {
    uint bits = (uint)data[idx*4] | ((uint)data[idx*4+1] << 8) |
                ((uint)data[idx*4+2] << 16) | ((uint)data[idx*4+3] << 24);
    return as_type<float>(bits);
}

inline float read_bf16(device const uchar *data, uint idx) {
    uint bits = (uint)data[idx*2] | ((uint)data[idx*2+1] << 8);
    return as_type<float>(bits << 16);
}

inline float read_f16(device const uchar *data, uint idx) {
    uint bits = (uint)data[idx*2] | ((uint)data[idx*2+1] << 8);
    uint sign = (bits >> 15) & 1;
    uint exp = (bits >> 10) & 0x1F;
    uint mant = bits & 0x3FF;
    uint f32_bits;
    if (exp == 0) {
        f32_bits = (mant == 0) ? (sign << 31) : (sign << 31 | 0x38000000 | (mant << 13));
    } else if (exp == 31) {
        f32_bits = sign << 31 | 0x7F800000 | (mant << 13);
    } else {
        f32_bits = sign << 31 | ((exp + 112) << 23) | (mant << 13);
    }
    return as_type<float>(f32_bits);
}

inline float read_scale(device const uchar *data, uint idx, uint dtype) {
    if (dtype == 0) return read_f32(data, idx);
    if (dtype == 1) return read_bf16(data, idx);
    return read_f16(data, idx);
}

// 4-bit GEMV with threadgroup x-cache
// Threadgroup size should be <= 256 for optimal shared memory usage
kernel void q4_gemv_opt(
    device const uint *packed [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const uchar *biases [[buffer(2)]],
    device const float *out_scale [[buffer(3)]],
    device const float *x [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    constant uint &group_size [[buffer(8)]],
    constant uint &has_out_scale [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]],
    constant uint &bias_dtype [[buffer(11)]],
    constant uint &x_cache_size [[buffer(12)]],  // cols, must fit in threadgroup memory
    uint row [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row >= rows) return;
    
    // Threadgroup memory for x (MPS-style: load once, share)
    threadgroup float x_cache[4096];  // Max 4096 cols; adjust based on actual
    
    // Cooperative load of x into threadgroup memory
    for (uint i = tid; i < cols; i += tg_size) {
        x_cache[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    const uint mask = 0xF;
    const uint values_per_word = 8;
    uint words_per_row = cols / values_per_word;
    uint num_groups = cols / group_size;
    
    float acc = 0.0f;
    float row_scale = has_out_scale ? out_scale[row] : 1.0f;
    device const uint *row_packed = packed + row * words_per_row;
    
    // Unrolled 4x for ILP
    uint col = 0;
    for (; col + 4 <= cols; col += 4) {
        float w0, w1, w2, w3;
        // Unpack 4 values
        uint b0 = col * 4, b1 = (col+1) * 4, b2 = (col+2) * 4, b3 = (col+3) * 4;
        uint q0 = (row_packed[b0 / 32] >> (b0 % 32)) & mask;
        uint q1 = (row_packed[b1 / 32] >> (b1 % 32)) & mask;
        uint q2 = (row_packed[b2 / 32] >> (b2 % 32)) & mask;
        uint q3 = (row_packed[b3 / 32] >> (b3 % 32)) & mask;
        
        uint g0 = col / group_size, g1 = (col+1) / group_size;
        uint g2 = (col+2) / group_size, g3 = (col+3) / group_size;
        uint s0 = row * num_groups + g0, s1 = row * num_groups + g1;
        uint s2 = row * num_groups + g2, s3 = row * num_groups + g3;
        
        w0 = (float(q0) * read_scale(scales, s0, scale_dtype) + read_scale(biases, s0, bias_dtype)) * row_scale;
        w1 = (float(q1) * read_scale(scales, s1, scale_dtype) + read_scale(biases, s1, bias_dtype)) * row_scale;
        w2 = (float(q2) * read_scale(scales, s2, scale_dtype) + read_scale(biases, s2, bias_dtype)) * row_scale;
        w3 = (float(q3) * read_scale(scales, s3, scale_dtype) + read_scale(biases, s3, bias_dtype)) * row_scale;
        
        // FMA accumulation
        acc = fma(w0, x_cache[col], acc);
        acc = fma(w1, x_cache[col+1], acc);
        acc = fma(w2, x_cache[col+2], acc);
        acc = fma(w3, x_cache[col+3], acc);
    }
    // Remainder
    for (; col < cols; ++col) {
        uint bitpos = col * 4;
        uint q = (row_packed[bitpos / 32] >> (bitpos % 32)) & mask;
        uint g = col / group_size;
        uint sb_idx = row * num_groups + g;
        float w = (float(q) * read_scale(scales, sb_idx, scale_dtype) + 
                   read_scale(biases, sb_idx, bias_dtype)) * row_scale;
        acc = fma(w, x_cache[col], acc);
    }
    
    out[row] = acc;
}

// 6-bit GEMV with threadgroup x-cache
kernel void q6_gemv_opt(
    device const uint *packed [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const uchar *biases [[buffer(2)]],
    device const float *out_scale [[buffer(3)]],
    device const float *x [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    constant uint &group_size [[buffer(8)]],
    constant uint &has_out_scale [[buffer(9)]],
    constant uint &scale_dtype [[buffer(10)]],
    constant uint &bias_dtype [[buffer(11)]],
    constant uint &x_cache_size [[buffer(12)]],
    uint row [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (row >= rows) return;
    
    threadgroup float x_cache[4096];
    
    for (uint i = tid; i < cols; i += tg_size) {
        x_cache[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    const uint mask = 0x3F;
    uint words_per_row = (cols * 6 + 31) / 32;
    uint num_groups = cols / group_size;
    
    float acc = 0.0f;
    float row_scale = has_out_scale ? out_scale[row] : 1.0f;
    device const uint *row_packed = packed + row * words_per_row;
    
    for (uint col = 0; col < cols; ++col) {
        uint bitpos = col * 6;
        uint word = bitpos / 32;
        uint shift = bitpos % 32;
        
        uint q = row_packed[word] >> shift;
        if (shift + 6 > 32) {
            q |= row_packed[word + 1] << (32 - shift);
        }
        q &= mask;
        
        uint g = col / group_size;
        uint sb_idx = row * num_groups + g;
        float w = (float(q) * read_scale(scales, sb_idx, scale_dtype) + 
                   read_scale(biases, sb_idx, bias_dtype)) * row_scale;
        acc = fma(w, x_cache[col], acc);
    }
    
    out[row] = acc;
}
