#include <metal_stdlib>
using namespace metal;

// Fused dequant + GEMV for OptiQ mixed-quantization (Muse Glimmer q4km-gs128)
// 
// Weight format:
//   - Packed ints, low bits first, in U32 (tightly packed, codes may cross word boundaries)
//   - Scales: F32 per group (group_size elements per group, groups run along rows)
//   - Biases: F32 per group (zero-point)
//   - Out_scale: F32 per row (applied after dequant)
//
// Dequant: w[i] = (q[i] * scale[i/group_size] + bias[i/group_size]) * out_scale[row]
//
// Each thread computes one output element: out[row] = dot(W[row,:], x[:])

// Helper: read F32 scale/bias (supports F32 only for now; F16/BF16 need conversion)
inline float read_f32(device const uchar *data, uint idx) {
    // Reinterpret 4 bytes as float (little-endian)
    uint bits = (uint)data[idx*4] | ((uint)data[idx*4+1] << 8) |
                ((uint)data[idx*4+2] << 16) | ((uint)data[idx*4+3] << 24);
    return as_type<float>(bits);
}

// 4-bit quantized GEMV
//   packed: U32 array, 8 values per word, low bits first
//   scales, biases: F32 arrays, one per group
//   out_scale: F32 array, one per row (may be null -> 1.0)
//   x: F32 input vector (cols elements)
//   out: F32 output vector (rows elements)
//   rows, cols, group_size, bits=4
kernel void q4_gemv(
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
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    
    const uint bits = 4;
    const uint mask = 0xF;
    const uint values_per_word = 8; // 32/4
    
    float acc = 0.0f;
    float row_scale = has_out_scale ? out_scale[row] : 1.0f;
    
    // Iterate over columns
    for (uint col = 0; col < cols; ++col) {
        // Unpack 4-bit value
        uint bitpos = col * bits;
        uint word = bitpos / 32;
        uint shift = bitpos % 32;
        // For 4-bit, values never cross word boundaries (4 divides 32)
        uint q = (packed[row * (cols / values_per_word) + word] >> shift) & mask;
        
        // Dequant
        uint g = col / group_size;
        // scales/biases are per-row? No - per group along the row
        // Actually scales shape is [rows, cols/group_size], so index is row * num_groups + g
        uint num_groups = cols / group_size;
        float scale = read_f32(scales, row * num_groups + g);
        float bias = read_f32(biases, row * num_groups + g);
        float w = (float(q) * scale + bias) * row_scale;
        
        acc += w * x[col];
    }
    
    out[row] = acc;
}

// 6-bit quantized GEMV
//   packed: U32 array, tightly packed, codes may cross word boundaries
kernel void q6_gemv(
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
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    
    const uint bits = 6;
    const uint mask = 0x3F;
    
    float acc = 0.0f;
    float row_scale = has_out_scale ? out_scale[row] : 1.0f;
    
    // Words per row for 6-bit packing
    uint total_bits = cols * bits;
    uint words_per_row = (total_bits + 31) / 32;
    
    uint num_groups = cols / group_size;
    
    for (uint col = 0; col < cols; ++col) {
        // Unpack 6-bit value (may cross word boundary)
        uint bitpos = col * bits;
        uint word = bitpos / 32;
        uint shift = bitpos % 32;
        uint word_idx = row * words_per_row + word;
        
        uint q = packed[word_idx] >> shift;
        if (shift + bits > 32) {
            // Crosses into next word
            q |= packed[word_idx + 1] << (32 - shift);
        }
        q &= mask;
        
        // Dequant
        uint g = col / group_size;
        float scale = read_f32(scales, row * num_groups + g);
        float bias = read_f32(biases, row * num_groups + g);
        float w = (float(q) * scale + bias) * row_scale;
        
        acc += w * x[col];
    }
    
    out[row] = acc;
}
