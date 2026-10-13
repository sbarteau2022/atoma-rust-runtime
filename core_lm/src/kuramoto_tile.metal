#include <metal_stdlib>
using namespace metal;

// FIXED: pi-unit Q32.31 table, corrected gain 1304065748
constant long CORDIC_GAIN = 1304065748;
constant long ATAN_TABLE[32] = {
    536870912, 316933406, 167458907, 85004756, 42667331,
    21354465, 10679838, 5340245, 2670163, 1335087,
    667544, 333772, 166886, 83443, 41722,
    20861, 10430, 5215, 2608, 1304,
    652, 326, 163, 81, 41,
    20, 10, 5, 3, 1, 1, 0
};

inline long cordic_sin(long theta) {
    long z = (long)(int)theta; // hardware torus
    long pi_half = 1l << 30;
    long pi = 1l << 31;
    long sign = 1; // sign stays 1 (bug fix: removed sign=-1)
    if (z > pi_half) { z = pi - z; }
    else if (z < -pi_half) { z = -pi - z; }
    long x = CORDIC_GAIN;
    long y = 0;
    for (int i = 0; i < 32; i++) {
        long x_shift = x >> i;
        long y_shift = y >> i;
        if (z >= 0) {
            x -= y_shift; y += x_shift; z -= ATAN_TABLE[i];
        } else {
            x += y_shift; y -= x_shift; z += ATAN_TABLE[i];
        }
    }
    return y * sign;
}

kernel void kuramoto_step(
    device long* theta [[buffer(0)]],
    device const long* omega [[buffer(1)]],
    device long* d_theta_out [[buffer(2)]],
    constant long& dt [[buffer(3)]],
    constant long& k_coupling [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    long my_theta = theta[id];
    long sum_coupling = 0;
    for (uint j = 0; j < grid_size; j++) {
        if (id == j) continue;
        long phase_diff = (long)(int)(theta[j] - my_theta);
        sum_coupling += cordic_sin(phase_diff);
    }
    long interaction = (k_coupling * (sum_coupling / grid_size)) >> 31;
    long total_derivative = omega[id] + interaction;
    long delta = (total_derivative * dt) >> 31;
    d_theta_out[id] = delta;
    theta[id] = (long)(int)(my_theta + delta);
}
