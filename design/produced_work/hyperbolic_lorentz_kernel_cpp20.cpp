// Autonomous Monster-Hybrid Synthesized Kernel: `hyperbolic_lorentz_kernel_cpp20.cpp`
// C++20 constexpr Q32.31 Hyperbolic Lorentz Inner Product & Geodesic Cursor Step.

#include <cstdint>
#include <array>

namespace atoma::geometry {

using Q32_31 = std::int64_t;
constexpr Q32_31 ONE_Q32 = (1LL << 31);

struct alignas(32) LorentzVecQ32 {
    Q32_31 x0;
    Q32_31 x1;
    Q32_31 x2;
    Q32_31 x3;

    static constexpr LorentzVecQ32 origin() noexcept {
        return LorentzVecQ32{ONE_Q32, 0, 0, 0};
    }
};

constexpr Q32_31 mul_q32(Q32_31 a, Q32_31 b) noexcept {
    __int128 prod = static_cast<__int128>(a) * static_cast<__int128>(b);
    return static_cast<Q32_31>(prod >> 31);
}

constexpr Q32_31 lorentz_inner_product(const LorentzVecQ32& u, const LorentzVecQ32& v) noexcept {
    __int128 t0 = static_cast<__int128>(u.x0) * static_cast<__int128>(v.x0);
    __int128 t1 = static_cast<__int128>(u.x1) * static_cast<__int128>(v.x1);
    __int128 t2 = static_cast<__int128>(u.x2) * static_cast<__int128>(v.x2);
    __int128 t3 = static_cast<__int128>(u.x3) * static_cast<__int128>(v.x3);
    return static_cast<Q32_31>((t0 - t1 - t2 - t3) >> 31);
}

static_assert(lorentz_inner_product(LorentzVecQ32::origin(), LorentzVecQ32::origin()) == ONE_Q32,
              "Origin Lorentz self-inner-product must equal 1.0 in Q32.31");

} // namespace atoma::geometry
