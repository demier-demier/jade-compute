#ifndef JADE_FP16_HPP
#define JADE_FP16_HPP

#include <cstdint>
#include <cstring>
#include <limits>

namespace jade {
// Host conversion: IEEE binary32 -> binary16, round to nearest, ties to even.
// NaNs are canonicalized, signed zeros and infinities are preserved.
inline uint16_t to_fp16(float value) noexcept {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "IEEE binary32 is required");
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 255) {
        return static_cast<uint16_t>(sign | (fraction ? 0x7e00u : 0x7c00u));
    }
    const int e = static_cast<int>(exponent) - 127;
    if (e > 15) return static_cast<uint16_t>(sign | 0x7c00u);
    if (e < -25) return sign;
    uint32_t mantissa;
    uint32_t shift;
    uint32_t base;
    if (e < -14) {
        mantissa = fraction | 0x800000u;
        shift = static_cast<uint32_t>(-e - 1);
        base = 0;
    } else {
        mantissa = fraction;
        shift = 13;
        base = static_cast<uint32_t>(e + 15) << 10;
    }
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t midpoint = 1u << (shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1u))) {
        ++rounded;
    }
    return static_cast<uint16_t>(sign | (base + rounded));
}
} // namespace jade
#endif
