#include "fp16.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

int main() {
    size_t checked = 0;
    // Independently reconstruct every finite binary16 value mathematically.
    for (unsigned raw = 0; raw <= 0xffffu; ++raw) {
        const unsigned exponent = (raw >> 10) & 31;
        const unsigned mantissa = raw & 1023;
        if (exponent == 31) continue;
        float value = exponent == 0
            ? std::ldexp(static_cast<float>(mantissa), -24)
            : std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                         static_cast<int>(exponent) - 15);
        if (raw & 0x8000u) value = -value;
        assert(jade::to_fp16(value) == raw);
        ++checked;
    }
    assert(jade::to_fp16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);
    assert(jade::to_fp16(1.0f + 3 * std::ldexp(1.0f, -11)) == 0x3c02);
    const float midpoint = std::ldexp(1.0f, -25);
    assert(jade::to_fp16(midpoint) == 0);
    assert(jade::to_fp16(std::nextafter(midpoint, 1.0f)) == 1);
    assert(jade::to_fp16(-midpoint) == 0x8000);
    assert(jade::to_fp16(65504.0f) == 0x7bff);
    assert(jade::to_fp16(65520.0f) == 0x7c00);
    assert(jade::to_fp16(std::numeric_limits<float>::infinity()) == 0x7c00);
    assert(jade::to_fp16(-std::numeric_limits<float>::infinity()) == 0xfc00);
    assert(jade::to_fp16(std::numeric_limits<float>::quiet_NaN()) == 0x7e00);
    std::printf("FP16 conversion: %zu finite roundtrips and boundary checks passed. No GPU/SDK exercised.\n", checked);
}
