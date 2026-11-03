// Jade original SIMD online attention: FP32, one query per SIMD group.
// Requires a 32-lane pipeline and a [32,1,1] threadgroup; checked by the host.
#include <metal_stdlib>
using namespace metal;
kernel void jade_main(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint query = group.x;
    const uint head = group.y;
    // This condition is uniform across the complete SIMD group.
    if (query >= 256u || head >= 4u) return;
    const uint head_base = head * 256u * 64u;
    const uint query_base = head_base + query * 64u;
    const float scale = 1.0f / sqrt(float(64u));
    float query_values[2];
    float weighted_values[2];
    #pragma unroll
    for (uint part = 0u; part < 2u; ++part) {
        const uint channel = lane + part * 32u;
        query_values[part] = channel < 64u ? q[query_base + channel] : 0.0f;
        weighted_values[part] = 0.0f;
    }
    float row_max = -INFINITY;
    float normalizer = 0.0f;
    const uint key_limit = query + 1u;
    for (uint key = 0u; key < key_limit; ++key) {
        const uint key_base = head_base + key * 64u;
        float partial_score = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < 2u; ++part) {
            const uint channel = lane + part * 32u;
            if (channel < 64u) {
                partial_score = fma(query_values[part], k[key_base + channel], partial_score);
            }
        }
        // Tail lanes contribute zero but still participate in this collective.
        const float score = simd_sum(partial_score) * scale;
        const float next_max = fmax(row_max, score);
        const float old_weight = exp(row_max - next_max);
        const float new_weight = exp(score - next_max);
        normalizer = fma(old_weight, normalizer, new_weight);
        #pragma unroll
        for (uint part = 0u; part < 2u; ++part) {
            const uint channel = lane + part * 32u;
            if (channel < 64u) {
                weighted_values[part] = fma(new_weight, v[key_base + channel],
                                            old_weight * weighted_values[part]);
            }
        }
        row_max = next_max;
    }
    // Every row has at least one valid key, including query zero in causal mode.
    #pragma unroll
    for (uint part = 0u; part < 2u; ++part) {
        const uint channel = lane + part * 32u;
        if (channel < 64u) out[query_base + channel] = weighted_values[part] / normalizer;
    }
}

