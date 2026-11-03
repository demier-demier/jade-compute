//! Original FP32 self-attention source generators for the native Metal backend.
//!
//! Q/K/V/O use contiguous [heads, sequence, head_dim] layout. Attention is
//! softmax(Q K^T / sqrt(head_dim)) V; a causal query includes its own key.
//! The online path retains a query and unnormalized output in lane registers
//! instead of materializing an [heads, sequence, sequence] score/probability
//! tensor. One full 32-lane SIMD group computes each query independently.
//!
//! The backend MUST check Apple-family SIMD-reduction support and the compiled
//! pipeline's threadExecutionWidth == ONLINE_SIMD_WIDTH before dispatch. These
//! are Metal-only kernels; a matching API name on another vendor is insufficient.
//! Callers must validate shapes before calling these infallible source helpers;
//! the assertions also protect standalone use against invalid dimensions.
//!
//! Design references (consulted 2026-11-03; no vendor kernel source copied):
//! - Dao et al., FlashAttention (2022), section 3.1 / Algorithm 1: stable online
//!   max/normalizer rescaling: <https://arxiv.org/abs/2205.14135>.
//! - Apple's documented SIMD reductions:
//!   <https://developer.apple.com/videos/play/tech-talks/10858/>.
//! - Metal's pipeline-dependent SIMD width and lane attribute:
//!   <https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups>.
//!
//! This is a small scalar-SIMD online forward kernel, not the tiled matrix
//! implementation or performance characteristics of FlashAttention. It has no
//! backward pass, dropout, padding mask, tensor/matrix instructions, or K/V reuse
//! across queries. Inputs and FP32 intermediate scores/accumulators must remain
//! finite. Compile with Metal fast math disabled, as the native runtime does.

/// Required pipeline SIMD width for the online kernel, checked by the backend.
pub const ONLINE_SIMD_WIDTH: u32 = 32;

fn check_shape(heads: u32, seq: u32, dim: u32) {
    assert!((1..=32).contains(&heads), "attention heads must be 1..=32");
    assert!(
        (1..=2048).contains(&seq),
        "attention sequence must be 1..=2048"
    );
    assert!(
        (1..=128).contains(&dim),
        "attention head dimension must be 1..=128"
    );
}

/// One threadgroup/SIMD group per (query, head). No shared-memory barriers are
/// required: every lane reaches every simd_sum and owns its output components.
pub fn online_launch(heads: u32, seq: u32, dim: u32, _causal: bool) -> ([u32; 3], [u32; 3]) {
    check_shape(heads, seq, dim);
    ([seq, heads, 1], [ONLINE_SIMD_WIDTH, 1, 1])
}

/// A simple scalar dot-product baseline: each thread computes one QK score.
pub fn scores_launch(heads: u32, seq: u32, dim: u32, _causal: bool) -> ([u32; 3], [u32; 3]) {
    check_shape(heads, seq, dim);
    ([seq.div_ceil(8), seq.div_ceil(8), heads], [8, 8, 1])
}

/// Each thread computes one output element from the materialized probabilities.
pub fn apply_launch(heads: u32, seq: u32, dim: u32) -> ([u32; 3], [u32; 3]) {
    check_shape(heads, seq, dim);
    ([dim.div_ceil(8), seq.div_ceil(8), heads], [8, 8, 1])
}

/// Buffers: Q=0, K=1, V=2, O=3, all contiguous FP32 [heads, seq, dim].
pub fn online_source(heads: u32, seq: u32, dim: u32, causal: bool) -> String {
    check_shape(heads, seq, dim);
    let fragments = dim.div_ceil(ONLINE_SIMD_WIDTH);
    let key_limit = if causal {
        "query + 1u".to_owned()
    } else {
        format!("{seq}u")
    };
    format!(
        r#"// Jade original SIMD online attention: FP32, one query per SIMD group.
// Requires a 32-lane pipeline and a [32,1,1] threadgroup; checked by the host.
#include <metal_stdlib>
using namespace metal;
kernel void jade_main(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {{
    const uint query = group.x;
    const uint head = group.y;
    // This condition is uniform across the complete SIMD group.
    if (query >= {seq}u || head >= {heads}u) return;
    const uint head_base = head * {seq}u * {dim}u;
    const uint query_base = head_base + query * {dim}u;
    const float scale = 1.0f / sqrt(float({dim}u));
    float query_values[{fragments}];
    float weighted_values[{fragments}];
    #pragma unroll
    for (uint part = 0u; part < {fragments}u; ++part) {{
        const uint channel = lane + part * 32u;
        query_values[part] = channel < {dim}u ? q[query_base + channel] : 0.0f;
        weighted_values[part] = 0.0f;
    }}
    float row_max = -INFINITY;
    float normalizer = 0.0f;
    const uint key_limit = {key_limit};
    for (uint key = 0u; key < key_limit; ++key) {{
        const uint key_base = head_base + key * {dim}u;
        float partial_score = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < {fragments}u; ++part) {{
            const uint channel = lane + part * 32u;
            if (channel < {dim}u) {{
                partial_score = fma(query_values[part], k[key_base + channel], partial_score);
            }}
        }}
        // Tail lanes contribute zero but still participate in this collective.
        const float score = simd_sum(partial_score) * scale;
        const float next_max = fmax(row_max, score);
        const float old_weight = exp(row_max - next_max);
        const float new_weight = exp(score - next_max);
        normalizer = fma(old_weight, normalizer, new_weight);
        #pragma unroll
        for (uint part = 0u; part < {fragments}u; ++part) {{
            const uint channel = lane + part * 32u;
            if (channel < {dim}u) {{
                weighted_values[part] = fma(new_weight, v[key_base + channel],
                                            old_weight * weighted_values[part]);
            }}
        }}
        row_max = next_max;
    }}
    // Every row has at least one valid key, including query zero in causal mode.
    #pragma unroll
    for (uint part = 0u; part < {fragments}u; ++part) {{
        const uint channel = lane + part * 32u;
        if (channel < {dim}u) out[query_base + channel] = weighted_values[part] / normalizer;
    }}
}}
"#
    )
}

/// Buffers: Q=0, K=1 ([heads, seq, dim]), scores=2 ([heads, seq, seq]).
/// Causal upper-triangle entries are -infinity; each row still has a finite key
/// for finite input scores, so the existing row-wise Softmax can consume them.
pub fn scores_source(heads: u32, seq: u32, dim: u32, causal: bool) -> String {
    check_shape(heads, seq, dim);
    let mask = if causal {
        "    if (key > query) { scores[score_index] = -INFINITY; return; }\n"
    } else {
        ""
    };
    format!(
        r#"// Jade materialized attention baseline: scaled QK scores.
#include <metal_stdlib>
using namespace metal;
kernel void jade_main(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* scores [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {{
    const uint key = gid.x, query = gid.y, head = gid.z;
    if (key >= {seq}u || query >= {seq}u || head >= {heads}u) return;
    const uint score_index = (head * {seq}u + query) * {seq}u + key;
{mask}    const uint q_base = (head * {seq}u + query) * {dim}u;
    const uint k_base = (head * {seq}u + key) * {dim}u;
    float dot = 0.0f;
    for (uint channel = 0u; channel < {dim}u; ++channel) {{
        dot = fma(q[q_base + channel], k[k_base + channel], dot);
    }}
    scores[score_index] = dot / sqrt(float({dim}u));
}}
"#
    )
}

/// Buffers: P=0 ([heads, seq, seq]), V=1 and O=2 ([heads, seq, dim]).
pub fn apply_source(heads: u32, seq: u32, dim: u32) -> String {
    check_shape(heads, seq, dim);
    format!(
        r#"// Jade materialized attention baseline: probability/value product.
#include <metal_stdlib>
using namespace metal;
kernel void jade_main(
    device const float* probabilities [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {{
    const uint channel = gid.x, query = gid.y, head = gid.z;
    if (channel >= {dim}u || query >= {seq}u || head >= {heads}u) return;
    const uint row_base = (head * {seq}u + query) * {seq}u;
    const uint v_base = head * {seq}u * {dim}u;
    float weighted_sum = 0.0f;
    for (uint key = 0u; key < {seq}u; ++key) {{
        weighted_sum = fma(probabilities[row_base + key],
                           v[v_base + key * {dim}u + channel], weighted_sum);
    }}
    out[(head * {seq}u + query) * {dim}u + channel] = weighted_sum;
}}
"#
    )
}
