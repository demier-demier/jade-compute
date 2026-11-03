//! FP32 GEMM built from Metal's public cooperative SIMD-group matrix operations.
//!
//! A 128-thread group computes a 32x32 output tile. Four SIMD groups each retain
//! four 8x8 FP32 accumulators, reusing two A and two B fragments per K step.
//! This is an original tiled kernel using documented API primitives, not a
//! translation of CUDA/WMMA code or a claim about Apple's physical matrix units.
//!
//! The caller must validate dimensions/index products, require Apple7 or newer
//! SIMD-group matrix support and a pipeline execution width of 32, dispatch full
//! 128-thread groups, disallow output/input aliasing, and disable fast math.
//! All storage, fragments, and accumulators remain float; no half conversion is
//! generated. Reduction order differs from scalar FP32 and FP64 references.
//!
//! References (Apple):
//! - MSL specification, "SIMD-group Matrix Data Types" / "SIMD-group Matrices":
//!   <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
//! - Feature availability (SIMD-scoped matrix multiply starts at Apple7):
//!   <https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf>
//! - Public matrix primitive demonstration at 24:56:
//!   <https://developer.apple.com/videos/play/tech-talks/10858/>

/// Return threadgroup counts and threads per threadgroup, respectively.
/// Dimension validation is the operation IR's responsibility.
pub fn launch(m: u32, n: u32, _k: u32) -> ([u32; 3], [u32; 3]) {
    ([n.div_ceil(32), m.div_ceil(32), 1], [128, 1, 1])
}

/// Emit row-major C[M,N] = A[M,K] * B[K,N], with three FP32 buffer bindings.
pub fn source(m: u32, n: u32, k: u32) -> String {
    format!(
        r#"#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void jade_main(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint linear_thread [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{{
    constexpr uint M = {m}u, N = {n}u, K = {k}u;
    constexpr uint TILE = 32u;
    threadgroup float tile_a[32 * 32];
    threadgroup float tile_b[32 * 32];
    const uint base_row = group.y * TILE;
    const uint base_col = group.x * TILE;
    const uint simd_row = (simd_group / 2u) * 16u;
    const uint simd_col = (simd_group % 2u) * 16u;

    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint base_k = 0; base_k < K; base_k += TILE) {{
        // Consecutive threads load consecutive elements. Zero padding makes
        // every collective load and multiply legal, including ragged K tails.
        for (uint i = linear_thread; i < TILE * TILE; i += 128u) {{
            const uint row = i / TILE;
            const uint col = i % TILE;
            tile_a[i] = (base_row + row < M && base_k + col < K)
                ? A[(base_row + row) * K + base_k + col] : 0.0f;
            tile_b[i] = (base_k + row < K && base_col + col < N)
                ? B[(base_k + row) * N + base_col + col] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Control flow and addresses are uniform within each SIMD group.
        // No assumptions are made about which lane owns a matrix element.
        #pragma unroll
        for (uint inner = 0; inner < TILE; inner += 8u) {{
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, tile_a + simd_row * TILE + inner, TILE);
            simdgroup_load(a1, tile_a + (simd_row + 8u) * TILE + inner, TILE);
            simdgroup_load(b0, tile_b + inner * TILE + simd_col, TILE);
            simdgroup_load(b1, tile_b + inner * TILE + simd_col + 8u, TILE);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }}
        // All readers finish before staging is overwritten by the next K tile
        // or reused for the boundary output path below. No early lane returns.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (base_row + TILE <= M && base_col + TILE <= N) {{
        // Interior tiles avoid the extra shared-memory copy and barrier.
        device float* out = C + (base_row + simd_row) * N + base_col + simd_col;
        simdgroup_store(c00, out, N);
        simdgroup_store(c01, out + 8u, N);
        simdgroup_store(c10, out + 8u * N, N);
        simdgroup_store(c11, out + 8u * N + 8u, N);
    }} else {{
        // A collective store has no lane mask. Store a complete tile in shared
        // memory first, then bounds-check each scalar global store.
        threadgroup float* out = tile_a + simd_row * TILE + simd_col;
        simdgroup_store(c00, out, TILE);
        simdgroup_store(c01, out + 8u, TILE);
        simdgroup_store(c10, out + 8u * TILE, TILE);
        simdgroup_store(c11, out + 8u * TILE + 8u, TILE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = linear_thread; i < TILE * TILE; i += 128u) {{
            const uint row = base_row + i / TILE;
            const uint col = base_col + i % TILE;
            if (row < M && col < N) C[row * N + col] = tile_a[i];
        }}
    }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::launch;

    #[test]
    fn launch_covers_both_ragged_axes_without_overflow() {
        assert_eq!(launch(1, 1, 1), ([1, 1, 1], [128, 1, 1]));
        assert_eq!(launch(33, 65, 17), ([3, 2, 1], [128, 1, 1]));
        assert_eq!(
            launch(u32::MAX, u32::MAX, 1),
            ([134_217_728, 134_217_728, 1], [128, 1, 1])
        );
    }
}
