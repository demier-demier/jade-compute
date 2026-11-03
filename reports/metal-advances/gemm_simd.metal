#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void jade_main(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint linear_thread [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr uint M = 257u, N = 263u, K = 67u;
    constexpr uint TILE = 32u;
    threadgroup float tile_a[32 * 32];
    threadgroup float tile_b[32 * 32];
    const uint base_row = group.y * TILE;
    const uint base_col = group.x * TILE;
    const uint simd_row = (simd_group / 2u) * 16u;
    const uint simd_col = (simd_group % 2u) * 16u;

    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint base_k = 0; base_k < K; base_k += TILE) {
        // Consecutive threads load consecutive elements. Zero padding makes
        // every collective load and multiply legal, including ragged K tails.
        for (uint i = linear_thread; i < TILE * TILE; i += 128u) {
            const uint row = i / TILE;
            const uint col = i % TILE;
            tile_a[i] = (base_row + row < M && base_k + col < K)
                ? A[(base_row + row) * K + base_k + col] : 0.0f;
            tile_b[i] = (base_k + row < K && base_col + col < N)
                ? B[(base_k + row) * N + base_col + col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Control flow and addresses are uniform within each SIMD group.
        // No assumptions are made about which lane owns a matrix element.
        #pragma unroll
        for (uint inner = 0; inner < TILE; inner += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, tile_a + simd_row * TILE + inner, TILE);
            simdgroup_load(a1, tile_a + (simd_row + 8u) * TILE + inner, TILE);
            simdgroup_load(b0, tile_b + inner * TILE + simd_col, TILE);
            simdgroup_load(b1, tile_b + inner * TILE + simd_col + 8u, TILE);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        // All readers finish before staging is overwritten by the next K tile
        // or reused for the boundary output path below. No early lane returns.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base_row + TILE <= M && base_col + TILE <= N) {
        // Interior tiles avoid the extra shared-memory copy and barrier.
        device float* out = C + (base_row + simd_row) * N + base_col + simd_col;
        simdgroup_store(c00, out, N);
        simdgroup_store(c01, out + 8u, N);
        simdgroup_store(c10, out + 8u * N, N);
        simdgroup_store(c11, out + 8u * N + 8u, N);
    } else {
        // A collective store has no lane mask. Store a complete tile in shared
        // memory first, then bounds-check each scalar global store.
        threadgroup float* out = tile_a + simd_row * TILE + simd_col;
        simdgroup_store(c00, out, TILE);
        simdgroup_store(c01, out + 8u, TILE);
        simdgroup_store(c10, out + 8u * TILE, TILE);
        simdgroup_store(c11, out + 8u * TILE + 8u, TILE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = linear_thread; i < TILE * TILE; i += 128u) {
            const uint row = base_row + i / TILE;
            const uint col = base_col + i % TILE;
            if (row < M && col < N) C[row * N + col] = tile_a[i];
        }
    }
}

