// Register-tiled FP32 GEMM. Dispatch ceil(n / 32), ceil(m / 32), 1.
// A 64-thread group produces a 32x32 tile. Each invocation owns 4x4 outputs.
// K tiles contain 16 elements; the two shared arrays use 4096 bytes total.
struct Params {
    m: u32,
    n: u32,
    k: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 512>;
var<workgroup> tile_b: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let thread = local.y * 8u + local.x;
    let tile_row = group.y * 32u;
    let tile_col = group.x * 32u;
    let local_row = local.y * 4u;
    let local_col = local.x * 4u;
    var accum: array<vec4<f32>, 4>;

    for (var base = 0u; base < params.k; base += 16u) {
        // Consecutive invocations load consecutive row-major elements.
        for (var load = 0u; load < 8u; load += 1u) {
            let slot = thread + load * 64u;
            let a_row = tile_row + slot / 16u;
            let a_col = base + slot % 16u;
            let b_row = base + slot / 32u;
            let b_col = tile_col + slot % 32u;
            tile_a[slot] = 0.0;
            tile_b[slot] = 0.0;
            if (a_row < params.m && a_col < params.k) {
                tile_a[slot] = a[a_row * params.k + a_col];
            }
            if (b_row < params.k && b_col < params.n) {
                tile_b[slot] = b[b_row * params.n + b_col];
            }
        }
        workgroupBarrier();

        for (var inner = 0u; inner < 16u; inner += 1u) {
            let b_offset = inner * 32u + local_col;
            let b_values = vec4<f32>(
                tile_b[b_offset],
                tile_b[b_offset + 1u],
                tile_b[b_offset + 2u],
                tile_b[b_offset + 3u],
            );
            for (var row = 0u; row < 4u; row += 1u) {
                let a_value = tile_a[(local_row + row) * 16u + inner];
                accum[row] = fma(vec4<f32>(a_value), b_values, accum[row]);
            }
        }
        // All reads of this tile must finish before the next cooperative load.
        workgroupBarrier();
    }

    for (var row = 0u; row < 4u; row += 1u) {
        let output_row = tile_row + local_row + row;
        for (var col = 0u; col < 4u; col += 1u) {
            let output_col = tile_col + local_col + col;
            if (output_row < params.m && output_col < params.n) {
                c[output_row * params.n + output_col] = accum[row][col];
            }
        }
    }
}
