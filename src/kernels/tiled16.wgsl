// Shared-memory FP32 GEMM. Dispatch ceil(n / 16), ceil(m / 16), 1.
// Out-of-range inputs become zero. Every invocation reaches both barriers.
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

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = group.y * 16u + local.y;
    let col = group.x * 16u + local.x;
    let slot = local.y * 16u + local.x;
    var sum = 0.0;

    for (var base = 0u; base < params.k; base += 16u) {
        tile_a[slot] = 0.0;
        tile_b[slot] = 0.0;
        if (row < params.m && base + local.x < params.k) {
            tile_a[slot] = a[row * params.k + base + local.x];
        }
        if (base + local.y < params.k && col < params.n) {
            tile_b[slot] = b[(base + local.y) * params.n + col];
        }
        workgroupBarrier();

        for (var inner = 0u; inner < 16u; inner += 1u) {
            sum = fma(tile_a[local.y * 16u + inner], tile_b[inner * 16u + local.x], sum);
        }
        // Prevent a fast invocation overwriting the next tile before peers read it.
        workgroupBarrier();
    }
    if (row < params.m && col < params.n) {
        c[row * params.n + col] = sum;
    }
}
