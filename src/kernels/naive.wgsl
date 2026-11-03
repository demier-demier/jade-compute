// Row-major FP32 GEMM baseline. Dispatch ceil(n / 8), ceil(m / 8), 1.
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) {
        return;
    }
    var sum = 0.0;
    for (var inner = 0u; inner < params.k; inner += 1u) {
        sum = fma(a[row * params.k + inner], b[inner * params.n + col], sum);
    }
    c[row * params.n + col] = sum;
}
