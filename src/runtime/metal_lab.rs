//! Reproducible native Metal experiments, including an Apple MPS baseline.
//! FP64 references are computed independently on the CPU, outside timed regions.
use super::{
    ir::KernelSpec,
    metal::{MetalBuffer, MetalRuntime},
    workloads::{compare, values},
    Dispatch, Execution, Runtime,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

fn resident(runtime: &MetalRuntime, data: &[f32]) -> Result<MetalBuffer> {
    let buffer = runtime.alloc(data.len())?;
    runtime.upload(&buffer, data)?;
    Ok(buffer)
}

fn gemm_reference(m: u32, n: u32, k: u32, a: &[f32], b: &[f32]) -> Vec<f64> {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut out = vec![0.0_f64; m * n];
    // Different traversal from the device kernel; f64 accumulation and no MSL code reuse.
    for row in 0..m {
        for p in 0..k {
            let av = a[row * k + p] as f64;
            for col in 0..n {
                out[row * n + col] += av * b[p * n + col] as f64;
            }
        }
    }
    out
}

fn attention_reference(
    heads: u32,
    seq: u32,
    dim: u32,
    causal: bool,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f64> {
    let (heads, seq, dim) = (heads as usize, seq as usize, dim as usize);
    let mut out = vec![0.0; heads * seq * dim];
    // Materialize each row in FP64, then normalize and apply V. This is not
    // the online recurrence used by the optimized GPU kernel.
    for h in 0..heads {
        for row in 0..seq {
            let count = if causal { row + 1 } else { seq };
            let mut scores = vec![0.0_f64; count];
            for (col, score) in scores.iter_mut().enumerate() {
                *score = (0..dim)
                    .map(|d| {
                        q[(h * seq + row) * dim + d] as f64 * k[(h * seq + col) * dim + d] as f64
                    })
                    .sum::<f64>()
                    / (dim as f64).sqrt();
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - max).exp();
            }
            let denom: f64 = scores.iter().sum();
            for d in 0..dim {
                out[(h * seq + row) * dim + d] = scores
                    .iter()
                    .enumerate()
                    .map(|(col, &p)| p / denom * v[(h * seq + col) * dim + d] as f64)
                    .sum();
            }
        }
    }
    out
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    } else {
        values[n / 2]
    }
}
fn summary(samples: &[Execution]) -> Value {
    let host: Vec<_> = samples
        .iter()
        .map(|e| e.host_ms / e.repeats as f64)
        .collect();
    let gpu: Option<Vec<_>> = samples
        .iter()
        .map(|e| e.gpu_ms.map(|t| t / e.repeats as f64))
        .collect();
    json!({
        "median_host_ms": median(host.clone()),
        "min_host_ms": host.iter().copied().fold(f64::INFINITY, f64::min),
        "max_host_ms": host.iter().copied().fold(0.0, f64::max),
        "median_gpu_ms": gpu.map(median),
        "samples": samples,
    })
}
fn gpu_median(samples: &[Execution]) -> Option<f64> {
    samples
        .iter()
        .map(|e| e.gpu_ms.map(|t| t / e.repeats as f64))
        .collect::<Option<Vec<_>>>()
        .map(median)
}
fn checked_rounds(repeats: u32, samples: u32) -> Result<()> {
    ensure!((1..=100).contains(&repeats), "repeats must be 1..=100");
    ensure!((3..=31).contains(&samples), "samples must be 3..=31");
    Ok(())
}

pub fn gemm(
    runtime: &MetalRuntime,
    m: u32,
    n: u32,
    k: u32,
    repeats: u32,
    samples: u32,
) -> Result<Value> {
    checked_rounds(repeats, samples)?;
    let simd_spec = KernelSpec::GemmSimd { m, n, k };
    simd_spec.validate()?;
    ensure!(
        u64::from(m) * u64::from(n) * u64::from(k) <= 2_147_483_648,
        "GEMM experiment limited to 2^31 reference multiply-adds"
    );
    let elements =
        u64::from(m) * u64::from(k) + u64::from(k) * u64::from(n) + 3 * u64::from(m) * u64::from(n);
    ensure!(
        elements <= 32 * 1024 * 1024,
        "GEMM experiment limited to 128 MiB of GPU tensors"
    );
    let a = values(m as usize * k as usize, 0x1234, 1.0);
    let b = values(k as usize * n as usize, 0xabcd, 1.0 / (k as f32).sqrt());
    let expected = gemm_reference(m, n, k, &a, &b);
    let ag = resident(runtime, &a)?;
    let bg = resident(runtime, &b)?;
    let outputs = (0..3)
        .map(|_| runtime.alloc(expected.len()))
        .collect::<Result<Vec<_>>>()?;
    let old = runtime.compile(&KernelSpec::Gemm {
        m,
        n,
        k,
        tiled: true,
    })?;
    let simd = runtime.compile(&simd_spec)?;
    let mps = runtime.compile_mps_gemm(m, n, k)?;
    let names = [
        "jade_register_tiled",
        "jade_simdgroup_fp32",
        "apple_mps_fp32",
    ];
    let execute = |index: usize, count: u32| -> Result<Execution> {
        if index == 2 {
            runtime.execute_mps_gemm(&mps, &ag, &bg, &outputs[index], count)
        } else {
            runtime.execute(
                &[Dispatch {
                    kernel: if index == 0 { &old } else { &simd },
                    buffers: vec![&ag, &bg, &outputs[index]],
                }],
                count,
            )
        }
    };
    let mut initial = Vec::new();
    for (index, output) in outputs.iter().enumerate() {
        execute(index, 3)?;
        initial.push(
            compare(&runtime.download(output)?, &expected, 5e-5, 5e-5)
                .with_context(|| format!("initial {}", names[index]))?,
        );
    }
    let mut timings: [Vec<Execution>; 3] = std::array::from_fn(|_| Vec::new());
    let mut order = Vec::new();
    for round in 0..samples {
        let mut round_order = Vec::new();
        for offset in 0..3 {
            let index = (round as usize + offset) % 3;
            timings[index].push(execute(index, repeats)?);
            round_order.push(names[index]);
        }
        order.push(round_order);
    }
    let mut final_checks = Vec::new();
    for (index, output) in outputs.iter().enumerate() {
        final_checks.push(
            compare(&runtime.download(output)?, &expected, 5e-5, 5e-5)
                .with_context(|| format!("final {}", names[index]))?,
        );
    }
    ensure!(
        runtime.download(&ag)? == a && runtime.download(&bg)? == b,
        "GEMM modified an input"
    );
    let variants: Vec<_> = (0..3)
        .map(|i| {
            json!({
                "name": names[i], "timing":summary(&timings[i]),
                "initial_check":initial[i], "final_check":final_checks[i],
            })
        })
        .collect();
    let med = |i: usize| {
        median(
            timings[i]
                .iter()
                .map(|t| t.host_ms / t.repeats as f64)
                .collect(),
        )
    };
    Ok(json!({
        "runtime": runtime.info(), "experiment":"native_fp32_gemm", "passed":true,
        "shape":{"m":m,"n":n,"k":k}, "variants":variants,
        "ratio_register_to_simd_host":med(0)/med(1),
        "ratio_mps_to_simd_host":med(2)/med(1),
        "ratio_register_to_simd_gpu":gpu_median(&timings[0]).zip(gpu_median(&timings[1])).map(|(a,b)|a/b),
        "ratio_mps_to_simd_gpu":gpu_median(&timings[2]).zip(gpu_median(&timings[1])).map(|(a,b)|a/b),
        "mps_baseline":"Native MPSMatrixMultiplication, packed row strides, Float32, options=None. No AllowReducedPrecision. Packed rows are valid but not necessarily Apple's fastest recommended stride.",
        "host_timing_caveat":"MPS default per-encode validation is inside its host timer; Jade binding validation is before its timer. GPU timestamps are the cleaner compute comparison.",
        "sample_order":order, "warmup_repeats_per_variant":3,
        "validation":"All output elements against independent FP64 CPU reference before and after timing; input immutability checked",
        "tolerance":{"atol":5e-5,"rtol":5e-5},
        "precision":"FP32 inputs, output and accumulation; no FP16 conversion or NVIDIA TF32 mode",
        "scope":"Resident row-major C=A*B. Same device, buffers and precision. Host encode/submit/wait and separate native GPU timestamps. Excludes compile, allocation, upload, readback, oracle and MPS object setup. No NVIDIA or full-model comparison.",
    }))
}

pub fn attention(
    runtime: &MetalRuntime,
    heads: u32,
    seq: u32,
    dim: u32,
    causal: bool,
    repeats: u32,
    samples: u32,
) -> Result<Value> {
    checked_rounds(repeats, samples)?;
    attention_run(
        runtime, heads, seq, dim, causal, repeats, samples, 1.0, false,
    )
}

#[allow(clippy::too_many_arguments)]
fn attention_run(
    runtime: &MetalRuntime,
    heads: u32,
    seq: u32,
    dim: u32,
    causal: bool,
    repeats: u32,
    samples: u32,
    amplitude: f32,
    zeros: bool,
) -> Result<Value> {
    let online_spec = KernelSpec::AttentionOnline {
        heads,
        seq,
        dim,
        causal,
    };
    online_spec.validate()?;
    let count = heads as usize * seq as usize * dim as usize;
    let score_count = heads as usize * seq as usize * seq as usize;
    ensure!(
        5 * count as u64 + 2 * score_count as u64 <= 32 * 1024 * 1024,
        "attention experiment limited to 128 MiB of GPU tensors"
    );
    ensure!(
        u64::from(heads) * u64::from(seq) * u64::from(seq) * u64::from(dim) <= 536_870_912,
        "attention experiment limited to 2^29 reference dot-product terms"
    );
    let q = if zeros {
        vec![0.0; count]
    } else {
        values(count, 0x1111, amplitude)
    };
    let k = if zeros {
        vec![0.0; count]
    } else {
        values(count, 0x2222, amplitude)
    };
    let v = values(count, 0x3333, 1.0);
    let expected = attention_reference(heads, seq, dim, causal, &q, &k, &v);
    let qg = resident(runtime, &q)?;
    let kg = resident(runtime, &k)?;
    let vg = resident(runtime, &v)?;
    let scores = runtime.alloc(score_count)?;
    let probabilities = runtime.alloc(score_count)?;
    let out_materialized = runtime.alloc(count)?;
    let out_online = runtime.alloc(count)?;
    let score_kernel = runtime.compile(&KernelSpec::AttentionScores {
        heads,
        seq,
        dim,
        causal,
    })?;
    let softmax = runtime.compile(&KernelSpec::Softmax {
        rows: heads * seq,
        cols: seq,
    })?;
    let apply = runtime.compile(&KernelSpec::AttentionApply { heads, seq, dim })?;
    let online = runtime.compile(&online_spec)?;
    let jobs = [
        Dispatch {
            kernel: &score_kernel,
            buffers: vec![&qg, &kg, &scores],
        },
        Dispatch {
            kernel: &softmax,
            buffers: vec![&scores, &probabilities],
        },
        Dispatch {
            kernel: &apply,
            buffers: vec![&probabilities, &vg, &out_materialized],
        },
    ];
    let online_jobs = [Dispatch {
        kernel: &online,
        buffers: vec![&qg, &kg, &vg, &out_online],
    }];
    let outputs = [&out_materialized, &out_online];
    let names = ["materialized_reference", "online_simd"];
    let execute = |i: usize, count: u32| {
        runtime.execute(if i == 0 { &jobs[..] } else { &online_jobs[..] }, count)
    };
    let mut initial = Vec::new();
    // Large logits test numerical stability; a slightly larger absolute tolerance
    // accounts for FP32 dot-product differences amplified by peaked softmax.
    let tolerance = if amplitude > 1.0 { 5e-4 } else { 5e-5 };
    for (i, output) in outputs.iter().enumerate() {
        execute(i, 3)?;
        initial.push(
            compare(&runtime.download(output)?, &expected, tolerance, tolerance).with_context(
                || {
                    format!(
                        "initial attention {} ({heads},{seq},{dim},{causal})",
                        names[i]
                    )
                },
            )?,
        );
    }
    let mut timings: [Vec<Execution>; 2] = std::array::from_fn(|_| Vec::new());
    let mut order = Vec::new();
    for round in 0..samples {
        let indices = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        for i in indices {
            timings[i].push(execute(i, repeats)?);
        }
        order.push(indices.map(|i| names[i]));
    }
    let mut final_checks = Vec::new();
    for (i, output) in outputs.iter().enumerate() {
        final_checks.push(
            compare(&runtime.download(output)?, &expected, tolerance, tolerance)
                .with_context(|| format!("final attention {}", names[i]))?,
        );
    }
    ensure!(
        runtime.download(&qg)? == q && runtime.download(&kg)? == k && runtime.download(&vg)? == v,
        "attention modified an input"
    );
    let variants: Vec<_> = (0..2)
        .map(|i| {
            json!({
                "name":names[i], "timing":summary(&timings[i]),
                "initial_check":initial[i], "final_check":final_checks[i],
            })
        })
        .collect();
    let med = |i: usize| {
        median(
            timings[i]
                .iter()
                .map(|e| e.host_ms / e.repeats as f64)
                .collect(),
        )
    };
    Ok(json!({
        "runtime":runtime.info(), "experiment":"native_fp32_self_attention", "passed":true,
        "shape":{"heads":heads,"sequence":seq,"head_dim":dim,"causal":causal},
        "input_amplitude":amplitude,"zero_qk":zeros,"variants":variants,
        "ratio_materialized_to_online_host":med(0)/med(1),
        "ratio_materialized_to_online_gpu":gpu_median(&timings[0]).zip(gpu_median(&timings[1])).map(|(a,b)|a/b),
        "sample_order":order,"warmup_repeats_per_variant":3,
        "logical_global_scratch_bytes":{"materialized_reference":2*score_count*4,"online_simd":0},
        "scratch_scope":"Algorithm-level score/probability buffers only, not measured process/device peak. Excludes common Q/K/V/O, driver allocations and registers. Benchmark keeps both paths allocated.",
        "validation":"All output values checked against independent materialized FP64 reference before and after timing; input immutability checked",
        "tolerance":{"atol":tolerance,"rtol":tolerance},
        "scope":"Resident FP32 self-attention forward only, scale=1/sqrt(head_dim), optional causal mask. Online softmax recurrence; no S*S score buffer. Baseline is this project's scalar materialized QK/softmax/PV, not optimized FlashAttention, MPS attention, MLX or a complete model. Both paths retain quadratic arithmetic. Excludes compile, allocation, transfer and CPU oracle.",
    }))
}

pub fn verify(runtime: &MetalRuntime) -> Result<Value> {
    let mut gemm_checks = Vec::new();
    for (m, n, k) in [
        (1, 1, 1),
        (7, 9, 13),
        (8, 8, 8),
        (31, 47, 19),
        (32, 32, 33),
        (33, 31, 65),
        (65, 67, 63),
        (1, 129, 33),
        (64, 64, 64),
        (127, 65, 129),
        (256, 256, 65),
        (257, 263, 67),
    ] {
        let report = gemm(runtime, m, n, k, 1, 3)?;
        eprintln!("PASS Metal FP32 GEMM ({m},{n},{k}) including Apple MPS");
        gemm_checks.push(report);
    }
    let mut attention_checks = Vec::new();
    for (h, s, d, causal, amplitude, zeros) in [
        (1, 1, 1, false, 1.0, false),
        (1, 1, 128, true, 1.0, false),
        (2, 7, 17, false, 1.0, false),
        (2, 7, 17, true, 1.0, false),
        (2, 33, 64, false, 1.0, false),
        (2, 33, 64, true, 1.0, false),
        (1, 65, 128, true, 1.0, false),
        (3, 17, 33, false, 1.0, true),
        (2, 17, 31, true, 1.0, true),
        (2, 17, 64, false, 20.0, false),
        (2, 17, 64, true, 20.0, false),
    ] {
        let report = attention_run(runtime, h, s, d, causal, 1, 3, amplitude, zeros)?;
        eprintln!("PASS Metal attention ({h},{s},{d}), causal={causal}, amplitude={amplitude}, zero_qk={zeros}");
        attention_checks.push(report);
    }
    let mut precision_checks = Vec::new();
    for (a, b) in [(1.0001_f32, 1.0003_f32), (70000.25_f32, 1.0000312_f32)] {
        let ag = resident(runtime, &[a])?;
        let bg = resident(runtime, &[b])?;
        let output = runtime.alloc(1)?;
        for spec in [
            KernelSpec::GemmSimd { m: 1, n: 1, k: 1 },
            KernelSpec::GemmMps { m: 1, n: 1, k: 1 },
        ] {
            let kernel = runtime.compile(&spec)?;
            runtime.execute(
                &[Dispatch {
                    kernel: &kernel,
                    buffers: vec![&ag, &bg, &output],
                }],
                1,
            )?;
            let check = compare(
                &runtime.download(&output)?,
                &[a as f64 * b as f64],
                1e-5,
                1e-6,
            )?;
            precision_checks.push(json!({"spec":spec,"a":a,"b":b,"check":check}));
        }
    }
    Ok(
        json!({"runtime":runtime.info(),"passed":true,"gemm_shapes":gemm_checks,"attention_shapes":attention_checks,"fp32_precision_probes":precision_checks,
        "scope":"Native Metal and native Apple MPS execution, every output compared against independent FP64 references. Verification timings are diagnostic, not performance results."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_attention_oracle_handles_uniform_and_causal_rows() {
        let q = vec![0.0; 6];
        let k = q.clone();
        let v = vec![1.0, 2.0, 3.0, 4.0, 8.0, 9.0];
        assert_eq!(
            attention_reference(1, 3, 2, false, &q, &k, &v),
            vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]
        );
        assert_eq!(
            attention_reference(1, 3, 2, true, &q, &k, &v),
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0]
        );
    }
}
