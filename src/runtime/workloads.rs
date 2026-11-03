//! Independently computed FP64 references and a synthetic resident MLP workload.
use super::{
    ir::{Expr, KernelSpec},
    Buffer, Dispatch, Execution, Runtime,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};

pub struct Case {
    pub name: String,
    pub spec: KernelSpec,
    pub inputs: Vec<Vec<f32>>,
    pub expected: Vec<f64>,
}

pub fn values(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / 16777216.0 * 2.0 - 1.0) * scale
        })
        .collect()
}
fn dot(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    (0..m * n)
        .map(|i| (0..k).map(|p| a[i / n * k + p] * b[p * n + i % n]).sum())
        .collect()
}
fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
fn normalize(rows: usize, cols: usize, x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    let mut output = vec![0.0; x.len()];
    for row in 0..rows {
        let offset = row * cols;
        let square: f64 = x[offset..offset + cols].iter().map(|v| v * v).sum();
        let inv = 1.0 / (square / cols as f64 + eps).sqrt();
        for col in 0..cols {
            output[offset + col] = x[offset + col] * inv * weight[col];
        }
    }
    output
}
pub fn cases() -> Vec<Case> {
    let mut all = Vec::new();
    for len in [1, 257, 1025] {
        let a = values(len, 0x1234, 2.0);
        let b = values(len, 0xabcd, 1.0);
        let expected = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| -0.375 * x as f64 + y as f64)
            .collect();
        all.push(Case {
            name: format!("axpy_{len}"),
            spec: KernelSpec::Axpy {
                len: len as u32,
                alpha: -0.375,
            },
            inputs: vec![a, b],
            expected,
        });
    }
    {
        let a = values(257, 0x1234, 2.0);
        let b = values(257, 0xabcd, 1.0);
        let expected = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| x as f64 + y as f64)
            .collect();
        all.push(Case {
            name: "add_257".into(),
            spec: KernelSpec::Add { len: 257 },
            inputs: vec![a, b],
            expected,
        });
    }
    {
        let gate = vec![
            -1000.0, -100.0, -20.0, -1.0, 0.0, 0.25, 1.0, 20.0, 100.0, 1000.0,
        ];
        let up = values(gate.len(), 0xaacc, 2.0);
        let expected = gate
            .iter()
            .zip(&up)
            .map(|(&x, &y)| x as f64 * sigmoid(x as f64) * y as f64)
            .collect();
        all.push(Case {
            name: "swiglu_extreme".into(),
            spec: KernelSpec::SwiGlu {
                len: gate.len() as u32,
            },
            inputs: vec![gate, up],
            expected,
        });
    }
    for (rows, cols) in [(1, 1), (3, 17), (2, 257)] {
        let x = values(rows * cols, 0x1234, 2.0);
        let weight = values(cols, 0xabcd, 1.0);
        let eps = 1e-5_f32;
        let expected = normalize(rows, cols, &widen(&x), &widen(&weight), eps as f64);
        all.push(Case {
            name: format!("rmsnorm_{rows}_{cols}"),
            spec: KernelSpec::RmsNorm {
                rows: rows as u32,
                cols: cols as u32,
                epsilon: eps,
            },
            inputs: vec![x, weight],
            expected,
        });
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| {
                if i / cols == 0 {
                    1000.0
                } else {
                    (i % cols) as f32 * 0.25 - 10000.0
                }
            })
            .collect();
        let mut expected = vec![0.0; x.len()];
        for row in 0..rows {
            let slice = &x[row * cols..(row + 1) * cols];
            let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = slice.iter().map(|&x| (x as f64 - max).exp()).sum();
            for col in 0..cols {
                expected[row * cols + col] = (slice[col] as f64 - max).exp() / sum;
            }
        }
        all.push(Case {
            name: format!("softmax_{rows}_{cols}"),
            spec: KernelSpec::Softmax {
                rows: rows as u32,
                cols: cols as u32,
            },
            inputs: vec![x],
            expected,
        });
    }
    for (m, n, k) in [(1, 1, 1), (31, 47, 19), (8, 33, 65)] {
        for tiled in [false, true] {
            let a = values(m * k, 0x1234, 1.0);
            let b = values(k * n, 0xabcd, 1.0);
            let expected = dot(m, n, k, &widen(&a), &widen(&b));
            all.push(Case {
                name: format!("gemm_{m}_{n}_{k}_tiled_{tiled}"),
                spec: KernelSpec::Gemm {
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    tiled,
                },
                inputs: vec![a, b],
                expected,
            });
        }
    }
    let expression = Expr::Maximum {
        left: Box::new(Expr::Add {
            left: Box::new(Expr::Multiply {
                left: Box::new(Expr::Input { binding: 0 }),
                right: Box::new(Expr::Constant { value: 2.5 }),
            }),
            right: Box::new(Expr::Input { binding: 1 }),
        }),
        right: Box::new(Expr::Constant { value: 0.0 }),
    };
    let a = values(257, 0x1234, 1.0);
    let b = values(257, 0xabcd, 1.0);
    let expected = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (x as f64 * 2.5 + y as f64).max(0.0))
        .collect();
    all.push(Case {
        name: "custom_fused_axpy_relu".into(),
        spec: KernelSpec::Elementwise {
            len: 257,
            inputs: 2,
            expression,
        },
        inputs: vec![a, b],
        expected,
    });
    let input = values(73, 0x9988, 1.0);
    let expected = input.iter().map(|&x| (-(x as f64)).exp()).collect();
    all.push(Case {
        name: "custom_negate_exp".into(),
        spec: KernelSpec::Elementwise {
            len: 73,
            inputs: 1,
            expression: Expr::Exp {
                value: Box::new(Expr::Negate {
                    value: Box::new(Expr::Input { binding: 0 }),
                }),
            },
        },
        inputs: vec![input],
        expected,
    });
    all
}

#[derive(Debug, Serialize)]
pub struct Check {
    pub elements: usize,
    pub max_abs_error: f64,
    pub max_relative_error: f64,
    pub all_finite: bool,
}
pub fn compare(actual: &[f32], expected: &[f64], atol: f64, rtol: f64) -> Result<Check> {
    ensure!(actual.len() == expected.len(), "oracle length mismatch");
    let mut result = Check {
        elements: actual.len(),
        max_abs_error: 0.0,
        max_relative_error: 0.0,
        all_finite: true,
    };
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        ensure!(a.is_finite() && e.is_finite(), "nonfinite value at {i}");
        let error = (a as f64 - e).abs();
        ensure!(
            error <= atol + rtol * e.abs(),
            "mismatch at {i}: got {a}, expected {e}, error {error}"
        );
        result.max_abs_error = result.max_abs_error.max(error);
        result.max_relative_error = result.max_relative_error.max(error / (e.abs().max(atol)));
    }
    Ok(result)
}

pub fn verify<R: Runtime>(runtime: &R) -> Result<Value> {
    let mut checks = Vec::new();
    for case in cases() {
        let kernel = runtime
            .compile(&case.spec)
            .with_context(|| format!("compile {}", case.name))?;
        let mut buffers = Vec::new();
        for input in &case.inputs {
            let buffer = runtime.alloc(input.len())?;
            runtime.upload(&buffer, input)?;
            buffers.push(buffer);
        }
        let output = runtime.alloc(case.expected.len())?;
        let mut refs: Vec<_> = buffers.iter().collect();
        refs.push(&output);
        let job = Dispatch {
            kernel: &kernel,
            buffers: refs,
        };
        let timing = runtime.execute(&[job], 2)?;
        let check = compare(&runtime.download(&output)?, &case.expected, 3e-5, 3e-5)
            .with_context(|| case.name.clone())?;
        eprintln!("PASS native {}: {} elements", case.name, check.elements);
        checks.push(json!({"case":case.name,"spec":case.spec,"check":check,"execution":timing}));
    }
    // Validation failures must be rejected before enqueueing any GPU work.
    let spec = KernelSpec::Add { len: 17 };
    let kernel = runtime.compile(&spec)?;
    let a = runtime.alloc(17)?;
    let b = runtime.alloc(17)?;
    let wrong = runtime.alloc(16)?;
    ensure!(
        runtime.upload(&a, &[0.0; 16]).is_err(),
        "short upload accepted"
    );
    ensure!(
        runtime
            .execute(
                &[Dispatch {
                    kernel: &kernel,
                    buffers: vec![&a, &b, &a]
                }],
                1
            )
            .is_err(),
        "aliased output accepted"
    );
    ensure!(
        runtime
            .execute(
                &[Dispatch {
                    kernel: &kernel,
                    buffers: vec![&a, &b, &wrong]
                }],
                1
            )
            .is_err(),
        "wrong output length accepted"
    );
    ensure!(runtime.execute(&[], 1).is_err(), "empty job list accepted");
    ensure!(
        runtime
            .execute(
                &[Dispatch {
                    kernel: &kernel,
                    buffers: vec![&a, &b, &wrong]
                }],
                0
            )
            .is_err(),
        "zero repeats accepted"
    );
    Ok(
        json!({"runtime":runtime.info(),"passed":true,"checks":checks,"rejected_invalid_calls":5,"scope":"Native GPU execution against independent FP64 references; not CUDA ABI or full-model conformance"}),
    )
}

fn mlp_elements(batch: u32, width: u32, hidden: u32) -> Result<u64> {
    let x = (u64::from(batch) * u64::from(width)).checked_mul(4);
    let weights = (u64::from(width) * u64::from(hidden)).checked_mul(3);
    let activation = (u64::from(batch) * u64::from(hidden)).checked_mul(3);
    x.zip(weights)
        .zip(activation)
        .and_then(|((a, b), c)| {
            a.checked_add(b)?
                .checked_add(c)?
                .checked_add(u64::from(width))
        })
        .context("MLP tensor size calculation overflow")
}

/// A synthetic six-operation LLM-style residual MLP block. Weights are fixed
/// generated numbers, not a trained checkpoint; there is no token generation.
pub fn mlp<R: Runtime>(
    runtime: &R,
    batch: u32,
    width: u32,
    hidden: u32,
    repeats: u32,
    samples: u32,
) -> Result<Value> {
    ensure!(
        batch > 0
            && width > 0
            && hidden > 0
            && repeats > 0
            && repeats <= 1000
            && (3..=31).contains(&samples),
        "invalid MLP dimensions/repeats/samples"
    );
    let total = mlp_elements(batch, width, hidden)?;
    ensure!(
        total <= 32 * 1024 * 1024,
        "MLP experiment limited to 128 MiB of tensors"
    );
    let (m, w, h) = (batch as usize, width as usize, hidden as usize);
    let x = values(m * w, 0x1234, 1.0);
    let norm = vec![1.0; w];
    let wg = values(w * h, 0x3333, 1.0 / (w as f32).sqrt());
    let wu = values(w * h, 0x4444, 1.0 / (w as f32).sqrt());
    let wd = values(h * w, 0x5555, 1.0 / (h as f32).sqrt());
    let resident = |v: &[f32]| -> Result<R::Buffer> {
        let b = runtime.alloc(v.len())?;
        runtime.upload(&b, v)?;
        Ok(b)
    };
    let x_gpu = resident(&x)?;
    let norm_gpu = resident(&norm)?;
    let wg_gpu = resident(&wg)?;
    let wu_gpu = resident(&wu)?;
    let wd_gpu = resident(&wd)?;
    let normalized = runtime.alloc(m * w)?;
    let gate = runtime.alloc(m * h)?;
    let up = runtime.alloc(m * h)?;
    let activated = runtime.alloc(m * h)?;
    let projected = runtime.alloc(m * w)?;
    let output = runtime.alloc(m * w)?;
    let eps = 1e-5_f32;
    let specs = [
        KernelSpec::RmsNorm {
            rows: batch,
            cols: width,
            epsilon: eps,
        },
        KernelSpec::Gemm {
            m: batch,
            n: hidden,
            k: width,
            tiled: true,
        },
        KernelSpec::Gemm {
            m: batch,
            n: hidden,
            k: width,
            tiled: true,
        },
        KernelSpec::SwiGlu {
            len: batch * hidden,
        },
        KernelSpec::Gemm {
            m: batch,
            n: width,
            k: hidden,
            tiled: true,
        },
        KernelSpec::Add { len: batch * width },
    ];
    let kernels = specs
        .iter()
        .map(|s| runtime.compile(s))
        .collect::<Result<Vec<_>>>()?;
    let jobs = vec![
        Dispatch {
            kernel: &kernels[0],
            buffers: vec![&x_gpu, &norm_gpu, &normalized],
        },
        Dispatch {
            kernel: &kernels[1],
            buffers: vec![&normalized, &wg_gpu, &gate],
        },
        Dispatch {
            kernel: &kernels[2],
            buffers: vec![&normalized, &wu_gpu, &up],
        },
        Dispatch {
            kernel: &kernels[3],
            buffers: vec![&gate, &up, &activated],
        },
        Dispatch {
            kernel: &kernels[4],
            buffers: vec![&activated, &wd_gpu, &projected],
        },
        Dispatch {
            kernel: &kernels[5],
            buffers: vec![&projected, &x_gpu, &output],
        },
    ];
    let reference_norm = normalize(m, w, &widen(&x), &widen(&norm), eps as f64);
    let reference_gate = dot(m, h, w, &reference_norm, &widen(&wg));
    let reference_up = dot(m, h, w, &reference_norm, &widen(&wu));
    let reference_activation: Vec<_> = reference_gate
        .iter()
        .zip(&reference_up)
        .map(|(&g, &u)| g * sigmoid(g) * u)
        .collect();
    let reference_projection = dot(m, w, h, &reference_activation, &widen(&wd));
    let expected: Vec<_> = reference_projection
        .iter()
        .zip(&x)
        .map(|(&p, &x)| p + x as f64)
        .collect();
    runtime.execute(&jobs, 3)?;
    let initial = compare(&runtime.download(&output)?, &expected, 5e-5, 5e-5)?;
    let mut batched = Vec::new();
    let mut separately = Vec::new();
    let separate = || -> Result<Execution> {
        let start = std::time::Instant::now();
        let mut gpu = Some(0.0);
        for _ in 0..repeats {
            for job in &jobs {
                let t = runtime.execute(std::slice::from_ref(job), 1)?;
                gpu = gpu.zip(t.gpu_ms).map(|(a, b)| a + b);
            }
        }
        Ok(Execution {
            host_ms: start.elapsed().as_secs_f64() * 1000.0,
            gpu_ms: gpu,
            dispatches: repeats as u64 * jobs.len() as u64,
            repeats,
        })
    };
    for round in 0..samples {
        if round % 2 == 0 {
            batched.push(runtime.execute(&jobs, repeats)?);
            separately.push(separate()?);
        } else {
            separately.push(separate()?);
            batched.push(runtime.execute(&jobs, repeats)?);
        }
    }
    let final_check = compare(&runtime.download(&output)?, &expected, 5e-5, 5e-5)?;
    let median = |v: &[Execution]| {
        let mut xs: Vec<_> = v.iter().map(|e| e.host_ms / e.repeats as f64).collect();
        xs.sort_by(f64::total_cmp);
        let n = xs.len();
        if n % 2 == 0 {
            (xs[n / 2 - 1] + xs[n / 2]) / 2.0
        } else {
            xs[n / 2]
        }
    };
    let batch_ms = median(&batched);
    let separate_ms = median(&separately);
    Ok(
        json!({"runtime":runtime.info(),"workload":"synthetic residual MLP: RMSNorm -> gate/up GEMM -> SwiGLU -> down GEMM -> residual add","trained_model":false,"shape":{"batch":batch,"width":width,"hidden":hidden},"specs":specs,"initial_check":initial,"final_check":final_check,"batched_samples":batched,"separate_submission_samples":separately,"median_batched_host_ms_per_block":batch_ms,"median_separate_host_ms_per_block":separate_ms,"ratio_separate_to_batched":separate_ms/batch_ms,"scope":"Same native kernels and resident tensors; one ordered batch versus synchronizing every dispatch. Excludes compilation, allocation, upload and readback. Not a CUDA/vendor-BLAS/full-model comparison.","output_elements":output.len()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extreme_mlp_dimensions_fail_without_wrapping() {
        assert!(mlp_elements(u32::MAX, u32::MAX, u32::MAX).is_err());
        assert_eq!(mlp_elements(1, 1, 1).unwrap(), 11);
    }
    #[test]
    fn oracle_rejects_wrong_and_nonfinite_values() {
        assert!(compare(&[0.0], &[1.0], 1e-5, 1e-5).is_err());
        assert!(compare(&[f32::NAN], &[0.0], 1e-5, 1e-5).is_err());
    }
}
