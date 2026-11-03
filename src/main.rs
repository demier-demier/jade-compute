use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use jade_compute::{
    corex::CorexPlan,
    portable::{self, Backend, Gpu, Kernel},
    validation, MatmulPlan, Shape, Timing,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(
    version,
    about = "Experimental Rust GPU GEMM: portable kernels and an explicit native CoreX bridge"
)]
struct Cli {
    #[arg(long, global = true, value_enum, default_value = "auto")]
    backend: BackendChoice,
    #[arg(long, global = true)]
    adapter: Option<String>,
    /// Trusted libjade_corex.so built against the target machine's CoreX SDK.
    #[arg(long, global = true)]
    corex_library: Option<PathBuf>,
    #[arg(long, global = true, default_value_t = 0)]
    device: i32,
    #[arg(long, global = true, value_enum, default_value = "fp32")]
    precision: Precision,
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum BackendChoice {
    Auto,
    Vulkan,
    Metal,
    Corex,
}
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Precision {
    Fp32,
    Fp16,
}
#[derive(Subcommand)]
enum Command {
    /// Enumerate portable GPU adapters; CoreX device is identified on plan creation.
    Devices,
    /// Compare every element of multiple odd/rectangular shapes against FP64 dots.
    Verify {
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Validate and benchmark resident matrices; selects the fastest measured kernel.
    Bench {
        #[arg(long, default_value_t = 512)]
        m: u32,
        #[arg(long, default_value_t = 512)]
        n: u32,
        #[arg(long, default_value_t = 512)]
        k: u32,
        #[arg(long, default_value_t = 10)]
        iterations: u32,
        #[arg(long, default_value_t = 7)]
        samples: u32,
        #[arg(long)]
        report: Option<PathBuf>,
    },
}
enum Engine {
    Portable(Gpu),
    Corex {
        path: PathBuf,
        device: i32,
        fp16: bool,
    },
}
impl Engine {
    fn new(cli: &Cli) -> Result<Self> {
        if cli.backend == BackendChoice::Corex {
            ensure!(cli.adapter.is_none(), "use --device for CoreX selection");
            ensure!(cli.device >= 0, "device index must be nonnegative");
            let path = cli.corex_library.clone().context(
                "--backend corex requires --corex-library /absolute/path/libjade_corex.so",
            )?;
            Ok(Self::Corex {
                path,
                device: cli.device,
                fp16: cli.precision == Precision::Fp16,
            })
        } else {
            ensure!(
                cli.precision == Precision::Fp32,
                "portable backend currently supports FP32 only"
            );
            ensure!(
                cli.corex_library.is_none() && cli.device == 0,
                "--corex-library and --device apply only to --backend corex"
            );
            Ok(Self::Portable(Gpu::new(
                portable_backend(cli.backend),
                cli.adapter.as_deref(),
            )?))
        }
    }
    fn plans(
        &self,
        shape: Shape,
        a: &[f32],
        b: &[f32],
    ) -> Result<(Vec<Box<dyn MatmulPlan>>, Value)> {
        match self {
            Self::Portable(gpu) => {
                let plans = Kernel::ALL
                    .into_iter()
                    .map(|kernel| {
                        gpu.plan(shape, kernel, a, b)
                            .map(|p| Box::new(p) as Box<dyn MatmulPlan>)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((plans, serde_json::to_value(&gpu.info)?))
            }
            Self::Corex { path, device, fp16 } => {
                // SAFETY: explicit user-supplied library path; documented trusted bridge requirement.
                let plan = unsafe { CorexPlan::load(path, shape, *device, *fp16, a, b)? };
                let info =
                    json!({"backend": "CoreX native", "device_info": plan.info, "bridge": path});
                Ok((vec![Box::new(plan)], info))
            }
        }
    }
}
fn portable_backend(b: BackendChoice) -> Backend {
    match b {
        BackendChoice::Auto => Backend::Auto,
        BackendChoice::Vulkan => Backend::Vulkan,
        BackendChoice::Metal => Backend::Metal,
        BackendChoice::Corex => unreachable!(),
    }
}
fn emit(value: &Value, report: Option<&PathBuf>) -> Result<()> {
    let content = serde_json::to_string_pretty(value)?;
    if let Some(path) = report {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, format!("{content}\n"))?;
        eprintln!("Report: {}", path.display());
    }
    println!("{content}");
    Ok(())
}
fn base_report(cli: &Cli) -> Value {
    json!({
        "schema_version": 1, "package_version": env!("CARGO_PKG_VERSION"),
        "unix_time": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        "host_os": std::env::consts::OS, "host_arch": std::env::consts::ARCH,
        "precision": if cli.precision == Precision::Fp16 { "FP16 inputs, FP32 accumulation/output" } else { "FP32" },
        "scope": "GEMM microbenchmark only; no complete model or CUDA comparison",
        "target_tiangai300_validated": false,
        "target_note": "No automatic certification. Establish exact model, driver and SDK separately from adapter identity."
    })
}
fn reference_inputs(cli: &Cli, a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    if cli.precision == Precision::Fp16 {
        let round = |v: &[f32]| v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
        (round(a), round(b))
    } else {
        (a.to_vec(), b.to_vec())
    }
}
fn oracle(
    cli: &Cli,
    shape: Shape,
    a: &[f32],
    b: &[f32],
    out: &[f32],
    budget: usize,
) -> Result<validation::Validation> {
    let mut result = validation::check(shape, a, b, out, budget, false)?;
    if cli.precision == Precision::Fp16 {
        result.reference =
            "FP64 CPU dot products over IEEE FP16-rounded inputs, testing FP32 accumulation";
    }
    Ok(result)
}
fn median(xs: &[f64]) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
    }
}
#[derive(Serialize)]
struct Measurement {
    kernel: String,
    validation: validation::Validation,
    samples: Vec<Timing>,
    median_wall_ms: f64,
    median_gpu_ms: Option<f64>,
    min_gpu_ms: Option<f64>,
    max_gpu_ms: Option<f64>,
    gflops: f64,
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.command, Command::Devices) {
        ensure!(cli.backend != BackendChoice::Corex, "CoreX: use verify --backend corex --corex-library PATH to identify the selected device; use vendor ixsmi to enumerate devices");
        return emit(
            &serde_json::to_value(portable::devices(portable_backend(cli.backend)))?,
            None,
        );
    }
    // Reject invalid requests before initializing GPU or allocating buffers.
    if let Command::Bench {
        m,
        n,
        k,
        iterations,
        samples,
        ..
    } = cli.command
    {
        let shape = Shape::new(m, n, k)?;
        ensure!(
            iterations > 0 && iterations <= 10_000,
            "iterations must be 1..10000"
        );
        ensure!((3..=101).contains(&samples), "samples must be 3..101");
        ensure!(
            (shape.a_len() as u64 + shape.b_len() as u64 + shape.c_len() as u64) * 4
                <= 128 * 1024 * 1024,
            "CLI limits each matrix set to 128 MiB to bound host/GPU allocation"
        );
    }
    let engine = Engine::new(&cli)?;
    let mut report = base_report(&cli);
    if cli.backend == BackendChoice::Corex {
        report["native_math_note"] = json!("Precision describes requested storage/compute types; CoreX math-mode semantics and exact target compatibility remain unvalidated.");
    }
    match &cli.command {
        Command::Devices => unreachable!(),
        Command::Verify { report: path } => {
            let mut checks = Vec::new();
            for (m, n, k) in [
                (1, 1, 1),
                (1, 17, 33),
                (17, 1, 35),
                (31, 47, 19),
                (64, 64, 64),
                (65, 33, 17),
                (3, 5, 257),
            ] {
                let shape = Shape::new(m, n, k)?;
                let (a, b) = validation::inputs(shape);
                let (ar, br) = reference_inputs(&cli, &a, &b);
                let (mut plans, info) = engine.plans(shape, &a, &b)?;
                report["device"] = info;
                for plan in &mut plans {
                    plan.run(1)?;
                    let check = oracle(&cli, shape, &ar, &br, &plan.download()?, usize::MAX)
                        .with_context(|| format!("{} shape {m}x{n}x{k}", plan.name()))?;
                    eprintln!(
                        "PASS {} {m}x{n}x{k}: {} elements",
                        plan.name(),
                        check.checked_elements
                    );
                    checks
                        .push(json!({"kernel": plan.name(), "shape": shape, "validation": check}));
                }
            }
            report["checks"] = json!(checks);
            report["passed"] = json!(true);
            emit(&report, path.as_ref())
        }
        Command::Bench {
            m,
            n,
            k,
            iterations,
            samples,
            report: path,
        } => {
            let shape = Shape::new(*m, *n, *k)?;
            let (a, b) = validation::inputs(shape);
            let (ar, br) = reference_inputs(&cli, &a, &b);
            let start = Instant::now();
            let (mut plans, info) = engine.plans(shape, &a, &b)?;
            report["plan_creation_host_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
            report["device"] = info;
            for plan in &mut plans {
                plan.run(1)?;
                oracle(&cli, shape, &ar, &br, &plan.download()?, 4096)
                    .with_context(|| format!("pre-benchmark validation {}", plan.name()))?;
                plan.run(3)?;
            }
            let mut timings: Vec<Vec<Timing>> = vec![Vec::new(); plans.len()];
            // Rotate first candidate each round to reduce fixed-order thermal bias.
            for round in 0..*samples as usize {
                for offset in 0..plans.len() {
                    let index = (round + offset) % plans.len();
                    timings[index].push(plans[index].run(*iterations)?);
                }
            }
            let mut results = Vec::new();
            for (plan, timings) in plans.iter_mut().zip(timings) {
                let check = oracle(&cli, shape, &ar, &br, &plan.download()?, 4096)?;
                let wall: Vec<f64> = timings.iter().map(|t| t.wall_ms_per_dispatch).collect();
                let gpu: Vec<f64> = timings
                    .iter()
                    .filter_map(|t| t.gpu_ms_per_dispatch)
                    .collect();
                let median_wall_ms = median(&wall);
                let median_gpu_ms = (!gpu.is_empty()).then(|| median(&gpu));
                let effective = median_gpu_ms.unwrap_or(median_wall_ms);
                ensure!(effective.is_finite() && effective > 0.0, "invalid timing");
                let gflops = shape.flops() / effective / 1e6;
                eprintln!(
                    "{}: {:.4} ms, {:.2} GFLOP/s ({})",
                    plan.name(),
                    effective,
                    gflops,
                    if median_gpu_ms.is_some() {
                        "GPU timestamps"
                    } else {
                        "host wall clock"
                    }
                );
                results.push(Measurement {
                    kernel: plan.name().to_string(),
                    validation: check,
                    samples: timings,
                    median_wall_ms,
                    median_gpu_ms,
                    min_gpu_ms: gpu.iter().copied().reduce(f64::min),
                    max_gpu_ms: gpu.iter().copied().reduce(f64::max),
                    gflops,
                });
            }
            let use_gpu = results.iter().all(|r| r.median_gpu_ms.is_some());
            let metric = |r: &Measurement| {
                if use_gpu {
                    r.median_gpu_ms.unwrap()
                } else {
                    r.median_wall_ms
                }
            };
            let best = results
                .iter()
                .min_by(|a, b| metric(a).total_cmp(&metric(b)))
                .unwrap();
            report["shape"] = json!(shape);
            report["iterations_per_sample"] = json!(iterations);
            report["warmup_dispatches"] = json!(4);
            report["selection_metric"] = json!(if use_gpu {
                "median GPU batch time / dispatch"
            } else {
                "median encode-submit-sync wall time / dispatch"
            });
            report["timing_scope"] = json!("Resident A/B/C, repeated same GEMM. Excludes allocation, compilation, initial upload and output download. Wall time includes encoding, submission, synchronization and timestamp resolve, but excludes timestamp mapping.");
            report["selected_kernel"] = json!(best.kernel);
            report["speedup_vs_naive_same_device"] = results
                .iter()
                .find(|r| r.kernel == "naive")
                .map(|r| json!(metric(r) / metric(best)))
                .unwrap_or(Value::Null);
            report["results"] = serde_json::to_value(results)?;
            emit(&report, path.as_ref())
        }
    }
}
