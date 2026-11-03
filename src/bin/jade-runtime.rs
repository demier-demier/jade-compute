use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use jade_compute::runtime::{corex::CorexRuntime, ir::KernelSpec, workloads, Runtime};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about = "Jade native runtime: checked kernel IR -> Metal or CoreX, resident memory and ordered execution"
)]
struct Cli {
    #[arg(long, global = true, value_enum, default_value = "metal")]
    backend: Backend,
    #[arg(long, global = true)]
    library: Option<PathBuf>,
    #[arg(long, global = true, default_value = "/usr/local/corex/bin/clang++")]
    compiler: PathBuf,
    #[arg(long, global = true, default_value = "/usr/local/corex")]
    sdk_root: PathBuf,
    /// Exact architecture value approved for the target board by Iluvatar.
    #[arg(long, global = true)]
    architecture: Option<String>,
    #[arg(long, global = true, default_value = "target/corex-kernels")]
    cache_dir: PathBuf,
    #[arg(long, global = true, default_value_t = 0)]
    device: i32,
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum Backend {
    Metal,
    Corex,
}
#[derive(Clone, Copy, ValueEnum)]
enum Target {
    Metal,
    Corex,
}
#[derive(Subcommand)]
enum Command {
    Info,
    /// Validate native SIMD-group GEMM, Apple MPS and online self-attention.
    MetalVerify {
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Interleave FP32 GEMM measurements against Jade's prior kernel and Apple MPS.
    MetalGemm {
        #[arg(long, default_value_t = 512)]
        m: u32,
        #[arg(long, default_value_t = 512)]
        n: u32,
        #[arg(long, default_value_t = 512)]
        k: u32,
        #[arg(long, default_value_t = 10)]
        repeats: u32,
        #[arg(long, default_value_t = 9)]
        samples: u32,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Compare fused online self-attention with a materialized reference on Metal.
    MetalAttention {
        #[arg(long, default_value_t = 4)]
        heads: u32,
        #[arg(long, default_value_t = 256)]
        sequence: u32,
        #[arg(long, default_value_t = 64)]
        head_dim: u32,
        #[arg(long)]
        causal: bool,
        #[arg(long, default_value_t = 10)]
        repeats: u32,
        #[arg(long, default_value_t = 9)]
        samples: u32,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Verify {
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Mlp {
        /// Use native Metal SIMD-group FP32 GEMM in the synthetic MLP.
        #[arg(long, conflicts_with = "mps_gemm")]
        simd_gemm: bool,
        /// Mix native Apple MPS GEMM with Jade kernels in the same ordered batch.
        #[arg(long)]
        mps_gemm: bool,
        #[arg(long, default_value_t = 8)]
        batch: u32,
        #[arg(long, default_value_t = 128)]
        width: u32,
        #[arg(long, default_value_t = 256)]
        hidden: u32,
        #[arg(long, default_value_t = 10)]
        repeats: u32,
        #[arg(long, default_value_t = 7)]
        samples: u32,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Generate vendor source from an explicit JSON kernel description, without executing it.
    Emit {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long, value_enum)]
        target: Target,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export the same conformance kernels to both MSL and CoreX C++.
    EmitSuite {
        #[arg(long)]
        directory: PathBuf,
    },
}
fn emit_report(mut value: Value, path: Option<&PathBuf>) -> Result<()> {
    value["schema_version"] = json!(1);
    value["unix_time"] = json!(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs());
    value["host_os"] = json!(std::env::consts::OS);
    value["host_arch"] = json!(std::env::consts::ARCH);
    let text = serde_json::to_string_pretty(&value)?;
    if let Some(path) = path {
        write(path, &text)?;
        eprintln!("Report: {}", path.display());
    }
    println!("{text}");
    Ok(())
}
fn write(path: &std::path::Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, format!("{content}\n"))?;
    Ok(())
}
fn run<R: Runtime>(runtime: &R, command: &Command) -> Result<()> {
    match command {
        Command::Info => emit_report(json!({"runtime":runtime.info()}), None),
        Command::Verify { report } => emit_report(workloads::verify(runtime)?, report.as_ref()),
        Command::Mlp {
            simd_gemm,
            mps_gemm,
            batch,
            width,
            hidden,
            repeats,
            samples,
            report,
        } => emit_report(
            workloads::mlp_with_gemm(
                runtime,
                *batch,
                *width,
                *hidden,
                *repeats,
                *samples,
                if *mps_gemm {
                    workloads::GemmMode::Mps
                } else if *simd_gemm {
                    workloads::GemmMode::Simd
                } else {
                    workloads::GemmMode::Register
                },
            )?,
            report.as_ref(),
        ),
        _ => unreachable!(),
    }
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    if matches!(
        cli.command,
        Command::MetalVerify { .. } | Command::MetalGemm { .. } | Command::MetalAttention { .. }
    ) {
        ensure!(
            cli.backend == Backend::Metal,
            "metal-* experiments require --backend metal; no CoreX fallback"
        );
    }
    if matches!(
        &cli.command,
        Command::Mlp {
            simd_gemm: true,
            ..
        } | Command::Mlp { mps_gemm: true, .. }
    ) {
        ensure!(
            cli.backend == Backend::Metal,
            "--simd-gemm and --mps-gemm require Metal"
        );
    }
    match &cli.command {
        Command::Emit {
            spec,
            target,
            output,
        } => {
            ensure!(
                std::fs::metadata(spec)?.len() <= 65536,
                "kernel JSON exceeds 64 KiB"
            );
            let spec: KernelSpec = serde_json::from_str(&std::fs::read_to_string(spec)?)?;
            let source = match target {
                Target::Metal => spec.metal_source()?,
                Target::Corex => spec.corex_source()?,
            };
            write(output, &source)?;
            println!("Generated {} (source only; no execution)", output.display());
            return Ok(());
        }
        Command::EmitSuite { directory } => {
            std::fs::create_dir_all(directory)?;
            let mut entries = Vec::new();
            for case in workloads::cases() {
                write(
                    &directory.join(format!("{}.json", case.name)),
                    &serde_json::to_string_pretty(&case.spec)?,
                )?;
                write(
                    &directory.join(format!("{}.metal", case.name)),
                    &case.spec.metal_source()?,
                )?;
                write(
                    &directory.join(format!("{}.cu", case.name)),
                    &case.spec.corex_source()?,
                )?;
                entries.push(json!({"case":case.name,"spec":case.spec,"launch":case.spec.launch()?,"bindings":case.spec.bindings()?}));
            }
            write(
                &directory.join("manifest.json"),
                &serde_json::to_string_pretty(
                    &json!({"schema_version":1,"scope":"Generated source, not compilation or hardware validation", "cases":entries}),
                )?,
            )?;
            println!("Generated Metal + CoreX kernels in {}", directory.display());
            return Ok(());
        }
        _ => {}
    }
    match cli.backend {
        Backend::Metal => {
            ensure!(
                cli.library.is_none() && cli.architecture.is_none() && cli.device == 0,
                "native library/architecture/device selection is only available for CoreX"
            );
            #[cfg(target_os = "macos")]
            {
                let runtime = jade_compute::runtime::metal::MetalRuntime::new()?;
                use jade_compute::runtime::metal_lab;
                match &cli.command {
                    Command::MetalVerify { report } => {
                        emit_report(metal_lab::verify(&runtime)?, report.as_ref())
                    }
                    Command::MetalGemm {
                        m,
                        n,
                        k,
                        repeats,
                        samples,
                        report,
                    } => emit_report(
                        metal_lab::gemm(&runtime, *m, *n, *k, *repeats, *samples)?,
                        report.as_ref(),
                    ),
                    Command::MetalAttention {
                        heads,
                        sequence,
                        head_dim,
                        causal,
                        repeats,
                        samples,
                        report,
                    } => emit_report(
                        metal_lab::attention(
                            &runtime, *heads, *sequence, *head_dim, *causal, *repeats, *samples,
                        )?,
                        report.as_ref(),
                    ),
                    _ => run(&runtime, &cli.command),
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                anyhow::bail!("native Metal requires macOS")
            }
        }
        Backend::Corex => {
            let library = cli
                .library
                .context("CoreX requires --library pointing to libjade_runtime_corex.so")?;
            let architecture=cli.architecture.context("CoreX requires --architecture from the vendor for the exact target; it is not inferred")?;
            // SAFETY: this CLI explicitly accepts a user-selected trusted bridge
            // and vendor SDK/compiler, documented in the native runtime guide.
            let runtime = unsafe {
                CorexRuntime::new(
                    &library,
                    &cli.compiler,
                    &cli.sdk_root,
                    &architecture,
                    &cli.cache_dir,
                    cli.device,
                )?
            };
            run(&runtime, &cli.command)
        }
    }
}
