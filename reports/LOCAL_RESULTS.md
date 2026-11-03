# Earlier portable validation — 2026-11-03

This report records the earlier Rust/wgpu GEMM stage. The later direct Metal
runtime and synthetic MLP results are separate in [NATIVE_RESULTS.md](NATIVE_RESULTS.md).

Hardware: Apple M4 Pro, integrated GPU, 16 GPU cores, macOS 27.2, Metal backend.
Rust 1.90.0; wgpu 27.0.1; release profile with thin LTO. This machine has no
Iluvatar GPU or CoreX SDK. No Chinese GPU compatibility or NVIDIA comparison
has been validated.

## Actual results

- Rust unit tests: 2 passed (invalid dimensions and a deliberately wrong/NaN oracle case).
- Clippy all targets with warnings denied: passed. Rustfmt: passed.
- Real Metal execution: 21 kernel/shape checks passed, covering all 23,244 output
  values across three kernels and seven shapes; each has an FP64 CPU reference.
- Native CPU FP16 helper: 63,488 finite roundtrips and rounding-boundary checks
  passed. This exercises neither CoreX headers nor a GPU.
- CLI rejection checks: missing bridge argument, absent library, unsupported
  portable FP16, and zero dimensions return errors.
- Native build shell syntax checked; its macOS guard rejects the unsupported host.
  The native C header was syntax checked separately. No native CoreX link or run.

## Resident-matrix throughput experiment

Times below are median **host encode-submit-synchronize milliseconds per GEMM**,
not GPU kernel timestamps or end-to-end model latency. Seven rotating rounds,
ten dispatches per sample, four untimed dispatches including validation/warmup.

| M × N × K | Naive ms | Tiled16 ms | Register32 ms | Selected | Ratio vs naive |
| --- | ---: | ---: | ---: | --- | ---: |
| 512 × 512 × 512 | 1.1720 | 1.3186 | 1.0360 | register32 | 1.131× |
| 1024 × 1024 × 1024 | 8.0261 | 9.4487 | 6.5252 | register32 | 1.230× |
| 1 × 1024 × 1024 | 0.3058 | 0.3061 | 0.6083 | naive | 1.000× |

A 1.230× ratio means 23.0% higher operation throughput in that local experiment
against this project's naive shader. It is not a 23% gain over CUDA, native BLAS,
or inference. The vector-like case selects the naive shader; register tiling is
almost twice as slow there. This is evidence against a universal-kernel claim.

GPU timestamp capability was advertised, but the actual returned values were
zero. The runtime rejected those timings and used synchronized wall clock.
Raw reports keep GPU time null. This is a single local experiment per shape,
not a multi-session statistical certification or a vendor performance result.

512 and 1024 cases check 4,096 outputs before and after timing and scan all outputs
for nonfinite values. The vector case checks all 1,024 outputs. Irregular small
shapes are fully checked by the separate verification command.

## Artifacts

- [Full GPU correctness report](apple-m4-pro-verify.json)
- [512-square samples](apple-m4-pro-512.json)
- [1024-square samples](apple-m4-pro-1024.json)
- [Vector-like samples](apple-m4-pro-vector.json)
- [Historical environment and source checksums](manifest.json)

The portable manifest preserves the source and executable hashes from that
earlier stage. Its `Cargo.toml`, `Cargo.lock`, `src/lib.rs`, and executable hashes
differ from the expanded project and rebuilt executable; they must not be read as
checksums of the current checkout. The later [native manifest](native-manifest.json)
matched its 28 source files, three listed reports, and native executable when
checked for publication. The original portable measurements and hashes are retained.

At this measurement stage, the implementation was a GEMM workbench. The project
subsequently gained a native operation/expression IR, direct Metal execution and
a validated synthetic MLP. Trained-model serving, partner pilots and the product
plan's commercial success thresholds remain future work.
