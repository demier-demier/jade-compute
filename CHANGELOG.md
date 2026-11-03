# Changelog

## 0.1.0-alpha.2 — 2026-11-03

Native Metal matrix and attention research preview. The Cargo package remains
`0.1.0`; this label identifies the repository preview.

### Added

- Checked `GemmSimd` operations with native FP32 SIMD-group matrix fragments,
  uniform cooperative operations and safe ragged-tile handling.
- Explicit `GemmMps` library operations, packed-row FP32 MPS baseline and ordered
  mixed MPS/custom-shader batches with full prevalidation and replay.
- Causal/noncausal forward self-attention with SIMD reductions and stable online
  softmax, plus a materialized scalar reference path.
- Apple7 capability / 32-lane pipeline checks; Metal-only operations fail
  explicitly for CoreX instead of silently changing backend.
- `metal-verify`, `metal-gemm`, `metal-attention` and synthetic MLP
  `--simd-gemm` / `--mps-gemm` choices.
- Reproduction script, independent FP64 checks, GPU validation tests, raw
  interleaved timing samples and source/report/binary hashes.

### Measured evidence

- 62 new case/implementation checks covering 509,710 output values; 12 host
  tests and 3 explicit GPU tests passed on the local Mac.
- At 1024³, custom SIMD GEMM measured 1.48× faster than the prior Jade kernel;
  Apple MPS remained faster than the custom kernel in all measured GPU timings.
- Causal attention (4 heads, 512 positions, dimension 64) measured 1.40× against
  the project's scalar materialized reference and avoids 8 MiB of algorithm-level
  score/probability buffers. This is not a FlashAttention/MLX/model benchmark.

See [full measurements and boundaries](reports/METAL_ADVANCES.md). No NVIDIA
performance, CUDA compatibility, CoreX execution or full-model support is claimed.

## 0.1.0-alpha.1 — 2026-11-03

Initial research prototype. This repository release label is separate from the
current Cargo package version, `0.1.0`.

### Added

- Rust runtime API for owned FP32 buffers, exact transfers, checked bindings,
  pipeline compilation, and synchronous ordered dispatch/replay.
- Operation and expression IR generating Metal Shading Language and CoreX C++.
  Operations include AXPY, Add, GEMM, RMSNorm, SwiGLU, softmax, and fused expressions.
- Direct Metal execution with resource-origin checks, alias rejection, pipeline
  caching, and native command-buffer timing.
- Prepared CoreX runtime, compiler integration, and versioned native bridge.
- Synthetic six-operation residual MLP with independent FP64 references.
- Portable wgpu GEMM workbench with three kernels and per-run measured selection.
- Separate CoreX vendor-BLAS GEMM bridge with FP32 and requested FP16/FP32-accumulation paths.
- Raw reports, source/binary manifests, generated kernel examples, architecture
  documentation, and English/Portuguese product plans.
- Linux/macOS host-check workflow and structured contribution/reporting guidance.

### Evidence and limits

- Local native Metal validation on Apple M4 Pro passed 19 numerical cases,
  comparing all 6,456 outputs; five invalid API calls were rejected.
- Earlier portable validation passed 21 kernel/shape checks across 23,244 outputs.
- Synthetic MLP measurements compare batching against synchronization after every
  dispatch of the same kernels. They are preliminary local scheduling experiments.
- CoreX SDK compilation and Iluvatar GPU execution remain unvalidated.
- No complete CUDA replacement, CUDA source/PTX compatibility, trained-model
  serving, training, or demonstrated NVIDIA speedup is claimed.
- Hosted CI checks Rust and CPU behavior; it does not run GPU benchmarks.

See [native results](reports/NATIVE_RESULTS.md) and
[historical portable results](reports/LOCAL_RESULTS.md) for the recorded measurements.
