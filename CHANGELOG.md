# Changelog

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
