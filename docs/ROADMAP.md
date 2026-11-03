# Roadmap

The next objective is to move from a validated native GPU prototype to a useful
model workload on a qualified target. Milestones are evidence gates, not release
dates or a promise of complete CUDA compatibility. Metal and CoreX development
can proceed in parallel; results remain specific to the backend tested.

## Implemented native foundation

- Checked expression frontend and operation IR with MSL/CoreX source generation.
- Owned buffers, resource validation, compilation reuse and ordered execution.
- AXPY, Add, GEMM, RMSNorm, SwiGLU, softmax and custom elementwise expressions.
- Native FP32 SIMD-group GEMM, Apple MPS GEMM and ordered mixed MPS/shader batches.
- Small causal/noncausal forward self-attention, with an online-softmax kernel
  and a materialized reference path. No KV cache or backward pass.
- Alpha 1 execution history: nineteen GPU numerical cases, regular/irregular
  synthetic MLPs and an explicitly executed GPU ownership/ordering test.

History: [alpha 1 native results](../reports/NATIVE_RESULTS.md).
Expansion: [implementation and limits](METAL_ADVANCES.md),
[validation and measurements](../reports/METAL_ADVANCES.md).
Implementation: [IR](../src/runtime/ir.rs), [runtime contract](../src/runtime/mod.rs),
[Metal](../src/runtime/metal.rs), [CoreX adapter](../src/runtime/corex.rs) and
[workloads](../src/runtime/workloads.rs).

## 1. Qualify the first CoreX target

**Work:** obtain a usable TianGai 300 Linux environment; record exact board,
driver, SDK and compiler versions; confirm the architecture flag; build the
native bridge and compile the emitted kernels.

**Done when:** the shared operation subset and MLP checks run successfully on
the device, with independent reference values, raw reports, unsupported cases
and a reproducible environment manifest. A source export or successful host
build alone does not satisfy this milestone. Metal-specific SIMD/MPS/attention
operations need separate CoreX implementations and qualification.

Starting points: [CoreX setup](COREX.md),
[native runtime guide](NATIVE_RUNTIME.md),
[bridge build script](../scripts/build-corex-runtime.sh).

## 2. Strengthen numerical and runtime qualification

**Work:** broaden shape and value coverage, exercise resource/error behavior,
inspect GPU traces and repeat measurements across sessions. Extend capability
reporting and document each supported backend/SDK combination.

**Done when:** failures have minimal reproducible cases, reported timings have
explicit scopes and source/environment records, and regressions can be detected on each
qualified target. Use the [Metal verification script](../scripts/verify-native-metal.sh)
and [current workload cases](../src/runtime/workloads.rs) as starting points.

## 3. Integrate one real model workload

**Work:** agree on a checkpoint, task, precision, batch/context and quality
target; identify required operators; implement the missing model integration
and compare against a relevant optimized native baseline.

**Done when:** the workload produces correct model outputs and task-quality
results, with end-to-end latency/throughput, memory and available power data.
Synthetic MLP timing is insufficient. See the [product plan](PRODUCT_PLAN.md)
and [possible commercial evaluation scope](COMMERCIAL.md).

## 4. Optimize the measured bottlenecks

**Work:** extend beyond the implemented SIMD GEMM, MPS integration and small
online-attention kernel: evaluate tiled/fused attention with K/V reuse,
device-specific schedules, memory reuse and execution plans. Add
correctness-gated tuning and a persistent cache with explicit
compiler/driver/device invalidation rules.

**Done when:** changes improve the agreed workload at the same quality and
precision, survive repeated runs and expose regressions. Retain the baseline
and raw samples; do not promote a single kernel result into a model-wide claim.

## 5. Expand only with a concrete workload

Candidate extensions include full-model attention/KV-cache integration, additional
precisions, backward operators, optimizers, collective communication and other
vendor backends. Each requires its own semantics, numerical tests and target
execution. Training additionally needs gradient checks and convergence/time-to-
quality evidence. Wider migration features follow the
[CUDA responsibility map](CUDA_RECONSTRUCTION.md).

## Useful ways to contribute

| Contribution | Information that makes it actionable |
| --- | --- |
| Numerical failure | Operation/specification, inputs, expected output and tolerance. |
| Runtime issue | Small reproduction, backend, OS/driver/SDK/compiler versions and error. |
| Hardware validation | Exact device/software configuration and complete generated reports. |
| Performance investigation | Workload, baseline, timing boundaries, correctness and raw samples. |
| Model request | Checkpoint/task, required operators, shapes, precision and quality criteria. |

Start with a technical issue. For business discussions, contact
[Bernardo Demier on LinkedIn](https://www.linkedin.com/in/bernardo-demier-3176001ab/).
Discuss scope and contribution terms before preparing substantial code: the project has not
granted an open-source license. See the [FAQ](FAQ.md) and
[commercial direction](COMMERCIAL.md).
