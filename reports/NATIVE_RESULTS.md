# Native runtime validation — 2026-11-03

This is the new direct Metal backend, not the previous wgpu backend.
Hardware: Apple M4 Pro. Rust 1.90.0, release build, generated FP32 MSL kernels,
Metal fast math disabled. Native CoreX execution is not available on this host.

## Verified execution

- 19 native GPU conformance cases passed; every one of 6,456 output
  elements was compared with independent FP64 CPU calculations.
- Coverage: AXPY, Add, naive/register-tiled GEMM, RMSNorm, SwiGLU, stable softmax,
  custom fused AXPY/ReLU and a custom exponential/negation expression.
- Five invalid calls rejected: short upload, aliased output, wrong output length,
  empty batch and zero repetitions.
- A separate hardware test passed ordering, replay, resource ownership and
  prevalidation of the entire batch before any dispatch.
- Cargo test: nine ordinary tests passed; the GPU-specific test was initially
  ignored by default and then explicitly executed successfully. Ten distinct
  unit/integration tests passed overall. Clippy denied warnings; rustfmt passed.
- CoreX host tests cover the unsupported-platform guard and private compiler
  directory ownership/cleanup. They do not exercise the SDK or hardware.
- Generated sources for all 19 cases were exported to both Metal and CoreX.
  This is source emission, not proof of CoreX compilation or execution.

## Synthetic residual MLP

The executed chain contains six dispatches: RMSNorm, gate GEMM, up GEMM, SwiGLU,
down GEMM and residual Add. All tensors remain allocated on the GPU. Weights are
fixed generated values; this is not a trained model, LLM server or token workload.

| Shape (batch, width, hidden) | Outputs checked | Maximum absolute error |
| --- | ---: | ---: |
| (8, 128, 256) | 1024 | 2.00790038e-07 |
| (3, 33, 65), irregular tails | 99 | 9.18883258e-08 |

For the first shape, seven alternating rounds of ten block repetitions gave:

- One ordered batch: **0.1940625 ms per block**, median host encode/submit/wait time.
- Synchronizing after each of the same six dispatches: **1.1256333 ms per block**.
- Ratio: **5.800×**, against this deliberately synchronization-heavy baseline.

Compilation, allocations, initial upload, CPU reference and output readback are
outside both timed intervals. Raw samples include valid native command-buffer
GPU times as well as host times. The ratio is not a CUDA, MPS, vendor BLAS,
full-model, Chinese-hardware, or cross-device comparison. The initial run's
7.813× ratio is retained separately; measurements vary between runs.

### Publication reproduction

Re-running the README commands on the same Mac before the public preview again
passed 19 numerical cases and the MLP's 1,024-output initial/final checks. The
MLP maximum absolute error remained 2.00790038e-07. No source or binary changed.

| Recorded run | Batched median, ms/block | Per-dispatch sync median, ms/block | Ratio |
| --- | ---: | ---: | ---: |
| Initial | 0.1109166 | 0.866625 | 7.813× |
| Figure source | 0.1940625 | 1.1256333 | 5.800× |
| Publication reproduction | 0.2094125 | 0.9312000 | 4.447× |

This variation is retained rather than presenting the best ratio as a stable
expectation. All runs use the same synthetic workload; they are not independent
hardware validations. The reproduction's report has its own hash manifest.

## Files and reproducibility

- [All operator checks](native-metal-verify.json)
- [MLP measurements used in the figure](native-metal-mlp.json)
- [Publication reproduction](native-metal-mlp-reproduction.json)
- [Reproduction provenance](publication-reproduction-manifest.json)
- [Irregular MLP measurements](native-metal-mlp-ragged.json)
- [Initial MLP measurements](native-metal-mlp-initial.json)
- [Generated kernel inventory](generated-native/manifest.json)
- [Source and binary hashes](native-manifest.json)

Run `./scripts/verify-native-metal.sh` to create a fresh timestamped report
folder. The script exercises the real GPU and exports CoreX source without
claiming to run it. The broader CUDA reconstruction is still incomplete:
[coverage map](../docs/CUDA_RECONSTRUCTION.md).
