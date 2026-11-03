# Native Metal advances — measured 2026-11-03

Apple M4 Pro, macOS 27.2, Rust/Cargo 1.90.0, release build. Source snapshot:
[`ad34926`](https://github.com/demier-demier/jade-compute/commit/ad349266616cc6cbcb51a2a4b46376a438944c4c).
The public preview is `v0.1.0-alpha.2`; the Cargo package remains `0.1.0`.

This run validates FP32 SIMD-group matrices, forward online self-attention and
mixed Apple MPS/custom-kernel batches. [Implementation and public API references](../docs/METAL_ADVANCES.md).
These are local operator and synthetic-block measurements, not full-model,
NVIDIA, CUDA, CoreX, training or energy-efficiency benchmarks.

## Validation

- **12 GEMM shapes × 3 implementations**: prior Jade register tiling, new native
  SIMD-group matrices, and Apple's packed-row FP32 MPSMatrixMultiplication.
- **11 attention configurations × 2 implementations**: materialized reference
  and online recurrence; causal/noncausal, ragged dimensions, uniform scores,
  and larger finite logits.
- **4 FP32 precision probes** through the actual runtime: operands not exactly
  representable in FP16, including a value above FP16's finite range.
- In total, **62 case/implementation checks, covering 509,710 output values**.
  GEMM/attention outputs were checked before and after timing; this count includes
  each output per case/implementation once, not twice for the repeated checks.
  Every value was finite and within its declared absolute/relative tolerance.
- In the expanded verification suite, largest absolute error was **4.8075e-7**
  for normalized GEMM and **1.3063e-5** for attention including large logits.
  The separately timed, larger GEMM cases reached **3.1108e-6** (1024³), also
  within the declared tolerance. The large-value precision
  probe had **0.001194** absolute error at an output near 70,002, or **1.71e-8**
  relative error. A large absolute value is not hidden behind the smaller-case result.
- **12 host tests + 3 explicit GPU tests passed**, including mixed MPS → shader
  → MPS ordering/replay, ownership, alias rejection and rejecting an invalid later
  job before the first submission. Rustfmt and Clippy with warnings denied passed.
- The original 19-operator suite also passed again on the new runtime.

Metal API and shader validation were enabled for correctness runs. Timings below
were collected separately with those instrumentation variables unset.
[Expanded verification](metal-advances/verify.json) ·
[Original-operator regression](metal-advances/native-verify.json).

## GEMM: actual GPU time

Dense row-major `C[M,N] = A[M,K] × B[K,N]`, FP32 inputs/output/accumulation
contract, no hidden FP16 conversion. Alpha=1, beta=0 for MPS. Metal fast math is
disabled for Jade; MPS uses options=None without AllowReducedPrecision.

**Median native GPU milliseconds per GEMM; lower is better.**

| M × N × K | Prior Jade | Jade SIMD | Apple MPS | Prior / SIMD | Raw samples |
|---|---:|---:|---:|---:|---|
| 128 × 128 × 128 | 0.120658 | 0.050187 | 0.025250 | 2.404× | [gemm-128.json](metal-advances/gemm-128.json) |
| 512 × 512 × 512 | 0.162946 | 0.116917 | 0.057263 | 1.394× | [gemm-512.json](metal-advances/gemm-512.json) |
| 1024 × 1024 × 1024 | 1.256200 | 0.850200 | 0.408200 | 1.478× | [gemm-1024.json](metal-advances/gemm-1024.json) |
| 127 × 257 × 129 | 0.029629 | 0.014167 | 0.006417 | 2.091× | [gemm-ragged.json](metal-advances/gemm-ragged.json) |
| 1 × 1024 × 1024 | 0.156663 | 0.063000 | 0.006987 | 2.487× | [gemm-vector.json](metal-advances/gemm-vector.json) |

The custom SIMD kernel improves on the prior Jade kernel for every measured
shape in this run. **MPS is faster in GPU time for every measured shape.** The
1024³ comparison is 1.48× for Jade SIMD versus prior Jade; using the native MPS
path gives 3.08× versus that same prior kernel. The latter is Apple's optimized
library executing through Jade, not an original Jade kernel outperforming Apple.

MPS uses packed row strides matching these tensors. Apple may prefer padded
strides for other workloads; this does not measure the best possible MPS setup.
Host timing has a different ordering on the ragged case because it includes
submission overhead: MPS's default per-encode validation is inside its host timer,
while Jade's binding validation is before its timer. GPU time is therefore the
primary compute metric here. Both host and GPU samples are retained in JSON.

## Self-attention: time and scratch storage

All cases use 4 heads and dimension 64, with scale `1/sqrt(64)`. The online path
has one fused operation; the materialized reference uses QK → softmax → PV.

**Median native GPU milliseconds per complete attention operation.**

| Sequence / mask | Scalar materialized | Online SIMD | Materialized / online | Score/probability scratch avoided | Raw samples |
|---|---:|---:|---:|---:|---|
| 256, noncausal | 0.126925 | 0.100788 | 1.259× | 2 MiB | [attention-256.json](metal-advances/attention-256.json) |
| 512, noncausal | 0.490517 | 0.437350 | 1.122× | 8 MiB | [attention-512.json](metal-advances/attention-512.json) |
| 512, causal | 0.349667 | 0.250108 | 1.398× | 8 MiB | [attention-causal-512.json](metal-advances/attention-causal-512.json) |

The reference is this project's simple scalar implementation, not optimized
FlashAttention, MLX, MPS attention or an inference server. The online path still
has quadratic arithmetic. It implements the stable online-softmax recurrence,
not the full tiled FlashAttention schedule or backward pass.

Scratch counts are exact algorithm-level sizes for two FP32 `[H,S,S]` buffers.
They are **not measured peak GPU/process memory** and exclude shared Q/K/V/O,
registers and driver allocations. Both paths remain allocated in the benchmark.
The online kernel requires no global score/probability buffer.

## Mixed execution in a synthetic AI block

The six-operation residual MLP (RMSNorm → gate/up GEMMs → SwiGLU → down GEMM
→ residual Add) now supports `--simd-gemm` and `--mps-gemm` explicitly. The MPS
mode uses three native MPS operations in the same ordered batch as Jade shaders.
It has fixed generated weights, no checkpoint/tokenizer, attention layer or training.

Both modes at batch 64 / width 512 / hidden 1024 validated all **32,768 outputs**
with maximum absolute error **6.7689e-7**. Recorded median host times per batched
block were 0.2404042 ms (SIMD) and 0.1355625 ms (MPS). These two mode runs were
separate, so no controlled cross-mode speedup is claimed from them. Their internal
batch-versus-synchronize-each-operation ratios remain scheduling experiments.
[SIMD MLP](metal-advances/mlp-simd.json) · [Mixed MPS MLP](metal-advances/mlp-mps.json).

## Method and reproducibility

Each benchmark warms up each implementation with three repeats, then collects
nine interleaved rounds with ten repeats per sample. GEMM rotates the first of
three implementations; attention alternates which of two runs first. The figures
show individual GPU samples and medians. No fastest-sample selection is used.
This was one desktop session, not an isolated multi-machine lab measurement.

Inputs are shared and resident; equally sized outputs are separate per variant.
Compilation, allocation, upload, readback and the independent FP64 CPU oracle
are outside the timed region. Host time covers encoding/submission/wait;
native GPU timestamps cover the complete command buffer, including dependencies.
MPS wrapper creation is before timing; its internal GPU kernel count is opaque.
Times are normalized by logical operation repeats, never vendor dispatch count.

```sh
bash scripts/verify-metal-advances.sh target/metal-reproduction-new
```

Choose a new directory: the script refuses to overwrite existing reports.
It builds, validates, measures, emits examples and writes source/report/binary
SHA-256 hashes. [Manifest](metal-advances/manifest.json). The figure is reproducible
with `python3 scripts/render-metal-results.py` and has its own
[provenance](../assets/metal-advances-provenance.json).

The original alpha.1 reports remain historical evidence tied to their earlier
source snapshot. They are not overwritten or presented as measurements of the
expanded runtime. No Apple partnership, endorsement or production certification
is claimed; the project remains a public research prototype with a separate
[commercial direction](../docs/COMMERCIAL.md).
