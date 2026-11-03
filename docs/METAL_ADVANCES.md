# Native Metal matrix and attention experiments

Jade now has an explicit FP32 SIMD-group GEMM, a forward online-attention
kernel, and an Apple Metal Performance Shaders (MPS) GEMM baseline. They run
through the native Rust/Metal backend, using the same checked buffers and
operation IR as the existing kernels.

This expansion was executed and validated on the local **Apple M4 Pro**.
The complete checks and raw timings are in [the measurement report](../reports/METAL_ADVANCES.md).
Capability checks alone do not establish correctness or performance on other
Apple devices, and these Metal operations do not establish CoreX support.

## Concepts and native implementation

| Familiar concept | Jade's Metal implementation | Compatibility boundary |
|---|---|---|
| Cooperative matrix fragments, as in CUDA WMMA | `simdgroup_float8x8`, cooperative loads/stores and `simdgroup_multiply_accumulate` | Native MSL operations; no WMMA ABI, PTX translation or NVIDIA Tensor Core emulation |
| Warp-wide dot-product reduction | Metal `simd_sum` across the full SIMD group | Uses Metal's actual pipeline execution width; no CUDA warp intrinsic is called |
| Avoid storing the attention score matrix | Stable online softmax with a running maximum, normalizer and weighted output | A small forward kernel inspired by published attention algorithms; not the full FlashAttention implementation |
| Vendor-optimized matrix multiplication | Public `MPSMatrixMultiplication` with FP32 descriptors | An Apple-library baseline on the same device, not a CUDA comparison |

NVIDIA documents WMMA as a cooperative warp-level matrix API. Jade uses that
programming concept through Apple's own matrix primitives; the two APIs and
their fragment representations are not interchangeable.
[NVIDIA WMMA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-matrix-functions)
and [Apple's SIMD-group matrix and reduction explanation](https://developer.apple.com/videos/play/tech-talks/10858/)
provide the respective native models.

The runtime requires **Apple GPU family 7 or newer** and a compiled pipeline
`threadExecutionWidth` of **32** for the new SIMD-group operations. These are
explicit checks, not assumptions inferred from a device name. Apple's current
[feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
list SIMD-scoped reductions and matrix multiplication from Apple7. See also
[Metal threadgroups and SIMD groups](https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups).

## FP32 GEMM

`GemmSimd` computes dense row-major `C[M,N] = A[M,K] × B[K,N]`. A group of
128 threads covers a 32×32 output tile. Its four SIMD groups each retain four
8×8 FP32 accumulators. Cooperatively loaded tiles are reused across the inner
dimension; out-of-range inputs are zero-filled. Boundary output tiles use
shared staging and checked scalar stores, so ragged dimensions are supported.
Collective operations stay in uniform control flow.

Inputs, output, fragments and accumulation remain FP32. The compiler's fast
math option is disabled. The reduction order differs from scalar accumulation,
so numerical comparison uses tolerances rather than bitwise identity. This
implementation makes no claim about Apple's undocumented physical execution
units. The native code is in [metal_simd.rs](../src/runtime/metal_simd.rs).

The experiment compares three implementations: the existing register-tiled
Jade GEMM, the SIMD-group GEMM, and Apple's MPS GEMM. The MPS path uses the
public [matrix multiplication API](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
with no transposition, `alpha=1`, `beta=0`, and `MPSDataTypeFloat32` for every
matrix. `MPSKernelOptionsNone` leaves reduced precision disabled. Descriptors
use the actual packed stride `rowBytes = 4 × columns`; no hidden repacking or
FP16 conversion is inserted. Apple's recommended padded stride is optional,
and this baseline deliberately keeps the same packed tensor layout.

MPS GEMM is also a first-class runtime operation, `KernelSpec::GemmMps`.
`Runtime::compile` prepares the native MPS object explicitly;
`Runtime::execute` can mix it with generated shader jobs in one ordered,
replayed command buffer. The complete batch is validated before encoding, and
all MPS matrix wrappers are prepared outside the clock and retained until GPU
completion. Tracked buffers preserve dependencies in both directions between
MPS and shader jobs. This does not replace existing `Gemm` operations or add a
silent fallback: callers select the MPS operation themselves.

## Forward self-attention

The attention contract is:

```text
Q, K, V, O: contiguous FP32 [H, S, D]
element offset: ((head * S + position) * D + channel)
O = softmax(Q Kᵀ / sqrt(D)) V
1 ≤ H ≤ 32; 1 ≤ S ≤ 2048; 1 ≤ D ≤ 128
```

All tensors have the same head count and sequence length. Causal mode permits
keys at or before the query position, including the diagonal; otherwise all
keys participate. Inputs and intermediate FP32 scores must remain finite.
Output must not alias an input. There is no batch axis, cross-attention,
grouped-query attention, padding mask, dropout, backward pass or KV cache.

One full 32-lane SIMD group handles one query. Lanes keep query/output
components in registers and use `simd_sum` to combine the dot product. As each
key arrives, the kernel updates its maximum, rescales the previous normalizer
and weighted output, and incorporates the new value. It does not allocate a
global `[H,S,S]` score or probability tensor.

This online-normalization and memory-traffic approach is credited to published
research, including Dao et al.'s original
[FlashAttention paper](https://arxiv.org/abs/2205.14135). It is an algorithmic
idea, not an NVIDIA API being emulated. Jade's implementation is a small
scalar-SIMD forward kernel: it does not reproduce the paper's tiled matrix
schedule, backward implementation or performance claims. It still performs
quadratic work in sequence length and does not reuse K/V tiles across queries.

The comparison path is this project's materialized QK → softmax → PV sequence,
using scalar kernels. It is not optimized FlashAttention, MPS attention, MLX
or a full-model baseline. The materialized path allocates two FP32 `[H,S,S]`
intermediates; the online path avoids those two algorithm-level scratch
buffers. The reported scratch count is not measured peak GPU/process memory:
common tensors, registers and driver resources are excluded, and the benchmark
keeps both paths allocated. Source: [metal_attention.rs](../src/runtime/metal_attention.rs).

## Reproduce the experiments

Run from the repository root on a supported Mac with Rust and the Apple Metal
toolchain. The commands below write new reports; they do not modify the
published evidence under `reports/`. Fresh outputs go under
`target/metal-advances/`.

```sh
bash scripts/cargo-local.sh run --release --locked --bin jade-runtime -- \
  --backend metal metal-verify \
  --report target/metal-advances/verify.json

bash scripts/cargo-local.sh run --release --locked --bin jade-runtime -- \
  --backend metal metal-gemm --m 512 --n 512 --k 512 \
  --repeats 10 --samples 9 --report target/metal-advances/gemm-512.json

bash scripts/cargo-local.sh run --release --locked --bin jade-runtime -- \
  --backend metal metal-attention --heads 4 --sequence 256 --head-dim 64 \
  --repeats 10 --samples 9 --report target/metal-advances/attention-256.json

bash scripts/cargo-local.sh run --release --locked --bin jade-runtime -- \
  --backend metal metal-attention --heads 4 --sequence 256 --head-dim 64 \
  --causal --repeats 10 --samples 9 \
  --report target/metal-advances/attention-256-causal.json

bash scripts/cargo-local.sh run --release --locked --bin jade-runtime -- \
  --backend metal mlp --batch 8 --width 128 --hidden 256 --mps-gemm \
  --repeats 10 --samples 9 --report target/metal-advances/mlp-mps.json
```

`mlp --mps-gemm` selects MPS for the GEMM nodes while normalization, SwiGLU
and residual addition remain generated shader jobs. `--simd-gemm` selects
the explicit SIMD-group variant instead; the flags are mutually exclusive.
Without either flag, the existing register-tiled GEMM remains the default.
The workload is still a synthetic MLP, not a trained-model inference result.

`metal-verify` exercises ragged and small GEMMs, native MPS, causal/noncausal
attention, uniform-score rows and larger finite logits. It compares every
output against independent FP64 CPU references before and after the measured runs.
Its timings are diagnostics, not performance results.

The benchmark commands warm up each variant and interleave measured samples
with a rotating first variant. They retain raw timings and numerical checks.
`--repeats` accepts 1–100 and `--samples` accepts 3–31. The experiment harness
also caps GPU tensor allocations at 128 MiB and CPU-reference work at `2^31`
GEMM multiply-add terms or `2^29` attention dot-product terms. These harness
limits can reject combinations that individually satisfy the shape contract.

## Timing and results

Host time covers command-buffer creation, encoding, submission and completion
wait. Native GPU timestamps cover the completed command buffer. Compilation,
allocation, upload, readback and the CPU oracle are outside the timed region.
Missing or unusable GPU timestamps are reported as absent.

MPS kernel/descriptors are created at preparation time, and its matrix wrappers
are created before the execution clock starts. Its first encode can still
trigger internal lazy setup, which is why warm-up is required. MPS and custom
kernels share the same submit/wait/timestamp boundary. For MPS,
`Execution.dispatches` counts logical GEMM encodes; the framework's internal
GPU kernel count is opaque. Times per repeat therefore describe one logical
GEMM or attention operation, not one hardware instruction or vendor dispatch.

[Measured results and validation status](../reports/METAL_ADVANCES.md) are
maintained separately from this implementation note, with published raw JSON
under `reports/metal-advances/`. Any eventual speed ratio
must identify its shape, precision, baseline and timing metric. Results from
one Apple M4 Pro do not establish a CUDA speedup, Chinese-GPU support,
training throughput or complete-model inference performance.
