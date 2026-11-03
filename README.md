# Jade Compute — experimental Rust GPU workbench

Jade is a Rust compute-runtime and compiler prototype targeting native Apple Metal
and Iluvatar CoreX. A checked operation/expression IR generates native GPU source
for both architectures. The Metal path executes locally; the CoreX source and
runtime integration await the matching SDK and device. It is not yet a complete
CUDA replacement, CUDA source/PTX translator, or a demonstrated NVIDIA speedup.

The new [native runtime guide](docs/NATIVE_RUNTIME.md) explains the compiler,
memory and execution APIs. The [CUDA reconstruction map](docs/CUDA_RECONSTRUCTION.md)
tracks the remaining ecosystem layers. Em português:
[engenharia e estado da reconstrução](docs/RECONSTRUCAO_CUDA_PT.md).

```sh
./scripts/cargo-local.sh build --release --locked --bins
./target/release/jade-runtime info
./target/release/jade-runtime verify --report reports/native-metal-verify.json
./target/release/jade-runtime mlp --report reports/native-metal-mlp.json
./target/release/jade-runtime emit-suite --directory reports/generated-native
```

## What exists

- Native Rust runtime with owned allocations, exact transfers, pipeline caching,
  resource-origin checks and ordered command batching/replay.
- Checked expression language and operation IR lowered to Metal Shading Language
  and CoreX C++; no string-replacement CUDA translation or guessed ISA flags.
- Native AXPY, Add, GEMM, RMSNorm, SwiGLU, Softmax, and fused expression kernels.
- A synthetic resident residual MLP executes six operations with an independent
  CPU oracle. This is a neural-network block, not a trained model or token server.
- Direct Metal execution has passed 19 full numerical cases. See
  [native validation and timing](reports/NATIVE_RESULTS.md).
- Generalized CoreX allocation/transfer/stream/event bridge and runtime kernel
  compilation, requiring an explicit vendor-approved architecture. Source prepared;
  no CoreX SDK build or Chinese-GPU execution has occurred.

The earlier portable GEMM workbench remains available:

- Rust library and CLI with resident GPU buffers and checked matrix dimensions.
- Three portable FP32 WGSL kernels: naive, shared-memory tiling, register tiling.
- Correctness checks against independent FP64 CPU dot products, including odd and
  rectangular shapes; finite-value checks cover the complete output.
- Interleaved benchmark rounds with warmup, raw samples, median selection and
  explicit timing scope. Selection is per run; there is no persistent tuning cache.
- A C ABI bridge for CoreX's CUDA-compatible runtime/BLAS APIs: FP32 and FP16 input
  with requested FP32 accumulation/output. The Rust loader is implemented.
  **The native bridge has not been built against CoreX or run on an Iluvatar GPU.**
- [Verified hardware landscape](docs/HARDWARE.md),
  [product development plan](docs/PRODUCT_PLAN.md), and
  [native CoreX setup and limits](docs/COREX.md).

Plano em português: [proposta de piloto e desenvolvimento](docs/PLANO_PRODUTO_PT.md).
Resultados prévios em português: [medições, validação e limites](reports/RESULTADOS_PREVIOS_PT.md).

Rust owns host orchestration. The earlier portable GPU kernels are WGSL; the new
native Metal backend runs Jade-generated MSL kernels directly. CoreX has two
prepared paths: the earlier GEMM bridge calls vendor BLAS, while the new runtime
compiles Jade-generated device kernels with the vendor SDK and launches them
through a C ABI bridge. Neither CoreX path has been validated on Iluvatar hardware.
Changing the host language alone does not accelerate the mathematical kernel.

## Run locally

Use Rust 1.90.0 (pinned) or the project-local toolchain already installed on this
Mac. `scripts/cargo-local.sh` chooses that toolchain when present and otherwise
uses your normal Cargo. It does not modify shell startup files or global defaults.

```sh
./scripts/cargo-local.sh build --release --locked
./target/release/jade-compute devices
./target/release/jade-compute --backend metal verify --report reports/verify.json
./target/release/jade-compute --backend metal bench \
  --m 512 --n 512 --k 512 --iterations 10 --samples 7 --report reports/bench.json
```

On Linux, use `--backend vulkan` only with a working Vulkan compute driver supported
by wgpu. No particular Chinese GPU's Vulkan compatibility is assumed. `--adapter`
selects a name substring; automatic selection prefers discrete then integrated
GPUs. Software/unknown/virtual adapter types are rejected.

For native Iluvatar execution, on the supported Linux machine:

```sh
COREX_ROOT=/usr/local/corex ./scripts/build-corex.sh
./scripts/cargo-local.sh build --release --locked
./target/release/jade-compute --backend corex \
  --corex-library "$PWD/target/corex/libjade_corex.so" --precision fp16 \
  verify --report reports/corex-fp16-verify.json
./target/release/jade-compute --backend corex \
  --corex-library "$PWD/target/corex/libjade_corex.so" --precision fp16 \
  bench --m 1024 --n 1024 --k 1024 --report reports/corex-fp16-bench.json
```

Only load the trusted bridge built from this source. It loads native code into the
process. Missing libraries or unsupported modes fail explicitly; CoreX never
falls back to Metal, a CPU, or another vendor.

## Measurement contract

The operation is row-major `C[m,n] = A[m,k] × B[k,n]`, with no transpose, bias or
activation. Repeated dispatches reuse A/B/C allocations and the same inputs.
Compilation, allocation, initial upload and output readback are outside the timed
batch. The host metric includes command encoding, submission and synchronization;
timestamp mapping is outside that metric. Native CoreX event timing includes its
stream's batch of library calls. These are not full-inference timings.

In the earlier wgpu experiments, the local Apple M4 Pro reported timestamp
support but returned `(0,0)` timestamps. That backend rejected those intervals and
used synchronized wall-clock timings; the portable JSON reports preserve
`gpu_ms_per_dispatch: null`. The newer direct Metal backend obtained valid,
positive command-buffer GPU timestamps, recorded separately in the native reports.

`verify` checks every output in seven small shapes for each kernel. `bench` checks
up to 4,096 evenly distributed output positions before and after timing, plus all
outputs for NaN/Inf. FP16 validation independently rounds inputs to IEEE FP16
before computing the FP64 oracle. Requested native math modes are not proof of
CoreX's internal arithmetic semantics.

Benchmark reports select the fastest candidate for that shape on that run. A
speedup against this project's naive shader is **not** a CUDA, BLAS, full-model, or
cross-hardware speedup. Larger/tiled kernels may be slower for some shapes.

## Current validation

See [local results](reports/LOCAL_RESULTS.md) and the accompanying raw JSON files.

```sh
./scripts/cargo-local.sh test --locked
./scripts/cargo-local.sh clippy --all-targets --locked -- -D warnings
./scripts/cargo-local.sh fmt --all -- --check
c++ -std=c++17 -O2 native/fp16_test.cpp -o target/fp16-test
./target/fp16-test
```

No model serving, training, attention, KV-cache, distributed communication,
autograd, CUDA source translation, or vendor ISA compiler is implemented here.
The proposed business product and its performance targets are research hypotheses
in the product plan. No company was contacted and no partnership is claimed.
