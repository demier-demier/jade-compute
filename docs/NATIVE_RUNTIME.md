# Native compute runtime and kernel compiler

The runtime API is [src/runtime/mod.rs](../src/runtime/mod.rs). It owns FP32
buffers, transfers exact slices, compiles a checked kernel description, and
executes an ordered list of dispatches before synchronizing. Buffers and kernels
are bound to one runtime instance and cannot be sent or shared between threads.
Every binding in the full batch is checked before submission. Output/input
aliasing is rejected, while read/read aliasing is allowed.

This implements a subset of CUDA-like compute concepts through a new API. It
does not implement CUDA's ABI, asynchronous stream semantics, graphs, memory
pooling, binary loader, full C++ compiler, PTX or SASS instructions.

## Frontend and code generation

`KernelSpec` is a serializable, validated Rust IR. Supported operations are
AXPY, element addition, row-major GEMM (naive and register-tiled), RMSNorm,
SwiGLU and row-wise stable softmax. The `Elementwise` variant accepts an
expression tree of inputs, finite constants, addition, multiplication, maximum,
negation and exponential, allowing a fused kernel to be built without writing
either vendor language. It supports up to eight element-aligned inputs, bounded
expression depth and size, and a separate output.

Tensor dimensions and element counts are checked against 31-bit indexing.
Tail loads/stores are guarded. Shared-memory kernels use uniform barriers and
do not assume a NVIDIA warp width for their reductions.

```sh
./target/release/jade-runtime emit \
  --spec examples/kernels/fused_axpy_relu.json --target metal \
  --output reports/fused_axpy_relu.metal
./target/release/jade-runtime emit \
  --spec examples/kernels/fused_axpy_relu.json --target corex \
  --output reports/fused_axpy_relu.cu
./target/release/jade-runtime emit-suite --directory reports/generated-native
```

Emission creates source only. It does not compile or execute CoreX code on the
Mac. The native backend invokes the actual target compiler when `compile` is
called. Generated source and launch/binding descriptions remain inspectable.

## Native Metal

The backend calls Metal directly through `MTLBuffer`, `MTLComputePipelineState`,
`MTLCommandQueue`, command buffers and compute encoders. It uses shared storage
and tracked resources; each ordered job has its own serial compute encoder.
Pipeline compilation disables fast math. Pipelines are cached per runtime.
This is the established command-queue API, not a claim to implement Metal 4's
different command/synchronization interfaces.

Native command-buffer GPU start/end timestamps are recorded after completion.
Invalid or unavailable GPU times remain absent. Host time measures encoding,
submission and completion. Neither metric includes pipeline compilation,
allocation, initial upload or output readback.

```sh
./target/release/jade-runtime --backend metal verify
./target/release/jade-runtime --backend metal mlp \
  --batch 8 --width 128 --hidden 256 --repeats 10 --samples 7
```

## Native CoreX

Status: Rust code and C/C++ bridge prepared; not built with the CoreX SDK or run
on an Iluvatar board. The constructor rejects non-Linux hosts before creating
compiler directories or loading code. Exact board architecture, SDK and driver
compatibility must be supplied by Iluvatar; no architecture is guessed.

On the matching Linux host:

```sh
COREX_ROOT=/usr/local/corex bash scripts/build-corex-runtime.sh
./scripts/cargo-local.sh build --release --locked --bins
./target/release/jade-runtime --backend corex \
  --library "$PWD/target/corex/libjade_runtime_corex.so" \
  --sdk-root /usr/local/corex --compiler /usr/local/corex/bin/clang++ \
  --architecture "$ILUVATAR_APPROVED_ARCH" verify
```

`ILUVATAR_APPROVED_ARCH` must be set to the exact compiler target supplied by the
vendor for that board. The bridge owns the native stream, allocations and timing
events. Generated modules are compiled with the selected CoreX compiler and
loaded behind a C launch ABI. The runtime keeps modules and contexts alive until
their kernels and buffers are released. Temporary compilation uses an exclusively
created private directory; cleanup removes only that owned directory.

The CoreX constructor is an unsafe Rust boundary: callers must supply a trusted
bridge and mutually compatible vendor SDK/compiler. The CLI exposes those paths
explicitly. It never silently switches to another device or precision.

## Synthetic MLP and evidence

The MLP workload is RMSNorm → gate/up GEMMs → SwiGLU → down GEMM → residual add.
It uses fixed generated weights and repeated resident inputs, without attention,
KV-cache, tokenization or training. Independent FP64 calculations validate all
output values. The benchmark compares one ordered batch against synchronizing
after every dispatch of the **same kernels**. That comparison measures scheduling
overhead and is not a comparison against CUDA, MPS, vendor BLAS or a full LLM.

[Native results](../reports/NATIVE_RESULTS.md) are separate from the older
[wgpu GEMM measurements](../reports/LOCAL_RESULTS.md). Current limitations and the
remaining CUDA subsystems are in the [reconstruction map](CUDA_RECONSTRUCTION.md).
