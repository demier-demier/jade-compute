# Native Iluvatar CoreX bridge

Status: source prepared; not compiled against CoreX or executed on an Iluvatar
GPU in this delivery. The available machine is an Apple M4 Pro. Local Metal
results do not validate Tiangai 300 compatibility or performance.

This bridge calls the CUDA-shaped C interfaces supplied by the **Iluvatar
CoreX SDK**. It does not install NVIDIA CUDA, translate PTX, replace the CoreX
driver, or provide a complete AI framework. The Rust API owns a native context;
the C++ shim keeps SDK-specific enum and structure layouts behind a small C ABI.

The implementation provides persistent device matrices, one nonblocking stream,
a reusable vendor BLAS handle, and reusable GPU timing events. Every GEMM writes
`C = A × B`; all matrices are dense and row-major. `cublasSgemm(n,m,k,B,A,C)`
uses the column-major transpose identity without transposing or copying data.

Precision 0 uses FP32 SGEMM. Precision 1 converts host FP32 inputs to IEEE FP16
(nearest, ties to even), calls `cublasGemmEx` with FP32 accumulation and FP32
output, and requests tensor math when that SDK exposes the option. The CPU
reference for this mode must use the FP16-rounded inputs. This path is prepared
for AI matrix operations, but it does not implement BF16, FP8/FP4, attention,
quantization, automatic differentiation, model loading, or distributed training.

## Build on the target Linux host

Use the vendor-provided driver and SDK version approved for the actual Tiangai
300 board. Do not infer the device architecture from an older BI-V100/V150
example. No GPU architecture flag is required here: the vendor BLAS library
contains its device kernels.

```sh
export COREX_ROOT=/usr/local/corex
bash scripts/build-corex.sh
```

The script requires `include/cuda_runtime.h`, `include/cublas_v2.h`, and the
SDK's `libcudart.so` and `libcublas.so` in `lib64`, `lib`, or `lib/x64`. It uses a
host C++17 compiler (`CXX`, the SDK's `clang++`, or system `c++`), checks that the libraries resolve
inside the chosen SDK, probes the actual GemmEx signature, and embeds SDK
library search paths. It produces `target/corex/libjade_corex.so`. The SDK and
its transitive runtime dependencies must remain installed at runtime.

Some SDK releases use `CUDA_R_32F` for GemmEx's compute type; newer interfaces
use `CUBLAS_COMPUTE_32F`. The build checks the function signature instead of
hard-coding a version guess. Unsupported headers or symbols cause compilation
or linking to fail. A successful build still requires numerical validation on
the target card; a `NOT_SUPPORTED` result for a precision/layout must remain a
failure, never a silently relabeled FP32 run.

## ABI and timing

[`native/jade_corex.h`](../native/jade_corex.h) is the complete ABI. Version is
`1`; Rust must check it before creating a context. Each call accepts a caller
owned bounded error buffer. Null handles, zero/oversized dimensions, out-of-range
devices, incomplete execution order, and incorrect element counts are rejected.
Callers still must pass valid C pointers and must not reuse a destroyed handle.

Calls on one context require serial access. Upload and download synchronize
before returning, so borrowed Rust slices do not outlive pending host transfers.
The bridge selects its device for each operation. Destruction releases resources
best-effort and is null-safe; it does not reset the device or affect other users.

`jade_corex_run` reports total GPU-event milliseconds over the requested number
of GEMMs. Divide by iterations for milliseconds per GEMM. Warm up before
collecting samples. Upload/FP16 conversion, download, initialization, and host
wall time are separate costs. The GPU-event interval can include submission
gaps for small workloads; it is not a guarantee of isolated kernel time.

The conversion helper can be checked without a GPU:

```sh
c++ -std=c++17 -O2 native/fp16_test.cpp -o /tmp/jade-fp16-test
/tmp/jade-fp16-test
```

This test covers all finite FP16 values and important rounding boundaries. It
does not validate the native SDK or any GPU behavior.

## Primary-source basis

Verified on 2026-11-03:

- [Iluvatar Tiangai 300 product page](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg300).
- [Iluvatar IxRT's CoreX library discovery](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/cmake/FindCuda.cmake)
  defaults to `/usr/local/corex` and links `cudart`.
- [Iluvatar stream and event helper](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/samples/common/include/memory_utils.h)
  uses `cuda_runtime.h`, streams, and events from that SDK.
- [Iluvatar's plugin build](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/samples/plugins/CMakeLists.txt)
  is a reference for the vendor's CUDA-compatible library integration.

API names shared with NVIDIA are source-interface compatibility, not proof that
every NVIDIA feature exists in a particular CoreX release. Exact SDK/driver,
GemmEx dtype support, and behavior on Tiangai 300 remain target validation work.
