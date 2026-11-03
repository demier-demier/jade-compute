# Reconstructing CUDA's responsibilities as Jade

Evidence reviewed on **2026-11-03**. This is a functional architecture study and
an implementation roadmap for an independent compute stack. It is **not** a
claim that CUDA has been fully reconstructed, that NVIDIA binaries run on Jade,
or that any Chinese GPU has passed the local tests.

The implementation direction is Rust host runtime + an explicit kernel
description + native backends for Apple Metal and Iluvatar CoreX. The initial
scope is a small set of numerical operations. Vendor drivers and compilers
remain dependencies. Rust ownership can improve resource management; changing
the host language alone does not accelerate GPU instructions.

## Evidence rules

| Label | Meaning |
| --- | --- |
| **Documented** | A linked primary source describes an interface or behavior; it has not necessarily been exercised here. |
| **Observed: source** | The repository contains an implementation that was inspected; this does not establish a successful target build or run. |
| **Observed: execution** | A local run has a recorded environment, inputs, correctness result, and output artifact. |
| **Inferred / planned** | Our proposed mapping or engineering decision; it is not vendor compatibility evidence. |
| **Unknown** | Required facts or artifacts are unavailable or unverified. |

Public API and compiler documentation supports reconstruction of responsibilities
and a new design. No NVIDIA or CoreX driver binary has been decompiled in this
work. No proprietary source, complete driver specification, or TianGai 300 ISA
has been recovered. A future binary investigation needs a concrete artifact,
version/hash, available inspection tools, and separately recorded findings.

## What CUDA contains, and what Jade must decide

Rows describe intended responsibilities, not interchangeable APIs. Every Jade
mapping below is **inferred / planned** unless identified as observed later.

| Layer | Documented CUDA responsibility | Jade / native Metal direction | CoreX direction and unresolved work |
| --- | --- | --- | --- |
| Device, driver, context | Driver API exposes device/context/module management; Runtime provides a higher-level programming interface. [Driver](https://docs.nvidia.com/cuda/cuda-driver-api/index.html), [Runtime](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html) | Own device object, capability report, explicit resource ownership and errors; call Apple's driver through Metal. | Load a versioned Jade C ABI backed by CoreX. Do not replace the Linux kernel driver. Verify SDK/device/context rules. |
| Allocation and transfer | Device and host allocation, copies, managed memory, arrays and specialized allocation APIs. [Memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html) | Buffers with lengths, access rules, resource lifetime and explicit upload/download. Choose shared/private storage deliberately. | Vendor allocation/copy calls; validate pinned memory, alignment and asynchronous lifetime rules from the installed SDK. |
| Unified / virtual memory | CUDA includes unified-memory and virtual-address mechanisms. [Guide contents](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html) | Apple shared system memory is not an implementation of CUDA managed-memory behavior. No transparent pointer migration promised. | Page faulting, oversubscription and virtual memory support remain unknown for this board/SDK. |
| Stream and event ordering | Ordered work queues, event dependencies, synchronization, capture and asynchronous error behavior. [Streams](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html) | Ordered dispatch sequence first; public completion is explicit. Future queues/events must define lifetime and visibility semantics. | Current bridge uses vendor stream/events internally. Cross-stream behavior needs native conformance tests. |
| Kernel launch | Grid/block dimensions, arguments and resource limits control kernel execution. [SIMT kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | Translate a validated operation specification to MSL, compile a pipeline, bind buffers, dispatch threadgroups. | Emit a restricted vendor-language kernel; compile with the installed CoreX compiler and verified target flags. |
| SIMT and subgroup | CUDA execution groups threads into 32-thread warps; divergent execution and synchronization have specified rules. [Execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#simt-execution-model) | Query Metal `threadExecutionWidth`; avoid a global warp-width constant and implicit lockstep assumptions. | Query/document actual subgroup width and intrinsic behavior; never infer them from CUDA-like symbol names. |
| Memory ordering | Address spaces, atomics, thread scopes and synchronization semantics affect correctness. [Memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html) | Define only the supported scopes; use explicit threadgroup barriers and resource hazards. Reject unsupported atomic/scope combinations. | Establish vendor memory model before lowering advanced synchronization. Host Rust safety cannot prove shader race freedom. |
| Graph execution | CUDA graphs have nodes, dependencies, instantiation, launch and update rules. [Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) | Start with ordered replay of a dispatch list; later add a validated dependency graph and reusable execution plans. A dispatch list is not CUDA Graph API compatibility. | Use stream replay first; native graph availability and update/capture semantics remain to be checked. |
| Compiler frontend | `nvcc` separates host/device compilation and packages device programs. [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) | Rust builds a deliberately limited typed kernel specification; backend emitters produce source for the vendor compiler. No arbitrary `.cu` parser. | CoreX compiler invocation and headers are SDK-specific. A source emitter alone is not a working toolchain. |
| PTX / native machine code | PTX is a virtual ISA; native binaries and SASS inspection are distinct layers. [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [Binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | Emit MSL and let Metal generate device code. No PTX interpreter, SASS translator, cubin loader, or binary ABI emulation. | Let CoreX compile source. TianGai 300 native ISA and a complete device-code ABI are unknown here. |
| Dense math / tensor units | cuBLAS and cuBLASLt provide tuned linear algebra and algorithm selection. [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) | Own kernel candidates plus a future native-library baseline; capability-aware matrix intrinsics later. | Existing bridge delegates GEMM to vendor BLAS. Wrapping BLAS alone cannot establish a performance advantage over it. |
| Neural-network operations | cuDNN supplies neural-network operators, graph execution choices and fused operations. [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/index.html) | Initial RMSNorm, softmax and MLP subset is detailed below; attention, quantization and backward kernels are future extensions. | Compare native libraries and selected custom kernels on actual models; no cuDNN replacement claimed. |
| Multi-GPU / distributed | NCCL supplies collective communication and communicator/ordering/failure semantics. [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) | Future collective interface, topology discovery and correctness tests; not implemented by a single-device runtime. | Requires multiple boards, interconnect information and vendor collective support; unknown today. |
| Diagnostics and profiling | CUDA tooling includes debuggers, timeline/kernel profilers and memory/race/synchronization checks. [Toolkit](https://docs.nvidia.com/cuda/index.html), [Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | Error propagation, reference validation, raw timing records, shader source retention; integrate Metal capture/profiling separately. | Need vendor tracing/profiling tools and supported counters. Reject unavailable/invalid GPU timings rather than manufacture them. |
| Interoperability / advanced features | CUDA also covers graphics/external resources, IPC, cooperative execution, dynamic parallelism and specialized hardware features. [Feature inventory](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html) | Explicit future capability extensions; not implied by GEMM or MLP support. | Per-feature SDK and hardware validation; no blanket compatibility declaration. |

## Native Metal is a real target, with its own semantics

**Documented:** Metal has command queues, command buffers, compute encoders,
pipelines and dispatch operations. Jade can express the supported workload
directly through these interfaces. [GPU calculation flow](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)

**Documented:** Apple GPUs share physical system memory, but shared and private
resources have different CPU access permissions. Shared storage still requires
correct ordering of CPU and GPU accesses. This is not permission to read an
unfinished GPU write. [Apple storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus), [Shared resources](https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared)

**Documented:** Metal's SIMD width is a queried pipeline property. Source
translation must preserve synchronization and workgroup behavior rather than
substitute CUDA's constant warp width. [Threads/threadgroups](https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups), [Apple silicon porting](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)

**Documented:** Metal source becomes Metal IR and then device-specific code.
Jade's operation specification is a separate, much smaller representation.
It is neither Metal IR nor PTX. [Metal libraries](https://developer.apple.com/documentation/metal/metal-libraries)

**Documented:** Metal events, fences and barriers have different scopes.
Metal 4 queue/resource synchronization differs from earlier command queues;
the backend must state which API family it implements. A newer OS does not
mean that the code uses Metal 4. [Resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)

**Documented:** Xcode's Metal debugger can inspect dispatches, dependencies,
timelines and shader performance. This is a future profiling workflow, not
an assertion that such a capture has already been inspected. [Metal debugger](https://developer.apple.com/documentation/xcode/metal-debugger)

## CoreX: documented starting point, target verification outstanding

The target's official product name is **TianGai 300 / 天垓300**. The spelling
"Tiangai 300" is treated as this product; **BI-V300**, installed VRAM, minimum
driver/SDK and compiler architecture flag have not been verified for the user's
board. No Vulkan support is assumed. [Official product](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg300)

**Documented:** the official ecosystem uses a Linux SDK/compiler and public
samples containing CUDA-style runtime/library interfaces. Samples show
allocation, streams and events, and link vendor libraries. These are a basis
for an adapter, not proof that every CUDA API or ABI is implemented. [CoreX stack](https://www.iluvatar.com/software?fullCode=cpjs-rj-rjz), [Runtime sample](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/samples/common/include/memory_utils.h), [Compiler setup](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/cmake/FindCompiler.cmake), [Library linkage](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/samples/plugins/CMakeLists.txt)

**Documented:** DeepSpark's quickstart says required headers should be obtained
through the application engineer. Its SDK 4.3.0 installation example and older
board IDs do not establish TianGai 300 support. [SDK quickstart](https://github.com/Deep-Spark/DeepSpark/blob/master/docs/quickstart.md)

**Unknown / blocked target verification:** this Mac has neither an Iluvatar GPU
nor the CoreX SDK. A Linux build, native loading, device execution, numerical
validation, profiling and speed comparisons require that environment. Code
generation and a locally tested ABI loader do not remove this dependency.

## Implemented native architecture

**Observed: source:** the new [`Runtime`](../src/runtime/mod.rs) trait defines
`info`, `alloc`, `upload`, `download`, `compile` and `execute`. Backends own their
buffers and compiled kernels. `execute` accepts ordered `Dispatch` records and
a repeat count, then waits for completion before returning host/GPU timing and
dispatch counts. Invalid sizes, incompatible ownership and prohibited writable
aliasing are rejected. This is not an asynchronous CUDA-compatible API.

**Observed: source:** [`KernelSpec` and `Expr`](../src/runtime/ir.rs) implement a
checked FP32 operation representation. Elementwise expressions contain inputs,
finite constants, addition, multiplication, maximum, negation and exponential.
The frontend validates input references and expression complexity. Additional
operations are AXPY, addition, naive/register-tiled GEMM, RMSNorm, SwiGLU and
row softmax. Arbitrary pointers, indexing and control flow are outside this
initial language. The same specification emits MSL and CoreX C++.

**Observed: source:** [`MetalRuntime`](../src/runtime/metal.rs) owns native Metal
buffers and compute pipelines, compiles generated MSL, caches pipelines within
the runtime, and submits ordered batches using `MTLCommandQueue` and
`MTLCommandBuffer`, not the Metal 4 queue API. [`CorexRuntime`](../src/runtime/corex.rs)
uses a versioned native ABI, generated source and a configured vendor compiler.
It requires an explicit vendor-confirmed architecture; it does not guess a
TianGai 300 target. Its compilation cache is session-scoped, not a persistent
cross-driver optimization database. Both paths remain dependent on vendor
compilers and drivers.

The [`workload implementation`](../src/runtime/workloads.rs) composes a synthetic
residual MLP block with **six dispatches**:

```text
input -> RMSNorm -> gate GEMM --+
                  up GEMM ----+-> SwiGLU -> down GEMM -> residual add
input -------------------------------------------------------^
```

Weights are deterministically generated numbers, not a trained checkpoint.
The example has no attention, KV cache, tokenizer, token generation, backward
pass or optimizer. It compares ordered batched submission with the same six
operations submitted and synchronized individually. That comparison measures
submission/completion overhead; it is not a comparison against a tuned vendor
inference stack or CUDA Graphs.

The [`jade-runtime` CLI](../src/bin/jade-runtime.rs) exposes these paths:

```sh
cargo run --release --bin jade-runtime -- info
cargo run --release --bin jade-runtime -- verify --report reports/native-metal-verify.json
cargo run --release --bin jade-runtime -- mlp --batch 8 --width 128 --hidden 256 --report reports/native-metal-mlp.json
cargo run --release --bin jade-runtime -- emit --spec examples/kernels/fused_axpy_relu.json --target metal --output target/fused_axpy_relu.metal
cargo run --release --bin jade-runtime -- emit-suite --directory target/generated-kernels
```

The default execution backend is Metal. `emit` also accepts `--target corex`;
`emit-suite` exports JSON specifications and both source dialects. These two
commands generate source without executing it. Native CoreX execution selects
`--backend corex` and requires `--library` plus `--architecture`; `--compiler`,
`--sdk-root`, `--cache-dir` and `--device` configure the installed environment.

## Execution evidence

**Observed: execution:** the existing Rust/wgpu path ran on this Mac's Apple M4
Pro through Metal. Three WGSL GEMM kernels passed 21 kernel/shape checks covering
23,244 values against an FP64 CPU oracle. Performance reports compare only this
project's kernels. Invalid zero GPU timestamps were discarded. See
[recorded local results](../reports/LOCAL_RESULTS.md).

**Observed: source:** the initial CoreX GEMM bridge and Rust loader exist, with
FP32 and FP16-input paths. They have not been linked or executed against CoreX.
The host FP16 conversion checks do not validate device arithmetic. See
[bridge scope](COREX.md).

**Observed: execution — native Metal:** the new IR-to-MSL path executed directly
on the Apple M4 Pro. **19 numerical cases passed, covering 6,456 output values**
against independent FP64 CPU references, and five invalid-call checks were
rejected. A separate native Metal test exercised ordered operations, ownership
and alias checks. Nine ordinary Cargo tests and the separately invoked GPU
test passed; Clippy with warnings denied and rustfmt also passed. See
[native verification report](../reports/native-metal-verify.json) and
[final native validation record](../reports/NATIVE_RESULTS.md).

**Observed: execution — synthetic MLP:** batch 8, width 128 and hidden width 256
produced 1,024 checked outputs before and after timing; maximum absolute error
was approximately **2.01e-7**. Across seven alternating rounds of ten block
repeats, median synchronized host time per block was **0.1940625 ms** with
batched submission and **1.1256333 ms** with synchronization after every dispatch:
a **5.800× ratio against that submission baseline only**. Compilation,
allocation, upload and readback are excluded. The native command-buffer GPU
timestamps were valid and positive, unlike the earlier wgpu timestamps; they
cover the whole command buffer, not isolated kernel instructions. See
[native MLP samples and scope](../reports/native-metal-mlp.json).

The irregular MLP shape (batch 3, width 33, hidden width 65) also passed, with
99 output values and maximum absolute error **9.18883258e-08**. See
[irregular MLP report](../reports/native-metal-mlp-ragged.json). The initial
regular-shape run's **7.813×** ratio is preserved in the
[initial report](../reports/native-metal-mlp-initial.json); the final run's
**5.800×** ratio demonstrates variation between measurements.

These are local experiments with synthetic weights, not a universal speed claim. They do not
establish a trained-model speedup, superiority over an optimized Metal runtime,
CUDA Graphs, vendor BLAS or NVIDIA hardware, or performance on a Chinese GPU.

**Observed: host-side tests only — CoreX:** two tests passed for the platform
and compiler-cache guards. The CoreX SDK compilation, device kernel execution
and target numerical/performance results remain **unknown**. These tests cannot
substitute for access to the Linux/Iluvatar target.

## Delivery sequence and acceptance evidence

Increment 1 has the local execution evidence recorded above. The remaining
increments still require their stated implementation and acceptance evidence.

| Increment | Concrete engineering work | Required evidence before declaring it delivered |
| --- | --- | --- |
| 1. Native vertical slice | Rust operation specification, validators, MSL emitter, native Metal buffers/pipelines, ordered replay; small MLP forward example. | Execute on the Mac, compare all small-shape outputs with independent CPU references, record actual backend and timing scope. |
| 2. CoreX parity | Same supported operations through a versioned ABI, source emitter and SDK compiler; validate target flags. | Compile and run on TianGai 300 Linux; record board/driver/SDK/precision, unsupported cases and raw results. |
| 3. Useful AI operators | Optimize the initial operators and add fused bias/activation, attention and selected quantization; choose a specific partner model workload. | Numerical/task-quality gates plus end-to-end gains against that vendor's tuned baseline; account for compilation and transfers separately. |
| 4. Execution planning | Dependency graph, buffer reuse, persistent cache and correctness-gated per-device tuning. | Repeated-shape and mixed-load tests, cache invalidation across driver/compiler revisions, no hidden quality reduction. |
| 5. Training and scale | Backward operators, optimizer, deterministic/random behavior, collectives and failure handling. | Gradient checks, training convergence/time-to-quality, multi-GPU correctness and scaling efficiency. |
| 6. Wider migration surface | A chosen source/API compatibility subset with published exclusions. | Versioned conformance suite and actual applications; no "full CUDA" label from a few working kernels. |

The potential product advantage is a reproducible optimization system across
supported devices: validated transformations, hardware-specific schedules and
reliable execution. This is an **inferred product hypothesis**, not demonstrated
novelty or a measured vendor advantage. See [product plan](PRODUCT_PLAN.md).

Report inference tokens/s, time to first token, p95 latency, batch/context,
precision, quality, VRAM and power together. For training, report steps/s and
time to a fixed quality target. Include cold/warm compilation, host dispatch,
GPU execution and transfer costs separately. A microbenchmark speedup over
Jade's naive kernel is not a CUDA, model, or hardware superiority result.
