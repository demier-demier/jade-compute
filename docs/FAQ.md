# Frequently asked questions

## What is Jade Compute?

A Rust compiler/runtime prototype with a small expression language and checked
GPU operation IR. It generates native Metal and CoreX source, manages GPU
resources and executes ordered operations. The current code and its limits are
described in the [native runtime guide](NATIVE_RUNTIME.md).

## Does it replace CUDA?

Not today. Jade implements its own restricted interface. It does not parse
arbitrary CUDA C++, translate PTX/SASS, load NVIDIA binaries or reproduce the
CUDA runtime ABI. Libraries, distributed communication and tooling represent
additional work in the [coverage map](CUDA_RECONSTRUCTION.md).

## Is it faster than NVIDIA or CUDA?

That has not been demonstrated. The historical alpha 1 native MLP experiment compares
batched submission with synchronization after each operation, on the same Mac
and with the same kernels. Its final 5.800× ratio applies to that baseline only;
the earlier 7.813× run also shows measurement variation. See the
[raw results and measurement scope](../reports/NATIVE_RESULTS.md). The later
[Metal experiments](METAL_ADVANCES.md) include an Apple MPS GEMM baseline;
their [results](../reports/METAL_ADVANCES.md) remain device- and workload-specific.

## Has it actually run on a GPU?

Yes: the direct Metal backend ran on an Apple M4 Pro. The alpha 1 record contains
nineteen numerical cases covering 6,456 values against independent FP64 CPU
references, plus regular and irregular synthetic MLP blocks. The expansion adds
SIMD-group GEMM, forward self-attention and native MPS, including mixed MPS/shader
batches; see its [validation record](../reports/METAL_ADVANCES.md).
CoreX has not been compiled against
the vendor SDK or executed on an Iluvatar GPU. Source emission is a separate
result from target validation.

## Is the GPU code written in Rust?

Rust implements the frontend and host runtime. Jade expressions and operation
descriptions generate MSL for Metal and C++ device source for CoreX's compiler.
The earlier portable path uses WGSL. This is not a general Rust-to-GPU compiler,
and changing the host language alone does not make GPU math faster.

## Can it run a trained LLM or train a model?

No complete model is integrated. There is a six-operation residual MLP with
synthetic weights and a separate small FP32 forward self-attention implementation.
Attention supports contiguous `[H,S,D]` tensors, including causal mode, within
the [documented limits](METAL_ADVANCES.md#forward-self-attention). There is no
tokenizer, checkpoint loader, KV cache, token server, backward pass, autograd,
optimizer or distributed training implementation. These require separate
engineering and model-level validation.

## Which Chinese GPUs are supported?

No Chinese GPU is validated by this repository yet. Iluvatar TianGai 300 is the
initial intended target. The CoreX path needs the matching Linux environment,
SDK, driver and vendor-confirmed compiler architecture. Other manufacturers
need their own backends and qualification; see the [hardware notes](HARDWARE.md).

## Why develop Metal and CoreX in parallel?

Metal provides an accessible native execution target for testing the frontend,
resource ownership, numerical behavior and scheduling. CoreX is a separate
target for integration work. A passing Mac test does not establish Iluvatar
compatibility, hardware behavior or performance.

## Why publish a prototype with commercial ambitions?

The repository lets people inspect the design, reproduce reported observations
where permitted, identify problems and discuss useful workloads. The intended
commercial direction is qualified model integration and optimization for
specific hardware. Product availability and commercial terms are not yet set.
See [commercial direction](COMMERCIAL.md).

## Is it open source, and may I use it commercially?

No open-source license has been granted for the project code. Public GitHub
visibility is not a general license for integration, redistribution or
commercial reuse. GitHub provides viewing/forking rights through its service;
dependencies have their own licenses. Contact the maintainer about additional
permissions. [GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

## How can I help or discuss a project?

Useful feedback includes reproducible bugs, reference outputs, hardware/SDK
details and concrete model workloads. Discuss code contributions and applicable
terms before investing in a substantial change. The [roadmap](ROADMAP.md)
identifies work with measurable completion criteria. For possible commercial
evaluation, licensing or partnership discussions, contact
[Bernardo Demier on LinkedIn](https://www.linkedin.com/in/bernardo-demier-3176001ab/).
