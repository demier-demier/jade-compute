# Commercial direction

Jade Compute is an early GPU compiler and runtime prototype. The ambition is a
commercial software stack that helps hardware makers and model teams run
selected AI workloads efficiently across supported GPU architectures.

The public repository makes the engineering approach and measured evidence
reviewable. A production product, supported hardware matrix, pricing and support
commitments have **not** been announced.

## What can be reviewed today

| Area | Current evidence |
| --- | --- |
| Compiler | Rust expression frontend and checked FP32 operation IR generate MSL and CoreX C++. |
| Runtime | Owned GPU buffers, validation, compilation, pipeline reuse and ordered execution, including mixed MPS GEMM/shader batches. |
| Apple Metal | Alpha 1 validated 19 numerical cases and a synthetic residual MLP on an Apple M4 Pro. The expansion implements FP32 SIMD-group GEMM, native MPS and small forward self-attention; its evidence is recorded separately. |
| Iluvatar CoreX | Adapter and generated source prepared; no SDK build or device execution verified. |
| Performance | Local measurements with raw samples and stated baselines; no demonstrated advantage over NVIDIA or a tuned vendor inference stack. |

Read the [native runtime guide](NATIVE_RUNTIME.md),
[alpha 1 validation history](../reports/NATIVE_RESULTS.md),
[Metal expansion](METAL_ADVANCES.md), its
[validation and measurements](../reports/METAL_ADVANCES.md), and
[architecture coverage map](CUDA_RECONSTRUCTION.md).

## Intended product scope

The commercial direction is model integration and measurable execution
improvements on specific hardware. Candidate capabilities include:

- Native kernel specialization and fusion for agreed model workloads.
- Runtime integration, memory reuse and execution planning.
- Correctness and model-quality checks alongside performance tuning.
- Qualification against named board, driver, SDK and compiler versions.
- Ongoing support for an explicitly agreed hardware/software matrix.

These are development goals. The implemented forward-attention subset has no
KV cache, dropout or backward pass. Full CUDA compatibility, TensorRT/cuDNN
replacement, arbitrary model support, training at scale and deployment support
are outside today's validated scope.

## A possible evaluation project

A scoped, potentially paid proof of concept could be discussed with a GPU
manufacturer, systems integrator or model deployment team. Scope, access,
deliverables, commercial terms and availability would need a separate agreement.

| Step | Concrete output |
| --- | --- |
| Define the workload | Exact model/checkpoint, task, precision, shapes, batch/context and quality target. |
| Establish the environment | Named GPU, Linux/OS image, SDK/compiler/driver versions and accessible baseline. |
| Reproduce the baseline | Raw correctness and performance records using the vendor's relevant optimized stack. |
| Optimize selected bottlenecks | Reviewed kernels/runtime changes and repeatable evaluation scripts. |
| Assess the result | Same-quality comparisons, regressions, limitations and a decision on further work. |

Success criteria would be agreed before optimization. Inference comparisons
should include throughput, first-token and tail latency, quality, memory and
power where measurable. Training work additionally needs convergence and time
to a fixed quality target. No percentage improvement is promised in advance.

## Public visibility and licensing

An open-source license has **not** been granted for Jade's project code; licensing
terms remain under consideration. Publishing a repository permits the viewing
and forking provided by GitHub's terms, and does not by itself grant general
commercial reuse rights. Dependencies retain their own licenses.
[GitHub's licensing explanation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

Discuss evaluation, integration, redistribution and commercial permissions with
the maintainer before relying on permissions not explicitly granted.

## Contact

Contact [Bernardo Demier on LinkedIn](https://www.linkedin.com/in/bernardo-demier-3176001ab/)
to discuss evaluation, licensing or a possible partnership. A useful introduction
identifies the hardware/SDK, model workload, bottleneck and desired outcome.
No vendor partnership or endorsement is claimed.

Continue with the [roadmap](ROADMAP.md) or [frequently asked questions](FAQ.md).
