# Jade Compute: proposed vendor pilot

Status: product hypothesis and engineering plan, prepared on 2026-11-03. No vendor has been contacted and no commercial commitment has been made.

The proposed product is a performance optimization SDK for selected AI workloads on Chinese accelerators. It would integrate with a manufacturer's existing inference stack and supply specialized kernels, measured dispatch policies, and reproducible deployment artifacts. Rust would manage resources, execution, tuning records, and packaging; GPU speed must come from better computation and data movement.

The first implementation was a GEMM prototype with portable FP32 kernels, benchmarking, validation, and a vendor-BLAS CoreX bridge. The repository has since gained its own operation/expression IR, direct Metal execution, and a six-operation synthetic residual MLP validated locally. A separate CoreX runtime for generated kernels is prepared but has not been compiled with the SDK or run on Iluvatar hardware. The project does not yet run this trained-model serving pilot, establish Tiangai 300 performance, or replace the CUDA ecosystem. Implementation and hardware-validation status remain distinct.

## Existing competition and potential differentiation

| Existing project | Relevant capability | Consequence for Jade |
| --- | --- | --- |
| [CubeCL](https://github.com/tracel-ai/cubecl) and [Burn](https://burn.dev/docs/burn/backend/fusion/index.html) | Rust kernels, multiple backends, tuning caches, and fusion | Rust and portability alone are insufficient differentiation. |
| [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | Specialized kernels, tiling, fusion, and autotuning | A fast matrix multiplication example is an engineering milestone. |
| [Apache TVM](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) | Hardware-measured schedule search, cost models, and persistent tuning records | Generic autotuning is already established. |
| [FlagOS](https://docs.flagos.io/en/latest/overview.html) | Compiler, operators, communication, and framework integrations | Cross-vendor AI software is an existing competitive category. |
| [Iluvatar DeepSpark](https://github.com/Deep-Spark/DeepSpark/blob/master/README_en.md) and [MUSA](https://www.mthreads.com/product/MUSASDK) | Vendor libraries, inference tooling, and framework/compiler integrations | The vendor's strongest supported implementation must be the baseline. |

Potential differentiation is a repeatable improvement on a valuable workload: original specialized kernels and weight layouts, selection policies derived from our measurements, and compatibility/regression evidence across SDK releases. These are proposed assets to develop, not established novelty or exclusivity claims. Existing compilers and libraries should be reused where appropriate.

The initial market hypothesis is latency-sensitive model serving where a hardware manufacturer needs better application results and easier deployment for customers. This is qualitative positioning; customer demand and willingness to pay require validation.

## First workload and baseline

- Start with the dense [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), chosen as a reproducible baseline rather than a claim about the newest model. Pin the exact model, tokenizer, and configuration revisions.
- Use one Iluvatar device or a precisely recorded board/device topology, with a vendor-supported CoreX SDK and existing vLLM deployment. Confirm Tiangai 300 access and support before scheduling target-specific work.
- Fix BF16 where the baseline supports it reliably; otherwise fix FP16 for both variants. No precision reduction, quantization, pruning, or quality trade changes inside the first comparison.
- Exercise decode batches of 1, 4, and 16, input lengths of 2,048, 8,192, and 16,384 tokens, and 256 generated tokens. Record actual scheduler batch sizes separately from client concurrency.
- Freeze tokenizer/chat template, thinking mode, sampling parameters, prefix-cache policy, graph execution, tensor parallelism, and memory limits. Use non-thinking mode for the first latency workload.
- For controlled timing, enforce the recorded output length consistently. Run a separate natural-completion quality evaluation so forced token counts do not masquerade as model quality.
- First reproduce the best vendor-approved serving path; compare applicable FlagGems or other supported optimized paths as well. A naive kernel or unoptimized eager framework is not an adequate commercial baseline.

Profile before selecting an optimization. Attribute time and memory traffic to MLP projections/SwiGLU, normalization, RoPE, KV-cache operations, attention, and host scheduling. Select only bottlenecks with enough measured contribution to affect end-to-end results. Candidate work includes fusion, persistent weights/buffers, layout selection, or shape-specific dispatch. Feed the resulting operators into the existing vLLM path.

## Measurement and acceptance

Use paired baseline/candidate measurements with alternating order and the same workload trace. Warm up compilation, tuning, graph capture, and caches before measurement; record the warmup rule and cold-start cost separately. Collect a named, stable 30-minute serving window per evaluated configuration and repeat it three times with independent process starts. Report failures and thermal/clock instability rather than silently dropping runs.

Define goodput before testing as successfully completed requests per second satisfying the agreed latency limits; also report their generated tokens per second. Report p50/p95/p99 time to first token, inter-token latency and full-request latency, raw throughput, error rate, and peak device memory. Goodput must not improve by excluding slow requests from latency statistics.

Before optimization begins, record the exact p95 latency limits, evaluation dataset and metric, numerical tolerances, and allowed model-quality difference. Freeze the primary workload mix and holdout cases. Use independent operator references, full checks on manageable irregular shapes, and explicitly labelled sampling for large tensors.

**Proposed commercial gate: at least 20% higher end-to-end goodput at the same p95 latency limits and within the prespecified quality budget, repeatable across all three runs of the primary mix.** This is a target for deciding whether to proceed, not a performance promise. Require no goodput regression greater than 5% on the agreed high-traffic holdout cases, no increased serving error rate, and adherence to the agreed memory cap. Report uncertainty and all workload cells, including losses.

Stop or revise the approach if profiling leaves too little opportunity, supported SDK access is unavailable, numerical/quality gates fail, or the integrated candidate cannot improve the strongest vendor baseline. Isolated kernel speedups do not override a failed application gate.

## Four engineering stages

The estimate is **8–12 engineering weeks**, conditional on usable GPU/SDK access from the start and competent GPU/kernel engineers working with Rust and model-serving integration. Discovery, driver issues, vendor turnaround, and second-device access can change scope and schedule; this is not a delivery commitment.

| Stage | Estimate | Deliverable and exit condition |
| --- | --- | --- |
| 1. Bring-up and profiling | 2 weeks | Reproduce the vendor baseline, freeze the benchmark/quality contract, and identify a bottleneck with an achievable application-level opportunity. |
| 2. Specialized optimization | 2–4 weeks | Implement and validate the selected kernels/layouts/dispatch policies; measure against the strongest native alternatives on tuning and holdout shapes. |
| 3. Serving integration | 2–3 weeks | Integrate with existing vLLM, run the repeated stable windows, and evaluate the 20% application gate plus regression limits. Stop if the gate cannot be justified. |
| 4. Second-vendor validation | 2–3 weeks | Port the same narrow workload to MetaX/MACA or Moore Threads/MUSA; measure against that device's own best baseline and document adaptation effort and retained gains. |

Second-vendor access may be arranged during earlier stages. Support must be demonstrated on an identified device/SDK pair; a common interface does not establish portability. A second-platform result supports the cross-vendor proposition only if correctness and application-level benefits survive the port.

## Benchmark manifest

Every result must include or reference:

- Run ID, UTC timestamp, source commit, local modifications, compiler/build flags, executable/container hashes, and command lines.
- GPU model, device count/topology, device/board identifiers, memory, CPU/RAM, OS/kernel, driver, SDK, libraries, framework fork/version, and serving configuration.
- Model/tokenizer/configuration revisions and hashes; precision/accumulation settings; kernel source/binary hashes; tuning-space version, budget, cache key and selected candidate.
- Dataset/workload hashes, seeds, prompt/output token counts, thinking/sampling settings, offered-load trace, concurrency, observed batch sizes, cache state and quality-evaluation definition.
- Baseline identity, p95 limits, quality budget, memory cap, holdout mix, warmup procedure, measurement window, run order, and excluded-run reasons.
- Per-request latency/status records, device timing traces, raw samples and aggregation method, peak memory, clock/temperature evidence where available, and measured power with its sampling method if reported.
- Correctness results, quality results, confidence/variability analysis, regressions, unsupported cases, cold-start/tuning cost, and exact reproduction instructions.

Device-event timings and resident-buffer GEMM throughput remain separate from client-observed serving metrics. A comparison with an NVIDIA system would require its own frozen hardware/software/workload baseline; success against a Chinese vendor's existing stack does not establish superiority over NVIDIA.

## Proposed partner engagement

Prepare a bounded paid proof of concept or co-development proposal after the technical scope is reviewable. Request a GPU loan or remote lab access, the matching SDK/documentation, a vendor SDK engineer, a reproducible baseline, and one agreed customer workload. A hardware purchase is not the prerequisite proposed here.

The offer would be a measured improvement package with integration and regression evidence, with acceptance tied to the agreed application metrics. This document does not send a proposal, contact any company, choose a project license, publish the repository, or claim a partnership.
