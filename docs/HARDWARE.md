# Hardware targets and evidence

Sources checked: **2026-11-03**. The links below establish product identities,
software stacks, and integration options. They do not establish Jade performance
or successful execution on those devices. No Chinese accelerator was available
for local validation.

## Primary target: Iluvatar TianGai 300

The official product is **TianGai 300 (天垓300)**:
[Iluvatar product page](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg300).
Its exact board identifier, minimum compatible SDK/driver, compiler architecture
flag, and VRAM configuration remain **unverified**. Do not infer a `BI-V300`
alias, reuse an older board's architecture flag, or assume Vulkan support.

Iluvatar's [software stack page](https://www.iluvatar.com/software?fullCode=cpjs-rj-rjz)
describes C/C++ interfaces, libraries, compilers, runtime, drivers, and Linux
support. Its [DeepSpark overview](https://github.com/Deep-Spark/DeepSpark/blob/master/README_en.md)
identifies the stack as IXUCA and points to the vendor developer portal.
CoreX is also used in SDK paths and packages.

The vendor's public source provides concrete integration evidence:

- [Compiler discovery](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/cmake/FindCompiler.cmake)
  checks `/usr/local/corex/bin/clang++`.
- [Plugin build](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/samples/plugins/CMakeLists.txt)
  links `cublasLt`, `cudart`, and `ixrt` from the vendor environment.
- [Memory utilities](https://github.com/Deep-Spark/iluvatar-corex-ixrt/blob/master/samples/common/include/data_utils.h)
  use `cudaMalloc` and `cudaMallocHost`.
- The [official quickstart](https://github.com/Deep-Spark/DeepSpark/blob/master/docs/quickstart.md)
  directs developers to their application engineer for required headers.
  Its SDK 4.3 example covers older devices and is **not** evidence that SDK 4.3
  supports TianGai 300.

Use a C ABI bridge compiled against the installed CoreX headers. This keeps
vendor structs, handles, enum values, and API-version differences inside the
vendor build. CUDA-style symbol names here refer to the vendor's compatible
interfaces; they do not imply NVIDIA binaries execute on this hardware.

## Other Chinese accelerator families

These are integration candidates, not a speed ranking or an availability list.
Only the CoreX bridge and portable test path are part of this initial project;
the other native backends below are not implemented.

| Family | Native software route | Verified source and project implication |
| --- | --- | --- |
| Moore Threads MTT S5000 / PH100 | MUSA | [Product documentation](https://docs.mthreads.com/driver-linux-server/driver-linux-server-doc-online/MTT_S5000/introduction/) and [Linux/MUSA 5.2 installation](https://docs.mthreads.com/driver-linux-server/driver-linux-server-doc-online/MTT_S5000/install_guide/). A separate Rust-to-MUSA bridge can use the documented [Driver API](https://docs.mthreads.com/musa-sdk/version-5.2.0/libraries/core_api/driver_api_reference/). |
| MetaX C600 | MXMACA | [Official product](https://www.metax-tech.com/en/goods/prod.html?cid=107&id=68) and [mcBLAS API](https://developer.metax-tech.com/api/client/document/preview/1080/index.html). Native BLAS, streams, and precision controls need a separate backend. |
| Biren 166M / 166L / 166C | BIRENSUPA | [Official product listing](https://www.birentech.com/), [software platform](https://www.birentech.com/product/software/birensupa/), and [documentation index](https://developer.birentech.com/Document_search.html), which lists suBLAS and suCCL. Obtain the matching SDK before implementing its ABI. |
| Hygon DCU | DTK / DAS / DAP | The [issuer's 2026 report](https://static.cninfo.com.cn/finalpage/2026-08-14/1225472509.PDF) identifies these software layers. Confirm the exact board and toolkit before selecting runtime bindings. |
| Huawei Ascend | CANN / AscendCL | An **NPU** route, not a generic Vulkan GPU target. The [official allocation API](https://www.hiascend.com/doc_center/source/en/canncommercial/601/inferapplicationdev/aclcppdevg/aclcppdevg_03_0072.html) documents device memory management. Ascend kernels and operator libraries require their own backend. |
| Cambricon | Neuware / BANG C | An **MLU** route. The [official SDK portal](https://developer.cambricon.com/index/sdk/index/classid/14.html) describes the compiler/runtime stack and SDK access process. Confirm current product support rather than treating older examples as a universal compatibility list. |

## Native execution and portable validation

The performance-oriented architecture is Rust orchestration plus native vendor
libraries or kernels: retain weights and intermediate tensors on the device,
reuse allocations and execution state, and optimize the measured bottleneck.
Rust can manage ownership and scheduling while a narrow C ABI bridge integrates
the vendor SDK. Wrapping BLAS in Rust alone does **not** make BLAS faster.

The portable path uses Rust/wgpu with WGSL compute shaders. wgpu documents
[Metal and Vulkan backends](https://docs.rs/wgpu/latest/wgpu/struct.Backends.html),
but it does not automatically supply MUSA, MACA, CoreX, CANN, or Neuware support.
Vulkan execution requires a working driver exposing the required compute
capabilities on the exact target. Mac/Metal results validate that local path;
they do not validate TianGai 300 or any other Chinese device.

Jade is an initial compute runtime and benchmark project, **not a full CUDA
replacement**, CUDA binary translator, complete training framework, or complete
LLM serving engine. Native execution and performance remain unconfirmed until
the matching SDK and target hardware run the validation workload.

## How to establish an AI performance improvement

Compare the native vendor baseline and Jade on the same board, driver, SDK,
model/checkpoint, tokenizer, input data, precision, quantization, batch/concurrency,
context lengths, output lengths, and accuracy criteria. Record power limits,
cooling, clocks, CPU, memory, and interconnect. Keep all baseline settings.

- **Inference:** report prompt/prefill and generated/decode tokens per second
  separately, TTFT (time to first token), p50/p95 latency, and model quality.
  State whether queueing and transfers are included. Record peak VRAM, measured
  power, and energy per output token when telemetry permits.
- **Training:** report steps per second, samples or tokens per second, global
  batch and accumulation, optimizer, precision, checkpointing, peak VRAM, and
  measured power. Compare time to the same quality/convergence target; faster
  steps with degraded training quality are not an equivalent result.
- **Microbenchmarks:** validate outputs first; record shapes and numerical
  tolerance, warm up compilation and caches, repeat measurements, and separate
  device execution from allocation, upload, submission, synchronization, and
  download. A faster GEMM does not establish faster end-to-end inference.

Publish raw measurements and environment metadata. A vendor marketing result,
local Mac result, or reduced-precision result without quality checks cannot
substitute for this target-specific comparison.

## Documented workloads and commercial focus

The strongest documented focus across these vendors is data-center AI training
and inference: language models, multimodal generation, recommendation, and
industry applications. This is an inference from the sources below, **not a
measured market-share claim**. Scientific computing, video analysis, rendering,
and cloud desktops also matter, with different product lines and requirements.

| Vendor | Concrete workload evidence | Evidence level and source date |
| --- | --- | --- |
| Iluvatar | Infini-AI deployment on a 100-card Zhikai 100 inference cluster; the platform lists Qwen2.5, Llama3.1, Stable Diffusion, and CogVideoX. | [Vendor-reported MaaS deployment, 2024-12-08](https://www.iluvatar.com/newsDetails?code=tszxlhwwxqwczkGPUbktljqcsykpzcdc&topicId=495). This concerns Zhikai 100, not TianGai 300. |
| Moore Threads | Wuxi/Huishan center with 2,000 S5000 GPUs; cloud training and industrial models deployed toward edge systems. | [Vendor deployment report, 2026-10-16](https://mthreads.com/news/353); its 20+ customers include both testing and production use. Full utilization is the vendor's report, not an independent measurement. |
| MetaX | siiRL 2.0 reinforcement-learning training on its cluster; dense and MoE model workloads. | [Vendor-reported framework integration and validation, 2025-09-23](https://www.metax-tech.com/en/ndetail/12522.html), not an independently reproduced performance comparison. |
| Biren | SGLang inference of SenseNova U1 multimodal understanding/generation on 166-series GPUs. | [Vendor model validation, 2026-04-29](https://www.birentech.com/news/otr0bf5evzdwwivrjt6h3322/); this is not evidence of a named customer's production volume. |
| Hygon | LLM training/inference and scientific AI; DTK/DAS/DAP software layers. | [Issuer report, 2026-08-14](https://static.cninfo.com.cn/finalpage/2026-08-14/1225472509.PDF) and [OneScience catalog](https://developer.sourcefind.cn/onescience) list weather, biomolecular, materials, and fluid-simulation workloads. Catalog availability does not prove customer deployment. |
| Huawei Ascend | WPS document-model pretraining and office inference; 360 multimodal search; ERNIE 4.5 post-training. | [Vendor customer case page](https://www.hiascend.com/industries/internet), accessed 2026-11-03; no publication date shown. Performance claims were not independently reproduced. |
| Cambricon | Cloud AI training/inference, with reported commercial deployments in finance and internet sectors; vision, speech, language, and recommendation software. | [Issuer's 2026 half-year report](https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF), published 2026-08-08. Cloud products account for approximately 99.98% of its reported revenue (p. 133); this says nothing about the training/inference split or the wider market. |

For non-LLM use, Moore Threads documents a
[scientific suite](https://www.mthreads.com/product/MUSAScientificComputingSuite)
with GROMACS, SPONGE, RELION, and DSDP adaptation as of June 2026, and a separate
[S3000 cloud-desktop product](https://docs.mthreads.com/s3000/s3000-doc-online/introduction/).
These are product/software support statements, not Jade compatibility tests.

There is a specific demand signal for inference optimization: MetaX's
[2026 challenge announcement](https://developer.metax-tech.com/forum/u/yang-yang/739/posts/)
(June 30 post) requests Fused MoE GEMM, MLA, sparse attention, and inference-library
optimization on MXMACA, with reproducible correctness and performance evidence.
It supports investigating those bottlenecks; it does not establish an open
purchase opportunity or an exclusive technical gap.

Portability alone is already addressed by
[CubeCL](https://github.com/tracel-ai/cubecl),
[FlagTree](https://github.com/flagos-ai/FlagTree/blob/main/README.md),
[FlagGems](https://github.com/flagos-ai/FlagGems/releases), and
[DeepLink DIPU](https://github.com/DeepLink-org/deeplink.framework/blob/main/dipu/README.md).
A proposed Jade differentiation must therefore be demonstrated on a narrower
workload: better end-to-end throughput or latency, memory use, or porting effort
at equivalent quality. Rust is an implementation choice, not evidence of that
advantage. Vendor interviews and target-device tests must validate the business
hypothesis before presenting it as a product benefit.
