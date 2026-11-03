//! Native, synchronous Metal execution. No wgpu objects are used in this backend.
//! Shared allocations never escape as host references, and every public owner is
//! thread-confined so safe callers cannot race CPU accesses with GPU execution.
#![allow(unexpected_cfgs)] // objc 0.2 macros use the historical cargo-clippy cfg.

use super::{ir::KernelSpec, Buffer, Dispatch, Execution, Runtime, RuntimeInfo};
use anyhow::{anyhow, bail, ensure, Context, Result};
use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use objc::{
    msg_send,
    rc::{autoreleasepool, StrongPtr},
    runtime::{Class, Object, BOOL, NO, YES},
    sel, sel_impl,
};
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{c_char, CStr},
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

// Public Apple API, declared in MetalPerformanceShaders.h. Linking the umbrella
// framework also registers the MPSMatrix Objective-C classes used below.
#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {
    #[link_name = "MPSSupportsMTLDevice"]
    fn mps_supports_device(device: *mut metal::MTLDevice) -> BOOL;
}

// MPSCoreTypes.h declares MPSDataType as NS_ENUM(uint32_t, ...).
// MPSDataTypeFloat32 = MPSDataTypeFloatBit | 32; not an NSUInteger enum.
const MPS_DATA_TYPE_FLOAT32: u32 = 0x1000_0000 | 32;

fn fresh_id() -> Result<u64> {
    NEXT_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| anyhow!("Metal object identifiers exhausted"))
}

fn allocation_bytes(elements: usize) -> Result<usize> {
    ensure!(
        elements > 0,
        "Metal allocations must contain at least one element"
    );
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .context("Metal allocation size overflow")?;
    ensure!(
        bytes <= isize::MAX as usize,
        "Metal allocation exceeds host slice limits"
    );
    Ok(bytes)
}

// The pointer must be a live NSString (or nil) inside the current autorelease pool.
unsafe fn string_value(value: *mut Object) -> String {
    if value.is_null() {
        return String::new();
    }
    let utf8: *const c_char = msg_send![value, UTF8String];
    if utf8.is_null() {
        return String::new();
    }
    CStr::from_ptr(utf8).to_string_lossy().into_owned()
}

unsafe fn error_description(error: *mut Object) -> String {
    if error.is_null() {
        return "Metal supplied no error description".into();
    }
    let description: *mut Object = msg_send![error, localizedDescription];
    string_value(description)
}

fn native_string(value: &str) -> Result<StrongPtr> {
    let class = Class::get("NSString").context("NSString class is unavailable")?;
    // SAFETY: alloc/init creates an owned NSString from a borrowed byte slice;
    // initWithBytes copies those bytes, and StrongPtr releases its +1 reference.
    unsafe {
        let allocated: *mut Object = msg_send![class, alloc];
        ensure!(!allocated.is_null(), "NSString allocation failed");
        let initialized: *mut Object = msg_send![allocated,
            initWithBytes: value.as_ptr().cast::<std::ffi::c_void>()
            length: value.len()
            encoding: 4_usize
        ];
        ensure!(!initialized.is_null(), "NSString initialization failed");
        Ok(StrongPtr::new(initialized))
    }
}

fn mps_descriptor(rows: u32, columns: u32) -> Result<StrongPtr> {
    ensure!(rows > 0, "MPS matrix rows must be positive");
    let row_bytes = allocation_bytes(columns as usize)?;
    let class = Class::get("MPSMatrixDescriptor").context("MPSMatrixDescriptor is unavailable")?;
    // SAFETY: rowBytes is a whole packed FP32 row. Apple's documented recommended
    // alignment is optional: the required stride is a multiple of element size
    // and at least columns*sizeof(element). Retain the autoreleased descriptor.
    unsafe {
        let descriptor: *mut Object = msg_send![class,
            matrixDescriptorWithRows: rows as usize
            columns: columns as usize
            rowBytes: row_bytes
            dataType: MPS_DATA_TYPE_FLOAT32
        ];
        ensure!(
            !descriptor.is_null(),
            "MPS matrix descriptor creation failed"
        );
        Ok(StrongPtr::retain(descriptor))
    }
}

fn mps_matrix(buffer: &MetalBuffer, descriptor: &StrongPtr) -> Result<StrongPtr> {
    let class = Class::get("MPSMatrix").context("MPSMatrix is unavailable")?;
    // SAFETY: called only after descriptor dimensions and buffer origin/length
    // were checked together. initWithBuffer:descriptor: uses the existing buffer
    // at offset zero. The returned +1 reference is balanced by StrongPtr.
    unsafe {
        let allocated: *mut Object = msg_send![class, alloc];
        ensure!(!allocated.is_null(), "MPS matrix allocation failed");
        let initialized: *mut Object = msg_send![allocated,
            initWithBuffer: &*buffer.native
            descriptor: **descriptor
        ];
        ensure!(!initialized.is_null(), "MPS matrix initialization failed");
        Ok(StrongPtr::new(initialized))
    }
}

// Both custom kernels and MPS use this identical submit/wait/timestamp boundary.
fn finish_command(
    command: &metal::CommandBufferRef,
    started: Instant,
    dispatches: u64,
    repeats: u32,
) -> Result<Execution> {
    command.commit();
    command.wait_until_completed();
    let host_ms = started.elapsed().as_secs_f64() * 1000.0;
    if command.status() != MTLCommandBufferStatus::Completed {
        let description = unsafe {
            let error: *mut Object = msg_send![command, error];
            error_description(error)
        };
        bail!(
            "Metal command buffer failed ({:?}): {description}",
            command.status()
        );
    }
    // Native command-buffer timestamps are seconds, valid after completion.
    // Missing/zero timestamps remain None, never infinite reported throughput.
    let gpu_ms = unsafe {
        let has_start: BOOL = msg_send![command, respondsToSelector: sel!(GPUStartTime)];
        let has_end: BOOL = msg_send![command, respondsToSelector: sel!(GPUEndTime)];
        if has_start == YES && has_end == YES {
            let start: f64 = msg_send![command, GPUStartTime];
            let end: f64 = msg_send![command, GPUEndTime];
            let elapsed = (end - start) * 1000.0;
            (start.is_finite()
                && start > 0.0
                && end.is_finite()
                && elapsed.is_finite()
                && elapsed > 0.0)
                .then_some(elapsed)
        } else {
            None
        }
    };
    Ok(Execution {
        host_ms,
        gpu_ms,
        dispatches,
        repeats,
    })
}

pub struct MetalBuffer {
    native: metal::Buffer,
    elements: usize,
    id: u64,
    owner: u64,
    _thread_confined: PhantomData<Rc<()>>,
}

impl Buffer for MetalBuffer {
    fn len(&self) -> usize {
        self.elements
    }
    fn id(&self) -> u64 {
        self.id
    }
}

pub struct MetalKernel {
    spec: KernelSpec,
    kind: MetalKernelKind,
    owner: u64,
    _thread_confined: PhantomData<Rc<()>>,
}

enum MetalKernelKind {
    Shader {
        pipeline: metal::ComputePipelineState,
        grid: MTLSize,
        block: MTLSize,
    },
    MpsGemm(MetalMpsGemm),
}

/// Apple's native FP32 matrix-multiplication baseline. Descriptors describe the
/// same packed row-major buffers as Jade GEMM; no transpose, repacking or dtype
/// conversion is inserted. Native objects stay private and thread-confined.
pub struct MetalMpsGemm {
    native: StrongPtr,
    left_descriptor: StrongPtr,
    right_descriptor: StrongPtr,
    output_descriptor: StrongPtr,
    spec: KernelSpec,
    owner: u64,
    _thread_confined: PhantomData<Rc<()>>,
}

struct MpsMatrices {
    left: StrongPtr,
    right: StrongPtr,
    result: StrongPtr,
}

impl MpsMatrices {
    // The calling execution path checks all buffer owners, dimensions and aliases
    // before constructing wrappers. No GPU work or storage allocation occurs.
    fn new(
        kernel: &MetalMpsGemm,
        a: &MetalBuffer,
        b: &MetalBuffer,
        out: &MetalBuffer,
    ) -> Result<Self> {
        Ok(Self {
            left: mps_matrix(a, &kernel.left_descriptor)?,
            right: mps_matrix(b, &kernel.right_descriptor)?,
            result: mps_matrix(out, &kernel.output_descriptor)?,
        })
    }
}

fn encode_mps(command: &metal::CommandBufferRef, kernel: &MetalMpsGemm, matrices: &MpsMatrices) {
    // SAFETY: private callers retain the checked kernel, matrix wrappers and
    // buffers until command completion. They all belong to the same device.
    // MPS encodes its own operations into the same tracked command buffer.
    unsafe {
        let _: () = msg_send![*kernel.native,
            encodeToCommandBuffer: command
            leftMatrix: *matrices.left
            rightMatrix: *matrices.right
            resultMatrix: *matrices.result
        ];
    }
}

// Preparing a typed sequence avoids a parallel optional-wrapper array. Every
// MPS job owns its wrappers before timing starts and through completion.
enum PreparedMetalJob<'a> {
    Shader {
        pipeline: &'a metal::ComputePipelineState,
        grid: MTLSize,
        block: MTLSize,
        buffers: &'a [&'a MetalBuffer],
    },
    MpsGemm {
        kernel: &'a MetalMpsGemm,
        matrices: MpsMatrices,
    },
}

pub struct MetalRuntime {
    device: metal::Device,
    queue: metal::CommandQueue,
    name: String,
    owner: u64,
    pipelines: RefCell<HashMap<String, metal::ComputePipelineState>>,
    _thread_confined: PhantomData<Rc<()>>,
}

impl MetalRuntime {
    pub fn new() -> Result<Self> {
        autoreleasepool(|| {
            let device =
                metal::Device::system_default().context("no native Metal device available")?;
            // SAFETY: newCommandQueue returns an owned (+1) object. Check nil
            // before constructing metal-rs's non-null owner.
            let queue = unsafe {
                let raw: *mut metal::MTLCommandQueue = msg_send![&*device, newCommandQueue];
                ensure!(!raw.is_null(), "Metal command queue allocation failed");
                metal::CommandQueue::from_ptr(raw)
            };
            // SAFETY: a Metal device's name is a live NSString.
            let name = unsafe {
                let name: *mut Object = msg_send![&*device, name];
                string_value(name)
            };
            Ok(Self {
                device,
                queue,
                name,
                owner: fresh_id()?,
                pipelines: RefCell::new(HashMap::new()),
                _thread_confined: PhantomData,
            })
        })
    }

    fn check_buffer(&self, buffer: &MetalBuffer) -> Result<()> {
        ensure!(
            buffer.owner == self.owner,
            "buffer belongs to a different Metal runtime"
        );
        Ok(())
    }

    /// Conservative capability gate for the explicit SIMD-group variants.
    pub fn supports_apple7(&self) -> bool {
        // SAFETY: the selector query is valid on every NSObject-backed device;
        // call supportsFamily only when the OS implements it.
        unsafe {
            let available: BOOL =
                msg_send![&*self.device, respondsToSelector: sel!(supportsFamily:)];
            available == YES && self.device.supports_family(metal::MTLGPUFamily::Apple7)
        }
    }

    /// Prepare Apple's MPSMatrixMultiplication with alpha=1, beta=0 and FP32
    /// packed descriptors. Object creation is outside execute_mps_gemm timing.
    /// Public API references: MPSMatrixMultiplication.h and MPSCore/MPSMatrix.h
    /// in the Apple SDK; no private framework interfaces are used.
    pub fn compile_mps_gemm(&self, m: u32, n: u32, k: u32) -> Result<MetalMpsGemm> {
        let spec = KernelSpec::Gemm {
            m,
            n,
            k,
            tiled: false,
        };
        spec.validate()?;
        // SAFETY: the device is a live owned MTLDevice and the public support
        // function does not take ownership or submit any GPU work.
        ensure!(
            unsafe { mps_supports_device(self.device.as_ptr()) } == YES,
            "Metal Performance Shaders does not support this device"
        );
        autoreleasepool(|| {
            let left_descriptor = mps_descriptor(m, k)?;
            let right_descriptor = mps_descriptor(k, n)?;
            let output_descriptor = mps_descriptor(m, n)?;
            let class = Class::get("MPSMatrixMultiplication")
                .context("MPSMatrixMultiplication class is unavailable")?;
            // SAFETY: alloc/init returns +1 ownership, checked for nil before
            // wrapping. Selector argument widths follow the public SDK header:
            // BOOL flags, NSUInteger dimensions, double alpha/beta.
            let native = unsafe {
                let allocated: *mut Object = msg_send![class, alloc];
                ensure!(!allocated.is_null(), "MPS GEMM allocation failed");
                let initialized: *mut Object = msg_send![allocated,
                    initWithDevice: &*self.device
                    transposeLeft: NO
                    transposeRight: NO
                    resultRows: m as usize
                    resultColumns: n as usize
                    interiorColumns: k as usize
                    alpha: 1.0_f64
                    beta: 0.0_f64
                ];
                ensure!(!initialized.is_null(), "MPS GEMM initialization failed");
                // MPSKernelOptionsNone. Do not opt into the public
                // MPSKernelOptionsAllowReducedPrecision performance tradeoff.
                let _: () = msg_send![initialized, setOptions: 0_usize];
                StrongPtr::new(initialized)
            };
            Ok(MetalMpsGemm {
                native,
                left_descriptor,
                right_descriptor,
                output_descriptor,
                spec,
                owner: self.owner,
                _thread_confined: PhantomData,
            })
        })
    }

    /// Run repeated C=A*B with Apple's native FP32 GEMM. Matrix wrapper creation
    /// and argument validation occur before the timer. Host time includes command
    /// buffer creation, every MPS encode, submission and completion wait. MPS may
    /// perform internal lazy setup on its first encode; benchmarks must warm up.
    /// Execution.dispatches counts logical MPS GEMM encodes; the framework's
    /// internal number of GPU dispatches is opaque.
    pub fn execute_mps_gemm(
        &self,
        kernel: &MetalMpsGemm,
        a: &MetalBuffer,
        b: &MetalBuffer,
        out: &MetalBuffer,
        repeats: u32,
    ) -> Result<Execution> {
        ensure!(
            kernel.owner == self.owner,
            "MPS kernel belongs to a different Metal runtime"
        );
        ensure!(repeats > 0, "MPS GEMM repeats must be positive");
        for buffer in [a, b, out] {
            self.check_buffer(buffer)?;
        }
        kernel
            .spec
            .validate_bindings(&[a.len(), b.len(), out.len()], &[a.id(), b.id(), out.id()])?;
        autoreleasepool(|| {
            // Wrappers retain existing buffers; they do not allocate matrix
            // storage or copy/repack the data. Their setup is not timed.
            let matrices = MpsMatrices::new(kernel, a, b, out)?;
            let started = Instant::now();
            let command = unsafe {
                // SAFETY: checked autoreleased command buffer lives through
                // completion inside this pool, as in the regular execute path.
                let raw: *mut metal::MTLCommandBuffer = msg_send![&*self.queue, commandBuffer];
                ensure!(!raw.is_null(), "Metal command buffer allocation failed");
                metal::CommandBufferRef::from_ptr(raw)
            };
            for _ in 0..repeats {
                encode_mps(command, kernel, &matrices);
            }
            finish_command(command, started, u64::from(repeats), repeats)
        })
    }

    fn build_pipeline(&self, spec: &KernelSpec) -> Result<metal::ComputePipelineState> {
        let source = native_string(&spec.metal_source()?)?;
        let options_class =
            Class::get("MTLCompileOptions").context("Metal compile options unavailable")?;
        // SAFETY: the objects returned by new* are owned. All nil results are
        // checked before wrapping them. NSError is borrowed within this pool.
        unsafe {
            let raw_options: *mut metal::MTLCompileOptions = msg_send![options_class, new];
            ensure!(
                !raw_options.is_null(),
                "Metal compile options allocation failed"
            );
            let options = metal::CompileOptions::from_ptr(raw_options);
            options.set_fast_math_enabled(false);
            let mut error: *mut Object = std::ptr::null_mut();
            let raw_library: *mut metal::MTLLibrary = msg_send![&*self.device,
                newLibraryWithSource: *source
                options: &*options
                error: &mut error
            ];
            ensure!(
                !raw_library.is_null(),
                "Metal source compilation failed for {}: {}",
                spec.name(),
                error_description(error)
            );
            let library = metal::Library::from_ptr(raw_library);
            let entry = native_string("jade_main")?;
            let raw_function: *mut metal::MTLFunction =
                msg_send![&*library, newFunctionWithName: *entry];
            ensure!(
                !raw_function.is_null(),
                "Metal source does not contain jade_main"
            );
            let function = metal::Function::from_ptr(raw_function);
            error = std::ptr::null_mut();
            let raw_pipeline: *mut metal::MTLComputePipelineState = msg_send![&*self.device,
                newComputePipelineStateWithFunction: &*function
                error: &mut error
            ];
            ensure!(
                !raw_pipeline.is_null(),
                "Metal pipeline creation failed: {}",
                error_description(error)
            );
            Ok(metal::ComputePipelineState::from_ptr(raw_pipeline))
        }
    }
}

impl Runtime for MetalRuntime {
    type Buffer = MetalBuffer;
    type Kernel = MetalKernel;

    fn info(&self) -> RuntimeInfo {
        RuntimeInfo {
            backend: "native-metal".into(),
            device: self.name.clone(),
            execution: "direct MTLBuffer/MTLComputePipelineState and explicit MPS GEMM; ordered synchronous command buffers".into(),
            limitations: vec![
                "FP32 Jade IR; explicit SIMD-group matrix operations where selected; no NVIDIA Tensor Core/WMMA ABI".into(),
                format!("Apple7+ SIMD-group capability gate: {}", self.supports_apple7()),
                "Shared storage and thread-confined synchronous API; no asynchronous stream interface".into(),
                "GPU timings cover the full command buffer, including dependencies between jobs".into(),
                "Dispatch counts are logical operations; MPS internal GPU kernel counts are opaque".into(),
                "Metal validation does not establish Chinese-GPU support or performance".into(),
            ],
        }
    }

    fn alloc(&self, elements: usize) -> Result<MetalBuffer> {
        let bytes = allocation_bytes(elements)?;
        ensure!(
            bytes as u64 <= self.device.max_buffer_length(),
            "allocation exceeds Metal maxBufferLength"
        );
        let id = fresh_id()?;
        autoreleasepool(|| {
            let options = MTLResourceOptions::StorageModeShared
                | MTLResourceOptions::HazardTrackingModeTracked;
            // SAFETY: newBufferWithLength owns the allocation. Check nil before
            // constructing the wrapper, and initialize bytes before host reads.
            let native = unsafe {
                let raw: *mut metal::MTLBuffer = msg_send![&*self.device,
                    newBufferWithLength: bytes as u64
                    options: options
                ];
                ensure!(
                    !raw.is_null(),
                    "Metal buffer allocation failed for {bytes} bytes"
                );
                let native = metal::Buffer::from_ptr(raw);
                ensure!(
                    native.length() >= bytes as u64,
                    "Metal returned a short allocation"
                );
                let contents = NonNull::new(native.contents().cast::<u8>())
                    .context("Metal shared buffer has no host mapping")?;
                std::ptr::write_bytes(contents.as_ptr(), 0, bytes);
                native
            };
            Ok(MetalBuffer {
                native,
                elements,
                id,
                owner: self.owner,
                _thread_confined: PhantomData,
            })
        })
    }

    fn upload(&self, buffer: &MetalBuffer, data: &[f32]) -> Result<()> {
        self.check_buffer(buffer)?;
        ensure!(
            data.len() == buffer.elements,
            "upload requires exactly {} elements, got {}",
            buffer.elements,
            data.len()
        );
        let dst = NonNull::new(buffer.native.contents().cast::<f32>())
            .context("Metal shared buffer is not mapped")?;
        // SAFETY: the allocation is sufficiently sized, initialized, private to
        // this thread, and no GPU work survives a synchronous execute call.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst.as_ptr(), data.len());
        }
        Ok(())
    }

    fn download(&self, buffer: &MetalBuffer) -> Result<Vec<f32>> {
        self.check_buffer(buffer)?;
        let src = NonNull::new(buffer.native.contents().cast::<f32>())
            .context("Metal shared buffer is not mapped")?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(buffer.elements)
            .context("host readback allocation failed")?;
        // SAFETY: memory is initialized at allocation, execute waits before
        // returning, and !Send/!Sync excludes concurrent uploads or execution.
        unsafe {
            output.extend_from_slice(std::slice::from_raw_parts(src.as_ptr(), buffer.elements));
        }
        Ok(output)
    }

    fn compile(&self, spec: &KernelSpec) -> Result<MetalKernel> {
        spec.validate()?;
        // MPS is an explicit IR operation, never a silent substitution for a
        // shader variant. It has no generated source or caller-specified grid.
        if let KernelSpec::GemmMps { m, n, k } = spec {
            return Ok(MetalKernel {
                spec: spec.clone(),
                kind: MetalKernelKind::MpsGemm(self.compile_mps_gemm(*m, *n, *k)?),
                owner: self.owner,
                _thread_confined: PhantomData,
            });
        }
        let requires_simdgroup = spec.requires_simdgroup_matrix() || spec.requires_simdgroup_ops();
        ensure!(
            !requires_simdgroup || self.supports_apple7(),
            "selected SIMD-group operation requires Apple7 or newer Metal GPU support"
        );
        let launch = spec.launch()?;
        ensure!(
            launch.grid.iter().chain(&launch.block).all(|&n| n > 0),
            "Metal launch dimensions must be positive"
        );
        let thread_count = launch
            .block
            .iter()
            .try_fold(1_u64, |n, &v| n.checked_mul(v as u64))
            .context("Metal threadgroup size overflow")?;
        let device_limit = self.device.max_threads_per_threadgroup();
        ensure!(
            launch.block[0] as u64 <= device_limit.width
                && launch.block[1] as u64 <= device_limit.height
                && launch.block[2] as u64 <= device_limit.depth,
            "kernel threadgroup dimensions exceed Metal device limits"
        );
        let key = serde_json::to_string(spec).context("serialize Metal pipeline cache key")?;
        let cached = self
            .pipelines
            .try_borrow()
            .context("Metal pipeline cache is already borrowed")?
            .get(&key)
            .cloned();
        let pipeline = if let Some(pipeline) = cached {
            pipeline
        } else {
            let pipeline = autoreleasepool(|| self.build_pipeline(spec))?;
            self.pipelines
                .try_borrow_mut()
                .context("Metal pipeline cache is already borrowed")?
                .insert(key, pipeline.clone());
            pipeline
        };
        ensure!(
            thread_count <= pipeline.max_total_threads_per_threadgroup(),
            "kernel exceeds pipeline threadgroup limit"
        );
        ensure!(
            !requires_simdgroup || pipeline.thread_execution_width() == 32,
            "selected SIMD-group operation requires a 32-thread execution width"
        );
        ensure!(
            pipeline.static_threadgroup_memory_length()
                <= self.device.max_threadgroup_memory_length(),
            "kernel exceeds Metal shared-memory limit"
        );
        let size = |v: [u32; 3]| MTLSize::new(v[0] as u64, v[1] as u64, v[2] as u64);
        Ok(MetalKernel {
            spec: spec.clone(),
            kind: MetalKernelKind::Shader {
                pipeline,
                grid: size(launch.grid),
                block: size(launch.block),
            },
            owner: self.owner,
            _thread_confined: PhantomData,
        })
    }

    fn execute(
        &self,
        jobs: &[Dispatch<'_, MetalBuffer, MetalKernel>],
        repeats: u32,
    ) -> Result<Execution> {
        ensure!(!jobs.is_empty(), "execute requires at least one job");
        ensure!(repeats > 0, "execute repeats must be positive");
        let dispatches = u64::try_from(jobs.len())?
            .checked_mul(repeats as u64)
            .context("dispatch count overflow")?;
        // Validate the entire sequence before submitting any work, so a bad
        // later job cannot leave partially executed output behind.
        for job in jobs {
            ensure!(
                job.kernel.owner == self.owner,
                "kernel belongs to a different Metal runtime"
            );
            ensure!(job.buffers.len() <= 31, "too many Metal buffer bindings");
            for buffer in &job.buffers {
                self.check_buffer(buffer)?;
            }
            let lengths: Vec<_> = job.buffers.iter().map(|b| b.len()).collect();
            let ids: Vec<_> = job.buffers.iter().map(|b| b.id()).collect();
            job.kernel.spec.validate_bindings(&lengths, &ids)?;
        }
        autoreleasepool(|| {
            let prepared = jobs
                .iter()
                .map(|job| -> Result<PreparedMetalJob<'_>> {
                    Ok(match &job.kernel.kind {
                        MetalKernelKind::Shader {
                            pipeline,
                            grid,
                            block,
                        } => PreparedMetalJob::Shader {
                            pipeline,
                            grid: *grid,
                            block: *block,
                            buffers: &job.buffers,
                        },
                        MetalKernelKind::MpsGemm(kernel) => PreparedMetalJob::MpsGemm {
                            kernel,
                            // Full-batch binding validation above established exactly
                            // three shape-matched buffers for this MPS operation.
                            matrices: MpsMatrices::new(
                                kernel,
                                job.buffers[0],
                                job.buffers[1],
                                job.buffers[2],
                            )?,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let started = Instant::now();
            // SAFETY: commandBuffer and computeCommandEncoder are autoreleased
            // objects used only within this pool. Standard command buffers
            // retain referenced resources; all execution finishes before return.
            let command = unsafe {
                let raw: *mut metal::MTLCommandBuffer = msg_send![&*self.queue, commandBuffer];
                ensure!(!raw.is_null(), "Metal command buffer allocation failed");
                metal::CommandBufferRef::from_ptr(raw)
            };
            for _ in 0..repeats {
                for job in &prepared {
                    match job {
                        PreparedMetalJob::Shader {
                            pipeline,
                            grid,
                            block,
                            buffers,
                        } => {
                            let encoder = unsafe {
                                let raw: *mut metal::MTLComputeCommandEncoder =
                                    msg_send![command, computeCommandEncoder];
                                ensure!(!raw.is_null(), "Metal compute encoder allocation failed");
                                metal::ComputeCommandEncoderRef::from_ptr(raw)
                            };
                            encoder.set_compute_pipeline_state(pipeline);
                            for (index, buffer) in buffers.iter().enumerate() {
                                encoder.set_buffer(index as u64, Some(&buffer.native), 0);
                            }
                            encoder.dispatch_thread_groups(*grid, *block);
                            encoder.end_encoding();
                        }
                        PreparedMetalJob::MpsGemm { kernel, matrices } => {
                            encode_mps(command, kernel, matrices);
                        }
                    }
                    // Separate serial encoders plus tracked resources establish
                    // write/read ordering across custom and MPS jobs/repetitions.
                }
            }
            finish_command(command, started, dispatches, repeats)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsafe_host_allocation_sizes() {
        assert!(allocation_bytes(0).is_err());
        assert!(allocation_bytes(usize::MAX).is_err());
        assert!(allocation_bytes(isize::MAX as usize / 4 + 1).is_err());
        assert_eq!(allocation_bytes(7).unwrap(), 28);
    }

    #[test]
    #[ignore = "requires a working native Metal GPU and Metal Performance Shaders"]
    fn native_mps_gemm_packed_fp32_and_ownership_checks() -> Result<()> {
        let runtime = MetalRuntime::new()?;
        let other = MetalRuntime::new()?;
        assert!(runtime.compile_mps_gemm(0, 2, 3).is_err());
        for (m, n, k) in [(1, 1, 1), (2, 3, 2), (17, 31, 19), (64, 64, 64)] {
            let shape = crate::Shape::new(m, n, k)?;
            let (av, bv) = crate::validation::inputs(shape);
            let a = runtime.alloc(av.len())?;
            let b = runtime.alloc(bv.len())?;
            let out = runtime.alloc(shape.c_len())?;
            let kernel = runtime.compile_mps_gemm(m, n, k)?;
            runtime.upload(&a, &av)?;
            runtime.upload(&b, &bv)?;
            let timing = runtime.execute_mps_gemm(&kernel, &a, &b, &out, 3)?;
            assert_eq!(timing.dispatches, 3);
            assert_eq!(timing.repeats, 3);
            assert!(timing.host_ms.is_finite() && timing.host_ms > 0.0);
            if let Some(ms) = timing.gpu_ms {
                assert!(ms.is_finite() && ms > 0.0);
            }
            crate::validation::check(shape, &av, &bv, &runtime.download(&out)?, usize::MAX, false)?;
        }
        let a = runtime.alloc(4)?;
        let b = runtime.alloc(4)?;
        let out = runtime.alloc(4)?;
        let short = runtime.alloc(3)?;
        let foreign = other.alloc(4)?;
        let kernel = runtime.compile_mps_gemm(2, 2, 2)?;
        let foreign_kernel = other.compile_mps_gemm(2, 2, 2)?;
        assert!(runtime.execute_mps_gemm(&kernel, &a, &b, &a, 1).is_err());
        assert!(runtime.execute_mps_gemm(&kernel, &a, &b, &b, 1).is_err());
        assert!(runtime
            .execute_mps_gemm(&kernel, &short, &b, &out, 1)
            .is_err());
        assert!(runtime
            .execute_mps_gemm(&kernel, &foreign, &b, &out, 1)
            .is_err());
        assert!(runtime
            .execute_mps_gemm(&foreign_kernel, &a, &b, &out, 1)
            .is_err());
        assert!(runtime.execute_mps_gemm(&kernel, &a, &b, &out, 0).is_err());
        assert_eq!(runtime.download(&out)?, vec![0.0; 4]);
        // Read/read aliasing is valid; only writes must not alias another binding.
        runtime.upload(&a, &[1.0, 2.0, 3.0, 4.0])?;
        runtime.execute_mps_gemm(&kernel, &a, &a, &out, 1)?;
        assert_eq!(runtime.download(&out)?, vec![7.0, 10.0, 15.0, 22.0]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a working native Metal GPU and Metal Performance Shaders"]
    fn native_mixed_mps_shader_batch_ordering_and_rejection() -> Result<()> {
        let runtime = MetalRuntime::new()?;
        let other = MetalRuntime::new()?;
        let a = runtime.alloc(4)?;
        let b = runtime.alloc(4)?;
        let bias = runtime.alloc(4)?;
        let ab = runtime.alloc(4)?;
        let shifted = runtime.alloc(4)?;
        let out = runtime.alloc(4)?;
        let foreign = other.alloc(4)?;
        runtime.upload(&a, &[1.0, 2.0, 3.0, 4.0])?;
        runtime.upload(&b, &[2.0, 0.0, 1.0, 2.0])?;
        runtime.upload(&bias, &[1.0, -1.0, 2.0, -2.0])?;
        runtime.upload(&ab, &[11.0, 12.0, 13.0, 14.0])?;
        let spec = KernelSpec::GemmMps { m: 2, n: 2, k: 2 };
        let gemm = runtime.compile(&spec)?;
        let foreign_gemm = other.compile(&spec)?;
        let add = runtime.compile(&KernelSpec::Add { len: 4 })?;

        // A later invalid shader job must reject before the first MPS operation
        // is encoded/submitted, preserving preexisting output bytes.
        assert!(runtime
            .execute(
                &[
                    Dispatch {
                        kernel: &gemm,
                        buffers: vec![&a, &b, &ab]
                    },
                    Dispatch {
                        kernel: &add,
                        buffers: vec![&ab, &foreign, &shifted]
                    },
                ],
                1
            )
            .is_err());
        assert_eq!(runtime.download(&ab)?, vec![11.0, 12.0, 13.0, 14.0]);
        assert!(runtime
            .execute(
                &[
                    Dispatch {
                        kernel: &gemm,
                        buffers: vec![&a, &b, &ab]
                    },
                    Dispatch {
                        kernel: &foreign_gemm,
                        buffers: vec![&ab, &b, &out]
                    },
                ],
                1
            )
            .is_err());
        assert_eq!(runtime.download(&ab)?, vec![11.0, 12.0, 13.0, 14.0]);
        assert!(runtime
            .execute(
                &[Dispatch {
                    kernel: &gemm,
                    buffers: vec![&a, &b, &a]
                },],
                1
            )
            .is_err());

        // Both MPS->shader and shader->MPS dependencies occur in one command
        // buffer. Replay must preserve their order across all repetitions.
        let jobs = [
            Dispatch {
                kernel: &gemm,
                buffers: vec![&a, &b, &ab],
            },
            Dispatch {
                kernel: &add,
                buffers: vec![&ab, &bias, &shifted],
            },
            Dispatch {
                kernel: &gemm,
                buffers: vec![&shifted, &b, &out],
            },
        ];
        let timing = runtime.execute(&jobs, 3)?;
        assert_eq!(timing.dispatches, 9); // logical operations, not MPS internals
        assert_eq!(timing.repeats, 3);
        assert_eq!(runtime.download(&ab)?, vec![4.0, 4.0, 10.0, 8.0]);
        assert_eq!(runtime.download(&shifted)?, vec![5.0, 3.0, 12.0, 6.0]);
        assert_eq!(runtime.download(&out)?, vec![13.0, 6.0, 30.0, 12.0]);
        assert_eq!(runtime.download(&a)?, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(runtime.download(&b)?, vec![2.0, 0.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a working native Metal GPU"]
    fn native_ordering_ownership_and_alias_checks() -> Result<()> {
        let runtime = MetalRuntime::new()?;
        let other = MetalRuntime::new()?;
        let a = runtime.alloc(5)?;
        let b = runtime.alloc(5)?;
        let intermediate = runtime.alloc(5)?;
        let out = runtime.alloc(5)?;
        let foreign = other.alloc(5)?;
        runtime.upload(&a, &[0., 1., -2., 3., 1.])?;
        runtime.upload(&b, &[-1., 2., 3., -4., 7.])?;
        assert!(runtime.upload(&a, &[1.]).is_err());
        assert!(runtime.upload(&foreign, &[0.; 5]).is_err());
        assert!(runtime.download(&foreign).is_err());
        let spec = KernelSpec::Add { len: 5 };
        let add = runtime.compile(&spec)?;
        let foreign_add = other.compile(&spec)?;
        assert!(runtime
            .execute(
                &[Dispatch {
                    kernel: &foreign_add,
                    buffers: vec![&a, &b, &out]
                }],
                1
            )
            .is_err());
        assert!(runtime
            .execute(
                &[Dispatch {
                    kernel: &add,
                    buffers: vec![&a, &b, &a]
                }],
                1
            )
            .is_err());
        // A bad later job must reject the whole sequence before the first runs.
        assert!(runtime
            .execute(
                &[
                    Dispatch {
                        kernel: &add,
                        buffers: vec![&a, &b, &intermediate]
                    },
                    Dispatch {
                        kernel: &add,
                        buffers: vec![&intermediate, &foreign, &out]
                    },
                ],
                1
            )
            .is_err());
        assert_eq!(runtime.download(&intermediate)?, vec![0.; 5]);
        let jobs = [
            Dispatch {
                kernel: &add,
                buffers: vec![&a, &b, &intermediate],
            },
            Dispatch {
                kernel: &add,
                buffers: vec![&intermediate, &b, &out],
            },
        ];
        assert!(runtime.execute(&jobs, 0).is_err());
        let timing = runtime.execute(&jobs, 3)?;
        assert_eq!(timing.dispatches, 6);
        assert_eq!(timing.repeats, 3);
        assert_eq!(runtime.download(&out)?, vec![-2., 5., 4., -5., 15.]);
        assert!(timing.host_ms.is_finite() && timing.host_ms > 0.);
        if let Some(ms) = timing.gpu_ms {
            assert!(ms.is_finite() && ms > 0.);
        }
        Ok(())
    }
}
