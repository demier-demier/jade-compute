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
    runtime::{Class, Object, BOOL, YES},
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
    pipeline: metal::ComputePipelineState,
    grid: MTLSize,
    block: MTLSize,
    owner: u64,
    _thread_confined: PhantomData<Rc<()>>,
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
            execution: "direct MTLBuffer/MTLComputePipelineState; ordered synchronous command buffers".into(),
            limitations: vec![
                "FP32 kernels generated from Jade IR; no tensor-core instructions".into(),
                "Shared storage and thread-confined synchronous API; no asynchronous stream interface".into(),
                "GPU timings cover the full command buffer, including dependencies between jobs".into(),
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
            pipeline.static_threadgroup_memory_length()
                <= self.device.max_threadgroup_memory_length(),
            "kernel exceeds Metal shared-memory limit"
        );
        let size = |v: [u32; 3]| MTLSize::new(v[0] as u64, v[1] as u64, v[2] as u64);
        Ok(MetalKernel {
            spec: spec.clone(),
            pipeline,
            grid: size(launch.grid),
            block: size(launch.block),
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
                for job in jobs {
                    let encoder = unsafe {
                        let raw: *mut metal::MTLComputeCommandEncoder =
                            msg_send![command, computeCommandEncoder];
                        ensure!(!raw.is_null(), "Metal compute encoder allocation failed");
                        metal::ComputeCommandEncoderRef::from_ptr(raw)
                    };
                    encoder.set_compute_pipeline_state(&job.kernel.pipeline);
                    for (index, buffer) in job.buffers.iter().enumerate() {
                        encoder.set_buffer(index as u64, Some(&buffer.native), 0);
                    }
                    encoder.dispatch_thread_groups(job.kernel.grid, job.kernel.block);
                    encoder.end_encoding();
                    // Separate serial encoders plus tracked resources establish
                    // write-to-read dependencies between jobs and repetitions.
                }
            }
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
            // Native command-buffer timestamps are in seconds and are valid
            // only after completion. Missing/zero timestamps remain None.
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
