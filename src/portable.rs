use crate::{MatmulPlan, Shape, Timing};
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::{sync::mpsc, time::Instant};
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Auto,
    Vulkan,
    Metal,
}
impl Backend {
    fn mask(self) -> wgpu::Backends {
        match self {
            Self::Auto => wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Metal => wgpu::Backends::METAL,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    pub timestamps: bool,
    pub max_storage_buffer_binding_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
}
fn describe(adapter: &wgpu::Adapter) -> DeviceInfo {
    let i = adapter.get_info();
    DeviceInfo {
        name: i.name,
        vendor_id: i.vendor,
        device_id: i.device,
        backend: format!("{:?}", i.backend),
        device_type: format!("{:?}", i.device_type),
        driver: i.driver,
        driver_info: i.driver_info,
        timestamps: adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
        max_compute_invocations_per_workgroup: adapter
            .limits()
            .max_compute_invocations_per_workgroup,
    }
}
fn instance(backend: Backend) -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: backend.mask(),
        ..Default::default()
    })
}
pub fn devices(backend: Backend) -> Vec<DeviceInfo> {
    instance(backend)
        .enumerate_adapters(backend.mask())
        .iter()
        .map(describe)
        .collect()
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub info: DeviceInfo,
}
impl Gpu {
    pub fn new(backend: Backend, name: Option<&str>) -> Result<Self> {
        let instance = instance(backend);
        let adapters = instance.enumerate_adapters(backend.mask());
        let mut candidates: Vec<_> = adapters
            .into_iter()
            .filter(|a| {
                matches!(
                    a.get_info().device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                ) && name
                    .is_none_or(|s| a.get_info().name.to_lowercase().contains(&s.to_lowercase()))
            })
            .collect();
        candidates.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        });
        let adapter = candidates
            .first()
            .context("no matching hardware GPU; CPU/software fallback is disabled")?;
        let info = describe(adapter);
        ensure!(
            info.max_compute_invocations_per_workgroup >= 256,
            "GPU does not support 256-thread tiled kernel"
        );
        let features = if info.timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jade-device"),
            required_features: features,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .context("request GPU device")?;
        Ok(Self {
            device,
            queue,
            info,
        })
    }

    pub fn plan(&self, shape: Shape, kernel: Kernel, a: &[f32], b: &[f32]) -> Result<PortablePlan> {
        Shape::new(shape.m, shape.n, shape.k)?;
        ensure!(
            a.len() == shape.a_len() && b.len() == shape.b_len(),
            "input shape mismatch"
        );
        let limits = self.device.limits();
        for len in [shape.a_len(), shape.b_len(), shape.c_len()] {
            let bytes = len as u64 * 4;
            ensure!(
                bytes <= limits.max_buffer_size
                    && bytes <= limits.max_storage_buffer_binding_size as u64,
                "matrix exceeds GPU buffer limits"
            );
        }
        let tile = kernel.tile();
        let dispatch = [shape.n.div_ceil(tile), shape.m.div_ceil(tile)];
        ensure!(
            dispatch
                .iter()
                .all(|&d| d <= limits.max_compute_workgroups_per_dimension),
            "dispatch exceeds GPU limits"
        );
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(kernel.name()),
                source: wgpu::ShaderSource::Wgsl(kernel.source().into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.name()),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            bail!("kernel compilation failed: {error}");
        }
        let a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A-resident"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B-resident"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C-resident"),
            size: shape.c_len() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dims = [shape.m, shape.n, shape.k, 0_u32];
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GEMM-shape"),
                contents: bytemuck::cast_slice(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident-matrices"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let timestamp = self.info.timestamps.then(|| {
            let query = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("GEMM-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (query, resolve, read)
        });
        Ok(PortablePlan {
            device: self.device.clone(),
            queue: self.queue.clone(),
            shape,
            kernel,
            pipeline,
            bind,
            output: c_buffer,
            timestamp,
            dispatch,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    Naive,
    Tiled16,
    Register32,
}
impl Kernel {
    pub const ALL: [Self; 3] = [Self::Naive, Self::Tiled16, Self::Register32];
    pub fn name(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Tiled16 => "tiled16",
            Self::Register32 => "register32",
        }
    }
    fn tile(self) -> u32 {
        match self {
            Self::Naive => 8,
            Self::Tiled16 => 16,
            Self::Register32 => 32,
        }
    }
    fn source(self) -> &'static str {
        match self {
            Self::Naive => include_str!("kernels/naive.wgsl"),
            Self::Tiled16 => include_str!("kernels/tiled16.wgsl"),
            Self::Register32 => include_str!("kernels/register32.wgsl"),
        }
    }
}

pub struct PortablePlan {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shape: Shape,
    kernel: Kernel,
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    output: wgpu::Buffer,
    timestamp: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)>,
    dispatch: [u32; 2],
}
impl MatmulPlan for PortablePlan {
    fn name(&self) -> &str {
        self.kernel.name()
    }
    fn run(&mut self, iterations: u32) -> Result<Timing> {
        ensure!(iterations > 0, "iterations must be positive");
        let start = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GEMM-batch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.name()),
                timestamp_writes: self.timestamp.as_ref().map(|(query, _, _)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set: query,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            for _ in 0..iterations {
                pass.dispatch_workgroups(self.dispatch[0], self.dispatch[1], 1);
            }
        }
        if let Some((query, resolve, read)) = &self.timestamp {
            encoder.resolve_query_set(query, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, read, 0, 16);
        }
        let submission = self.queue.submit([encoder.finish()]);
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        let wall_ms_per_dispatch = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let gpu_ms_per_dispatch = if let Some((_, _, read)) = &self.timestamp {
            let bytes = read_bytes(&self.device, read)?;
            let first = u64::from_ne_bytes(bytes[0..8].try_into()?);
            let last = u64::from_ne_bytes(bytes[8..16].try_into()?);
            if last > first {
                Some(
                    (last - first) as f64 * self.queue.get_timestamp_period() as f64
                        / 1e6
                        / iterations as f64,
                )
            } else {
                eprintln!("{}: unusable GPU timestamp interval ({first}, {last}); using synchronized wall clock for this plan", self.kernel.name());
                None
            }
        } else {
            None
        };
        if gpu_ms_per_dispatch.is_none() {
            self.timestamp = None;
        }
        Ok(Timing {
            wall_ms_per_dispatch,
            gpu_ms_per_dispatch,
        })
    }
    fn download(&self) -> Result<Vec<f32>> {
        let bytes = self.shape.c_len() as u64 * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.output, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        Ok(read_bytes(&self.device, &staging)?
            .chunks_exact(4)
            .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
            .collect())
    }
}

fn read_bytes(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>> {
    let (tx, rx) = mpsc::sync_channel(1);
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    rx.recv().context("readback callback missing")??;
    let bytes = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    Ok(bytes)
}
