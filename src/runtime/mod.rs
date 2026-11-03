//! Native compute layer. Execution is synchronous at the public boundary.
pub mod corex;
pub mod ir;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod workloads;

use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeInfo {
    pub backend: String,
    pub device: String,
    pub execution: String,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Execution {
    pub host_ms: f64,
    pub gpu_ms: Option<f64>,
    pub dispatches: u64,
    pub repeats: u32,
}

pub trait Buffer {
    fn len(&self) -> usize;
    fn id(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Dispatch<'a, B, K> {
    pub kernel: &'a K,
    pub buffers: Vec<&'a B>,
}

/// Backends own native memory and kernel pipelines. `execute` enqueues jobs in
/// order, replays that sequence `repeats` times, and waits before returning.
/// It is not a CUDA ABI, CUDA Graph API, or asynchronous stream interface.
pub trait Runtime {
    type Buffer: Buffer;
    type Kernel;
    fn info(&self) -> RuntimeInfo;
    fn alloc(&self, elements: usize) -> Result<Self::Buffer>;
    fn upload(&self, buffer: &Self::Buffer, data: &[f32]) -> Result<()>;
    fn download(&self, buffer: &Self::Buffer) -> Result<Vec<f32>>;
    fn compile(&self, spec: &ir::KernelSpec) -> Result<Self::Kernel>;
    fn execute(
        &self,
        jobs: &[Dispatch<'_, Self::Buffer, Self::Kernel>],
        repeats: u32,
    ) -> Result<Execution>;
}
