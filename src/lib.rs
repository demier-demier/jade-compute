//! Experimental resident-buffer GEMM runtime. CoreX is loaded explicitly at runtime.
pub mod corex;
pub mod portable;
pub mod runtime;
pub mod validation;

use anyhow::{ensure, Result};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Shape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl Shape {
    pub fn new(m: u32, n: u32, k: u32) -> Result<Self> {
        ensure!(m > 0 && n > 0 && k > 0, "dimensions must be positive");
        for (x, y) in [(m, k), (k, n), (m, n)] {
            // Shader indices are u32; checked before host allocation.
            ensure!(
                u64::from(x) * u64::from(y) <= u64::from(u32::MAX),
                "matrix exceeds u32 indexing"
            );
        }
        Ok(Self { m, n, k })
    }
    pub fn a_len(self) -> usize {
        self.m as usize * self.k as usize
    }
    pub fn b_len(self) -> usize {
        self.k as usize * self.n as usize
    }
    pub fn c_len(self) -> usize {
        self.m as usize * self.n as usize
    }
    pub fn flops(self) -> f64 {
        2.0 * self.m as f64 * self.n as f64 * self.k as f64
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Timing {
    /// Includes host encoding, submit and synchronization. Excludes initial upload.
    pub wall_ms_per_dispatch: f64,
    /// Device timestamps over the batch, divided by dispatch count.
    pub gpu_ms_per_dispatch: Option<f64>,
}

pub trait MatmulPlan {
    fn name(&self) -> &str;
    fn run(&mut self, iterations: u32) -> Result<Timing>;
    fn download(&self) -> Result<Vec<f32>>;
}
