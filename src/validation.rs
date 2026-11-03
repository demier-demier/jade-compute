use crate::Shape;
use anyhow::{bail, ensure, Result};
use serde::Serialize;

/// Deterministic mixed-sign non-symmetric inputs, independent of backend.
pub fn inputs(shape: Shape) -> (Vec<f32>, Vec<f32>) {
    fn fill(len: usize, mut state: u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
            })
            .collect()
    }
    (
        fill(shape.a_len(), 0x12345678),
        fill(shape.b_len(), 0x9abcdef1),
    )
}

#[derive(Debug, Serialize)]
pub struct Validation {
    pub checked_elements: usize,
    pub total_elements: usize,
    pub all_elements_finite: bool,
    pub max_abs_error: f64,
    pub max_scaled_error: f64,
    pub reference: &'static str,
}

/// FP64 dot products. Bound uses sum(abs(a*b)), so cancellation does not hide error.
/// FP16 allowance is for input quantization, not an FP16 accumulator.
pub fn check(
    shape: Shape,
    a: &[f32],
    b: &[f32],
    out: &[f32],
    budget: usize,
    fp16: bool,
) -> Result<Validation> {
    ensure!(budget > 0, "validation budget must be positive");
    ensure!(
        a.len() == shape.a_len() && b.len() == shape.b_len() && out.len() == shape.c_len(),
        "matrix length mismatch"
    );
    if let Some(i) = out.iter().position(|x| !x.is_finite()) {
        bail!("nonfinite output at {i}");
    }
    let count = budget.min(out.len());
    let mut max_abs_error = 0.0_f64;
    let mut max_scaled_error = 0.0_f64;
    for sample in 0..count {
        // Even spread includes both endpoints and boundary rows.
        let idx = if count <= 1 {
            0
        } else {
            sample * (out.len() - 1) / (count - 1)
        };
        let row = idx / shape.n as usize;
        let col = idx % shape.n as usize;
        let mut expected = 0.0;
        let mut sum_abs = 0.0;
        for inner in 0..shape.k as usize {
            let product =
                a[row * shape.k as usize + inner] as f64 * b[inner * shape.n as usize + col] as f64;
            expected += product;
            sum_abs += product.abs();
        }
        let error = (out[idx] as f64 - expected).abs();
        // FP32 accumulation error grows with K; keep a nonzero absolute floor.
        let coefficient = if fp16 {
            0.0012
        } else {
            (shape.k as f64 * f32::EPSILON as f64 * 2.0).max(2e-6)
        };
        let tolerance = 2e-5 + coefficient * sum_abs;
        max_abs_error = max_abs_error.max(error);
        max_scaled_error = max_scaled_error.max(error / tolerance);
        ensure!(error <= tolerance, "mismatch at row {row}, col {col}: got {}, reference {expected}, absolute error {error}, allowed {tolerance}", out[idx]);
    }
    Ok(Validation {
        checked_elements: count,
        total_elements: out.len(),
        all_elements_finite: true,
        max_abs_error,
        max_scaled_error,
        reference: "FP64 CPU dot products over original FP32 input",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_dimensions() {
        assert!(Shape::new(0, 1, 1).is_err());
        assert!(Shape::new(u32::MAX, 2, 1).is_err());
    }
    #[test]
    fn oracle_catches_wrong_and_nonfinite_output() {
        let shape = Shape::new(2, 3, 2).unwrap();
        let a = vec![1., 2., -3., 4.];
        let b = vec![1., 2., 3., 4., 5., 6.];
        let mut c = vec![9., 12., 15., 13., 14., 15.];
        assert!(check(shape, &a, &b, &c, usize::MAX, false).is_ok());
        c[5] = 0.;
        assert!(check(shape, &a, &b, &c, usize::MAX, false).is_err());
        c[5] = f32::NAN;
        assert!(check(shape, &a, &b, &c, 1, false).is_err());
    }
}
