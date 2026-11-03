//! Explicitly loaded C ABI bridge; no guessed vendor structs or enum discriminants.
use crate::{MatmulPlan, Shape, Timing};
use anyhow::{bail, ensure, Context, Result};
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_void},
    path::Path,
    ptr::NonNull,
    time::Instant,
};

type Create =
    unsafe extern "C" fn(u32, u32, u32, c_int, u32, *mut *mut c_void, *mut c_char, usize) -> c_int;
type Upload = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    usize,
    *const f32,
    usize,
    *mut c_char,
    usize,
) -> c_int;
type Run = unsafe extern "C" fn(*mut c_void, u32, *mut f64, *mut c_char, usize) -> c_int;
type Download = unsafe extern "C" fn(*mut c_void, *mut f32, usize, *mut c_char, usize) -> c_int;
type Info = unsafe extern "C" fn(*mut c_void, *mut c_char, usize, *mut c_char, usize) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void);

pub struct CorexPlan {
    ctx: NonNull<c_void>,
    run_fn: Run,
    download_fn: Download,
    destroy_fn: Destroy,
    shape: Shape,
    fp16: bool,
    pub info: String,
    // Must remain loaded until the context has been destroyed.
    _library: Library,
}
fn text(bytes: &[c_char]) -> String {
    let bytes: Vec<u8> = bytes
        .iter()
        .take_while(|&&v| v != 0)
        .map(|&v| v as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
fn status(code: c_int, error: &[c_char]) -> Result<()> {
    if code != 0 {
        bail!("CoreX bridge error {code}: {}", text(error));
    }
    Ok(())
}
impl CorexPlan {
    /// # Safety
    /// `path` must identify a trusted native library implementing jade_corex.h ABI v1.
    /// Dynamic libraries execute code on load. Use the bridge built from this project.
    pub unsafe fn load(
        path: &Path,
        shape: Shape,
        device: i32,
        fp16: bool,
        a: &[f32],
        b: &[f32],
    ) -> Result<Self> {
        Shape::new(shape.m, shape.n, shape.k)?;
        ensure!(device >= 0, "device index must be nonnegative");
        ensure!(
            a.len() == shape.a_len() && b.len() == shape.b_len(),
            "input shape mismatch"
        );
        let path = path.canonicalize().context(
            "CoreX bridge not found; build scripts/build-corex.sh on the target Linux host",
        )?;
        // SAFETY: caller explicitly supplies and trusts this native bridge.
        let library =
            Library::new(&path).with_context(|| format!("load CoreX bridge {}", path.display()))?;
        let abi = *library.get::<unsafe extern "C" fn() -> u32>(b"jade_corex_abi_version\0")?;
        ensure!(abi() == 1, "unsupported CoreX bridge ABI");
        let create = *library.get::<Create>(b"jade_corex_create\0")?;
        let upload = *library.get::<Upload>(b"jade_corex_upload\0")?;
        let run_fn = *library.get::<Run>(b"jade_corex_run\0")?;
        let download_fn = *library.get::<Download>(b"jade_corex_download\0")?;
        let destroy_fn = *library.get::<Destroy>(b"jade_corex_destroy\0")?;
        let info_fn = *library.get::<Info>(b"jade_corex_device_info\0")?;
        let mut error = [0 as c_char; 1024];
        let mut raw = std::ptr::null_mut();
        status(
            create(
                shape.m,
                shape.n,
                shape.k,
                device,
                u32::from(fp16),
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            ),
            &error,
        )?;
        let ctx = NonNull::new(raw).context("CoreX returned a null context")?;
        // Construct owner before any further fallible operation to avoid leaking context.
        let mut plan = Self {
            ctx,
            shape,
            fp16,
            run_fn,
            download_fn,
            destroy_fn,
            info: String::new(),
            _library: library,
        };
        status(
            upload(
                raw,
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
                error.as_mut_ptr(),
                error.len(),
            ),
            &error,
        )?;
        let mut info = [0 as c_char; 2048];
        status(
            info_fn(
                raw,
                info.as_mut_ptr(),
                info.len(),
                error.as_mut_ptr(),
                error.len(),
            ),
            &error,
        )?;
        plan.info = text(&info);
        Ok(plan)
    }
}
impl MatmulPlan for CorexPlan {
    fn name(&self) -> &str {
        if self.fp16 {
            "corex-blas-fp16-fp32acc"
        } else {
            "corex-blas-fp32"
        }
    }
    fn run(&mut self, iterations: u32) -> Result<Timing> {
        ensure!(iterations > 0, "iterations must be positive");
        let mut error = [0 as c_char; 1024];
        let mut ms = 0.0;
        let start = Instant::now();
        // SAFETY: context owned, live, serial access; bridge ABI checked at construction.
        status(
            unsafe {
                (self.run_fn)(
                    self.ctx.as_ptr(),
                    iterations,
                    &mut ms,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            &error,
        )?;
        let wall_ms_per_dispatch = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        ensure!(ms.is_finite() && ms > 0.0, "invalid CoreX event time");
        Ok(Timing {
            wall_ms_per_dispatch,
            gpu_ms_per_dispatch: Some(ms / iterations as f64),
        })
    }
    fn download(&self) -> Result<Vec<f32>> {
        let mut error = [0 as c_char; 1024];
        let mut output = vec![0.0; self.shape.c_len()];
        // SAFETY: output allocation has exactly the element count passed to the bridge.
        status(
            unsafe {
                (self.download_fn)(
                    self.ctx.as_ptr(),
                    output.as_mut_ptr(),
                    output.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            &error,
        )?;
        Ok(output)
    }
}
impl Drop for CorexPlan {
    fn drop(&mut self) {
        // SAFETY: destroy runs once before the native library is dropped.
        unsafe {
            (self.destroy_fn)(self.ctx.as_ptr());
        }
    }
}
