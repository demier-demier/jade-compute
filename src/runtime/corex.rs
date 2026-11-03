//! Native CoreX runtime and generated-kernel loader. No Vulkan/WebGPU fallback.
//!
//! The SDK, compiler, architecture and bridge are explicit inputs. Only kernels
//! emitted by validated KernelSpec are compiled. Device execution has not been
//! validated on Iluvatar hardware in the local macOS development environment.
use super::{ir::KernelSpec, Buffer, Dispatch, Execution, Runtime, RuntimeInfo};
use anyhow::{bail, ensure, Context as _, Result};
use libloading::Library;
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{c_char, c_int, c_void, OsString},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    ptr::NonNull,
    rc::{Rc, Weak},
    sync::atomic::{AtomicU64, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

type Create = unsafe extern "C" fn(c_int, *mut *mut c_void, *mut c_char, usize) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Info = unsafe extern "C" fn(*mut c_void, *mut c_char, usize, *mut c_char, usize) -> c_int;
type Alloc =
    unsafe extern "C" fn(*mut c_void, usize, *mut *mut c_void, *mut c_char, usize) -> c_int;
type Free = unsafe extern "C" fn(*mut c_void, *mut c_void);
type Upload = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const c_void,
    usize,
    *mut c_char,
    usize,
) -> c_int;
type Download = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *mut c_void,
    usize,
    *mut c_char,
    usize,
) -> c_int;
type Launch = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> c_int;
type Execute = unsafe extern "C" fn(
    *mut c_void,
    *const NativeJob,
    usize,
    u32,
    *mut f64,
    *mut c_char,
    usize,
) -> c_int;

#[repr(C)]
struct NativeJob {
    launch: Launch,
    buffers: *mut *mut c_void,
    buffer_count: usize,
}

// Rc deliberately keeps runtime state, buffers and kernels !Send and !Sync.
struct NativeContext {
    raw: NonNull<c_void>,
    alloc: Alloc,
    free: Free,
    upload: Upload,
    download: Download,
    execute: Execute,
    destroy: Destroy,
    device_info: String,
    device: i32,
    _library: Library,
}
impl Drop for NativeContext {
    fn drop(&mut self) {
        // SAFETY: Rc ownership ensures buffers/kernels have already released this
        // context. The bridge remains loaded until all fields are dropped.
        unsafe { (self.destroy)(self.raw.as_ptr()) };
    }
}

pub struct CorexBuffer {
    raw: NonNull<c_void>,
    elements: usize,
    identity: u64,
    context: Rc<NativeContext>,
}
impl Buffer for CorexBuffer {
    fn len(&self) -> usize {
        self.elements
    }
    fn id(&self) -> u64 {
        self.identity
    }
}
impl Drop for CorexBuffer {
    fn drop(&mut self) {
        // SAFETY: pointer is owned by this buffer and belongs to the live context.
        unsafe { (self.context.free)(self.context.raw.as_ptr(), self.raw.as_ptr()) };
    }
}

struct OwnedDirectory(PathBuf);
impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        // This path was created exclusively by this runtime. Never remove the
        // caller's cache root or any directory that predated the session.
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct CompilerState {
    compiler: PathBuf,
    sdk: PathBuf,
    cudart: PathBuf,
    architecture: String,
    identity: String,
    directory: OwnedDirectory,
    next_module: AtomicU64,
    // Full strings are compared; no truncated-hash collision can select a kernel.
    cache: RefCell<HashMap<String, Weak<CompiledKernel>>>,
}

struct CompiledKernel {
    launch: Launch,
    spec: KernelSpec,
    // Drop the loaded module before releasing its context and private files.
    _library: Library,
    context: Rc<NativeContext>,
    _compiler: Rc<CompilerState>,
}

#[derive(Clone)]
pub struct CorexKernel(Rc<CompiledKernel>);

pub struct CorexRuntime {
    context: Rc<NativeContext>,
    compiler: Rc<CompilerState>,
}

static BUFFER_ID: AtomicU64 = AtomicU64::new(1);
static DIRECTORY_ID: AtomicU64 = AtomicU64::new(1);

fn message(bytes: &[c_char]) -> String {
    let bytes: Vec<u8> = bytes
        .iter()
        .take_while(|&&x| x != 0)
        .map(|&x| x as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
fn status(code: c_int, error: &[c_char]) -> Result<()> {
    ensure!(
        code == 0,
        "CoreX runtime bridge error {code}: {}",
        message(error)
    );
    Ok(())
}
fn file_identity(path: &Path) -> Result<String> {
    let meta = fs::metadata(path)?;
    ensure!(meta.is_file(), "expected a file: {}", path.display());
    let modified = meta.modified()?.duration_since(UNIX_EPOCH)?.as_nanos();
    Ok(format!("{}:{}:{modified}", path.display(), meta.len()))
}
fn private_directory(root: &Path) -> Result<OwnedDirectory> {
    fs::create_dir_all(root)
        .with_context(|| format!("create compiler cache root {}", root.display()))?;
    let root = root.canonicalize()?;
    ensure!(root.is_dir(), "compiler cache root is not a directory");
    let time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    for _ in 0..32 {
        let sequence = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "jade-corex-{}-{time}-{sequence}",
            std::process::id()
        ));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&path) {
            Ok(()) => return Ok(OwnedDirectory(path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).context("create private compiler directory"),
        }
    }
    bail!("cannot create a unique private compiler directory")
}

impl CorexRuntime {
    /// Loads an explicitly selected trusted native bridge and SDK compiler.
    /// Both paths execute native code; use the bridge built from this project and
    /// an authentic vendor SDK. No SDK, driver, or compiler is installed here.
    ///
    /// # Safety
    /// `library` must be a trusted implementation of jade_runtime_corex.h ABI 1.
    /// `compiler` and `sdk_root` must identify a trusted, mutually compatible
    /// Iluvatar SDK toolchain. Loading the bridge and generated libraries executes
    /// native code; an untrusted or ABI-incompatible implementation can violate
    /// Rust memory safety even if its reported ABI version is 1.
    pub unsafe fn new(
        library: &Path,
        compiler: &Path,
        sdk_root: &Path,
        architecture: &str,
        cache_dir: &Path,
        device: i32,
    ) -> Result<Self> {
        ensure!(cfg!(target_os = "linux"), "CoreX runtime requires Linux with the Iluvatar SDK and GPU; this host cannot validate it");
        ensure!(device >= 0, "device index must be nonnegative");
        ensure!(!architecture.is_empty() && architecture.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')),
                "provide the explicit vendor architecture from the target SDK; only letters, numbers, _, - and . are allowed");
        let sdk = sdk_root
            .canonicalize()
            .context("CoreX SDK directory not found")?;
        ensure!(sdk.is_dir(), "CoreX SDK root must be a directory");
        let header = sdk
            .join("include/cuda_runtime.h")
            .canonicalize()
            .context("CoreX cuda_runtime.h is missing")?;
        ensure!(
            header.starts_with(&sdk),
            "CoreX runtime header resolves outside SDK"
        );
        let compiler = compiler
            .canonicalize()
            .context("CoreX compiler not found")?;
        // An explicit external compiler is supported, but its identity and output
        // are captured. The driver must support the vendor's ivcore target.
        let compiler_identity = file_identity(&compiler)?;
        let version = Command::new(&compiler)
            .arg("--version")
            .output()
            .context("run CoreX compiler --version")?;
        ensure!(
            version.status.success(),
            "CoreX compiler --version failed: {}",
            String::from_utf8_lossy(&version.stderr)
        );
        let cudart = ["lib64", "lib", "lib/x64"]
            .iter()
            .map(|p| sdk.join(p).join("libcudart.so"))
            .find(|p| p.is_file())
            .context("CoreX libcudart.so missing under SDK lib64/lib/lib/x64")?
            .canonicalize()?;
        ensure!(
            cudart.starts_with(&sdk),
            "CoreX runtime library resolves outside SDK"
        );
        let identity = format!("compiler={compiler_identity}\nversion={}\nSDK={}\nheader={}\nruntime={}\narch={architecture}",
            String::from_utf8_lossy(&version.stdout), sdk.display(), file_identity(&header)?, file_identity(&cudart)?);
        let library_path = library.canonicalize().context(
            "native runtime bridge not found; build scripts/build-corex-runtime.sh on Linux",
        )?;
        ensure!(library_path.is_file(), "runtime bridge path must be a file");
        // SAFETY: explicit trusted bridge path; every loaded signature matches
        // native/jade_runtime_corex.h, whose ABI version is checked first.
        let context = unsafe {
            let library = Library::new(&library_path)?;
            let version = *library.get::<unsafe extern "C" fn() -> u32>(b"jr_version\0")?;
            ensure!(version() == 1, "unsupported native CoreX runtime ABI");
            let create = *library.get::<Create>(b"jr_create\0")?;
            let destroy = *library.get::<Destroy>(b"jr_destroy\0")?;
            let info = *library.get::<Info>(b"jr_info\0")?;
            let alloc = *library.get::<Alloc>(b"jr_alloc\0")?;
            let free = *library.get::<Free>(b"jr_free\0")?;
            let upload = *library.get::<Upload>(b"jr_upload\0")?;
            let download = *library.get::<Download>(b"jr_download\0")?;
            let execute = *library.get::<Execute>(b"jr_execute\0")?;
            let mut raw = std::ptr::null_mut();
            let mut error = [0 as c_char; 1024];
            status(
                create(device, &mut raw, error.as_mut_ptr(), error.len()),
                &error,
            )?;
            let raw = NonNull::new(raw).context("native bridge returned a null context")?;
            let mut ctx = NativeContext {
                raw,
                alloc,
                free,
                upload,
                download,
                execute,
                destroy,
                device_info: String::new(),
                device,
                _library: library,
            };
            let mut output = [0 as c_char; 2048];
            status(
                info(
                    raw.as_ptr(),
                    output.as_mut_ptr(),
                    output.len(),
                    error.as_mut_ptr(),
                    error.len(),
                ),
                &error,
            )?;
            ctx.device_info = message(&output);
            Rc::new(ctx)
        };
        let directory = private_directory(cache_dir)?;
        let compiler = Rc::new(CompilerState {
            compiler,
            sdk,
            cudart,
            architecture: architecture.to_string(),
            identity,
            directory,
            next_module: AtomicU64::new(0),
            cache: RefCell::new(HashMap::new()),
        });
        Ok(Self { context, compiler })
    }

    fn check_buffer(&self, buffer: &CorexBuffer) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.context, &buffer.context),
            "buffer belongs to another runtime/device context"
        );
        ensure!(
            self.context.device == buffer.context.device,
            "buffer device mismatch"
        );
        Ok(())
    }
}

impl Runtime for CorexRuntime {
    type Buffer = CorexBuffer;
    type Kernel = CorexKernel;

    fn info(&self) -> RuntimeInfo {
        RuntimeInfo {
            backend: "CoreX native generated kernels".to_string(),
            device: self.context.device_info.clone(),
            execution: format!("CoreX ivcore compiler; explicit architecture {}; ordered launches on one persistent stream", self.compiler.architecture),
            limitations: vec![
                "CoreX SDK compilation and GPU execution were not validated on the development Mac".to_string(),
                "Only validated KernelSpec operations and FP32 buffers; no CUDA binary compatibility".to_string(),
                "Exact internal arithmetic and target support require SDK/hardware validation".to_string(),
                "Synchronous API; ordered replay uses host launches, not a CUDA/CoreX Graph API".to_string(),
            ],
        }
    }

    fn alloc(&self, elements: usize) -> Result<CorexBuffer> {
        ensure!(elements > 0, "buffer allocation must be nonempty");
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .context("buffer size overflow")?;
        ensure!(
            bytes <= isize::MAX as usize,
            "buffer exceeds Rust allocation limit"
        );
        let identity = BUFFER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .map_err(|_| anyhow::anyhow!("buffer identity space exhausted"))?;
        let mut raw = std::ptr::null_mut();
        let mut error = [0 as c_char; 1024];
        // SAFETY: live context, writable output/error pointers and checked size.
        status(
            unsafe {
                (self.context.alloc)(
                    self.context.raw.as_ptr(),
                    bytes,
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            &error,
        )?;
        let raw = NonNull::new(raw).context("native allocator returned null")?;
        Ok(CorexBuffer {
            raw,
            elements,
            identity,
            context: self.context.clone(),
        })
    }

    fn upload(&self, buffer: &CorexBuffer, data: &[f32]) -> Result<()> {
        self.check_buffer(buffer)?;
        ensure!(
            data.len() == buffer.elements,
            "upload element count must match buffer exactly"
        );
        let mut error = [0 as c_char; 1024];
        // SAFETY: buffer provenance/length checked; bridge synchronizes before
        // returning, so no borrowed host pointer is retained by GPU work.
        status(
            unsafe {
                (self.context.upload)(
                    self.context.raw.as_ptr(),
                    buffer.raw.as_ptr(),
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data),
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            &error,
        )
    }

    fn download(&self, buffer: &CorexBuffer) -> Result<Vec<f32>> {
        self.check_buffer(buffer)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(buffer.elements)
            .context("host output allocation failed")?;
        output.resize(buffer.elements, 0.0);
        let mut error = [0 as c_char; 1024];
        // SAFETY: valid exclusive output memory; exact allocation length and
        // buffer provenance checked. Native transfer completes before return.
        status(
            unsafe {
                (self.context.download)(
                    self.context.raw.as_ptr(),
                    buffer.raw.as_ptr(),
                    output.as_mut_ptr().cast(),
                    output.len() * std::mem::size_of::<f32>(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            &error,
        )?;
        Ok(output)
    }

    fn compile(&self, spec: &KernelSpec) -> Result<CorexKernel> {
        spec.validate()?;
        let source = spec.corex_source()?;
        let key = format!(
            "{}\nsource={}\n{}",
            self.compiler.identity,
            source.len(),
            source
        );
        if let Some(compiled) = self
            .compiler
            .cache
            .borrow()
            .get(&key)
            .and_then(Weak::upgrade)
        {
            ensure!(
                Rc::ptr_eq(&self.context, &compiled.context),
                "cached kernel context mismatch"
            );
            return Ok(CorexKernel(compiled));
        }
        let serial = self
            .compiler
            .next_module
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .map_err(|_| anyhow::anyhow!("compiled module identity space exhausted"))?;
        let stem = format!("module-{serial}");
        let source_path = self.compiler.directory.0.join(format!("{stem}.cu"));
        let library_path = self.compiler.directory.0.join(format!("{stem}.so"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&source_path)?;
        file.write_all(source.as_bytes())?;
        drop(file);
        let mut sdk_option = OsString::from("--cuda-path=");
        sdk_option.push(&self.compiler.sdk);
        let output = Command::new(&self.compiler.compiler)
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-shared")
            .arg(sdk_option)
            .arg(format!("--cuda-gpu-arch={}", self.compiler.architecture))
            .arg("-I")
            .arg(self.compiler.sdk.join("include"))
            .arg("-x")
            .arg("ivcore")
            .arg(&source_path)
            .arg("-x")
            .arg("none")
            .arg(&self.compiler.cudart)
            .arg("-Xlinker")
            .arg("-rpath")
            .arg("-Xlinker")
            .arg(
                self.compiler
                    .cudart
                    .parent()
                    .context("runtime library has no directory")?,
            )
            .arg("-Wl,-z,defs")
            .arg("-o")
            .arg(&library_path)
            .output()
            .context("launch CoreX ivcore compiler")?;
        ensure!(
            output.status.success(),
            "CoreX kernel compilation failed for {:?} (architecture {}):\n{}\n{}",
            spec.name(),
            self.compiler.architecture,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let canonical = library_path
            .canonicalize()
            .context("compiler succeeded without producing kernel library")?;
        ensure!(
            canonical.starts_with(&self.compiler.directory.0) && canonical.is_file(),
            "compiled library escaped private output directory"
        );
        // SAFETY: library was produced from validated IR by the explicitly
        // selected trusted compiler, inside this runtime's private directory.
        let (library, launch) = unsafe {
            let library = Library::new(&canonical).context("load compiled CoreX module")?;
            let launch = *library
                .get::<Launch>(b"jade_generated_launch\0")
                .context("compiled module missing jade_generated_launch")?;
            (library, launch)
        };
        let compiled = Rc::new(CompiledKernel {
            launch,
            spec: spec.clone(),
            _library: library,
            context: self.context.clone(),
            _compiler: self.compiler.clone(),
        });
        self.compiler
            .cache
            .borrow_mut()
            .insert(key, Rc::downgrade(&compiled));
        Ok(CorexKernel(compiled))
    }

    fn execute(
        &self,
        jobs: &[Dispatch<'_, CorexBuffer, CorexKernel>],
        repeats: u32,
    ) -> Result<Execution> {
        ensure!(
            !jobs.is_empty() && repeats > 0,
            "execution requires jobs and positive repeats"
        );
        let dispatches = u64::try_from(jobs.len())?
            .checked_mul(u64::from(repeats))
            .context("dispatch count overflow")?;
        let mut bindings: Vec<Vec<*mut c_void>> = Vec::with_capacity(jobs.len());
        for job in jobs {
            ensure!(
                Rc::ptr_eq(&self.context, &job.kernel.0.context),
                "kernel belongs to another runtime/device context"
            );
            for buffer in &job.buffers {
                self.check_buffer(buffer)?;
            }
            let lengths: Vec<_> = job.buffers.iter().map(|b| b.len()).collect();
            let ids: Vec<_> = job.buffers.iter().map(|b| b.id()).collect();
            job.kernel.0.spec.validate_bindings(&lengths, &ids)?;
            bindings.push(job.buffers.iter().map(|b| b.raw.as_ptr()).collect());
        }
        // Complete all allocations before taking pointers into the binding arrays.
        let native_jobs: Vec<_> = jobs
            .iter()
            .zip(bindings.iter_mut())
            .map(|(job, ptrs)| NativeJob {
                launch: job.kernel.0.launch,
                buffers: ptrs.as_mut_ptr(),
                buffer_count: ptrs.len(),
            })
            .collect();
        let mut error = [0 as c_char; 1024];
        let mut gpu_ms = 0.0;
        let started = Instant::now();
        // SAFETY: all origins, dimensions and aliases were validated before any
        // launch. Jobs, buffers and loaded modules remain alive until the bridge
        // synchronizes. Rc prevents concurrent context operations across threads.
        status(
            unsafe {
                (self.context.execute)(
                    self.context.raw.as_ptr(),
                    native_jobs.as_ptr(),
                    native_jobs.len(),
                    repeats,
                    &mut gpu_ms,
                    error.as_mut_ptr(),
                    error.len(),
                )
            },
            &error,
        )?;
        let host_ms = started.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            gpu_ms.is_finite() && gpu_ms >= 0.0,
            "invalid GPU event time"
        );
        Ok(Execution {
            host_ms,
            // Zero is below the device timer's resolution, not a usable duration.
            gpu_ms: (gpu_ms > 0.0).then_some(gpu_ms),
            dispatches,
            repeats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiler_directories_are_private_and_cleanup_is_scoped() {
        let root = private_directory(&std::env::temp_dir()).unwrap();
        let sentinel = root.0.join("unrelated-file");
        fs::write(&sentinel, "preserve").unwrap();
        let first = private_directory(&root.0).unwrap();
        let second = private_directory(&root.0).unwrap();
        assert_ne!(first.0, second.0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&first.0).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        let first_path = first.0.clone();
        drop(first);
        assert!(!first_path.exists());
        assert!(second.0.is_dir());
        assert_eq!(fs::read_to_string(&sentinel).unwrap(), "preserve");
    }

    #[test]
    fn unsupported_host_fails_before_loading_or_compiling() {
        if !cfg!(target_os = "linux") {
            let missing = Path::new("/this-path-must-not-be-opened");
            // SAFETY: this branch runs only on unsupported hosts, whose guard
            // returns before any path is opened or native code is loaded.
            let result = unsafe {
                CorexRuntime::new(missing, missing, missing, "ivcore-explicit", missing, 0)
            };
            assert!(result.err().unwrap().to_string().contains("requires Linux"));
        }
    }
}
