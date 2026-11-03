#include "jade_runtime_corex.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <new>

namespace {
struct Context {
    int device = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
};

int fail(int code, char *err, size_t cap, const char *format, ...) {
    if (err && cap) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(err, cap, format, args);
        va_end(args);
        err[cap - 1] = '\0';
    }
    return code;
}
void clear(char *err, size_t cap) { if (err && cap) err[0] = '\0'; }
int cuda_fail(cudaError_t status, char *err, size_t cap, const char *operation) {
    return fail(2, err, cap, "%s: %s (%d)", operation,
                cudaGetErrorString(status), static_cast<int>(status));
}
int select(Context *ctx, char *err, size_t cap) {
    if (!ctx) return fail(1, err, cap, "null runtime context");
    const auto status = cudaSetDevice(ctx->device);
    return status == cudaSuccess ? 0 : cuda_fail(status, err, cap, "cudaSetDevice");
}
void destroy(Context *ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->device);
    if (ctx->stream) cudaStreamSynchronize(ctx->stream);
    if (ctx->start) cudaEventDestroy(ctx->start);
    if (ctx->stop) cudaEventDestroy(ctx->stop);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete ctx;
}
} // namespace

extern "C" uint32_t jr_version(void) { return 1; }
extern "C" int jr_create(int device, void **out, char *err, size_t cap) {
    clear(err, cap);
    if (!out) return fail(1, err, cap, "null output handle");
    *out = nullptr;
    if (device < 0) return fail(1, err, cap, "negative device index");
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "cudaGetDeviceCount");
    if (device >= count) return fail(1, err, cap, "device %d outside device count %d", device, count);
    auto *ctx = new (std::nothrow) Context;
    if (!ctx) return fail(3, err, cap, "runtime context allocation failed");
    ctx->device = device;
    int rc = select(ctx, err, cap);
    if (rc) { destroy(ctx); return rc; }
    status = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (status == cudaSuccess) status = cudaEventCreate(&ctx->start);
    if (status == cudaSuccess) status = cudaEventCreate(&ctx->stop);
    if (status != cudaSuccess) {
        rc = cuda_fail(status, err, cap, "create stream/timing events");
        destroy(ctx);
        return rc;
    }
    *out = ctx;
    return 0;
}
extern "C" void jr_destroy(void *ctx) { destroy(static_cast<Context *>(ctx)); }

extern "C" int jr_info(void *handle, char *out, size_t out_cap, char *err, size_t cap) {
    clear(err, cap);
    if (out && out_cap) out[0] = '\0';
    if (!out || !out_cap) return fail(1, err, cap, "missing device-info output");
    auto *ctx = static_cast<Context *>(handle);
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    cudaDeviceProp prop{};
    auto status = cudaGetDeviceProperties(&prop, ctx->device);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "cudaGetDeviceProperties");
    int runtime = 0, driver = 0;
    status = cudaRuntimeGetVersion(&runtime);
    if (status == cudaSuccess) status = cudaDriverGetVersion(&driver);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "runtime/driver version");
    const int n = std::snprintf(out, out_cap,
        "device=%d name=%.255s memory_bytes=%llu runtime_api=%d driver_api=%d",
        ctx->device, prop.name, static_cast<unsigned long long>(prop.totalGlobalMem), runtime, driver);
    if (n < 0 || static_cast<size_t>(n) >= out_cap) {
        out[out_cap - 1] = '\0';
        return fail(1, err, cap, "device-info output too small");
    }
    return 0;
}
extern "C" int jr_alloc(void *handle, size_t bytes, void **out, char *err, size_t cap) {
    clear(err, cap);
    if (!out) return fail(1, err, cap, "null allocation output");
    *out = nullptr;
    if (!bytes) return fail(1, err, cap, "allocation must be nonempty");
    auto *ctx = static_cast<Context *>(handle);
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    auto status = cudaMalloc(out, bytes);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "cudaMalloc");
    // Public allocations contain defined zeros, even before the first upload.
    status = cudaMemsetAsync(*out, 0, bytes, ctx->stream);
    const auto sync = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess || sync != cudaSuccess) {
        cudaFree(*out);
        *out = nullptr;
        return cuda_fail(status != cudaSuccess ? status : sync, err, cap, "initialize allocation");
    }
    return 0;
}
extern "C" void jr_free(void *handle, void *ptr) {
    if (!handle || !ptr) return;
    auto *ctx = static_cast<Context *>(handle);
    if (cudaSetDevice(ctx->device) != cudaSuccess) return;
    cudaStreamSynchronize(ctx->stream);
    cudaFree(ptr);
}
extern "C" int jr_upload(void *handle, void *device_ptr, const void *host_ptr,
                           size_t bytes, char *err, size_t cap) {
    clear(err, cap);
    if (!device_ptr || !host_ptr || !bytes) return fail(1, err, cap, "invalid upload buffers/size");
    auto *ctx = static_cast<Context *>(handle);
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    const auto status = cudaMemcpyAsync(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice, ctx->stream);
    const auto sync = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "upload");
    return sync == cudaSuccess ? 0 : cuda_fail(sync, err, cap, "upload synchronization");
}
extern "C" int jr_download(void *handle, const void *device_ptr, void *host_ptr,
                             size_t bytes, char *err, size_t cap) {
    clear(err, cap);
    if (!device_ptr || !host_ptr || !bytes) return fail(1, err, cap, "invalid download buffers/size");
    auto *ctx = static_cast<Context *>(handle);
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    const auto status = cudaMemcpyAsync(host_ptr, device_ptr, bytes, cudaMemcpyDeviceToHost, ctx->stream);
    const auto sync = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "download");
    return sync == cudaSuccess ? 0 : cuda_fail(sync, err, cap, "download synchronization");
}
extern "C" int jr_execute(void *handle, const jr_job *jobs, size_t job_count,
                            uint32_t repeats, double *gpu_ms, char *err, size_t cap) {
    clear(err, cap);
    if (gpu_ms) *gpu_ms = 0;
    if (!jobs || !job_count || !repeats || !gpu_ms) return fail(1, err, cap, "empty dispatch sequence or missing timing output");
    // Validate every job before any work is submitted.
    for (size_t j = 0; j < job_count; ++j) {
        if (!jobs[j].launch || !jobs[j].buffers || !jobs[j].buffer_count) {
            return fail(1, err, cap, "invalid job %zu", j);
        }
        for (size_t b = 0; b < jobs[j].buffer_count; ++b) {
            if (!jobs[j].buffers[b]) return fail(1, err, cap, "null buffer in job %zu", j);
        }
    }
    auto *ctx = static_cast<Context *>(handle);
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    auto status = cudaEventRecord(ctx->start, ctx->stream);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "record start event");
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
        for (size_t j = 0; j < job_count; ++j) {
            status = static_cast<cudaError_t>(jobs[j].launch(static_cast<void *>(ctx->stream), jobs[j].buffers));
            if (status != cudaSuccess) {
                cudaStreamSynchronize(ctx->stream);
                return cuda_fail(status, err, cap, "generated kernel launch");
            }
        }
    }
    status = cudaEventRecord(ctx->stop, ctx->stream);
    if (status != cudaSuccess) {
        cudaStreamSynchronize(ctx->stream);
        return cuda_fail(status, err, cap, "record stop event");
    }
    status = cudaEventSynchronize(ctx->stop);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "kernel synchronization");
    float elapsed = 0;
    status = cudaEventElapsedTime(&elapsed, ctx->start, ctx->stop);
    if (status != cudaSuccess) return cuda_fail(status, err, cap, "event elapsed time");
    if (!std::isfinite(elapsed) || elapsed < 0) return fail(2, err, cap, "invalid GPU-event duration");
    *gpu_ms = static_cast<double>(elapsed);
    return 0;
}
