#include "jade_corex.h"
#include "fp16.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <limits>
#include <new>

namespace {
struct Context {
    int m = 0, n = 0, k = 0, device = 0;
    uint32_t precision = 0;
    size_t a_len = 0, b_len = 0, c_len = 0;
    size_t a_bytes = 0, b_bytes = 0, c_bytes = 0;
    void *a = nullptr, *b = nullptr;
    float *c = nullptr;
    uint16_t *host_a = nullptr, *host_b = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cublasHandle_t blas = nullptr;
    bool uploaded = false, computed = false;
};

void clear_error(char *err, size_t cap) {
    if (err && cap) err[0] = '\0';
}

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

int runtime_error(cudaError_t status, char *err, size_t cap, const char *op) {
    return fail(2, err, cap, "%s: %s (runtime code %d)", op,
                cudaGetErrorString(status), static_cast<int>(status));
}

int blas_error(cublasStatus_t status, char *err, size_t cap, const char *op) {
    return fail(3, err, cap, "%s: BLAS status %d", op, static_cast<int>(status));
}

int select(Context *ctx, char *err, size_t cap) {
    if (!ctx) return fail(1, err, cap, "null context");
    const auto status = cudaSetDevice(ctx->device);
    return status == cudaSuccess ? 0 : runtime_error(status, err, cap, "cudaSetDevice");
}

bool matrix_size(uint32_t x, uint32_t y, size_t element_size,
                 size_t &count, size_t &bytes) {
    const auto maximum = std::numeric_limits<size_t>::max();
    if (!x || !y || static_cast<size_t>(x) > maximum / y) return false;
    count = static_cast<size_t>(x) * y;
    if (count > maximum / element_size) return false;
    bytes = count * element_size;
    return true;
}

void cleanup(Context *ctx) {
    if (!ctx) return;
    // Best effort: destroy must not throw across a C/Rust boundary.
    cudaSetDevice(ctx->device);
    if (ctx->stream) cudaStreamSynchronize(ctx->stream);
    if (ctx->blas) cublasDestroy(ctx->blas);
    if (ctx->start) cudaEventDestroy(ctx->start);
    if (ctx->stop) cudaEventDestroy(ctx->stop);
    if (ctx->a) cudaFree(ctx->a);
    if (ctx->b) cudaFree(ctx->b);
    if (ctx->c) cudaFree(ctx->c);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete[] ctx->host_a;
    delete[] ctx->host_b;
    delete ctx;
}

cublasStatus_t gemm(Context *ctx) {
    const float alpha = 1.0f, beta = 0.0f;
    // BLAS is column-major. Row-major C = A B is column-major C^T = B^T A^T.
    // The same buffers are reused: dimensions (n,m,k), leading dims (n,k,n).
    if (ctx->precision == 0) {
        return cublasSgemm(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                          ctx->n, ctx->m, ctx->k, &alpha,
                          static_cast<const float *>(ctx->b), ctx->n,
                          static_cast<const float *>(ctx->a), ctx->k,
                          &beta, ctx->c, ctx->n);
    }
#ifdef JADE_COREX_MODERN_COMPUTE_TYPE
    constexpr auto compute = CUBLAS_COMPUTE_32F;
#else
    constexpr auto compute = CUDA_R_32F;
#endif
    return cublasGemmEx(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        ctx->n, ctx->m, ctx->k, &alpha,
                        ctx->b, CUDA_R_16F, ctx->n,
                        ctx->a, CUDA_R_16F, ctx->k,
                        &beta, ctx->c, CUDA_R_32F, ctx->n,
                        compute, CUBLAS_GEMM_DEFAULT);
}
} // namespace

extern "C" uint32_t jade_corex_abi_version(void) { return 1; }

extern "C" int jade_corex_create(uint32_t m, uint32_t n, uint32_t k,
                                  int device, uint32_t precision, void **out,
                                  char *err, size_t cap) {
    clear_error(err, cap);
    if (!out) return fail(1, err, cap, "null output handle");
    *out = nullptr;
    if (!m || !n || !k || m > INT_MAX || n > INT_MAX || k > INT_MAX || device < 0 || precision > 1) {
        return fail(1, err, cap, "dimensions must be 1..INT_MAX; device >= 0; precision 0 or 1");
    }
    Context *ctx = new (std::nothrow) Context();
    if (!ctx) return fail(4, err, cap, "context allocation failed");
    ctx->m = static_cast<int>(m);
    ctx->n = static_cast<int>(n);
    ctx->k = static_cast<int>(k);
    ctx->device = device;
    ctx->precision = precision;
    const size_t input_size = precision ? sizeof(uint16_t) : sizeof(float);
    if (!matrix_size(m, k, input_size, ctx->a_len, ctx->a_bytes) ||
        !matrix_size(k, n, input_size, ctx->b_len, ctx->b_bytes) ||
        !matrix_size(m, n, sizeof(float), ctx->c_len, ctx->c_bytes) ||
        ctx->a_len > std::numeric_limits<size_t>::max() / sizeof(float) ||
        ctx->b_len > std::numeric_limits<size_t>::max() / sizeof(float)) {
        delete ctx;
        return fail(1, err, cap, "matrix byte count overflows size_t");
    }
    const auto initialize = [&]() -> int {
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaGetDeviceCount");
        if (device >= count) return fail(1, err, cap, "device %d outside detected count %d", device, count);
        int rc = select(ctx, err, cap);
        if (rc) return rc;
        status = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
        if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaStreamCreateWithFlags");
        status = cudaEventCreate(&ctx->start);
        if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaEventCreate(start)");
        status = cudaEventCreate(&ctx->stop);
        if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaEventCreate(stop)");
        auto bs = cublasCreate(&ctx->blas);
        if (bs != CUBLAS_STATUS_SUCCESS) return blas_error(bs, err, cap, "cublasCreate");
        bs = cublasSetStream(ctx->blas, ctx->stream);
        if (bs != CUBLAS_STATUS_SUCCESS) return blas_error(bs, err, cap, "cublasSetStream");
        bs = cublasSetPointerMode(ctx->blas, CUBLAS_POINTER_MODE_HOST);
        if (bs != CUBLAS_STATUS_SUCCESS) return blas_error(bs, err, cap, "cublasSetPointerMode");
#ifdef JADE_COREX_HAS_MATH_MODE
        bs = cublasSetMathMode(ctx->blas, precision ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
        if (bs != CUBLAS_STATUS_SUCCESS) return blas_error(bs, err, cap, "cublasSetMathMode");
#endif
        status = cudaMalloc(&ctx->a, ctx->a_bytes);
        if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaMalloc(A)");
        status = cudaMalloc(&ctx->b, ctx->b_bytes);
        if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaMalloc(B)");
        status = cudaMalloc(reinterpret_cast<void **>(&ctx->c), ctx->c_bytes);
        if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaMalloc(C)");
        if (precision) {
            ctx->host_a = new (std::nothrow) uint16_t[ctx->a_len];
            ctx->host_b = new (std::nothrow) uint16_t[ctx->b_len];
            if (!ctx->host_a || !ctx->host_b) return fail(4, err, cap, "FP16 staging allocation failed");
        }
        return 0;
    };
    const int rc = initialize();
    if (rc) { cleanup(ctx); return rc; }
    *out = ctx;
    return 0;
}

extern "C" int jade_corex_upload(void *handle, const float *a, size_t a_len,
                                  const float *b, size_t b_len, char *err, size_t cap) {
    clear_error(err, cap);
    auto *ctx = static_cast<Context *>(handle);
    if (!ctx || !a || !b || a_len != ctx->a_len || b_len != ctx->b_len) {
        return fail(1, err, cap, "upload requires non-null buffers and exact A/B element counts");
    }
    ctx->uploaded = false;
    ctx->computed = false;
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    const void *src_a = a, *src_b = b;
    if (ctx->precision) {
        for (size_t i = 0; i < a_len; ++i) ctx->host_a[i] = jade::to_fp16(a[i]);
        for (size_t i = 0; i < b_len; ++i) ctx->host_b[i] = jade::to_fp16(b[i]);
        src_a = ctx->host_a;
        src_b = ctx->host_b;
    }
    auto status = cudaMemcpyAsync(ctx->a, src_a, ctx->a_bytes, cudaMemcpyHostToDevice, ctx->stream);
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(ctx->b, src_b, ctx->b_bytes, cudaMemcpyHostToDevice, ctx->stream);
    }
    // Synchronize even on a partial-copy failure before releasing borrowed input.
    const auto sync = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaMemcpyAsync(upload)");
    if (sync != cudaSuccess) return runtime_error(sync, err, cap, "cudaStreamSynchronize(upload)");
    ctx->uploaded = true;
    return 0;
}

extern "C" int jade_corex_run(void *handle, uint32_t iterations,
                               double *elapsed_ms, char *err, size_t cap) {
    clear_error(err, cap);
    if (elapsed_ms) *elapsed_ms = 0;
    auto *ctx = static_cast<Context *>(handle);
    if (!ctx || !iterations || !elapsed_ms) return fail(1, err, cap, "run requires context, iterations > 0 and elapsed output");
    ctx->computed = false;
    if (!ctx->uploaded) return fail(1, err, cap, "upload A/B before running");
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    auto status = cudaEventRecord(ctx->start, ctx->stream);
    if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaEventRecord(start)");
    for (uint32_t i = 0; i < iterations; ++i) {
        const auto bs = gemm(ctx);
        if (bs != CUBLAS_STATUS_SUCCESS) {
            cudaStreamSynchronize(ctx->stream);
            return blas_error(bs, err, cap, "matrix multiplication");
        }
    }
    status = cudaEventRecord(ctx->stop, ctx->stream);
    if (status != cudaSuccess) {
        cudaStreamSynchronize(ctx->stream);
        return runtime_error(status, err, cap, "cudaEventRecord(stop)");
    }
    status = cudaEventSynchronize(ctx->stop);
    if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaEventSynchronize");
    float milliseconds = 0;
    status = cudaEventElapsedTime(&milliseconds, ctx->start, ctx->stop);
    if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaEventElapsedTime");
    if (!std::isfinite(milliseconds) || milliseconds < 0) return fail(2, err, cap, "invalid GPU event duration");
    *elapsed_ms = static_cast<double>(milliseconds);
    ctx->computed = true;
    return 0;
}

extern "C" int jade_corex_download(void *handle, float *out, size_t out_len,
                                    char *err, size_t cap) {
    clear_error(err, cap);
    auto *ctx = static_cast<Context *>(handle);
    if (!ctx || !out || out_len != ctx->c_len) return fail(1, err, cap, "download requires exact C element count");
    if (!ctx->computed) return fail(1, err, cap, "run successfully before downloading C");
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    const auto status = cudaMemcpyAsync(out, ctx->c, ctx->c_bytes, cudaMemcpyDeviceToHost, ctx->stream);
    const auto sync = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaMemcpyAsync(download)");
    return sync == cudaSuccess ? 0 : runtime_error(sync, err, cap, "cudaStreamSynchronize(download)");
}

extern "C" int jade_corex_device_info(void *handle, char *out, size_t out_cap,
                                       char *err, size_t cap) {
    clear_error(err, cap);
    if (out && out_cap) out[0] = '\0';
    auto *ctx = static_cast<Context *>(handle);
    if (!ctx || !out || !out_cap) return fail(1, err, cap, "device info requires context and output buffer");
    const int rc = select(ctx, err, cap);
    if (rc) return rc;
    cudaDeviceProp prop{};
    auto status = cudaGetDeviceProperties(&prop, ctx->device);
    if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaGetDeviceProperties");
    int runtime = 0, driver = 0, blas = 0;
    status = cudaRuntimeGetVersion(&runtime);
    if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaRuntimeGetVersion");
    status = cudaDriverGetVersion(&driver);
    if (status != cudaSuccess) return runtime_error(status, err, cap, "cudaDriverGetVersion");
    const auto bs = cublasGetVersion(ctx->blas, &blas);
    if (bs != CUBLAS_STATUS_SUCCESS) return blas_error(bs, err, cap, "cublasGetVersion");
    const int needed = std::snprintf(out, out_cap,
        "device=%d name=%.255s memory_bytes=%llu multiprocessors=%d runtime_api=%d driver_api=%d blas_api=%d precision=%s",
        ctx->device, prop.name, static_cast<unsigned long long>(prop.totalGlobalMem),
        prop.multiProcessorCount, runtime, driver, blas,
        ctx->precision ? "fp16_inputs_fp32_accum_output" : "fp32");
    if (needed < 0 || static_cast<size_t>(needed) >= out_cap) {
        out[out_cap - 1] = '\0';
        return fail(1, err, cap, "device info output buffer too small");
    }
    return 0;
}

extern "C" void jade_corex_destroy(void *handle) {
    cleanup(static_cast<Context *>(handle));
}
