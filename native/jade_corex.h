#ifndef JADE_COREX_H
#define JADE_COREX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI v1. Contexts require serial access. Every matrix is dense, row-major.
 * 0 = success; 1 = invalid input/state; 2 = runtime; 3 = BLAS; 4 = allocation.
 * Errors are copied into caller storage and NUL-terminated when capacity > 0.
 * The caller retains all host arrays. Upload/run/download synchronize before
 * returning; no caller-owned pointer is retained. Destroy is null-safe.
 * precision: 0 = FP32; 1 = FP16 A/B, FP32 accumulation and C. */
uint32_t jade_corex_abi_version(void);
int jade_corex_create(uint32_t m, uint32_t n, uint32_t k, int device,
                     uint32_t precision, void **out, char *err, size_t cap);
int jade_corex_upload(void *ctx, const float *a, size_t a_len,
                     const float *b, size_t b_len, char *err, size_t cap);
/* elapsed_ms is total GPU-event time for all iterations, excluding transfers. */
int jade_corex_run(void *ctx, uint32_t iterations, double *elapsed_ms,
                  char *err, size_t cap);
int jade_corex_download(void *ctx, float *out, size_t out_len,
                       char *err, size_t cap);
int jade_corex_device_info(void *ctx, char *out, size_t out_cap,
                          char *err, size_t cap);
void jade_corex_destroy(void *ctx);

#ifdef __cplusplus
}
#endif
#endif
