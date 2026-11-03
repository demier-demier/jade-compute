#ifndef JADE_RUNTIME_COREX_H
#define JADE_RUNTIME_COREX_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef int (*jr_launch_fn)(void *stream, void **buffers);
typedef struct jr_job {
    jr_launch_fn launch;
    void **buffers;
    size_t buffer_count;
} jr_job;

/* ABI 1. All contexts and their allocations require serial access. The caller
 * retains host buffers and compiled launcher libraries through execute. */
uint32_t jr_version(void);
int jr_create(int device, void **out, char *err, size_t cap);
void jr_destroy(void *ctx);
int jr_info(void *ctx, char *out, size_t out_cap, char *err, size_t cap);
int jr_alloc(void *ctx, size_t bytes, void **out, char *err, size_t cap);
void jr_free(void *ctx, void *ptr);
int jr_upload(void *ctx, void *device_ptr, const void *host_ptr, size_t bytes,
              char *err, size_t cap);
int jr_download(void *ctx, const void *device_ptr, void *host_ptr, size_t bytes,
                char *err, size_t cap);
/* Executes the ordered jobs sequence repeats times. GPU-event ms covers the
 * complete sequence including any submission gaps, excluding transfers. */
int jr_execute(void *ctx, const jr_job *jobs, size_t job_count, uint32_t repeats,
               double *gpu_ms, char *err, size_t cap);
#ifdef __cplusplus
}
#endif
#endif
