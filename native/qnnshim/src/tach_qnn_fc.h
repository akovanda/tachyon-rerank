// tach_qnn_fc.h  — very small C ABI for a single MatMul (A @ q)
// Build against Qualcomm AI Engine Direct SDK ("QNN SDK").
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tach_qnn_fc_ctx tach_qnn_fc_ctx;

// Create a context for (n,d). A and q are provided at run time.
// backend: "htp" (default) or "cpu".
// use_fp16: 1 to prefer FP16 on HTP if supported (best-effort).
tach_qnn_fc_ctx* tach_qnn_fc_create(int32_t n, int32_t d,
                                    const char* backend,
                                    int use_fp16);
tach_qnn_fc_ctx* tach_qnn_fc_create_batched(int32_t n, int32_t d, int32_t q_batch,
                                            const char* backend,
                                            int use_fp16);

// Create a context with static A (copied at creation). q is provided at run time.
tach_qnn_fc_ctx* tach_qnn_fc_create_static_a(int32_t n, int32_t d,
                                             const float* A,
                                             const char* backend,
                                             int use_fp16);
tach_qnn_fc_ctx* tach_qnn_fc_create_static_a_batched(int32_t n, int32_t d, int32_t q_batch,
                                                     const float* A,
                                                     const char* backend,
                                                     int use_fp16);

// Run y = A @ q. A is (n,d) row-major, q is (d,1), out is (n,1).
// Returns 0 on success; <0 on error.
int tach_qnn_fc_run(tach_qnn_fc_ctx* ctx,
                    const float* A,
                    const float* q,
                    float* out);

// Run y = A @ q for static-A contexts (q only).
int tach_qnn_fc_run_q(tach_qnn_fc_ctx* ctx,
                      const float* q,
                      float* out);

// Destroy and free resources.
void tach_qnn_fc_destroy(tach_qnn_fc_ctx* ctx);

#ifdef __cplusplus
}
#endif
