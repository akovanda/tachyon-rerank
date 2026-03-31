// TAG: direct-qnn-matmul-v1 (simple MatMul graph via QNN interface)
#if defined(__clang__) || defined(__GNUC__)
#pragma message("Compiling tach_qnn_fc.c: TAG=direct-qnn-matmul-v1")
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include "QNN/QnnInterface.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnTensor.h"
#include "QNN/QnnOpDef.h"

#define LOGE(...) fprintf(stderr, "[tach_qnn_fc] " __VA_ARGS__)
#define LOGI(...) fprintf(stderr, "[tach_qnn_fc] " __VA_ARGS__)
#define QIF(i) ((i)->QNN_INTERFACE_VER_NAME)

// ---------- dlsym helpers ----------
typedef Qnn_ErrorHandle_t (*PFN_QnnInterface_getProviders)(const QnnInterface_t***, uint32_t*);

static void* try_dlopen(const char* soname) {
  void* h = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    LOGE("dlopen(%s) failed: %s\n", soname, dlerror());
  }
  return h;
}

static int open_interface(const char* prefer, void** out_lib, const QnnInterface_t** out_qif) {
  const char* candidates[2];
  if (prefer && strcmp(prefer, "cpu") == 0) {
    candidates[0] = "libQnnCpu.so";
    candidates[1] = "libQnnHtp.so";
  } else {
    candidates[0] = "libQnnHtp.so";
    candidates[1] = "libQnnCpu.so";
  }

  for (int i = 0; i < 2; ++i) {
    void* lib = try_dlopen(candidates[i]);
    if (!lib) continue;

    PFN_QnnInterface_getProviders getProviders =
        (PFN_QnnInterface_getProviders)dlsym(lib, "QnnInterface_getProviders");
    if (!getProviders) {
      LOGE("dlsym(QnnInterface_getProviders) failed in %s: %s\n", candidates[i], dlerror());
      dlclose(lib);
      continue;
    }

    const QnnInterface_t** providers = NULL;
    uint32_t num = 0;
    Qnn_ErrorHandle_t err = getProviders(&providers, &num);
    if (err != QNN_SUCCESS || !providers || num == 0 || !providers[0]) {
      LOGE("QnnInterface_getProviders failed in %s (err=%lu, num=%u)\n",
           candidates[i], (unsigned long)err, num);
      dlclose(lib);
      continue;
    }

    *out_lib = lib;
    *out_qif = providers[0];
    LOGI("Using QNN provider from %s\n", candidates[i]);
    return 0;
  }

  return -1;
}

// ---------- context ----------
struct tach_qnn_fc_ctx {
  void* lib;
  const QnnInterface_t* qif;
  Qnn_BackendHandle_t backend;
  Qnn_DeviceHandle_t device;
  Qnn_ContextHandle_t context;
  Qnn_GraphHandle_t graph;

  Qnn_Tensor_t graph_inputs[2];
  Qnn_Tensor_t graph_outputs[1];
  Qnn_Tensor_t exec_inputs[2];
  Qnn_Tensor_t exec_outputs[1];

  uint32_t dims_a[2];
  uint32_t dims_q[2];
  uint32_t dims_y[2];

  uint32_t num_inputs;
  uint32_t num_outputs;
  int use_static_a;
  int use_fp16;
  uint16_t* a_fp16;
  uint16_t* q_fp16;
  uint16_t* y_fp16;
  size_t a_fp16_elems;
  size_t q_fp16_elems;
  size_t y_fp16_elems;

  int32_t n;
  int32_t d;
};

typedef struct tach_qnn_fc_ctx tach_qnn_fc_ctx;

static void cleanup_ctx(tach_qnn_fc_ctx* ctx) {
  if (!ctx) return;
  if (ctx->qif) {
    if (ctx->context && QIF(ctx->qif).contextFree) {
      QIF(ctx->qif).contextFree(ctx->context, NULL);
    }
    if (ctx->device && QIF(ctx->qif).deviceFree) {
      QIF(ctx->qif).deviceFree(ctx->device);
    }
    if (ctx->backend && QIF(ctx->qif).backendFree) {
      QIF(ctx->qif).backendFree(ctx->backend);
    }
  }
  if (ctx->a_fp16) free(ctx->a_fp16);
  if (ctx->q_fp16) free(ctx->q_fp16);
  if (ctx->y_fp16) free(ctx->y_fp16);
  if (ctx->lib) {
    dlclose(ctx->lib);
  }
  free(ctx);
}

static void init_tensor(Qnn_Tensor_t* t, const char* name, Qnn_TensorType_t type, uint32_t* dims, uint32_t rank, Qnn_DataType_t dtype) {
  *t = (Qnn_Tensor_t)QNN_TENSOR_INIT;
  t->version = QNN_TENSOR_VERSION_1;
  t->v1.name = name;
  t->v1.type = type;
  t->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  t->v1.dataType = dtype;
  t->v1.rank = rank;
  t->v1.dimensions = dims;
  t->v1.memType = QNN_TENSORMEMTYPE_RAW;
  t->v1.clientBuf.data = NULL;
  t->v1.clientBuf.dataSize = 0;
}

static inline uint32_t f32_to_u32(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return u;
}

static inline float u32_to_f32(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

static inline uint16_t float_to_half(float f) {
  uint32_t x = f32_to_u32(f);
  uint32_t sign = (x >> 16) & 0x8000u;
  int exp = (int)((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;

  if (exp <= 0) {
    if (exp < -10) {
      return (uint16_t)sign;
    }
    mant = (mant | 0x800000u) >> (1 - exp);
    return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
  } else if (exp >= 31) {
    if (mant == 0) {
      return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | 0x7c00u | (mant >> 13));
  } else {
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
  }
}

static inline float half_to_float(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t out;

  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ffu;
      exp = exp + (127 - 15);
      out = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    exp = exp + (127 - 15);
    out = sign | (exp << 23) | (mant << 13);
  }
  return u32_to_f32(out);
}

static void f32_to_f16_buf(const float* src, uint16_t* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = float_to_half(src[i]);
  }
}

static void f16_to_f32_buf(const uint16_t* src, float* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = half_to_float(src[i]);
  }
}

// ---------- public warmup ----------
int tach_qnn_warmup_open_close(void) {
  void* lib = NULL;
  const QnnInterface_t* qif = NULL;
  const char* prefer = getenv("TACHANN_QNN_BACKEND");
  if (open_interface(prefer, &lib, &qif) != 0) {
    return -10;
  }
  if (!qif || !QIF(qif).backendCreate || !QIF(qif).contextCreate) {
    if (lib) dlclose(lib);
    return -11;
  }

  Qnn_BackendHandle_t backend = NULL;
  Qnn_DeviceHandle_t device = NULL;
  Qnn_ContextHandle_t context = NULL;

  Qnn_ErrorHandle_t err = QIF(qif).backendCreate(NULL, NULL, &backend);
  if (err != QNN_SUCCESS || !backend) {
    if (lib) dlclose(lib);
    return -12;
  }

  if (QIF(qif).deviceCreate) {
    err = QIF(qif).deviceCreate(NULL, NULL, &device);
    if (err != QNN_SUCCESS && err != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      if (QIF(qif).backendFree) QIF(qif).backendFree(backend);
      if (lib) dlclose(lib);
      return -13;
    }
    if (err == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      device = NULL;
    }
  }

  err = QIF(qif).contextCreate(backend, device, NULL, &context);
  if (err != QNN_SUCCESS || !context) {
    if (QIF(qif).deviceFree && device) QIF(qif).deviceFree(device);
    if (QIF(qif).backendFree) QIF(qif).backendFree(backend);
    if (lib) dlclose(lib);
    return -14;
  }

  if (QIF(qif).contextFree) QIF(qif).contextFree(context, NULL);
  if (QIF(qif).deviceFree && device) QIF(qif).deviceFree(device);
  if (QIF(qif).backendFree) QIF(qif).backendFree(backend);
  if (lib) dlclose(lib);
  return 0;
}

// ---------- public graph API ----------

static tach_qnn_fc_ctx* tach_qnn_fc_create_internal(
    int32_t n,
    int32_t d,
    int32_t q_batch,
    const char* backend,
    int use_fp16,
    const float* a_static,
    int use_static_a) {
  if (n <= 0 || d <= 0) return NULL;
  if (q_batch <= 0) q_batch = 1;
  if (use_static_a && !a_static) return NULL;

  tach_qnn_fc_ctx* ctx = (tach_qnn_fc_ctx*)calloc(1, sizeof(*ctx));
  if (!ctx) return NULL;

  ctx->n = n;
  ctx->d = d;
  ctx->dims_a[0] = (uint32_t)n;
  ctx->dims_a[1] = (uint32_t)d;
  ctx->dims_q[0] = (uint32_t)d;
  ctx->dims_q[1] = (uint32_t)q_batch;
  ctx->dims_y[0] = (uint32_t)n;
  ctx->dims_y[1] = (uint32_t)q_batch;
  ctx->use_static_a = use_static_a ? 1 : 0;
  ctx->num_inputs = use_static_a ? 1u : 2u;
  ctx->num_outputs = 1u;
  ctx->use_fp16 = use_fp16 ? 1 : 0;
  if (ctx->use_fp16) {
    ctx->a_fp16_elems = (size_t)n * (size_t)d;
    ctx->q_fp16_elems = (size_t)d * (size_t)q_batch;
    ctx->y_fp16_elems = (size_t)n * (size_t)q_batch;
    ctx->a_fp16 = (uint16_t*)malloc(ctx->a_fp16_elems * sizeof(uint16_t));
    ctx->q_fp16 = (uint16_t*)malloc(ctx->q_fp16_elems * sizeof(uint16_t));
    ctx->y_fp16 = (uint16_t*)malloc(ctx->y_fp16_elems * sizeof(uint16_t));
    if (!ctx->a_fp16 || !ctx->q_fp16 || !ctx->y_fp16) {
      LOGE("fp16 buffer allocation failed\n");
      cleanup_ctx(ctx);
      return NULL;
    }
    if (use_static_a) {
      f32_to_f16_buf(a_static, ctx->a_fp16, ctx->a_fp16_elems);
    }
  }

  const char* prefer = backend ? backend : getenv("TACHANN_QNN_BACKEND");
  if (open_interface(prefer, &ctx->lib, &ctx->qif) != 0) {
    cleanup_ctx(ctx);
    return NULL;
  }

  if (!QIF(ctx->qif).backendCreate || !QIF(ctx->qif).contextCreate || !QIF(ctx->qif).graphCreate ||
      !QIF(ctx->qif).tensorCreateGraphTensor || !QIF(ctx->qif).graphAddNode || !QIF(ctx->qif).graphFinalize ||
      !QIF(ctx->qif).graphExecute) {
    LOGE("Required QNN interface functions missing\n");
    cleanup_ctx(ctx);
    return NULL;
  }

  Qnn_ErrorHandle_t err = QIF(ctx->qif).backendCreate(NULL, NULL, &ctx->backend);
  if (err != QNN_SUCCESS || !ctx->backend) {
    LOGE("backendCreate failed (%lu)\n", (unsigned long)err);
    cleanup_ctx(ctx);
    return NULL;
  }

  if (QIF(ctx->qif).deviceCreate) {
    err = QIF(ctx->qif).deviceCreate(NULL, NULL, &ctx->device);
    if (err != QNN_SUCCESS && err != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      LOGE("deviceCreate failed (%lu)\n", (unsigned long)err);
      cleanup_ctx(ctx);
      return NULL;
    }
    if (err == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      ctx->device = NULL;
    }
  }

  err = QIF(ctx->qif).contextCreate(ctx->backend, ctx->device, NULL, &ctx->context);
  if (err != QNN_SUCCESS || !ctx->context) {
    LOGE("contextCreate failed (%lu)\n", (unsigned long)err);
    cleanup_ctx(ctx);
    return NULL;
  }

  err = QIF(ctx->qif).graphCreate(ctx->context, "tach_fc", NULL, &ctx->graph);
  if (err != QNN_SUCCESS || !ctx->graph) {
    LOGE("graphCreate failed (%lu)\n", (unsigned long)err);
    cleanup_ctx(ctx);
    return NULL;
  }

  const Qnn_DataType_t dtype = ctx->use_fp16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
  if (use_static_a) {
    const uint32_t a_bytes = (uint32_t)((uint64_t)n * (uint64_t)d * (ctx->use_fp16 ? sizeof(uint16_t) : sizeof(float)));
    init_tensor(&ctx->graph_inputs[0], "A", QNN_TENSOR_TYPE_STATIC, ctx->dims_a, 2, dtype);
    if (ctx->use_fp16) {
      ctx->graph_inputs[0].v1.clientBuf.data = (void*)ctx->a_fp16;
    } else {
      ctx->graph_inputs[0].v1.clientBuf.data = (void*)a_static;
    }
    ctx->graph_inputs[0].v1.clientBuf.dataSize = a_bytes;
  } else {
    init_tensor(&ctx->graph_inputs[0], "A", QNN_TENSOR_TYPE_APP_WRITE, ctx->dims_a, 2, dtype);
  }
  init_tensor(&ctx->graph_inputs[1], "q", QNN_TENSOR_TYPE_APP_WRITE, ctx->dims_q, 2, dtype);
  init_tensor(&ctx->graph_outputs[0], "y", QNN_TENSOR_TYPE_APP_READ, ctx->dims_y, 2, dtype);

  err = QIF(ctx->qif).tensorCreateGraphTensor(ctx->graph, &ctx->graph_inputs[0]);
  if (err != QNN_SUCCESS) { LOGE("tensorCreateGraphTensor(A) failed (%lu)\n", (unsigned long)err); cleanup_ctx(ctx); return NULL; }
  err = QIF(ctx->qif).tensorCreateGraphTensor(ctx->graph, &ctx->graph_inputs[1]);
  if (err != QNN_SUCCESS) { LOGE("tensorCreateGraphTensor(q) failed (%lu)\n", (unsigned long)err); cleanup_ctx(ctx); return NULL; }
  err = QIF(ctx->qif).tensorCreateGraphTensor(ctx->graph, &ctx->graph_outputs[0]);
  if (err != QNN_SUCCESS) { LOGE("tensorCreateGraphTensor(y) failed (%lu)\n", (unsigned long)err); cleanup_ctx(ctx); return NULL; }

  Qnn_Param_t params[2];
  params[0] = (Qnn_Param_t)QNN_PARAM_INIT;
  params[0].paramType = QNN_PARAMTYPE_SCALAR;
  params[0].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
  params[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  params[0].scalarParam.bool8Value = 0;

  params[1] = (Qnn_Param_t)QNN_PARAM_INIT;
  params[1].paramType = QNN_PARAMTYPE_SCALAR;
  params[1].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
  params[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  params[1].scalarParam.bool8Value = 0;

  Qnn_OpConfig_t op = (Qnn_OpConfig_t)QNN_OPCONFIG_INIT;
  op.version = QNN_OPCONFIG_VERSION_1;
  op.v1.name = "matmul";
  op.v1.packageName = "qti.aisw";
  op.v1.typeName = QNN_OP_MAT_MUL;
  op.v1.numOfParams = 2;
  op.v1.params = params;
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = ctx->graph_inputs;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = ctx->graph_outputs;

  err = QIF(ctx->qif).graphAddNode(ctx->graph, op);
  if (err != QNN_SUCCESS) {
    LOGE("graphAddNode(MatMul) failed (%lu)\n", (unsigned long)err);
    cleanup_ctx(ctx);
    return NULL;
  }

  err = QIF(ctx->qif).graphFinalize(ctx->graph, NULL, NULL);
  if (err != QNN_SUCCESS) {
    LOGE("graphFinalize failed (%lu)\n", (unsigned long)err);
    cleanup_ctx(ctx);
    return NULL;
  }

  LOGI("QNN MatMul graph ready (n=%d d=%d b=%d%s)\n", n, d, q_batch, use_static_a ? ", static A" : "");
  return ctx;
}

// Create a QNN graph for MatMul: A (nxd) @ q (dx1) -> y (nx1)
tach_qnn_fc_ctx* tach_qnn_fc_create(int32_t n, int32_t d, const char* backend, int use_fp16) {
  return tach_qnn_fc_create_internal(n, d, 1, backend, use_fp16, NULL, 0);
}

// Create a QNN graph with static A.
tach_qnn_fc_ctx* tach_qnn_fc_create_static_a(int32_t n, int32_t d, const float* A, const char* backend, int use_fp16) {
  return tach_qnn_fc_create_internal(n, d, 1, backend, use_fp16, A, 1);
}

tach_qnn_fc_ctx* tach_qnn_fc_create_batched(int32_t n, int32_t d, int32_t q_batch, const char* backend, int use_fp16) {
  return tach_qnn_fc_create_internal(n, d, q_batch, backend, use_fp16, NULL, 0);
}

tach_qnn_fc_ctx* tach_qnn_fc_create_static_a_batched(int32_t n, int32_t d, int32_t q_batch, const float* A, const char* backend, int use_fp16) {
  return tach_qnn_fc_create_internal(n, d, q_batch, backend, use_fp16, A, 1);
}

int tach_qnn_fc_run_q(tach_qnn_fc_ctx* ctx, const float* q, float* out) {
  if (!ctx || !q || !out) return -1;
  if (!ctx->qif || !QIF(ctx->qif).graphExecute) return -2;

  const uint32_t q_cols = ctx->dims_q[1];
  const uint32_t y_cols = ctx->dims_y[1];
  const uint32_t q_bytes = (uint32_t)((uint64_t)ctx->d * (uint64_t)q_cols * (ctx->use_fp16 ? sizeof(uint16_t) : sizeof(float)));
  const uint32_t y_bytes = (uint32_t)((uint64_t)ctx->n * (uint64_t)y_cols * (ctx->use_fp16 ? sizeof(uint16_t) : sizeof(float)));

  ctx->exec_inputs[0] = ctx->graph_inputs[1];
  if (ctx->use_fp16) {
    f32_to_f16_buf(q, ctx->q_fp16, ctx->q_fp16_elems);
    ctx->exec_inputs[0].v1.clientBuf.data = (void*)ctx->q_fp16;
  } else {
    ctx->exec_inputs[0].v1.clientBuf.data = (void*)q;
  }
  ctx->exec_inputs[0].v1.clientBuf.dataSize = q_bytes;

  ctx->exec_outputs[0] = ctx->graph_outputs[0];
  if (ctx->use_fp16) {
    ctx->exec_outputs[0].v1.clientBuf.data = (void*)ctx->y_fp16;
  } else {
    ctx->exec_outputs[0].v1.clientBuf.data = (void*)out;
  }
  ctx->exec_outputs[0].v1.clientBuf.dataSize = y_bytes;

  Qnn_ErrorHandle_t err = QIF(ctx->qif).graphExecute(
      ctx->graph,
      ctx->exec_inputs,
      1,
      ctx->exec_outputs,
      1,
      NULL,
      NULL);

  if (err != QNN_SUCCESS) {
    LOGE("graphExecute failed (%lu)\n", (unsigned long)err);
    return -3;
  }

  if (ctx->use_fp16) {
    f16_to_f32_buf(ctx->y_fp16, out, ctx->y_fp16_elems);
  }
  return 0;
}

int tach_qnn_fc_run(tach_qnn_fc_ctx* ctx, const float* A, const float* q, float* out) {
  if (!ctx || !q || !out) return -1;
  if (!ctx->qif || !QIF(ctx->qif).graphExecute) return -2;
  if (ctx->use_static_a) {
    return tach_qnn_fc_run_q(ctx, q, out);
  }
  if (!A) return -1;

  const uint32_t a_bytes = (uint32_t)((uint64_t)ctx->n * (uint64_t)ctx->d * (ctx->use_fp16 ? sizeof(uint16_t) : sizeof(float)));
  const uint32_t q_cols = ctx->dims_q[1];
  const uint32_t y_cols = ctx->dims_y[1];
  const uint32_t q_bytes = (uint32_t)((uint64_t)ctx->d * (uint64_t)q_cols * (ctx->use_fp16 ? sizeof(uint16_t) : sizeof(float)));
  const uint32_t y_bytes = (uint32_t)((uint64_t)ctx->n * (uint64_t)y_cols * (ctx->use_fp16 ? sizeof(uint16_t) : sizeof(float)));

  ctx->exec_inputs[0] = ctx->graph_inputs[0];
  if (ctx->use_fp16) {
    f32_to_f16_buf(A, ctx->a_fp16, ctx->a_fp16_elems);
    ctx->exec_inputs[0].v1.clientBuf.data = (void*)ctx->a_fp16;
  } else {
    ctx->exec_inputs[0].v1.clientBuf.data = (void*)A;
  }
  ctx->exec_inputs[0].v1.clientBuf.dataSize = a_bytes;

  ctx->exec_inputs[1] = ctx->graph_inputs[1];
  if (ctx->use_fp16) {
    f32_to_f16_buf(q, ctx->q_fp16, ctx->q_fp16_elems);
    ctx->exec_inputs[1].v1.clientBuf.data = (void*)ctx->q_fp16;
  } else {
    ctx->exec_inputs[1].v1.clientBuf.data = (void*)q;
  }
  ctx->exec_inputs[1].v1.clientBuf.dataSize = q_bytes;

  ctx->exec_outputs[0] = ctx->graph_outputs[0];
  if (ctx->use_fp16) {
    ctx->exec_outputs[0].v1.clientBuf.data = (void*)ctx->y_fp16;
  } else {
    ctx->exec_outputs[0].v1.clientBuf.data = (void*)out;
  }
  ctx->exec_outputs[0].v1.clientBuf.dataSize = y_bytes;

  Qnn_ErrorHandle_t err = QIF(ctx->qif).graphExecute(
      ctx->graph,
      ctx->exec_inputs,
      2,
      ctx->exec_outputs,
      1,
      NULL,
      NULL);

  if (err != QNN_SUCCESS) {
    LOGE("graphExecute failed (%lu)\n", (unsigned long)err);
    return -3;
  }
  if (ctx->use_fp16) {
    f16_to_f32_buf(ctx->y_fp16, out, ctx->y_fp16_elems);
  }
  return 0;
}

void tach_qnn_fc_destroy(tach_qnn_fc_ctx* ctx) {
  cleanup_ctx(ctx);
}
