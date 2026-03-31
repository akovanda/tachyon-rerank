# tachyon_qnnshim

`tachyon_qnnshim` is the thin C-ABI layer that the Rust service loads with `dlopen`.

It exposes a small stable surface so the service can use the direct QNN path without linking the full SDK into the main binary.

**What It Does**
1. Detects whether QNN runtime libraries are visible.
2. Builds and executes a direct QNN MatMul graph on supported hardware.
3. Supports batched query execution for the retrieval path.
4. Falls back to CPU when QNN initialization or execution fails.

**Current Runtime Behavior**
1. Direct QNN execution is enabled by default when QNN is detected.
2. FP32 is the stable public path.
3. FP16 may be requested, but the shim disables it and retries FP32 if the HTP path rejects the graph.
4. The service remains usable because CPU fallback is built into the higher-level runtime router.

**Key Exports**
1. `tachyon_qnn_avail`: returns `1` if the QNN runtime is visible.
2. `tachyon_qnn_matmul`: single-query matmul path.
3. `tachyon_qnn_matmul_batched`: batched query path.
4. `tachann_qnn_warmup`: optional startup probe.
5. `tachann_qnn_cleanup`: explicit cleanup hook used by the service on shutdown.

**Important Environment Variables**
1. `QNN_SDK_ROOT`
2. `LD_LIBRARY_PATH`
3. `ADSP_LIBRARY_PATH`
4. `TACHANN_QNN_BACKEND`
5. `TACHANN_QNN_FP16`
6. `TACHANN_QNN_STATIC_A`

For the full setup, use `docs/DEPENDENCIES.md` and `docs/QNN_SETUP.md`.
