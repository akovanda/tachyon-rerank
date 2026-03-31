# Environment Variables

This doc lists the primary runtime knobs. Use `docs/DEPENDENCIES.md` for the external runtime setup and `./scripts/check_runtime.sh` to validate it.

**Core**
1. `TACHANN_BACKEND`: `auto` | `adaptive` | `cpu` | `ort` | `qnn`. Default: `auto`.
2. `TACHANN_BIND`: bind address. Default: `0.0.0.0:8080`.
3. `TACHANN_MAX_DIM`: max dimensionality. Default: `8192`.
4. `TACHANN_MAX_CAND`: max candidate rows. Default: `200000`.
5. `TACHANN_CHUNK`: chunk size used by CPU/QNN fallback. Default: `4096`.
6. `RUST_LOG`: logging level. Default: `info`.

**QNN (direct shim)**
1. `QNN_SDK_ROOT`: path to QAIRT/QNN SDK root.
2. `TACHANN_QNN_LIB`: path to the built shim library.
3. `TACHANN_QNN_BACKEND`: `htp` or `cpu`. Use `htp` on-device.
4. `TACHANN_QNN_FP16`: `1` to request FP16. The recommended public default is `0`, and the stable public path is still `fp32`.
5. `TACHANN_QNN_STATIC_A`: `1` to cache A and run q-only graphs.
6. `TACHANN_QNN_WARMUP`: `1` to warm the QNN path during server startup. Use `0` for benchmark sweeps.
7. `TACHANN_QNN_TIMING`: `1` or `all` to print timing breakdowns.
8. `ADSP_LIBRARY_PATH`: DSP search path. On Tachyon Particle this is semicolon-separated, for example `"$QNN_SDK_ROOT/lib/hexagon-v68/unsigned;/usr/lib/rfsa/adsp;/dsp"`.

**Adaptive Routing**
1. `TACHANN_BACKEND=adaptive`: use the Tachyon Particle heuristic profile.
2. Current heuristic: prefer QNN for batched workloads, but keep CPU available for the known large high-dimension singleton case.

**ORT**
1. `ORT_DYLIB_PATH`: path to `libonnxruntime.so`.
2. `MODELS_DIR`: directory containing ONNX models.
3. `TACHANN_ORT_EP`: `cpu` or `qnn`. Keep `cpu` unless you are explicitly testing the ORT QNN EP.
4. `TACHANN_ORT_QNN_MODEL`: ONNX model for the ORT QNN EP.
5. `TACHANN_ORT_STATIC_N` and `TACHANN_ORT_STATIC_D`: static dims for the ORT QNN EP.

**Precision Summary**
1. Clients should send `f32` vectors.
2. Clients should expect `f32` scores back.
3. `fp64` is not part of the public API or the supported serving story.
