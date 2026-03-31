# External Dependencies

Tachyon ANN is usable in three modes, and each mode has different runtime requirements.

**CPU-only**
1. No external inference runtime is required.
2. `cargo build -p tachyon-ann` and `TACHANN_BACKEND=cpu` are enough.

**QNN / HTP**
1. You must provide a local QAIRT/QNN SDK copy.
2. You must build or point to the QNN shim via `TACHANN_QNN_LIB`.
3. You must expose the host-side QNN shared libraries on `LD_LIBRARY_PATH`.
4. You must expose the DSP skels on `ADSP_LIBRARY_PATH`.
5. The device must expose `/dev/adsprpc-smd` and `/dev/ion`.

**ORT**
1. You must provide your own `libonnxruntime.so`.
2. ORT is a comparison/debug backend in this repo, not the primary runtime story.
3. Set `MODELS_DIR` to the repo `models/` directory or another directory containing the required ONNX model files.

**Required QNN Pieces**
1. `QNN_SDK_ROOT`
2. Host libs such as `libQnnSystem.so`, `libQnnHtp.so`, `libQnnHtpPrepare.so`, and one `libQnnHtpV*Stub.so`
3. DSP skels such as `libQnnHtpV68Skel.so` under `hexagon-v68/unsigned`
4. `TACHANN_QNN_LIB`, typically `./target/debug/libtachyon_qnnshim.so`

**Verified Tachyon Particle Environment**
```bash
export QNN_SDK_ROOT=/path/to/qairt
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4:$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4/arch64
export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v68/unsigned;/usr/lib/rfsa/adsp;/dsp"
export TACHANN_QNN_LIB=./target/debug/libtachyon_qnnshim.so
export TACHANN_QNN_BACKEND=htp
export TACHANN_QNN_FP16=0
export TACHANN_QNN_STATIC_A=1
```

**How To Verify**
1. CPU-only:
```bash
cargo build -p tachyon-ann --bin tachyon-ann
TACHANN_BACKEND=cpu ./target/debug/tachyon-ann
```
2. QNN / HTP:
```bash
./scripts/check_runtime.sh --qnn
```
3. ORT:
```bash
./scripts/check_runtime.sh --ort
```

**Common Failures**
1. `QNN: shim reports unavailable`
   - Usually means `QNN_SDK_ROOT`, `LD_LIBRARY_PATH`, or `TACHANN_QNN_LIB` is wrong.
2. `deviceCreate failed (14001)` or `Transport layer setup failed: 14001`
   - Usually means `ADSP_LIBRARY_PATH` is wrong or the DSP device nodes are missing/inaccessible.
3. `graphExecute failed (6001)` on QNN FP16
   - Use `TACHANN_QNN_FP16=0`. FP32 is the stable path in this repo.
4. ORT init failures or panics
   - Usually means `ORT_DYLIB_PATH` is unset or points to the wrong `libonnxruntime.so`.

**What This Repo Does Not Ship**
1. QAIRT/QNN binaries
2. ONNX Runtime binaries
3. External embedding models

The service scores vectors. It does not bundle the proprietary runtimes or a text/image embedding model.

**Precision Expectations**
1. The public API is `f32` in and `f32` out.
2. QNN should be treated as a stable `fp32` path today.
3. `TACHANN_QNN_FP16=1` is optional and may fall back to `fp32`.
4. `fp64` is not a supported serving format.
