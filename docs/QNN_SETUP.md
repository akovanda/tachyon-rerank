# QNN Setup

Use this doc when you want the direct QNN HTP path on a Tachyon Particle device.

**Start Here**
1. Read `docs/DEPENDENCIES.md` for the full list of required runtime pieces.
2. Run `./scripts/check_runtime.sh --qnn` before blaming the service.

**Build**
1. Set `QNN_SDK_ROOT` to your local QAIRT/QNN SDK.
2. Build the shim.
```bash
export QNN_SDK_ROOT=/path/to/qairt
cargo build -p tachyon_qnnshim
```

**Verified Runtime Environment**
```bash
export QNN_SDK_ROOT=/path/to/qairt
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4:$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4/arch64
export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v68/unsigned;/usr/lib/rfsa/adsp;/dsp"
export TACHANN_QNN_LIB=./target/debug/libtachyon_qnnshim.so
export TACHANN_QNN_BACKEND=htp
export TACHANN_QNN_FP16=0
export TACHANN_QNN_STATIC_A=1
./scripts/check_runtime.sh --qnn
./target/debug/tachyon-rerank
```

**Runtime Notes**
1. The service API remains `f32` in and `f32` out even when QNN is used underneath.
2. Direct QNN execution is enabled by default when QNN is detected.
3. `TACHANN_QNN_FP16=0` is the stable path today.
4. `TACHANN_QNN_FP16=1` is experimental; the shim retries `fp32` if the HTP graph rejects FP16.
5. `TACHANN_QNN_STATIC_A=1` is the preferred repeated-query mode.
6. `GET /info` reports whether QNN was actually selected or whether the request fell back to CPU.
7. `POST /score_batch` is the important retrieval workload path.

**Benchmarking Notes**
1. For benchmark sweeps, set `TACHANN_QNN_WARMUP=0`.
2. The bench binary already performs its own warmup iterations.
3. See `benchmarks/README.md` for the checked-in Tachyon Particle report.
