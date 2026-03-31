# Tachyon Rerank

Tachyon Rerank is a retrieval-first vector scoring microservice for edge hardware.

It is designed for workloads that already have embeddings and need exact scoring over a shared candidate set:
1. semantic search over a local corpus
2. code search reranking
3. RAG reranking over a top-k candidate set
4. secondary fits such as face/image retrieval and recommendation reranking

On Tachyon Particle, the service prefers the direct QNN HTP path. Everywhere else, CPU is the portable fallback.

**What It Is Good At**
1. Reusing the same candidate matrix across many requests
2. Batched query scoring via `POST /score_batch`
3. Exact cosine, inner-product, and L2 scoring on-device
4. `f32` vector scoring with a stable `fp32` QNN path

**What It Is Not**
1. A vector database or ANN indexing engine
2. An embedding model
3. A good fit for huge dynamic indexes or sparse singleton traffic
4. A generic mixed-precision inference framework; this service is an `f32` scoring API

**Quickstart: CPU-only**
1. Install Rust `1.85.0`.
```bash
rustup toolchain install 1.85.0
rustup component add --toolchain 1.85.0 rustfmt clippy
```
2. Build and run the service.
```bash
cargo build -p tachyon-rerank --bin tachyon-rerank
TACHANN_BACKEND=cpu ./target/debug/tachyon-rerank
```
3. Try a request.
```bash
curl -s http://localhost:8080/score \
  -H 'Content-Type: application/json' \
  -d '{
    "q": [1.0, 0.0],
    "a": [[1.0, 0.0], [0.0, 1.0]],
    "metric": "ip"
  }'
```

**Quickstart: Tachyon Particle / QNN**
1. Build the server and shim.
```bash
cargo build -p tachyon-rerank --bin tachyon-rerank
QNN_SDK_ROOT=/path/to/qairt cargo build -p tachyon_qnnshim
```
2. Export the verified runtime environment.
```bash
export QNN_SDK_ROOT=/path/to/qairt
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4:$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4/arch64
export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v68/unsigned;/usr/lib/rfsa/adsp;/dsp"
export TACHANN_QNN_LIB=./target/debug/libtachyon_qnnshim.so
export TACHANN_QNN_BACKEND=htp
export TACHANN_QNN_FP16=0
export TACHANN_QNN_STATIC_A=1
```
3. Validate the runtime before starting the server.
```bash
./scripts/check_runtime.sh --qnn
```
4. Run the service.
```bash
./target/debug/tachyon-rerank
```
5. Inspect the selected runtime.
```bash
curl -s http://localhost:8080/info
```

**Supplying External Runtimes**
This repo does not ship QAIRT/QNN, ONNX Runtime, or any embedding model. Use these docs instead of guessing:
1. `docs/DEPENDENCIES.md`: what libraries you need and how to verify them
2. `docs/QNN_SETUP.md`: direct QNN path on Tachyon Particle
3. `docs/ENV_VARS.md`: runtime knobs and defaults

**Examples**
1. `examples/semantic_search/`: local semantic search
2. `examples/code_search/`: code search reranking with batched queries
3. `examples/rag_rerank/`: RAG reranking over retrieved passages
4. `examples/README.md`: overview plus optional public dataset download

The examples use tiny precomputed embeddings so the docs and tests stay deterministic.

**Precision Model**
1. The public API accepts `f32` vectors and returns `f32` scores.
2. The stable QNN serving path is `fp32`.
3. `TACHANN_QNN_FP16=1` is best treated as experimental; the shim retries `fp32` if HTP rejects the graph.
4. `fp64` is not a public serving format in this project.

**Performance Proof**
1. Checked-in report: `benchmarks/REPORT.md`
2. Full benchmark notes: `benchmarks/README.md`

Small rerank latency:

![Small rerank latency](benchmarks/bench_charts/n1024_d128_latency.svg)

High-dimension speedup:

![High-dimension speedup](benchmarks/bench_charts/n8192_d768_speedup.svg)

**API**
1. `GET /info`: requested mode, actual backend, fallback reason, QNN status
2. `POST /score`: one query against a candidate matrix
3. `POST /score_batch`: many queries against a shared candidate matrix

Batched example:
```bash
curl -s http://localhost:8080/score_batch \
  -H 'Content-Type: application/json' \
  -d '{
    "qs": [[1.0, 0.0], [0.0, 1.0]],
    "a": [[1.0, 0.0], [0.0, 1.0]],
    "metric": "ip"
  }'
```

**Repository Layout**
1. `services/tachann`: service, runtime router, and benchmark binary
2. `native/qnnshim`: direct QNN shim used by the service
3. `examples`: retrieval-first mini demos and fixtures
4. `benchmarks`: checked-in benchmark report, raw data, and charts
5. `scripts`: runtime check, dataset fetch, and benchmark helpers

**Additional Docs**
1. `docs/ARCHITECTURE.md`: request flow, backend roles, and batching model
2. `docs/DEPENDENCIES.md`: external runtime setup and common failures
3. `docs/QNN_SETUP.md`: QNN-specific build and runtime notes
4. `benchmarks/README.md`: benchmark reproduction and interpretation

**License**
MIT. See `LICENSE`.
