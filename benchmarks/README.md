# Benchmarks

This folder contains the checked-in Tachyon Particle benchmark report and the scripts used to regenerate it.

**Artifacts**
1. `benchmarks/REPORT.md`: doc-ready summary of the current checked-in run
2. `benchmarks/bench_results.csv`: raw per-backend latency and throughput data
3. `benchmarks/bench_summary.md`: best backend by case/batch
4. `benchmarks/bench_charts/*.svg`: latency and speedup curves used in the docs

**How These Cases Map To Real Workloads**
1. `n=1024,d=128`: small semantic-search or note-search reranking
2. `n=8192,d=128`: larger code-search or document-rerank workload
3. `n=8192,d=768`: high-dimension RAG or dense-retrieval reranking

**Recommended Device Flow**
1. Build the bench binary.
```bash
cargo build -p tachyon-rerank --bin tachyon-rerank-bench
```
2. Validate the runtime.
```bash
./scripts/check_runtime.sh --qnn
```
3. Run a single comparison.
```bash
./target/debug/tachyon-rerank-bench --backend cpu --backend qnn --n 1024 --d 128 --q-batch 4
```
4. Regenerate the full report.
```bash
TACHANN_QNN_WARMUP=0 python3 scripts/bench_sweep.py --out benchmarks --backends cpu,qnn
```
5. Add ORT only if you want an explicit comparison baseline.
```bash
TACHANN_QNN_WARMUP=0 python3 scripts/bench_sweep.py --out benchmarks --backends cpu,ort,qnn
```

**Current Readout**
1. Small rerank set: QNN wins across the tested batch range.
2. `8K x 128`: crossover case; QNN becomes the better choice once batching grows.
3. `8K x 768`: singleton latency is near parity, then QNN scales much better with batching.

**Notes**
1. The checked-in report was collected on a Tachyon Particle device with QNN HTP available.
2. `ADSP_LIBRARY_PATH` is semicolon-separated on this target.
3. The sweep records both the requested backend and the actual backend label, so fallbacks are visible in the CSV.
4. The bench binary already performs warmup iterations; `TACHANN_QNN_WARMUP=0` keeps the sweep setup cleaner.
