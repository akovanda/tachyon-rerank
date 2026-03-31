# Architecture

Tachyon ANN is an exact vector scoring microservice designed for edge retrieval and reranking workloads.

**What It Does**
1. Accepts one query with `/score` or many queries with `/score_batch`.
2. Scores those query vectors against a shared candidate matrix.
3. Routes the request to QNN, ORT, or CPU depending on runtime mode and availability.
4. Falls back to CPU if an accelerated backend cannot initialize or execute.
5. Exposes an `f32` scoring API regardless of backend choice.

**Why The Service Exists**
1. Retrieval systems often already have embeddings.
2. The hot path is usually scoring one or many query vectors against a candidate set.
3. On Tachyon Particle, QNN becomes attractive when the candidate matrix is reused and queries can be batched.

**Backend Roles**
1. QNN: primary hardware path on supported Tachyon Particle devices.
2. CPU: zero-dependency fallback and public CI baseline.
3. ORT: comparison/debug path only.

**Precision Model**
1. Request vectors are `f32`.
2. Response scores are `f32`.
3. CPU may use some `f64` intermediate accumulation internally for stability, but `fp64` is not the API format.
4. The stable QNN path should be thought of as `fp32`; `fp16` is opportunistic rather than guaranteed.

**Request Flow**
1. Client sends vectors to `/score` or `/score_batch`.
2. `RuntimeRouter` decides which backend to use.
3. Backend computes distances or negative similarities.
4. The response returns raw scores in candidate order.
5. Callers do ranking/top-k outside the service.

**Batching and Static-A**
1. `score_batch` is the important path for retrieval workloads.
2. QNN static-A mode caches the candidate matrix and reuses it across repeated queries.
3. This is why semantic search, code search, and RAG reranking are the primary fits.

**Good Fits**
1. Semantic search over a local corpus
2. Code search reranking
3. RAG reranking over a top-k candidate set
4. Secondary fits: face/image retrieval and recommendation reranking

**Poor Fits**
1. Huge dynamic vector indexes where ANN indexing dominates
2. Workloads that are mostly single, tiny requests with no batching opportunity
3. Pipelines where embedding generation is the real bottleneck
