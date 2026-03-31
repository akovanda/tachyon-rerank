# Code Search Mini Demo

This example models a code-search or documentation-search reranker.

**Expected Result**
1. `q_batching` should rank `chunk_score_batch` first.
2. `q_setup` should rank `chunk_qnn_env` first.

**Run It**
```bash
curl -s http://localhost:8080/score_batch \
  -H 'Content-Type: application/json' \
  -d @examples/code_search/request.score_batch.json
```

**Why It Fits Tachyon ANN**
1. Code search often reranks a small top-k candidate set.
2. Editor and search workflows naturally produce short bursts of queries.
3. Batched reranking is exactly where the QNN path becomes more attractive.
