# RAG Rerank Mini Demo

This example models reranking over an already retrieved set of passages.

**Expected Result**
1. `q_sdk_root` should rank `pass_qnn_sdk` first.
2. `q_dsp_path` should rank `pass_adsp_path` first.

**Run It**
```bash
curl -s http://localhost:8080/score_batch \
  -H 'Content-Type: application/json' \
  -d @examples/rag_rerank/request.score_batch.json
```

**Why It Fits Tachyon ANN**
1. RAG pipelines usually retrieve a top-k set first, then rerank.
2. That rerank stage is a small exact-scoring problem.
3. Multiple user questions or agent subqueries can be batched together.
