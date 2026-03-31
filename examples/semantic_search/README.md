# Semantic Search Mini Demo

This example models a small local semantic-search corpus.

**Expected Result**
1. `doc_embeddings` should rank first for `q_semantic`.
2. `doc_sourdough` should rank last because it is unrelated noise.

**Run It**
```bash
curl -s http://localhost:8080/score \
  -H 'Content-Type: application/json' \
  -d @examples/semantic_search/request.score.json
```

**Why It Fits Tachyon ANN**
1. The corpus embeddings are static or slow-moving.
2. The same candidate matrix can be reused for many local queries.
3. This is the exact retrieval pattern where QNN batching becomes interesting.
