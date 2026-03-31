# Examples

These examples are retrieval-first mini demos for Tachyon Rerank.

Each example includes:
1. A tiny deterministic corpus with precomputed embeddings
2. One or more query vectors
3. Expected ranking order
4. A ready-to-send API payload

The examples are intentionally small so they can double as test fixtures.

**Examples**
1. `examples/semantic_search/`: local semantic search over a tiny document set
2. `examples/code_search/`: code-chunk reranking with batched queries
3. `examples/rag_rerank/`: RAG reranking over a retrieved candidate set

**Important**
1. The service scores vectors. It does not embed raw text.
2. The text in these examples exists to explain why the ranking should look the way it does.
3. The vectors are precomputed fixtures used by the docs and tests.

**Optional Public Data**
1. To download a real public retrieval corpus for manual evaluation:
```bash
python3 scripts/fetch_public_demo_data.py --dataset scifact
```
2. That download is not used in CI and is not required for first-run examples.
