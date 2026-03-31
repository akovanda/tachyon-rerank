# Contributing

Thanks for your interest in contributing.

**Project Focus**
1. Retrieval-first workloads: semantic search, code search, and RAG reranking
2. QNN-first on Tachyon Particle, with CPU as the portable fallback
3. Small, deterministic examples and tests that stay honest

**Before You Start**
1. Read `README.md`, `docs/ARCHITECTURE.md`, and `docs/DEPENDENCIES.md`.
2. If you plan a larger change, open an issue first.

**Development Setup**
1. Install Rust `1.94.1`.
2. CPU-only build:
```bash
cargo build -p tachyon-rerank
```
3. QNN work requires `QNN_SDK_ROOT` plus the runtime pieces documented in `docs/DEPENDENCIES.md`.
4. Validate the local runtime before running QNN-specific work:
```bash
./scripts/check_runtime.sh --qnn
```

**Tests and Checks**
1. Run the service tests, including the example-fixture tests.
```bash
cargo test -p tachyon-rerank
```
2. Run the black-box end-to-end server tests.
```bash
cargo test -p tachyon-rerank --test e2e_server
```
3. Run format checks.
```bash
cargo fmt --all --check
```
4. Run clippy.
```bash
cargo clippy -p tachyon-rerank --all-targets -- -D warnings
```
5. Optional accelerator parity and QNN end-to-end tests are opt-in:
```bash
TACHANN_RUN_OPTIONAL_ACCEL_TESTS=1 cargo test -p tachyon-rerank optional_
```

**Examples and Docs**
1. Keep `examples/` and the tests in sync.
2. If you change ranking behavior or setup, update the README and the relevant doc.
3. If you change performance behavior, update the benchmark report or explain why not.

**Releases**
1. The repo starts at `v0.1.0`.
2. After CI succeeds on a push to `main`, the release workflow automatically bumps the patch version, commits the manifest update, and creates the next `vX.Y.Z` tag.
3. Keep manual version edits out of normal feature commits unless the release flow itself is changing.

**License**
By contributing, you agree that your contributions are licensed under the MIT license.
