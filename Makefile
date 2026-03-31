.PHONY: build build-release build-qnn test bench bench-sweep check-runtime fetch-public-data clean

build:
	cargo build -p tachyon-ann

build-release:
	cargo build -p tachyon-ann --release

build-qnn:
	cargo build -p tachyon_qnnshim

test:
	cargo test -p tachyon-ann

bench:
	cargo run -p tachyon-ann --bin tachann-bench -- --backend cpu --n 1024 --d 128 --q-batch 4

bench-sweep:
	python3 scripts/bench_sweep.py --out benchmarks --backends cpu

check-runtime:
	./scripts/check_runtime.sh --auto

fetch-public-data:
	python3 scripts/fetch_public_demo_data.py --dataset scifact

clean:
	cargo clean
