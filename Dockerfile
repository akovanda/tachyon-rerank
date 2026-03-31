# syntax=docker/dockerfile:1.6

FROM rust:1.85.0-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates pkg-config clang build-essential libssl-dev git \
 && rm -rf /var/lib/apt/lists/*

ENV CC=clang CXX=clang++
WORKDIR /src

COPY Cargo.toml Cargo.lock ./
COPY services/tachann/Cargo.toml services/tachann/Cargo.toml
COPY native/qnnshim/Cargo.toml native/qnnshim/Cargo.toml

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo fetch

COPY services services

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p tachyon-rerank --bin tachyon-rerank

FROM debian:bookworm-slim AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    ca-certificates curl bash \
    libgcc-s1 libstdc++6 \
    libc++1 libc++abi1 libunwind8 \
    libgomp1 libnuma1 \
    libssl3 \
  ; rm -rf /var/lib/apt/lists/*

RUN useradd --system --create-home --home-dir /app appuser

WORKDIR /app
COPY --from=builder /src/target/release/tachyon-rerank /app/tachyon-rerank

EXPOSE 8080
USER appuser
ENTRYPOINT ["/app/tachyon-rerank"]
