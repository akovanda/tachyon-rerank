#!/usr/bin/env python3
"""Fetch a small public retrieval dataset for manual evaluation.

This script intentionally downloads raw text data only. Tachyon Rerank scores vectors;
you still need an embedding model of your choice to turn the corpus and queries
into embeddings before sending them to the service.
"""

from __future__ import annotations

import argparse
import pathlib
import urllib.request
import zipfile

DATASETS = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "files": ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"],
    }
}


def fetch(url: str, dest: pathlib.Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as out:
        out.write(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="scifact")
    parser.add_argument("--out", default=".cache/public-data")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    out_dir = pathlib.Path(args.out).resolve() / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{args.dataset}.zip"

    print(f"downloading {args.dataset} -> {archive}")
    fetch(dataset["url"], archive)

    with zipfile.ZipFile(archive) as zf:
        for name in dataset["files"]:
            member = next((m for m in zf.namelist() if m.endswith(name)), None)
            if member is None:
                raise SystemExit(f"missing {name} in archive")
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            print(f"wrote {target}")

    print()
    print("download complete")
    print("next step: embed the downloaded corpus and queries with your model of choice")
    print("then feed the resulting vectors into Tachyon Rerank via /score or /score_batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
