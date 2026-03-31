#!/usr/bin/env python3
"""Bump package versions in Cargo manifests.

Usage:
  python3 scripts/bump_version.py 0.1.1
  python3 scripts/bump_version.py 0.1.1 --root /path/to/repo
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_RE = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"\s*$', re.MULTILINE)
TARGETS = [
    Path("services/tachann/Cargo.toml"),
    Path("native/qnnshim/Cargo.toml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="New semantic version, e.g. 0.1.1")
    parser.add_argument("--root", default=".", help="Repository root")
    return parser.parse_args()


def validate_version(version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit(f"invalid semantic version: {version}")


def update_manifest(path: Path, version: str) -> None:
    text = path.read_text()
    updated, count = VERSION_RE.subn(f'version="{version}"', text, count=1)
    if count != 1:
        raise SystemExit(f"failed to update version in {path}")
    path.write_text(updated)


def main() -> None:
    args = parse_args()
    validate_version(args.version)
    root = Path(args.root).resolve()
    for relative in TARGETS:
        update_manifest(root / relative, args.version)


if __name__ == "__main__":
    main()
