"""Record an audit trail for artifacts before deleting regenerable heavy files.

Usage:
  python -I scripts/manifest.py <out.json> <label>=<path> [<label>=<path> ...]

For every file under each path it stores relative path, size and sha256, plus the
Python package versions, so a deleted model or dataset can be re-created and checked
byte-for-byte later. Appends to <out.json> if it exists.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from importlib import metadata
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = Path(sys.argv[1])
    doc = json.loads(out.read_text()) if out.exists() else {"entries": {}}
    doc["packages"] = {k: metadata.version(k) for k in ("mlx", "mlx-lm", "transformers", "numpy", "pandas")
                       if _has(k)}
    for arg in sys.argv[2:]:
        label, _, path = arg.partition("=")
        root = Path(path).expanduser()
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        doc["entries"][label] = {
            "path": str(root), "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_bytes": sum(p.stat().st_size for p in files),
            "files": [{"rel": str(p.relative_to(root)) if root.is_dir() else p.name,
                       "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
        }
        print(f"{label}: {len(files)} files, {doc['entries'][label]['total_bytes'] / 1e9:.2f} GB")
    out.write_text(json.dumps(doc, indent=1))


def _has(pkg: str) -> bool:
    try:
        metadata.version(pkg)
        return True
    except metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
