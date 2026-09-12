"""Copy result files (JSON/JSONL/YAML) from local dir to S3 mount.

Uses manual read+write instead of shutil.copy because S3 FUSE
does not support chmod (shutil.copy calls copymode which fails).

Usage:
  python scripts/copy_results_to_s3.py /tmp/fogen/pareto /s3-data/fogen/pareto
"""
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

for f in src.rglob("*"):
    if f.is_file() and f.suffix in (".json", ".jsonl", ".yaml"):
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = f.read_bytes()
        out.write_bytes(data)
        print(f"  {f.relative_to(src)} -> {out}")
