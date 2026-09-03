"""Copy ALL files from local dir to S3 mount (including checkpoints).

Uses write_bytes — no shutil.copy, no chmod, no permissions issues.

Usage:
  python scripts/copy_all_to_s3.py /tmp/fogen/pareto/430m_foo /s3-data/fogen/pareto/430m_foo
"""
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

for f in src.rglob("*"):
    if f.is_file():
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.write_bytes(f.read_bytes())
            print(f"  {f.relative_to(src)} ({f.stat().st_size // 1024}KB)")
        except Exception as e:
            print(f"  FAILED {f.relative_to(src)}: {e}")
