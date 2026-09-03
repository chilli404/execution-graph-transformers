"""Background checkpoint sync: watches a local dir, copies new checkpoint files to S3.

Uses aws s3 cp for large files (safetensors/pt) since shutil.copy fails on
S3 FUSE for files >1GB. Falls back to shutil.copy for small files.

Usage:
  python scripts/sync_checkpoints_to_s3.py /tmp/fogen/pareto/1b_foo s3://grainger-mlops-pimmachinelearning-dev/fogen/pareto/1b_foo &

Runs until killed. Checks every 60 seconds for new checkpoint files.
"""
import subprocess
import sys
import time
from pathlib import Path

src = Path(sys.argv[1])
dst_str = sys.argv[2]
seen = set()

while True:
    if src.exists():
        for f in src.rglob("*"):
            if f.is_file() and f.suffix in (".safetensors", ".pt") and f not in seen:
                rel = f.relative_to(src)
                if dst_str.startswith("s3://"):
                    s3_path = f"{dst_str}/{rel}"
                else:
                    s3_path = f"{dst_str}/{rel}"
                try:
                    subprocess.run(
                        ["aws", "s3", "cp", str(f), s3_path, "--quiet"],
                        check=True, timeout=600
                    )
                    print(f"  synced {rel}", flush=True)
                    seen.add(f)
                except Exception as e:
                    print(f"  sync failed {rel}: {e}", flush=True)
    time.sleep(60)
