#!/bin/bash
# Prepare 10B tokens of ClimbMix training data (fast version).
#
# Streams from HF, tokenizes with parallel workers, writes uint16 shards.
#
# Usage: bash blackwell/prep_data_10b.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.huggingface/token)
fi

SHARD_DIR="data/climbmix/bpe8192/shards_10b"

if [ -d "$SHARD_DIR" ] && ls "$SHARD_DIR"/shard_*.bin 1>/dev/null 2>&1; then
    echo "Shards already exist in $SHARD_DIR, skipping"
    exit 0
fi

echo "$(date): Preparing 10B-token ClimbMix dataset"
echo "Shard output dir: $(pwd)/$SHARD_DIR"

uv run python -u -c "
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
from datasets import load_dataset
from fogen.data import load_tokenizer

tok_dir = Path('data/climbmix/bpe8192')
shard_dir = Path('$SHARD_DIR')
shard_dir.mkdir(parents=True, exist_ok=True)

tokenizer = load_tokenizer(str(tok_dir))

TARGET_TOKENS = 10_500_000_000
SHARD_TOKENS = 100_000_000
N_WORKERS = os.cpu_count() or 8

print(f'Streaming ClimbMix, tokenizing with {N_WORKERS} threads...')
ds = load_dataset('karpathy/climbmix-400b-shuffle', split='train', streaming=True)

t0 = time.time()
total_tokens = 0
doc_count = 0
shard_idx = 0
buf = []
hashes = []

def flush():
    global buf, shard_idx
    if not buf:
        return
    arr = np.asarray(buf, dtype=np.uint16)
    p = shard_dir / f'shard_{shard_idx:05d}.bin'
    arr.tofile(p)
    hashes.append({'file': p.name, 'tokens': len(arr),
                   'sha256': hashlib.sha256(arr.tobytes()).hexdigest()})
    buf = []
    shard_idx += 1

# Batch docs, tokenize in parallel with threads
# (tokenizers library releases the GIL, so threads give real parallelism)
BATCH_SIZE = 1000

def tokenize_batch(texts):
    return [tokenizer.encode(t).ids for t in texts]

batch = []
reached_target = False

for row in ds:
    batch.append(row['text'])
    doc_count += 1

    if len(batch) >= BATCH_SIZE:
        # tokenizers.Tokenizer.encode_batch is even faster
        encoded = tokenizer.encode_batch(batch)
        for enc in encoded:
            ids = enc.ids
            buf.extend(ids)
            buf.append(0)  # eot_id
            total_tokens += len(ids) + 1
            if len(buf) >= SHARD_TOKENS:
                flush()
        batch = []

        if doc_count % 50_000 < BATCH_SIZE:
            elapsed = time.time() - t0
            tok_s = total_tokens / elapsed
            eta_min = (TARGET_TOKENS - total_tokens) / tok_s / 60 if tok_s > 0 else 0
            print(f'  {doc_count:>10,} docs  '
                  f'{total_tokens/1e9:.2f}B/{TARGET_TOKENS/1e9:.1f}B tok  '
                  f'{shard_idx} shards  '
                  f'{elapsed/60:.1f}min  '
                  f'{tok_s/1e6:.1f}M tok/s  '
                  f'ETA {eta_min:.0f}min', flush=True)

        if total_tokens >= TARGET_TOKENS:
            reached_target = True
            break

# Flush remaining
if batch:
    encoded = tokenizer.encode_batch(batch)
    for enc in encoded:
        ids = enc.ids
        buf.extend(ids)
        buf.append(0)
        total_tokens += len(ids) + 1
        if len(buf) >= SHARD_TOKENS:
            flush()
flush()

manifest = {'total_tokens': total_tokens, 'shards': hashes, 'dtype': 'uint16'}
(shard_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

elapsed = time.time() - t0
print(f'Done: {total_tokens:,} tokens in {len(hashes)} shards ({elapsed/60:.1f}min)')
print(f'Avg throughput: {total_tokens/elapsed/1e6:.1f}M tok/s')
"

echo "$(date): 10B data prep complete"
