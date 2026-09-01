#!/bin/bash
# Prepare 10B tokens of ClimbMix training data.
#
# 1. Downloads ~250 parquet files in parallel via huggingface-cli
# 2. Tokenizes from local disk with progress
# 3. Cleans up parquet files
#
# Usage: bash blackwell/prep_data_10b.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.huggingface/token)
fi

SHARD_DIR="data/climbmix/bpe8192/shards_10b"
RAW_DIR="data/climbmix/raw_10b"

if [ -d "$SHARD_DIR" ] && ls "$SHARD_DIR"/shard_*.bin 1>/dev/null 2>&1; then
    echo "Shards already exist in $SHARD_DIR, skipping"
    exit 0
fi

# Step 1: Download parquet files in parallel
# ClimbMix has ~6543 files, ~70k docs each. 250 files ≈ 17.5M docs ≈ 12B+ tokens.
echo "$(date): Downloading parquet files..."
uv run huggingface-cli download karpathy/climbmix-400b-shuffle \
    --repo-type dataset \
    --include "data/train-0000[0-2]*.parquet" \
    --local-dir "$RAW_DIR"

echo "$(date): Download complete. Tokenizing..."

# Step 2: Tokenize from local disk
uv run python -u -c "
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pyarrow.parquet as pq

from fogen.data import load_tokenizer, write_shards

tok_dir = Path('data/climbmix/bpe8192')
shard_dir = Path('$SHARD_DIR')
raw_dir = Path('$RAW_DIR')

tokenizer = load_tokenizer(str(tok_dir))

parquet_files = sorted(glob.glob(str(raw_dir / 'data' / 'train-0000[0-2]*.parquet')))
if not parquet_files:
    parquet_files = sorted(glob.glob(str(raw_dir / '**' / 'train-0000[0-2]*.parquet'), recursive=True))
print(f'Found {len(parquet_files)} parquet files')

TARGET_TOKENS = 10_500_000_000  # slight overshoot to ensure 10B after trimming
t0 = time.time()
doc_count = 0
tok_estimate = 0

def doc_iter():
    global doc_count, tok_estimate
    for fi, pf in enumerate(parquet_files):
        table = pq.read_table(pf, columns=['text'])
        texts = table.column('text').to_pylist()
        for text in texts:
            doc_count += 1
            tok_estimate += len(text) // 4  # rough bytes-to-tokens estimate
            if doc_count % 100_000 == 0:
                elapsed = time.time() - t0
                docs_s = doc_count / elapsed
                print(f'  {doc_count:>10,} docs  '
                      f'~{tok_estimate/1e9:.1f}B tok (est)  '
                      f'{elapsed/60:.1f}min  '
                      f'{docs_s:.0f} docs/s  '
                      f'file {fi+1}/{len(parquet_files)}', flush=True)
            if tok_estimate > TARGET_TOKENS:
                return
            yield text

print('Tokenizing and writing shards...')
manifest = write_shards(
    doc_iter(),
    tokenizer,
    out_dir=str(shard_dir),
    shard_tokens=100_000_000,
)
elapsed = time.time() - t0
print(f'Done: {manifest[\"total_tokens\"]:,} tokens in {len(manifest[\"shards\"])} shards ({elapsed/60:.1f}min)')
"

# Step 3: Clean up parquet files
echo "$(date): Cleaning up parquet download..."
rm -rf "$RAW_DIR"

echo "$(date): 10B data prep complete"
