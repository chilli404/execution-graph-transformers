#!/bin/bash
# Prepare 10B tokens of ClimbMix training data.
#
# Uses load_dataset (non-streaming) which downloads parquet files
# in parallel and caches them, then tokenizes from local cache.
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

uv run python -u -c "
import time
from pathlib import Path
from datasets import load_dataset
from fogen.data import load_tokenizer, write_shards

tok_dir = Path('data/climbmix/bpe8192')
shard_dir = Path('$SHARD_DIR')
tokenizer = load_tokenizer(str(tok_dir))

# Download first 3% of ClimbMix-400B (~12B tokens).
# load_dataset downloads parquet files in parallel and caches to disk.
print('Downloading ClimbMix (first 3%, ~12B tokens)...')
print('This downloads in parallel — much faster than streaming.')
ds = load_dataset('karpathy/climbmix-400b-shuffle', split='train[:3%]')
print(f'Downloaded {len(ds):,} documents')

TARGET_TOKENS = 10_500_000_000
t0 = time.time()
doc_count = 0
tok_estimate = 0
total_docs = len(ds)

def doc_iter():
    global doc_count, tok_estimate
    for row in ds:
        doc_count += 1
        text = row['text']
        tok_estimate += len(text) // 4
        if doc_count % 100_000 == 0:
            elapsed = time.time() - t0
            docs_s = doc_count / elapsed
            pct = 100 * doc_count / total_docs
            print(f'  {doc_count:>10,}/{total_docs:,} ({pct:.1f}%)  '
                  f'~{tok_estimate/1e9:.1f}B tok (est)  '
                  f'{elapsed/60:.1f}min  '
                  f'{docs_s:.0f} docs/s', flush=True)
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

echo "$(date): 10B data prep complete"
