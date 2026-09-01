#!/bin/bash
# Prepare 10B tokens of ClimbMix training data.
#
# Streams from HF (downloads only what it consumes), tokenizes,
# writes uint16 shards to data/climbmix/bpe8192/shards_10b/
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
import time
from pathlib import Path
from datasets import load_dataset
from fogen.data import load_tokenizer, write_shards

tok_dir = Path('data/climbmix/bpe8192')
shard_dir = Path('$SHARD_DIR')
tokenizer = load_tokenizer(str(tok_dir))

print('Streaming ClimbMix (downloads only what it consumes)...')
ds = load_dataset('karpathy/climbmix-400b-shuffle', split='train', streaming=True)

TARGET_TOKENS = 10_500_000_000
t0 = time.time()
doc_count = 0
tok_count = 0

def doc_iter():
    global doc_count, tok_count
    for row in ds:
        doc_count += 1
        text = row['text']
        tok_count += len(tokenizer.encode(text).ids)
        if doc_count % 50_000 == 0:
            elapsed = time.time() - t0
            docs_s = doc_count / elapsed
            eta_min = (TARGET_TOKENS - tok_count) / (tok_count / elapsed) / 60 if tok_count > 0 else 0
            n_shards = len(list(shard_dir.glob('shard_*.bin'))) if shard_dir.exists() else 0
            print(f'  {doc_count:>10,} docs  '
                  f'{tok_count/1e9:.2f}B/{TARGET_TOKENS/1e9:.1f}B tok  '
                  f'{n_shards} shards  '
                  f'{elapsed/60:.1f}min  '
                  f'{docs_s:.0f} docs/s  '
                  f'ETA {eta_min:.0f}min', flush=True)
        if tok_count >= TARGET_TOKENS:
            print(f'Reached {tok_count/1e9:.2f}B tokens, stopping.', flush=True)
            return
        yield text

print(f'Tokenizing to {shard_dir} ...')
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
