#!/bin/bash
# Prepare 10B tokens of ClimbMix training data.
# Queue with: tsp bash blackwell/prep_data_10b.sh
#
# Downloads from karpathy/climbmix-400b-shuffle (streaming),
# reuses the existing BPE-8192 tokenizer, writes new shards to
# data/climbmix/bpe8192/shards_10b/
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.huggingface/token)
fi

echo "$(date): Preparing 10B-token ClimbMix dataset"

uv run python -u -c "
from pathlib import Path
from datasets import load_dataset
from fogen.data import load_tokenizer, write_shards

tok_dir = Path('data/climbmix/bpe8192')
shard_dir = tok_dir / 'shards_10b'

if shard_dir.exists() and list(shard_dir.glob('shard_*.bin')):
    print('Shards already exist, skipping')
    exit(0)

tokenizer = load_tokenizer(str(tok_dir))

# Stream ClimbMix and write shards on the fly.
# Target: ~10B tokens. At ~750 tokens/doc, need ~13.3M docs.
# Over-fetch slightly to ensure we hit 10B after tokenization.
TARGET_DOCS = 14_000_000

print(f'Streaming {TARGET_DOCS:,} docs from ClimbMix...')
ds = load_dataset('karpathy/climbmix-400b-shuffle', split='train', streaming=True)

def doc_iter():
    for i, row in enumerate(ds):
        if i >= TARGET_DOCS:
            break
        if i % 1_000_000 == 0 and i > 0:
            print(f'  streamed {i:,} docs...')
        yield row['text']

print('Tokenizing and writing shards...')
manifest = write_shards(
    doc_iter(),
    tokenizer,
    out_dir=str(shard_dir),
    shard_tokens=100_000_000,  # 100M tokens per shard
)
print(f'Done: {manifest[\"total_tokens\"]:,} tokens in {len(manifest[\"shards\"])} shards')
print(f'Data at: {shard_dir}')
"

echo "$(date): 10B data prep complete"
