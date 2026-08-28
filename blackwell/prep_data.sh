#!/bin/bash
# Prepare training data (tokenizer + shards)
# Queue with: tsp bash blackwell/prep_data.sh
#
# Downloads dataset from HuggingFace, trains BPE-8192 tokenizer,
# writes uint16 shards.
#
# Set DATASET below: "climbmix" (default, matches paper) or "tinystories"
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-climbmix}"

# Set HF_TOKEN if not already in environment
# Either export it before running, or place it in ~/.huggingface/token
if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.huggingface/token)
fi

echo "$(date): Preparing $DATASET data"

uv run python -c "
import sys
from pathlib import Path
from datasets import load_dataset

from fogen.data import train_tokenizer, write_shards, load_tokenizer

dataset = '$DATASET'

if dataset == 'climbmix':
    out_base = Path('data/climbmix/bpe8192')
    print('Downloading ClimbMix (streaming, first 100M tokens worth)...')
    ds = load_dataset('karpathy/climbmix-400b-shuffle', split='train', streaming=True)
    # ClimbMix is huge (400B tokens). Take enough for ~100M tokens after BPE.
    # Rough estimate: 1 doc ~ 500 tokens, so 200k docs ~ 100M tokens
    docs = []
    for i, row in enumerate(ds):
        docs.append(row['text'])
        if i >= 200_000:
            break
    print(f'  Downloaded {len(docs)} documents')
elif dataset == 'tinystories':
    out_base = Path('data/tinystories/bpe8192')
    print('Downloading TinyStories...')
    ds = load_dataset('roneneldan/TinyStories', split='train')
    docs = [row['text'] for row in ds]
    print(f'  {len(docs)} documents')
else:
    print(f'Unknown dataset: {dataset}')
    sys.exit(1)

tok_dir = out_base
shard_dir = out_base / 'shards'

# Train tokenizer
if not (tok_dir / 'tokenizer.json').exists():
    print('Training BPE-8192 tokenizer...')
    tok_dir.mkdir(parents=True, exist_ok=True)
    train_tokenizer(iter(docs), vocab_size=8192, out_dir=str(tok_dir))
    print('  Done')
else:
    print('Tokenizer already exists, skipping')

# Write shards
if not shard_dir.exists() or not list(shard_dir.glob('shard_*.bin')):
    print('Tokenizing and writing shards...')
    tokenizer = load_tokenizer(str(tok_dir))
    manifest = write_shards(
        iter(docs),
        tokenizer,
        out_dir=str(shard_dir),
        shard_tokens=50_000_000,
    )
    print(f'  {manifest[\"total_tokens\"]} tokens in {len(manifest[\"shards\"])} shards')
else:
    print('Shards already exist, skipping')

print('Done. Data at:', out_base)
"

echo "$(date): Data prep complete"
