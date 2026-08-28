#!/bin/bash
# Prepare TinyStories training data (tokenizer + shards)
# Queue with: tsp bash blackwell/prep_data.sh
#
# Downloads TinyStories from HuggingFace, trains BPE-8192 tokenizer,
# writes uint16 shards. ~10 minutes, ~2GB disk.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "$(date): Preparing TinyStories data"

uv run python -c "
from pathlib import Path
from datasets import load_dataset

from fogen.data import train_tokenizer, write_shards, load_tokenizer

out_base = Path('data/tinystories/bpe8192')
tok_dir = out_base
shard_dir = out_base / 'shards'

# Download
print('Downloading TinyStories...')
ds = load_dataset('roneneldan/TinyStories', split='train')
print(f'  {len(ds)} documents')

# Train tokenizer
if not (tok_dir / 'tokenizer.json').exists():
    print('Training BPE-8192 tokenizer...')
    tok_dir.mkdir(parents=True, exist_ok=True)
    train_tokenizer((doc['text'] for doc in ds), vocab_size=8192, out_dir=str(tok_dir))
    print('  Done')
else:
    print('Tokenizer already exists, skipping')

# Write shards
if not shard_dir.exists() or not list(shard_dir.glob('shard_*.bin')):
    print('Tokenizing and writing shards...')
    tokenizer = load_tokenizer(str(tok_dir))
    manifest = write_shards(
        (doc['text'] for doc in ds),
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
echo "Update config paths to: data/tinystories/bpe8192"
