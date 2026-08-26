# RTX PRO 6000 Blackwell Work Plan

## Objective

Use the 96GB RTX PRO 6000 Blackwell to answer the two largest remaining questions:

1. Does execution-graph equivalence continue to improve at 350–500M scale?
2. Does the fused parallel DAG provide real benefits in a native vLLM continuous-batching runtime?

This is a confirmation and deployment phase, not a new hyperparameter exploration phase.

---

# Work Package A: Environment and correctness

## A1. Freeze the software stack

Use an isolated environment rather than modifying a shared system Python.

Recommended starting point:

```text
Python: 3.12
vLLM: 0.27.1
CUDA/Torch: use the vLLM-supported pinned combination
Model dtype: BF16
```

Record:

- NVIDIA driver
- CUDA runtime
- PyTorch version
- vLLM version and commit
- GPU name and memory
- tokenizer hash
- checkpoint hash

## A2. Export the verified 120M checkpoint

Use:

```bash
PYTHONPATH=src python scripts/export_hf_checkpoint.py \
  --ckpt <checkpoint.safetensors> \
  --config <config_used.yaml> \
  --tokenizer_dir <tokenizer_dir> \
  --output <hf_model_dir>
```

The current HF wrapper has already passed export/reload tests with zero sequential and parallel logit difference.

## A3. Correctness gates

Before performance testing, verify:

1. first-token logits;
2. prefill logits;
3. cached decode logits;
4. greedy generation tokens;
5. sequential mode;
6. ordinary parallel mode;
7. fused parallel mode.

Required tolerance:

```text
FP32 reference vs BF16 runtime: define and report max/mean logit error
ordinary parallel vs fused parallel: exact or numerically negligible difference
```

Do not benchmark a runtime that fails correctness.

---

# Work Package B: Native vLLM plugin

## B1. Out-of-tree package

Suggested structure:

```text
fogen_vllm/
  pyproject.toml
  fogen_vllm/
    __init__.py
    model.py
    attention.py
    weight_loader.py
    fused_block.py
```

Register through `vllm.general_plugins` and `ModelRegistry`.

## B2. Model interface

Implement:

- flattened input tokens;
- explicit positions;
- vLLM intermediate tensors;
- pipeline-parallel placeholders even if unused initially;
- weight loading from the exported HF checkpoint;
- execution-mode selection in model config.

## B3. PagedAttention

Replace the local PyTorch SDPA cache path with vLLM PagedAttention while preserving:

- rotary positions;
- QK normalization;
- alternating value embeddings;
- value-mixing scalars;
- causal behavior;
- logit softcap.

## B4. Fused parallel block

The main kernel opportunity is

\[
[Q,K,V,F_{up}]=N(x)W_{fused}.
\]

The output must be numerically equivalent to the ordinary parallel graph before any speed claims are accepted.

---

# Work Package C: Serving benchmark

## C1. Modes

Benchmark the same graph-consistent checkpoint under:

1. sequential;
2. ordinary parallel;
3. fused parallel;
4. optional compiler-selected mixed DAG.

## C2. Workloads

At minimum cover:

```text
Concurrency: 1, 8, 32, 64
Prompt length: 128, 512, 2048, 4096
Output length: 32, 128, 512
```

Use both:

- fixed synthetic workloads for controlled comparison;
- a ShareGPT-like request-length distribution for realistic continuous batching.

## C3. Metrics

Report:

- requests/second;
- input tokens/second;
- output tokens/second;
- p50/p95 TTFT;
- p50/p95 TPOT;
- inter-token latency;
- peak GPU memory;
- KV-cache utilization;
- graph compilation/profiling overhead.

## C4. Serving success gate

Proceed with a systems claim only if:

- logits/generation correctness passes;
- fused TPOT improves by at least 10% under multiple realistic workloads;
- benefits survive concurrency, not only batch 1;
- peak memory does not materially regress;
- no workload has a severe latency tail regression hidden by the mean.

---

# Work Package D: 350–500M scale

## D1. Proposed model

A useful target near 430M parameters is approximately:

```yaml
model:
  vocab_size: 8192
  n_layer: 20
  d_model: 1152
  n_head: 18
  ctx_len: 1024
```

The exact parameter count must be verified before training.

## D2. Models to train

Do not repeat the full exploratory matrix. Train only:

1. sequential specialist;
2. graph-consistent model.

Add a parallel specialist only if cross-graph interpretation requires it.

## D3. Staged gates

### Gate 1: memory and numerical smoke

- 20 steps;
- both graph forwards;
- backward and optimizer states;
- fused inference path;
- no OOM or NaNs.

### Gate 2: optimization smoke

- several hundred steps;
- loss must decrease comparably to the sequential baseline;
- consistency defect must not explode;
- choose learning rate before long training.

### Gate 3: bounded-token scaling pilot

Start with a bounded pilot rather than compute-optimal pretraining. Use enough tokens to compare graph specialization and equivalence, but do not claim frontier language-model quality.

### Gate 4: full selected run

Only after Gates 1--3 pass, extend token budget.

## D4. Data

Available TinyStories preparation contains approximately 466M tokens. For ClimbMix, prepare additional deterministic shards before claiming a larger-data result.

Record unique-token budget separately from total sampled training tokens.

## D5. Scaling success criteria

- polymorphic sequential BPB within 3% of sequential specialist;
- parallel-mode BPB within 3% of sequential or parallel reference;
- cross-graph top-1 agreement at least 90%;
- single-layer to multi-graph KL Pearson at least 0.9;
- fused runtime speedup persists;
- no substantially worse training instability.

---

# Work Package E: Optional 1B pilot

Do this only if 350–500M succeeds.

The 1B experiment should be a short scaling probe, not a broad sweep. Use activation checkpointing and reduce batch size as needed. Its purpose is to test trends:

- Does graph equivalence become easier with scale?
- Does defect composition remain stable?
- Does fused speedup increase when matmuls dominate overhead?

Do not claim a fully trained 1B language model from an insufficient token budget.

---

# Work Package F: Prompt-adaptive routing

Current exact prompt defects are expensive and prompt-level coverage undercovers on ClimbMix.

A high-value extension is a lightweight learned defect predictor using cheap features:

- residual norm;
- Attention update norm;
- residual/Attention cosine;
- attention entropy;
- FFN input statistics;
- layer index.

Train a linear model, boosted tree, and small MLP to predict exact per-layer defects. Evaluate:

- held-out defect Spearman;
- cross-corpus transfer;
- compiler KL relative to random masks;
- predictor overhead;
- end-to-end serving speedup.

This is optional and should not block the static/aggregate compiler paper.

---

# Deliverables

The collaborator should return:

1. reproducible environment lock;
2. HF and vLLM correctness report;
3. native vLLM plugin code;
4. raw serving benchmark JSON/CSV;
5. 350–500M configs and checkpoints;
6. training curves and evaluation artifacts;
7. failures and negative workloads, not only best-case numbers;
8. concise recommendation on whether the vLLM and scale claims are publication-ready.

---

# Stop conditions

Stop or narrow a branch if:

- vLLM correctness cannot match the reference;
- fused speedup disappears under continuous batching;
- the 500M model does not learn under a small, principled LR check;
- graph consistency causes more than 5% persistent quality degradation;
- the experiment requires repeated post-hoc tuning to produce a positive result.
