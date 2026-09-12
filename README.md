# One Model, Many DAGs

Research code and collaborator materials for **execution-graph equivalent Transformers**: training one checkpoint that can run under sequential, parallel, and mixed Attention–FFN execution DAGs while preserving quality.

> **Status:** ICLR 2027 submission in progress. All experiments complete (120M–7B primary architecture + Llama 430M/1B). See `docs/EXPERIMENT_STATUS.md` for the full results inventory.

## Core idea

A standard Transformer block is sequential:

```text
x -> Attention -> FFN -> output
```

A parallel block lets Attention and FFN read the same normalized input:

```text
              -> Attention -
x -> Norm(x)                 + -> output
              -> FFN -------
```

Ordinary Transformer weights specialize to the graph used during training. Changing only the execution dependency can substantially degrade quality. We train with a graph-consistency objective so the same weights implement approximately equivalent functions across execution DAGs.

## Headline results

| Scale | Agreement | Sym KL | ΔBPB | Architecture |
|-------|-----------|--------|------|--------------|
| 120M | 95.4% | 5.86 | 0.0014 | Primary |
| 430M | 96.5% | 4.68 | 0.0016 | Primary |
| 1B | 96.1% | 4.10 | 0.0012 | Primary |
| 3B (gradnorm) | 98.3% | 0.92 | 0.0012 | Primary |
| 7B | 95.6% | 3.78 | 0.0017 | Primary |
| 430M | 93.6% | 10.4 | 0.003 | Llama |
| 1B | 91.2% | 26.1 | 0.002 | Llama |

- **Specialist collapse:** 430M specialist gets 54% agreement (KL=1273) under graph switch. Polymorphic model: 96.5% (KL=4.68).
- **Composition law:** single-layer defects predict 10k+ held-out mixed DAGs with Pearson 0.97 (primary) and 0.85 (Llama). Calibrated compiler reduces budget violations from 33–47% to <1%.
- **Hardware compilation:** L40S 1.20–1.54×, Apple M4 1.14×, 7B TP=2 1.20× from communication overlap.
- **Downstream parity:** 7B lm-eval (HellaSwag, PIQA, WinoGrande, ARC) — all execution graphs within 1 stderr.
- **Loss analysis:** MSE and KL both achieve graph consistency. KL gradients are 10–58× larger than LM gradients; gradient-normalized weighting resolves scale sensitivity.

See `docs/EXPERIMENT_STATUS.md` for complete results with interpretation.

## Repository map

```text
src/fogen/
  model.py                Transformer with 4 execution modes (seq/par/fused/cuda)
                          Supports both primary (ReLU², QK-norm, value embeddings)
                          and Llama-style (RMSNorm, SwiGLU) architectures
  training/train.py       Config-driven training with polymorphic loss,
                          multiple consistency objectives, gradient-normalized cw
  execution_graph.py      Defect-budget graph compiler (greedy + DP)
  hf_model.py             Hugging Face AutoModel interface
  evals/                  BPB evaluation and forced-choice scoring

configs/                  120M training and ablation configs
  pareto/                 430M and 1B loss-objective sweep configs
                          Naming: {scale}_{objective}_{temperature}_cw{weight}

blackwell/                Scale-up experiments (430M–7B)
  configs/                Training configs for 430M, 1B, 3B, 7B, Llama variants
  vllm_plugin/            Native vLLM out-of-tree plugin with PagedAttention + TP
  results/organized/      All experiment results (see below)

scripts/                  Evaluation, analysis, and export tools
  eval_graph_rewrites.py        Full graph-rewrite sweep (agreement, KL, BPB)
  eval_composition_holdout.py   10k+ mask holdout composition test
  eval_pairwise_mechanism.py    C(L,2) pairwise interaction analysis
  measure_gradient_ratio.py     ||∇consistency|| / ||∇LM|| measurement
  eval_compiler_calibration.py  Calibrated compiler violation rates
  export_hf_checkpoint.py       Export to HuggingFace format

docs/
  EXPERIMENT_STATUS.md    Complete results inventory with interpretation

paper/                    Theory and exposition
  theory.tex              Formal theorems and proofs
  LAYMAN_GUIDE_{EN,CN}.md
  THEORY_GUIDE_{EN,CN}.md

results/                  Original 120M figures and JSON artifacts
```

### Results structure

All scale-up results live in `blackwell/results/organized/`:

```text
organized/
  scale_series/       120M–7B primary architecture graph rewrites and composition
  llama/              Llama 430M + 1B (specialist collapse, composition, calibration)
  pareto_430m/        12 loss-objective configs (MSE × 5λ, KL T=1 × 4λ, KL T=4 × 3λ)
  pareto_1b/          12 loss-objective configs (MSE × 2, KL × 10)
  ablation_120m/      6 objectives × 3 seeds
  gradient_analysis/  ||∇MSE||/||∇LM|| and ||∇KL||/||∇LM|| at 120M–7B
  mechanism/          Full pairwise interaction + defect cosine (430M, 7B)
  serving/            vLLM, TP=2, hardware profiles
  downstream/         7B lm-eval (seq/par/compiled)
```

## Suggested reading order

1. This README
2. `docs/EXPERIMENT_STATUS.md` — all experiments and results with interpretation
3. `paper/theory.tex` — formal theorems and proofs

## Main research question

> Can a neural network learn an equivalence class of execution graphs, so deployment can compile one checkpoint into a hardware-specific DAG without retraining?

## Training objective

For sequential and parallel logits \(z_s,z_p\):

\[
\mathcal L
=\tfrac12\mathcal L_{\mathrm{seq}}
+\tfrac12\mathcal L_{\mathrm{par}}
+\lambda\|\bar z_s-\bar z_p\|_2^2,
\]

where centered logits remove vocabulary-wide shifts that do not affect softmax probabilities.

## Theory in one line

The local graph defect is exactly

\[
d_l(x)=g_l(x+a_l(x))-g_l(x)
=\int_0^1Jg_l(x+t a_l(x))a_l(x)\,dt.
\]

This connects Attention updates, FFN steering, execution-graph divergence, and defect-guided graph compilation.

## Current status

All experiments complete through 7B (primary architecture) and 1B (Llama). Results include: scale series 120M–7B, second architecture validation, loss-objective Pareto sweeps at 430M and 1B, gradient ratio analysis, gradient-normalized training at 3B, pairwise mechanism analysis, compiler calibration, vLLM serving, TP=2 communication overlap, and downstream benchmarks. Paper writing in progress for ICLR 2027 (deadline Sep 25).

## Quick start

```bash
uv sync --extra dev
uv run pytest -q
uv run python scripts/make_execution_paper_figures.py
```

Or without uv:

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/make_execution_paper_figures.py
```

Large training data and checkpoints are intentionally not committed. See `docs/EXPERIMENT_STATUS.md` for the full results inventory.
