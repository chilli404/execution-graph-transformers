# One Model, Many DAGs

Research code and collaborator materials for **execution-graph equivalent Transformers**: training one checkpoint that can run under sequential, parallel, and mixed Attention–FFN execution DAGs while preserving quality.

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

## Verified headline results

Experiments use 117.5M-parameter decoder-only Transformers trained on ClimbMix and TinyStories.

- **Graph specialization:** sequential specialists degrade by roughly 35–38% BPB when switched to parallel; parallel specialists degrade by roughly 11–14% when switched to sequential.
- **Near-specialist quality:** the graph-consistent model is within 0.1% of the sequential specialist on ClimbMix and slightly outperforms both specialists on TinyStories.
- **Cross-graph agreement:** 90.5% on ClimbMix and 94.8% on TinyStories.
- **Mixed-DAG generalization:** more than 50 sampled 12-layer DAGs remain within 0.002 BPB of the reference graph.
- **Composition law:** sums of single-layer effects predict multi-layer graph KL with Pearson 0.96–0.97 and approximately 11% relative RMSE.
- **Serving:** fused parallel execution improves mean KV-cache TPOT by about 24–26%; fixed-shape `torch.compile` speedups reach 1.45–1.54x.
- **Necessary consistency:** joint CE and random-mask baselines preserve average BPB but have much worse agreement and 6x+ larger KL divergence.

These are exploratory but verified single-seed research results, not final publication claims. See the claim boundaries in `docs/COLLABORATOR_BRIEF.md`.

## Repository map

```text
docs/
  COLLABORATOR_BRIEF.md   Current state, verified results, limitations
  MESSAGE_TO_COLLABORATOR.md
paper/
  LAYMAN_GUIDE_CN.md      Chinese non-technical guide
  THEORY_GUIDE_CN.md      Chinese progressive theory guide
  theory.tex              Formal theorem/proof section
notebooks/
  01_results_overview.ipynb
  02_composition_and_compiler.ipynb
  03_serving_analysis.ipynb
blackwell/
  README.md               RTX PRO 6000 Blackwell work plan
src/fogen/
  model.py                Sequential/parallel/fused model and KV cache
  execution_graph.py      Defect-budget graph compiler
  hf_model.py             Hugging Face model interface
scripts/
  Analysis, evaluation, export, and plotting tools
configs/
  Successful 120M training and ablation configurations
results/
  Verified JSON artifacts and generated figures
```

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

The method, theory probes, cross-corpus results, aggregate graph compiler, prompt-level ranker, HF export, and PyTorch/KV-cache benchmarks are implemented. A native vLLM out-of-tree plugin and larger-model Blackwell validation are the primary next work packages.

## Quick start

```bash
python -m pip install torch transformers safetensors numpy scipy matplotlib pyyaml tokenizers datasets
PYTHONPATH=src python -m pytest src scripts -q
python scripts/make_execution_paper_figures.py
```

Large training data and checkpoints are intentionally not committed. See `docs/COLLABORATOR_BRIEF.md` and `blackwell/README.md` for artifact and replication guidance.
