# Collaborator Brief: Execution-Graph Equivalent Transformers

## Project status

The project has progressed beyond initial feasibility. We have a working learning objective, two-corpus 117.5M experiments, mechanism probes, a defect-budget compiler, serving benchmarks, and critical baselines. The next stage is larger-scale confirmation and native vLLM integration, not another broad hyperparameter sweep.

## The problem

A standard block computes Attention before FFN. A parallel block computes both from the same normalized residual state. Although the parameters and major operators are unchanged, specialist checkpoints fail under the alternate DAG.

### ClimbMix specialists

| Training graph | Sequential BPB | Parallel BPB |
|---|---:|---:|
| Sequential specialist | **1.1934** | 1.6488 |
| Parallel specialist | 1.3687 | **1.2006** |

### TinyStories specialists

| Training graph | Sequential BPB | Parallel BPB |
|---|---:|---:|
| Sequential specialist | **0.5369** | 0.7250 |
| Parallel specialist | 0.6006 | **0.5414** |

This is evidence of execution-DAG specialization.

## Main method

The graph-consistent model jointly optimizes sequential and parallel next-token losses plus centered-logit consistency:

\[
\mathcal L=\tfrac12\mathcal L_s+\tfrac12\mathcal L_p+
\lambda\|\bar z_s-\bar z_p\|^2.
\]

### Main model results

| Corpus | Poly sequential BPB | Poly parallel BPB | Argmax agreement |
|---|---:|---:|---:|
| ClimbMix | 1.1946 | 1.1960 | 90.5% |
| TinyStories | 0.5357 | 0.5368 | 94.8% |

The quality–consistency trade-off observed at 11.5M largely disappears at 117.5M.

## Critical baselines

| Method | Seq BPB | Par BPB | Agreement | Symmetric KL (nats/token) |
|---|---:|---:|---:|---:|
| Joint CE, no consistency | 1.1956 | 1.1990 | 81.3% | 31.5 |
| Random mask, 1x compute | 1.2010 | 1.2030 | 78.0% | 31.7 |
| Random mask, 2x compute | **1.1611** | **1.1631** | 71.8% | 53.7 |
| Graph consistency | 1.1946 | 1.1960 | **90.5%** | **4.82** |

More compute and lower BPB do not create graph equivalence. Consistency specifically reduces cross-graph functional divergence.

### A note on KL normalization

All symmetric KL values in this project are **per-token, in nats** (natural log). They are computed as `(KL(p||q) + KL(q||p)) / 2` using PyTorch's `F.kl_div(..., reduction="batchmean")`, which sums over the vocabulary dimension and averages over batch × sequence positions. At high agreement (>95% top-token match) the KL is dominated by tail disagreements across the full 8192-token vocabulary; a value of 4.82 nats/token is consistent with <0.002 BPB degradation because BPB measures average log-loss on the correct token while KL measures distributional divergence across all tokens. The manuscript must always state "per-token symmetric KL (nats)" — never bare numbers.

## Mixed-DAG generalization

The 12-layer model supports 4,096 possible sequential/parallel masks. More than 50 representative graphs were evaluated per corpus.

- Maximum BPB degradation is below 0.002.
- Minimum agreement for the main model is above 95% on the evaluated graph sets.
- Defect-to-KL Spearman is approximately 0.97 on both corpora.

## Theory results

For normalized FFN map \(g_l\) and Attention update \(a_l\), the exact local defect is

\[
d_l(x)=g_l(x+a_l(x))-g_l(x)
=\int_0^1Jg_l(x+t a_l(x))a_l(x)\,dt.
\]

First-order JVPs capture the defect direction with mean cosine 0.83–0.85. Linearization improves strongly with layer depth.

After completing all 12 single-layer probes, sums of single-layer effects predict multi-layer effects:

| Corpus | KL Spearman | KL Pearson | Relative RMSE | Slope |
|---|---:|---:|---:|---:|
| ClimbMix | 0.929 | 0.962 | 11.7% | 0.715 |
| TinyStories | 0.950 | 0.969 | 11.2% | 0.702 |

The observed composition is highly predictive and mildly subadditive.

## Graph compiler

The compiler solves

\[
\max_m\sum_lm_l
\quad\text{s.t.}\quad
\alpha\sum_lm_ld_l\le\epsilon.
\]

**Important caveats:**

- The composition law is empirical, not an exact bound. The budget ε is a predicted quality cost, not a guaranteed upper limit on actual KL divergence. The manuscript should describe ε as a "predicted divergence budget" rather than a "guarantee."
- For uniform latency savings, greedy sort-by-cost is provably optimal. For non-uniform savings, the DP solver is exact for its discretized cost grid (resolution-dependent); the optimality gap vs. the continuous problem is at most one bin width per selected layer.
- For deployment, a calibrated prediction margin (e.g., from held-out residual quantiles) should be added to convert the empirical prediction into a conservative budget. See `scripts/eval_compiler_calibration.py` for the calibration procedure.

At the aggregate graph level:

- held-out Spearman is 0.94–0.97;
- cross-corpus ClimbMix calibration to TinyStories test remains 0.97;
- selected graphs are within at most one parallel layer of the sampled oracle.

At prompt level, defects are useful rankers but not reliable 95% certificates on held-out ClimbMix prompts. This must remain a stated limitation.

## Serving results

### Eager and compiled full forward

- Eager fused speedup: approximately 1.20–1.29x.
- Fixed-shape `torch.compile`, batch 1: 1.45x.
- Fixed-shape `torch.compile`, batch 4: 1.54x.

### Real KV-cache decoding

| Corpus | Mean TTFT speedup | Mean TPOT speedup |
|---|---:|---:|
| ClimbMix | 1.22x | 1.26x |
| TinyStories | 1.19x | 1.24x |

The fused path is logit-equivalent to the ordinary parallel path.

## Standard capability check

LAMBADA exact match:

- ClimbMix specialists: 15.0%; polymorphic modes: 14.6% / 14.6%.
- TinyStories specialists: 11.2% / 9.6%; polymorphic modes: 10.6% / 10.5%.

No substantial new capability is claimed; the purpose is to verify that graph equivalence does not materially destroy language-model ability.

## Training cost

Polymorphic training executes two forward passes per step (sequential + parallel), so forward FLOPs are approximately 2× a standard run. Backward is 1× (single `.backward()` on the combined loss). Total training FLOPs per step are approximately 1.5× a specialist at the same scale.

The standard FLOPs estimate for a decoder-only Transformer forward pass is `6 × N × T` (where N = non-embedding parameters, T = tokens per batch). For polymorphic training, forward becomes `2 × 6 × N × T`, backward remains `2 × 6 × N × T`, giving `4 × 6NT` vs the specialist's `3 × 6NT` — a **1.33× total FLOPs overhead** per step.

At 7B scale, `memory_efficient` mode is used: sequential and parallel paths each run forward+backward separately with gradient checkpointing. This reduces peak activation memory from ~2× to ~1× (enabling 7B on a single 96GB GPU) but adds recomputation overhead from checkpointing, bringing the effective overhead closer to **1.5–1.7×**.

All scales (120M–7B) use identical step counts and data budgets for polymorphic and specialist configs. The economic argument is not "cheaper to train" but: **one polymorphic run replaces an otherwise hardware-specific family of specialist checkpoints and retains deployment optionality after training.**

| Scale | Specialist tok/s | Polymorphic tok/s | Wall-clock overhead |
|---|---:|---:|---:|
| 120M (L40S) | — | — | ~1.4× (estimated from FLOPs) |
| 430M (Blackwell) | — | ~9,319 | ~1.4× (estimated from FLOPs) |
| 7B (Blackwell) | — | — | ~1.6× (estimated, memory-efficient mode) |

*Note: specialist tok/s not yet measured on the same hardware. The table will be completed during manuscript assembly with matched wall-clock comparisons from training logs.*

## Negative and boundary results

- CUDA streams were slower on one GPU due to resource contention.
- A moving stop-gradient sequential teacher was unstable.
- Short retrofit training recovered parallel BPB but not high cross-graph agreement.
- Prompt-level calibration undercovers on ClimbMix; call it a ranker, not a certificate.
- vLLM has not yet been benchmarked.
- Most principal 120M results are from one locked seed per corpus.

## What is ready

- Sequential/parallel/mixed graph model implementation
- Graph-consistency training
- Exact KV-cache path
- Fused QKV+MLP-up inference
- Hugging Face export/reload with zero logit difference
- Theory section and proofs
- Aggregate and prompt-level compiler code
- Paper figures and raw JSON artifacts

## Highest-value collaborator tasks

1. Native vLLM out-of-tree model plugin and PagedAttention integration.
2. 350–500M replication on RTX PRO 6000 Blackwell.
3. Optional 1B short scaling pilot after data and runtime checks.
4. Production-style continuous-batching TTFT/TPOT benchmark.
5. Final manuscript assembly and related-work audit.

See `blackwell/README.md` for the operational work plan.

## Quantitative claims style guide

The following corrections must be observed during manuscript writing:

1. **C=200 serving is a regression, not neutral.** The measured TPOT is 2.55ms (sequential) → 2.73ms (fused), which is a **7% latency regression**. Do not describe this as "~0%" or "negligible." Report it honestly: fused parallel helps at interactive concurrency but regresses slightly under heavy batching.

2. **50.1% specialist agreement is not random chance.** For a vocabulary of 8192, uniform random agreement would be 0.012%. 50.1% top-1 agreement means the specialist retains substantial token-level overlap despite graph switching — it is a "catastrophic loss of graph agreement" or "near-total functional divergence," not chance-level.

3. **The consistency objective is necessary among tested methods, not universally sufficient.** Write: "the consistency term is necessary among the training objectives we evaluate and is sufficient to produce high graph agreement across the tested execution family." Do not claim bare "necessary and sufficient" — the claim is bounded by the ablation set and model family tested.
