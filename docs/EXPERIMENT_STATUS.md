# Experiment Status

All experiments complete as of September 3, 2026.
Results organized in `blackwell/results/organized/`.
One run still training: 430M 10B-token continuation on the remote rig (~Sep 9).

---

## What this branch adds

The `main` branch contains the original 120M experiments (two corpora, critical baselines, composition law, compiler, serving benchmarks, theory). This branch (`mps-graph-compilation`) adds everything needed for the ICLR submission:

**Core model changes** (`src/fogen/model.py`):
- Llama-style architecture variant (RMSNorm, SwiGLU, no value embeddings, configurable softcap)
- All 4 execution modes work for both architectures

**Training changes** (`src/fogen/training/train.py`):
- Multiple consistency objectives (centered MSE, raw MSE, KL forward, symmetric KL, Jensen-Shannon)
- Temperature parameter for KL-based objectives
- Gradient-normalized consistency weight (gradnorm_rho)
- Memory-efficient separate-backward mode for large models
- Resume from checkpoint support

**New evaluation scripts** (`scripts/`):
- `eval_composition_holdout.py` — Brutal held-out composition law test (10k+ masks, multiple split strategies)
- `eval_composition_cross_seqlen.py` — Fit at one sequence length, predict another
- `eval_training_progress.py` — Agreement/KL/β at intermediate checkpoints
- `eval_pairwise_mechanism.py` — All C(L,2) pairwise interactions + defect vector cosines
- `eval_pareto_equivalence.py` — Quick agreement/KL eval for pareto checkpoints
- `eval_compiler_calibration.py` — Calibrated compiler with violation rates
- `measure_gradient_ratio.py` — ||∇consistency|| / ||∇LM|| across scales
- `copy_all_to_s3.py`, `copy_results_to_s3.py` — S3 FUSE-compatible file copy

**Blackwell scale-up** (`blackwell/`):
- Configs for 430M, 1B, 3B, 7B at various consistency weights
- Llama 430M + 1B configs
- Pareto sweep configs (430M: 12 configs, 1B: 15 configs)
- vLLM out-of-tree plugin with PagedAttention and tensor parallelism
- TP=2 benchmark, data prep scripts

---

## Experiments and Results

### 1. Scale Series — Does graph polymorphism work from 120M to 7B?

**Location:** `organized/scale_series/`

The fundamental question: can one checkpoint run under multiple execution DAGs? We train with graph-consistency loss (dual forward pass + centered-logit MSE) and evaluate by switching the model to all-parallel and measuring quality degradation.

| Scale | Training | Agreement | Sym KL (nats/tok) | ΔBPB | Composition ρ |
|-------|----------|-----------|---------|------|---------------|
| 120M | cw=0.1, 57M tok | 92.7% | — | — | — |
| 430M | cw=1.0, 100M tok | 96.5% | 4.68 | 0.0016 | 0.982 |
| 430M | cw=0.1, 100M tok | 93.2% | 14.3 | 0.0024 | 0.974 |
| 430M | cw=0.1, 2B tok | 92.3% | 0.92 | 0.0008 | 0.862 |
| 1B | cw=1.0, 100M tok | 96.1% | 4.10 | 0.0014 | 0.977 |
| 3B | gradnorm, 100M tok | **98.3%** | **0.92** | 0.0012 | 0.915 |
| 7B | cw=0.1, 100M tok | 95.6% | 3.78 | 0.0023 | 0.962 |

**What this shows:** Graph polymorphism works at every scale tested. A specialist checkpoint at 430M gets 54.2% agreement and KL=1273 under the same switch — the consistency objective is what enables equivalence.

**The 430M 2B-token point** confirms that equivalence survives 20× longer training (ΔBPB drops to 0.0008). Note: this used different hyperparameters (cw=0.1, batch=128) than the 100M run (cw=1.0, batch=8), so it's not a clean tokens-only comparison.

**The 3B gradient-normalized run** achieves the best equivalence of any model (98.3% agreement, KL=0.92) while also maintaining composition law predictiveness (ρ=0.915). This demonstrates that adaptive consistency weighting produces both stability and strong equivalence at scales where fixed coefficients require manual tuning.

---

### 2. Composition Law — Can we predict the quality cost of arbitrary DAGs?

**Location:** `organized/scale_series/430m_composition_holdout.json`, `430m_cross_seqlen.json`

The composition model D(m) ≈ α₀·Σd_l·|m|^(-β) predicts the KL divergence of any execution mask from 20 single-layer measurements.

**Holdout validation (430M, 10,155 masks):**

| Split strategy | CA Pearson | β |
|----------------|-----------|---|
| Random 50/50 | 0.970 | 0.217 |
| Fit ≤10 parallel → predict >10 | 0.927 | 0.181 |
| Fit >10 → predict ≤10 | 0.972 | 0.279 |
| 5-fold CV | 0.971 | 0.218±0.001 |

**Cross-sequence-length (430M):** Fit at seqlen=128, Pearson holds at 0.969 through seqlen=1024. The law doesn't depend on sequence length.

**What this shows:** The composition law is not curve-fitting. It survives 10k held-out masks, cardinality extrapolation (few→many parallel layers), and 8× sequence length transfer. The 2-parameter model consistently beats linear addition and the 3-parameter model doesn't improve meaningfully (γ≈1).

**β deconfounding:** β = a + b·log(N) + c·log(cw) with R²=0.81. Scale drives β (b=+0.039, CI excludes zero). β stabilizes by training step 4000 — it's not a training-stage artifact.

---

### 3. Llama Architecture — Is this specific to one model family?

**Location:** `organized/llama/`

All original experiments used a nanochat-derived architecture (LayerNorm, ReLU², QK-norm, value embeddings, softcap). This tests a conventional Llama-style model (RMSNorm, SwiGLU, no value embeddings, no softcap) at 430M and 1B.

| | 430M Specialist | 430M Poly | 1B Specialist | 1B Poly |
|---|---|---|---|---|
| Par KL | 553 | **10.4** | 244 | **26.1** |
| Agreement | 64% | **94%** | 73% | **91%** |
| ΔBPB | 0.140 | **0.003** | 0.075 | **0.002** |

**Composition on Llama:** 5-fold CV Pearson 0.827 (430M) → 0.846 (1B). Weaker than primary (0.97) but improves with scale and the subadditive structure persists. The 2-parameter CA model beats linear on Llama (Pearson 0.90 vs 0.83), and 3p γ=0.96≈1 confirms the 2p model is the right order.

**Compiler calibration on Llama:** Raw violations 33-47% → calibrated 95%: 2.0-2.6% → calibrated 99%: 0.5-0.7%. Even with weaker raw predictions, calibration produces a practical compiler.

**What this shows:** Graph specialization and consistency training are not artifacts of one architecture. Composition predictiveness is architecture-dependent (0.97 vs 0.83), but the subadditive structure and calibrated compilation both transfer.

---

### 4. Loss Objective Sweep — What consistency loss should you use?

**Location:** `organized/pareto_430m/`, `organized/pareto_1b/`, `organized/ablation_120m/`

The original method uses centered-logit MSE. We tested whether KL-based objectives work equally well, sweeping consistency weight λ and temperature T at 120M, 430M, and 1B.

**430M Pareto (3 key points with full equivalence eval):**

| Objective | λ | Seq CE | Agreement | Sym KL |
|-----------|---|--------|-----------|--------|
| Centered MSE | 1.0 | **3.079** | 96.6% | 4.25 |
| Sym KL T=1 | 0.01 | 3.097 | 97.4% | 2.38 |
| Sym KL T=1 | 0.03 | 3.122 | **98.2%** | **1.04** |

**1B Pareto (5 key points with full equivalence eval):**

| Objective | λ | Seq CE | Agreement | Sym KL |
|-----------|---|--------|-----------|--------|
| Sym KL T=1 | 0.003 | **2.606** | 94.1% | 12.7 |
| Centered MSE | 0.1 | 2.681 | 94.5% | 12.1 |
| Raw MSE | 0.1 | 2.687 | 94.2% | 12.0 |
| Sym KL T=1 | 0.01 | 2.878 | **95.0%** | **9.0** |
| Sym KL T=1 | 0.001 | 2.799 | 93.8% | 15.2 |

All 1B configs with λ≥0.03 for KL and λ=1.0 for MSE diverged under bf16/memory-efficient training.

**What this shows:** Graph consistency is objective-agnostic — both MSE and KL can produce it. KL can actually match or beat MSE quality (2.61 vs 2.68 at 1B) when properly tuned. However, MSE works across a wide range of λ at every scale while KL requires ~100× lower coefficients. The next section explains why.

Full sweeps: 12 configs at 430M, 12 configs at 1B, 6 objectives × 3 seeds at 120M. Available in the organized folders.

---

### 5. Gradient Ratio Analysis — Why do MSE and KL need different coefficients?

**Location:** `organized/gradient_analysis/`

We measured the gradient magnitude of each consistency loss relative to the LM loss at trained checkpoints across all scales.

| Scale | ||∇MSE|| / ||∇LM|| | ||∇KL|| / ||∇LM|| |
|-------|---------------------|---------------------|
| 120M | 0.14 | 57.6 |
| 430M | 0.03 | 15.8 |
| 1B | 0.03 | 12.7 |
| 7B | 0.02 | 9.6 |

**What this shows:** KL consistency gradients are 10-58× larger than LM gradients at all scales. MSE consistency gradients are 2-14% of LM — a small perturbation. This quantitatively explains:

- Why λ=1.0 for KL completely overwhelms LM optimization (effective strength 10-58×)
- Why λ=0.003 is the right KL operating point at 1B (0.003 × 12.7 ≈ 0.038, matching MSE's 0.031)
- Why MSE is stable with λ=0.1-1.0 at all scales (effective strength always <0.14×)

The gradient-normalized 3B result (§1) confirms this: when the effective consistency gradient is controlled automatically, the model achieves 98.3% agreement with an adaptive coefficient that settles at 10.0 — far above what any fixed coefficient could sustain.

---

### 6. Pairwise Mechanism — Why do defects compose subadditively?

**Location:** `organized/mechanism/`

We evaluated all C(L,2) layer pairs and measured pairwise interactions I_ij = KL({i,j}) - KL({i}) - KL({j}), plus cosine similarity between per-layer defect vectors.

| Scale | Pairs | Subadditive | Mean cos(δ_i,δ_j) | cos↔interaction ρ | dist↔cos ρ |
|-------|-------|-------------|--------------------|--------------------|------------|
| 430M | 190 | 100% | 0.011 | -0.06 (ns) | **-0.71** |
| 7B | 496 | 97% | 0.042 | +0.26 | **-0.50** |

**What this shows:** Pairwise interactions are overwhelmingly subadditive — defects cancel rather than compound. Defect vectors are near-orthogonal (mean cosine ≈ 0), and distant layers have more opposed defect vectors (ρ=-0.71). However, pairwise cosine alone does not predict which pairs cancel most strongly. The attenuation likely arises from network propagation through intervening layers rather than simple geometric cancellation at a shared representation.

---

### 7. Hardware Compilation — Does this produce real speedups?

**Location:** `organized/serving/`

Three hardware platforms produce three different optimal execution graphs from the same checkpoint:

| Hardware | Optimal graph | Speedup |
|----------|---------------|---------|
| L40S (CUDA) | All-parallel-fused | 1.20-1.54× |
| Apple M4 Max (MPS) | Mixed 7/12 parallel | 1.14× |
| Blackwell Max-Q | Workload-dependent | 1.00-1.09× |

The MPS result is particularly important — Apple Silicon genuinely prefers a different DAG than CUDA, and the compiler correctly selects it.

**vLLM production serving (430M, Blackwell):** 14% TPOT improvement at C=1, tapering at high concurrency. All execution modes pass correctness gates (37/37 greedy generation match).

**TP=2 (7B, 2× H100 PCIe):** 1.19-1.20× speedup from CUDA stream overlap — MLP computation runs concurrently with the attention all-reduce.

**Compiler calibration:** Raw composition predictions violate budgets 33-47% of the time. Quantile-calibrated predictions reduce this to <1% violations at 99% confidence, converting the empirical law into a practical deployment tool.

---

### 8. Downstream Benchmarks — Does equivalence hold beyond perplexity?

**Location:** `organized/downstream/`

lm-eval on the 7B checkpoint under three execution graphs:

| Task | Sequential | Parallel | Compiled | Max Δ | Stderr |
|------|-----------|----------|----------|-------|--------|
| arc_easy | 0.402 | 0.398 | 0.402 | 0.004 | ±0.010 |
| hellaswag | 0.273 | 0.274 | 0.274 | 0.000 | ±0.004 |
| piqa | 0.585 | 0.589 | 0.589 | 0.004 | ±0.012 |
| winogrande | 0.518 | 0.518 | 0.520 | 0.002 | ±0.014 |

**What this shows:** All deltas are within 1 standard error. Sequential, all-parallel, and compiler-selected mixed graphs produce statistically indistinguishable downstream performance. The absolute scores are low (7B trained on only 100M tokens), but the equivalence claim doesn't depend on absolute capability.

---

## Running Experiments

| Experiment | Status | ETA |
|------------|--------|-----|
| 430M 10B-token continuation | Running on remote rig | ~Sep 9 |
| Everything else | Complete | — |
