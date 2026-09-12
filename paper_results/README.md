# Paper Results

All results for the ICLR 2027 submission, organized by paper section.

## core_scaling/
Primary architecture (QK-norm, ReLU², value embeddings) at 120M–7B.
Graph-rewrite JSONs contain per-mask BPB, agreement, symmetric KL, layer defects.
- 430M cw=1.0 and cw=0.1
- 1B cw=1.0 and cw=0.1
- 3B cw=0.1 (fixed) and gradnorm (adaptive)
- 7B cw=0.1 (12k steps)
- 430M sequential specialist (superadditive control, η≈1.23)

## llama/
Second architecture (RMSNorm, SwiGLU, no value embeddings, no softcap).
- 430M poly + seq specialist + composition holdout + calibration
- 1B poly + seq specialist + composition holdout + calibration

## composition/
Composition law validation.
- 430M 10k-mask holdout (5-fold CV Pearson 0.971, β=0.218±0.001)
- Cross-sequence-length transfer (fit 128 → test 1024, Pearson 0.969)
- Compiler calibration (raw 36% violations → 99% calibrated 0.6%)
- Training progress (β stabilizes by step 4000)

## compiler/
Hardware latency profiles for graph compilation.
- Per-layer profiling on L40S and Blackwell
- Additive latency model: R²=0.09 (honest limitation)

## serving/
Production serving benchmarks.
- vLLM 430M continuous batching (C=1 to C=200)
- vLLM 7B single-GPU
- TP=2 raw PyTorch 7B on 2× H100 (1.19-1.20× with stream overlap)

## downstream/
lm-eval benchmark results (HellaSwag, PIQA, WinoGrande, ARC-Easy).
- 7B: seq vs par vs compiled (all within stderr)
- 430M 3.5B tokens: seq vs par (all within 1-2 stderr)
- 430M 9B tokens: seq vs par (PIQA 70%, HellaSwag 42%)

## loss_ablation/
Consistency objective comparison.
- 430M + 1B Pareto: centered MSE vs raw MSE vs symmetric KL
- Equivalence evals on key Pareto checkpoints
- Key finding: MSE robust default, KL matches at λ=0.003 (100× lower)

## gradient_analysis/
Gradient ratio measurements (||∇L_con||/||∇L_LM||) at 120M–7B.
- MSE: 0.02-0.14× (scale-invariant)
- KL: 10-58× (explains tuning asymmetry)

## mechanism/
Pairwise defect analysis and layer-skip robustness.
- 430M: 190/190 pairs subadditive, mean cos=0.011
- 7B: 481/496 pairs subadditive, mean cos=0.042
- Layer skip: poly wins 16/20 vs specialist (39% less degradation)

## ternary/
Ternary execution mask ({seq, par, skip}) — second graph rewrite type.
- 120M: 5000 random 3^12 masks, mean ΔBPB=0.040 (vs 0.434 binary-only)
- 90% of ternary masks under 0.10 ΔBPB
- 430M ternary: running (results pending)

## generation/
Autoregressive generation divergence test.
- 430M 100M tokens: 19.5% identical, median divergence token 13
- 430M 9B tokens: 7.0% identical, median divergence token 7
- Limitation: distributional equivalence ≠ rollout equivalence

## long_training/
430M trained from 0.4B to 9B tokens — β trajectory over training.
- ΔBPB decreases monotonically (0.0014 → 0.0002)
- η decreases (0.42 → 0.38)
- β increases (0.48 → 0.53)
- Equivalence survives Chinchilla-scale training (~21 tokens/param)
