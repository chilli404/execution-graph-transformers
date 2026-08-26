# Ready-to-send collaborator message

Hi — I’ve been working on a project about **execution-graph equivalent Transformers**. The core observation is that ordinary Transformer weights strongly specialize to whether each block runs Attention→FFN sequentially or Attention‖FFN in parallel. Changing only that dependency graph can degrade BPB by roughly 11–38%, even though the parameters and major operators are unchanged.

We trained 117.5M models with a graph-consistency objective so one checkpoint can run sequential, parallel, or mixed layer-level DAGs. On ClimbMix and TinyStories, the polymorphic model matches specialist BPB and reaches 90.5–94.8% cross-graph top-1 agreement. More than 50 mixed DAGs per corpus remain within 0.002 BPB of the reference.

There is also a theory/compiler component. A layer’s sequential-to-parallel defect is the FFN response to the current Attention update and has an exact Jacobian integral form. Single-layer effects predict multi-layer graph KL with Pearson around 0.96–0.97 and about 11% relative RMSE. We use those costs in a graph compiler that selects the maximum number of parallel layers under a divergence budget.

The system results are promising: fused parallel execution improves real KV-cache TPOT by about 24–26%, and fixed-shape `torch.compile` speedups reach 1.45–1.54x. Critical baselines show that joint CE or random graph augmentation can preserve BPB but do not learn functional graph equivalence; consistency reduces KL by more than 6x.

The project now needs a stronger deployment and scale phase. The highest-value work for the RTX PRO 6000 Blackwell is:

1. Build the native vLLM out-of-tree plugin with PagedAttention and the fused QKV+MLP-up path.
2. Re-run serving benchmarks under continuous batching, including p50/p95 TTFT, TPOT, throughput, and memory.
3. Train a 350–500M sequential baseline and graph-consistent model to validate scaling beyond 120M.
4. If the 500M run is clean, run a short 1B scaling pilot rather than a large sweep.
5. Help turn the existing theorem, compiler, figures, and ablations into a full paper.

The repository is organized so you can begin with `README.md`, then `docs/COLLABORATOR_BRIEF.md`, and finally `blackwell/README.md`. The current claim boundaries are explicit: prompt-level defects are useful graph rankers but not yet reliable 95% per-prompt certificates, and vLLM results have not yet been produced.
