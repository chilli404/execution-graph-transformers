# Literature Review: Execution-Graph-Equivalent Transformers

## Purpose, scope, and evidence policy

This document is a collaborator- and reviewer-facing investigation of work most relevant to a model that can execute the same learned parameters under sequential, parallel, and layerwise-mixed Attention–FFN dependency graphs.

**Evidence policy.** Statements labeled **Verified literature fact** are limited to claims made by the cited paper or its official metadata page. Statements labeled **Project interpretation** explain how that work motivates or bounds this project; they are not claims made by the cited authors. **Not covered** records the distinction on which the project's novelty depends. Bibliographic metadata and links were checked against arXiv, OpenReview, ACL Anthology, JMLR, PMLR, MLSys, ICLR, or NeurIPS pages. Repository and blog records are used only where they are the primary publication record, notably GPT-J/Mesh Transformer JAX.

**Terminology.** For a normalized residual state `x`, write the attention update as `a(x)` and the FFN update as `g(x)`. Ignoring normalization details for exposition:

- a serialized block evaluates `x + a(x) + g(x + a(x))`;
- a parallel block evaluates `x + a(x) + g(x)`;
- their local residual-map difference is `d(x) = g(x + a(x)) - g(x)`.

This identity is an algebraic comparison, not an assertion that the two architectures are generally equivalent.

---

## 1. Direct motivation: *Feed-Forward Steering in Transformer Residual Dynamics*

### Mudarisov, Burtsev, and State (2026), arXiv:2608.02071

**Verified literature fact.** The paper models attention as a non-local aggregation field and the FFN as a local steering field in residual-direction dynamics. It separates radial and tangential FFN effects, analyzes nonlinear projective equilibria, and introduces a sequential-to-parallel defect for finite Attention–FFN blocks. Its experiments cover GPT-2, Pythia, Mistral, and Llama models. The paper reports that tangential-only FFN interventions preserve much more quality than radial-only interventions and that layers with smaller measured defects tolerate approximate parallelization better than layers with larger defects. This is an arXiv v1 preprint, not a peer-reviewed proceedings paper. [arXiv](https://arxiv.org/abs/2608.02071)

**Project interpretation.** This is the closest conceptual precursor. It supplies both the mechanism-level intuition—attention moves the point seen by the FFN, while the FFN steers residual direction—and the correct local object for asking whether a serial edge can be removed. It motivates treating execution-graph changes as functional perturbations rather than merely scheduling changes.

**Not covered.** The paper studies diagnosis and approximate post-training parallelization of selected layers. It does not train one checkpoint with an explicit cross-graph consistency objective; establish simultaneous quality under all-sequential, all-parallel, and arbitrary layerwise-mixed DAGs; learn graph equivalence as a model property; or compile execution graphs under a calibrated aggregate defect budget. Its layerwise intervention evidence therefore should not be presented as prior evidence for whole-model graph equivalence.

---

## 2. Fixed parallel Attention/FFN architectures

The papers in this section establish that changing Attention–FFN dependencies can improve training or serving efficiency when the model is designed or trained for a **fixed** altered graph. That is important feasibility evidence, but fixed-graph success is not the same objective as one checkpoint remaining functionally consistent across graphs.

### GPT-J-6B and Mesh Transformer JAX (Wang; Wang and Komatsuzaki, 2021)

**Verified literature fact.** Mesh Transformer JAX is a model-parallel JAX implementation. Its source comments that the feed-forward and attention projections are combined and computed in parallel to minimize all-reduces. The GPT-J release describes placing attention and feed-forward layers in parallel to reduce communication. The primary records identify Ben Wang as the Mesh Transformer JAX author and Ben Wang and Aran Komatsuzaki as the GPT-J-6B authors; these are software/model releases rather than archival papers. [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax) · [GPT-J release note](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/)

**Project interpretation.** GPT-J is early large-scale evidence that a parallel Attention/FFN graph is practical and motivated by collective-communication costs, not only by architectural aesthetics.

**Not covered.** GPT-J is trained for its parallel graph. The release does not test the same checkpoint under the serialized graph, optimize output agreement across graphs, support mixed per-layer graph selection, or provide a defect-based graph-selection rule.

### PaLM (Chowdhery et al., arXiv 2022; JMLR 2023)

**Verified literature fact.** PaLM uses a parallel Transformer-layer formulation, citing Wang and Komatsuzaki (2021): attention and MLP read the same normalized state and their outputs are added. The paper reports roughly 15% faster training at large scale because input matrix multiplications can be fused. Its ablations found a small quality degradation at 8B and no degradation at 62B, from which the authors extrapolated quality neutrality at 540B. [JMLR](https://jmlr.org/papers/v24/22-1144.html) · [arXiv](https://arxiv.org/abs/2204.02311)

**Project interpretation.** PaLM demonstrates that a fixed parallel graph can scale to a 540B-parameter dense language model and that the graph change has a concrete systems rationale. Its scale-dependent ablation also warns against treating equivalence as automatic or scale-independent.

**Not covered.** PaLM does not claim that one trained checkpoint can be switched back to serialized execution, does not evaluate arbitrary mixed DAGs, and does not use an explicit cross-graph consistency loss or local graph-defect compiler.

### *Efficiently Scaling Transformer Inference* (Pope et al., MLSys 2023)

**Verified literature fact.** Pope et al. develop an analytical model and multidimensional partitioning strategies for generative Transformer inference on TPU v4. On PaLM 540B they report 29 ms/token at low batch size with int8 weights and 76% model-FLOPs utilization for large-batch input processing at context length 2,048; they also analyze the context-length advantage of multi-query attention. This is principally a systems and partitioning paper, not the introduction of a new Attention/FFN dependency graph. [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html) · [arXiv](https://arxiv.org/abs/2211.05102)

**Project interpretation.** The paper motivates connecting architectural DAG choices to measured serving regimes, partitioning, KV-cache behavior, latency, and hardware utilization. It is a reminder that an algebraically available parallel branch matters only if kernels and partitioning realize the overlap or fusion.

**Not covered.** It does not train graph-polymorphic weights, compare serialized and parallel outputs from one checkpoint, or select per-layer execution dependencies under a quality budget.

### PAF: *Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design* (Sonkar and Baraniuk, 2023)

**Verified literature fact.** The paper compares Parallel Attention and Feed-Forward Net Design (PAF) with Series Attention and Feed-Forward Net Design (SAF), training PAF variants of RoBERTa-large and BERT-large. It investigates two hypotheses: that the FFN helps preserve embedding isotropy and that the attention residual is small relative to the input embedding. [arXiv](https://arxiv.org/abs/2305.13297)

**Project interpretation.** PAF reinforces that the parallel block is a meaningful object for mechanistic comparison, not just an implementation trick, and suggests residual magnitude and representation geometry as useful diagnostics of graph sensitivity.

**Not covered.** PAF compares separately trained architectural variants. It does not require a shared checkpoint to agree across SAF and PAF, study decoder-only mixed DAGs, or derive and calibrate a deployment-time graph defect.

### *Simplifying Transformer Blocks* (He and Hofmann, ICLR 2024)

**Verified literature fact.** He and Hofmann combine signal-propagation analysis and experiments to remove several standard block components, including sequential sub-block structure, normalization layers, skip connections, and some value/projection parameters. They report matching standard models' per-iteration training behavior and performance while obtaining 16% higher training throughput and 15% fewer parameters in their experiments. Their simplified block is more extensive than the conventional parallel block. [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/24fd58f52ff8d0496add8da3991644e9-Abstract-Conference.html) · [OpenReview](https://openreview.net/forum?id=RtDok9eS3s)

**Project interpretation.** This work shows that block topology can be redesigned systematically and that signal propagation is a useful criterion when changing dependencies. It also supplies a strong architectural-design comparator: efficiency can come from changing the model, whereas this project asks whether one set of weights can tolerate multiple dependency graphs.

**Not covered.** Its simplified architecture is fixed and changes more than execution order. It does not preserve the same operators and parameters while switching between serialized and parallel DAGs, and it does not target cross-graph functional agreement.

### FAL: *First Attentions Last* (Kim et al., NeurIPS 2025)

**Verified literature fact.** FAL redirects the first layer's multi-head-attention output to the MLP inputs of later layers, bypassing per-block MHA–MLP connections. FAL+ additionally augments later MLP inputs with normalized first-attention information. The paper reports up to 44% lower multi-GPU training time and up to 1.18× single-GPU throughput, while reporting better perplexity than its GPT baseline in the evaluated settings. [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/ce46df2baa9d8ef0513b2f03109e6fc8-Abstract-Conference.html) · [OpenReview](https://openreview.net/forum?id=iyu4sLQZvW)

**Project interpretation.** FAL makes the communication cost of the local MHA-to-MLP edge explicit and demonstrates that rerouting that edge can expose useful parallelism. It broadens the design space beyond a binary standard-versus-parallel block.

**Not covered.** FAL trains a particular rerouted architecture; it does not allow the same checkpoint to choose the original or bypassed edge independently at deployment, enforce output consistency across those choices, or certify mixed execution graphs.

### Kraken (Prabhakar, Zhang, and Wentzlaff, NeurIPS 2024)

**Verified literature fact.** Kraken introduces a fixed degree of intra-layer model parallelism intended to overlap collective communication with computation in multi-device inference. Trained on OpenWebText, Kraken models are reported to reach similar perplexity to standard Transformers and preserve language-modeling capability on SuperGLUE. TensorRT-LLM experiments report a mean 35.6% time-to-first-token speedup across tested sizes, context lengths, and tensor-parallel degrees. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f4d1fc085b7504c140e66bb26ed8842-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2408.07802)

**Project interpretation.** Kraken is strong evidence for hardware-aware co-design: altered residual topology can hide collectives that ordinary tensor parallelism exposes. It motivates evaluating execution graphs on real multi-device serving stacks rather than relying only on FLOP counts.

**Not covered.** Kraken's innate model-parallel topology is fixed at training time. It does not seek sequential/parallel/mixed graph interchangeability for one checkpoint or use a learned consistency objective and defect-budget compiler.

### Ladder-Residual (Zhang et al., ICML 2025)

**Verified literature fact.** Ladder-Residual reroutes residual dependencies in residual networks to decouple communication from computation. The paper focuses on tensor parallelism and reports a 29% end-to-end inference speedup for a 70B Transformer sharded over eight devices. It trains 1B and 3B Ladder Transformers from scratch and reports comparable performance to dense standard-Transformer baselines; it also reports partial conversion of Llama-3.1-8B with 3B retraining tokens and minimal degradation. [PMLR](https://proceedings.mlr.press/v267/zhang25bg.html) · [OpenReview](https://openreview.net/forum?id=bJnSplWSCL)

**Project interpretation.** Ladder-Residual most directly supports the broader premise that residual dependency edges are an architecture–systems interface. It motivates treating the execution DAG as something a compiler might choose for a target interconnect.

**Not covered.** Ladder-Residual defines and trains or adapts to a specific rerouted architecture. It does not demonstrate zero-retraining switching among the standard graph, a parallel Attention/FFN graph, and arbitrary layerwise mixtures from one checkpoint.

---

## 3. Elastic and Once-for-All models

These works are the clearest precedent for the **one trained supernet, many deployment configurations** principle. Their elasticity axes, however, differ from Attention–FFN dependency rewiring.

### LayerDrop: *Reducing Transformer Depth on Demand with Structured Dropout* (Fan, Grave, and Joulin, ICLR 2020)

**Verified literature fact.** LayerDrop applies structured dropout during Transformer training. The paper reports that subnetworks of different depths can be selected at inference without fine-tuning and with limited performance impact, across machine translation, language modeling, summarization, question answering, and language-understanding experiments. [OpenReview](https://openreview.net/forum?id=SylO2yStDr) · [arXiv](https://arxiv.org/abs/1909.11556)

**Project interpretation.** LayerDrop establishes the most relevant training lesson: expose the model to structural variation during training if deployment will remove computation. It motivates sampling graph variants rather than expecting an ordinary specialist to tolerate a changed DAG.

**Not covered.** LayerDrop removes whole layers; it does not preserve both Attention and FFN while changing their within-block dependency, optimize paired output agreement, or model the local noncommutativity of the retained operators.

### Once-for-All (Cai et al., ICLR 2020)

**Verified literature fact.** Once-for-All (OFA) trains one supernet and specializes subnetworks for target devices without additional training. Progressive shrinking covers depth, width, kernel size, and input resolution; the paper evaluates image-classification deployment across heterogeneous edge hardware. [OpenReview](https://openreview.net/forum?id=HylxE1HKwS) · [arXiv](https://arxiv.org/abs/1908.09791)

**Project interpretation.** OFA supplies the deployment-level analogy: amortize training once, then select a hardware-appropriate subnetwork. The proposed execution-graph model similarly separates expensive training from deployment-specific graph selection.

**Not covered.** OFA is a vision supernet and specializes by selecting nested architectural dimensions. It does not rewire the order/dependency of two retained nonlinear operators, target token-level output consistency, or reason about operator-order defects.

### MatFormer: *Nested Transformer for Elastic Inference* (Devvrit et al., 2024)

**Verified literature fact.** MatFormer jointly trains nested FFN widths inside a Transformer, allowing smaller models and layerwise Mix'n'Match granularities to be extracted. The NeurIPS 2024 paper covers decoder and encoder models in language and vision and includes speculative decoding with nested submodels. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe066022bab2a6c6a3c57032a1623c70-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2310.07707)

**Project interpretation.** MatFormer is the nearest Transformer-specific analogy to combinatorial layerwise deployment choices. It shows that jointly optimized choices can compose into many configurations not individually optimized and that agreement among nested models can have serving value.

**Not covered.** MatFormer varies FFN capacity while retaining the block's dependency pattern. It does not switch whether an FFN sees the pre-attention or post-attention state, and it does not analyze the resulting local graph defect.

### LayerShuffle (Freiberger et al., 2024)

**Verified literature fact.** LayerShuffle trains vision Transformers with randomized execution order of stacked Attention–FFN modules. The paper reports tolerance to arbitrary layer orders with an approximately 20% accuracy reduction at the same model size, as well as graceful degradation under layer pruning. It also analyzes how layers use execution-position information. [arXiv](https://arxiv.org/abs/2407.04513)

**Project interpretation.** LayerShuffle is direct evidence that execution order can be made a training-time random variable and that ordinary networks are not robust to such changes by default. It is an important comparator against any broad claim of “graph polymorphism.”

**Not covered.** LayerShuffle permutes whole vision-Transformer modules rather than changing Attention-to-FFN edges within language-model blocks. It accepts a substantial same-size accuracy cost and does not optimize paired graph agreement, derive an operator defect, or compile graphs under a measured quality budget.

---

## 4. Dynamic and recurrent execution

### Universal Transformer (Dehghani et al., ICLR 2019)

**Verified literature fact.** The Universal Transformer repeatedly applies a shared self-attentive transition over depth and includes a dynamic per-position halting mechanism. The paper presents it as combining recurrence with parallel processing across sequence positions and reports gains on algorithmic and language tasks. [OpenReview](https://openreview.net/forum?id=HyzdRiR9Y7) · [arXiv](https://arxiv.org/abs/1807.03819)

**Project interpretation.** Universal Transformer separates logical depth from parameter count and establishes input-dependent execution depth as a learnable decision. It motivates viewing a Transformer as an executable transition system rather than a single immutable stack.

**Not covered.** Its runtime choice is how many recurrent steps to apply, not which dependency graph to use inside independently parameterized layers. It does not seek equivalence between serialized and parallel Attention/FFN evaluations.

### *Polymorphic Universal Transformer* (Chen et al., ACL 2026)

**Verified literature fact.** The paper studies ultra-deep recurrent Transformers and proposes a Polymorphic Transformer combining conditional sparse subspaces, SiLU Attention, and an uncertainty-aware depth scheduler. The ACL Anthology abstract reports improved representation rank and robustness and a 64.7% computation reduction while attaining reasoning performance comparable to the paper's baseline. [ACL Anthology](https://aclanthology.org/2026.acl-long.1809/)

**Project interpretation.** This work is terminologically close but technically orthogonal. Its “polymorphism” concerns functional diversity and sparse, dynamically scheduled recurrent depth under shared parameters. It motivates making the distinction between **dynamic compute** and **execution-graph equivalence** explicit in reviews.

**Not covered.** It does not compare standard serialized and parallel Attention/FFN residual maps, enforce their logits to agree, support a layerwise binary dependency mask over a conventional decoder stack, or compile static hardware-specific DAGs from local defects.

---

## 5. Residual dynamics and operator-splitting background

### Residual networks as dynamics: Neural ODEs (Chen et al., NeurIPS 2018)

**Verified literature fact.** Neural ODEs parameterize the derivative of a hidden state and use an ODE solver instead of a fixed discrete layer sequence. The paper explicitly connects residual updates with Euler discretization and demonstrates adaptive evaluation and continuous-depth models. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html) · [arXiv](https://arxiv.org/abs/1806.07366)

**Project interpretation.** The residual-as-discretized-flow view legitimizes asking how two vector-field-like updates compose and what changes when their discretization/order changes.

**Not covered.** Neural ODEs do not study Transformer Attention/FFN dependencies, cross-discretization agreement of one checkpoint, or graph compilation.

### Transformer splitting and Macaron Net (Lu et al., ICLR 2020)

**Verified literature fact.** Lu et al. interpret a Transformer as a numerical solver for a convection–diffusion equation in a multi-particle system. They associate the standard architecture with Lie–Trotter splitting and motivate a Strang–Marchuk-inspired FFN–attention–FFN “Macaron” block, reporting improvements on supervised and unsupervised tasks. [OpenReview](https://openreview.net/forum?id=SJl1o2NFwS) · [arXiv](https://arxiv.org/abs/1906.02762)

**Project interpretation.** This is the key operator-splitting antecedent. It establishes that Attention/FFN order is mathematically meaningful: sequential composition is not generally interchangeable with an additive/parallel update. The local quantity `g(x+a(x))−g(x)` is the finite residual-map discrepancy induced by removing the Attention-to-FFN dependency; in a small-step smooth limit it is related to directional derivatives and, for exact flows, to noncommutativity terms familiar from splitting analysis.

**Not covered.** Macaron Net chooses a more accurate fixed splitting architecture. It does not train one discrete Transformer to realize multiple splitting schemes with matched outputs or use empirical defects to choose a layerwise deployment graph.

### *A Neural ODE Interpretation of Transformer Layers* (Zhong et al., NeurIPS DLDE Workshop 2022)

**Verified literature fact.** This extended abstract interprets residual Transformer layers as numerical integration and proposes putting multi-head attention and MLP sublayers in parallel. It reports improvements on multiple evaluated tasks and an additional image-classification gain from a more sophisticated ODE solver. [arXiv](https://arxiv.org/abs/2212.06011)

**Project interpretation.** It provides a direct bridge between continuous-depth reasoning and the fixed parallel block, supporting the view that parallel addition is a different integration rule rather than a free scheduling transformation.

**Not covered.** It compares architectural/integration choices, not one checkpoint's agreement across choices, arbitrary mixed graphs, or defect-budget graph selection.

### Self-attention as interacting-particle dynamics (Geshkovski et al., NeurIPS 2023)

**Verified literature fact.** Geshkovski et al. analyze self-attention as an interacting particle system and prove clustering behavior for a simplified setting with time-independent weights; limiting behavior depends on the value-matrix spectrum. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2305.05465)

**Project interpretation.** This is background for the aggregation side of the attention-as-aggregation/FFN-as-steering picture. It helps explain why omitting the FFN from residual-dynamics analysis can miss a local directional field that changes the realized block trajectory.

**Not covered.** The analysis abstracts away the learned FFN and does not study discrete Attention/FFN graph substitutions, language-model output agreement, or deployment compilation.

---

## 6. Explicit comparison and novelty gap

### Comparison table

| Work | Verified execution variation | One checkpoint / shared weights? | Deployment-time choice | Explicit cross-configuration agreement? | Attention–FFN dependency defect? |
|---|---|---:|---|---:|---:|
| Feed-Forward Steering | Selective post-training serial-to-parallel layer intervention | Yes, pretrained models | Diagnostically selected layers | No training objective | Yes, local defect |
| GPT-J / PaLM / PAF | Fixed parallel Attention + FFN | Yes within the trained fixed graph | No alternate graph demonstrated | No | No compiler |
| Efficiently Scaling Transformer Inference | Systems partitioning and kernels for PaLM inference | Yes | Hardware partition/layout | No | No |
| Simplifying Transformer Blocks | Fixed simplified block | Yes within that architecture | No | No | No |
| FAL | Fixed rerouting of MHA-to-MLP dependencies | Yes within FAL/FAL+ | No original-graph fallback shown | No | No |
| Kraken | Fixed innate intra-layer model parallelism | Yes within Kraken | Fixed degree/topology | No | No |
| Ladder-Residual | Fixed rerouted residual dependencies | From-scratch or adapted model | No zero-retraining graph switching shown | No | No |
| LayerDrop | Drop whole Transformer layers | Yes | Select depth | Not paired logit equivalence | No |
| Once-for-All | Select depth/width/kernel/resolution subnetworks | Yes | Hardware-specialized subnetwork | Accuracy-oriented supernet training | No |
| MatFormer | Select nested FFN width per model/layer | Yes | Width and Mix'n'Match granularity | Joint nested losses, not graph agreement | No |
| LayerShuffle | Permute or prune whole ViT modules | Yes | Arbitrary trained-for order | Robustness, with reported accuracy cost | No within-block defect |
| Universal Transformer | Recurrent depth / per-position halting | Shared transition | Input-dependent steps | No graph-pair agreement | No |
| Polymorphic Universal Transformer | Sparse recurrent subspaces and dynamic depth | Shared recurrent framework | Input-dependent scheduler | No serial/parallel agreement | No |
| Neural ODE / Macaron / Transformer ODE | Change integration/splitting architecture | Usually separately trained choice | Solver choice in Neural ODE; fixed Transformer variants | No | Splitting theory, not a deployment compiler |
| **This project** | **Per-layer serialized vs parallel Attention/FFN edge, including mixed DAGs** | **One checkpoint** | **Static hardware/budget-specific graph** | **Explicit centered-logit consistency objective** | **Exact local residual-map defect and aggregate graph budget** |

### Novelty gap

**Verified synthesis of the literature.** Existing work covers four neighboring capabilities:

1. **Fixed efficient graphs:** GPT-J, PaLM, PAF, FAL, Kraken, and Ladder-Residual show that models can be designed and trained around altered dependencies for fusion, overlap, or reduced communication.
2. **Elastic subnetworks:** LayerDrop, Once-for-All, and MatFormer show that one training run can support many depth/width/capacity choices.
3. **Order or dynamic-depth robustness:** LayerShuffle and recurrent/dynamic Transformers vary whole-module order or iteration count.
4. **Mechanism and mathematics:** residual-dynamics and operator-splitting work explains why operator order matters, while Feed-Forward Steering supplies a directly measurable serial-to-parallel defect.

**Project interpretation and proposed novelty.** The unfilled intersection is a conventional decoder-style Transformer trained so that **the same parameters implement approximately the same token-prediction function under multiple Attention–FFN dependency DAGs**, including arbitrary layerwise mixtures, with:

- both major operators retained rather than pruned or width-sliced;
- the dependency edge, not only capacity or depth, chosen at deployment;
- an explicit output-consistency objective, rather than average task loss alone;
- a local finite-map defect connected to graph-induced output divergence; and
- a budgeted compiler that selects a hardware-appropriate graph without retraining.

This novelty statement must remain scoped. The project should **not** claim to invent parallel Attention/FFN blocks, elastic inference, dynamic execution, residual-as-ODE analysis, operator splitting, or defect-guided approximate layer parallelization. The narrower claim is the combination of cross-DAG training, mixed-DAG generalization, functional agreement, and defect-budget deployment for one checkpoint.

### Reviewer-facing falsification criteria

The proposed gap would be weakened or closed by prior work that demonstrates all or most of the following in one method: (i) unchanged Attention and FFN weights, (ii) zero-retraining switching between serialized and parallel blocks, (iii) layerwise mixed graph selection, (iv) explicit output-distribution agreement across graphs, and (v) a validated graph-level quality predictor or budget. None of the works verified above establishes that combination.

For this project's own claims, average language-model quality is insufficient. A convincing evaluation must separately show specialist graph sensitivity, cross-graph logit/distribution agreement, mixed-DAG behavior, the necessity of the consistency term, defect-to-divergence prediction, and realized serving benefit on the intended hardware/runtime.

---

## References

1. Timur Mudarisov, Mikhail Burtsev, and Radu State. “Feed-Forward Steering in Transformer Residual Dynamics.” arXiv:2608.02071, 2026. https://arxiv.org/abs/2608.02071
2. Ben Wang. “Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX.” Software release, 2021. https://github.com/kingoflolz/mesh-transformer-jax
3. Ben Wang and Aran Komatsuzaki. “GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model.” Model release, 2021. https://github.com/kingoflolz/mesh-transformer-jax
4. Aakanksha Chowdhery et al. “PaLM: Scaling Language Modeling with Pathways.” *Journal of Machine Learning Research* 24(240):1–113, 2023; arXiv first posted 2022. https://jmlr.org/papers/v24/22-1144.html
5. Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. “Efficiently Scaling Transformer Inference.” *Proceedings of Machine Learning and Systems* 5, 2023. https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html
6. Shashank Sonkar and Richard G. Baraniuk. “Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design.” arXiv:2305.13297, 2023. https://arxiv.org/abs/2305.13297
7. Bobby He and Thomas Hofmann. “Simplifying Transformer Blocks.” *International Conference on Learning Representations*, 2024. https://openreview.net/forum?id=RtDok9eS3s
8. Gyudong Kim, Hyukju Na, Jin Kyu Kim, Hyunsung Jang, Jaemin Park, Jaegi Hwang, Namkoo Ha, Seungryong Kim, and Young Geun Kim. “First Attentions Last: Better Exploiting First Attentions for Efficient Parallel Training.” *Advances in Neural Information Processing Systems* 38, 2025. https://papers.nips.cc/paper_files/paper/2025/hash/ce46df2baa9d8ef0513b2f03109e6fc8-Abstract-Conference.html
9. Rohan Baskar Prabhakar, Hengrui Zhang, and David Wentzlaff. “Kraken: Inherently Parallel Transformers For Efficient Multi-Device Inference.” *Advances in Neural Information Processing Systems* 37, 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f4d1fc085b7504c140e66bb26ed8842-Abstract-Conference.html
10. Muru Zhang, Mayank Mishra, Zhongzhu Zhou, William Brandon, Jue Wang, Yoon Kim, Jonathan Ragan-Kelley, Shuaiwen Leon Song, Ben Athiwaratkun, and Tri Dao. “Ladder-Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference with Communication Overlapping.” *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267:75779–75792, 2025. https://proceedings.mlr.press/v267/zhang25bg.html
11. Angela Fan, Edouard Grave, and Armand Joulin. “Reducing Transformer Depth on Demand with Structured Dropout.” *International Conference on Learning Representations*, 2020. https://openreview.net/forum?id=SylO2yStDr
12. Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. “Once-for-All: Train One Network and Specialize it for Efficient Deployment.” *International Conference on Learning Representations*, 2020. https://openreview.net/forum?id=HylxE1HKwS
13. Devvrit, Sneha Kudugunta, Aditya Kusupati, Tim Dettmers, Kaifeng Chen, Inderjit Dhillon, Yulia Tsvetkov, Hannaneh Hajishirzi, Sham Kakade, Ali Farhadi, and Prateek Jain. “MatFormer: Nested Transformer for Elastic Inference.” *Advances in Neural Information Processing Systems* 37, 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe066022bab2a6c6a3c57032a1623c70-Abstract-Conference.html
14. Matthias Freiberger, Peter Kun, Anders Sundnes Løvlie, and Sebastian Risi. “LayerShuffle: Enhancing Robustness in Vision Transformers by Randomizing Layer Execution Order.” arXiv:2407.04513, 2024. https://arxiv.org/abs/2407.04513
15. Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. “Universal Transformers.” *International Conference on Learning Representations*, 2019. https://openreview.net/forum?id=HyzdRiR9Y7
16. Yilong Chen, Zitian Gao, Yihao Xiao, Jason Klein Liu, Xinyu Yang, Yifan Luo, Haoming Luo, Zhengmao Ye, Tingwen Liu, Ran Tao, and Bryan Dai. “Polymorphic Universal Transformer.” *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 39001–39013, 2026. https://aclanthology.org/2026.acl-long.1809/
17. Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K. Duvenaud. “Neural Ordinary Differential Equations.” *Advances in Neural Information Processing Systems* 31, 2018. https://proceedings.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html
18. Yiping Lu, Zhuohan Li, Di He, Zhiqing Sun, Bin Dong, Tao Qin, Liwei Wang, and Tie-Yan Liu. “Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View.” *International Conference on Learning Representations*, 2020. https://openreview.net/forum?id=SJl1o2NFwS
19. Yaofeng Desmond Zhong, Tongtao Zhang, Amit Chakraborty, and Biswadip Dey. “A Neural ODE Interpretation of Transformer Layers.” NeurIPS DLDE Workshop, 2022. https://arxiv.org/abs/2212.06011
20. Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. “The Emergence of Clusters in Self-Attention Dynamics.” *Advances in Neural Information Processing Systems* 36, 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html
