# One Set of Weights, Many Computational Routes: A Plain-Language Guide to the Paper

## The One-Minute Version

When today's Transformer models are trained, they learn not only *what* to compute, but also come to depend on *the order in which the computation is performed*.

A standard Transformer block usually runs Attention first, then passes the result of Attention to the feed-forward network (FFN):

```text
Input → Attention → FFN → Output
```

To make better use of hardware, we might instead let Attention and the FFN start at the same time from the same input:

```text
                 ┌→ Attention ─┐
Input ────────────┤              ├→ Output
                 └→ FFN ───────┘
```

The model's parameters have not changed, and neither have its main operators. What has changed is the dependency structure of the computation.

Our central finding is that ordinary models depend strongly on the computational route used during training. Directly switching a trained sequential model to a parallel graph can seriously degrade its quality.

We introduce **graph-consistency training**, which allows the same checkpoint to retain nearly the same capabilities under sequential, parallel, and mixed computational routes. We then measure how much changing each layer to parallel execution affects the model, and use those measurements to build a **graph compiler**: given a budget for acceptable output change, it automatically selects the layers that can safely be parallelized.

The ultimate goal is:

> Train a model once, then automatically compile it at deployment time into different computational DAGs for different hardware and latency requirements—without retraining a separate model for every platform.

---

# 1. Why Study This Problem?

## 1.1 Large models are constrained not only by parameter count, but also by computation order

Many inference optimizations for large models focus on:

- quantizing parameters;
- removing layers;
- reducing Attention computation;
- using smaller submodels;
- distributing the model across multiple GPUs.

But another issue is often overlooked: a computational graph contains many dependencies that force one operation to wait for another.

For example, in a standard Transformer block:

1. Attention must finish first;
2. the FFN reads the hidden state updated by Attention;
3. only then can the FFN begin.

Even if the GPU still has idle resources, the FFN cannot run early.

If Attention and the FFN could run in parallel—or if some layers could run in parallel while others remained sequential—the critical path of inference could potentially be shortened.

## 1.2 Why not simply modify an existing model?

At first glance, one might assume that:

- Attention and the FFN both have residual connections;
- the parallel graph is only a small approximation;
- changing the graph directly should not do much harm.

Our experiments show that this intuition is wrong.

Once training is complete, the weights have adapted to a particular set of computational dependencies. Even if the parameters and operators stay the same, changing only the execution order can cause a clear loss in quality.

It is like a team that has worked together for years using a fixed workflow:

```text
Engineer A finishes the design → Engineer B builds on that design
```

If the two engineers are suddenly asked to work entirely in parallel, the people and tasks remain unchanged—but B can no longer use A's intermediate result, so the final output may differ.

---

# 2. What Exactly Is the Difference Between Sequential and Parallel Execution?

Let Attention be denoted by A and the FFN by F.

## Sequential block

```text
x → Attention → produces x + A(x)
              → FFN reads the updated state
              → final output
```

The FFN sees the representation after it has been updated by Attention.

## Parallel block

```text
Attention and the FFN both read the same original x
Their two results are added together at the end
```

The FFN cannot see the new information produced by Attention in that layer.

The entire difference between the two comes from:

> The FFN's response to the information newly added by Attention.

We call this difference the **execution-graph defect**.

---

# 3. What Did We Discover First?

## 3.1 Ordinary models become “bound” to the execution graph used during training

For models with about 120M parameters:

| Training setup | Evaluated with sequential graph | Evaluated with parallel graph |
|---|---:|---:|
| Sequential specialist | 1.193 | 1.649 |
| Parallel specialist | 1.369 | 1.201 |

Lower BPB is better.

The pattern is clear:

- the sequential model becomes substantially worse when run with the parallel graph;
- the parallel model also becomes worse when switched back to the sequential graph.

This shows that model weights learn not only language, but also the computational DAG used during training—and come to depend on it.

We call this phenomenon:

> **Execution-DAG specialization.**

## 3.2 Merely exposing the model to both graphs is not enough

A natural baseline is to train with:

- half the loss coming from sequential execution;
- half the loss coming from parallel execution;
- no requirement that the two graphs produce the same answer.

This model achieves good average BPB under both graphs, but their outputs still differ:

| Method | Seq BPB | Par BPB | Agreement between graphs | KL difference |
|---|---:|---:|---:|---:|
| Joint CE, no consistency | 1.196 | 1.199 | 81.3% | 31.5 |
| Graph consistency | 1.195 | 1.196 | 90.5% | 4.82 |

Therefore:

> “Both graphs are capable” does not mean “both graphs implement the same function.”

---

# 4. What Is Our Method?

During training, we run both:

- the sequential graph;
- the parallel graph.

The model must not only predict the correct token under both graphs, but also keep their **centered logits** close to one another.

Why centered logits?

Adding the same constant to every vocabulary logit does not change the softmax probabilities. Subtracting the mean ensures that the loss constrains only the part that can actually change the output distribution.

In plain language, the objective is:

> It is not enough for both routes to reach the destination; they should arrive at almost exactly the same place.

---

# 5. The Most Important Quality Results

## 5.1 ClimbMix

| Model | Sequential BPB | Parallel BPB | Agreement between graphs |
|---|---:|---:|---:|
| Sequential specialist | 1.193 | 1.649 | N/A |
| Parallel specialist | 1.369 | 1.201 | N/A |
| Polymorphic model | 1.195 | 1.196 | 90.5% |

For the polymorphic model:

- sequential quality is almost identical to that of the sequential specialist;
- parallel quality is slightly better than that of the parallel specialist;
- the two graphs make the same top-1 prediction for most tokens.

## 5.2 TinyStories

| Model | Sequential BPB | Parallel BPB | Agreement between graphs |
|---|---:|---:|---:|
| Sequential specialist | 0.537 | 0.725 | N/A |
| Parallel specialist | 0.601 | 0.541 | N/A |
| Polymorphic model | 0.536 | 0.537 | 94.8% |

On the second corpus, both execution modes of the polymorphic model slightly outperform their corresponding specialists.

This suggests that the result is not an accident specific to a single dataset.

---

# 6. How Many Computational Graphs Can the Model Support?

In a 12-layer model, each layer can be either sequential or parallel, giving a theoretical total of:

```text
2^12 = 4096 execution graphs
```

We evaluated dozens of representative mixed graphs, including:

- fully sequential;
- fully parallel;
- first half parallel;
- second half parallel;
- alternating odd and even layers;
- layers selected by defect;
- fixed random masks;
- different combinations containing 0 through 12 parallel layers.

The results show that:

- on ClimbMix, the maximum BPB loss across all sampled graphs is less than 0.002;
- on TinyStories, the maximum loss is about 0.0013;
- most mixed graphs achieve top-1 agreement above 95%.

This indicates that the model has not merely memorized the two endpoints. It has learned to generalize compositionally to graphs in between.

---

# 7. Why Can We Predict Whether a Graph Is Safe?

## 7.1 Every layer has its own defect

In some layers, the FFN depends heavily on the new output produced by Attention in the same layer. Switching such a layer to parallel execution has a larger effect.

In other layers, the FFN is less sensitive to that Attention update. Those layers are better candidates for parallelization.

## 7.2 Mathematical explanation

The local defect can be written as the integral of the FFN Jacobian along the Attention update.

In plain language:

> Attention nudges the hidden state in a particular direction; the defect measures how sensitive the FFN is to that nudge.

A first-order approximation can be written as:

```text
FFN Jacobian × Attention update
```

Experiments show that:

- this first-order approximation captures the direction of the defect fairly well;
- it becomes more accurate in later layers;
- earlier layers contain more higher-order nonlinear interactions.

## 7.3 The effects of multi-layer graphs are approximately additive

If we know the effect of rewriting each individual layer, we can predict the effect of rewriting several layers together.

Across the two corpora:

- Pearson correlation for KL prediction is about 0.96–0.97;
- rank correlation is about 0.93–0.95;
- relative prediction error is about 11%;
- the multi-layer effect is about 70% of the sum of the single-layer effects.

In other words:

> The effects of multiple layers are highly compositional, but slightly sublinear overall.

---

# 8. What Is the Graph Compiler?

The graph compiler takes as input:

- the defect cost of each layer;
- the user's budget for acceptable output change;
- the parallelization benefit offered by each layer.

It then chooses how to:

```text
Parallelize as many layers as possible without exceeding the output-change budget
```

A 12-layer model has only 4096 DAGs, so we can enumerate them exactly rather than relying on approximate search.

In aggregate graph-level tests:

- held-out prediction correlation is about 0.94–0.97;
- calibration on ClimbMix still achieves about 0.97 after transfer to TinyStories;
- none of the tested budgets are violated;
- compared with an oracle, the compiler parallelizes at most one fewer layer.

## At the prompt level

When selecting a graph dynamically for each individual prompt, the compiler usually outperforms a random mask with the same number of parallel layers:

- win rate on ClimbMix is about 76%–92%;
- cross-corpus win rate on TinyStories is about 80%–100%.

However, prompt-level budget coverage on ClimbMix is only about 77%–89%, falling short of the 95% target.

The accurate description of the current system is therefore:

> A **prompt-level defect-guided ranker**

rather than:

> A certificate with a strict guarantee for every prompt.

---

# 9. What Do the Speed Results Mean?

## 9.1 Standard eager PyTorch

Fused parallel execution provides approximately:

- **1.20×–1.29× full-forward speedup.**

## 9.2 `torch.compile`

With fixed shapes:

- batch 1: about **1.45×**;
- batch 4: about **1.54×**.

This shows that compilation does not automatically eliminate the advantage of the fused parallel graph.

## 9.3 Real KV-cache decoding

Models trained on both corpora achieve:

- average TTFT improvement of about **1.19×–1.22×**;
- average TPOT improvement of about **1.24×–1.26×**.

TPOT is the more stable and more persuasive systems result.

---

# 10. Why Can't the Random-Mask Baseline Replace Our Method?

Random-mask training executes a randomly chosen graph for each batch, in a manner similar to LayerDrop.

It can give many graphs good average BPB, but it cannot make them implement the same function.

| Method | Agreement between graphs | KL |
|---|---:|---:|
| Random mask, 1× compute | 78.0% | 31.7 |
| Random mask, 2× compute | 71.8% | 53.7 |
| Joint CE | 81.3% | 31.5 |
| Graph consistency | 90.5% | 4.82 |

Notably, doubling the training compute for random-mask training lowers BPB, yet makes agreement between graphs even worse.

This demonstrates that:

> Stronger language-modeling ability does not automatically produce execution-graph equivalence.

The consistency constraint addresses a different problem.

---

# 11. What Are the Main Novel Contributions?

## 11.1 A new dimension of model elasticity

Existing elastic models typically vary:

- depth;
- width;
- number of tokens;
- number of experts;
- subnetwork size.

What we vary is:

> The structure of data dependencies between operators.

## 11.2 Train once, compile many DAGs

At deployment time, a single checkpoint can be compiled into a:

- sequential graph;
- parallel graph;
- mixed graph;
- defect-selected graph;
- hardware-specific graph.

## 11.3 A complete loop connecting theory, method, and system

This is not merely an isolated speed trick. The work:

1. identifies graph specialization;
2. trains for graph equivalence;
3. explains the defect mathematically;
4. predicts multi-layer rewrites;
5. compiles the graph;
6. demonstrates speed gains in real KV-cache inference.

---

# 12. Potential Impact

## For model training

Future models may no longer need to be tied to a single execution graph. Instead, they could be trained as a family of approximately equivalent DAGs.

## For deployment

A single checkpoint could use a different DAG depending on:

- GPU type;
- batch size;
- latency SLA;
- memory budget;
- online or offline workload.

## For inference systems

The graph compiler could form a layer between the model and systems such as vLLM or TensorRT:

```text
Checkpoint
    ↓
Defect profiler
    ↓
Graph compiler
    ↓
Hardware-specific execution DAG
    ↓
vLLM / TensorRT / torch.compile
```

## For research

This work raises a new question:

> To what extent do neural-network weights depend on the computational graph itself, rather than only on the parameters and operators?

The same question could be extended to:

- layer reordering;
- communication–computation overlap;
- pipeline execution;
- MoE expert scheduling;
- multimodal branch parallelization;
- speculative execution.

---

# 13. What Has Not Yet Been Achieved?

It is important to be candid about the current limitations:

1. a formal vLLM plugin has not yet been completed;
2. the 95% prompt-level coverage certificate has not yet been achieved;
3. retrofitting existing models can quickly recover parallel quality, but cannot quickly restore high agreement;
4. the current experiments focus mainly on models with about 120M parameters;
5. the main positive experimental results come primarily from a single training seed after the method was finalized;
6. evaluation on standard capability benchmarks is currently centered mainly on LAMBADA;
7. the approach has not yet been validated in a production-grade continuous-batching environment.

These are directions for future work and should not be hidden.

---

# 14. The Clearest Way to Explain the Work

For a non-specialist, the story can be told this way:

> In the past, a model's weights could safely run only through the fixed computational workflow used during training. We found that changing only the order of computation can make a model substantially worse. We introduce a training method that lets the same set of weights adapt to several computational workflows. We then measure the “risk of changing the workflow” for each layer and automatically choose a computational route suited to the current hardware and latency requirements. The results show that meaningful inference speedups are possible with almost no loss in model quality.

The most important point is not simply “how much faster” the model becomes, but that:

> The model changes from one fixed program into a family of programs that can be recompiled.

---

# 15. Impact and Novelty in One Sentence Each

## Motivation

A fixed execution DAG limits a Transformer's ability to adapt to different hardware and workloads, while ordinary weights cannot safely switch between DAGs.

## Result

Graph-consistency training enables one checkpoint to maintain nearly the same quality across sequential, parallel, and mixed DAGs, while a defect-based compiler selects low-risk graphs.

## Impact

This could enable “train once, deploy with many graphs,” reducing the need to retrain models for different hardware platforms.

## Novelty

This work is the first to make the execution dependency graph a central object of Transformer elasticity and compilation, unifying training, a mathematical theory of defects, a graph compiler, and real serving speedups.
