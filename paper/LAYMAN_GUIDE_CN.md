# 一套权重，多种计算路线：论文通俗指南

## 一分钟版本

今天的 Transformer 模型在训练时，不仅学会了“做什么”，还会悄悄依赖“计算必须按什么顺序做”。

一个标准 Transformer block 通常先运行 Attention，再把 Attention 的结果交给 FFN：

```text
输入 → Attention → FFN → 输出
```

如果为了提高硬件速度，我们让 Attention 和 FFN 同时从同一个输入开始计算：

```text
                 ┌→ Attention ─┐
输入 ────────────┤              ├→ 输出
                 └→ FFN ───────┘
```

模型使用的参数没有改变，主要算子也没有改变，但计算依赖关系变了。

我们的核心发现是：普通模型会强烈依赖训练时的计算路线。直接把训练好的 sequential 模型改成 parallel，模型质量可能严重下降。

我们提出一种 graph-consistency training，让同一个 checkpoint 在 sequential、parallel 和混合计算路线下都能保持接近相同的能力。然后，我们测量每一层“改成 parallel 会造成多大影响”，并用这些影响构建一个 graph compiler：给定允许的输出变化预算，自动选择可以安全并行的层。

最终目标是：

> 模型只训练一次，部署时根据硬件和延迟要求，自动编译成不同的计算 DAG，而不必为每种硬件重新训练一份模型。

---

# 1. 为什么要研究这个问题？

## 1.1 大模型不仅受参数量限制，也受计算顺序限制

很多大模型推理优化集中在：

- 量化参数；
- 删除层；
- 减少 Attention 计算；
- 使用更小的子模型；
- 把模型拆到多张 GPU。

但还有一个经常被忽略的问题：计算图中存在很多必须等待前一步完成的依赖。

例如，在标准 Transformer block 中：

1. Attention 必须先完成；
2. FFN 读取 Attention 更新后的 hidden state；
3. FFN 才能开始。

即使 GPU 还有空闲资源，FFN 也不能提前执行。

如果 Attention 和 FFN 可以并行，或者某些层可以并行、某些层保持 sequential，就可能降低推理关键路径。

## 1.2 为什么不能直接修改已有模型？

直觉上，人们可能认为：

- Attention 和 FFN 都有 residual connection；
- parallel graph 只是一个小近似；
- 直接修改应该不会太严重。

实验发现这个直觉是错的。

模型训练完成后，权重已经适应了特定的计算依赖关系。即使参数和算子都不变，只修改执行顺序，也可能造成明显质量损失。

这类似于一支团队长期按照固定工作流程合作：

```text
工程师 A 先完成设计 → 工程师 B 根据设计继续工作
```

如果突然要求两个人完全并行工作，虽然人员和任务都没有变化，但 B 无法再利用 A 的中间结果，最终输出就可能不同。

---

# 2. Sequential 和 Parallel 到底有什么区别？

将 Attention 记为 A，将 FFN 记为 F。

## Sequential block

```text
x → Attention → 得到 x + A(x)
              → FFN 读取更新后的状态
              → 最终输出
```

FFN 看到的是 Attention 更新后的表示。

## Parallel block

```text
Attention 和 FFN 都读取相同的原始 x
最后把两个结果加起来
```

FFN 看不到本层 Attention 刚产生的新信息。

两者的差异，完全来自：

> FFN 对 Attention 新增信息的响应。

我们把这个差异称为 execution-graph defect。

---

# 3. 我们首先发现了什么？

## 3.1 普通模型会“绑定”训练时的 execution graph

在约 120M 参数模型上：

| 训练方式 | 用 sequential graph 评估 | 用 parallel graph 评估 |
|---|---:|---:|
| Sequential specialist | 1.193 | 1.649 |
| Parallel specialist | 1.369 | 1.201 |

BPB 越低越好。

可以看到：

- sequential 模型使用 parallel graph 后明显变差；
- parallel 模型切回 sequential graph 后也会变差。

这说明模型权重不仅学习语言，还学习并依赖训练时的计算 DAG。

我们把这个现象称为：

> Execution-DAG specialization。

## 3.2 只让模型见过两种 graph，还不够

一个自然 baseline 是：

- 一半 loss 来自 sequential；
- 一半 loss 来自 parallel；
- 不要求两个 graph 给出相同答案。

这个模型在两种 graph 下的平均 BPB 都不错，但两个 graph 的输出仍然不同：

| 方法 | Seq BPB | Par BPB | 两图答案一致率 | KL 差异 |
|---|---:|---:|---:|---:|
| Joint CE，没有 consistency | 1.196 | 1.199 | 81.3% | 31.5 |
| Graph consistency | 1.195 | 1.196 | 90.5% | 4.82 |

因此：

> “两种 graph 都有能力”不等于“两种 graph 实现了相同函数”。

---

# 4. 我们的方法是什么？

训练时同时运行：

- sequential graph；
- parallel graph。

模型不仅要在两种 graph 下预测正确 token，还要让两个 graph 的 centered logits 接近。

为什么使用 centered logits？

因为给所有 vocabulary logits 同时加一个常数，并不会改变 softmax 概率。减去平均值后，loss 只约束真正会改变输出分布的部分。

方法目标可以通俗理解为：

> 不只是让两条路线都能到达目的地，还要求两条路线最终到达几乎相同的位置。

---

# 5. 最重要的质量结果

## 5.1 ClimbMix

| 模型 | Sequential BPB | Parallel BPB | 两图答案一致率 |
|---|---:|---:|---:|
| Sequential specialist | 1.193 | 1.649 | 不适用 |
| Parallel specialist | 1.369 | 1.201 | 不适用 |
| Polymorphic model | 1.195 | 1.196 | 90.5% |

Polymorphic model：

- sequential 质量几乎等于 sequential specialist；
- parallel 质量略优于 parallel specialist；
- 两种 graph 大部分 token 给出相同 top-1 预测。

## 5.2 TinyStories

| 模型 | Sequential BPB | Parallel BPB | 两图答案一致率 |
|---|---:|---:|---:|
| Sequential specialist | 0.537 | 0.725 | 不适用 |
| Parallel specialist | 0.601 | 0.541 | 不适用 |
| Polymorphic model | 0.536 | 0.537 | 94.8% |

在第二个语料上，polymorphic model 的两种执行方式都略优于对应 specialists。

因此结果不是单一数据集上的偶然现象。

---

# 6. 模型能支持多少种计算图？

12 层模型中，每层都可以选择 sequential 或 parallel，因此理论上共有：

```text
2^12 = 4096 种 execution graphs
```

我们评估了几十种代表性 mixed graphs，包括：

- 全 sequential；
- 全 parallel；
- 前半 parallel；
- 后半 parallel；
- 奇偶层交替；
- 按 defect 选择；
- 固定随机 masks；
- 0 到 12 个 parallel layers 的不同组合。

结果显示：

- ClimbMix 上所有 sampled graphs 的最大 BPB 损失小于 0.002；
- TinyStories 上最大损失约 0.0013；
- 大部分 mixed graphs 的 top-1 agreement 超过 95%。

这说明模型不仅记住了两个 endpoint，而是对中间 graph 产生了组合泛化。

---

# 7. 为什么可以预测一个 graph 是否安全？

## 7.1 每层都有自己的 defect

有些层的 FFN 非常依赖本层 Attention 的新输出；把这种层改成 parallel，影响较大。

另一些层的 FFN 对本层 Attention 更新不太敏感；这些层更适合 parallelize。

## 7.2 数学解释

局部 defect 可以写成 FFN Jacobian 沿 Attention update 的积分。

通俗地说：

> Attention 把 hidden state 推动了一小步；defect 衡量 FFN 对这一步有多敏感。

一阶近似可以写成：

```text
FFN Jacobian × Attention update
```

实验发现：

- 这个一阶近似能较好捕获 defect 的方向；
- 越靠后的层，一阶近似越准确；
- 前层包含更多高阶非线性交互。

## 7.3 多层 graph 的影响近似可加

如果已知每个 single-layer rewrite 的影响，就能预测多个层一起 rewrite 的影响。

在两个语料上：

- KL prediction Pearson 约 0.96–0.97；
- 排序相关约 0.93–0.95；
- 相对预测误差约 11%；
- multi-layer effect 大约是 single-layer effects 之和的 70%。

即：

> 多层影响高度可组合，但整体略微次线性。

---

# 8. Graph compiler 是什么？

Graph compiler 接收：

- 每层 defect cost；
- 用户允许的输出变化 budget；
- 每层可带来的并行收益。

然后选择：

```text
在不超过输出变化预算的条件下，尽可能多地 parallelize layers
```

12 层只有 4096 个 DAG，因此可以精确枚举，不需要近似搜索。

在 aggregate graph-level test 中：

- held-out prediction correlation 约 0.94–0.97；
- ClimbMix 校准结果迁移到 TinyStories 后仍约 0.97；
- 测试 budgets 没有 violation；
- 与 oracle 相比最多只少并行一层。

## Prompt-level 情况

对单个 prompt 动态选择 graph 时，compiler 通常比相同 parallel-layer 数的随机 mask 更好：

- ClimbMix win rate 约 76%–92%；
- 跨 TinyStories 约 80%–100%。

但 ClimbMix prompt-level budget coverage 只有约 77%–89%，没有达到 95% 目标。

因此当前正确说法是：

> Prompt-level defect-guided ranker

而不是：

> 对每个 prompt 都有严格保证的 certificate。

---

# 9. 速度结果意味着什么？

## 9.1 普通 eager PyTorch

Fused parallel execution 大约获得：

- 1.20×–1.29× full-forward speedup。

## 9.2 torch.compile

固定 shape 下：

- batch 1：约 1.45×；
- batch 4：约 1.54×。

这说明 compiler 并不会自动消除 fused parallel graph 的优势。

## 9.3 真实 KV-cache decoding

两个语料训练出的模型都得到：

- TTFT 平均约 1.19×–1.22×；
- TPOT 平均约 1.24×–1.26×。

TPOT 是更稳定、也更有说服力的系统结果。

---

# 10. 为什么 random-mask baseline 不能替代我们？

Random-mask training 每个 batch 随机执行一个 graph，类似 LayerDrop。

它能让很多 graph 都有不错的平均 BPB，但不能让它们实现相同函数。

| 方法 | 两图一致率 | KL |
|---|---:|---:|
| Random mask，1× compute | 78.0% | 31.7 |
| Random mask，2× compute | 71.8% | 53.7 |
| Joint CE | 81.3% | 31.5 |
| Graph consistency | 90.5% | 4.82 |

值得注意：给 random-mask 两倍训练计算后，BPB 更低，但 graph agreement 反而更差。

这说明：

> 更强的语言建模能力不会自动产生 execution-graph equivalence。

Consistency 约束解决的是不同问题。

---

# 11. 这项工作的主要 novelty 是什么？

## 11.1 新的 model elasticity 维度

现有 elastic models 通常改变：

- 深度；
- 宽度；
- token 数；
- expert 数；
- 子网络大小。

我们改变的是：

> Operators 之间的数据依赖结构。

## 11.2 Train once, compile many DAGs

一个 checkpoint 可以在部署时被编译为：

- sequential graph；
- parallel graph；
- mixed graph；
- defect-selected graph；
- hardware-specific graph。

## 11.3 Theory、method、system 形成闭环

不是单独的速度技巧：

1. 发现 graph specialization；
2. 训练 graph equivalence；
3. 用数学解释 defect；
4. 预测多层 rewrite；
5. 编译 graph；
6. 在真实 KV-cache 推理中获得速度收益。

---

# 12. 潜在影响

## 对模型训练

未来模型可以不再固定一个 execution graph，而是训练成一组近似等价的 DAG。

## 对部署

同一个 checkpoint 可以针对：

- GPU 类型；
- batch size；
- latency SLA；
- memory budget；
- 在线/离线 workload；

选择不同 DAG。

## 对推理系统

Graph compiler 可以成为模型与 vLLM/TensorRT 之间的一层：

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

## 对研究

这项工作提出一个新的问题：

> 神经网络权重在多大程度上依赖计算图本身，而不仅依赖参数和算子？

这个问题可以推广到：

- layer reordering；
- communication-compute overlap；
- pipeline execution；
- MoE expert scheduling；
- multimodal branch parallelization；
- speculative execution。

---

# 13. 当前还没有完成什么？

必须诚实说明：

1. 尚未完成正式 vLLM plugin；
2. prompt-level 95% coverage certificate 尚未达到；
3. retrofit 已有模型只能快速恢复 parallel quality，不能快速恢复高 agreement；
4. 当前主要规模是约 120M 参数；
5. 主实验有效结果主要来自锁定方法后的单个训练 seed；
6. 标准能力 benchmark 目前主要是 LAMBADA；
7. 还没有在生产级 continuous batching 环境验证。

这些是后续工作，不应该隐藏。

---

# 14. 最适合对外讲的故事

可以这样向不了解技术的人解释：

> 过去，一个模型的权重只能安全地按照训练时的固定计算流程运行。我们发现，只改变计算顺序，就可能让模型明显变差。我们提出一种训练方法，让同一套权重能够适应多种计算流程。然后，我们为每一层测量“改变流程的风险”，自动选择适合当前硬件和延迟要求的计算路线。结果表明，在几乎不损失模型质量的情况下，可以获得可观的推理加速。

最关键的不是“快了多少”，而是：

> 模型从一个固定程序，变成了可以被重新编译的程序族。

---

# 15. 一句话总结影响与新颖性

## Motivation

固定 execution DAG 限制了 Transformer 对不同硬件和 workload 的适应能力，而普通权重不能安全切换 DAG。

## Result

Graph-consistency training 让一个 checkpoint 在 sequential、parallel 和 mixed DAGs 下保持近似相同质量，并通过 defect compiler 选择低风险 graph。

## Impact

未来可以实现“一次训练、多种部署图”，减少针对不同硬件重复训练模型的需要。

## Novelty

工作首次把 execution dependency graph 作为 Transformer elasticity 和 compilation 的核心对象，并将训练、数学 defect theory、graph compiler 与真实 serving speedup 统一起来。
