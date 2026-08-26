# Execution-Graph Theory：从 Residual Stream 到 Graph Compiler

## 1. 最基础：Transformer 在不断更新 residual stream

对每个 token，Transformer 维护一个向量：

\[
x_l\in\mathbb R^d,
\]

其中 \(l\) 是 layer index。

每层不会完全替换这个向量，而是在原向量上增加 update：

\[
x_{l+1}=x_l+\Delta_l(x_l).
\]

这个持续流经所有层的向量空间通常称为 **residual stream**。

可以把它想象成模型正在修改的一份共享工作草稿：

- 每一层读取当前草稿；
- 产生一个修改建议；
- 把修改加回草稿；
- 下一层继续处理。

---

## 2. Attention 和 FFN 做的事情不同

### Attention：跨 token 收集信息

Attention 让一个 token 从其他 tokens 读取信息：

\[
a_l(x)=A_l(N_l(x)).
\]

它主要决定：

> 当前 token 应该从上下文中的哪些位置获取什么信息？

因此 Attention 是 non-local token-mixing mechanism。

### FFN：对每个 token 做局部变换

FFN 对每个 token 独立应用同一个非线性函数：

\[
f_l(x)=F_l(N_l(x)).
\]

它主要决定：

> 看到当前 token state 后，应该在 residual space 中向哪个方向更新？

因此 FFN 可以看成 residual space 中的 local vector field。

---

## 3. 为什么说 FFN 是 residual-direction steering field？

设当前 residual vector 是 \(x\)，FFN update 是 \(f(x)\)。

FFN update 可以分解成两部分。

### Radial component

与 \(x\) 平行的部分：

\[
f_{\parallel}(x)
=
\frac{\langle f(x),x\rangle}{\|x\|^2}x.
\]

它主要改变 residual vector 的长度。

### Tangential component

与 \(x\) 正交的部分：

\[
f_{\perp}(x)=f(x)-f_{\parallel}(x).
\]

它主要改变 residual vector 的方向。

很多 downstream predictions 更依赖 residual direction，而不仅是 norm。因此 tangential FFN update 像 steering wheel：

- radial update 更像加速或减速；
- tangential update 改变行驶方向。

这就是 **FFN steering field** 的直观含义。

---

## 4. Attention–FFN interaction 是什么？

标准 sequential block 是：

\[
a=A(N(x)),
\]

\[
y=x+a,
\]

\[
S(x)=y+F(N(y)).
\]

FFN 读取的是 Attention 已更新的 state \(y=x+a\)。

因此 Attention 不仅直接更新 residual stream，还会改变：

- FFN 的输入方向；
- FFN neurons 的 activation pattern；
- ReLU/ReLU² gating；
- FFN 最终 steering direction。

这就是 Attention–FFN interaction：

> Attention update 会改变 FFN 随后产生的 update。

如果 FFN 对 Attention 新增的信息高度敏感，这种 interaction 就很强。

---

## 5. Parallel block 去掉了什么？

Parallel block 让 Attention 和 FFN 都读取相同的原始输入：

\[
P(x)=x+A(N(x))+F(N(x)).
\]

Attention 和 FFN 之间不再有直接数据依赖，因此可以同时计算。

Sequential 和 parallel 的差异是：

\[
d(x)=S(x)-P(x),
\]

即：

\[
d(x)
=
F(N(x+A(N(x))))-F(N(x)).
\]

这个 \(d(x)\) 称为 **sequential-to-parallel defect**。

它准确测量：

> 去掉本层 Attention→FFN dependency 后，FFN update 改变了多少。

---

## 6. Defect 的数学解释

为了简化记号，令：

\[
a(x)=A(N(x)),
\qquad
g(x)=F(N(x)).
\]

那么：

\[
d(x)=g(x+a(x))-g(x).
\]

### Exact identity

根据微积分基本定理：

\[
d(x)
=
\int_0^1J_g(x+t a(x))a(x)\,dt.
\]

因此 defect 由两件事决定：

1. Attention update \(a(x)\) 有多大；
2. FFN+Norm 沿这个方向的 Jacobian sensitivity 有多大。

### First-order approximation

当 Attention update 较小时：

\[
d(x)\approx J_g(x)a(x).
\]

这说明 defect 的一阶来源是：

> FFN Jacobian × Attention update。

如果 \(g\) 的 Jacobian 变化很快，则还会出现较大的高阶 nonlinear remainder。

我们的实验发现：

- 后层 defect 更接近一阶近似；
- 前层包含更强的高阶非线性 interaction。

---

## 7. Defect 如何实际计算？

对一个具体 prompt，在第 \(l\) 层分别计算：

\[
S_l(x_l),\qquad P_l(x_l).
\]

然后使用 normalized RMS difference：

\[
d_l(x)
=
\frac{
\sqrt{\operatorname{mean}[(S_l(x)-P_l(x))^2]}
}{
\sqrt{\operatorname{mean}[(S_l(x)-x)^2]}
}.
\]

12 层模型会输出12个 defect values：

```text
[0.043, 0.031, 0.049, ..., 0.036, 0.025]
```

低 defect 层通常更适合被改成 parallel。

---

## 8. 如何根据 pretrained model 的 defect 做 post-hoc selection？

最简单的方法不需要重新训练模型。

### Step 1：收集 calibration prompts

从目标数据分布采样一批文本。

### Step 2：测量每层平均 defect

\[
\bar d_l
=
\mathbb E_x[d_l(x)].
\]

### Step 3：排序 layers

从 defect 最小的层开始选择。

### Step 4：只 parallelize 低 defect layers

例如：

```text
Layer:   0 1 2 3 4 5 6 7 8 9 10 11
Mode:    S S P S S P P S P S P  P
```

这种方法的优点：

- 不修改权重；
- 不重新训练；
- 能避免最敏感的 layers。

局限：

- 单层 effects 可能相互作用；
- calibration 平均值可能隐藏 prompt-specific tail；
- pretrained specialist 本身可能仍高度 graph-dependent。

---

## 9. 为什么 single-layer effects 可以预测 multi-layer graphs？

设 graph mask \(m_l=1\) 表示第 \(l\) 层 parallel。

在 defect 较小时，最终 logit perturbation 可做一阶展开：

\[
\Delta z(m)
\approx
\sum_lm_lv_l,
\]

其中 \(v_l\) 是第 \(l\) 层 defect 传播到最终 logits 后的影响。

因此 multi-layer graph effect 应近似由 single-layer effects 相加得到。

实验上：

\[
D(m)
\approx
\alpha\sum_{l:m_l=1}D(e_l),
\qquad \alpha\approx0.7.
\]

\(\alpha<1\) 表明 interaction 整体略微 subadditive：多个 defects 一起出现时，部分影响会互相抵消或被后续 normalization 缩小。

---

## 10. 原始 post-hoc defect 方法的不足

对于普通 pretrained sequential model：

- 低 defect layers 可能安全；
- 高 defect layers 仍不能 parallelize；
- all-parallel 通常会明显损坏模型；
- 可选择的 graph 范围有限。

也就是说，post-hoc selection 只能寻找模型已经偶然具有的 graph flexibility。

---

## 11. 我们的方法进一步做了什么？

我们不只测量 defect，而是在训练时要求：

- sequential graph 做好 language modeling；
- parallel graph 做好 language modeling；
- 两种 graph 的 centered logits 接近。

训练目标：

\[
L
=
\frac12L_{\mathrm{seq}}
+
\frac12L_{\mathrm{par}}
+
\lambda\|\bar z_{\mathrm{seq}}-\bar z_{\mathrm{par}}\|^2.
\]

结果是模型主动学习：

> 不依赖某一个特定 Attention→FFN execution dependency。

这使：

- all-sequential 可用；
- all-parallel 可用；
- 大量 mixed graphs 可用；
- single-layer defect 更适合用于 graph compilation。

---

## 12. 从 defect ranking 到 graph compiler

给每层一个 predicted divergence cost \(c_l\)，并给用户一个 KL budget \(\epsilon\)。

Compiler 求解：

\[
\max_m\sum_lm_l
\]

subject to：

\[
\alpha\sum_lm_lc_l\le\epsilon.
\]

含义是：

> 在不超过输出变化预算的条件下，尽可能多地 parallelize layers。

这把 defect 从一个分析工具变成 deployment decision variable。

---

## 13. 整条逻辑链

```text
Residual stream
    ↓
Attention 收集跨 token 信息
    ↓
FFN 根据新 state 改变 residual direction
    ↓
Attention→FFN interaction
    ↓
Sequential-to-parallel defect
    ↓
Single-layer defect probes
    ↓
Multi-layer composition law
    ↓
Graph-consistency training
    ↓
Defect-budget graph compiler
    ↓
Hardware-specific execution DAG
```

一句话总结：

> Defect 测量的是 FFN 对本层 Attention update 的敏感度；graph-consistency training 让这种敏感度不再限制模型只能使用一个固定 DAG；graph compiler 则把剩余风险转化为可控制的部署预算。
