# Execution-Graph Theory: From the Residual Stream to a Graph Compiler

## 1. The Foundation: A Transformer Continuously Updates the Residual Stream

For each token, a Transformer maintains a vector:

\[
x_l\in\mathbb R^d,
\]

where \(l\) is the layer index.

Rather than replacing this vector entirely, each layer adds an update to it:

\[
x_{l+1}=x_l+\Delta_l(x_l).
\]

The vector space that flows continuously through all layers is commonly called the **residual stream**.

You can think of it as a shared working draft that the model continually revises:

- each layer reads the current draft;
- produces a proposed revision;
- adds that revision back to the draft;
- and passes it to the next layer for further processing.

---

## 2. Attention and the FFN Serve Different Roles

### Attention: Gathering Information Across Tokens

Attention allows one token to read information from other tokens:

\[
a_l(x)=A_l(N_l(x)).
\]

Its main role is to determine:

> What information should the current token retrieve, and from which positions in the context?

Attention is therefore a non-local token-mixing mechanism.

### FFN: Applying a Local Transformation to Each Token

The FFN applies the same nonlinear function independently to each token:

\[
f_l(x)=F_l(N_l(x)).
\]

Its main role is to determine:

> Given the current token state, in which direction should the residual stream be updated?

The FFN can therefore be viewed as a local vector field in residual space.

---

## 3. Why Call the FFN a Residual-Direction Steering Field?

Let the current residual vector be \(x\), and let the FFN update be \(f(x)\).

The FFN update can be decomposed into two components.

### Radial Component

The component parallel to \(x\) is:

\[
f_{\parallel}(x)
=
\frac{\langle f(x),x\rangle}{\|x\|^2}x.
\]

It primarily changes the magnitude of the residual vector.

### Tangential Component

The component orthogonal to \(x\) is:

\[
f_{\perp}(x)=f(x)-f_{\parallel}(x).
\]

It primarily changes the direction of the residual vector.

Many downstream predictions depend more strongly on residual direction than on norm alone. The tangential FFN update is therefore analogous to a steering wheel:

- a radial update is more like accelerating or decelerating;
- a tangential update changes the direction of travel.

This is the intuition behind the **FFN steering field**.

---

## 4. What Is the Attention–FFN Interaction?

A standard sequential block is:

\[
a=A(N(x)),
\]

\[
y=x+a,
\]

\[
S(x)=y+F(N(y)).
\]

The FFN reads the state \(y=x+a\), which has already been updated by Attention.

Attention therefore not only updates the residual stream directly, but also changes:

- the direction of the FFN input;
- the activation pattern of FFN neurons;
- ReLU/ReLU² gating;
- the FFN's final steering direction.

This is the Attention–FFN interaction:

> The Attention update changes the update subsequently produced by the FFN.

If the FFN is highly sensitive to the information newly introduced by Attention, this interaction is strong.

---

## 5. What Does a Parallel Block Remove?

A parallel block makes Attention and the FFN read the same original input:

\[
P(x)=x+A(N(x))+F(N(x)).
\]

There is no longer a direct data dependency between Attention and the FFN, so they can be computed concurrently.

The difference between the sequential and parallel blocks is:

\[
d(x)=S(x)-P(x),
\]

that is:

\[
d(x)
=
F(N(x+A(N(x))))-F(N(x)).
\]

This \(d(x)\) is called the **sequential-to-parallel defect**.

It measures exactly:

> How much the FFN update changes when the within-layer Attention→FFN dependency is removed.

---

## 6. A Mathematical Interpretation of the Defect

To simplify notation, let:

\[
a(x)=A(N(x)),
\qquad
g(x)=F(N(x)).
\]

Then:

\[
d(x)=g(x+a(x))-g(x).
\]

### Exact Identity

By the fundamental theorem of calculus:

\[
d(x)
=
\int_0^1J_g(x+t a(x))a(x)\,dt.
\]

The defect is therefore determined by two factors:

1. the magnitude of the Attention update \(a(x)\);
2. the Jacobian sensitivity of the FFN+Norm map along that direction.

### First-Order Approximation

When the Attention update is small:

\[
d(x)\approx J_g(x)a(x).
\]

This shows that the first-order source of the defect is:

> FFN Jacobian × Attention update.

If the Jacobian of \(g\) changes rapidly, there will also be a large higher-order nonlinear remainder.

Our experiments find that:

- defects in later layers are better approximated by the first-order term;
- earlier layers contain stronger higher-order nonlinear interactions.

---

## 7. How Is the Defect Computed in Practice?

For a particular prompt, compute the following separately at layer \(l\):

\[
S_l(x_l),\qquad P_l(x_l).
\]

Then use the normalized RMS difference:

\[
d_l(x)
=
\frac{
\sqrt{\operatorname{mean}[(S_l(x)-P_l(x))^2]}
}{
\sqrt{\operatorname{mean}[(S_l(x)-x)^2]}
}.
\]

A 12-layer model produces 12 defect values:

```text
[0.043, 0.031, 0.049, ..., 0.036, 0.025]
```

Layers with low defect are generally better candidates for conversion to parallel execution.

---

## 8. How Can Defects in a Pretrained Model Be Used for Post-Hoc Selection?

The simplest method does not require retraining the model.

### Step 1: Collect Calibration Prompts

Sample a set of texts from the target data distribution.

### Step 2: Measure the Mean Defect of Each Layer

\[
\bar d_l
=
\mathbb E_x[d_l(x)].
\]

### Step 3: Rank the Layers

Select layers in ascending order of defect.

### Step 4: Parallelize Only Low-Defect Layers

For example:

```text
Layer:   0 1 2 3 4 5 6 7 8 9 10 11
Mode:    S S P S S P P S P S P  P
```

Advantages of this method:

- it does not modify the weights;
- it does not require retraining;
- it avoids the most sensitive layers.

Limitations:

- single-layer effects may interact;
- calibration averages may hide prompt-specific tails;
- the pretrained specialist itself may remain highly graph-dependent.

---

## 9. Why Can Single-Layer Effects Predict Multi-Layer Graphs?

Let the graph mask satisfy \(m_l=1\) when layer \(l\) is parallel.

When defects are small, the final logit perturbation admits a first-order expansion:

\[
\Delta z(m)
\approx
\sum_lm_lv_l,
\]

where \(v_l\) is the effect of the layer-\(l\) defect after it propagates to the final logits.

The effect of a multi-layer graph should therefore be approximately the sum of its single-layer effects.

Empirically:

\[
D(m)
\approx
\alpha\sum_{l:m_l=1}D(e_l),
\qquad \alpha\approx0.7.
\]

The fact that \(\alpha<1\) indicates that the interactions are slightly subadditive overall: when multiple defects occur together, some of their effects cancel one another or are attenuated by subsequent normalization.

---

## 10. Limitations of the Original Post-Hoc Defect Method

For an ordinary pretrained sequential model:

- low-defect layers may be safe to parallelize;
- high-defect layers still cannot be parallelized;
- an all-parallel graph usually causes substantial model degradation;
- the range of selectable graphs is limited.

In other words, post-hoc selection can only discover graph flexibility that the model already happens to possess.

---

## 11. What Does Our Method Add?

We do not merely measure the defect. During training, we require that:

- the sequential graph perform language modeling well;
- the parallel graph perform language modeling well;
- the centered logits of the two graphs remain close.

The training objective is:

\[
L
=
\frac12L_{\mathrm{seq}}
+
\frac12L_{\mathrm{par}}
+
\lambda\|\bar z_{\mathrm{seq}}-\bar z_{\mathrm{par}}\|^2.
\]

As a result, the model actively learns:

> Not to depend on one particular Attention→FFN execution dependency.

This makes it possible for:

- the all-sequential graph to remain usable;
- the all-parallel graph to remain usable;
- many mixed graphs to remain usable;
- single-layer defects to become more suitable for graph compilation.

---

## 12. From Defect Ranking to a Graph Compiler

Assign each layer a predicted divergence cost \(c_l\), and let the user specify a KL budget \(\epsilon\).

The compiler solves:

\[
\max_m\sum_lm_l
\]

subject to:

\[
\alpha\sum_lm_lc_l\le\epsilon.
\]

In other words:

> Parallelize as many layers as possible without exceeding the output-change budget.

This turns the defect from an analysis tool into a deployment decision variable.

---

## 13. The Complete Chain of Reasoning

```text
Residual stream
    ↓
Attention gathers information across tokens
    ↓
The FFN changes residual direction based on the new state
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

In one sentence:

> The defect measures the FFN's sensitivity to the within-layer Attention update; graph-consistency training prevents that sensitivity from restricting the model to a single fixed DAG; and the graph compiler converts the remaining risk into a controllable deployment budget.
