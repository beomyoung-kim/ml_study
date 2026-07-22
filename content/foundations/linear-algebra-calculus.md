# Linear Algebra & Calculus for Deep Learning

> [!NOTE] Goal of this chapter
> Do not be intimidated. The mathematics required for deep learning reduces to **two core intuitions**: (1) how to group numbers and operate on them at once—**linear algebra**, including vectors, matrices, and matrix multiplication—and (2) how to measure "how much the result changes when I change this number slightly"—**calculus**, including derivatives, gradients, and the chain rule. Together they make it possible to "update parameters with gradients" in [What Is Machine Learning?](#/foundations/what-is-ml). Sections §0–§2 are for complete beginners; the later sections go deeper for interviews.

## §0 · Why vectors and matrices?

In [Neural Networks from First Principles](#/foundations/neural-networks-basics), one neuron computed $z = w_1x_1 + w_2x_2 + \dots$. With thousands of neurons, we cannot write each term separately, so we group the numbers.

- **Scalar:** one number, such as `3.14`
- **Vector:** numbers in a row, such as `[2, 5, 1]`; for example, one person's height, weight, and age
- **Matrix:** numbers in a grid; for example, data for several people or all weights in one layer
- **Tensor:** three or more dimensions; for example, an image batch with shape `(N, C, H, W)`

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="55" y="20" text-anchor="middle" fill="#98a3b2">Scalar</text>
  <rect x="35" y="35" width="40" height="40" rx="6" fill="#6366f1"/><text x="55" y="60" text-anchor="middle" fill="#fff">7</text>
  <text x="180" y="20" text-anchor="middle" fill="#98a3b2">Vector (3,)</text>
  <g fill="none" stroke="#0ea5e9" stroke-width="1.6"><rect x="150" y="35" width="40" height="40" rx="6"/><rect x="150" y="75" width="40" height="40" rx="6"/><rect x="150" y="115" width="40" height="0" rx="6"/></g>
  <rect x="150" y="35" width="40" height="120" rx="6" fill="none" stroke="#0ea5e9" stroke-width="1.6"/>
  <line x1="150" y1="75" x2="190" y2="75" stroke="#0ea5e9"/><line x1="150" y1="115" x2="190" y2="115" stroke="#0ea5e9"/>
  <text x="170" y="60" text-anchor="middle" fill="currentColor">2</text><text x="170" y="100" text-anchor="middle" fill="currentColor">5</text><text x="170" y="140" text-anchor="middle" fill="currentColor">1</text>
  <text x="360" y="20" text-anchor="middle" fill="#98a3b2">Matrix (2×3)</text>
  <rect x="290" y="35" width="140" height="80" rx="6" fill="none" stroke="#e0533f" stroke-width="1.6"/>
  <line x1="290" y1="75" x2="430" y2="75" stroke="#e0533f"/><line x1="337" y1="35" x2="337" y2="115" stroke="#e0533f"/><line x1="384" y1="35" x2="384" y2="115" stroke="#e0533f"/>
  <text x="500" y="20" text-anchor="middle" fill="#98a3b2">Tensor (3D+)</text>
  <g fill="none" stroke="#12a150" stroke-width="1.4"><rect x="480" y="45" width="70" height="55" rx="4"/><rect x="490" y="38" width="70" height="55" rx="4"/><rect x="500" y="31" width="70" height="55" rx="4"/></g>
</svg>
<figcaption>Each object simply adds another dimension. Half the bugs in deep-learning code come from a mismatched <b>shape</b>, so always say the shapes out loud.</figcaption>
</figure>

### Dot product—the most important operation

The **dot product** of two vectors multiplies corresponding elements and sums the results. This is exactly a neuron's weighted sum:

$$
w \cdot x = \sum_i w_i x_i = w_1x_1 + w_2x_2 + \dots
$$

A dot product also measures how closely two vectors point in the same direction. **Attention's $QK^\top$, similarity in a recommender system, and a neuron's weighted sum are all dot products.**

### Matrix multiplication—a bundle of dot products

Matrix product $C = AB$ fills a grid with the dot product of each row of $A$ and each column of $B$. There is one rule: **the inner dimensions must match.** $(m \times k) \cdot (k \times n) = (m \times n)$.

<figure>
<svg viewBox="0 0 560 170" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="20" y="50" width="110" height="70" rx="6" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="75" y="40" text-anchor="middle" fill="#0ea5e9">A (m×k)</text>
  <rect x="45" y="55" width="100" height="18" fill="rgba(14,165,233,.25)"/><text x="150" y="68" fill="#0ea5e9">row</text>
  <text x="160" y="90" text-anchor="middle" font-size="18" fill="currentColor">×</text>
  <rect x="190" y="30" width="90" height="110" rx="6" fill="none" stroke="#e0533f" stroke-width="1.6"/><text x="235" y="20" text-anchor="middle" fill="#e0533f">B (k×n)</text>
  <rect x="205" y="35" width="18" height="100" fill="rgba(224,83,63,.25)"/><text x="235" y="155" fill="#e0533f">column</text>
  <text x="320" y="90" text-anchor="middle" font-size="18" fill="currentColor">=</text>
  <rect x="360" y="50" width="120" height="70" rx="6" fill="none" stroke="#12a150" stroke-width="1.8"/><text x="420" y="40" text-anchor="middle" fill="#12a150">C (m×n)</text>
  <circle cx="378" cy="68" r="7" fill="#12a150"/><text x="470" y="135" text-anchor="end" fill="#98a3b2">C[i,j] = row i of A · column j of B</text>
</svg>
<figcaption>One green cell is the dot product of a blue row and a red column. One neural-network layer multiplies an input vector by a weight matrix, $Wx$.</figcaption>
</figure>

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"matmul","packages":["numpy"],"approx":true,"starter":"def matmul(A, B):\n    # Return the matrix product C:(m,n) of A:(m,k) and B:(k,n).\n    # C[i][j] = row i of A dot column j of B.\n    # You may use NumPy (A @ B) or implement it with three nested loops.\n    pass","tests":[{"args":[[[1,2],[3,4]],[[5,6],[7,8]]],"expect":[[19.0,22.0],[43.0,50.0]]},{"args":[[[1,0,2]],[[3],[4],[5]]],"expect":[[13.0]]},{"args":[[[2,0],[0,2]],[[1,2],[3,4]]],"expect":[[2.0,4.0],[6.0,8.0]]}],"solution":"import numpy as np\n\ndef matmul(A, B):\n    return (np.asarray(A, float) @ np.asarray(B, float)).tolist()"}
</script>
</div>

> [!TIP] Broadcasting in one line
> When NumPy operates on arrays with different shapes, it can align them automatically through broadcasting. For example, `(N,1)` + `(1,M)` → `(N,M)`. [ML Coding](#/ml-coding/intro) uses this trick for pairwise calculations without loops.

## §1 · Derivatives: "How much does the result change if I change this slightly?"

A **derivative** is a slope. $\frac{df}{dx}$ asks, "If I increase $x$ by a tiny amount, by what multiple does $f$ change?" When [What Is Machine Learning?](#/foundations/what-is-ml) moves parameters to reduce the loss, the gradient answers the question "In which direction?"

<figure>
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <path d="M40 200 Q320 -40 600 200" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <!-- tangent at left (steep) -->
  <line x1="90" y1="185" x2="220" y2="90" stroke="#e0533f" stroke-width="2"/>
  <circle cx="140" cy="140" r="5" fill="#e0533f"/>
  <text x="150" y="135" fill="#e0533f">large slope → large move</text>
  <!-- tangent at bottom (flat) -->
  <line x1="270" y1="40" x2="370" y2="40" stroke="#12a150" stroke-width="2"/>
  <circle cx="320" cy="40" r="5" fill="#12a150"/>
  <text x="320" y="30" text-anchor="middle" fill="#12a150">slope ≈ 0 (minimum)</text>
  <text x="320" y="215" text-anchor="middle" fill="#98a3b2">At a point on the loss surface, move opposite the slope—downhill</text>
</svg>
<figcaption>The gradient points in the direction of steepest ascent. To <b>reduce</b> the loss, move in the opposite direction. This is gradient descent.</figcaption>
</figure>

With several inputs, the vector of partial derivatives with respect to each input is the **gradient** $\nabla f$:

$$
\nabla f = \left[\frac{\partial f}{\partial x_1},\ \frac{\partial f}{\partial x_2},\ \dots\right]
$$

### The chain rule—the heart of deep-learning training

When functions are nested, as in $f(g(h(x)))$, the overall gradient is the **product** of the gradients at each step:

$$
\frac{df}{dx} = \frac{df}{dg}\cdot\frac{dg}{dh}\cdot\frac{dh}{dx}
$$

A neural network is a composition of layers. If we send the output error backward toward the input, multiplying as we go, we obtain the gradient for every layer. This is **backpropagation**.

### Backpropagation by hand, with small numbers

Let $x=2$ and $L = (3x)^2$. Then $g = 3x = 6$ and $L = g^2 = 36$. By the chain rule:

$$
\frac{dL}{dx} = \underbrace{\frac{dL}{dg}}_{2g=12}\cdot\underbrace{\frac{dg}{dx}}_{3} = 12\times 3 = 36
$$

Check it directly: $L=(3x)^2=9x^2$, so $dL/dx=18x=36$ ✓. The intuition that we **multiply gradients working from the back toward the front** is all of backpropagation.

## §2 · Neural-network backpropagation in matrix form

Now for a real dense layer: $z = Wx + b$, $a=\phi(z)$, and $L=L(a)$. Given upstream gradient $\bar a = \partial L/\partial a$, the backward pass is only four lines:

$$
\bar z = \bar a \odot \phi'(z),\qquad
\bar W = \bar z\, x^\top,\qquad
\bar b = \bar z,\qquad
\bar x = W^\top \bar z
$$

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="20" y="50" width="70" height="34" rx="6" fill="#6366f1"/><text x="55" y="72" fill="#fff" text-anchor="middle">x</text>
  <rect x="150" y="50" width="90" height="34" rx="6" fill="none" stroke="#98a3b2" stroke-width="1.5"/><text x="195" y="72" text-anchor="middle" fill="currentColor">Wx+b</text>
  <rect x="300" y="50" width="70" height="34" rx="6" fill="none" stroke="#98a3b2" stroke-width="1.5"/><text x="335" y="72" text-anchor="middle" fill="currentColor">φ</text>
  <rect x="430" y="50" width="70" height="34" rx="6" fill="#e0533f"/><text x="465" y="72" fill="#fff" text-anchor="middle">L</text>
  <path d="M90 67 H150 M240 67 H300 M370 67 H430" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#fa)"/>
  <text x="255" y="42" text-anchor="middle" fill="#0ea5e9">forward pass →</text>
  <path d="M465 100 C 400 130, 260 130, 195 100" stroke="#12a150" stroke-width="1.5" fill="none" marker-end="url(#ga)"/>
  <text x="330" y="128" text-anchor="middle" fill="#12a150">← backward: multiply local Jᵀ (VJP)</text>
  <defs>
    <marker id="fa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#0ea5e9"/></marker>
    <marker id="ga" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
  </defs>
</svg>
<figcaption>The forward pass stores activations; the backward pass multiplies the gradient by each block's transposed Jacobian. It never materializes the full Jacobian. This operation is a <b>vector–Jacobian product (VJP)</b>.</figcaption>
</figure>

**Check the shapes:** $\bar W$ must have the same shape as $W$, and $\bar z\,x^\top$ does. Saying "$\bar W$ has the same shape as $W$" aloud in an interview prevents sign and transpose mistakes.

The full network composes these VJPs:

$$
\frac{\partial L}{\partial x} = J_1^\top J_2^\top \cdots J_\ell^\top \frac{\partial L}{\partial y}
$$

**Why reverse mode?** The loss is one scalar, while the network can have billions of parameters. A reverse sweep composes only VJPs and obtains every parameter gradient without forming a full $J$. The exact cost depends on the operation and implementation, but it is usually the same order as the forward pass. Forward-mode JVPs can instead be advantageous when there are few input dimensions and many output dimensions.

### Check a gradient in code with a numerical derivative

When you doubt a formula, approximate the derivative directly from its definition with $\frac{f(x+h)-f(x-h)}{2h}$ and compare. This is the standard way to validate a custom backward implementation. The function below computes the gradient of $f(x)=x^3$ **without using its derivative formula**; the answer should be close to $3x^2$.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"grad_cube","packages":["numpy"],"approx":true,"starter":"def grad_cube(x, h=1e-5):\n    # Approximate the derivative of f(x) = x**3 at x with centered differences:\n    # (f(x+h) - f(x-h)) / (2h). The result should be close to 3*x**2,\n    # even though you do not use the derivative formula.\n    pass","tests":[{"args":[2.0],"expect":12.0,"tol":1e-3},{"args":[3.0],"expect":27.0,"tol":1e-3},{"args":[0.0],"expect":0.0,"tol":1e-3}],"solution":"def grad_cube(x, h=1e-5):\n    f = lambda t: t ** 3\n    return (f(x + h) - f(x - h)) / (2 * h)"}
</script>
</div>

The key intuition is that **even without knowing a derivative formula, you can approximate a small problem's gradient numerically and check your backward pass.** Autograd does not automate finite differences, however. A framework records the derivative rule for each operation in the computation graph and composes those rules with the chain rule to compute VJPs or JVPs. Finite differences independently verify that result as a gradient-checking tool.

## §3 · Eigenvalues and SVD—the spectral view (Advanced)

Any matrix $A\in\mathbb{R}^{m\times n}$ can be factored as

$$
A = U\Sigma V^\top,\qquad \sigma_1\ge\sigma_2\ge\cdots\ge 0
$$

$U$ and $V$ are orthonormal. Intuitively, **every linear transformation decomposes into rotation → axis-wise scaling → rotation.** Each singular value $\sigma_i$ tells us how much one axis is stretched.

<dl class="kv">
<dt>PCA (dimensionality reduction)</dt><dd>For centered data $X=U\Sigma V^\top$, the columns of $V$ are principal directions, the projection is $XV$, and $\sigma_i^2/(n-1)$ is the explained variance.</dd>
<dt>Low rank / LoRA</dt><dd>Eckart–Young states that truncated SVD is the optimal rank-$k$ approximation of an <em>already given matrix</em>. LoRA does not directly apply this theorem; it parameterizes the fine-tuning update as $\Delta W=BA$ and learns only a low-rank update. It therefore does not automatically inherit truncated SVD's optimality guarantee.</dd>
<dt>Pseudo-inverse</dt><dd>$A^+ = V\Sigma^+ U^\top$, the least-squares solution. Zero singular values expose the null space and rank deficiency.</dd>
<dt>Optimizer geometry</dt><dd>Matrix optimizers in the Muon family—see <a href="#/foundations/optimization">Optimization</a>—treat weight updates through an orthogonalization or spectral perspective.</dd>
</dl>

## §4 · Norms and conditioning (Advanced)

$$
\|x\|_1=\textstyle\sum_i|x_i|,\quad \|x\|_2=\sqrt{\textstyle\sum_i x_i^2},\quad \|A\|_2=\sigma_{\max}(A)
$$

Applications include weight decay ($\ell_2$), gradient clipping with global $\ell_2$, and Lipschitz control and stability through the spectral norm. The **condition number** $\kappa=\sigma_{\max}/\sigma_{\min}$ determines optimization difficulty. When $\kappa$ is large, one direction is steep and another is flat, so any single learning rate is wrong for one of them and optimization zigzags. Input normalization, whitening, and BatchNorm all **improve conditioning** and speed learning—an optimization benefit distinct from their statistical interpretations.

## Interview Q&A

<details class="qa"><summary>Derive backpropagation through one dense layer as matrix operations.</summary>
<div class="qa-body">

**Short:** Given $\bar a$, compute $\bar z=\bar a\odot\phi'(z)$, $\bar W=\bar z\,x^\top$, $\bar b=\bar z$, and $\bar x=W^\top\bar z$.

**Deep:** Each expression is a VJP. An elementwise nonlinearity has a diagonal Jacobian, so it becomes a Hadamard product with $\phi'(z)$. For the linear map $z=Wx$, the Jacobian with respect to $x$ is $W$, giving $W^\top\bar z$, while the Jacobian with respect to $W$ has an outer-product structure, giving $\bar z\,x^\top$. It is rank one per example and summed across a batch. Shape check: $\bar W$ must match $W$, and $\bar z\,x^\top$ does.
</div></details>

<details class="qa"><summary>Why is the softmax + cross-entropy gradient so simple?</summary>
<div class="qa-body">

**Short:** Their Jacobians cancel, reducing $\partial L/\partial z$ to $p-y$, predicted probability minus target.

**Deep:** The softmax and cross-entropy Jacobians cancel exactly and reduce to $p-y$. This is why frameworks fuse `log_softmax + nll_loss` into a single `cross_entropy` operation: it is more stable and simpler. For the full step-by-step derivation, see the canonical owner, [Losses & Gradients](#/ml-coding/losses-gradients).
</div></details>

<details class="qa"><summary>What is SVD, and where does it appear in ML?</summary>
<div class="qa-body">

**Short:** $A=U\Sigma V^\top$, a rotation–scaling–rotation decomposition defined for any matrix.

**Deep:** It underlies PCA—the eigendecomposition of a covariance matrix is the SVD of centered data—optimal low-rank approximation through Eckart–Young and truncated SVD, LoRA and model compression, the pseudo-inverse for least squares, and spectral analysis of Jacobians and Hessians. A zero singular value means rank deficiency or a null-space direction the model cannot distinguish.
</div></details>

<details class="qa"><summary>Why scale attention by $1/\sqrt{d}$?</summary>
<div class="qa-body">

**Short:** It keeps dot-product logits at unit variance so softmax does not saturate.

**Deep:** If the entries of $q$ and $k$ are i.i.d., zero-mean, and unit-variance, then $q^\top k$ has variance $d$. Dividing by $\sqrt d$ returns it to $\mathcal{O}(1)$. Without scaling, large logits saturate softmax and kill gradients. See [Attention from Scratch](#/ml-coding/attention) for the full derivation and code.
</div></details>

**Expected follow-ups**

- *JVP vs VJP?* Forward mode computes $Jv$, associated with columns; reverse mode computes $v^\top J$, associated with rows. Autodiff libraries provide both.
- *A Hessian-vector product without constructing the Hessian?* Pearlmutter's trick: differentiate $\nabla f\cdot v$ once more, for roughly one backward pass.
- *Is convolution a matrix multiplication?* Yes. im2col converts it to GEMM, and self-attention is two GEMMs around a softmax.
- *What is backpropagation's real bottleneck?* Activation memory for cached forward values, not just compute. This is why gradient checkpointing and FSDP exist. See [Distributed Training](#/foundations/distributed-training).

## Cheat sheet

| Fact | In one line |
| --- | --- |
| Dot product | $w\cdot x=\sum w_ix_i$: a neuron's weighted sum and a similarity measure |
| Matrix multiplication | $C[i,j]$ = row i of A · column j of B; inner dimensions match |
| Gradient | Direction of steepest ascent; descend in the opposite direction to reduce loss |
| Chain rule | Gradient of a composition = product of stepwise gradients → backpropagation |
| Dense backward | $\bar z=\bar a\odot\phi'(z)$, $\bar W=\bar z x^\top$, $\bar x=W^\top\bar z$ |
| Reverse mode | Scalar loss ⇒ all parameter gradients, usually at the same computational order as the forward pass; constants and memory depend on graph and kernels |
| Softmax + CE | Gradient = $p-y$ because the two Jacobians cancel |
| SVD | $A=U\Sigma V^\top$; top $k$ gives the optimal low-rank approximation by Eckart–Young |
| Conditioning | Large $\kappa$ ⇒ slow zigzagging; normalization and preconditioning help |

**Next:** [Optimization](#/foundations/optimization) · [Probability & Statistics](#/foundations/probability-statistics) · [Neural Network Architectures](#/foundations/architectures) · [Implement Attention](#/ml-coding/attention)
