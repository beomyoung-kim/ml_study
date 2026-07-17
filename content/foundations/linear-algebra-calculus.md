# Linear Algebra & Calculus for Deep Learning

<div class="tag-row"><span class="tag">Jacobian</span><span class="tag">SVD</span><span class="tag">backprop</span><span class="tag">matrix calculus</span><span class="tag">conditioning</span></div>

> [!TIP] Why this chapter exists
> Microsoft-style research screens explicitly list *linear algebra, probability, optimization* as the math bar. But the real value is **fluency**: being able to write backprop as a chain of matrix products, connect SVD to PCA/LoRA/attention, and reason about *conditioning* when training misbehaves. Attention's $QK^\top/\sqrt{d}$, LoRA's low-rank $\Delta W$, and Adam's per-coordinate scaling are all one language. Speak it.

## The objects you must never confuse

| Object | Maps | Shape | Meaning |
| --- | --- | --- | --- |
| Gradient $\nabla f$ | $\mathbb{R}^n\to\mathbb{R}$ | $n$ | steepest-ascent direction |
| Jacobian $J_f$ | $\mathbb{R}^n\to\mathbb{R}^m$ | $m\times n$ | best local linear map |
| Hessian $H_f$ | $\mathbb{R}^n\to\mathbb{R}$ | $n\times n$ | curvature (2nd order) |

$$
\nabla f = \begin{bmatrix}\partial f/\partial x_1\\\vdots\\\partial f/\partial x_n\end{bmatrix},\quad
[J_f]_{ij}=\frac{\partial f_i}{\partial x_j},\quad
[H_f]_{ij}=\frac{\partial^2 f}{\partial x_i\partial x_j}
$$

> [!WARNING] Layout conventions bite
> Numerator vs denominator layout flips gradients between row and column vectors, so textbooks disagree by a transpose. In an interview, **state your shapes out loud** ("$\bar W$ is the same shape as $W$") and you'll never contradict yourself.

## Backprop is just repeated VJPs

The single most important derivation. One dense layer:

$$
z = Wx + b,\qquad a=\phi(z),\qquad L=L(a)
$$

Given the upstream gradient $\bar a = \partial L/\partial a$, push it back element-by-element and then through the linear map:

$$
\bar z = \bar a \odot \phi'(z),\qquad
\bar W = \bar z\, x^\top,\qquad
\bar b = \bar z,\qquad
\bar x = W^\top \bar z
$$

A full network composes these **vector–Jacobian products (VJPs)**:

$$
\frac{\partial L}{\partial x} = J_1^\top J_2^\top \cdots J_\ell^\top \frac{\partial L}{\partial y}
$$

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="20" y="50" width="70" height="34" rx="6" fill="#6366f1"/><text x="55" y="72" fill="#fff" text-anchor="middle">x</text>
  <rect x="150" y="50" width="90" height="34" rx="6" fill="none" stroke="#232b36" stroke-width="1.5"/><text x="195" y="72" text-anchor="middle" fill="currentColor">Wx+b</text>
  <rect x="300" y="50" width="70" height="34" rx="6" fill="none" stroke="#232b36" stroke-width="1.5"/><text x="335" y="72" text-anchor="middle" fill="currentColor">φ</text>
  <rect x="430" y="50" width="70" height="34" rx="6" fill="#e0533f"/><text x="465" y="72" fill="#fff" text-anchor="middle">L</text>
  <path d="M90 67 H150 M240 67 H300 M370 67 H430" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#fa)"/>
  <path d="M465 100 C 400 130, 260 130, 195 100" stroke="#12a150" stroke-width="1.5" fill="none" marker-end="url(#ga)"/>
  <text x="330" y="128" text-anchor="middle" fill="#12a150">reverse pass: multiply local Jᵀ (VJP)</text>
  <defs>
    <marker id="fa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker>
    <marker id="ga" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
  </defs>
</svg>
<figcaption>Forward stores activations; the backward pass applies each block's transposed Jacobian to the incoming gradient. No full Jacobian is ever materialized.</figcaption>
</figure>

**Why reverse-mode?** With a scalar loss, #params $\gg$ #outputs. Reverse-mode AD computes *all* parameter gradients at a small constant multiple of the forward cost, because it only ever forms $v^\top J$ — never $J$ itself. Forward-mode wins the opposite regime (few inputs, many outputs).

## The matrix-calculus toolkit

$$
\frac{\partial (a^\top x)}{\partial x}=a,\qquad
\frac{\partial (x^\top A x)}{\partial x}=(A+A^\top)x\ \xrightarrow{A=A^\top}\ 2Ax
$$

$$
\frac{\partial}{\partial A}\operatorname{tr}(AB^\top)=B,\qquad
\frac{\partial}{\partial A}\|Y-AX\|_F^2=-2(Y-AX)X^\top
$$

> [!EXAMPLE] Softmax + cross-entropy collapses to $p-y$
> Softmax's Jacobian is $\operatorname{diag}(p)-pp^\top$. Composed with the cross-entropy derivative $-y/p$, the two cancel to the famously clean $\partial L/\partial z = p - y$. This is *why* we fuse `log_softmax + nll_loss` into one `cross_entropy` op — it is both stabler and simpler than doing them separately.

## Eigen / SVD — the spectral view

Any $A\in\mathbb{R}^{m\times n}$ factors as

$$
A = U\Sigma V^\top,\qquad \sigma_1\ge\sigma_2\ge\cdots\ge 0
$$

with $U,V$ orthonormal. Symmetric matrices further diagonalize as $S=Q\Lambda Q^\top$ with real eigenvalues. The bridge: $A^\top A = V\Sigma^\top\Sigma V^\top$, so the right singular vectors are eigenvectors of $A^\top A$ with eigenvalues $\sigma_i^2$.

<dl class="kv">
<dt>PCA</dt><dd>Centered data $X=U\Sigma V^\top$: columns of $V$ are principal directions, projections are $XV$, and $\sigma_i^2/(n-1)$ are explained variances.</dd>
<dt>Low-rank / LoRA</dt><dd>Eckart–Young: truncating to the top $k$ singular values gives the optimal rank-$k$ approximation (Frobenius & spectral). LoRA exploits this by learning $\Delta W\approx BA$.</dd>
<dt>Pseudo-inverse</dt><dd>$A^+ = V\Sigma^+ U^\top$ — the least-squares solver; zero singular values reveal the null space / rank deficiency.</dd>
<dt>Optimizer geometry</dt><dd>Matrix optimizers (Muon and kin, see <a href="#/foundations/optimization">Optimization</a>) treat weight updates through an orthogonalization / spectral lens.</dd>
</dl>

## Norms and conditioning

$$
\|x\|_1=\textstyle\sum_i|x_i|,\quad \|x\|_2=\sqrt{\textstyle\sum_i x_i^2},\quad \|x\|_\infty=\max_i|x_i|
$$
$$
\|A\|_F=\sqrt{\textstyle\sum_{ij}a_{ij}^2}=\sqrt{\textstyle\sum_k\sigma_k^2},\qquad \|A\|_2=\sigma_{\max}(A)
$$

Uses: weight decay ($\ell_2$), gradient clipping (global $\ell_2$), Lipschitz/stability (spectral norm), adversarial budgets ($\ell_p$ balls). The **condition number** $\kappa=\sigma_{\max}/\sigma_{\min}$ (or $\lambda_{\max}/\lambda_{\min}$ for the Hessian/data covariance) controls how hard the problem is to optimize: a large $\kappa$ means one direction is steep and another flat, so any single learning rate is wrong for one of them. Input normalization, whitening, and BatchNorm all *improve conditioning* — a purely-optimization reason they help, distinct from the statistical story.

## Interview Q&A

<details class="qa"><summary>Derive backprop through one dense layer as matrix operations.</summary>
<div class="qa-body">

**Short:** given $\bar a$, the backward pass is $\bar z=\bar a\odot\phi'(z)$, $\bar W=\bar z\,x^\top$, $\bar b=\bar z$, $\bar x=W^\top\bar z$.

**Deep:** each is a VJP. The elementwise nonlinearity has a diagonal Jacobian, so it becomes a Hadamard product with $\phi'(z)$. The linear map $z=Wx$ has Jacobian $W$ w.r.t. $x$ (hence $W^\top\bar z$) and an outer-product structure w.r.t. $W$ (hence $\bar z\,x^\top$, which is rank-1 per example and summed over the batch). Sanity-check shapes: $\bar W$ must match $W$, and $\bar z\,x^\top$ does exactly that.
</div></details>

<details class="qa"><summary>Why is backprop efficient? Answer in linear-algebra terms.</summary>
<div class="qa-body">

**Short:** for a scalar loss, reverse-mode AD gets *every* parameter gradient at ~2–3× the forward cost, because it computes $v^\top J$ without ever forming $J$.

**Deep:** the cost of AD scales with $\min(n_\text{in}, n_\text{out})$ passes. A loss is a single scalar output, so one reverse sweep suffices regardless of parameter count — the opposite of forward-mode, which needs one pass per input. The real bottleneck isn't compute but **activation memory** (you must cache forward activations to reuse them backward), which is why gradient checkpointing and FSDP exist. See [Distributed Training](#/foundations/distributed-training).
</div></details>

<details class="qa"><summary>What is SVD and where does it show up in ML?</summary>
<div class="qa-body">

**Short:** $A=U\Sigma V^\top$ — orthonormal $\times$ singular values $\times$ orthonormal, defined for *any* matrix.

**Deep:** it underlies PCA (eigen-decomposition of the covariance = SVD of centered data), optimal low-rank approximation (Eckart–Young → truncated SVD, LoRA, model compression), the Moore–Penrose pseudo-inverse for least squares, and spectral analysis of Jacobians/Hessians (sharp vs. flat minima). A zero singular value signals rank deficiency / a null space direction the model can't distinguish.
</div></details>

<details class="qa"><summary>Why the $1/\sqrt{d}$ scaling in attention?</summary>
<div class="qa-body">

**Short:** it keeps the dot-product logits at unit variance so softmax doesn't saturate.

**Deep:** if $q,k$ have i.i.d. zero-mean unit-variance entries, $q^\top k=\sum_{i=1}^{d} q_i k_i$ has variance $d$ (sum of $d$ independent unit-variance terms). Without rescaling, logits grow like $\sqrt{d}$, pushing softmax toward one-hot and killing gradients through the other keys. Dividing by $\sqrt{d}$ restores $\mathcal{O}(1)$ variance. This is the same additive-variance fact behind $\|x\|_2\!\approx\!\sqrt d$ for random vectors.
</div></details>

<details class="qa"><summary>How does ill-conditioning slow training, and what fixes it?</summary>
<div class="qa-body">

**Short:** a large Hessian condition number $\kappa$ forces a compromise learning rate — too big for the steep direction, too small for the flat one — causing zig-zag and slow progress.

**Deep:** vanilla gradient descent's convergence rate degrades like $(\kappa-1)/(\kappa+1)$ per step. Fixes attack $\kappa$ from different angles: **momentum** dampens oscillation along steep directions; **Adam's** diagonal preconditioner rescales each coordinate; **normalization** (input, Batch/LayerNorm) reshapes the loss surface toward isotropy; true **second-order** methods multiply by $H^{-1}$ to sphere the landscape but are too expensive at scale. See [Optimization](#/foundations/optimization).
</div></details>

**Follow-ups you should expect**

- *When is the Hessian symmetric?* When mixed partials are continuous (Clairaut/Schwarz).
- *JVP vs. VJP?* Forward-mode gives $Jv$ (columns of $J$); reverse-mode gives $v^\top J$ (rows). Autodiff libraries expose both.
- *How do you compute a Hessian-vector product without the Hessian?* Pearlmutter's trick: differentiate $\nabla f \cdot v$ once more — cost of one extra backward pass.
- *PCA vs. autoencoder?* Linear vs. nonlinear compression; a linear AE with MSE recovers the PCA subspace.
- *Is conv a matrix multiply?* Yes — im2col turns it into a (Toeplitz-structured) GEMM; self-attention is two GEMMs around a softmax.

## Cheat-sheet

| Fact | One-liner |
| --- | --- |
| Grad/Jac/Hess | scalar→vec grad; vec→vec Jacobian $m\times n$; scalar curvature $n\times n$. |
| Dense-layer backward | $\bar z=\bar a\odot\phi'(z)$, $\bar W=\bar z x^\top$, $\bar x=W^\top\bar z$. |
| Reverse-mode wins | scalar loss ⇒ all grads at ~2–3× forward; only forms $v^\top J$. |
| Softmax+CE gradient | $p-y$ — the two Jacobians cancel; fuse the ops. |
| SVD | $A=U\Sigma V^\top$; top-$k$ = optimal low-rank (Eckart–Young). |
| Eigen↔SVD | $V$ = eigenvectors of $A^\top A$, eigenvalues $\sigma_i^2$. |
| Norms | $\ell_1$ sparse, $\ell_2$ energy, $\|A\|_2=\sigma_{\max}$ (Lipschitz). |
| Attention scale | $1/\sqrt d$ keeps logit variance $\mathcal O(1)$ ⇒ no softmax saturation. |
| Conditioning | large $\kappa$ ⇒ slow/zig-zag; normalization & preconditioning help. |

**Related:** [Optimization](#/foundations/optimization) · [Probability & Statistics](#/foundations/probability-statistics) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [Normalization & Stability](#/foundations/normalization-stability) · [Attention From Scratch](#/ml-coding/attention)
