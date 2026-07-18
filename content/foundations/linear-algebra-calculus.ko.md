# Linear Algebra & Calculus for Deep Learning

<div class="tag-row"><span class="tag">Jacobian</span><span class="tag">SVD</span><span class="tag">backprop</span><span class="tag">matrix calculus</span><span class="tag">conditioning</span></div>

> [!TIP] 이 장이 존재하는 이유
> Microsoft 스타일의 research 스크리닝은 *linear algebra, probability, optimization*을 수학 기준선으로 명시적으로 제시합니다. 하지만 진짜 가치는 **유창함**에 있습니다. backprop을 matrix product의 chain으로 쓸 줄 알고, SVD를 PCA/LoRA/attention과 연결하며, 학습이 이상하게 굴 때 *conditioning* 관점에서 추론할 수 있는 능력이죠. Attention의 $QK^\top/\sqrt{d}$, LoRA의 low-rank $\Delta W$, Adam의 좌표별 스케일링은 모두 하나의 언어입니다. 그 언어를 구사하세요.

## 절대 헷갈리면 안 되는 대상들

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

> [!WARNING] Layout convention은 발목을 잡는다
> Numerator layout이냐 denominator layout이냐에 따라 gradient가 row vector와 column vector 사이를 오가고, 그래서 교재마다 transpose 하나 차이로 서로 다릅니다. 면접에서는 **shape를 소리 내어 명시**하세요("$\bar W$는 $W$와 같은 shape입니다"). 그러면 스스로 모순되는 일이 절대 없습니다.

## Backprop은 그저 VJP의 반복이다

가장 중요한 유도입니다. dense layer 하나:

$$
z = Wx + b,\qquad a=\phi(z),\qquad L=L(a)
$$

upstream gradient $\bar a = \partial L/\partial a$가 주어지면, 이를 원소별로 뒤로 밀고 그다음 linear map을 통과시킵니다:

$$
\bar z = \bar a \odot \phi'(z),\qquad
\bar W = \bar z\, x^\top,\qquad
\bar b = \bar z,\qquad
\bar x = W^\top \bar z
$$

전체 network는 이 **vector–Jacobian product (VJP)**들을 합성합니다:

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
<figcaption>Forward pass는 activation을 저장하고, backward pass는 각 block의 transposed Jacobian을 들어오는 gradient에 적용합니다. 전체 Jacobian은 절대 실체화되지 않습니다.</figcaption>
</figure>

**왜 reverse-mode인가?** scalar loss에서는 #params $\gg$ #outputs입니다. Reverse-mode AD는 $J$ 자체를 만들지 않고 오직 $v^\top J$만 형성하기 때문에, forward 비용의 작은 상수 배로 *모든* parameter gradient를 계산합니다. Forward-mode는 반대 상황(input이 적고 output이 많은 경우)에서 유리합니다.

## Matrix-calculus 툴킷

$$
\frac{\partial (a^\top x)}{\partial x}=a,\qquad
\frac{\partial (x^\top A x)}{\partial x}=(A+A^\top)x\ \xrightarrow{A=A^\top}\ 2Ax
$$

$$
\frac{\partial}{\partial A}\operatorname{tr}(AB^\top)=B,\qquad
\frac{\partial}{\partial A}\|Y-AX\|_F^2=-2(Y-AX)X^\top
$$

> [!EXAMPLE] Softmax + cross-entropy는 $p-y$로 붕괴한다
> Softmax의 Jacobian은 $\operatorname{diag}(p)-pp^\top$입니다. 여기에 cross-entropy의 derivative $-y/p$를 합성하면 둘이 상쇄되어 그 유명하고 깔끔한 $\partial L/\partial z = p - y$가 나옵니다. 이것이 바로 우리가 `log_softmax + nll_loss`를 하나의 `cross_entropy` op으로 fuse하는 *이유*입니다 — 따로 하는 것보다 더 안정적이면서도 더 간단합니다.

## Eigen / SVD — spectral 관점

임의의 $A\in\mathbb{R}^{m\times n}$는 다음과 같이 분해됩니다:

$$
A = U\Sigma V^\top,\qquad \sigma_1\ge\sigma_2\ge\cdots\ge 0
$$

여기서 $U,V$는 orthonormal입니다. Symmetric matrix는 한발 더 나아가 실수 eigenvalue를 갖는 $S=Q\Lambda Q^\top$로 diagonalize됩니다. 다리 역할을 하는 관계: $A^\top A = V\Sigma^\top\Sigma V^\top$이므로, right singular vector는 eigenvalue $\sigma_i^2$를 갖는 $A^\top A$의 eigenvector입니다.

<dl class="kv">
<dt>PCA</dt><dd>Centering한 데이터 $X=U\Sigma V^\top$: $V$의 column이 principal direction이고, projection은 $XV$이며, $\sigma_i^2/(n-1)$이 explained variance입니다.</dd>
<dt>Low-rank / LoRA</dt><dd>Eckart–Young: 상위 $k$개 singular value로 truncate하면 optimal rank-$k$ approximation을 얻습니다(Frobenius & spectral). LoRA는 $\Delta W\approx BA$를 학습해 이를 활용합니다.</dd>
<dt>Pseudo-inverse</dt><dd>$A^+ = V\Sigma^+ U^\top$ — least-squares solver이며, 0인 singular value가 null space / rank deficiency를 드러냅니다.</dd>
<dt>Optimizer geometry</dt><dd>Matrix optimizer(Muon 계열, <a href="#/foundations/optimization">Optimization</a> 참고)는 weight update를 orthogonalization / spectral 렌즈로 다룹니다.</dd>
</dl>

## Norm과 conditioning

$$
\|x\|_1=\textstyle\sum_i|x_i|,\quad \|x\|_2=\sqrt{\textstyle\sum_i x_i^2},\quad \|x\|_\infty=\max_i|x_i|
$$
$$
\|A\|_F=\sqrt{\textstyle\sum_{ij}a_{ij}^2}=\sqrt{\textstyle\sum_k\sigma_k^2},\qquad \|A\|_2=\sigma_{\max}(A)
$$

용도: weight decay($\ell_2$), gradient clipping(global $\ell_2$), Lipschitz/안정성(spectral norm), adversarial budget($\ell_p$ ball). **condition number** $\kappa=\sigma_{\max}/\sigma_{\min}$(Hessian/데이터 covariance의 경우 $\lambda_{\max}/\lambda_{\min}$)는 문제를 optimize하기가 얼마나 어려운지를 결정합니다. 큰 $\kappa$는 한 방향은 가파르고 다른 방향은 평평하다는 뜻이라, 어떤 단일 learning rate를 써도 둘 중 하나에는 틀린 값이 됩니다. Input normalization, whitening, BatchNorm은 모두 *conditioning을 개선*합니다 — 이것들이 도움이 되는 순수하게 optimization적인 이유로, statistical한 설명과는 별개입니다.

## Interview Q&A

<details class="qa"><summary>Dense layer 하나를 통과하는 backprop을 matrix 연산으로 유도하라.</summary>
<div class="qa-body">

**Short:** $\bar a$가 주어지면 backward pass는 $\bar z=\bar a\odot\phi'(z)$, $\bar W=\bar z\,x^\top$, $\bar b=\bar z$, $\bar x=W^\top\bar z$입니다.

**Deep:** 각각이 VJP입니다. Elementwise nonlinearity는 diagonal Jacobian을 가지므로 $\phi'(z)$와의 Hadamard product가 됩니다. Linear map $z=Wx$는 $x$에 대해 Jacobian이 $W$이고(따라서 $W^\top\bar z$), $W$에 대해서는 outer-product 구조를 가집니다(따라서 $\bar z\,x^\top$, example마다 rank-1이고 batch에 대해 합산). shape로 sanity-check: $\bar W$는 반드시 $W$와 일치해야 하는데, $\bar z\,x^\top$이 정확히 그렇습니다.
</div></details>

<details class="qa"><summary>Backprop은 왜 효율적인가? linear-algebra 용어로 답하라.</summary>
<div class="qa-body">

**Short:** scalar loss에서 reverse-mode AD는 $J$를 형성하지 않고 $v^\top J$를 계산하기 때문에, forward 비용의 약 2~3배로 *모든* parameter gradient를 얻습니다.

**Deep:** AD의 비용은 $\min(n_\text{in}, n_\text{out})$번의 pass에 비례합니다. loss는 단일 scalar output이므로 parameter 수와 무관하게 reverse sweep 한 번이면 충분합니다 — input마다 pass 한 번이 필요한 forward-mode와 정반대죠. 진짜 병목은 compute가 아니라 **activation memory**(backward에서 재사용하려고 forward activation을 캐시해야 함)이며, 그래서 gradient checkpointing과 FSDP가 존재합니다. [Distributed Training](#/foundations/distributed-training)을 참고하세요.
</div></details>

<details class="qa"><summary>SVD란 무엇이고 ML 어디에 등장하는가?</summary>
<div class="qa-body">

**Short:** $A=U\Sigma V^\top$ — orthonormal $\times$ singular value $\times$ orthonormal, *임의의* matrix에 대해 정의됩니다.

**Deep:** SVD는 PCA(covariance의 eigen-decomposition = centering한 데이터의 SVD), optimal low-rank approximation(Eckart–Young → truncated SVD, LoRA, model compression), least squares를 위한 Moore–Penrose pseudo-inverse, Jacobian/Hessian의 spectral 분석(sharp minima vs flat minima)의 기반입니다. 0인 singular value는 rank deficiency / model이 구분할 수 없는 null space 방향을 신호합니다.
</div></details>

<details class="qa"><summary>attention에서 왜 $1/\sqrt{d}$ 스케일링인가?</summary>
<div class="qa-body">

**Short:** dot-product logit을 unit variance로 유지해서 softmax가 saturate되지 않게 합니다.

**Deep:** $q,k$가 i.i.d. zero-mean unit-variance 원소를 가지면 $q^\top k=\sum_{i=1}^{d} q_i k_i$는 variance가 $d$입니다(independent unit-variance 항 $d$개의 합). rescaling이 없으면 logit이 $\sqrt{d}$처럼 커져 softmax를 one-hot 쪽으로 밀어붙이고 다른 key들로 가는 gradient를 죽입니다. $\sqrt{d}$로 나누면 $\mathcal{O}(1)$ variance가 복원됩니다. 이것은 random vector에 대한 $\|x\|_2\!\approx\!\sqrt d$의 배경에 있는 것과 동일한 additive-variance 사실입니다.
</div></details>

<details class="qa"><summary>ill-conditioning은 학습을 어떻게 늦추고, 무엇이 이를 고치는가?</summary>
<div class="qa-body">

**Short:** 큰 Hessian condition number $\kappa$는 타협적인 learning rate를 강요합니다 — 가파른 방향에는 너무 크고 평평한 방향에는 너무 작아서 — zig-zag와 느린 진행을 유발합니다.

**Deep:** vanilla gradient descent의 수렴 속도는 step마다 $(\kappa-1)/(\kappa+1)$처럼 나빠집니다. 해결책은 각각 다른 각도에서 $\kappa$를 공략합니다: **momentum**은 가파른 방향의 진동을 감쇠하고, **Adam**의 diagonal preconditioner는 각 좌표를 rescale하며, **normalization**(input, Batch/LayerNorm)은 loss surface를 isotropy 쪽으로 재구성하고, 진정한 **second-order** 방법은 $H^{-1}$을 곱해 landscape를 구형으로 만들지만 대규모에서는 너무 비쌉니다. [Optimization](#/foundations/optimization)을 참고하세요.
</div></details>

**예상해야 할 follow-up**

- *Hessian은 언제 symmetric인가?* mixed partial이 연속일 때(Clairaut/Schwarz).
- *JVP vs. VJP?* Forward-mode는 $Jv$($J$의 column)를, reverse-mode는 $v^\top J$($J$의 row)를 줍니다. Autodiff 라이브러리는 둘 다 노출합니다.
- *Hessian 없이 Hessian-vector product를 어떻게 계산하나?* Pearlmutter의 트릭: $\nabla f \cdot v$를 한 번 더 미분 — backward pass 한 번의 비용.
- *PCA vs. autoencoder?* Linear vs. nonlinear compression; MSE를 쓰는 linear AE는 PCA subspace를 복원합니다.
- *Conv는 matrix multiply인가?* 그렇습니다 — im2col이 이를 (Toeplitz 구조의) GEMM으로 바꾸고, self-attention은 softmax를 둘러싼 두 개의 GEMM입니다.

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
