# 선형대수 & 미적분 (딥러닝을 위한)

> [!NOTE] 이 챕터의 목표
> 겁먹지 마세요. 딥러닝에 필요한 수학은 **딱 두 가지 직관**으로 압축됩니다: (1) 숫자를 묶어서 한 번에 계산하는 법(**선형대수** — 벡터·행렬·행렬곱), (2) "이 숫자를 조금 바꾸면 결과가 얼마나 변하나"를 재는 법(**미분** — 기울기·연쇄법칙). 이 둘이 [머신러닝이란?](#/foundations/what-is-ml)의 "기울기(gradient)로 파라미터를 갱신"을 실제로 가능하게 만듭니다. 앞부분(§0~§2)은 완전 입문용, 뒤부분은 면접용 심화입니다.

## §0 · 왜 벡터와 행렬인가

[신경망 첫걸음](#/foundations/neural-networks-basics)에서 뉴런 하나는 $z = w_1x_1 + w_2x_2 + \dots$ 였습니다. 뉴런이 수천 개면 이걸 하나하나 쓸 수 없습니다. 그래서 숫자를 묶습니다.

- **스칼라(scalar)**: 숫자 하나. `3.14`
- **벡터(vector)**: 숫자를 한 줄로. `[2, 5, 1]` — 예: 한 사람의 (키, 몸무게, 나이)
- **행렬(matrix)**: 숫자를 표(격자)로. 예: 여러 사람의 데이터, 또는 한 층의 모든 가중치
- **텐서(tensor)**: 3차원 이상. 예: 이미지 배치 `(N, C, H, W)`

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="55" y="20" text-anchor="middle" fill="#98a3b2">스칼라</text>
  <rect x="35" y="35" width="40" height="40" rx="6" fill="#6366f1"/><text x="55" y="60" text-anchor="middle" fill="#fff">7</text>
  <text x="180" y="20" text-anchor="middle" fill="#98a3b2">벡터 (3,)</text>
  <g fill="none" stroke="#0ea5e9" stroke-width="1.6"><rect x="150" y="35" width="40" height="40" rx="6"/><rect x="150" y="75" width="40" height="40" rx="6"/><rect x="150" y="115" width="40" height="0" rx="6"/></g>
  <rect x="150" y="35" width="40" height="120" rx="6" fill="none" stroke="#0ea5e9" stroke-width="1.6"/>
  <line x1="150" y1="75" x2="190" y2="75" stroke="#0ea5e9"/><line x1="150" y1="115" x2="190" y2="115" stroke="#0ea5e9"/>
  <text x="170" y="60" text-anchor="middle" fill="currentColor">2</text><text x="170" y="100" text-anchor="middle" fill="currentColor">5</text><text x="170" y="140" text-anchor="middle" fill="currentColor">1</text>
  <text x="360" y="20" text-anchor="middle" fill="#98a3b2">행렬 (2×3)</text>
  <rect x="290" y="35" width="140" height="80" rx="6" fill="none" stroke="#e0533f" stroke-width="1.6"/>
  <line x1="290" y1="75" x2="430" y2="75" stroke="#e0533f"/><line x1="337" y1="35" x2="337" y2="115" stroke="#e0533f"/><line x1="384" y1="35" x2="384" y2="115" stroke="#e0533f"/>
  <text x="500" y="20" text-anchor="middle" fill="#98a3b2">텐서 (3D+)</text>
  <g fill="none" stroke="#12a150" stroke-width="1.4"><rect x="480" y="45" width="70" height="55" rx="4"/><rect x="490" y="38" width="70" height="55" rx="4"/><rect x="500" y="31" width="70" height="55" rx="4"/></g>
</svg>
<figcaption>차원이 하나씩 늘어나는 것뿐입니다. 딥러닝 코드의 버그 절반은 이 <b>shape(모양)</b>가 안 맞아서 생깁니다 — 항상 shape를 소리 내어 확인하세요.</figcaption>
</figure>

### 내적(dot product) — 가장 중요한 연산

두 벡터의 **내적**은 원소끼리 곱해서 다 더한 것입니다. 뉴런의 가중합이 바로 이것입니다:

$$
w \cdot x = \sum_i w_i x_i = w_1x_1 + w_2x_2 + \dots
$$

내적은 "두 벡터가 얼마나 같은 방향인가"를 재는 유사도이기도 합니다. **Attention의 $QK^\top$, 추천 시스템의 유사도, 뉴런의 가중합이 전부 내적**입니다.

### 행렬 곱(matmul) — 내적의 묶음

행렬 곱 $C = AB$는 "$A$의 각 행과 $B$의 각 열의 내적"을 격자로 채운 것입니다. 규칙은 하나: **안쪽 차원이 맞아야 함.** $(m \times k) \cdot (k \times n) = (m \times n)$.

<figure>
<svg viewBox="0 0 560 170" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="20" y="50" width="110" height="70" rx="6" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="75" y="40" text-anchor="middle" fill="#0ea5e9">A (m×k)</text>
  <rect x="45" y="55" width="100" height="18" fill="rgba(14,165,233,.25)"/><text x="150" y="68" fill="#0ea5e9">행</text>
  <text x="160" y="90" text-anchor="middle" font-size="18" fill="currentColor">×</text>
  <rect x="190" y="30" width="90" height="110" rx="6" fill="none" stroke="#e0533f" stroke-width="1.6"/><text x="235" y="20" text-anchor="middle" fill="#e0533f">B (k×n)</text>
  <rect x="205" y="35" width="18" height="100" fill="rgba(224,83,63,.25)"/><text x="235" y="155" fill="#e0533f">열</text>
  <text x="320" y="90" text-anchor="middle" font-size="18" fill="currentColor">=</text>
  <rect x="360" y="50" width="120" height="70" rx="6" fill="none" stroke="#12a150" stroke-width="1.8"/><text x="420" y="40" text-anchor="middle" fill="#12a150">C (m×n)</text>
  <circle cx="378" cy="68" r="7" fill="#12a150"/><text x="470" y="135" text-anchor="end" fill="#98a3b2">C[i,j] = (A의 i행)·(B의 j열)</text>
</svg>
<figcaption>초록 칸 하나 = 파란 행과 빨간 열의 내적. 신경망 한 층 = 입력 벡터에 가중치 행렬을 곱하는 것($Wx$)입니다.</figcaption>
</figure>

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"matmul","packages":["numpy"],"approx":true,"starter":"def matmul(A, B):\n    # A:(m,k), B:(k,n) 두 행렬의 곱 C:(m,n) 을 반환. C[i][j] = A의 i행 · B의 j열.\n    # 힌트: numpy 를 써도 되고(A @ B), 삼중 루프로 직접 구현해도 됩니다.\n    pass","tests":[{"args":[[[1,2],[3,4]],[[5,6],[7,8]]],"expect":[[19.0,22.0],[43.0,50.0]]},{"args":[[[1,0,2]],[[3],[4],[5]]],"expect":[[13.0]]},{"args":[[[2,0],[0,2]],[[1,2],[3,4]]],"expect":[[2.0,4.0],[6.0,8.0]]}],"solution":"import numpy as np\n\ndef matmul(A, B):\n    return (np.asarray(A, float) @ np.asarray(B, float)).tolist()"}
</script>
</div>

> [!TIP] 브로드캐스팅 한 줄 예고
> NumPy에서 shape가 다른 배열끼리 연산하면 자동으로 모양을 맞춰 줍니다(broadcasting). 예: `(N,1)` + `(1,M)` → `(N,M)`. [ML 코딩](#/ml-coding/intro)에서 이 트릭으로 반복문 없이 pairwise 계산을 합니다.

## §1 · 미분: "조금 바꾸면 얼마나 변하나"

**미분(derivative)** 은 기울기입니다. $\frac{df}{dx}$는 "$x$를 아주 조금 늘리면 $f$가 몇 배로 변하나"입니다. [머신러닝이란?](#/foundations/what-is-ml)에서 손실을 줄이려고 파라미터를 옮길 때, "어느 방향으로?"의 답이 바로 이 기울기(gradient)입니다.

<figure>
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <path d="M40 200 Q320 -40 600 200" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <!-- tangent at left (steep) -->
  <line x1="90" y1="185" x2="220" y2="90" stroke="#e0533f" stroke-width="2"/>
  <circle cx="140" cy="140" r="5" fill="#e0533f"/>
  <text x="150" y="135" fill="#e0533f">기울기 큼 → 크게 이동</text>
  <!-- tangent at bottom (flat) -->
  <line x1="270" y1="40" x2="370" y2="40" stroke="#12a150" stroke-width="2"/>
  <circle cx="320" cy="40" r="5" fill="#12a150"/>
  <text x="320" y="30" text-anchor="middle" fill="#12a150">기울기 ≈ 0 (최소점)</text>
  <text x="320" y="215" text-anchor="middle" fill="#98a3b2">손실 함수 위의 한 점에서, 기울기의 반대쪽(내리막)으로 내려갑니다</text>
</svg>
<figcaption>gradient는 "가장 가파른 오르막" 방향입니다. 그래서 손실을 <b>줄이려면</b> gradient의 반대 방향으로 움직입니다 — 이게 경사하강법(gradient descent)입니다.</figcaption>
</figure>

여러 입력이 있으면 각 입력에 대한 편미분을 모은 벡터가 **gradient** $\nabla f$입니다:

$$
\nabla f = \left[\frac{\partial f}{\partial x_1},\ \frac{\partial f}{\partial x_2},\ \dots\right]
$$

### 연쇄법칙(chain rule) — 딥러닝 학습의 심장

함수가 겹겹이 쌓이면($f(g(h(x)))$), 전체 기울기는 각 단계 기울기의 **곱**입니다:

$$
\frac{df}{dx} = \frac{df}{dg}\cdot\frac{dg}{dh}\cdot\frac{dh}{dx}
$$

신경망은 층이 겹겹이 쌓인 합성함수입니다. 그래서 출력 쪽 오차를 입력 쪽으로 "곱하며 거꾸로 흘려보내면" 모든 층의 gradient가 나옵니다 — 이것이 **역전파(backpropagation)** 입니다.

### 손으로 해보는 역전파 (작은 숫자로)

$x=2$, 함수 $L = (3x)^2$을 생각합시다. $g = 3x = 6$, $L = g^2 = 36$. 연쇄법칙으로:

$$
\frac{dL}{dx} = \underbrace{\frac{dL}{dg}}_{2g=12}\cdot\underbrace{\frac{dg}{dx}}_{3} = 12\times 3 = 36
$$

검산: 실제로 $L=(3x)^2=9x^2$, $dL/dx=18x=36$ ✓. **"뒤에서부터 기울기를 곱해 온다"** 는 이 감각이 backprop의 전부입니다.

## §2 · 신경망 backprop을 행렬로

이제 진짜입니다. dense layer 하나: $z = Wx + b$, $a=\phi(z)$, $L=L(a)$. 위쪽(출력)에서 내려온 gradient $\bar a = \partial L/\partial a$가 주어지면, 아래 네 줄로 backward가 끝납니다:

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
  <text x="255" y="42" text-anchor="middle" fill="#0ea5e9">순전파 →</text>
  <path d="M465 100 C 400 130, 260 130, 195 100" stroke="#12a150" stroke-width="1.5" fill="none" marker-end="url(#ga)"/>
  <text x="330" y="128" text-anchor="middle" fill="#12a150">← 역전파: 국소 Jᵀ 를 곱함 (VJP)</text>
  <defs>
    <marker id="fa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#0ea5e9"/></marker>
    <marker id="ga" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
  </defs>
</svg>
<figcaption>순전파는 activation을 저장하고, 역전파는 각 블록의 전치 Jacobian을 gradient에 곱합니다. 전체 Jacobian을 실제로 만들지는 않습니다 — 이걸 <b>vector–Jacobian product(VJP)</b>라 합니다.</figcaption>
</figure>

**shape로 검산하세요:** $\bar W$는 반드시 $W$와 같은 모양이어야 하고, $\bar z\,x^\top$가 정확히 그렇습니다. 면접에서도 "$\bar W$는 $W$와 같은 shape"라고 소리 내어 말하면 부호/전치 실수를 안 합니다.

전체 network는 이 VJP들을 합성한 것입니다:

$$
\frac{\partial L}{\partial x} = J_1^\top J_2^\top \cdots J_\ell^\top \frac{\partial L}{\partial y}
$$

**왜 reverse-mode(역방향)인가?** 손실은 스칼라 하나, 파라미터는 수십억 개입니다. 역방향으로 훑으면 $J$ 전체를 만들지 않고 VJP만 합성해 모든 파라미터의 gradient를 한 번에 얻습니다. 실제 비용은 연산과 구현에 따라 달라지지만 보통 순전파와 같은 차수입니다. 입력 차원은 작고 출력 차원이 큰 문제라면 반대로 forward-mode의 JVP가 유리할 수 있습니다.

### gradient를 코드로 검산하기 (numerical gradient)

수식이 맞는지 의심될 때, 미분 정의 그대로 $\frac{f(x+h)-f(x-h)}{2h}$로 근사해 비교하면 됩니다. 실무에서 backward 구현을 검증하는 표준 기법입니다. 아래는 $f(x)=x^3$의 기울기를 **미분 공식 없이** 수치로 구하는 함수입니다 — 정답은 $3x^2$과 거의 같아야 합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"grad_cube","packages":["numpy"],"approx":true,"starter":"def grad_cube(x, h=1e-5):\n    # f(x) = x**3 의 x 지점 도함수를 중심차분으로 근사: (f(x+h) - f(x-h)) / (2h)\n    # 결과는 3*x**2 과 거의 같아야 합니다 (미분 공식을 안 써도!).\n    pass","tests":[{"args":[2.0],"expect":12.0,"tol":1e-3},{"args":[3.0],"expect":27.0,"tol":1e-3},{"args":[0.0],"expect":0.0,"tol":1e-3}],"solution":"def grad_cube(x, h=1e-5):\n    f = lambda t: t ** 3\n    return (f(x + h) - f(x - h)) / (2 * h)"}
</script>
</div>

핵심 감각: **미분 공식을 몰라도 작은 문제의 기울기를 수치로 근사해 backward를 검산할 수 있다.** 다만 autograd가 유한차분을 자동화하는 것은 아닙니다. 프레임워크는 계산 그래프에 기록된 각 연산의 미분 규칙을 chain rule로 합성해 VJP/JVP를 계산하고, 유한차분은 그 결과를 독립적으로 확인하는 gradient-check 도구입니다.

## §3 · Eigen / SVD — 스펙트럼 관점 (심화)

임의의 행렬 $A\in\mathbb{R}^{m\times n}$는 이렇게 분해됩니다:

$$
A = U\Sigma V^\top,\qquad \sigma_1\ge\sigma_2\ge\cdots\ge 0
$$

$U,V$는 orthonormal(직교)입니다. 직관: **어떤 선형 변환이든 "회전 → 축별 늘이기 → 회전"으로 분해된다.** $\sigma_i$(singular value)가 각 축을 얼마나 늘이는지입니다.

<dl class="kv">
<dt>PCA (차원 축소)</dt><dd>centering한 데이터 $X=U\Sigma V^\top$: $V$의 column이 주성분 방향, projection은 $XV$, $\sigma_i^2/(n-1)$이 설명 분산.</dd>
<dt>Low-rank / LoRA</dt><dd>Eckart–Young은 <em>이미 주어진 행렬</em>의 최적 rank-$k$ 근사가 truncated SVD임을 말합니다. LoRA는 이 정리를 직접 적용하는 것이 아니라, 미세조정 중 업데이트를 $\Delta W=BA$로 매개화해 낮은 rank만 학습합니다. 따라서 truncated-SVD 최적성을 자동으로 보장하지 않습니다.</dd>
<dt>Pseudo-inverse</dt><dd>$A^+ = V\Sigma^+ U^\top$ — least-squares 해. 0인 singular value가 null space / rank 부족을 드러냅니다.</dd>
<dt>Optimizer geometry</dt><dd>matrix optimizer(Muon 계열, <a href="#/foundations/optimization">Optimization</a> 참고)는 weight update를 orthogonalization / 스펙트럼 관점으로 다룹니다.</dd>
</dl>

## §4 · Norm과 conditioning (심화)

$$
\|x\|_1=\textstyle\sum_i|x_i|,\quad \|x\|_2=\sqrt{\textstyle\sum_i x_i^2},\quad \|A\|_2=\sigma_{\max}(A)
$$

용도: weight decay($\ell_2$), gradient clipping(global $\ell_2$), Lipschitz/안정성(spectral norm). **condition number** $\kappa=\sigma_{\max}/\sigma_{\min}$는 최적화 난이도를 결정합니다. $\kappa$가 크면 한 방향은 가파르고 다른 방향은 평평해서, 어떤 learning rate를 써도 한쪽엔 틀립니다(지그재그). Input normalization, whitening, BatchNorm은 모두 이 **conditioning을 개선**해 학습을 빠르게 합니다 — 통계적 설명과는 별개인, 순수하게 최적화적인 이득입니다.

## 면접 Q&A

<details class="qa"><summary>Dense layer 하나를 통과하는 backprop을 행렬 연산으로 유도하라.</summary>
<div class="qa-body">

**Short:** $\bar a$가 주어지면 $\bar z=\bar a\odot\phi'(z)$, $\bar W=\bar z\,x^\top$, $\bar b=\bar z$, $\bar x=W^\top\bar z$.

**Deep:** 각각이 VJP입니다. 원소별 비선형성은 diagonal Jacobian이라 $\phi'(z)$와의 Hadamard product가 됩니다. 선형 사상 $z=Wx$는 $x$에 대한 Jacobian이 $W$($\Rightarrow W^\top\bar z$), $W$에 대해서는 outer-product 구조($\Rightarrow \bar z\,x^\top$, 예제마다 rank-1이고 batch에 대해 합산). shape 검산: $\bar W$는 $W$와 일치해야 하고 $\bar z\,x^\top$가 그렇습니다.
</div></details>

<details class="qa"><summary>softmax + cross-entropy의 gradient가 왜 그렇게 깔끔한가?</summary>
<div class="qa-body">

**Short:** 둘의 Jacobian이 상쇄되어 $\partial L/\partial z = p - y$ (예측 확률 − 정답)로 붕괴합니다.

**Deep:** softmax와 cross-entropy의 Jacobian이 정확히 상쇄되어 $p-y$로 붕괴하며, 그래서 프레임워크가 `log_softmax + nll_loss`를 하나의 `cross_entropy` op으로 fuse합니다(더 안정적이고 더 간단). 단계별 전체 유도는 canonical owner인 [손실 & gradient](#/ml-coding/losses-gradients) 참고.
</div></details>

<details class="qa"><summary>SVD란 무엇이고 ML 어디에 등장하는가?</summary>
<div class="qa-body">

**Short:** $A=U\Sigma V^\top$ — 임의의 행렬에 대해 정의되는 "회전·늘이기·회전" 분해.

**Deep:** PCA(공분산의 eigen-분해 = centering한 데이터의 SVD), 최적 low-rank 근사(Eckart–Young → truncated SVD, LoRA, 모델 압축), least squares용 pseudo-inverse, Jacobian/Hessian의 스펙트럼 분석의 기반. 0인 singular value는 rank 부족 / 모델이 구분 못 하는 null space 방향을 뜻합니다.
</div></details>

<details class="qa"><summary>attention에서 왜 $1/\sqrt{d}$ 스케일링인가?</summary>
<div class="qa-body">

**Short:** dot-product logit을 unit variance로 유지해 softmax가 saturate(포화)되지 않게 합니다.

**Deep:** $q,k$가 i.i.d. zero-mean unit-variance면 $q^\top k$의 variance가 $d$라서, $\sqrt d$로 나눠 $\mathcal{O}(1)$로 되돌립니다(안 하면 logit이 커져 softmax가 saturate되고 gradient가 죽음). 전체 유도와 코드는 [Attention (밑바닥)](#/ml-coding/attention) 참고.
</div></details>

**예상 follow-up**

- *JVP vs VJP?* forward-mode는 $Jv$(열), reverse-mode는 $v^\top J$(행). autodiff 라이브러리는 둘 다 제공.
- *Hessian 없이 Hessian-vector product?* Pearlmutter 트릭: $\nabla f\cdot v$를 한 번 더 미분 — backward 한 번 비용.
- *Conv는 행렬곱인가?* 예 — im2col이 GEMM으로 바꾸고, self-attention은 softmax를 감싼 두 GEMM.
- *backprop의 진짜 병목?* compute가 아니라 activation memory(캐시) — gradient checkpointing/FSDP가 그래서 존재. [분산 학습](#/foundations/distributed-training).

## Cheat-sheet

| Fact | 한 줄 |
| --- | --- |
| 내적 | $w\cdot x=\sum w_ix_i$ — 뉴런의 가중합이자 유사도 |
| 행렬곱 | $C[i,j]$ = A의 i행 · B의 j열; 안쪽 차원 일치 |
| gradient | 가장 가파른 오르막 방향; 손실은 반대로 내려감 |
| 연쇄법칙 | 합성함수의 기울기 = 단계별 기울기의 곱 → 역전파 |
| dense backward | $\bar z=\bar a\odot\phi'(z)$, $\bar W=\bar z x^\top$, $\bar x=W^\top\bar z$ |
| reverse-mode | 스칼라 손실 ⇒ 모든 파라미터 gradient를 보통 순전파와 같은 계산 차수로; 상수 비용·메모리는 graph와 kernel에 의존 |
| softmax+CE | gradient = $p-y$ (두 Jacobian 상쇄) |
| SVD | $A=U\Sigma V^\top$; 상위 $k$ = 최적 low-rank (Eckart–Young) |
| conditioning | 큰 $\kappa$ ⇒ 느린 지그재그; normalization/preconditioning이 도움 |

**다음:** [Optimization](#/foundations/optimization) · [확률 & 통계](#/foundations/probability-statistics) · [신경망 아키텍처](#/foundations/architectures) · [Attention 직접 구현](#/ml-coding/attention)
