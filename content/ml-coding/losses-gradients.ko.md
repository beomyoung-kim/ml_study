# Loss & Gradient (밑바닥부터)

> [!NOTE] 이 챕터의 목표
> [머신러닝이란?](#/foundations/what-is-ml)에서 "손실(loss)을 줄이는 방향으로 파라미터를 갱신한다"고 했습니다. 이 챕터는 그 **손실을 직접 코드로 만들고**, 손실의 **gradient(기울기)를 손으로 구해** 봅니다. 이게 backprop(역전파)의 실체입니다. 직관 → 짧은 수식 → 실행 가능한 NumPy 랩 순서로, 겁먹지 않고 갑니다.

> [!TIP] 면접 한 줄
> "이 영역 전체를 여는 하나의 사실: logit에 대한 softmax-cross-entropy의 gradient는 그냥 $p - \text{onehot}(y)$ — **예측 확률 빼기 정답**입니다. backward pass, focal loss, label smoothing은 전부 이것의 변형입니다." 이렇게 말하고, 물어보면 유도하세요.

## §0 · 손실(loss)이 뭐고 왜 gradient가 필요한가

**손실**은 "예측이 정답에서 얼마나 틀렸는지"를 하나의 숫자로 만든 것입니다. 작을수록 좋습니다. 학습은 이 숫자를 줄이는 게 전부이고, 줄이려면 "각 파라미터를 어느 방향으로 움직여야 손실이 줄까?"를 알아야 합니다 — 그 방향이 바로 **gradient(기울기)** 입니다.

그래서 손실 함수는 두 가지를 줘야 합니다: **① 값**(얼마나 틀렸나)과 **② gradient**(어떻게 고칠까). 이 챕터의 모든 함수가 이 둘을 반환합니다.

- **회귀(regression)** — 숫자를 맞힘(집값, 좌표): **MSE**
- **이진/멀티라벨 분류** — 독립적인 yes/no: **BCE**
- **다중 클래스 분류** — 상호 배타적 한 개 선택: **cross-entropy(교차 엔트로피)**

| Loss | 한 샘플 공식 | 언제 |
| --- | --- | --- |
| MSE | $\tfrac12(\hat y - y)^2$ | regression |
| BCE | $-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ | binary / multi-label |
| Cross-entropy | $-\log p_y,\ \ p=\text{softmax}(z)$ | multi-class |
| Focal | $-\alpha_t(1-p_t)^\gamma\log p_t$ | 클래스 불균형 detection |

## §1 · softmax와 cross-entropy를 직관으로

분류 모델의 마지막 층은 클래스마다 **점수(logit, 로짓)** 를 하나씩 냅니다. 이건 확률이 아닙니다 — 음수일 수도, 아주 클 수도 있습니다. **softmax** 가 이 점수들을 **합이 1인 확률**로 바꿉니다: 큰 점수는 큰 확률로, 지수함수로 강조하면서.

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

<figure>
<svg viewBox="0 0 600 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- logits -->
  <text x="95" y="20" text-anchor="middle" fill="#98a3b2">logit z (점수)</text>
  <g fill="#6366f1">
    <rect x="40"  y="120" width="34" height="40"/><text x="57"  y="176" text-anchor="middle" fill="currentColor">1</text>
    <rect x="90"  y="90"  width="34" height="70"/><text x="107" y="176" text-anchor="middle" fill="currentColor">2</text>
    <rect x="140" y="55"  width="34" height="105"/><text x="157" y="176" text-anchor="middle" fill="currentColor">3</text>
  </g>
  <!-- arrow -->
  <text x="300" y="95" text-anchor="middle" fill="#e0533f" font-weight="700">softmax</text>
  <path d="M215 100 H385" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#sm)"/>
  <text x="300" y="120" text-anchor="middle" fill="#98a3b2" font-size="10">지수화 후 합=1 로 정규화</text>
  <!-- probs -->
  <text x="500" y="20" text-anchor="middle" fill="#98a3b2">확률 p (합=1.0)</text>
  <g fill="#12a150">
    <rect x="430" y="150" width="34" height="10"/><text x="447" y="176" text-anchor="middle" fill="currentColor">.09</text>
    <rect x="480" y="122" width="34" height="38"/><text x="497" y="176" text-anchor="middle" fill="currentColor">.24</text>
    <rect x="530" y="55"  width="34" height="105"/><text x="547" y="176" text-anchor="middle" fill="currentColor">.67</text>
  </g>
  <defs><marker id="sm" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>logit $[1,2,3]$ → softmax → 확률 $[0.09, 0.24, 0.67]$. 점수 차이를 지수로 벌려 "가장 큰 것"에 확률을 몰아주되, 나머지도 0은 아닙니다(soft-max).</figcaption>
</figure>

이제 **cross-entropy(교차 엔트로피)** 는 아주 단순합니다: **정답 클래스에 모델이 준 확률에 $-\log$를 씌운 것**. 정답에 0.9를 줬으면 손실 $-\log 0.9\approx 0.1$(작음, 좋음), 0.1밖에 안 줬으면 $-\log 0.1\approx 2.3$(큼, 나쁨).

$$
L = -\log p_y
$$

> [!NOTE] $-\log$ 가 하는 일
> 정답 확률이 1에 가까우면 손실 ≈ 0, 0에 가까우면 손실 → ∞. 즉 **"확신을 갖고 틀린 것"을 아주 강하게 벌합니다.** 이 성질이 아래 "왜 MSE가 아닌가"의 핵심입니다.

## §2 · 왜 cross-entropy인가 (CE vs KL · BCE vs CE · 왜 L1/L2 아닌가)

### Cross-entropy vs KL divergence
참 분포 $p$와 예측 분포 $q$에 대해 $D_{\mathrm{KL}}(p\Vert q)=H(p,q)-H(p)$입니다. 지도 학습에서는 label $p$가 고정이라 $H(p)$가 상수(one-hot이면 0)이므로 — **CE 최소화 = $D_{\mathrm{KL}}(p\Vert q)$ 최소화**, 즉 예측을 정답 쪽으로 당기는 것입니다. target이 고정 label이면 CE를, **양쪽 모두 실제 분포**면(knowledge distillation·VAE·RLHF/PPO) KL을 직접 씁니다. KL의 비대칭성(forward = mode-covering, reverse = mode-seeking)과 상세한 용례는 [확률 & 통계](#/foundations/probability-statistics) 참고.

### BCE vs CE — 기능적으로 다름
- **BCE (출력별 sigmoid):** 각 출력이 **독립적인** yes/no. **multi-label**("cat"과 "outdoor"가 둘 다 참일 수 있음)에 사용 — $C$개의 독립 Bernoulli 문제이며, 확률의 합이 1일 **필요가 없습니다**.
- **CE (클래스에 대한 softmax):** **상호 배타적** 클래스에 대한 하나의 **categorical**. softmax가 logit들을 *결합*해서 확률의 합이 **1이 되게** 합니다. single-label multi-class에 사용.
- 2-클래스에서는 둘이 일치: **2-way softmax-CE ≡ 하나의 logit에 대한 BCE**. 요령: *상호 배타적 → softmax-CE; 독립/multi-label → BCE.*

### 분류에 왜 L1/L2 (MSE)를 쓰지 않나?
출력에 sigmoid/softmax를 붙이면 그냥 MSE로 regression하고 싶은 유혹이 생깁니다. 그것이 잘못된 도구인 세 가지 이유:

**1) 가장 크게 틀렸을 때 정확히 gradient가 사라집니다.** sigmoid + MSE에서는 $L=\tfrac12(\sigma(z)-y)^2$이고

$$
\frac{\partial L}{\partial z}=(\sigma(z)-y)\,\sigma'(z),\qquad \sigma'(z)=\sigma(z)(1-\sigma(z))
$$

정답이 $y=1$인데 모델이 $z=-5$($\sigma\approx0.007$)라고 하면, $\sigma'\approx0.007$이므로 gradient는 $\approx(-0.993)(0.007)\approx-0.007$ — 확신을 갖고 틀렸는데도 **아주 작습니다**. sigmoid + **BCE**(또는 softmax + CE)에서는 $\sigma'$이 **상쇄**됩니다:

$$
\frac{\partial L_{\text{BCE}}}{\partial z}=\sigma(z)-y\ \ (\text{여기서}\approx-0.993)
$$

오차에 비례하는 **큰** 신호입니다. CE는 실수로부터 빠르게 학습하고, MSE-on-sigmoid는 정체됩니다.

**2) 확신에 찬 오답에 큰 penalty.** $p_y\to0$일 때 $-\log p_y\to\infty$이므로, CE는 정답에 거의 0의 확률을 준 예측을 강하게 벌합니다. 모집단·모델이 이상적이면 CE는 proper scoring rule이지만, 유한 데이터와 distribution shift에서 신경망 확률이 자동으로 calibration되는 것은 아닙니다. 필요하면 ECE와 temperature scaling을 별도로 평가합니다.

<figure>
<svg viewBox="0 0 360 226" font-family="Inter, sans-serif" font-size="11">
  <line x1="40" y1="200" x2="335" y2="200" stroke="#98a3b2"/>
  <line x1="40" y1="200" x2="40" y2="24" stroke="#98a3b2"/>
  <polyline points="54,30 69,70 98,109 127,132 185,161 243,180 330,200" fill="none" stroke="#e0533f" stroke-width="2.4"/>
  <polyline points="40,143 98,164 156,180 214,191 272,198 330,200" fill="none" stroke="#6366f1" stroke-width="2.4"/>
  <text x="60" y="42" fill="#f4917f" font-weight="700">CE = −log p</text>
  <text x="150" y="140" fill="#a5b4fc" font-weight="700">MSE = (1−p)²</text>
  <text x="188" y="218" text-anchor="middle" fill="#98a3b2">predicted prob of true class p →</text>
  <text x="20" y="115" fill="#98a3b2" transform="rotate(-90 20,115)">loss</text>
  <text x="46" y="20" fill="#6b7686" font-size="9.5">→∞</text>
</svg>
<figcaption><b>정답</b> 클래스에 부여된 확률 대비 loss. CE는 $p\to0$일 때 폭발하고(확신에 찬 오답으로부터 크게 밀어냄), MSE는 평평해지며(약한 신호) 그 sigmoid gradient는 더 작아집니다.</figcaption>
</figure>

**3) 올바른 likelihood / convexity.** CE는 Bernoulli/Categorical 모델 하에서의 negative log-likelihood — 클래스 label에 대한 *올바른* 확률적 loss입니다. MSE는 **Gaussian** likelihood에 대응합니다(regression엔 맞고 category엔 틀림). 또한 CE는 logit에 대해 convex인 반면, sigmoid를 거친 MSE는 non-convex이고 나쁜 local minima를 가집니다. **L1/L2가 *맞는* 경우:** regression / 연속 target, 그리고 bounding-box 좌표 regression(smooth-L1 / Huber).

## §3 · softmax-CE gradient 유도 — 왜 $p - y$ 인가

logit $z$에 대해 $p_i = e^{z_i}/\sum_j e^{z_j}$, loss $L = -\log p_y$. softmax의 Jacobian(야코비안, 편미분 행렬)은 $\partial p_i/\partial z_k = p_i(\delta_{ik} - p_k)$. 이를 $L$의 미분에 넣으면 두 항이 상쇄됩니다:

$$
\frac{\partial L}{\partial z_k} = -\frac{1}{p_y}\frac{\partial p_y}{\partial z_k}
= -\frac{1}{p_y}\,p_y(\delta_{yk} - p_k) = p_k - \delta_{yk}
$$

따라서 $\boxed{\nabla_z L = p - \text{onehot}(y)}$입니다. 프레임워크가 softmax와 CE를 logit 기반 연산 하나로 제공하는 주된 이유는 log-sum-exp로 수치 안정성을 확보하고 중간 확률 저장·kernel launch를 줄일 수 있기 때문입니다. (같은 미분은 [선형대수 & 미적분](#/foundations/linear-algebra-calculus)에도 나옵니다.)

## §4 · Cross-entropy (안정 버전)와 gradient

```python
import numpy as np


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)     # stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(logits, targets, reduction="mean"):
    """logits:(N,C)  targets:(N,) int -> (loss, grad wrt logits (N,C))."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("expected logits (N,C) and targets (N,)")
    N = logits.shape[0]
    if N == 0 or logits.shape[1] == 0:
        raise ValueError("empty batch/classes are undefined")
    if np.any((targets < 0) | (targets >= logits.shape[1])):
        raise ValueError("target index out of range")
    m = logits.max(axis=1, keepdims=True)
    z = logits - m
    logsumexp = m[:, 0] + np.log(np.exp(z).sum(axis=1))
    loss = logsumexp - logits[np.arange(N), targets]
    p = np.exp(logits - logsumexp[:, None])
    grad = p.copy()
    grad[np.arange(N), targets] -= 1.0              # p - onehot(y)
    if reduction == "mean":
        loss, grad = loss.mean(), grad / N
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
    return float(loss), grad
```

## §5 · BCE, MSE, focal — 모두 수치적으로 안정

```python
def mse(pred, y):
    diff = pred - y
    return 0.5 * np.mean(diff ** 2), diff / y.size   # loss, grad wrt pred


def bce_with_logits(z, y):
    """Stable BCE from logits. grad wrt z is sigmoid(z) - y."""
    # -log(1-sigmoid) = softplus(z); log(sigmoid) = z - softplus(z)
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))  # LSE-stable
    sig = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    sig[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    sig[~pos] = ez / (1.0 + ez)
    grad = sig - y
    return float(loss.mean()), grad / z.size


def binary_focal_loss(z, y, gamma=2.0, alpha=0.25):
    """Focal loss from logits (RetinaNet). Stable via logsigmoid."""
    z, y = z.reshape(-1).astype(np.float64), y.reshape(-1).astype(np.float64)
    softplus = np.logaddexp(0.0, z)                 # log(1+e^z)
    log_p, log_1mp = z - softplus, -softplus        # logsigmoid, log(1-sigmoid)
    p = np.exp(log_p)
    p_t = np.where(y == 1, p, 1 - p)
    log_pt = np.where(y == 1, log_p, log_1mp)
    a_t = np.where(y == 1, alpha, 1 - alpha)
    return float((-a_t * (1 - p_t) ** gamma * log_pt).mean())
```

> [!WARNING] log과 exp를 순진하게 합성하지 마세요
> `np.log(sigmoid(z))`는 $|z|$가 크면 overflow합니다. `logaddexp`, `log1p`, `softplus`, 그리고 항등식 $\log\sigma(z)=z-\text{softplus}(z)$를 사용하세요. 이것이 면접관이 loss 코드에서 듣고 싶어하는 수치 안정성 신호입니다.

## §6 · Autograd 없는 backward: linear layer

backprop의 핵심입니다. $Y = XW + b$ ($X\in\mathbb{R}^{N\times D_\text{in}}$, $W\in\mathbb{R}^{D_\text{in}\times D_\text{out}}$)에서, 위에서 내려온 gradient $\partial L/\partial Y$(`dY`, shape $(N,D_\text{out})$)가 주어지면:

$$
\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y},\quad
\frac{\partial L}{\partial b} = \sum_n \frac{\partial L}{\partial Y_n},\quad
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}\, W^\top
$$

<figure>
<svg viewBox="0 0 620 130" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <rect x="30" y="45" width="70" height="34" rx="6" fill="#6366f1"/><text x="65" y="67" text-anchor="middle" fill="#fff">X (N,Dᵢ)</text>
  <rect x="180" y="45" width="90" height="34" rx="6" fill="none" stroke="#98a3b2" stroke-width="1.5"/><text x="225" y="67" text-anchor="middle" fill="currentColor">Y=XW+b</text>
  <rect x="360" y="45" width="70" height="34" rx="6" fill="#e0533f"/><text x="395" y="67" text-anchor="middle" fill="#fff">L</text>
  <path d="M100 62 H180 M270 62 H360" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#f)"/>
  <text x="140" y="38" text-anchor="middle" fill="#0ea5e9">순전파 →</text>
  <path d="M395 92 C 320 118, 180 118, 120 92" stroke="#12a150" stroke-width="1.5" fill="none" marker-end="url(#g)"/>
  <text x="255" y="115" text-anchor="middle" fill="#12a150">← 역전파: dW=Xᵀ·dY, dX=dY·Wᵀ</text>
  <defs>
    <marker id="f" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#0ea5e9"/></marker>
    <marker id="g" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
  </defs>
</svg>
<figcaption>순전파는 입력 X를 캐시하고, 역전파는 dY에 전치 행렬을 곱해 dW·dX를 만듭니다. shape가 곧 정답표입니다.</figcaption>
</figure>

```python
class Linear:
    def __init__(self, d_in, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d_in, d_out)) * (1 / np.sqrt(d_in))
        self.b = np.zeros(d_out)

    def forward(self, x):
        self.x = x                        # cache input for backward
        return x @ self.W + self.b        # (N, d_out)

    def backward(self, dY):
        self.dW = self.x.T @ dY           # (d_in, d_out)  — check shapes!
        self.db = dY.sum(0)               # (d_out,)
        return dY @ self.W.T              # dX (N, d_in) to pass downstream
```

**Shape 확인이 곧 디버깅 도구:** `dW`는 `W`와 일치해야 하므로 `X.T @ dY`일 수밖에 없습니다. 세 규칙 — *선형 사상엔 matmul, 다른 피연산자를 transpose, bias엔 batch 축을 따라 sum* — 이 모든 layer에 일반화됩니다.

## §7 · Gradient check (시키지 않아도 항상 하세요)

해석적으로 구한 gradient가 맞는지, 정의 그대로의 수치 미분과 비교해 검산합니다.

```python
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4, 3)); t = np.array([0, 1, 2, 1])
    loss, grad = cross_entropy(z, t)

    eps, num = 1e-5, np.zeros_like(z)     # central finite differences
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            e = np.zeros_like(z); e[i, j] = eps
            num[i, j] = (cross_entropy(z + e, t)[0]
                         - cross_entropy(z - e, t)[0]) / (2 * eps)
    assert np.allclose(grad, num, atol=1e-4)     # analytic == numeric
    print("CE grad check passed; loss =", round(loss, 4))
```

> [!NOTE] 프레임워크 한 줄
> `F.cross_entropy(logits, targets)`(log-softmax + NLL을 fuse, 확률을 결코 materialize하지 않음), `F.binary_cross_entropy_with_logits`, `F.mse_loss`; focal은 `torchvision.ops.sigmoid_focal_loss`.

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **NumPy가 준비된 라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

elementwise activation부터, 그다음 loss들, 마지막으로 이들을 잇는 softmax-CE gradient 순서로 내려가며 풀어보세요.

### 1. Sigmoid <span class="badge badge-easy">Easy</span>

실수 logit을 elementwise로 $(0,1)$ 확률로 매핑합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sigmoid","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef sigmoid(z):\n    # stable elementwise sigmoid; split non-negative and negative inputs to avoid exp overflow\n    pass","tests":[{"args":[[0.0,2.0,-2.0]],"expect":[0.5,0.8807970779778823,0.11920292202211755]},{"args":[[[0.0,1.0],[-1.0,3.0]]],"expect":[[0.5,0.7310585786300049],[0.2689414213699951,0.9525741268224334]]},{"args":[[-1000.0,1000.0]],"expect":[0.0,1.0]}],"solution":"import numpy as np\n\ndef sigmoid(z):\n    z = np.asarray(z, dtype=float)\n    out = np.empty_like(z)\n    pos = z >= 0\n    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))\n    ez = np.exp(z[~pos])\n    out[~pos] = ez / (1.0 + ez)\n    return out"}
</script>
</div>

### 2. Stable Softmax <span class="badge badge-easy">Easy</span>

큰 logit이 overflow하지 않도록 지수화 전에 행별 최댓값을 뺍니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(z, axis=-1):\n    # subtract the max along axis for stability, exponentiate, normalize\n    pass","tests":[{"args":[[[1.0,2.0,3.0]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[0.0,0.0]]],"expect":[[0.5,0.5]]},{"args":[[[1000.0,1001.0,1002.0]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]}],"solution":"import numpy as np\n\ndef softmax(z, axis=-1):\n    z = np.asarray(z, dtype=float)\n    z = z - np.max(z, axis=axis, keepdims=True)\n    e = np.exp(z)\n    return e / e.sum(axis=axis, keepdims=True)"}
</script>
</div>

### 3. Softmax Cross-Entropy Loss <span class="badge badge-med">Medium</span>

logit 배치에 대한 정답 클래스의 평균 negative log-likelihood.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"cross_entropy_loss","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef cross_entropy_loss(logits, targets):\n    # logits:(N,C), targets:(N,); compute mean logsumexp(logits)-true_logit without clipping probabilities\n    pass","tests":[{"args":[[[1.0,2.0,3.0],[1.0,1.0,1.0]],[2,0]],"expect":0.7531091265562451},{"args":[[[0.0,0.0],[0.0,0.0]],[0,1]],"expect":0.6931471805599453},{"args":[[[2.0,1.0,0.0]],[0]],"expect":0.40760596444438046},{"args":[[[-1000.0,0.0]],[0]],"expect":1000.0}],"solution":"import numpy as np\n\ndef cross_entropy_loss(logits, targets):\n    logits = np.asarray(logits, dtype=float)\n    targets = np.asarray(targets, dtype=np.int64)\n    if logits.ndim != 2 or targets.shape != (logits.shape[0],) or logits.shape[0] == 0 or logits.shape[1] == 0:\n        raise ValueError('expected non-empty logits (N,C) and targets (N,)')\n    if np.any((targets < 0) | (targets >= logits.shape[1])):\n        raise ValueError('target index out of range')\n    m = logits.max(axis=1)\n    lse = m + np.log(np.exp(logits - m[:, None]).sum(axis=1))\n    return float(np.mean(lse - logits[np.arange(len(targets)), targets]))"}
</script>
</div>

### 4. Softmax-CE Gradient <span class="badge badge-med">Medium</span>

그 유명한 $p - \text{onehot}(y)$를 batch 크기로 mean-reduce한 값.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax_ce_grad","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax_ce_grad(logits, targets):\n    # gradient wrt logits: softmax(logits) minus one-hot(targets), divided by N\n    pass","tests":[{"args":[[[1.0,2.0,3.0],[1.0,1.0,1.0]],[2,0]],"expect":[[0.04501528658519023,0.12236423552739882,-0.1673795221125891],[-0.33333333333333337,0.16666666666666666,0.16666666666666666]]},{"args":[[[0.0,0.0],[0.0,0.0]],[0,1]],"expect":[[-0.25,0.25],[0.25,-0.25]]},{"args":[[[2.0,1.0,0.0]],[0]],"expect":[[-0.3347590442251782,0.24472847105479764,0.09003057317038046]]}],"solution":"import numpy as np\n\ndef softmax_ce_grad(logits, targets):\n    logits = np.asarray(logits, dtype=float)\n    targets = np.asarray(targets)\n    N = logits.shape[0]\n    z = logits - logits.max(axis=1, keepdims=True)\n    p = np.exp(z)\n    p /= p.sum(axis=1, keepdims=True)\n    grad = p.copy()\n    grad[np.arange(N), targets] -= 1.0\n    return grad / N"}
</script>
</div>

### 5. Mean Squared Error <span class="badge badge-easy">Easy</span>

평균 제곱 차이의 절반 — $\tfrac12$ 덕분에 gradient가 $\hat y - y$가 됩니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"mse","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef mse(pred, y):\n    # 0.5 * mean((pred - y)^2)\n    pass","tests":[{"args":[[1.0,2.0,3.0],[1.0,2.0,3.0]],"expect":0.0},{"args":[[2.0,0.0,2.0],[0.0,0.0,0.0]],"expect":1.3333333333333333},{"args":[[[1.0,2.0],[3.0,4.0]],[[0.0,0.0],[0.0,0.0]]],"expect":3.75}],"solution":"import numpy as np\n\ndef mse(pred, y):\n    pred = np.asarray(pred, dtype=float)\n    y = np.asarray(y, dtype=float)\n    diff = pred - y\n    return float(0.5 * np.mean(diff ** 2))"}
</script>
</div>

### 6. BCE With Logits (stable) <span class="badge badge-med">Medium</span>

logit에서 바로 계산하는 수치적으로 안정된 binary cross-entropy.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"bce_with_logits","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef bce_with_logits(z, y):\n    # stable: max(z,0) - z*y + log1p(exp(-|z|)), then mean\n    pass","tests":[{"args":[[0.0,0.0,0.0],[1.0,0.0,1.0]],"expect":0.6931471805599453},{"args":[[10.0,-10.0],[1.0,0.0]],"expect":4.539889921686465e-05},{"args":[[2.0,-1.0,0.5],[1.0,1.0,0.0]],"expect":0.8047555609137674}],"solution":"import numpy as np\n\ndef bce_with_logits(z, y):\n    z = np.asarray(z, dtype=float)\n    y = np.asarray(y, dtype=float)\n    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))\n    return float(loss.mean())"}
</script>
</div>

### 7. Binary Focal Loss <span class="badge badge-med">Medium</span>

$(1-p_t)^\gamma$로 쉬운 예제를 down-weight — logsigmoid로 안정화.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"binary_focal_loss","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef binary_focal_loss(z, y, gamma=2.0, alpha=0.25):\n    # logsigmoid via softplus; p_t, log_pt, alpha_t by class; mean of -a_t*(1-p_t)^gamma*log_pt\n    pass","tests":[{"args":[[2.0,-1.0,0.5],[1.0,0.0,0.0]],"expect":0.10016771149490598},{"args":[[0.0,0.0],[1.0,0.0]],"expect":0.08664339756999316},{"args":[[3.0,-3.0,1.0,-1.0],[1.0,0.0,1.0,0.0],1.0,0.5],"expect":0.02163833526892106}],"solution":"import numpy as np\n\ndef binary_focal_loss(z, y, gamma=2.0, alpha=0.25):\n    z = np.asarray(z, dtype=float).reshape(-1)\n    y = np.asarray(y, dtype=float).reshape(-1)\n    sp = np.logaddexp(0.0, z)\n    log_p, log_1mp = z - sp, -sp\n    p = np.exp(log_p)\n    p_t = np.where(y == 1, p, 1 - p)\n    log_pt = np.where(y == 1, log_p, log_1mp)\n    a_t = np.where(y == 1, alpha, 1 - alpha)\n    return float((-a_t * (1 - p_t) ** gamma * log_pt).mean())"}
</script>
</div>

## 면접관이 지켜보는 흔한 버그

- **`cross_entropy`는 확률이 아니라 logit을 기대** — softmax를 두 번 하지 마세요.
- **mean으로 reduce한 gradient에서 `1/N`을 잊음**.
- **`log(0)`** — log 전에 확률을 clamp.
- **BCE vs CE 혼동:** multi-label(독립 클래스)엔 BCE/sigmoid, 상호 배타적 클래스엔 softmax-CE.
- **`dW`의 shape:** `W.shape`와 같아야 하며, transpose 위치는 거기서 따라옵니다.

## Q&A

<details class="qa"><summary>왜 focal loss가 클래스 불균형에 도움이 되나?</summary>
<div class="qa-body">

**짧게:** $(1-p_t)^\gamma$ 조절 인자가 쉽고 잘 분류된 예제의 loss(및 gradient)를 줄여서 학습이 어려운 예제에 집중하게 합니다 — background box가 객체보다 1000:1로 많을 때 결정적입니다.

**깊게:** dense detection에서는 anchor의 절대 다수가 쉬운 negative입니다. 이들 각각의 작은 CE loss도 합산하면 여전히 gradient를 지배하여 드문 positive를 압도합니다. Focal은 confidence $p_t$인 예제를 $(1-p_t)^\gamma$로 down-weight합니다: $p_t=0.9$, $\gamma=2$인 쉬운 예제는 $0.01$로 스케일되고, $p_t=0.1$인 어려운 예제는 거의 건드려지지 않습니다($0.81$). $\alpha$는 추가로 positive/negative 클래스 prior를 rebalance합니다. 덕분에 RetinaNet은 hard-negative mining 없이 two-stage detector에 맞먹었습니다. [Object Detection](#/cv/detection) 참고.
</div></details>

<details class="qa"><summary>왜 softmax와 cross-entropy를 하나의 op로 fuse하나?</summary>
<div class="qa-body">

**짧게:** 안정성과 속도 — fuse된 log-softmax-후-NLL은 확률을 materialize하거나 `log(exp(...))`를 계산하는 일을 아예 피하고, gradient가 깔끔한 $p-\text{onehot}(y)$로 collapse합니다.

**깊게:** `softmax` 후 `log`를 따로 계산하면 under/overflow 위험이 있고 중복 작업을 합니다. $\log\text{-softmax}(z) = z - \text{LSE}(z)$는 max-subtraction을 통해 바로 안정적이고, NLL은 정답 클래스만 gather합니다. Backward에서도 softmax Jacobian을 log와 chaining하면 샘플당 $O(C^2)$지만, 해석적 단순화가 이를 $O(C)$로 만듭니다. 수치적으로도 계산적으로도 엄격히 낫기 때문에 `F.cross_entropy`가 logit을 받습니다.
</div></details>

<details class="qa"><summary>Dice loss와 CE — segmentation에서 왜 둘을 결합하나?</summary>
<div class="qa-body">

**짧게:** CE는 픽셀별 calibration과 안정된 gradient를 주고, Dice는 region overlap을 직접 최적화하며 foreground/background 불균형에 robust합니다. 상보적이라 segmentation 모델은 종종 둘을 합산합니다.

**깊게:** soft Dice는 $1 - \frac{2\sum p g}{\sum p + \sum g}$로, mask 전체에 대한 *non-decomposable*한 loss이며 객체 크기에 자연스럽게 scale-invariant합니다 — 픽셀별 CE와 달리 작은 객체가 큰 객체만큼 기여합니다. 하지만 예측이 0에 가까운 학습 초기에는 Dice gradient가 noisy하므로, CE가 최적화를 안정화하는 동안 Dice가 목적함수를 mIoU 지표에 정렬시킵니다. [Segmentation](#/cv/segmentation) 참고.
</div></details>

### Follow-ups
- **Label smoothing?** one-hot target을 $(1-\epsilon)\cdot\text{onehot} + \epsilon/C$로 대체하여 과확신을 줄이고 calibration을 개선.
- **Huber / smooth-L1?** 0 근처에서는 quadratic, 꼬리에서는 linear — robust한 box-regression loss(Fast R-CNN).
- **MSE에 왜 $\tfrac12$?** 미분에서 나오는 2를 상쇄하여 $\nabla = \hat y - y$가 되게 함; 순전히 미관용.
- **Class weight?** 불균형에 맞춰 각 클래스의 CE 항을 역빈도로 스케일.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Softmax-CE grad | $p - \text{onehot}(y)$ |
| Sigmoid-BCE grad | $\sigma(z) - y$ |
| MSE grad | $\hat y - y$ |
| Linear backward | $dW=X^\top dY$, $db=\sum dY$, $dX=dY\,W^\top$ |
| Shape rule | `dW.shape == W.shape` fixes the transpose |
| Stability | subtract max, `clip` before log, `logaddexp`/`softplus` |
| Focal | $(1-p_t)^\gamma$ down-weights easy examples |
| Grad check | central diff, `atol≈1e-4` |

**다음:** [Optimization](#/foundations/optimization) · [CNN · RNN · Transformer](#/foundations/architectures) · [Object Detection](#/cv/detection) · [Segmentation](#/cv/segmentation) · [ML 코딩 라운드](#/ml-coding/intro)
