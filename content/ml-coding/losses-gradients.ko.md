# Loss와 Gradient

> [!TIP] 이렇게 먼저 말하세요
> "이 영역 전체를 여는 하나의 사실: logits에 대한 softmax-cross-entropy의 gradient는 그냥 $p - \text{onehot}(y)$입니다. 예측 확률 빼기 정답. 나머지는 전부 — backward pass, focal loss, label smoothing — 그것의 변형입니다." 이걸 말하고, 물어보면 유도하세요.

MSE, BCE, cross-entropy, focal loss를 **수치적으로 안정된** 코드로 구현하고, softmax-CE gradient를 손으로 유도하고, linear layer에 대한 **autograd 없는 backward**를 작성합니다. 이 라운드는 수학을 코드로, 그리고 코드를 gradient로 바꾸는 양방향 기술을 봅니다 — backprop 이해의 핵심입니다([Optimization](#/foundations/optimization) 참고).

## Loss 한눈에

| Loss | Formula (per sample) | Use |
| --- | --- | --- |
| MSE | $\tfrac12(\hat y - y)^2$ | regression |
| BCE | $-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ | binary / multi-label |
| Cross-entropy | $-\log p_y,\ \ p=\text{softmax}(z)$ | multi-class |
| Focal | $-\alpha_t(1-p_t)^\gamma\log p_t$ | class-imbalanced detection |

## softmax-CE gradient 유도

logits $z$에 대해 $p_i = e^{z_i}/\sum_j e^{z_j}$, loss $L = -\log p_y$. softmax Jacobian은 $\partial p_i/\partial z_k = p_i(\delta_{ik} - p_k)$. 그러면:

$$
\frac{\partial L}{\partial z_k} = -\frac{1}{p_y}\frac{\partial p_y}{\partial z_k}
= -\frac{1}{p_y}\,p_y(\delta_{yk} - p_k) = p_k - \delta_{yk}
$$

따라서 $\boxed{\nabla_z L = p - \text{onehot}(y)}$ — 깔끔하고, 저렴하며, 모든 프레임워크가 logits+CE를 fuse하는 이유입니다. *(verifiable)*

## Cross-entropy (stable)와 gradient

```python
import numpy as np


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)     # stability
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy(logits, targets, reduction="mean"):
    """logits:(N,C)  targets:(N,) int -> (loss, grad wrt logits (N,C))."""
    N = logits.shape[0]
    p = softmax(logits, axis=1)
    py = p[np.arange(N), targets]                   # gather true-class probs
    loss = -np.log(np.clip(py, 1e-12, 1.0))         # clamp: avoid log(0)
    grad = p.copy()
    grad[np.arange(N), targets] -= 1.0              # p - onehot(y)
    if reduction == "mean":
        loss, grad = loss.mean(), grad / N
    else:
        loss = loss.sum()
    return float(loss), grad
```

## BCE, MSE, focal — 모두 수치적으로 안정

```python
def mse(pred, y):
    diff = pred - y
    return 0.5 * np.mean(diff ** 2), diff / y.size   # loss, grad wrt pred


def bce_with_logits(z, y):
    """Stable BCE from logits. grad wrt z is sigmoid(z) - y."""
    # -log(1-sigmoid) = softplus(z); log(sigmoid) = z - softplus(z)
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))  # LSE-stable
    grad = 1.0 / (1.0 + np.exp(-z)) - y
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

## Autograd 없는 backward: linear layer

backprop의 핵심. $Y = XW + b$, $X\in\mathbb{R}^{N\times D_\text{in}}$, $W\in\mathbb{R}^{D_\text{in}\times D_\text{out}}$에 대해, upstream $\partial L/\partial Y$(`dY`, shape $(N,D_\text{out})$)가 주어지면:

$$
\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y},\quad
\frac{\partial L}{\partial b} = \sum_n \frac{\partial L}{\partial Y_n},\quad
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}\, W^\top
$$

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

**Shape 확인이 곧 디버깅 도구:** `dW`는 `W`와 일치해야 하므로 `X.T @ dY`일 수밖에 없습니다. 세 가지 규칙 — *선형 사상에는 matmul, 다른 피연산자를 transpose, bias에는 broadcast된(batch) 축을 따라 sum* — 이 모든 layer에 일반화됩니다.

## Gradient check (시키지 않아도 항상 하세요)

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
> 아래 각 문제에는 **NumPy가 준비된 라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

elementwise activation부터, 그다음 loss들, 마지막으로 이들을 잇는 softmax-CE gradient 순서로 내려가며 풀어보세요.

### 1. Sigmoid <span class="badge badge-easy">Easy</span>

실수 logit을 elementwise로 $(0,1)$ 확률로 매핑합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sigmoid","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef sigmoid(z):\n    # elementwise 1/(1+e^-z); accept a list or nested list\n    pass","tests":[{"args":[[0.0,2.0,-2.0]],"expect":[0.5,0.8807970779778823,0.11920292202211755]},{"args":[[[0.0,1.0],[-1.0,3.0]]],"expect":[[0.5,0.7310585786300049],[0.2689414213699951,0.9525741268224334]]},{"args":[[0.0]],"expect":[0.5]}],"solution":"import numpy as np\n\ndef sigmoid(z):\n    z = np.asarray(z, dtype=float)\n    return 1.0 / (1.0 + np.exp(-z))"}
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
{"func":"cross_entropy_loss","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef cross_entropy_loss(logits, targets):\n    # logits:(N,C), targets:(N,) int; softmax rows, gather true-class prob, -log, mean\n    pass","tests":[{"args":[[[1.0,2.0,3.0],[1.0,1.0,1.0]],[2,0]],"expect":0.7531091265562451},{"args":[[[0.0,0.0],[0.0,0.0]],[0,1]],"expect":0.6931471805599453},{"args":[[[2.0,1.0,0.0]],[0]],"expect":0.40760596444438046}],"solution":"import numpy as np\n\ndef cross_entropy_loss(logits, targets):\n    logits = np.asarray(logits, dtype=float)\n    targets = np.asarray(targets)\n    N = logits.shape[0]\n    z = logits - logits.max(axis=1, keepdims=True)\n    p = np.exp(z)\n    p /= p.sum(axis=1, keepdims=True)\n    py = p[np.arange(N), targets]\n    return float(-np.log(np.clip(py, 1e-12, 1.0)).mean())"}
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

- **`cross_entropy`는 확률이 아니라 logits를 기대** — softmax를 두 번 하지 마세요.
- **mean으로 reduce한 gradient에서 `1/N`을 잊음**.
- **`log(0)`** — log 전에 확률을 clamp.
- **BCE vs CE 혼동:** multi-label(독립적인 클래스)에는 BCE/sigmoid, 상호 배타적 클래스에는 softmax-CE.
- **`dW`의 shape:** `W.shape`와 같아야 하며, transpose 위치는 거기서 따라옵니다.

## Q&A

<details class="qa"><summary>왜 focal loss가 클래스 불균형에 도움이 되나?</summary>
<div class="qa-body">

**짧게:** $(1-p_t)^\gamma$ 조절 인자가 쉽고 잘 분류된 예제의 loss(및 gradient)를 줄여서 학습이 어려운 예제에 집중하게 합니다 — background box가 객체보다 1000:1로 많을 때 결정적입니다.

**깊게:** dense detection에서는 anchor의 절대 다수가 쉬운 negative입니다. 이들 각각의 작은 CE loss도 합산하면 여전히 gradient를 지배하여 드문 positive를 압도합니다. Focal은 confidence $p_t$인 예제를 $(1-p_t)^\gamma$로 down-weight합니다: $p_t=0.9$, $\gamma=2$인 쉬운 예제는 $0.01$로 스케일되고, $p_t=0.1$인 어려운 예제는 거의 건드려지지 않습니다($0.81$). $\alpha$는 추가로 positive/negative 클래스 prior를 rebalance합니다. 덕분에 RetinaNet은 hard-negative mining 없이 two-stage detector에 맞먹었습니다.
</div></details>

<details class="qa"><summary>왜 softmax와 cross-entropy를 하나의 op로 fuse하나?</summary>
<div class="qa-body">

**짧게:** 안정성과 속도 — fuse된 log-softmax-후-NLL은 확률을 materialize하거나 `log(exp(...))`를 계산하는 일을 아예 피하고, gradient가 깔끔한 $p-\text{onehot}(y)$로 collapse합니다.

**깊게:** `softmax` 후 `log`를 따로 계산하면 under/overflow 위험이 있고 중복 작업을 합니다. `log_softmax(z) = z - \text{LSE}(z)$는 max-subtraction을 통해 바로 안정적이고, NLL은 정답 클래스만 gather합니다. Backward 쪽에서는 softmax의 Jacobian을 log와 chaining하면 샘플당 $O(C^2)$ product가 되지만, 해석적 단순화가 이를 $O(C)$로 만듭니다. 수치적으로도 계산적으로도 엄격히 낫기 때문에 `F.cross_entropy`가 logits를 받는 것입니다.
</div></details>

<details class="qa"><summary>Dice loss와 CE — segmentation에서 왜 둘을 결합하나?</summary>
<div class="qa-body">

**짧게:** CE는 픽셀별 calibration과 안정된 gradient를 주고, Dice는 region overlap을 직접 최적화하며 foreground/background 불균형에 robust합니다. 상보적이라 segmentation 모델은 종종 둘을 합산합니다.

**깊게:** soft Dice는 $1 - \frac{2\sum p g}{\sum p + \sum g}$로, mask 전체에 대한 *non-decomposable*한 loss이며, 이 때문에 객체 크기에 자연스럽게 scale-invariant합니다 — 픽셀별 CE와 달리 작은 객체가 큰 객체만큼 기여합니다. 하지만 예측이 0에 가까운 학습 초기에는 Dice gradient가 noisy하므로, CE가 최적화를 안정화하는 동안 Dice가 목적함수를 mIoU 지표에 정렬시킵니다. [Segmentation](#/cv/segmentation) 참고.
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

**Cross-links:** [Optimization](#/foundations/optimization) · [Object Detection](#/cv/detection) · [Segmentation](#/cv/segmentation) · [Evaluation Metrics](#/foundations/evaluation-metrics) · [The ML Coding Round](#/ml-coding/intro)
