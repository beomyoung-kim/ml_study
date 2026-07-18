# Losses & Gradients

> [!TIP] Say this first
> "The single fact that unlocks this whole area: the gradient of softmax-cross-entropy with respect to the logits is just $p - \text{onehot}(y)$. Predicted probability minus the truth. Everything else — the backward pass, focal loss, label smoothing — is a variation on that." Say it, then derive it if asked.

Implement MSE, BCE, cross-entropy, and focal loss with **numerically stable** code, derive the softmax-CE gradient by hand, and write an **autograd-free backward** for a linear layer. This round tests the bidirectional skill of turning math into code *and* code into gradients — the core of understanding backprop (see [Optimization](#/foundations/optimization)).

## The losses at a glance

| Loss | Formula (per sample) | Use |
| --- | --- | --- |
| MSE | $\tfrac12(\hat y - y)^2$ | regression |
| BCE | $-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ | binary / multi-label |
| Cross-entropy | $-\log p_y,\ \ p=\text{softmax}(z)$ | multi-class |
| Focal | $-\alpha_t(1-p_t)^\gamma\log p_t$ | class-imbalanced detection |

## Deriving the softmax-CE gradient

For logits $z$, $p_i = e^{z_i}/\sum_j e^{z_j}$, loss $L = -\log p_y$. The softmax Jacobian is $\partial p_i/\partial z_k = p_i(\delta_{ik} - p_k)$. Then:

$$
\frac{\partial L}{\partial z_k} = -\frac{1}{p_y}\frac{\partial p_y}{\partial z_k}
= -\frac{1}{p_y}\,p_y(\delta_{yk} - p_k) = p_k - \delta_{yk}
$$

So $\boxed{\nabla_z L = p - \text{onehot}(y)}$ — clean, cheap, and the reason logits+CE are fused in every framework. *(verifiable)*

## Cross-entropy (stable) with gradient

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

## BCE, MSE, focal — all numerically stable

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

> [!WARNING] Never compose log and exp naively
> `np.log(sigmoid(z))` overflows for large $|z|$. Use `logaddexp`, `log1p`, `softplus`, and the identity $\log\sigma(z)=z-\text{softplus}(z)$. This is the numerical-stability signal interviewers listen for in loss code.

## Autograd-free backward: a linear layer

The heart of backprop. For $Y = XW + b$ with $X\in\mathbb{R}^{N\times D_\text{in}}$, $W\in\mathbb{R}^{D_\text{in}\times D_\text{out}}$, given upstream $\partial L/\partial Y$ (`dY`, shape $(N,D_\text{out})$):

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

**Shape check is the debugging tool:** `dW` must match `W`, so it can only be `X.T @ dY`. The three rules — *matmul for the linear map, transpose the other operand, sum over the broadcasted (batch) axis for bias* — generalize to every layer.

## Gradient check (always do this unprompted)

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

> [!NOTE] Framework one-liner
> `F.cross_entropy(logits, targets)` (fuses log-softmax + NLL, never materializes probs), `F.binary_cross_entropy_with_logits`, `F.mse_loss`; focal via `torchvision.ops.sigmoid_focal_loss`.

## Practice — implement, run, test

> [!TIP] How to use this section
> Each problem below has a **live Python editor** with **NumPy** available. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime plus NumPy (~15 MB); later runs are instant.

Work down the list — the elementwise activations first, then the losses, then the softmax-CE gradient that ties them together.

### 1. Sigmoid <span class="badge badge-easy">Easy</span>

Map real-valued logits to $(0,1)$ probabilities elementwise.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sigmoid","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef sigmoid(z):\n    # elementwise 1/(1+e^-z); accept a list or nested list\n    pass","tests":[{"args":[[0.0,2.0,-2.0]],"expect":[0.5,0.8807970779778823,0.11920292202211755]},{"args":[[[0.0,1.0],[-1.0,3.0]]],"expect":[[0.5,0.7310585786300049],[0.2689414213699951,0.9525741268224334]]},{"args":[[0.0]],"expect":[0.5]}],"solution":"import numpy as np\n\ndef sigmoid(z):\n    z = np.asarray(z, dtype=float)\n    return 1.0 / (1.0 + np.exp(-z))"}
</script>
</div>

### 2. Stable Softmax <span class="badge badge-easy">Easy</span>

Subtract the row max before exponentiating so large logits don't overflow.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(z, axis=-1):\n    # subtract the max along axis for stability, exponentiate, normalize\n    pass","tests":[{"args":[[[1.0,2.0,3.0]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[0.0,0.0]]],"expect":[[0.5,0.5]]},{"args":[[[1000.0,1001.0,1002.0]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]}],"solution":"import numpy as np\n\ndef softmax(z, axis=-1):\n    z = np.asarray(z, dtype=float)\n    z = z - np.max(z, axis=axis, keepdims=True)\n    e = np.exp(z)\n    return e / e.sum(axis=axis, keepdims=True)"}
</script>
</div>

### 3. Softmax Cross-Entropy Loss <span class="badge badge-med">Medium</span>

Mean negative log-likelihood of the true class over a batch of logits.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"cross_entropy_loss","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef cross_entropy_loss(logits, targets):\n    # logits:(N,C), targets:(N,) int; softmax rows, gather true-class prob, -log, mean\n    pass","tests":[{"args":[[[1.0,2.0,3.0],[1.0,1.0,1.0]],[2,0]],"expect":0.7531091265562451},{"args":[[[0.0,0.0],[0.0,0.0]],[0,1]],"expect":0.6931471805599453},{"args":[[[2.0,1.0,0.0]],[0]],"expect":0.40760596444438046}],"solution":"import numpy as np\n\ndef cross_entropy_loss(logits, targets):\n    logits = np.asarray(logits, dtype=float)\n    targets = np.asarray(targets)\n    N = logits.shape[0]\n    z = logits - logits.max(axis=1, keepdims=True)\n    p = np.exp(z)\n    p /= p.sum(axis=1, keepdims=True)\n    py = p[np.arange(N), targets]\n    return float(-np.log(np.clip(py, 1e-12, 1.0)).mean())"}
</script>
</div>

### 4. Softmax-CE Gradient <span class="badge badge-med">Medium</span>

The famous $p - \text{onehot}(y)$, mean-reduced by the batch size.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax_ce_grad","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax_ce_grad(logits, targets):\n    # gradient wrt logits: softmax(logits) minus one-hot(targets), divided by N\n    pass","tests":[{"args":[[[1.0,2.0,3.0],[1.0,1.0,1.0]],[2,0]],"expect":[[0.04501528658519023,0.12236423552739882,-0.1673795221125891],[-0.33333333333333337,0.16666666666666666,0.16666666666666666]]},{"args":[[[0.0,0.0],[0.0,0.0]],[0,1]],"expect":[[-0.25,0.25],[0.25,-0.25]]},{"args":[[[2.0,1.0,0.0]],[0]],"expect":[[-0.3347590442251782,0.24472847105479764,0.09003057317038046]]}],"solution":"import numpy as np\n\ndef softmax_ce_grad(logits, targets):\n    logits = np.asarray(logits, dtype=float)\n    targets = np.asarray(targets)\n    N = logits.shape[0]\n    z = logits - logits.max(axis=1, keepdims=True)\n    p = np.exp(z)\n    p /= p.sum(axis=1, keepdims=True)\n    grad = p.copy()\n    grad[np.arange(N), targets] -= 1.0\n    return grad / N"}
</script>
</div>

### 5. Mean Squared Error <span class="badge badge-easy">Easy</span>

Half the mean squared difference — the $\tfrac12$ makes the gradient $\hat y - y$.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"mse","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef mse(pred, y):\n    # 0.5 * mean((pred - y)^2)\n    pass","tests":[{"args":[[1.0,2.0,3.0],[1.0,2.0,3.0]],"expect":0.0},{"args":[[2.0,0.0,2.0],[0.0,0.0,0.0]],"expect":1.3333333333333333},{"args":[[[1.0,2.0],[3.0,4.0]],[[0.0,0.0],[0.0,0.0]]],"expect":3.75}],"solution":"import numpy as np\n\ndef mse(pred, y):\n    pred = np.asarray(pred, dtype=float)\n    y = np.asarray(y, dtype=float)\n    diff = pred - y\n    return float(0.5 * np.mean(diff ** 2))"}
</script>
</div>

### 6. BCE With Logits (stable) <span class="badge badge-med">Medium</span>

Numerically stable binary cross-entropy straight from logits.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"bce_with_logits","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef bce_with_logits(z, y):\n    # stable: max(z,0) - z*y + log1p(exp(-|z|)), then mean\n    pass","tests":[{"args":[[0.0,0.0,0.0],[1.0,0.0,1.0]],"expect":0.6931471805599453},{"args":[[10.0,-10.0],[1.0,0.0]],"expect":4.539889921686465e-05},{"args":[[2.0,-1.0,0.5],[1.0,1.0,0.0]],"expect":0.8047555609137674}],"solution":"import numpy as np\n\ndef bce_with_logits(z, y):\n    z = np.asarray(z, dtype=float)\n    y = np.asarray(y, dtype=float)\n    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))\n    return float(loss.mean())"}
</script>
</div>

### 7. Binary Focal Loss <span class="badge badge-med">Medium</span>

Down-weight easy examples by $(1-p_t)^\gamma$ — stable via logsigmoid.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"binary_focal_loss","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef binary_focal_loss(z, y, gamma=2.0, alpha=0.25):\n    # logsigmoid via softplus; p_t, log_pt, alpha_t by class; mean of -a_t*(1-p_t)^gamma*log_pt\n    pass","tests":[{"args":[[2.0,-1.0,0.5],[1.0,0.0,0.0]],"expect":0.10016771149490598},{"args":[[0.0,0.0],[1.0,0.0]],"expect":0.08664339756999316},{"args":[[3.0,-3.0,1.0,-1.0],[1.0,0.0,1.0,0.0],1.0,0.5],"expect":0.02163833526892106}],"solution":"import numpy as np\n\ndef binary_focal_loss(z, y, gamma=2.0, alpha=0.25):\n    z = np.asarray(z, dtype=float).reshape(-1)\n    y = np.asarray(y, dtype=float).reshape(-1)\n    sp = np.logaddexp(0.0, z)\n    log_p, log_1mp = z - sp, -sp\n    p = np.exp(log_p)\n    p_t = np.where(y == 1, p, 1 - p)\n    log_pt = np.where(y == 1, log_p, log_1mp)\n    a_t = np.where(y == 1, alpha, 1 - alpha)\n    return float((-a_t * (1 - p_t) ** gamma * log_pt).mean())"}
</script>
</div>

## Common bugs interviewers watch for

- **`cross_entropy` expects logits, not probabilities** — don't softmax twice.
- **Forgetting the `1/N`** in the mean-reduced gradient.
- **`log(0)`** — clamp probabilities before `log`.
- **BCE vs CE confusion:** BCE/sigmoid for multi-label (independent classes), softmax-CE for mutually exclusive classes.
- **Shape of `dW`:** must equal `W.shape`; the transpose placement follows from that.

## Q&A

<details class="qa"><summary>Why does focal loss help with class imbalance?</summary>
<div class="qa-body">

**Short:** the $(1-p_t)^\gamma$ modulating factor shrinks the loss (and gradient) from easy, well-classified examples so training focuses on hard ones — crucial when background boxes outnumber objects 1000:1.

**Deep:** in dense detection the vast majority of anchors are easy negatives; summed, their small individual CE losses still dominate the gradient and drown out rare positives. Focal down-weights an example with confidence $p_t$ by $(1-p_t)^\gamma$: an easy example at $p_t=0.9$, $\gamma=2$ is scaled by $0.01$, while a hard one at $p_t=0.1$ is barely touched ($0.81$). $\alpha$ additionally rebalances the positive/negative class prior. This let RetinaNet match two-stage detectors without hard-negative mining.
</div></details>

<details class="qa"><summary>Why fuse softmax and cross-entropy in one op?</summary>
<div class="qa-body">

**Short:** stability and speed — the fused log-softmax-then-NLL avoids ever materializing probabilities or computing `log(exp(...))`, and the gradient collapses to the clean $p-\text{onehot}(y)$.

**Deep:** computing `softmax` then `log` separately risks under/overflow and does redundant work; `log_softmax(z) = z - \text{LSE}(z)$ is directly stable via the max-subtraction, and the NLL just gathers the true class. On the backward side, chaining softmax's Jacobian with the log would be an $O(C^2)$ per-sample product, but the analytic simplification makes it $O(C)$. Both numerically and computationally it's strictly better, which is why `F.cross_entropy` takes logits.
</div></details>

<details class="qa"><summary>Dice loss and CE — why combine them for segmentation?</summary>
<div class="qa-body">

**Short:** CE gives per-pixel calibration and stable gradients; Dice directly optimizes region overlap and is robust to foreground/background imbalance. They're complementary, so segmentation models often sum them.

**Deep:** soft Dice is $1 - \frac{2\sum p g}{\sum p + \sum g}$, a *non-decomposable* loss over the whole mask, which makes it naturally scale-invariant to object size — a tiny object contributes as much as a large one, unlike per-pixel CE. But Dice gradients are noisy early in training when predictions are near zero, so CE stabilizes optimization while Dice aligns the objective with the mIoU metric. See [Segmentation](#/cv/segmentation).
</div></details>

### Follow-ups
- **Label smoothing?** Replace the one-hot target with $(1-\epsilon)\cdot\text{onehot} + \epsilon/C$ to reduce overconfidence and improve calibration.
- **Huber / smooth-L1?** Quadratic near zero, linear in the tails — robust box-regression loss (Fast R-CNN).
- **Why $\tfrac12$ in MSE?** Cancels the 2 from differentiation, so $\nabla = \hat y - y$; purely cosmetic.
- **Class weights?** Scale each class's CE term by inverse frequency for imbalance.

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
