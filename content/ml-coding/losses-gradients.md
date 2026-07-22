# Losses & Gradients

> [!NOTE] Goal of this chapter
> In [What Is Machine Learning?](#/foundations/what-is-ml), training meant updating parameters in a direction that reduces a loss. Here you will **implement that loss in code** and derive its **gradient by hand**. That is the substance of backpropagation. We will move from intuition to short equations to runnable NumPy labs.

> [!TIP] Interview one-liner
> “The single fact that unlocks this whole area: the gradient of softmax cross-entropy with respect to a logit is just $p-\text{onehot}(y)$—**predicted probability minus the truth**. The backward pass, focal loss, and label smoothing are all variations on it.” Say that, then derive it if asked.

## §0 · What is a loss, and why do we need a gradient?

**Loss** turns “how wrong was the prediction?” into one number. Smaller is better. Training is the process of reducing that number, which requires knowing “in which direction should each parameter move to lower the loss?” That direction is the **gradient**.

A loss function therefore has to provide two things: **① a value**, measuring how wrong the model is, and **② a gradient**, telling it how to improve. Every implementation in this chapter exposes both.

- **Regression:** predict a number such as a house price or coordinate—use **MSE**.
- **Binary or multi-label classification:** independent yes/no decisions—use **BCE**.
- **Multi-class classification:** choose one mutually exclusive class—use **cross-entropy**.

| Loss | Per-sample formula | When to use it |
| --- | --- | --- |
| MSE | $\tfrac12(\hat y - y)^2$ | regression |
| BCE | $-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ | binary / multi-label |
| Cross-entropy | $-\log p_y,\ \ p=\text{softmax}(z)$ | multi-class |
| Focal | $-\alpha_t(1-p_t)^\gamma\log p_t$ | class-imbalanced detection |

## §1 · Softmax and cross-entropy by intuition

A classifier's final layer emits one **score, or logit**, per class. Logits are not probabilities: they can be negative or arbitrarily large. **Softmax** converts them into **probabilities that sum to 1**, exponentially emphasizing larger scores:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

<figure>
<svg viewBox="0 0 600 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- logits -->
  <text x="95" y="20" text-anchor="middle" fill="#98a3b2">logit z (score)</text>
  <g fill="#6366f1">
    <rect x="40"  y="120" width="34" height="40"/><text x="57"  y="176" text-anchor="middle" fill="currentColor">1</text>
    <rect x="90"  y="90"  width="34" height="70"/><text x="107" y="176" text-anchor="middle" fill="currentColor">2</text>
    <rect x="140" y="55"  width="34" height="105"/><text x="157" y="176" text-anchor="middle" fill="currentColor">3</text>
  </g>
  <!-- arrow -->
  <text x="300" y="95" text-anchor="middle" fill="#e0533f" font-weight="700">softmax</text>
  <path d="M215 100 H385" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#sm)"/>
  <text x="300" y="120" text-anchor="middle" fill="#98a3b2" font-size="10">exponentiate, then normalize to sum=1</text>
  <!-- probs -->
  <text x="500" y="20" text-anchor="middle" fill="#98a3b2">probability p (sum=1.0)</text>
  <g fill="#12a150">
    <rect x="430" y="150" width="34" height="10"/><text x="447" y="176" text-anchor="middle" fill="currentColor">.09</text>
    <rect x="480" y="122" width="34" height="38"/><text x="497" y="176" text-anchor="middle" fill="currentColor">.24</text>
    <rect x="530" y="55"  width="34" height="105"/><text x="547" y="176" text-anchor="middle" fill="currentColor">.67</text>
  </g>
  <defs><marker id="sm" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>Logits $[1,2,3]$ become probabilities $[0.09,0.24,0.67]$ through softmax. Exponentiation concentrates probability on the largest score without forcing the others to exactly zero—a soft maximum.</figcaption>
</figure>

**Cross-entropy** is now simple: take the probability assigned to the correct class and apply $-\log$. If the correct class gets 0.9, the loss is $-\log 0.9\approx0.1$, which is small; if it gets only 0.1, the loss is $-\log 0.1\approx2.3$, which is large.

$$
L=-\log p_y
$$

> [!NOTE] What $-\log$ does
> When the correct-class probability approaches 1, the loss approaches 0. As it approaches 0, the loss goes to infinity. In other words, it penalizes **confidently wrong predictions** very strongly. This is central to the explanation below of why MSE is not the default classification loss.

## §2 · Why cross-entropy? (CE vs KL, BCE vs CE, and why not L1/L2)

### Cross-entropy vs KL divergence
For a true distribution $p$ and prediction $q$, $D_{\mathrm{KL}}(p\Vert q)=H(p,q)-H(p)$. In supervised learning, the label distribution $p$ is fixed, so $H(p)$ is constant—and zero for a one-hot target. Therefore, **minimizing CE is equivalent to minimizing $D_{\mathrm{KL}}(p\Vert q)$**, pulling the prediction toward the target. Use CE for a fixed label; use KL directly when **both sides are distributions**, as in knowledge distillation, VAEs, or an RLHF/PPO reference-policy penalty. For KL's asymmetry—forward mode-covering versus reverse mode-seeking—and more examples, see [Probability & Statistics](#/foundations/probability-statistics).

### BCE vs CE — functionally different
- **BCE (sigmoid per output):** each output is an **independent** yes/no. Use for **multi-label** ("cat" *and* "outdoor" can both be true) — $C$ independent Bernoulli problems, probabilities need **not** sum to 1.
- **CE (softmax over classes):** one **categorical** over **mutually exclusive** classes; softmax *couples* the logits so probabilities **sum to 1**. Use for single-label multi-class.
- They coincide in the 2-class case: **2-way softmax-CE ≡ BCE on one logit**. Rule of thumb: *mutually exclusive → softmax-CE; independent/multi-label → BCE.*

### Why not L1/L2 (MSE) for classification?
Put a sigmoid/softmax on the output and it's tempting to just regress with MSE. Three reasons it's the wrong tool:

**1) The gradient vanishes exactly when you're most wrong.** With sigmoid + MSE, $L=\tfrac12(\sigma(z)-y)^2$ and

$$
\frac{\partial L}{\partial z}=(\sigma(z)-y)\,\sigma'(z),\qquad \sigma'(z)=\sigma(z)(1-\sigma(z))
$$

If the truth is $y=1$ but the model says $z=-5$ ($\sigma\approx0.007$), then $\sigma'\approx0.007$, so the gradient is $\approx(-0.993)(0.007)\approx-0.007$ — **tiny**, even though the prediction is confidently wrong. With sigmoid + **BCE** (or softmax + CE) the $\sigma'$ **cancels**:

$$
\frac{\partial L_{\text{BCE}}}{\partial z}=\sigma(z)-y\ \ (\approx-0.993\text{ here})
$$

a **large** signal proportional to the error. CE learns fast from mistakes; MSE-on-sigmoid stalls.

**2) A large penalty for confident wrongness.** As $p_y\to0$, $-\log p_y\to\infty$, so CE strongly punishes assigning almost zero probability to the correct class. CE is a proper scoring rule under ideal population and model assumptions, but finite data and distribution shift do not make neural-network probabilities automatically calibrated. Evaluate ECE and temperature scaling separately when calibration matters.

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
<figcaption>Loss vs. probability assigned to the <b>true</b> class. CE blows up as $p\to0$ (huge push away from confident-wrong); MSE flattens (weak signal), and its sigmoid gradient shrinks further.</figcaption>
</figure>

**3) Right likelihood / convexity.** CE is the negative log-likelihood under a Bernoulli/Categorical model — the *correct* probabilistic loss for class labels; MSE corresponds to a **Gaussian** likelihood (right for regression, wrong for categories). CE is also convex in the logits, while MSE-through-a-sigmoid is non-convex with poor local minima. **When L1/L2 *is* right:** regression / continuous targets, and bounding-box coordinate regression (smooth-L1 / Huber).

## §3 · Deriving the softmax-CE gradient—why $p-y$?

For logits $z$, let $p_i=e^{z_i}/\sum_j e^{z_j}$ and $L=-\log p_y$. The softmax Jacobian is $\partial p_i/\partial z_k=p_i(\delta_{ik}-p_k)$. Substituting it into the derivative of $L$ makes two terms cancel:

$$
\frac{\partial L}{\partial z_k} = -\frac{1}{p_y}\frac{\partial p_y}{\partial z_k}
= -\frac{1}{p_y}\,p_y(\delta_{yk} - p_k) = p_k - \delta_{yk}
$$

Thus $\boxed{\nabla_z L=p-\text{onehot}(y)}$. Frameworks expose softmax plus CE as one logit-based operation primarily to use a stable log-sum-exp formulation and avoid extra probability storage and kernel launches. The same derivative also appears in [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus).

## §4 · Cross-entropy (stable) with gradient

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

## §5 · BCE, MSE, focal—all numerically stable

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

> [!WARNING] Never compose log and exp naively
> `np.log(sigmoid(z))` overflows for large $|z|$. Use `logaddexp`, `log1p`, `softplus`, and the identity $\log\sigma(z)=z-\text{softplus}(z)$. This is the numerical-stability signal interviewers listen for in loss code.

## §6 · Autograd-free backward: a linear layer

The heart of backprop. For $Y = XW + b$ with $X\in\mathbb{R}^{N\times D_\text{in}}$, $W\in\mathbb{R}^{D_\text{in}\times D_\text{out}}$, given upstream $\partial L/\partial Y$ (`dY`, shape $(N,D_\text{out})$):

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
  <text x="140" y="38" text-anchor="middle" fill="#0ea5e9">forward →</text>
  <path d="M395 92 C 320 118, 180 118, 120 92" stroke="#12a150" stroke-width="1.5" fill="none" marker-end="url(#g)"/>
  <text x="255" y="115" text-anchor="middle" fill="#12a150">← backward: dW=Xᵀ·dY, dX=dY·Wᵀ</text>
  <defs>
    <marker id="f" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#0ea5e9"/></marker>
    <marker id="g" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
  </defs>
</svg>
<figcaption>The forward pass caches input X; the backward pass multiplies dY by transposed matrices to form dW and dX. The shapes are the answer key.</figcaption>
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

**Shape check is the debugging tool:** `dW` must match `W`, so it can only be `X.T @ dY`. The three rules — *matmul for the linear map, transpose the other operand, sum over the broadcasted (batch) axis for bias* — generalize to every layer.

## §7 · Gradient check (always do this unprompted)

Compare an analytic gradient with finite differences computed directly from the definition.

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
{"func":"sigmoid","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef sigmoid(z):\n    # stable elementwise sigmoid; split non-negative and negative inputs to avoid exp overflow\n    pass","tests":[{"args":[[0.0,2.0,-2.0]],"expect":[0.5,0.8807970779778823,0.11920292202211755]},{"args":[[[0.0,1.0],[-1.0,3.0]]],"expect":[[0.5,0.7310585786300049],[0.2689414213699951,0.9525741268224334]]},{"args":[[-1000.0,1000.0]],"expect":[0.0,1.0]}],"solution":"import numpy as np\n\ndef sigmoid(z):\n    z = np.asarray(z, dtype=float)\n    out = np.empty_like(z)\n    pos = z >= 0\n    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))\n    ez = np.exp(z[~pos])\n    out[~pos] = ez / (1.0 + ez)\n    return out"}
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
{"func":"cross_entropy_loss","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef cross_entropy_loss(logits, targets):\n    # logits:(N,C), targets:(N,); compute mean logsumexp(logits)-true_logit without clipping probabilities\n    pass","tests":[{"args":[[[1.0,2.0,3.0],[1.0,1.0,1.0]],[2,0]],"expect":0.7531091265562451},{"args":[[[0.0,0.0],[0.0,0.0]],[0,1]],"expect":0.6931471805599453},{"args":[[[2.0,1.0,0.0]],[0]],"expect":0.40760596444438046},{"args":[[[-1000.0,0.0]],[0]],"expect":1000.0}],"solution":"import numpy as np\n\ndef cross_entropy_loss(logits, targets):\n    logits = np.asarray(logits, dtype=float)\n    targets = np.asarray(targets, dtype=np.int64)\n    if logits.ndim != 2 or targets.shape != (logits.shape[0],) or logits.shape[0] == 0 or logits.shape[1] == 0:\n        raise ValueError('expected non-empty logits (N,C) and targets (N,)')\n    if np.any((targets < 0) | (targets >= logits.shape[1])):\n        raise ValueError('target index out of range')\n    m = logits.max(axis=1)\n    lse = m + np.log(np.exp(logits - m[:, None]).sum(axis=1))\n    return float(np.mean(lse - logits[np.arange(len(targets)), targets]))"}
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

**Deep:** in dense detection the vast majority of anchors are easy negatives; summed, their small individual CE losses still dominate the gradient and drown out rare positives. Focal down-weights an example with confidence $p_t$ by $(1-p_t)^\gamma$: an easy example at $p_t=0.9$, $\gamma=2$ is scaled by $0.01$, while a hard one at $p_t=0.1$ is barely touched ($0.81$). $\alpha$ additionally rebalances the positive/negative class prior. This let RetinaNet match two-stage detectors without hard-negative mining. See [Object Detection](#/cv/detection).
</div></details>

<details class="qa"><summary>Why fuse softmax and cross-entropy in one op?</summary>
<div class="qa-body">

**Short:** stability and speed — the fused log-softmax-then-NLL avoids ever materializing probabilities or computing `log(exp(...))`, and the gradient collapses to the clean $p-\text{onehot}(y)$.

**Deep:** computing `softmax` then `log` separately risks under/overflow and repeats work. $\log\text{-softmax}(z)=z-\text{LSE}(z)$ is directly stable via max subtraction, and NLL gathers only the true class. On backward, chaining softmax's Jacobian with log naively gives an $O(C^2)$ product per sample; the analytic simplification makes it $O(C)$. It is strictly better numerically and computationally, which is why `F.cross_entropy` takes logits.
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

**Next:** [Optimization](#/foundations/optimization) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [Object Detection](#/cv/detection) · [Segmentation](#/cv/segmentation) · [The ML Coding Round](#/ml-coding/intro)
