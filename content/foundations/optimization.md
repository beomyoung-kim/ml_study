# Optimization

> [!NOTE] Goal of this chapter
> [What Is Machine Learning?](#/foundations/what-is-ml) described learning as moving parameters a little at a time in a direction that reduces loss. Optimization is the art of making those moves **well**. This chapter begins with the intuition of rolling a ball downhill, then explains why momentum accelerates, why Adam became a standard choice, and what to inspect first when training explodes. The opening sections are introductory; §4 onward goes deeper for interviews.

## §0 · Gradient descent = rolling a ball downhill

Think of the loss function as a **bumpy landscape**. Height is the loss, so lower is better, and position is the parameter value. We want to find the lowest valley. The method is simple: from the current position, take a step in the **steepest downhill direction**, opposite the gradient. Repeating this is **gradient descent**.

$$\theta_{t+1}=\theta_t-\eta\,g_t \qquad (g_t=\nabla_\theta L,\ \eta=\text{learning rate})$$

The step-size parameter $\eta$ is the **learning rate**. The fastest way to build intuition is to move the slider yourself: see how a high learning rate diverges and how slowly a tiny one progresses.

<div class="widget" data-widget="gradient-descent"></div>

> [!TIP] Interview one-liner
> An interviewer is not really testing whether you can recite the Adam equations. They want to know *why* momentum accelerates, *why* AdamW separates weight decay, *what* warmup plus cosine buys, and *what you inspect first* when a run diverges. Saying that "most divergence is explained by the update rule plus numerical stability and data health, not optimizer magic" sounds like experience with large-scale training.

## §1 · Four steps from SGD to Adam

In practice, we estimate a gradient from a **minibatch**, a small subset of the data, rather than from the full dataset. Its direction is useful but noisy, so a direct update can zigzag through a valley. The following four steps progressively control that zigzag. Let $g_t=\nabla_\theta L_t(\theta_t)$ be the minibatch gradient.

**1. SGD (Stochastic Gradient Descent)**—follow the current gradient:
$$\theta_{t+1}=\theta_t-\eta\,g_t$$

**2. Momentum**—accumulate past movement in a velocity, like the inertia of a ball rolling downhill:
$$v_{t+1}=\mu v_t+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_{t+1}\quad(\mu\approx0.9)$$
In a narrow valley, side-to-side components cancel while the consistent downhill component accumulates, increasing the effective stride and accelerating convergence. **Nesterov momentum** measures the gradient at the look-ahead point $\theta_t-\eta\mu v_t$ to reduce overshoot.

<figure>
<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- valley contours (elongated ellipses = ill-conditioned) -->
  <g fill="none" stroke="#98a3b2" stroke-width="1" opacity="0.5">
    <ellipse cx="330" cy="120" rx="290" ry="55"/>
    <ellipse cx="330" cy="120" rx="210" ry="38"/>
    <ellipse cx="330" cy="120" rx="130" ry="23"/>
    <ellipse cx="330" cy="120" rx="55" ry="11"/>
  </g>
  <text x="330" y="124" text-anchor="middle" fill="#12a150" font-size="11">minimum ●</text>
  <circle cx="330" cy="120" r="4" fill="#12a150"/>
  <!-- SGD zig-zag path (red) -->
  <polyline points="60,70 100,168 150,78 200,160 250,86 300,150 340,96 370,138" fill="none" stroke="#e0533f" stroke-width="2"/>
  <text x="90" y="55" fill="#e0533f">SGD: side-to-side zigzag</text>
  <!-- momentum smooth path (indigo) -->
  <path d="M60 190 Q 180 175, 260 150 T 330 122" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <text x="120" y="212" fill="#6366f1">momentum: cancel oscillation → descend directly</text>
</svg>
<figcaption>In a long, narrow, ill-conditioned valley, SGD in red bounces across the steep walls and descends slowly. Momentum in blue cancels this oscillation and follows the valley directly toward the minimum.</figcaption>
</figure>

**3. RMSprop**—divide by recent gradient magnitude for each coordinate, taking smaller steps in steep directions and larger steps in flat ones:
$$v_t=\rho v_{t-1}+(1-\rho)g_t^{\odot 2},\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t$$

**4. Adam**—combine momentum, the first moment, with per-coordinate scaling, the second moment, and bias correction. It is now a de facto default:
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^{\odot2}
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$
Typical defaults are $\beta_1{=}0.9,\ \beta_2{=}0.999,\ \epsilon{=}10^{-8}$. Bias correction removes the early-step bias caused by initializing $m$ and $v$ at zero.

| | First moment | Second moment | Bias correction | Primary use |
| --- | --- | --- | --- | --- |
| SGD + momentum | velocity | — | — | CNN classification; still strong |
| RMSprop | — | ✓ | usually no | legacy / RNN niche |
| Adam / AdamW | ✓ | ✓ | ✓ | Transformers, ViTs, LLMs, fine-tuning |

### Implement one momentum step

Translate the momentum update into code. Implement one step that accumulates velocity and updates the parameter. This is the scalar version, but vectors work identically.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sgd_momentum_step","packages":["numpy"],"approx":true,"starter":"def sgd_momentum_step(w, grad, v, lr=0.1, mu=0.9):\n    # One momentum step: v_new = mu*v + grad; w_new = w - lr*v_new\n    # Return [w_new, v_new].\n    pass","tests":[{"args":[1.0,2.0,0.0,0.1,0.9],"expect":[0.8,2.0]},{"args":[0.8,1.0,2.0,0.1,0.9],"expect":[0.52,2.8]},{"args":[0.0,0.0,1.0,0.5,0.5],"expect":[-0.25,0.5]}],"solution":"def sgd_momentum_step(w, grad, v, lr=0.1, mu=0.9):\n    v_new = mu * v + grad\n    w_new = w - lr * v_new\n    return [w_new, v_new]"}
</script>
</div>

On the first step, $v=0$, so the update matches SGD. Over later steps, velocity grows—2.0, then 2.8, and so on—accelerating in a consistent direction.

## §2 · Adam vs AdamW—decoupled weight decay

Adding an L2 penalty to the loss changes the gradient to $g_t\leftarrow \nabla L+\lambda\theta_t$. Adam then divides by $\sqrt{\hat v_t}$, however, so coordinates with large second moments receive **less decay**. Regularization becomes distorted across coordinates. **AdamW** fixes this by applying weight decay *outside* the preconditioner:

$$
\theta_{t+1}=\theta_t-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\Big)-\eta\lambda\theta_t
$$

> [!NOTE] In one sentence
> "AdamW applies weight decay outside gradient preconditioning, so the intended $\ell_2$ shrinkage acts uniformly across coordinates." It is a practical default for Transformer, ViT, and LLM fine-tuning, and teams usually **disable decay** for biases and normalization gains $\gamma,\beta$. The decay $\lambda$ and learning rate $\eta$ interact: cosine lowers $\eta$, which also lowers effective decay, so tune them together. For plain SGD, L2 and weight decay are nearly equivalent; the distinction is especially important for adaptive optimizers.

## §3 · Learning-rate schedules

**Warmup** limits the update-to-weight ratio during the early phase, when representations and gradient statistics change rapidly, and thereby reduces instability. With bias correction, Adam's first step does not become arbitrarily large merely because $\hat v$ is small. A larger batch reduces sampling noise, but early stabilization can become more important when it is paired with a larger peak LR, greater model width, and a distributed-training recipe. **Cosine decay** smoothly reduces the late-stage learning rate.

<div class="widget" data-widget="lr-schedule"></div>

$$
\text{warmup: } \eta_t=\eta_\text{peak}\frac{t}{T_w}\quad(t\le T_w)
$$
$$
\text{cosine: } \eta_t=\eta_\text{min}+\tfrac12(\eta_\text{peak}-\eta_\text{min})\Big(1+\cos\pi\frac{t-T_w}{T-T_w}\Big)
$$

**Warmup + cosine, with an optional cooldown**, is a common default in modern CV and LLM training. Alternatives include inverse-square-root schedules in classic Transformers, one-cycle from fast.ai, and SGDR warm restarts, which are less common in production than a single cosine schedule.

## §4 · Convexity and the second-order perspective (Advanced)

<figure>
<svg viewBox="0 0 640 170" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <path d="M30 140 C 120 -10, 200 -10, 290 140" fill="none" stroke="#12a150" stroke-width="2.5"/>
  <text x="160" y="160" text-anchor="middle" fill="#12a150">convex: one global minimum</text>
  <path d="M350 120 C 390 40, 420 150, 460 90 C 500 30, 540 150, 610 60" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="480" y="160" text-anchor="middle" fill="#e0533f">non-convex: saddles · plateaus · many minima</text>
</svg>
<figcaption>At a stationary point, a positive-definite Hessian indicates a strict local minimum, a negative-definite Hessian a strict local maximum, and a mix of positive and negative eigenvalues a saddle. If an eigenvalue is zero, this test alone is inconclusive.</figcaption>
</figure>

The Newton step $\theta_{t+1}=\theta_t-H_t^{-1}g_t$ uses curvature, but $H$ is a $d\times d$ matrix, is noisy under minibatching, and can be indefinite in a non-convex region. K-FAC and Shampoo/SOAP exploit approximations to curvature or matrix structure. Adam is a diagonal preconditioner based on gradient second moments, while Muon orthogonalizes matrix updates. Do not call all of them "Hessian approximations." Saddles, plateaus, and poor conditioning are all important difficulties in high dimensions.

> [!WARNING] The 2026 tone for new optimizers
> AdamW remains the default. Matrix optimizers such as Muon claim improvements in token efficiency, but when compared with a *properly tuned* AdamW baseline, practical speedups often shrink from 2× to roughly 1.1–1.4×. Ask for (1) a fairly tuned AdamW baseline, (2) reproduction at scale, and (3) matched hyperparameter budgets. Sharing hyperparameters between optimizers is a classic way to create a false win.

## §5 · Gradient clipping

$$
g\leftarrow g\cdot\min\!\Big(1,\frac{c}{\|g\|_2}\Big)\qquad(\text{global-norm clip})
$$

Use clipping to limit gradient spikes in RNNs, long sequences, and LLM pretraining, where $c=1.0$ is a common starting point. Global-norm clipping preserves the original gradient direction better than clipping each component separately. **Clipping on almost every step is a diagnostic signal**: inspect LR and warmup, loss and gradient scaling, outlier batches, exploding paths, and implementation bugs, while monitoring both clip fraction and the unclipped norm.

## §6 · Batch-size effects

Under an independent-sample approximation, minibatch-gradient variance decreases roughly as $1/B$. Batch-coupled layers, duplicate or correlated samples, and the sampling method can change this relationship. **Linear LR scaling is a useful starting point for some optimizers and training regimes, not a law.** When changing the batch size, validate it experimentally together with the optimizer, token or step budget, and schedule. Gradient accumulation also does not increase the micro-batch size used to compute BatchNorm statistics.

## Interview Q&A

<details class="qa"><summary>Why does momentum converge faster than plain SGD?</summary>
<div class="qa-body">

**Short:** It accumulates a velocity that cancels oscillation across high-curvature axes and reinforces the consistent direction down the valley, increasing the effective stride in a useful direction.

**Deep:** On an ill-conditioned quadratic, SGD slows with condition number $\kappa$; heavy-ball momentum improves this dependence—to roughly $\kappa\to\sqrt\kappa$ in an ideal quadratic. Because velocity is an EMA of gradients, it also low-pass filters minibatch noise. Excessive $\mu$ can overshoot and diverge, while Nesterov's look-ahead reduces that overshoot.
</div></details>

<details class="qa"><summary>Adam versus AdamW—exactly what differs, and why does it matter?</summary>
<div class="qa-body">

**Short:** Adam with L2 folds decay into the gradient, so adaptive $1/\sqrt{\hat v}$ scaling makes decay unequal across coordinates. AdamW applies decay directly to the weights, decoupled from the preconditioner.

**Deep:** Under Adam + L2, decay is divided down for a high-second-moment coordinate, so effective regularization is no longer the intended $\ell_2$. AdamW's $-\eta\lambda\theta$ term restores uniform shrinkage and empirically gives better generalization and cleaner LR/weight-decay tuning. It is a default for Transformers, ViTs, and LLMs, with decay usually disabled for biases and LayerNorm or RMSNorm gains.
</div></details>

<details class="qa"><summary>Why warmup followed by cosine?</summary>
<div class="qa-body">

**Short:** Warmup limits update magnitude while representations and gradients change rapidly at the start, while cosine smoothly decreases the late-stage LR. Both are useful defaults, not universal optima.

**Deep:** In bias-corrected Adam's first step, $\hat m/\sqrt{\hat v}$ is approximately the magnitude of the gradient's sign, so it is incorrect to claim that a small $v$ alone must create a huge step. Actual early instability comes from update magnitude relative to parameter scale, unsettled activation and gradient statistics, and interactions among a large peak LR and numerical precision. Warmup passes through this interval at a lower LR.
</div></details>

<details class="qa"><summary>Training diverged at step 300. In what order do you investigate?</summary>
<div class="qa-body">

**Short:** LR and warmup → gradient norm and clip fraction → mixed precision and loss scale → a bad batch with NaN/inf input or incorrect label scale → initialization, normalization, and residual paths → distributed issues such as incorrect LR scaling or a bad all-reduce.

**Deep:** Plot loss and gradient norm over the first few hundred steps. A spike points first to peak LR and warmup, but clipping on every step can also come from loss scaling, outlier batches, badly scaled targets, or a structural bug. Under FP16, verify dynamic loss scaling; see [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency). Rule out `inf` values, incorrect targets, pathological augmentation, and normalization or residual-scaling problems. The point is to isolate LR, numerics, data, and implementation before changing the optimizer's name.
</div></details>

**Expected follow-ups**

- *What if you lower $\beta_2$?* The second moment adapts faster, becoming more responsive to a nonstationary objective but less stable.
- *Batch size × $k$ → LR?* Begin with linear scaling and a longer warmup; if unstable, back off toward $\sqrt k$ scaling.
- *LR finder?* Sweep $\eta$ exponentially for several hundred steps and select before divergence. It is noisy and often replaced by a proxy-model run at LLM scale.
- *Sharp vs flat minima?* Better generalization from flat minima is a *hypothesis* and motivates SAM, but the definition is sensitive to reparameterization; present it as unsettled.

## Cheat sheet

| Question | In one line |
| --- | --- |
| Gradient descent | Move one step opposite the gradient—downhill—on the loss landscape |
| SGD → momentum | Velocity is an EMA of gradients; cancel oscillation and accelerate down the valley |
| Adam | Momentum + per-coordinate $1/\sqrt{\hat v}$ + bias correction; $\beta=(0.9,0.999)$ |
| Adam vs AdamW | Decoupled decay gives uniform shrinkage; default for Transformers |
| No-decay parameters | Biases and LayerNorm/RMSNorm gains |
| Warmup | Ramp LR from zero to avoid excessive early updates and unstable gradients |
| Cosine | Smooth annealing with few knobs; warmup + cosine is a common default |
| Second-order methods | $H$ is too large, noisy, or indefinite, so use structured or diagonal preconditioning |
| Gradient clipping | Start around global norm $c{\approx}1$; frequent clipping is a diagnostic signal |
| Batch size | Variance is roughly $1/B$ under assumptions; validate LR, schedule, and budget jointly |

**Next:** [Regularization & Generalization](#/foundations/regularization-generalization) · [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus) · [Normalization & Training Stability](#/foundations/normalization-stability) · [Distributed Training](#/foundations/distributed-training) · [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency)
