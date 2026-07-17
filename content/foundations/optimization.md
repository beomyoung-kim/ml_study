# Optimization

<div class="tag-row"><span class="tag">SGD & momentum</span><span class="tag">Adam / AdamW</span><span class="tag">LR schedules</span><span class="tag">warmup</span><span class="tag">grad clipping</span><span class="tag">batch size</span></div>

> [!TIP] What they're really probing
> Not the ability to recite the Adam update — the ability to explain *why* momentum accelerates, *why* AdamW decouples weight decay, *what* warmup+cosine buys you, and *what you check first when a run diverges*. Say "the update rule and numerical stability + data health explain most divergences, not optimizer magic," and you sound like someone who has actually trained large models.

## Play with it first

The fastest way to build intuition for step size, momentum, and getting stuck is to move the sliders.

<div class="widget" data-widget="gradient-descent"></div>

## From SGD to Adam in four steps

Let $g_t=\nabla_\theta L_t(\theta_t)$ be the minibatch gradient.

**1. SGD** — follow the current gradient:
$$\theta_{t+1}=\theta_t-\eta\,g_t$$

**2. Momentum (heavy-ball)** — accumulate a velocity (EMA of gradients):
$$v_{t+1}=\mu v_t+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_{t+1}\quad(\mu\approx0.9)$$
In a narrow ravine, oscillating components cancel while the consistent down-valley direction accumulates → larger effective steps, faster convergence. **Nesterov** evaluates the gradient at the look-ahead point $\theta_t-\eta\mu v_t$, a small correction that improves stability near the optimum.

**3. RMSprop** — normalize each coordinate by its recent gradient magnitude:
$$v_t=\rho v_{t-1}+(1-\rho)g_t^{\odot 2},\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t$$

**4. Adam** — momentum *and* per-coordinate scaling, with bias correction:
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^{\odot2}
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$
Defaults $\beta_1{=}0.9,\ \beta_2{=}0.999,\ \epsilon{=}10^{-8}$. Bias correction undoes the zero-initialization bias of $m,v$ in the first steps.

| | 1st moment | 2nd moment | bias corr. | default use |
| --- | --- | --- | --- | --- |
| SGD+mom | velocity | — | — | CNN classification, still strong |
| RMSprop | — | ✓ | usually no | legacy / RNN niche |
| Adam / AdamW | ✓ | ✓ | ✓ | Transformers, ViT, LLM, fine-tuning |

## Adam vs AdamW — decoupled weight decay

Adding L2 to the loss gives $g_t\leftarrow \nabla L+\lambda\theta_t$. But Adam then divides by $\sqrt{\hat v_t}$, so coordinates with large second moments get **weaker** decay — the regularization is distorted per-coordinate. AdamW fixes this by decaying the parameters *outside* the preconditioner:

$$
\theta_{t+1}=\theta_t-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\Big)-\eta\lambda\theta_t
$$

> [!NOTE] Say this in one line
> "AdamW applies weight decay outside the gradient preconditioning, so the intended $\ell_2$ shrinkage hits every coordinate uniformly." In practice: it's the default for Transformers/ViT/LLM fine-tuning, and teams usually **disable decay on biases and norm gains** ($\gamma,\beta$). Note $\lambda$ and $\eta$ interact — as cosine shrinks $\eta$, effective decay shrinks too, so tune them together. In plain SGD, L2 and weight decay are (nearly) equivalent; the distinction matters specifically for adaptive optimizers.

## Learning-rate schedules

Warmup avoids early instability (Adam's small initial $\hat v$ can produce huge steps; large-batch / large-model runs have noisy early gradients). Cosine decays smoothly for fine late-stage search with few hyperparameters. Step decay is simple but the milestones are finicky.

<div class="widget" data-widget="lr-schedule"></div>

$$
\text{warmup: } \eta_t=\eta_\text{peak}\frac{t}{T_w}\quad(t\le T_w)
$$
$$
\text{cosine: } \eta_t=\eta_\text{min}+\tfrac12(\eta_\text{peak}-\eta_\text{min})\Big(1+\cos\pi\frac{t-T_w}{T-T_w}\Big)
$$

**Warmup + cosine (+ optional cooldown)** is the modern CV/LLM default. Other options: inverse-sqrt (classic Transformer), one-cycle (fast.ai), and SGDR-style warm restarts (rarer in production than a single cosine).

## Convexity & the second-order view

<figure>
<svg viewBox="0 0 640 170" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <path d="M30 140 C 120 -10, 200 -10, 290 140" fill="none" stroke="#12a150" stroke-width="2.5"/>
  <text x="160" y="160" text-anchor="middle" fill="#12a150">convex: one global min</text>
  <path d="M350 120 C 390 40, 420 150, 460 90 C 500 30, 540 150, 610 60" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="480" y="160" text-anchor="middle" fill="#e0533f">non-convex: saddles, plateaus, many minima</text>
</svg>
<figcaption>Deep learning lives on the right. A Hessian's eigenvalue signs classify a critical point: all positive = min, all negative = max, mixed = saddle.</figcaption>
</figure>

A Newton step $\theta_{t+1}=\theta_t-H_t^{-1}g_t$ uses curvature to size each direction — powerful, but $H$ is $d\times d$ (infeasible for $d\sim10^9$), noisy under minibatching, and indefinite in non-convex regions. So we use *structured* approximations: Adam's diagonal preconditioner, K-FAC, Shampoo/SOAP, and matrix-orthogonalization methods (**Muon** and kin). In high dimensions the practical enemy is rarely a bad local minimum (which requires *all* directions to curve up at once — measure-small); it's **saddle points and plateaus**, which SGD noise and momentum help escape.

> [!WARNING] 2026 tone on new optimizers
> AdamW remains the default. Matrix optimizers like Muon claim token-efficiency gains, but ICLR-2026-era benchmark analyses show that against a *properly tuned* AdamW baseline the real speedup often shrinks to ~1.1–1.4×, not 2×. When comparing, insist on (1) a fairly tuned AdamW, (2) reproduction at scale, (3) matched hyperparameter budgets — sharing HPs across optimizers is the classic way to manufacture an illusory win.

## Gradient clipping

$$
g\leftarrow g\cdot\min\!\Big(1,\frac{c}{\|g\|_2}\Big)\qquad(\text{global-norm clip})
$$

Used in RNNs, long sequences, and LLM pretraining ($c$ often $1.0$) to survive loss spikes and FP16 overflow. Global-norm is preferred over per-element value clipping, which can change the update direction. **If clipping fires every step, your LR is too high** — monitor the clip fraction as a diagnostic.

## Batch size effects

Minibatch gradient variance scales like $1/B$. Bigger batches → less noise, more stable, but past a "critical batch size" you get diminishing returns and need care with generalization. Rules of thumb: the **linear scaling rule** ($\eta'=k\eta$ for $k\times$ batch) is the starting point, with **longer warmup**; if it destabilizes, back off toward $\sqrt k$ scaling. With gradient accumulation, sum/average micro-batch gradients and step once — and match the LR to the *effective* batch. See [Probability & Statistics](#/foundations/probability-statistics) for the variance argument.

## Interview Q&A

<details class="qa"><summary>Why does momentum converge faster than plain SGD?</summary>
<div class="qa-body">

**Short:** it accumulates a velocity that cancels oscillation across the steep (high-curvature) axis and reinforces the consistent direction down the valley, so the effective step in the useful direction grows.

**Deep:** in an ill-conditioned quadratic, SGD's rate degrades with the condition number $\kappa$; heavy-ball momentum improves the dependence (roughly $\kappa\to\sqrt\kappa$ in the ideal quadratic case). The velocity is an EMA of gradients, so it low-pass-filters minibatch noise too. Too-large $\mu$ overshoots and can diverge; Nesterov's look-ahead gradient reduces that overshoot.
</div></details>

<details class="qa"><summary>Adam vs AdamW — what exactly differs and why care?</summary>
<div class="qa-body">

**Short:** Adam-with-L2 folds decay into the gradient, so the adaptive $1/\sqrt{\hat v}$ scaling makes decay uneven across coordinates; AdamW applies decay directly to the weights, decoupled from the preconditioner.

**Deep:** high-second-moment coordinates get their decay divided down under Adam+L2, so the effective regularization is not the $\ell_2$ you intended. AdamW's $-\eta\lambda\theta$ term restores uniform shrinkage and, empirically, better generalization and cleaner LR/WD tuning. It's the Transformer/ViT/LLM default; decay is typically switched off for biases and LayerNorm/RMSNorm gains.
</div></details>

<details class="qa"><summary>Why warmup, then cosine?</summary>
<div class="qa-body">

**Short:** warmup prevents early blow-ups from an oversized first step (small $\hat v$, noisy early gradients); cosine anneals smoothly to a small LR for stable late-stage convergence with almost no tuning.

**Deep:** at step 1 Adam's second-moment estimate is tiny, so the effective step $\eta\hat m/\sqrt{\hat v}$ can be enormous; ramping $\eta$ from 0 over a few hundred/thousand steps tames it. Large batches and large models amplify early gradient noise, making warmup more important, not less — even Pre-LN Transformers usually benefit. Cosine (vs step) removes the "which epoch do I drop?" hyperparameter and avoids the abrupt dynamics-breaking of a sharp step drop.
</div></details>

<details class="qa"><summary>Why don't we use full second-order methods in deep learning?</summary>
<div class="qa-body">

**Short:** the Hessian is $d\times d$ — too big to store or invert at billions of parameters, noisy under minibatching, and indefinite in non-convex regions.

**Deep:** even a Hessian-vector product (cheap via Pearlmutter's trick) doesn't make full Newton affordable per step, and near saddles a raw Newton step moves the *wrong way* along negative-curvature directions (needs damping/trust-region). So we settle for structured curvature: diagonal (Adam), Kronecker-factored (K-FAC), or spectral/orthogonalization (Shampoo, Muon). The interview point isn't "I can't do Newton" — it's "first-order + adaptive preconditioning wins at scale, and here's why."
</div></details>

<details class="qa"><summary>Training diverged at step 300. What do you check, in order?</summary>
<div class="qa-body">

**Short:** LR/warmup → grad-norm & clip fraction → mixed-precision/loss-scale → bad batch (NaN/inf inputs, label scale) → init/norm/residual → distributed (wrong LR scaling, bad all-reduce).

**Deep:** plot loss and grad-norm for the first few hundred steps; a spike says lower the peak LR or lengthen warmup. If clipping fires constantly, the LR is too high. On FP16, verify dynamic loss scaling (prefer BF16 on modern hardware, see [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency)). Rule out data (infinities, mis-scaled targets, pathological augmentation) and structural bugs (missing Pre-LN/RMSNorm, bad residual scaling — see [Normalization & Stability](#/foundations/normalization-stability)). In distributed runs, a mis-scaled LR or a bad reduction is a classic culprit. The headline: divergence is usually LR/numerics/data, not the optimizer choice.
</div></details>

**Follow-ups you should expect**

- *Effect of decreasing $\beta_2$?* Second moment adapts faster → more responsive on non-stationary objectives, less stable.
- *Batch size $\times k$ → LR?* Linear scaling as a start, longer warmup; back off to $\sqrt k$ if unstable.
- *LR finder?* Sweep $\eta$ exponentially for a few hundred steps, pick just below divergence; noisy, and often replaced by proxy-model runs at LLM scale.
- *Saddle vs plateau vs vanishing gradient?* Mixed-sign curvature vs near-zero curvature vs exponentially shrinking signal through depth.
- *Sharp vs flat minima?* Flat is *hypothesized* to generalize better (motivates SAM), but the definition is reparameterization-sensitive — hold it as contested.

## Cheat-sheet

| Ask | One-liner |
| --- | --- |
| SGD → momentum | velocity = EMA of grads; cancels oscillation, accelerates the valley. |
| Adam | momentum + per-coord $1/\sqrt{\hat v}$ + bias correction; $\beta=(0.9,0.999)$. |
| Adam vs AdamW | decoupled decay ⇒ uniform shrinkage; default for Transformers. |
| No-decay params | biases, LayerNorm/RMSNorm gains. |
| Warmup | ramps LR from 0 to avoid huge early Adam steps / noisy grads. |
| Cosine | smooth anneal, fewest knobs; warmup+cosine is the default. |
| 2nd order | $H$ too big/noisy/indefinite ⇒ use diagonal/KFAC/Muon approximations. |
| Grad clip | global-norm to $c{\approx}1$; always-on ⇒ LR too high. |
| Batch size | var $\sim1/B$; linear-scale LR + longer warmup; watch critical batch. |
| New optimizers 2026 | tune AdamW fairly first; real gains often ~1.1–1.4×, not 2×. |

**Related:** [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus) · [Regularization & Generalization](#/foundations/regularization-generalization) · [Normalization & Stability](#/foundations/normalization-stability) · [Distributed Training](#/foundations/distributed-training) · [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) · [The 2026 Landscape](#/start/landscape-2026)
