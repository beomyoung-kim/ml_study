# Optimization

<div class="tag-row"><span class="tag">SGD & momentum</span><span class="tag">Adam / AdamW</span><span class="tag">LR schedules</span><span class="tag">warmup</span><span class="tag">grad clipping</span><span class="tag">batch size</span></div>

> [!TIP] 그들이 진짜로 파고드는 것
> Adam update를 암송하는 능력이 아니라 — *왜* momentum이 가속하는지, *왜* AdamW가 weight decay를 decouple하는지, warmup+cosine이 *무엇을* 벌어주는지, 그리고 run이 diverge할 때 *가장 먼저 무엇을 확인하는지*를 설명하는 능력입니다. "divergence 대부분은 optimizer의 마법이 아니라 update rule과 numerical stability + data health가 설명한다"고 말하면, 대규모 model을 실제로 학습해 본 사람처럼 들립니다.

## 먼저 직접 만져보기

step size, momentum, 그리고 갇히는 현상에 대한 직관을 가장 빨리 쌓는 방법은 슬라이더를 움직여 보는 것입니다.

<div class="widget" data-widget="gradient-descent"></div>

## SGD에서 Adam까지 네 단계로

$g_t=\nabla_\theta L_t(\theta_t)$를 minibatch gradient라고 합시다.

**1. SGD** — 현재 gradient를 따라갑니다:
$$\theta_{t+1}=\theta_t-\eta\,g_t$$

**2. Momentum (heavy-ball)** — velocity(gradient의 EMA)를 누적합니다:
$$v_{t+1}=\mu v_t+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_{t+1}\quad(\mu\approx0.9)$$
좁은 ravine에서는 진동하는 성분이 상쇄되는 동안 골짜기 아래로 향하는 일관된 방향은 누적됩니다 → 더 큰 effective step, 더 빠른 수렴. **Nesterov**는 look-ahead 지점 $\theta_t-\eta\mu v_t$에서 gradient를 평가하는데, 이 작은 보정이 optimum 근처에서 안정성을 높입니다.

**3. RMSprop** — 각 좌표를 최근 gradient 크기로 정규화합니다:
$$v_t=\rho v_{t-1}+(1-\rho)g_t^{\odot 2},\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t$$

**4. Adam** — momentum *과* 좌표별 스케일링, 그리고 bias correction:
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^{\odot2}
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$
기본값 $\beta_1{=}0.9,\ \beta_2{=}0.999,\ \epsilon{=}10^{-8}$. Bias correction은 초기 몇 step에서 $m,v$의 zero-initialization bias를 되돌립니다.

| | 1st moment | 2nd moment | bias corr. | default use |
| --- | --- | --- | --- | --- |
| SGD+mom | velocity | — | — | CNN classification, still strong |
| RMSprop | — | ✓ | usually no | legacy / RNN niche |
| Adam / AdamW | ✓ | ✓ | ✓ | Transformers, ViT, LLM, fine-tuning |

## Adam vs AdamW — decoupled weight decay

loss에 L2를 더하면 $g_t\leftarrow \nabla L+\lambda\theta_t$가 됩니다. 그런데 Adam은 이후 $\sqrt{\hat v_t}$로 나누므로, 큰 second moment를 가진 좌표는 **더 약한** decay를 받습니다 — regularization이 좌표별로 왜곡되는 것이죠. AdamW는 parameter를 preconditioner *바깥에서* decay시켜 이를 고칩니다:

$$
\theta_{t+1}=\theta_t-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\Big)-\eta\lambda\theta_t
$$

> [!NOTE] 한 줄로 말하면
> "AdamW는 gradient preconditioning 바깥에서 weight decay를 적용하므로, 의도한 $\ell_2$ shrinkage가 모든 좌표에 균일하게 작용한다." 실무에서는: Transformer/ViT/LLM fine-tuning의 기본값이고, 팀들은 보통 bias와 norm gain($\gamma,\beta$)에 대해 **decay를 끕니다**. $\lambda$와 $\eta$는 상호작용한다는 점에 유의하세요 — cosine이 $\eta$를 줄이면 effective decay도 줄어들므로 둘을 함께 튜닝해야 합니다. 순수 SGD에서는 L2와 weight decay가 (거의) 동등합니다; 이 구분은 adaptive optimizer에서만 특히 중요합니다.

## Learning-rate schedule

Warmup은 초기 불안정성을 피합니다(Adam의 작은 초기 $\hat v$는 거대한 step을 낼 수 있고, large-batch / large-model run은 초기 gradient가 noisy합니다). Cosine은 매끄럽게 decay하여 hyperparameter 몇 개만으로 후반의 정밀한 탐색을 합니다. Step decay는 단순하지만 milestone이 까다롭습니다.

<div class="widget" data-widget="lr-schedule"></div>

$$
\text{warmup: } \eta_t=\eta_\text{peak}\frac{t}{T_w}\quad(t\le T_w)
$$
$$
\text{cosine: } \eta_t=\eta_\text{min}+\tfrac12(\eta_\text{peak}-\eta_\text{min})\Big(1+\cos\pi\frac{t-T_w}{T-T_w}\Big)
$$

**Warmup + cosine (+ 선택적 cooldown)**은 현대 CV/LLM의 기본값입니다. 다른 선택지: inverse-sqrt(고전적 Transformer), one-cycle(fast.ai), SGDR 스타일 warm restart(단일 cosine보다 프로덕션에서 드묾).

## Convexity & second-order 관점

<figure>
<svg viewBox="0 0 640 170" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <path d="M30 140 C 120 -10, 200 -10, 290 140" fill="none" stroke="#12a150" stroke-width="2.5"/>
  <text x="160" y="160" text-anchor="middle" fill="#12a150">convex: one global min</text>
  <path d="M350 120 C 390 40, 420 150, 460 90 C 500 30, 540 150, 610 60" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="480" y="160" text-anchor="middle" fill="#e0533f">non-convex: saddles, plateaus, many minima</text>
</svg>
<figcaption>Deep learning은 오른쪽에 삽니다. Hessian의 eigenvalue 부호가 critical point를 분류합니다: 전부 양수 = min, 전부 음수 = max, 혼합 = saddle.</figcaption>
</figure>

Newton step $\theta_{t+1}=\theta_t-H_t^{-1}g_t$는 curvature를 이용해 각 방향의 크기를 정합니다 — 강력하지만 $H$는 $d\times d$이고($d\sim10^9$에서는 불가능), minibatching 하에서 noisy하며, non-convex 영역에서는 indefinite합니다. 그래서 우리는 *구조화된* 근사를 씁니다: Adam의 diagonal preconditioner, K-FAC, Shampoo/SOAP, matrix-orthogonalization 방법(**Muon** 계열). 고차원에서 실질적인 적은 나쁜 local minimum이 아니라(그건 *모든* 방향이 동시에 위로 휘어야 하는데 — measure가 작음), **saddle point와 plateau**이며, SGD noise와 momentum이 이를 탈출하는 데 도움을 줍니다.

> [!WARNING] 새 optimizer에 대한 2026년의 톤
> AdamW는 여전히 기본값입니다. Muon 같은 matrix optimizer는 token 효율 이득을 주장하지만, ICLR-2026 시기의 benchmark 분석에 따르면 *제대로 튜닝된* AdamW baseline과 비교하면 실제 speedup은 2×가 아니라 종종 ~1.1–1.4×로 줄어듭니다. 비교할 때는 (1) 공정하게 튜닝된 AdamW, (2) 대규모에서의 재현, (3) 맞춰진 hyperparameter 예산을 요구하세요 — optimizer 간에 HP를 공유하는 것은 허구의 승리를 만들어내는 전형적인 방법입니다.

## Gradient clipping

$$
g\leftarrow g\cdot\min\!\Big(1,\frac{c}{\|g\|_2}\Big)\qquad(\text{global-norm clip})
$$

RNN, 긴 sequence, LLM pretraining($c$는 흔히 $1.0$)에서 loss spike와 FP16 overflow를 견디기 위해 씁니다. Global-norm은 update 방향을 바꿀 수 있는 per-element value clipping보다 선호됩니다. **clipping이 매 step 발동하면 LR이 너무 높은 것입니다** — clip fraction을 진단 지표로 모니터하세요.

## Batch size 효과

Minibatch gradient의 variance는 $1/B$처럼 스케일됩니다. 큰 batch → 적은 noise, 더 안정적, 하지만 "critical batch size"를 넘어서면 수익이 감소하고 generalization에 주의가 필요합니다. 경험칙: **linear scaling rule**($k\times$ batch에 $\eta'=k\eta$)이 출발점이고, **더 긴 warmup**을 함께 씁니다; 불안정해지면 $\sqrt k$ 스케일링 쪽으로 물러납니다. Gradient accumulation에서는 micro-batch gradient를 합/평균하고 한 번 step하며 — LR을 *effective* batch에 맞추세요. variance 논증은 [Probability & Statistics](#/foundations/probability-statistics)를 참고하세요.

## Interview Q&A

<details class="qa"><summary>Momentum은 왜 순수 SGD보다 빨리 수렴하는가?</summary>
<div class="qa-body">

**Short:** 가파른(high-curvature) 축을 가로지르는 진동을 상쇄하고 골짜기 아래로 향하는 일관된 방향을 강화하는 velocity를 누적하므로, 유용한 방향의 effective step이 커집니다.

**Deep:** ill-conditioned quadratic에서 SGD의 속도는 condition number $\kappa$에 따라 나빠집니다; heavy-ball momentum은 이 의존성을 개선합니다(이상적인 quadratic에서 대략 $\kappa\to\sqrt\kappa$). velocity는 gradient의 EMA이므로 minibatch noise도 low-pass filtering합니다. 너무 큰 $\mu$는 overshoot하고 diverge할 수 있는데, Nesterov의 look-ahead gradient가 그 overshoot을 줄입니다.
</div></details>

<details class="qa"><summary>Adam vs AdamW — 정확히 무엇이 다르고 왜 신경 쓰는가?</summary>
<div class="qa-body">

**Short:** Adam-with-L2는 decay를 gradient에 접어 넣으므로 adaptive $1/\sqrt{\hat v}$ 스케일링이 좌표별로 decay를 불균등하게 만듭니다; AdamW는 preconditioner와 decouple된 채로 weight에 직접 decay를 적용합니다.

**Deep:** high-second-moment 좌표는 Adam+L2 하에서 decay가 나눠 줄어들어, effective regularization이 의도한 $\ell_2$가 아니게 됩니다. AdamW의 $-\eta\lambda\theta$ 항은 균일한 shrinkage를 복원하고, 경험적으로 더 나은 generalization과 더 깔끔한 LR/WD 튜닝을 줍니다. Transformer/ViT/LLM의 기본값이며; decay는 보통 bias와 LayerNorm/RMSNorm gain에 대해 꺼둡니다.
</div></details>

<details class="qa"><summary>왜 warmup 다음에 cosine인가?</summary>
<div class="qa-body">

**Short:** warmup은 과도하게 큰 첫 step(작은 $\hat v$, noisy한 초기 gradient)에서 오는 초기 폭발을 막고; cosine은 매끄럽게 작은 LR로 anneal하여 거의 튜닝 없이 후반의 안정적 수렴을 줍니다.

**Deep:** step 1에서 Adam의 second-moment estimate는 아주 작으므로 effective step $\eta\hat m/\sqrt{\hat v}$가 엄청날 수 있습니다; $\eta$를 수백/수천 step에 걸쳐 0에서 올리면 이를 길들입니다. 큰 batch와 큰 model은 초기 gradient noise를 증폭하므로 warmup이 덜이 아니라 더 중요해집니다 — Pre-LN Transformer조차 보통 이득을 봅니다. Cosine(vs step)은 "어느 epoch에서 떨어뜨리지?" hyperparameter를 없애고, 가파른 step drop이 dynamics를 갑자기 깨뜨리는 것을 피합니다.
</div></details>

<details class="qa"><summary>Deep learning에서 왜 full second-order 방법을 쓰지 않는가?</summary>
<div class="qa-body">

**Short:** Hessian은 $d\times d$입니다 — 수십억 parameter에서 저장하거나 역행렬을 구하기엔 너무 크고, minibatching 하에서 noisy하며, non-convex 영역에서 indefinite합니다.

**Deep:** Hessian-vector product(Pearlmutter 트릭으로 저렴)조차 step당 full Newton을 감당 가능하게 만들지 못하고, saddle 근처에서 raw Newton step은 negative-curvature 방향을 따라 *잘못된 쪽*으로 움직입니다(damping/trust-region 필요). 그래서 우리는 구조화된 curvature로 만족합니다: diagonal(Adam), Kronecker-factored(K-FAC), spectral/orthogonalization(Shampoo, Muon). 면접에서의 핵심은 "Newton을 못 한다"가 아니라 — "first-order + adaptive preconditioning이 대규모에서 이기며, 그 이유는 이것이다"입니다.
</div></details>

<details class="qa"><summary>Step 300에서 학습이 diverge했다. 어떤 순서로 확인하는가?</summary>
<div class="qa-body">

**Short:** LR/warmup → grad-norm & clip fraction → mixed-precision/loss-scale → 나쁜 batch(NaN/inf input, label scale) → init/norm/residual → distributed(잘못된 LR 스케일링, 나쁜 all-reduce).

**Deep:** 첫 수백 step의 loss와 grad-norm을 plot하세요; spike는 peak LR을 낮추거나 warmup을 늘리라는 신호입니다. Clipping이 계속 발동하면 LR이 너무 높은 것입니다. FP16에서는 dynamic loss scaling을 검증하세요(현대 하드웨어에서는 BF16 선호, [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) 참고). 데이터(무한대, 잘못 스케일된 target, 병적인 augmentation)와 구조적 버그(Pre-LN/RMSNorm 누락, 나쁜 residual 스케일링 — [Normalization & Stability](#/foundations/normalization-stability) 참고)를 배제하세요. Distributed run에서는 잘못 스케일된 LR이나 나쁜 reduction이 전형적인 범인입니다. 요지: divergence는 보통 optimizer 선택이 아니라 LR/numerics/data입니다.
</div></details>

**예상해야 할 follow-up**

- *$\beta_2$를 줄이면?* Second moment가 더 빨리 적응 → non-stationary objective에 더 반응적, 덜 안정적.
- *Batch size $\times k$ → LR?* 시작은 linear scaling에 더 긴 warmup; 불안정하면 $\sqrt k$로 물러남.
- *LR finder?* 수백 step 동안 $\eta$를 지수적으로 sweep하고 divergence 직전을 선택; noisy하며, LLM 규모에서는 종종 proxy-model run으로 대체됨.
- *Saddle vs plateau vs vanishing gradient?* 혼합 부호 curvature vs 거의 0인 curvature vs depth를 통해 지수적으로 줄어드는 신호.
- *Sharp vs flat minima?* Flat이 더 잘 generalize한다고 *가설*됨(SAM의 동기), 그러나 정의가 reparameterization에 민감함 — 논쟁 중인 것으로 두세요.

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
