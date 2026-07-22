# Optimization

> [!NOTE] 이 챕터의 목표
> [머신러닝이란?](#/foundations/what-is-ml)에서 "손실을 줄이는 방향으로 파라미터를 조금씩 옮긴다"고 했습니다. 그 "조금씩 옮기기"를 **잘** 하는 방법이 optimization(최적화)입니다. 이 챕터는 공을 언덕 아래로 굴리는 직관에서 시작해, 왜 momentum이 빨라지고 Adam이 표준이 되었는지, 그리고 학습이 터질 때 무엇을 먼저 보는지까지 이어집니다. 앞부분은 입문, 뒤부분(§4~)은 면접 심화입니다.

## §0 · 경사하강법 = 공을 언덕 아래로 굴리기

손실 함수를 **울퉁불퉁한 지형**이라고 생각하세요. 높이 = 손실(작을수록 좋음), 위치 = 파라미터 값. 우리는 이 지형에서 가장 낮은 골짜기를 찾고 싶습니다. 방법은 단순합니다: 지금 서 있는 자리에서 **가장 가파른 내리막 방향**(= gradient의 반대)으로 한 걸음 내딛기. 이걸 반복하는 것이 **경사하강법(gradient descent)** 입니다.

$$\theta_{t+1}=\theta_t-\eta\,g_t \qquad (g_t=\nabla_\theta L,\ \eta=\text{학습률})$$

한 걸음의 크기를 정하는 $\eta$가 **learning rate(학습률)** 입니다. 여기서 직관을 가장 빨리 얻는 방법은 직접 슬라이더를 움직여 보는 것입니다 — 학습률을 키우면 어떻게 발산(diverge)하고, 너무 작으면 얼마나 느린지 보세요.

<div class="widget" data-widget="gradient-descent"></div>

> [!TIP] 면접 한 줄
> 면접관이 진짜 확인하는 건 Adam 수식 암송이 아니라, *왜* momentum이 가속하는지, *왜* AdamW가 weight decay를 분리하는지, warmup+cosine이 *무엇을* 벌어주는지, 그리고 run이 diverge할 때 *가장 먼저 무엇을 보는지*입니다. "divergence 대부분은 optimizer의 마법이 아니라 update rule + numerical stability + data health가 설명한다"고 말하면 대규모 학습을 해 본 사람처럼 들립니다.

## §1 · SGD에서 Adam까지 네 단계로

실제 학습에서는 데이터 전체가 아니라 **minibatch(미니배치)**, 즉 데이터의 작은 묶음으로 gradient를 추정합니다. 이 gradient는 방향이 맞긴 하지만 노이즈가 있어서, 그대로 쓰면 골짜기를 향해 지그재그로 헤맵니다. 아래 네 단계는 그 지그재그를 점점 잘 다스리는 이야기입니다. $g_t=\nabla_\theta L_t(\theta_t)$를 minibatch gradient라 합시다.

**1. SGD (Stochastic Gradient Descent)** — 그냥 현재 gradient를 따라갑니다:
$$\theta_{t+1}=\theta_t-\eta\,g_t$$

**2. Momentum(관성)** — 과거 이동 방향을 velocity(속도)로 누적합니다. 언덕을 구르는 공이 관성을 갖는 것과 같습니다:
$$v_{t+1}=\mu v_t+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_{t+1}\quad(\mu\approx0.9)$$
좁은 골짜기에서 좌우로 튀는 성분은 서로 상쇄되고, 골짜기 아래로 향하는 일관된 방향은 쌓입니다 → 유효 보폭이 커지고 수렴이 빨라집니다. **Nesterov**는 한 발 앞 지점 $\theta_t-\eta\mu v_t$에서 gradient를 재서 overshoot을 줄입니다.

<figure>
<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- valley contours (elongated ellipses = ill-conditioned) -->
  <g fill="none" stroke="#98a3b2" stroke-width="1" opacity="0.5">
    <ellipse cx="330" cy="120" rx="290" ry="55"/>
    <ellipse cx="330" cy="120" rx="210" ry="38"/>
    <ellipse cx="330" cy="120" rx="130" ry="23"/>
    <ellipse cx="330" cy="120" rx="55" ry="11"/>
  </g>
  <text x="330" y="124" text-anchor="middle" fill="#12a150" font-size="11">최소점 ●</text>
  <circle cx="330" cy="120" r="4" fill="#12a150"/>
  <!-- SGD zig-zag path (red) -->
  <polyline points="60,70 100,168 150,78 200,160 250,86 300,150 340,96 370,138" fill="none" stroke="#e0533f" stroke-width="2"/>
  <text x="90" y="55" fill="#e0533f">SGD: 좌우로 튐 (지그재그)</text>
  <!-- momentum smooth path (indigo) -->
  <path d="M60 190 Q 180 175, 260 150 T 330 122" fill="none" stroke="#6366f1" stroke-width="2.5"/>
  <text x="120" y="212" fill="#6366f1">momentum: 진동 상쇄 → 곧장 내려감</text>
</svg>
<figcaption>가늘고 긴 골짜기(ill-conditioned)에서 SGD(빨강)는 가파른 벽을 따라 좌우로 튀며 느리게 내려가고, momentum(파랑)은 그 진동을 상쇄해 골짜기를 따라 곧장 최소점으로 향합니다.</figcaption>
</figure>

**3. RMSprop** — 좌표(방향)마다 최근 gradient 크기로 나눠 보폭을 맞춥니다(가파른 방향은 작게, 평평한 방향은 크게):
$$v_t=\rho v_{t-1}+(1-\rho)g_t^{\odot 2},\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{v_t}+\epsilon}g_t$$

**4. Adam** — momentum(1차 모멘트) *과* 좌표별 스케일링(2차 모멘트), 그리고 bias correction(초기 보정)을 합칩니다. 오늘날 사실상의 기본값입니다:
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^{\odot2}
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$
기본값 $\beta_1{=}0.9,\ \beta_2{=}0.999,\ \epsilon{=}10^{-8}$. bias correction은 $m,v$를 0에서 시작해 생기는 초기 몇 step의 편향을 되돌립니다.

| | 1차 모멘트 | 2차 모멘트 | bias corr. | 주 사용처 |
| --- | --- | --- | --- | --- |
| SGD+momentum | velocity | — | — | CNN 분류, 여전히 강력 |
| RMSprop | — | ✓ | 보통 X | 레거시 / RNN 틈새 |
| Adam / AdamW | ✓ | ✓ | ✓ | Transformer, ViT, LLM, fine-tuning |

### 직접 구현 — momentum 한 스텝

말로 들은 momentum 업데이트를 코드로 옮겨 봅시다. velocity를 누적하고 파라미터를 갱신하는 한 스텝을 구현하세요. (스칼라 버전이지만 벡터도 똑같습니다.)

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sgd_momentum_step","packages":["numpy"],"approx":true,"starter":"def sgd_momentum_step(w, grad, v, lr=0.1, mu=0.9):\n    # momentum 한 스텝: v_new = mu*v + grad ; w_new = w - lr*v_new\n    # [w_new, v_new] 를 리스트로 반환하세요.\n    pass","tests":[{"args":[1.0,2.0,0.0,0.1,0.9],"expect":[0.8,2.0]},{"args":[0.8,1.0,2.0,0.1,0.9],"expect":[0.52,2.8]},{"args":[0.0,0.0,1.0,0.5,0.5],"expect":[-0.25,0.5]}],"solution":"def sgd_momentum_step(w, grad, v, lr=0.1, mu=0.9):\n    v_new = mu * v + grad\n    w_new = w - lr * v_new\n    return [w_new, v_new]"}
</script>
</div>

첫 스텝에선 $v=0$이라 SGD와 같지만, 스텝이 쌓이면 velocity가 커져(2.0 → 2.8 …) 일관된 방향으로 가속하는 게 보입니다.

## §2 · Adam vs AdamW — decoupled weight decay

loss에 L2 벌점을 더하면 gradient가 $g_t\leftarrow \nabla L+\lambda\theta_t$가 됩니다. 그런데 Adam은 이후 $\sqrt{\hat v_t}$로 나누므로, 2차 모멘트가 큰 좌표는 **decay를 덜** 받습니다 — regularization(정규화)이 좌표마다 왜곡되는 것이죠. **AdamW**는 weight decay(가중치 감쇠)를 preconditioner *바깥에서* 적용해 이를 바로잡습니다:

$$
\theta_{t+1}=\theta_t-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\Big)-\eta\lambda\theta_t
$$

> [!NOTE] 한 줄로 말하면
> "AdamW는 gradient preconditioning 바깥에서 weight decay를 적용하므로, 의도한 $\ell_2$ shrinkage가 모든 좌표에 균일하게 작용한다." 실무 기본값은 Transformer/ViT/LLM fine-tuning이고, 보통 bias와 norm gain($\gamma,\beta$)에는 **decay를 끕니다**. $\lambda$와 $\eta$는 상호작용합니다 — cosine이 $\eta$를 줄이면 유효 decay도 줄어드니 함께 튜닝하세요. 순수 SGD에서는 L2와 weight decay가 (거의) 같고, 이 구분은 adaptive optimizer에서만 특히 중요합니다.

## §3 · Learning-rate schedule

**warmup(워밍업)** 은 초기 representation과 gradient statistics가 빠르게 변하는 구간에서 update-to-weight ratio를 제한해 불안정을 줄입니다. bias correction을 쓰는 Adam의 첫 step이 단지 $\hat v$가 작다는 이유만으로 무한히 커지는 것은 아닙니다. 큰 batch는 sampling noise를 줄이지만, 큰 peak LR·모델 폭·분산 학습 레시피와 함께 쓰일 때 초기 안정화가 더 중요해질 수 있습니다. **cosine(코사인 감쇠)** 은 후반 LR을 매끄럽게 줄입니다.

<div class="widget" data-widget="lr-schedule"></div>

$$
\text{warmup: } \eta_t=\eta_\text{peak}\frac{t}{T_w}\quad(t\le T_w)
$$
$$
\text{cosine: } \eta_t=\eta_\text{min}+\tfrac12(\eta_\text{peak}-\eta_\text{min})\Big(1+\cos\pi\frac{t-T_w}{T-T_w}\Big)
$$

**Warmup + cosine (+ 선택적 cooldown)** 은 현대 CV/LLM의 기본값입니다. 다른 선택지: inverse-sqrt(고전 Transformer), one-cycle(fast.ai), SGDR warm restart(단일 cosine보다 프로덕션에서 드묾).

## §4 · Convexity & second-order 관점 (심화)

<figure>
<svg viewBox="0 0 640 170" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <path d="M30 140 C 120 -10, 200 -10, 290 140" fill="none" stroke="#12a150" stroke-width="2.5"/>
  <text x="160" y="160" text-anchor="middle" fill="#12a150">convex: global min 하나</text>
  <path d="M350 120 C 390 40, 420 150, 460 90 C 500 30, 540 150, 610 60" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="480" y="160" text-anchor="middle" fill="#e0533f">non-convex: saddle · plateau · 여러 minima</text>
</svg>
<figcaption>stationary point에서 Hessian이 positive definite면 strict local minimum, negative definite면 strict local maximum, 양·음 고유값이 섞이면 saddle입니다. 0 고유값이 있으면 이 검사만으로는 결론을 낼 수 없습니다.</figcaption>
</figure>

Newton step $\theta_{t+1}=\theta_t-H_t^{-1}g_t$는 curvature를 이용하지만 $H$는 $d\times d$이고 minibatch에서는 noisy하며 non-convex에서는 indefinite할 수 있습니다. K-FAC·Shampoo/SOAP은 curvature 또는 행렬 구조를 활용하는 근사이고, Adam은 gradient second moment를 쓰는 diagonal preconditioner, Muon은 matrix update를 orthogonalize하는 optimizer입니다. 모두를 "Hessian 근사"라고 부르지는 마세요. 고차원에서는 saddle·plateau와 나쁜 conditioning도 중요한 난점입니다.

> [!WARNING] 새 optimizer에 대한 2026년의 톤
> AdamW는 여전히 기본값입니다. Muon 같은 matrix optimizer는 token 효율 이득을 주장하지만, *제대로 튜닝된* AdamW baseline과 비교하면 실제 speedup은 2×가 아니라 종종 ~1.1–1.4×로 줄어듭니다. 비교할 때는 (1) 공정하게 튜닝된 AdamW, (2) 대규모 재현, (3) 맞춰진 hyperparameter 예산을 요구하세요 — optimizer 간 HP 공유는 허구의 승리를 만드는 전형적 방법입니다.

## §5 · Gradient clipping

$$
g\leftarrow g\cdot\min\!\Big(1,\frac{c}{\|g\|_2}\Big)\qquad(\text{global-norm clip})
$$

RNN, 긴 sequence, LLM pretraining($c$는 흔히 $1.0$인 출발점)에서 gradient spike를 제한하기 위해 씁니다. Global-norm은 각 성분을 따로 자르는 clipping보다 원래 gradient 방향을 보존합니다. **거의 매 step clipping이 발동하면 진단 신호**입니다 — LR·warmup, loss/gradient scaling, outlier batch, exploding path 또는 구현 버그를 함께 점검하고 clip fraction과 unclipped norm을 모니터하세요.

## §6 · Batch size 효과

독립 표본 근사에서 minibatch gradient variance는 대략 $1/B$로 줄어듭니다. 그러나 batch-coupled layer, 중복·상관 샘플, sampling 방식에 따라 달라집니다. **Linear LR scaling은 특정 optimizer·학습 구간에서 유용한 출발점이지 법칙이 아닙니다.** batch를 바꿀 때는 optimizer, token/step 예산, schedule과 함께 실험으로 확인하세요. Gradient accumulation도 BN 통계의 micro-batch 크기를 키우지는 않습니다.

## 면접 Q&A

<details class="qa"><summary>Momentum은 왜 순수 SGD보다 빨리 수렴하는가?</summary>
<div class="qa-body">

**Short:** 가파른(high-curvature) 축의 진동을 상쇄하고 골짜기 아래로 향하는 일관된 방향을 강화하는 velocity를 누적하므로, 유용한 방향의 유효 보폭이 커집니다.

**Deep:** ill-conditioned quadratic에서 SGD 속도는 condition number $\kappa$에 따라 나빠지고, heavy-ball momentum은 이 의존성을 개선합니다(이상적 quadratic에서 대략 $\kappa\to\sqrt\kappa$). velocity는 gradient의 EMA라 minibatch noise도 low-pass filtering합니다. $\mu$가 너무 크면 overshoot·diverge할 수 있는데 Nesterov의 look-ahead가 이를 줄입니다.
</div></details>

<details class="qa"><summary>Adam vs AdamW — 정확히 무엇이 다르고 왜 신경 쓰는가?</summary>
<div class="qa-body">

**Short:** Adam-with-L2는 decay를 gradient에 접어 넣어 adaptive $1/\sqrt{\hat v}$ 스케일링이 좌표별로 decay를 불균등하게 만듭니다; AdamW는 preconditioner와 decouple된 채 weight에 직접 decay를 적용합니다.

**Deep:** high-second-moment 좌표는 Adam+L2에서 decay가 나눠 줄어들어 유효 regularization이 의도한 $\ell_2$가 아니게 됩니다. AdamW의 $-\eta\lambda\theta$ 항이 균일 shrinkage를 복원하고, 경험적으로 더 나은 generalization과 깔끔한 LR/WD 튜닝을 줍니다. Transformer/ViT/LLM 기본값이며, decay는 보통 bias와 LayerNorm/RMSNorm gain에 대해 꺼둡니다.
</div></details>

<details class="qa"><summary>왜 warmup 다음에 cosine인가?</summary>
<div class="qa-body">

**Short:** warmup은 초기의 급격한 representation·gradient 변화에서 update 크기를 제한하고, cosine은 후반 LR을 매끄럽게 줄입니다. 둘 다 유용한 기본 후보지만 항상 최적인 것은 아닙니다.

**Deep:** bias-corrected Adam의 첫 step에서 $\hat m/\sqrt{\hat v}$는 대략 gradient의 부호 크기이므로 "작은 $v$ 때문에 무조건 거대한 step"이라는 설명은 틀립니다. 실제 초기 불안정은 parameter scale 대비 update, 아직 안정되지 않은 activation/gradient statistics, 큰 peak LR과 수치 정밀도의 상호작용에서 옵니다. warmup은 이 구간을 낮은 LR로 통과하게 합니다.
</div></details>

<details class="qa"><summary>Step 300에서 학습이 diverge했다. 어떤 순서로 확인하는가?</summary>
<div class="qa-body">

**Short:** LR/warmup → grad-norm & clip fraction → mixed-precision/loss-scale → 나쁜 batch(NaN/inf input, label scale) → init/norm/residual → distributed(잘못된 LR 스케일링, 나쁜 all-reduce).

**Deep:** 첫 수백 step의 loss와 grad-norm을 plot하세요. Spike가 있으면 peak LR·warmup을 먼저 점검하되, clipping이 계속 발동하는 원인은 LR뿐 아니라 loss scaling, outlier batch, 잘못 스케일된 target이나 구조적 버그일 수 있습니다. FP16이면 dynamic loss scaling을 검증하세요([Mixed Precision & 효율화](#/foundations/mixed-precision-efficiency)). 데이터의 `inf`/잘못된 target/병적 augmentation과 normalization·residual scaling 문제도 배제하세요. 요지: optimizer 이름을 바꾸기 전에 LR·numerics·data·구현을 분리 진단하는 것입니다.
</div></details>

**예상 follow-up**

- *$\beta_2$를 줄이면?* 2차 모멘트가 더 빨리 적응 → non-stationary objective에 더 반응적, 덜 안정적.
- *Batch size $\times k$ → LR?* 시작은 linear scaling + 더 긴 warmup; 불안정하면 $\sqrt k$로.
- *LR finder?* 수백 step 동안 $\eta$를 지수적으로 sweep하고 divergence 직전을 선택; noisy하며 LLM 규모에선 proxy-model run으로 대체.
- *Sharp vs flat minima?* flat이 더 잘 generalize한다는 *가설*(SAM의 동기), 그러나 정의가 reparameterization에 민감 — 논쟁 중.

## Cheat-sheet

| Ask | 한 줄 |
| --- | --- |
| 경사하강법 | 손실 지형에서 gradient 반대(내리막)로 한 걸음씩 |
| SGD → momentum | velocity = gradient의 EMA; 진동 상쇄, 골짜기 가속 |
| Adam | momentum + 좌표별 $1/\sqrt{\hat v}$ + bias correction; $\beta=(0.9,0.999)$ |
| Adam vs AdamW | decoupled decay ⇒ 균일 shrinkage; Transformer 기본값 |
| no-decay 대상 | bias, LayerNorm/RMSNorm gain |
| warmup | LR을 0에서 올려 초기 거대 step / noisy grad 방지 |
| cosine | 매끄러운 anneal, knob 최소; warmup+cosine이 기본값 |
| 2차 방법 | $H$가 너무 크고/noisy/indefinite ⇒ diagonal/KFAC/Muon 근사 |
| grad clip | global-norm $c{\approx}1$; 항상 발동 ⇒ LR 과다 |
| batch size | var $\sim1/B$; linear-scale LR + 더 긴 warmup; critical batch 주의 |

**다음:** [Regularization & 일반화](#/foundations/regularization-generalization) · [선형대수 & 미적분](#/foundations/linear-algebra-calculus) · [Normalization & 학습 안정성](#/foundations/normalization-stability) · [분산 학습](#/foundations/distributed-training) · [Mixed Precision & 효율화](#/foundations/mixed-precision-efficiency)
