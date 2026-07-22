# 디버깅 & 실험

> [!NOTE] 이 챕터의 목표
> 첫 학습 코드를 돌리면 거의 항상 "loss가 안 내려가요" 또는 "왜 안 되는지 모르겠어요" 상황을 만납니다. 신경망은 **조용히 틀립니다** — 에러 없이 그냥 학습이 안 됩니다. 이 챕터는 그럴 때 찍어보지 않고 **체계적으로 원인을 좁히는 순서(playbook)** 를 줍니다. [머신러닝이란?](#/foundations/what-is-ml)의 학습 루프를 이미 돌려본 사람을 위한 실전 안내입니다.

> [!TIP] 면접 한 줄
> 좋은 첫 문장: *"먼저 augmentation과 regularization을 끄고 아주 작은 배치를 overfit할 수 있는지 확인해, 데이터·loss·역전파·optimizer 경로를 좁힙니다."* 실패가 버그의 강한 신호이기는 하지만 capacity나 optimization 문제도 가능하므로 증거를 하나씩 분리합니다.

## 첫 번째 무기: 한 배치를 과적합시켜라

무언가 안 될 때 가장 먼저 할 일은 학습률을 바꾸는 게 아닙니다. **아주 작은 데이터(예: 8개 샘플)를 잡아서 loss를 거의 0까지 몰아붙이는 것**입니다(Karpathy의 레시피). 이 한 가지 테스트가 문제를 절반으로 갈라 줍니다:

- **과적합에 성공** → 최소한 이 작은 샘플들에서 forward/backward/optimizer가 loss를 줄일 수 있다는 증거입니다. 전체 파이프라인의 누수·평가 오류까지 증명하지는 않습니다.
- **과적합에 실패** → label/loss mismatch, frozen parameter, 잘못된 mode 같은 버그부터 확인하되, 부족한 capacity, 너무 강한 regularization·augmentation, 부적절한 LR, stochastic/impossible target 가능성도 함께 점검합니다.

무엇보다 먼저 훑을 빠른 체크리스트:

1. `model.train()` / `eval()`이 올바르게 토글됐는지 — BN(배치 정규화)/dropout(드롭아웃)이 모드에 따라 동작이 다릅니다.
2. Loss가 유한한지 — `NaN`(숫자 아님)/`inf`(무한대)가 없어야 합니다.
3. Label 범위와 `ignore_index`가 올바른지.
4. 학습 가능한 파라미터 수 > 0, 그리고 LR ≠ 0.
5. 입력 정규화(mean/std)가 checkpoint와 일치하는지.
6. 한 배치를 **눈으로** 시각화 — image/mask/label 정렬 확인.

> [!NOTE] init 시 loss 값도 검산하세요
> 초기 logits가 모든 클래스에서 거의 같아 softmax가 균등하다면, class frequency와 무관하게 샘플당 cross-entropy는 $\ln C$ 근처입니다. 이 전제가 맞는데도 크게 다르면 label 범위, reduction, initialization, masking을 점검하세요. pretrained·biased head나 class prior로 초기화한 모델에는 이 검산을 그대로 적용하지 않습니다.

## loss 곡선 읽는 법 — 모양이 곧 진단

디버깅의 8할은 loss 곡선을 보는 눈입니다. 네 가지 전형적인 모양과 그 의미:

<figure>
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <!-- panel 1: healthy -->
  <text x="90" y="18" text-anchor="middle" fill="#12a150" font-weight="700">건강함</text>
  <line x1="30" y1="30" x2="30" y2="120" stroke="#98a3b2"/><line x1="30" y1="120" x2="160" y2="120" stroke="#98a3b2"/>
  <path d="M35 40 C 70 55, 110 100, 155 112" fill="none" stroke="#12a150" stroke-width="2.5"/>
  <text x="95" y="140" text-anchor="middle" fill="#98a3b2">완만히 감소</text>
  <!-- panel 2: diverging -->
  <text x="250" y="18" text-anchor="middle" fill="#e0533f" font-weight="700">발산</text>
  <line x1="190" y1="30" x2="190" y2="120" stroke="#98a3b2"/><line x1="190" y1="120" x2="320" y2="120" stroke="#98a3b2"/>
  <path d="M195 90 C 220 95, 245 90, 265 70 C 285 45, 300 35, 315 30" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="255" y="140" text-anchor="middle" fill="#98a3b2">LR ↑ / NaN 위험</text>
  <!-- panel 3: plateau -->
  <text x="410" y="18" text-anchor="middle" fill="#d97706" font-weight="700">정체</text>
  <line x1="350" y1="30" x2="350" y2="120" stroke="#98a3b2"/><line x1="350" y1="120" x2="480" y2="120" stroke="#98a3b2"/>
  <path d="M355 55 L475 55" fill="none" stroke="#d97706" stroke-width="2.5"/>
  <text x="415" y="140" text-anchor="middle" fill="#98a3b2">LR=0 / 얼어붙음</text>
  <!-- panel 4: overfit -->
  <text x="570" y="18" text-anchor="middle" fill="#6366f1" font-weight="700">과대적합</text>
  <line x1="510" y1="30" x2="510" y2="120" stroke="#98a3b2"/><line x1="510" y1="120" x2="635" y2="120" stroke="#98a3b2"/>
  <path d="M515 45 C 545 70, 575 100, 630 110" fill="none" stroke="#12a150" stroke-width="2"/>
  <path d="M515 55 C 545 80, 575 78, 630 60" fill="none" stroke="#e0533f" stroke-width="2" stroke-dasharray="4 3"/>
  <text x="572" y="140" text-anchor="middle" fill="#98a3b2">train↓ val↑</text>
  <text x="572" y="160" text-anchor="middle" fill="#12a150">train</text><text x="620" y="160" text-anchor="middle" fill="#e0533f">val</text>
</svg>
<figcaption>왼쪽부터: 건강한 하강 / 위로 튀는 발산(LR이 너무 큼 → NaN 임박) / 평평한 정체(LR=0, 파라미터 얼어붙음, 잘못된 loss) / train은 내려가는데 val이 올라가는 과대적합. 곡선 모양만 봐도 첫 대응이 갈립니다.</figcaption>
</figure>

## LR range test

Schedule을 확정하기 전에 **LR range test(학습률 범위 탐색)** 를 돌립니다: 수백 step에 걸쳐 LR을 지수적으로 올리며 loss vs. LR을 그립니다.

<figure>
<svg viewBox="0 0 520 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="50" y1="20" x2="50" y2="160" stroke="#98a3b2"/><line x1="50" y1="160" x2="490" y2="160" stroke="#98a3b2"/>
  <text x="30" y="90" fill="#98a3b2" transform="rotate(-90 30 90)">loss</text>
  <text x="260" y="185" fill="#98a3b2" text-anchor="middle">learning rate (log scale) →</text>
  <path d="M60 60 C 150 62, 200 70, 250 95 C 300 120, 330 135, 360 130 C 400 120, 430 60, 470 25" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <line x1="300" y1="20" x2="300" y2="160" stroke="#12a150" stroke-dasharray="4 3"/>
  <text x="300" y="16" fill="#12a150" text-anchor="middle">가장 가파른 하강 → 여기 선택</text>
  <text x="440" y="18" fill="#e0533f" text-anchor="middle">발산</text>
</svg>
<figcaption>좋은 LR은 대략 loss가 가장 가파르게 하강하는 지점으로, loss가 폭발하는 지점보다 한 자릿수 아래입니다. 찍는 것보다 싸고 신뢰할 만합니다.</figcaption>
</figure>

이것을 grad-norm(gradient 크기) 및 weight-norm 로깅과 짝지으면, NaN이 터지기 전에 불안정이 쌓이는 것을 눈으로 볼 수 있습니다(메커니즘은 [Normalization & 학습 안정성](#/foundations/normalization-stability)).

## 직접 해보기 — 노이즈 낀 loss 곡선 매끈하게 보기

실제 loss 곡선은 배치마다 튀어서 추세를 읽기 어렵습니다. 그래서 **이동 평균(moving average)** 으로 매끈하게 만들어 봅니다 — 디버깅 대시보드가 늘 하는 일입니다. 아래에서 창 크기 $k$의 이동 평균(각 위치에서 직전 $k$개의 평균)을 구현하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"moving_average","packages":["numpy"],"approx":true,"starter":"def moving_average(xs, k):\n    # xs: loss 값 리스트, k: 창 크기.  각 위치에서 '직전 k개'의 평균을 담은 리스트를 반환.\n    # 출력 길이는 len(xs) - k + 1.  예: moving_average([1,2,3,4], 2) -> [1.5, 2.5, 3.5]\n    pass","tests":[{"args":[[1,2,3,4],2],"expect":[1.5,2.5,3.5]},{"args":[[2,2,2,2,2],3],"expect":[2.0,2.0,2.0]},{"args":[[10,0,10,0],2],"expect":[5.0,5.0,5.0]}],"solution":"import numpy as np\n\ndef moving_average(xs, k):\n    x = np.asarray(xs, dtype=float)\n    out = [float(x[i:i+k].mean()) for i in range(len(x) - k + 1)]\n    return out"}
</script>
</div>

노이즈를 걷어내면 위 "loss 곡선 읽는 법"의 네 모양 중 어느 것인지 훨씬 또렷이 보입니다.

<details class="qa"><summary>Training loss가 안 내려갑니다. 처음 10분을 설명해 보세요.</summary>
<div class="qa-body">

**짧게:** 하나의 작은 batch를 overfit해 봅니다. 실패하면 파이프라인 버그(잘못된 loss reduction, frozen param, `eval()` 켜둠, label/target mismatch, LR=0). 성공하면 모델은 학습할 수 있고 문제는 scale/optimization입니다.

**깊게:** 순서화된 probe — (1) `requires_grad` param이 존재하고 `backward()` 후 grad가 non-zero인지 확인, (2) 같은 batch에서 loss를 반복 출력 — 0을 향해 떨어져야 함, (3) init 시 loss 값 sanity check(예: 균형 CE에 대해 $\ln(\text{num\_classes})$), (4) LR이 non-zero이고 optimizer가 step했는지 검증, (5) input+target을 시각화해 label을 손상시킨 transform 포착. "overfit one batch"가 통과한 뒤에만 LR schedule, capacity, regularization을 건드립니다. **후속 질문:** *overfit은 되는데 일반화가 안 되면?* — regularization/data 문제. [Regularization & 일반화](#/foundations/regularization-generalization) 참고.
</div></details>

## "Loss는 내려가는데 metric이 안 움직인다"

전형적인 경우이고, 진단은 결정 표(decision table)입니다:

| 관찰 | 가설 | 조치 |
| --- | --- | --- |
| train metric ↑, val metric flat | overfitting | augmentation, weight decay, 더 많은 데이터 |
| train metric도 flat | loss ≠ target metric | metric-aware loss, error analysis |
| val loss ↓ but val metric flat | threshold/decoding 문제 | NMS / threshold / post-proc 튜닝 |
| 둘 다 좋은데 deploy는 나쁨 | domain shift | target-distribution 데이터, recalibrate |
| random보다 나쁨 | eval 버그 | prediction 덤프, metric unit-test |

CV 예시: cross-entropy는 떨어지는데 **mIoU**가 그대로 → 모델이 그냥 background를 예측. KD(지식 증류) loss는 떨어지는데 student task metric은 떨어짐 → student가 teacher의 *오류*를 복사 중. 이것을 잡는 습관: training loss만이 아니라 **중간 품질 신호를 지켜보기**.

<details class="qa"><summary>Validation loss는 개선되는데 target metric이 그대로입니다. 무슨 일이죠?</summary>
<div class="qa-body">

**짧게:** loss는 metric과 분리된 surrogate입니다 — 보통 threshold/decoding mismatch, loss가 악용하는 class imbalance, 또는 eval 버그. prediction을 덤프하고 oracle upper bound와 비교해 진단합니다.

**깊게:** CE 같은 loss는 픽셀/token별 likelihood를 최적화하고, mIoU/mAP/F1 같은 metric은 set 또는 ranking 기반이며 threshold에 민감합니다. CE가 떨어지는데 mIoU가 그대로면, 모델이 쉬운 다수 픽셀을 이기는 반면 경계/희소 class는 정체됐을 가능성 — metric-aligned loss(Dice/Lovász/focal)로 고치고 per-class score를 점검하세요. 항상 ground truth를 prediction으로 넣어 harness를 sanity-check하세요: metric이 상한에 도달해야 합니다. 아니면 eval 코드가 깨진 것입니다. **후속 질문:** *어느 신호로 early stopping?* — surrogate loss가 아니라 최종 보고 metric.
</div></details>

## Reproducibility (재현성)

```python
import random, numpy as np, torch
random.seed(s); np.random.seed(s)
torch.manual_seed(s); torch.cuda.manual_seed_all(s)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # 속도를 내주고 결정성을 얻음
```

고정된 하드웨어·소프트웨어와 deterministic 구현이 있는 연산만 쓰면 bit-exact 실행이 가능할 수 있지만, 모든 연산·플랫폼 조합에서 보장되지는 않습니다. DataLoader generator와 worker별 Python/NumPy RNG도 함께 seed하고, 지원되지 않는 deterministic 연산은 명시적으로 실패하게 하세요. 연구 결론은 commit·환경·data snapshot·config를 고정하고 여러 seed의 paired 차이와 신뢰구간으로 확인합니다.

<details class="qa"><summary>왜 GPU에서 항상 bit-exact 재현성을 얻을 수 없나요?</summary>
<div class="qa-body">

**짧게:** floating-point 덧셈은 결합적이지 않고, GPU는 non-deterministic 순서로 합산합니다(atomic add, reduction 스케줄링). 그래서 동일한 입력이 학습마다 약간 다른 결과를 낼 수 있습니다 — TF32, autotune된 kernel, distributed all-reduce 순서가 이를 가중합니다.

**깊게:** `cudnn.benchmark=True`는 input shape별로 kernel을 autotune(빠르지만 non-deterministic)하고, `deterministic=True`는 속도 비용을 치르고 재현 가능한 kernel을 강제하며, 일부 연산은 deterministic 구현이 없습니다. rank 간 all-reduce는 partial sum을 미지정 순서로 결합합니다. 그래서 bit-exactness를 쫓는 건 보통 잘못된 목표입니다 — 대신 seed를 고정하고, commit/data/config를 로깅하며, **≥3 seed에 걸친 mean ± std**를 보고해 결론이 이 noise에 robust하게 하세요. **후속 질문:** *Dataset versioning?* — content hash / DVC / immutable snapshot 경로로 "데이터"를 고정.
</div></details>

## Experiment tracking (실험 추적)

**config**(hyperparameter, git hash, data version), **curve**(train/val loss, target metric, LR, grad-norm, weight-norm), **system**(GPU util, samples/s), **artifact**(checkpoint), **note**(가설 + 결론)를 로깅하세요. 학습 이름은 `YYYYMMDD_hypothesis_short`로 짓고 태그(`baseline`, `bugfix`, `sweep`)를 답니다. Anti-pattern: 이름 없는 수십 개의 학습, 한 머신에만 있는 best checkpoint, variance 없이 단일 test score 보고. *당신의 미래 논문 ablation table은 바로 이 로그에서 나옵니다.*

## Ablation discipline (제거 실험 규율)

<div class="proscons"><div><div class="pros-t">좋은 ablation</div>

- 명확하고 재현 가능한 **baseline**
- 한 번에 **한 factor**만 변경(또는 해석 가능한 factorial)
- **동일한 예산** — 모든 arm에 같은 epoch/data/tuning
- mean ± std를 가진 **multiple seed**
- 맥락을 위한 **oracle/upper bound**
- **claim**이 의존하는 module만

</div><div><div class="cons-t">나쁜 ablation</div>

- data + loss + architecture를 함께 변경
- **새 방법만 grid-search**
- **test set**에서 model selection
- 작은 toy 세팅에서만 gain
- variance 미보고
- 진짜 driver를 숨기는 table bloat

</div></div>

면접에서의 수: *"먼저 gain을 설명할 수 있는 축들을 나열한 뒤, 각각을 배제하도록 ablation을 설계합니다 — 예를 들어 개선이 새 loss에서 왔나, 아니면 그냥 더 큰 backbone에서 왔나?"* Negative result(부정적 결과)는 과학입니다: 실패 조건과 범위를 보고하세요. 상세 방법론은 [Experiment Design](#/research/experiment-design).

<details class="qa"><summary>Reviewer가 당신의 gain이 방법이 아니라 그냥 더 많은 compute에서 온 것일 수 있다고 합니다. 어떻게 답하나요?</summary>
<div class="qa-body">

**짧게:** **compute-matched** ablation을 보여줍니다 — baseline과 당신의 방법을 동일한 epoch, data, tuning 예산으로 학습 — 여기에 scaling curve를 더해 개선이 운 좋은 한 세팅이 아니라 여러 compute 지점에서 보이게 합니다.

**깊게:** 실패 모드는 새 방법의 hyperparameter만 튜닝하고 baseline은 stock으로 돌리는 것입니다. 이는 방법과 search 예산을 혼동시킵니다. Control: (1) equal-budget 학습과 arm당 equal HP search, (2) model size / token count에 걸쳐 같은 metric을 보고해 gap이 일관됨을 보임, (3) headroom을 bound하는 oracle upper bound, (4) delta가 noise를 넘도록 variance를 가진 ≥3 seed. compute-matching 하에서 gain이 사라지면, 그것은 정직하고 가치 있는 negative result입니다. **후속 질문:** *언제 ablation을 멈추나?* — marginal 정보 < 기회비용이거나, claim에 대한 causal story가 완성됐을 때.
</div></details>

## Quick scenarios (빠른 시나리오)

<dl class="kv">
<dt>NaN from step 0</dt><dd>Non-finite input, LR too high, FP16, `log(0)` → BF16, anomaly detect, batch 덤프.</dd>
<dt>Curve dead flat</dt><dd>LR=0, frozen param, 잘못된 loss, 나쁜 label → `requires_grad` 합산, manual forward.</dd>
<dt>Train perfect, val random</dt><dd>깨진 eval 또는 극단적 overfit/domain shift → val 시각화, split 확인.</dd>
<dt>Only multi-GPU degrades</dt><dd>Sampler, mean-vs-sum reduction, BN → 1-GPU parity test (<a href="#/foundations/distributed-training">Distributed Training</a> 참고).</dd>
<dt>Won't reproduce</dt><dd>Seed, data 순서, 몰래 섞인 bugfix → commit + container 고정.</dd>
</dl>

## Cheat-sheet

| 질문 | 한 줄 요약 |
| --- | --- |
| First move | augmentation/regularization을 끄고 작은 배치를 overfit해 원인 범위를 축소. 실패는 버그·capacity·optimization을 구분해 조사. |
| Init loss sanity | 초기 예측이 균등하면 CE는 $\ln(\text{num\_classes})$ 근처. class balance 조건은 불필요. |
| 곡선 모양 | 하강=건강 · 위로 튐=LR↑/NaN · 평평=LR0/얼어붙음 · train↓val↑=과대적합. |
| LR range test | LR을 올리며 sweep, loss plot. ~steepest descent, divergence보다 10× 아래를 선택. |
| Loss↓ metric flat | Surrogate ≠ metric, threshold, 또는 eval 버그. harness 테스트엔 GT를 넣기. |
| Reproducibility | bit-exact 말고 statistical(seed 전반 mean±std) + 고정된 commit/data/config를 목표. |
| Tracking | config/curve/system/artifact/note 로깅. 모든 학습에 이름 + 태그. |
| Ablation | 한 factor, 동일 예산, multiple seed, oracle. 새 방법만 튜닝하지 말 것. |
| Multi-GPU divergence | 1-GPU parity test로 sampler/reduction/sync 버그 격리. |
| Compute-matched | arm 전반에 epoch/data/tuning을 맞춰 gain이 그냥 예산이 아니게. |

**다음:** [Normalization & 학습 안정성](#/foundations/normalization-stability) · [Distributed Training](#/foundations/distributed-training) · [Optimization](#/foundations/optimization) · [Regularization & 일반화](#/foundations/regularization-generalization) · [평가 지표](#/foundations/evaluation-metrics) · [Experiment Design](#/research/experiment-design)
