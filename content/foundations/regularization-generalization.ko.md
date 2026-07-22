# Regularization & 일반화

<div class="tag-row"><span class="tag">일반화</span><span class="tag">bias–variance</span><span class="tag">L1/L2 & weight decay</span><span class="tag">dropout</span><span class="tag">early stopping</span><span class="tag">data aug</span><span class="tag">double descent</span></div>

> [!NOTE] 이 챕터의 목표
> [머신러닝이란?](#/foundations/what-is-ml)에서 학습의 진짜 목표는 "외우기가 아니라 **일반화(generalization)**"라고 했습니다. 이 챕터는 모델이 훈련 데이터를 통째로 외워버리는 **과대적합(overfitting)** 을 어떻게 알아채고, 어떤 도구(regularization, 정칙화)로 막는지를 그림·코드로 다룹니다.

## 과대적합이 무엇이고 왜 싸우나

모델을 너무 오래/너무 크게 학습하면, 데이터의 진짜 패턴뿐 아니라 **잡음(noise)** 까지 외웁니다. 그러면 훈련 데이터에선 완벽하지만 처음 보는 데이터에선 무너집니다. 그래서 데이터를 셋으로 나눕니다:

<dl class="kv">
<dt>훈련(train)</dt><dd>모델이 학습에 쓰는 데이터.</dd>
<dt>검증(validation)</dt><dd>학습 중 성능을 재고 하이퍼파라미터를 고르는 데이터. 모델이 직접 학습하진 않음.</dd>
<dt>테스트(test)</dt><dd>마지막에 딱 한 번, 진짜 일반화 성능을 재는 데이터. 반복해서 보면 안 됨.</dd>
</dl>

### Data leakage — 성능을 가짜로 올리는 가장 흔한 버그

**Data leakage(데이터 누수)** 는 실제 예측 시점에는 알 수 없는 정보가 학습·모델 선택 과정에 들어가는 일입니다. 높은 validation score가 배포 성능으로 이어지지 않는다면 regularization보다 먼저 split과 feature 생성 과정을 감사하세요.

| 누수 유형 | 잘못된 예 | 안전한 설계 |
| --- | --- | --- |
| Target leakage | 퇴원 후 생성된 진단 코드로 입원 시 예후 예측 | 각 feature의 **생성 시점**이 예측 시점 이전인지 확인 |
| Split leakage | 전체 데이터로 평균·표준편차·PCA·feature selection을 fit | 먼저 split하고, 전처리는 train에만 `fit`; pipeline/CV fold 안에서 수행 |
| Entity leakage | 같은 환자·사용자·영상의 near-duplicate가 train과 val에 동시 존재 | entity/group 단위 split, perceptual hash·ID로 중복 검사 |
| Temporal leakage | 미래 데이터를 섞은 random split, centered rolling statistic | 시간순 split; feature는 해당 시점까지의 과거만 사용 |
| Augmentation leakage | split 전에 원본을 증강해 변형본이 양쪽에 들어감 | 원본을 먼저 split한 뒤 train에만 stochastic augmentation |
| Validation overfitting | 같은 val에 수백 번 튜닝한 뒤 그 점수를 최종 성능으로 보고 | 독립 test를 봉인; 작은 데이터의 모델 선택 편향은 nested CV 고려 |

Oversampling·SMOTE, vocabulary 구축, 결측값 대치, target encoding도 모두 **학습 데이터에 맞춰지는 변환**입니다. scikit-learn이라면 `Pipeline` 안에 넣고 cross-validation을 돌려야 각 fold의 validation 정보가 fit에 섞이지 않습니다. 단, production에서 실제로 이용 가능한 사전학습 통계나 외부 데이터까지 무조건 누수인 것은 아닙니다. 핵심 질문은 **“배포 시점에 이 정보가 존재하는가, 평가 집단에서 독립적인가?”** 입니다.

> **개념 코드 — split을 먼저, fit은 fold 안에서**

```python
train_idx, test_idx = group_split(entity_id)       # 같은 사용자는 한쪽에만
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

pipeline = Pipeline([
    ("impute", Imputer()),                         # 아직 fit하지 않음
    ("scale", StandardScaler()),
    ("model", Classifier()),
])
scores = cross_validate(pipeline, X_train, y_train) # 매 fold train에만 fit
pipeline.fit(X_train, y_train)
final_score = metric(y_test, pipeline.predict(X_test))  # 봉인 test는 마지막 1회
```

핵심 신호는 **훈련 성능과 검증 성능의 벌어짐(generalization gap, 일반화 격차)** 입니다. 이 격차가 벌어지기 시작하는 순간이 과대적합의 신호입니다.

<figure>
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="55" y1="210" x2="590" y2="210" stroke="#98a3b2" stroke-width="1.4"/>
  <line x1="55" y1="20" x2="55" y2="210" stroke="#98a3b2" stroke-width="1.4"/>
  <text x="320" y="238" text-anchor="middle" fill="#98a3b2">학습이 진행됨 (epoch) →</text>
  <text x="30" y="115" fill="#98a3b2" transform="rotate(-90 30 115)">손실(loss)</text>
  <!-- train loss: monotonic down -->
  <path d="M60 60 C 180 150, 320 185, 585 200" fill="none" stroke="#0ea5e9" stroke-width="2.5"/>
  <text x="470" y="195" fill="#0ea5e9">훈련 손실 ↓ (계속 내려감)</text>
  <!-- val loss: down then up -->
  <path d="M60 75 C 190 150, 250 158, 300 158 C 400 158, 480 120, 585 55" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="120" y="55" fill="#e0533f">검증 손실 (내려갔다 다시 ↑)</text>
  <!-- early stopping line at val min -->
  <line x1="300" y1="35" x2="300" y2="210" stroke="#12a150" stroke-width="1.6" stroke-dasharray="5 4"/>
  <circle cx="300" cy="158" r="5" fill="#12a150"/>
  <text x="308" y="45" fill="#12a150">← 여기서 멈춰라 (early stopping)</text>
  <!-- overfit region shading label -->
  <text x="450" y="150" text-anchor="middle" fill="#98a3b2" font-size="11">이 구간 = 과대적합</text>
</svg>
<figcaption>훈련 손실은 계속 내려가지만 검증 손실은 어느 순간 다시 올라갑니다. 그 <b>최저점</b>이 일반화가 가장 좋은 지점 — 여기서 학습을 멈추는 것이 <b>early stopping(조기 종료)</b> 입니다.</figcaption>
</figure>

> [!TIP] 면접 한 줄
> Regularization은 "$\lambda\|\theta\|^2$ 항을 더하는 것"이 아니라 **일반화 격차를 체계적으로 줄이는 활동**입니다. "데이터 · 모델 · 최적화 중 어떤 레버를 먼저 당길지, 무슨 신호를 보고 그렇게 하는지"를 말할 수 있으면 강한 답입니다. weakly/semi-supervised, continual learning은 근본적으로 일반화 문제입니다.

## Bias–variance: 과소적합 vs 과대적합의 언어

먼저 직관입니다. 예측이 틀리는 데는 두 가지 서로 다른 이유가 있습니다:

- **Bias(편향)가 높다** = 모델이 **너무 단순**해서 패턴 자체를 못 잡음 → **과소적합(underfitting)**. (예: 곡선 데이터에 직선)
- **Variance(분산)가 높다** = 모델이 **너무 민감**해서 데이터가 조금만 바뀌어도 예측이 출렁임 → **과대적합**. (훈련 잡음까지 외움)

목표는 이 둘의 **균형**입니다. 수식으로는, 제곱오차의 기대값이 정확히 셋으로 쪼개집니다:

$$
\mathbb E\big[(y-\hat f(x))^2\big]=\underbrace{\text{Bias}^2}_{\text{너무 경직}}+\underbrace{\text{Variance}}_{\text{너무 민감}}+\underbrace{\text{Noise}}_{\text{줄일 수 없음}}
$$

<figure>
<svg viewBox="0 0 560 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="50" y1="180" x2="530" y2="180" stroke="#98a3b2"/><line x1="50" y1="20" x2="50" y2="180" stroke="#98a3b2"/>
  <text x="290" y="202" text-anchor="middle" fill="#98a3b2">모델 복잡도(capacity) →</text>
  <text x="16" y="100" fill="#98a3b2" transform="rotate(-90 16 100)">error</text>
  <path d="M60 40 C 160 120, 260 168, 520 176" fill="none" stroke="#0ea5e9" stroke-width="2"/>
  <text x="150" y="60" fill="#0ea5e9">bias² (단순할수록 큼)</text>
  <path d="M60 176 C 260 170, 380 120, 520 30" fill="none" stroke="#e0533f" stroke-width="2"/>
  <text x="440" y="60" fill="#e0533f">variance (복잡할수록 큼)</text>
  <path d="M60 90 C 200 70, 240 66, 300 84 C 380 108, 460 70, 520 44" fill="none" stroke="#12a150" stroke-width="2.5"/>
  <text x="300" y="56" fill="#12a150" text-anchor="middle">test error (고전적 U)</text>
</svg>
<figcaption>고전적 U-curve: 너무 단순하면 bias, 너무 복잡하면 variance. Deep net은 이 곡선을 구부리지만(아래 double descent), "과소적합 vs 과대적합" 언어는 여전히 가장 빠른 디버깅 도구입니다.</figcaption>
</figure>

**현대적 반전:** 파라미터가 극단적으로 많은(**over-parameterized**) 신경망은 훈련 데이터를 완벽히 외우고도 일반화를 잘합니다. SGD·초기화·augmentation이 눈에 안 보이는 곳에서 variance를 통제하는 **암묵적 정칙화(implicit regularization)** 로 작동하기 때문입니다(아래 심화). 그래도 이 표는 여전히 가장 빠른 진단표입니다:

| 훈련 | 검증 | 진단 | 첫 번째 레버 |
| --- | --- | --- | --- |
| 나쁨 | 나쁨 | 과소적합 / 버그 / 나쁜 LR | 더 큰 모델, 더 긴 학습, 버그 확인 |
| 좋음 | 나쁨 | 과대적합 / 누수 / shift | augmentation, weight decay, 데이터 추가 |
| 좋음 | 좋음 | 건강함 | 배포 후 drift 모니터 |
| 나쁨 | 좋음 | 거의 확실히 metric/누수 버그 | eval 파이프라인 감사 |

> [!TIP] Tiny-overfit 테스트 (가장 먼저 할 것)
> Regularization에 손대기 전에, 정칙화를 끄고 모델이 **작은 batch를 거의 외울 수 있는지** 확인하세요. 충분한 capacity와 적절한 목적함수인데도 못 외우면 data loader, label, train/eval 모드, learning rate, 얼어붙은 파라미터를 우선 의심합니다. 다만 label noise·확률적 target·강한 augmentation·작은 모델·최적화 난이도 때문에 loss가 정확히 0이 아닐 수도 있으므로, 실패 하나만으로 버그를 증명하지는 못합니다. [gradient-descent 위젯](#/foundations/optimization)으로 step size의 영향을 확인하세요.

## 과대적합을 막는 도구 상자

<dl class="kv">
<dt>L2 / weight decay(가중치 감쇠)</dt><dd>모든 weight를 조금씩 0 쪽으로 당김(Gaussian prior, <a href="#/foundations/probability-statistics">확률 & 통계</a> 참고). 주의: Adam+L2 ≠ AdamW의 decoupled decay — <a href="#/foundations/optimization">Optimization</a> 참고.</dd>
<dt>L1</dt><dd>sparsity(희소성)를 유도합니다. subgradient SGD의 유한 step에서는 정확한 0이 보장되지 않으며, 좌표를 정확히 0으로 만드는 성질은 soft-thresholding을 쓰는 proximal 방법에서 가장 명확합니다. 배포용 압축은 보통 structured pruning도 함께 검토.</dd>
<dt>Dropout(드롭아웃)</dt><dd>학습 중 뉴런을 무작위로 꺼서(0으로) 뉴런들이 서로 의존(co-adaptation)하지 못하게 함 → 여러 얇은 망의 ensemble 효과.</dd>
<dt>Early stopping(조기 종료)</dt><dd>검증 지표가 더 안 좋아지면 멈춤. 위 그림의 초록 지점.</dd>
<dt>Data augmentation(데이터 증강)</dt><dd>입력을 변형(뒤집기·자르기·색변화)해 데이터를 넓힘 → CV에서 종종 가장 가성비 높은 regularizer.</dd>
<dt>Label smoothing(라벨 스무딩)</dt><dd>one-hot 정답을 살짝 뭉갬($1-\varepsilon$ / $\tfrac{\varepsilon}{K-1}$) → 과신을 줄일 수 있습니다. 정확도·calibration 개선은 데이터와 설정에 따라 달라 별도로 측정해야 합니다.</dd>
</dl>

**L1 vs L2 기하 직관:** 같은 예산 안에서 손실을 줄일 때, $\ell_1$ 제약은 마름모(diamond)라 **꼭짓점(=0인 좌표)** 에 먼저 닿아 sparsity가 생기고, $\ell_2$는 둥근 공이라 모든 좌표를 골고루 줄입니다(dense shrink).

### 직접 구현: inverted dropout

가장 헷갈리는 dropout의 "스케일링"을 직접 만들어 봅시다. 학습 시 살아남은 뉴런을 $1/(1-p)$로 키워두면(inverted dropout), **추론 시에는 아무것도 안 해도** 기대값이 맞습니다. 아래 랩에서 마스크가 주어졌을 때의 dropout을 구현하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"apply_dropout","packages":["numpy"],"approx":true,"starter":"def apply_dropout(x, mask, p):\n    # x: 입력 배열, mask: 살아남을 자리=1 / 꺼질 자리=0 (같은 shape), p: drop 확률.\n    # inverted dropout: 꺼진 자리는 0, 살아남은 자리는 1/(1-p) 배로 스케일.\n    # 반환: 리스트.\n    import numpy as np\n    x = np.asarray(x, float); mask = np.asarray(mask, float)\n    # TODO\n    return out.tolist()","tests":[{"args":[[1.0,2.0,3.0,4.0],[1,0,1,0],0.5],"expect":[2.0,0.0,6.0,0.0]},{"args":[[10.0,10.0],[1,1],0.5],"expect":[20.0,20.0]},{"args":[[5.0,5.0,5.0,5.0],[1,1,1,1],0.2],"expect":[6.25,6.25,6.25,6.25]}],"solution":"import numpy as np\n\ndef apply_dropout(x, mask, p):\n    x = np.asarray(x, float); mask = np.asarray(mask, float)\n    out = x * mask / (1.0 - p)\n    return out.tolist()"}
</script>
</div>

세 번째 테스트: $p=0.2$면 살아남은 값은 $5/(1-0.2)=6.25$로 커집니다. 이렇게 학습 때 미리 키워두므로 추론 때 전체 망을 그대로 써도 크기가 맞습니다.

**Data augmentation이 regularization인 이유:** $x$ 대신 변형 $T(x)$를 보여주면 모델이 그 변형에 **불변(invariance)** 하도록 배웁니다(vicinal risk 최소화). 스펙트럼: 기본(flip/crop/jitter) → 강함(RandAugment/TrivialAugment) → 섞기(MixUp/CutMix/Copy-Paste). 너무 강하면 오히려 과소적합하거나 라벨을 파괴합니다. Segmentation에선 기하 변형을 마스크에도 똑같이 적용해야 하고, 색 변형은 그러면 안 됩니다.

## 암묵적 정칙화 (심화)

모든 regularizer가 손실 식에 명시되는 건 아닙니다. **옵티마이저 자체**가 훈련 손실 0인 해가 여럿일 때 어디에 착지할지를 편향시킵니다:

- **SGD의 minibatch noise**는 해 선택에 영향을 주며 평평한 해와 연관된다는 경험적·이론적 결과가 있습니다. 다만 작은 batch의 잡음이 언제나 이롭거나 작은 norm을 보장하는 것은 아닙니다.
- **선형 분리 가능 데이터의 로지스틱 회귀**처럼 특정 조건에서는 gradient descent가 명시적 penalty 없이도 방향상 max-margin 해로 수렴합니다. 모든 신경망에 그대로 일반화되는 정리는 아닙니다.
- **Early stopping**은 학습 궤적을 제한하는 정칙화입니다. 선형 least squares 등 제한된 설정에서는 $\ell_2$와 연결할 수 있지만 일반 신경망에서 둘이 동치인 것은 아닙니다.
- **아키텍처**도 inductive bias를 줍니다 — conv의 locality·weight sharing, equivariant 구조, bottleneck, normalization 등입니다.

실무 요점: 큰 모델이 파라미터 수에도 불구하고 일반화한다면, 대개 당신이 코드로 안 짠 일을 암묵적 정칙화가 해주고 있는 것입니다.

## Double descent (심화)

```mermaid
flowchart LR
  A["파라미터 부족<br/>고전적 U-curve"] --> B["interpolation 임계점<br/>error가 정점"]
  B --> C["파라미터 과잉<br/>두 번째 하강: error 다시 ↓"]
```

Test error는 interpolation 임계점(train error → 0) 근처에서 정점을 찍은 뒤, capacity가 더 커지면 **다시 떨어질 수 있습니다**(Belkin et al.; Nakkiran et al.). *epoch 단위* 버전도 있어 순진한 early stopping과 충돌할 수 있습니다. 면접 안전 프레이밍:

> "double-descent 곡선을 매일 그리진 않지만 교훈은 남습니다: capacity를 줄이는 것만이 일반화로 가는 유일한 길이 아니며, 데이터·regularization·학습 예산을 함께 봐야 한다."

## Calibration & 일반화 (심화)

모델은 정확도(accuracy)는 높으면서도 **calibration이 나쁠**(confidence ≠ 실제 정답률) 수 있습니다. Temperature scaling은 고정된 validation set에서 logit scale을 맞추는 대표적 post-hoc 방법입니다. Label smoothing은 과신을 줄일 수 있지만 calibration을 항상 개선하지는 않습니다. discrimination과 calibration을 별도로 측정하세요. 자세히는 [평가 지표](#/foundations/evaluation-metrics).

## 면접 Q&A

<details class="qa"><summary>L1과 L2를 기하까지 포함해 대비하라.</summary>
<div class="qa-body">

**Short:** L2는 모든 weight를 매끄럽게 줄이고(dense, Gaussian prior), L1은 일부를 정확히 0으로 만듭니다(sparse, Laplace prior).

**Deep:** L2의 gradient는 $\propto\theta$라 비례적으로만 줄여 좌표를 0으로 강제하지 못합니다. L1의 subgradient는 크기가 일정해 좌표를 0으로 고정할 수 있습니다. 기하적으로 손실 등고선이 $\ell_1$ 마름모의 축 꼭짓점에서 먼저 만납니다. 딥러닝에서 L2는 weight decay로 나타나고(Adam/AdamW 주의), 배포용 sparsity는 보통 L1 학습이 아니라 structured pruning에서 옵니다.
</div></details>

<details class="qa"><summary>Dropout은 무엇을 하고, train과 inference는 왜 다른가?</summary>
<div class="qa-body">

**Short:** train 시 뉴런을 무작위로 꺼 co-adaptation을 막고 ensemble을 근사하며, inference에선 전체 망을 씁니다(inverted dropout이라 rescaling 불필요 — 학습이 이미 $1/(1-p)$로 스케일).

**Deep:** 각 minibatch가 서로 다른 "얇은" subnet을 학습한다는 ensemble 직관이 유용합니다. 강한 augmentation을 쓰는 큰 데이터셋에선 dropout을 줄이거나 없애기도 합니다(과잉 정칙화 위험). CV 전용 변형(SpatialDropout, DropBlock)은 인접 픽셀이 상관되어 있어 연속 영역을 통째로 drop합니다. Transformer는 attention/FFN dropout과 **stochastic depth(DropPath)** 를 씁니다. 추론 때 dropout을 켠 채 여러 번 표본화하는 **MC-Dropout**은 근사적 epistemic uncertainty 신호이지만, 보정된 불확실도를 보장하지는 않습니다.
</div></details>

<details class="qa"><summary>과잉 파라미터 신경망에서 bias–variance를 설명하라.</summary>
<div class="qa-body">

**Short:** 고전적 U-curve는 capacity↑ ⇒ bias↓/variance↑라 하지만, 데이터를 외우는 거대 망도 일반화합니다. 암묵적 정칙화(SGD·init·augmentation)가 variance를 길들이기 때문입니다.

**Deep:** 그래도 이 분해는 디버깅을 이끕니다 — train≈val이 둘 다 나쁘면 high bias/capacity 부족 또는 최적화 실패, 큰 train–val 격차는 high variance/과대적합 → augmentation·decay·데이터. 깨지는 가정은 "파라미터 많음 = 과대적합 심함"입니다. scaling law와 double descent가 반대 영역을 보여줍니다. Ensemble은 상관 낮은 오차를 평균해 variance를 줄입니다.
</div></details>

<details class="qa"><summary>새 프로젝트에서 regularization을 어떤 순서로 적용하나?</summary>
<div class="qa-body">

**Short:** metric/eval 먼저 맞추고 → tiny-overfit 되는 버그 없는 baseline → augmentation → AdamW weight decay + cosine + early stop/EMA → 그다음 dropout/label-smoothing/DropPath를 ablation으로 → capacity 축소는 최후의 수단.

**Deep:** "dropout부터 켠다"는 anti-pattern입니다 — 효과를 귀속시킬 수 없죠. 하나씩 추가하고 ablate하세요. 같은 val set에 반복 튜닝해 생기는 간접적 test 과대적합을 조심하세요. 좌우명: regularization은 loss 항을 뿌리는 게 아니라 일반화 격차를 겨냥한 *시스템 설계* 활동입니다. [디버깅 & 실험](#/foundations/debugging-experimentation) 참고.
</div></details>

**예상 follow-up**

- *Elastic Net?* L1+L2 결합 — sparsity에 안정성을 더함.
- *TTA는 regularization?* 아님 — test-time augmentation은 추론 시 ensembling이지 학습 regularizer가 아님.
- *Ensemble은 어떤 항을 줄이나?* Variance(상관 낮은 오차 평균).
- *무한한 데이터면 regularize 불필요?* 최적화 안정성·효율·견고한 평가는 여전히 필요.
- *Fine-tuning에서 과잉 정칙화 증상?* train과 val이 둘 다 정체(과소적합), 격차가 벌어지는 게 아니라.

## Cheat-sheet

| Fact | 한 줄 |
| --- | --- |
| 목표 | penalty 추가가 아니라 일반화 격차 축소 |
| Tiny-overfit 테스트 | 정칙화를 끈 작은 batch도 못 외우면 pipeline·capacity·optimization을 우선 점검 |
| L1 vs L2 | L1 sparse(마름모 꼭짓점), L2 dense shrink(둥근 공) |
| Weight decay | Adam+L2 ≠ AdamW; decoupled decay 사용 |
| Dropout | train 시 얇게 ≈ ensemble; inverted면 추론 rescale 불필요 |
| Augmentation | 대개 가성비 최고; 마스크/박스도 일관되게 변형 |
| Early stopping | 소량 데이터에 강력; epoch 단위 double descent 주의 |
| Bias–variance | 여전한 디버깅 틀; 과잉 파라미터는 "많음=과대적합" 반례 |
| Double descent | interpolation 정점 이후 error가 다시 하강 가능 |

**다음:** [Optimization](#/foundations/optimization) · [평가 지표](#/foundations/evaluation-metrics) · [확률 & 통계](#/foundations/probability-statistics) · [디버깅 & 실험](#/foundations/debugging-experimentation) · [Weak & Semi-Supervised](#/cv/weak-semi-supervised)
