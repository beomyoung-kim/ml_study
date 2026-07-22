# Continual Learning (연속 학습)

<div class="tag-row"><span class="tag">catastrophic forgetting</span><span class="tag">stability–plasticity</span><span class="tag">regularization / replay / isolation</span><span class="tag">class-incremental seg</span><span class="tag">prompt tuning</span><span class="tag">ECLIPSE</span></div>

> [!NOTE] 이 챕터의 목표
> 사람은 새것을 배워도 옛것을 웬만해선 안 잊습니다. 그런데 신경망은 새 task를 학습하면 **예전 task를 급격히 잊어버립니다**(catastrophic forgetting, 파국적 망각). 이 챕터는 왜 그런지, 그리고 어떻게 막는지를 그림과 직관으로 잡습니다. [전이학습·백본 freeze](#/cv/backbones-transfer)와 [Segmentation](#/cv/segmentation)을 먼저 봤다면 매끄럽습니다.

## 무엇을 / 왜

**상황:** 데이터가 한꺼번에 오지 않고 시간에 따라 조금씩 옵니다. step 1에서는 "고양이·개"를, step 2에서는 "자동차·버스"를 배웁니다. 그런데 step 2를 학습하면 모델이 고양이·개를 거의 못 맞히게 됩니다 — 공유하던 가중치(weight)를 새 클래스에 맞춰 덮어썼기 때문입니다.

**왜 어려운가:** 예전 데이터를 다시 저장해 두고 같이 학습하면 강한 기준선이 되지만, 저장 공간·프라이버시·비용 때문에 제한될 수 있습니다. **제한된 과거 접근 하에서 옛 성능과 새 학습을 함께 관리하는 것**이 continual learning의 목표입니다. Continual, incremental, lifelong learning은 문헌에서 겹쳐 쓰이지만 task boundary·streaming·label-space 가정이 서로 다를 수 있으므로 protocol을 먼저 정의합니다.

<figure>
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- axes -->
  <line x1="55" y1="215" x2="600" y2="215" stroke="#98a3b2" stroke-width="1.5"/>
  <line x1="55" y1="25" x2="55" y2="215" stroke="#98a3b2" stroke-width="1.5"/>
  <text x="330" y="245" text-anchor="middle" fill="#98a3b2">학습 step (새 클래스가 계속 추가됨) →</text>
  <text x="20" y="120" fill="#98a3b2" transform="rotate(-90 20 120)">Task 1 정확도</text>
  <!-- x ticks -->
  <g fill="#98a3b2" text-anchor="middle"><text x="120" y="232">1</text><text x="240" y="232">2</text><text x="360" y="232">3</text><text x="480" y="232">4</text><text x="560" y="232">5</text></g>
  <!-- naive fine-tuning: steep drop (red) -->
  <polyline points="120,45 240,150 360,190 480,205 560,210" fill="none" stroke="#e0533f" stroke-width="3"/>
  <circle r="5" fill="#e0533f"><animateMotion dur="3s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" path="M120,45 L240,150 L360,190 L480,205 L560,210"/></circle>
  <text x="470" y="200" fill="#e0533f">순진한 fine-tuning → 망각</text>
  <!-- CL method: stays high (green) -->
  <polyline points="120,45 240,55 360,62 480,68 560,72" fill="none" stroke="#12a150" stroke-width="3"/>
  <circle r="5" fill="#12a150"><animateMotion dur="3s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" path="M120,45 L240,55 L360,62 L480,68 L560,72"/></circle>
  <text x="300" y="48" fill="#12a150">좋은 CL 방법 → 유지</text>
</svg>
<figcaption>파국적 망각의 핵심 그림: Task 1을 배운 뒤 새 task들을 계속 학습할 때, 순진한 fine-tuning(빨강)은 Task 1 정확도가 급락하지만, 좋은 continual learning 방법(초록)은 이를 높게 유지하면서 새것도 배웁니다.</figcaption>
</figure>

> [!TIP] 면접 한 줄
> 지원자는 여기서 두 편의 1저자/공동저자 논문을 갖고 있습니다: **SSUL**(NeurIPS 2021, exemplar class-incremental semantic seg) → **ECLIPSE**(CVPR 2024, visual prompt tuning으로 continual *panoptic* seg). 강한 답은 일반적인 **stability–plasticity** 프레이밍 + segmentation 특유의 **background/no-object shift**를 엮고, *distillation-free·prompt-based* 방법이 replay/KD를 언제 이기는지 논증하는 것입니다.

## 1 · 파국적 망각 & 안정성–가소성

<strong>Catastrophic forgetting(파국적 망각):</strong> 새 task를 학습하면 과거 task를 인코딩하던 공유 weight를 덮어써서, 과거 task 정확도가 무너지는 현상. 이는 **stability–plasticity dilemma(안정성–가소성 딜레마)의** 첨예한 끝입니다.

- **안정성(stability):** 배운 것을 지키는 힘. 너무 크면(예: 완전 freeze) 새것을 못 배움.
- **가소성(plasticity):** 새것을 배우는 유연함. 너무 크면(예: 자유 fine-tuning) 옛것을 잊음.

<figure>
<svg viewBox="0 0 620 130" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="60" y1="70" x2="560" y2="70" stroke="#98a3b2" stroke-width="2"/>
  <circle cx="60" cy="70" r="6" fill="#0ea5e9"/><text x="60" y="100" text-anchor="middle" fill="#0ea5e9">최대 stability</text>
  <text x="60" y="42" text-anchor="middle" fill="#98a3b2">freeze: 망각 없음,</text>
  <text x="60" y="28" text-anchor="middle" fill="#98a3b2">학습도 없음</text>
  <circle cx="560" cy="70" r="6" fill="#e0533f"/><text x="560" y="100" text-anchor="middle" fill="#e0533f">최대 plasticity</text>
  <text x="560" y="42" text-anchor="middle" fill="#98a3b2">fine-tune: 새것 학습,</text>
  <text x="560" y="28" text-anchor="middle" fill="#98a3b2">옛것 망각</text>
  <circle cx="360" cy="70" r="7" fill="#12a150"/><text x="360" y="100" text-anchor="middle" fill="#12a150">좋은 CL 방법</text>
</svg>
<figcaption>너무 stable하면 새 클래스를 못 배우고, 너무 plastic하면 잊습니다. 방법이 이 축의 어디에 있는지 보려면 base / new / all 지표를 <b>따로</b> 보고하세요.</figcaption>
</figure>

Segmentation에는 픽셀 라벨 누락과 background 의미 변화라는 추가 난점이 있습니다. 이 때문에 동일 조건의 classification보다 어려울 수 있지만, 망각 크기는 dataset·protocol·model에 따라 측정해야 합니다(§4).

## 2 · 세 가지 방법군

| 방법군 | 아이디어 | 대표 | 비용 / 약점 |
| --- | --- | --- | --- |
| **Regularization(정규화)** | 중요한 weight/출력을 보호 | EWC, LwF, MiB, PLOP | 제약끼리 충돌; 긴 시퀀스에서 성능 저하 |
| **Replay(재현)** | 옛 샘플/특징을 다시 학습(rehearsal) | iCaRL, RECALL, exemplar set | 저장 공간 + **프라이버시**; panoptic 설계 어려움 |
| **Parameter isolation(파라미터 격리)** | 작은 task 전용 파라미터만 추가, 나머지 freeze | PNN, VPT, **ECLIPSE** | 모듈 증가; 다중 forward 추론 |

<dl class="kv">
<dt>EWC</dt><dd>과거 optimum 주변의 posterior를 diagonal quadratic으로 근사해 weight를 붙잡는 방법입니다. 실전의 $F_i$는 Fisher diagonal을 parameter importance proxy로 쓴 값이지 중요도의 완전한 척도는 아닙니다: $\mathcal{L}=\mathcal{L}_t+\tfrac{\lambda}{2}\sum_i F_i(\theta_i-\theta_i^{*})^2$.</dd>
<dt>LwF / KD (knowledge distillation, 지식 증류)</dt><dd>새 데이터에 대한 <b>예전 모델의 출력</b>을 흉내 내도록 학습(옛 데이터 불필요). Seg 변형: <b>MiB</b>(background 모델링), <b>PLOP</b>(multi-scale feature distillation), <b>CoMFormer</b>(panoptic query distillation).</dd>
<dt>Replay</dt><dd>소수 exemplar를 저장(iCaRL herding)하거나 생성/특징 replay. 종종 가장 강력하지만 이미지 저장이 금지될 수 있음.</dd>
</dl>

## 3 · Class-incremental vs task-incremental (그리고 seg의 반전)

- **Task-incremental:** 테스트 때 task ID가 주어짐(더 쉬움).
- **Class-incremental:** task ID 없이 지금까지 본 *모든* 클래스에 대해 예측(더 어려움) — 표준 seg 세팅.
- **Disjoint vs overlap:** 어떤 클래스의 이미지가 어느 step에 나타날 수 있는지와 비현재 클래스 pixel을 어떻게 label하는지를 정합니다. Overlap protocol에서는 미래 클래스가 이전 step 이미지에 나타나도 background로 주어질 수 있어 background shift가 생깁니다. “현실적/더 어려움”은 데이터 분포와 annotation policy에 따라 달라집니다.

## 4 · Background / no-object shift (segmentation 특유)

> [!QUESTION] "continual *segmentation*이 continual classification과 뭐가 다른가?"
> **짧게:** 현재 step에서 annotation되지 않은 과거·미래 class pixel이 semantic **background**에 섞일 수 있습니다. Query 모델의 **no-object**는 unmatched query target이라 pixel background와 동일하지 않지만, step별 query/class 구성이 바뀌며 그 score의 의미도 drift할 수 있습니다. **깊게:** old/future pixel을 background로 학습하면 나중에 foreground로 받아들여야 할 표현을 거부하는 gradient가 생깁니다. MiB는 background modeling을, SSUL은 Unknown + exemplar를 사용합니다. ECLIPSE는 query의 no-object head를 학습하는 대신 inference score에서 재구성하는 별도 메커니즘을 다룹니다. 두 shift를 같은 label이라고 뭉개지 마세요.

ECLIPSE의 **logit manipulation(로짓 조작)**: drift(표류)하는 학습된 no-object head 대신, 추론 시 다른 step들의 logit으로부터 no-object 점수를 계산합니다:

$$s^{\text{no-obj}}_t = \delta\Big(\sum_{k<t} s^{\mathcal{C}^k}_t + \sum_{k>t} s^{\mathcal{C}^k}_t\Big)$$

$\delta$는 *재학습 없이* 튜닝되는 사후(post-hoc) scalar(기본 ~0.5)입니다 — no-object는 본질적으로 모든 step 클래스 점수의 함수이기 때문입니다.

## 5 · ECLIPSE — panoptic CL을 위한 visual prompt tuning

> [!EXAMPLE] 메커니즘
> **Step 1:** $\mathcal{C}^1$에 대해 전체 **Mask2Former**를 학습한 뒤 **전부 freeze**. **Step $t>1$:** 작은 **prompt set** $\mathbf{Q}^t$(query) + **step별 MLP head만** 학습($N^t \approx |\mathcal{C}^t|$, 최소 10). 추론에서는 $\mathbf{Q}^{1:t}$ 출력을 집계. 학습 가능 파라미터 ≈ 모델의 **1.3%**(논문: ~0.6M vs ~44.9M). Distillation-free, replay-free.

ECLIPSE의 보고된 ablation에서는 여러 decoder layer에 주입하는 **deep prompt**가 shallow prompt보다 new-class 품질이 좋았습니다. 클래스별 sigmoid는 서로 다른 step class를 하나의 softmax denominator에서 직접 경쟁시키지 않습니다. Frozen path는 parameter drift를 막지만 combined inference의 interference까지 없애지는 않으므로 “완벽한 stability”를 보장하지 않습니다. 전체 ablation·FLOPs·CoMFormer 비교는 <strong><a href="#/resume/eclipse">ECLIPSE 딥다이브</a>에</strong> 있습니다.

<div class="proscons"><div><div class="pros-t">prompt-tuning이 여기서 이기는 이유</div>
튜닝할 distillation weight / pseudo-label threshold가 없고; 학습 가능한 footprint·메모리가 아주 작으며; forgetting에 극도로 강하고; 자연히 distillation-free·replay-free(프라이버시 친화적)입니다.
</div><div><div class="cons-t">비용</div>
step-1 feature가 약하면 frozen backbone이 plasticity를 제한(강한 pretraining, 예: Swin-L/COCO로 완화); 추론이 여러 번의 prompt forward를 수행; step-1 오분류가 고정됩니다.
</div></div>

## 6 · Prompt 기반 continual learning (일반적 흐름)

강한 pretrained backbone을 freeze하고 prompt만 학습하는 것은 중요한 CL 설계군 중 하나입니다:

- **Classification:** **L2P**(prompt *pool*을 학습하고 입력마다 선택), **DualPrompt**(general + expert prompt), CODA-Prompt.
- **Segmentation:** **ECLIPSE**(Mask2Former 위의 step별 visual prompt).
- **왜 통하나:** prompt는 공유 weight의 drift를 막고 task별 저장량을 줄일 수 있습니다. **한계:** frozen representation의 ceiling, prompt 선택·통합 오류, task 수에 따른 parameter/latency 증가가 남습니다. Foundation backbone + prompt/LoRA는 후보이지 유일한 recipe가 아닙니다. [Vision Foundation Models](#/cv/foundation-models) 참고.

## 7 · Panoptic continual이 "하드 모드"인 이유

Things(instance matching) + stuff + no-object drift가 한꺼번에 오고, PQ는 recognition 오류(RQ)에 가차없습니다. **CoMFormer**가 query distillation으로 panoptic CL을 개척했고, ECLIPSE는 최초의 *distillation-free* panoptic-CL 결과를 주장하며 **long sequence**(짧은 step이 많은 경우)에서 격차를 벌립니다 — regularization/KD 방법이 무너지는 영역입니다.

## 7b · 이름을 알아둘 벤치마크 & 프로토콜

| 벤치마크 | task | 대표 프로토콜(base-step) |
| --- | --- | --- |
| Pascal VOC | semantic CL | 15-5, 15-1, 10-1 (disjoint / overlap) |
| ADE20K | semantic & panoptic CL | 100-50, 100-10, 100-5, 50-50 |
| Cityscapes (domain-incremental) | semantic | 도시/조건 shift |

표기 "X-Y" = base 클래스 X개, 이후 Y개씩 증분. 작은 Y와 많은 step(예: 100-5, 11 task)은 forgetting이 누적되는 *long-sequence* 스트레스 테스트입니다.

### 비교 전에 고정할 evaluation contract

Continual result는 숫자 하나보다 protocol에 민감합니다. Task boundary를 아는지, 과거 raw sample/feature/generator를 얼마나 저장하는지, pretraining과 외부 데이터가 허용되는지, single-pass online인지 step별 여러 epoch인지, test 때 task ID가 있는지를 표에 함께 적으세요.

Step $t$까지 학습한 뒤 task $k$에서의 score를 $A_{t,k}$라 두면 대표 요약은 다음입니다.

$$
\text{FinalAvg}=\frac1T\sum_{k=1}^{T} A_{T,k},\qquad
\text{Forgetting}=\frac1{T-1}\sum_{k<T}\left(\max_{\ell\in\{k,\dots,T-1\}}A_{\ell,k}-A_{T,k}\right)
$$

Final average가 같아도 새 task를 못 배운 모델과 옛 task를 잊은 모델은 다릅니다. 따라서 base/new/all, 각 step accuracy matrix, memory·train FLOPs·inference passes·parameter growth를 함께 보고하고, joint-training upper bound와 naive sequential fine-tuning lower baseline을 둡니다. Negative forgetting은 backward transfer일 수 있으므로 무조건 0으로 clamp할지도 명시합니다.

> **개념 코드 — score matrix $A_{t,k}$를 채우는 순서**

```python
A = np.full((T, T), np.nan)
for t in range(T):
    train_on_step(model, train_stream[t])       # 현재 step 데이터만 허용
    model.eval()
    with torch.no_grad():
        for k in range(t + 1):                  # 지금까지 본 모든 task 재평가
            A[t, k] = evaluate(model, test_sets[k])
    model.train()

final_avg = np.mean(A[T - 1, :])
forgetting = np.mean([np.max(A[k:T - 1, k]) - A[T - 1, k]
                      for k in range(T - 1)])
```

## 8 · Q&A

<details class="qa"><summary>언제 prompt-tuning 대신 replay나 KD를 쓰나요?</summary>
<div class="qa-body">

**짧게:** 데이터를 저장할 수 있고, 최대 plasticity가 필요하며, backbone이 약할 때.

**깊게:** replay는 종종 가장 강력한 raw 성능을 내고 네트워크 전체를 adapt하게 합니다(높은 plasticity). 프라이버시/저장이 문제 안 되고 소수의 큰 step이 예상되면, 작은 exemplar buffer + KD가 new-class 정확도에서 prompt-tuning을 이길 수 있습니다. Prompt-tuning은 프라이버시 제약·long sequence·빠듯한 compute·강한 frozen backbone에서 빛납니다.
</div></details>

<details class="qa"><summary>freeze가 망각을 막는다면, 그 대가는?</summary>
<div class="qa-body">

**짧게:** step-1 오류를 고정하고 plasticity를 제한합니다.

**깊게:** step 1이 "car"만 알았다면 frozen path에서 "bus"는 "car"로 고정됩니다. ECLIPSE는 logit manipulation으로 완화하지만(이후 step logit이 상호 경쟁으로 잘못된 클래스를 억제), 일반 교훈은 stability와 plasticity가 trade-off라는 것 — frozen 모델은 plasticity를 내주고 stability를 삽니다.
</div></details>

<details class="qa"><summary>세 방법군은 긴 시퀀스에서 어떻게 다르게 무너지나요?</summary>
<div class="qa-body">

**짧게:** Regularization은 누적 제약, replay는 buffer coverage, isolation은 module growth와 shared inference interference가 병목이 될 수 있습니다. 어느 것이 먼저 무너지는지는 protocol에 따라 다릅니다.

**깊게:** Step이 많아지면 EWC/LwF의 근사와 teacher error가 누적될 수 있고, 고정 replay buffer는 class별 coverage가 얇아집니다. Parameter isolation은 옛 parameter drift는 막지만 shared normalization/head, prompt selection, class competition 때문에 end-to-end forgetting이 0이라고 보장할 수 없고 parameter/추론 예산도 커집니다.
</div></details>

### Follow-ups
- *"지표는?"* stability/plasticity 분리를 위해 **base / new / all**(PQ 또는 mIoU)을 보고; long sequence엔 *forgetting*(최고점 대비 하락)을 추가.
- *"Overlap vs disjoint?"* 미래 클래스가 이전 image에 나타나 background로 label될 수 있는 overlap은 background shift를 만듭니다. 난이도 비교에는 동일 data/order/annotation policy를 명시하세요.
- *"제품 관점?"* On-device 기능은 no-replay(프라이버시) 제약 하에 시간이 지나며 클래스 커버리지를 확장 → isolation/prompt 방법이 자연스러운 선택.

## Cheat-sheet

| 용어 | 한 줄 |
| --- | --- |
| Catastrophic forgetting | 새 task 학습이 옛 task 성능을 지움 |
| Stability–plasticity | 옛것 유지 vs 새것 학습 — 핵심 트레이드오프 |
| EWC / LwF | Fisher 페널티 / 출력 distillation (regularization) |
| Replay | 저장한 exemplar/특징을 다시 학습 |
| Isolation / VPT | 작은 prompt만 추가·학습, 나머지 freeze |
| Background shift | "background/no-obj" 의미가 step마다 변함 |
| Logit manipulation | 다른 step logit으로 no-obj 재구성 (ECLIPSE) |
| base/new/all | 트레이드오프를 드러내려 따로 보고 |

**다음:** [Segmentation](#/cv/segmentation) · [Weak & Semi-Supervised](#/cv/weak-semi-supervised) · [Vision Foundation Models](#/cv/foundation-models) · [ECLIPSE 딥다이브](#/resume/eclipse)
