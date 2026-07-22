# 디코딩 & 샘플링 전략

> [!NOTE] 이 챕터의 목표
> [다음 토큰 예측](#/llm/next-token)에서 모델은 다음 토큰의 **확률 분포**를 내놓습니다. 그런데 그 확률에서 실제로 **어떤 토큰을 고를 것인가?** 이 선택 방법(디코딩·샘플링)이 같은 모델을 "정확하지만 지루하게" 또는 "창의적이지만 산만하게" 만듭니다. temperature·top-k·top-p가 정확히 무엇을 하는지 그림과 코드로 잡습니다.

## 무엇을, 왜

매 스텝마다 LLM은 어휘(vocabulary) 전체에 대한 확률 분포를 냅니다(예: "고양이" 0.6, "개" 0.2, …). **디코딩(decoding)** 은 이 분포에서 다음 토큰을 고르는 규칙입니다. 규칙에 따라 출력이 완전히 달라집니다:

- 항상 최고 확률만 고르면(greedy) → 결정적이지만 다양성이 낮고 반복에 빠질 수 있음
- 확률대로 무작위로 뽑으면(sampling) → 다양하지만 가끔 엉뚱함

이 사이의 균형을 **temperature, top-k, top-p** 같은 손잡이로 조절합니다.

> [!TIP] 면접 한 줄
> "temperature는 분포의 뾰족함을, top-k/top-p는 후보 집합을 조절한다. 정답 검증이 쉬운 작업은 보통 낮은 다양성에서 시작하고, 창작·후보 탐색은 샘플링 폭을 넓힌다." 최적값은 모델·API·작업·검증기에 따라 달라진다고 덧붙이세요.

## Greedy — 항상 1등

$$\hat{x}_t = \arg\max_i\ p(x_t = i \mid x_{<t})$$

가장 단순합니다. 확률이 가장 높은 토큰을 매번 고릅니다. **알고리즘에는 난수가 없지만**, 같은 입력의 비트 단위 재현성은 서버의 모델 버전·커널·배칭·동점 처리에 따라 깨질 수 있습니다. 다양성이 없고 종종 같은 구절을 반복하며, 짧은 추출·분류처럼 출력 공간이 제한된 작업의 한 출발점입니다.

## Temperature — 분포의 뾰족함 조절

softmax 전에 logit을 temperature $T$로 나눕니다:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T \to 0^+$: 분포가 **뾰족**해짐 → 유일한 최댓값이면 greedy 분포에 수렴
- $T = 1$: 모델이 학습한 원래 분포
- $T > 1$: 분포가 **평평**해짐 → 더 다양하고 무작위

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="110" y="20" text-anchor="middle" font-weight="700" fill="#0ea5e9">낮은 T (뾰족)</text>
  <g fill="#0ea5e9">
    <rect x="60" y="40" width="24" height="120"/><rect x="92" y="140" width="24" height="20"/><rect x="124" y="150" width="24" height="10"/><rect x="156" y="155" width="24" height="5"/>
  </g>
  <text x="110" y="180" text-anchor="middle" fill="#98a3b2">1등 확률↑ (여전히 샘플링)</text>

  <text x="330" y="20" text-anchor="middle" font-weight="700" fill="#12a150">T = 1 (원래)</text>
  <g fill="#12a150">
    <rect x="280" y="70" width="24" height="90"/><rect x="312" y="105" width="24" height="55"/><rect x="344" y="130" width="24" height="30"/><rect x="376" y="145" width="24" height="15"/>
  </g>
  <text x="330" y="180" text-anchor="middle" fill="#98a3b2">학습된 분포 그대로</text>

  <text x="550" y="20" text-anchor="middle" font-weight="700" fill="#e0533f">높은 T (평평)</text>
  <g fill="#e0533f">
    <rect x="500" y="95" width="24" height="65"/><rect x="532" y="105" width="24" height="55"/><rect x="564" y="112" width="24" height="48"/><rect x="596" y="118" width="24" height="42"/>
  </g>
  <text x="550" y="180" text-anchor="middle" fill="#98a3b2">고르게 → 다양·무작위</text>
</svg>
<figcaption>temperature는 같은 logit을 서로 다른 확률 분포로 바꿉니다. 양의 T가 낮으면 1등에 확률이 몰리고, 높으면 후보들에 더 고르게 퍼집니다. 샘플링을 계속하는 한 낮은 T도 엄밀히 결정적이지 않으며, T=0은 보통 수식에 대입하지 않고 API가 greedy로 별도 처리합니다.</figcaption>
</figure>

## Top-k — 상위 k개만 후보로

확률 상위 $k$개 토큰만 남기고 나머지는 0으로 만든 뒤, 그 안에서 (재정규화 후) 샘플링합니다. 꼬리의 말도 안 되는 토큰이 뽑히는 사고를 막습니다. 단점: 적절한 $k$가 문맥마다 다릅니다(분포가 뾰족할 땐 $k$가 과함, 평평할 땐 부족).

## Top-p (nucleus) — 누적 확률 p까지만

확률을 큰 순서로 더해 **누적 합이 $p$(예: 0.9)에 도달할 때까지**의 토큰만 후보로 씁니다. 후보 개수가 문맥에 따라 자동으로 늘고 줄어드는 게 top-k보다 나은 점입니다 — 확신할 땐 1~2개, 애매할 땐 수십 개.

<figure>
<svg viewBox="0 0 620 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <g fill="#12a150"><rect x="40" y="40" width="40" height="80"/><rect x="90" y="70" width="40" height="50"/><rect x="140" y="95" width="40" height="25"/></g>
  <g fill="#98a3b2" opacity="0.4"><rect x="190" y="108" width="40" height="12"/><rect x="240" y="113" width="40" height="7"/><rect x="290" y="116" width="40" height="4"/></g>
  <path d="M185 30 V130" stroke="#e0533f" stroke-width="2" stroke-dasharray="5 4"/>
  <text x="110" y="30" fill="#12a150" font-weight="700">nucleus (누적 0.9)</text>
  <text x="360" y="80" fill="#98a3b2">← 잘라냄 (꼬리)</text>
  <text x="180" y="145" text-anchor="middle" fill="#98a3b2">확률 큰 순으로 더해 0.9 넘는 순간까지만 후보</text>
</svg>
<figcaption>top-p는 "확률 질량의 90%를 차지하는 최소 집합(nucleus)"만 후보로 남깁니다. 초록 = 후보, 회색 = 잘린 꼬리.</figcaption>
</figure>

**min-p**는 "최고 확률의 일정 비율 이상"인 토큰만 남기는 문맥 적응형 변형입니다. 일부 모델·설정에서 유용하지만 top-p보다 항상 강건하다고 단정할 수는 없습니다. **Beam search**는 여러 후보 시퀀스를 동시에 유지해 높은 누적 점수의 문장을 찾습니다. 번역 등에서는 유용할 수 있지만 길이 편향·반복·다양성 저하가 있어 length penalty와 task 평가가 필요합니다.

<details class="concept-code">
<summary>개념 코드로 보기</summary>

> 아래는 한 decode step의 필터 순서를 보이는 PyTorch식 **의사코드**입니다. 실제 API는 penalty와 top-k/top-p의 적용 순서가 다를 수 있습니다.

```python
def sample_next(logits, temperature, top_k, top_p, banned_ids, rng):
    # logits: [B,V]. temperature=0은 이 함수가 아니라 argmax 경로로 보낸다.
    assert temperature > 0
    z = logits.float() / temperature                # fp32에서 필터링
    z[:, banned_ids] = -inf                         # EOS 금지 등 정책 mask

    if top_k is not None:
        kth = topk(z, k=top_k).values[:, -1, None]
        z = z.masked_fill(z < kth, -inf)

    if top_p is not None:
        z_sorted, ids_sorted = sort(z, descending=True)
        p_sorted = softmax(z_sorted, dim=-1)
        remove = cumsum(p_sorted, dim=-1) > top_p
        # threshold를 처음 넘긴 토큰까지는 남겨 nucleus가 비지 않게 한다.
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        z = scatter_back(z_sorted.masked_fill(remove, -inf), ids_sorted)

    if any_row_has_no_finite_logit(z):
        raise InvalidDecodingPolicy("모든 후보가 가려짐")
    probs = softmax(z, dim=-1)
    return multinomial(probs, num_samples=1, generator=rng)  # seed 기록
```

</details>

## 언제 무엇을?

| 상황 | 추천 설정 |
| --- | --- |
| 수학·코드·추출(검증 가능) | 낮은 다양성에서 시작하고 테스트/스키마 검증 |
| 일반 대화·QA | 제공자 기본값을 기준으로 temperature 또는 top-p를 한 축씩 조정 |
| 창작·브레인스토밍 | 더 넓게 샘플하되 품질·안전 필터로 검증 |
| 여러 후보 뽑아 검증(self-consistency, best-of-N) | $T>0$으로 N개 샘플 후 검증 → [Reasoning](#/llm/reasoning) |

표의 값은 고정 레시피가 아닙니다. 제공자마다 temperature 범위와 필터 적용 순서가 다르고, instruction-tuned/reasoning 모델은 권장 기본값도 다릅니다. 정확도뿐 아니라 pass@k, 다양성, 지연·토큰 비용을 함께 측정하세요.

> [!NOTE] 추론(reasoning)과의 연결
> [test-time compute](#/llm/reasoning)의 self-consistency나 best-of-N은 **하나의 프롬프트로 여러 답을 샘플링**한 뒤 다수결·검증으로 고릅니다. 이때 다양한 답이 나오도록 T>0으로 샘플링하는 것이 전제입니다 — 그래서 디코딩 전략이 추론 성능과 직결됩니다.

## 직접 돌려보기 — temperature softmax

logit과 temperature $T$가 주어졌을 때 확률 분포를 반환하세요. $T$로 나눈 뒤 (수치 안정) softmax를 적용합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax_temperature","packages":["numpy"],"approx":true,"starter":"def softmax_temperature(logits, T=1.0):\n    # T는 양수여야 합니다. T=0/음수면 ValueError.\n    # logits 를 T 로 나눈 뒤 수치 안정 softmax 확률을 반환하세요.\n    pass","tests":[{"args":[[1.0,1.0,1.0],1.0],"expect":[0.3333333,0.3333333,0.3333333],"tol":1e-4},{"args":[[2.0,1.0,0.0],1.0],"expect":[0.6652409,0.2447284,0.0900306],"tol":1e-4},{"args":[[2.0,1.0,0.0],0.5],"expect":[0.8668132,0.1173104,0.0158764],"tol":1e-4}],"solution":"import numpy as np\n\ndef softmax_temperature(logits, T=1.0):\n    if T <= 0:\n        raise ValueError('T must be positive; use argmax separately for greedy')\n    z = np.asarray(logits, dtype=float) / T\n    z = z - z.max()          # 수치 안정\n    e = np.exp(z)\n    return (e / e.sum()).tolist()"}
</script>
</div>

세 번째 테스트를 보세요: $T=0.5$(낮음)로 낮추니 1등 확률이 0.665 → 0.867로 더 뾰족해졌습니다. temperature가 분포를 어떻게 조이는지 눈으로 확인됩니다.

## Q&A

<details class="qa"><summary>top-k와 top-p를 함께 쓰나요?</summary>
<div class="qa-body">

**짧게:** 네, 실무 API는 대개 temperature + top-p(때로 top-k까지)를 동시에 적용합니다.

**깊게:** 함께 쓸 수 있지만 **적용 순서는 구현/API마다 다릅니다**. 예를 들어 temperature 조정 뒤 top-k와 top-p를 적용하고 재정규화할 수 있습니다. 손잡이를 겹치면 상호작용 때문에 원인 분석이 어려우므로 하나씩 바꾸고, 사용하는 라이브러리의 실제 순서를 확인하세요.
</div></details>

<details class="qa"><summary>greedy가 왜 종종 반복(repetition)에 빠지나요?</summary>
<div class="qa-body">

**짧게:** 매번 국소 최적(1등)만 고르면 자기가 만든 고확률 구절로 되돌아오는 루프에 갇히기 쉽습니다.

**깊게:** greedy는 전체 문장 확률을 최대화하지 않고 매 스텝 탐욕적으로 고릅니다. 한 번 반복 패턴에 들어가면 계속 강화될 수 있습니다. repetition penalty나 no-repeat n-gram은 분포를 인위적으로 바꿔 정상적인 반복·코드·고유명사까지 해칠 수 있으므로 검증해야 합니다. 종료 조건, 프롬프트, 학습 데이터 문제도 함께 진단하세요.
</div></details>

## Cheat-sheet

| 손잡이 | 무엇을 조절 | 효과 |
| --- | --- | --- |
| greedy | — | 항상 1등; 난수 없음(환경 전체의 비트 재현성은 별개), 반복 위험 |
| temperature $T$ | 분포의 뾰족함($T>0$) | ↓1등 집중 / ↑다양; T=0은 보통 별도 greedy |
| top-k | 후보 개수(고정 k) | 꼬리 제거, k 튜닝 필요 |
| top-p (nucleus) | 누적 확률 p까지 | 문맥 적응적 후보 |
| min-p | 최고 대비 비율 컷 | 문맥 적응형 후보 제한; 모델별 검증 필요 |
| beam search | 여러 시퀀스 동시 | 번역·요약 등 |

**다음:** [프롬프팅 & ICL](#/llm/prompting) · [Reasoning & Test-Time Compute](#/llm/reasoning)
