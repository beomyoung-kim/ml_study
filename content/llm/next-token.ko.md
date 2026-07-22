# 다음 토큰 예측 직관

> [!NOTE] 이 챕터의 목표
> 많은 생성형 LLM의 **사전학습 목표**는 "다음에 올 토큰의 분포를 맞히기"입니다. 규모·데이터·아키텍처·사후학습이 결합되며 지식·번역·코딩·일부 추론 능력이 나타나지만, 목표 하나가 모든 능력을 자동 보장하는 것은 아닙니다. 토큰은 [토크나이제이션 & BPE](#/llm/tokenization), 실제 선택은 [디코딩 & 샘플링](#/llm/decoding-sampling)을 참고하세요.

## 무엇을, 왜

모델에 앞부분(문맥, prefix)을 주면, 모델은 **어휘(vocabulary)에 있는 모든 토큰마다 확률**을 매깁니다. 그게 전부입니다. 출력은 "다음에 올 토큰에 대한 확률 분포(probability distribution)" 하나입니다.

예를 들어 "the cat sat on the ___" 다음에 올 토큰의 확률 분포는 이런 식입니다:

<figure>
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="13">
  <text x="20" y="28" fill="currentColor">문맥(prefix): "the cat sat on the __"</text>
  <text x="20" y="48" fill="#98a3b2" font-size="11">→ 모델이 매긴 다음 토큰 확률 분포:</text>
  <g>
    <rect x="130" y="70" width="300" height="20" fill="#e0533f"/><text x="70" y="85" text-anchor="end" fill="currentColor">mat</text><text x="440" y="85" fill="#e0533f">0.55</text>
    <rect x="130" y="98" width="110" height="20" fill="#6366f1"/><text x="70" y="113" text-anchor="end" fill="currentColor">floor</text><text x="250" y="113" fill="#6366f1">0.20</text>
    <rect x="130" y="126" width="82" height="20" fill="#0ea5e9"/><text x="70" y="141" text-anchor="end" fill="currentColor">table</text><text x="222" y="141" fill="#0ea5e9">0.15</text>
    <rect x="130" y="154" width="55" height="20" fill="#98a3b2"/><text x="70" y="169" text-anchor="end" fill="currentColor">roof …</text><text x="195" y="169" fill="#98a3b2">0.10</text>
  </g>
  <text x="280" y="205" text-anchor="middle" fill="#98a3b2">확률의 합 = 1.0 · 자연스러운 다음 토큰에 높은 확률</text>
</svg>
<figcaption>다음 토큰 예측: 문맥이 주어지면 어휘 전체에 대한 확률 분포를 출력합니다. 학습이 잘 된 모델은 "mat"처럼 자연스러운 다음 토큰에 높은 확률을 줍니다.</figcaption>
</figure>

왜 확률 **분포**일까요? 언어에는 정답이 하나가 아니기 때문입니다. "the cat sat on the ___" 뒤에는 mat도, floor도, table도 올 수 있습니다. 모델은 "무조건 이거 하나"가 아니라 "각 후보가 얼마나 그럴듯한가"를 배웁니다.

수식으로는 문맥 $x_{<t}$가 주어졌을 때 다음 토큰 $x_t$의 조건부 확률 $p_\theta(x_t \mid x_{<t})$를 예측하는 것이고, 학습은 전체 코퍼스에서 정답 토큰의 로그 확률을 최대화(= 음의 로그 우도, negative log-likelihood를 최소화)합니다:

$$
\mathcal{L}(\theta) = -\sum_{t} \log p_\theta(x_t \mid x_{<t})
$$

> [!TIP] 면접 한 줄
> "현대 LLM은 수조 개 토큰으로 **next-token prediction** 하나를 학습한 모델이다. long context·저렴한 추론·MoE·RLHF 같은 주제는 전부 그 하나의 목표가 **데이터**·**serving 비용**·**정렬(alignment)** 제약과 만나 생긴 결과다." 목표를 먼저 꺼내고 거기서 짚어 나가면 좋습니다.

## 자기회귀 생성: 예측 → 붙이기 → 반복

문장을 **생성**할 때는 이 예측을 반복합니다. ① 다음 토큰의 확률 분포를 구하고 ② 그중 하나를 고르고(sample) ③ 그걸 문맥 뒤에 붙이고 ④ 다시 ①로. 이 순환을 **자기회귀(autoregressive) 생성**이라 합니다.

<figure>
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- prefix -->
  <rect x="20" y="30" width="180" height="34" rx="6" fill="none" stroke="#6366f1" stroke-width="1.5"/>
  <text x="110" y="52" text-anchor="middle" fill="currentColor">"the cat sat on the"</text>
  <text x="110" y="22" text-anchor="middle" fill="#98a3b2" font-size="10">① 문맥(prefix)</text>
  <path d="M200 47 H236" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#nt)"/>
  <!-- model -->
  <rect x="240" y="30" width="70" height="34" rx="6" fill="#e0533f"/>
  <text x="275" y="52" text-anchor="middle" fill="#fff">모델</text>
  <path d="M310 47 H346" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#nt)"/>
  <!-- prob bars -->
  <text x="350" y="22" fill="#98a3b2" font-size="10">② 확률 분포</text>
  <g>
    <rect x="410" y="30" width="130" height="12" fill="#e0533f"/><text x="404" y="40" text-anchor="end" fill="currentColor" font-size="10">mat</text>
    <rect x="410" y="46" width="47" height="12" fill="#6366f1"/><text x="404" y="56" text-anchor="end" fill="currentColor" font-size="10">floor</text>
    <rect x="410" y="62" width="35" height="12" fill="#0ea5e9"/><text x="404" y="72" text-anchor="end" fill="currentColor" font-size="10">table</text>
  </g>
  <!-- sample highlight (animated) -->
  <rect x="408" y="28" width="134" height="16" rx="3" fill="none" stroke="#12a150" stroke-width="2">
    <animate attributeName="opacity" values="0.25;1;0.25" dur="1.8s" repeatCount="indefinite"/>
  </rect>
  <text x="560" y="40" fill="#12a150" font-size="10">③ 샘플 → "mat"</text>
  <!-- append box -->
  <rect x="150" y="150" width="240" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.5"/>
  <text x="270" y="172" text-anchor="middle" fill="currentColor">"the cat sat on the <tspan fill="#12a150" font-weight="700">mat</tspan>"</text>
  <text x="270" y="142" text-anchor="middle" fill="#98a3b2" font-size="10">④ 붙여서 새 문맥</text>
  <!-- arrow from sample down to append -->
  <path d="M500 66 C 500 130, 400 120, 392 155" stroke="#12a150" stroke-width="1.5" fill="none" marker-end="url(#ntg)"/>
  <!-- loop back to prefix -->
  <path d="M150 167 C 60 167, 40 100, 60 64" stroke="#6366f1" stroke-width="1.5" fill="none" stroke-dasharray="5 4" marker-end="url(#ntb)">
    <animate attributeName="stroke-dashoffset" values="18;0" dur="0.9s" repeatCount="indefinite"/>
  </path>
  <text x="40" y="120" text-anchor="middle" fill="#6366f1" font-size="10" transform="rotate(-90 40 120)">↺ 반복</text>
  <!-- traveling dot along the loop -->
  <circle r="3.5" fill="#6366f1">
    <animateMotion path="M150 167 C 60 167, 40 100, 60 64" dur="1.8s" repeatCount="indefinite"/>
  </circle>
  <defs>
    <marker id="nt" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker>
    <marker id="ntg" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
    <marker id="ntb" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#6366f1"/></marker>
  </defs>
</svg>
<figcaption>자기회귀 생성 루프: 문맥 → 모델이 확률 분포 출력 → 하나를 샘플 → 문맥 뒤에 붙임 → 다시 반복. 매 스텝 딱 한 토큰씩 자랍니다. 그래서 긴 답일수록 forward pass가 그만큼 많이 필요합니다.</figcaption>
</figure>

②의 확률 분포에서 "어떻게 하나를 고르는가"(가장 확률 높은 것? 무작위 샘플링? 온도(temperature) 조절?)가 바로 [디코딩 & 샘플링](#/llm/decoding-sampling)의 주제입니다. 이 챕터는 그 앞 단계, 즉 **분포 자체를 어떻게 예측하고 배우는가**에 집중합니다.

## 학습은 병렬로: teacher forcing

생성은 한 토큰씩 순차적이라 느립니다. 그런데 **학습**은 훨씬 빠릅니다. 정답 문장 전체를 이미 알고 있으므로, 모든 위치의 "다음 토큰"을 **한 번의 forward pass로 동시에** 예측하고 채점할 수 있기 때문입니다.

"the cat sat on the mat"를 학습한다고 합시다. 모델은 아래 예측을 **병렬로 한꺼번에** 냅니다:

| 위치에서 본 문맥 | 맞혀야 할 정답 |
| --- | --- |
| the | cat |
| the cat | sat |
| the cat sat | on |
| the cat sat on | the |
| the cat sat on the | mat |

각 위치에서 모델에게는 (모델 자신의 이전 출력이 아니라) **정답 prefix를 그대로** 보여줍니다 — 이를 **teacher forcing(교사 강요)** 이라 합니다. 덕분에 위치들이 서로를 기다리지 않아 GPU에서 한 번에 계산됩니다. (한 위치가 미래 토큰을 훔쳐보지 못하게 막는 것이 [Attention](#/foundations/architectures)의 **causal mask**입니다.)

> [!NOTE] 단순한 목표가 왜 지능처럼 보일까 — 압축 직관
> 확률 모델의 예측이 좋아지면 산술 부호화 관점의 기대 부호 길이도 짧아지므로, 예측과 **압축**은 수학적으로 연결됩니다. 이를 "문법·사실·패턴을 내부 표현에 담도록 압력을 준다"는 직관으로 쓸 수 있습니다. 다만 이 비유가 인과 이해·신뢰성·안전을 증명하지는 않습니다. 낮은 예측 오차도 유용한 답을 보장하지 않으므로 [Post-Training & Alignment](#/llm/alignment)와 작업별 평가가 따로 필요합니다.

## 손실은 cross-entropy(교차 엔트로피)

각 위치의 예측이 얼마나 좋은지 채점하는 손실이 **cross-entropy(교차 엔트로피)** 입니다. 아주 단순합니다: **모델이 정답 토큰에 준 확률에 $-\log$를 씌운 것**입니다.

$$
L = -\log p_\theta(x_t \mid x_{<t})
$$

정답 토큰에 0.9를 줬으면 $-\log 0.9 \approx 0.1$(작음, 좋음), 0.1밖에 안 줬으면 $-\log 0.1 \approx 2.3$(큼, 나쁨)입니다. $-\log$는 "확신을 갖고 틀린 것"을 아주 강하게 벌합니다. 이게 왜 분류에 딱 맞는 손실인지, softmax와 gradient가 왜 깔끔하게 $p - \text{onehot}(y)$가 되는지는 [Loss & Gradient (밑바닥부터)](#/ml-coding/losses-gradients)에서 코드로 유도합니다 — 다음 토큰 예측은 사실 **어휘 크기만큼 클래스가 있는 초대형 분류 문제**입니다.

## 직접 돌려보기 — 다음 토큰 cross-entropy

모델이 다음 토큰에 대해 예측한 확률 분포 `probs`와 실제 정답 토큰의 인덱스 `target`이 주어졌을 때, 그 위치의 cross-entropy 손실 $-\log(\text{probs}[\text{target}])$를 구해 봅시다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"next_token_ce","packages":["numpy"],"approx":true,"starter":"def next_token_ce(probs, target):\n    # probs: 다음 토큰에 대한 확률 분포(리스트, 합=1).\n    # target: 실제 정답 토큰의 인덱스(정수).\n    # cross-entropy = -log(probs[target]) 를 반환하세요.\n    pass","tests":[{"args":[[0.6,0.3,0.1],0],"expect":0.5108256237659907,"tol":1e-6},{"args":[[0.25,0.25,0.25,0.25],2],"expect":1.3862943611198906,"tol":1e-6},{"args":[[0.9,0.1],0],"expect":0.10536051565782628,"tol":1e-6},{"args":[[0.1,0.9],0],"expect":2.302585092994046,"tol":1e-6}],"solution":"import numpy as np\n\ndef next_token_ce(probs, target):\n    p = np.asarray(probs, dtype=float)\n    return float(-np.log(p[target]))"}
</script>
</div>

정답에 확률 1.0을 주면 손실 0(완벽), 확률이 낮을수록 손실이 커집니다. 문장 전체의 학습 손실은 이 값을 모든 위치에서 평균 낸 것입니다.

## 얼마나 잘 맞히나: perplexity(당혹도)

한 문장의 평균 cross-entropy에 $\exp$를 씌우면 **perplexity(퍼플렉서티, 당혹도)** 가 됩니다 — "모델이 다음 토큰에 평균적으로 얼마나 놀라는가"를 재는 고전 지표입니다.

$$
\text{perplexity} = \exp\!\Big(-\frac{1}{N}\sum_{i=1}^{N}\log p_i\Big)
$$

여기서 $p_i$는 각 위치에서 **정답 토큰에 준 확률**입니다. perplexity=10을 "평균적으로 10개 후보 사이에서 헷갈리는 정도"로 직관화할 수 있지만 실제 후보 수라는 뜻은 아닙니다. 낮을수록 같은 평가 단위에서는 좋습니다. **서로 다른 tokenizer는 토큰 분할과 $N$이 달라 raw perplexity를 직접 비교하기 어렵고**, 데이터·전처리도 같아야 합니다. 실제 유용성은 downstream 평가로 따로 잽니다.

## Q&A

<details class="qa"><summary>모델은 단어가 아니라 토큰을 예측한다는데, 차이가 뭔가요?</summary>
<div class="qa-body">

**짧게:** 토큰은 단어보다 작은 조각(subword)일 때가 많습니다.

**깊게:** 모델은 문자도 단어도 아닌 **토큰 ID**(정수)를 다룹니다. "cats"는 `cat` + `s` 두 토큰일 수 있고, 한국어 "먹었다"도 여러 조각으로 쪼개질 수 있습니다. 어휘 구성 방식은 [토크나이제이션 & BPE](#/llm/tokenization)에서 다룹니다. "다음 토큰 예측"이라고 해야 정확하지만, 직관적으로는 "다음 조각 예측"으로 이해하면 됩니다. 정수 ID를 벡터로 바꾸는 첫 단계가 [임베딩](#/llm/embeddings)입니다.
</div></details>

<details class="qa"><summary>teacher forcing으로 학습하는데 생성은 스스로 이어붙이면, 문제가 없나요?</summary>
<div class="qa-body">

**짧게:** 문제가 있습니다 — **exposure bias(노출 편향)**.

**깊게:** 학습 때는 정답 prefix를 보지만 생성 때는 모델 자신의 출력에 이어서 예측하므로 오류가 누적될 수 있습니다. 이를 exposure bias라고 설명하는 것이 흔하지만, 모든 생성 오류의 단일 원인은 아니며 일반적인 alignment나 디코딩이 이를 "해결"한다고 단정할 수도 없습니다. 데이터·목표·시퀀스 수준 학습과 추론 정책을 작업별로 비교해야 합니다.
</div></details>

<details class="qa"><summary>왜 확률 하나가 아니라 분포 전체를 예측하나요?</summary>
<div class="qa-body">

**짧게:** 언어는 정답이 여럿이고, 분포가 있어야 학습 신호(cross-entropy)와 다양한 생성(sampling)이 가능합니다.

**깊게:** 하나만 찍으면 불확실성을 표현하기 어렵고 표준 cross-entropy 학습 신호도 쓸 수 없습니다. 어휘 전체 softmax 분포는 학습과 sampling을 가능하게 합니다([디코딩 & 샘플링](#/llm/decoding-sampling)). 마지막 출력 행렬을 입력 임베딩과 공유하는 **weight tying**은 흔하지만 모든 모델의 필수 구조는 아닙니다.
</div></details>

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| 목표 | 문맥(prefix) → 다음 토큰 **확률 분포** 예측 |
| 손실 | 정답 토큰의 음의 로그 우도 = cross-entropy $-\log p_y$ |
| 생성 | 자기회귀: 예측 → 샘플 → 붙이기 → 반복 (한 토큰씩) |
| teacher forcing | 학습 시 정답 prefix로 전 위치 **동시(병렬)** 예측 |
| exposure bias | 학습(정답 prefix) vs 생성(자기 출력)의 간극 |
| perplexity | $\exp(-\text{평균}\log p)$; 같은 tokenizer·데이터·전처리에서 비교 |
| 능력의 출현 | 예측 목표가 표현 학습 압력을 주지만 규모·데이터·구조·사후학습과 평가가 함께 좌우 |

**다음:** [디코딩 & 샘플링](#/llm/decoding-sampling) · [LLM Fundamentals](#/llm/fundamentals) · [Loss & Gradient](#/ml-coding/losses-gradients) · [토크나이제이션 & BPE](#/llm/tokenization)
