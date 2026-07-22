# 프롬프팅 & In-Context Learning

> [!NOTE] 이 챕터의 목표
> LLM을 **재학습하지 않고** 입력 문맥으로 원하는 행동을 유도하는 법을 잡습니다. 예시 몇 개로 과제·형식·라벨 관계를 추론하는 **in-context learning**을 보고, next-token pretraining이 주는 학습 압력과 아직 논쟁 중인 내부 메커니즘을 구분합니다.

## 무엇을, 왜

[LLM Fundamentals](#/llm/fundamentals)에서 봤듯 LLM은 앞의 토큰들을 보고 **다음 토큰**을 예측하는 모델입니다. 프롬프팅은 이 예측기를 **문맥(context)으로 조종**하는 기술입니다 — 앞에 무엇을 써 주느냐로 이어질 답을 바꾸는 것이죠. 파인튜닝(재학습)과 달리 **추가 학습 비용 없이 즉시** 적용할 수 있지만, 긴 지시·예시는 매 요청의 입력 token·latency·API 비용을 늘립니다.

<div class="proscons"><div><div class="pros-t">프롬프팅</div>

입력(문맥)만 바꿉니다. 재학습 없이 즉시 실험할 수 있습니다. 대부분의 문제는 여기서 먼저 시도합니다.

</div><div><div class="cons-t">파인튜닝</div>

가중치(파라미터)를 바꿉니다. 데이터·GPU·시간이 들지만, 형식·스타일을 모델에 **내재화**할 수 있습니다.

</div></div>

한 문장으로: **프롬프팅은 "무엇을 물어보느냐"를 설계하는 일**이고, 모델 자체는 그대로 둡니다.

> [!TIP] 면접 한 줄
> "프롬프팅은 파라미터를 안 바꾸고 문맥으로 조건을 거는 것이다. Few-shot에서는 모델이 예시의 형식·과제·라벨 관계를 문맥 안에서 추론해 continuation을 만든다." ICL의 내부 메커니즘은 pattern completion, meta-learning, implicit Bayesian inference 등 여러 관점이 연구 중이므로 하나로 확정하지 않습니다.

## Zero-shot vs Few-shot

가장 기본이 되는 두 가지 방식입니다.

- **Zero-shot(제로샷)**: 예시 없이 **지시만** 줍니다. "다음을 한 문장으로 요약해:"
- **Few-shot(퓨샷)**: 입력→출력 **정답 예시를 $k$개**(shots) 넣고, 그 뒤에 새 입력을 이어 붙입니다. 모델은 그 **형식(format)** 과 기준을 따라 답합니다.

<figure>
<svg viewBox="0 0 660 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="150" y="22" text-anchor="middle" font-weight="700" fill="#6366f1">Zero-shot (예시 0개)</text>
  <rect x="30" y="35" width="240" height="70" rx="8" fill="none" stroke="#6366f1" stroke-width="1.5"/>
  <text x="45" y="58" fill="currentColor">"이 리뷰 감정은? →"</text>
  <text x="45" y="80" fill="#98a3b2" font-size="11">지시만 주고 바로 답을 요구</text>
  <rect x="30" y="120" width="240" height="30" rx="6" fill="#12a150"/>
  <text x="150" y="140" text-anchor="middle" fill="#fff">→ "긍정"</text>
  <text x="510" y="22" text-anchor="middle" font-weight="700" fill="#e0533f">Few-shot (예시 몇 개)</text>
  <rect x="390" y="35" width="240" height="100" rx="8" fill="none" stroke="#e0533f" stroke-width="1.5"/>
  <text x="405" y="55" fill="#98a3b2" font-size="11">"최고였어 → 긍정</text>
  <text x="405" y="72" fill="#98a3b2" font-size="11"> 별로다 → 부정</text>
  <text x="405" y="89" fill="#98a3b2" font-size="11"> 돈 아까움 → 부정"</text>
  <text x="405" y="112" fill="currentColor">"이 리뷰는? →"</text>
  <text x="405" y="128" fill="#e0533f" font-size="11">예시로 형식·기준을 먼저 보여줌</text>
  <rect x="390" y="150" width="240" height="30" rx="6" fill="#12a150"/>
  <text x="510" y="170" text-anchor="middle" fill="#fff">→ 형식·기준을 조건화</text>
</svg>
<figcaption>Zero-shot은 지시만, few-shot은 정답 예시를 함께 넣어 형식과 기준을 보여줍니다. Few-shot의 이득은 예시의 대표성·순서·라벨 균형과 모델에 따라 달라지며, 나쁜 예시는 zero-shot보다 해로울 수 있습니다.</figcaption>
</figure>

## In-Context Learning — 왜, 어떻게 놀라운가

few-shot에서 모델은 **가중치를 전혀 바꾸지 않습니다.** 학습(경사하강, backprop)이 일어나지 않는데도 프롬프트 안 예시만으로 새 작업을 수행합니다. 이것을 **in-context learning(문맥 내 학습, ICL)** 이라 부릅니다. 진짜 파라미터가 바뀌는 [머신러닝의 학습 루프](#/foundations/what-is-ml)와는 전혀 다른 종류의 "학습"입니다.

**왜 next-token 예측에서 가능할까요?** pretraining data에는 예시 뒤 같은 패턴이 이어지는 구조가 많고, 이를 예측하려면 문맥의 규칙성과 mapping을 활용하는 편이 유리합니다. 이것이 ICL을 가능하게 하는 한 직관이지만 충분한 설명은 아닙니다. Pattern completion, meta-learning, induction head, implicit Bayesian inference 등 서로 보완·경쟁하는 설명이 연구 중입니다.

<figure>
<svg viewBox="0 0 640 180" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="13">
  <rect x="20" y="55" width="90" height="34" rx="6" fill="none" stroke="#6366f1" stroke-width="1.6"/>
  <text x="65" y="77" text-anchor="middle" fill="currentColor">A → B</text>
  <rect x="130" y="55" width="90" height="34" rx="6" fill="none" stroke="#6366f1" stroke-width="1.6"/>
  <text x="175" y="77" text-anchor="middle" fill="currentColor">C → D</text>
  <rect x="240" y="55" width="90" height="34" rx="6" fill="none" stroke="#6366f1" stroke-width="1.6"/>
  <text x="285" y="77" text-anchor="middle" fill="currentColor">E → ?</text>
  <text x="175" y="30" text-anchor="middle" fill="#98a3b2" font-size="11">프롬프트 속 예시 (문맥)</text>
  <path d="M340 72 H430" stroke="#98a3b2" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="385" y="60" text-anchor="middle" fill="#98a3b2" font-size="11">패턴 이어받기</text>
  <rect x="440" y="55" width="180" height="34" rx="6" fill="#e0533f"/>
  <text x="530" y="77" text-anchor="middle" fill="#fff">F 예측 (continuation)</text>
  <text x="530" y="120" text-anchor="middle" fill="#98a3b2" font-size="11">가중치 갱신 없음 — 그냥 다음 토큰 예측</text>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>모델은 프롬프트가 만든 관계(A→B, C→D …)를 문맥 안에서 추론해 continuation으로 완성합니다. 가중치 갱신은 없으며, 예시의 <b>형식·순서·대표성</b>이 정확도에 크게 영향을 줍니다. 이를 가능하게 하는 내부 메커니즘은 여전히 연구 중입니다.</figcaption>
</figure>

## 채팅 템플릿: system / developer / user / assistant / tool

실무 chat 모델은 하나의 긴 글이 아니라 **역할(role)이 구분된 메시지들**을 받습니다. 이 구조를 **chat template(챗 템플릿)** 이라 하며, 내부적으로는 `<|system|>` 같은 특수 토큰으로 감싸져 다시 하나의 토큰 열로 합쳐집니다. 아래 세 역할은 공통적인 최소 멘탈 모델입니다. 실제 API에는 **developer**(애플리케이션 규칙)와 **tool**(도구 결과) 역할도 있고, 우선순위와 허용 역할은 provider별 사양을 따라야 합니다.

<figure>
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- system -->
  <rect x="60" y="20" width="500" height="52" rx="10" fill="none" stroke="#6366f1" stroke-width="1.8"/>
  <rect x="60" y="20" width="90" height="52" rx="10" fill="#6366f1"/>
  <text x="105" y="50" text-anchor="middle" fill="#fff" font-weight="700">system</text>
  <text x="170" y="42" fill="currentColor">규칙·페르소나·톤 설정</text>
  <text x="170" y="60" fill="#98a3b2" font-size="11">"너는 친절한 한국어 번역가다"</text>
  <!-- user -->
  <rect x="60" y="88" width="500" height="52" rx="10" fill="none" stroke="#0ea5e9" stroke-width="1.8"/>
  <rect x="60" y="88" width="90" height="52" rx="10" fill="#0ea5e9"/>
  <text x="105" y="118" text-anchor="middle" fill="#fff" font-weight="700">user</text>
  <text x="170" y="110" fill="currentColor">실제 요청·질문</text>
  <text x="170" y="128" fill="#98a3b2" font-size="11">"'Hello, world'를 번역해줘"</text>
  <!-- assistant -->
  <rect x="60" y="156" width="500" height="52" rx="10" fill="none" stroke="#e0533f" stroke-width="1.8" stroke-dasharray="5 4"/>
  <rect x="60" y="156" width="90" height="52" rx="10" fill="#e0533f"/>
  <text x="105" y="186" text-anchor="middle" fill="#fff" font-weight="700">assistant</text>
  <text x="170" y="178" fill="#e0533f">모델이 채울 자리 (여기부터 생성 시작)</text>
  <text x="170" y="196" fill="#98a3b2" font-size="11">"안녕, 세상"</text>
  <path d="M310 72 V88" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#a2)"/>
  <path d="M310 140 V156" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#a2)"/>
  <text x="310" y="232" text-anchor="middle" fill="#98a3b2" font-size="11">세 역할이 특수 토큰으로 감싸져 하나의 입력 열이 됩니다</text>
  <defs><marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>대표적인 chat template에서 system은 상위 규칙, user는 요청, assistant는 생성 위치를 나타냅니다. 실제 역할 이름·우선순위·tool message 형식은 provider와 model template별 계약을 따릅니다.</figcaption>
</figure>

- **system**: 전역 규칙·역할·톤. 대화 내내 유지.
- **developer**: 애플리케이션이 넣는 세부 규칙. 지원 여부와 system과의 우선순위는 API별로 다름.
- **user**: 실제 요청.
- **assistant**: 모델이 채워야 할 자리(여기부터 생성 시작).
- **tool**: 모델이 호출한 도구의 결과. 신뢰 경계와 provenance를 별도로 관리.

> [!WARNING] 흔한 함정
> 학습 때 쓰인 chat template과 다른 형식은 성능 저하·이상 종료·역할 혼동을 일으킬 수 있습니다. 모델 카드의 template과 tokenizer API를 사용하고, special token 중복·assistant generation marker·tool turn 경계를 실제 token id로 검사하세요. ([VLM 구현 세부](#/vlm/practical)에서도 같은 함정을 다룹니다.)

## Chain-of-Thought — reasoning으로 가는 다리

일부 수학·다단계 추론 과제에서는 중간 풀이를 유도하는 **chain-of-thought(CoT, 사고 사슬)** 프롬프팅이 정확도를 높입니다. 다만 효과는 모델과 과제에 따라 다르고, 최신 reasoning API는 내부 추론을 숨기거나 요약만 반환하기도 합니다. 실무에서는 원시 내부 사고를 요구하기보다 **검증 가능한 중간 결과, 간결한 근거, 최종 답의 재검산**을 요청하는 편이 안전합니다.

왜 도움이 될 수 있을까요? 중간 결과를 token으로 펼치면 이후 생성 step이 그것을 다시 읽어 추가 직렬 계산과 scratch space로 사용할 수 있습니다. 하지만 더 긴 설명이 faithful하거나 정확하다는 보장은 없으므로 final answer checker와 task metric으로 검증합니다.

```text
[질문] 사과 3개에 2400원. 7개는 얼마?

(zero-shot 요구)      → "4800원"          (틀림)

(+ "단계별로 생각해보자") → "사과 1개 = 2400 / 3 = 800원.
                          7개 = 800 × 7 = 5600원."   → 중간 계산을 검산할 수 있음
```

Prompted CoT와 별도로, reasoning trace·verifiable reward·test-time search를 활용해 더 긴 계산을 학습·선택하는 계열이 발전했습니다. 같은 현상으로 단순화하지 말고 [Reasoning & Test-Time Compute](#/llm/reasoning)에서 학습과 inference를 나눠 봅니다.

## 구체적 예: few-shot 프롬프트 한 덩어리

few-shot이 실제로 프롬프트에서 어떻게 생겼는지 봅시다. 아래는 감정 분류를 위한 **완성된 프롬프트 텍스트**입니다.

```text
아래 리뷰의 감정을 "긍정" 또는 "부정" 중 하나로만 답하라.

리뷰: 최고였어, 또 올게요
감정: 긍정

리뷰: 별로다, 다신 안 감
감정: 부정

리뷰: 가격 대비 돈이 아까움
감정: 부정

리뷰: 웨이팅은 길었지만 맛은 훌륭했다
감정:
```

마지막 줄 `감정:` 뒤가 비어 있고, 모델은 앞의 세 예시 형식을 이어받아 여기에 `긍정`을 채웁니다 — 가중치는 그대로인 채로요.

## 직접 만들어보기 — few-shot 프롬프트 조립기

few-shot 프롬프트는 결국 **문자열을 규칙대로 이어 붙인 것**뿐입니다. 예시 목록과 새 질문을 받아 위와 같은 프롬프트 문자열을 만드는 함수를 완성해 보세요. (막히면 **Solution**을 열어 보세요.)

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"build_few_shot","packages":[],"starter":"def build_few_shot(examples, query):\n    # examples: [[리뷰, 감정], ...]  형태의 예시 목록\n    # query: 감정을 물어볼 새 리뷰 문자열\n    # 아래 형식의 프롬프트 문자열을 만들어 반환하세요.\n    #   각 예시:  '리뷰: {리뷰}\\n감정: {감정}\\n'\n    #   마지막:   '리뷰: {query}\\n감정:'   (감정 뒤는 비움)\n    # 예시 블록과 마지막 질문은 '\\n' 로 이어 붙입니다.\n    # TODO\n    pass","tests":[{"args":[[["최고였어","긍정"],["별로다","부정"]],"그냥 그래"],"expect":"리뷰: 최고였어\n감정: 긍정\n\n리뷰: 별로다\n감정: 부정\n\n리뷰: 그냥 그래\n감정:"},{"args":[[["맛있다","긍정"]],"별로"],"expect":"리뷰: 맛있다\n감정: 긍정\n\n리뷰: 별로\n감정:"}],"solution":"def build_few_shot(examples, query):\n    blocks = [f\"리뷰: {r}\\n감정: {s}\" for r, s in examples]\n    blocks.append(f\"리뷰: {query}\\n감정:\")\n    return \"\\n\\n\".join(blocks)"}
</script>
</div>

## 실무 프롬프트 패턴과 함정

<dl class="kv">
<dt>지시를 명확·구체적으로</dt><dd>"요약해"보다 "핵심을 3개의 불릿으로, 각 20자 이내로 요약해"가 훨씬 안정적입니다. 모호함은 곧 분산(variance)입니다.</dd>
<dt>출력 형식을 못 박기</dt><dd>"JSON만 출력하고 설명은 붙이지 마라"처럼 형식을 명시하고, 필요하면 예시로 고정(few-shot)합니다.</dd>
<dt>역할·제약을 system에</dt><dd>페르소나·금지사항·톤은 system 메시지로. user에는 그때그때의 요청만.</dd>
<dt>어려우면 단계적 사고</dt><dd>추론이 필요한 작업엔 CoT를 유도. 단, 최종 형식이 필요하면 "마지막 줄에만 답을" 식으로 분리 요청.</dd>
</dl>

> [!WARNING] 프롬프트로 "요청"과 "보장"은 다르다
> "반드시 JSON으로"라고 써도 모델이 가끔 어길 수 있습니다. 문법을 강제하려면 constrained decoding / JSON schema를 사용하세요([Instruction Tuning & Decoding](#/vlm/instruction-tuning)). 그래도 **schema-valid JSON이 사실·업무 규칙까지 맞는 것은 아니며**, max-token truncation이나 도구 오류도 별도 처리해야 합니다. 프롬프트는 유도, constrained decoding은 구문 보장, validator는 의미 검증을 담당합니다.

> [!WARNING] 신뢰할 수 없는 입력과 prompt injection
> 검색 문서·웹페이지·tool output 안의 텍스트는 **데이터**이지 상위 지시가 아닙니다. 경계를 명시하고, 권한 있는 지시와 외부 콘텐츠를 분리하며, tool에는 최소 권한·allowlist·승인 단계를 적용하세요. “이전 지시를 무시하라” 같은 문장을 필터링하는 것만으로는 충분하지 않습니다.

## Q&A

<details class="qa"><summary>Few-shot이 항상 zero-shot보다 좋나요?</summary>
<div class="qa-body">

**짧게:** 아니요 — 강력한 instruct 모델은 zero-shot도 잘하고, 나쁜 예시는 오히려 해롭습니다.

**깊게:** 예시는 토큰(=비용·context 길이)을 먹고, 대표성이 떨어지거나 편향된 예시는 모델을 잘못된 형식/분포로 유도합니다. 심지어 예시의 **순서**나 라벨 분포만 바꿔도 결과가 흔들립니다. Few-shot은 (1) 출력 **형식**이 까다롭거나, (2) 작업이 모호해 예시로 기준을 못 박아야 할 때 특히 유용합니다.
</div></details>

<details class="qa"><summary>In-context learning은 진짜 "학습"인가요?</summary>
<div class="qa-body">

**짧게:** 파라미터가 바뀌지 않으므로 통상적 의미의 학습은 아닙니다. 문맥에 조건을 건 것에 가깝습니다.

**깊게:** [what-is-ml](#/foundations/what-is-ml)의 학습은 경사하강으로 가중치를 갱신합니다. ICL은 그런 갱신 없이, 이미 학습된 모델이 프롬프트 속 패턴을 인식해 이어가는 것입니다. 새 대화를 시작하면(=문맥이 사라지면) "배운" 것도 사라집니다. 그래서 지속적으로 능력을 내재화하려면 파인튜닝이 필요합니다.
</div></details>

<details class="qa"><summary>프롬프팅으로 안 되면 언제 파인튜닝하나요?</summary>
<div class="qa-body">

**짧게:** 프롬프트로도 형식/품질이 안 잡히거나, 같은 작업을 대량·저지연으로 반복해야 할 때.

**깊게:** 이들은 고정된 순서가 아니라 다른 문제를 풉니다. Instruction ambiguity는 prompting/few-shot, 외부의 최신·권한 있는 사실은 [RAG](#/llm/rag), 반복되는 behavior·style·domain adaptation은 fine-tuning 후보입니다. Baseline을 만든 뒤 quality·latency·token cost·update frequency·security를 비교하고 조합하세요.
</div></details>

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| 프롬프팅 | 재학습 없이 입력 문맥으로 행동 조종; 즉시 실험 가능하지만 token·latency 비용은 발생 |
| Zero vs Few-shot | 지시만 vs 정답 예시 $k$개를 함께 |
| In-context learning | 가중치 갱신 없이 프롬프트 속 패턴을 이어받음 |
| 왜 가능한가 | 문맥 규칙을 쓰는 pretraining 압력; 내부 메커니즘은 여러 가설이 연구 중 |
| Chat template | system(규칙)/user(요청)/assistant(생성) — 형식 정확히 |
| Chain-of-thought | 일부 추론 과제에서 유용; 모델·과제별 검증, 내부 사고 대신 간결한 근거·검산 요청 |
| 요청 vs 보장 | 프롬프트=유도 · constrained decoding=구문 · validator=의미 검증 |
| 언제 파인튜닝 | 프롬프트로 형식/품질 안 될 때, 대량·저지연 |

**다음:** [LLM Fundamentals](#/llm/fundamentals) · [Reasoning & Test-Time Compute](#/llm/reasoning)
