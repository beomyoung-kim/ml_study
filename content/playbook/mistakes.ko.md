# Common Mistakes & Red Flags

<div class="tag-row"><span class="tag">rejection reasons</span><span class="tag">per-round pitfalls</span><span class="tag">RS/AS-specific</span><span class="tag">how to avoid</span></div>

> [!TIP] 자료만이 아니라 실패 모드를 배우세요
> 탈락 사유는 공개되지 않는 경우가 많아 비율을 단정할 수 없습니다. 다만 mock과 회고로 줄일 수 있는 반복적 위험은 있습니다: 시간 압박 속 구현 실패, 자기 연구 방어 부족, 기여 경계의 모호함, 모르는 것을 추측으로 덮는 습관. 이 챕터는 그런 **관찰 가능한 실패 모드**를 진단하는 체크리스트입니다.

## RS/AS 후보에게 반복되는 위험

아래 항목이 실제 회사의 공식 순위나 rubric이라는 뜻은 아닙니다. 이번 role의 prep guide와 recruiter 설명을 우선하고, 자신의 mock에서 관찰된 항목을 해결책에 매핑하세요.

```mermaid
flowchart TB
  R(("No-hire")) --> A["Can't implement from scratch<br/>under time pressure"]
  R --> B["Breadth gap<br/>(RL / diffusion / LLM primitive)"]
  R --> C["Can't defend own published work"]
  R --> D["Obscured contribution<br/>('we' everywhere)"]
  R --> E["Weak communication / coding hygiene"]
  R --> F["Behavioral misses<br/>(no result, no reflection, blame)"]
  R --> G["Fit mismatch at senior level"]
```

| # | 탈락 사유 | 강한 사람에게도 일어나는 이유 | 해결책 |
| --- | --- | --- | --- |
| 1 | **처음부터 구현 못 함** | "매일 ML 사용"과 제한된 환경에서 핵심 연산을 구현하는 능력은 다릅니다. | [ML-coding from scratch](#/ml-coding/intro)를 시간 제한과 실제 도구 정책에 맞춰 연습. |
| 2 | **Breadth gap** | 역할의 핵심 기초에서 설명이 끊기면 다른 강점을 보여줄 시간이 줄어듭니다. | JD와 prep guide 기준으로 [foundations](#/foundations/optimization) 맵의 우선순위를 정함. |
| 3 | **자기 작업 방어 못 함** | 대표 작업의 baseline, ablation, 한계를 설명하지 못하면 ownership 판단이 어려워집니다. | 모든 대표 논문을 pre-mortem: "왜 baseline X가 아닌가?", 실패 사례, 무엇을 다시 할지. [job talk](#/research/job-talk) 참고. |
| 4 | **기여를 가림** | "we"를 남용해 패널이 *당신이* 한 것을 분리 못 함. | 깔끔한 [I-vs-we 분리](#/behavioral/star): 목표에는 "we", 모든 결정에는 "I". |
| 5 | **coding 위생 / 디버깅 못 함** | 특히 AS/RE 트랙 — 지저분, 테스트 없음, 라이브 디버깅 못 함. | 타입, 테스트, 작은 함수, 디버깅을 내레이션. |
| 6 | **Behavioral 실책** | 구체적 결과 없음, 회고 없음, 협업 signal 부실, 남 탓. | 모든 story를 정량화, ownership + 학습으로 마무리. |
| 7 | **Fit 불일치** | 강한 일반 역량과 이번 팀이 필요한 scope는 다를 수 있습니다. | *현재* 문제가 맞는 팀을 타겟하고 [HM 스크린](#/process/recruiter-hm)에서 fit을 확인. |
| 8 | **협상 integrity 훼손** | 부풀리거나 위조한 경쟁 offer는 신뢰·관계·문서상 위험을 만듭니다. | 실제 사실과 deadline만 공유. [Negotiation](#/process/negotiation) 참고. |

> [!WARNING] research 직군에서 특히 점검할 두 가지
> **자기 출판 작업을 방어하는 능력**과 **기여 경계를 명확히 하는 능력**은 ownership을 판단할 핵심 근거입니다. 훌륭한 결과를 나열하는 데서 멈추지 말고, 질문 아래에서도 결정·대안·검증·한계를 설명할 수 있게 준비하세요.

## 라운드별 실수

### Coding 라운드

<div class="proscons"><div><div class="pros-t">하기</div>

- 코딩 전 clarify + 가정 진술
- 접근 내레이션, 막히면 fallback 예고
- 작은 추적으로 테스트, edge case 포함
- 시계 관리, 깔끔한 부분 완성이 망가진 "완성"보다 낫다

</div><div><div class="cons-t">피하기</div>

- 몇 분간 침묵 코딩 (부분 점수 불가)
- 문제 이해 전에 코드로 점프
- 물을 때까지 복잡도 무시
- 힌트를 받는 대신 면접관의 힌트에 맞서기

</div></div>

[Coding Round Strategy](#/coding/strategy)와 [Communication](#/playbook/communication) 참고.

### ML depth / fundamentals

- **Breadth gap을 허세로 메우기.** RLHF나 diffusion에서 확신에 찬 오답은 솔직한 "형태는 알지만 구현은 안 해봤습니다"보다 나쁩니다. → ["I don't know" 다루기](#/playbook/communication).
- **Depth 없는 breadth (또는 반대).** vision은 깊은데 LLM primitive에서 막힐 수 있습니다. 맵을 메워 0점이 없게.
- **추론 없는 암송.** "Transformer가 쓰니까 LayerNorm"은 실패합니다, 가변 길이 sequence에 왜 BN이 아니라 LN인지 설명하세요.

### Research deep-dive / job talk

- **자료가 너무 많음.** 논문 전체를 욱여넣으면 → 패널은 아무것도 기억 못 합니다. 중요성 + 설명 가능성으로 선별.
- **잘못된 고도.** 혼합 패널에 너무 많은 jargon, 또는 전문가에게 너무 얕음. 방을 읽으세요.
- **선택을 방어 못 함.** "왜 baseline X가 아닌가?"나 자기 실패 사례에서 머뭇거리면 치명적. [Failure & Negative Results](#/research/failure) 참고.
- **시간 관리 부실.** Q&A 전에 시간 초과. 시계 놓고 리허설.

### Behavioral

- **"저는 실패한 적 없어요."** 부정직하거나 야망이 낮은 걸로 읽힙니다. research 경력은 *곧* 실패한 실험입니다.
- **남 탓하기.** ownership에 대한 anti-signal, "까다로운 팀원" story는 *당신의* 적응을 보여줘야 합니다.
- **숫자 없음, 회고 없음.** 결과나 교훈 없는 story는 그냥 일화입니다.
- **맥락 앞에 몰아넣기.** 40% Situation은 signal을 묻습니다. Action은 50~60%. → [STAR 시간 배분](#/behavioral/star).

### Fit / motivation

- **뻔한 "why us."** 뿌리고 기도하는 signal. 조직마다 정직한 "저는 ___를 존경했습니다" 하나면 해결.
- **연봉만 또는 남 탓하는 "why leave."** push가 아니라 pull(70%)로 프레이밍. → [HM 스크린](#/process/recruiter-hm).
- **미공개 product·roadmap 추측하기.** 공개 JD와 공식 자료의 경계를 넘지 마세요.

## 전방위 behavioral & communication red flag

이것들은 라운드와 무관하게 debrief에 기록됩니다:

<dl class="kv">
<dt>힌트에 대한 방어성</dt><dd>넛지에 맞서는 건 "같이 일하기 어려움"으로 읽힙니다. 힌트를 우아하게 받으세요 — 당신을 *돕는* 것입니다.</dd>
<dt>허세</dt><dd>확신에 찬 오답은 인정보다 신뢰를 더 무너뜨립니다. dig-in이 어차피 폭로합니다.</dd>
<dt>에고 / 무시</dt><dd>이전 팀원, baseline, "뻔한" 질문을 깎아내리기. 협업 방식과 판단의 신뢰를 약화합니다.</dd>
<dt>안 듣기</dt><dd>물은 질문이 아니라 준비한 질문에 답하기, 면접관의 방향 조종을 놓치기.</dd>
<dt>자기 논평</dt><dd>면접관/recruiter에게 "그 라운드 망친 것 같아요"는 부정적 해석을 유도하고, 당신은 자신에 대해 노이즈 많은 심판입니다.</dd>
</dl>

> [!WARNING] intellectual honesty는 라운드를 가리지 않는 기본 원칙입니다
> 정확한 rubric은 회사마다 다르지만, 아는 것·자기 연구·선택한 접근의 경계를 명확히 하면 검증 가능한 추론을 보여줄 수 있습니다. 모르는 부분은 확인 경로와 함께 인정하세요.

## PhD → 산업 번역 함정

학계는 면접에서 역효과를 내는 습관을 훈련시킵니다. 의도적으로 번역하세요.

| 학계 습관 | 면접 현실 | 재배선 |
| --- | --- | --- |
| 뉘앙스 & 단서로 시작 | 명확한 결정에 보상 | **결정 먼저**, 단서는 요청 시 |
| "We" (lab 규범, 겸손) | 패널이 *당신을* 분리해야 함 | **당신의 결정에는 "I"** |
| 성공 = 논문 | 성공 = 결정 + 측정 가능한 impact | **정량화된 결과 + 출시** |
| 완전함의 소진 | Time-box, 우선순위 | **headline 먼저, 중요성으로 선별** |
| 권위/인용으로 방어 | 증거 & 추론으로 방어 | **데이터, ablation, first principles** |

## loop 전에 자기 실패 모드를 진단하세요

볼 수 없는 실수는 고칠 수 없습니다. 대부분의 후보는 여덟 개 전부가 아니라 **한두 개의 지배적 실패 모드**를 가집니다. 값싼 계측으로 당신의 것을 찾으세요:

- **Mock을 녹화하세요** (coding, 논문 방어, behavioral story 하나). 되돌려 보세요: 내레이션했나? STAR에서 Action이 지배했나? 낯선 사람이 *당신이* 무엇을 만들었는지 알 수 있었나?
- **Mock 파트너에게 한 질문:** "저를 채용하기 가장 망설이게 만드는 단 하나는 뭔가요?" 솔직한 답이 보통 당신의 지배적 모드입니다.
- **Debrief 로그를 추적하세요.** 실제 라운드마다 질문과 스스로 평가한 약한 순간을 적으세요. 3~4 라운드에 걸쳐 패턴이 드러납니다 — 그 패턴이 당신의 fix-list입니다.
- **스스로 시간을 재세요.** 습관적으로 시간이 부족하다면 실수는 지식이 아니라 [페이싱](#/playbook/communication)입니다.

> [!NOTE] 한 번에 하나씩 고치기
> 한 번에 다 고치려 하면 아무것도 못 고칩니다. 최상위 모드를 골라 일주일 드릴하고, 다시 녹화하고, 다음으로 넘어가세요. 여기선 복리가 폭보다 낫습니다.

## 후속 질문

<details class="qa"><summary>라운드 초반에 틀린 답을 준 걸 깨달았어요. 정정해야 하나요?</summary>
<div class="qa-body">

**짧게:** 네 — 자기 정정은 *긍정* signal입니다.

**깊게:** "아까 X라고 했는데 정정하고 싶습니다: 사실은 Y입니다, 왜냐하면…"은 intellectual honesty와 rigor로 읽힙니다, 정확히 research 패널이 보상하는 것. 자기 오류를 잡는 게 그들이 못 봤길 바라는 것보다 낫습니다, 알고 있는 오류를 놔두는 게 진짜 위험입니다.
</div></details>

<details class="qa"><summary>거만하게 들리지 않으면서 "기여를 가림" 함정을 어떻게 피하죠?</summary>
<div class="qa-body">

**짧게:** 목표는 팀에 공을 돌리고, 결정은 당신이 가져가세요.

**깊게:** "팀이 X를 출시했고, *제가* architecture, loss, data 결정을 맡았으며, 구체적으로 Y를 결정했습니다." 협업자에게 공을 돌리면서(거만하지 않음) 당신의 역할을 모호하지 않게(가려지지 않음) 했습니다. 각 대표 프로젝트에 이 분리를 리허설하세요 — 가장 레버리지 큰 behavioral 해결책입니다. [I-vs-we](#/behavioral/star) 참고.
</div></details>

## 치트시트

| 실수 | 한 줄 해결책 |
| --- | --- |
| 처음부터 코딩 못 함 | attention / NMS / training loop을 제한된 환경과 시간 안에서 드릴 |
| Breadth gap | foundations 맵 전반에 0점 없기 |
| 자기 논문 방어 못 함 | baseline, ablation, 실패 사례 pre-mortem |
| 기여를 가림 | 목표에는 "we", 모든 결정에는 "I" |
| 허세 | 추론으로 도달 / 범위로 묶기 / "모르겠습니다" + 경로 |
| 결과 / 회고 없음 | 정량화 + ownership & 학습으로 마무리 |
| 맥락 앞에 몰아넣기 | Action = 50~60%, headline 먼저 |
| 뻔한 "why us" | 조직마다 정직한 "저는 ___를 존경했습니다" 하나 |
| 날조한 오퍼 | 범위 공유, 절대 지어내지 말 것 — 걸리면 치명적 |
| 힌트에 맞서기 / 에고 | 넛지를 받기, low-ego 협업이 채점됨 |

**관련:** [STAR & The Story Bank](#/behavioral/star) · [Common Questions & Answers](#/behavioral/questions) · [Communication & Whiteboarding](#/playbook/communication) · [Day-Of Tactics & Recovery](#/playbook/tactics) · [The Research Job Talk](#/research/job-talk) · [Failure & Negative Results](#/research/failure) · [Recruiter & HM Screens](#/process/recruiter-hm) · [The ML Coding Round](#/ml-coding/intro)
