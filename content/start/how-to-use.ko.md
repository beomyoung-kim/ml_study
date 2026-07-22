# How to Use This Book

> [!TIP] 30초 요약
> **가장 약한 축**에 해당하는 part를 먼저 읽고 검색(`/`)을 적극적으로 활용하세요. 대부분의 개념 chapter는 면접 당일에 다시 볼 수 있는 Q&A·follow-up·cheat-sheet를 제공합니다. 허브와 reference chapter는 탐색성을 우선합니다.

## 누구를 위한 책인가

이 책은 빅테크 랩과 AI 회사의 **Research Scientist (RS)**, **Applied Scientist (AS)**, 그리고 **ML Engineer / MLE-research** 직무 면접을 준비하는 후보자를 중심 독자로 삼습니다. 기초 chapter부터 시작할 수 있지만, 목표는 지식을 *면접에 맞는 형태*로 바꾸는 것입니다: 빠르게 떠올리고, 말로 구조화하며, 가정과 한계를 정확히 밝히는 형태입니다.

특히 **computer vision·multimodal·VLM·agent** 사례가 많습니다. 범용 학습 경로 위에 이 책의 개인 이력서 패킷과 사례가 해당 영역에 집중되어 있기 때문입니다. 순수 NLP나 systems 쪽이라면 coding, foundations, system-design, research, behavioral part를 중심으로 읽으세요.

## 멘탈 모델

loop가 측정하는 것은 네 가지입니다. 모든 chapter를 그중 하나에 매핑하세요:

| 축 | 실제로 테스트하는 것 | Parts |
| --- | --- | --- |
| **Coding** | 시간 압박 속에서 소리 내어 생각하면서 spec을 정확하고 깔끔한 코드로 옮길 수 있는가? | Coding · ML Coding From Scratch |
| **ML depth & breadth** | 무언가가 *왜* 작동하는지 이해하고, 자신의 전문 분야를 깊게 파고들면서 다른 모든 곳에서도 신뢰를 유지할 수 있는가? | Foundations · CV · LLM/VLM/Agents |
| **System design** | 모호한 문제의 범위를 정하고, 방어 가능한 trade-off를 내리며, data/serving/scale에 대해 추론할 수 있는가? | ML System Design |
| **Research & behavioral** | 자신의 연구가 준 임팩트를 설명하고, 과학자처럼 사고하며, 협업할 수 있는가? | Research Interviews · Behavioral · CV Deep-Dives |

## 각 chapter는 어떻게 구성되어 있나

- **Lead-in** 은 그 주제가 추상적으로가 아니라 *면접에서* 왜 중요한지를 짚어줍니다.
- **다이어그램과 인터랙티브 위젯** — attention heatmap, gradient-descent 애니메이션, precision/recall 슬라이더 — 조작할 수 있는 그림은 기억에 남기 때문입니다.
- **Q&A 블록** 은 자주 쓰이는 면접 질문 형식으로, *짧은 답변*(처음에 말하는 것)과 *깊은 답변*(더 밀어붙일 때 가는 곳)으로 구성됩니다.
- **"Follow-ups"** — 첫 답변 *이후에* 나오는 질문들. 면접은 여기서 이기거나 집니다.
- **cheat-sheet** 는 마지막에: 혀끝에 준비해 둬야 할 5–10개의 사실.

위젯을 써보세요 — 실제로 동작합니다:

<div class="widget" data-widget="activation"></div>

## 읽기 경로

<div class="card-grid">
  <a class="card" href="#/start/prep-plan"><div class="card-emoji">📅</div><div class="card-title">곧 loop가 있어요</div><div class="card-desc">2·4·8주 경로 중 남은 시간에 맞는 계획을 고르고, 실제 loop와 가장 약한 축을 기준으로 우선순위를 정하세요.</div></a>
  <a class="card" href="#/foundations/what-is-ml"><div class="card-emoji">🌱</div><div class="card-title">기초부터 복습할래요</div><div class="card-desc">머신러닝의 문제 설정부터 시작하고, CV가 처음이면 이미지·tensor 기초로 이어가세요.</div></a>
  <a class="card" href="#/coding/strategy"><div class="card-emoji">⌨️</div><div class="card-title">Coding이 약점이에요</div><div class="card-desc">Strategy → patterns → 반복 연습. ML-from-scratch part가 research 직무의 반전 요소입니다.</div></a>
  <a class="card" href="#/llm/reasoning"><div class="card-emoji">🤖</div><div class="card-title">2026 트렌드에 뒤처졌어요</div><div class="card-desc">Reasoning, alignment, agents, 그리고 VLM chapter가 빠르게 최신 상태로 만들어줍니다.</div></a>
  <a class="card" href="#/research/job-talk"><div class="card-emoji">📄</div><div class="card-title">Research 직무예요</div><div class="card-desc">Job talk·deep-dive 비중이 큰 loop라면 해당 준비를 우선하세요. 실제 라운드 구성은 먼저 확인합니다.</div></a>
</div>

### 학습용과 인터뷰용은 순서가 다릅니다

- **학습 모드:** 목차 순서를 따르세요. 수식은 손으로 다시 유도하고, `from scratch` 코드는 빈 파일에서 재작성한 뒤 작은 입력으로 검증합니다.
- **ML/CV 입문:** [머신러닝이란?](#/foundations/what-is-ml) → [수학 기초](#/foundations/linear-algebra-calculus) → [신경망 첫걸음](#/foundations/neural-networks-basics) 순서로 읽고, vision은 [이미지 기초](#/cv/image-basics)부터 시작합니다.
- **인터뷰 모드:** 먼저 [준비도 점수표](#/start/prep-plan)로 약한 축을 찾고, 해당 chapter의 질문에 **소리 내어** 답합니다. 읽는 시간보다 retrieval·mock·오답 수정 시간을 더 크게 잡으세요.
- **개인화 모드:** 범용 개념을 익힌 뒤 [이력서 맵](#/resume/overview)에서 자신의 프로젝트 주장과 연결하고, [단계별 예시 답변](#/resume/interview-stage-answers)을 30초·90초 버전으로 소리 내어 리허설합니다. 교재의 문장을 그대로 암기하지 말고 본인이 공개 가능한 근거로 바꾸세요.

## 관례

- **언어:** 설명은 한국어로 쓰되, 실제 문헌과 인터뷰에서 통용되는 기술 용어는 English를 병기하거나 그대로 유지합니다. 답변 연습은 지원 회사의 인터뷰 언어로 다시 하세요.
- **근거의 세 층:** 교과서적이고 안정적인 사실, 날짜·protocol에 민감한 frontier snapshot, 개인 이력/PDF에 근거한 주장으로 나눠 읽으세요. 두 번째와 세 번째는 출처와 공개 가능 범위를 반드시 다시 확인합니다.
- **목차 badge:** `primer`는 입문 선행지식, `core`는 핵심 개념, `lab`은 직접 구현, `hub`는 당일 routing, `practice`는 답변 연습, `2026`은 날짜에 민감한 frontier snapshot입니다.
- **수식**은 KaTeX(`$…$`), **다이어그램**은 Mermaid로 렌더링됩니다. **실행 코드**와 별개로 데이터 흐름을 설명하는 `개념 코드`·`PyTorch식 pseudocode`가 있으며, 긴 보조 코드는 클릭해 펼칩니다. 코드 블록에는 syntax highlight와 복사 버튼이 있고 callout은 tips/warnings를 표시합니다.
- **Cross-links** 가 관련 chapter를 연결합니다 — 따라가세요. 그 그래프가 *곧* 지식입니다.

> [!WARNING] 암기하지 말고 — 내재화하세요
> 이 책을 그대로 외워서 읊으면 짜인 대본처럼 들립니다. 이 책을 추론의 출발점이 될 *모델*을 만드는 데 쓰고, 그다음 소리 내어(가능하면 mock interview에서) 연습하세요. [Playbook](#/playbook/communication)이 전달 방식을 다룹니다.
