# How to Use This Book

> [!TIP] 30초 요약
> **가장 약한 축**에 해당하는 part를 먼저 읽으세요. 검색(`/`)을 적극적으로 활용하세요. 모든 chapter는 면접 당일 아침에 훑어볼 수 있는 압축된 cheat-sheet로 끝납니다.

## 누구를 위한 책인가

이 책은 빅테크 랩과 frontier AI 회사의 **Research Scientist (RS)**, **Applied Scientist (AS)**, 그리고 시니어 **ML Engineer / MLE-research** 직무 면접을 준비하는 후보자를 대상으로 합니다. 여러분이 이미 ML을 안다고 가정합니다 — 목표는 여러분의 지식을 *면접에 맞는 형태*로 만드는 것입니다: 빠르게 떠올릴 수 있고, 말로 잘 구조화되며, 한계에 대해 솔직한 형태로요.

특히 **VLMs와 agents**로 나아가는 **computer-vision / multimodal** 배경에 맞춰 튜닝되어 있습니다. 그게 이 분야가 향하는 방향이자 저자가 일하는 곳이기 때문입니다. 순수 NLP나 systems 쪽이라면, coding, foundations, system-design, research, behavioral part는 모두 여전히 직접적으로 유용합니다.

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
- **Q&A 블록** 은 면접관이 실제로 묻는 형태 그대로, *짧은 답변*(처음에 말하는 것)과 *깊은 답변*(더 밀어붙일 때 가는 곳)으로 구성됩니다.
- **"Follow-ups"** — 첫 답변 *이후에* 나오는 질문들. 면접은 여기서 이기거나 집니다.
- **cheat-sheet** 는 마지막에: 혀끝에 준비해 둬야 할 5–10개의 사실.

위젯을 써보세요 — 실제로 동작합니다:

<div class="widget" data-widget="activation"></div>

## 읽기 경로

<div class="card-grid">
  <a class="card" href="#/start/prep-plan"><div class="card-emoji">📅</div><div class="card-title">곧 loop가 있어요</div><div class="card-desc">8-week 계획으로 가서, company playbook과 가장 약한 축을 기준으로 우선순위를 정하세요.</div></a>
  <a class="card" href="#/coding/strategy"><div class="card-emoji">⌨️</div><div class="card-title">Coding이 약점이에요</div><div class="card-desc">Strategy → patterns → 반복 연습. ML-from-scratch part가 research 직무의 반전 요소입니다.</div></a>
  <a class="card" href="#/llm/reasoning"><div class="card-emoji">🤖</div><div class="card-title">2026 트렌드에 뒤처졌어요</div><div class="card-desc">Reasoning, alignment, agents, 그리고 VLM chapter가 빠르게 최신 상태로 만들어줍니다.</div></a>
  <a class="card" href="#/research/job-talk"><div class="card-emoji">📄</div><div class="card-title">Research 직무예요</div><div class="card-desc">Job talk과 deep-dive 준비가 LeetCode medium 하나 더 푸는 것보다 중요합니다.</div></a>
</div>

## 관례

- **언어:** 영어 중심입니다. 실제 답변을 영어로 하게 될 테니까요. 물론 기술 용어, 수식, 코드는 만국 공통입니다.
- **정직성 표시:** 주장이 *검증 가능한 것*인지, *방어 가능한 의견*인지, *추측성 방향*인지 태그로 표시합니다. 면접에서는 보정된 확신이 거짓 확신을 이깁니다.
- **수식** 은 KaTeX(`$…$`)로, **다이어그램** 은 Mermaid로 렌더링되고, **코드** 는 syntax-highlight에 복사 버튼이 있으며, callout이 tips/warnings를 표시합니다.
- **Cross-links** 가 관련 chapter를 연결합니다 — 따라가세요. 그 그래프가 *곧* 지식입니다.

> [!WARNING] 암기하지 말고 — 내재화하세요
> 이 책을 그대로 외워서 읊으면 짜인 대본처럼 들립니다. 이 책을 추론의 출발점이 될 *모델*을 만드는 데 쓰고, 그다음 소리 내어(가능하면 mock interview에서) 연습하세요. [Playbook](#/playbook/communication)이 전달 방식을 다룹니다.
