<div class="hero">
  <div class="eyebrow">Research &amp; Applied Scientist · Computer Vision · VLMs · Agents</div>
  <h1>The ML Interview Codex</h1>
  <p class="sub">최고 수준 AI 랩의 research 또는 applied-scientist 자리를 얻기 위한, 종합적이고 지속적으로 업데이트되는 현장 가이드 — coding과 ML 기초부터 VLMs, agents, system design, research job talk, 그리고 CV 특화 심화까지. <b>2026년 7월</b> 기준 최신.</p>
  <div class="stat-row">
    <div class="stat"><div class="n">13</div><div class="l">Parts</div></div>
    <div class="stat"><div class="n">70+</div><div class="l">Chapters</div></div>
    <div class="stat"><div class="n">2026</div><div class="l">Frontier-current</div></div>
    <div class="stat"><div class="n">∞</div><div class="l">Living document</div></div>
  </div>
</div>

> [!TIP] 처음 오셨나요? 먼저 **[How to Use This Book](#/start/how-to-use)**을 읽고, 그다음 **[The 2026 Landscape](#/start/landscape-2026)**를 훑어보며 지금 면접관들이 무엇을 기대하는지 감을 잡으세요. 다음 주에 loop가 잡혀 있다면 **[8-Week Prep Plan](#/start/prep-plan)**으로 바로 가서 우선순위를 정리하세요.

## 이 책이 무엇인가

대부분의 면접 자료는 (a) coding만 갈아 넣는 문제 리스트이거나 (b) 2021년에 쓰인 일반적인 ML-101 플래시카드입니다. 이 책은 2026년의 **research/applied scientist** 트랙을 위해 만들어졌습니다 — reasoning 모델, RLVR, native multimodality, agentic tool use에 대해 추론할 수 있어야 *하면서도* 여전히 non-max suppression을 화이트보드에 깔끔하게 그려내고, 자신의 research를 설득력 있게 발표하며, behavioral loop를 통과해야 하는 트랙 말이죠.

이 책은 **처음부터 끝까지 읽을 수 있는 책**으로도, **주제별로 골라 읽는 책**으로도 구성되어 있습니다. 모든 chapter는 독립적이며, 다이어그램과 실전 예제가 풍부하고, 무엇이 검증 가능한 사실이고 무엇이 방어 가능한 의견인지에 대해 솔직합니다.

## research-scientist loop의 네 가지 축

<figure>
<svg viewBox="0 0 720 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif">
  <defs>
    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e0533f"/><stop offset="1" stop-color="#6366f1"/></linearGradient>
  </defs>
  <g>
    <rect x="10" y="20" width="165" height="170" rx="12" fill="none" stroke="#e0533f" stroke-width="2"/>
    <text x="92" y="48" text-anchor="middle" font-size="30">⌨️</text>
    <text x="92" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#e0533f">Coding</text>
    <text x="92" y="104" text-anchor="middle" font-size="11" fill="#98a3b2">DSA patterns</text>
    <text x="92" y="122" text-anchor="middle" font-size="11" fill="#98a3b2">ML-from-scratch</text>
    <text x="92" y="140" text-anchor="middle" font-size="11" fill="#98a3b2">live problem-solving</text>
  </g>
  <g>
    <rect x="188" y="20" width="165" height="170" rx="12" fill="none" stroke="#6366f1" stroke-width="2"/>
    <text x="270" y="48" text-anchor="middle" font-size="30">🧠</text>
    <text x="270" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#6366f1">ML depth &amp; breadth</text>
    <text x="270" y="104" text-anchor="middle" font-size="11" fill="#98a3b2">DL foundations</text>
    <text x="270" y="122" text-anchor="middle" font-size="11" fill="#98a3b2">CV · LLM · VLM · agents</text>
    <text x="270" y="140" text-anchor="middle" font-size="11" fill="#98a3b2">2026 frontier</text>
  </g>
  <g>
    <rect x="366" y="20" width="165" height="170" rx="12" fill="none" stroke="#0ea5e9" stroke-width="2"/>
    <text x="448" y="48" text-anchor="middle" font-size="30">🏗️</text>
    <text x="448" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#0ea5e9">System design</text>
    <text x="448" y="104" text-anchor="middle" font-size="11" fill="#98a3b2">ML pipelines</text>
    <text x="448" y="122" text-anchor="middle" font-size="11" fill="#98a3b2">LLM/agent systems</text>
    <text x="448" y="140" text-anchor="middle" font-size="11" fill="#98a3b2">serving &amp; scale</text>
  </g>
  <g>
    <rect x="544" y="20" width="165" height="170" rx="12" fill="none" stroke="#12a150" stroke-width="2"/>
    <text x="626" y="48" text-anchor="middle" font-size="30">📄</text>
    <text x="626" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#12a150">Research + behavioral</text>
    <text x="626" y="104" text-anchor="middle" font-size="11" fill="#98a3b2">job talk</text>
    <text x="626" y="122" text-anchor="middle" font-size="11" fill="#98a3b2">deep-dive on your work</text>
    <text x="626" y="140" text-anchor="middle" font-size="11" fill="#98a3b2">STAR stories</text>
  </g>
</svg>
<figcaption>네 가지 평가 축. Research 직무는 표준 MLE loop보다 오른쪽 두 축에 훨씬 큰 가중치를 둡니다.</figcaption>
</figure>

## 읽기 시작하기

<div class="card-grid">
  <a class="card" href="#/start/landscape-2026"><div class="card-emoji">🛰️</div><div class="card-title">The 2026 Landscape</div><div class="card-desc">무엇이 바뀌었나: reasoning 모델, RLVR, native multimodal, agents. 기대치를 보정하세요.</div></a>
  <a class="card" href="#/coding/patterns"><div class="card-emoji">⌨️</div><div class="card-title">Coding Patterns</div><div class="card-desc">대부분의 coding 라운드를 커버하는 ~15개 패턴, cue→pattern 조회표 포함.</div></a>
  <a class="card" href="#/foundations/optimization"><div class="card-emoji">📐</div><div class="card-title">DL Foundations</div><div class="card-desc">Optimization, normalization, architectures — 인터랙티브 시각화와 함께.</div></a>
  <a class="card" href="#/llm/reasoning"><div class="card-emoji">🤖</div><div class="card-title">Reasoning &amp; Agents</div><div class="card-desc">Test-time compute, RLVR, tool use, visual agents — 2026년의 핫존.</div></a>
  <a class="card" href="#/system-design/framework"><div class="card-emoji">🏗️</div><div class="card-title">ML System Design</div><div class="card-desc">반복 사용 가능한 framework와 research/applied 직무를 위한 실전 케이스 스터디.</div></a>
  <a class="card" href="#/resume/predicted-questions"><div class="card-emoji">🎯</div><div class="card-title">Your CV Q&amp;A</div><div class="card-desc">Beomyoung의 CV에서 예측한 질문들과 여러 강력한 모범 답안.</div></a>
</div>

> [!NOTE] 살아있는 문서
> 모델, benchmark, 채용 기준이 움직이는 대로 내용은 계속 개정됩니다. 각 part에는 "last-reviewed" 노트가 있고, [Changelog](#/resources/changelog)가 새로운 내용을 추적합니다. 오래되거나 틀린 부분을 발견하셨나요? Markdown 파일 하나일 뿐이니 — 고쳐서 push하세요.
