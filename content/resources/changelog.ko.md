# Changelog

<div class="tag-row"><span class="tag">living document</span><span class="tag">versioned</span><span class="tag">append-only</span></div>

> [!TIP] 이 페이지가 존재하는 이유
> 인터뷰 준비는 금방 낡습니다 — 2023년처럼 말하는 2026년 후보자는 뒤처져 보입니다. **이 책은 살아있는 문서(living document)**로, 프론티어가 움직일 때마다 업데이트됩니다. 이 페이지는 무엇이 언제 바뀌었는지를 기록해, 지난달에 공부한 내용이 여전히 최신인지 한눈에 파악할 수 있게 합니다. 모든 항목에는 날짜가 붙고, 가장 최신 항목이 맨 위에 옵니다.

> [!IMPORTANT] 최신성(Currency)
> 콘텐츠는 **2026년 7월** 기준 최신입니다. 모델 이름, benchmark, "SOTA" 주장은 시간이 지나면 낡습니다 — 항상 최신 수치를 primary source와 교차 확인하세요(참고: [Curated Resources → Track the frontier](#/resources/open-source)). 시간에 민감한 사실은 챕터에서 의도적으로 유보적으로 표현했습니다. 그 유보는 결함이 아니라 기능입니다.

## Conventions

- **Append-only.** 과거 항목을 다시 쓰거나 삭제하지 마세요 — 히스토리 자체가 핵심입니다. 변경을 반영하려면 맨 위에 날짜가 붙은 *새* 항목을 추가하세요.
- **역순 시간 정렬(Reverse chronological).** 최신 항목이 먼저입니다. 각 항목은 `## YYYY-MM-DD — title` 섹션입니다.
- **의미 기반 그룹화(Semantic-ish grouping).** 한 항목 안에서는 노트를 `**Added**`, `**Changed**`, `**Fixed**`, `**Removed**` 아래로 묶습니다([Keep a Changelog](https://keepachangelog.com/) 관례를 차용). 빈 그룹은 생략합니다.
- **변경 하나당 한 줄.** 영향을 받은 챕터를 해시 라우터로 링크해(예: `[Segmentation](#/cv/segmentation)`) 독자가 곧장 이동할 수 있게 합니다.
- **정직성 태그는 이어집니다.** 확실성이 중요한 주장을 기록한 항목이라면 [AUTHORING.md](#/resources/open-source)에 따라 *(verifiable)* / *(defensible opinion)* / *(speculative)* 태그를 붙입니다.

---

## 2026-07-19 — CLIP/DINO 학습 · contrastive learning · MLSD 사례 추가

**Added**
- [VLM Pretraining](#/vlm/pretraining): "**CLIP은 실제로 어떻게 학습되고 zero-shot을 하나**"(듀얼 인코더, ~400M 쌍, learnable temperature, 6줄 학습 의사코드, 프롬프트 기반 zero-shot) + 신규 **§1.5 Contrastive learning** 섹션(InfoNCE 일반형; SimCLR / MoCo / CLIP / Triplet 표; temperature; 고전 contrastive·triplet loss; representation collapse와 비대조 BYOL/SimSiam/DINO의 회피 방식).
- [Vision Foundation Models](#/cv/foundation-models): "**DINO는 어떻게 학습되나**" — self-distillation, EMA teacher, multi-crop, 목적식, **centering + sharpening**(negative 없이 collapse 회피), iBOT/KoLeo, student–teacher 도식.
- [ML System Design Case Studies](#/system-design/case-studies): 신규 사례 3개 — **E · 얼굴 인증 & liveness(anti-spoofing)**(FaceSign 연결), **F · 추천/랭킹**, **G · OCR/문서 이해** — tag-row·cheat-sheet 갱신.

**Changed**
- 세 챕터의 한국어(`.ko.md`) 버전도 동일하게 갱신.

---

## 2026-07-19 — 아키텍처 심화 · 손실 이론 · 시각 자료

**Added**
- [Architectures](#/foundations/architectures): depthwise-separable FLOP 유도(~9× 절감과 *이유*); residual connection이 vanishing gradient를 왜 해소하는지 **gradient 관점 증명**; activation **수식 표**(라이브 위젯 옆); **RNN/LSTM/GRU** 전체 섹션(unrolled 도식, cell-state highway SVG, GRU 수식, 장단점); **Transformer encoder–decoder 아키텍처 SVG**(원본 재현); **positional encoding** 직관(순서 무지·absolute vs relative·sinusoid fingerprint); **Mamba/SSM 심화**(selective SSM, O(1) 메모리·KV cache 없음, Transformer 대비 장단점).
- [Losses & Gradients](#/ml-coding/losses-gradients): "**왜 cross-entropy인가?**" — CE vs KL ($CE=H(p)+D_{KL}$), BCE vs CE, classification에 왜 L1/L2를 안 쓰는지(vanishing gradient 수식 + loss 곡선 그림).
- [Curated Open-Source Resources](#/resources/open-source): "**시각·애니메이션·인터랙티브 설명 자료**" 디렉터리(conv GIF, CNN/Transformer Explainer, Illustrated Transformer, 3Blue1Brown, colah LSTM, distill momentum, positional encoding, A Visual Guide to Mamba…), 링크 live 확인.

**Changed**
- 세 챕터의 한국어(`.ko.md`) 버전도 동일하게 갱신.

---

## 2026-07-18 — 코드 랩 전체 확장 (coding + ML-coding)

**Added**
- 인터랙티브 **코드 랩**이 이제 **모든 coding 패턴 챕터와 모든 ML-coding 챕터**를 커버합니다 — 총 **168개의 실행 가능한 랩**(영어 + 한국어). 각 랩은 starter 코드, 숨겨진 참고 **Solution**, 브라우저에서 즉시 채점되는 테스트 케이스를 갖습니다. coding 랩은 LeetCode로 연결되고, ML-coding 랩은 **NumPy**를 로드해 `numpy.allclose`로 채점합니다.

**Changed**
- 코드 랩 위젯 업그레이드: 채점을 Pyodide 내부의 Python 하네스로 옮기고(exact / 순서 무관 / 수치 근사), 패키지(numpy)를 지연 로드할 수 있게 했습니다. class 기반·training loop·plot 스니펫은 의도적으로 정적 참고 코드로 유지했습니다.
- 모든 참고 solution이 자체 테스트를 통과하는지(168/168) 배포 전에 검증했습니다.

---

## 2026-07-18 — 인터랙티브 코드 랩 (파일럿)

**Added**
- **실행 가능한 Python 코드 랩**(Pyodide, 지연 로드): 브라우저에서 직접 풀이를 작성하고 **Run tests**로 테스트 케이스에 대해 채점(pass/fail)하며, 참고용 **Solution**을 열어볼 수 있습니다 — attempt-first 연습. [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) 챕터에 파일럿 적용(문제 6개, 영어+한국어), LeetCode 링크 포함. `assets/widgets.js`의 새 `data-widget="code"` 위젯으로 구동.

**Planned**
- 파일럿 UX 확인 후 나머지 코딩 패턴 챕터와 ML-coding 챕터로 코드 랩 확장.

---

## 2026-07-18 — 한국어 / 영어 언어 토글

**Added**
- 상단 바에 **한 / EN 언어 토글** (theme 토글 옆) — 책 전체를 영어/한국어로 전환하며, 선택을 기억하고 기본값은 영어입니다. 한국어 모드에서도 technical 용어·code·math는 영어로 유지합니다.
- **전체 83개 챕터의 한국어 번역(`.ko.md`)** — KaTeX / Mermaid / callout / Q&A가 그대로 렌더링됩니다. 번역이 없는 챕터는 안내와 함께 영어로 fallback 되어 책이 깨지지 않습니다.

**Changed**
- 엔진(`assets/app.js`)이 한국어 모드에서 `<chapter>.ko.md`를 가져오고 없으면 영어로 fallback 합니다. UI(사이드바·이전/다음·검색·footer)도 현지화되었고, manifest(`assets/book.js`)에 한국어 part/chapter 제목(`titleKo`)이 추가되었습니다. 번역 추가·수정 방법은 `AUTHORING.md` 참고.

---

## 2026-07-18 — PEFT + deeper RL detail

**Added**
- [Post-Training & Alignment](#/llm/alignment)에 **PEFT 섹션** — full FT vs parameter-efficient tuning, LoRA($W_0+\frac{\alpha}{r}BA$ 수식과 왜 $B$를 0으로 초기화하는지 포함), QLoRA/DoRA/adapters/prefix/prompt/IA³ 비교 표, "무엇을 고를지" 가이드, 그리고 LoRA vs full-FT Q&A. [ECLIPSE](#/resume/eclipse)(visual prompt tuning) 및 [VLM practical](#/vlm/practical)과 상호 링크.

**Changed**
- [Post-Training & Alignment](#/llm/alignment)의 **GRPO**를 full clipped-surrogate + KL objective와 per-token importance ratio, 그리고 Dr. GRPO / GSPO가 이를 어떻게 수정하는지로 보강. cheat-sheet에 PEFT/LoRA/QLoRA 행 추가.
- (RLVR, RLHF, SFT, DPO, GRPO는 이미 깊이 다뤄져 있었음 — 이번 패스는 PEFT 공백을 메우고 method-level RL 수식을 추가함.)

---

## 2026-07-17 — CV deep-dive: NeurIPS'26 spatial-reasoning agent

**Added**
- [Deep-Dive: Spatial-Reasoning Agent (NeurIPS'26)](#/resume/neurips26-spatial-reasoning) — under-review 제출물의 편집된(redacted) 공개 write-up(메커니즘, architecture, ablation 통찰, 한계, 어려운 후속 Q&A). Codename과 정확한 수치는 double-blind 리뷰 동안 오프라인으로 보류. [Grounded VLM/Agents](#/resume/grounded-vlm-agents) 및 [CV overview](#/resume/overview)와 상호 링크.

---

## 2026-07-17 — Initial comprehensive edition

기존 스크래치 노트를 대체하는 client-side Markdown SPA 온라인 책으로의 전면 재구축. 모든 파트를 집필했고 2026년 7월 기준 최신.

**Added**
- **Getting Started** — [How to Use This Book](#/start/how-to-use), [The 2026 Landscape](#/start/landscape-2026), [Prep Plan](#/start/prep-plan): 프론티어 프레이밍(reasoning/RLVR, native multimodal, MoE, agents, test-time compute)과 시간 배분형 학습 계획.
- **Foundations** — [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus), [Probability & Statistics](#/foundations/probability-statistics), [Architectures](#/foundations/architectures): 수학과 model-family의 기반.
- **Coding** — [Strategy](#/coding/strategy), [Patterns](#/coding/patterns), [Arrays & Strings](#/coding/arrays-strings), [Binary Search](#/coding/binary-search), [Trees & BST](#/coding/trees-bst): pattern-first 코딩 뼈대.
- **ML Coding** — [Intro](#/ml-coding/intro), [Conv & Pooling](#/ml-coding/conv-pooling), [NMS & IoU](#/ml-coding/nms-iou): 밑바닥부터 구현하는 ML primitive.
- **Computer Vision** — [Segmentation](#/cv/segmentation): semantic / instance / panoptic, metric, 그리고 현대 foundation model.
- **LLM** — [Fundamentals](#/llm/fundamentals): transformer, attention, tokenization, decoding, 그리고 post-training 스택.
- **Process** — [Interview Pipeline](#/process/pipeline), [Recruiter & Hiring Manager](#/process/recruiter-hm): 루프 구조와 비기술 라운드.
- **Resources** — [Curated Open-Source Resources](#/resources/open-source), [Glossary](#/resources/glossary), 그리고 이 changelog.

**Changed**
- 기존 `resources.md` 링크 목록을 "best for" + "steal this" 노트, "if you only use 5" 요약, frontier-tracking 섹션을 갖춘 주석 달린 디렉터리로 이식·업그레이드.

**Notes**
- 대상 독자는 Meta, NVIDIA, Apple, Adobe, ByteDance, Mistral, Microsoft에서 인터뷰하는 CV / VLM / visual agents 분야의 **PhD 수준 Applied/Research Scientist**.
- 최신(2025–2026) 모델의 benchmark 수치는 벤더 발표인 경우가 많음. 챕터는 정확한 점수보다 메커니즘/역량을 우선하고 그에 맞게 유보적으로 표현함. *(defensible opinion)*

---

## How to add an entry

챕터를 업데이트할 때는 여기에 기록하세요 — 미래의 자신(그리고 협업자)에게 그 자취가 필요합니다.

1. **챕터 작성 또는 편집** `content/**` 아래의 `.md`를 [AUTHORING.md](#/resources/open-source)를 정확히 따라(voice, chapter shape, callouts, diagrams, cross-links) 작성합니다.
2. **새 챕터 등록**(있는 경우) `assets/book.js`의 올바른 파트에 한 줄을 추가 — `{ id, title, file }`. 기존 챕터는 변경 불필요.
3. **날짜 붙은 항목을 맨 위에 추가** 이전 항목 위에, `## YYYY-MM-DD — title` 형식과 Added/Changed/Fixed/Removed 그룹을 사용. 손댄 모든 챕터를 링크하세요.
4. **최신성 갱신** — 편집이 "current to" 날짜를 옮긴다면, 이 페이지 상단의 currency callout을, 필요하면 [The 2026 Landscape](#/start/landscape-2026)도 업데이트하세요.
5. **append-only 유지** — 과거 항목을 편집하지 말고 새 항목으로 대체(supersede)하세요.

> [!NOTE] Contributing
> 책을 정의하는 두 파일은 콘텐츠(`content/**.md`)와 목차(`assets/book.js`)입니다. 전체 스타일 계약 — voice, 필수 chapter shape, 사용 가능한 Markdown 기능(callouts, Q&A accordions, KaTeX, Mermaid, widgets, SVG figures, cards, badges), 그리고 하지 말아야 할 것들 — 은 `AUTHORING.md`에 있습니다. 쓰기 전에 읽고, 그다음 변경을 여기에 기록하세요.

**Related:** [How to Use This Book](#/start/how-to-use) · [Curated Open-Source Resources](#/resources/open-source) · [Glossary](#/resources/glossary)
