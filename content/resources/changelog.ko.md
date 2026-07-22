# Changelog

<div class="tag-row"><span class="tag">living document</span><span class="tag">versioned</span><span class="tag">append-only</span></div>

> [!TIP] 이 페이지가 존재하는 이유
> 인터뷰 준비는 금방 낡습니다 — 2023년처럼 말하는 2026년 후보자는 뒤처져 보입니다. <strong>이 책은 살아있는 문서(living document)</strong>로, 프론티어가 움직일 때마다 업데이트됩니다. 이 페이지는 무엇이 언제 바뀌었는지를 기록해, 지난달에 공부한 내용이 여전히 최신인지 한눈에 파악할 수 있게 합니다. 모든 항목에는 날짜가 붙고, 가장 최신 항목이 맨 위에 옵니다.

> [!IMPORTANT] 최신성(Currency)
> 콘텐츠는 **2026년 7월** 기준 최신입니다. 모델 이름, benchmark, "SOTA" 주장은 시간이 지나면 낡습니다 — 항상 최신 수치를 primary source와 교차 확인하세요(참고: [Curated Resources → Track the frontier](#/resources/open-source)). 시간에 민감한 사실은 챕터에서 의도적으로 유보적으로 표현했습니다. 그 유보는 결함이 아니라 기능입니다.

## Conventions

- **Append-only.** 과거 항목을 다시 쓰거나 삭제하지 마세요 — 히스토리 자체가 핵심입니다. 변경을 반영하려면 맨 위에 날짜가 붙은 *새* 항목을 추가하세요.
- **역순 시간 정렬(Reverse chronological).** 최신 항목이 먼저입니다. 각 항목은 `## YYYY-MM-DD — title` 섹션입니다.
- **의미 기반 그룹화(Semantic-ish grouping).** 한 항목 안에서는 노트를 `**Added**`, `**Changed**`, `**Fixed**`, `**Removed**` 아래로 묶습니다([Keep a Changelog](https://keepachangelog.com/) 관례를 차용). 빈 그룹은 생략합니다.
- **변경 하나당 한 줄.** 영향을 받은 챕터를 해시 라우터로 링크해(예: `[Segmentation](#/cv/segmentation)`) 독자가 곧장 이동할 수 있게 합니다.
- **정직성 태그는 이어집니다.** 확실성이 중요한 주장을 기록한 항목이라면 repo의 `AUTHORING.md`에 따라 *(verifiable)* / *(defensible opinion)* / *(speculative)* 태그를 붙입니다.

---

## 2026-07-21 — 최종 QA · 이력서 사실 재대조 · UI hardening

같은 날의 UI·번역 항목을 보충합니다. 최종 통합 검사와 `local/resume.pdf` 재대조에서 발견한 마지막 정합성·안전성 문제를 반영했습니다.

**Changed**
- 이력서에 실제로 명시된 ownership은 과도하게 흐리지 않도록 [FaceSign](#/resume/facesign)의 anti-spoofing model 구축, [On-Device Segmentation](#/resume/on-device-segmentation)의 독자적 개발·약 10 ms·ONNX serving, foreground API의 model·dataset 개발과 내부 비교를 답변에 복원했습니다. 동시에 system 전체 ownership, 비공개 protocol·수치, 사용 기법은 별도 확인하도록 경계를 유지했습니다.
- [단계별 예시 답변](#/resume/interview-stage-answers)과 [예상 질문](#/resume/predicted-questions)을 위 사실 수준에 맞춰 다시 개인화하고 한·영 문서를 함께 동기화했습니다.

**Fixed**
- mobile overflow table의 번역 함수 shadowing, skip link와 hash router 충돌, unknown route의 HTML injection, 빠른 route·언어 전환의 stale render, 검색 index의 언어 race와 중복 fetch, navigation 뒤 focus handoff, callout 제목의 inline code 손실, light-theme focus/faint-text contrast를 수정했습니다.
- 최종 검증에서 105개 한·영 route, 언어별 119개 widget·448개 test, 38개 conceptual code·30개 answer card, 전체 Python fence 180개를 통과했습니다.

---

## 2026-07-21 — UI · 개념 코드 · 이력서 답변 · 영문 동기화

같은 날의 앞선 재구성 기록을 수정하지 않고 보충합니다. 현재 manifest 기준은 <strong>105개 챕터 · 15개 파트</strong>이며, 모든 route에 한국어·영어 문서가 함께 있습니다.

**Added**
- ML·CV·LLM·VLM·system/research의 수식 중심 설명에 <strong>38개 conceptual Python/PyTorch 블록</strong>을 추가했습니다. 그중 긴 21개는 [이 책 사용법](#/start/how-to-use)의 규약에 따라 클릭해서 펼치는 개념 코드로 표시합니다.
- 현재 이력서를 기반으로 recruiter부터 HM, coding/ML depth, technical deep-dive, system design, job talk, behavioral, closing까지 <strong>30개 클릭형 예시 답변</strong>을 담은 [단계별 예시 답변](#/resume/interview-stage-answers)을 추가했습니다. 이력서 확인·공개 근거·회사별 치환·본인 확인 필요 상태를 분리합니다.
- 105개 한·영 문서의 누락, 구조 parity, 내부 route/anchor, code-widget JSON, 영문 한글 잔여를 검사하는 저장소 검증 명령을 추가했습니다.

**Changed**
- landing hero, sidebar의 책 위치, prose·callout·table·figure·code·widget, 답변 카드와 개념 코드, light/dark contrast, 검색 modal, mobile drawer를 하나의 반응형 UI 체계로 재정리했습니다.
- 한국어에서 완료한 재구성·팩트 교정·개념 코드를 105개 영어 문서에 동기화하고, 번역 과정에서 이력서가 보장하지 않는 개인 ownership·내부 성과·갈등 서사를 만들지 않도록 동일한 사실 경계를 적용했습니다.
- <code>AUTHORING.md</code>를 한국어 우선 편집 → 영문 동기화 workflow와 전용 answer/concept-code 규약에 맞게 갱신했습니다.

**Fixed**
- 검색을 다시 열 때 query와 결과가 어긋나던 문제, 전체 결과와 표시 상한의 혼동, route 전환 뒤 modal 잔류, 접기 카드의 불필요한 접근성 텍스트, mobile drawer 뒤 본문 focus 노출을 수정했습니다.
- sidebar·검색·언어/테마·목차의 정적 접근성 label을 현재 언어에 맞게 현지화하고 390–1440px 대표 viewport에서 수평 overflow와 focus 복원을 점검했습니다.

---

## 2026-07-21 — 재구성 QA · 한국어 근거 교정 · renderer 보강

같은 날 먼저 기록한 재구성 항목을 <strong>수정하지 않고 보충(supersede)</strong>합니다. 현재 manifest 기준은 <strong>104개 챕터 · 15개 파트</strong>입니다.

**Added**
- [머신러닝이란?](#/foundations/what-is-ml)의 고전 ML 기준선 지도, [신경망 첫걸음](#/foundations/neural-networks-basics)의 canonical train loop, [CNN · RNN · Transformer](#/foundations/architectures)의 diffusion·flow-matching 구분을 보강했습니다.
- [Two Pointers · Sliding Window](#/coding/two-pointers-sliding-window)에 linked-list 정본을, [Trees · BST](#/coding/trees-bst)에 backtracking 정본과 드릴을 추가했습니다.

**Changed**
- 홈과 [이 책 사용법](#/start/how-to-use)에 학습·인터뷰·개인 이력서 경로를 분리하고, [준비 플랜](#/start/prep-plan)을 **2·4·8주** 경로로 정리했습니다.
- `questions-to-ask`는 범용 <strong>Process</strong>가 아니라 당일 실행·질문·오퍼를 묶은 후반부 파트에 두었습니다. From-Scratch는 설명을 완전히 배제한 “lab 전용”이 아니라 <strong>직접 구현과 검증 중심</strong>으로 읽습니다.
- [2026 지형도](#/start/landscape-2026), [behavioral](#/behavioral/star), [개인 이력서 패킷](#/resume/overview)의 한국어 문서를 검토해 날짜 의존 주장, 회사 고정관념, 검증되지 않은 ownership·갈등·내부 수치, 심사 중 결과를 source/status-aware 표현으로 바꿨습니다. 이번 내용 교정은 한국어 문서에만 적용했습니다.

**Fixed**
- [Attention](#/ml-coding/attention)·[LLM Fundamentals](#/llm/fundamentals)의 attention/KV 설명, [IoU & NMS](#/ml-coding/nms-iou)·[mAP & mIoU](#/ml-coding/metrics-map-miou)·[K-Means](#/ml-coding/kmeans)의 수식·edge case를 교정했습니다.
- [Alignment](#/llm/alignment)의 PPO·outcome reward model, [RoPE](#/ml-coding/positional-encoding-rope), [VLM pretraining](#/vlm/pretraining)의 fusion, [Grounding](#/vlm/grounding)의 답-근거 평가, [Agents](#/llm/agents)의 MCP transport·evaluation 보안을 최신 근거와 범위에 맞게 교정했습니다.
- 한국어 heading의 Unicode slug, 같은 이름 heading의 중복 ID suffix, percent-encoded 한글 anchor 복원, 의미 기반 badge label을 renderer에서 보강했습니다.
- 변경된 asset이 오래된 browser cache에 남지 않도록 asset URL cache-bust를 갱신했습니다.
- [Phoenix](#/resume/phoenix-mask-refinement)의 uncertainty 설명과 ECLIPSE backbone 연결, [ECLIPSE](#/resume/eclipse)의 freezing 표현, 심사 중 [Spatial-Reasoning Agent](#/resume/neurips26-spatial-reasoning)의 비공개 비교 결론, 이력서 기반 behavioral 답변의 사실/초안 경계를 교정했습니다.
- 외부 자료 설명·내부 hash route·Q&A HTML nesting을 다시 점검하고, 현재 접근이 제한된 링크에는 검증 주의문을 붙였습니다.

---

## 2026-07-21 — 대규모 재구성: 104개 챕터 · 15개 파트

책 전체를 재편해 이제 **104개 챕터 (15개 파트)** 구조가 되었습니다. 챕터 수를 명시하는 곳은 모두 104로 갱신하세요.

**Changed**
- **파트 14 → 15로 확장.** 비대해진 "**LLMs · VLMs · Agents**"(18챕터)를 3개 파트로 분할 — **LLM Core** / **LLM 정렬 · 추론 · 에이전트** / **VLM · 비주얼 에이전트**. 각 파트가 한 번에 읽을 만한 크기가 됩니다.
- <strong>"Advanced: Scaling & Efficiency"(2챕터)</strong>를 독립 파트에서 <strong>ML · DL Foundations</strong>의 **심화(선택)** 섹션으로 흡수. Foundations 안에서 기본 → 심화로 자연스럽게 이어집니다.
- <strong>architectures</strong>와 <strong>normalization-stability</strong>를 <strong>Foundations</strong>로 이동 — model-family와 학습 안정성 기초를 한 파트에 모았습니다.
- **From-Scratch 파트는 이제 lab 전용**(실행 가능한 코드 랩만) — 설명 텍스트는 해당 주제 챕터로 흡수하고 중복을 제거했습니다.
- **Computer Vision 재정렬** — upsampling 챕터를 segmentation과 <strong>인접</strong>하게 배치해 decoder/upsampling → segmentation 흐름이 이어지도록.
- **Research 파트 재정렬** — presenting → job-talk 순서로 재배치해 발표 준비에서 job talk으로 자연스럽게 이어집니다.
- **questions-to-ask** 챕터를 <strong>Company Playbooks</strong>에서 <strong>Process</strong>로 이동 — 프로세스 준비의 일부로 더 잘 맞습니다.

**Fixed**
- 콘텐츠 **중복 제거**, **phone-screens / behavioral** 커버리지 통합, **한국어 / 영어 톤 통일**, 그리고 여러 **팩트 및 링크 수정**(끊긴 hash route, cross-link 대상, ZIM 데이터 수치 등).

---

## 2026-07-19 — RAG/MCP · 모델 네이밍 · 극한 이미지 · 비디오 패러다임

**Added**
- [Agentic AI & Tool Use](#/llm/agents): **MCP**(Model Context Protocol) 서브섹션(M×N→M+N, host/client/server, primitive, 공격면) + **RAG** 섹션(chunk/embed/index/rerank/generate, RAG vs long-context vs fine-tune, 실패 모드, agentic RAG).
- [LLM Fundamentals](#/llm/fundamentals) §8: **모델 이름 읽기** — Base / Instruct / **Thinking** / -Zero 학습 접미사와 dense / **`A3B`**(MoE active) / **`E4B`**(MatFormer effective footprint) 크기 태그.
- [VLM Implementation Details](#/vlm/practical) §3½: **극한 입력** — 작은 물체·1:100 aspect ratio·초대형 이미지; *fixed resize가 왜 실패하는지*, native-res / AnyRes tiling / coarse-to-fine / bucketing / sliding-window, 그리고 5단계 **ML 시스템 디자인 답변 프레이밍**.
- [Video-Language Models](#/vlm/video): **비디오 3대 패러다임** — sparse-sampling → VLM(understanding) vs **streaming/recurrent memory(SAM 2)** vs **clip/window bidirectional**(3D conv / spatio-temporal attn) vs tracking-by-detection.

**Changed**
- 네 챕터의 한국어(`.ko.md`) 버전도 동일하게 갱신.

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
- 인터랙티브 <strong>코드 랩</strong>이 이제 <strong>모든 coding 패턴 챕터와 모든 ML-coding 챕터</strong>를 커버합니다 — 총 **168개의 실행 가능한 랩**(영어 + 한국어). 각 랩은 starter 코드, 숨겨진 참고 **Solution**, 브라우저에서 즉시 채점되는 테스트 케이스를 갖습니다. coding 랩은 LeetCode로 연결되고, ML-coding 랩은 <strong>NumPy</strong>를 로드해 `numpy.allclose`로 채점합니다.

**Changed**
- 코드 랩 위젯 업그레이드: 채점을 Pyodide 내부의 Python 하네스로 옮기고(exact / 순서 무관 / 수치 근사), 패키지(numpy)를 지연 로드할 수 있게 했습니다. class 기반·training loop·plot 스니펫은 의도적으로 정적 참고 코드로 유지했습니다.
- 모든 참고 solution이 자체 테스트를 통과하는지(168/168) 배포 전에 검증했습니다.

---

## 2026-07-18 — 인터랙티브 코드 랩 (파일럿)

**Added**
- **실행 가능한 Python 코드 랩**(Pyodide, 지연 로드): 브라우저에서 직접 풀이를 작성하고 <strong>Run tests</strong>로 테스트 케이스에 대해 채점(pass/fail)하며, 참고용 <strong>Solution</strong>을 열어볼 수 있습니다 — attempt-first 연습. [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) 챕터에 파일럿 적용(문제 6개, 영어+한국어), LeetCode 링크 포함. `assets/widgets.js`의 새 `data-widget="code"` 위젯으로 구동.

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
- [Post-Training & Alignment](#/llm/alignment)의 <strong>GRPO</strong>를 full clipped-surrogate + KL objective와 per-token importance ratio, 그리고 Dr. GRPO / GSPO가 이를 어떻게 수정하는지로 보강. cheat-sheet에 PEFT/LoRA/QLoRA 행 추가.
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

1. **챕터 작성 또는 편집** `content/**` 아래의 `.md`를 repo의 `AUTHORING.md`를 정확히 따라(voice, chapter shape, callouts, diagrams, cross-links) 작성합니다.
2. **새 챕터 등록**(있는 경우) `assets/book.js`의 올바른 파트에 한 줄을 추가 — `{ id, title, file }`. 기존 챕터는 변경 불필요.
3. **날짜 붙은 항목을 맨 위에 추가** 이전 항목 위에, `## YYYY-MM-DD — title` 형식과 Added/Changed/Fixed/Removed 그룹을 사용. 손댄 모든 챕터를 링크하세요.
4. **최신성 갱신** — 편집이 "current to" 날짜를 옮긴다면, 이 페이지 상단의 currency callout을, 필요하면 [The 2026 Landscape](#/start/landscape-2026)도 업데이트하세요.
5. **append-only 유지** — 과거 항목을 편집하지 말고 새 항목으로 대체(supersede)하세요.

> [!NOTE] Contributing
> 책을 정의하는 두 파일은 콘텐츠(`content/**.md`)와 목차(`assets/book.js`)입니다. 전체 스타일 계약 — voice, 필수 chapter shape, 사용 가능한 Markdown 기능(callouts, Q&A accordions, KaTeX, Mermaid, widgets, SVG figures, cards, badges), 그리고 하지 말아야 할 것들 — 은 `AUTHORING.md`에 있습니다. 쓰기 전에 읽고, 그다음 변경을 여기에 기록하세요.

**Related:** [How to Use This Book](#/start/how-to-use) · [Curated Open-Source Resources](#/resources/open-source) · [Glossary](#/resources/glossary)
