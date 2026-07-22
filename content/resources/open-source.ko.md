# 엄선한 공개 학습 자료

<div class="tag-row"><span class="tag">free</span><span class="tag">open-source</span><span class="tag">coding</span><span class="tag">ML depth</span><span class="tag">system design</span><span class="tag">LLM/GenAI</span><span class="tag">behavioral</span><span class="tag">mock</span></div>

> [!TIP] 이 페이지 사용법
> 이건 *주석 달린* 디렉터리지 링크 덤프가 아닙니다. 각 resource가 어떤 상황에 유용한지와 재사용할 학습 구조를 적었습니다. 일부는 open-source가 아니라 무료 공개 자료이거나 유료 기능을 함께 제공하므로 license·paywall을 각 사이트에서 확인하세요. 공통 교훈은 **무작위 질문 덤프가 아니라 재사용 가능한 pattern으로 조직하라**는 것입니다.

> [!NOTE] 링크 상태
> 마지막 목록 검토일은 **2026-07-21**입니다. URL·가격·license·course 범위는 바뀔 수 있습니다. GitHub star 수처럼 학습 가치와 무관하고 빠르게 변하는 숫자는 의도적으로 제거했으며, 미러보다 저자/기관의 원본을 우선합니다.

## If you only use 5 resources

PhD 수준 Applied/Research-Scientist 루프(CV / VLM / visual agents)를 위한, 최소한의 고레버리지 세트입니다.

| # | Resource | Why it's on the shortlist |
| --- | --- | --- |
| 1 | [Chip Huyen — Introduction to ML Interviews](https://huyenchip.com/ml-interviews-book/) | ML interview의 기술 질문과 career/process를 한 흐름에서 훑는 공개 책. 정확한 문항 수보다 edition과 목차를 확인하세요. |
| 2 | [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/machine-learning-interviews) | **9단계 ML system-design 프레임워크**와 인터뷰 전체를 떠받치는 4-pillar 루프. |
| 3 | [NeetCode roadmap + 150](https://neetcode.io/roadmap) | 널리 쓰이는 DSA topic roadmap과 의존성 기반 학습 경로. 현재 무료·유료 범위를 확인하세요. |
| 4 | [Amidi Stanford CS229 / CS230 cheatsheets](https://stanford.edu/~shervine/teaching/cs-229/) | 수식 밀도가 높은 한 페이지 recall sheet로 마지막 복습에 유용. |
| 5 | [aishwaryanr/awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) | 역할 기반 준비 + VLM 독자에 1:1로 대응되는 multimodal/embeddings 질문 taxonomy. |

아래의 모든 것은 이 5개를 확장하고 공백(behavioral, mock, 회사 추적)을 메웁니다.

## Coding interview prep

<details class="qa"><summary>PhD 후보자가 정말 코딩 훈련(grind)을 해야 하나요?</summary>
<div class="qa-body">

**짧게:** 실제 loop에 coding이 포함된다면 범위를 좁혀 연습하세요. 라운드 수와 난도는 role마다 다르므로 recruiter에게 확인하고, 문제 수보다 반복되는 pattern의 coverage와 설명·검증 습관을 우선합니다.

**깊게:** NeetCode/Grokking/sean-prashad의 전체 가치 제안은, 인터뷰 문제의 공간이 반복되는 소수의 형태로 붕괴한다는 것입니다. 그 형태를 내재화하고, 각각 5~10문제씩 풀면 일반화됩니다. [Coding Strategy](#/coding/strategy)와 [Patterns](#/coding/patterns)를 보세요.
</div></details>

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **NeetCode** (Blind 75 → 150 → 250) | Navdeep Singh | [neetcode.io](https://neetcode.io) · [roadmap](https://neetcode.io/roadmap) | 패턴별 문제 roadmap과 해설; 무료·유료 범위는 현재 사이트에서 확인 | **의존성 그래프로서의 roadmap** — topic 선행 관계와 pattern별 practice list를 학습 계획에 재사용합니다. |
| **Tech Interview Handbook + Grind75** | Yangshun Tay | [techinterviewhandbook.org](https://www.techinterviewhandbook.org) · [grind75](https://www.techinterviewhandbook.org/grind75) · [repo](https://github.com/yangshun/tech-interview-handbook) | 간결한 coding·behavioral·resume·negotiation 안내 | **설정 가능한 학습 플래너** — 주당 시간과 기간에 맞춰 문제 순서와 난이도를 조정합니다. |
| **Blind 75** | 익명 Blind post에서 시작된 community list | via [NeetCode](https://neetcode.io) | 자주 재사용되는 75문제 shortlist | 목록 분포를 실제 지원 회사의 출제 확률로 해석하지 말고, 빠른 pattern-coverage baseline으로 사용합니다. |
| **sean-prashad/leetcode-patterns** | Sean Prashad | [site](https://seanprashad.com/leetcode-patterns/) · [repo](https://github.com/seanprashad/leetcode-patterns) | pattern별 문제와 난이도 filter | **"cue → pattern" 결정 치트시트**("input sorted → binary search / two pointers"; "top-K → heap"; "all permutations → backtracking"). [Patterns](#/coding/patterns)에서 대응물을 만듭니다. |
| **Grokking the Coding Interview (14 patterns)** | Design Gurus / Fahim ul Haq | [free article](https://hackernoon.com/14-patterns-to-ace-any-coding-interview-question-c5bb3357f6ed) | 바이럴한 "14 patterns" 사고 모델(코스는 유료, 프레임워크는 무료로 유명) | **"N patterns to ace any X"** 패키징 — 이 분야에서 가장 재현성 높은 프레이밍. |
| **jwasham/coding-interview-university** | John Washam | [repo](https://github.com/jwasham/coding-interview-university) | 체크박스가 있는 방대한 수개월짜리 CS 커리큘럼 | **추적 가능한 커리큘럼 포맷** — "전체 syllabus를 순서대로, 진행 체크박스와 함께." |

## ML / DL question banks

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **Introduction to ML Interviews (MLIB)** | Chip Huyen | [free book](https://huyenchip.com/ml-interviews-book/) · [repo](https://github.com/chiphuyen/ml-interviews-book) | 골드 스탠더드. 200개 이상의 지식 질문 + 30개의 개방형 판단 질문 | **커리어 절반 + 질문 절반 분할**, 그리고 **Research Scientist vs Research Engineer** 기대치를 분리한 점 — 이 독자에게 정확히 들어맞습니다. |
| **alirezadir/Machine-Learning-Interviews** | alirezadir | [repo](https://github.com/alirezadir/machine-learning-interviews) | 폭넓은 MLE interview repository | **4-pillar 루프** 분해: coding / ML depth / ML system design / behavioral. |
| **khangich/machine-learning-interview** | Khang Pham | [repo](https://github.com/khangich/machine-learning-interview) · [mlengineer.io](https://mlengineer.io) | 저자의 interview 경험을 바탕으로 정리한 community guide | 회사 귀속 질문은 시점·팀·기억 편향이 있으므로 낮은 신뢰도의 practice prompt로 쓰고, `design.md`, `appliedml.md`, `how.md` 구조를 참고합니다. |
| **interviews.ai — Deep Learning Interviews** | Shlomo Kashani (BoltzmannEntropy) | [repo](https://github.com/BoltzmannEntropy/interviews.ai) · [arXiv 2201.00650](https://arxiv.org/abs/2201.00650) | 공개 PDF와 deep-learning interview question/solution 모음 | 짧은 정의보다 derivation과 follow-up을 연습할 때 사용하되, 난도 label과 해답은 독립적으로 검산합니다. |
| **andrewekhalel/MLQuestions** | Andrew Ekhalel | [repo](https://github.com/andrewekhalel/MLQuestions) | 명시적으로 **Computer Vision** 성향이 있는 간결한 Q&A | CV 성향을 가진 몇 안 되는 뱅크 중 하나 — CV depth(detection/segmentation 세부)를 위한 좋은 시드 목록. |
| **zafstojano/ml-interview Q&A** | community | [repo](https://github.com/zafstojano/ml-interview-questions-and-answers) | Chip Huyen MLIB에 대한 커뮤니티 풀이 | MLIB에 대한 정답지 / 검산용으로 사용하세요. |

## ML system design

프레임워크로 가장 학습하기 좋은 라운드. 이 책의 모든 case study는 아래 **9-step 등뼈**를 따릅니다.

<dl class="kv">
<dt>alirezadir — 9-step MLSD framework</dt><dd>가장 인용하기 좋은 오픈소스 프레임워크. <a href="https://github.com/alirezadir/machine-learning-interviews/blob/main/src/MLSD/ml-system-design.md">ml-system-design.md</a>: (1) Problem formulation → (2) Metrics (offline + online) → (3) Architectural components / MVP → (4) Data collection & prep → (5) Feature engineering → (6) Model dev & offline eval → (7) Prediction service (batch/online/hybrid) → (8) Online testing & deploy (A/B, rollout) → (9) Scaling, monitoring & continual training. <b>이걸 외우세요.</b></dd>
<dt>Chip Huyen — Designing ML Systems (DMLS)</dt><dd>책은 유료지만, <a href="https://huyenchip.com/machine-learning-systems-design/">무료 primer</a>(TOC + case studies + ~27개 연습문제)와 <a href="https://github.com/chiphuyen/dmls-book">repo 요약</a>은 금맥입니다. <b>Steal:</b> 요구사항 rubric — <i>reliability, scalability, maintainability, adaptability</i>.</dd>
<dt>Eugene Yan — applied-ml & applyingml</dt><dd><a href="https://github.com/eugeneyan/applied-ml">applied-ml</a> = 문제 유형별(RecSys, search, CV, fraud…)로 큐레이션된 500개 이상의 산업 논문/블로그. <a href="https://applyingml.com">applyingml.com</a> = ML 프로세스 + 멘토 인터뷰. <b>Steal:</b> take-home과 design 답변을 위한 <b>Why / What / How</b> design-doc 골격.</dd>
</dl>

> [!NOTE] 지형 파악용(일부 유료 — 인용하되 복사하지 말 것)
> ByteByteGo ML System Design과 Hello Interview "ML System Design in a Hurry"는 널리 참조되지만 일부 유료입니다. 존재는 알아두되, 위의 무료 프레임워크 위에 쌓으세요.

## LLM / GenAI

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **mlabonne/llm-course** | Maxime Labonne | [repo](https://github.com/mlabonne/llm-course) | 공개 LLM curriculum, roadmap와 notebook | **LLM Scientist vs LLM Engineer track 분할** — 학습 목표를 research와 deployment로 나누는 방식. |
| **awesome-generative-ai-guide** | Aishwarya Naresh Reganti | [repo](https://github.com/aishwaryanr/awesome-generative-ai-guide) | 역할 기반 준비·연구 업데이트·question collection을 모은 community hub | 역할 기반 폴더와 Multimodal/Embeddings 섹션을 routing 참고로 쓰고, 최신성은 각 원문에서 확인합니다. |
| **llmgenai/LLMInterviewQuestions** | llmgenai | [repo](https://github.com/llmgenai/LLMInterviewQuestions) | 100개 이상의 질문, 16개 카테고리 **applied/production** taxonomy(RAG, vector DB, agents) | 경쟁 리소스가 빠뜨리는 **vector-DB + advanced-search + agent** 카테고리 — "visual agents"와 관련. |
| **aman.ai primers** | Aman Chadha | [primers](https://aman.ai/primers/ai/) · [interview Qs](https://aman.ai/primers/ai/interview/) | 매우 깊고 자주 업데이트되는 DL/NLP primer + 100개 이상의 인터뷰 질문 | 모델에 대한 **"pros and cons" 비교 포맷**, 그리고 depth 레퍼런스로서의 transformer/attention 심층 분석. |
| **microsoft/generative-ai-for-beginners** | Microsoft | [repo](https://github.com/microsoft/generative-ai-for-beginners) | 공개 입문 course | 인터뷰 Q-bank보다 교육 과정에 가까우므로 기초 복습용으로 사용. |
| **KalyanKS-NLP LLM Engineer Toolkit** | Kalyan KS | [repo](https://github.com/KalyanKS-NLP/llm-engineer-toolkit) | LLM engineering library·tool을 범주별로 모은 curated toolkit | Q&A bank가 아니라 stack discovery용입니다. 선택한 library의 현재 공식 문서·license·maintenance 상태를 다시 확인하세요. |

## Cheat sheets

Amidi Stanford VIP 시트가 본받아야 할 포맷입니다: **한 페이지, 수식 밀도 높음, 시각적으로 깔끔.** 우리는 바로 이 스타일로 자체 CV/VLM 시트를 만듭니다.

- **CS229 (ML)** — [stanford.edu/~shervine/teaching/cs-229](https://stanford.edu/~shervine/teaching/cs-229/) · [repo](https://github.com/afshinea/stanford-cs-229-machine-learning). 시트: supervised, unsupervised, deep learning, tips & tricks(metrics, bias/variance), + prob/stats 및 linear-algebra 복습. 10개 이상의 언어로 번역됨.
- **CS230 (Deep Learning)** — [stanford.edu/~shervine/teaching/cs-230](https://stanford.edu/~shervine/teaching/cs-230/) · [repo](https://github.com/afshinea/stanford-cs-230-deep-learning). 시트: CNNs(filter arithmetic, architectures), RNNs(LSTM/GRU, attention, embeddings), DL tips & tricks.
- **CS231n (CV) course notes** — [cs231n.github.io](https://cs231n.github.io/). 널리 쓰이는 CV 기초 자료(CNNs, backprop, detection).
- **CS224n (NLP)** — [web.stanford.edu/class/cs224n](https://web.stanford.edu/class/cs224n/). NLP/transformer 대응편.
- **d2l.ai — Dive into Deep Learning** — [d2l.ai](https://d2l.ai/). 실행 가능한 코드 교과서. 유도 공백을 메우기에 훌륭.

## Visual, animated & interactive explainers

*움직이는 걸 직접 볼 때* 더 빨리 이해되는 개념들입니다. 아래 URL은 2026-07-21에 목록 수준으로 검토했지만, 외부 사이트 가용성은 보장하지 않습니다.

**Convolution / CNN**
- [conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) — Dumoulin & Visin. Conv / transposed / dilated convolution을 비교하는 대표적인 애니메이션 GIF.
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — Georgia Tech (Polo Chau). 실제 activation과 함께 라이브 CNN을 레이어별로 클릭해가며 봅니다.
- [Animated AI](https://animatedai.github.io/) — stride, padding, grouped/depthwise-separable convolution의 깔끔한 애니메이션.

**Attention & Transformer**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar. 가장 많이 인용되는 self-attention 그림 설명.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — 브라우저에서 라이브 GPT-2를 실행. 텍스트를 입력하고 attention map을 관찰하세요.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Lilian Weng. attention 변형들에 대한 결정판 서베이.
- [3Blue1Brown — Attention, step by step](https://www.3blue1brown.com/lessons/attention) — Q/K/V에 대한 Manim 애니메이션 기하학적 직관.

**ViT (Vision Transformer)**
- [A Visual Guide to Vision Transformers](https://blog.mdturp.ch/posts/2024-04-05-visual_guide_to_vision_transformer.html) — MDTURP. patch에서 training까지 14단계 애니메이션 스크롤 스토리.

**Backpropagation**
- [3Blue1Brown — Backpropagation](https://www.3blue1brown.com/lessons/backpropagation) & [Backprop calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 애니메이션으로 보는 gradient 흐름 + chain-rule 메커니즘.

**Gradient descent & optimizers**
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) — Distill (Gabriel Goh). 인터랙티브. 슬라이더를 드래그해 momentum을 체감하세요.
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — Sebastian Ruder. 표준 SGD/Momentum/RMSprop/Adam 레퍼런스.
- [TensorFlow Playground](https://playground.tensorflow.org/) — LR / activation / data를 바꿔가며 net이 라이브로 학습하는 걸 관찰.

**RNN / LSTM / GRU**
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Chris Olah. gate별로 그린 고전적 그림 설명.

**Positional encoding**
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) — Amirhossein Kazemnejad. sinusoid에 대한 시각적 + 수학적 직관.

**Mamba / State-Space Models**
- [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) — Maarten Grootendorst. SSM 기초부터 selective scan까지 50개 이상의 커스텀 비주얼.
- [The Annotated S4](https://srush.github.io/annotated-s4/) — Sasha Rush. 실행 가능한 JAX와 함께하는 주석 논문 walkthrough.
- [Mamba & S4 Explained](https://www.youtube.com/watch?v=8Q_tqwpTpVU) — Umar Jamil. 제1원리 기반 애니메이션 유도.

**General interactive playgrounds**
- [Distill.pub](https://distill.pub/) — 인터랙티브하고 애니메이션이 풍부한 ML 설명글의 본거지.
- [Explained Visually (setosa.io)](https://setosa.io/ev/) — PCA, eigenvector, image kernel, Markov chain 등.

## Behavioral & research-scientist-specific

> [!WARNING] 서비스가 부족한 카테고리
> 공개 자료는 coding·ML question bank보다 **job talk, research deep-dive, paper-impact storytelling**을 덜 구조화해 다루는 경우가 많습니다. 아래 자료를 출발점으로 쓰고 실제 role의 rubric과 mock feedback으로 보완하세요.

- **STAR / STAR-L method** — [MIT CAPD worksheet](https://capd.mit.edu/resources/the-star-method-for-behavioral-interviews/)는 STAR의 기본 골격을 제공합니다. 이 책은 여기에 **Learning**을 덧붙이고, 2–3분 답변의 시작점으로 Action 비중을 크게 두는 휴리스틱과 I-vs-we 구분을 사용합니다. 이는 MIT의 고정 규칙이 아닙니다.
- **PhD → industry 전환** — Rutgers "From Bench to Offer", NIH OITE 블로그, Cheeky Scientist. 핵심 재구성: *학계는 뉘앙스와 단서(caveat)를 보상하지만, 인터뷰는 명확한 결정과 측정 가능한 결과를 보상한다.* 연구를 **결정 + 측정 가능한 임팩트** 언어로 번역하세요.
- **research job talk** — recruiter와 포맷을 확인하고, 문제의 중요성과 산출물로 시작하며, 과학 커뮤니케이션 + depth를 보여주고, 깊은 Q&A를 예상하고, 프레이밍을 팀의 제품에 맞추세요.

## Mock interviews

- **[Pramp](https://www.pramp.com/)** — peer-to-peer mock interview service. 무료·유료 범위와 현재 제공 category는 예약 전에 확인하세요.
- **[interviewing.io](https://interviewing.io/)** — 시니어 엔지니어와의 익명 모의. 녹화되며, 큰 무료 technical-talk / 질문 라이브러리 제공.
- **신뢰할 수 있는 동료·동문** — 대상 role을 이해하는 사람에게 research-depth mock을 부탁하세요. 질문 유출이나 회사 기밀을 요구하지 말고, 평가 rubric과 전달 방식에 대한 피드백만 받습니다.

## Track the frontier (company blogs & leaderboards)

인터뷰어는 *현재*의 프론티어에 대비해 당신을 평가합니다. 어떤 루프든 시작 전에 이것들을 훑어보세요 — 이유는 [The 2026 Landscape](#/start/landscape-2026)를 보세요.

<div class="proscons"><div><div class="pros-t">회사 연구 블로그</div>

- [Meta AI Research](https://ai.meta.com/research/)
- [NVIDIA Research](https://research.nvidia.com/)
- [Apple ML Research](https://machinelearning.apple.com/)
- [Adobe Research](https://research.adobe.com/)
- [Microsoft Research](https://www.microsoft.com/en-us/research/)
- [ByteDance Seed](https://seed.bytedance.com/)
- [Mistral AI News](https://mistral.ai/news)

</div><div><div class="cons-t">트렌드 & 리더보드 트래커</div>

- [Papers with Code](https://paperswithcode.com/) — SOTA + benchmark 추적
- [LMArena](https://lmarena.ai/) — human-preference LLM 리더보드(비판적으로 읽을 것)
- [Hugging Face](https://huggingface.co/models) — models, datasets, Open LLM leaderboard
- [arXiv cs.CV / cs.CL / cs.LG](https://arxiv.org/list/cs.CV/recent) — primary firehose
- [Distill.pub](https://distill.pub/) — 보관됨(archived)이지만, 직관을 길러주는 최고의 설명글

</div></div>

## Interview-experience sources (real questions)

회사·role별 경험담을 찾는 보조 자료입니다. 익명 게시물은 sampling bias·시점·지역·팀 차이가 크므로 recruiter의 현재 안내와 공식 prep page보다 낮은 신뢰도로 다루세요.

- [Glassdoor Interview Questions](https://www.glassdoor.com/Interview/index.htm) — "Meta Research Scientist interview" 등으로 검색.
- [Blind (teamblind.com)](https://www.teamblind.com/) — 익명 빅테크 직원 리포트.
- [1point3acres](https://www.1point3acres.com/bbs/) — MSRA 및 중국 오피스 루프에 강함.
- [LeetCode Discuss — Interview Experience](https://leetcode.com/discuss/interview-experience) — 회사 태그가 붙은 실제 코딩 테스트.
- [levels.fyi](https://www.levels.fyi/) — self-reported level/comp의 참고 분포. 표본·통화·주가·지역·시점을 확인하고 오퍼/세무 판단의 단일 근거로 쓰지 마세요.

## 개인 research asset (인터뷰 패킷)

이 절만 특정 후보자용입니다. 공개 논문·code·발표를 [research deep-dive](#/research/job-talk)와 STAR-L story bank의 evidence로 사용하되, citation 수와 profile 정보는 지원 직전에 갱신합니다.

<dl class="kv">
<dt>Google Scholar</dt><dd><a href="https://scholar.google.co.kr/citations?user=n_TR1LcAAAAJ">All publications</a> — 임팩트 스토리를 위한 citation 맥락.</dd>
<dt>GitHub</dt><dd><a href="https://github.com/qjadud1994">qjadud1994</a> — 공개 코드. 엔지니어링 엄밀성의 증거.</dd>
<dt>ZIM</dt><dd>Zero-shot Image Matting — <a href="https://naver-ai.github.io/ZIM/">project</a> · <a href="https://arxiv.org/pdf/2411.00626">paper</a>. SAM / alpha-matte / promptable segmentation 이야깃거리와 직결.</dd>
<dt>ECLIPSE</dt><dd>Panoptic segmentation — <a href="https://arxiv.org/abs/2403.20126">arXiv 2403.20126</a>. Continual / catastrophic-forgetting 내러티브.</dd>
<dt>PointWSSIS</dt><dd>Weakly-supervised instance segmentation — <a href="https://arxiv.org/abs/2303.15062">arXiv 2303.15062</a>. Label-efficiency 스토리.</dd>
<dt>BESTIE</dt><dd>Weakly-supervised instance segmentation — <a href="https://arxiv.org/abs/2109.09477">arXiv 2109.09477</a>. Image-level supervision과 pseudo-instance refinement 사례.</dd>
</dl>

## Cheat-sheet

| Need | Go-to (free) |
| --- | --- |
| Coding patterns | NeetCode roadmap + sean-prashad "cue → pattern" 시트 |
| Study plan / scheduling | Grind75 설정 가능한 플래너 |
| ML depth Q-bank | Chip Huyen MLIB + interviews.ai (PhD 수준 풀이) |
| ML system design | alirezadir 9-step 프레임워크 + Eugene Yan Why/What/How |
| LLM/GenAI | mlabonne llm-course + awesome-generative-ai-guide |
| CV slant | andrewekhalel/MLQuestions + CS231n notes |
| Last-mile recall | Amidi CS229/CS230 cheatsheets |
| Behavioral | MIT CAPD STAR-L + story-bank 매트릭스 |
| Mocks | Pramp / interviewing.io / 당신의 네트워크 |
| Real questions | Glassdoor / Blind / 1point3acres / LeetCode Discuss |

**Related:** [How to Use This Book](#/start/how-to-use) · [The 2026 Landscape](#/start/landscape-2026) · [Prep Plan](#/start/prep-plan) · [Coding Strategy](#/coding/strategy) · [Interview Pipeline](#/process/pipeline)
