# Curated Open-Source Resources

<div class="tag-row"><span class="tag">free</span><span class="tag">open-source</span><span class="tag">coding</span><span class="tag">ML depth</span><span class="tag">system design</span><span class="tag">LLM/GenAI</span><span class="tag">behavioral</span><span class="tag">mock</span></div>

> [!TIP] 이 페이지 사용법
> 이건 *주석 달린* 디렉터리이지 링크 덤프가 아닙니다. 각 리소스마다 **무엇에 가장 좋은지(what it's best for)**와 **훔쳐올 단 하나의 최고 아이디어(the single best idea to steal)** — 콘텐츠 자체가 아니라 재사용 가능한 프레임워크나 패키징 기법 — 를 얻습니다. 모든 top 리소스를 관통하는 메타 교훈: **무작위 질문 덤프가 아니라 재사용 가능한 패턴으로 조직하라.** NeetCode, sean-prashad, Chip Huyen, alirezadir가 모두 이기는 이유는, 처음 보는 문제에도 일반화되는 작고 기억하기 쉬운 taxonomy를 손에 쥐여주기 때문입니다. 이 책 전체가 그 원칙 위에 지어졌습니다.

> [!NOTE] star 수와 링크에 관하여
> 아래 star 수는 2026년 중반 기준 근사치입니다 — 출처에서 직접 확인하세요. 모든 외부 링크는 새 탭에서 열립니다. 미러보다 primary source를 우선하세요.

## If you only use 5 resources

PhD 수준 Applied/Research-Scientist 루프(CV / VLM / visual agents)를 위한, 최소한의 고레버리지 세트입니다.

| # | Resource | Why it's on the shortlist |
| --- | --- | --- |
| 1 | [Chip Huyen — Introduction to ML Interviews](https://huyenchip.com/ml-interviews-book/) | 최고의 전체 구조. 경쟁 리소스가 빠뜨리는 *커리어 절반*(RS vs RE, 프로세스, offer)에 200개 이상의 질문. |
| 2 | [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/machine-learning-interviews) | **9단계 ML system-design 프레임워크**와 인터뷰 전체를 떠받치는 4-pillar 루프. |
| 3 | [NeetCode roadmap + 150](https://neetcode.io/roadmap) | 18개 토픽 DSA ontology와 의존성 그래프 학습 경로 — 사실상의 코딩 표준. |
| 4 | [Amidi Stanford CS229 / CS230 cheatsheets](https://stanford.edu/~shervine/teaching/cs-229/) | 한 페이지, 수식이 빽빽한 recall 시트 — 마지막 마무리 복습의 골드 스탠더드. |
| 5 | [aishwaryanr/awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) | 역할 기반 준비 + VLM 독자에 1:1로 대응되는 multimodal/embeddings 질문 taxonomy. |

아래의 모든 것은 이 5개를 확장하고 공백(behavioral, mock, 회사 추적)을 메웁니다.

## Coding interview prep

<details class="qa"><summary>PhD 후보자가 정말 코딩 훈련(grind)을 해야 하나요?</summary>
<div class="qa-body">

**짧게:** 네, 하지만 범위를 좁혀서. Research-scientist 루프는 보통 1~2개의 코딩 라운드를 포함합니다. 500문제가 필요한 게 아니라, 시간 압박 아래 처음 보는 문제를 풀 만큼 탄탄한 ~15~20개의 *패턴*이 필요합니다. 문제 수가 아니라 패턴 커버리지를 최적화하세요.

**깊게:** NeetCode/Grokking/sean-prashad의 전체 가치 제안은, 인터뷰 문제의 공간이 반복되는 소수의 형태로 붕괴한다는 것입니다. 그 형태를 내재화하고, 각각 5~10문제씩 풀면 일반화됩니다. [Coding Strategy](#/coding/strategy)와 [Patterns](#/coding/patterns)를 보세요.
</div></details>

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **NeetCode** (Blind 75 → 150 → 250) | Navdeep Singh (ex-Google) | [neetcode.io](https://neetcode.io) · [roadmap](https://neetcode.io/roadmap) | 사실상의 코딩 표준. 모든 문제에 무료 비디오 풀이 | **의존성 그래프로서의 roadmap** — 토픽이 선행 조건을 풀어줍니다(unlock). 18개 카테고리 taxonomy는 구할 수 있는 가장 깔끔한 DSA ontology. |
| **Tech Interview Handbook + Grind75** | Yangshun Tay (ex-Meta) | [techinterviewhandbook.org](https://www.techinterviewhandbook.org) · [grind75](https://www.techinterviewhandbook.org/grind75) · [repo (~140k★)](https://github.com/yangshun/tech-interview-handbook) | "Cracking the Coding Interview를 간결하게" + behavioral, resume, negotiation | **설정 가능한 학습 플래너** — 주당 시간 + 주 수를 입력하면 순서가 잡히고 난이도가 점증하는 계획이 나옵니다. 목록이 아니라 상호작용성이 이걸 유명하게 만들었습니다. |
| **Blind 75** | Yangshun (orig. teamblind) | via [NeetCode](https://neetcode.io) | 10개 카테고리에 걸친 정전(canonical)의 최소 75문제 목록 | 분포 신호: **Trees + DP가 목록의 ~1/3** → 고효율 토픽에 준비 비중을 실으세요. |
| **sean-prashad/leetcode-patterns** | Sean Prashad | [site](https://seanprashad.com/leetcode-patterns/) · [repo (~10k★)](https://github.com/seanprashad/leetcode-patterns) | 패턴별로 묶인 ~179문제, 회사 + 난이도 필터 | **"cue → pattern" 결정 치트시트**("input sorted → binary search / two pointers"; "top-K → heap"; "all permutations → backtracking"). [Patterns](#/coding/patterns)에서 CV/ML 대응물을 만듭니다. |
| **Grokking the Coding Interview (14 patterns)** | Design Gurus / Fahim ul Haq | [free article](https://hackernoon.com/14-patterns-to-ace-any-coding-interview-question-c5bb3357f6ed) | 바이럴한 "14 patterns" 사고 모델(코스는 유료, 프레임워크는 무료로 유명) | **"N patterns to ace any X"** 패키징 — 이 분야에서 가장 재현성 높은 프레이밍. |
| **jwasham/coding-interview-university** | John Washam | [repo (~337k★)](https://github.com/jwasham/coding-interview-university) | 체크박스가 있는 방대한 수개월짜리 CS 커리큘럼 | **추적 가능한 커리큘럼 포맷** — "전체 syllabus를 순서대로, 진행 체크박스와 함께." |

## ML / DL question banks

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **Introduction to ML Interviews (MLIB)** | Chip Huyen | [free book](https://huyenchip.com/ml-interviews-book/) · [repo](https://github.com/chiphuyen/ml-interviews-book) | 골드 스탠더드. 200개 이상의 지식 질문 + 30개의 개방형 판단 질문 | **커리어 절반 + 질문 절반 분할**, 그리고 **Research Scientist vs Research Engineer** 기대치를 분리한 점 — 이 독자에게 정확히 들어맞습니다. |
| **alirezadir/Machine-Learning-Interviews** | alirezadir (ex-Amazon/Google) | [repo (~10k★)](https://github.com/alirezadir/machine-learning-interviews) | 최고의 단일 올인원 MLE repo | **4-pillar 루프** 분해: coding / ML depth / ML system design / behavioral — 이 책의 등뼈입니다. |
| **khangich/machine-learning-interview** | Khang Pham | [repo](https://github.com/khangich/machine-learning-interview) · [mlengineer.io](https://mlengineer.io) | 실제 offer(Snapchat, Coupang, Stitchfix)에서 만든 진짜 FAANG 질문 | **특정 회사에 귀속된 질문** — 신뢰도가 높고 타깃 회사 목록과 맞습니다. `design.md`, `appliedml.md`, `how.md`를 캐내세요. |
| **interviews.ai — Deep Learning Interviews** | Shlomo Kashani (BoltzmannEntropy) | [repo](https://github.com/BoltzmannEntropy/interviews.ai) · [arXiv 2201.00650](https://arxiv.org/abs/2201.00650) | 무료 PDF. 수백 개의 **완전히 풀린, 시험 스타일, PhD 수준** 문제 | 엄밀한 **한 줄 요약이 아닌 유도(derivation)** 레지스터 — research-scientist 청중에 맞는 깊이. |
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
| **mlabonne/llm-course** | Maxime Labonne | [repo (~80k★)](https://github.com/mlabonne/llm-course) | 최고의 무료 LLM 커리큘럼. roadmap + Colab 노트북 | **LLM Scientist vs LLM Engineer 트랙 분할** — 응용 배포까지 다뤄야 하는 연구서에 완벽. |
| **awesome-generative-ai-guide** | Aishwarya Naresh Reganti | [repo](https://github.com/aishwaryanr/awesome-generative-ai-guide) | 최고의 무료 GenAI 허브. **역할 기반** 준비 + 연구 업데이트 + "60 GenAI questions" | **역할 기반 폴더**("역할 선택 → overview → rounds → bank → plan")와 VLM에 1:1로 대응되는 **Multimodal + Embeddings** 섹션. |
| **llmgenai/LLMInterviewQuestions** | llmgenai | [repo](https://github.com/llmgenai/LLMInterviewQuestions) | 100개 이상의 질문, 16개 카테고리 **applied/production** taxonomy(RAG, vector DB, agents) | 경쟁 리소스가 빠뜨리는 **vector-DB + advanced-search + agent** 카테고리 — "visual agents"와 관련. |
| **aman.ai primers** | Aman Chadha | [primers](https://aman.ai/primers/ai/) · [interview Qs](https://aman.ai/primers/ai/interview/) | 매우 깊고 자주 업데이트되는 DL/NLP primer + 100개 이상의 인터뷰 질문 | 모델에 대한 **"pros and cons" 비교 포맷**, 그리고 depth 레퍼런스로서의 transformer/attention 심층 분석. |
| **microsoft/generative-ai-for-beginners** | Microsoft | [repo (~90k★)](https://github.com/microsoft/generative-ai-for-beginners) | 믿을 만한 무료 기초(21-lesson 코스) | 인터뷰 준비보다는 교육에 가까움 — 초심자를 여기로 안내하되, 복사할 소스는 아님. |
| **KalyanKS-NLP LLM-Interview Hub** | Kalyan KS | [repo](https://github.com/KalyanKS-NLP/llm-engineer-toolkit) | 보조용 100개 이상의 LLM Q&A | LLM 폭을 위한 빠른 보완 뱅크. |

## Cheat sheets

Amidi Stanford VIP 시트가 본받아야 할 포맷입니다: **한 페이지, 수식 밀도 높음, 시각적으로 깔끔.** 우리는 바로 이 스타일로 자체 CV/VLM 시트를 만듭니다.

- **CS229 (ML)** — [stanford.edu/~shervine/teaching/cs-229](https://stanford.edu/~shervine/teaching/cs-229/) · [repo](https://github.com/afshinea/stanford-cs-229-machine-learning). 시트: supervised, unsupervised, deep learning, tips & tricks(metrics, bias/variance), + prob/stats 및 linear-algebra 복습. 10개 이상의 언어로 번역됨.
- **CS230 (Deep Learning)** — [stanford.edu/~shervine/teaching/cs-230](https://stanford.edu/~shervine/teaching/cs-230/) · [repo](https://github.com/afshinea/stanford-cs-230-deep-learning). 시트: CNNs(filter arithmetic, architectures), RNNs(LSTM/GRU, attention, embeddings), DL tips & tricks.
- **CS231n (CV) course notes** — [cs231n.github.io](https://cs231n.github.io/). 정전의 CV depth 기초(CNNs, backprop, detection).
- **CS224n (NLP)** — [web.stanford.edu/class/cs224n](http://web.stanford.edu/class/cs224n/). NLP/transformer 대응편.
- **d2l.ai — Dive into Deep Learning** — [d2l.ai](https://d2l.ai/). 실행 가능한 코드 교과서. 유도 공백을 메우기에 훌륭.

## Visual, animated & interactive explainers

*움직이는 걸 직접 볼 때* 더 빨리 이해되는 개념들. 모든 링크는 라이브 검증됨(2026-07).

**Convolution / CNN**
- [conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) — Dumoulin & Visin. conv / transposed / dilated convolution의 정전이 된 애니메이션 GIF(파랑=input, 청록=output).
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
> Research-scientist 인터뷰 — **job talk, research deep-dive, paper-impact 스토리텔링** — 는 오픈소스 리소스에서 가장 취약한 영역입니다(대부분 일반적인 커리어 서비스 페이지). 이 책의 [Behavioral & Research](#/process/pipeline) 커버리지는 의도적인 차별화입니다. 아래 링크는 발판으로 삼고, 그 너머로 나아가세요.

- **STAR / STAR-L method** — [MIT CAPD worksheet](https://capd.mit.edu/resources/the-star-method-for-behavioral-interviews/). Situation → Task → Action → Result (+ **Learning**, 과학자에게 적합). 채택할 규칙: 시간 배분(Action ~50~60%), **"we"가 아닌 "I"** 사용, 그리고 자신의 연구에서 미리 채운 **story-bank 매트릭스**(행 = 역량, 열 = STAR-L 슬롯).
- **PhD → industry 전환** — Rutgers "From Bench to Offer", NIH OITE 블로그, Cheeky Scientist. 핵심 재구성: *학계는 뉘앙스와 단서(caveat)를 보상하지만, 인터뷰는 명확한 결정과 측정 가능한 결과를 보상한다.* 연구를 **결정 + 측정 가능한 임팩트** 언어로 번역하세요.
- **research job talk** — recruiter와 포맷을 확인하고, 문제의 중요성과 산출물로 시작하며, 과학 커뮤니케이션 + depth를 보여주고, 깊은 Q&A를 예상하고, 프레이밍을 팀의 제품에 맞추세요.

## Mock interviews

- **[Pramp](https://www.pramp.com/)** — 무료 peer-to-peer 모의 인터뷰(coding + 일부 behavioral). 반복 연습과 캘리브레이션에 최적.
- **[interviewing.io](https://interviewing.io/)** — 시니어 엔지니어와의 익명 모의. 녹화되며, 큰 무료 technical-talk / 질문 라이브러리 제공.
- **당신의 네트워크** — 가장 신호가 강한 옵션. KAIST MLAI / SIIT 동문과 이제 빅테크에 있는 NAVER Cloud 동료에게 진짜 research-depth 모의를 부탁하세요. 루프를 직접 진행하는 사람만큼 루프를 시뮬레이션하는 건 없습니다.

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

특정 루프가 무엇을 물을지에 대한 가장 신뢰도 높은 신호인, 회사·역할에 귀속된 질문을 찾는 데 사용하세요.

- [Glassdoor Interview Questions](https://www.glassdoor.com/Interview/index.htm) — "Meta Research Scientist interview" 등으로 검색.
- [Blind (teamblind.com)](https://www.teamblind.com/) — 익명 빅테크 직원 리포트.
- [1point3acres](https://www.1point3acres.com/bbs/) — MSRA 및 중국 오피스 루프에 강함.
- [LeetCode Discuss — Interview Experience](https://leetcode.com/discuss/interview-experience) — 회사 태그가 붙은 실제 코딩 테스트.
- [levels.fyi](https://www.levels.fyi/) — offer 협상을 위한 level/comp 맥락.

## Your own research assets (use as evidence)

research 인터뷰에서 가장 강력한 "소스"는 *당신 자신의 작업*입니다. 이것들을 [research deep-dive](#/process/pipeline)에서 인용하고 STAR-L story bank의 씨앗으로 삼을 수 있게 준비해두세요.

<dl class="kv">
<dt>Google Scholar</dt><dd><a href="https://scholar.google.co.kr/citations?user=n_TR1LcAAAAJ">All publications</a> — 임팩트 스토리를 위한 citation 맥락.</dd>
<dt>GitHub</dt><dd><a href="https://github.com/qjadud1994">qjadud1994</a> — 공개 코드. 엔지니어링 엄밀성의 증거.</dd>
<dt>ZIM</dt><dd>Zero-shot Image Matting — <a href="https://naver-ai.github.io/ZIM/">project</a> · <a href="https://arxiv.org/pdf/2411.00626">paper</a>. SAM / alpha-matte / promptable segmentation 이야깃거리와 직결.</dd>
<dt>ECLIPSE</dt><dd>Panoptic segmentation — <a href="https://arxiv.org/abs/2403.20126">arXiv 2403.20126</a>. Continual / catastrophic-forgetting 내러티브.</dd>
<dt>PointWSSIS</dt><dd>Weakly-supervised instance segmentation — <a href="https://arxiv.org/abs/2303.15062">arXiv 2303.15062</a>. Label-efficiency 스토리.</dd>
<dt>BESTIE</dt><dd>Weakly-supervised semantic segmentation — <a href="https://arxiv.org/abs/2109.09477">arXiv 2109.09477</a>. 기반이 되는 segmentation depth.</dd>
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
