# Company Playbooks

<div class="tag-row"><span class="tag">Meta / FAIR</span><span class="tag">NVIDIA</span><span class="tag">Apple</span><span class="tag">Adobe</span><span class="tag">ByteDance Seed</span><span class="tag">Mistral</span><span class="tag">Microsoft / MSR</span></div>

> [!TIP] 이 chapter가 존재하는 이유
> [pipeline](#/process/pipeline)은 공유되지만; *강조점*은 그렇지 않습니다. Meta의 near-SWE coding, NVIDIA의 faculty-style job talk, Mistral의 take-home restitution, Microsoft의 "As Appropriate" 라운드 — 각각이 다른 필터입니다. 이것은 **CV / VLM / agents** 후보자를 위한 회사별 지도입니다: loop 형태, 무엇에 가중하는지, 현재 research 방향, 문화 signal, 그리고 여러분의 이야기를 어떻게 매핑할지.

> [!WARNING] 사실 대 lore — 이것부터 읽으세요
> **[Verifiable]** 태그가 붙은 주장은 회사 페이지, 공식 블로그, 또는 levels.fyi aggregate에서 나옵니다. **[Lore]** 태그가 붙은 주장은 Glassdoor/Blind/1point3acres/포럼 일화에서 나옵니다 — 방향성 있게 유용하지만 **보장되지 않고**, 팀/분기에 따라 다릅니다. **어떤 면접관 이름이나 기밀 내부 rubric도 재현되지 않습니다.** loop 구성은 분기마다 바뀝니다 — recruiter에게 확인하세요.

## 한눈에 비교

| Company | Coding weight | Job-talk centrality | 특징적 signal | RS comp *(levels.fyi aggregate, 2025–26)* |
| --- | --- | --- | --- | --- |
| **Meta / FAIR** | 높음 — near-SWE, 2–3 LeetCode 라운드 | 두드러짐 (FAIR/Applied) | 스케일에서 publish *하고* ship; 모호함 속 ownership | ~$305K–$581K+ (IC4–IC6) |
| **NVIDIA Research** | 중간; 일부 랩은 생략; C++/CUDA 중시 | 중심 (~1시간, faculty-style) | GPU/systems + research-engineering; intellectual honesty | ~$205K–$533K+, median ~$455K |
| **Apple MLR/AIML** | 중간–높음 (1–2+ coding) | MLR에는 중심 | On-device / privacy / HW co-design; 비밀주의 | ~$334K–$476K+ (ICT3–5) |
| **Adobe Research** | 높음 — PyTorch + algo 라운드 | 중심 | Publication + PyTorch 엄밀함 + product sense | median ~$320K, base-heavy |
| **ByteDance Seed** | 매우 높음 — RS에도 **Hard** LeetCode | 기술 라운드에 접힘 | 속도 + coding + generation/VLM depth | median ~$395K (US), private equity |
| **Mistral AI** | ML-primitives from scratch + LeetCode-w/-twist | **Take-home + restitution** | MoE/transformer 내부; open-weights 신념 | Paris ~€200–500K TC; BSPCE equity |
| **Microsoft MSR** | 가벼움 (Medium + ML-impl) | **Central job talk** + researcher 패널 | Research agenda + 협업; AA 라운드 | ~$156K–$625K+ (L59–67) |

*모든 comp는 self-reported levels.fyi aggregate이며 변동합니다; coding-weight/talk-centrality 등급은 [Lore] loop 리포트를 종합한 것으로 팀마다 다릅니다.*

---

## Meta — AI Research / FAIR (VLM 포함)

**Loop** *[Lore, 한 명의 1차 증언]*. Meta는 하나의 획일적 RS loop를 운영하지 않습니다 — "Research Scientist" title은 여러 트랙에 걸쳐 있습니다.
- Recruiter screen (~30–45 min): 배경, publication, 트랙 정렬, level 보정.
- Technical phone screen (~45 min, CoderPad): 거의 항상 coding — 1–2 LeetCode Medium(가끔 Hard).
- Full loop (virtual, **4–6 라운드**): 종종 **2–3 coding 라운드**(각각 Medium/Hard 둘 — Meta의 coding 기준은 research title에도 near-SWE이며, 이것이 *결정적* 특징), **1 ML system design**, **research deep-dive / job talk**(전체 lifecycle을 훑기: 문제 → 실험 설계 → data → architecture → eval → 결과), 그리고 **behavioral**(종종 여러분의 research 궤적을 파고드는 PhD가 진행).

**트랙 분류** *[Lore]*: RS–Systems & Infrastructure(SWE-flavored), RS–Central Applied Science(stats/econ + stats 시험 + EDA coding + research presentation), 그리고 **RS, ML / FAIR**(AI/ML research). **FAIR는 postdoc/faculty 채용처럼 운영되며** 다수의 first-author NeurIPS/ICML/CVPR-tier 논문을 기대합니다. 대표적 LLM 연구(Llama)는 점점 FAIR 본체가 *아니라* **Meta GenAI / Superintelligence Labs**에 있다는 점에 유의 — recruiter에게 FAIR 대 GenAI를 명확히 하세요.

**2025–26 방향** *[Verifiable, Meta AI]*: **Llama 4**(Scout & Maverick — natively multimodal MoE, long context); **Segment Anything** family(SAM 2.1, 그리고 promptable *concept* segmentation을 갖춘 것으로 보고된 **SAM 3**); self-supervised vision(**DINO/DINOv2/DINOv3**, LeCun 아래의 **V-JEPA** world-model agenda); **Perception Encoders**; **Meta Spirit LM**(text+speech).

**문화 / behavioral** *[Lore]*: Meta 가치 — move fast, ownership, bias for action, direct communication. 파고드는 것: 모호함 속 ownership("요구사항이 완전히 백지"), 자기 인식(자기 실패를 소유), 정량화된 impact. STAR story ~3분.

> [!QUOTE] 이야기 매핑
> Beomyoung → FAIR/VLM: **grounded VLM + region-level 근거**(SAM/DINO/perception 라인과 정렬)와 **ZIM → CLOVA-X** research-to-product 증명으로 시작하세요. near-SWE coding 기준을 위해, Medium을 ~20분으로 과잉 준비하고 from-scratch attention/sampling을 리허설하세요.

<div class="proscons"><div><div class="pros-t">강점을 살림</div>Vision-foundation + grounding depth; shipped-at-scale 이야기; 강한 first-author record</div><div><div class="cons-t">주의</div>research title치고 coding 기준이 유난히 높음; "완벽한" 솔루션도 협업 signal 없이는 여전히 no-hire일 수 있음</div></div>

**Leveling / comp** *[levels.fyi]*: E-ladder E4…E8; IC5/E5가 흔한 senior 타겟. RS ~$305K (IC4) → $581K (IC6). Meta는 **total comp**를 벤치마크 — equity + refresher를 최적화하고, 경쟁 offer 수치를 인용하세요.

---

## NVIDIA Research

**Loop** *[Lore]*. Academic/faculty-style; **랩마다 크게 다름**(일부는 전용 coding 라운드를 생략).
- Recruiter phone screen → **technical screen**(엔지니어링-인접 role에는 종종 75분 HackerRank; 순수 research에는 관심/프로젝트에 대한 대화와 약간의 coding/ML) → **job talk**(~1시간, 자기 research 발표) → **final loop**(~6회 one-on-one): coding/systems(Python/C++/**CUDA**/PyTorch, distributed training, GPU trade-off — 일부 랩은 생략), DL-depth 라운드(transformer 내부, parallel training, optimization), 다수의 research deep-dive, behavioral(보통 HM). Timeline ~4–6주.

**2025–26 방향** *[Verifiable, NVIDIA]*: generative AI & LLM(video/vision generation, LLM/agent용 RL, 효율적 inference 예: Blackwell 위의 **Dynamo**); **Physical AI & robotics** — **Cosmos** world model, **Isaac GR00T** humanoid, **BioNeMo**; autonomous driving; neural graphics. 주목할 랩: **GEAR**(Generalist Embodied Agent Research; Jim Fan & Yuke Zhu) — planning을 위한 LLM, VLM, internet-scale world model. *(특정 "Learning & Perception / Toronto" 팀은 확인되지 않음 — recruiter에게 확인.)*

**문화 / behavioral** *[Verifiable values + Lore]*: 다섯 가지 가치 — **Innovation, Speed & Agility, Intellectual Honesty, Excellence, One Team**; 만트라 "the mission is the boss". **Intellectual honesty**(모르는 것을 인정)는 *명시적으로* 평가되는 signal — 우아한 "모르지만, 이렇게 접근하겠습니다"가 점수가 됩니다.

**알려진 질문 스타일** *[Lore]*: coding — reverse linked list, number-of-islands(grid BFS/DFS), C++ 세부(memory mgmt, virtual function); ML-from-scratch — dropout/BN/softmax/attention; **LayerNorm vs BatchNorm**(수식 + transformer에서 왜 LN인지); quantization/distillation; text↔video embedding을 정렬하는 loss 설계.

> [!QUOTE] 이야기 매핑
> Beomyoung → NVIDIA: **효율을 과학적 목표로** 프레임하세요 — ~10 ms의 on-device seg, ONNX serving, multi-node PyTorch training — 그리고 generative-vision 야망과 짝지으세요. CUDA를 익힐 의향은 정직하게 말해도 괜찮습니다; intellectual-honesty 가치가 그것을 보상합니다.

**Leveling / comp** *[levels.fyi]*: IC1–IC6+; RSU는 **"NSU"** 브랜드(분기 vest). RS ~$205K (IC1) → $533K (IC6), median ~$455K. signed-grant 대 current value를 모델링하세요(주가 상승으로 큰 괴리).

---

## Apple — Machine Learning Research (MLR) & AIML

**Loop** *[Lore; 팀별로 매우 다름 — MLR pure-research 대 AIML applied]*. RS/MLR (~3–5주): recruiter → 1–2 technical phone screen(coding + domain theory) → virtual onsite: **45–60분 research talk** + **5–6 researcher 면접**(coding, research-theory deep-dive, ML system design, behavioral). AIML/MLE (~4–6주): recruiter → 1–2 phone screen → 때로 take-home → **~5–8 라운드 onsite** → HM 대화.

**Signal** *[Lore + Verifiable framing]*: 실제 제약 **아래에서의** applied ML 추론 — on-device("이게 폰에서 돌아갈까?"), privacy-first(federated learning, differential privacy, **Private Cloud Compute**), accuracy 대 latency/battery/memory, Apple Silicon을 위한 software–hardware co-design. MLR = 호기심 주도, publish, talk + novelty로 평가; AIML = product-embedded, coding + system design 더 많음. Apple은 research↔product를 의도적으로 흐릿하게 합니다("research must be production-ready").

**2025–26 방향** *[Verifiable]*: **Apple Foundation Models / Apple Intelligence** — ~3B on-device 모델 + Private Cloud Compute 위의 더 큰 server 모델; **Foundation Models framework**가 on-device LLM을 개발자에게 노출; **Manzano** — hybrid vision tokenizer를 갖춘 통합 multimodal 모델. On-device/privacy/efficiency; Apple Silicon을 위한 **MLX**. 팀: **MLR**, **AIML**, **SIML**(Siri / "Hey Siri" trigger detection), Health.

**문화 / behavioral** *[Mixed]*: **비밀주의** — MLR은 제품을 위협하지 않을 때만 publish; publication 수보다 product impact; discretion과 외부 가시성 없이 일하는 것에 대한 파고들기 예상. 하드웨어/제품 팀과 cross-functional.

**알려진 질문 스타일** *[Lore]*: Non-Maximum Suppression 구현, IoU 계산, K-Means from scratch, LeetCode Medium; Batch 대 Layer Norm과 언제; forward/reverse diffusion; binary-classification loss 유도; "Hey Siri" trigger 설계와 false positive 처리; 저조도에서 Face ID 개선.

> [!QUOTE] 이야기 매핑
> Beomyoung → Apple: **on-device human-seg 모델(~10 ms, ONNX)**은 효율/privacy 내러티브에 거의 완벽한 fit입니다; foundation-model depth를 위해 ZIM과 짝지으세요. 미공개 제품에 대해 **추측하지** 마세요 — 공개 JD 언어에 머무르세요.

**Leveling / comp** *[levels.fyi]*: **ICT2–ICT6**; ML은 ICT4/ICT5를 타겟. RS ~$334K (ICT3) → $476K+ (ICT5). Base 밴드가 비교적 경직됨 → **equity + level**이 lever; leveling은 committee/team-driven이고 밀어붙일 가치가 있습니다.

---

## Adobe Research (Firefly Applied Science 포함)

**Loop** *[Lore]*. 4–6주에 걸쳐 ~5 라운드: recruiter → **research presentation**(peer-reviewed publication을 패널에 발표, 엄밀하게 방어) → technical problem-solving(coding + PyTorch + ML theory) → behavioral → **final onsite**(deep coding, ML theory, ML system design, behavioral). 반복되는 공식: *"뛰어난 publication record가 문을 열어주지만, offer를 받으려면 엄밀한 PyTorch + algorithmic coding 라운드를 통과해야 한다."*

**2025–26 방향** *[Verifiable]*: **Firefly** generative AI가 flagship. MAX 2025: **Firefly Image Model 5**(native 4MP photorealistic), 새 audio/video 도구, **Firefly Foundry**(custom enterprise model). Adobe Research의 **Video AI Lab** + **Foundation Model Team**이 **Firefly Video Model**과 **GenExtend**(Premiere Pro)를 만들었습니다. 면접 테마: diffusion(DDPM, DDIM, classifier-free guidance), ViT/VLM, attention 내부, video gen의 temporal consistency, 대규모 multimodal data curation/de-dup, distributed training(DDP/FSDP/tensor-parallel), fine-tuning/alignment.

**문화 / behavioral** *[Mixed]*: publication-friendly, academic-style 랩(활발히 publish, 대학 협업, 인턴); Apple보다 비밀주의 덜함. Behavioral은 협업, 커뮤니케이션, 이론을 creative 제품으로 옮기기를 파고듦 — **product sense**가 차별점.

**알려진 질문 스타일** *[Lore]*: "reverse-diffusion 수식을 훑어라 — user editing prompt에 condition하라"; "memory에 최적화된 multi-head attention을 구현하라", custom layer/loss, 스케일에서 OOM 디버깅; "1B video clip을 고품질 text alignment와 de-dup으로 curate하는 pipeline을 설계하라"; LambdaMART 대 RankNet, 왜 NDCG인지.

> [!QUOTE] 이야기 매핑
> Beomyoung → Adobe: **단일 최고의 fit** — ZIM은 editing-grade 경계이고, CLOVA-X image editing은 직접적인 research-to-product creative-tool 루프입니다. research talk를 ZIM으로 시작하고; PyTorch 구현 세부를 라이브로 방어할 준비를 하세요.

**Leveling / comp** *[levels.fyi/Glassdoor]*: IC research는 **P-level**(~P30–P60)을 쓰고, Principal Scientist에서 정점. RS median TC ~$320K, range ~$179K–$585K. Comp mix ~**71% base / 21% RSU / 8% bonus** — FAANG 대비 **base-heavy, equity 가벼움**; 같은 "level"에서 TC가 FAANG에 못 미칠 수 있으므로, 경쟁 offer + level/equity 밀어붙이기가 lever.

---

## ByteDance Seed (Vision Foundation Model)

**Loop** *[Lore; 팀/지역별로 다름]*. **Online Assessment**(60–120분 HackerRank, 종종 카메라 켜기; 추천 경력직은 생략 가능) → recruiter screen(~30분) → **technical loop(2–4 라운드 ~45분)** 종종 **project/research deep-dive와 매 라운드 라이브 LeetCode 문제를 페어링** → **HM/final**(팀 프로젝트, mission alignment; director 포함 가능). **heavy-LeetCode 평판은 확인됨 — research role에도 질문이 Hard 쪽으로 치우침(Google/Amazon 중앙값보다 어려움).** graph search, DP, string, heap을 Python/C++로 준비하세요.

**2025–26 방향** *[Verifiable + 일부 언론/루머]*: **ByteDance Seed** = foundation-model 부문(2024 설립). **Wu Yonghui**(ex-Google)가 2025년 2월 Seed를 맡음 *[언론]*; 공격적 채용; **Top Seed** early-career PhD 프로그램. LLM / **Doubao**(Seed 2.0가 2026년 2월 보고됨 — *param 수는 루머 수준*; **Seed-OSS** ~36B open-weights). VL: **Seed1.5-VL**(~8.7B open-weights, 다수 공개 benchmark에서 SOTA). **Vision-generation 팀(타겟)** — **Seedream**(image gen), **Seedance**(video gen; unified audio-video), **SeedEdit**(NL image editing). Agentic: **UI-TARS**(GUI agents).

**문화 / behavioral** *[Lore]*: 빠른 속도, **OKR-driven, output-oriented, 낮은 실패 관용**; WLB 낮게 평가됨(~2.3–2.7/5; "996"/"007" 언급, 팀 의존); comp는 강점으로 평가됨(~4.0/5). 파고드는 것: 동기, 모호함/속도 속 운영, cross-timezone 협업, ownership, mission alignment.

**알려진 질문 스타일** *[Lore]*: Hard LeetCode(DP, graph, string, heap)와 낯선 follow-up; ML — bias/variance, transformer/BERT pretraining, ROC/AUC; vision → diffusion/VLM 내부; Seed의 generation/multimodal 방향에 묶인 research deep-dive.

> [!QUOTE] 이야기 매핑
> Beomyoung → Seed Vision: **promptable discriminative vision(ZIM, SAM 계보) → generative multimodal**로 이어지되 여전히 구조/제어를 존중하는 arc를 pitch하세요. product surface가 짧은 research→users 루프를 줍니다. **준비의 대부분을 Hard LeetCode에 투자**하세요 — 여기서 강한 researcher가 가장 자주 넘어지는 지점입니다.

**Leveling / comp** *[levels.fyi, US]*: 숫자 ladder(1-1, 2-1, 2-2, 3-1…). RS (US) ~$168K (L1-2) → $851K (L3-2), median ~$395K. **Private/pre-IPO** — RSU가 크지만 **internal mark에서 illiquid**; buyback/liquidity 조건이 중요; level 배치가 delta의 대부분을 좌우.

---

## Mistral AI

**Loop** *[Lore/coaching-sourced]*. Recruiter(20–30분: **"왜 하필 Mistral인가"**) → technical phone screen(60분, medium-hard coding 문제 하나; Python default, inference에는 C++/CUDA, infra에는 Rust) → **take-home(4–8시간)** research/senior에 흔함 — NN 컴포넌트/training primitive를 from scratch로 구현하거나, near-paper 스타일의 작은 LLM 실험을 설계 → **restitution / research presentation**(팀이 take-home의 방법론을 파고듦; ~15–20분 발표 + 20분+ 어려운 Q&A — take-home + follow-up은 **하나로 연결된 평가**) → **onsite(4–5 라운드)**: 1–2 coding, 1 system design(LLM-infra), 1–2 ML/research depth, behavioral/HM. Timeline ~4–8주. 반복되는 불만: **take-home 리뷰 중 느린 follow-up**.

**Signal** *[Lore]*: 진짜 research 유창함 + transformer/MoE 내부; **깔끔하고 typed되고 tested된 코드**(type hint, Pydantic); system design에서 실제 수치(H100 memory, throughput, p95); **open-weight** 전략에 대한 실제 의견. Research role: 두 ML/research 라운드, PhD-or-equivalent 기대("transformer를 API로만 *써본* 후보는 고전한다"). Applied/eng role: system-design 라운드 하나; **PhD 불필요**.

**2025–26 방향** *[Verifiable]*: open-weight LLM — Mistral 7B, **Mixtral**(sparse MoE), Mistral Large, **Mistral Large 3**(open MoE); efficiency/MoE(top-k routing, 작은 모델); reasoning — **Magistral**; multimodal — **Pixtral / Pixtral Large**; consumer **Le Chat**. 2026 통합: **Mistral Small 4**(Magistral reasoning + Pixtral vision + Devstral coding 병합, 설정 가능한 `reasoning_effort`). 준비: **Mistral 7B & Mixtral 논문을 읽으세요**; MoE trade-off(2-of-8 expert, capacity factor, load balancing), transformer 내부(GQA, sliding-window attention, RoPE, BPE), inference(continuous batching, KV-cache, speculative decoding, quantization; vLLM 학습), post-training(SFT/DPO/RLHF).

**문화 / behavioral** *[Lore]*: Paris-HQ 유럽 랩; US always-on 대비 "sustainable high-intensity"; 강한 노동 보호; 일상 엔지니어링에서 EU AI Act 준수. 두드러진 파고들기: **"US frontier 랩이 아니라 왜 Mistral인가?"** Red flag: 모호한 MoE 이해, 특정 Mistral 논문에 대한 관여 없음, open-weights 무시, Paris에서 US comp 기대.

**알려진 질문 스타일** *[Lore]*: from-scratch — multi-head attention(올바른 Q/K/V split, 1/√dₖ scaling, causal mask), RoPE, top-k/top-p sampling, BPE, sliding-window mask, KV-cache — type/test와 함께 <30분에 깔끔하게; system design — "70B MoE를 10K req/s, p95 <1s로 serve하라", "checkpoint recovery와 함께 2K H100에 걸쳐 70B dense 모델을 train하라"; ML — "Mixtral은 왜 2-of-8로 route하나?", step ~10k–40k에서의 loss spike 디버깅.

> [!QUOTE] 이야기 매핑
> Beomyoung → Mistral (Seoul, Applied Scientist): **research-to-product 루프**와 **code ownership**으로 시작하세요; Seoul role은 bilingual(KR+EN)이고 customer-facing입니다. transformer를 *from scratch*로 구현해 본 적 없다면 정직하게 말하고 — 그다음 take-home 전에 격차를 메우세요, restitution이 그것을 *반드시* 파고들 테니까요.

**Leveling / comp** *[Lore/coaching; levels.fyi 희소]*: 스타트업 — **equity가 중요**. Paris 밴드(coaching-site): Senior/Staff RS ~€200–280K base / ~€300–500K TC; 상단 RS TC는 $490K–$950K로 보고됨 *[낙관적/불확실]*. Equity = **BSPCE**(프랑스 tax-advantaged option; 4년 vest, 1년 cliff), **IPO/secondary 전까지 illiquid**. Paris base는 US 랩보다 ~30–50% 낮음(더 낮은 COL + 유리한 BSPCE 세금으로 상쇄). BSPCE **수량 + strike**를 협상하고; valuation 기준 + liquidity를 명확히 하세요.

---

## Microsoft — MSR & AI Frontiers / Microsoft AI (MAI)

**두 개의 구별되는 트랙 — 분리하세요.** **MSR** = academic-style, publication-driven; **product AI org**(Microsoft AI / Copilot / Azure AI) = shipping-first.

**A) MSR Researcher loop** *[Lore, 확인됨]*: recruiter(~30분) → HM/technical screen(~30–45분, research deep-dive + 가벼운 case/ML coding) → **onsite(4–6+ 라운드)**: **research presentation / job talk**(주제를 직접 선택) + **4–6 researcher 1:1**(deep dive, future vision) + **1–2 가벼운 coding 라운드**(~Medium, 때로 k-means 같은 ML-impl) + ML theory/breadth + culture/behavioral + **"As Appropriate" (AA)** 라운드. **job talk이 MSR의 시그니처 요소** — 평가자가 실제로 묻는 건 *"이 사람과 협업하고 싶은가?"*; 한 시간 중 ~⅓을 Q&A로 예상하고, 우아한 "모르지만, 이게 제 접근입니다"가 *강점*입니다.

**B) Applied Scientist loop (product org)** *[Lore]*: recruiter → HM(~45분) → technical phone screen(ML-grounded coding + project deep-dive; MS는 **순수 algorithm 퍼즐에서 ML 추론 쪽으로** 이동함 — 명확한 추론이 있으면 pseudocode 허용) → **onsite(4–6)**: ML fundamentals + coding, **ML system design**(~60분 end-to-end, 종종 절반은 behavioral), case study / research presentation, behavioral, 그리고 **AA** 라운드.

**"As Appropriate" 라운드** *[Verifiable mechanism, Lore details]*: *직속 팀 외부*의 시니어 면접관이 채용 기준과 장기 잠재력을 지키며; 그들의 투표가 상당한 무게를 가짐(bar-raiser 유사물). Level 의존성: L59–61은 DSA에 더 가중; L63+는 system design + research-vision/leadership signal을 추가.

**2025–26 방향** *[Verifiable]*: **Phi** small language model(Phi-4, Phi-4-mini, Phi-4-multimodal) — performance-per-parameter, scale보다 data quality; **AI Frontiers lab**(Ahmed Awadallah 주도)에서의 **agentic AI & reasoning** 포함 **Fara-7B** open-weight computer-use agent; multimodal foundation model; AI4Science / Healthcare AI; systems & inference efficiency, RAG; Responsible AI. 별개로, **MAI** 자체 모델 노력(MAI-Voice-1, 2025년 MAI-1-preview; 2026까지 추가 reasoning/coding/image 모델 보고됨 — *언론 보도, 보도된 대로 취급*).

**문화 / behavioral** *[Verifiable framing]*: **growth mindset**(Nadella/Dweck) + **"Model, Coach, Care"** + **"One Microsoft."** 비관리자도 leadership-through-influence, mentoring, ownership, learning-from-failure를 보여야 함. curiosity, cross-org 협업, 좌절로부터의 성장을 강조하는 STAR story.

**알려진 질문 스타일** *[Lore]*: LeetCode Medium(array, tree/graph, string); ML-from-scratch(k-means, bag-of-words); transformer 내부, batch-vs-layer norm, RLHF, imbalanced data; system design — recommendation, object-detection pipeline; research — "논문을 훑어라 — 당신의 구체적 기여와 *왜* 그 결정을 했나?"; behavioral/AA — technical conflict, 잘 안 풀린 research 방향, mentoring/influence without authority.

> [!QUOTE] 이야기 매핑
> Beomyoung → MSR / AI Frontiers: **grounding을 agent의 perception layer로** 프레임하세요 — "단순 caption이 아니라 검증 가능한 region" — grounded VLM + visual-reasoning agent를 agentic-AI agenda와 정렬. 동기에서 구체적인 lab/job ID를 짚으세요. 여러분의 Centum Digital Week "Beyond AI, Into Agents" 발표를 언급하세요.

**Leveling / comp** *[ladder Verifiable; comp levels.fyi]*: 밴드 ~59–60(Researcher), 61–62, 63–64(Senior Researcher), 65–66(Principal), 67+(Partner/Fellow). RS ~$156K (L59) → $625K+ (L67), median ~$292K; AI Researcher(L63–66) ~$315K–$474K+. **Level 밀어붙이기가 가장 큰 lever**; 경쟁 offer(OpenAI/DeepMind/Anthropic/Meta)가 stock/sign-on을 움직임; 특별 AI-talent stock award가 보고된 요인.

---

## 이들 중 어디로든 걸어 들어가는 법

<details class="qa"><summary>그들이 묻습니다: "왜 하필 우리인가요?" — 어떻게 뻔하게 들리지 않을까요?</summary>
<div class="qa-body">

**짧게:** 그 org의 *구체적인* 발표된 모델/논문을 짚고, 왜 인상 깊었는지 정직한 한 문장, 그리고 당신 연구가 그것에 당기는 lever.

**깊게:** "multimodal 모델을 하고 싶어요"는 모두에게 맞고 누구에게도 신호하지 않습니다. 대조: *"SAM의 promptable interface가 바로 ZIM이 matting quality를 위해 확장한 것이라, SAM-3 concept-segmentation 방향이 정확히 제 region-level 연구가 기여할 지점입니다."* 그 문장은 당신이 그들의 연구를 읽었고, 자기 연구를 알고, 그 접점을 본다는 걸 증명합니다. 회사마다 하나씩 만드세요 — [Recruiter & HM Screens](#/process/recruiter-hm) 참조.
</div></details>

<details class="qa"><summary>어느 회사가 "안전한" 첫 면접이고 어느 곳이 도전(reach)인가요?</summary>
<div class="qa-body">

**짧게:** 명성이 아니라 *당신의* fit으로 순서를 정하세요. shipped-product record를 가진 CV/VLM 후보에게 **Adobe**(ZIM↔editing)와 **Apple**(on-device efficiency)은 자연스러운 fit의 워밍업입니다; **ByteDance**(Hard coding)와 **Mistral**(from-scratch transformer take-home)은 *형식*이 전용 준비를 가장 필요로 하는 곳입니다.

**깊게:** 여기서 "reach"는 당신이 충분히 뛰어난지가 아니라 *format risk*에 관한 것입니다. 당신의 research depth는 어디서나 강합니다; 차별화된 risk는 Meta/ByteDance coding 기준과 Mistral의 연결된 take-home+restitution입니다. 이야기가 깔끔하게 들어맞는 곳에서 워밍업한 뒤, coding이 몸에 익으면 format-heavy loop를 치세요. 타임라인 클러스터링은 [pipeline](#/process/pipeline) 참조.
</div></details>

### 회사를 가로질러 반복되는 follow-up

- *"당신의 최고 논문의 한계는 무엇인가요?"* — 모든 research org가 이 질문의 버전을 묻습니다. ZIM(또는 flagship)의 정직한 failure mode를 준비하세요. [Failure & Negative Results](#/research/failure) 참조.
- *"이게 우리 data/compute로 어떻게 스케일하나요?"* — 당신의 방법을 distributed training + serving 현실에 연결하세요. [Design Framework](#/system-design/framework) 참조.
- *"다음에 무엇을 할지 어떻게 정하나요?"* — research taste. label-efficient/continual → grounded VLM → agents를 표류가 아니라 *의도된* 궤적으로 묶으세요.

## Cheat-sheet

| Company | 한 줄 준비 우선순위 |
| --- | --- |
| Meta / FAIR | Near-SWE coding + job talk; FAIR 대 GenAI 명확히; grounding + shipped ZIM으로 시작 |
| NVIDIA | Job talk + DL-depth; efficiency-as-science; intellectual honesty; CUDA willingness |
| Apple | On-device/privacy 프레이밍; NMS/IoU/K-means from scratch; 제품 추측 금지 |
| Adobe | Research talk(ZIM) + 엄밀한 PyTorch coding; diffusion 수식; product sense |
| ByteDance Seed | **Hard LeetCode가 게이트**; discriminative→generative 이야기; speed/OKR 문화 |
| Mistral | Take-home용 from-scratch transformer; MoE 의견; "왜 Mistral, US 랩 아니고"; 깔끔한 typed 코드 |
| Microsoft MSR | Central job talk + AA 라운드; grounding-as-agent-perception; growth mindset 이야기 |

**Related:** [The RS/AS Pipeline](#/process/pipeline) · [Recruiter & HM Screens](#/process/recruiter-hm) · [Offers & Negotiation](#/process/negotiation) · [The Research Job Talk](#/research/job-talk) · [Coding Round Strategy](#/coding/strategy) · [Design Framework](#/system-design/framework) · [The 2026 Landscape](#/start/landscape-2026)
