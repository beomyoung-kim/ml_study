# Company Playbooks

<div class="tag-row"><span class="tag">Meta / FAIR</span><span class="tag">NVIDIA</span><span class="tag">Apple</span><span class="tag">Adobe</span><span class="tag">ByteDance Seed</span><span class="tag">Mistral</span><span class="tag">Microsoft / MSR</span></div>

> [!TIP] Why this chapter exists
> The [pipeline](#/process/pipeline) is shared; the *emphasis* is not. Meta near-SWE coding, NVIDIA's faculty-style job talk, Mistral's take-home restitution, Microsoft's "As Appropriate" round — each is a different filter. This is the per-company map for a **CV / VLM / agents** candidate: loop shape, what they weight, current research directions, culture signals, and how to map your story.

> [!WARNING] Facts vs. lore — read this first
> Claims tagged **[Verifiable]** come from company pages, official blogs, or levels.fyi aggregates. Claims tagged **[Lore]** come from Glassdoor/Blind/1point3acres/forum anecdotes — directionally useful, **not** guaranteed, and team/quarter-dependent. **No interviewer names or confidential internal rubrics are reproduced.** Loop composition changes quarterly — confirm with your recruiter.

## Comparison at a glance

| Company | Coding weight | Job-talk centrality | Distinctive signal | RS comp *(levels.fyi aggregate, 2025–26)* |
| --- | --- | --- | --- | --- |
| **Meta / FAIR** | High — near-SWE, 2–3 LeetCode rounds | Prominent (FAIR/Applied) | Publish *and* ship at scale; ownership in ambiguity | ~$305K–$581K+ (IC4–IC6) |
| **NVIDIA Research** | Moderate; some labs skip; C++/CUDA valued | Central (~1 hr, faculty-style) | GPU/systems + research-engineering; intellectual honesty | ~$205K–$533K+, median ~$455K |
| **Apple MLR/AIML** | Moderate–high (1–2+ coding) | Central for MLR | On-device / privacy / HW co-design; secrecy | ~$334K–$476K+ (ICT3–5) |
| **Adobe Research** | High — PyTorch + algo rounds | Central | Publications + PyTorch rigor + product sense | median ~$320K, base-heavy |
| **ByteDance Seed** | Very high — **Hard** LeetCode even for RS | Folded into technical rounds | Speed + coding + generation/VLM depth | median ~$395K (US), private equity |
| **Mistral AI** | ML-primitives from scratch + LeetCode-w/-twist | **Take-home + restitution** | MoE/transformer internals; open-weights conviction | Paris ~€200–500K TC; BSPCE equity |
| **Microsoft MSR** | Lighter (Medium + ML-impl) | **Central job talk** + researcher panel | Research agenda + collaboration; AA round | ~$156K–$625K+ (L59–67) |

*All comp is self-reported levels.fyi aggregate and fluctuates; coding-weight/talk-centrality ratings synthesize [Lore] loop reports and vary by team.*

---

## Meta — AI Research / FAIR (incl. VLM)

**Loop** *[Lore, one first-hand account]*. Meta does not run one uniform RS loop — the "Research Scientist" title spans tracks.
- Recruiter screen (~30–45 min): background, publications, track alignment, level calibration.
- Technical phone screen (~45 min, CoderPad): almost always coding — 1–2 LeetCode Mediums (occasionally a Hard).
- Full loop (virtual, **4–6 rounds**): often **2–3 coding rounds** (two Medium/Hard each — Meta's coding bar is near-SWE even for research titles, a *defining* feature), **1 ML system design**, a **research deep-dive / job talk** (walk the full lifecycle: problem → experimental design → data → architecture → eval → results), and **behavioral** (often run by a PhD probing your research trajectory).

**Track taxonomy** *[Lore]*: RS–Systems & Infrastructure (SWE-flavored), RS–Central Applied Science (stats/econ + a stats exam + EDA coding + research presentation), and **RS, ML / FAIR** (AI/ML research). **FAIR operates like a postdoc/faculty hire** expecting multiple first-author NeurIPS/ICML/CVPR-tier papers. Note flagship LLM work (Llama) increasingly sits in **Meta GenAI / Superintelligence Labs**, *not* FAIR proper — clarify FAIR-vs-GenAI with your recruiter.

**2025–26 directions** *[Verifiable, Meta AI]*: **Llama 4** (Scout & Maverick — natively multimodal MoE, long context); the **Segment Anything** family (SAM 2.1, and reported **SAM 3** with promptable *concept* segmentation); self-supervised vision (**DINO/DINOv2/DINOv3**, **V-JEPA** world-model agenda under LeCun); **Perception Encoders**; **Meta Spirit LM** (text+speech).

**Culture / behavioral** *[Lore]*: Meta values — move fast, ownership, bias for action, direct communication. Probes: ownership in ambiguity ("requirements completely blank"), self-awareness (own your failures), quantified impact. STAR stories ~3 min.

> [!QUOTE] Map your story
> Beomyoung → FAIR/VLM: lead with **grounded VLMs + region-level evidence** (aligns with SAM/DINO/perception lines) and the **ZIM → CLOVA-X** research-to-product proof. For the near-SWE coding bar, over-prepare Mediums to ~20 min and rehearse from-scratch attention/sampling.

<div class="proscons"><div><div class="pros-t">Plays to your strengths</div>Vision-foundation + grounding depth; shipped-at-scale story; strong first-author record</div><div><div class="cons-t">Watch out</div>Coding bar is unusually high for a research title; a "perfect" solution can still be a no-hire without collaboration signal</div></div>

**Leveling / comp** *[levels.fyi]*: E-ladder E4…E8; IC5/E5 is the common senior target. RS ~$305K (IC4) → $581K (IC6). Meta benchmarks **total comp** — optimize equity + refreshers, cite competing-offer numbers.

---

## NVIDIA Research

**Loop** *[Lore]*. Academic/faculty-style; **varies strongly by lab** (some skip a dedicated coding round).
- Recruiter phone screen → **technical screen** (often a 75-min HackerRank for engineering-adjacent roles; for pure research, a conversation on interests/projects w/ some coding/ML) → **job talk** (~1 hr, present your research) → **final loop** (~6 one-on-ones): coding/systems (Python/C++/**CUDA**/PyTorch, distributed training, GPU trade-offs — some labs skip), a DL-depth round (transformer internals, parallel training, optimization), multiple research deep-dives, and behavioral (usually the HM). Timeline ~4–6 weeks.

**2025–26 directions** *[Verifiable, NVIDIA]*: generative AI & LLMs (video/vision generation, RL for LLMs/agents, efficient inference e.g. **Dynamo** on Blackwell); **Physical AI & robotics** — **Cosmos** world models, **Isaac GR00T** humanoid, **BioNeMo**; autonomous driving; neural graphics. Notable lab: **GEAR** (Generalist Embodied Agent Research; Jim Fan & Yuke Zhu) — LLMs for planning, VLMs, internet-scale world models. *(A specific "Learning & Perception / Toronto" team was not confirmed — verify with recruiter.)*

**Culture / behavioral** *[Verifiable values + Lore]*: five values — **Innovation, Speed & Agility, Intellectual Honesty, Excellence, One Team**; mantra "the mission is the boss." **Intellectual honesty** (admitting what you don't know) is an *explicit* evaluated signal — a graceful "I don't know, but here's how I'd approach it" scores.

**Known question styles** *[Lore]*: coding — reverse linked list, number-of-islands (grid BFS/DFS), C++ specifics (memory mgmt, virtual functions); ML-from-scratch — dropout/BN/softmax/attention; **LayerNorm vs BatchNorm** (math + why LN in transformers); quantization/distillation; design a loss aligning text↔video embeddings.

> [!QUOTE] Map your story
> Beomyoung → NVIDIA: frame **efficiency as a scientific goal** — on-device seg at ~10 ms, ONNX serving, multi-node PyTorch training — and pair it with generative-vision ambition. Willingness to ramp CUDA is fine to state honestly; the intellectual-honesty value rewards it.

**Leveling / comp** *[levels.fyi]*: IC1–IC6+; RSUs branded **"NSU"** (quarterly vests). RS ~$205K (IC1) → $533K (IC6), median ~$455K. Model signed-grant vs current value (large divergence from stock appreciation).

---

## Apple — Machine Learning Research (MLR) & AIML

**Loop** *[Lore; highly team-specific — MLR pure-research vs AIML applied]*. RS/MLR (~3–5 wks): recruiter → 1–2 technical phone screens (coding + domain theory) → virtual onsite: **45–60 min research talk** + **5–6 researcher interviews** (coding, research-theory deep-dive, ML system design, behavioral). AIML/MLE (~4–6 wks): recruiter → 1–2 phone screens → sometimes a take-home → **onsite of ~5–8 rounds** → HM conversation.

**Signals** *[Lore + Verifiable framing]*: applied ML reasoning **under real constraints** — on-device ("could this run on a phone?"), privacy-first (federated learning, differential privacy, **Private Cloud Compute**), accuracy vs latency/battery/memory, software–hardware co-design for Apple Silicon. MLR = curiosity-driven, publishes, judged on the talk + novelty; AIML = product-embedded, more coding + system design. Apple deliberately blurs research↔product ("research must be production-ready").

**2025–26 directions** *[Verifiable]*: **Apple Foundation Models / Apple Intelligence** — a ~3B on-device model + a larger server model on Private Cloud Compute; **Foundation Models framework** exposes the on-device LLM to developers; **Manzano** — unified multimodal model with a hybrid vision tokenizer. On-device/privacy/efficiency; **MLX** for Apple Silicon. Teams: **MLR**, **AIML**, **SIML** (Siri / "Hey Siri" trigger detection), Health.

**Culture / behavioral** *[Mixed]*: **secrecy** — MLR publishes only if it doesn't compromise product; product impact > publication count; expect probing on discretion and working without external visibility. Cross-functional with hardware/product teams.

**Known question styles** *[Lore]*: implement Non-Maximum Suppression, compute IoU, K-Means from scratch, LeetCode Medium; Batch vs Layer Norm and when; forward/reverse diffusion; derive binary-classification loss; design "Hey Siri" trigger and handle false positives; improve Face ID in low light.

> [!QUOTE] Map your story
> Beomyoung → Apple: the **on-device human-seg model (~10 ms, ONNX)** is a near-perfect fit for the efficiency/privacy narrative; pair with ZIM for foundation-model depth. Do **not** speculate about unreleased products — stay in public JD language.

**Leveling / comp** *[levels.fyi]*: **ICT2–ICT6**; ML targets ICT4/ICT5. RS ~$334K (ICT3) → $476K+ (ICT5). Base bands relatively rigid → **equity + level** are the levers; leveling is committee/team-driven and worth pushing.

---

## Adobe Research (incl. Firefly Applied Science)

**Loop** *[Lore]*. ~5 rounds over 4–6 wks: recruiter → **research presentation** (present peer-reviewed publications to a panel, defend with rigor) → technical problem-solving (coding + PyTorch + ML theory) → behavioral → **final onsite** (deep coding, ML theory, ML system design, behavioral). The recurring formula: *"a stellar publication record gets you in the door, but you must pass rigorous PyTorch + algorithmic coding rounds to get the offer."*

**2025–26 directions** *[Verifiable]*: **Firefly** generative AI is flagship. MAX 2025: **Firefly Image Model 5** (native 4MP photorealistic), new audio/video tools, **Firefly Foundry** (custom enterprise models). Adobe Research's **Video AI Lab** + **Foundation Model Team** built the **Firefly Video Model** and **GenExtend** (Premiere Pro). Interview themes: diffusion (DDPM, DDIM, classifier-free guidance), ViTs/VLMs, attention internals, temporal consistency in video gen, large-scale multimodal data curation/de-dup, distributed training (DDP/FSDP/tensor-parallel), fine-tuning/alignment.

**Culture / behavioral** *[Mixed]*: publication-friendly, academic-style lab (publishes heavily, university collaborations, interns); less secrecy than Apple. Behavioral probes collaboration, communication, translating theory into creative products — **product sense** is a differentiator.

**Known question styles** *[Lore]*: "walk through reverse-diffusion math — condition it on user editing prompts"; "implement multi-head attention optimized for memory," custom layers/loss, OOM debugging at scale; "design a pipeline to curate 1B video clips with high-quality text alignment and de-dup"; LambdaMART vs RankNet, why NDCG.

> [!QUOTE] Map your story
> Beomyoung → Adobe: the **strongest single fit** — ZIM is editing-grade boundaries, and CLOVA-X image editing is a direct research-to-product creative-tool loop. Lead the research talk with ZIM; be ready to defend PyTorch implementation details live.

**Leveling / comp** *[levels.fyi/Glassdoor]*: IC research uses **P-levels** (~P30–P60), topping at Principal Scientist. RS median TC ~$320K, range ~$179K–$585K. Comp mix ~**71% base / 21% RSU / 8% bonus** — **base-heavy, lighter equity** vs FAANG; TC can trail FAANG at the same "level," so competing offers + pushing level/equity are the levers.

---

## ByteDance Seed (Vision Foundation Model)

**Loop** *[Lore; varies by team/region]*. **Online Assessment** (60–120 min HackerRank, camera often on; referred experienced hires may skip) → recruiter screen (~30 min) → **technical loop (2–4 rounds ~45 min)** often pairing a **project/research deep-dive with a live LeetCode problem each round** → **HM/final** (team projects, mission alignment; may include a director). **The heavy-LeetCode reputation is confirmed — questions skew Hard (harder than Google/Amazon medians) even for research roles.** Prep graph search, DP, strings, heaps in Python/C++.

**2025–26 directions** *[Verifiable + some press/rumor]*: **ByteDance Seed** = the foundation-model division (est. 2024). **Wu Yonghui** (ex-Google) took over Seed Feb 2025 *[press]*; aggressive hiring; **Top Seed** early-career PhD program. LLMs / **Doubao** (Seed 2.0 reported Feb 2026 — *param counts are rumor-tier*; **Seed-OSS** ~36B open-weights). VL: **Seed1.5-VL** (~8.7B open-weights, SOTA on many public benchmarks). **Vision-generation team (target)** — **Seedream** (image gen), **Seedance** (video gen; unified audio-video), **SeedEdit** (NL image editing). Agentic: **UI-TARS** (GUI agents).

**Culture / behavioral** *[Lore]*: fast pace, **OKR-driven, output-oriented, low failure tolerance**; WLB rated low (~2.3–2.7/5; "996"/"007" referenced, team-dependent); comp rated a strength (~4.0/5). Probes: motivation, operating in ambiguity/speed, cross-timezone collaboration, ownership, mission alignment.

**Known question styles** *[Lore]*: Hard LeetCode (DP, graphs, strings, heaps) with unfamiliar follow-ups; ML — bias/variance, transformer/BERT pretraining, ROC/AUC; vision → diffusion/VLM internals; research deep-dive tied to Seed's generation/multimodal directions.

> [!QUOTE] Map your story
> Beomyoung → Seed Vision: pitch the arc from **promptable discriminative vision (ZIM, SAM lineage) → generative multimodal** that still respects structure/control. The product surface gives a short research→users loop. **Invest most prep in Hard LeetCode** — this is where strong researchers most often fall here.

**Leveling / comp** *[levels.fyi, US]*: numeric ladder (1-1, 2-1, 2-2, 3-1…). RS (US) ~$168K (L1-2) → $851K (L3-2), median ~$395K. **Private/pre-IPO** — RSUs are large but **illiquid at internal marks**; buyback/liquidity terms matter; level placement drives most of the delta.

---

## Mistral AI

**Loop** *[Lore/coaching-sourced]*. Recruiter (20–30 min: **"why Mistral specifically"**) → technical phone screen (60 min, one medium-hard coding problem; Python default, C++/CUDA for inference, Rust for infra) → **take-home (4–8 hrs)** common for research/senior — implement an NN component/training primitive from scratch, *or* design a small LLM experiment in near-paper style → **restitution / research presentation** (team drills the take-home's methodology; ~15–20 min present + 20+ min hard Q&A — the take-home + follow-up are **one linked evaluation**) → **onsite (4–5 rounds)**: 1–2 coding, 1 system design (LLM-infra), 1–2 ML/research depth, behavioral/HM. Timeline ~4–8 wks. Recurring complaint: **slow follow-up during take-home review**.

**Signals** *[Lore]*: genuine research fluency + transformer/MoE internals; **clean, typed, tested code** (type hints, Pydantic); real numbers in system design (H100 memory, throughput, p95); a real opinion on the **open-weight** strategy. Research roles: two ML/research rounds, PhD-or-equivalent expected ("candidates who only *used* transformers via APIs struggle"). Applied/eng roles: one system-design round; **PhD not required**.

**2025–26 directions** *[Verifiable]*: open-weight LLMs — Mistral 7B, **Mixtral** (sparse MoE), Mistral Large, **Mistral Large 3** (open MoE); efficiency/MoE (top-k routing, small models); reasoning — **Magistral**; multimodal — **Pixtral / Pixtral Large**; consumer **Le Chat**. 2026 consolidation: **Mistral Small 4** (merges Magistral reasoning + Pixtral vision + Devstral coding, configurable `reasoning_effort`). Prep: **read the Mistral 7B & Mixtral papers**; MoE trade-offs (2-of-8 experts, capacity factors, load balancing), transformer internals (GQA, sliding-window attention, RoPE, BPE), inference (continuous batching, KV-cache, speculative decoding, quantization; study vLLM), post-training (SFT/DPO/RLHF).

**Culture / behavioral** *[Lore]*: Paris-HQ European lab; "sustainable high-intensity" vs US always-on; strong labor protections; EU AI Act compliance in daily engineering. Prominent probe: **"Why Mistral rather than US frontier labs?"** Red flags: vague MoE understanding, no engagement with a specific Mistral paper, dismissing open-weights, expecting US comp in Paris.

**Known question styles** *[Lore]*: from-scratch — multi-head attention (correct Q/K/V split, 1/√dₖ scaling, causal mask), RoPE, top-k/top-p sampling, BPE, sliding-window masks, KV-cache — cleanly with types/tests in <30 min; system design — "serve a 70B MoE at 10K req/s, p95 <1 s," "train a 70B dense model across 2K H100s with checkpoint recovery"; ML — "why does Mixtral route 2-of-8?", debug a loss spike at step ~10k–40k.

> [!QUOTE] Map your story
> Beomyoung → Mistral (Seoul, Applied Scientist): lead with the **research-to-product loop** and **code ownership**; the Seoul role is bilingual (KR+EN) and customer-facing. Be honest if you haven't implemented a transformer *from scratch* — then close the gap before the take-home, because the restitution *will* probe it.

**Leveling / comp** *[Lore/coaching; levels.fyi sparse]*: startup — **equity matters**. Paris bands (coaching-site): Senior/Staff RS ~€200–280K base / ~€300–500K TC; higher-end RS TC reportedly $490K–$950K *[optimistic/uncertain]*. Equity = **BSPCE** (France tax-advantaged options; 4-yr vest, 1-yr cliff), **illiquid until IPO/secondary**. Paris base ~30–50% below US labs (offset by lower COL + favorable BSPCE tax). Negotiate BSPCE **quantity + strike**; clarify valuation basis + liquidity.

---

## Microsoft — MSR & AI Frontiers / Microsoft AI (MAI)

**Two distinct tracks — separate them.** **MSR** = academic-style, publication-driven; **product AI orgs** (Microsoft AI / Copilot / Azure AI) = shipping-first.

**A) MSR Researcher loop** *[Lore, corroborated]*: recruiter (~30 min) → HM/technical screen (~30–45 min, research deep-dive + light case/ML coding) → **onsite (4–6+ rounds)**: **research presentation / job talk** (you pick the topic) + **4–6 researcher 1:1s** (deep dives, future vision) + **1–2 lighter coding rounds** (~Medium, sometimes ML-impl like k-means) + ML theory/breadth + culture/behavioral + the **"As Appropriate" (AA)** round. The **job talk is the signature MSR element** — evaluators are really asking *"do I want to collaborate with this person?"*; expect ~⅓ of the hour in Q&A, and a graceful "I don't know, but here's my approach" is a *strength*.

**B) Applied Scientist loop (product orgs)** *[Lore]*: recruiter → HM (~45 min) → technical phone screen (ML-grounded coding + project deep-dive; MS has shifted **away from pure algorithm puzzles toward ML reasoning** — pseudocode acceptable with clear reasoning) → **onsite (4–6)**: ML fundamentals + coding, **ML system design** (~60 min end-to-end, often half behavioral), case study / research presentation, behavioral, and the **AA** round.

**The "As Appropriate" round** *[Verifiable mechanism, Lore details]*: a senior interviewer from *outside* the immediate team guards the hiring bar and long-term potential; their vote carries significant weight (bar-raiser analog). Level-dependence: L59–61 weight DSA more; L63+ adds system design + research-vision/leadership signal.

**2025–26 directions** *[Verifiable]*: **Phi** small language models (Phi-4, Phi-4-mini, Phi-4-multimodal) — performance-per-parameter, data quality > scale; **agentic AI & reasoning** at the **AI Frontiers lab** (led by Ahmed Awadallah) incl. **Fara-7B** open-weight computer-use agent; multimodal foundation models; AI4Science / Healthcare AI; systems & inference efficiency, RAG; Responsible AI. Separately, the **MAI** in-house model effort (MAI-Voice-1, MAI-1-preview in 2025; further reasoning/coding/image models reported into 2026 — *press-reported, treat as reported*).

**Culture / behavioral** *[Verifiable framing]*: **growth mindset** (Nadella/Dweck) + **"Model, Coach, Care"** + **"One Microsoft."** Even non-managers must show leadership-through-influence, mentoring, ownership, learning-from-failure. STAR stories emphasizing curiosity, cross-org collaboration, growth from setbacks.

**Known question styles** *[Lore]*: LeetCode Medium (arrays, tree/graph, strings); ML-from-scratch (k-means, bag-of-words); transformer internals, batch-vs-layer norm, RLHF, imbalanced data; system design — recommendation, object-detection pipeline; research — "walk through a paper — your specific contributions and *why* those decisions?"; behavioral/AA — technical conflict, a research direction that didn't pan out, mentoring/influence without authority.

> [!QUOTE] Map your story
> Beomyoung → MSR / AI Frontiers: frame **grounding as an agent's perception layer** — "not just captions, but verifiable regions" — aligning grounded VLMs + visual-reasoning agents with the agentic-AI agenda. Name the specific lab/job ID in your motivation. Reference your Centum Digital Week "Beyond AI, Into Agents" talk.

**Leveling / comp** *[ladder Verifiable; comp levels.fyi]*: bands ~59–60 (Researcher), 61–62, 63–64 (Senior Researcher), 65–66 (Principal), 67+ (Partner/Fellow). RS ~$156K (L59) → $625K+ (L67), median ~$292K; AI Researcher (L63–66) ~$315K–$474K+. **Pushing level is the biggest lever**; competing offers (OpenAI/DeepMind/Anthropic/Meta) move stock/sign-on; special AI-talent stock awards are a reported factor.

---

## How to walk into any of these

<details class="qa"><summary>They ask: "Why us, specifically?" — how do I not sound generic?</summary>
<div class="qa-body">

**Short:** Name a *specific* released model/paper from that org, one honest sentence on why it impressed you, and the lever your work pulls on it.

**Deep:** "I want to work on multimodal models" fits everyone and signals nobody. Contrast: *"SAM's promptable interface is what ZIM extended for matting quality, so the SAM-3 concept-segmentation direction is exactly where my region-level work would contribute."* That sentence proves you read their work, know your own, and see the join. Build one per company — see [Recruiter & HM Screens](#/process/recruiter-hm).
</div></details>

<details class="qa"><summary>Which company is the "safe" first interview and which is the reach?</summary>
<div class="qa-body">

**Short:** Sequence by *your* fit, not prestige. For a CV/VLM candidate with a shipped-product record, **Adobe** (ZIM↔editing) and **Apple** (on-device efficiency) are natural-fit warm-ups; **ByteDance** (Hard coding) and **Mistral** (from-scratch transformer take-home) are the ones whose *format* most needs dedicated prep.

**Deep:** "Reach" here is about *format risk*, not whether you're good enough. Your research depth is strong everywhere; the differentiated risk is Meta/ByteDance coding bars and Mistral's linked take-home+restitution. Warm up where your story lands cleanly, then hit the format-heavy loops once your coding is grooved. See the [pipeline](#/process/pipeline) on clustering timelines.
</div></details>

### Follow-ups that recur across companies

- *"What's a limitation of your own best paper?"* — every research org asks a version. Have the honest failure mode of ZIM (or your flagship) ready. See [Failure & Negative Results](#/research/failure).
- *"How would this scale to our data/compute?"* — connect your method to distributed training + serving realities. See [Design Framework](#/system-design/framework).
- *"How do you decide what to work on next?"* — research taste. Tie label-efficient/continual → grounded VLM → agents as a *deliberate* trajectory, not drift.

## Cheat-sheet

| Company | One-line prep priority |
| --- | --- |
| Meta / FAIR | Near-SWE coding + job talk; clarify FAIR vs GenAI; lead with grounding + shipped ZIM |
| NVIDIA | Job talk + DL-depth; efficiency-as-science; intellectual honesty; CUDA willingness |
| Apple | On-device/privacy framing; NMS/IoU/K-means from scratch; don't speculate on products |
| Adobe | Research talk (ZIM) + rigorous PyTorch coding; diffusion math; product sense |
| ByteDance Seed | **Hard LeetCode is the gate**; discriminative→generative story; speed/OKR culture |
| Mistral | From-scratch transformer for the take-home; MoE opinion; "why Mistral not US labs"; clean typed code |
| Microsoft MSR | Central job talk + AA round; grounding-as-agent-perception; growth mindset stories |

**Related:** [The RS/AS Pipeline](#/process/pipeline) · [Recruiter & HM Screens](#/process/recruiter-hm) · [Offers & Negotiation](#/process/negotiation) · [The Research Job Talk](#/research/job-talk) · [Coding Round Strategy](#/coding/strategy) · [Design Framework](#/system-design/framework) · [The 2026 Landscape](#/start/landscape-2026)
