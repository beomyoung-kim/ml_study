# Predicted Questions & Answers

<div class="tag-row"><span class="tag">research narrative</span><span class="tag">project probes</span><span class="tag">research→product</span><span class="tag">motivation</span><span class="tag">curveballs</span><span class="tag badge-new">flagship</span></div>

> [!TIP] Why this chapter matters
> This is the chapter that wins loops. Interviewers rarely quiz you on your CV to check facts — they use it to test whether you can **narrate a research arc**, **own a specific contribution**, and **defend choices under pressure**. Below is a bank of the questions most likely to come from *this* CV, each with the interviewer's hidden intent, **2–3 distinct model answers by angle**, a probable **follow-up**, and a one-line **rebuttal tip**. Internalize the *structure and the trade-offs*, not the sentences.

## How to use these answers

<div class="proscons"><div><div class="pros-t">Do</div>
<b>Internalize, don't memorize.</b> Learn the <i>spine</i> of each answer — the claim, the evidence, the trade-off — then rebuild it live in your own words. A recited answer dies on the first follow-up; a understood one survives ten.
<br><br>
<b>Pick the angle to the room.</b> Each question offers a <b>technical-depth</b>, an <b>impact/product</b>, and a <b>concise/executive</b> variant. Read the interviewer: a FAIR panel wants depth and defensibility; a hiring manager wants impact and fit; a recruiter wants the 30-second version.
</div><div><div class="cons-t">Don't</div>
<b>Don't overclaim.</b> For confidential work (FaceSign, the foreground API, ongoing/under-review), speak to <b>framing and impact</b>, never invented numbers: <i>"I can share the framing; the specific internal metrics are confidential."</i> That line reads as integrity, not evasion.
<br><br>
<b>Don't blur "I" and "we".</b> Credit collaborators, but make <i>your</i> contribution unambiguous — the single most common reason research-track candidates get dinged.
</div></div>

**Tailor per company** using the [Company Playbooks](#/process/companies): Meta wants publish-and-ship ownership; NVIDIA rewards systems/GPU fluency and intellectual honesty; Apple probes on-device/privacy constraints; Adobe wants PyTorch rigor + product sense; ByteDance folds it into hard coding rounds; Mistral asks "why us, not a US lab?"; Microsoft MSR centers the job talk. Every answer below can be re-pointed at one of those.

Deep-dives for the raw material: [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [FaceSign](#/resume/facesign) · [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM/Agents](#/resume/grounded-vlm-agents). Behavioral framing: [STAR & The Story Bank](#/behavioral/star). Map: [Your CV → Interview Map](#/resume/overview).

---

## A · Research narrative & motivation

The opener of almost every research-track loop. They are testing whether there's a *thesis* to your career or just a list of papers.

<details class="qa"><summary>"Walk me through your research."</summary>
<div class="qa-body">

**Testing:** can you compress 14+ papers into a coherent arc with a through-line, at the right altitude, in ~2–3 minutes — while making *your* role clear.

**Answer A — the through-line (technical narrative).**
"There's one thread: **making visual perception accurate and trustworthy under real-world constraints.** I started in **weakly/semi-supervised segmentation** — DRS, BESTIE, PointWSSIS — because the bottleneck to deploying perception is labels, so I attacked the label-efficiency frontier: how to get instance masks from image tags or a few points. Then **continual learning** — SSUL and ECLIPSE — because deployed models must absorb new classes without forgetting or storing old data. Then I scaled up to **foundation models** with ZIM, a promptable zero-shot matting model built on SAM. Across all of it the pattern is the same: *define the real bottleneck, then solve it with a data + architecture co-design rather than brute force.* Most recently I've moved up the stack to **grounded VLMs and training-free visual reasoning agents** — connecting language reasoning to pixel- and region-level evidence, and diagnosing when perception tools silently fail. So: pixels → labels → continual → foundation → grounded reasoning. Same obsession, higher abstraction."

**Answer B — impact-first (for a hiring manager).**
"Two-sentence version: I build perception that ships. Seven first-author papers at CVPR/ICCV/ECCV/NeurIPS, an ICCV 2025 Highlight for ZIM, and a consistent record of carrying that research into NAVER Cloud products — a foreground-segmentation API that beat commercial alternatives in internal eval, the anti-spoofing model behind a government-certified face-auth service, and an on-device segmenter running at ~10 ms. The research and the product feed each other: production tells me which problems are real, papers make the solutions rigorous."

**Answer C — concise/executive (30 sec).**
"I'm a computer-vision applied scientist. My arc goes from label-efficient and continual segmentation, to a segmentation/matting foundation model (ZIM, ICCV 2025 Highlight), and now to grounded vision-language models and visual reasoning agents. The constant is perception that's accurate, efficient, and verifiable enough to put in front of millions of users — which I've done at NAVER Cloud."

**Follow-up:** "Which of these do you consider your best work, and why?"
**Rebuttal tip:** Name ZIM for *reach* (foundation + Highlight + product), or PointWSSIS for *elegance* (redefining the proposal bottleneck) — but justify by **the idea**, not the venue. "Best" = clearest problem redefinition, not highest citation count.

</div></details>

<details class="qa"><summary>"What's the thread connecting your papers? Or is it just whatever your team needed?"</summary>
<div class="qa-body">

**Testing:** research *taste* — do you choose problems, or do problems choose you? The cynical framing is bait; don't get defensive.

**Answer A — the intellectual thread.**
"The thread is **cheap or weak signals → reliable structured perception.** DRS and BESTIE: image tags → instance masks. PointWSSIS: a few points → full masks. ECLIPSE: a frozen model + tiny prompts → new classes without forgetting. ZIM: coarse SA-1B labels → fine alpha mattes via a label converter. Even the agents work is the same move at the top of the stack: noisy specialist tool outputs → a verifiable, self-diagnosing reasoning program. I'm consistently drawn to the gap between *what supervision you can afford* and *what fidelity the product needs*, and I close it with data + architecture design."

**Answer B — honest about the environment (owns the constraint).**
"Both, honestly, and I think that's a strength. Working at NAVER Cloud meant real products surfaced real bottlenecks — labeling cost, category drift, latency, boundary quality. But *which* bottleneck to elevate into a paper, and *how* to frame it so it generalizes beyond our product, was my call. PointWSSIS came from a labeling-budget pain point, but the contribution — that the proposal stage, not the mask head, is the true bottleneck in semi-supervised instance segmentation — is a general insight. That's the taste part."

**Follow-up:** "So what problem would you pick next, with total freedom?"
**Rebuttal tip:** Have a real answer ready — "verifiable spatial reasoning: perception tools that report their own uncertainty so an agent knows when to distrust them." It proves the thread is forward-looking, not a retrofit.

</div></details>

<details class="qa"><summary>"Why the pivot from perception to grounded VLMs and agents? Isn't that chasing the trend?"</summary>
<div class="qa-body">

**Testing:** whether the move is opportunistic (following LLM hype) or a principled continuation. This is *the* narrative question for 2025–26 vision hires.

**Answer A — continuation, not pivot (depth).**
"It's not a pivot away from perception — it's perception finding its role in the new stack. End-to-end VLMs are strong but they **hallucinate** and produce *unsupported textual descriptions*: the answer can be right while the visual evidence is wrong — spurious success. My whole background is high-fidelity, region- and pixel-level perception. That's exactly the missing ingredient: grounding language reasoning in verifiable visual evidence, and giving agents *specialist tools* (like ZIM-quality masks or depth) so they can reason about 3D and time. The interesting new problem is that those tools **fail silently** — they return a plausible-but-wrong box or mask with no exception — so my current work turns those silent failures into *typed diagnoses* that drive targeted program repair. I'm not leaving perception; I'm making it the trustworthy tool layer under reasoning."

**Answer B — market/product logic (impact).**
"Products moved. Image editing, robotics, UI agents — they all need to *resolve a region from a language query* and act on it verifiably. A pure end-to-end model that describes an image isn't enough when a user says 'remove the person on the left' or a robot must reason about which object is behind another. My perception work is the substrate for that. The pivot is following where the value is, and I happen to have the exact background the value requires."

**Answer C — concise.**
"Because the frontier moved from 'can a model perceive?' to 'can a model reason over perception without lying about it?' Grounding and agentic tool-use are how you answer that, and my perception depth is the differentiator, not a sunk cost."

**Follow-up:** "If frontier VLMs keep improving, doesn't your modular/agent approach become obsolete?"
**Rebuttal tip:** Concede they're strong, then name the residual: **precise measurement, verifiable evidence, upgrading a module without retraining the whole model, and diagnostic repair.** Hybrid is the practical default even as end-to-end improves — that's your defensible position, not a bet against scale.

</div></details>

<details class="qa"><summary>"You did a PhD while working full-time. Doesn't that mean your research is a side project?"</summary>
<div class="qa-body">

**Testing:** rigor and independence — is the industry job the "real" work and the PhD a formality, or are they genuinely one program?

**Answer A — integration (reframe).**
"They're the same program, which is unusual and I think it's my edge. My PhD problems *come from* production — PointWSSIS from labeling cost, ECLIPSE from category drift in a live API, ZIM from an editing product needing fine masks. So my papers are pre-validated against reality, and my products get academic rigor: ablations, baselines, honest failure analysis. Seven first-author top-venue papers and an ICCV Highlight *while shipping* is the proof it's not a side project — most people do one or the other."

**Answer B — independence signal.**
"I drive the research. I'm first author on seven papers, I set the problem framing, ran the experiments, wrote the code that's open-sourced and adopted. Prof. Sung Ju Hwang at KAIST MLAI advises the direction and sharpens the framing; the execution and the specific contributions are mine. The full-time role gives me data and product context academia rarely has."

**Follow-up:** "How do you manage the time and the two sets of priorities?"
**Rebuttal tip:** Don't romanticize the hustle. Frame it as *alignment* — you chose problems where the product need and the publishable question overlap, so you're not paying the cost twice. That's judgment, not just stamina.

</div></details>

---

## B · Per-project probes

Expect the interviewer to pick one project — usually ZIM or whichever matches their team — and drill until you either show ownership or crack. Full technical depth lives in the deep-dives; here are the *entry* probes and how to answer at interview altitude.

<details class="qa"><summary>ZIM — "Why couldn't you just fine-tune SAM on matting data?"</summary>
<div class="qa-body">

**Testing:** do you understand *why* the naive baseline fails — the core motivation of your own Highlight paper. Fumbling here is fatal.

**Answer A — the failure mode (technical).**
"We tried; it collapses. Public matting data is dominated by **macro** targets — full portraits, salient whole objects. Fine-tune SAM on that and it *forgets its micro granularity*: give it a point prompt on a strand of hair or a small part and it regurgitates the whole person. You trade away the zero-shot promptability that made SAM valuable. Our ablation makes it stark — same architecture trained on public mattes gives fine-grained SAD around 120 versus about 10 for our converted data. **Data granularity dominates architecture.** So instead of fine-tuning on the wrong distribution, we built a **label converter** that turns SA-1B's ~1B coarse masks into fine alpha mattes, with Spatial Generalization Augmentation and Selective Transformation Learning to control noise — then trained on that."

**Answer B — the design principle (concise).**
"Because the problem isn't the loss, it's the data contract. SAM's supervision is coarse and hard-masked; matting needs soft, micro-level boundaries. Fine-tuning on macro portrait data destroys the promptable zero-shot behavior. We generated micro-granular matte labels at SA-1B scale via a converter, plus a hierarchical pixel decoder and prompt-aware masked attention to actually render fine detail. That's why it stays zero-shot *and* produces mattes."

**Follow-up:** "What does the hierarchical pixel decoder buy you, concretely?"
**Rebuttal tip:** SAM's stride-4 decoder + naive upsample causes **checkerboard artifacts** and loses fine structure; the multi-level (stride 2/4/8) fusion recovers high-res detail for ~10 ms overhead. Have that number and the mechanism ready — see [ZIM](#/resume/zim).

</details>
</div>

<details class="qa"><summary>ZIM — "Prompt-aware masked attention: where do you apply it, and why not everywhere?"</summary>
<div class="qa-body">

**Testing:** precision on your own architectural detail — a favorite way to catch "we"-authors who didn't do the work.

**Answer A — the mechanism.**
"Only on the **token-to-image (T2I)** cross-attention. A box prompt becomes a binary mask that sets attention logits outside the box to −∞; a point becomes a Gaussian (σ≈21) that modulates the QK scores. The intuition: I want the *output tokens* to attend to the prompted region. Crucially I do **not** apply it to image-to-token attention — forcing that corrupts the global image features and hurts performance. That asymmetry is the non-obvious part; we validated it by ablation."

**Answer B — concise.**
"On T2I attention: box → hard mask, point → Gaussian. It focuses the decoder on the prompt region without damaging the global features, which is why applying it to the I2T direction backfires."

**Follow-up:** "Why a Gaussian for points instead of a hard disk?"
**Rebuttal tip:** Soft falloff tolerates prompt-location ambiguity and gives smoother gradients near boundaries — matting is a soft-label task, so a hard disk would fight the objective.

</div></details>

<details class="qa"><summary>ECLIPSE — "Freezing the whole backbone to avoid forgetting — isn't that giving up on learning new classes?"</summary>
<div class="qa-body">

**Testing:** the stability–plasticity trade-off, and whether you can defend an extreme design choice.

**Answer A — the trade + the levers (depth).**
"Freezing Mask2Former after step 1 makes forgetting *structurally impossible* — old weights never move — which is why we beat distillation-based methods like CoMFormer on all-PQ over long class sequences, with only ~1.3% trainable parameters and no need to store old data. The cost is exactly what you'd fear: **error propagation** (a step-1 mistake is frozen in) and limited plasticity. We recover plasticity two ways: **deep visual prompts** at every transformer layer (deep vs shallow lifts new-class PQ meaningfully for ~100K params), and a stronger frozen init (Swin-L / COCO-pretrained) which raises the ceiling. And **logit manipulation** handles the no-obj semantic drift — the meaning of 'background' changes every step, so a fixed no-obj classifier is untrustworthy; we recompute it from the other classes' logits post-hoc. So it's not 'give up on new classes,' it's 'buy stability for free, then spend a little to rent back plasticity.'"

**Answer B — product framing (impact).**
"In a live segmentation API, you add categories constantly and you often *can't* keep old training data for privacy/storage reasons. ECLIPSE lets you ship a small adapter per new category batch instead of retraining a 45M-param model or replaying old data. That maps directly to on-device or privacy-constrained update stories — Apple-flavored."

**Follow-up:** "Why panoptic, not just semantic continual learning?"
**Rebuttal tip:** Panoptic is strictly harder — instance matching *and* stuff *and* the shifting no-obj definition all at once, and PQ punishes recognition failures (RQ). Continual panoptic was under-studied vs semantic; that gap is part of the contribution. See [ECLIPSE](#/resume/eclipse).

</div></details>

<details class="qa"><summary>PointWSSIS — "What's the actual insight? It sounds like just another semi-supervised trick."</summary>
<div class="qa-body">

**Testing:** whether you redefined a problem or just tuned one — the difference between a strong CVPR paper and an incremental one.

**Answer A — the redefinition (depth).**
"The insight is that in instance segmentation the bottleneck isn't the mask head, it's the **proposal stage**. Semi-supervised methods fight a false-negative / false-positive trade-off on pseudo-labels: lower the confidence threshold and you drown in false positives, raise it and you erase real instances — because *if there's no proposal, an expressive mask branch still outputs nothing.* So I changed the supervision: a few full masks plus many cheap **point** labels, where the point directly resolves the proposal ambiguity — it filters teacher proposals down to true positives. Then Adaptive Pyramid-Level Selection handles the missing scale information a point doesn't carry, and MaskRefineNet cleans rough masks when full labels are scarce. Result: COCO at 5% full labels goes from ~24.9 AP (prior SSIS) to 33.7, and at 50% we nearly match fully-supervised. The move is *relocating the bottleneck*, not tuning a threshold."

**Answer B — the lineage (concise).**
"BESTIE showed you can push image-tags → instances but you're capped by pseudo-label drift. PointWSSIS asks: what's the cheapest signal that removes the *proposal* bottleneck specifically? A point. It's a targeted intervention on the real bottleneck, which is why the gap at low label budgets is large, not marginal."

**Follow-up:** "Does the point have to be the centroid?"
**Rebuttal tip:** No — random-in-mask works about as well as centroid (SOLOv2 tolerates diverse in-region points), which makes the labeling even cheaper. That robustness is itself an ablation-backed selling point. See [PointWSSIS & BESTIE](#/resume/pointwssis-bestie).

</div></details>

<details class="qa"><summary>BESTIE — "You still use CAMs and pseudo-labels. How do you handle the noise?"</summary>
<div class="qa-body">

**Testing:** command of the weakly-supervised failure mode (semantic drift) and your specific fix.

**Answer A — the drift + fixes.**
"The killer failure is **semantic drift**: a missed instance gets learned as background while a visually identical instance is foreground, so the model learns a contradiction and degrades. I attack it three ways. **Instance-aware guidance** only applies the center/offset losses on labeled-instance regions, so background regions don't poison the signal. **Self-refinement** runs online per mini-batch — the network's own grouped output is fed back as soft-weighted refined labels, promoting false-negatives to true-positives over training. And **PAM** (Peak Attention Module) sharpens genuine instance cues while suppressing CAM noise — the opposite direction from DRS, which *spreads* activation for semantic coverage. Same author, opposite tool for the opposite need."

**Answer B — concise.**
"Semantic drift from missing pseudo-instances is the enemy. I mask the loss to trusted regions, refine labels online so misses get recovered, and use peak-based cues instead of raw noisy CAMs. The WSSS quality caps the ceiling, but WSIS degrades gracefully as WSSS drops — I show that sensitivity."

**Follow-up:** "How is PAM different from your earlier DRS work?"
**Rebuttal tip:** DRS *suppresses* discriminative regions to expand coverage for semantic segmentation; PAM *amplifies* peaks to localize instances. Framing them as two sides of the same CAM-shaping insight tells a strong research-taste story.

</div></details>

<details class="qa"><summary>DRS / SSUL — "These are older papers. Are they still relevant, or just early-career warm-ups?"</summary>
<div class="qa-body">

**Testing:** whether you can extract durable principles from early work rather than dismiss it.

**Answer A — principles that carried forward.**
"They're the roots of everything after. **DRS** (AAAI 2021) is about *shaping* CAM attention — suppressing the discriminative region so activation spreads to the whole object. That CAM-shaping intuition became PAM in BESTIE. **SSUL** (NeurIPS 2021, co-first) introduced modeling the **unknown/future class** explicitly in class-incremental semantic segmentation plus exemplar memory — that 'the background's meaning is unstable across steps' insight is *exactly* what ECLIPSE's logit manipulation later solved for panoptic. So they're not warm-ups; they're where my two long-running threads — CAM manipulation and continual segmentation — were seeded."

**Answer B — concise.**
"Every later paper is traceable to these two: DRS → attention shaping → BESTIE; SSUL → unstable background → ECLIPSE. Early, but load-bearing."

**Follow-up:** "SSUL was co-first-author — what was *your* part?"
**Rebuttal tip:** Be specific and generous: name the piece you drove (the segmentation modeling / unknown-class handling) and credit the co-author's part (the incremental-learning framing) without diluting your ownership. Vague "we did it together" reads as a red flag on a co-authored paper.

</div></details>

<details class="qa"><summary>FaceSign — "Tell me about the anti-spoofing model. What accuracy did you hit?"</summary>
<div class="qa-body">

**Testing:** two things at once — do you understand face anti-spoofing deeply, *and* do you handle a confidentiality boundary gracefully (a real signal for Apple/regulated roles).

**Answer A — decline-then-deliver (the model to imitate).**
"The specific accuracy, attack-set composition, and architecture are confidential — it's a government-certified payment and badge-auth service, so I'm contractually and ethically bound there. What I *can* give you is the framing. Anti-spoofing sits between capture and recognition: detect/align → **liveness / presentation-attack detection** → recognize. If a presentation attack — print, replay, 3D mask, and separately digital injection — gets through, it doesn't matter how good recognition is; the system fails. The hard part isn't the easy print attack, it's **domain generalization**: unseen phones, lighting, demographics, and novel attack instruments. We evaluate with the APCER/BPCER/ACER frame (ISO/IEC 30107) and choose an operating point against the recognition FRR based on the required security level. False rejects cost you conversions, so step-up fallback (password) matters as much as the detector."

**Answer B — the systems view (concise).**
"I owned the anti-spoofing component. I can't share numbers on a certified security product, but I can talk about the threat model, the sensor trade-offs (RGB-only vs depth/IR), the evaluation frame, and the operating-point trade-off between security and approval rate. It's a *system* problem — sensing + model + risk policy — not just a classifier."

**Follow-up:** "How would you compare it to Apple Face ID?"
**Rebuttal tip:** Don't invent a comparison. Face ID is famous for *hardware co-design* (structured-light dot projector); your work is service-side model + operations. Shared DNA: "sensing + model + policy as one system." No head-to-head numbers exist — say so. See [FaceSign](#/resume/facesign).

</div></details>

<details class="qa"><summary>On-device seg — "Why is 10 ms the target? Why not just use a bigger model on the server?"</summary>
<div class="qa-body">

**Testing:** systems/efficiency reasoning and whether you understand *why* on-device matters — not just that you hit a number.

**Answer A — frame-budget engineering (depth).**
"10 ms isn't a magic number, it's a **frame-budget** result. Real-time UX is ~30 fps ≈ 33 ms/frame, and segmentation shares that budget with camera preprocessing, post-processing, other models, and the UI. Give segmentation ~10 ms and the whole pipeline holds; give it 20 and you drop frames. The reason it's *on mobile CPU* rather than server is privacy (frames never leave the device) and cost (no per-inference server bill), plus offline capability. I designed for the worst common denominator — CPU, not a specific NPU — so it runs everywhere. I got there by cutting the biggest levers first: input resolution, channel width, decoder depth; then distillation from a heavy teacher to keep boundary quality; QAT last, because quantizing first kills hair and fine boundaries before anything else."

**Answer B — the two-tier story (impact, Apple-flavored).**
"It's the on-device half of a two-tier strategy: a heavy cloud foundation model (ZIM-class) provides quality and acts as a distillation teacher; a tiny distilled + quantized student runs on-device at ~10 ms for the frame budget and privacy. That 'cloud foundation + on-device specialist' split is exactly the pattern modern products need, and I've built both ends of it."

**Follow-up:** "You said ONNX serving — is it really on the phone CPU or on a server?"
**Rebuttal tip:** Be precise, because the CV wording invites this: the *model* was designed and measured for a mobile-CPU budget; the *serving path* was ONNX-based in-house infra. Distinguish design target from deployment infra rather than overclaiming a phone deployment you can't detail. See [On-Device Seg](#/resume/on-device-segmentation).

</div></details>

<details class="qa"><summary>Ongoing agents — "Your NeurIPS submission is under review. Walk me through it."</summary>
<div class="qa-body">

**Testing:** can you present unpublished work with real substance while respecting the "don't overclaim / don't leak" line — and without sounding vaporous.

**Answer A — problem-first framing (the safe, strong version).**
"It's under review, so I'll give you the problem and the framing rather than final numbers or claim acceptance. The setting is **visual program synthesis** for 3D spatial reasoning — an LLM writes an executable program that calls specialist vision tools (detection, depth, segmentation). The failure I care about is **silent perception failure**: a tool returns a plausible-but-wrong output — a bad box, a wrong depth — and because there's no exception, the program runs to completion and emits a *confidently wrong* answer. My framework turns those silent failures into **typed diagnoses**, and uses the diagnosis to drive **targeted program repair**. The goal is to rival frontier VLMs on 3D spatial reasoning **without task-specific training**. The public lineage is VisProg / ViperGPT (program synthesis) and VADAR (dynamic API generation for 3D); my angle is the diagnose-and-repair loop on silent failures."

**Answer B — why-it-matters (concise).**
"Agents that call vision tools break not because the LLM plans badly but because the *tools* lie quietly. I make those silent failures observable and repairable, targeting frontier-level 3D spatial reasoning training-free. Details and numbers are pre-publication."

**Follow-up:** "How do you evaluate it?"
**Rebuttal tip:** Speak to the *philosophy* using public benchmarks (Omni3D-Bench, Spatial457, SpatialAct) and the right metrics — answer accuracy *plus* grounding IoU, program executability, and failure-type recall — without naming any internal bench. "The point is measuring evidence, not just the final answer." See [Grounded VLM/Agents](#/resume/grounded-vlm-agents).

</div></details>

<details class="qa"><summary>Grounded VLMs — "How does grounding actually reduce hallucination?"</summary>
<div class="qa-body">

**Testing:** conceptual depth on the current frontier, beyond buzzwords.

**Answer A — the mechanism.**
"Hallucination in VLMs is largely *unsupported generation* — the language head produces a fluent claim with no obligation to point at evidence. Grounding changes the contract: the model must **resolve a region or pixels** for the claim, so the answer is tied to a verifiable location. Two benefits. First, you can *check* it — grounding IoU / pointing accuracy exposes 'right answer, wrong evidence' (spurious success) that pure text metrics hide. Second, you can force the reasoning to *operate on* the evidence — pixel-grounded chain-of-thought, crop-and-zoom loops (grounded RL like ViGoRL/MGPO), mask operations mid-reasoning. It doesn't eliminate hallucination, but it makes the model's claims falsifiable, which is the precondition for trust in editing, robotics, and UI agents."

**Answer B — concise.**
"By requiring the model to *point*, not just *say*. If every claim has to resolve to a region, you can verify it and you can catch spurious successes — answers that are right for the wrong visual reason."

**Follow-up:** "Isn't that just adding a detector? Where's the research?"
**Rebuttal tip:** A detector gives boxes; the research is making reasoning *consume and verify* those boxes end-to-end, handling when they're wrong, and doing region-query resolution the detector alone can't. Grounding is a reasoning contract, not a bolt-on box.

</div></details>

---

## C · Research → product & impact

The Applied-Scientist differentiator. RS candidates who can't ship get filtered to engineering tracks; you have the opposite problem's cure — prove the shipping is real and rigorous.

<details class="qa"><summary>"Your foreground-segmentation API beat Photoroom, Remove.bg, and Adobe. Prove it — how did you measure that?"</summary>
<div class="qa-body">

**Testing:** rigor and honesty behind a big claim. A bold benchmark claim with no methodology is a red flag; with methodology it's a green one.

**Answer A — methodology-first (rigor).**
"It was an **internal evaluation**, so I'll be precise about scope: I can share the *framing*, not the confidential per-competitor numbers. We built an evaluation set representative of our product's real inputs — the failure-prone cases matter most: hair, fine boundaries, semi-transparency, multi-person, hard lighting. We scored with boundary-focused metrics (boundary F-score, SAD-style boundary error) plus human visual review, because mIoU alone hides exactly the boundary quality customers notice. Against the same inputs, our model + data pipeline came out ahead of those commercial APIs on *our* distribution. The honest caveat is it's our test set and our domain — I wouldn't claim universal superiority, and I'd say that in the room."

**Answer B — the moat (product).**
"The win wasn't a bigger model, it was **data**. The differentiator was curating boundary-hard training data matched to our product distribution and a pipeline tuned for the boundaries users actually see. Off-the-shelf APIs optimize for a generic distribution; we optimized for ours. That's a defensible, repeatable advantage, and it's the same data-first lesson as ZIM — the data contract decides quality."

**Follow-up:** "Is this the same model as ZIM?"
**Rebuttal tip:** No — don't conflate them. The API is a product line (internal serving); ZIM is a promptable zero-shot foundation model. Shared DNA (boundary quality, data curation), different artifacts. Claiming they're one model is the kind of overclaim that unravels under follow-up.

</div></details>

<details class="qa"><summary>"You got ZIM into CLOVA-X Image Editing. What broke when research met production?"</summary>
<div class="qa-body">

**Testing:** whether you've actually shipped research, or just tossed a checkpoint over the wall. The interesting answer is the friction.

**Answer A — the handoff frictions (systems).**
"The public detail I can share is that it shipped and I presented it at TEAM NAVER DAN 24; the exact SLAs are internal. The real work was the *handoff*, and that's where research-to-product usually breaks: (1) **representation mismatch** — the editor expects a specific alpha format (premultiplied vs not, color spill handling), not just a raw matte; (2) **latency** — a ViT-B matte model runs on the order of ~180 ms on a V100, which is fine for a batch tool but must be managed for interactive UX via resolution control, model tiering, and caching; (3) **failure monitoring** — hair and semi-transparency are where mattes fail, so you need a fallback (e.g. hard-threshold) and monitoring; (4) **API compatibility** — existing clients expected hard masks, so the soft-matte output had to slot in without breaking them. Shipping is 20% model, 80% these contracts."

**Answer B — concise (impact).**
"I open-sourced ZIM with a demo and integrated it into CLOVA-X image editing. The lesson: the model was the easy part. Format handoff to the editor, interactive-latency budgeting, and failure fallbacks for hair/transparency were where the real engineering went. Internal SLAs are confidential, but the framing is standard research-to-product hardening."

**Follow-up:** "How would you extend it to video editing?"
**Rebuttal tip:** SAM2-style memory + temporal consistency is the direction, and it's genuinely unsolved for matting — say so. Flickering alpha across frames is the hard part; don't pretend it's a trivial extension.

</div></details>

<details class="qa"><summary>"You open-sourced most of your work. Doesn't that give away NAVER's advantage?"</summary>
<div class="qa-body">

**Testing:** product/business judgment and whether you understand *why* a company publishes.

**Answer A — the strategic case.**
"The published methods aren't the moat; the *data operations, integration, latency engineering, and domain adaptation* are. Open-sourcing ZIM, ECLIPSE, PointWSSIS, BESTIE bought us research brand, recruiting, and an ecosystem — Grounded-ZIM and downstream adopters (Inpaint-Anything, medical, 3D) came *because* it was open, which validates and extends the work for free. Meanwhile the foreground API and FaceSign — the things with direct commercial or security value — stayed internal. So the split is deliberate: publish the general method, protect the product-specific data and operations."

**Answer B — concise.**
"Because the paper isn't the product. The advantage is in the data pipeline and the integration, which we don't publish. Open-sourcing the method is net-positive for brand, recruiting, and ecosystem — that's why the confidential things (the API, FaceSign) aren't open."

**Follow-up:** "So what *would* you keep closed here, if we hired you?"
**Rebuttal tip:** Show you can reason about *their* IP, not just recite NAVER's policy — "product-specific data, eval sets tied to real customers, and anything safety/abuse-related; general methods I'd push to publish." That's the maturity they're checking for.

</div></details>

---

## D · Career, motivation & fit

Recruiter and HM territory. These decide *fit*, and sloppy answers end loops even when the technical rounds are strong. Keep them honest, specific, and non-defensive.

<details class="qa"><summary>"Why leave NAVER after 5+ years?"</summary>
<div class="qa-body">

**Testing:** are you running *toward* something (good) or *away* from something (risk)? Bitterness or vagueness both hurt.

**Answer A — toward the frontier (positive pull).**
"Five years at NAVER Cloud gave me something rare — a full research-to-product loop, from foundation models to APIs used by millions. I'm proud of it and I'm not leaving unhappy. I'm moving because the problems I now care most about — grounded, verifiable multimodal reasoning and visual agents at frontier scale — are where [company] is pushing hardest, with the compute, the multimodal data, and the peers to do it at a level I can't reach alone. I want to go from 'strong regional product research' to 'global frontier research,' and bring my perception-to-product discipline into that environment."

**Answer B — scope and scale (concise).**
"I've maxed out the loop I can run inside one org. I want frontier-scale multimodal problems, a bigger research community around me, and the chance to work on grounded reasoning at a scale that needs the infrastructure [company] has. It's a pull toward the work, not a push away from NAVER."

**Follow-up:** "What specifically can't you do at NAVER that you could here?"
**Rebuttal tip:** Be concrete and flattering-but-true: scale of multimodal pretraining, a specific team/agenda (name it — SAM/perception at Meta, physical-AI at NVIDIA, on-device foundation models at Apple, Firefly at Adobe), and the peer density. Avoid anything that sounds like a complaint about NAVER.

</div></details>

<details class="qa"><summary>"Why this company specifically?" (per-company)</summary>
<div class="qa-body">

**Testing:** homework and genuine fit. Generic answers are lethal here. Use the [Company Playbooks](#/process/companies).

**Meta (FAIR/VLM):** "My perception-to-grounding thread is the missing piece for product VLMs — SAM lineage is literally my starting point for ZIM, and grounded, verifiable reasoning is FAIR-adjacent. I publish *and* ship, which is the FAIR/Applied bar."

**NVIDIA Research:** "Spatial reasoning and perception tools are the substrate for physical AI and robotics (GEAR/GR00T), and I care about efficient, deployable stacks — my on-device and ONNX work shows the systems side, and I'd lean into the GPU/systems fluency your labs value."

**Apple (MLR / Multimodal):** "My two-tier story is your story: cloud foundation quality (ZIM) + on-device specialists at a real frame budget (~10 ms), under privacy and hardware constraints. I've shipped perception under exactly the accuracy-vs-latency-vs-battery trade-offs you optimize for, and I'm comfortable working without external visibility (FaceSign)."

**Adobe (Firefly/Research):** "Grounded, region-controllable editing is precisely where ZIM and grounding meet Firefly — precise masks for generative fill, region-grounded edit programs. I have the PyTorch rigor and the creative-product sense the role needs."

**ByteDance Seed:** "Vision generative foundation + controllable perception is where my matting/segmentation foundation work plugs in — SeedEdit/Seedream-style editing needs exactly the boundary and region quality I build, and I move fast."

**Mistral AI:** "I value the open-weight conviction — I open-sourced most of my work for the same reasons. My contribution is bringing rigorous multimodal/perception grounding into a lab that's mostly LLM-centric, and I'm comfortable with the take-home-plus-defense format because defending my own work is what I do."

**Microsoft MSR:** "MSR's job-talk, agenda-driven model fits how I already work — I've run a coherent multi-year agenda (label-efficient → continual → foundation → grounded) while collaborating across academia and product. I'd bring a perception-grounding agenda and strong collaboration signal."

**Follow-up:** "What would you want to work on in your first year?"
**Rebuttal tip:** Name a real project on *their* roadmap that your skills uniquely serve — not a generic "learn the codebase." Specificity proves the fit is real.

</div></details>

<details class="qa"><summary>"You're in Seoul. Are you actually willing to relocate?" / visa & logistics"</summary>
<div class="qa-body">

**Testing:** logistical seriousness — recruiters kill candidates who wobble here late.

**Answer A — clear commitment.**
"Yes — my CV says open to relocation and I mean it. I'm happy to discuss timing, visa sponsorship, and the KAIST PhD logistics openly; I've structured the PhD to be compatible with a move, and I'd want to align start dates so nothing is left ambiguous. If the role is remote-friendly or hybrid, I'm flexible on that too."

**Answer B — concise.**
"Fully willing to relocate; I'd just want to coordinate visa timeline and my PhD completion so we set a realistic start date. No hesitation on the move itself."

**Follow-up:** "Does the PhD need to finish first?"
**Rebuttal tip:** Have a real plan — remote completion, a defense timeline, or advisor flexibility. "It's in parallel and I've planned for continuity" beats a vague "we'll figure it out."

</div></details>

<details class="qa"><summary>"Do you see yourself as a researcher or an engineer?"</summary>
<div class="qa-body">

**Testing:** self-awareness and fit to the specific track (RS vs AS vs MLE). The wrong self-label mismatches you to the loop.

**Answer A — the applied-scientist identity.**
"Applied scientist, genuinely both. I find the accuracy–cost Pareto with real experiments — ablations, baselines, honest failure analysis — *and* I close the last mile: ONNX export issues, latency budgets, production handoffs. My papers come from product bottlenecks and my products carry academic rigor. If your role is pure long-horizon RS I can do the research depth; if it's AS I'm already living that hybrid. I'd rather be judged on 'ships rigorous research' than on a title."

**Answer B — concise.**
"Both, by design. Seven first-author papers say researcher; a shipped foundation model, a production API, and a certified auth component say engineer. Applied science is the intersection, and it's deliberate."

**Follow-up:** "Which do you enjoy more?"
**Rebuttal tip:** Honest but role-aware — lean research for an RS panel, lean shipping for AS/MLE, but always tie back to "the loop between them is what I actually love." Don't disown either half.

</div></details>

---

## E · Hard, curveball & weakness probes

The rounds that separate offers from near-misses. The meta-rule: **name the weakness before they do, then show the compensating strength or the plan.** Defensiveness is the only fatal answer.

<details class="qa"><summary>"Your work looks incremental — a lot of segmentation variants. Where's the big idea?"</summary>
<div class="qa-body">

**Testing:** can you defend the *significance* of your work without getting rattled — a deliberate provocation.

**Answer A — reframe to problem-redefinition (confident).**
"I'd push back on 'variant.' Each of these *redefined the bottleneck*, which is the opposite of incremental. PointWSSIS didn't tune a segmenter — it showed the proposal stage, not the mask head, is the true bottleneck in semi-supervised instance segmentation. ECLIPSE didn't add a continual-learning trick — it showed you can drop distillation entirely and beat it by freezing + prompting, which inverts the field's assumption. ZIM showed *data granularity dominates architecture* for matting foundations, earning an ICCV Highlight (top ~3%) precisely because reviewers saw it as a reframing, not a fine-tune. The unifying big idea is 'define the real bottleneck and solve it with data+architecture co-design.' That's a research philosophy, not a pile of variants."

**Answer B — the arc as the big idea.**
"The big idea is the *arc*, not any single paper: driving perception up the ladder from cheap supervision to trustworthy reasoning. Individually each paper is a sharp contribution; together they're a coherent bet that perception's value is in being *label-efficient, updatable, and verifiable*. That's exactly the bet the field is now making with grounded agents."

**Follow-up:** "Fine — which single result would survive if the field forgot everything else?"
**Rebuttal tip:** Pick one and commit: "data granularity beats architecture for foundation models" (ZIM) or "relocate the bottleneck, don't tune the head" (PointWSSIS). Waffling here confirms the accusation; a crisp pick refutes it.

</div></details>

<details class="qa"><summary>"You're a vision person. This role touches LLMs/NLP — do you have the depth?"</summary>
<div class="qa-body">

**Testing:** honest self-assessment of a real gap, plus evidence you can close it.

**Answer A — own the boundary, show the bridge.**
"Fair — my *publication* depth is in vision, not in LLM pretraining or NLP theory. But two things. First, my current work lives at the seam: grounded VLMs and agentic program synthesis require me to reason about LLM planning, tool-use, chain-of-thought, hallucination, and evaluation — I'm operating there now, not aspirationally. Second, I know exactly what I don't know — I'd be careful before claiming I can derive RLHF or build a tokenizer from scratch cold. I'd rather tell you the boundary honestly and show I close gaps fast (I taught myself the SAM stack and the diffusion-conditioning literature for product work) than bluff transformer internals and get caught."

**Answer B — the T-shape (concise).**
"I'm T-shaped: deep in perception, broad and current across multimodal — attention/transformers, VLM training and eval, grounding, agents. The frontier I want is vision-language, where my depth is the scarce half and the LLM side is learnable and already part of my daily work."

**Follow-up:** "OK — implement multi-head attention / explain RoPE right now."
**Rebuttal tip:** *Practice the from-scratch primitives cold* before the loop — attention (correct head split, 1/√d_k scaling, causal mask), KV-cache, LoRA, top-k/top-p, RoPE. This is the #1 rejection reason for strong researchers. Honesty about depth only helps if you can back it when tested. Prep via the LLM and ML-coding chapters.

</div></details>

<details class="qa"><summary>"All your production work is single-domain — image, at one company. Can you generalize?"</summary>
<div class="qa-body">

**Testing:** breadth risk and adaptability — a real concern for a 5-year single-employer CV.

**Answer A — depth-transfers argument.**
"Two responses. First, within 'image at one company' I actually spanned a *lot*: weakly-supervised, continual, foundation models, matting, detection, on-device efficiency, biometric security, and now multimodal agents — different sub-problems, metrics, and constraints. Second, the transferable skill isn't the domain, it's the *method*: define the bottleneck, co-design data and architecture, run honest ablations, and close the production last mile. That method moved me from CAMs to foundation models to agents without friction, and it'll move to video, 3D, or a new modality the same way. I also have earlier exposure — OCR, autonomous-driving perception, face — from internships, so it's not as narrow as one line suggests."

**Answer B — concise.**
"Single company, but not single problem — I went from image tags to foundation models to agents. The constant is the research method, and that's what transfers. I'd treat a new domain as another bottleneck to define."

**Follow-up:** "Why never switch companies or domains before now?"
**Rebuttal tip:** Frame stability as *depth earned*, not inertia: "staying let me run the full research-to-product loop end to end, repeatedly — rare and valuable. Now I want the domain and scale jump." Own it as a choice.

</div></details>

<details class="qa"><summary>"Tell me about a research bet that failed."</summary>
<div class="qa-body">

**Testing:** honesty, reflection, and research judgment — a canned success dressed as failure fails the round.

**Answer A — real failure + the lesson (STAR-lite).**
"Early on I over-invested in *offline iterative* pseudo-label refinement for weakly-supervised instance seg — the intuition being more refinement rounds = better labels. It barely moved the needle and cost a lot of compute; the gains flattened almost immediately. The lesson reshaped BESTIE and PointWSSIS: I switched to **online, per-mini-batch self-refinement**, which was cheaper and actually worked, and I learned to *test the assumption cheaply first* before building the expensive version. Now I prototype the smallest ablation that could kill an idea before committing engineering to it."

**Answer B — a framing/scope failure (concise).**
"I once chased a quantize-first path for the on-device model because it was the fastest lever — and it killed boundary quality (hair, fingers) before anything else. The failure taught me the *order* of compression matters: resolution/width/decoder and distillation first, quantization last. It's now a rule I bring to every efficiency project."

**Follow-up:** "What would you do differently now?"
**Rebuttal tip:** The answer *is* the differently — "kill ideas with the cheapest possible experiment first." Make sure your failure has a concrete, transferable rule attached, not just "I learned to persevere."

</div></details>

<details class="qa"><summary>"You publish a lot. Will you actually ship here, or just chase papers?"</summary>
<div class="qa-body">

**Testing:** the flip side of the RS worry — whether your publishing habit crowds out product delivery. Common at product-AI orgs (Apple AIML, MS MAI, ByteDance applied).

**Answer A — evidence of shipping.**
"My publishing *comes from* shipping, not instead of it. The foreground API in production, FaceSign in a certified service, an on-device model at ~10 ms, ZIM integrated into CLOVA-X — those are shipped, not papers. I publish because rigor makes the product better and because I chose problems where the two align. I don't have a paper-vs-product tension; I have a habit of turning one into the other. If a quarter needs pure delivery, I'll deliver — I've done far more unpublished product work than published papers."

**Answer B — concise.**
"Look at the ratio: everything I published, I also shipped or open-sourced, and plenty I shipped I never published (the API, FaceSign). Papers are a byproduct of doing the product work rigorously, not a competing goal."

**Follow-up:** "What if we asked you to not publish for a year?"
**Rebuttal tip:** Accept it cleanly — "product and impact come first; I've worked under strict non-disclosure already (FaceSign)." Any hesitation reads as prioritizing your CV over the team.

</div></details>

<details class="qa"><summary>"There's a gap: no papers between some years / a slow patch. What happened?"</summary>
<div class="qa-body">

**Testing:** whether apparent gaps hide a problem — usually they don't, but the interviewer wants to see you not flinch.

**Answer A — the confidential-work explanation.**
"The 'quiet' stretches are when I was heaviest on *unpublishable* product work — FaceSign is a government-certified security product I can't publish, and the foreground API and on-device model are internal. So the paper timeline undercounts the actual output. I was shipping the whole time; some of the most demanding work simply doesn't show up on Google Scholar."

**Answer B — concise.**
"No gap in output — a gap in *publishable* output. Certified-security and internal-product work (FaceSign, the API) filled those periods and by nature can't be published."

**Follow-up:** "How do I verify impact I can't see?"
**Rebuttal tip:** Offer what's checkable — the DAN 24 talk, the public FaceSign product page, the "used by millions" scale, and references — while holding the confidential numbers. "I can point you to public evidence and references; the internal metrics stay internal" is the credible line.

</div></details>

<details class="qa"><summary>"Rapid-fire fundamentals: BatchNorm vs LayerNorm — why LN in transformers?"</summary>
<div class="qa-body">

**Testing:** breadth under pressure. One or two whiffs on fundamentals can sink an otherwise strong loop — this is a stand-in for the whole breadth round.

**Answer A — crisp mechanism.**
"BatchNorm normalizes across the batch dimension per feature, so it depends on batch statistics — bad for variable-length sequences and small batches, and it couples examples. LayerNorm normalizes across the feature dimension per token, independent of batch and sequence length, which is exactly what transformers need: per-token stability regardless of batch size or sequence position, and it behaves identically at train and inference (no running stats). That's why transformers use LN — and why RMSNorm (LN without mean-centering) is now common for efficiency."

**Follow-up chain:** "Pre-norm vs post-norm? Why does pre-norm train more stably?"
**Rebuttal tip:** This question is a *probe for the whole breadth round* — treat it as a signal to over-prepare fundamentals (optimization, normalization, attention, RoPE, scaling laws, RLHF, diffusion, MoE). Depth on your specialty won't save a breadth gap. Drill the foundations and LLM chapters cold.

</div></details>

<details class="qa"><summary>"Curveball: if you had one year and unlimited compute, what would you build?"</summary>
<div class="qa-body">

**Testing:** vision, ambition, and whether your research taste scales beyond incremental papers.

**Answer A — the coherent moonshot.**
"A **self-diagnosing perception stack for agents**: vision tools that don't just output a box or mask but report *calibrated uncertainty and a typed failure signal*, and a reasoning layer that consumes those signals to decide when to trust, re-run, or repair. Today agents fail silently because tools lie confidently. If perception could say 'I'm unsure about this occlusion' or 'this depth is unreliable,' agents could reason verifiably about 3D and time at frontier level *without* task-specific training. It's the natural top of my arc — from label-efficient perception to *trustworthy* perception — and unlimited compute lets me build the diagnostic data and the tool ensemble at scale."

**Answer B — concise.**
"Verifiable multimodal reasoning: perception tools that expose their own failure modes, and agents that repair around them. It's where grounding and agents converge, and it's the trust problem that gates real deployment."

**Follow-up:** "What's the first experiment you'd run?"
**Rebuttal tip:** Have a concrete week-one experiment — "instrument an existing VisProg/VADAR pipeline to log every tool's silent errors on a 3D benchmark, and measure how much of the final error is silent-perception vs planning." Vision without a first step reads as daydreaming.

</div></details>

---

## Cheat-sheet

| Situation | The move | The one-liner |
| --- | --- | --- |
| "Walk me through your research" | Give the **through-line**, then let them drill | "pixels → labels → continual → foundation → grounded reasoning" |
| Confidential work (FaceSign, API, ongoing) | **Decline-then-deliver** framing | "I can share the framing; the internal metrics are confidential." |
| "Your work is incremental" | Reframe as **bottleneck redefinition** | "Each paper relocated the bottleneck, not tuned the head." |
| "You're a vision person / LLM depth?" | **Own the boundary**, show the bridge, then *prove it cold* | "T-shaped; the vision half is the scarce one." |
| Co-authored paper (SSUL) | Specific **I vs we**, credit generously | Name your part, name theirs, no blur. |
| "Why leave NAVER?" | **Pull, not push** | "Toward frontier-scale, not away from NAVER." |
| Rapid-fire fundamentals | Over-prepare breadth; specialty won't cover a gap | One whiff can sink the loop. |
| Failed bet | Real failure + **transferable rule** | "Kill ideas with the cheapest experiment first." |
| Big-idea challenge | Commit to **one** durable result | "Data granularity beats architecture" (ZIM). |
| Every answer | Pick the **angle** to the room; internalize don't memorize | Depth for FAIR, impact for HM, 30 sec for recruiter. |

**Golden rules:** (1) Never invent a number — describe the mechanism or say it's confidential. (2) Make *your* contribution unambiguous. (3) Name a weakness before they do. (4) Tailor every answer with the [Company Playbooks](#/process/companies).

## Cross-links

- Deep-dives (the raw technical material behind every answer above): [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [FaceSign](#/resume/facesign) · [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM/Agents](#/resume/grounded-vlm-agents)
- [Your CV → Interview Map](#/resume/overview) — which project to lead with, per company
- [STAR & The Story Bank](#/behavioral/star) — behavioral framing for the failure/conflict/ownership questions
- [Company Playbooks](#/process/companies) — what each target company probes, to re-point every answer
