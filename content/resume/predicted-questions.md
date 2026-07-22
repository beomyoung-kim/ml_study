# Predicted Questions & Answer Drafts

<div class="tag-row"><span class="tag">research narrative</span><span class="tag">project probes</span><span class="tag">research→product</span><span class="tag">motivation</span><span class="tag">curveballs</span><span class="tag badge-new">flagship</span></div>

> [!TIP] Why this chapter matters
> This chapter is a question bank for preparing both CV fact checks and research deep-dives. Interviewers may cross-check claims, personal contributions, design choices, and limitations. Each question includes a possible intent, an **unverified answer draft**, a follow-up, and a defense point. Do not memorize the wording; rewrite it around real evidence and trade-offs you are allowed to disclose.

> [!IMPORTANT] These are answer drafts, not a system of record
> The sentences below are rehearsal drafts derived from the resume, public papers, and local submissions. Use ownership that the resume states explicitly—creating ZIM, building the FaceSign anti-spoofing model, and independently developing the on-device model—while personally verifying any more detailed role, product requirement, conflict, internal comparison, or figure. Delete unverified narratives. Always disclose work under review as **under review**, and discuss methods, evaluation design, and trade-offs only within the approved public scope.

## How to use these answers

<div class="proscons"><div><div class="pros-t">Do</div>
<b>Internalize, don't memorize.</b> Learn the <i>spine</i> of each answer — the claim, the evidence, the trade-off — then rebuild it live in your own words. A recited answer dies on the first follow-up; a understood one survives ten.
<br><br>
<b>Pick the angle for the round.</b> A technical deep-dive may emphasize mechanism and ablations, a hiring-manager conversation scope and impact, and a recruiter conversation short facts and logistics. Confirm the actual agenda; do not infer a question style from a person's or company's name.
</div><div><div class="cons-t">Don't</div>
<b>Don't overclaim.</b> For confidential work (FaceSign, the foreground API, ongoing/under-review), speak to <b>framing and impact</b>, never invented numbers: <i>"I can share the framing; the specific internal metrics are confidential."</i> That line reads as integrity, not evasion.
<br><br>
<b>Don't blur "I" and "we".</b> Credit collaborators, but make <i>your actual</i> contribution clear. There is no basis for ranking how often this issue causes rejection, but an answer with no evidence of ownership is a weak signal.
</div></div>

Use the [Company Playbooks](#/process/companies) as a starting point for **company tailoring**, then retarget the answer using the actual job description, recent public research, and a recruiter-confirmed loop. Do not infer interview format or value weights from the company name alone.

Deep-dives for the raw material: [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [Phoenix](#/resume/phoenix-mask-refinement) · [FaceSign](#/resume/facesign) · [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM/Agents](#/resume/grounded-vlm-agents) · [Spatial-Reasoning Agent (under review)](#/resume/neurips26-spatial-reasoning). For short answers by round, use [Resume-Based Interview-Stage Answers](#/resume/interview-stage-answers); for behavioral framing, [STAR & The Story Bank](#/behavioral/star); for the full map, [Your CV → Interview Map](#/resume/overview).

---

## A · Research narrative & motivation

A common opener in research-track conversations. It tests whether you can explain consistent problem selection and a next direction beyond listing papers.

<details class="qa"><summary>"Walk me through your research."</summary>
<div class="qa-body">

**Testing:** can you compress 14+ papers into a coherent arc with a through-line, at the right altitude, in ~2–3 minutes — while making *your* role clear.

**Answer A — the through-line (technical narrative).**
"There's one thread: **making visual perception accurate and trustworthy under real-world constraints.** I started in **weakly/semi-supervised segmentation** — DRS, BESTIE, PointWSSIS — because the bottleneck to deploying perception is labels, so I attacked the label-efficiency frontier: how to get instance masks from image tags or a few points. Then **continual learning** — SSUL and ECLIPSE — because deployed models must absorb new classes without forgetting or storing old data. Then I scaled up to **foundation models** with ZIM, a promptable zero-shot matting model built on SAM. Across all of it the pattern is the same: *define the real bottleneck, then solve it with a data + architecture co-design rather than brute force.* Most recently I've moved up the stack to **grounded VLMs and training-free visual reasoning agents** — connecting language reasoning to pixel- and region-level evidence, and diagnosing when perception tools silently fail. So: pixels → labels → continual → foundation → grounded reasoning. Same obsession, higher abstraction."

**Answer B — impact-first (for a hiring manager).**
"Two-sentence version: I build perception that ships. I have seven first/co-first-author papers at major venues, an ICCV 2025 Highlight for ZIM, and a record of contributing research to NAVER Cloud products — a foreground-segmentation API, anti-spoofing for a certified face-authentication service, and an on-device segmenter at about 10 ms. Within the public scope, the consistent link is that research rigor and product constraints make each other's problems clearer."

**Answer C — concise/executive (30 sec).**
"I'm a computer-vision applied scientist. My arc goes from label-efficient and continual segmentation to a segmentation/matting foundation model, ZIM, which received an ICCV 2025 Highlight, and now to grounded vision-language models and visual reasoning agents. The constant is treating accuracy, efficiency, and verifiability together with real product constraints."

**Follow-up:** "Which of these do you consider your best work, and why?"
**Rebuttal tip:** Name ZIM for *reach* (foundation + Highlight + product), or PointWSSIS for *elegance* (redefining the proposal bottleneck) — but justify by **the idea**, not the venue. "Best" = clearest problem redefinition, not highest citation count.

</div></details>

<details class="qa"><summary>"What's the thread connecting your papers? Or is it just whatever your team needed?"</summary>
<div class="qa-body">

**Testing:** research *taste* — do you choose problems, or do problems choose you? The cynical framing is bait; don't get defensive.

**Answer A — the intellectual thread.**
"The thread is **cheap or weak signals → reliable structured perception.** DRS and BESTIE: image tags → instance masks. PointWSSIS: a few points → full masks. ECLIPSE: a frozen model + tiny prompts → new classes without forgetting. ZIM: coarse SA-1B labels → fine alpha mattes via a label converter. Even the agents work is the same move at the top of the stack: noisy specialist tool outputs → a verifiable, self-diagnosing reasoning program. I'm consistently drawn to the gap between *what supervision you can afford* and *what fidelity the product needs*, and I close it with data + architecture design."

**Answer B — honest about the environment (owns the constraint).**
"Both the industry environment and my research interests influenced the work. For each project, I will verify from records whether it came from a specific product pain point and whether I owned the problem framing. The verifiable commonality is that the papers evaluated practical constraints such as labeling cost, continual updates, and boundary fidelity as generalizable research problems."

**Follow-up:** "So what problem would you pick next, with total freedom?"
**Rebuttal tip:** Have a real answer ready — "verifiable spatial reasoning: perception tools that report their own uncertainty so an agent knows when to distrust them." It proves the thread is forward-looking, not a retrofit.

</div></details>

<details class="qa"><summary>"Why the pivot from perception to grounded VLMs and agents? Isn't that chasing the trend?"</summary>
<div class="qa-body">

**Testing:** whether the move is opportunistic (following LLM hype) or a principled continuation. This is *the* narrative question for 2025–26 vision hires.

**Answer A — continuation, not pivot (depth).**
"It's not a pivot away from perception — it's perception finding its role in the new stack. End-to-end VLMs are strong but they **hallucinate** and produce *unsupported textual descriptions*: the answer can be right while the visual evidence is wrong — spurious success. My whole background is high-fidelity, region- and pixel-level perception. That's exactly the missing ingredient: grounding language reasoning in verifiable visual evidence, and giving agents *specialist tools* (like ZIM-quality masks or depth) so they can reason about 3D and time. The interesting new problem is that those tools **fail silently** — they return a plausible-but-wrong box or mask with no exception — so my current work turns those silent failures into *typed diagnoses* that drive targeted program repair. I'm not leaving perception; I'm making it the trustworthy tool layer under reasoning."

**Answer B — market/product logic (impact).**
"Applications such as image editing, robotics, and UI agents need to connect a language query to a region, object, or action and then inspect the result. Whether an end-to-end model or modular tool use is better depends on the task, but my perception research provides a foundation for analyzing grounding and tool reliability."

**Answer C — concise.**
"Alongside perception quality, it has become important to connect outputs to evidence and detect failures. Grounding and agentic tool use are design axes for that problem, and my perception depth is the bridge."

**Follow-up:** "If frontier VLMs keep improving, doesn't your modular/agent approach become obsolete?"
**Rebuttal tip:** Acknowledge the strengths of end-to-end models, then discuss both the potential benefits and costs of modular approaches: precise measurement, inspectable traces, and module upgrades versus orchestration cost and tool-error accumulation. A hybrid is a design candidate to compare, not an automatic default.

</div></details>

<details class="qa"><summary>"You did a PhD while working full-time. Doesn't that mean your research is a side project?"</summary>
<div class="qa-body">

**Testing:** rigor and independence — is the industry job the "real" work and the PhD a formality, or are they genuinely one program?

**Answer A — integration (reframe).**
"They run on separate schedules, but form a program in which problem selection and validation practice reinforce each other. Product experience exposes practical constraints such as labeling cost, category drift, latency, and boundary quality; research training enforces ablations, baselines, and failure analysis. My strength is being able to explain seven first/co-first-author papers, an ICCV Highlight, and deployment experience together. I will connect a paper directly to a product only when that relationship is factual."

**Answer B — independence signal.**
"First/co-first-author status is public, but it does not automatically prove contribution scope. For each project, I will create a table of which parts of architecture, data, evaluation, and writing I actually owned and which parts collaborators owned, then answer only with verifiable decisions."

**Follow-up:** "How do you manage the time and the two sets of priorities?"
**Rebuttal tip:** Don't romanticize the hustle. Frame it as *alignment* — you chose problems where the product need and the publishable question overlap, so you're not paying the cost twice. That's judgment, not just stamina.

</div></details>

---

## B · Per-project probes

Expect the interviewer to pick one project — usually ZIM or whichever matches their team — and drill until you either show ownership or crack. Full technical depth lives in the deep-dives; here are the *entry* probes and how to answer at interview altitude.

<details class="qa"><summary>ZIM — "Why couldn't you just fine-tune SAM on matting data?"</summary>
<div class="qa-body">

**Testing:** can you explain the paper's core motivation and the failure mode of the public-matte baseline accurately?

**Answer A — the failure mode (technical).**
"In the paper's public-matte baseline, micro-level generalization degrades sharply because public matting data is biased toward macro targets. In the same-architecture comparison, fine-grained SAD is about 120 versus about 10 when trained on SA1B-Matte, showing that data granularity is load-bearing. We therefore use a **label converter** with SGA and STL to turn SA-1B-derived coarse masks into fine alpha mattes. I would not mix image counts with mask-instance counts; the public paper reports training on 1% and about 2.2M matte labels, and I would state those units explicitly."

**Answer B — the design principle (concise).**
"Because the problem isn't the loss, it's the data contract. SAM's supervision is coarse and hard-masked; matting needs soft, micro-level boundaries. Fine-tuning on macro portrait data destroys the promptable zero-shot behavior. We generated micro-granular matte labels at SA-1B scale via a converter, plus a hierarchical pixel decoder and prompt-aware masked attention to actually render fine detail. That's why it stays zero-shot *and* produces mattes."

**Follow-up:** "What does the hierarchical pixel decoder buy you, concretely?"
**Rebuttal tip:** SAM's stride-4 decoder + naive upsample causes **checkerboard artifacts** and loses fine structure; the multi-level (stride 2/4/8) fusion recovers high-res detail for ~10 ms overhead. Have that number and the mechanism ready — see [ZIM](#/resume/zim).

</div></details>

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
"Freezing Mask2Former parameters after step 1 prevents old-weight drift and reduces forgetting, but it does not preserve final old-class outputs exactly. Error propagation can still arise through new prompts and aggregation, and plasticity is limited. The paper addresses this trade-off with deep visual prompts, stronger frozen initialization, and logit manipulation for the changing meaning of no-object. I would keep claims within the public ADE20K protocol and the result using about 1.3% trainable parameters."

**Answer B — product framing (impact).**
"A product hypothesis is that a new-class adapter can help in settings where replaying old data or fully retraining is difficult. The paper alone, however, does not prove deployment in an actual API, on-device suitability, or privacy compliance. I would connect it to this trade-off only when the job description actually contains that constraint."

**Follow-up:** "Why panoptic, not just semantic continual learning?"
**Rebuttal tip:** Panoptic adds instance matching, stuff, and a changing no-object definition to semantic prediction, while PQ is also sensitive to recognition failures. Explain these as additional problem structure rather than a simplistic difficulty ranking. See [ECLIPSE](#/resume/eclipse).

</div></details>

<details class="qa"><summary>PointWSSIS — "What's the actual insight? It sounds like just another semi-supervised trick."</summary>
<div class="qa-body">

**Testing:** whether you redefined a problem or just tuned one — the difference between a strong CVPR paper and an incremental one.

**Answer A — the redefinition (depth).**
"The insight is that one major bottleneck in semi-supervised instance segmentation is the proposal stage. Lowering the threshold increases false positives; raising it can miss instances. Point labels add location information and reduce proposal ambiguity, APS handles scale selection, and MaskRefineNet refines rough masks. The paper reports 33.7 AP versus a 24.9 comparison baseline in the COCO 5% full-label setting, and 38.8 versus the 39.7 fully supervised reference in the 50% setting. I would cite the protocol and label budget together and would not claim that points eliminate error entirely."

**Answer B — the lineage (concise).**
"BESTIE showed you can push image-tags → instances but you're capped by pseudo-label drift. PointWSSIS asks: what's the cheapest signal that removes the *proposal* bottleneck specifically? A point. It's a targeted intervention on the real bottleneck, which is why the gap at low label budgets is large, not marginal."

**Follow-up:** "Does the point have to be the centroid?"
**Rebuttal tip:** No — random-in-mask works about as well as centroid (SOLOv2 tolerates diverse in-region points), which makes the labeling even cheaper. That robustness is itself an ablation-backed selling point. See [PointWSSIS & BESTIE](#/resume/pointwssis-bestie).

</div></details>

<details class="qa"><summary>Phoenix — "Using adversarial attacks to generate data — isn't that just a fancier version of morphological noise?"</summary>
<div class="qa-body">

**Testing:** can you explain why existing refiners are insufficient and defend the adversarial reinterpretation as a principled choice rather than a trick — the core motivation of the ECCV '26 paper?

**Answer A — failure mode + principle (technical).**
"Morphological noise transforms ground-truth boundaries independently of the image or current model, whereas AMP creates hard perturbations along embedding-space directions that increase the current decoder loss. The paper's edge-correlation analysis and ablations report a different distribution and effect from morphology-only noise. I would not equate gradient magnitude with calibrated predictive uncertainty. A guidance mask controls expansion, contraction, and inversion, while CMRL learns the relationship among ground-truth, noisy, and refined masks."

**Answer B — impact/lineage (concise).**
"This work generalizes the pseudo-label-quality problem addressed by PointWSSIS's MaskRefineNet into training-pair generation for refiners. It combines AMP and CMRL and reports gains of up to 16.1 AP points under the paper's weakly and semi-supervised refinement protocols. Phoenix and ZIM use the SAM family while ECLIPSE uses Mask2Former, but they connect at a higher-level design axis: preserving a pretrained trunk and adapting with a small module."

**Follow-up:** "Why attack embedding space rather than pixels?"
**Rebuttal tip:** Efficiency plus semantic-level perturbation. It reuses the frozen 637M-parameter encoder's image embedding and updates only a lightweight decoder in about 6 ms per iteration, while perturbing high-level semantics rather than pixel noise. AMP alone gives +4.6, CMRL alone +1.5, and together +6.4, a **synergy** larger than the individual gains' sum because realistic noise makes the three-way contrast more meaningful. See [Phoenix](#/resume/phoenix-mask-refinement).

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
**Rebuttal tip:** Fill in `[design, experiments, and writing I owned] / [the co-author's part] / [joint decisions]` from source records. Do not preassign segmentation modeling or unknown-class handling to yourself.

</div></details>

<details class="qa"><summary>FaceSign — "Tell me about the anti-spoofing model. What accuracy did you hit?"</summary>
<div class="qa-body">

**Testing:** do you understand the basic threats and evaluation in face anti-spoofing, and can you handle both confidentiality boundaries and actual ownership accurately?

**Answer A — decline-then-deliver (the model to imitate).**
"Within the scope stated on my resume, I built the anti-spoofing model behind FaceSign. My detailed scope inside the model was `[confirmed architecture, training, and evaluation scope]`. I will not share accuracy, attack-set composition, or architecture that is not approved for disclosure. In general, presentation-attack detection covers print, replay, and 3D masks, while digital injection needs a separate pipeline-integrity layer; APCER and BPCER must also be distinguished from recognition FAR and FRR."

**Answer B — the systems view (concise).**
"The public resume explicitly records that I built the anti-spoofing model. I will separate that model-level ownership from ownership of the whole FaceSign system, data, and deployment. I can explain `[one or two real model decisions]` within the record and disclosure boundary, then discuss the general FAS threat model, sensing, and operating-point trade-offs without presenting them as internal implementation facts."

**Follow-up:** "How would you compare it with a publicly documented hardware-co-designed biometric system?"
**Rebuttal tip:** Use public architectures only as general examples; do not infer FaceSign's sensors, pipeline, or your role. If there is no head-to-head protocol, do not rank them. See [FaceSign](#/resume/facesign).

</div></details>

<details class="qa"><summary>On-device seg — "Why is 10 ms the target? Why not just use a bigger model on the server?"</summary>
<div class="qa-body">

**Testing:** systems/efficiency reasoning and whether you understand *why* on-device matters — not just that you hit a number.

**Answer A — frame-budget engineering (depth).**
"My resume records that I independently developed this human-segmentation model and achieved about 10 ms on a mobile CPU. In a 30-fps pipeline's roughly 33 ms budget, I will attach the actual device, input, runtime, thread setting, and statistic to that figure. I will explain the real product reason for the CPU target and only the levers I actually compared—such as resolution, width, or decoder depth—along with the target-device quality–latency trade-off at each verified stage."

**Answer B — the two-tier story (impact).**
"Separating a cloud foundation model from an on-device specialist can optimize quality, latency, and cost differently. What my resume verifies is independent model development, about 10 ms on a mobile CPU, and ONNX-based in-house serving. I will not infer that ZIM was the teacher, that distillation or quantization was used, or that the model shipped directly inside a phone app without separate records."

**Follow-up:** "You said ONNX serving — is it really on the phone CPU or on a server?"
**Rebuttal tip:** Be precise, because the CV wording invites this: the *model* was designed and measured for a mobile-CPU budget; the *serving path* was ONNX-based in-house infra. Distinguish design target from deployment infra rather than overclaiming a phone deployment you can't detail. See [On-Device Seg](#/resume/on-device-segmentation).

</div></details>

<details class="qa"><summary>Ongoing agents — "Your NeurIPS submission is under review. Walk me through it."</summary>
<div class="qa-body">

**Testing:** can you explain the problem and mechanism clearly while respecting the status and disclosure boundaries of work under review?

**Answer A — problem-first framing (the safe, strong version).**
"Because it is under review, I will not discuss comparative conclusions, models, benchmarks, or figures; I will explain only the approved problem and mechanism. The setting is visual program synthesis for 3D spatial reasoning, and the failure of interest is a **silent perception failure** in which a tool returns an incorrect box or depth without an exception. The submission compares a structured scene hypothesis with an execution trace to create a **typed diagnosis** and connect it to targeted repair. Its public lineage is VisProg, ViperGPT, and VADAR, and any exact differentiation claim will stay within the submission's related-work scope."

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

Applied-scientist roles may test research quality and delivery experience together. Prove public research results, product artifacts, and personal ownership separately.

<details class="qa"><summary>"How would you defend the internal-comparison claim for the foreground-segmentation API?"</summary>
<div class="qa-body">

**Testing:** rigor and honesty behind a big claim. A bold benchmark claim with no methodology is a red flag; with methodology it's a green one.

**Answer A — verification first (rigor).**
"My resume states that I developed the model and dataset for the foreground-segmentation API, that it is served internally, and that it outperformed the Photoroom, Remove.bg, and Adobe segmentation APIs in an internal evaluation. I can state that resume-backed headline while bounding it to a particular time, domain, and internal protocol rather than claiming universal superiority. Within the approved scope, I will explain the dataset source, sampling date, API versions, input/output normalization, metrics and human review, and confidence intervals; I will not invent undisclosed figures or slices."

**Answer B — when the public scope is narrow.**
"I will clearly state the scope already disclosed on my resume—model and dataset development, internal serving, and the named three-API internal comparison. If the protocol or figures are not approved for disclosure, I will not improvise them; instead I will explain why foreground quality should include boundaries, transparency, failure slices, and latency in addition to mean IoU, along with the data and model decisions I actually owned."

**Follow-up:** "Is this the same model as ZIM?"
**Rebuttal tip:** No — don't conflate them. The API is a product line (internal serving); ZIM is a promptable zero-shot foundation model. Shared DNA (boundary quality, data curation), different artifacts. Claiming they're one model is the kind of overclaim that unravels under follow-up.

</div></details>

<details class="qa"><summary>"You got ZIM into CLOVA-X Image Editing. What broke when research met production?"</summary>
<div class="qa-body">

**Testing:** whether you've actually shipped research, or just tossed a checkpoint over the wall. The interesting answer is the friction.

**Answer A — the handoff frictions (systems).**
"The publicly verifiable facts are the TEAM NAVER DAN 24 presentation and the CLOVA-X Image Editing integration. My direct production scope is `[confirmed ownership]`. A general matte-integration checklist includes alpha representation, resolution, latency, API compatibility, monitoring, and fallback, but I will not claim without evidence that this project used caching, tiering, or particular thresholds. I will choose one real, shareable friction and explain the decision and outcome."

**Answer B — concise (impact).**
"ZIM's public code and demo, and its CLOVA-X integration, are verifiable. Between them, I will describe only handoff and failure-hardening work confirmed by actual records; if it cannot be disclosed, I will distinguish a general checklist from public project results."

**Follow-up:** "How would you extend it to video editing?"
**Rebuttal tip:** SAM2-style memory + temporal consistency is the direction, and it's genuinely unsolved for matting — say so. Flickering alpha across frames is the hard part; don't pretend it's a trivial extension.

</div></details>

<details class="qa"><summary>"You open-sourced most of your work. Doesn't that give away NAVER's advantage?"</summary>
<div class="qa-body">

**Testing:** product/business judgment and whether you understand *why* a company publishes.

**Answer A — draft strategic rationale.**
"ZIM, ECLIPSE, PointWSSIS, and BESTIE have public code, which supports reproduction, follow-on use, and research adoption. Product data, customer evaluations, and security details, by contrast, may need protection. If I actually participated in deciding the disclosure boundary, I will explain the criteria and trade-offs. Otherwise I will not present the company's intent as my own decision; I will distinguish public outcomes from general IP principles."

**Answer B — concise.**
"Open-source scope balances research reproducibility with product and security IP. I will describe the effect of public code through verifiable adoption and the nonpublic scope through company policy, without overstating it as a strategy I personally set."

**Follow-up:** "So what *would* you keep closed here, if we hired you?"
**Rebuttal tip:** Show you can reason about *their* IP, not just recite NAVER's policy — "product-specific data, eval sets tied to real customers, and anything safety/abuse-related; general methods I'd push to publish." That's the maturity they're checking for.

</div></details>

---

## D · Career, motivation & fit

Recruiter and hiring-manager territory. These questions decide *fit*, and a loose answer can end a loop even when the technical rounds are strong. Keep answers honest, specific, and non-defensive.

<details class="qa"><summary>"Why leave NAVER after 5+ years?"</summary>
<div class="qa-body">

**Testing:** are you moving *toward* something or merely *away* from something? Both bitterness and vagueness are risky.

**Answer A — toward the frontier (positive pull).**
"My time at NAVER Cloud gave me experience across public research and product touchpoints. I am not leaving by inferring or criticizing the limits of my current organization; I am interested because `[the target team's latest official research or job description]` aligns specifically with my next question in grounded multimodal reasoning. The evidence I bring is `[public project or verified role]`, and the point I want to learn and contribute in the new environment is `[confirmed team agenda]`."

**Answer B — scope and scale (concise).**
"Rather than answering that I want to get away from my current role, I am looking for the intersection between `[a role characteristic supported by official evidence]` and my `[verified experience]`. I will offer a concrete hypothesis for growth and contribution and confirm the actual team scope in the interview."

**Follow-up:** "What specifically can't you do at NAVER that you could here?"
**Rebuttal tip:** Cite one recent official paper, product, or job description; connect it to your evidence and a question you still need to verify. Do not merely list brands or assert what your current employer cannot do.

</div></details>

<details class="qa"><summary>"Why this company specifically?" (per-company)</summary>
<div class="qa-body">

**Testing:** can you explain the real intersection between the role and your experience using current official evidence? Use the [Company Research Playbook](#/process/companies).

Fill these four fields for each company, then combine them into one paragraph.

| Evidence | What to fill in |
| --- | --- |
| **Official, dated signal** | One recent job description, team page, paper, or product |
| **My evidence** | A public result such as ZIM, ECLIPSE, or PointWSSIS, or a verified product role |
| **Contribution hypothesis** | The actual intersection in precise masks, low-label adaptation, grounding, or efficiency |
| **Uncertainty to verify** | Team scope, publication policy, compute/data access, or loop format |

Answer scaffold: *"In `[dated official source]`, I saw the specific agenda `[X]`. What I learned from my verified project `[Y]` connects to the concrete gap `[Z]` in that problem. In the first year, I would like to test `[a falsifiable hypothesis]`, while confirming the team's actual priorities with you."* Do not infer company culture, interview format, or IP strategy from a name alone.

**Follow-up:** "What would you want to work on in your first year?"
**Rebuttal tip:** Name a real project on *their* roadmap that your skills uniquely serve — not a generic "learn the codebase." Specificity proves the fit is real.

</div></details>

<details class="qa"><summary>"You're in Seoul. Are you actually willing to relocate?" / visa & logistics"</summary>
<div class="qa-body">

**Testing:** logistical seriousness — recruiters kill candidates who wobble here late.

**Answer A — clear commitment.**
"I will complete this answer only after confirming my actual intent and constraints: relocation `[possible / not possible / conditional]`, visa-sponsorship need `[ ]`, PhD defense and graduation timeline `[ ]`, and possible start-date range `[ ]`. I will not state unconfirmed adviser flexibility or a remote option as fact."

**Answer B — concise.**
"The conditions currently confirmed are `[ ]`, and I will coordinate the required visa and possible start date with the recruiter." Change this to a categorical answer only after the decision is final.

**Follow-up:** "Does the PhD need to finish first?"
**Rebuttal tip:** Confirm the defense timeline, earliest employment date, visa requirements, and physical-presence requirements with the relevant parties. Do not promise remote completion or adviser approval before confirmation.

</div></details>

<details class="qa"><summary>"Do you see yourself as a researcher or an engineer?"</summary>
<div class="qa-body">

**Testing:** self-awareness and fit to the specific track (RS vs AS vs MLE). The wrong self-label mismatches you to the loop.

**Answer A — the applied-scientist identity.**
"Applied scientist is the closest identity. In public papers I worked with ablations, baselines, and failure analysis; in resume-listed product work I handled latency and deployment constraints. I will mention a specific ONNX issue or handoff ownership only when there is a verified example. I will also confirm the target role's expected research-versus-delivery balance first."

**Answer B — concise.**
"The two evidence axes are my first/co-first-author research record and the resume-listed contributions to models, APIs, and a certified service. I will verify each product's release status and my contribution scope, then explain the applied-science intersection."

**Follow-up:** "Which do you enjoy more?"
**Rebuttal tip:** Honest but role-aware — lean research for an RS panel, lean shipping for AS/MLE, but always tie back to "the loop between them is what I actually love." Don't disown either half.

</div></details>

---

## E · Hard, curveball & weakness probes

For hard questions, state the scope and mitigation plan precisely instead of hiding weaknesses. Both defensiveness and inventing facts or fake failures are major risks.

<details class="qa"><summary>"Your work looks incremental — a lot of segmentation variants. Where's the big idea?"</summary>
<div class="qa-body">

**Testing:** can you defend the *significance* of your work without getting rattled — a deliberate provocation.

**Answer A — reframe to problem-redefinition (confident).**
"I would answer that concern with each paper's problem reformulation and evidence. PointWSSIS tests the proposal bottleneck, ECLIPSE a freezing-plus-prompting alternative, and ZIM the combination of data granularity and promptable matting. I would not infer the reason for a venue decision as though I knew the reviewers' minds. The common research philosophy is to state the real bottleneck, co-design data and architecture, and separate the effects with ablations."

**Answer B — the arc as the big idea.**
"The big idea is the *arc*, not any single paper: driving perception up the ladder from cheap supervision to trustworthy reasoning. Individually each paper is a sharp contribution; together they're a coherent bet that perception's value is in being *label-efficient, updatable, and verifiable*. That's exactly the bet the field is now making with grounded agents."

**Follow-up:** "Fine — which single result would survive if the field forgot everything else?"
**Rebuttal tip:** Choose one and justify it with a public ablation: data granularity was load-bearing in ZIM, while PointWSSIS reformulated the proposal bottleneck through supervision. Do not turn either into an exaggerated universal law.

</div></details>

<details class="qa"><summary>"You're a vision person. This role touches LLMs/NLP — do you have the depth?"</summary>
<div class="qa-body">

**Testing:** honest self-assessment of a real gap, plus evidence you can close it.

**Answer A — own the boundary, show the bridge.**
"Fair — my *publication* depth is in vision, not in LLM pretraining or NLP theory. But two things. First, my current work lives at the seam: grounded VLMs and agentic program synthesis require me to reason about LLM planning, tool-use, chain-of-thought, hallucination, and evaluation — I'm operating there now, not aspirationally. Second, I know exactly what I don't know — I'd be careful before claiming I can derive RLHF or build a tokenizer from scratch cold. I'd rather tell you the boundary honestly and show I close gaps fast (I taught myself the SAM stack and the diffusion-conditioning literature for product work) than bluff transformer internals and get caught."

**Answer B — the T-shape (concise).**
"I'm T-shaped: deep in perception, broad and current across multimodal — attention/transformers, VLM training and eval, grounding, agents. The frontier I want is vision-language, where my depth is the scarce half and the LLM side is learnable and already part of my daily work."

**Follow-up:** "OK — implement multi-head attention / explain RoPE right now."
**Rebuttal tip:** Implement and explain the from-scratch primitives the role requires — attention with correct head splitting, $1/\sqrt{d_k}$ scaling and causal masks, plus KV cache, LoRA, sampling, and RoPE. There is no basis for calling any one item the number-one rejection reason, but fundamentals outside your specialty may be evaluated separately.

</div></details>

<details class="qa"><summary>"All your production work is single-domain — image, at one company. Can you generalize?"</summary>
<div class="qa-body">

**Testing:** breadth risk and adaptability — a real concern for a 5-year single-employer CV.

**Answer A — depth-transfers argument.**
"Two responses. First, within 'image at one company' I actually spanned a *lot*: weakly-supervised, continual, foundation models, matting, detection, on-device efficiency, biometric security, and now multimodal agents — different sub-problems, metrics, and constraints. Second, the transferable skill isn't the domain, it's the *method*: define the bottleneck, co-design data and architecture, run honest ablations, and close the production last mile. That method moved me from CAMs to foundation models to agents without friction, and it'll move to video, 3D, or a new modality the same way. I also have earlier exposure — OCR, autonomous-driving perception, face — from internships, so it's not as narrow as one line suggests."

**Answer B — concise.**
"Single company, but not single problem — I went from image tags to foundation models to agents. The constant is the research method, and that's what transfers. I'd treat a new domain as another bottleneck to define."

**Follow-up:** "Why never switch companies or domains before now?"
**Rebuttal tip:** Explain the depth and repeated responsibilities you actually gained during the tenure with concrete examples. Use only verified facts for any claim of complete end-to-end ownership or motivation for the next move.

</div></details>

<details class="qa"><summary>"Tell me about a research bet that failed."</summary>
<div class="qa-body">

**Testing:** honesty, reflection, and research judgment — a canned success dressed as failure fails the round.

**Answer A — research-failure scaffold.**
"In `[project]`, the hypothesis I owned was `[ ]`. Under `[matched experimental condition or log]`, I observed `[falsifying signal]` and stopped at `[time]`. My pivot was `[ ]`, and the result was `[a public result or the remaining failure]`. Since then, I apply `[a specific change in experiment order]`." Use the offline-refinement example only if it was a real event that you can explain from records.

**Answer B — engineering-failure scaffold.**
"I first tried `[actual optimization or deployment decision]`, but `[quality or latency failure]` appeared under `[measurement protocol]`. I isolated the cause with `[ablation or profile]` and changed to `[alternative]`." A quantize-first path, boundary collapse, and distillation order are not established by the resume; use them only if actual logs support them.

**Follow-up:** "What would you do differently now?"
**Rebuttal tip:** The answer *is* the differently — "kill ideas with the cheapest possible experiment first." Make sure your failure has a concrete, transferable rule attached, not just "I learned to persevere."

</div></details>

<details class="qa"><summary>"You publish a lot. Will you actually ship here, or just chase papers?"</summary>
<div class="qa-body">

**Testing:** how do you set priorities and ownership when publication goals conflict with product delivery?

**Answer A — evidence of shipping.**
"Public and resume evidence includes the foreground API, a FaceSign contribution, an on-device model at about 10 ms, and the touchpoint between ZIM and CLOVA-X. For each, I will distinguish my role from the actual release status. I will not claim that research and delivery are always aligned; if there is a real case of conflict, I will explain how I adjusted scope, deadline, and publication."

**Answer B — concise.**
"I have research with public code and product contributions listed on the resume. Rather than claim that every paper shipped or compare the volume of unpublished work, I will distinguish the research artifact and product artifact in two verifiable examples."

**Follow-up:** "What if we asked you to not publish for a year?"
**Rebuttal tip:** Accept it cleanly — "product and impact come first; I've worked under strict non-disclosure already (FaceSign)." Any hesitation reads as prioritizing your CV over the team.

</div></details>

<details class="qa"><summary>"There's a gap: no papers between some years / a slow patch. What happened?"</summary>
<div class="qa-body">

**Testing:** whether apparent gaps hide a problem — usually they don't, but the interviewer wants to see you not flinch.

**Answer A — explain the actual timeline.**
"During `[period]`, I focused on `[approved resume-listed role or artifact]`. I will attribute the gap to security or internal-product disclosure limits only when the actual policy and timeline confirm that causal story. For activity without public evidence, I will state only the role and dates and will not inflate its impact."

**Answer B — concise.**
"The work during the interval between papers was `[verified project and responsibility]`. I will distinguish public artifacts from the nonpublic scope."

**Follow-up:** "How do I verify impact I can't see?"
**Rebuttal tip:** Start with verifiable evidence such as the DAN 24 talk, a public product page, and papers or code. Provide references only with the person's prior consent and within company procedure, and do not disclose internal metrics. "I can point you to public evidence; internal metrics stay inside the company" is credible.

</div></details>

<details class="qa"><summary>"Rapid-fire fundamentals: BatchNorm vs LayerNorm — why LN in transformers?"</summary>
<div class="qa-body">

**Testing:** breadth under pressure. Getting one question wrong does not automatically mean rejection, but the interviewer may evaluate both conceptual precision and how you correct the limits of what you know.

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
| Rapid-fire fundamentals | Practice retrieval for role-relevant breadth | State assumptions and correct errors; do not overread one question as an automatic pass/fail rule. |
| Failed bet | Real failure + **transferable rule** | "Kill ideas with the cheapest experiment first." |
| Big-idea challenge | Commit to **one** durable result | "The ablation showed data granularity was load-bearing" (ZIM). |
| Every answer | Pick the **angle** for the round; internalize, don't memorize | Deep-dive: mechanism; HM: scope/impact; recruiter: short facts/logistics. |

**Golden rules:** (1) Never invent a number — describe the mechanism or say it's confidential. (2) Make *your* contribution unambiguous. (3) Name a weakness before they do. (4) Tailor every answer with the [Company Playbooks](#/process/companies).

## Cross-links

- Deep-dives (the raw technical material behind every answer above): [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [Phoenix](#/resume/phoenix-mask-refinement) · [FaceSign](#/resume/facesign) · [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM/Agents](#/resume/grounded-vlm-agents) · [Spatial-Reasoning Agent (under review)](#/resume/neurips26-spatial-reasoning)
- [Your CV → Interview Map](#/resume/overview) — which project to lead with, per company
- [STAR & The Story Bank](#/behavioral/star) — behavioral framing for the failure/conflict/ownership questions
- [Company Playbooks](#/process/companies) — what each target company probes, to re-point every answer
