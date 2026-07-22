# Your Resume → Interview Map

<div class="tag-row"><span class="tag">deep-dive round</span><span class="tag">research narrative</span><span class="tag">perception → grounding</span><span class="tag">NAVER Cloud · KAIST</span></div>

> [!TIP] What this part is for
> In the **research deep-dive round**, interviewers pick lines off your CV and drill until they hit bedrock: *"Why this design? What breaks it? What's the honest limitation?"* This page is the router. Each CV line maps to the questions it invites, the deep-dive chapter that rehearses them, and the topical chapter that supplies the textbook backing. Rehearse the arc, not just the papers — a coherent story is worth more than any single number.

## The one-sentence story

You build **perception that is accurate *and* deployable**, then extend it toward **verifiable, grounded multimodal reasoning**. Concretely: segmentation/detection/matting → label-efficient and continual learning → the perceptual *tool layer* beneath grounded VLMs and visual reasoning agents. The resume directly records several product touchpoints and ownership claims: building the FaceSign anti-spoofing model, independently developing on-device segmentation, developing the foreground API's model and dataset, and integrating ZIM into CLOVA-X. Verify only the more detailed system, data, and deployment boundaries—and the disclosure scope—beyond those sentences.

## Research narrative arc

```mermaid
flowchart LR
  A["<b>Act 1 — Perception</b><br/>segmentation · detection<br/>matting · face"] --> B["<b>Act 2 — Label-efficient &amp; continual</b><br/>weak/semi-sup · point sup.<br/>class-incremental"]
  B --> C["<b>Act 3 — Foundation → grounded &amp; agentic</b><br/>zero-shot matting foundation (ZIM)<br/>grounded VLM · visual agents"]
  A -.->|"DRS, TricubeNet, FaceSign"| A
  B -.->|"SSUL, BESTIE, PointWSSIS, ECLIPSE"| B
  C -.->|"ZIM (foundation) · grounded VLM · under-review work"| C
  style A fill:#0ea5e9,color:#fff
  style B fill:#12a150,color:#fff
  style C fill:#e0533f,color:#fff
```

The through-line to say out loud: *"I kept asking how to get correct pixels/regions for less — less label, less compute, less retraining — and that pushed me from supervised masks, to weak/continual signals, to promptable foundations, and now to agents that must **know when their perception is wrong**."*

## Project timeline

<figure>
<svg viewBox="0 0 760 250" role="img" aria-label="Timeline of publications and products 2021-2026" style="max-width:100%;height:auto;font-family:inherit">
  <line x1="40" y1="130" x2="720" y2="130" stroke="currentColor" stroke-width="1.5" opacity="0.4"/>
  <!-- year ticks -->
  <g fill="currentColor" font-size="12" text-anchor="middle" opacity="0.75">
    <text x="70" y="150">2021</text>
    <text x="200" y="150">2022</text>
    <text x="330" y="150">2023</text>
    <text x="460" y="150">2024</text>
    <text x="590" y="150">2025</text>
    <text x="700" y="150">2026</text>
  </g>
  <!-- research (above line) -->
  <g font-size="11" text-anchor="middle">
    <circle cx="70"  cy="130" r="5" fill="#0ea5e9"/><text x="70"  y="112" fill="#0ea5e9">DRS · SSUL</text>
    <circle cx="200" cy="130" r="5" fill="#12a150"/><text x="200" y="112" fill="#12a150">BESTIE</text>
    <circle cx="200" cy="130" r="5" fill="#0ea5e9"/><text x="200" y="96"  fill="#0ea5e9">TricubeNet</text>
    <circle cx="330" cy="130" r="5" fill="#12a150"/><text x="330" y="112" fill="#12a150">PointWSSIS</text>
    <circle cx="460" cy="130" r="5" fill="#12a150"/><text x="460" y="112" fill="#12a150">ECLIPSE</text>
    <circle cx="460" cy="130" r="5" fill="#12a150"/><text x="460" y="96"  fill="#0ea5e9">EResFD · WSSHM</text>
    <circle cx="590" cy="130" r="6" fill="#e0533f"/><text x="590" y="110" fill="#e0533f" font-weight="700">ZIM ★Highlight</text>
    <circle cx="680" cy="130" r="5" fill="#e0533f"/><text x="680" y="112" fill="#e0533f">Phoenix · ECCV'26</text>
    <circle cx="720" cy="130" r="5" fill="#e0533f"/><text x="720" y="96" fill="#e0533f">Visual agent*</text>
  </g>
  <!-- products (below line) -->
  <g font-size="11" text-anchor="middle" fill="#6366f1">
    <rect x="120" y="170" width="120" height="20" rx="4" fill="#6366f1" opacity="0.15"/><text x="180" y="184">FaceSign (gov-certified)</text>
    <rect x="300" y="196" width="150" height="20" rx="4" fill="#6366f1" opacity="0.15"/><text x="375" y="210">On-device seg · FG-seg API</text>
    <rect x="520" y="170" width="150" height="20" rx="4" fill="#6366f1" opacity="0.15"/><text x="595" y="184">CLOVA-X Image Editing</text>
  </g>
  <text x="40" y="30" font-size="12" font-weight="700" fill="#0ea5e9">● perception</text>
  <text x="180" y="30" font-size="12" font-weight="700" fill="#12a150">● label-efficient/continual</text>
  <text x="430" y="30" font-size="12" font-weight="700" fill="#e0533f">● grounded/agentic</text>
  <text x="620" y="30" font-size="12" font-weight="700" fill="#6366f1">▬ products</text>
</svg>
<figcaption>Publications above the line, product touchpoints below. The authors' public page lists Phoenix as ECCV 2026; the *visual-agent paper is under review at NeurIPS 2026, so never describe it as accepted.</figcaption>
</figure>

## CV line → questions it invites → what to review

| CV line | What they'll drill | Deep-dive | Topical backing |
| --- | --- | --- | --- |
| **ZIM** — zero-shot matting foundation, ICCV'25 Highlight | Why SAM fails at matting; data engine (SGA/STL); decoder + masked attention; MicroMat-3K; why it's a *Highlight* | [ZIM](#/resume/zim) | [Matting](#/cv/matting), [Foundation Models](#/cv/foundation-models) |
| **ECLIPSE** — continual panoptic, VPT | Why panoptic continual is hard; freeze + prompts vs KD/replay; logit manipulation; plasticity gap | [ECLIPSE](#/resume/eclipse) | [Continual Learning](#/cv/continual-learning) |
| **PointWSSIS / BESTIE** — point & weakly-sup instance seg | Proposal bottleneck; point vs image-level; MaskRefineNet; semantic drift; budget–AP Pareto | [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) | [Weak & Semi-Supervised](#/cv/weak-semi-supervised) |
| **Phoenix** — adversarial mask refinement, ECCV'26 | Limits of morphological noise; adversarial noise generation (AMP); tri-directional contrastive learning (CMRL); self-improvement; up to +16.1 AP points reported under the paper's weak/semi refinement protocols | [Phoenix](#/resume/phoenix-mask-refinement) | [Segmentation](#/cv/segmentation), [Weak & Semi-Supervised](#/cv/weak-semi-supervised) |
| **DRS / SSUL** (earlier) | Saliency-guided WSSS; unknown-label class-incremental seg | [PointWSSIS & BESTIE](#/resume/pointwssis-bestie), [ECLIPSE](#/resume/eclipse) | [Weak & Semi-Supervised](#/cv/weak-semi-supervised), [Continual Learning](#/cv/continual-learning) |
| **FaceSign** — built anti-spoofing model for a gov-certified service | Model-level ownership vs system boundary; threat model (print/replay/3D mask/injection); APCER/BPCER; RGB vs depth | [FaceSign](#/resume/facesign) | [Detection](#/cv/detection) |
| **On-device human seg** — independently developed, ~10ms mobile CPU, ONNX | Quality–latency levers actually used; measurement protocol; ONNX export pitfalls; team interfaces | [On-Device Seg](#/resume/on-device-segmentation) | [Efficiency](#/foundations/mixed-precision-efficiency), [Segmentation](#/cv/segmentation) |
| **Foreground-seg API** — developed model and dataset; internal serving and a three-API internal comparison | Data curation; boundary quality; comparison protocol, domain, date, and approved disclosure scope | [ZIM](#/resume/zim), [On-Device Seg](#/resume/on-device-segmentation) | [Matting](#/cv/matting) |
| **Grounded Multimodal AI** *(ongoing)* | Why grounding; verifiability vs hallucination; region queries | [Grounded VLM/Agents](#/resume/grounded-vlm-agents) | [Grounding & Region Reasoning](#/vlm/grounding), [VLM Pretraining](#/vlm/pretraining) |
| **Visual Reasoning Agents** *(ongoing, under review at NeurIPS 2026)* | Training-free program synthesis; silent failure → typed diagnosis → repair; 3D spatial | [Grounded VLM/Agents](#/resume/grounded-vlm-agents) | [Visual Agents](#/vlm/visual-agents), [Agentic AI](#/llm/agents) |
| **EResFD / TricubeNet** (co/first-author) | Lightweight face detection; kernel-based oriented detection | [FaceSign](#/resume/facesign) | [Detection](#/cv/detection) |

## How to use this in the room

<div class="proscons"><div><div class="pros-t">Do</div>
Open with the <b>30-second pitch</b>, then let them steer. Volunteer the <b>limitation</b> before they find it — it reads as maturity. Anchor claims with the <b>3 numbers</b> that matter per project. Connect research → product when you can.
</div><div><div class="cons-t">Don't</div>
Recite the abstract. Fabricate internal metrics. For FaceSign, on-device work, and the foreground API, <b>verify the approved disclosure scope first</b>; if a detail cannot be shared, discuss only the evaluation design and trade-offs. Do not claim the ongoing NeurIPS 2026 paper is accepted. Do not exaggerate that separate products share a model or data when they do not.
</div></div>

## The deep-dives

<div class="card-grid">
  <a class="card" href="#/resume/interview-stage-answers"><div class="card-emoji">🗣️</div><div class="card-title">Stage-by-Stage Sample Answers</div><div class="card-desc">Clickable speaking drafts grounded in the current resume, from recruiter and HM screens through technical, system-design, job-talk, and behavioral rounds.</div></a>
  <a class="card" href="#/resume/predicted-questions"><div class="card-emoji">🎯</div><div class="card-title">Predicted Questions & Answer Drafts</div><div class="card-desc">A question bank predicted from this resume. Answers are drafts until corrected for the actual role, evidence, and disclosure scope.</div></a>
  <a class="card" href="#/resume/zim"><div class="card-emoji">✂️</div><div class="card-title">ZIM</div><div class="card-desc">Zero-shot image matting foundation. SAM → soft masks via a data engine + hierarchical decoder. ICCV'25 Highlight.</div></a>
  <a class="card" href="#/resume/eclipse"><div class="card-emoji">🌗</div><div class="card-title">ECLIPSE</div><div class="card-desc">Continual panoptic segmentation with visual prompt tuning. Distillation-free, ~1.3% trainable params.</div></a>
  <a class="card" href="#/resume/pointwssis-bestie"><div class="card-emoji">📍</div><div class="card-title">PointWSSIS & BESTIE</div><div class="card-desc">Point- and image-level supervised instance segmentation. Attacking the proposal bottleneck.</div></a>
  <a class="card" href="#/resume/phoenix-mask-refinement"><div class="card-emoji">🔥</div><div class="card-title">Phoenix (ECCV'26)</div><div class="card-desc">Mask refinement through adversarial noise generation (AMP) and tri-directional contrastive learning (CMRL). Up to +16.1 AP points under the paper's protocols.</div></a>
  <a class="card" href="#/resume/facesign"><div class="card-emoji">🛡️</div><div class="card-title">FaceSign</div><div class="card-desc">Government-certified face anti-spoofing in production. Threat models & compliance framing.</div></a>
  <a class="card" href="#/resume/on-device-segmentation"><div class="card-emoji">⚡</div><div class="card-title">On-Device Seg</div><div class="card-desc">~10ms mobile-CPU human segmentation via ONNX. Frame-budget engineering.</div></a>
  <a class="card" href="#/resume/grounded-vlm-agents"><div class="card-emoji">🧭</div><div class="card-title">Grounded VLM & Agents</div><div class="card-desc">Ongoing: verifiable grounding + training-free visual reasoning agents that diagnose their own failures.</div></a>
  <a class="card" href="#/resume/neurips26-spatial-reasoning"><div class="card-emoji">🔺</div><div class="card-title">Spatial-Reasoning Agent (Under Review)</div><div class="card-desc">Under review at NeurIPS 2026: typed diagnosis + program repair for 3D spatial reasoning. Limit comparisons to the submitted manuscript's protocol.</div></a>
</div>

## Cheat-sheet — headline facts

| Fact | Value |
| --- | --- |
| Publications / citations / h-index | 14+ · 572 · 9 *(values in the current resume snapshot; attach a lookup date and recheck before applying)* |
| First/co-first-author papers | 7 *(including CVPR, ICCV, ECCV, NeurIPS, AAAI, and WACV; recheck author notation)* |
| Signature honor | ZIM, ICCV 2025 **Highlight** *(cite only the official designation)* |
| Affiliation | Applied Scientist, NAVER Cloud (5+ yrs) · Ph.D. candidate, KAIST MLAI (Sung Ju Hwang) |
| Prior advisor (M.S.) | Prof. Junmo Kim, KAIST SIIT |
| Arc | perception → label-efficient/continual → grounded VLM/visual agents |
| Golden rule | Public facts only; verify the disclosure scope of internal metrics; state ongoing work with its status and protocol |

## Cross-links
- Speaking by round: [Stage-by-Stage Resume Answers](#/resume/interview-stage-answers) · [Predicted Questions & Answers](#/resume/predicted-questions)
- Home turf: [Segmentation](#/cv/segmentation) · [Object Detection](#/cv/detection) · [Image Matting](#/cv/matting)
- Efficiency & label: [Weak & Semi-Supervised](#/cv/weak-semi-supervised) · [Continual Learning](#/cv/continual-learning) · [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency)
- Frontier: [Grounding & Region Reasoning](#/vlm/grounding) · [Visual Reasoning Agents](#/vlm/visual-agents) · [Agentic AI & Tool Use](#/llm/agents)
