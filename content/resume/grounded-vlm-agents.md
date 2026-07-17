# Deep-Dive: Grounded VLM & Visual Reasoning Agents (Ongoing)

<div class="tag-row"><span class="tag">ongoing</span><span class="tag">NeurIPS 2026 (under review)</span><span class="tag">grounded VLM</span><span class="tag">visual agents</span><span class="tag">training-free</span><span class="tag">framing over specifics</span></div>

> [!DANGER] Ground rules for this chapter
> This work is **unpublished / under review**. Never fabricate methods, numbers, dataset names, or an acceptance decision. Sell it on **motivation, problem definition, position among public trends, and the bridge from my public work (ZIM, ECLIPSE)**. Method specifics are "pre-release." Use the decline-and-redirect script at the end when pushed.

> [!TIP] 30-second pitch
> Two threads. **(1) Grounded Multimodal AI:** tie language reasoning to **pixel/region-level evidence** so a model resolves region-based queries and reasons *verifiably* instead of emitting unsupported text. **(2) Visual Reasoning Agents:** **training-free** agentic program synthesis that dynamically composes specialist vision models into query-specific executable workflows for multi-step **spatial/temporal** reasoning that end-to-end VLMs still fumble. My NeurIPS'26 submission sits in thread 2: a **diagnostic framework for 3D spatial reasoning** that turns *silent perception failures* into **typed diagnoses**, driving targeted **program repair** — aiming to rival frontier VLMs *without task-specific training*.

**Résumé anchors (do not over-specify beyond these):** grounded VLMs connecting language reasoning with pixel/region evidence, resolving region queries, reasoning verifiably; training-free program synthesis building query-specific workflows from specialist models; a 3D-spatial diagnostic framework: silent failures → typed diagnoses → targeted repair. Backing chapters: [Grounding & Region Reasoning](#/vlm/grounding), [Visual Reasoning Agents](#/vlm/visual-agents), [Agentic AI & Tool Use](#/llm/agents), [VLM Pretraining](#/vlm/pretraining).

## The core research question

> *When a language model produces a visually plausible sentence, how do we (a) bind it to pixel/region evidence and executable perception tools, and (b) when a tool is silently wrong, **diagnose and repair** rather than fail silently?*

Two failure modes motivate the two threads:

```mermaid
flowchart TB
  subgraph G ["Grounded Multimodal — fight hallucination"]
    LQ[Language / region query] --> GR["Ground to regions / pixels"]
    GR --> EV[Visual evidence]
    EV --> VA["Verifiable answer"]
  end
  subgraph A ["Visual Agents — fight silent tool failure"]
    Q[Query] --> SYN["Program synthesis (training-free)"]
    SYN --> SP["Specialist tools (detect / depth / seg / matte)"]
    SP --> EX[Execute]
    EX -->|silent wrong output| DIAG["Typed diagnosis"]
    DIAG --> REP["Targeted program repair"]
    REP --> SYN
  end
  style VA fill:#12a150,color:#fff
  style DIAG fill:#e0533f,color:#fff
```

## Public trend map (2025–2026) — where I place myself

| Direction | Public exemplars | One line |
| --- | --- | --- |
| Program synthesis | VisProg, ViperGPT | LLM → code/DSL + vision tools |
| Dynamic API agents | VADAR (CVPR 2025) | agent generates 3D-spatial APIs on the fly |
| Verifier training | VALOR | improve logic/grounding without labels |
| Grounded RL | ViGoRL, MGPO | coordinate/crop loops via RL |
| Pixel-grounded CoT | TerraScope, Pixelis | reasoning with masks/pixel ops |
| Concept segmentation | SAM 3 | text concept → mask/track |
| Diagnostic benches | Spatial457, Omni3D-Bench, SpatialAct | measure spatial / repair gaps |

**My stated position:** on top of specialist perception quality (ZIM/seg) and a label-efficient/continual background, I explore **verifiable grounding** and **failure-diagnosable agents** — *adjacent* to VADAR's dynamic-API/3D framing, but I won't present comparison numbers on unpublished work.

## Predicted deep-dive Q&A (framing answers)

<details class="qa"><summary>Why grounded VLMs — what's wrong with end-to-end?</summary>
<div class="qa-body">

**Short:** End-to-end VLMs hallucinate and give **unsupported descriptions**; even a correct answer can rest on wrong evidence (*spurious success*).

**Deep:** For products (editing, robotics, UI agents) you must *resolve a region* and justify the answer with visual evidence, not prose. Grounding makes the reasoning auditable and lets you detect when the model is right for the wrong reason — which end-to-end accuracy alone hides.
</div></details>

<details class="qa"><summary>Why training-free agents rather than fine-tuning one big VLM?</summary>
<div class="qa-body">

**Short:** Compose and swap specialist tools (including our own matte/seg) without retraining; fine-tuning per task can degrade the foundation/product model.

**Deep:** It mirrors the ZIM lesson — task-specific fine-tuning can erode general capability. Training-free composition keeps each specialist at full strength and lets you upgrade a module without touching the rest. The cost is **tool error**, which is exactly why diagnosis/repair is the research contribution rather than an afterthought.
</div></details>

<details class="qa"><summary>What is a "silent perception failure," and what's a "typed diagnosis"?</summary>
<div class="qa-body">

**Short:** A tool returns a wrong box/mask/depth but raises no exception, so the program finishes and emits a wrong answer *quietly*. A typed diagnosis categorizes *how* it failed so repair can be targeted.

**Deep:** In a synthesized program, the LLM assumes each tool's output is correct. If depth is off or a detection is missing, errors propagate to a confident wrong answer. Turning that into a **typed** failure signal (at the résumé-wording level) lets the system apply a different repair policy per failure class instead of blind re-planning. Specifics are pre-release.
</div></details>

<details class="qa"><summary>Why is 3D spatial reasoning hard for end-to-end VLMs?</summary>
<div class="qa-body">

Metric distances, occlusion, object-centric orientation, and multi-hop spatial composition. SpatialVLM-style models show real limits here. Programs with depth/detection tools help, but **tool-error accumulation** is the crux — hence the diagnostic angle.
</div></details>

<details class="qa"><summary>How does ZIM fit in?</summary>
<div class="qa-body">

Specialist mask/matte quality is a **lower bound** on any editing/agent chain that relies on it, and **Grounded-ZIM** (text → box → matte) is a working prototype of grounded UX. But the ongoing work is *not* just "ZIM again" — ZIM is an **asset** I can plug in and diagnose, not the same paper.
</div></details>

### Hard-pressure follow-ups

<details class="qa"><summary>End-to-end frontier models keep getting better — won't they eat modular agents?</summary>
<div class="qa-body">

They're strong, but four things keep modularity valuable: (1) **precise measurement** (metric 3D), (2) **verifiable evidence**, (3) **module upgrade without retraining**, (4) **diagnostic repair** of failures. In practice a **hybrid** is the default — a frontier reasoner orchestrating diagnosable specialists — and that's where I'm betting.
</div></details>

<details class="qa"><summary>Isn't this just ideas without results?</summary>
<div class="qa-body">

My execution is proven on public, peer-reviewed work (ZIM ICCV Highlight, ECLIPSE, PointWSSIS, BESTIE, SSUL). The ongoing work is under review, so I won't quote numbers or claim acceptance — but I can state the problem definition precisely and place it accurately in the public literature. That distinction *is* the honest answer.
</div></details>

<details class="qa"><summary>How would you evaluate it? (general)</summary>
<div class="qa-body">

Answer accuracy **plus** grounding IoU / pointing accuracy, program executability, failure-type recall, and public 3D-spatial benchmarks (Omni3D-Bench, Spatial457, SpatialAct). Internal benchmark names stay private if unpublished. The point of the eval is to catch *spurious success* — right answer, wrong evidence.
</div></details>

<details class="qa"><summary>What are the risks / likely negative results?</summary>
<div class="qa-body">

LLMs hallucinating tool APIs, infinite re-planning loops, tool-version breakage, 3D scale ambiguity, and evaluation overfitting. Public surveys also flag agents' weak *tool-awareness*. Naming these unprompted is part of a credible pitch.
</div></details>

## The bridge from public work (say this)

- **Perception quality** (ZIM, on-device seg) → the specialist tool layer agents depend on.
- **Label-efficiency / continual** (PointWSSIS, ECLIPSE) → the instinct to avoid task-specific retraining and to upgrade modules incrementally.
- **Safety/verifiability mindset** (FaceSign) → wanting answers backed by evidence, not prose.

## Company-specific one-liners

| Company | One line |
| --- | --- |
| Meta | Grounded, verifiable reasoning + tool-use agents for product VLMs |
| Apple | Region evidence + efficient on-device specialists + alignment |
| Adobe | Region-grounded editing/generation programs |
| ByteDance | Generative FM + controllable perception |
| NVIDIA | Spatial agents for robotics; efficient specialist stacks |

## Guardrails — allowed vs forbidden phrasing

<div class="proscons"><div><div class="pros-t">Allowed</div>
Restating résumé wording; the silent-failure → typed-diagnosis → repair problem framing; naming public lineage (VisProg, ViperGPT, VADAR, ViGoRL, SAM 3); connecting motivation to ZIM/ECLIPSE.
</div><div><div class="cons-t">Forbidden</div>
Any unpublished accuracy/% gain; "we outperform VADAR"-style claims; stating the NeurIPS'26 paper is accepted; internal dataset names or scale.
</div></div>

## Decline-and-redirect script

> *"The method and numbers aren't public yet, so I can't share those. What I can do is give you the precise problem framing — silent perception failure, typed diagnosis, targeted repair — how it sits relative to public work like VADAR and ViperGPT, and what my ZIM/continual research taught me that led here."*

## Honest limitations

- Under review → no verifiable results to cite; sell framing, not outcomes.
- Two threads overlap but have **different failure modes** (hallucination vs. silent tool error) — keep them distinct in the room.
- Training-free composition inherits **tool error**; the whole bet is that diagnosis/repair can manage it.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Thread 1 | Grounded Multimodal — bind language to pixel/region evidence; verifiable region queries |
| Thread 2 | Visual Reasoning Agents — training-free program synthesis from specialist tools |
| NeurIPS'26 (under review) | 3D-spatial diagnostic framework: silent failure → **typed diagnosis** → **program repair** |
| Public lineage | VisProg, ViperGPT, VADAR, ViGoRL/MGPO, TerraScope, SAM 3 |
| Bridge | ZIM/seg (tools) · ECLIPSE/PointWSSIS (no-retrain instinct) · FaceSign (verifiability) |
| Golden rule | Framing over specifics; no numbers; not accepted-yet |

## Cross-links
- Topical: [Grounding & Region Reasoning](#/vlm/grounding) · [Visual Reasoning Agents](#/vlm/visual-agents) · [Agentic AI & Tool Use](#/llm/agents) · [Vision-Language Pretraining](#/vlm/pretraining) · [Video-Language Models](#/vlm/video)
- Deep-dives: [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [On-Device Seg](#/resume/on-device-segmentation) · back to the [CV → Interview Map](#/resume/overview)
