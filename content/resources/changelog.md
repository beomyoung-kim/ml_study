# Changelog

<div class="tag-row"><span class="tag">living document</span><span class="tag">versioned</span><span class="tag">append-only</span></div>

> [!TIP] Why this page exists
> Interview preparation goes stale fast — a 2026 candidate speaking as if it were 2023 looks behind. <strong>This book is a living document</strong>, updated whenever the frontier moves. This page records what changed and when, so you can tell at a glance whether material you studied last month is still current. Every entry is dated, with the newest first.

> [!IMPORTANT] Currency
> Content is current to **July 2026**. Model names, benchmarks, and "SOTA" claims age — always cross-check the latest figures against primary sources (see [Curated Resources → Track the frontier](#/resources/open-source)). Time-sensitive claims are intentionally phrased conservatively throughout the chapters. That restraint is a feature, not a defect.

## Conventions

- **Append-only.** Do not rewrite or delete past entries — the history itself matters. To reflect a change, add a *new* dated entry at the top.
- **Reverse chronological.** Newest entries come first. Each entry is a \`## YYYY-MM-DD — title\` section.
- **Semantic-ish grouping.** Within an entry, group notes under \`**Added**\`, \`**Changed**\`, \`**Fixed**\`, and \`**Removed**\`, borrowing the [Keep a Changelog](https://keepachangelog.com/) convention. Omit empty groups.
- **One line per change.** Link the affected chapter through its hash route, for example [Segmentation](#/cv/segmentation), so readers can jump straight there.
- **Honesty tags carry through.** When an entry records a claim whose certainty matters, tag it *(verifiable)*, *(defensible opinion)*, or *(speculative)* according to the repository's \`AUTHORING.md\`.

---

## 2026-07-21 — Final QA · resume fact reconciliation · UI hardening

This supplements the UI and translation entry from the same day. It records the final consistency and safety fixes found by integrated validation and a second comparison against `local/resume.pdf`.

**Changed**
- Restored ownership that the resume actually states rather than over-softening it: building the [FaceSign](#/resume/facesign) anti-spoofing model; independently developing [On-Device Segmentation](#/resume/on-device-segmentation) with about 10 ms mobile-CPU inference and ONNX serving; and developing the foreground API's model and dataset with its internal comparison. The text still requires separate verification for whole-system ownership, nonpublic protocols or figures, and specific techniques used.
- Re-personalized [Interview-Stage Example Answers](#/resume/interview-stage-answers) and [Predicted Questions](#/resume/predicted-questions) at that evidence level and synchronized the Korean and English documents together.

**Fixed**
- Fixed translation-function shadowing in mobile overflow tables, skip-link/hash-router collision, HTML injection through unknown routes, stale rendering during rapid route or language changes, search-index language races and duplicate fetches, focus handoff after navigation, lost inline code in callout titles, and light-theme focus/faint-text contrast.
- Final validation passed 105 Korean/English routes, 119 widgets and 448 tests per language, 38 conceptual-code blocks, 30 answer cards, and 180 Python fences in total.

---

## 2026-07-21 — UI · conceptual code · resume answers · English sync

This supplements, without editing, the earlier restructuring record from the same day. The current manifest baseline is <strong>105 chapters · 15 parts</strong>, with both Korean and English documents for every route.

**Added**
- Added <strong>38 conceptual Python/PyTorch blocks</strong> to equation-centered explanations across ML, CV, LLM, VLM, systems, and research. The 21 longer blocks are marked as click-to-expand conceptual code under the convention in [How to Use This Book](#/start/how-to-use).
- Added [Interview-Stage Example Answers](#/resume/interview-stage-answers), with <strong>30 click-to-expand example answers</strong> covering the path from recruiter and HM conversations through coding/ML depth, technical deep-dives, system design, job talk, behavioral, and closing. It separates resume verification, public evidence, company-specific substitutions, and items requiring personal confirmation.
- Added a repository validation command that checks omissions, structural parity, internal routes and anchors, code-widget JSON, and residual Korean text across all 105 Korean/English document pairs.

**Changed**
- Reworked the landing hero, the book's location in the sidebar, prose, callouts, tables, figures, code, widgets, answer cards, conceptual code, light/dark contrast, search modal, and mobile drawer into one responsive UI system.
- Synchronized the restructuring, factual corrections, and conceptual code completed in Korean across all 105 English documents, applying the same factual boundaries so translation would not invent personal ownership, internal results, conflicts, or other narratives not guaranteed by the resume.
- Updated <code>AUTHORING.md</code> for a Korean-first editing → English-sync workflow and dedicated answer/concept-code conventions.

**Fixed**
- Fixed query/result mismatch when reopening search, confusion between the full result count and display cap, modals lingering after route changes, unnecessary accessibility text on collapsible cards, and page content remaining focusable behind the mobile drawer.
- Localized static accessibility labels for the sidebar, search, language/theme controls, and table of contents to the active language, and checked horizontal overflow and focus restoration at representative 390–1440px viewports.

---

## 2026-07-21 — Restructuring QA · Korean evidence corrections · renderer hardening

This <strong>supplements, without editing, and supersedes where necessary</strong> the earlier restructuring entry from the same day. The manifest baseline at this point was <strong>104 chapters · 15 parts</strong>.

**Added**
- Expanded the classical-ML baseline map in [What Is Machine Learning?](#/foundations/what-is-ml), the canonical training loop in [Neural Networks: First Steps](#/foundations/neural-networks-basics), and the diffusion-versus-flow-matching distinction in [CNN · RNN · Transformer](#/foundations/architectures).
- Added the canonical linked-list treatment to [Two Pointers · Sliding Window](#/coding/two-pointers-sliding-window), and the canonical backtracking treatment plus drills to [Trees · BST](#/coding/trees-bst).

**Changed**
- Split the learning, interview, and personal-resume paths on the home page and in [How to Use This Book](#/start/how-to-use), and reorganized [Prep Plan](#/start/prep-plan) into **2-, 4-, and 8-week** paths.
- Placed \`questions-to-ask\` in the late-stage part that combines day-of execution, questions, and offers rather than in the generic <strong>Process</strong> part. From-Scratch is framed not as a "lab-only" part that excludes all explanation, but as a section centered on <strong>direct implementation and verification</strong>.
- Reviewed the Korean documents for [The 2026 Landscape](#/start/landscape-2026), [behavioral](#/behavioral/star), and the [Personal Resume Packet](#/resume/overview), replacing date-dependent claims, company stereotypes, unverified ownership/conflict/internal figures, and under-review results with source- and status-aware language. These content corrections were applied only to the Korean documents in this pass.

**Fixed**
- Corrected the attention/KV explanations in [Attention](#/ml-coding/attention) and [LLM Fundamentals](#/llm/fundamentals), and the equations and edge cases in [IoU & NMS](#/ml-coding/nms-iou), [mAP & mIoU](#/ml-coding/metrics-map-miou), and [K-Means](#/ml-coding/kmeans).
- Corrected PPO and outcome reward models in [Alignment](#/llm/alignment), [RoPE](#/ml-coding/positional-encoding-rope), fusion in [VLM Pretraining](#/vlm/pretraining), answer/evidence evaluation in [Grounding](#/vlm/grounding), and MCP transport and evaluation security in [Agents](#/llm/agents), matching current evidence and scope.
- Hardened the renderer for Unicode slugs in Korean headings, duplicate-ID suffixes for repeated headings, percent-encoded Korean anchor restoration, and semantic badge labels.
- Updated asset-URL cache busting so changed assets would not remain stuck in stale browser caches.
- Corrected the uncertainty explanation and ECLIPSE-backbone connection in [Phoenix](#/resume/phoenix-mask-refinement), freezing language in [ECLIPSE](#/resume/eclipse), nonpublic comparison conclusions in the under-review [Spatial-Reasoning Agent](#/resume/neurips26-spatial-reasoning), and the fact-versus-draft boundary in resume-based behavioral answers.
- Rechecked external-resource descriptions, internal hash routes, and Q&A HTML nesting, and added verification warnings to links whose current access was restricted.

---

## 2026-07-21 — Major restructuring: 104 chapters · 15 parts

The whole book was reorganized into **104 chapters across 15 parts**. Every place that states the chapter count should be updated to 104.

**Changed**
- **Expanded 14 parts → 15.** Split the oversized **LLMs · VLMs · Agents** part (18 chapters) into three: **LLM Core** / **LLM Alignment · Reasoning · Agents** / **VLM · Visual Agents**. Each part is now a manageable reading unit.
- Absorbed <strong>Advanced: Scaling & Efficiency (2 chapters)</strong> into the **Advanced (Optional)** section of <strong>ML · DL Foundations</strong>, creating a natural fundamentals → advanced flow.
- Moved <strong>architectures</strong> and <strong>normalization-stability</strong> into <strong>Foundations</strong>, gathering model families and training-stability fundamentals in one place.
- Made **From-Scratch a lab-only part** with runnable code labs; explanatory prose moved into the relevant topic chapters and duplication was removed.
- **Reordered Computer Vision** so the upsampling chapter sits <strong>next to</strong> segmentation, preserving a decoder/upsampling → segmentation flow.
- **Reordered Research** from presenting → job talk, creating a natural path from presentation preparation into the job talk.
- Moved **questions-to-ask** from <strong>Company Playbooks</strong> to <strong>Process</strong>, where it fits process preparation better.

**Fixed**
- Removed duplicated content, consolidated phone-screen and behavioral coverage, unified Korean/English tone, and fixed several facts and links, including broken hash routes, cross-link targets, and ZIM data figures.

---

## 2026-07-19 — RAG/MCP · model naming · extreme images · video paradigms

**Added**
- [Agentic AI & Tool Use](#/llm/agents): an **MCP** (Model Context Protocol) subsection covering M×N→M+N, host/client/server, primitives, and attack surfaces, plus a **RAG** section covering chunk/embed/index/rerank/generate, RAG vs long context vs fine-tuning, failure modes, and agentic RAG.
- [LLM Fundamentals](#/llm/fundamentals) §8: **How to read model names** — Base / Instruct / **Thinking** / -Zero training suffixes, and dense / **\`A3B\`** (MoE active) / **\`E4B\`** (MatFormer effective footprint) size tags.
- [VLM Implementation Details](#/vlm/practical) §3½: **Extreme inputs** — small objects, 1:100 aspect ratios, and ultra-high-resolution images; *why fixed resizing fails*, native-resolution / AnyRes tiling / coarse-to-fine / bucketing / sliding-window methods, and a five-step **ML system-design answer frame**.
- [Video-Language Models](#/vlm/video): **three video paradigms** — sparse sampling → VLM for understanding, **streaming/recurrent memory (SAM 2)**, **clip/window bidirectional** models using 3D convolution or spatiotemporal attention, and tracking by detection.

**Changed**
- Updated the Korean (\`.ko.md\`) versions of all four chapters to match.

---

## 2026-07-19 — CLIP/DINO training · contrastive learning · MLSD cases

**Added**
- [VLM Pretraining](#/vlm/pretraining): "**How CLIP actually trains and performs zero-shot inference**" — dual encoders, roughly 400M pairs, learnable temperature, six-line training pseudocode, and prompt-based zero-shot inference — plus a new **§1.5 Contrastive learning** section covering generalized InfoNCE; a SimCLR / MoCo / CLIP / Triplet table; temperature; classical contrastive and triplet losses; representation collapse; and how non-contrastive BYOL/SimSiam/DINO avoids it.
- [Vision Foundation Models](#/cv/foundation-models): "**How DINO trains**" — self-distillation, an EMA teacher, multi-crop, the objective, **centering + sharpening** to avoid collapse without negatives, iBOT/KoLeo, and a student–teacher diagram.
- [ML System Design Case Studies](#/system-design/case-studies): three new cases — **E · Face authentication & liveness (anti-spoofing)** linked to FaceSign, **F · Recommendation/ranking**, and **G · OCR/document understanding** — plus tag-row and cheat-sheet updates.

**Changed**
- Updated the Korean (\`.ko.md\`) versions of all three chapters to match.

---

## 2026-07-19 — Architecture depth · loss theory · visual material

**Added**
- [Architectures](#/foundations/architectures): a depthwise-separable FLOP derivation showing roughly 9× savings and *why*; a **gradient-perspective proof** of why residual connections alleviate vanishing gradients; an **activation-equation table** beside the live widget; a full **RNN/LSTM/GRU** section with an unrolled diagram, cell-state-highway SVG, GRU equations, and trade-offs; a reconstructed **Transformer encoder–decoder architecture SVG**; positional-encoding intuition covering order blindness, absolute vs relative, and sinusoidal fingerprints; and a **Mamba/SSM deep dive** covering selective SSMs, O(1) memory, no KV cache, and trade-offs against Transformers.
- [Losses & Gradients](#/ml-coding/losses-gradients): "**Why cross-entropy?**" — CE vs KL (\(CE=H(p)+D_{KL}\)), BCE vs CE, and why not to use L1/L2 for classification, with a vanishing-gradient equation and loss-curve figure.
- [Curated Open-Source Resources](#/resources/open-source): a directory of "**visual, animated, and interactive explanations**" — convolution GIFs, CNN/Transformer Explainer, The Illustrated Transformer, 3Blue1Brown, colah's LSTM, Distill momentum, positional encoding, A Visual Guide to Mamba, and more — with links checked live.

**Changed**
- Updated the Korean (\`.ko.md\`) versions of all three chapters to match.

---

## 2026-07-18 — Full code-lab expansion (coding + ML coding)

**Added**
- Interactive <strong>code labs</strong> now cover <strong>every coding-pattern chapter and every ML-coding chapter</strong>: **168 runnable labs** in English and Korean. Each lab includes starter code, a hidden reference **Solution**, and test cases graded instantly in the browser. Coding labs link to LeetCode; ML-coding labs load <strong>NumPy</strong> and grade with \`numpy.allclose\`.

**Changed**
- Upgraded the code-lab widget: grading now runs through a Python harness inside Pyodide with exact, order-insensitive, and numeric-approximate modes, and packages such as NumPy can load lazily. Class-based, training-loop, and plotting snippets intentionally remain static reference code.
- Verified before deployment that every reference solution passes its own tests (168/168).

---

## 2026-07-18 — Interactive code labs (pilot)

**Added**
- **Runnable Python code labs** using lazily loaded Pyodide: write a solution in the browser, grade it against test cases with <strong>Run tests</strong>, see pass/fail, and reveal a reference <strong>Solution</strong> for attempt-first practice. Piloted in [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) with six problems in English and Korean plus LeetCode links, powered by the new \`data-widget="code"\` widget in \`assets/widgets.js\`.

**Planned**
- Expand code labs to the remaining coding-pattern and ML-coding chapters after validating the pilot UX.

---

## 2026-07-18 — Korean / English language toggle

**Added**
- Added a **한 / EN language toggle** to the top bar beside the theme toggle. It switches the entire book between English and Korean, remembers the selection, and defaults to English. Technical terms, code, and math remain in English even in Korean mode.
- Added **Korean translations (\`.ko.md\`) for all 83 chapters**. KaTeX, Mermaid, callouts, and Q&A render unchanged. Chapters without a translation fall back to English with a notice rather than breaking the book.

**Changed**
- The engine in \`assets/app.js\` fetches \`<chapter>.ko.md\` in Korean mode and falls back to English if absent. The UI — sidebar, previous/next controls, search, and footer — was localized, and Korean part/chapter titles (\`titleKo\`) were added to the manifest in \`assets/book.js\`. See \`AUTHORING.md\` for how to add or update translations.

---

## 2026-07-18 — PEFT + deeper RL detail

**Added**
- Added a **PEFT section** to [Post-Training & Alignment](#/llm/alignment): full fine-tuning vs parameter-efficient tuning; LoRA, including the \(W_0+\frac{\alpha}{r}BA\) equation and why \(B\) starts at zero; a QLoRA/DoRA/adapters/prefix/prompt/IA³ comparison table; a "what to choose" guide; and LoRA-vs-full-FT Q&A. Cross-linked [ECLIPSE](#/resume/eclipse) for visual prompt tuning and [VLM Practical](#/vlm/practical).

**Changed**
- Expanded <strong>GRPO</strong> in [Post-Training & Alignment](#/llm/alignment) with the full clipped-surrogate-plus-KL objective, per-token importance ratios, and how Dr. GRPO and GSPO modify them. Added PEFT/LoRA/QLoRA rows to the cheat sheet.
- RLVR, RLHF, SFT, DPO, and GRPO were already covered in depth; this pass filled the PEFT gap and added method-level RL equations.

---

## 2026-07-17 — CV deep-dive: NeurIPS '26 spatial-reasoning agent

**Added**
- [Deep-Dive: Spatial-Reasoning Agent (NeurIPS '26)](#/resume/neurips26-spatial-reasoning) — a redacted public write-up of an under-review submission, covering mechanism, architecture, ablation insights, limitations, and hard follow-up Q&A. The codename and exact figures remain offline during double-blind review. Cross-linked [Grounded VLM/Agents](#/resume/grounded-vlm-agents) and the [CV Overview](#/resume/overview).

---

## 2026-07-17 — Initial comprehensive edition

A full rebuild into a client-side Markdown SPA book, replacing the original scratch notes. Every part was written and current to July 2026.

**Added**
- **Getting Started** — [How to Use This Book](#/start/how-to-use), [The 2026 Landscape](#/start/landscape-2026), and [Prep Plan](#/start/prep-plan): frontier framing around reasoning/RLVR, native multimodality, MoE, agents, and test-time compute, plus time-budgeted study plans.
- **Foundations** — [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus), [Probability & Statistics](#/foundations/probability-statistics), and [Architectures](#/foundations/architectures): the mathematical and model-family base.
- **Coding** — [Strategy](#/coding/strategy), [Patterns](#/coding/patterns), [Arrays & Strings](#/coding/arrays-strings), [Binary Search](#/coding/binary-search), and [Trees & BST](#/coding/trees-bst): a pattern-first coding spine.
- **ML Coding** — [Intro](#/ml-coding/intro), [Conv & Pooling](#/ml-coding/conv-pooling), and [NMS & IoU](#/ml-coding/nms-iou): ML primitives implemented from scratch.
- **Computer Vision** — [Segmentation](#/cv/segmentation): semantic, instance, and panoptic segmentation, metrics, and modern foundation models.
- **LLM** — [Fundamentals](#/llm/fundamentals): Transformers, attention, tokenization, decoding, and the post-training stack.
- **Process** — [Interview Pipeline](#/process/pipeline) and [Recruiter & Hiring Manager](#/process/recruiter-hm): loop structure and nontechnical rounds.
- **Resources** — [Curated Open-Source Resources](#/resources/open-source), [Glossary](#/resources/glossary), and this changelog.

**Changed**
- Migrated and upgraded the old \`resources.md\` link list into an annotated directory with "best for" and "steal this" notes, an "if you only use 5" summary, and a frontier-tracking section.

**Notes**
- The intended reader is a **PhD-level Applied/Research Scientist** in CV, VLMs, or visual agents interviewing at Meta, NVIDIA, Apple, Adobe, ByteDance, Mistral, or Microsoft.
- Benchmark figures for current 2025–2026 models often come from vendor reports. Chapters prioritize mechanisms and capabilities over exact scores and use appropriately conservative phrasing. *(defensible opinion)*

---

## How to add an entry

When you update a chapter, record it here — your future self and collaborators need the trail.

1. **Write or edit the chapter.** Author the \`.md\` file under \`content/**\`, following the repository's \`AUTHORING.md\` exactly for voice, chapter shape, callouts, diagrams, and cross-links.
2. **Register a new chapter**, if applicable, by adding one line to the correct part in \`assets/book.js\`: \`{ id, title, file }\`. Existing chapters need no change.
3. **Add a dated entry at the top**, above previous entries, using \`## YYYY-MM-DD — title\` and Added/Changed/Fixed/Removed groups. Link every chapter touched.
4. **Update currency.** If the edit advances the "current to" date, update the currency callout at the top of this page and, if necessary, [The 2026 Landscape](#/start/landscape-2026).
5. **Keep it append-only.** Do not edit old entries; supersede them with a new one.

> [!NOTE] Contributing
> The two files that define the book are the content (\`content/**.md\`) and table of contents (\`assets/book.js\`). The full style contract — voice, required chapter shape, supported Markdown features including callouts, Q&A accordions, KaTeX, Mermaid, widgets, SVG figures, cards, and badges, plus what not to do — lives in \`AUTHORING.md\`. Read it before writing, then record the change here.

**Related:** [How to Use This Book](#/start/how-to-use) · [Curated Open-Source Resources](#/resources/open-source) · [Glossary](#/resources/glossary)
