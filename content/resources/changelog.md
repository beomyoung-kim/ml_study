# Changelog

<div class="tag-row"><span class="tag">living document</span><span class="tag">versioned</span><span class="tag">append-only</span></div>

> [!TIP] Why this page exists
> Interview prep goes stale fast — a 2026 candidate who talks like it's 2023 reads as behind. **This book is a living document**, updated as the frontier moves. This page is the record of what changed and when, so you can tell at a glance whether what you studied last month is still current. Every entry is dated; the newest sits at the top.

> [!IMPORTANT] Currency
> Content is current to **July 2026**. Model names, benchmarks, and "SOTA" claims decay — always cross-check the newest numbers against primary sources (see [Curated Resources → Track the frontier](#/resources/open-source)). Where a fact is time-sensitive, the chapters hedge deliberately; that hedging is a feature, not a gap.

## Conventions

- **Append-only.** Never rewrite or delete a past entry — history is the point. To reflect a change, add a *new* dated entry at the top.
- **Reverse chronological.** Newest entry first. Each entry is an `## YYYY-MM-DD — title` section.
- **Semantic-ish grouping.** Within an entry, group notes under `**Added**`, `**Changed**`, `**Fixed**`, and `**Removed**` (borrowing [Keep a Changelog](https://keepachangelog.com/) conventions). Omit empty groups.
- **One line per change.** Link the affected chapter with the hash router (e.g. `[Segmentation](#/cv/segmentation)`) so a reader can jump straight to it.
- **Honesty tags carry over.** If an entry records a claim whose certainty matters, tag it *(verifiable)* / *(defensible opinion)* / *(speculative)* per [AUTHORING.md](#/resources/open-source).

---

## 2026-07-19 — Architectures deep-dive, loss theory & visual explainers

**Added**
- [Architectures](#/foundations/architectures): depthwise-separable FLOP derivation (the ~9× saving, *why*); residual-connection **gradient proof** of why vanishing stops; activation **formula table** beside the live widget; full **RNN/LSTM/GRU** section (unrolled diagram, cell-state-highway SVG, GRU equations, pros/cons); an **original SVG recreation of the Transformer encoder–decoder**; **positional-encoding** intuition (order-blindness, absolute vs relative, sinusoid fingerprint); and a **Mamba/SSM deep-dive** (selective SSM, O(1) memory / no KV cache, pros/cons vs Transformer).
- [Losses & Gradients](#/ml-coding/losses-gradients): "**Why cross-entropy?**" — CE vs KL ($CE=H(p)+D_{KL}$), BCE vs CE, and why not L1/L2 for classification (vanishing-gradient math + a loss-curve figure).
- [Curated Open-Source Resources](#/resources/open-source): "**Visual, animated & interactive explainers**" directory (conv GIFs, CNN/Transformer Explainers, Illustrated Transformer, 3Blue1Brown, colah's LSTMs, distill momentum, positional-encoding, A Visual Guide to Mamba…), links verified live.

**Changed**
- Korean (`.ko.md`) siblings of all three chapters updated to match.

---

## 2026-07-18 — Code lab rollout (all coding + ML-coding)

**Added**
- The interactive **code lab** now covers **every coding pattern chapter and every ML-coding chapter** — **168 runnable labs** total (English + Korean), each with starter code, a hidden reference **Solution**, and test cases graded live in the browser. Coding labs link out to LeetCode; ML-coding labs load **NumPy** and grade with `numpy.allclose`.

**Changed**
- Code-lab widget upgraded: grading moved into an in-Pyodide Python harness (exact / order-insensitive / numeric-closeness) that can lazy-load packages (numpy). Class-based, training-loop, and plotting snippets were intentionally kept as static reference code.
- Every reference solution is verified to pass its own tests (168/168) before shipping.

---

## 2026-07-18 — Interactive code lab (pilot)

**Added**
- A **runnable Python code lab** (Pyodide, lazy-loaded): write a solution in the browser, hit **Run tests** to check it against test cases (pass/fail), and reveal a reference **Solution** — attempt-first practice. Piloted on [Two Pointers & Sliding Window](#/coding/two-pointers-sliding-window) (6 problems, EN + KO) with LeetCode links. Powered by a new `data-widget="code"` widget in `assets/widgets.js`.

**Planned**
- Roll the code lab out to the remaining coding pattern chapters and the ML-coding chapters after the pilot's UX is confirmed.

---

## 2026-07-18 — Korean / English language toggle

**Added**
- **한 / EN language toggle** in the top bar (next to the theme toggle) — switches the whole book between English and Korean, persists your choice, and defaults to English. Korean mode keeps technical terms, code, and math in English by design.
- **Korean translations (`.ko.md`) for all 83 chapters**, rendered natively (KaTeX / Mermaid / callouts / Q&A preserved). A chapter with no translation falls back to English with a small notice, so the book never breaks.

**Changed**
- Engine (`assets/app.js`) fetches `<chapter>.ko.md` in Korean mode with English fallback; UI chrome (sidebar, prev/next, search, footer) is localized. Manifest (`assets/book.js`) gained Korean part/chapter titles (`titleKo`). See `AUTHORING.md` for how to add or update a translation.

---

## 2026-07-18 — PEFT + deeper RL detail

**Added**
- **PEFT section** in [Post-Training & Alignment](#/llm/alignment) — full FT vs parameter-efficient tuning, LoRA (with the $W_0+\frac{\alpha}{r}BA$ math and why $B$ inits to zero), a QLoRA/DoRA/adapters/prefix/prompt/IA³ comparison table, a "which to pick" guide, and a LoRA-vs-full-FT Q&A. Cross-linked to [ECLIPSE](#/resume/eclipse) (visual prompt tuning) and [VLM practical](#/vlm/practical).

**Changed**
- Enriched **GRPO** in [Post-Training & Alignment](#/llm/alignment) with the full clipped-surrogate + KL objective and per-token importance ratio, and how Dr. GRPO / GSPO modify it. Added PEFT/LoRA/QLoRA rows to the cheat-sheet.
- (RLVR, RLHF, SFT, DPO, GRPO were already covered in depth — this pass filled the PEFT gap and added method-level RL math.)

---

## 2026-07-17 — CV deep-dive: NeurIPS'26 spatial-reasoning agent

**Added**
- [Deep-Dive: Spatial-Reasoning Agent (NeurIPS'26)](#/resume/neurips26-spatial-reasoning) — redacted public write-up of the under-review submission (mechanism, architecture, ablation insight, limitations, hard-follow-up Q&A). Codename/exact numbers held offline during double-blind review. Cross-linked from [Grounded VLM/Agents](#/resume/grounded-vlm-agents) and the [CV overview](#/resume/overview).

---

## 2026-07-17 — Initial comprehensive edition

Full rebuild as an online book — a client-side Markdown SPA replacing the prior scratch notes. All parts written and current to July 2026.

**Added**
- **Getting Started** — [How to Use This Book](#/start/how-to-use), [The 2026 Landscape](#/start/landscape-2026), [Prep Plan](#/start/prep-plan): the frontier framing (reasoning/RLVR, native multimodal, MoE, agents, test-time compute) and a time-boxed study plan.
- **Foundations** — [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus), [Probability & Statistics](#/foundations/probability-statistics), [Architectures](#/foundations/architectures): the math and model-family bedrock.
- **Coding** — [Strategy](#/coding/strategy), [Patterns](#/coding/patterns), [Arrays & Strings](#/coding/arrays-strings), [Binary Search](#/coding/binary-search), [Trees & BST](#/coding/trees-bst): a pattern-first coding scaffold.
- **ML Coding** — [Intro](#/ml-coding/intro), [Conv & Pooling](#/ml-coding/conv-pooling), [NMS & IoU](#/ml-coding/nms-iou): implement-from-scratch ML primitives.
- **Computer Vision** — [Segmentation](#/cv/segmentation): semantic / instance / panoptic, metrics, and modern foundation models.
- **LLM** — [Fundamentals](#/llm/fundamentals): transformers, attention, tokenization, decoding, and the post-training stack.
- **Process** — [Interview Pipeline](#/process/pipeline), [Recruiter & Hiring Manager](#/process/recruiter-hm): the loop structure and non-technical rounds.
- **Resources** — [Curated Open-Source Resources](#/resources/open-source), [Glossary](#/resources/glossary), and this changelog.

**Changed**
- Ported and upgraded the legacy `resources.md` link list into an annotated directory with "best for" + "steal this" notes, an "if you only use 5" shortlist, and a frontier-tracking section.

**Notes**
- Register targets a **PhD-level Applied/Research Scientist** in CV / VLM / visual agents, interviewing at Meta, NVIDIA, Apple, Adobe, ByteDance, Mistral, and Microsoft.
- Benchmark figures for the newest (2025–2026) models are frequently vendor-reported; chapters favor mechanisms/capabilities over exact scores and hedge accordingly. *(defensible opinion)*

---

## How to add an entry

When you update a chapter, record it here — future-you (and any collaborator) needs the trail.

1. **Write or edit the chapter** `.md` under `content/**`, following [AUTHORING.md](#/resources/open-source) exactly (voice, chapter shape, callouts, diagrams, cross-links).
2. **Register a new chapter** (if any) by adding one line to the correct part in `assets/book.js` — `{ id, title, file }`. Existing chapters need no change.
3. **Prepend a dated entry** at the top of this file, above the previous one, using the `## YYYY-MM-DD — title` format and the Added/Changed/Fixed/Removed groups. Link every touched chapter.
4. **Bump currency** — if the edit moves the "current to" date, update the currency callout near the top of this page and, if relevant, [The 2026 Landscape](#/start/landscape-2026).
5. **Keep it append-only** — do not edit past entries; supersede them with a new one.

> [!NOTE] Contributing
> The two files that define the book are the content (`content/**.md`) and the table of contents (`assets/book.js`). The full style contract — voice, required chapter shape, available Markdown features (callouts, Q&A accordions, KaTeX, Mermaid, widgets, SVG figures, cards, badges), and the don'ts — lives in `AUTHORING.md`. Read it before writing, then log your change here.

**Related:** [How to Use This Book](#/start/how-to-use) · [Curated Open-Source Resources](#/resources/open-source) · [Glossary](#/resources/glossary)
