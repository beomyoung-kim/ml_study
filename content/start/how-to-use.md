# How to Use This Book

> [!TIP] The 30-second version
> Start with the part that matches your **weakest axis**, and use search (`/`) liberally. Most concept chapters provide Q&A, follow-ups, and a cheat sheet you can revisit on interview day. Hub and reference chapters prioritize navigation.

## Who this is for

This book is written primarily for candidates interviewing for **Research Scientist (RS)**, **Applied Scientist (AS)**, and **ML Engineer / MLE-research** roles at big-tech labs and AI companies. You can begin with the foundational chapters, but the goal is to make your knowledge *interview-shaped*: quick to recall, structured out loud, and precise about assumptions and limits.

It contains especially many **computer vision, multimodal, VLM, and agent** examples. The personalized resume packet and examples in this book concentrate on those areas, on top of a general learning path. If you work primarily in NLP or systems, focus on the coding, foundations, system-design, research, and behavioral parts.

## The mental model

There are four things a loop measures. Map every chapter to one:

| Axis | What they're really testing | Parts |
| --- | --- | --- |
| **Coding** | Can you turn a spec into correct, clean code under time pressure while thinking aloud? | Coding · ML Coding From Scratch |
| **ML depth & breadth** | Do you understand *why* things work, can you go deep on your specialty and stay credible everywhere else? | Foundations · CV · LLM/VLM/Agents |
| **System design** | Can you scope an ambiguous problem, make defensible trade-offs, and reason about data/serving/scale? | ML System Design |
| **Research & behavioral** | Can you explain your work's impact, think like a scientist, and collaborate? | Research Interviews · Behavioral · CV Deep-Dives |

## How each chapter is built

- **Lead-in** framing why the topic matters *in an interview*, not just in the abstract.
- **Diagrams and interactive widgets** — attention heatmaps, gradient-descent animations, precision/recall sliders — because a picture you can manipulate sticks.
- **Q&A blocks** in commonly used interview-question formats, with a *short answer* (what you say first) and a *deep answer* (where you go if they push).
- **"Follow-ups"** — the questions that come *after* your first answer. Interviews are won or lost here.
- **A cheat-sheet** at the end: the 5–10 facts to have on the tip of your tongue.

Try the widgets — they're live:

<div class="widget" data-widget="activation"></div>

## Reading paths

<div class="card-grid">
  <a class="card" href="#/start/prep-plan"><div class="card-emoji">📅</div><div class="card-title">I have a loop soon</div><div class="card-desc">Choose the 2-, 4-, or 8-week path that fits your remaining time, then prioritize by the actual loop and your weakest axis.</div></a>
  <a class="card" href="#/foundations/what-is-ml"><div class="card-emoji">🌱</div><div class="card-title">I want to review the basics</div><div class="card-desc">Start with machine-learning problem formulation; if CV is new to you, continue to image and tensor fundamentals.</div></a>
  <a class="card" href="#/coding/strategy"><div class="card-emoji">⌨️</div><div class="card-title">Coding is my weak spot</div><div class="card-desc">Strategy → patterns → drill. The ML-from-scratch part is the research-role twist.</div></a>
  <a class="card" href="#/llm/reasoning"><div class="card-emoji">🤖</div><div class="card-title">I'm behind on 2026</div><div class="card-desc">Reasoning, alignment, agents, and the VLM chapters get you current fast.</div></a>
  <a class="card" href="#/research/job-talk"><div class="card-emoji">📄</div><div class="card-title">It's a research role</div><div class="card-desc">If the loop emphasizes a job talk or deep dive, prioritize that preparation. Confirm the actual round composition first.</div></a>
</div>

### Study mode and interview mode use different orders

- **Study mode:** Follow the table-of-contents order. Re-derive equations by hand, rewrite `from scratch` code in a blank file, and verify it on small inputs.
- **ML/CV beginner:** Read [What Is Machine Learning?](#/foundations/what-is-ml) → [Math Foundations](#/foundations/linear-algebra-calculus) → [Neural Networks: The Basics](#/foundations/neural-networks-basics), then begin vision with [Image Fundamentals](#/cv/image-basics).
- **Interview mode:** First use the [readiness scorecard](#/start/prep-plan) to find your weakest axis, then answer that chapter's questions **out loud**. Allocate more time to retrieval, mocks, and error correction than to reading.
- **Personalization mode:** After learning the general concepts, connect them to your own project claims in the [resume map](#/resume/overview), then rehearse the [stage-by-stage sample answers](#/resume/interview-stage-answers) aloud in 30- and 90-second versions. Do not memorize the book's wording; replace it with evidence you are permitted to disclose.

## Conventions

- **Language:** Explanations are written in Korean, while technical terms commonly used in papers and interviews are shown in English or left untranslated. Practice answers again in the language your target company will use for the interview.
- **Three evidence layers:** Read textbook-stable facts, date- and protocol-sensitive frontier snapshots, and claims grounded in personal records or PDFs as separate layers. Reconfirm sources and disclosure boundaries for the second and third layers.
- **Table-of-contents badges:** `primer` marks introductory prerequisites, `core` core concepts, `lab` hands-on implementation, `hub` day-of routing, `practice` answer rehearsal, and `2026` a date-sensitive frontier snapshot.
- **Math** renders with KaTeX (`$…$`) and **diagrams** with Mermaid. Separate from **executable code**, `conceptual code` and `PyTorch-style pseudocode` explain data flow; long supporting code is collapsible. Code blocks include syntax highlighting and a copy button, and callouts flag tips/warnings.
- **Cross-links** connect related chapters — follow them; the graph *is* the knowledge.

> [!WARNING] Don't memorize — internalize
> Reciting this book verbatim will read as canned. Use it to build a *model* you can reason from, then practice out loud (ideally in mock interviews). The [Playbook](#/playbook/communication) covers delivery.
