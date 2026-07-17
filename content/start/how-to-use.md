# How to Use This Book

> [!TIP] The 30-second version
> Read the part that matches your **weakest axis** first. Use search (`/`) liberally. Every chapter ends with a compressed cheat-sheet you can review the morning of an interview.

## Who this is for

This book targets candidates interviewing for **Research Scientist (RS)**, **Applied Scientist (AS)**, and senior **ML Engineer / MLE-research** roles at big-tech labs and frontier AI companies. It assumes you already know ML — the goal is to make your knowledge *interview-shaped*: fast to recall, well-structured out loud, and honest about limits.

It is especially tuned for a **computer-vision / multimodal** background moving toward **VLMs and agents**, because that's both where the field is going and where the author works. If you're a pure-NLP or systems person, the coding, foundations, system-design, research, and behavioral parts are all still directly useful.

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
- **Q&A blocks** in the exact form an interviewer asks, with a *short answer* (what you say first) and a *deep answer* (where you go if they push).
- **"Follow-ups"** — the questions that come *after* your first answer. Interviews are won or lost here.
- **A cheat-sheet** at the end: the 5–10 facts to have on the tip of your tongue.

Try the widgets — they're live:

<div class="widget" data-widget="activation"></div>

## Reading paths

<div class="card-grid">
  <a class="card" href="#/start/prep-plan"><div class="card-emoji">📅</div><div class="card-title">I have a loop soon</div><div class="card-desc">Go to the 8-week plan, then triage by company playbook and your weakest axis.</div></a>
  <a class="card" href="#/coding/strategy"><div class="card-emoji">⌨️</div><div class="card-title">Coding is my weak spot</div><div class="card-desc">Strategy → patterns → drill. The ML-from-scratch part is the research-role twist.</div></a>
  <a class="card" href="#/llm/reasoning"><div class="card-emoji">🤖</div><div class="card-title">I'm behind on 2026</div><div class="card-desc">Reasoning, alignment, agents, and the VLM chapters get you current fast.</div></a>
  <a class="card" href="#/research/job-talk"><div class="card-emoji">📄</div><div class="card-title">It's a research role</div><div class="card-desc">The job talk and deep-dive prep matter more than another LeetCode medium.</div></a>
</div>

## Conventions

- **Language:** English-primary, since your real answers will be in English. Technical terms, math, and code are, of course, universal.
- **Honesty markers:** claims are tagged when they're *verifiable* vs. a *defensible opinion* vs. *speculative direction*. In interviews, calibrated confidence beats false certainty.
- **Math** renders with KaTeX (`$…$`), **diagrams** with Mermaid, **code** is syntax-highlighted with a copy button, and callouts flag tips/warnings.
- **Cross-links** connect related chapters — follow them; the graph *is* the knowledge.

> [!WARNING] Don't memorize — internalize
> Reciting this book verbatim will read as canned. Use it to build a *model* you can reason from, then practice out loud (ideally in mock interviews). The [Playbook](#/playbook/communication) covers delivery.
