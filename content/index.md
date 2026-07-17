<div class="hero">
  <div class="eyebrow">Research &amp; Applied Scientist · Computer Vision · VLMs · Agents</div>
  <h1>The ML Interview Codex</h1>
  <p class="sub">A comprehensive, continuously-updated field guide for landing a research or applied-scientist role at a top AI lab — from coding and ML foundations to VLMs, agents, system design, the research job talk, and CV-specific deep-dives. Current to <b>July 2026</b>.</p>
  <div class="stat-row">
    <div class="stat"><div class="n">13</div><div class="l">Parts</div></div>
    <div class="stat"><div class="n">70+</div><div class="l">Chapters</div></div>
    <div class="stat"><div class="n">2026</div><div class="l">Frontier-current</div></div>
    <div class="stat"><div class="n">∞</div><div class="l">Living document</div></div>
  </div>
</div>

> [!TIP] New here? Read **[How to Use This Book](#/start/how-to-use)** first, then skim **[The 2026 Landscape](#/start/landscape-2026)** to calibrate what interviewers now expect. If you have a loop next week, jump to the **[8-Week Prep Plan](#/start/prep-plan)** and triage.

## What this book is

Most interview resources are either (a) coding-only grind lists or (b) generic ML-101 flashcards written for 2021. This one is built for the **research/applied scientist** track in 2026 — where you're expected to reason about reasoning models, RLVR, native multimodality, and agentic tool use *and* still whiteboard non-max suppression cleanly, present your own research convincingly, and pass a behavioral loop.

It's organized as a **book you can read front-to-back** or **dip into by topic**. Every chapter is self-contained, heavy on diagrams and worked examples, and honest about what's a verifiable fact versus a defensible opinion.

## The four axes of a research-scientist loop

<figure>
<svg viewBox="0 0 720 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif">
  <defs>
    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e0533f"/><stop offset="1" stop-color="#6366f1"/></linearGradient>
  </defs>
  <g>
    <rect x="10" y="20" width="165" height="170" rx="12" fill="none" stroke="#e0533f" stroke-width="2"/>
    <text x="92" y="48" text-anchor="middle" font-size="30">⌨️</text>
    <text x="92" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#e0533f">Coding</text>
    <text x="92" y="104" text-anchor="middle" font-size="11" fill="#98a3b2">DSA patterns</text>
    <text x="92" y="122" text-anchor="middle" font-size="11" fill="#98a3b2">ML-from-scratch</text>
    <text x="92" y="140" text-anchor="middle" font-size="11" fill="#98a3b2">live problem-solving</text>
  </g>
  <g>
    <rect x="188" y="20" width="165" height="170" rx="12" fill="none" stroke="#6366f1" stroke-width="2"/>
    <text x="270" y="48" text-anchor="middle" font-size="30">🧠</text>
    <text x="270" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#6366f1">ML depth &amp; breadth</text>
    <text x="270" y="104" text-anchor="middle" font-size="11" fill="#98a3b2">DL foundations</text>
    <text x="270" y="122" text-anchor="middle" font-size="11" fill="#98a3b2">CV · LLM · VLM · agents</text>
    <text x="270" y="140" text-anchor="middle" font-size="11" fill="#98a3b2">2026 frontier</text>
  </g>
  <g>
    <rect x="366" y="20" width="165" height="170" rx="12" fill="none" stroke="#0ea5e9" stroke-width="2"/>
    <text x="448" y="48" text-anchor="middle" font-size="30">🏗️</text>
    <text x="448" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#0ea5e9">System design</text>
    <text x="448" y="104" text-anchor="middle" font-size="11" fill="#98a3b2">ML pipelines</text>
    <text x="448" y="122" text-anchor="middle" font-size="11" fill="#98a3b2">LLM/agent systems</text>
    <text x="448" y="140" text-anchor="middle" font-size="11" fill="#98a3b2">serving &amp; scale</text>
  </g>
  <g>
    <rect x="544" y="20" width="165" height="170" rx="12" fill="none" stroke="#12a150" stroke-width="2"/>
    <text x="626" y="48" text-anchor="middle" font-size="30">📄</text>
    <text x="626" y="80" text-anchor="middle" font-size="15" font-weight="700" fill="#12a150">Research + behavioral</text>
    <text x="626" y="104" text-anchor="middle" font-size="11" fill="#98a3b2">job talk</text>
    <text x="626" y="122" text-anchor="middle" font-size="11" fill="#98a3b2">deep-dive on your work</text>
    <text x="626" y="140" text-anchor="middle" font-size="11" fill="#98a3b2">STAR stories</text>
  </g>
</svg>
<figcaption>The four evaluation axes. Research roles weight the right two far more than a standard MLE loop does.</figcaption>
</figure>

## Start reading

<div class="card-grid">
  <a class="card" href="#/start/landscape-2026"><div class="card-emoji">🛰️</div><div class="card-title">The 2026 Landscape</div><div class="card-desc">What changed: reasoning models, RLVR, native multimodal, agents. Calibrate your expectations.</div></a>
  <a class="card" href="#/coding/patterns"><div class="card-emoji">⌨️</div><div class="card-title">Coding Patterns</div><div class="card-desc">The ~15 patterns that cover most coding rounds, with a cue→pattern lookup.</div></a>
  <a class="card" href="#/foundations/optimization"><div class="card-emoji">📐</div><div class="card-title">DL Foundations</div><div class="card-desc">Optimization, normalization, architectures — with interactive visualizations.</div></a>
  <a class="card" href="#/llm/reasoning"><div class="card-emoji">🤖</div><div class="card-title">Reasoning &amp; Agents</div><div class="card-desc">Test-time compute, RLVR, tool use, visual agents — the 2026 hot zone.</div></a>
  <a class="card" href="#/system-design/framework"><div class="card-emoji">🏗️</div><div class="card-title">ML System Design</div><div class="card-desc">A repeatable framework plus worked case studies for research/applied roles.</div></a>
  <a class="card" href="#/resume/predicted-questions"><div class="card-emoji">🎯</div><div class="card-title">Your CV Q&amp;A</div><div class="card-desc">Predicted questions from Beomyoung's CV, with multiple strong model answers.</div></a>
</div>

> [!NOTE] A living document
> Content is revised continuously as models, benchmarks, and hiring bars move. Each part has a "last-reviewed" note, and the [Changelog](#/resources/changelog) tracks what's new. Found something stale or wrong? It's one Markdown file — fix it and push.
