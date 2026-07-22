# 🧠 ML Interview Codex

A comprehensive, continuously-updated **105-chapter bilingual online book** for ML / AI **Research & Applied Scientist** interviews — coding, ML/DL foundations, computer vision, LLMs/VLMs/agents, ML system design, research & behavioral rounds, plus CV-specific deep-dives. **Current to July 2026.**

Built as a lightweight, dependency-free **client-side Markdown single-page app** — no build step, deploys to GitHub Pages, and updating content is just editing a `.md` file.

> Live site (project page): **https://beomyoung-kim.github.io/ml_study/**

---

## ✨ Features

- **Responsive online-book UX** — hierarchical chapter navigation, current-book position, on-page table of contents, prev/next paging, keyboard focus states, and a mobile drawer.
- **Full-text search** across every chapter (`/` to open).
- **Light / dark theme** and **Korean / English toggle** (both remember your choice; Korean keeps technical terms/code/math in English).
- **Rich rendering** — KaTeX math, Mermaid diagrams, syntax-highlighted code with copy buttons, GitHub-style callouts, Q&A accordions, collapsible conceptual pseudocode, and resume-grounded answer cards.
- **Interactive widgets** — activation functions, gradient descent, LR schedules, precision/recall threshold, attention heatmaps (all self-contained canvas/SVG).
- **Self-contained visuals** — inline SVG + Mermaid + CSS, theme-aware, no external image hosting.

## 🗂️ Structure

```
ml_study/
├── index.html            # app shell
├── assets/
│   ├── book.js           # ← TABLE OF CONTENTS (edit to add chapters/parts)
│   ├── app.js            # engine: routing, markdown, math, search, TOC
│   ├── widgets.js        # interactive canvas/SVG widgets
│   └── book.css          # design system (light/dark)
├── content/**/*.md       # ← ALL CONTENT lives here, one file per chapter
├── AUTHORING.md          # the style contract for writing chapters
├── .nojekyll             # tell GitHub Pages to serve files as-is
└── .github/workflows/pages.yml  # optional: deploy via GitHub Actions
```

## 🚀 Run locally

Any static file server works (needed because the app `fetch`es Markdown):

```bash
cd ml_study
python3 -m http.server 8811
# open http://localhost:8811
```

Before publishing, run the bilingual content checks:

```bash
node scripts/validate-content.mjs
```

The validator checks manifest/file coverage, Korean/English structural parity, balanced
Markdown/HTML containers, internal routes and section anchors, code-widget JSON, and accidental
Hangul left in English prose.

## 📦 Deploy to GitHub Pages

One-time setup: repo *Settings → Pages → Build and deployment → Source: **Deploy from a branch** → Branch: `main` / `/ (root)` → Save*. The included `.nojekyll` ensures files serve verbatim. After that, every push to `main` republishes automatically — no build step, no workflow.

Because links use hash routing (`#/path`), deep links work without any server rewrite rules.

## ✍️ Updating / adding content

This is a **living document**. To edit a chapter, just edit its Markdown file in `content/` and push.

To **add a new chapter**:
1. Write and review `content/<part>/<name>.ko.md`, then create its synchronized English sibling at `content/<part>/<name>.md`.
2. Add one line to the matching part in `assets/book.js`:
   ```js
   { id: "my-id", title: "My Chapter", file: "part/name" },
   ```
3. (Optional) add a badge: `badge: "2026"` or `badge: "new"`.

See **[`AUTHORING.md`](AUTHORING.md)** for the full style contract (callouts, Q&A blocks, math, Mermaid, widgets, cross-links). Record notable updates in `content/resources/changelog.md`.

## 🎯 Scope

Written for a PhD-level Research/Applied Scientist candidate in **computer vision, VLMs, and visual agents**, but the coding, foundations, system-design, research, and behavioral parts apply to any ML research/applied role.

## 📄 License / use

Personal study material. External resources are credited in [Curated Open-Source Resources](content/resources/open-source.md). Diagrams and text are original unless cited.
