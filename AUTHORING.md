# Authoring Guide — ML Interview Codex

This is the style contract for every chapter. The book is a client-side Markdown SPA:
content lives in `content/**.md`, the table of contents in `assets/book.js`. To add a
chapter: write the `.md`, then add one line to the right part in `assets/book.js`.

## Voice & language
- **English-primary.** Technical terms, math, and code as usual.
- Write for a **PhD-level research/applied-scientist candidate**. Assume ML maturity; the value is *interview-shaping* knowledge: crisp recall, structure, trade-offs, failure modes.
- Be **honest about certainty.** Tag claims when useful: *(verifiable)*, *(defensible opinion)*, *(speculative / direction)*. Never invent benchmark numbers, dates, or model names. When citing 2025–2026 models, prefer capabilities/architecture over exact scores; hedge vendor-reported figures.
- Concise and high-density. No filler. Every sentence should earn its place in an interview.

## Required chapter shape
1. `# Title` (H1, once).
2. A short **lead-in**: why this matters *in an interview* (1–3 sentences), often as a `> [!TIP]` box.
3. Body with `##`/`###` sections, heavy on diagrams, tables, and worked examples.
4. **Q&A blocks** phrased exactly as an interviewer asks — short answer first, then depth.
5. **"Follow-ups"** — the sharper questions that come after your first answer.
6. A closing **cheat-sheet** (`## Cheat-sheet`): 5–12 must-know facts, ideally a table.
7. **Cross-links** to related chapters.

## Markdown features available
- **Callouts** (GitHub-style). First line of a blockquote is `[!TYPE] optional title`:
  ```
  > [!TIP] Say this first
  > body...
  ```
  Types: `NOTE`, `TIP`, `WARNING`, `DANGER`, `QUESTION`, `EXAMPLE`, `QUOTE` (also `IMPORTANT`→note, `CAUTION`→warning).
- **Q&A accordion** — use a `<details class="qa">`:
  ```html
  <details class="qa"><summary>Why does BN help optimization?</summary>
  <div class="qa-body">

  **Short:** it smooths the loss landscape... 

  **Deep:** ...
  </div></details>
  ```
- **Math**: inline `$…$` and display `$$…$$` (KaTeX). Also `\( … \)` / `\[ … \]`.
- **Mermaid**: ```` ```mermaid ```` fenced blocks (flowchart/sequence/gantt/etc.).
- **Code**: fenced with a language; auto-highlighted with a copy button.
- **Interactive widgets** (embed raw HTML): `<div class="widget" data-widget="NAME"></div>`.
  Available: `activation`, `gradient-descent`, `lr-schedule`, `metrics-threshold`, `attention`.
  Add new ones in `assets/widgets.js`.
- **Inline SVG figures**: wrap in `<figure>…<figcaption>…</figcaption></figure>`. Prefer SVG/CSS/Mermaid over raster images (self-contained, theme-aware, diffable). Use `currentColor` or the palette (`#e0533f` accent, `#6366f1` indigo, `#0ea5e9` sky, `#12a150` green) so figures read in light & dark.
- **Cards** for navigation hubs: `<div class="card-grid"><a class="card" href="#/path"><div class="card-emoji">🧩</div><div class="card-title">…</div><div class="card-desc">…</div></a></div>`.
- **Badges**: `<span class="badge badge-hard">Hard</span>` (`-hard/-med/-easy/-new/-2026`).
- **Pros/cons**: `<div class="proscons"><div><div class="pros-t">Pros</div>…</div><div><div class="cons-t">Cons</div>…</div></div>`.
- **Key–value spec**: `<dl class="kv"><dt>…</dt><dd>…</dd></dl>`.

## Linking
- Internal links use the hash router: `[Optimization](#/foundations/optimization)` where the path is the chapter's `file` in `assets/book.js`.
- External links get opened in a new tab automatically.

## Diagrams: quality bar
- Every conceptual chapter should have **at least one** diagram or widget.
- Mermaid for flows/architectures/timelines; inline SVG for precise custom figures (loss curves, matrices, pipelines); widgets for anything the reader benefits from *manipulating*.

## Translations (Korean)

The book has a **한 / EN toggle** (default English). Each English chapter `content/<path>.md`
may have a Korean sibling `content/<path>.ko.md`. In Korean mode the engine loads the `.ko.md`
and **falls back to English** (with a small notice) when it's missing — so a partial translation
never breaks the book.

To translate or update a chapter:
- Copy the English file to `<name>.ko.md` and translate **prose only**. Keep **verbatim**: all
  code, math (`$…$`/`$$…$$`), ```mermaid blocks, inline `<svg>`, widget divs, HTML tags/classes,
  callout markers (`> [!TIP]`), `<details class="qa">` structure, link *targets* (`#/…`, URLs),
  badges, and tags. Preserve blank-line layout exactly (the engine relies on it).
- **Keep technical terms in English** (attention, transformer, RLVR, LoRA, mIoU, …) — Korean
  particles attach naturally (`attention을`, `transformer는`). Translate only the visible link
  *text*, not the target.
- Sidebar/UI labels: add `titleKo` on the part/chapter in `assets/book.js` for a Korean nav label;
  omit it to keep the English (technical) title. UI chrome strings live in the `I18N` object in
  `assets/app.js`.
- After writing a `.ko.md`, sanity-check it renders (math/Mermaid intact) before committing.

## Don'ts
- Don't dump prose walls. Prefer tables, lists, and figures.
- Don't fabricate. If unsure of a number, describe the mechanism instead.
- Don't duplicate content across chapters — cross-link instead.
- Don't use raster screenshots of papers; redraw the idea as SVG/Mermaid.
