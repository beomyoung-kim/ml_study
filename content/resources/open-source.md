# Curated Open-Source Resources

<div class="tag-row"><span class="tag">free</span><span class="tag">open-source</span><span class="tag">coding</span><span class="tag">ML depth</span><span class="tag">system design</span><span class="tag">LLM/GenAI</span><span class="tag">behavioral</span><span class="tag">mock</span></div>

> [!TIP] How to use this page
> This is an *annotated* directory, not a link dump. For each resource you get **what it's best for** and **the single best idea to steal** — the reusable framework or packaging trick, not just the content. The meta-lesson across every top resource: **organize by reusable pattern, not by random question dumps.** NeetCode, sean-prashad, Chip Huyen, and alirezadir all win because they hand you a small, memorable taxonomy that generalizes to unseen problems. This whole book is built on that principle.

> [!NOTE] On star counts and links
> Star counts below are approximate as of mid-2026 — verify at the source. All external links open in a new tab. Prefer the primary source over mirrors.

## If you only use 5 resources

For a PhD-level Applied/Research-Scientist loop (CV / VLM / visual agents), this is the minimal high-leverage set.

| # | Resource | Why it's on the shortlist |
| --- | --- | --- |
| 1 | [Chip Huyen — Introduction to ML Interviews](https://huyenchip.com/ml-interviews-book/) | Best overall structure; the *career half* (RS vs RE, process, offers) that competitors omit + 200+ Qs. |
| 2 | [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/machine-learning-interviews) | The **9-step ML system-design framework** and the 4-pillar loop that scaffolds an entire interview. |
| 3 | [NeetCode roadmap + 150](https://neetcode.io/roadmap) | The 18-topic DSA ontology and the dependency-graph learning path — the de-facto coding standard. |
| 4 | [Amidi Stanford CS229 / CS230 cheatsheets](https://stanford.edu/~shervine/teaching/cs-229/) | One-page, formula-dense recall sheets — the gold standard for last-mile review. |
| 5 | [aishwaryanr/awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) | Role-based prep + the multimodal/embeddings question taxonomy that maps 1:1 to a VLM reader. |

Everything below expands on these and fills the gaps (behavioral, mocks, company tracking).

## Coding interview prep

<details class="qa"><summary>Do PhD candidates really need the coding grind?</summary>
<div class="qa-body">

**Short:** yes, but scoped. Research-scientist loops usually include 1–2 coding rounds; you don't need 500 problems, you need ~15–20 *patterns* solid enough to solve unseen ones under time pressure. Optimize for pattern coverage, not problem count.

**Deep:** the entire value proposition of NeetCode/Grokking/sean-prashad is that the space of interview problems collapses onto a small set of recurring shapes. Internalize the shapes, do 5–10 problems each, and you generalize. See [Coding Strategy](#/coding/strategy) and [Patterns](#/coding/patterns).
</div></details>

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **NeetCode** (Blind 75 → 150 → 250) | Navdeep Singh (ex-Google) | [neetcode.io](https://neetcode.io) · [roadmap](https://neetcode.io/roadmap) | The de-facto coding standard; free video walkthrough for every problem | **Roadmap as a dependency graph** — topics unlock prerequisites. The 18-category taxonomy is the cleanest DSA ontology available. |
| **Tech Interview Handbook + Grind75** | Yangshun Tay (ex-Meta) | [techinterviewhandbook.org](https://www.techinterviewhandbook.org) · [grind75](https://www.techinterviewhandbook.org/grind75) · [repo (~140k★)](https://github.com/yangshun/tech-interview-handbook) | "Cracking the Coding Interview, but concise" + behavioral, resume, negotiation | The **configurable study planner** — input hours/week + weeks, get an ordered, difficulty-ramped plan. The interactivity, not the list, made it famous. |
| **Blind 75** | Yangshun (orig. teamblind) | via [NeetCode](https://neetcode.io) | The canonical minimal 75-problem list across 10 categories | The distribution signal: **Trees + DP are ~1/3 of the list** → weight your prep toward high-payoff topics. |
| **sean-prashad/leetcode-patterns** | Sean Prashad | [site](https://seanprashad.com/leetcode-patterns/) · [repo (~10k★)](https://github.com/seanprashad/leetcode-patterns) | ~179 problems grouped by pattern, with company + difficulty filters | The **"cue → pattern" decision cheat sheet** ("input sorted → binary search / two pointers"; "top-K → heap"; "all permutations → backtracking"). We build a CV/ML analogue in [Patterns](#/coding/patterns). |
| **Grokking the Coding Interview (14 patterns)** | Design Gurus / Fahim ul Haq | [free article](https://hackernoon.com/14-patterns-to-ace-any-coding-interview-question-c5bb3357f6ed) | The viral "14 patterns" mental model (course is paid; the framework is free-famous) | The **"N patterns to ace any X"** packaging — the most repeatable framing in the space. |
| **jwasham/coding-interview-university** | John Washam | [repo (~337k★)](https://github.com/jwasham/coding-interview-university) | An exhaustive, multi-month CS syllabus with checkboxes | The **trackable-curriculum format** — "here is the whole syllabus, in order, with progress checkboxes." |

## ML / DL question banks

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **Introduction to ML Interviews (MLIB)** | Chip Huyen | [free book](https://huyenchip.com/ml-interviews-book/) · [repo](https://github.com/chiphuyen/ml-interviews-book) | The gold standard; 200+ knowledge Qs + 30 open-ended judgment Qs | The **career-half + question-half split**, and disaggregating **Research Scientist vs Research Engineer** expectations — directly on-target for this reader. |
| **alirezadir/Machine-Learning-Interviews** | alirezadir (ex-Amazon/Google) | [repo (~10k★)](https://github.com/alirezadir/machine-learning-interviews) | Best single all-in-one MLE repo | The **4-pillar loop** decomposition: coding / ML depth / ML system design / behavioral — the spine of this book. |
| **khangich/machine-learning-interview** | Khang Pham | [repo](https://github.com/khangich/machine-learning-interview) · [mlengineer.io](https://mlengineer.io) | Real FAANG questions built from actual offers (Snapchat, Coupang, Stitchfix) | **Questions attributed to specific companies** — credible and matches the target-company list. Mine `design.md`, `appliedml.md`, `how.md`. |
| **interviews.ai — Deep Learning Interviews** | Shlomo Kashani (BoltzmannEntropy) | [repo](https://github.com/BoltzmannEntropy/interviews.ai) · [arXiv 2201.00650](https://arxiv.org/abs/2201.00650) | Free PDF; hundreds of **fully-solved, exam-style, PhD-level** problems | The rigorous **derivation-not-one-liner** register — the right depth for a research-scientist audience. |
| **andrewekhalel/MLQuestions** | Andrew Ekhalel | [repo](https://github.com/andrewekhalel/MLQuestions) | Concise Q&A with an explicit **Computer Vision** slant | One of the few banks with a CV lean — a good seed list for CV depth (detection/segmentation specifics). |
| **zafstojano/ml-interview Q&A** | community | [repo](https://github.com/zafstojano/ml-interview-questions-and-answers) | Community solutions to Chip Huyen's MLIB | Use as an answer key / sanity check against MLIB. |

## ML system design

The most learnable-by-framework round. Every case study in this book follows the **9-step spine** below.

<dl class="kv">
<dt>alirezadir — 9-step MLSD framework</dt><dd>The most citable open-source framework. <a href="https://github.com/alirezadir/machine-learning-interviews/blob/main/src/MLSD/ml-system-design.md">ml-system-design.md</a>: (1) Problem formulation → (2) Metrics (offline + online) → (3) Architectural components / MVP → (4) Data collection & prep → (5) Feature engineering → (6) Model dev & offline eval → (7) Prediction service (batch/online/hybrid) → (8) Online testing & deploy (A/B, rollout) → (9) Scaling, monitoring & continual training. <b>Memorize this.</b></dd>
<dt>Chip Huyen — Designing ML Systems (DMLS)</dt><dd>Book is paid, but the <a href="https://huyenchip.com/machine-learning-systems-design/">free primer</a> (TOC + case studies + ~27 exercises) and <a href="https://github.com/chiphuyen/dmls-book">repo summaries</a> are gold. <b>Steal:</b> the requirements rubric — <i>reliability, scalability, maintainability, adaptability</i>.</dd>
<dt>Eugene Yan — applied-ml & applyingml</dt><dd><a href="https://github.com/eugeneyan/applied-ml">applied-ml</a> = 500+ curated industry papers/blogs by problem type (RecSys, search, CV, fraud…). <a href="https://applyingml.com">applyingml.com</a> = the ML process + mentor interviews. <b>Steal:</b> the <b>Why / What / How</b> design-doc skeleton for take-home and design answers.</dd>
</dl>

> [!NOTE] Landscape-only (partly paid — cite, don't copy)
> ByteByteGo ML System Design and Hello Interview "ML System Design in a Hurry" are widely referenced but partly paid. Know they exist; build on the free frameworks above.

## LLM / GenAI

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **mlabonne/llm-course** | Maxime Labonne | [repo (~80k★)](https://github.com/mlabonne/llm-course) | The best free LLM curriculum; roadmaps + Colab notebooks | The **LLM Scientist vs LLM Engineer track split** — perfect for a research book that must also cover applied deployment. |
| **awesome-generative-ai-guide** | Aishwarya Naresh Reganti | [repo](https://github.com/aishwaryanr/awesome-generative-ai-guide) | Best free GenAI hub; **role-based** prep + research updates + "60 GenAI questions" | The **role-based folder** ("pick your role → overview → rounds → bank → plan") and the **Multimodal + Embeddings** sections that map 1:1 to VLMs. |
| **llmgenai/LLMInterviewQuestions** | llmgenai | [repo](https://github.com/llmgenai/LLMInterviewQuestions) | 100+ Qs, 16-category **applied/production** taxonomy (RAG, vector DB, agents) | The **vector-DB + advanced-search + agent** categories competitors omit — relevant to "visual agents." |
| **aman.ai primers** | Aman Chadha | [primers](https://aman.ai/primers/ai/) · [interview Qs](https://aman.ai/primers/ai/interview/) | Extremely deep, frequently-updated DL/NLP primers + 100+ interview Qs | The **"pros and cons" comparison format** for models, and the transformer/attention deep-dive as a depth reference. |
| **microsoft/generative-ai-for-beginners** | Microsoft | [repo (~90k★)](https://github.com/microsoft/generative-ai-for-beginners) | A credible free foundation (21-lesson course) | More teaching than interview prep — point beginners here; not a source to copy. |
| **KalyanKS-NLP LLM-Interview Hub** | Kalyan KS | [repo](https://github.com/KalyanKS-NLP/llm-engineer-toolkit) | Supplementary 100+ LLM Q&A | A quick complementary bank for LLM breadth. |

## Cheat sheets

The Amidi Stanford VIP sheets are the format to emulate: **one page, formula-dense, visually clean.** We produce our own CV/VLM sheets in this exact style.

- **CS229 (ML)** — [stanford.edu/~shervine/teaching/cs-229](https://stanford.edu/~shervine/teaching/cs-229/) · [repo](https://github.com/afshinea/stanford-cs-229-machine-learning). Sheets: supervised, unsupervised, deep learning, tips & tricks (metrics, bias/variance), + prob/stats and linear-algebra refreshers. Translated into 10+ languages.
- **CS230 (Deep Learning)** — [stanford.edu/~shervine/teaching/cs-230](https://stanford.edu/~shervine/teaching/cs-230/) · [repo](https://github.com/afshinea/stanford-cs-230-deep-learning). Sheets: CNNs (filter arithmetic, architectures), RNNs (LSTM/GRU, attention, embeddings), DL tips & tricks.
- **CS231n (CV) course notes** — [cs231n.github.io](https://cs231n.github.io/). Canonical CV depth foundation (CNNs, backprop, detection).
- **CS224n (NLP)** — [web.stanford.edu/class/cs224n](http://web.stanford.edu/class/cs224n/). The NLP/transformer counterpart.
- **d2l.ai — Dive into Deep Learning** — [d2l.ai](https://d2l.ai/). Runnable-code textbook; great for filling derivation gaps.

## Behavioral & research-scientist-specific

> [!WARNING] The under-served category
> Research-scientist interviewing — the **job talk, research deep-dive, and paper-impact storytelling** — is the weakest area in open-source resources (mostly generic career-services pages). This book's [Behavioral & Research](#/process/pipeline) coverage is a deliberate differentiation. Treat the links below as scaffolding, then go beyond them.

- **STAR / STAR-L method** — [MIT CAPD worksheet](https://capd.mit.edu/resources/the-star-method-for-behavioral-interviews/). Situation → Task → Action → Result (+ **Learning**, apt for scientists). Rules to adopt: time-allocation (Action ~50–60%), use **"I" not "we"**, and a **story-bank matrix** (rows = competencies, columns = STAR-L slots) pre-filled from your own research.
- **PhD → industry transition** — Rutgers "From Bench to Offer", NIH OITE blog, Cheeky Scientist. Key reframe: *academia rewards nuance and caveats; interviews reward clear decisions and measurable outcomes.* Translate research into **decision + measurable impact** language.
- **The research job talk** — confirm format with the recruiter; lead with problem significance and deliverables; demonstrate science communication + depth; anticipate deep Q&A; tailor framing to the team's product.

## Mock interviews

- **[Pramp](https://www.pramp.com/)** — free peer-to-peer mock interviews (coding + some behavioral). Best for reps and calibration.
- **[interviewing.io](https://interviewing.io/)** — anonymous mocks with senior engineers; recorded, with a large free technical-talk / question library.
- **Your network** — the highest-signal option. Ask KAIST MLAI / SIIT alumni and NAVER Cloud colleagues who are now at big tech for a real research-depth mock. Nothing simulates the loop like someone who runs one.

## Track the frontier (company blogs & leaderboards)

Interviewers calibrate you against the *current* frontier. Skim these before any loop — see [The 2026 Landscape](#/start/landscape-2026) for why.

<div class="proscons"><div><div class="pros-t">Company research blogs</div>

- [Meta AI Research](https://ai.meta.com/research/)
- [NVIDIA Research](https://research.nvidia.com/)
- [Apple ML Research](https://machinelearning.apple.com/)
- [Adobe Research](https://research.adobe.com/)
- [Microsoft Research](https://www.microsoft.com/en-us/research/)
- [ByteDance Seed](https://seed.bytedance.com/)
- [Mistral AI News](https://mistral.ai/news)

</div><div><div class="cons-t">Trend & leaderboard trackers</div>

- [Papers with Code](https://paperswithcode.com/) — SOTA + benchmark tracking
- [LMArena](https://lmarena.ai/) — human-preference LLM leaderboard (read critically)
- [Hugging Face](https://huggingface.co/models) — models, datasets, Open LLM leaderboard
- [arXiv cs.CV / cs.CL / cs.LG](https://arxiv.org/list/cs.CV/recent) — the primary firehose
- [Distill.pub](https://distill.pub/) — archived, but the best intuition-building explainers

</div></div>

## Interview-experience sources (real questions)

Use these to find company- and role-attributed questions — the highest-credibility signal for what a specific loop will ask.

- [Glassdoor Interview Questions](https://www.glassdoor.com/Interview/index.htm) — search "Meta Research Scientist interview", etc.
- [Blind (teamblind.com)](https://www.teamblind.com/) — anonymous big-tech employee reports.
- [1point3acres](https://www.1point3acres.com/bbs/) — strong for MSRA and China-office loops.
- [LeetCode Discuss — Interview Experience](https://leetcode.com/discuss/interview-experience) — company-tagged real coding tests.
- [levels.fyi](https://www.levels.fyi/) — level/comp context for offer negotiation.

## Your own research assets (use as evidence)

The strongest "source" in a research interview is *your own work*. Keep these ready to cite in the [research deep-dive](#/process/pipeline) and to seed your STAR-L story bank.

<dl class="kv">
<dt>Google Scholar</dt><dd><a href="https://scholar.google.co.kr/citations?user=n_TR1LcAAAAJ">All publications</a> — citation context for impact stories.</dd>
<dt>GitHub</dt><dd><a href="https://github.com/qjadud1994">qjadud1994</a> — public code; proof of engineering rigor.</dd>
<dt>ZIM</dt><dd>Zero-shot Image Matting — <a href="https://naver-ai.github.io/ZIM/">project</a> · <a href="https://arxiv.org/pdf/2411.00626">paper</a>. Ties directly to SAM / alpha-matte / promptable segmentation talking points.</dd>
<dt>ECLIPSE</dt><dd>Panoptic segmentation — <a href="https://arxiv.org/abs/2403.20126">arXiv 2403.20126</a>. Continual / catastrophic-forgetting narrative.</dd>
<dt>PointWSSIS</dt><dd>Weakly-supervised instance segmentation — <a href="https://arxiv.org/abs/2303.15062">arXiv 2303.15062</a>. Label-efficiency story.</dd>
<dt>BESTIE</dt><dd>Weakly-supervised semantic segmentation — <a href="https://arxiv.org/abs/2109.09477">arXiv 2109.09477</a>. Foundational segmentation depth.</dd>
</dl>

## Cheat-sheet

| Need | Go-to (free) |
| --- | --- |
| Coding patterns | NeetCode roadmap + sean-prashad "cue → pattern" sheet |
| Study plan / scheduling | Grind75 configurable planner |
| ML depth Q-bank | Chip Huyen MLIB + interviews.ai (PhD-level solved) |
| ML system design | alirezadir 9-step framework + Eugene Yan Why/What/How |
| LLM/GenAI | mlabonne llm-course + awesome-generative-ai-guide |
| CV slant | andrewekhalel/MLQuestions + CS231n notes |
| Last-mile recall | Amidi CS229/CS230 cheatsheets |
| Behavioral | MIT CAPD STAR-L + story-bank matrix |
| Mocks | Pramp / interviewing.io / your network |
| Real questions | Glassdoor / Blind / 1point3acres / LeetCode Discuss |

**Related:** [How to Use This Book](#/start/how-to-use) · [The 2026 Landscape](#/start/landscape-2026) · [Prep Plan](#/start/prep-plan) · [Coding Strategy](#/coding/strategy) · [Interview Pipeline](#/process/pipeline)
