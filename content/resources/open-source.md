# Curated Public Learning Resources

<div class="tag-row"><span class="tag">free</span><span class="tag">open-source</span><span class="tag">coding</span><span class="tag">ML depth</span><span class="tag">system design</span><span class="tag">LLM/GenAI</span><span class="tag">behavioral</span><span class="tag">mock</span></div>

> [!TIP] How to use this page
> This is an *annotated* directory, not a link dump. It explains where each resource is useful and which reusable learning structure to borrow. Some entries are freely available rather than open source, or combine free and paid features, so check each site's license and paywall. The common lesson is to **organize around reusable patterns, not random question dumps.**

> [!NOTE] Link status
> This list was last reviewed on **2026-07-21**. URLs, prices, licenses, and course scope can change. Fast-moving numbers unrelated to learning value, such as GitHub star counts, are intentionally omitted; prefer the author's or institution's primary source over mirrors.

## If you only use 5 resources

For a PhD-level Applied/Research-Scientist loop (CV / VLM / visual agents), this is the minimal high-leverage set.

| # | Resource | Why it's on the shortlist |
| --- | --- | --- |
| 1 | [Chip Huyen — Introduction to ML Interviews](https://huyenchip.com/ml-interviews-book/) | A public book that covers technical questions and career/process guidance for ML interviews in one flow. Check the current edition and contents rather than relying on an exact question count. |
| 2 | [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/machine-learning-interviews) | The **9-step ML system-design framework** and the 4-pillar loop that scaffolds an entire interview. |
| 3 | [NeetCode roadmap + 150](https://neetcode.io/roadmap) | A widely used DSA-topic roadmap with a dependency-based learning path. Check the current split between free and paid material. |
| 4 | [Amidi Stanford CS229 / CS230 cheatsheets](https://stanford.edu/~shervine/teaching/cs-229/) | Formula-dense, one-page recall sheets useful for final review. |
| 5 | [aishwaryanr/awesome-generative-ai-guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) | Role-based prep + the multimodal/embeddings question taxonomy that maps 1:1 to a VLM reader. |

Everything below expands on these and fills the gaps (behavioral, mocks, company tracking).

## Coding interview prep

<details class="qa"><summary>Do PhD candidates really need the coding grind?</summary>
<div class="qa-body">

**Short:** if the actual loop includes coding, practice within a narrow scope. Round count and difficulty vary by role, so confirm them with the recruiter and prioritize recurring-pattern coverage plus explanation and verification habits over raw problem count.

**Deep:** the entire value proposition of NeetCode/Grokking/sean-prashad is that the space of interview problems collapses onto a small set of recurring shapes. Internalize the shapes, do 5–10 problems each, and you generalize. See [Coding Strategy](#/coding/strategy) and [Patterns](#/coding/patterns).
</div></details>

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **NeetCode** (Blind 75 → 150 → 250) | Navdeep Singh | [neetcode.io](https://neetcode.io) · [roadmap](https://neetcode.io/roadmap) | Pattern-organized problem roadmap and explanations; check the current free/paid split on the site | **Roadmap as a dependency graph** — reuse its topic prerequisites and pattern-specific practice lists in your study plan. |
| **Tech Interview Handbook + Grind75** | Yangshun Tay | [techinterviewhandbook.org](https://www.techinterviewhandbook.org) · [grind75](https://www.techinterviewhandbook.org/grind75) · [repo](https://github.com/yangshun/tech-interview-handbook) | Concise guidance on coding, behavioral interviews, résumés, and negotiation | The **configurable study planner** — adjust problem order and difficulty to your weekly time and preparation window. |
| **Blind 75** | Community list originating in an anonymous Blind post | via [NeetCode](https://neetcode.io) | A frequently reused 75-problem shortlist | Use it as a quick pattern-coverage baseline, not as evidence of the question distribution at a target company. |
| **sean-prashad/leetcode-patterns** | Sean Prashad | [site](https://seanprashad.com/leetcode-patterns/) · [repo](https://github.com/seanprashad/leetcode-patterns) | Problems and difficulty filters organized by pattern | The **"cue → pattern" decision cheat sheet** ("input sorted → binary search / two pointers"; "top-K → heap"; "all permutations → backtracking"). We build the corresponding structure in [Patterns](#/coding/patterns). |
| **Grokking the Coding Interview (14 patterns)** | Design Gurus / Fahim ul Haq | [free article](https://hackernoon.com/14-patterns-to-ace-any-coding-interview-question-c5bb3357f6ed) | The viral "14 patterns" mental model (course is paid; the framework is free-famous) | The **"N patterns to ace any X"** packaging — the most repeatable framing in the space. |
| **jwasham/coding-interview-university** | John Washam | [repo](https://github.com/jwasham/coding-interview-university) | An exhaustive, multi-month CS syllabus with checkboxes | The **trackable-curriculum format** — "here is the whole syllabus, in order, with progress checkboxes." |

## ML / DL question banks

| Resource | Author | Link | Best for | Steal this |
| --- | --- | --- | --- | --- |
| **Introduction to ML Interviews (MLIB)** | Chip Huyen | [free book](https://huyenchip.com/ml-interviews-book/) · [repo](https://github.com/chiphuyen/ml-interviews-book) | The gold standard; 200+ knowledge Qs + 30 open-ended judgment Qs | The **career-half + question-half split**, and disaggregating **Research Scientist vs Research Engineer** expectations — directly on-target for this reader. |
| **alirezadir/Machine-Learning-Interviews** | alirezadir | [repo](https://github.com/alirezadir/machine-learning-interviews) | Broad MLE interview repository | The **4-pillar loop** decomposition: coding / ML depth / ML system design / behavioral. |
| **khangich/machine-learning-interview** | Khang Pham | [repo](https://github.com/khangich/machine-learning-interview) · [mlengineer.io](https://mlengineer.io) | Community guide based on the author's interview experience | Treat company-attributed questions as low-confidence practice prompts because of timing, team, and recall bias; use the structure of `design.md`, `appliedml.md`, and `how.md`. |
| **interviews.ai — Deep Learning Interviews** | Shlomo Kashani (BoltzmannEntropy) | [repo](https://github.com/BoltzmannEntropy/interviews.ai) · [arXiv 2201.00650](https://arxiv.org/abs/2201.00650) | Public PDF and collection of deep-learning interview questions and solutions | Use it to practice derivations and follow-ups rather than short definitions, while independently checking difficulty labels and answers. |
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
| **mlabonne/llm-course** | Maxime Labonne | [repo](https://github.com/mlabonne/llm-course) | Public LLM curriculum with roadmaps and notebooks | The **LLM Scientist vs LLM Engineer track split** — a way to separate research and deployment learning goals. |
| **awesome-generative-ai-guide** | Aishwarya Naresh Reganti | [repo](https://github.com/aishwaryanr/awesome-generative-ai-guide) | Community hub collecting role-based preparation, research updates, and questions | Use its role-based folders and Multimodal/Embeddings sections as routing references, then verify freshness at the primary sources. |
| **llmgenai/LLMInterviewQuestions** | llmgenai | [repo](https://github.com/llmgenai/LLMInterviewQuestions) | 100+ Qs, 16-category **applied/production** taxonomy (RAG, vector DB, agents) | The **vector-DB + advanced-search + agent** categories competitors omit — relevant to "visual agents." |
| **aman.ai primers** | Aman Chadha | [primers](https://aman.ai/primers/ai/) · [interview Qs](https://aman.ai/primers/ai/interview/) | Extremely deep, frequently-updated DL/NLP primers + 100+ interview Qs | The **"pros and cons" comparison format** for models, and the transformer/attention deep-dive as a depth reference. |
| **microsoft/generative-ai-for-beginners** | Microsoft | [repo](https://github.com/microsoft/generative-ai-for-beginners) | Public introductory course | Use it for foundation review; it is closer to a course than an interview question bank. |
| **KalyanKS-NLP LLM Engineer Toolkit** | Kalyan KS | [repo](https://github.com/KalyanKS-NLP/llm-engineer-toolkit) | Curated toolkit grouping LLM-engineering libraries and tools by category | This is for stack discovery, not a Q&A bank. Recheck the current official documentation, license, and maintenance status of any library you select. |

## Cheat sheets

The Amidi Stanford VIP sheets are the format to emulate: **one page, formula-dense, visually clean.** We produce our own CV/VLM sheets in this exact style.

- **CS229 (ML)** — [stanford.edu/~shervine/teaching/cs-229](https://stanford.edu/~shervine/teaching/cs-229/) · [repo](https://github.com/afshinea/stanford-cs-229-machine-learning). Sheets: supervised, unsupervised, deep learning, tips & tricks (metrics, bias/variance), + prob/stats and linear-algebra refreshers. Translated into 10+ languages.
- **CS230 (Deep Learning)** — [stanford.edu/~shervine/teaching/cs-230](https://stanford.edu/~shervine/teaching/cs-230/) · [repo](https://github.com/afshinea/stanford-cs-230-deep-learning). Sheets: CNNs (filter arithmetic, architectures), RNNs (LSTM/GRU, attention, embeddings), DL tips & tricks.
- **CS231n (CV) course notes** — [cs231n.github.io](https://cs231n.github.io/). A widely used CV foundation (CNNs, backprop, detection).
- **CS224n (NLP)** — [web.stanford.edu/class/cs224n](https://web.stanford.edu/class/cs224n/). The NLP/transformer counterpart.
- **d2l.ai — Dive into Deep Learning** — [d2l.ai](https://d2l.ai/). Runnable-code textbook; great for filling derivation gaps.

## Visual, animated & interactive explainers

Concepts that click faster when you *see them move*. The URLs below were reviewed at the list level on 2026-07-21, but external-site availability is not guaranteed.

**Convolution / CNN**
- [conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) — Dumoulin & Visin. Representative animated GIFs comparing convolution, transposed convolution, and dilated convolution.
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — Georgia Tech (Polo Chau). Click through a live CNN layer-by-layer with real activations.
- [Animated AI](https://animatedai.github.io/) — clean animations of stride, padding, and grouped/depthwise-separable convolution.

**Attention & Transformer**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar. The most-cited illustrated walkthrough of self-attention.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — runs a live GPT-2 in your browser; type text and watch the attention maps.
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Lilian Weng. The definitive attention-variants survey.
- [3Blue1Brown — Attention, step by step](https://www.3blue1brown.com/lessons/attention) — Manim-animated geometric intuition for Q/K/V.

**ViT (Vision Transformer)**
- [A Visual Guide to Vision Transformers](https://blog.mdturp.ch/posts/2024-04-05-visual_guide_to_vision_transformer.html) — MDTURP. A 14-step animated scroll-story from patches to training.

**Backpropagation**
- [3Blue1Brown — Backpropagation](https://www.3blue1brown.com/lessons/backpropagation) & [Backprop calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — animated gradient flow + the chain-rule mechanics.

**Gradient descent & optimizers**
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) — Distill (Gabriel Goh). Interactive; drag sliders to feel momentum.
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — Sebastian Ruder. Standard SGD/Momentum/RMSprop/Adam reference.
- [TensorFlow Playground](https://playground.tensorflow.org/) — tweak LR / activation / data and watch a net train live.

**RNN / LSTM / GRU**
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Chris Olah. The classic gate-by-gate illustrated explainer.

**Positional encoding**
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) — Amirhossein Kazemnejad. Visual + mathematical intuition for the sinusoids.

**Mamba / State-Space Models**
- [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) — Maarten Grootendorst. 50+ custom visuals from SSM basics to selective scan.
- [The Annotated S4](https://srush.github.io/annotated-s4/) — Sasha Rush. Annotated-paper walkthrough with runnable JAX.
- [Mamba & S4 Explained](https://www.youtube.com/watch?v=8Q_tqwpTpVU) — Umar Jamil. First-principles animated derivation.

**General interactive playgrounds**
- [Distill.pub](https://distill.pub/) — home of interactive, animation-rich ML explainers.
- [Explained Visually (setosa.io)](https://setosa.io/ev/) — PCA, eigenvectors, image kernels, Markov chains, and more.

## Behavioral & research-scientist-specific

> [!WARNING] The under-served category
> Public resources often cover the **job talk, research deep-dive, and paper-impact storytelling** less systematically than coding and ML question banks. Use the resources below as a starting point, then supplement them with the actual role's rubric and mock-interview feedback.

- **STAR / STAR-L method** — the [MIT CAPD worksheet](https://capd.mit.edu/resources/the-star-method-for-behavioral-interviews/) provides the basic STAR structure. This book adds **Learning**, uses a larger Action share as a starting heuristic for a 2–3 minute answer, and distinguishes "I" from "we." These are not fixed MIT rules.
- **PhD → industry transition** — Rutgers "From Bench to Offer", NIH OITE blog, Cheeky Scientist. Key reframe: *academia rewards nuance and caveats; interviews reward clear decisions and measurable outcomes.* Translate research into **decision + measurable impact** language.
- **The research job talk** — confirm format with the recruiter; lead with problem significance and deliverables; demonstrate science communication + depth; anticipate deep Q&A; tailor framing to the team's product.

## Mock interviews

- **[Pramp](https://www.pramp.com/)** — peer-to-peer mock-interview service. Check the current free/paid split and available categories before booking.
- **[interviewing.io](https://interviewing.io/)** — anonymous mocks with senior engineers; recorded, with a large free technical-talk / question library.
- **Trusted peers and alumni** — ask someone who understands the target role for a research-depth mock. Do not request leaked questions or company-confidential information; ask only for feedback on the evaluation rubric and your delivery.

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

Use these as secondary sources for company- and role-specific experiences. Anonymous posts have substantial sampling, timing, region, and team bias, so rank them below current recruiter guidance and official preparation pages.

- [Glassdoor Interview Questions](https://www.glassdoor.com/Interview/index.htm) — search "Meta Research Scientist interview", etc.
- [Blind (teamblind.com)](https://www.teamblind.com/) — anonymous big-tech employee reports.
- [1point3acres](https://www.1point3acres.com/bbs/) — strong for MSRA and China-office loops.
- [LeetCode Discuss — Interview Experience](https://leetcode.com/discuss/interview-experience) — company-tagged real coding tests.
- [levels.fyi](https://www.levels.fyi/) — reference distribution of self-reported levels and compensation. Check the sample, currency, equity price, region, and date; do not use it as the sole basis for offer or tax decisions.

## Personal research assets (interview packet)

This section alone is candidate-specific. Use public papers, code, and talks as evidence in the [research deep-dive](#/research/job-talk) and the STAR-L story bank; refresh citation counts and profile details immediately before applying.

<dl class="kv">
<dt>Google Scholar</dt><dd><a href="https://scholar.google.co.kr/citations?user=n_TR1LcAAAAJ">All publications</a> — citation context for impact stories.</dd>
<dt>GitHub</dt><dd><a href="https://github.com/qjadud1994">qjadud1994</a> — public code; proof of engineering rigor.</dd>
<dt>ZIM</dt><dd>Zero-shot Image Matting — <a href="https://naver-ai.github.io/ZIM/">project</a> · <a href="https://arxiv.org/pdf/2411.00626">paper</a>. Ties directly to SAM / alpha-matte / promptable segmentation talking points.</dd>
<dt>ECLIPSE</dt><dd>Panoptic segmentation — <a href="https://arxiv.org/abs/2403.20126">arXiv 2403.20126</a>. Continual / catastrophic-forgetting narrative.</dd>
<dt>PointWSSIS</dt><dd>Weakly-supervised instance segmentation — <a href="https://arxiv.org/abs/2303.15062">arXiv 2303.15062</a>. Label-efficiency story.</dd>
<dt>BESTIE</dt><dd>Weakly-supervised instance segmentation — <a href="https://arxiv.org/abs/2109.09477">arXiv 2109.09477</a>. Example of image-level supervision and pseudo-instance refinement.</dd>
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
