# Presenting Research

<div class="tag-row"><span class="tag">whiteboarding your work</span><span class="tag">2-min / 10-min / 30-min</span><span class="tag">tailoring to audience</span><span class="tag">figures that land</span></div>

> [!TIP] The meta-skill
> Presenting is not a slide-design task — it's **audience-modeling under a time budget**. The same work becomes an elevator pitch, a whiteboard chalk-talk, or a full job talk depending on *who's listening and for how long*. This chapter is the general skill; [The Research Job Talk](#/research/job-talk) is the specific 45-minute format. Beomyoung has real stage reps to draw on — DAN 24, Centum Digital Week 2025, NeurIPS 2021 Social — so speak from *verified habits*, not theory.

```mermaid
flowchart LR
  M[Motivation<br/>why care] --> P[Problem<br/>precise task]
  P --> I[Insight / Method]
  I --> E[Evidence<br/>results + ablation]
  E --> L[Limitations]
  L --> F[Future / fit]
  F --> Q[Q&A]
```

## One work, three lengths

The hardest skill is **graceful degradation**: cutting content without losing the thread. Build the 30-min version, then derive the shorter ones by *dropping detail, never the spine*.

| | **2-minute** (elevator / hallway) | **10-minute** (chalk-talk / screen) | **30-minute** (seminar / job talk) |
| --- | --- | --- | --- |
| Goal | Make them want more | Convey **one idea** + evidence | Depth + defensibility + trajectory |
| Content | Problem + your one result | Motivation → 1 insight → 2 evidences → next | Full arc + ablations + limitations + future |
| Slides/board | none / 1 | ~6–10 | ~18–22 + backup |
| Cut first | everything but the hook + result | background, secondary results | nothing — this is the full version |
| Keep always | the pain and your delta | the *mechanism* | the honest limitation |

### The 2-minute version (memorize)

> "I make pixel-level perception **label-efficient**, and I ship it: my work became a segmentation API that beats commercial tools and an image-matting model (ZIM, ICCV 2025 Highlight) inside CLOVA-X. Now I'm connecting that pixel-level grounding to language models so they reason from **visual evidence**, not guesses."

Pattern: **Pain → what you did → why it matters → where you're going**, in four breaths. No jargon, no acronym you don't immediately unpack.

### The 10-minute version

Sacrifice background; keep exactly one mechanism.

```mermaid
flowchart TB
  T0["0–1m Hook: a broken result"] --> T1["1–3m Problem + gap"]
  T1 --> T2["3–7m The ONE insight"]
  T2 --> T3["7–9m Evidence / demo"]
  T3 --> T4["9–10m Takeaway + next"]
```

### The 30-minute version

The [job-talk arc](#/research/job-talk) at full resolution: motivation → prior art → **contributions up front** → one deep-dive → results + ablations → impact → future/fit, with a backup deck for defense.

## Tailoring to the audience

> [!WARNING] Wrong altitude sinks talks
> The same slide is too shallow for an expert panel and too dense for a mixed room. **Read the room first**, then set altitude — this is a graded signal in the job talk.

| Audience | Set altitude to… | Lead with | Beomyoung's rep |
| --- | --- | --- | --- |
| Mixed / product (execs, PMs) | User value & impact first | A visible product pain, a demo | **DAN 24** — CLOVA-X Image Editing |
| Trend / broad tech | Narrative + one concrete anchor | Where the field is going, then *your* work as proof | **Centum Digital Week 2025** — agents |
| Expert research panel | Mechanism & evidence | The gap and your delta; defend choices | **NeurIPS 2021 Social** — SSUL |
| Hiring committee | Contribution clarity + trajectory | What *you* did, and the next question | the [job talk](#/research/job-talk) |

> [!EXAMPLE] The line to say about audience-tailoring
> "At DAN, user value came before research detail; at the NeurIPS Social, the technical core came first. I practice changing **altitude** for the room without changing the truth of the result."

<details class="qa"><summary>"You have a mixed audience — non-experts and experts. What's your first two minutes?"</summary>
<div class="qa-body">

**Short:** Open with a concrete, visible failure everyone understands (a jagged cut-out edge; a VLM confidently mislabeling an object), state your one-line promise, and *signal* the depth is coming so experts stay patient.

**Deep:** Give a single "on-ramp" everyone boards, then climb. Define each acronym once. Put a "for the experts, details in backup" pointer so you don't lose either group. Never open with the AI grand-narrative or a market-size chart — start with the *pain*.
</div></details>

## Whiteboarding your own work

Some rounds have **no slides** — you get a marker and "explain your best paper." Different skill: no polish to hide behind, all structure and clarity.

```mermaid
flowchart TD
  A[Announce the plan<br/>'I'll draw the pipeline, then the one trick'] --> B[Draw the spine<br/>input → model → output]
  B --> C[Add the ONE contribution<br/>in a different color]
  C --> D[Annotate the loss / data as you speak]
  D --> E[Pause: invite questions]
```

<details class="qa"><summary>"Whiteboard your most important result for me."</summary>
<div class="qa-body">

**Short:** Say the plan, draw the **pipeline spine** left-to-right, then add your contribution in a second color so it's visually obvious what's *yours*, narrating the loss/data as you go.

**Deep:** Manage the board like slides: reserve space (don't run off the edge), write the *thesis sentence* at the top and leave it, box the one equation that matters. Talk while you draw — silence reads as uncertainty. Invite questions at natural breaks; a whiteboard makes Q&A conversational, which favors you. → [Communication & Whiteboarding](#/playbook/communication).
</div></details>

> [!NOTE] Whiteboard hygiene
> Legible large letters · one diagram per board-wipe · thesis sentence stays up top · your contribution in a distinct color · don't erase what a questioner is pointing at.

> [!QUESTION] "When should you switch to the whiteboard mid-talk?"
> **Short:** The moment a question is about *mechanism* and your slide only shows the *result* — derive it live. **Deep:** Whiteboarding a follow-up ("let me draw why the loss behaves that way") signals you understand the work beyond the deck, and turns a defensive Q&A into a collaborative one. Keep a clean board reserved for exactly this.

## Figures that land

> [!TIP] The one-job rule
> Every figure answers **one** question; if it needs two, split it. The reader should get the point in ~5 seconds, then you narrate the nuance.

| Figure | Does one job | Trap to avoid |
| --- | --- | --- |
| **Teaser (Fig 1)** | "Here's the idea in one picture" | Marketing gloss with no information |
| **Pipeline** | Data/tensor flow; where *your* block sits | Every layer drawn = nothing emphasized |
| **Qualitative** | Show the claim (and a **failure** case) | Consistent crops; success-only cherry-pick |
| **Main table** | The comparison, with your row highlighted | Unreadable 12-column dump; no backbone/data noted |
| **Ablation** | Attribution of the gain | Curve with no axis labels / error bars |

**Design for the projector and both themes:** high contrast, few words, **highlight the number that matters** (bold/color only your row). For a before/after (ZIM's binary-vs-soft edge), place them adjacent at the same scale so the difference is undeniable. Prefer a redrawn schematic over a screenshotted paper figure. → [Experiment Design](#/research/experiment-design) for what makes an ablation figure *honest*.

## Delivery mechanics

- **Openers/closers memorized**, middle spoken freely — the two moments audiences remember.
- ~1 slide/minute; slide text ≤ ~6 lines; the thesis sentence large on the slide.
- Live demo? Always have a **muted video / static image fallback** — demos fail on stage.
- Q&A: restate → answer → "does that address it?"; never bluff. See the [job-talk Q&A frame](#/research/job-talk).
- English talk: rehearse **transition phrases** ("which brings me to…", "the key insight here is…") so momentum doesn't stall on word-finding.

### Rehearsal checklist (night before)

```
[ ] 2-min, 10-min, 30-min versions all runnable
[ ] Opening 30s + closing 30s memorized verbatim
[ ] Timed twice on the clock; marked where to cut at the 5-min warning
[ ] "What's the weakness?" answer ready  (see Reading & Critiquing Papers)
[ ] "Tell me about a failure" answer ready  (see Failure & Negative Results)
[ ] Team-fit slide reflects the JD keywords (grounding / agents / on-device)
[ ] Co-author credit accurate; 'I' vs 'we' clean
[ ] Demo fallback image loaded; screen-share + timer tested
```

### Follow-ups they'll push

- *"Compress your whole PhD into one sentence."* — the trajectory line: label-efficient perception → matting foundation model + product → grounded VLMs / visual agents.
- *"What was the hardest question you ever got, and how'd you handle it?"* — pick a real one; show composure and a follow-up.
- *"Your agenda sounds broad — what's the through-line?"* — region-verifiable visual grounding, from pixels to language.
- *"How do you prep differently for a Korean vs English talk?"* — content identical; English adds memorized transitions + slower pace.

## Opening 30 seconds (practice draft)

> "I work on making pixel-level perception **label-efficient** and getting it into real products — a segmentation API that outperforms commercial tools, and a matting model, ZIM, inside CLOVA-X image editing. Now I'm connecting that pixel- and region-level grounding to language-model agents, so they reason from *visual evidence* instead of unsupported description. Today I'll walk one idea from that trajectory in depth, its limitations, and the next question I'd open with this team."

## Cheat-sheet

| Item | One-liner |
| --- | --- |
| Degrade gracefully | Build 30-min, derive 10/2-min by dropping detail, never the spine |
| 2-min | Pain → what you did → why it matters → where you're going |
| Altitude | Product room = value first; expert panel = mechanism first |
| Whiteboard | Announce plan → draw spine → contribution in 2nd color → talk while drawing |
| Figures | One job each; ~5-sec readable; highlight only your row/number |
| Demos | Always a muted-video / static fallback |
| Openers/closers | Memorized verbatim; middle spoken freely |
| Q&A | Restate → answer → confirm; never bluff |

**Related:** [The Research Job Talk](#/research/job-talk) · [Reading & Critiquing Papers](#/research/papers) · [Failure & Negative Results](#/research/failure) · [Experiment Design & Ablations](#/research/experiment-design) · [Communication & Whiteboarding](#/playbook/communication) · [CV deep-dives →](#/resume/overview) · [Deep-Dive: ZIM](#/resume/zim) · [Agentic AI & Tool Use](#/llm/agents)
