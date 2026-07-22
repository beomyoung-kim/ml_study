# The Design Framework

<div class="tag-row"><span class="tag">9-step spine</span><span class="tag">problem → metric → data → model → serve → monitor</span><span class="tag">research vs product framing</span><span class="tag">45–60 min</span></div>

> [!TIP] Say this in the first 60 seconds
> "Before I design anything, let me clarify the use case, constraints, and how we'll measure success—then I'll sketch a baseline system and deep-dive wherever you want." Rubrics vary by organization, but this opening immediately demonstrates problem framing and structured thinking.

The worked cases in the [next chapter](#/system-design/case-studies) use this spine as a common starting point. The framework adapts [alirezadir's 9-step ML system-design outline](https://github.com/alirezadir/machine-learning-interviews/blob/main/src/MLSD/ml-system-design.md) into a loop that can be walked on a whiteboard. Do not memorize the order rigidly; reallocate time around the risks in the question.

> [!WARNING] The research/applied twist
> Product-MLE framing assumes garden-variety infra exists and rewards a clean data→features→ranking→serve→A/B pipeline. **Research/applied framing shifts the weight** toward *problem formulation, metric design, experimental rigor (ablations, baselines, failure analysis), and modeling novelty.* You still draw the whole system, but you spend your extra minutes where a scientist adds value: **what to measure and how to know you're right.** *(defensible — synthesized from RS/AS loop reports)*

## The 9 steps as one loop

```mermaid
flowchart TD
  S1[1 · Problem formulation<br/>use case, constraints, ML objective] --> S2[2 · Metrics<br/>offline ML + online business]
  S2 --> S3[3 · Architecture / MVP<br/>ML and non-ML components]
  S3 --> S4[4 · Data<br/>sources, labels, quality]
  S4 --> S5[5 · Features / representation<br/>preprocessing, encoders]
  S5 --> S6[6 · Model + offline eval<br/>baseline → ambitious, ablations]
  S6 --> S7[7 · Prediction service<br/>batch / online / on-device]
  S7 --> S8[8 · Online testing<br/>shadow → canary → A/B]
  S8 --> S9[9 · Scale, monitor, update<br/>drift, retrain, failure modes]
  S9 -.->|learnings re-scope| S1
  style S1 fill:#6366f1,color:#fff
  style S2 fill:#6366f1,color:#fff
  style S6 fill:#e0533f,color:#fff
  style S8 fill:#12a150,color:#fff
```

The dashed edge is the point: an ML system is **iterative**, not a waterfall. Say so out loud — "I'd ship the simplest thing that beats the current baseline, then let monitoring and error analysis re-scope the problem." That single sentence reads as production maturity.

## Time budget (illustrative 45-minute round)

| Phase | Min | What you must produce |
| --- | --- | --- |
| Clarify + success metrics | 6–8 | Assumptions written down; ML objective; 2–3 offline + 1–2 online metrics |
| High-level architecture | 6–8 | One box diagram, request path, offline vs online split |
| Data + features | 8–10 | Sources, labeling strategy, leakage/PII, representation |
| Model + offline eval | 8–10 | Baseline → ambitious model; **ablations + failure analysis** |
| Serving + scale | 6–8 | Latency budget, batch vs online, cost back-of-envelope |
| Online eval + monitoring + failure | 5–7 | A/B design, drift, rollback, degraded mode |

> [!NOTE] Read the room, don't recite
> A strong interviewer will interrupt to deep-dive one box. **Let them steer** — the budget above is your default when *they* don't. Never sprint through all nine steps to "finish"; depth on the box they care about beats breadth they didn't ask for.

Treat the table as one illustrative allocation, not a universal interview duration or company rubric. Ask how much time is available and which section the interviewer wants to explore, then scale the phases proportionally.

## Step 1 — Problem formulation (where research candidates win)

Translate a fuzzy goal into a precise ML problem, then interrogate it. The clarifying questions below are the highest-leverage minutes of the whole round.

| Axis | Ask | Why it forks the design |
| --- | --- | --- |
| **Users / scale** | DAU? QPS peak? request size (image res, tokens)? | batch vs online; model size ceiling |
| **Latency** | p50 / p99 target? mobile or server? interactive? | on-device vs cloud; distillation need |
| **Quality bar** | acceptable error rate? human in the loop? asymmetric cost? | metric choice; calibration; fail-open vs fail-closed |
| **Data** | labels exist? cold start? privacy/consent? | supervised vs weak/self-sup; data engine |
| **Cost** | GPU budget? cost per 1k requests? | quantization, cascades, caching |
| **Scope / horizon** | MVP now, or 18-month vision? | how ambitious a model to propose |

Then state the **ML framing** explicitly: is this binary/multi-label classification, dense prediction (segmentation/matting), retrieval, ranking, regression, or generation? Naming the framing — and the *reduction* you chose — is a research-taste signal. *"I'd frame moderation as multi-label detection with a per-policy threshold rather than one binary head, because policies have independent cost trade-offs."*

> [!QUESTION] "How is designing for a research role different from product MLE here?"
> **Short:** Same skeleton; I spend the extra minutes on *what to measure and how I'd know I'm right*, not on a candidate-generation → ranking funnel.
>
> **Deep:** A product answer optimizes a business KPI through existing infra. A research/applied answer treats the system as an **experimental apparatus**: I'll define the metric so it can't be gamed, propose a baseline the fancy model must beat, design the **ablations** that isolate *why* it works, and plan the **failure analysis** that tells me where it breaks. Infra awareness (FSDP, mixed precision, throughput) still matters — foundation-model scale demands it — but my differentiator is rigor about the objective and the evidence.

## Step 2 — Metrics: offline, online, guardrail

The most common failure is conflating them. Separate three tiers and connect them with a hypothesis.

<dl class="kv">
<dt>Offline ML metrics</dt><dd>What you use to select a model before deployment; they must match the <b>decision and operating point</b>. For imbalanced or rare positives, a PR curve and precision/recall at a threshold are often more direct, but ROC-AUC is not always inferior. Do not reduce calibration to ECE alone: inspect reliability diagrams, NLL/Brier score, class- and slice-level calibration, and risk/coverage at the target threshold. For dense prediction use mIoU, boundary-F, and SAD/MSE; for retrieval/ranking use Recall@k, nDCG, and MRR; for generation combine task metrics with validated human or judge evaluation.</dd>
<dt>Online business metrics</dt><dd>What the product actually cares about: edit-completion rate, retention, CTR, report rate, task success. You cannot A/B on these until you ship, so you need an offline <b>proxy</b> and a stated hypothesis linking them.</dd>
<dt>Guardrail metrics</dt><dd>Must-not-regress constraints: p99 latency, cost/req, crash rate, fairness across slices, safety violation rate. A win on the primary metric that trips a guardrail is <b>not</b> a win.</dd>
</dl>

> [!EXAMPLE] Metric design as a research signal
> Don't just name a metric — **defend it against gaming**. "mIoU averages over pixels, so a model can score well while destroying thin structures like hair; I'd add a **boundary-F** metric and a hard-case slice (fine detail, transparency) so the number I optimize matches the quality users see." That is exactly the judgment a research panel probes for. See [Evaluation Metrics](#/foundations/evaluation-metrics).

## Step 3 — Architecture / MVP

Separate the **online path**—input → preprocessing → model → postprocessing → storage/response—from the **offline path**—collection → labels → training → registry → deployment. Authentication, caches, queues, human review, and fallbacks are first-class components outside the model. Add rough QPS, payload, and latency budgets to the arrows, and mark single points of failure and backpressure boundaries.

## Steps 4–5 — Data and features

Data quality dominates model choice at this level. Cover, briefly:

- **Sources & labels** — licensed vs scraped vs synthetic; who labels, guideline versioning, inter-annotator agreement; weak/self-supervised signals when labels are scarce.
- **The data engine** — sampled production data → hard-case mining → active learning → re-label → retrain. Draw this loop; it's often the real product moat.
- **Splits & leakage** — split by *entity* (user/scene/identity), not by row, or your metric lies. Temporal splits for anything with drift.
- **PII / consent / retention** — design for lawful basis or consent, purpose limitation, data minimization, retention and deletion, and access auditing. Verify the concrete requirements for the jurisdiction, data type, and organization policy.
- **Representation** — which encoder/features; normalization; class imbalance handling (resampling, focal/reweighted loss).

## Step 6 — Model & offline evaluation

Rather than naming only one candidate, you can often propose a comparable **model ladder**:

```mermaid
flowchart LR
  B[Baseline<br/>heuristic / small model<br/>cheap, ships day 1] --> A[Ambitious<br/>SOTA-class model<br/>quality ceiling]
  A --> D[Distilled / quantized<br/>meets latency + cost]
  B -. must beat .-> A
  D -. within ε of .-> A
  style B fill:#0ea5e9,color:#fff
  style A fill:#e0533f,color:#fff
  style D fill:#12a150,color:#fff
```

- **Baseline first.** A rule or a small model that ships in a week. It sets the bar the ambitious model must clear and de-risks the whole project. Knowing *when a simpler baseline wins* is a senior signal.
- **Ablations.** The research heart of the answer: what does each component *buy*? Isolate the encoder, the loss term, the data source. "I'd ablate the boundary loss on the hard-hair slice to confirm the gain isn't just from more data."
- **Failure analysis.** Slice the errors (by demographic, resolution, class, scene). Report *where* it fails, not just aggregate accuracy — this is what distinguishes a scientist from a leaderboard chaser.
- **Distillation/quantization** to hit the serving budget without giving up the quality ceiling. Cross-link: [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency), [Distributed Training](#/foundations/distributed-training).

## Steps 7–9 — Serve, test online, monitor

**Serving pattern** follows the latency/cost/privacy constraints from Step 1:

| Pattern | Use when | Cost of getting it wrong |
| --- | --- | --- |
| Synchronous online API | interactive edits, auth, search | p99 blowups, capacity outages |
| Async queue / batch | video, bulk indexing, offline scoring | staleness, backlog |
| On-device | privacy, offline, ultra-low latency | model-size ceiling, fragmentation, no hotfix |
| Cascade (cheap → expensive) | cost control at scale | calibration of the router/threshold |

**Online testing and rollout** can proceed in stages when appropriate: **shadow** (mirror traffic only when safety and privacy permit) → **canary** (a small fraction, with rollback on guardrail breach) → **A/B** (when interference, ethics, and statistical-power conditions are satisfied) → ramp. Not every product can use this sequence unchanged; for high-risk or rare events, simulation, prospective validation, and a human gate may matter more. Predeclare the **hypothesis, primary metric, guardrails, and stopping rule**.

<details class="concept-code">
<summary>View as conceptual code</summary>

> This is Python-like **pseudocode** that separates deployment from experimental decisions. It is not a real statistical-testing or deployment-platform API.

```python
def staged_rollout(candidate, baseline, preregistered_plan):
    plan = validate_plan(preregistered_plan)  # Fix analysis unit, primary, guardrails, and horizon
    verify_artifact_hash(candidate)

    if privacy_and_side_effect_policy.allows_shadow:
        shadow_report = shadow(candidate, sampled_traffic(), actions_disabled=True)
        require_no_operational_regression(shadow_report)

    deployment = canary(candidate, traffic_fraction=0.01)
    while deployment.in_canary_window():
        health = monitor_guardrails(deployment)  # Latency, error, safety, cost
        if health.crosses_emergency_boundary:    # Predeclared immediate-stop boundary
            deployment.rollback()
            return "rolled_back"

    experiment = assign_stably_by_unit(          # Analysis unit such as user, not request
        units=eligible_population(), variants=[baseline, candidate], seed=plan.seed
    )
    results = run_until_planned_horizon(experiment, plan)  # No ad hoc daily peeking or early stop
    effect, interval = paired_or_cluster_aware_estimate(results, plan)
    if plan.primary_wins(effect, interval) and plan.guardrails_pass(results):
        return deployment.ramp_with_monitoring()
    deployment.rollback()
    return "no_ship"
```

</details>

**Monitoring & failure modes** — the box juniors skip and seniors lead with:

- *Operational:* QPS, error rate, latency, GPU util, cost.
- *Capacity & queues:* arrival rate, queue depth/age, saturation, admission control, timeouts, and retry storms. Under overload, specify backpressure, load shedding, and degraded mode instead of allowing an unbounded wait.
- *ML health:* prediction distribution, confidence drift, **data/concept drift** vs training, per-slice metrics.
- *Failure modes & degraded mode:* what happens on a bad model (rollback via registry), bad input (validation, fallback UI), abuse (rate limits + policy model), or outage. When possible, keep a **reduced-functionality but safer fallback**; if no safe fallback exists, fail closed, escalate to a human, or surface an explicit outage.

<details class="qa"><summary>Walk me through your reusable design checklist — the one you'd run in any design round.</summary>
<div class="qa-body">

**Short:** Nine boxes, and I say out loud which one the interviewer wants to deep-dive.

**Deep:**

1. **Problem formulation** — ask about users, scale, latency, cost, and privacy; choose the objective and reduction among classification, dense prediction, retrieval, ranking, and generation.
2. **Metrics** — connect offline decision/operating-point metrics, online KPIs, and latency/cost/fairness/safety guardrails with a proxy hypothesis.
3. **Architecture / MVP** — draw the online request path, offline data/training path, non-ML baseline, authentication, queues, and fallback.
4. **Data** — cover sources, labeling guidelines, entity/temporal splits, leakage, PII/consent, and the data-engine loop.
5. **Features / representation** — specify encoder, preprocessing, normalization, and train/serve parity.
6. **Model + offline eval** — baseline → ambitious → distilled; ablations, uncertainty, and failure slices.
7. **Serving** — batch/online/on-device/cascade; latency, capacity, cost, and backpressure.
8. **Online testing** — shadow → canary → A/B where valid; hypothesis, power, stopping rule, and rollback.
9. **Scale / monitor / update** — drift, queue and SLO health, slice metrics, retraining triggers, degraded mode, and governance.

Then: *"Which box do you want me to go deep on?"*
</div></details>

<details class="qa"><summary>You've got 45 minutes and the interviewer stays quiet. How do you spend them?</summary>
<div class="qa-body">

**Short:** If the scheduled round is about 45 minutes, I might use 8 minutes for clarification and metrics, 8 for architecture, about 20 for data and model, and about 8 for serving and monitoring—while narrating trade-offs throughout.

**Deep:** Do not infer intent from silence alone; ask, "Would you like me to go deeper here or continue end to end?" I front-load problem framing and metrics because they are costly to redo, and I share timeboxes aloud ("I'll spend about two more minutes on data, then move to the model") so the interviewer can redirect. If time is tight, I compress serving or modeling detail according to the role and the question.
</div></details>

### Follow-ups they'll push after your first answer

- *"Your metric went up but the product KPI didn't move — what happened?"* → proxy/KPI mismatch, or the metric was gameable; re-examine the offline metric's validity and the experiment power.
- *"How would you know your training/eval split is leaking?"* → suspiciously high offline vs online gap; check for entity overlap, temporal leakage, near-duplicates.
- *"What's the cheapest thing that beats today's system?"* → they're testing whether you reach for the baseline before the SOTA model.
- *"Which single component would you ablate first, and what result would change your mind?"* → name a falsifiable prediction; that's research maturity.

## Cheat-sheet

| Step | One-liner | Research emphasis |
| --- | --- | --- |
| 1 Problem | Clarify + state the ML objective and reduction | **High** — framing is taste |
| 2 Metrics | Offline · online · guardrail, with a proxy→KPI hypothesis | **High** — design against gaming |
| 3 Architecture | One box diagram; offline vs online split | Med |
| 4 Data | Sources, labels, splits/leakage, PII, data engine | High |
| 5 Features | Encoder, preprocessing, imbalance | Med |
| 6 Model + eval | Baseline → ambitious → distilled; **ablations + failure analysis** | **Highest** |
| 7 Serving | Batch / online / on-device / cascade; latency + cost | Med (infra-aware) |
| 8 Online test | Shadow → canary → A/B; hypothesis + stopping rule | High |
| 9 Monitor/scale | Drift, per-slice metrics, rollback, degraded mode | Med |

> [!TIP] The one-sentence close
> "I'd ship the baseline behind a shadow test, prove the ambitious model beats it on the hard-case slice *and* the guardrails, distill it to the latency budget, then let per-slice monitoring tell me what to fix next." That sentence contains all nine steps.

**Related:** [Resume-based stage-by-stage answer examples](#/resume/interview-stage-answers) · [Worked Case Studies](#/system-design/case-studies) · [Designing LLM/Agent Systems](#/system-design/llm-systems) · [Evaluation Metrics](#/foundations/evaluation-metrics) · [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) · [Distributed Training](#/foundations/distributed-training) · [Experiment Design & Ablations](#/research/experiment-design) · [The RS/AS Pipeline](#/process/pipeline)
