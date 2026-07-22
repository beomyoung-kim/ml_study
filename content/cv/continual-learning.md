# Continual Learning

<div class="tag-row"><span class="tag">catastrophic forgetting</span><span class="tag">stability–plasticity</span><span class="tag">regularization / replay / isolation</span><span class="tag">class-incremental seg</span><span class="tag">prompt tuning</span><span class="tag">ECLIPSE</span></div>

> [!NOTE] Goal of this chapter
> Humans generally do not forget what they already know when they learn something new. Neural networks, however, can **rapidly forget earlier tasks** after training on a new one—catastrophic forgetting. This chapter builds an intuitive, visual understanding of why that happens and how to prevent it. It will read most smoothly after [Transfer Learning & Backbone Freezing](#/cv/backbones-transfer) and [Segmentation](#/cv/segmentation).

## What and why

**Scenario:** data does not arrive all at once; it arrives gradually over time. At step 1, the model learns “cat” and “dog.” At step 2, it learns “car” and “bus.” After step 2, however, the model may barely recognize cats and dogs because the shared weights have been overwritten to fit the new classes.

**Why it is difficult:** keeping old data and retraining on it is a strong baseline, but storage, privacy, or cost constraints may limit that option. The goal of continual learning is to **retain old performance while learning new information under restricted access to the past**. The terms continual, incremental, and lifelong learning overlap in the literature, but their assumptions about task boundaries, streaming, and label spaces can differ, so define the protocol first.

<figure>
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- axes -->
  <line x1="55" y1="215" x2="600" y2="215" stroke="#98a3b2" stroke-width="1.5"/>
  <line x1="55" y1="25" x2="55" y2="215" stroke="#98a3b2" stroke-width="1.5"/>
  <text x="330" y="245" text-anchor="middle" fill="#98a3b2">training step (new classes keep arriving) →</text>
  <text x="20" y="120" fill="#98a3b2" transform="rotate(-90 20 120)">Task 1 accuracy</text>
  <!-- x ticks -->
  <g fill="#98a3b2" text-anchor="middle"><text x="120" y="232">1</text><text x="240" y="232">2</text><text x="360" y="232">3</text><text x="480" y="232">4</text><text x="560" y="232">5</text></g>
  <!-- naive fine-tuning: steep drop (red) -->
  <polyline points="120,45 240,150 360,190 480,205 560,210" fill="none" stroke="#e0533f" stroke-width="3"/>
  <circle r="5" fill="#e0533f"><animateMotion dur="3s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" path="M120,45 L240,150 L360,190 L480,205 L560,210"/></circle>
  <text x="470" y="200" fill="#e0533f">naive fine-tuning → forgetting</text>
  <!-- CL method: stays high (green) -->
  <polyline points="120,45 240,55 360,62 480,68 560,72" fill="none" stroke="#12a150" stroke-width="3"/>
  <circle r="5" fill="#12a150"><animateMotion dur="3s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" path="M120,45 L240,55 L360,62 L480,68 L560,72"/></circle>
  <text x="300" y="48" fill="#12a150">good CL method → retained</text>
</svg>
<figcaption>The central picture of catastrophic forgetting: after learning Task 1, naive fine-tuning (red) causes Task 1 accuracy to collapse as new tasks arrive, while a good continual-learning method (green) keeps it high and still learns new information.</figcaption>
</figure>

> [!TIP] Interview one-liner
> The candidate has two first-/co-author papers in this area: **SSUL** (NeurIPS 2021, exemplar class-incremental semantic segmentation) → **ECLIPSE** (CVPR 2024, continual *panoptic* segmentation via visual prompt tuning). A strong answer combines the general **stability–plasticity** framing with the segmentation-specific **background/no-object shift**, then argues when a *distillation-free, prompt-based* method beats replay or KD.

## 1 · Catastrophic forgetting & stability–plasticity

<strong>Catastrophic forgetting:</strong> learning a new task overwrites shared weights that encoded earlier tasks, causing old-task accuracy to collapse. It is the sharp end of the **stability–plasticity dilemma**.

- **Stability:** the ability to preserve what has already been learned. Too much—for example, freezing everything—prevents new learning.
- **Plasticity:** the flexibility to learn something new. Too much—for example, unrestricted fine-tuning—causes old knowledge to be forgotten.

<figure>
<svg viewBox="0 0 620 130" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="60" y1="70" x2="560" y2="70" stroke="#98a3b2" stroke-width="2"/>
  <circle cx="60" cy="70" r="6" fill="#0ea5e9"/><text x="60" y="100" text-anchor="middle" fill="#0ea5e9">max stability</text>
  <text x="60" y="42" text-anchor="middle" fill="#98a3b2">freeze: no forgetting,</text>
  <text x="60" y="28" text-anchor="middle" fill="#98a3b2">but no learning</text>
  <circle cx="560" cy="70" r="6" fill="#e0533f"/><text x="560" y="100" text-anchor="middle" fill="#e0533f">max plasticity</text>
  <text x="560" y="42" text-anchor="middle" fill="#98a3b2">fine-tune: learns new,</text>
  <text x="560" y="28" text-anchor="middle" fill="#98a3b2">forgets old</text>
  <circle cx="360" cy="70" r="7" fill="#12a150"/><text x="360" y="100" text-anchor="middle" fill="#12a150">good CL method</text>
</svg>
<figcaption>Too much stability prevents learning new classes; too much plasticity causes forgetting. To see where a method lies on this axis, report <b>base / new / all</b> metrics separately.</figcaption>
</figure>

Segmentation adds the challenges of missing pixel labels and shifts in the meaning of background. This can make it harder than classification under comparable conditions, but the magnitude of forgetting must be measured for the particular dataset, protocol, and model (§4).

## 2 · The three method families

| Family | Idea | Representatives | Cost / weakness |
| --- | --- | --- | --- |
| **Regularization** | protect important weights or outputs | EWC, LwF, MiB, PLOP | constraints can conflict; performance degrades over long sequences |
| **Replay** | rehearse old samples or features | iCaRL, RECALL, exemplar sets | storage + **privacy**; difficult panoptic design |
| **Parameter isolation** | add small task-specific parameters and freeze the rest | PNN, VPT, **ECLIPSE** | module growth; multi-forward inference |

<dl class="kv">
<dt>EWC</dt><dd>Approximates the posterior near the old optimum with a diagonal quadratic and anchors the weights there. In practice, $F_i$ uses the diagonal Fisher as a proxy for parameter importance, not a complete measure of importance: $\mathcal{L}=\mathcal{L}_t+\tfrac{\lambda}{2}\sum_i F_i(\theta_i-\theta_i^{*})^2$.</dd>
<dt>LwF / KD (knowledge distillation)</dt><dd>Train the model to imitate the <b>old model's outputs</b> on new data, without requiring old data. Segmentation variants include <b>MiB</b> (background modeling), <b>PLOP</b> (multi-scale feature distillation), and <b>CoMFormer</b> (panoptic query distillation).</dd>
<dt>Replay</dt><dd>Store a small set of exemplars, such as iCaRL herding, or use generative/feature replay. It is often among the strongest approaches, but storing images may be prohibited.</dd>
</dl>

## 3 · Class-incremental vs task-incremental (and the segmentation twist)

- **Task-incremental:** the task ID is provided at test time, making the problem easier.
- **Class-incremental:** predict over *all* classes seen so far without a task ID, making the problem harder—the standard segmentation setting.
- **Disjoint vs overlap:** specifies which step's images may contain a class and how pixels of non-current classes are labeled. In an overlap protocol, future classes may appear in earlier-step images but be labeled as background, creating background shift. Whether overlap is “more realistic” or “harder” depends on the data distribution and annotation policy.

## 4 · Background / no-object shift (segmentation-specific)

> [!QUESTION] “What makes continual *segmentation* different from continual classification?”
> **Short:** pixels from unannotated old and future classes at the current step can be folded into semantic **background**. A query model's **no-object** target belongs to unmatched queries and is not the same as pixel background, although its score can also drift as the query and class composition changes across steps. **Deep:** training old or future pixels as background produces gradients that reject representations the model must later accept as foreground. MiB models the background; SSUL uses an Unknown class plus exemplars. Instead of learning a query no-object head, ECLIPSE reconstructs its score at inference through a separate mechanism. Do not collapse these two shifts into the same label.

ECLIPSE's **logit manipulation**: rather than using a learned no-object head that drifts, compute the no-object score from the logits of other steps at inference:

$$s^{\text{no-obj}}_t = \delta\Big(\sum_{k<t} s^{\mathcal{C}^k}_t + \sum_{k>t} s^{\mathcal{C}^k}_t\Big)$$

$\delta$ is a post-hoc scalar, approximately 0.5 by default, tuned *without retraining*—because no-object is inherently a function of the class scores from all steps.

## 5 · ECLIPSE — visual prompt tuning for panoptic CL

> [!EXAMPLE] The mechanism
> **Step 1:** train the full **Mask2Former** on $\mathcal{C}^1$, then **freeze everything**. **Step $t>1$:** learn only a small **prompt set** $\mathbf{Q}^t$ (queries) plus a **per-step MLP head**, with $N^t \approx |\mathcal{C}^t|$ and a minimum of 10. At inference, aggregate the outputs of $\mathbf{Q}^{1:t}$. Trainable parameters are about **1.3%** of the model (paper: ~0.6M vs ~44.9M). Distillation-free, replay-free.

In ECLIPSE's reported ablations, **deep prompts** injected at multiple decoder layers produced better new-class quality than shallow prompts. Per-class sigmoid scores avoid directly competing classes from different steps within one softmax denominator. The frozen path prevents parameter drift, but it does not eliminate all interference during combined inference, so it does not guarantee “perfect stability” end to end. The full ablations, FLOPs, and CoMFormer comparison are in the <strong><a href="#/resume/eclipse">ECLIPSE deep dive</a></strong>.

<div class="proscons"><div><div class="pros-t">Why prompt tuning wins here</div>
No distillation weight or pseudo-label threshold to tune; tiny trainable footprint and memory; extremely strong resistance to forgetting; naturally distillation-free and replay-free, making it privacy-friendly.
</div><div><div class="cons-t">Costs</div>
A weak step-1 representation means the frozen backbone limits plasticity, although strong pretraining such as Swin-L/COCO can help; inference performs multiple prompt forwards; step-1 misclassifications are locked in.
</div></div>

## 6 · Prompt-based continual learning (the general trend)

Freezing a strong pretrained backbone and learning only prompts is an important family of continual-learning designs:

- **Classification:** **L2P** learns a prompt *pool* and selects prompts per input; **DualPrompt** uses general and expert prompts; CODA-Prompt is another representative.
- **Segmentation:** **ECLIPSE** adds per-step visual prompts on Mask2Former.
- **Why it works:** prompts can prevent shared-weight drift and reduce per-task storage. **Limitations:** the ceiling of the frozen representation, prompt-selection or aggregation errors, and parameter/latency growth with the number of tasks remain. A foundation backbone with prompts or LoRA is a candidate design, not the only recipe. See [Vision Foundation Models](#/cv/foundation-models).

## 7 · Why panoptic continual learning is “hard mode”

Things, including instance matching, stuff, and no-object drift all arrive at once, and PQ is unforgiving of recognition errors in RQ. **CoMFormer** pioneered panoptic CL through query distillation. ECLIPSE claims the first *distillation-free* panoptic-CL result and widens the gap on **long sequences**, where many short steps cause regularization and KD methods to erode.

## 7b · Benchmarks & protocols you should name

| Benchmark | Task | Typical protocols (base-step) |
| --- | --- | --- |
| Pascal VOC | semantic CL | 15-5, 15-1, 10-1 (disjoint / overlap) |
| ADE20K | semantic & panoptic CL | 100-50, 100-10, 100-5, 50-50 |
| Cityscapes (domain-incremental) | semantic | city/condition shifts |

Notation “X-Y” means X base classes followed by increments of Y. A small Y and many steps—for example, 100-5 with 11 tasks—form a *long-sequence* stress test in which forgetting compounds.

### Fix the evaluation contract before comparing

Continual-learning results are more sensitive to the protocol than a single score suggests. In the comparison table, state whether task boundaries are known; how much past raw data, features, or generator state may be stored; whether pretraining and external data are allowed; whether training is single-pass online or uses multiple epochs per step; and whether the task ID is available at test time.

Let $A_{t,k}$ be the score on task $k$ after training through step $t$. Two common summaries are:

$$
\text{FinalAvg}=\frac1T\sum_{k=1}^{T} A_{T,k},\qquad
\text{Forgetting}=\frac1{T-1}\sum_{k<T}\left(\max_{\ell\in\{k,\dots,T-1\}}A_{\ell,k}-A_{T,k}\right)
$$

Two models can have the same final average even when one failed to learn new tasks and the other forgot old tasks. Report base/new/all scores, the per-step accuracy matrix, memory, training FLOPs, inference passes, and parameter growth together. Include a joint-training upper bound and a naive sequential fine-tuning lower baseline. Negative forgetting may indicate backward transfer, so state whether it is clamped to zero.

> **Concept code — filling the score matrix $A_{t,k}$**

```python
A = np.full((T, T), np.nan)
for t in range(T):
    train_on_step(model, train_stream[t])       # current-step data only
    model.eval()
    with torch.no_grad():
        for k in range(t + 1):                  # reevaluate every task seen so far
            A[t, k] = evaluate(model, test_sets[k])
    model.train()

final_avg = np.mean(A[T - 1, :])
forgetting = np.mean([np.max(A[k:T - 1, k]) - A[T - 1, k]
                      for k in range(T - 1)])
```

## 8 · Q&A

<details class="qa"><summary>When would you use replay or KD instead of prompt tuning?</summary>
<div class="qa-body">

**Short:** when you can store data, need maximum plasticity, and have a weak backbone.

**Deep:** replay is often among the strongest raw performers and lets the whole network adapt, providing high plasticity. If privacy and storage are not constraints and you expect a few large steps, a small exemplar buffer plus KD may beat prompt tuning on new-class accuracy. Prompt tuning shines under privacy constraints, long sequences, tight compute, and a strong frozen backbone.
</div></details>

<details class="qa"><summary>Freezing prevents forgetting—what does it cost?</summary>
<div class="qa-body">

**Short:** it locks in step-1 errors and caps plasticity.

**Deep:** if step 1 knows only “car,” the frozen path will continue mapping “bus” toward “car.” ECLIPSE mitigates this with logit manipulation, where later-step logits suppress the wrong class through mutual competition, but the general lesson is that stability and plasticity trade off: a frozen model buys stability with plasticity.
</div></details>

<details class="qa"><summary>How do the three families degrade differently over long sequences?</summary>
<div class="qa-body">

**Short:** regularization can bottleneck on accumulated constraints, replay on buffer coverage, and isolation on module growth and shared-inference interference. Which fails first depends on the protocol.

**Deep:** as the number of steps grows, approximations and teacher errors in EWC/LwF can accumulate, while a fixed replay buffer provides thinner coverage per class. Parameter isolation prevents old-parameter drift, but shared normalization or heads, prompt selection, and class competition mean end-to-end forgetting is not guaranteed to be zero; its parameter and inference budgets also grow.
</div></details>

### Follow-ups
- *“Metrics?”* Report **base / new / all** scores, using PQ or mIoU, to separate stability from plasticity; on long sequences, add *forgetting*, the drop from the best previous score.
- *“Overlap vs disjoint?”* If future classes appear in earlier images but are labeled as background, overlap creates background shift. State the same data, ordering, and annotation policy when comparing difficulty.
- *“Product angle?”* An on-device feature may expand class coverage over time under a no-replay privacy constraint, making isolation or prompt methods natural candidates.

## Cheat-sheet

| Term | One-liner |
| --- | --- |
| Catastrophic forgetting | new-task learning erases old-task performance |
| Stability–plasticity | retain old vs learn new—the central trade-off |
| EWC / LwF | Fisher penalty / output distillation (regularization) |
| Replay | rehearse stored exemplars or features |
| Isolation / VPT | add and train small prompts; freeze the rest |
| Background shift | the meaning of “background/no-obj” changes by step |
| Logit manipulation | reconstruct no-obj from other steps' logits (ECLIPSE) |
| base/new/all | report separately to expose the trade-off |

**Next:** [Segmentation](#/cv/segmentation) · [Weak & Semi-Supervised](#/cv/weak-semi-supervised) · [Vision Foundation Models](#/cv/foundation-models) · [ECLIPSE deep dive](#/resume/eclipse)
