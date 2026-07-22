# Debugging & Experimentation

> [!NOTE] Goal of this chapter
> The first time you run training code, you will almost certainly encounter "the loss will not go down" or "I do not know why this is failing." Neural networks can be **silently wrong**—they simply fail to learn without throwing an error. This chapter gives you a systematic **playbook** for narrowing down the cause instead of guessing. It is a practical guide for readers who have already run the training loop from [What Is Machine Learning?](#/foundations/what-is-ml).

> [!TIP] Interview one-liner
> A strong opening is: *"First I disable augmentation and regularization and check whether the model can overfit a very small batch, narrowing the problem to the data, loss, backpropagation, or optimizer path."* Failure is strong evidence of a bug, but capacity or optimization may also be responsible, so isolate the evidence one step at a time.

## Your first weapon: overfit one batch

When something fails, changing the learning rate is not the first move. **Take a tiny dataset—perhaps eight examples—and drive its loss almost to zero** (Karpathy's recipe). This one test divides the problem in half:

- **Overfitting succeeds** → evidence that, at least on these few examples, the forward pass, backward pass, and optimizer can reduce the loss. It does not prove that the full pipeline is free of leakage or evaluation errors.
- **Overfitting fails** → begin with bugs such as a label/loss mismatch, frozen parameters, or an incorrect mode, while also checking insufficient capacity, overly strong regularization or augmentation, an unsuitable learning rate, and stochastic or impossible targets.

Run through this fast checklist before anything else:

1. Are `model.train()` and `eval()` toggled correctly? Batch normalization and dropout behave differently by mode.
2. Is the loss finite, with no `NaN` or `inf`?
3. Are the label range and `ignore_index` correct?
4. Is the trainable-parameter count greater than zero, and is the learning rate nonzero?
5. Does input normalization—mean and standard deviation—match the checkpoint?
6. **Look at** one batch and confirm that images, masks, and labels are aligned.

> [!NOTE] Sanity-check the initial loss as well
> If the initial logits are nearly equal for every class and softmax is uniform, per-example cross-entropy should be near $\ln C$, regardless of class frequency. If those assumptions hold but the value is far away, inspect label ranges, reduction, initialization, and masking. Do not apply this check unchanged to a pretrained model, a biased head, or a model initialized with class priors.

## Reading loss curves—the shape is the diagnosis

Most debugging comes down to recognizing the shape of the loss curve. Here are four typical shapes and what they mean:

<figure>
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <!-- panel 1: healthy -->
  <text x="90" y="18" text-anchor="middle" fill="#12a150" font-weight="700">healthy</text>
  <line x1="30" y1="30" x2="30" y2="120" stroke="#98a3b2"/><line x1="30" y1="120" x2="160" y2="120" stroke="#98a3b2"/>
  <path d="M35 40 C 70 55, 110 100, 155 112" fill="none" stroke="#12a150" stroke-width="2.5"/>
  <text x="95" y="140" text-anchor="middle" fill="#98a3b2">smooth decline</text>
  <!-- panel 2: diverging -->
  <text x="250" y="18" text-anchor="middle" fill="#e0533f" font-weight="700">diverging</text>
  <line x1="190" y1="30" x2="190" y2="120" stroke="#98a3b2"/><line x1="190" y1="120" x2="320" y2="120" stroke="#98a3b2"/>
  <path d="M195 90 C 220 95, 245 90, 265 70 C 285 45, 300 35, 315 30" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="255" y="140" text-anchor="middle" fill="#98a3b2">LR ↑ / NaN risk</text>
  <!-- panel 3: plateau -->
  <text x="410" y="18" text-anchor="middle" fill="#d97706" font-weight="700">plateau</text>
  <line x1="350" y1="30" x2="350" y2="120" stroke="#98a3b2"/><line x1="350" y1="120" x2="480" y2="120" stroke="#98a3b2"/>
  <path d="M355 55 L475 55" fill="none" stroke="#d97706" stroke-width="2.5"/>
  <text x="415" y="140" text-anchor="middle" fill="#98a3b2">LR=0 / frozen</text>
  <!-- panel 4: overfit -->
  <text x="570" y="18" text-anchor="middle" fill="#6366f1" font-weight="700">overfitting</text>
  <line x1="510" y1="30" x2="510" y2="120" stroke="#98a3b2"/><line x1="510" y1="120" x2="635" y2="120" stroke="#98a3b2"/>
  <path d="M515 45 C 545 70, 575 100, 630 110" fill="none" stroke="#12a150" stroke-width="2"/>
  <path d="M515 55 C 545 80, 575 78, 630 60" fill="none" stroke="#e0533f" stroke-width="2" stroke-dasharray="4 3"/>
  <text x="572" y="140" text-anchor="middle" fill="#98a3b2">train↓ val↑</text>
  <text x="572" y="160" text-anchor="middle" fill="#12a150">train</text><text x="620" y="160" text-anchor="middle" fill="#e0533f">val</text>
</svg>
<figcaption>From left: healthy descent; upward divergence, often from an excessive learning rate and approaching NaN; a flat plateau from LR=0, frozen parameters, or a wrong loss; and overfitting, where training loss falls while validation loss rises. The curve's shape determines your first response.</figcaption>
</figure>

## LR range test

Before committing to a schedule, run an **LR range test**: increase the learning rate exponentially over a few hundred steps and plot loss against LR.

<figure>
<svg viewBox="0 0 520 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="50" y1="20" x2="50" y2="160" stroke="#98a3b2"/><line x1="50" y1="160" x2="490" y2="160" stroke="#98a3b2"/>
  <text x="30" y="90" fill="#98a3b2" transform="rotate(-90 30 90)">loss</text>
  <text x="260" y="185" fill="#98a3b2" text-anchor="middle">learning rate (log scale) →</text>
  <path d="M60 60 C 150 62, 200 70, 250 95 C 300 120, 330 135, 360 130 C 400 120, 430 60, 470 25" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <line x1="300" y1="20" x2="300" y2="160" stroke="#12a150" stroke-dasharray="4 3"/>
  <text x="300" y="16" fill="#12a150" text-anchor="middle">steepest descent → choose here</text>
  <text x="440" y="18" fill="#e0533f" text-anchor="middle">diverges</text>
</svg>
<figcaption>A good LR is around the point of steepest loss descent, roughly an order of magnitude below where the loss explodes. This is cheaper and more reliable than guessing.</figcaption>
</figure>

Pair the test with gradient-norm and weight-norm logging so you can see instability accumulate before a NaN appears. See [Normalization & Training Stability](#/foundations/normalization-stability) for the mechanism.

## Try it yourself—smooth a noisy loss curve

Real loss curves jump from batch to batch and obscure the trend. We therefore smooth them with a **moving average**, just as a debugging dashboard does. Implement a moving average with window size $k$: at each position, average the preceding $k$ values.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"moving_average","packages":["numpy"],"approx":true,"starter":"def moving_average(xs, k):\n    # xs: list of loss values; k: window size. Return the mean of each consecutive window of k values.\n    # Output length is len(xs) - k + 1. Example: moving_average([1,2,3,4], 2) -> [1.5, 2.5, 3.5]\n    pass","tests":[{"args":[[1,2,3,4],2],"expect":[1.5,2.5,3.5]},{"args":[[2,2,2,2,2],3],"expect":[2.0,2.0,2.0]},{"args":[[10,0,10,0],2],"expect":[5.0,5.0,5.0]}],"solution":"import numpy as np\n\ndef moving_average(xs, k):\n    x = np.asarray(xs, dtype=float)\n    out = [float(x[i:i+k].mean()) for i in range(len(x) - k + 1)]\n    return out"}
</script>
</div>

Removing the noise makes it much easier to see which of the four shapes above describes the curve.

<details class="qa"><summary>Your training loss will not go down. Walk me through your first ten minutes.</summary>
<div class="qa-body">

**Short:** Try to overfit a single tiny batch. If it fails, suspect a pipeline bug such as the wrong loss reduction, frozen parameters, `eval()` left on, a label/target mismatch, or LR=0. If it succeeds, the model can learn and the problem is scale or optimization.

**Deep:** Use ordered probes: (1) confirm that `requires_grad` parameters exist and their gradients are nonzero after `backward()`; (2) repeatedly print the loss on the same batch—it should approach zero; (3) sanity-check the initial loss, such as $\ln(\text{num\_classes})$ for uniform predictions under CE; (4) verify that LR is nonzero and the optimizer steps; and (5) visualize inputs and targets to catch transforms that corrupted labels. Only after "overfit one batch" passes should you touch the LR schedule, capacity, or regularization. **Follow-up:** *It overfits but does not generalize?* That is a regularization or data problem; see [Regularization & Generalization](#/foundations/regularization-generalization).
</div></details>

## "The loss goes down, but the metric does not"

This is common, and the diagnosis is a decision table:

| Observation | Hypothesis | Action |
| --- | --- | --- |
| train metric ↑, val metric flat | overfitting | augmentation, weight decay, more data |
| train metric also flat | loss ≠ target metric | metric-aware loss, error analysis |
| val loss ↓ but val metric flat | threshold/decoding issue | tune NMS / threshold / post-proc |
| both good, deploy bad | domain shift | target-distribution data, recalibrate |
| worse than random | eval bug | dump predictions, unit-test the metric |

CV examples: cross-entropy falls but **mIoU** stays flat, so the model may simply predict background. Knowledge-distillation loss falls while the student's task metric falls, so the student may be copying the teacher's *errors*. The habit that catches both is **monitoring intermediate quality signals**, not only training loss.

<details class="qa"><summary>Validation loss improves, but your target metric is flat. What is happening?</summary>
<div class="qa-body">

**Short:** The loss is a surrogate decoupled from the metric—usually a threshold or decoding mismatch, class imbalance that the loss exploits, or an evaluation bug. Diagnose it by dumping predictions and comparing against an oracle upper bound.

**Deep:** Losses such as CE optimize per-pixel or per-token likelihood, while metrics such as mIoU, mAP, and F1 are set- or ranking-based and threshold-sensitive. If CE falls while mIoU remains flat, the model may improve on easy majority pixels while boundaries and rare classes stagnate. Try a metric-aligned loss such as Dice, Lovász, or focal loss, and inspect per-class scores. Always sanity-check the harness by feeding ground truth as predictions: the metric should reach its ceiling, or the evaluation code is broken. **Follow-up:** *Which signal should drive early stopping?* The final reported metric, not the surrogate loss.
</div></details>

## Reproducibility

```python
import random, numpy as np, torch
random.seed(s); np.random.seed(s)
torch.manual_seed(s); torch.cuda.manual_seed_all(s)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # trade speed for determinism
```

Bit-exact execution may be possible when hardware and software are fixed and every operation has a deterministic implementation, but it is not guaranteed for every operation and platform combination. Also seed the DataLoader generator and each worker's Python and NumPy RNG, and make unsupported deterministic operations fail explicitly. Validate research conclusions by pinning the commit, environment, data snapshot, and configuration, then examining paired differences and confidence intervals across several seeds.

<details class="qa"><summary>Why can you not always obtain bit-exact reproducibility on GPUs?</summary>
<div class="qa-body">

**Short:** Floating-point addition is not associative, and GPUs can sum values in nondeterministic orders through atomic additions and reduction scheduling. Identical inputs can therefore produce slightly different results across runs, compounded by TF32, autotuned kernels, and distributed all-reduce ordering.

**Deep:** `cudnn.benchmark=True` autotunes kernels for each input shape—it is fast but can be nondeterministic—while `deterministic=True` forces reproducible kernels at a speed cost, and some operations lack deterministic implementations. Across ranks, all-reduce may combine partial sums in different orders. Chasing bit-exactness is therefore often the wrong goal. Fix seeds, log the commit, data, and configuration, and report **mean ± standard deviation across at least three seeds** so conclusions are robust to this noise. **Follow-up:** *Dataset versioning?* Use a content hash, DVC, or immutable snapshot path to pin "the data."
</div></details>

## Experiment tracking

Log **configuration** (hyperparameters, Git hash, data version), **curves** (training and validation loss, target metric, LR, gradient norm, weight norm), **system** statistics (GPU utilization, samples per second), **artifacts** (checkpoints), and **notes** (hypothesis and conclusion). Name runs `YYYYMMDD_hypothesis_short` and tag them with labels such as `baseline`, `bugfix`, and `sweep`. Anti-patterns include dozens of nameless runs, a best checkpoint stored on only one machine, and a single test score reported without variance. *Your future paper's ablation table comes directly from these logs.*

## Ablation discipline

<div class="proscons"><div><div class="pros-t">Good ablation</div>

- A clear, reproducible **baseline**
- **One factor** changed at a time, or an interpretable factorial design
- **Equal budget**—the same epochs, data, and tuning for every arm
- **Multiple seeds** with mean ± standard deviation
- An **oracle or upper bound** for context
- Only the modules on which the **claim** depends

</div><div><div class="cons-t">Bad ablation</div>

- Changing data, loss, and architecture together
- **Grid-searching only the new method**
- Model selection on the **test set**
- Gains only in a tiny toy setting
- No variance reported
- Table bloat that hides the real driver

</div></div>

The interview move is: *"First I list the axes that could explain the gain, then design ablations to rule each one out—for example, did the improvement come from the new loss or simply a larger backbone?"* A negative result is science: report the failure conditions and scope. See [Experiment Design](#/research/experiment-design) for detailed methodology.

<details class="qa"><summary>A reviewer says your gains might come from more compute rather than your method. How do you answer?</summary>
<div class="qa-body">

**Short:** Show a **compute-matched** ablation: train the baseline and your method with equal epochs, data, and tuning budget, then add a scaling curve so the improvement appears at several compute levels rather than at one lucky setting.

**Deep:** The failure mode is tuning only the new method's hyperparameters while running the baseline with stock settings, which confounds method and search budget. Controls are (1) equal-budget training and equal hyperparameter-search budget per arm; (2) the same metric across model sizes or token counts to show a consistent gap; (3) an oracle upper bound that limits the headroom; and (4) at least three seeds with variance so the delta exceeds the noise. If the gain disappears after compute matching, that is an honest and valuable negative result. **Follow-up:** *When do you stop ablating?* When marginal information falls below opportunity cost, or when the causal story supporting the claim is complete.
</div></details>

## Quick scenarios

<dl class="kv">
<dt>NaN from step 0</dt><dd>Non-finite input, LR too high, FP16, `log(0)` → BF16, anomaly detection, dump the batch.</dd>
<dt>Curve dead flat</dt><dd>LR=0, frozen parameters, wrong loss, bad labels → count `requires_grad`, run a manual forward pass.</dd>
<dt>Train perfect, val random</dt><dd>Broken evaluation or extreme overfitting/domain shift → visualize validation data, check the split.</dd>
<dt>Only multi-GPU degrades</dt><dd>Sampler, mean-versus-sum reduction, BatchNorm → one-GPU parity test; see <a href="#/foundations/distributed-training">Distributed Training</a>.</dd>
<dt>Will not reproduce</dt><dd>Seed, data order, a bug fix silently mixed in → pin the commit and container.</dd>
</dl>

## Cheat sheet

| Question | One-line summary |
| --- | --- |
| First move | Disable augmentation and regularization, then overfit a small batch to narrow the cause; distinguish bugs, capacity, and optimization when it fails |
| Initial-loss sanity | If initial predictions are uniform, CE should be near $\ln(\text{num\_classes})$; class balance is not required |
| Curve shape | descending = healthy · rising = high LR/NaN · flat = LR 0/frozen · train↓ val↑ = overfitting |
| LR range test | Sweep LR upward and plot loss; choose around the steepest descent, roughly 10× below divergence |
| Loss↓, metric flat | Surrogate ≠ metric, threshold, or evaluation bug; feed GT to test the harness |
| Reproducibility | Aim for statistical reproducibility—mean ± standard deviation across seeds—with a pinned commit, data, and configuration, not merely bit-exact output |
| Tracking | Log configuration, curves, system, artifacts, and notes; name and tag every run |
| Ablation | One factor, equal budget, multiple seeds, and an oracle; never tune only the new method |
| Multi-GPU divergence | Use a one-GPU parity test to isolate sampler, reduction, or synchronization bugs |
| Compute matched | Match epochs, data, and tuning across arms so a gain is not merely extra budget |

**Next:** [Normalization & Training Stability](#/foundations/normalization-stability) · [Distributed Training](#/foundations/distributed-training) · [Optimization](#/foundations/optimization) · [Regularization & Generalization](#/foundations/regularization-generalization) · [Evaluation Metrics](#/foundations/evaluation-metrics) · [Experiment Design](#/research/experiment-design)
