# Deep-Dive: Phoenix — Adversarial Mask Refinement

<div class="tag-row"><span class="tag">ECCV 2026</span><span class="tag">mask refinement</span><span class="tag">adversarial perturbation</span><span class="tag">contrastive learning</span><span class="tag">SAM-based</span><span class="tag">first author</span></div>

> [!TIP] 30-second pitch
> Phoenix is a **mask-refinement** framework that improves a coarse mask without retraining its input segmentation model. It starts from the observation that the morphological noise used by previous refiners—dilation and erosion—does not adequately resemble real model errors. Phoenix proposes (1) **AMP**, which creates model-aware hard perturbations through an embedding-space adversarial attack, and (2) **CMRL**, which learns the relation among ground-truth, noisy, and refined masks. Under the paper's evaluated protocols, it reports higher results than prior refiners and gains of up to **+16.1 AP points** for weakly/semi-supervised pseudo-label refinement.

**Public status/reference:** The author's [publication page](https://beomyoung-kim.github.io/) lists the work as **ECCV 2026** (checked July 21, 2026). The separate [project page](https://phoenix-eccv26.github.io) may change availability, so check it again immediately before applying. See also [Segmentation](#/cv/segmentation), [Weak & Semi-Supervised](#/cv/weak-semi-supervised), and [Vision Foundation Models](#/cv/foundation-models).

## Problem and motivation

Even strong segmentation models such as Mask2Former and SAM still produce masks with **inaccurate boundaries, semantic confusion, and broken structure**. **Mask refinement** repairs those masks as postprocessing without changing the source architecture. This is especially useful when the source model cannot be modified because of proprietary or compute constraints, or when weakly/semi-supervised learning needs to turn a **pseudo-label into trustworthy supervision**—directly connected to the motivation in [PointWSSIS](#/resume/pointwssis-bestie).

The heart of refinement training is how to create **(noisy, clean) mask pairs**. This is also the weakness of earlier methods:

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="160" y="20" text-anchor="middle" font-weight="700" fill="#98a3b2">Morphological noise (prior work)</text>
  <rect x="60" y="35" width="200" height="120" rx="8" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <!-- blob with random jagged noise -->
  <path d="M120 70 Q160 55 200 75 Q215 100 195 125 Q160 140 125 122 Q108 98 120 70 Z" fill="none" stroke="#6366f1" stroke-width="2"/>
  <g stroke="#6366f1" stroke-width="1" opacity="0.6">
    <circle cx="118" cy="72" r="3"/><circle cx="205" cy="90" r="3"/><circle cx="130" cy="130" r="3"/><circle cx="195" cy="128" r="3"/>
  </g>
  <text x="160" y="178" text-anchor="middle" fill="#6366f1">random · semantics-agnostic</text>
  <text x="160" y="195" text-anchor="middle" fill="#98a3b2" font-size="10">edge correlation ≈ 0</text>

  <text x="480" y="20" text-anchor="middle" font-weight="700" fill="#e0533f">Adversarial (Phoenix)</text>
  <rect x="380" y="35" width="200" height="120" rx="8" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <path d="M440 70 Q480 55 520 75 Q535 100 515 125 Q480 140 445 122 Q428 98 440 70 Z" fill="none" stroke="#e0533f" stroke-width="2"/>
  <!-- noise concentrated near a semantically hard boundary (top-right) -->
  <path d="M505 62 Q525 70 528 88" fill="none" stroke="#e0533f" stroke-width="3" opacity="0.8"/>
  <path d="M512 66 Q532 76 534 92" fill="none" stroke="#e0533f" stroke-width="2" opacity="0.5"/>
  <text x="480" y="178" text-anchor="middle" fill="#e0533f">focused on boundaries · ambiguity</text>
  <text x="480" y="195" text-anchor="middle" fill="#98a3b2" font-size="10">edge correlation [−0.6, 0.8]</text>
</svg>
<figcaption>A conceptual comparison of prior morphological noise (left) and Phoenix's model-aware perturbation (right). The paper reports a wider distribution of Pearson correlation between adversarial noise and image edges, but that result must not be interpreted as calibrated predictive uncertainty.</figcaption>
</figure>

> [!NOTE] The one-line thesis
> “Real segmentation errors are **structured and context-dependent**; morphological noise is not. I therefore repurposed adversarial perturbation from a destructive attack into a constructive data-generation tool.” This is the sentence to say aloud.

## Contribution 1 — Adversarial Mask Perturbation (AMP)

**Idea:** Create a realistic noisy mask from a ground-truth mask, but guide the noise toward **locations that confuse the model**. Phoenix freezes a pretrained SAM decoder and runs an **FGSM-style adversarial attack** in embedding space.

<dl class="kv">
<dt>What is attacked?</dt><dd>Not the input pixels, but a <b>trainable perturbation embedding $\mathbf{E}_p$ in embedding space</b>. This provides (1) compute efficiency—the lightweight 4M decoder is reused, and one perturbation update takes about 6 ms on a V100—and (2) perturbation at the level of <b>higher-level semantics</b>, not raw pixels.</dd>
<dt>FGSM update</dt><dd>$\mathbf{E}_p \leftarrow \mathbf{E}_p + \alpha\cdot\mathrm{sign}(\nabla_{\mathbf{E}_p}\mathcal{L}_{adv})$. The loss gradient signals an embedding-space direction to which the current decoder is sensitive. A large gradient is not the same as calibrated predictive uncertainty, and the FGSM sign update does not retain the magnitude itself. It is therefore more accurate to say that AMP <b>creates model-aware hard perturbations</b> than that it “measures uncertainty precisely.”</dd>
<dt>Controlling direction with a guidance mask</dt><dd>Change the guidance mask $\mathcal{M}_g$ in the adversarial objective $\mathcal{L}_{adv}=-\mathcal{D}(f_{dec}(\cdot),\mathcal{M}_g)$ to <b>control the error type</b>: <b>Expansion</b> ($\mathcal{M}_g=\mathbf{1}$) creates false positives by expanding outside the boundary; <b>Contraction</b> ($\mathcal{M}_g=\mathbf{0}$) creates false negatives through under-segmentation; and <b>Inversion</b> ($\mathcal{M}_g=1-\mathcal{M}_t$) balances the two.</dd>
<dt>IoU-adaptive strength</dt><dd>Repeat FGSM until the IoU between generated noisy mask and target enters $[\tau,\tau+\epsilon]$. If it does not, reduce the step size with $\alpha\!\leftarrow\!\alpha/10$ and retry (Algorithm 1). During training, sample $\tau\!\sim\!\mathcal{U}(0.3,0.9)$ to cover difficulty from mild to aggressive noise.</dd>
</dl>

> [!IMPORTANT] Why attack embeddings rather than pixels?
> A conventional adversarial attack perturbs the input or representation in the direction that increases prediction loss. Phoenix reuses that direction to generate hard noise for training a refiner. Reusing the frozen encoder embedding reduces computation, and the authors interpret the resulting perturbations as closer to model errors than morphology-only noise. The defensible framing is: “We repurposed adversarial perturbation for training-data generation.”

## Contribution 2 — Contrastive Mask Refinement Learning (CMRL)

**Idea:** Conventional refinement treats the task as per-pixel classification, independently at each pixel. Phoenix instead models the relation among **three masks—ground truth ($\mathcal{M}_t$), noisy ($\mathcal{M}_n$), and refined ($\mathcal{M}_r$)—with three-way contrast in feature space**.

First, classify pixels into **six regions** according to their states in the three masks: True, Success, and Failure × {foreground, background}:

$$
\mathcal{T}_{fg}=(\mathcal{M}_t{=}1)\wedge(\mathcal{M}_n{=}1)\wedge(\mathcal{M}_r{=}1),\quad
\mathcal{S}_{fg}=(\mathcal{M}_t{=}1)\wedge(\mathcal{M}_n{=}0)\wedge(\mathcal{M}_r{=}1),\quad
\mathcal{F}_{fg}=(\mathcal{M}_t{=}1)\wedge(\mathcal{M}_n{=}0)\wedge(\mathcal{M}_r{=}0)
$$

(The background regions $\mathcal{T}_{bg},\mathcal{S}_{bg},\mathcal{F}_{bg}$ are defined symmetrically.) Three losses operate on these regions:

<dl class="kv">
<dt>Intra-Class Consistency $\mathcal{L}_{intra}$</dt><dd>Uses InfoNCE to <b>pull together</b> features within the same semantic region—foreground or background—drawing failed-pixel features toward correct same-class features.</dd>
<dt>Inter-Class Contrast $\mathcal{L}_{inter}$</dt><dd><b>Pushes foreground-failure features away from every background feature</b>, bidirectionally, sharpening the foreground/background boundary where errors cluster.</dd>
<dt>Self-Improvement $\mathcal{L}_{self}$</dt><dd>The key innovation. It makes features in a current <b>failure region ($\mathcal{F}$)</b> resemble features in an already <b>corrected region ($\mathcal{S}$)</b>, explicitly learning the wrong→correct transition by <b>bootstrapping</b> from successes within the same image.</dd>
</dl>

$$\mathcal{L}_{CMRL}=\lambda_{intra}\mathcal{L}_{intra}+\lambda_{inter}\mathcal{L}_{inter}+\lambda_{self}\mathcal{L}_{self}\quad(\lambda=0.4,0.4,0.2)$$

Training adds $\mathcal{L}_{CMRL}$ to SAM's Dice plus Focal loss.

> [!EXAMPLE] Why three-way? How it differs from conventional contrastive learning
> SimCLR/CLIP-style methods learn representations through two directions—positive and negative. CMRL models the relation among **three masks: ground truth, noisy, and refined**. In particular, $\mathcal{L}_{self}$ pulls a still-wrong pixel toward a just-corrected pixel, directly modeling the **wrong→correct transition** in refinement. This self-improvement signal is absent from a conventional pixel loss.

## Architecture and training

<dl class="kv">
<dt>Backbone</dt><dd>Based on pretrained <b>SAM (ViT-H)</b>. The 637M encoder is <b>frozen</b>, and only the lightweight <b>4M decoder is fine-tuned</b>. [ZIM](#/resume/zim) is in the SAM lineage while [ECLIPSE](#/resume/eclipse) is based on Mask2Former, but all connect at the higher-level principle of limiting parameter drift in a pretrained trunk and adapting with a small module.</dd>
<dt>Training</dt><dd>AdamW, learning rate $1\times10^{-4}$ with cosine decay, 10K iterations, batch 16, under <b>10 hours on 8×V100</b>. AMP uses a Dice objective, $N{=}10$, $\alpha_0{=}0.01$, and $\epsilon{=}5\%$.</dd>
<dt>Data/evaluation</dt><dd>Train on LVIS under the same setup as SegRefiner. Evaluate instance segmentation on COCO and fine-grained masks on DIS5K/ThinObject-5K. Metrics include AP, boundary AP, IoU, and Boundary-F.</dd>
</dl>

## Results to remember

<dl class="kv">
<dt>Weak/semi-supervised pseudo-label refinement</dt><dd>Under the relevant paper protocols, up to <b>+16.1 AP<sup>mask</sup> points and +17.3 AP<sup>boundary</sup> points</b>. Cite these only with the dataset, label setting, and baseline from the corresponding table.</dd>
<dt>Refining strong models</dt><dd>Consistent improvement across outputs from Mask R-CNN, SOLO, Mask2Former, ViTDet, MaskDINO, and other architectures—for example, +6.5 AP on Mask R-CNN R101.</dd>
<dt>Fine-grained (DIS)</dt><dd>On thin structures and complex topology, <b>mean-IoU gains of 11–21%</b> over SAMRefiner and SegRefiner.</dd>
<dt>Components (4f)</dt><dd>AMP alone +4.6 AP<sup>1</sup>, CMRL alone +1.5, and <b>+6.4</b> together—a <b>synergy</b> larger than the individual sum, interpreted as realistic noise making three-way contrast more meaningful.</dd>
<dt>Testing “realism” (4e)</dt><dd>AMP noise trains a better refiner than using actual errors from UNet/ISNet outputs, which the paper attributes to broader error diversity and controllable strength—a functional-substitutability argument.</dd>
</dl>

## Q&A

<details class="qa"><summary>In one sentence, why is morphological noise such as SegRefiner's insufficient?</summary>
<div class="qa-body">

**Short:** Dilation and erosion randomly disturb a ground-truth boundary without semantics; they do not model *where and why* a real model fails, such as a complex boundary or confusion between similar objects.

**Deep:** The paper's Pearson-correlation analysis is the quantitative evidence. Morphological noise has a narrow correlation distribution around zero with image edges or texture, whereas adversarial noise spans roughly [−0.6, 0.8]. In ablation 4d, training Phoenix with morphological noise sharply reduces performance, while supplying AMP noise to SegRefiner substantially improves it. The same noise helps across architectures, isolating it as a cause of the gain.
</div></details>

<details class="qa"><summary>Why attack embedding space instead of pixels?</summary>
<div class="qa-body">

**Short:** Efficiency plus semantic-level perturbation. Reuse the frozen encoder's image embedding and run only the 4M decoder, so one update takes about 6 ms, and perturb higher-level semantics rather than raw pixels.

**Deep:** The adversarial objective's gradient provides the local direction that increases the current decoder loss fastest. This motivates a **model-aware failure pattern** beyond morphology-only noise; it does not make the gradient norm a calibrated estimate of uncertainty. The perturbation embedding $\mathbf{E}_p$ is used only to generate training noise and is absent from the inference path, so it adds no deployment cost.
</div></details>

<details class="qa"><summary>What exactly does CMRL's self-improvement loss do?</summary>
<div class="qa-body">

**Short:** Within one image, it pulls the feature of a still-wrong pixel ($\mathcal{F}$) toward that of a just-corrected pixel ($\mathcal{S}$), bootstrapping the wrong→correct transition from the model's own successes.

**Deep:** Conventional contrast aligns against static positives and negatives. $\mathcal{L}_{self}$ instead models the **dynamic transition** of refinement. AMP makes the contrast between corrected regions $\mathcal{S}$ and uncorrected regions $\mathcal{F}$ more meaningful through a realistic failure distribution, motivating the joint gain (+6.4) beyond the individual gains (+4.6 and +1.5).
</div></details>

<details class="qa"><summary>How does this work connect to your earlier research?</summary>
<div class="qa-body">

**Short:** It is a conceptual successor and generalization of [PointWSSIS](#/resume/pointwssis-bestie)'s **MaskRefineNet**, which refined coarse point-supervised proposals. It pushes the same “pseudo-label quality is the bottleneck” thesis into generation of the refiner's own training pairs.

**Deep:** ZIM and Phoenix use SAM-lineage representations, while ECLIPSE builds on Mask2Former. The common design axis is not one backbone name, but preserving a pretrained representation and adapting it through a small module. Phoenix's weak/semi-supervised evaluation directly meets PointWSSIS's pseudo-label-quality problem.
</div></details>

### Likely follow-ups

- *Why sample $\tau$ from $\mathcal{U}(0.3,0.9)$ instead of fixing it?* To cover both mild and aggressive noise. High $\tau$ favors high-quality masks and low $\tau$ favors low-quality masks, so randomization balances robustness across real error strength.
- *Why Dice for the adversarial objective?* In ablation 4b it gives the best balance over L1, MSE, BCE, and Focal and is relatively robust across noise types.
- *Why can AMP outperform training on real model errors in 4e?* Errors from one model are biased toward that model's failure modes. AMP offers broader error diversity and controllable strength, improving generalization; call this functional substitutability, not proof that it is literally more realistic in every sense.
- *Limitations?* Dependence on randomized perturbation, though five repeats report AP¹ 28.7±0.5, plus the shared limitation of refiners: recovery is difficult when the input mask is completely wrong.

## Cheat-sheet

| Concept | One line |
| --- | --- |
| Problem | Morphological noise does not mimic real segmentation errors |
| AMP | Embedding-space FGSM creates model-aware hard noise that increases the current decoder loss |
| Principle | Uses a loss-increasing local direction; gradient magnitude is not calibrated uncertainty |
| Guidance mask | Expansion, Contraction, and Inversion control error type |
| CMRL | Three-way GT/noisy/refined contrast: intra, inter, and **self** |
| Self-loss | Bootstrap failure pixels toward corrected pixels—the wrong→correct transition |
| Backbone | SAM ViT-H with frozen encoder; fine-tune only the 4M decoder |
| Result | Under paper protocols, up to +16.1 AP points in weak/semi refinement; cite DIS deltas with their metric definition |
| Synergy | AMP (+4.6) + CMRL (+1.5) → +6.4 jointly, larger than the sum |
| Lineage | Generalizes PointWSSIS's refinement problem and connects to the freeze-plus-small-adaptation design axis across different backbones |

**Related:** [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [Segmentation](#/cv/segmentation) · [Weak & Semi-Supervised](#/cv/weak-semi-supervised) · [Predicted Questions & Answers](#/resume/predicted-questions)
