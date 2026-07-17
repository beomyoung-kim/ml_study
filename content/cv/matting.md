# Image Matting

<div class="tag-row"><span class="tag">alpha matte</span><span class="tag">trimap-free</span><span class="tag">SAM-guided</span><span class="tag">SAD / Grad / Conn</span><span class="tag">ZIM</span><span class="tag">BiRefNet</span></div>

> [!TIP] Why this chapter matters
> Matting is the candidate's strongest **research × product** intersection: **ZIM** (ICCV 2025 Highlight), WSSHM (weakly-semi human matting), a foreground-segmentation API beating commercial ones, and CLOVA-X Image Editing. The interview lever is articulating *why segmentation is not matting* and *why naively fine-tuning SAM on matting data destroys its zero-shot ability*.

## The problem

Decompose an observed image $I$ into foreground $F$, background $B$, and a per-pixel opacity (alpha matte) $\alpha \in [0,1]$:

$$I_i = \alpha_i F_i + (1-\alpha_i) B_i$$

Per pixel there are 7 unknowns ($F_i, B_i \in \mathbb{R}^3$, $\alpha_i$) and 3 observations — the problem is **massively under-constrained**. Priors come from a trimap, a coarse mask/prompt, or a learned foundation model.

```mermaid
flowchart LR
  I[Image] --> Net[Matting network]
  P["Guidance:\ntrimap / mask / point / box / text"] --> Net
  Net --> A["α ∈ [0,1]"]
  A --> C[Compositing / editing / green-screen]
  style A fill:#e0533f,color:#fff
```

## 1 · Matting vs segmentation

| | Segmentation | Matting |
| --- | --- | --- |
| Output | hard label {0,1} / class-id | soft $\alpha \in [0,1]$ |
| Boundary tolerance | a few px forgiven (IoU) | hair / fur / motion / glass must be exact |
| Metrics | mIoU / AP | SAD, MSE, **Grad**, **Conn** |
| Data | relatively abundant | high-quality mattes are rare & expensive |
| Resolution need | moderate | high — sub-pixel edges |

The composition equation is the whole reason $\alpha$ must be continuous: at a hair strand a pixel is *partly* foreground. A thresholded segmentation mask cannot represent partial coverage, so compositing it leaves **halos, color spill, and jagged edges**.

> [!QUESTION] "Why can't I just evaluate matting with IoU?"
> IoU thresholds $\alpha$ to a binary mask, discarding exactly the soft-transition information matting exists to model. A model that nails the torso but smears every hair strand can still post a high IoU. You must use SAD/MSE for magnitude and **Grad/Conn** for boundary structure.

## 2 · Guidance regimes

<dl class="kv">
<dt>Trimap-based</dt><dd>User (or a model) provides FG / BG / <b>Unknown</b> regions; the network only solves the Unknown band. Most accurate, highest UX cost. Classic: <b>Deep Image Matting (DIM)</b>.</dd>
<dt>Mask-guided</dt><dd>A coarse binary mask + image (e.g. MGMatting). Cheaper than a trimap; ZIM's label converter builds on this idea.</dd>
<dt>Trimap-free (auto)</dt><dd>Image only. Portrait/human specialists (MODNet) for video calls; salient-object matting for general scenes.</dd>
<dt>Promptable / zero-shot</dt><dd>Point/box/text prompts, no trimap: <b>ZIM</b>. Interactive matting inherits SAM's UX.</dd>
</dl>

Product reality: trimaps exist only in pro tools. Consumer editing and APIs need **trimap-free or prompt-based** matting — which is exactly where ZIM and the foreground API sit.

## 3 · Metrics

- $\text{SAD} = \sum_i |\alpha_i - \hat\alpha_i|$ — sum of absolute differences (often reported /1000).
- $\text{MSE} = \frac{1}{N}\sum_i (\alpha_i - \hat\alpha_i)^2$.
- **Grad** — difference of *spatial gradients* of predicted vs GT alpha; sensitive to edge sharpness/over-smoothing.
- **Conn** — connectivity-based structural error (Rhemann et al.).

Grad and Conn are what separate "looks segmented" from "looks matted." ZIM's loss deliberately includes a gradient term for this reason (see §6).

## 4 · Why SAM alone doesn't do matting

The ZIM argument, which is a great interview narrative:

```mermaid
flowchart TB
  subgraph problem [The trap]
    Pub["Public matting data:\nmostly MACRO (whole subject)"] --> FT
    SA1B["SA-1B masks:\nMICRO granularity but COARSE edges"] --> FT
    FT["Fine-tune SAM on public mattes\n(Matte-Anything / MAM style)"] --> Collapse["Micro prompts → macro output\nzero-shot collapses"]
  end
  subgraph fix [ZIM's fix]
    LC["Label Converter:\nseg masks → fine mattes"] --> SA1BM["SA1B-Matte\n(micro + fine)"]
    SA1BM --> Train[Train ZIM]
  end
```

1. SAM's **pixel decoder** is a shallow stride-4 upsampler (two transposed convs) → checkerboard artifacts, no fine structure.
2. SAM was trained toward **hard-ish masks** on coarse SA-1B labels.
3. Fine-tuning SAM on the small pool of *public* matting datasets (mostly whole-object "macro") makes it **overfit to macro** — it loses SAM's micro/part-level promptability. Zero-shot breaks.

The fix is **data granularity**, not just a bigger decoder: build micro-level *and* fine-boundary mattes at scale.

## 5 · ZIM — the two-axis contribution

> [!EXAMPLE] ZIM = Data + Architecture
> **Data:** a *label converter* (MGMatting+Hiera, trained with L1+Grad) turns SA-1B segmentation masks into fine mattes → **SA1B-Matte**. Two tricks keep it honest: **Spatial Generalization Augmentation (SGA)** (random cut-out pairs so the converter generalizes beyond macro) and **Selective Transformation Learning (STL)** (don't hallucinate hair on rigid objects like cars/desks, using non-transformable ADE20K samples). **Architecture:** a **Hierarchical Pixel Decoder** (multi-resolution stride 2/4/8, ~+10ms) + **Prompt-Aware Masked Attention** (box → binary attention mask; point → Gaussian soft mask injected into token-to-image cross-attention).

ZIM keeps SAM's promptable interface but outputs soft $\alpha$, and it *retains* zero-shot micro/part matting because its training data has the right granularity. Full architecture, ablations, and downstream results in the **[ZIM deep-dive](#/resume/zim)**.

## 6 · Losses

$$\mathcal{L} = \mathcal{L}_{\ell_1} + \lambda\,\mathcal{L}_{\text{grad}}, \qquad \mathcal{L}_{\text{grad}} = \|\nabla_x \hat\alpha - \nabla_x \alpha\|_1 + \|\nabla_y \hat\alpha - \nabla_y \alpha\|_1$$

- $\ell_1$ fixes magnitude; the **gradient term** enforces edge structure (ZIM uses $\lambda = 10$).
- Composition loss ($\|\hat\alpha F + (1-\hat\alpha)B - I\|$) ties $\alpha$ back to appearance.
- Laplacian/pyramid losses for multi-scale detail; perceptual (LPIPS) or adversarial terms can sharpen but add instability/pipeline cost.
- Segmentation **Dice** is a poor fit for soft targets — it assumes near-binary masks.

## 7 · The 2025–2026 landscape

- **BiRefNet** — high-resolution *dichotomous image segmentation* with bilateral reference; strong at fine structure and popular as a background-removal/matting-adjacent backbone (dynamic-resolution variants up to ~2K).
- **Matting Anything (MAM)** — SAM-guided universal matting: SAM mask as guidance to an alpha head. The archetype ZIM critiques for zero-shot fragility.
- **ZIM** — promptable zero-shot *matting* foundation (ICCV 2025 Highlight); Grounded-ZIM = Grounding DINO text→box→ZIM for text-driven matting.
- **SAM 3** provides sharper promptable masks/tracking that upstream a matting stage; matte quality still needs a dedicated soft-alpha head. See [Vision Foundation Models](#/cv/foundation-models).
- **Diffusion editing coupling** — precise $\alpha$ is a strong conditioning signal for inpainting / generative fill / video object editing (the [2026 landscape](#/start/landscape-2026) editing wave: FLUX Kontext, Nano-Banana). A clean matte in, fewer artifacts out.

> [!NOTE] Video & temporal consistency
> Video matting adds a **temporal coherence** requirement: per-frame mattes flicker without it. Approaches range from recurrent state (RVM) to memory-propagation (SAM 2-style) + a matting head. Product-critical for green-screen-free video calls and video editing; the metric adds temporal-stability terms on top of per-frame SAD/Grad.

<figure>
<svg viewBox="0 0 640 130" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <text x="90" y="20" text-anchor="middle" fill="#12a150">soft α (matting)</text>
  <rect x="30" y="30" width="120" height="70" rx="6" fill="none" stroke="#12a150" stroke-width="2"/>
  <path d="M40 100 q40 -70 100 -55" stroke="#12a150" stroke-width="2" fill="none"/>
  <text x="90" y="118" text-anchor="middle" fill="#6b7686">clean composite, no halo</text>
  <text x="330" y="20" text-anchor="middle" fill="#e0533f">hard mask (segmentation)</text>
  <rect x="270" y="30" width="120" height="70" rx="6" fill="none" stroke="#e0533f" stroke-width="2"/>
  <path d="M280 100 L280 55 L390 45" stroke="#e0533f" stroke-width="2" fill="none"/>
  <text x="330" y="118" text-anchor="middle" fill="#6b7686">jagged edge, halo/spill</text>
  <text x="520" y="55" fill="#6b7686">Same boundary,</text>
  <text x="520" y="72" fill="#6b7686">different downstream</text>
  <text x="520" y="89" fill="#6b7686">artifact budget.</text>
</svg>
<figcaption>The composition equation forces a continuous α at semi-transparent edges; a thresholded mask cannot represent partial coverage, so it leaves visible artifacts when composited.</figcaption>
</figure>

## 8 · Human matting & label efficiency (WSSHM)

People are the highest-value, hardest case: hair, fingers, semi-transparent clothing, plus huge product demand. **WSSHM** (candidate, arXiv 2024) is a weakly-**semi**-supervised, trimap-free human-matting baseline — few full mattes + many weak labels — porting the label-efficiency DNA of PointWSSIS into matting. The **foreground-segmentation API** (beating Photoroom / Remove.bg / Adobe in internal eval) is the productized sibling. See [Weak & Semi-Supervised](#/cv/weak-semi-supervised).

## 9 · Q&A

<details class="qa"><summary>Macro vs micro matting — why does it matter for a foundation model?</summary>
<div class="qa-body">

**Short:** macro = whole subject; micro = part/instance (a hand, a bag, hair). Interactive matting needs both, and traditional benchmarks only test macro.

**Deep:** classic AIM/P3M test sets are salient whole-objects, so a micro-capable model can look *worse* there with a single box prompt (it may segment a part) yet be strictly more useful. ZIM reports needing multi-point prompts to recover macro on those sets — an honest domain-shift caveat, not a defect. "Micro-level matte foundation" was the open gap ZIM filled.
</div></details>

<details class="qa"><summary>How do you turn cheap segmentation labels into matting supervision without lying?</summary>
<div class="qa-body">

**Short:** a label converter + two safeguards (SGA, STL).

**Deep:** the converter is trained on the little real matte data you have, so it over-specializes to macro and can *invent* soft edges. SGA (random cut-out pairs) forces it to handle arbitrary regions; STL withholds soft-edge learning on rigid objects that shouldn't have hairy boundaries. Ablations show SGA+STL is what makes SA1B-Matte trustworthy.
</div></details>

<details class="qa"><summary>Where does matte quality show up downstream?</summary>
<div class="qa-body">

**Short:** compositing artifacts, and error amplification in editing / 3D lifting.

**Deep:** a bad matte leaves halos and color spill; feed that into inpainting or a NeRF/3D-lift and errors compound. ZIM's downstream experiments (inpainting, SA3D NeRF, HQ-SAM replacement) show research alpha-metric gains translate to product-level artifact reduction — the "research metric → product metric" bridge interviewers love.
</div></details>

### Follow-ups
- *"Transparent objects (glass, smoke)?"* Different data/guidance regime; ZIM is tuned for open-world fine masks and is weak here even after fine-tuning on Transparent-460 — being candid about this is the Highlight-paper norm.
- *"Prompts: point vs box?"* Point → Gaussian soft attention (ambiguity handled by multi-mask); box → hard attention mask; text → Grounding DINO → box → matte.
- *"Serving?"* Server ViT-B matting is ~100s of ms on a V100-class GPU; products use a separate lightweight model (on-device human seg ~10ms). Role separation is the point — see [On-Device Seg](#/resume/on-device-segmentation).

## Cheat-sheet

| Term | Meaning |
| --- | --- |
| Composition eq. | $I = \alpha F + (1-\alpha)B$ |
| Trimap | FG / BG / Unknown regions |
| Trimap-free | image/prompt only, no trimap |
| SAD / MSE | magnitude error |
| Grad / Conn | boundary-structure error |
| SA1B-Matte | ZIM's converted micro+fine mattes |
| Macro vs micro | whole subject vs part/instance |
| Halo / spill | artifacts from a hard/soft-wrong matte |

**Related:** [Segmentation](#/cv/segmentation) · [Vision Foundation Models](#/cv/foundation-models) · [Weak & Semi-Supervised](#/cv/weak-semi-supervised) · [ZIM deep-dive](#/resume/zim) · [The 2026 Landscape](#/start/landscape-2026)
