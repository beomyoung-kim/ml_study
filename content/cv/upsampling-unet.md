# Upsampling & Encoder–Decoder Networks (U-Net)

> [!NOTE] Goal of this chapter
> Tasks such as segmentation and matting that produce an answer **for every pixel**, known as dense prediction, face a special problem. A CNN repeatedly **reduces** resolution to compress information, but the final output must be a pixel map at the original resolution. We therefore need techniques for **upsampling** the reduced representation and an architecture that organizes them: the **encoder–decoder**, exemplified by **U-Net**. The “pixel decoder,” “stride-4 upsampler,” and “checkerboard artifact” used throughout the segmentation and matting chapters all begin here.

## Why restore the resolution?

Image classification returns a **single answer** such as “cat,” so it can compress the features into a small vector and stop. Segmentation, however, must assign a label to **every pixel**, which requires an output of the same $H\times W$ size as the input.

The problem is that a CNN backbone reduces resolution to 1/8, 1/16, or 1/32 through pooling and strided operations. To produce a pixel map, we must **enlarge** that small feature map back to the original size. This enlargement is upsampling.

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="90" y="20" text-anchor="middle" fill="#0ea5e9">input (high resolution)</text>
  <rect x="40" y="30" width="100" height="100" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.8"/>
  <path d="M150 80 H210" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="180" y="72" text-anchor="middle" fill="#98a3b2" font-size="11">↓ compress</text>
  <text x="290" y="60" text-anchor="middle" fill="#6366f1">small features</text>
  <rect x="255" y="70" width="40" height="40" rx="4" fill="#6366f1"/>
  <path d="M310 80 H370" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="340" y="72" text-anchor="middle" fill="#98a3b2" font-size="11">↑ upsample</text>
  <text x="470" y="20" text-anchor="middle" fill="#12a150">output (pixel map)</text>
  <rect x="420" y="30" width="100" height="100" rx="4" fill="none" stroke="#12a150" stroke-width="1.8"/>
  <text x="560" y="84" fill="#98a3b2" font-size="11">= same size</text>
  <text x="560" y="100" fill="#98a3b2" font-size="11">  as input</text>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>The basic dense-prediction pattern: reduce resolution with an encoder, then enlarge it with a decoder. The central challenge is how to recover fine positional information that was lost while shrinking the representation.</figcaption>
</figure>

## Three upsampling methods

<dl class="kv">
<dt>Nearest-neighbor</dt><dd>The simplest method: replicate each pixel into a $2\times2$ block. It is fast but produces stair-step edges.</dd>
<dt>Bilinear interpolation</dt><dd>Fill values smoothly with a <b>weighted average</b> of neighboring pixels. It has no learned parameters and is common in resize+convolution decoders. Mismatched framework <code>align_corners</code> settings or coordinate conventions change pixel alignment.</dd>
<dt>Transposed convolution</dt><dd>A learned operation that applies the <b>transpose</b> of the matrix representing a convolution. It is often called “deconvolution,” but it is not the inverse of the original convolution. Uneven overlap between kernel and stride can produce <b>checkerboard artifacts</b>, so compare it with resize+convolution.</dd>
</dl>

<figure>
<svg viewBox="0 0 640 170" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <!-- source 2x2 -->
  <text x="70" y="20" text-anchor="middle" fill="#6366f1">2×2 (source)</text>
  <g stroke="#6366f1" stroke-width="1.5" fill="none">
    <rect x="40" y="30" width="30" height="30"/><rect x="70" y="30" width="30" height="30"/>
    <rect x="40" y="60" width="30" height="30"/><rect x="70" y="60" width="30" height="30"/>
  </g>
  <text x="55" y="50" fill="currentColor">1</text><text x="85" y="50" fill="currentColor">2</text>
  <text x="55" y="80" fill="currentColor">3</text><text x="85" y="80" fill="currentColor">4</text>
  <path d="M115 60 H175" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#a2)"/>
  <text x="145" y="52" text-anchor="middle" fill="#98a3b2">nearest ×2</text>
  <!-- nearest 4x4 -->
  <g stroke="#12a150" stroke-width="1.2" fill="none">
    <rect x="200" y="20" width="20" height="20"/><rect x="220" y="20" width="20" height="20"/><rect x="240" y="20" width="20" height="20"/><rect x="260" y="20" width="20" height="20"/>
    <rect x="200" y="40" width="20" height="20"/><rect x="220" y="40" width="20" height="20"/><rect x="240" y="40" width="20" height="20"/><rect x="260" y="40" width="20" height="20"/>
    <rect x="200" y="60" width="20" height="20"/><rect x="220" y="60" width="20" height="20"/><rect x="240" y="60" width="20" height="20"/><rect x="260" y="60" width="20" height="20"/>
    <rect x="200" y="80" width="20" height="20"/><rect x="220" y="80" width="20" height="20"/><rect x="240" y="80" width="20" height="20"/><rect x="260" y="80" width="20" height="20"/>
  </g>
  <text x="240" y="120" text-anchor="middle" fill="#12a150" font-size="10">replicate each pixel into 2×2 (blocky)</text>
  <!-- bilinear note -->
  <rect x="360" y="30" width="240" height="70" rx="8" fill="none" stroke="#e0533f" stroke-width="1.4"/>
  <text x="480" y="55" text-anchor="middle" fill="#e0533f">bilinear = weighted average of neighbors</text>
  <text x="480" y="78" text-anchor="middle" fill="#98a3b2" font-size="10">→ a smooth gradient without stair steps</text>
  <defs><marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>At an integer scale, nearest-neighbor interpolation replicates pixels, while bilinear interpolation blends nearby values according to a defined coordinate convention. A transposed convolution is learnable, but it is not an inverse convolution and may create uneven overlap.</figcaption>
</figure>

## Encoder–decoder networks and skip connections

Simply shrinking and enlarging a representation creates a problem: the compression process discards fine position and boundary information, so upsampling alone tends to produce blurry boundaries. The solution is a **skip connection**.

The key idea is to pass **early, high-resolution features** from the encoder directly to the corresponding decoder stage, then concatenate or add them. This lets the decoder use deep semantic context together with shallow spatial detail, but it cannot mathematically recover every piece of information lost during downsampling. When cropping or padding makes feature sizes differ, define an explicit alignment rule.

> **PyTorch-style pseudocode — merge each skip at the same spatial scale**

```python
skips = []
x = image                                      # [B,C,H,W]
for enc in encoder_blocks:
    x = enc(x); skips.append(x)
    x = downsample(x)

for dec, skip in zip(decoder_blocks, reversed(skips)):
    x = interpolate(x, size=skip.shape[-2:], mode="bilinear")
    x = torch.cat([x, skip], dim=1)            # H,W match; concatenate on channels
    x = dec(x)
logits = output_head(x)                        # [B,num_classes,H,W]
```

<figure>
<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <!-- encoder (left, going down) -->
  <rect x="40"  y="30"  width="90" height="26" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="85" y="47" text-anchor="middle" fill="currentColor">enc 1 (H)</text>
  <rect x="60"  y="80"  width="70" height="26" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="95" y="97" text-anchor="middle" fill="currentColor">enc 2 (H/2)</text>
  <rect x="80"  y="130" width="50" height="26" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="105" y="147" text-anchor="middle" fill="currentColor" font-size="10">enc 3</text>
  <!-- bottleneck -->
  <rect x="270" y="180" width="100" height="30" rx="6" fill="#6366f1"/><text x="320" y="200" text-anchor="middle" fill="#fff">bottleneck</text>
  <!-- decoder (right, going up) -->
  <rect x="510" y="130" width="50" height="26" rx="4" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="535" y="147" text-anchor="middle" fill="currentColor" font-size="10">dec 3</text>
  <rect x="510" y="80"  width="70" height="26" rx="4" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="545" y="97" text-anchor="middle" fill="currentColor">dec 2 (H/2)</text>
  <rect x="510" y="30"  width="90" height="26" rx="4" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="555" y="47" text-anchor="middle" fill="currentColor">dec 1 (H)</text>
  <!-- down path -->
  <path d="M95 56 L110 80 M115 106 L120 130 M130 156 L280 185" stroke="#0ea5e9" stroke-width="1.4" fill="none" marker-end="url(#a3)"/>
  <!-- up path -->
  <path d="M370 185 L520 156 M545 130 L555 106 M545 80 L550 56" stroke="#12a150" stroke-width="1.4" fill="none" marker-end="url(#a3)"/>
  <!-- skip connections (dashed) -->
  <path d="M130 43 L510 43" stroke="#e0533f" stroke-width="1.6" stroke-dasharray="5 4"/>
  <path d="M130 93 L510 93" stroke="#e0533f" stroke-width="1.6" stroke-dasharray="5 4"/>
  <path d="M130 143 L510 143" stroke="#e0533f" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text x="320" y="35" text-anchor="middle" fill="#e0533f">skip connection (direct boundary/position details)</text>
  <text x="90" y="225" fill="#0ea5e9">Encoder: shrink to learn “what”</text>
  <text x="410" y="225" fill="#12a150">Decoder: enlarge to recover “where”</text>
  <defs><marker id="a3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>The U shape of U-Net: the encoder on the left compresses the representation as it descends; the decoder on the right builds a dense output as it ascends; and <b>skip connections</b> (red dashed lines) transfer features between matching scales. This design appears in medical imaging, segmentation, and many diffusion U-Nets, although Transformer architectures such as DiT are also common diffusion backbones.</figcaption>
</figure>

> [!TIP] Interview one-liner
> “The heart of U-Net is the **same-scale skip connection**: it combines the *semantics* of deep features with the *spatial detail* of shallow features.” Also distinguish the architecture from the learning process: both U-Net and DiT are possible diffusion backbones, while diffusion or flow defines a generative objective and dynamics rather than an architecture. See [Architectures](#/foundations/architectures).

## Implement it — 2× nearest-neighbor upsampling

Implement the simplest upsampling operation. Replicate each pixel into a `scale × scale` block. Hint: apply `np.repeat` along each axis.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"upsample_nearest","packages":["numpy"],"starter":"def upsample_nearest(x, scale=2):\n    # x: a 2D list. Enlarge it by replicating each value into a scale×scale block,\n    # and return the enlarged 2D list.\n    # Example: [[1,2],[3,4]], scale=2 -> [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]\n    pass","tests":[{"args":[[[1,2],[3,4]],2],"expect":[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]},{"args":[[[5]],3],"expect":[[5,5,5],[5,5,5],[5,5,5]]},{"args":[[[1,2,3]],2],"expect":[[1,1,2,2,3,3],[1,1,2,2,3,3]]},{"args":[[[1,2],[3,4]],1],"expect":[[1,2],[3,4]]}],"solution":"import numpy as np\n\ndef upsample_nearest(x, scale=2):\n    a = np.asarray(x)\n    if a.ndim != 2 or 0 in a.shape:\n        raise ValueError(\"x must be a non-empty 2D array\")\n    if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or scale <= 0:\n        raise ValueError(\"scale must be a positive integer\")\n    a = np.repeat(a, int(scale), axis=0)\n    a = np.repeat(a, int(scale), axis=1)\n    return a.tolist()"}
</script>
</div>

Bilinear interpolation extends this operation by replacing replication with a distance-weighted average of neighboring values. A transposed convolution replaces the fixed enlargement rule with a learnable kernel.

## Q&A

<details class="qa"><summary>Why does transposed convolution create checkerboard artifacts, and how can we avoid them?</summary>
<div class="qa-body">

**Short answer:** When kernel size is not divisible by stride, uneven overlap gives different output positions different numbers of contributions, making grid-like artifacts more likely.

**In depth:** With kernel 3 and stride 2, for example, the overlap count can vary by output position. Mitigations include (1) resize followed by a normal convolution, (2) choosing a kernel that is a multiple of stride and checking initialization and training, and (3) inspecting artifacts after PixelShuffle. Kernel divisibility alone does not guarantee artifact-free output, and bilinear resize has its own blur and alignment trade-offs, so combine validation metrics with visual inspection.
</div></details>

<details class="qa"><summary>What gets worse without skip connections?</summary>
<div class="qa-body">

**Short answer:** Boundaries become blurry and small objects are missed.

**In depth:** Low-resolution bottleneck features contain limited fine positional information. Skip connections pass high-resolution encoder features to the decoder so it can use boundary cues. They can also pass shallow-feature noise and domain bias, however, and final quality still depends on the loss, resolution, and annotation quality.
</div></details>

## Cheat sheet

| Concept | In one line |
| --- | --- |
| Why upsample? | A dense-prediction output must match the input size |
| Nearest | Pixel replication—fast but blocky |
| Bilinear | Weighted average of neighbors—smooth and parameter-free (common default) |
| Transposed convolution | Learned transpose of a convolution matrix; not its inverse, and uneven overlap matters |
| Encoder–decoder | Shrink to learn “what,” then enlarge to recover “where” |
| Skip connection | Send high-resolution boundary information directly to the decoder—the heart of U-Net |
| Avoid checkerboards | Bilinear upsampling + regular convolution |

**Next:** [Vision Data Augmentation](#/cv/augmentation) · [Segmentation](#/cv/segmentation) · [Convolution & Pooling from Scratch](#/ml-coding/conv-pooling)
