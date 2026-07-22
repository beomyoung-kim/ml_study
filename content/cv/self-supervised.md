# Introduction to Self-Supervised Learning

> [!NOTE] Goal of this chapter
> This chapter develops the big picture of **self-supervised learning (SSL)**, which learns features without human-provided task labels. The central idea is to create learning targets from the structure of the data itself. DINO and MAE are representative vision SSL methods. CLIP, by contrast, uses web image–text pairs as supervision and is better described as **natural-language-supervised or weakly supervised multimodal pretraining** than strictly label-free SSL.

## What it is and why it matters

[Image classification](#/cv/classification) and [detection](#/cv/detection) learn from human-provided **labels**. Dense annotations are particularly expensive. Unlabeled data is far more abundant, but it is neither free nor unlimited: collection, deduplication, quality, copyright, privacy, harmful content, storage, and compute all affect training quality and deployment viability.

Self-supervised learning uses this **unlabeled data**. Instead of human labels, it automatically constructs a problem from the **data itself**, known as a pretext task. One example is to hide part of an image and reconstruct the original. Solving this task well requires the model to learn useful representations such as what a cat looks like. Those features can then **transfer** to classification, detection, and segmentation with relatively few labels; see [Transfer Learning](#/cv/backbones-transfer).

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="30" y="45" width="120" height="60" rx="8" fill="none" stroke="#0ea5e9" stroke-width="1.6"/>
  <text x="90" y="35" text-anchor="middle" fill="#0ea5e9">unlabeled data</text>
  <text x="90" y="80" text-anchor="middle" fill="currentColor">🖼️ 🖼️ 🖼️ …</text>
  <path d="M150 75 H210" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#s1)"/>
  <rect x="215" y="45" width="150" height="60" rx="8" fill="#6366f1"/>
  <text x="290" y="70" text-anchor="middle" fill="#fff" font-size="11">create a pretext task</text>
  <text x="290" y="90" text-anchor="middle" fill="#fff" font-size="11">(masking/view matching …)</text>
  <path d="M365 75 H425" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#s1)"/>
  <rect x="430" y="45" width="120" height="60" rx="8" fill="none" stroke="#12a150" stroke-width="1.8"/>
  <text x="490" y="70" text-anchor="middle" fill="#12a150">useful features</text>
  <text x="490" y="90" text-anchor="middle" fill="#98a3b2" font-size="11">→ transfer with few labels</text>
  <defs><marker id="s1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>Self-supervised learning creates problems from the data and solves them without human labels, learning features that can transfer to downstream tasks.</figcaption>
</figure>

## Two major approaches

### 1. Contrastive learning

The key intuition is to take two differently augmented views of the **same image** and pull them close together in embedding space, while pushing views of **different images** apart. The model learns what is essential to the image and what is an incidental transformation such as a crop or color change.

<figure>
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <circle cx="320" cy="100" r="88" fill="none" stroke="#98a3b2" stroke-width="1" stroke-dasharray="4 4"/>
  <text x="320" y="24" text-anchor="middle" fill="#98a3b2">embedding space</text>
  <circle cx="300" cy="95" r="7" fill="#e0533f"/><text x="300" y="84" text-anchor="middle" fill="#e0533f">view A</text>
  <circle cx="345" cy="110" r="7" fill="#e0533f"/><text x="360" y="114" fill="#e0533f">view B</text>
  <path d="M307 97 L338 108" stroke="#12a150" stroke-width="2" marker-end="url(#p1)"/>
  <text x="322" y="138" text-anchor="middle" fill="#12a150" font-size="11">pull together (positive)</text>
  <circle cx="150" cy="60" r="7" fill="#0ea5e9"/><circle cx="180" cy="160" r="7" fill="#0ea5e9"/><circle cx="480" cy="70" r="7" fill="#0ea5e9"/><circle cx="470" cy="150" r="7" fill="#0ea5e9"/>
  <path d="M295 92 L165 63" stroke="#e0533f" stroke-width="1.3" stroke-dasharray="3 3" marker-end="url(#p2)"/>
  <path d="M348 112 L465 145" stroke="#e0533f" stroke-width="1.3" stroke-dasharray="3 3" marker-end="url(#p2)"/>
  <text x="150" y="185" text-anchor="middle" fill="#0ea5e9" font-size="11">different images = push apart (negative)</text>
  <defs>
    <marker id="p1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
    <marker id="p2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#e0533f"/></marker>
  </defs>
</svg>
<figcaption>Contrastive learning pulls two augmented views of one image (red) together as a positive pair and pushes views of other images (blue) away as negatives. This push and pull produces meaningful embeddings.</figcaption>
</figure>

<dl class="kv">
<dt>SimCLR</dt><dd>Uses other images in a large batch as negatives. The augmentation combination, especially crop plus color transforms, strongly affects performance.</dd>
<dt>MoCo</dt><dd>Uses a momentum encoder and a queue to provide many negatives, enabling contrastive learning with smaller batches.</dd>
<dt>CLIP</dt><dd>Applies a contrastive objective to <b>image↔text</b> pairs, pulling matched pairs together and treating the remaining combinations in the batch as negatives. Because the text pairs are themselves an external supervision signal, CLIP is closer to natural-language supervision or weak supervision than narrowly defined vision-only SSL. See <a href="#/vlm/pretraining">VLM Pretraining</a> and <a href="#/vlm/vlm-101">VLM 101</a>.</dd>
</dl>

The loss is commonly **InfoNCE**, which treats the problem as softmax classification: choose one positive for an anchor from many candidates. Similarity is usually measured with **cosine similarity**, which you will implement below.

> **PyTorch-style pseudocode — the diagonal of the batch contains the positives**

```python
view1, view2 = augment(images), augment(images)     # each [B,C,H,W]
z1 = normalize(projector(encoder(view1)), dim=-1)  # [B,D]
z2 = normalize(projector(encoder(view2)), dim=-1)  # [B,D]

logits = z1 @ z2.T / temperature                    # [B,B]
positive = torch.arange(B, device=logits.device)    # (i,i) shares the same source image
loss = 0.5 * (cross_entropy(logits, positive)
              + cross_entropy(logits.T, positive))
loss.backward()                                     # gradients flow through both view paths
```

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"cosine_sim","packages":["numpy"],"approx":true,"starter":"def cosine_sim(a, b):\n    # Cosine similarity of two vectors = (a·b) / (|a| |b|). Values near 1 mean similar directions.\n    # The shapes must match, and cosine similarity is undefined for a zero vector.\n    pass","tests":[{"args":[[1,0],[1,0]],"expect":1.0},{"args":[[1,0],[0,1]],"expect":0.0},{"args":[[1,2,3],[2,4,6]],"expect":1.0},{"args":[[1,0],[-1,0]],"expect":-1.0}],"solution":"import numpy as np\n\ndef cosine_sim(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:\n        raise ValueError(\"a and b must be non-empty vectors with the same shape\")\n    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):\n        raise ValueError(\"vectors must be finite\")\n    denom = np.linalg.norm(a) * np.linalg.norm(b)\n    if denom == 0:\n        raise ValueError(\"cosine similarity is undefined for a zero vector\")\n    return float(np.clip((a @ b) / denom, -1.0, 1.0))"}
</script>
</div>

> [!WARNING] Collapse and false negatives
> If we minimize only a simple positive-pair agreement loss, mapping every input to the same vector can become a collapsed solution. However, “no negatives” does not mean collapse is inevitable. BYOL and SimSiam use stop-gradient, a predictor, and optimization dynamics; DINO uses a teacher plus centering and sharpening to avoid collapse without contrastive negatives. Conversely, treating every other image in the batch as a negative can push apart images from the same semantic class, creating **false negatives**.

### 2. Masked image modeling

This approach brings fill-in-the-blank learning from NLP to images. Split an image into patches, **mask some of them, and reconstruct the missing content**. A representative method is the **Masked Autoencoder (MAE)**, which learns strong representations while masking as much as 75% of the patches.

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="70" y="20" text-anchor="middle" fill="#98a3b2">some patches masked</text>
  <g stroke="#98a3b2" stroke-width="1">
    <rect x="20" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.5"/><rect x="50" y="30" width="30" height="30" fill="#3a3f4b"/><rect x="80" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.5"/>
    <rect x="20" y="60" width="30" height="30" fill="#3a3f4b"/><rect x="50" y="60" width="30" height="30" fill="#0ea5e9" opacity="0.5"/><rect x="80" y="60" width="30" height="30" fill="#3a3f4b"/>
    <rect x="20" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.5"/><rect x="50" y="90" width="30" height="30" fill="#3a3f4b"/><rect x="80" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.5"/>
  </g>
  <path d="M120 75 H190" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#m1)"/>
  <rect x="195" y="55" width="130" height="40" rx="8" fill="#6366f1"/><text x="260" y="80" text-anchor="middle" fill="#fff" font-size="11">encoder → decoder</text>
  <path d="M325 75 H395" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#m1)"/>
  <text x="470" y="20" text-anchor="middle" fill="#12a150">reconstruction</text>
  <g stroke="#98a3b2" stroke-width="1">
    <rect x="420" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="450" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="480" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.6"/>
    <rect x="420" y="60" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="450" y="60" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="480" y="60" width="30" height="30" fill="#0ea5e9" opacity="0.6"/>
    <rect x="420" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="450" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="480" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.6"/>
  </g>
  <text x="545" y="79" fill="#12a150" font-size="11">≈ original</text>
  <defs><marker id="m1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>MAE masks most patches (gray) and learns to reconstruct them (blue). Reconstruction requires modeling scene structure, which can naturally produce strong features.</figcaption>
</figure>

<div class="proscons"><div><div class="pros-t">Contrastive</div>

- Strong image-level, global representations → useful for classification and retrieval
- Augmentation and negative design strongly affect performance
- Extends naturally to multimodal systems such as CLIP

</div><div><div class="cons-t">Masked (MIM)</div>

- Patch-level targets can benefit dense features
- Less burden from augmentation and negative design
- Reconstruction targets may remain overly low-level

</div></div>

### 3. Self-distillation — one step further

The **DINO** family uses neither negatives nor reconstruction. A **teacher**, updated slowly through an exponential moving average, and a student learn to match their outputs on different views of the same image. Centering and sharpening prevent collapse. Attention maps that respond to object boundaries emerge without labels. [Vision Foundation Models](#/cv/foundation-models) covers the training mechanism in more detail.

## Q&A

<details class="qa"><summary>Are self-supervised learning and unsupervised learning the same thing?</summary>
<div class="qa-body">

**Short answer:** SSL is broadly a form of unsupervised learning, but its defining trait is that it creates targets from the data automatically and solves the resulting task with supervised-learning machinery.

**In depth:** Clustering optimizes within-cluster distance, and PCA optimizes reconstruction or explained variance; both have **explicit objectives** without human-provided target labels. SSL constructs a target or positive relationship from the input itself through masking or view matching, then uses losses and gradients much like supervised learning. Not every SSL target needs to be described as a fixed “pseudo-label.”
</div></details>

<details class="qa"><summary>When should I use contrastive learning versus masked modeling?</summary>
<div class="qa-body">

**Short answer:** Contrastive learning naturally fits global alignment, while masked modeling naturally provides patch-level learning signals. Compare downstream performance under a shared architecture, pretraining dataset, and fine-tuning protocol.

**In depth:** Contrasting globally pooled embeddings aligns well with retrieval and classification, while MIM provides a target at each patch. That does not mean pixel reconstruction automatically guarantees semantic dense features, nor that contrastive methods are inherently weak on dense tasks. Evaluate frozen dense features and full fine-tuning in addition to linear probing; see [Vision Foundation Models](#/cv/foundation-models).
</div></details>

## Cheat sheet

| Concept | In one line |
| --- | --- |
| SSL core | Learn features without labels by constructing problems from the data |
| Contrastive | Pull two views of the same image together and push others apart (SimCLR/MoCo/CLIP) |
| InfoNCE | Softmax over candidates to select the positive; often uses cosine similarity |
| Collapse | Without negatives or another mechanism, everything can map to one vector → must be prevented |
| Masked (MAE) | Mask patches and reconstruct a target; target and decoder design shape the representation |
| Self-distillation | Match teacher (EMA) and student across views (DINO), without labels or negatives |

**Next:** [Object Detection](#/cv/detection) · [Vision Foundation Models](#/cv/foundation-models) · [VLM 101](#/vlm/vlm-101)
