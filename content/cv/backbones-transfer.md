# Backbones & Transfer Learning (ResNet → ViT)

> [!NOTE] Goal of this chapter
> In many vision projects with limited labels or compute, it is better to start with a pretrained **backbone (feature extractor)** and attach a task-specific head than to train from scratch. Scratch training remains worth comparing when the dataset is very large or the domain and input channels differ substantially. The idea of **reusing previously learned features** is the foundation that connects detection, segmentation, foundation models, and continual learning.

## What is a backbone?

As we saw in [CNNs for Vision](#/cv/cnns-for-vision), the early and middle layers of a neural network extract a hierarchy of image representations. We call this **feature-extraction component** the backbone. The same backbone weights can transfer to many tasks, but their features are not completely task-agnostic. Transfer quality depends on the pretraining data and objective as well as the spatial resolution the downstream task needs. The **head** or decoder on top must also be designed for the task.

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="20" y="80" width="70" height="46" rx="6" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="55" y="107" text-anchor="middle" fill="currentColor">image</text>
  <rect x="120" y="70" width="200" height="66" rx="8" fill="#6366f1"/>
  <text x="220" y="98" text-anchor="middle" fill="#fff" font-weight="700">Backbone (shared)</text>
  <text x="220" y="118" text-anchor="middle" fill="#fff" font-size="11">pretrained feature extractor</text>
  <path d="M90 103 H120" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#ar)"/>
  <path d="M320 103 H370" stroke="#98a3b2" stroke-width="1.5"/>
  <rect x="380" y="20" width="150" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="455" y="42" text-anchor="middle" fill="#12a150">classification head → cat</text>
  <rect x="380" y="86" width="150" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="455" y="108" text-anchor="middle" fill="#12a150">detection head → boxes</text>
  <rect x="380" y="152" width="150" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="455" y="174" text-anchor="middle" fill="#12a150">segmentation head → mask</text>
  <path d="M370 103 C 375 40, 375 40, 380 37" stroke="#98a3b2" stroke-width="1.3" fill="none"/>
  <path d="M370 103 H380" stroke="#98a3b2" stroke-width="1.3"/>
  <path d="M370 103 C 375 166, 375 166, 380 169" stroke="#98a3b2" stroke-width="1.3" fill="none"/>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>One set of backbone weights or features can be shared across several task heads. When multiple heads process the same input together, the backbone features need to be computed only once; separate inputs or services still require a backbone forward pass for every request.</figcaption>
</figure>

## ResNet — making very deep networks trainable

When plain networks were made deeper, they sometimes failed to achieve even a lower training error—the **degradation problem**. This is an optimization problem, not merely overfitting. ResNet's residual connection, $y=\mathcal{F}(x)+x$, makes the identity map easy to represent and adds an identity term to the gradient, helping optimize very deep networks. It does not guarantee that gradients flow “without loss”; normalization, initialization, and the optimizer still matter. See the canonical treatment in [CNNs, RNNs, and Transformers](#/foundations/architectures) for the detailed derivation.

## ViT — split an image into tokens for a Transformer

The **Vision Transformer (ViT)** uses a [Transformer](#/foundations/architectures) instead of a CNN. It divides an image into **patches** (for example, 16×16), turns each patch into a token, adds positional information, and lets the tokens attend to one another through [self-attention](#/ml-coding/attention), just like words in a sentence. For an illustration and details of patch→token tokenization, see the canonical treatment in [CNNs, RNNs, and Transformers](#/foundations/architectures). The same “image→tokens” idea carries directly into [vision-language models](#/vlm/vlm-101).

<div class="proscons"><div><div class="pros-t">CNN (for example, ResNet)</div>

- Built-in **inductive biases**: locality, weight sharing, and translation equivariance → can work well with **less data**
- Strong on small/medium datasets and on-device workloads

</div><div><div class="cons-t">ViT</div>

- A basic ViT has a weaker locality bias than a CNN, so training from scratch may require more data and regularization; it performs strongly after large-scale pretraining
- Models global relationships flexibly and extends naturally to multimodal systems

</div></div>

> [!TIP] Interview one-liner
> “A CNN's locality bias makes it a strong starting point for small datasets and edge budgets, while a ViT excels at large-scale pretraining and global mixing. Measure the winner for your data, resolution, latency target, and checkpoint.” Swin is a windowed hierarchical Transformer, ConvNeXt is a modernized CNN, and Hiera is a hierarchical Transformer; do not flatten all three into the same vague “hybrid” category.

## Transfer learning — reuse what has already been learned

**Transfer learning** takes a backbone **pretrained** on a large dataset—ImageNet, for example, or via [self-supervised learning](#/cv/self-supervised)—and adapts it to your smaller downstream dataset and task. There are two main approaches:

<dl class="kv">
<dt>Feature extraction / freezing</dt><dd><b>Freeze</b> the backbone and train only the newly attached head. This is fast and reduces overfitting risk when data is <b>scarce</b>.</dd>
<dt>Fine-tuning</dt><dd>Train the backbone <b>as well</b>, or unfreeze only its later stages. This often improves performance when there is <b>enough data</b> and the domain differs from pretraining. Use a <b>smaller learning rate</b> in most cases.</dd>
</dl>

> [!WARNING] Freezing has two layers
> <code>requires_grad_(False)</code> prevents parameter updates, but it does not stop BatchNorm running statistics or Dropout behavior while <code>model.train()</code> is active. For a fixed feature extractor, decide explicitly whether the backbone should stay in <code>eval()</code> while only the head is in train mode. Adapting BN statistics to the new domain can also be a valid strategy, but it should be intentional.

> **PyTorch-style pseudocode — a fully fixed feature extractor**

```python
backbone.requires_grad_(False)
backbone.eval()                             # also freeze BN running stats and Dropout behavior
head.train()

for images, labels in train_loader:
    with torch.no_grad():                   # do not retain a backbone computation graph
        features = backbone(images)         # [B,C,H,W] or [B,T,D]
    logits = head(pool(features))           # gradients still flow through the head
    loss = criterion(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward(); optimizer.step()       # optimizer contains only head parameters
```

<figure>
<svg viewBox="0 0 620 130" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="90" y="22" text-anchor="middle" fill="#98a3b2">① Pretrain</text>
  <rect x="30" y="35" width="120" height="40" rx="6" fill="#6366f1"/><text x="90" y="60" text-anchor="middle" fill="#fff" font-size="11">backbone (large-scale)</text>
  <path d="M155 55 H195" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#a5)"/>
  <text x="300" y="22" text-anchor="middle" fill="#98a3b2">② Transfer (freeze + new head)</text>
  <rect x="205" y="35" width="120" height="40" rx="6" fill="#6366f1" opacity="0.55"/><text x="265" y="55" text-anchor="middle" fill="#fff" font-size="10">backbone (frozen ❄)</text>
  <text x="265" y="68" text-anchor="middle" fill="#fff" font-size="9">weights fixed</text>
  <rect x="335" y="35" width="90" height="40" rx="6" fill="#12a150"/><text x="380" y="55" text-anchor="middle" fill="#fff" font-size="10">new head</text>
  <text x="380" y="68" text-anchor="middle" fill="#fff" font-size="9">train only this</text>
  <path d="M425 55 H455" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#a5)"/>
  <text x="540" y="59" text-anchor="middle" fill="currentColor">my task's output</text>
  <text x="310" y="110" text-anchor="middle" fill="#98a3b2">Little data → freeze · more data and a different domain → fine-tune (small LR)</text>
  <defs><marker id="a5" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>Transfer learning: reuse the expensive backbone—either frozen or gently adapted—and train a lightweight head on your data, saving substantial data, time, and computation.</figcaption>
</figure>

## Try it yourself — turn backbone features into a vector with global average pooling

To feed a backbone's final feature map `(C, H, W)` into a head, we commonly average over the spatial dimensions and summarize it as one value per channel, producing a `(C,)` vector. The head then classifies this vector.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"global_avg_pool","packages":["numpy"],"approx":true,"starter":"def global_avg_pool(fmap):\n    # fmap: a (C, H, W) feature map. Average over H×W independently for each channel C\n    # and return the resulting (C,) vector as a list.\n    pass","tests":[{"args":[[[[1,3],[3,1]]]],"expect":[2.0]},{"args":[[[[1,1],[1,1]],[[2,4],[4,2]]]],"expect":[1.0,3.0]},{"args":[[[[0,0,0],[6,6,6]]]],"expect":[3.0]}],"solution":"import numpy as np\n\ndef global_avg_pool(fmap):\n    x = np.asarray(fmap, dtype=float)\n    if x.ndim != 3 or any(size == 0 for size in x.shape):\n        raise ValueError(\"fmap must be a non-empty (C,H,W) array\")\n    if not np.all(np.isfinite(x)):\n        raise ValueError(\"fmap must contain finite values\")\n    return x.mean(axis=(1, 2)).tolist()"}
</script>
</div>

If the backbone is frozen and in `eval()` mode, and the input preprocessing is deterministic, you can **precompute and cache** features through this `global_avg_pool` step. You cannot safely reuse the same cache when random augmentation changes every epoch, the backbone's BN/Dropout state changes, or the task requires spatial features. Compare storage and I/O costs against the computation saved.

## Q&A

<details class="qa"><summary>Why freeze with very little data, but fine-tune with more?</summary>
<div class="qa-body">

**Short answer:** With little data, training the backbone too creates a greater risk of overfitting and forgetting useful pretrained features.

**In depth:** A backbone may have millions or hundreds of millions of parameters. If you update all of them from only a few hundred labeled examples, the model can overfit that small dataset and destroy the useful general features learned during pretraining—a cousin of catastrophic forgetting. With little data, freeze the backbone and train the head; with more data and a different domain, such as medical or satellite imagery, fine-tune using a smaller learning rate. Unfreezing **only the later stages** is a common middle ground.
</div></details>

<details class="qa"><summary>ImageNet pretraining or self-supervised pretraining—which is better?</summary>
<div class="qa-body">

**Short answer:** Treat both as candidates. A self-supervised backbone can exploit large unlabeled datasets and transfer well to dense tasks, but compare downstream transfer scores using the same architecture, data, compute, and protocol.

**In depth:** Supervised ImageNet pretraining is a strong and reproducible baseline, but its objective is tied to a 1,000-class taxonomy. Self-supervised learning can learn representations from larger datasets without human class labels, yet data curation and compute costs remain, and it is not superior on every downstream task. Compare linear probing, full or partial fine-tuning, and frozen dense features under a shared protocol. Continue to [Self-Supervised Learning](#/cv/self-supervised) and [Vision Foundation Models](#/cv/foundation-models).
</div></details>

## Cheat sheet

| Concept | In one line |
| --- | --- |
| Backbone | A pretrained feature extractor shared across tasks |
| Head | A lightweight task-specific predictor placed on the backbone |
| ResNet | Residual connections ($y=F(x)+x$) make identity maps easy and help optimize deep networks |
| ViT | Image→patches→tokens→Transformer; evaluate together with pretraining, resolution, and latency |
| CNN vs ViT | Inductive bias (less data) vs flexibility (more data) |
| Freeze | Keep the backbone fixed and train only the head; useful with little data |
| Fine-tune | Train the backbone too (small LR); useful with more data and a different domain |

**Next:** [Upsampling & U-Net](#/cv/upsampling-unet) · [Self-Supervised Learning](#/cv/self-supervised) · [CNNs, RNNs, and Transformers](#/foundations/architectures)
