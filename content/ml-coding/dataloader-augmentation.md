# Dataloaders & Augmentation

> [!NOTE] Goal of this chapter
> Training requires data to flow **a little at a time, shuffled, and in batches**. A **DataLoader** does that job; **augmentation** creates diversity by transforming samples. Here you will build both from the ground up and see how a training pipeline actually works. The [NumPy Primer](#/ml-coding/numpy-primer) is enough prerequisite.

## What and why

A neural network does not consume the entire dataset at once. It learns incrementally from **batches**, or small groups of examples; see minibatches in [Optimization](#/foundations/optimization). Two components create and deliver those groups:

- A **Dataset** is a store that returns “sample $i$.” Its entire interface is `__len__`, the number of examples, and `__getitem__(i)`, which returns one.
- A **DataLoader** is an iterator that **shuffles** indices, **slices** them into batches, fetches each sample, and **collates** the list into arrays.

**Data augmentation** creates a new version of a sample on the fly by flipping or cropping an image or changing its color. More varied training examples make rote memorization harder and can improve **generalization**; see [Regularization & Generalization](#/foundations/regularization-generalization).

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- dataset -->
  <text x="70" y="20" text-anchor="middle" fill="#0ea5e9" font-weight="700">Dataset</text>
  <g fill="none" stroke="#0ea5e9" stroke-width="1.6">
    <rect x="30" y="30" width="80" height="26" rx="4"/><rect x="30" y="60" width="80" height="26" rx="4"/>
    <rect x="30" y="90" width="80" height="26" rx="4"/><rect x="30" y="120" width="80" height="26" rx="4"/>
    <rect x="30" y="150" width="80" height="26" rx="4"/>
  </g>
  <text x="70" y="48" text-anchor="middle" fill="currentColor">sample 0</text>
  <text x="70" y="78" text-anchor="middle" fill="currentColor">sample 1</text>
  <text x="70" y="108" text-anchor="middle" fill="currentColor">sample 2</text>
  <text x="70" y="138" text-anchor="middle" fill="currentColor">sample 3</text>
  <text x="70" y="168" text-anchor="middle" fill="currentColor">sample 4</text>
  <!-- shuffle -->
  <path d="M115 95 H175" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#d1)"/>
  <text x="215" y="90" text-anchor="middle" fill="#6366f1" font-weight="700">shuffle</text>
  <text x="215" y="106" text-anchor="middle" fill="#98a3b2">permute indices</text>
  <text x="215" y="122" text-anchor="middle" fill="#98a3b2">[3,0,4,1,2]</text>
  <!-- augment -->
  <path d="M258 95 H318" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#d1)"/>
  <text x="360" y="90" text-anchor="middle" fill="#e0533f" font-weight="700">augment</text>
  <text x="360" y="106" text-anchor="middle" fill="#98a3b2">flip·crop·norm</text>
  <!-- batch -->
  <path d="M402 95 H452" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#d1)"/>
  <rect x="460" y="60" width="150" height="70" rx="8" fill="none" stroke="#12a150" stroke-width="1.8"/>
  <text x="535" y="52" text-anchor="middle" fill="#12a150" font-weight="700">batch</text>
  <g fill="#12a150"><rect x="472" y="75" width="18" height="40" rx="3"/><rect x="496" y="75" width="18" height="40" rx="3"/><rect x="520" y="75" width="18" height="40" rx="3"/></g>
  <text x="535" y="150" text-anchor="middle" fill="#98a3b2">→ model</text>
  <defs><marker id="d1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>A training input pipeline: shuffle indices from the Dataset, apply augmentation, collate samples into a batch, and feed it to the model. The DataLoader repeats this process every epoch.</figcaption>
</figure>

> [!TIP] Interview one-liner
> “A `DataLoader` is an iterator. It shuffles indices once per epoch, slices them into batches, fetches each sample by index, and `collate`s the list into stacked arrays. Augmentation is a `Compose` of transforms applied per sample; the subtlety is keeping labels—boxes and masks—synchronized with geometric transforms.” Lead with the **interface**, not the code. The tells that you have built real pipelines are `drop_last`, `collate_fn`, label synchronization, and `num_workers`.

## Dataset + DataLoader from scratch

The following is a minimal NumPy implementation of PyTorch's `Dataset`/`DataLoader`. Real frameworks use the same skeleton.

```python
from __future__ import annotations
import math
from typing import Any, Callable, Iterator
import numpy as np


class Dataset:
    """Map-style protocol: __len__ + __getitem__."""
    def __len__(self) -> int: raise NotImplementedError
    def __getitem__(self, i: int) -> Any: raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(len(t) == len(tensors[0]) for t in tensors)
        self.tensors = tensors
    def __len__(self): return len(self.tensors[0])
    def __getitem__(self, i): return tuple(t[i] for t in self.tensors)


def default_collate(batch: list) -> Any:
    """Stack a list of samples into batched arrays, field by field."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)
    cols = zip(*batch)                                  # transpose list-of-tuples
    return tuple(np.stack(c, 0) if isinstance(c[0], np.ndarray) else list(c)
                 for c in cols)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 drop_last=False, collate_fn=default_collate, seed=0):
        self.ds, self.bs = dataset, batch_size
        self.shuffle, self.drop_last = shuffle, drop_last
        self.collate_fn = collate_fn
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator:
        idx = np.arange(len(self.ds))
        if self.shuffle:
            self.rng.shuffle(idx)                        # new order each epoch
        for start in range(0, len(idx), self.bs):
            batch_idx = idx[start:start + self.bs]
            if len(batch_idx) < self.bs and self.drop_last:
                break                                    # drop ragged tail
            yield self.collate_fn([self.ds[int(i)] for i in batch_idx])

    def __len__(self):
        n = len(self.ds)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)
```

**Key design choices, in plain language:**

- **`shuffle`:** reorder the **indices**, not the data itself. This is cheap and simple.
- **`drop_last`:** discard the partial final batch. It matters when BatchNorm statistics or compilation require a fixed batch size.
- **`collate_fn`:** combine a list of samples into arrays. Customize it for ragged data, such as a variable number of detection boxes.

## Augmentation: the label-sync subtlety

The most common trap: when you flip or crop an image, **apply the corresponding transform to its boxes and masks as well**. Omitting it silently misaligns the target, producing a bug that looks like a mysterious accuracy ceiling.

<figure>
<svg viewBox="0 0 640 190" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- original -->
  <text x="150" y="20" text-anchor="middle" fill="#98a3b2">original</text>
  <rect x="60" y="30" width="180" height="120" rx="6" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <circle cx="110" cy="95" r="26" fill="#0ea5e9" opacity="0.5"/>
  <rect x="82" y="67" width="56" height="56" rx="4" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="150" y="170" text-anchor="middle" fill="#e0533f">box: [x1,y1,x2,y2]</text>
  <!-- arrow -->
  <path d="M255 90 H320" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#f1)"/>
  <text x="288" y="80" text-anchor="middle" fill="#6366f1">flip →</text>
  <!-- flipped: image AND box both flip -->
  <text x="480" y="20" text-anchor="middle" fill="#12a150">horizontal flip (synced ✓)</text>
  <rect x="360" y="30" width="180" height="120" rx="6" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <circle cx="490" cy="95" r="26" fill="#0ea5e9" opacity="0.5"/>
  <rect x="462" y="67" width="56" height="56" rx="4" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="450" y="170" text-anchor="middle" fill="#e0533f">box: [W−x2, y1, W−x1, y2]</text>
  <defs><marker id="f1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>Flipping an image also moves the object, so its box must flip with it. Flip only the image and the ground truth becomes misaligned.</figcaption>
</figure>

```python
class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, s):
        for t in self.transforms:
            s = t(s)
        return s


class RandomHorizontalFlip:
    def __init__(self, p=0.5, seed=None):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.p, self.rng = p, np.random.default_rng(seed)
    def __call__(self, s):
        if self.rng.random() < self.p:
            s["image"] = s["image"][:, :, ::-1].copy()   # (C,H,W); .copy()!
            if "mask" in s:
                s["mask"] = s["mask"][:, ::-1].copy()
            if "boxes" in s:                             # xyxy must be remapped
                W = s["image"].shape[-1]
                b = s["boxes"].copy()
                b[:, 0], b[:, 2] = W - s["boxes"][:, 2], W - s["boxes"][:, 0]
                s["boxes"] = b
        return s


class RandomCrop:
    def __init__(self, size, seed=None):
        self.size, self.rng = size, np.random.default_rng(seed)
    def __call__(self, s):
        _, H, W = s["image"].shape
        if not 0 < self.size <= min(H, W):
            raise ValueError("crop size must fit inside the image")
        top = int(self.rng.integers(0, H - self.size + 1))
        left = int(self.rng.integers(0, W - self.size + 1))
        sl = (slice(top, top + self.size), slice(left, left + self.size))
        s["image"] = s["image"][:, sl[0], sl[1]]
        if "mask" in s:
            s["mask"] = s["mask"][sl]                     # same crop window
        if "boxes" in s:
            boxes = np.asarray(s["boxes"], dtype=np.float32).copy()
            boxes -= np.array([left, top, left, top], dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.size)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.size)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            s["boxes"] = boxes[keep]
            if "labels" in s:                              # aligned instance fields
                labels = np.asarray(s["labels"])
                if len(labels) != len(keep):
                    raise ValueError("labels and boxes must have equal length")
                s["labels"] = labels[keep]
        return s


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, np.float32)[:, None, None]
        self.std = np.array(std, np.float32)[:, None, None]
        if np.any(self.std <= 0):
            raise ValueError("std must be positive")
    def __call__(self, s):
        s["image"] = (s["image"] - self.mean) / self.std   # per-channel
        return s
```

> [!DANGER] The `.copy()` bug
> `x[:, :, ::-1]` returns a **view** with negative strides, not new memory. Later in-place ops (or PyTorch's `from_numpy`) can fail or alias. Always `.copy()` after a reversed/strided slice you intend to mutate or hand to a framework. This is the single most common augmentation bug.

## Putting it together

```python
class AugmentedDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images, self.masks, self.tf = images, masks, transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        s = {"image": self.images[i].astype(np.float32).copy(),
             "mask": self.masks[i].copy()}
        if self.tf: s = self.tf(s)
        return s["image"], s["mask"]


if __name__ == "__main__":
    x = np.arange(20, dtype=np.float32).reshape(20, 1)
    y = (np.arange(20) % 3).astype(np.int64)
    dl = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True,
                    drop_last=True, seed=1)
    assert len(dl) == 5
    xb, yb = next(iter(dl)); assert xb.shape == (4, 1) and yb.shape == (4,)

    imgs = np.random.rand(8, 3, 32, 32).astype(np.float32)
    masks = np.random.randint(0, 2, (8, 32, 32)).astype(np.int64)
    tf = Compose([RandomHorizontalFlip(1.0, 0), RandomCrop(28, 0),
                  Normalize([0.5] * 3, [0.5] * 3)])
    ib, mb = next(iter(DataLoader(AugmentedDataset(imgs, masks, tf), 2)))
    assert ib.shape == (2, 3, 28, 28) and mb.shape == (2, 28, 28)
    print("OK", ib.shape, mb.shape)
```

> [!NOTE] Framework one-liner
> `torch.utils.data.Dataset`/`DataLoader` + `torchvision.transforms.v2` (which transforms image + boxes + masks together) or `albumentations`. At scale: WebDataset shards or NVIDIA DALI for GPU-side decode/augment.

## Practice — implement, run, test

> [!TIP] How to use this section
> Each problem below has a **live Python editor** with NumPy preloaded. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime + NumPy (~15 MB); later runs are instant. These are the deterministic, pure-function core of the pipeline above (the `Dataset`/`DataLoader` classes stay as reference code); randomness is seeded so outputs are reproducible.

### 1. Per-channel normalize

Normalize a `(C, H, W)` image with per-channel `mean` and `std`: `(x − mean) / std`, broadcasting each stat over its channel.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"normalize","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef normalize(image, mean, std):\n    # image is (C,H,W); reshape mean/std to (C,1,1) and apply (x - mean) / std\n    pass","tests":[{"args":[[[[0,2],[4,6]]],[2],[2]],"expect":[[[-1,0],[1,2]]]},{"args":[[[[0,0],[0,0]],[[1,1],[1,1]]],[0,1],[1,1]],"expect":[[[0,0],[0,0]],[[0,0],[0,0]]]},{"args":[[[[2,4]]],[0],[2]],"expect":[[[1,2]]]}],"solution":"import numpy as np\n\ndef normalize(image, mean, std):\n    image = np.asarray(image, np.float32)\n    mean = np.asarray(mean, np.float32)[:, None, None]\n    std = np.asarray(std, np.float32)[:, None, None]\n    return (image - mean) / std"}
</script>
</div>

*Per-channel broadcast.* **Pitfall:** normalize *after* geometric transforms and *after* converting to float in $[0,1]$.

### 2. Horizontal flip (image)

Flip a `(C, H, W)` image left-right. Reverse the width axis — and `.copy()` so you return real memory, not a negative-stride view.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"horizontal_flip","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef horizontal_flip(image):\n    # reverse the last (width) axis of a (C,H,W) image; return a real copy\n    pass","tests":[{"args":[[[[1,2,3],[4,5,6]]]],"expect":[[[3,2,1],[6,5,4]]]},{"args":[[[[1,2],[3,4]]]],"expect":[[[2,1],[4,3]]]},{"args":[[[[1,2],[3,4]],[[5,6],[7,8]]]],"expect":[[[2,1],[4,3]],[[6,5],[8,7]]]}],"solution":"import numpy as np\n\ndef horizontal_flip(image):\n    image = np.asarray(image)\n    return image[:, :, ::-1].copy()"}
</script>
</div>

> [!DANGER] The `.copy()` bug
> `x[:, :, ::-1]` returns a **view** with negative strides, not new memory. Later in-place ops (or PyTorch's `from_numpy`) can fail or alias. Always `.copy()` after a reversed/strided slice you intend to mutate or hand to a framework.

### 3. Flip boxes with the image (label sync)

Here we use continuous, half-open `xyxy` coordinates with $0\le x_1\le x_2\le W$. Under this convention, a horizontal flip maps `[x1,y1,x2,y2]` to `[W−x2,y1,W−x1,y2]`. Older inclusive integer-pixel boxes use `W-1-x`; do not mix the conventions.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"flip_boxes","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef flip_boxes(boxes, width):\n    # boxes:(N,4) xyxy -> horizontally flipped: x1'=W-x2, x2'=W-x1, y unchanged\n    pass","tests":[{"args":[[[2,3,5,7]],10],"expect":[[5,3,8,7]]},{"args":[[[0,0,4,4],[1,1,2,2]],8],"expect":[[4,0,8,4],[6,1,7,2]]},{"args":[[[3,3,6,6]],10],"expect":[[4,3,7,6]]}],"solution":"import numpy as np\n\ndef flip_boxes(boxes, width):\n    boxes = np.asarray(boxes, float)\n    out = boxes.copy()\n    out[:, 0] = width - boxes[:, 2]\n    out[:, 2] = width - boxes[:, 0]\n    return out"}
</script>
</div>

*Getting this wrong silently corrupts labels — the mysterious accuracy ceiling.*

### 4. Seeded random crop

Crop a `size × size` window from a `(C, H, W)` image. Seed a fresh generator so a given `seed` always crops the same window, drawing `top` then `left`. For detection, apply the same pair to boxes and keypoints, clip boxes to the crop, and jointly remove zero-area instances and their aligned labels. The `RandomCrop` reference above implements this policy.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"random_crop","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef random_crop(image, size, seed):\n    # rng = np.random.default_rng(seed); draw top then left in [0, H-size] / [0, W-size]\n    pass","tests":[{"args":[[[[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]]],2,0],"expect":[[[21,22],[31,32]]]},{"args":[[[[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]]],2,1],"expect":[[[11,12],[21,22]]]},{"args":[[[[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]]],3,5],"expect":[[[11,12,13],[21,22,23],[31,32,33]]]}],"solution":"import numpy as np\n\ndef random_crop(image, size, seed):\n    rng = np.random.default_rng(seed)\n    image = np.asarray(image)\n    _, H, W = image.shape\n    top = int(rng.integers(0, H - size + 1))\n    left = int(rng.integers(0, W - size + 1))\n    return image[:, top:top + size, left:left + size]"}
</script>
</div>

*Seed the RNG inside the transform (or per worker), else every worker replays identical "random" crops.*

## Common bugs interviewers watch for

- **Missing `.copy()`** after reversed slices (see above).
- **Label desync:** flipping the image but not the boxes/mask.
- **Normalization order:** normalize *after* geometric transforms and *after* converting to float in $[0,1]$.
- **`drop_last` semantics:** off-by-one in `__len__`, or forgetting it changes batch count.
- **Reproducibility:** seed per-worker RNGs, else workers replay identical "random" augmentations.

## Q&A

<details class="qa"><summary>What does num_workers actually do, and what breaks?</summary>
<div class="qa-body">

**Short:** it forks/spawns worker processes that each hold a copy of the Dataset and prefetch batches in parallel, hiding data-loading latency behind GPU compute.

**Deep:** workers produce batches into a queue; `prefetch_factor` controls buffering and `pin_memory=True` helps host→device copies. PyTorch assigns each worker a default PyTorch seed, but if a Dataset or transform constructs its own NumPy generator or uses another library's RNG, seed it explicitly from `get_worker_info().seed`. With persistent workers, also define how desired randomness and reproducibility are refreshed each epoch.
</div></details>

<details class="qa"><summary>How do you handle variable-size targets (detection)?</summary>
<div class="qa-body">

**Short:** a custom `collate_fn` that keeps targets as a list (or pads them) instead of stacking, since each image has a different number of boxes.

**Deep:** images can be stacked after resize/pad to a common size, but per-image box/label counts differ, so `np.stack` on targets fails. The convention is to return images as a batched tensor and targets as a `list[dict]` of length `B` (Detectron2/torchvision detection style). Alternatively pad to the max count with an ignore flag. The model's loss then iterates or masks per image.
</div></details>

<details class="qa"><summary>Distributed training — how is data split across GPUs?</summary>
<div class="qa-body">

**Short:** a `DistributedSampler` shards indices so each rank sees a disjoint subset, and you call `sampler.set_epoch(e)` each epoch to keep shuffling synchronized and different across epochs.

**Deep:** under DDP, every rank processes a different shard of the same global permutation. If the dataset length is not divisible by the number of ranks and `drop_last=False`, `DistributedSampler` may pad indices and duplicate a few samples. With `drop_last=True`, it drops the tail instead. Check whether duplication biases evaluation metrics. `set_epoch(e)` changes the permutation each epoch.
</div></details>

### Follow-ups
- **Advanced augmentation?** RandAugment/TrivialAugment (policy search), Mosaic & Copy-Paste (YOLO/instance-seg)—all require careful label transforms. See [Vision Data Augmentation](#/cv/augmentation).
- **Class imbalance?** `WeightedRandomSampler` or per-class oversampling in the sampler, not the collate.
- **Deterministic runs?** Seed Python, NumPy, and torch RNGs + `set_epoch`; disable nondeterministic cudnn kernels.
- **I/O bottleneck?** Move JPEG decode into workers or GPU (DALI), use memory-mapped/sharded formats.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Dataset | `__len__` + `__getitem__(i)` |
| DataLoader | shuffle idx → batch → fetch → collate |
| `drop_last` | drop ragged final batch (BN/compile) |
| `collate_fn` | extension point for ragged targets |
| Augmentation | `Compose` of per-sample transforms |
| Label sync | flip/crop image **and** boxes/mask together |
| `.copy()` | after reversed/strided slices |
| `num_workers` | parallel prefetch; seed per worker |
| Distributed | `DistributedSampler` + `set_epoch` |

**Next:** [NumPy Primer](#/ml-coding/numpy-primer) · [Vision Data Augmentation](#/cv/augmentation) · [Regularization & Generalization](#/foundations/regularization-generalization) · [Object Detection](#/cv/detection) · [The ML Coding Round](#/ml-coding/intro)
