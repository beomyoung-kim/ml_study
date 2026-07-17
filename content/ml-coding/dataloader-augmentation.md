# Dataloaders & Augmentation

> [!TIP] Say this first
> "A `DataLoader` is an iterator: shuffle indices once per epoch, slice them into batches, fetch each sample by index, and `collate` the list into stacked arrays. Augmentation is a `Compose` of transforms applied per sample — the one subtlety is keeping labels (boxes, masks) in sync with geometric transforms." Lead with the interface, not the code.

This problem tests **interface design and edge cases** more than any algorithm. Interviewers use it to see whether you've actually written training pipelines — the tells are `drop_last`, `collate_fn`, label-synced augmentation, and knowing what `num_workers` really does.

## Dataset + DataLoader

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

**Key design choices:** `shuffle` reorders indices, never the data. `drop_last` avoids a partial final batch (matters for BatchNorm stats and fixed-shape compilation). `collate_fn` is the extension point for ragged data (variable-size detection targets).

## Augmentation: the label-sync subtlety

The trap: a flip or crop applied to the image **must apply the matching transform to boxes and masks**. Getting this wrong silently corrupts labels — a bug that looks like a mysterious accuracy ceiling.

```python
class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, s):
        for t in self.transforms:
            s = t(s)
        return s


class RandomHorizontalFlip:
    def __init__(self, p=0.5, seed=None):
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
        top = int(self.rng.integers(0, H - self.size + 1))
        left = int(self.rng.integers(0, W - self.size + 1))
        sl = (slice(top, top + self.size), slice(left, left + self.size))
        s["image"] = s["image"][:, sl[0], sl[1]]
        if "mask" in s:
            s["mask"] = s["mask"][sl]                     # same crop window
        return s


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, np.float32)[:, None, None]
        self.std = np.array(std, np.float32)[:, None, None]
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

**Deep:** the main process would otherwise block on disk I/O and augmentation between GPU steps. Workers produce batches into a queue; `prefetch_factor` controls buffering and `pin_memory=True` speeds host→device copies. Pitfalls: each worker inherits the parent RNG state, so without per-worker seeding (via `worker_init_fn` or the base seed + worker id) they generate identical augmentations; too many workers thrash RAM/CPU; and non-fork-safe objects (open file handles, CUDA tensors) must be created *inside* the worker, not passed in.
</div></details>

<details class="qa"><summary>How do you handle variable-size targets (detection)?</summary>
<div class="qa-body">

**Short:** a custom `collate_fn` that keeps targets as a list (or pads them) instead of stacking, since each image has a different number of boxes.

**Deep:** images can be stacked after resize/pad to a common size, but per-image box/label counts differ, so `np.stack` on targets fails. The convention is to return images as a batched tensor and targets as a `list[dict]` of length `B` (Detectron2/torchvision detection style). Alternatively pad to the max count with an ignore flag. The model's loss then iterates or masks per image.
</div></details>

<details class="qa"><summary>Distributed training — how is data split across GPUs?</summary>
<div class="qa-body">

**Short:** a `DistributedSampler` shards indices so each rank sees a disjoint subset, and you call `sampler.set_epoch(e)` each epoch to keep shuffling synchronized and different across epochs.

**Deep:** with DDP, each of $N$ ranks runs its own DataLoader over $1/N$ of the dataset — no sample is seen twice per epoch. `set_epoch` seeds the shuffle deterministically from the epoch number so all ranks agree on the permutation before sharding. If the dataset isn't divisible by $N$, the sampler pads (duplicates a few samples) to keep batch counts equal, which matters for gradient all-reduce staying in lockstep. See [Distributed Training](#/foundations/distributed-training).
</div></details>

### Follow-ups
- **Advanced augmentation?** RandAugment/TrivialAugment (policy search), Mosaic & Copy-Paste (YOLO/instance-seg) — all require careful label transforms.
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

**Cross-links:** [Distributed Training](#/foundations/distributed-training) · [Object Detection](#/cv/detection) · [Segmentation](#/cv/segmentation) · [The ML Coding Round](#/ml-coding/intro)
