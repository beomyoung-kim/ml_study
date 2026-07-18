# Dataloader와 Augmentation

> [!TIP] 이렇게 먼저 말하세요
> "`DataLoader`는 iterator입니다. epoch마다 한 번 인덱스를 shuffle하고, 배치 단위로 슬라이스한 뒤, 각 샘플을 인덱스로 가져와서 리스트를 `collate`로 쌓아 배열로 만듭니다. Augmentation은 샘플마다 적용되는 transform들의 `Compose`인데, 유일한 미묘함은 label(box, mask)을 geometric transform과 동기화된 상태로 유지하는 것입니다." 코드가 아니라 인터페이스부터 말하세요.

이 문제는 어떤 알고리즘보다도 **인터페이스 설계와 edge case**를 봅니다. 면접관은 이 문제로 지원자가 실제로 학습 파이프라인을 짜본 적이 있는지 확인합니다 — `drop_last`, `collate_fn`, label 동기화 augmentation, 그리고 `num_workers`가 실제로 무슨 일을 하는지 아는 것이 그 단서입니다.

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

**핵심 설계 선택:** `shuffle`은 데이터가 아니라 인덱스를 재정렬합니다. `drop_last`는 마지막의 부분 배치를 방지합니다(BatchNorm 통계와 고정 shape 컴파일에 중요). `collate_fn`은 ragged 데이터(가변 크기 detection target)를 위한 확장 지점입니다.

## Augmentation: label 동기화의 미묘함

함정: 이미지에 적용한 flip이나 crop은 **box와 mask에도 대응되는 transform을 적용해야** 합니다. 이걸 틀리면 label이 조용히 오염되는데 — 원인 모를 정확도 상한처럼 보이는 버그입니다.

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

> [!DANGER] `.copy()` 버그
> `x[:, :, ::-1]`는 새 메모리가 아니라 음수 stride를 가진 **view**를 반환합니다. 이후의 in-place 연산(또는 PyTorch의 `from_numpy`)이 실패하거나 aliasing을 일으킬 수 있습니다. 변경하거나 프레임워크에 넘길 의도로 뒤집기/strided 슬라이스를 했다면 항상 `.copy()`를 하세요. 이것이 가장 흔한 augmentation 버그 하나입니다.

## 하나로 합치기

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

> [!NOTE] 프레임워크 한 줄
> `torch.utils.data.Dataset`/`DataLoader` + `torchvision.transforms.v2`(이미지 + box + mask를 함께 변환) 또는 `albumentations`. 대규모에서는 WebDataset shard나 GPU-side decode/augment를 위한 NVIDIA DALI.

## 면접관이 지켜보는 흔한 버그

- **뒤집힌 슬라이스 뒤 `.copy()` 누락**(위 참고).
- **Label 비동기화:** 이미지는 flip하면서 box/mask는 안 함.
- **Normalization 순서:** geometric transform *이후*, 그리고 $[0,1]$ float로 변환한 *이후*에 normalize.
- **`drop_last` 시맨틱:** `__len__`에서의 off-by-one, 또는 배치 개수가 바뀐다는 걸 잊음.
- **재현성:** worker별 RNG를 seed하지 않으면 worker들이 동일한 "random" augmentation을 반복 재생.

## Q&A

<details class="qa"><summary>num_workers는 실제로 무엇을 하고, 무엇이 깨지나?</summary>
<div class="qa-body">

**짧게:** Dataset의 복사본을 각각 가진 worker 프로세스를 fork/spawn하여 배치를 병렬로 prefetch함으로써, 데이터 로딩 지연을 GPU 연산 뒤에 숨깁니다.

**깊게:** 그렇지 않으면 메인 프로세스는 GPU step 사이에 디스크 I/O와 augmentation에서 block됩니다. Worker들은 배치를 큐에 생산하고, `prefetch_factor`가 버퍼링을, `pin_memory=True`가 host→device 복사 속도를 제어합니다. 함정: 각 worker는 부모의 RNG 상태를 상속받으므로, worker별 seeding(`worker_init_fn`이나 base seed + worker id를 통해) 없이는 동일한 augmentation을 생성합니다. worker가 너무 많으면 RAM/CPU를 thrash하고, fork-safe하지 않은 객체(열린 파일 핸들, CUDA tensor)는 넘겨받는 게 아니라 worker *내부*에서 생성해야 합니다.
</div></details>

<details class="qa"><summary>가변 크기 target(detection)은 어떻게 처리하나?</summary>
<div class="qa-body">

**짧게:** 이미지마다 box 개수가 다르므로, target을 stack하지 않고 리스트로 유지(또는 pad)하는 custom `collate_fn`을 씁니다.

**깊게:** 이미지는 resize/pad로 공통 크기로 맞춘 뒤 stack할 수 있지만, 이미지별 box/label 개수가 다르므로 target에 `np.stack`을 하면 실패합니다. 관례는 이미지를 배치 tensor로, target을 길이 `B`의 `list[dict]`로 반환하는 것입니다(Detectron2/torchvision detection 스타일). 대안으로 ignore flag와 함께 최대 개수까지 pad할 수도 있습니다. 그러면 모델의 loss가 이미지별로 iterate하거나 mask 처리합니다.
</div></details>

<details class="qa"><summary>분산 학습 — 데이터는 GPU 간에 어떻게 나뉘나?</summary>
<div class="qa-body">

**짧게:** `DistributedSampler`가 인덱스를 shard하여 각 rank가 서로 겹치지 않는 subset을 보게 하고, epoch마다 `sampler.set_epoch(e)`를 호출해 shuffle이 동기화되면서 epoch마다 달라지도록 유지합니다.

**깊게:** DDP에서 $N$개 rank 각각은 데이터셋의 $1/N$에 대해 자기 DataLoader를 돌립니다 — epoch당 어떤 샘플도 두 번 보지 않습니다. `set_epoch`는 epoch 번호로부터 shuffle을 deterministic하게 seed하므로, shard하기 전에 모든 rank가 동일한 permutation에 합의합니다. 데이터셋이 $N$으로 나누어떨어지지 않으면 sampler가 pad(샘플 몇 개를 중복)하여 배치 개수를 동일하게 유지하는데, 이는 gradient all-reduce가 lockstep을 유지하는 데 중요합니다. [분산 학습](#/foundations/distributed-training) 참고.
</div></details>

### Follow-ups
- **고급 augmentation?** RandAugment/TrivialAugment(policy search), Mosaic & Copy-Paste(YOLO/instance-seg) — 모두 세심한 label 변환이 필요합니다.
- **클래스 불균형?** collate가 아니라 sampler에서 `WeightedRandomSampler`나 클래스별 oversampling.
- **Deterministic 실행?** Python, NumPy, torch RNG + `set_epoch`를 seed하고, nondeterministic한 cudnn 커널을 비활성화.
- **I/O 병목?** JPEG decode를 worker나 GPU(DALI)로 옮기고, memory-mapped/sharded 포맷을 사용.

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

**Cross-links:** [분산 학습](#/foundations/distributed-training) · [Object Detection](#/cv/detection) · [Segmentation](#/cv/segmentation) · [The ML Coding Round](#/ml-coding/intro)
