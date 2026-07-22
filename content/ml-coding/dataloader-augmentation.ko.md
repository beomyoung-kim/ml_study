# Dataloader와 Augmentation

> [!NOTE] 이 챕터의 목표
> 모델을 학습시키려면 데이터를 **한 번에 조금씩, 섞어서, 배치 단위로** 흘려보내야 합니다. 그 일을 하는 게 **DataLoader**이고, 데이터에 변형을 줘 다양성을 늘리는 게 **augmentation(증강)** 입니다. 이 챕터는 이 둘을 밑바닥부터 짜 보며 "학습 파이프라인이 실제로 어떻게 돌아가는지"를 잡습니다. 사전 지식은 [NumPy 프라이머](#/ml-coding/numpy-primer) 정도면 충분합니다.

## 무엇을, 왜

신경망은 데이터 전체를 한 번에 먹지 않습니다. **배치(batch, 작은 묶음)** 단위로 나눠서 조금씩 학습합니다(이유는 [Optimization](#/foundations/optimization)의 minibatch 참고). 그 묶음을 만들어 공급하는 부품이 두 개입니다:

- **Dataset**: "$i$번째 샘플 하나"를 꺼내 주는 창고. `__len__`(총 개수)과 `__getitem__(i)`(i번째 반환) 두 메서드가 전부입니다.
- **DataLoader**: 그 창고에서 인덱스를 **섞고(shuffle)**, **배치로 잘라**, 각 샘플을 가져와 하나의 배열로 **쌓는(collate)** 반복자(iterator).

**Augmentation(데이터 증강)** 은 이미지를 뒤집거나 자르거나 색을 바꿔 "같은 데이터의 다른 버전"을 즉석에서 만드는 것입니다. 데이터가 다양해질수록 모델이 통째로 외우지(과대적합) 못하고 **일반화**가 좋아집니다 — 이유는 [Regularization & 일반화](#/foundations/regularization-generalization) 참고.

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- dataset -->
  <text x="70" y="20" text-anchor="middle" fill="#0ea5e9" font-weight="700">Dataset</text>
  <g fill="none" stroke="#0ea5e9" stroke-width="1.6">
    <rect x="30" y="30" width="80" height="26" rx="4"/><rect x="30" y="60" width="80" height="26" rx="4"/>
    <rect x="30" y="90" width="80" height="26" rx="4"/><rect x="30" y="120" width="80" height="26" rx="4"/>
    <rect x="30" y="150" width="80" height="26" rx="4"/>
  </g>
  <text x="70" y="48" text-anchor="middle" fill="currentColor">샘플 0</text>
  <text x="70" y="78" text-anchor="middle" fill="currentColor">샘플 1</text>
  <text x="70" y="108" text-anchor="middle" fill="currentColor">샘플 2</text>
  <text x="70" y="138" text-anchor="middle" fill="currentColor">샘플 3</text>
  <text x="70" y="168" text-anchor="middle" fill="currentColor">샘플 4</text>
  <!-- shuffle -->
  <path d="M115 95 H175" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#d1)"/>
  <text x="215" y="90" text-anchor="middle" fill="#6366f1" font-weight="700">shuffle</text>
  <text x="215" y="106" text-anchor="middle" fill="#98a3b2">인덱스 섞기</text>
  <text x="215" y="122" text-anchor="middle" fill="#98a3b2">[3,0,4,1,2]</text>
  <!-- augment -->
  <path d="M258 95 H318" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#d1)"/>
  <text x="360" y="90" text-anchor="middle" fill="#e0533f" font-weight="700">augment</text>
  <text x="360" y="106" text-anchor="middle" fill="#98a3b2">flip·crop·norm</text>
  <!-- batch -->
  <path d="M402 95 H452" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#d1)"/>
  <rect x="460" y="60" width="150" height="70" rx="8" fill="none" stroke="#12a150" stroke-width="1.8"/>
  <text x="535" y="52" text-anchor="middle" fill="#12a150" font-weight="700">batch (묶음)</text>
  <g fill="#12a150"><rect x="472" y="75" width="18" height="40" rx="3"/><rect x="496" y="75" width="18" height="40" rx="3"/><rect x="520" y="75" width="18" height="40" rx="3"/></g>
  <text x="535" y="150" text-anchor="middle" fill="#98a3b2">→ 모델로</text>
  <defs><marker id="d1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>학습 데이터 파이프라인: 창고(Dataset)에서 인덱스를 섞고 → 증강을 적용하고 → 배치로 쌓아 모델에 공급합니다. DataLoader가 이 전체를 반복(epoch마다)합니다.</figcaption>
</figure>

> [!TIP] 면접 한 줄
> "`DataLoader`는 iterator입니다. epoch마다 한 번 인덱스를 shuffle하고, 배치 단위로 슬라이스한 뒤, 각 샘플을 인덱스로 가져와서 리스트를 `collate`로 쌓아 배열로 만듭니다. Augmentation은 샘플마다 적용되는 transform들의 `Compose`인데, 유일한 미묘함은 label(box, mask)을 geometric transform과 동기화된 상태로 유지하는 것입니다." — 코드가 아니라 **인터페이스**부터 말하면 실제로 학습 파이프라인을 짜 본 사람으로 들립니다. 면접관이 보는 단서는 `drop_last`, `collate_fn`, label 동기화, `num_workers`를 아는지입니다.

## Dataset + DataLoader 밑바닥 구현

아래는 PyTorch의 `Dataset`/`DataLoader`를 NumPy로 최소 구현한 것입니다. 실제 프레임워크도 골격은 똑같습니다.

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

**핵심 설계 선택 (용어 풀이):**
- **shuffle(섞기)**: 데이터 자체가 아니라 **인덱스**를 재정렬합니다 — 싸고 간단합니다.
- **drop_last(마지막 버림)**: 마지막에 남는 반쪽 배치를 버립니다. 배치 크기가 고정돼야 하는 BatchNorm 통계나 컴파일에 중요합니다.
- **collate_fn(쌓기 함수)**: 샘플 리스트를 하나의 배열로 합치는 지점. 크기가 제각각인 데이터(가변 개수 detection box)를 다룰 때 여기를 커스터마이즈합니다.

## Augmentation: label 동기화의 미묘함

가장 흔한 함정: 이미지에 flip이나 crop을 적용했으면 **그에 대응하는 변형을 box와 mask에도 똑같이** 적용해야 합니다. 이걸 빠뜨리면 정답(label)이 조용히 어긋나 — 원인 모를 정확도 상한처럼 보이는 버그가 됩니다.

<figure>
<svg viewBox="0 0 640 190" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- original -->
  <text x="150" y="20" text-anchor="middle" fill="#98a3b2">원본</text>
  <rect x="60" y="30" width="180" height="120" rx="6" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <circle cx="110" cy="95" r="26" fill="#0ea5e9" opacity="0.5"/>
  <rect x="82" y="67" width="56" height="56" rx="4" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="150" y="170" text-anchor="middle" fill="#e0533f">box: [x1,y1,x2,y2]</text>
  <!-- arrow -->
  <path d="M255 90 H320" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#f1)"/>
  <text x="288" y="80" text-anchor="middle" fill="#6366f1">flip →</text>
  <!-- flipped: image AND box both flip -->
  <text x="480" y="20" text-anchor="middle" fill="#12a150">좌우 뒤집기 (동기화 ✓)</text>
  <rect x="360" y="30" width="180" height="120" rx="6" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <circle cx="490" cy="95" r="26" fill="#0ea5e9" opacity="0.5"/>
  <rect x="462" y="67" width="56" height="56" rx="4" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <text x="450" y="170" text-anchor="middle" fill="#e0533f">box: [W−x2, y1, W−x1, y2]</text>
  <defs><marker id="f1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>이미지를 뒤집으면 그 안의 물체 위치(box)도 함께 뒤집어야 합니다. 이미지만 뒤집고 box를 그대로 두면 정답이 어긋납니다.</figcaption>
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

> [!DANGER] `.copy()` 버그
> `x[:, :, ::-1]`는 새 메모리가 아니라 음수 stride를 가진 **view(뷰)** 를 반환합니다. 이후의 in-place 연산(또는 PyTorch의 `from_numpy`)이 실패하거나 aliasing(같은 메모리 공유)을 일으킬 수 있습니다. 변형하거나 프레임워크에 넘길 의도로 뒤집기/strided 슬라이스를 했다면 항상 `.copy()`를 하세요. 가장 흔한 augmentation 버그 하나입니다.

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

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 NumPy가 미리 로드된 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다. 위 파이프라인의 결정적이고 순수한 함수 핵심들이며(`Dataset`/`DataLoader` 클래스는 참고 코드로 유지), randomness는 seed되어 출력이 재현 가능합니다.

### 1. 채널별 normalize

`(C, H, W)` 이미지를 채널별 `mean`, `std`로 normalize합니다: `(x − mean) / std`, 각 통계를 해당 채널에 broadcast합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"normalize","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef normalize(image, mean, std):\n    # image is (C,H,W); reshape mean/std to (C,1,1) and apply (x - mean) / std\n    pass","tests":[{"args":[[[[0,2],[4,6]]],[2],[2]],"expect":[[[-1,0],[1,2]]]},{"args":[[[[0,0],[0,0]],[[1,1],[1,1]]],[0,1],[1,1]],"expect":[[[0,0],[0,0]],[[0,0],[0,0]]]},{"args":[[[[2,4]]],[0],[2]],"expect":[[[1,2]]]}],"solution":"import numpy as np\n\ndef normalize(image, mean, std):\n    image = np.asarray(image, np.float32)\n    mean = np.asarray(mean, np.float32)[:, None, None]\n    std = np.asarray(std, np.float32)[:, None, None]\n    return (image - mean) / std"}
</script>
</div>

*채널별 broadcast.* **함정:** geometric transform *이후*, 그리고 $[0,1]$ float로 변환한 *이후*에 normalize.

### 2. Horizontal flip (이미지)

`(C, H, W)` 이미지를 좌우로 뒤집습니다. width 축을 뒤집고 — 음수 stride view가 아니라 실제 메모리를 반환하도록 `.copy()`합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"horizontal_flip","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef horizontal_flip(image):\n    # reverse the last (width) axis of a (C,H,W) image; return a real copy\n    pass","tests":[{"args":[[[[1,2,3],[4,5,6]]]],"expect":[[[3,2,1],[6,5,4]]]},{"args":[[[[1,2],[3,4]]]],"expect":[[[2,1],[4,3]]]},{"args":[[[[1,2],[3,4]],[[5,6],[7,8]]]],"expect":[[[2,1],[4,3]],[[6,5],[8,7]]]}],"solution":"import numpy as np\n\ndef horizontal_flip(image):\n    image = np.asarray(image)\n    return image[:, :, ::-1].copy()"}
</script>
</div>

> [!DANGER] `.copy()` 버그
> `x[:, :, ::-1]`는 새 메모리가 아니라 음수 stride를 가진 **view**를 반환합니다. 이후의 in-place 연산(또는 PyTorch의 `from_numpy`)이 실패하거나 aliasing을 일으킬 수 있습니다. 변경하거나 프레임워크에 넘길 의도로 뒤집기/strided 슬라이스를 했다면 항상 `.copy()`를 하세요.

### 3. 이미지와 함께 box 뒤집기 (label 동기화)

여기서는 연속/half-open `xyxy` 좌표($0\le x_1\le x_2\le W$)를 사용합니다. 이 convention에서 좌우 flip은 `[x1,y1,x2,y2] → [W−x2,y1,W−x1,y2]`입니다. 옛 inclusive integer-pixel box는 `W-1-x`를 써야 하므로 두 convention을 섞지 마세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"flip_boxes","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef flip_boxes(boxes, width):\n    # boxes:(N,4) xyxy -> horizontally flipped: x1'=W-x2, x2'=W-x1, y unchanged\n    pass","tests":[{"args":[[[2,3,5,7]],10],"expect":[[5,3,8,7]]},{"args":[[[0,0,4,4],[1,1,2,2]],8],"expect":[[4,0,8,4],[6,1,7,2]]},{"args":[[[3,3,6,6]],10],"expect":[[4,3,7,6]]}],"solution":"import numpy as np\n\ndef flip_boxes(boxes, width):\n    boxes = np.asarray(boxes, float)\n    out = boxes.copy()\n    out[:, 0] = width - boxes[:, 2]\n    out[:, 2] = width - boxes[:, 0]\n    return out"}
</script>
</div>

*이걸 틀리면 label이 조용히 오염됩니다 — 원인 모를 정확도 상한.*

### 4. Seed된 random crop

`(C, H, W)` 이미지에서 `size × size` window를 crop합니다. 새 generator를 seed하여 주어진 `seed`가 항상 같은 window를 crop하도록 합니다(`top` 다음 `left` 순서로 draw). Detection에서는 같은 `(top,left)`을 box/keypoint에도 적용하고, crop 밖으로 잘린 box는 clip한 뒤 넓이가 0인 instance와 정렬된 label을 함께 제거해야 합니다. 위 `RandomCrop` 참고 코드가 이 정책을 구현합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"random_crop","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef random_crop(image, size, seed):\n    # rng = np.random.default_rng(seed); draw top then left in [0, H-size] / [0, W-size]\n    pass","tests":[{"args":[[[[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]]],2,0],"expect":[[[21,22],[31,32]]]},{"args":[[[[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]]],2,1],"expect":[[[11,12],[21,22]]]},{"args":[[[[0,1,2,3],[10,11,12,13],[20,21,22,23],[30,31,32,33]]],3,5],"expect":[[[11,12,13],[21,22,23],[31,32,33]]]}],"solution":"import numpy as np\n\ndef random_crop(image, size, seed):\n    rng = np.random.default_rng(seed)\n    image = np.asarray(image)\n    _, H, W = image.shape\n    top = int(rng.integers(0, H - size + 1))\n    left = int(rng.integers(0, W - size + 1))\n    return image[:, top:top + size, left:left + size]"}
</script>
</div>

*RNG를 transform 내부(또는 worker별)에서 seed하세요. 그렇지 않으면 모든 worker가 동일한 "random" crop을 반복 재생합니다.*

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

**깊게:** Worker들은 배치를 큐에 생산하고 `prefetch_factor`가 버퍼링을, `pin_memory=True`가 host→device 복사를 돕습니다. PyTorch는 worker마다 기본 PyTorch seed를 부여하지만, Dataset/transform이 별도 NumPy generator나 외부 라이브러리 RNG를 만들면 `get_worker_info().seed`를 이용해 명시적으로 seed해야 합니다. persistent worker에서는 epoch마다 원하는 randomness와 재현성을 어떻게 갱신할지도 정합니다.
</div></details>

<details class="qa"><summary>가변 크기 target(detection)은 어떻게 처리하나?</summary>
<div class="qa-body">

**짧게:** 이미지마다 box 개수가 다르므로, target을 stack하지 않고 리스트로 유지(또는 pad)하는 custom `collate_fn`을 씁니다.

**깊게:** 이미지는 resize/pad로 공통 크기로 맞춘 뒤 stack할 수 있지만, 이미지별 box/label 개수가 다르므로 target에 `np.stack`을 하면 실패합니다. 관례는 이미지를 배치 tensor로, target을 길이 `B`의 `list[dict]`로 반환하는 것입니다(Detectron2/torchvision detection 스타일). 대안으로 ignore flag와 함께 최대 개수까지 pad할 수도 있습니다. 그러면 모델의 loss가 이미지별로 iterate하거나 mask 처리합니다.
</div></details>

<details class="qa"><summary>분산 학습 — 데이터는 GPU 간에 어떻게 나뉘나?</summary>
<div class="qa-body">

**짧게:** `DistributedSampler`가 인덱스를 shard하여 각 rank가 서로 겹치지 않는 subset을 보게 하고, epoch마다 `sampler.set_epoch(e)`를 호출해 shuffle이 동기화되면서 epoch마다 달라지도록 유지합니다.

**깊게:** DDP에서 각 rank는 같은 global permutation의 서로 다른 shard를 처리합니다. 다만 데이터셋 길이가 rank 수로 나누어떨어지지 않고 `drop_last=False`이면 `DistributedSampler`가 index를 padding해 일부 샘플을 중복할 수 있습니다. 반대로 `drop_last=True`는 tail을 버립니다. 평가에서 이 중복이 metric을 편향하지 않는지도 확인하세요. `set_epoch(e)`는 epoch별 permutation이 달라지게 합니다.
</div></details>

### Follow-ups
- **고급 augmentation?** RandAugment/TrivialAugment(policy search), Mosaic & Copy-Paste(YOLO/instance-seg) — 모두 세심한 label 변환이 필요합니다. [비전 데이터 증강](#/cv/augmentation) 참고.
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

**다음:** [NumPy 프라이머](#/ml-coding/numpy-primer) · [비전 데이터 증강](#/cv/augmentation) · [Regularization & 일반화](#/foundations/regularization-generalization) · [Object Detection](#/cv/detection) · [ML 코딩 라운드](#/ml-coding/intro)
