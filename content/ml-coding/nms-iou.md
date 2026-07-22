# IoU & Non-Max Suppression

> [!NOTE] Goal of this chapter
> You will implement two indispensable object-detection tools. **IoU** is a ruler for how much two boxes overlap; **NMS** is a cleanup step that keeps the best among duplicate boxes around the same object. Both use broadcasting from the [NumPy Primer](#/ml-coding/numpy-primer) to avoid inner Python loops. This chapter leads naturally into [Object Detection](#/cv/detection).

## What and why

A detector usually emits **several overlapping boxes** around one object. That creates two problems:

1. We need a ruler for **“Do these boxes refer to the same object?”** → **IoU**.
2. We need a way to **remove duplicates** → **NMS (Non-Max Suppression)**.

**IoU (Intersection over Union)** divides the overlap area of two boxes by their combined area. It ranges from 0, no overlap, to 1, identical boxes.

<figure>
<svg viewBox="0 0 640 230" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="13">
  <!-- intersection shaded first (under strokes) -->
  <rect x="200" y="95" width="90" height="80" fill="rgba(18,161,80,.35)"/>
  <!-- box A -->
  <rect x="70" y="45" width="220" height="130" fill="rgba(99,102,241,.10)" stroke="#6366f1" stroke-width="2.5"/>
  <text x="110" y="38" fill="#6366f1" font-weight="700">box A</text>
  <!-- box B -->
  <rect x="200" y="95" width="200" height="110" fill="rgba(224,83,63,.10)" stroke="#e0533f" stroke-width="2.5"/>
  <text x="360" y="222" fill="#e0533f" font-weight="700">box B</text>
  <!-- intersection label -->
  <text x="245" y="140" text-anchor="middle" fill="#12a150" font-weight="700" font-size="12">intersection</text>
  <text x="245" y="156" text-anchor="middle" fill="#12a150" font-size="10">(∩)</text>
  <!-- formula -->
  <text x="470" y="90" fill="currentColor" font-size="14">IoU =</text>
  <line x1="470" y1="105" x2="600" y2="105" stroke="currentColor" stroke-width="1.2"/>
  <text x="535" y="98" text-anchor="middle" fill="#12a150">intersection area</text>
  <text x="535" y="125" text-anchor="middle" fill="#98a3b2">union area (∪)</text>
  <text x="470" y="165" fill="#98a3b2" font-size="11">0 = no overlap</text>
  <text x="470" y="182" fill="#98a3b2" font-size="11">1 = identical</text>
</svg>
<figcaption>IoU divides the <b>intersection</b>, shown in green, by the <b>union</b> of the boxes. Detection evaluation and the overlap decision in NMS both use this one number.</figcaption>
</figure>

**NMS** cleans up a cluster of overlapping boxes: keep the highest-scoring box, drop boxes whose IoU with it exceeds a threshold, and repeat on the survivors.

<figure>
<svg viewBox="0 0 640 190" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="150" y="20" text-anchor="middle" fill="#98a3b2" font-weight="700">before NMS—duplicates</text>
  <rect x="70" y="45" width="150" height="95" fill="none" stroke="#98a3b2" stroke-width="1.5"/><text x="78" y="60" fill="#98a3b2">0.72</text>
  <rect x="85" y="55" width="150" height="95" fill="none" stroke="#98a3b2" stroke-width="1.5"/><text x="200" y="70" fill="#98a3b2">0.68</text>
  <rect x="60" y="35" width="150" height="95" fill="none" stroke="#e0533f" stroke-width="2.5"/><text x="66" y="30" fill="#e0533f" font-weight="700">0.95 ★</text>
  <text x="320" y="95" text-anchor="middle" font-size="22" fill="#98a3b2">→</text>
  <text x="500" y="20" text-anchor="middle" fill="#98a3b2" font-weight="700">after NMS—keep the best</text>
  <rect x="425" y="35" width="150" height="95" fill="rgba(224,83,63,.12)" stroke="#e0533f" stroke-width="2.5"/><text x="431" y="30" fill="#e0533f" font-weight="700">0.95 ✓</text>
</svg>
<figcaption>Of three boxes around the same object, NMS keeps the 0.95 box and removes the others that overlap it too much.</figcaption>
</figure>

> [!TIP] Interview one-liner
> “IoU broadcasts the `max` of top-left corners and `min` of bottom-right corners to form the intersection, then clamps negative side lengths to zero. NMS is a greedy loop: keep the top score, drop boxes that overlap it too much, and repeat.” Interviewers watch **vectorization** and **edge cases**—zero-area boxes, full containment, and empty input—more than the basic algorithm.

## The math

Boxes use continuous `[x1,y1,x2,y2]` coordinates, and their area is $(x_2-x_1)(y_2-y_1)$ with no `+1`; this differs from the old VOC integer-pixel convention. For two boxes $A,B$:

$$
\text{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A\cap B|}{|A| + |B| - |A\cap B|}
$$

The intersection rectangle has corners $\big(\max(x_1^A,x_1^B),\,\max(y_1^A,y_1^B)\big)$ and $\big(\min(x_2^A,x_2^B),\,\min(y_2^A,y_2^B)\big)$; if either side is negative the boxes don't overlap, so clamp to $0$.

## Practice — implement, run, test

> [!TIP] How to use this section
> Each problem below has a **live Python editor**. Write your solution, hit **▶ Run tests**, and see which cases pass. Stuck? Reveal a reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime plus NumPy (~15 MB); later runs are instant.

Build them in order: area first (it guards the degenerate boxes), then the broadcasted IoU grid, then the greedy loop that sits on top of it.

### 1. Box area <span class="badge badge-easy">Easy</span>

Area of each `[x1, y1, x2, y2]` box, clamping negative side lengths to zero so degenerate boxes score `0`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"box_area","packages":["numpy"],"approx":true,"starter":"def box_area(boxes):\n    # (N,4) xyxy boxes -> (N,) areas; clamp negative side lengths to 0\n    pass","tests":[{"args":[[[0,0,10,10],[1,1,4,5],[5,5,2,2]]],"expect":[100.0,12.0,0.0]},{"args":[[[0,0,2,3],[0,0,0,9]]],"expect":[6.0,0.0]},{"args":[[[2,2,5,7]]],"expect":[15.0]},{"args":[[]],"expect":[]}],"solution":"import numpy as np\n\ndef box_area(boxes):\n    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)\n    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])"}
</script>
</div>

*O(N), fully vectorized.* **Pitfall:** no `+1` — these are continuous `xyxy` coordinates, not VOC integer pixels.

### 2. Pairwise IoU by broadcasting <span class="badge badge-med">Medium</span>

The full $(N,M)$ IoU grid: broadcast `max` of the top-lefts and `min` of the bottom-rights, clamp, then divide by the union.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"iou_matrix","packages":["numpy"],"approx":true,"starter":"def iou_matrix(a, b):\n    # a:(N,4), b:(M,4) -> (N,M); broadcast max of top-lefts, min of bottom-rights, clamp\n    pass","tests":[{"args":[[[0,0,10,10]],[[0,0,10,10]]],"expect":[[1.0]]},{"args":[[[0,0,10,10]],[[1,1,11,11]]],"expect":[[0.680672268907563]]},{"args":[[[0,0,10,10],[20,20,30,30]],[[0,0,10,10],[5,5,15,15]]],"expect":[[1.0,0.14285714285714285],[0.0,0.0]]},{"args":[[[0,0,2,2]],[[10,10,12,12]]],"expect":[[0.0]]},{"args":[[[0,0,1,1],[2,2,3,3]],[]],"expect":[[],[]]}],"solution":"import numpy as np\n\ndef iou_matrix(a, b):\n    a = np.asarray(a, dtype=float).reshape(-1, 4)\n    b = np.asarray(b, dtype=float).reshape(-1, 4)\n    def area(x):\n        return np.maximum(0.0, x[:, 2] - x[:, 0]) * np.maximum(0.0, x[:, 3] - x[:, 1])\n    lt = np.maximum(a[:, None, :2], b[None, :, :2])\n    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])\n    wh = np.maximum(0.0, rb - lt)\n    inter = wh[..., 0] * wh[..., 1]\n    union = area(a)[:, None] + area(b)[None, :] - inter\n    return inter / np.maximum(union, 1e-8)"}
</script>
</div>

The `[:, None]` / `[None, :]` axis insertions are the whole game: they line up the $(N,1,\cdot)$ and $(1,M,\cdot)$ tensors so broadcasting produces the full $(N,M)$ grid without a loop. Review broadcasting in the [NumPy Primer](#/ml-coding/numpy-primer). **Complexity:** $O(NM)$ time and memory.

### 3. Greedy NMS <span class="badge badge-med">Medium</span>

Sort by score, keep the top box, drop every survivor whose IoU with it exceeds the threshold, and repeat. For equal scores, this educational implementation uses a stable sort to preserve input-index order. Actual CPU and GPU NMS implementations may resolve ties differently, so product logic should not depend on tie order.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"nms","packages":["numpy"],"starter":"def nms(boxes, scores, iou_thr=0.5):\n    # greedy: keep the top score, drop boxes with IoU > thr, repeat; return kept indices\n    pass","tests":[{"args":[[[0,0,10,10],[1,1,11,11],[20,20,30,30],[0,0,10,10]],[0.9,0.8,0.7,0.95],0.5],"expect":[3,2]},{"args":[[[0,0,10,10],[1,1,11,11]],[0.9,0.8],0.5],"expect":[0]},{"args":[[[0,0,10,10],[1,1,11,11]],[0.9,0.8],0.95],"expect":[0,1]},{"args":[[[0,0,10,10],[100,100,110,110]],[0.5,0.9],0.5],"expect":[1,0]},{"args":[[],[],0.5],"expect":[]}],"solution":"import numpy as np\n\ndef nms(boxes, scores, iou_thr=0.5):\n    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)\n    scores = np.asarray(scores, dtype=float).reshape(-1)\n    if len(boxes) != len(scores):\n        raise ValueError('boxes and scores must have equal length')\n    def iou_m(a, b):\n        lt = np.maximum(a[:, None, :2], b[None, :, :2])\n        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])\n        wh = np.maximum(0.0, rb - lt)\n        inter = wh[..., 0] * wh[..., 1]\n        area = lambda x: np.maximum(0.0, x[:, 2] - x[:, 0]) * np.maximum(0.0, x[:, 3] - x[:, 1])\n        union = area(a)[:, None] + area(b)[None, :] - inter\n        return inter / np.maximum(union, 1e-8)\n    if boxes.size == 0:\n        return []\n    order = np.argsort(-scores, kind='stable')\n    keep = []\n    while order.size > 0:\n        i = int(order[0])\n        keep.append(i)\n        if order.size == 1:\n            break\n        ious = iou_m(boxes[i][None], boxes[order[1:]])[0]\n        order = order[1:][ious <= iou_thr]\n    return keep"}
</script>
</div>

The loop is **inherently sequential** — each kept box changes which candidates survive — but each step is vectorized. **Complexity:** $O(N^2)$ worst case; each iteration is one broadcasted IoU against the survivors.

## Soft-NMS (the common follow-up)

Instead of hard-deleting overlapping boxes, *decay* their scores — better recall for crowded scenes (Bodla et al., 2017).

```python
def soft_nms(boxes, scores, iou_thr=0.5, sigma=0.5,
             score_thr=1e-3, method="gaussian"):
    """Returns (kept_boxes, kept_scores). Decays scores instead of removing."""
    boxes, scores = boxes.copy(), scores.copy().astype(np.float64)
    N = boxes.shape[0]
    for i in range(N):
        m = i + int(np.argmax(scores[i:]))          # bring current max to front
        boxes[[i, m]], scores[[i, m]] = boxes[[m, i]], scores[[m, i]]
        if i + 1 >= N:
            break
        ious = iou_matrix(boxes[i][None], boxes[i + 1:])[0]
        if method == "linear":
            w = np.where(ious > iou_thr, 1.0 - ious, 1.0)
        else:  # gaussian
            w = np.exp(-(ious ** 2) / sigma)
        scores[i + 1:] *= w                          # decay, don't delete
    keep = scores >= score_thr
    return boxes[keep], scores[keep]
```

## Sanity check

```python
if __name__ == "__main__":
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11],
                      [20, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.7, 0.95])
    # closed-form check: 9x9 overlap over (100+100-81) union
    assert abs(iou_matrix(boxes[0][None], boxes[1][None])[0, 0]
               - 81 / 119) < 1e-9
    keep = nms(boxes, scores, iou_thr=0.5)
    assert 2 in keep and 3 in keep          # far box + top-scoring duplicate
    print("keep:", keep, "OK")
```

> [!NOTE] Framework one-liner
> Production uses `torchvision.ops.box_iou`, `torchvision.ops.nms(boxes, scores, iou_thr)`, and `torchvision.ops.batched_nms(boxes, scores, idxs, iou_thr)` for class-aware suppression.

## Common bugs interviewers watch for

<div class="proscons"><div><div class="pros-t">Do</div>

- Use `argsort()[::-1]` — `argsort` is **ascending**.
- Clamp `rb - lt` to `0` (non-overlapping boxes give negative width).
- Guard the union with `np.maximum(union, eps)`.
- `.copy()` scores in Soft-NMS before mutating.

</div><div><div class="cons-t">Don't</div>

- Add `+1` to widths — that's the old VOC integer-pixel convention, not `xyxy` continuous.
- Materialize the full $N\times N$ matrix inside the NMS loop (compute winner-vs-rest only).
- Forget the empty-input guard (`boxes.size == 0`).
- Confuse class-aware vs class-agnostic NMS.

</div></div>

## Q&A

<details class="qa"><summary>How do you make NMS class-aware without a per-class Python loop?</summary>
<div class="qa-body">

**Short:** offset each box's coordinates by a large per-class constant so boxes of different classes can never overlap, then run one global NMS — that's exactly `batched_nms`.

**Deep:** under a contract with non-negative coordinates and enough dtype range, add `class_id * (max_coordinate + 1)` to every coordinate. If negative coordinates are allowed, base the offset on the full coordinate span; also avoid FP16 overflow. When those assumptions are unclear, explicit per-class NMS is safer.
</div></details>

<details class="qa"><summary>Soft-NMS vs hard NMS — when and why?</summary>
<div class="qa-body">

**Short:** Soft-NMS decays overlapping scores instead of deleting them, improving recall in crowded scenes at a small latency cost.

**Deep:** hard NMS can erase a true positive that happens to overlap a higher-scoring neighbor (two people standing together). Soft-NMS instead multiplies the neighbor's score by a Gaussian $e^{-\text{IoU}^2/\sigma}$ or linear $1-\text{IoU}$ weight, so a genuinely separate object survives with a reduced-but-nonzero score and can still be recovered above threshold. It's a drop-in change to the ranking, no retraining. Matrix-NMS (SOLOv2) parallelizes the same idea.
</div></details>

<details class="qa"><summary>What about rotated boxes or mask IoU?</summary>
<div class="qa-body">

**Short:** rotated IoU needs polygon-clipping intersection area; mask IoU is a pixelwise AND over OR.

**Deep:** axis-aligned `min/max` no longer bounds the intersection once boxes rotate — you compute the intersection polygon (Sutherland–Hodgman clipping) and its area via the shoelace formula. Mask IoU is simpler and more robust for segmentation: `(pred & gt).sum() / (pred | gt).sum()` per instance — the same quantity mIoU accumulates in a confusion matrix (see [mAP & mIoU](#/ml-coding/metrics-map-miou)).
</div></details>

### Follow-ups
- **Memory at large $N$?** The full $(N,M)$ matrix is $O(N^2)$; for very large candidate sets prefer the greedy winner-vs-rest form or tile the matrix.
- **GIoU / DIoU / CIoU?** Variants that add a penalty for the enclosing box / center distance / aspect ratio — used as *regression losses*, not just for NMS.
- **Where does this sit in your work?** Natural bridge to your detection/segmentation post-processing experience.

## Cheat-sheet

| Item | Formula / trick | Note |
| --- | --- | --- |
| Area (xyxy) | $(x_2-x_1)(y_2-y_1)$, clamp $\ge 0$ | no `+1` for continuous coords |
| Intersection | `max(0, min(rb) - max(lt))` | broadcast lt/rb to $(N,M,2)$ |
| IoU | $\dfrac{I}{A+B-I}$, guard denom | `/ max(union, 1e-8)` |
| Greedy NMS | sort desc, keep top, drop IoU > thr | $O(N^2)$, sequential |
| Soft-NMS | decay: $e^{-\text{IoU}^2/\sigma}$ | better recall, small cost |
| Class-aware | coordinate offset per class | one global pass (`batched_nms`) |

**Next:** [Object Detection](#/cv/detection) · [mAP & mIoU](#/ml-coding/metrics-map-miou) · [NumPy Primer](#/ml-coding/numpy-primer)
