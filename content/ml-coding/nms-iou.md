# IoU & Non-Max Suppression

> [!TIP] Say this first
> "IoU is intersection over union; the whole trick is computing the intersection rectangle by broadcasting `max` of the top-lefts and `min` of the bottom-rights, then clamping negatives to zero. NMS is a greedy loop: keep the top score, drop everything that overlaps it too much, repeat." Then code it.

The bread-and-butter of any detection/segmentation loop. Boxes are `[x1, y1, x2, y2]` in continuous coordinates, so area is $(x_2-x_1)(y_2-y_1)$ (no `+1`). The interviewer is watching **vectorization** and **edge cases** (zero-area boxes, full containment, empty input) far more than the algorithm itself.

## The math

For two boxes $A, B$:

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
{"func":"box_area","packages":["numpy"],"approx":true,"starter":"def box_area(boxes):\n    # (N,4) xyxy boxes -> (N,) areas; clamp negative side lengths to 0\n    pass","tests":[{"args":[[[0,0,10,10],[1,1,4,5],[5,5,2,2]]],"expect":[100.0,12.0,0.0]},{"args":[[[0,0,2,3],[0,0,0,9]]],"expect":[6.0,0.0]},{"args":[[[2,2,5,7]]],"expect":[15.0]}],"solution":"import numpy as np\n\ndef box_area(boxes):\n    boxes = np.asarray(boxes, dtype=float)\n    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])"}
</script>
</div>

*O(N), fully vectorized.* **Pitfall:** no `+1` — these are continuous `xyxy` coordinates, not VOC integer pixels.

### 2. Pairwise IoU by broadcasting <span class="badge badge-med">Medium</span>

The full $(N,M)$ IoU grid: broadcast `max` of the top-lefts and `min` of the bottom-rights, clamp, then divide by the union.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"iou_matrix","packages":["numpy"],"approx":true,"starter":"def iou_matrix(a, b):\n    # a:(N,4), b:(M,4) -> (N,M); broadcast max of top-lefts, min of bottom-rights, clamp\n    pass","tests":[{"args":[[[0,0,10,10]],[[0,0,10,10]]],"expect":[[1.0]]},{"args":[[[0,0,10,10]],[[1,1,11,11]]],"expect":[[0.680672268907563]]},{"args":[[[0,0,10,10],[20,20,30,30]],[[0,0,10,10],[5,5,15,15]]],"expect":[[1.0,0.14285714285714285],[0.0,0.0]]},{"args":[[[0,0,2,2]],[[10,10,12,12]]],"expect":[[0.0]]}],"solution":"import numpy as np\n\ndef iou_matrix(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    def area(x):\n        return np.maximum(0.0, x[:, 2] - x[:, 0]) * np.maximum(0.0, x[:, 3] - x[:, 1])\n    lt = np.maximum(a[:, None, :2], b[None, :, :2])\n    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])\n    wh = np.maximum(0.0, rb - lt)\n    inter = wh[..., 0] * wh[..., 1]\n    union = area(a)[:, None] + area(b)[None, :] - inter\n    return inter / np.maximum(union, 1e-8)"}
</script>
</div>

The `[:, None]` / `[None, :]` axis insertions are the whole game: they line up the $(N,1,\cdot)$ and $(1,M,\cdot)$ tensors so broadcasting produces the full $(N,M)$ grid without a loop. **Complexity:** $O(NM)$ time and memory.

### 3. Greedy NMS <span class="badge badge-med">Medium</span>

Sort by score, keep the top box, drop every survivor whose IoU with it exceeds the threshold, repeat. Returns kept indices in decreasing-score order.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"nms","packages":["numpy"],"starter":"def nms(boxes, scores, iou_thr=0.5):\n    # greedy: keep the top score, drop boxes with IoU > thr, repeat; return kept indices\n    pass","tests":[{"args":[[[0,0,10,10],[1,1,11,11],[20,20,30,30],[0,0,10,10]],[0.9,0.8,0.7,0.95],0.5],"expect":[3,2]},{"args":[[[0,0,10,10],[1,1,11,11]],[0.9,0.8],0.5],"expect":[0]},{"args":[[[0,0,10,10],[1,1,11,11]],[0.9,0.8],0.95],"expect":[0,1]},{"args":[[[0,0,10,10],[100,100,110,110]],[0.5,0.9],0.5],"expect":[1,0]},{"args":[[],[],0.5],"expect":[]}],"solution":"import numpy as np\n\ndef nms(boxes, scores, iou_thr=0.5):\n    boxes = np.asarray(boxes, dtype=float)\n    scores = np.asarray(scores, dtype=float)\n    def iou_m(a, b):\n        lt = np.maximum(a[:, None, :2], b[None, :, :2])\n        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])\n        wh = np.maximum(0.0, rb - lt)\n        inter = wh[..., 0] * wh[..., 1]\n        area = lambda x: np.maximum(0.0, x[:, 2] - x[:, 0]) * np.maximum(0.0, x[:, 3] - x[:, 1])\n        union = area(a)[:, None] + area(b)[None, :] - inter\n        return inter / np.maximum(union, 1e-8)\n    if boxes.size == 0:\n        return []\n    order = scores.argsort()[::-1]\n    keep = []\n    while order.size > 0:\n        i = int(order[0])\n        keep.append(i)\n        if order.size == 1:\n            break\n        ious = iou_m(boxes[i][None], boxes[order[1:]])[0]\n        order = order[1:][ious <= iou_thr]\n    return keep"}
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

**Deep:** add `class_id * (max_coordinate + 1)` to every coordinate. Boxes from different classes are now displaced into disjoint regions of the plane, so their IoU is 0 and they never suppress each other, while within-class geometry is preserved. One sort, one NMS pass, no Python-level class loop.
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

**Cross-links:** [Object Detection](#/cv/detection) · [mAP & mIoU](#/ml-coding/metrics-map-miou) · [The ML Coding Round](#/ml-coding/intro)
