# mAP & mIoU From Scratch

> [!NOTE] What and why
> A **metric** compresses how well a model performs into a number. Here you will implement the two most common vision metrics: **mIoU** for segmentation—how well pixels overlap—and **mAP** for detection—how well boxes are found. This is much easier if you first review precision and recall in [Evaluation Metrics](#/foundations/evaluation-metrics).

**Both metrics in one sentence each:**

- **mIoU** measures, for every class, how much the predicted and ground-truth regions overlap—their IoU—and averages across classes. It evaluates segmentation quality.
- **mAP** ranks detections by confidence, draws a precision–recall curve, computes the **area under that curve, AP**, and averages across classes. It evaluates detection quality.

<figure>
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- IoU illustration (left) -->
  <text x="110" y="20" text-anchor="middle" font-weight="700" fill="#0ea5e9">IoU = overlap ÷ union</text>
  <rect x="45" y="40" width="90" height="90" rx="4" fill="rgba(14,165,233,.18)" stroke="#0ea5e9" stroke-width="1.6"/>
  <rect x="95" y="65" width="90" height="90" rx="4" fill="rgba(224,83,63,.18)" stroke="#e0533f" stroke-width="1.6"/>
  <rect x="95" y="65" width="40" height="65" fill="rgba(18,161,80,.45)"/>
  <text x="60" y="55" fill="#0ea5e9">ground truth</text><text x="165" y="150" fill="#e0533f">prediction</text>
  <text x="115" y="180" text-anchor="middle" fill="#12a150">intersection ÷ union = IoU</text>
  <!-- PR curve (right) -->
  <text x="470" y="20" text-anchor="middle" font-weight="700" fill="#6366f1">AP = area under the PR curve</text>
  <line x1="360" y1="160" x2="600" y2="160" stroke="#98a3b2" stroke-width="1.5"/>
  <line x1="360" y1="40" x2="360" y2="160" stroke="#98a3b2" stroke-width="1.5"/>
  <text x="600" y="178" text-anchor="end" fill="#98a3b2">recall →</text>
  <text x="352" y="45" text-anchor="end" fill="#98a3b2">precision</text>
  <path d="M360 55 L420 58 L420 95 L500 100 L500 135 L580 140 L580 160 L360 160 Z" fill="rgba(99,102,241,.22)"/>
  <path d="M360 55 L420 58 L420 95 L500 100 L500 135 L580 140" fill="none" stroke="#6366f1" stroke-width="2.2"/>
  <text x="480" y="120" text-anchor="middle" fill="#6366f1">area = AP</text>
</svg>
<figcaption>Left: IoU (intersection over union) measures the overlap between two regions from 0 to 1. Right: detection AP is the area under the precision–recall curve. mIoU and mAP average these quantities across classes.</figcaption>
</figure>

Implement these metrics from their definitions rather than treating them as paper-table numbers, and your experimental reports become much more credible. Box IoU itself is also covered in [IoU & NMS](#/ml-coding/nms-iou).

> [!TIP] Interview one-liner
> "mIoU is a confusion-matrix reduction: accumulate all pixels with one `bincount`, compute $\text{TP}/(\text{TP}+\text{FP}+\text{FN})$ per class, then average. mAP ranks detections by score, greedily matches them to ground truth with IoU, builds a PR curve, and integrates its area." These are very different pipelines; distinguish them first.

## Practice—implement, run, and test it

> [!TIP] How to use this section
> Every problem below has a **live Python editor with NumPy preloaded**. Write your solution and press **▶ Run tests** to see which cases pass. Open the reference **Solution** if you are stuck, but try first. The first run downloads a small Python runtime and NumPy (about 15 MB); later runs are immediate. Inputs are small and deterministic, and outputs are checked with `numpy.allclose`. If NumPy is new to you, read the [NumPy Primer](#/ml-coding/numpy-primer) first.

Proceed in order: build the segmentation metric in Part 1, then the detection-AP machinery in Part 2.

### Part 1—mIoU through a confusion matrix

A **confusion matrix** is a $C\times C$ table counting pixels whose ground truth is A but prediction is B; rows are ground truth and columns are predictions. The diagonal contains correct predictions, or TPs, and the rest are errors. Per class,
$\text{IoU}_c = \dfrac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$, and mIoU is the class mean.

An elegant trick encodes every `(gt, pred)` pixel pair as the integer `gt*C + pred`, then uses `bincount` to construct the $C\times C$ matrix in one operation—no loop over classes, following the [broadcasting/vectorization](#/ml-coding/numpy-primer) mindset. Exclude pixels with `ignore_index`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"confusion_matrix","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef confusion_matrix(pred, target, num_classes, ignore_index=255):\n    # flatten both, drop ignore_index pixels, then bincount(gt*C + pred) -> (C,C)\n    pass","tests":[{"args":[[[0,0,1],[1,1,1]],[[0,1,1],[1,1,0]],2,-1],"expect":[[1,1],[1,3]]},{"args":[[[0,1],[1,0]],[[0,255],[1,255]],2,255],"expect":[[1,0],[0,1]]},{"args":[[[0,0],[1,2]],[[0,1],[1,2]],3],"expect":[[1,0,0],[1,1,0],[0,0,1]]}],"solution":"import numpy as np\n\ndef confusion_matrix(pred, target, num_classes, ignore_index=255):\n    pred = np.asarray(pred, dtype=np.int64); target = np.asarray(target, dtype=np.int64)\n    if pred.shape != target.shape or num_classes <= 0:\n        raise ValueError('pred/target shapes must match and num_classes must be positive')\n    pred, target = pred.reshape(-1), target.reshape(-1)\n    keep = target != ignore_index\n    pred, target = pred[keep], target[keep]\n    if np.any((target < 0) | (target >= num_classes) | (pred < 0) | (pred >= num_classes)):\n        raise ValueError('class index out of range')\n    cm = np.bincount(target * num_classes + pred, minlength=num_classes ** 2).reshape(num_classes, -1)\n    return cm.astype(np.float64)"}
</script>
</div>

Next compute mIoU from the confusion matrix: per class, $\text{TP}=\text{diagonal}$, $\text{FP}=\text{column sum}-\text{TP}$, and $\text{FN}=\text{row sum}-\text{TP}$. Average $\text{TP}/(\text{TP}+\text{FP}+\text{FN})$ over classes that are present.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"mean_iou","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef mean_iou(pred, target, num_classes, ignore_index=255):\n    # build the confusion matrix, then per-class TP/(TP+FP+FN); mean over present classes\n    pass","tests":[{"args":[[[0,0,1],[1,1,1]],[[0,1,1],[1,1,0]],2,-1],"expect":0.4666666666666667},{"args":[[[0,1],[1,0]],[[0,1],[1,0]],2,-1],"expect":1.0},{"args":[[[0,0]],[[0,0]],2,-1],"expect":1.0}],"solution":"import numpy as np\n\ndef confusion_matrix(pred, target, num_classes, ignore_index=255):\n    pred = np.asarray(pred, dtype=np.int64); target = np.asarray(target, dtype=np.int64)\n    if pred.shape != target.shape or num_classes <= 0:\n        raise ValueError('pred/target shapes must match and num_classes must be positive')\n    pred, target = pred.reshape(-1), target.reshape(-1)\n    keep = target != ignore_index\n    pred, target = pred[keep], target[keep]\n    if np.any((target < 0) | (target >= num_classes) | (pred < 0) | (pred >= num_classes)):\n        raise ValueError('class index out of range')\n    return np.bincount(target * num_classes + pred, minlength=num_classes ** 2).reshape(num_classes, -1).astype(np.float64)\n\ndef mean_iou(pred, target, num_classes, ignore_index=255):\n    cm = confusion_matrix(pred, target, num_classes, ignore_index)\n    tp = np.diag(cm)\n    fp = cm.sum(0) - tp\n    fn = cm.sum(1) - tp\n    denom = tp + fp + fn\n    iou = np.where(denom > 0, tp / np.maximum(denom, 1e-8), np.nan)\n    return float(np.nanmean(iou))"}
</script>
</div>

**Why a confusion matrix?** It is additive across images, so you can accumulate `cm` over the entire validation set and compute IoU once at the end. `bincount` vectorizes all $HW$ pixels with no class loop. **Complexity:** $O(HW)$ per image.

> [!WARNING] Check the benchmark's aggregation contract first
> Semantic-segmentation benchmarks in the MMSeg family usually sum one confusion matrix over the entire dataset before computing class IoUs. Some medical or video benchmarks instead specify an average of per-image or per-case Dice/IoU values. Those are different metrics. Do not mix them based on the name alone; aggregate at exactly the unit used by the official evaluator.

### Part 2—AP / mAP

AP, or Average Precision, is the area under a precision–recall curve for one class. Rank detections by confidence. A detection is a **true positive** if it matches an as-yet-unmatched ground-truth box with IoU at least the threshold; otherwise it is a **false positive**.

AP matching requires box IoU. The canonical [IoU & NMS implementation](#/ml-coding/nms-iou) contains the full treatment—vectorized `iou_matrix`, edge cases, and greedy `nms`. Here we keep only a **minimal inline version** so the mAP pipeline remains self-contained. It compares one box with `N` boxes and returns `N` overlaps.

```python
import numpy as np

def box_iou(box, boxes):
    """box:(4,) vs boxes:(N,4) xyxy -> (N,). Minimal inline copy; full treatment in nms-iou."""
    box = np.asarray(box, float); boxes = np.asarray(boxes, float)
    if boxes.size == 0:
        return np.zeros(0)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, 0] * wh[:, 1]
    a1 = np.prod(np.maximum(0.0, box[2:] - box[:2]))
    a2 = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    return inter / np.maximum(a1 + a2 - inter, 1e-8)
```

Next, VOC all-point AP: pad the PR curve, make precision monotone from the right—the **precision envelope**—then sum the area of every recall step.

<figure>
<svg viewBox="0 0 640 190" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="160" y="18" text-anchor="middle" fill="#98a3b2">raw PR (jagged)</text>
  <line x1="40" y1="150" x2="300" y2="150" stroke="#98a3b2"/><line x1="40" y1="40" x2="40" y2="150" stroke="#98a3b2"/>
  <polyline points="40,55 100,65 100,110 170,95 170,130 250,120 250,145" fill="none" stroke="#e0533f" stroke-width="2"/>
  <text x="480" y="18" text-anchor="middle" fill="#98a3b2">right-to-left maximum (envelope)</text>
  <line x1="360" y1="150" x2="620" y2="150" stroke="#98a3b2"/><line x1="360" y1="40" x2="360" y2="150" stroke="#98a3b2"/>
  <polyline points="360,55 420,55 420,95 490,95 490,120 570,120 570,145" fill="none" stroke="#12a150" stroke-width="2.4"/>
  <text x="490" y="175" text-anchor="middle" fill="#12a150">best attainable precision at each recall</text>
</svg>
<figcaption>The precision envelope makes precision monotonically non-increasing with recall and removes local oscillations. AP then measures the best attainable precision at each recall level.</figcaption>
</figure>

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"average_precision","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef average_precision(recall, precision):\n    # VOC all-point: pad the curve, make precision monotonic from the right, sum step areas\n    pass","tests":[{"args":[[0.5,1.0],[1.0,1.0]],"expect":1.0},{"args":[[1.0],[1.0]],"expect":1.0},{"args":[[0.5,1.0],[1.0,0.5]],"expect":0.75},{"args":[[0.33,0.66,1.0],[1.0,0.5,0.6]],"expect":0.732}],"solution":"import numpy as np\n\ndef average_precision(recall, precision):\n    mrec = np.concatenate(([0.0], recall, [1.0]))\n    mpre = np.concatenate(([0.0], precision, [0.0]))\n    for i in range(mpre.size - 1, 0, -1):\n        mpre[i - 1] = max(mpre[i - 1], mpre[i])\n    i = np.where(mrec[1:] != mrec[:-1])[0]\n    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))"}
</script>
</div>

#### Greedy matching per image, then integration

The final piece labels detections as TP or FP. It is long enough to keep as static reference code. Remember the rule: **globally sort detections by descending score → match each detection only to ground truth in its own image → allow each ground-truth box to be claimed once → form cumulative precision and recall → pass them to `average_precision`.**

```python
def mean_average_precision(preds, gts, num_classes, iou_thr=0.5):
    """
    preds[i] = {'boxes':(P,4), 'scores':(P,), 'labels':(P,)}
    gts[i]   = {'boxes':(G,4), 'labels':(G,)}
    """
    aps = []
    for c in range(num_classes):
        dets, gt_boxes, gt_used, npos = [], {}, {}, 0
        for img, (pr, gt) in enumerate(zip(preds, gts)):
            gmask = gt["labels"] == c
            gt_boxes[img] = gt["boxes"][gmask]
            gt_used[img] = np.zeros(int(gmask.sum()), bool)
            npos += int(gmask.sum())
            pmask = pr["labels"] == c
            for b, s in zip(pr["boxes"][pmask], pr["scores"][pmask]):
                dets.append((float(s), img, b))
        if npos == 0:
            continue
        dets.sort(key=lambda d: -d[0])              # global rank by score
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for i, (_, img, b) in enumerate(dets):
            gb = gt_boxes[img]
            if len(gb) == 0:
                fp[i] = 1; continue
            ious = box_iou(b, gb)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not gt_used[img][j]:
                tp[i] = 1; gt_used[img][j] = True   # each gt matched once
            else:
                fp[i] = 1                           # low IoU or duplicate
        tpc, fpc = np.cumsum(tp), np.cumsum(fp)
        recall = tpc / npos
        precision = tpc / np.maximum(tpc + fpc, 1e-8)
        aps.append(average_precision(recall, precision))
    return float(np.mean(aps)) if aps else 0.0
```

**Critical detail:** matching is **per image**, and each ground-truth box can be claimed **only once**. A second detection of the same object is a false positive, which is why [NMS](#/ml-coding/nms-iou) matters. **Complexity:** $O(P\log P)$ sorting plus $O(P\cdot G)$ matching per class.

One more trap: a low-score false positive added after recall has already reached 1 may not lower interpolated AP. A test that assumes "adding any FP must reduce AP" is therefore wrong. Put the FP before a TP, or place it while recall is still incomplete, to test ranking behavior.

## Sanity check

```python
if __name__ == "__main__":
    pred = np.array([[0, 0, 1], [1, 1, 1]]); tgt = np.array([[0, 1, 1], [1, 1, 0]])
    assert 0 < mean_iou(pred, tgt, 2, ignore_index=-1) <= 1

    gt = [{"boxes": np.array([[0, 0, 10, 10.]]), "labels": np.array([0])}]
    perfect = [{"boxes": np.array([[0, 0, 10, 10.]]),
                "scores": np.array([0.9]), "labels": np.array([0])}]
    assert abs(mean_average_precision(perfect, gt, 1) - 1.0) < 1e-6
    with_early_fp = [{"boxes": np.array([[0, 0, 10, 10.], [50, 50, 60, 60.]]),
                      "scores": np.array([0.8, 0.9]), "labels": np.array([0, 0])}]
    assert mean_average_precision(with_early_fp, gt, 1) < 1.0
    # A low-score FP after full recall may not reduce interpolated AP.
    print("mIoU/mAP sanity OK")
```

> [!NOTE] Framework one-liner
> Official evaluation uses `pycocotools` for COCO mAP, `torchmetrics.detection.MeanAveragePrecision`, and `torchmetrics` or MMSeg for mIoU. Reproduce their numbers to validate your own loop.

## Common bugs interviewers watch for

- **Averaging per-image IoU** instead of aggregating the confusion matrix; see the warning above.
- **Matching a ground truth twice:** mark it used after the first match.
- **Global vs. per-image matching:** a detection may match only ground truth from *its own image*.
- **Wrong `argsort` direction:** rank detections by *descending* score.
- **`ignore_index`:** exclude ignored pixels from the confusion matrix.

## Q&A

<details class="qa"><summary>What exactly is "COCO mAP"?</summary>
<div class="qa-body">

**Short:** The primary COCO AP, or mAP@[.5:.95], averages AP over IoU thresholds 0.50 through 0.95 in steps of 0.05, then across classes. It uses 101-point interpolation and a per-image cap on detections.

**Deep:** VOC used one IoU threshold, 0.5, and later all-point interpolation. COCO tightened the criterion: ten IoU thresholds reward precise localization rather than rough overlap. It also reports AP$_{50}$, AP$_{75}$, size-specific AP$_S$/AP$_M$/AP$_L$, and `maxDets` limits. Its 101-point interpolation samples recall at $0, 0.01, \dots, 1.0$. It is the same PR machinery with threshold and class averaging layered on top.
</div></details>

<details class="qa"><summary>Why aggregate mIoU, and what does it hide?</summary>
<div class="qa-body">

**Short:** Aggregating TP/FP/FN over the dataset gives every class equal weight regardless of image count, but a rare class still receives one full class IoU and can influence the mean strongly.

**Deep:** Because mIoU averages class IoUs equally, a class present in only a few images still owns $\frac1C$ of the score. That exposes tail-class failure but can be noisy when the class has few pixels. Frequency-weighted IoU weights classes by pixel count and can hide tail failures; pixel accuracy is dominated by majority classes. Best practice is to report per-class IoU alongside mIoU.
</div></details>

<details class="qa"><summary>How does mask AP differ from box AP?</summary>
<div class="qa-body">

**Short:** The PR pipeline is identical, but matching uses mask IoU—pixelwise intersection over union—instead of box IoU.

**Deep:** Instance-segmentation evaluation, such as Mask R-CNN under COCO's `segm` metric, replaces geometric box IoU with `(pred_mask & gt_mask).sum() / (pred_mask | gt_mask).sum()` when deciding TP versus FP. Everything else—score ranking, one-ground-truth-one-match, and averaging over IoU thresholds—stays the same. Understanding the box-AP loop transfers directly to segmentation evaluation.
</div></details>

### Follow-ups

- **Precision–recall vs. ROC?** PR is the right curve for imbalanced detection with few positives. When negatives dominate, ROC's FPR can be misleading.
- **Why does bad NMS hurt mAP?** Duplicate detections of one object become false positives and lower precision across recall levels.
- **Vectorize mIoU across a batch?** Sum confusion matrices; `bincount` already processes every pixel at once.
- **Panoptic quality (PQ)?** It combines segmentation IoU and recognition errors—TP/FP/FN—in one number. See [Segmentation](#/cv/segmentation).

## Cheat-sheet

| Item | Value |
| --- | --- |
| IoU (class) | $\text{TP}/(\text{TP}+\text{FP}+\text{FN})$ |
| mIoU | mean class IoU, **aggregated over the full dataset** |
| Confusion trick | `bincount(gt*C + pred)` → $C\times C$ |
| AP | area under the PR curve; VOC all-point / COCO 101-point |
| TP rule | IoU $\ge$ threshold **and** unmatched ground truth, per image |
| Duplicate detection | false positive → why NMS is necessary |
| COCO mAP | average over IoU 0.50:0.05:0.95, then classes |
| Mask AP | same pipeline, mask IoU for matching |

**Next:** [IoU & NMS](#/ml-coding/nms-iou) · [Evaluation Metrics](#/foundations/evaluation-metrics) · [Object Detection](#/cv/detection) · [Segmentation](#/cv/segmentation) · [ML Coding Round](#/ml-coding/intro)
