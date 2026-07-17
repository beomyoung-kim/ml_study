# mAP & mIoU From Scratch

> [!TIP] Say this first
> "mIoU is a confusion-matrix reduction — accumulate every pixel with one `bincount`, then per class $\text{IoU} = \text{TP}/(\text{TP}+\text{FP}+\text{FN})$ and average. mAP is per-class: sort detections by score, greedily match to ground truth by IoU, build the precision–recall curve, and integrate its area." Two very different pipelines — say which is which up front.

Implement mIoU (semantic segmentation) and AP/mAP (detection). This proves you know the metrics *by definition*, not just as numbers in a paper — the credibility difference when you present ablations. Box IoU is defined in [IoU & NMS](#/ml-coding/nms-iou); the conceptual treatment is in [Evaluation Metrics](#/foundations/evaluation-metrics).

## Part 1 — mIoU via a confusion matrix

Per class, $\text{IoU}_c = \dfrac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$, and mIoU is the mean over classes. The elegant trick: encode every (gt, pred) pixel pair as a single integer and `bincount` it into a $C\times C$ matrix in one shot.

```python
import numpy as np


def confusion_matrix(pred, target, num_classes, ignore_index=255):
    """pred,target: int label maps. Rows=gt, cols=pred. -> (C,C)."""
    pred, target = pred.reshape(-1), target.reshape(-1)
    keep = target != ignore_index                  # drop void/unlabeled pixels
    pred, target = pred[keep], target[keep]
    # each pair (g,p) -> unique index g*C + p; count all at once
    cm = np.bincount(target * num_classes + pred,
                     minlength=num_classes ** 2).reshape(num_classes, -1)
    return cm.astype(np.float64)


def mean_iou(pred, target, num_classes, ignore_index=255):
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp                            # predicted c, gt not c
    fn = cm.sum(1) - tp                            # gt c, predicted not c
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1e-8), np.nan)
    return float(np.nanmean(iou))                  # ignore classes absent everywhere
```

**Why a confusion matrix:** it's additive across images, so you accumulate `cm` over the whole validation set and compute IoU once at the end. `bincount` vectorizes all $HW$ pixels — no per-class loop. **Complexity:** $O(HW)$ per image.

> [!WARNING] Don't average per-image IoU
> mIoU is computed from **dataset-aggregated** TP/FP/FN, not by averaging each image's IoU. Per-image averaging over-weights images where a class occupies few pixels and is the classic reported-number mismatch.

## Part 2 — AP / mAP

AP is the area under the precision–recall curve for one class. Detections are ranked by confidence; each is a **true positive** if it matches an unmatched ground-truth box with IoU $\ge$ threshold, else a **false positive**.

```python
def box_iou(box, boxes):
    """box:(4,)  boxes:(N,4) xyxy -> (N,)."""
    if boxes.size == 0:
        return np.zeros(0)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, 0] * wh[:, 1]
    a1 = np.prod(np.maximum(0.0, box[2:] - box[:2]))
    a2 = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    return inter / np.maximum(a1 + a2 - inter, 1e-8)


def average_precision(recall, precision):
    """VOC 2010+ all-point interpolation: area under the PR envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):           # make precision monotonic
        mpre[i - 1] = max(mpre[i - 1], mpre[i])     #   (envelope from the right)
    i = np.where(mrec[1:] != mrec[:-1])[0]          # recall steps
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
```

The precision envelope (making precision monotonically non-increasing as recall grows) removes the "wiggles" so AP measures the best achievable precision at each recall level.

### Per-image greedy matching, then integrate

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

**The crucial detail:** matching is **per image** and each ground-truth box can be claimed **once** — a second detection of the same object is a false positive (which is why NMS matters). Sorting detections globally by score gives the ranking that generates the PR curve. **Complexity:** $O(P\log P)$ sort + $O(P\cdot G)$ matching per class.

## Sanity check

```python
if __name__ == "__main__":
    pred = np.array([[0, 0, 1], [1, 1, 1]]); tgt = np.array([[0, 1, 1], [1, 1, 0]])
    assert 0 < mean_iou(pred, tgt, 2, ignore_index=-1) <= 1

    gt = [{"boxes": np.array([[0, 0, 10, 10.]]), "labels": np.array([0])}]
    perfect = [{"boxes": np.array([[0, 0, 10, 10.]]),
                "scores": np.array([0.9]), "labels": np.array([0])}]
    assert abs(mean_average_precision(perfect, gt, 1) - 1.0) < 1e-6
    with_fp = [{"boxes": np.array([[0, 0, 10, 10.], [50, 50, 60, 60.]]),
                "scores": np.array([0.9, 0.8]), "labels": np.array([0, 0])}]
    assert mean_average_precision(with_fp, gt, 1) < 1.0   # FP lowers AP
    print("mIoU/mAP sanity OK")
```

> [!NOTE] Framework one-liner
> Official evaluation uses `pycocotools` (COCO mAP), `torchmetrics.detection.MeanAveragePrecision`, and `torchmetrics`/MMSeg for mIoU. Reproduce their number to validate your loop.

## Common bugs interviewers watch for

- **Averaging per-image IoU** instead of aggregating the confusion matrix (see warning).
- **Matching a ground truth twice** — must mark it used after the first match.
- **Global vs per-image matching:** a detection can only match ground truth *in its own image*.
- **`argsort` direction:** rank detections by *descending* score.
- **`ignore_index`** pixels must be dropped from the confusion matrix.

## Q&A

<details class="qa"><summary>What exactly is "COCO mAP"?</summary>
<div class="qa-body">

**Short:** AP averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05 (that's the primary "AP" / mAP@[.5:.95]), then averaged over classes, with 101-point interpolation and caps on detections per image.

**Deep:** VOC used a single IoU=0.5 and (later) all-point interpolation. COCO tightened this: requiring correctness across ten IoU thresholds rewards precise localization, not just rough overlap. It also reports AP$_{50}$, AP$_{75}$, and size-stratified AP$_S$/AP$_M$/AP$_L$, plus `maxDets` limits. The 101-point interpolation samples recall at $0, 0.01, \dots, 1.0$. Same PR machinery as above, wrapped in threshold and class averaging.
</div></details>

<details class="qa"><summary>Why is mIoU aggregated, and what does it hide?</summary>
<div class="qa-body">

**Short:** aggregating TP/FP/FN over the whole set gives each class equal weight regardless of image, but a rare class still gets a single IoU that can dominate the *mean*.

**Deep:** because mIoU averages per-class IoUs equally, a class present in only a few images still counts as $\frac1C$ of the score — good for surfacing tail-class failure, but noisy when a class has few pixels. Alternatives: frequency-weighted IoU (weight each class by its pixel count) hides tail failures; pixel accuracy is dominated by the majority class. Reporting per-class IoU alongside mIoU is best practice, and it's exactly the kind of nuance interviewers probe when you present segmentation results.
</div></details>

<details class="qa"><summary>How does mask AP differ from box AP?</summary>
<div class="qa-body">

**Short:** identical PR pipeline, but the IoU used for matching is mask IoU (pixelwise intersection over union) instead of box IoU.

**Deep:** instance segmentation (Mask R-CNN, COCO `segm` metric) swaps the geometric box-IoU for `(pred_mask & gt_mask).sum() / (pred_mask | gt_mask).sum()` when deciding TP vs FP, everything else — score ranking, one-gt-one-match, per-IoU-threshold averaging — stays the same. This is why understanding the box-AP loop transfers directly to segmentation evaluation.
</div></details>

### Follow-ups
- **Precision–recall vs ROC?** PR is the right curve for imbalanced detection (few positives); ROC's FPR is misleading when negatives dominate.
- **Why does poor NMS hurt mAP?** Duplicate detections of one object become false positives, dragging precision down at every recall level.
- **Vectorize mIoU across a batch?** Sum confusion matrices; `bincount` already handles all pixels at once.
- **Panoptic quality (PQ)?** Combines segmentation IoU and recognition (TP/FP/FN) into one number for panoptic segmentation.

## Cheat-sheet

| Item | Value |
| --- | --- |
| IoU (class) | $\text{TP}/(\text{TP}+\text{FP}+\text{FN})$ |
| mIoU | mean of per-class IoU, **dataset-aggregated** |
| Confusion trick | `bincount(gt*C + pred)` → $C\times C$ |
| AP | area under PR curve (VOC all-point / COCO 101-pt) |
| TP rule | IoU $\ge$ thr **and** gt unmatched, per image |
| Duplicate detection | false positive → motivates NMS |
| COCO mAP | avg over IoU 0.50:0.05:0.95, per class |
| Mask AP | same pipeline, mask IoU for matching |

**Cross-links:** [IoU & NMS](#/ml-coding/nms-iou) · [Evaluation Metrics](#/foundations/evaluation-metrics) · [Object Detection](#/cv/detection) · [Segmentation](#/cv/segmentation) · [The ML Coding Round](#/ml-coding/intro)
