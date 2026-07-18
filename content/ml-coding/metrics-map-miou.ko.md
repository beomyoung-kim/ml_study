# mAP & mIoU 밑바닥부터

> [!TIP] 이렇게 먼저 말하세요
> "mIoU는 confusion-matrix reduction입니다 — 모든 픽셀을 `bincount` 하나로 누적한 뒤, 클래스별 $\text{IoU} = \text{TP}/(\text{TP}+\text{FP}+\text{FN})$를 계산해 평균 냅니다. mAP는 클래스별입니다: detection을 score로 정렬하고, IoU로 ground truth에 greedy하게 매칭하고, precision–recall 곡선을 만들어 그 아래 넓이를 적분합니다." 매우 다른 두 파이프라인 — 어느 것이 어느 것인지 먼저 말하세요.

mIoU(semantic segmentation)와 AP/mAP(detection)를 구현합니다. 이는 지표를 논문 속 숫자로서가 아니라 *정의 그대로* 안다는 것을 증명합니다 — ablation을 발표할 때의 신뢰도 차이입니다. Box IoU는 [IoU & NMS](#/ml-coding/nms-iou)에서 정의되고, 개념적 설명은 [Evaluation Metrics](#/foundations/evaluation-metrics)에 있습니다.

## Part 1 — confusion matrix를 통한 mIoU

클래스별로 $\text{IoU}_c = \dfrac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$이고, mIoU는 클래스에 대한 평균입니다. 우아한 트릭: 모든 (gt, pred) 픽셀 쌍을 하나의 정수로 인코딩하고 `bincount`로 한 방에 $C\times C$ 행렬로 만드는 것입니다.

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

**왜 confusion matrix인가:** 이미지에 걸쳐 additive하므로, 전체 validation set에 대해 `cm`을 누적하고 마지막에 IoU를 한 번만 계산합니다. `bincount`는 모든 $HW$ 픽셀을 벡터화합니다 — 클래스별 loop가 없습니다. **복잡도:** 이미지당 $O(HW)$.

> [!WARNING] 이미지별 IoU를 평균 내지 마세요
> mIoU는 각 이미지의 IoU를 평균 내는 게 아니라 **데이터셋 전체로 집계된** TP/FP/FN으로 계산합니다. 이미지별 평균은 특정 클래스가 픽셀을 적게 차지하는 이미지에 과도한 가중치를 주며, 이것이 전형적인 보고 수치 불일치의 원인입니다.

## Part 2 — AP / mAP

AP는 한 클래스에 대한 precision–recall 곡선 아래 넓이입니다. Detection은 confidence로 순위를 매기고, 각각은 아직 매칭되지 않은 ground-truth box에 IoU $\ge$ threshold로 매칭되면 **true positive**, 아니면 **false positive**입니다.

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

precision envelope(recall이 커질 때 precision을 단조 비증가하게 만드는 것)는 "출렁임"을 제거하여, AP가 각 recall 수준에서 달성 가능한 최선의 precision을 측정하게 합니다.

### 이미지별 greedy 매칭 후 적분

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

**결정적 디테일:** 매칭은 **이미지별**이고 각 ground-truth box는 **한 번만** 차지될 수 있습니다 — 같은 객체에 대한 두 번째 detection은 false positive입니다(그래서 NMS가 중요합니다). Detection을 score로 전역 정렬하면 PR 곡선을 생성하는 순위가 나옵니다. **복잡도:** 클래스당 $O(P\log P)$ 정렬 + $O(P\cdot G)$ 매칭.

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

> [!NOTE] 프레임워크 한 줄
> 공식 평가는 `pycocotools`(COCO mAP), `torchmetrics.detection.MeanAveragePrecision`, mIoU에는 `torchmetrics`/MMSeg를 씁니다. 이들의 숫자를 재현하여 자신의 loop를 검증하세요.

## 면접관이 지켜보는 흔한 버그

- **이미지별 IoU를 평균 냄** — confusion matrix 집계 대신(위 warning 참고).
- **ground truth를 두 번 매칭** — 첫 매칭 후 used로 표시해야 함.
- **전역 vs 이미지별 매칭:** detection은 *자기 이미지 내의* ground truth에만 매칭 가능.
- **`argsort` 방향:** detection을 *내림차순* score로 순위 매김.
- **`ignore_index`** 픽셀은 confusion matrix에서 제외해야 함.

## Q&A

<details class="qa"><summary>"COCO mAP"란 정확히 무엇인가?</summary>
<div class="qa-body">

**짧게:** IoU threshold를 0.50부터 0.95까지 0.05 간격으로 두고 평균 낸 AP(그것이 주된 "AP" / mAP@[.5:.95])를, 클래스에 대해 다시 평균 낸 것이며, 101-point interpolation과 이미지당 detection 개수 상한을 씁니다.

**깊게:** VOC는 단일 IoU=0.5와 (나중에) all-point interpolation을 썼습니다. COCO는 이를 조였습니다: 열 개의 IoU threshold에서 정확성을 요구하여 대략적인 overlap이 아니라 정밀한 localization을 보상합니다. 또한 AP$_{50}$, AP$_{75}$, 크기별 AP$_S$/AP$_M$/AP$_L$, 그리고 `maxDets` 제한도 보고합니다. 101-point interpolation은 recall을 $0, 0.01, \dots, 1.0$에서 샘플링합니다. 위와 동일한 PR 기계에 threshold와 클래스 평균을 씌운 것입니다.
</div></details>

<details class="qa"><summary>왜 mIoU는 집계하며, 무엇을 숨기나?</summary>
<div class="qa-body">

**짧게:** 전체 집합에 대해 TP/FP/FN을 집계하면 이미지와 무관하게 각 클래스에 동등한 가중치를 주지만, 드문 클래스도 여전히 *평균*을 지배할 수 있는 단일 IoU를 갖습니다.

**깊게:** mIoU가 클래스별 IoU를 동등하게 평균 내기 때문에, 몇 개 이미지에만 존재하는 클래스도 여전히 점수의 $\frac1C$를 차지합니다 — tail-class 실패를 드러내는 데는 좋지만, 클래스의 픽셀이 적으면 noisy합니다. 대안: frequency-weighted IoU(각 클래스를 픽셀 수로 가중)는 tail 실패를 숨기고, pixel accuracy는 다수 클래스에 지배됩니다. mIoU와 함께 클래스별 IoU를 보고하는 것이 best practice이며, segmentation 결과를 발표할 때 면접관이 파고드는 바로 그 nuance입니다.
</div></details>

<details class="qa"><summary>mask AP는 box AP와 어떻게 다른가?</summary>
<div class="qa-body">

**짧게:** PR 파이프라인은 동일하지만, 매칭에 쓰는 IoU가 box IoU가 아니라 mask IoU(픽셀별 intersection over union)입니다.

**깊게:** instance segmentation(Mask R-CNN, COCO `segm` metric)은 TP vs FP를 결정할 때 geometric box-IoU를 `(pred_mask & gt_mask).sum() / (pred_mask | gt_mask).sum()`로 바꾸며, 나머지 — score 순위, one-gt-one-match, IoU-threshold별 평균 — 는 그대로입니다. 그래서 box-AP loop를 이해하면 segmentation 평가로 바로 전이됩니다.
</div></details>

### Follow-ups
- **Precision–recall vs ROC?** 불균형 detection(positive가 적음)에는 PR이 맞는 곡선입니다. negative가 지배할 때 ROC의 FPR은 오해를 부릅니다.
- **왜 나쁜 NMS가 mAP를 해치나?** 한 객체의 중복 detection이 false positive가 되어 모든 recall 수준에서 precision을 끌어내립니다.
- **배치에 걸쳐 mIoU를 벡터화?** confusion matrix를 합산; `bincount`가 이미 모든 픽셀을 한 번에 처리합니다.
- **Panoptic quality (PQ)?** segmentation IoU와 recognition(TP/FP/FN)을 하나의 숫자로 결합하여 panoptic segmentation에 사용.

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
