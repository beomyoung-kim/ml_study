# IoU & Non-Max Suppression

> [!TIP] 이것부터 말하세요
> "IoU는 intersection over union이고, 전체 트릭은 top-left들의 `max`와 bottom-right들의 `min`을 broadcasting으로 계산해서 intersection 사각형을 구한 다음, 음수를 0으로 clamp하는 것입니다. NMS는 greedy 루프입니다: 최고 score를 keep하고, 그것과 너무 많이 겹치는 것을 전부 drop하고, 반복합니다." 그다음 코딩하세요.

모든 detection/segmentation 루프의 기본 요소입니다. Box는 연속 좌표에서 `[x1, y1, x2, y2]`이므로 area는 $(x_2-x_1)(y_2-y_1)$입니다(`+1` 없음). 인터뷰어는 알고리즘 자체보다 **vectorization**과 **edge case**(zero-area box, 완전 포함, 빈 입력)를 훨씬 더 지켜봅니다.

## 수학

두 box $A, B$에 대해:

$$
\text{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A\cap B|}{|A| + |B| - |A\cap B|}
$$

Intersection 사각형은 corner $\big(\max(x_1^A,x_1^B),\,\max(y_1^A,y_1^B)\big)$와 $\big(\min(x_2^A,x_2^B),\,\min(y_2^A,y_2^B)\big)$를 갖습니다; 어느 한 변이라도 음수면 box가 겹치지 않으므로 $0$으로 clamp 합니다.

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

순서대로 만드세요: area 먼저(degenerate box를 방어합니다), 그다음 broadcast된 IoU grid, 그 위에 얹히는 greedy 루프.

### 1. Box area <span class="badge badge-easy">Easy</span>

각 `[x1, y1, x2, y2]` box의 area, 음의 변 길이를 0으로 clamp하여 degenerate box는 `0`이 되게 합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"box_area","packages":["numpy"],"approx":true,"starter":"def box_area(boxes):\n    # (N,4) xyxy boxes -> (N,) areas; clamp negative side lengths to 0\n    pass","tests":[{"args":[[[0,0,10,10],[1,1,4,5],[5,5,2,2]]],"expect":[100.0,12.0,0.0]},{"args":[[[0,0,2,3],[0,0,0,9]]],"expect":[6.0,0.0]},{"args":[[[2,2,5,7]]],"expect":[15.0]}],"solution":"import numpy as np\n\ndef box_area(boxes):\n    boxes = np.asarray(boxes, dtype=float)\n    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])"}
</script>
</div>

*O(N), 완전 vectorized.* **Pitfall:** `+1` 없음 — 이건 연속 `xyxy` 좌표이지 VOC integer pixel이 아닙니다.

### 2. Pairwise IoU by broadcasting <span class="badge badge-med">Medium</span>

전체 $(N,M)$ IoU grid: top-left들의 `max`와 bottom-right들의 `min`을 broadcast하고, clamp한 다음, union으로 나눕니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"iou_matrix","packages":["numpy"],"approx":true,"starter":"def iou_matrix(a, b):\n    # a:(N,4), b:(M,4) -> (N,M); broadcast max of top-lefts, min of bottom-rights, clamp\n    pass","tests":[{"args":[[[0,0,10,10]],[[0,0,10,10]]],"expect":[[1.0]]},{"args":[[[0,0,10,10]],[[1,1,11,11]]],"expect":[[0.680672268907563]]},{"args":[[[0,0,10,10],[20,20,30,30]],[[0,0,10,10],[5,5,15,15]]],"expect":[[1.0,0.14285714285714285],[0.0,0.0]]},{"args":[[[0,0,2,2]],[[10,10,12,12]]],"expect":[[0.0]]}],"solution":"import numpy as np\n\ndef iou_matrix(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    def area(x):\n        return np.maximum(0.0, x[:, 2] - x[:, 0]) * np.maximum(0.0, x[:, 3] - x[:, 1])\n    lt = np.maximum(a[:, None, :2], b[None, :, :2])\n    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])\n    wh = np.maximum(0.0, rb - lt)\n    inter = wh[..., 0] * wh[..., 1]\n    union = area(a)[:, None] + area(b)[None, :] - inter\n    return inter / np.maximum(union, 1e-8)"}
</script>
</div>

`[:, None]` / `[None, :]` axis 삽입이 전부입니다: $(N,1,\cdot)$과 $(1,M,\cdot)$ tensor를 정렬해서 broadcasting이 루프 없이 완전한 $(N,M)$ grid를 만들게 합니다. **Complexity:** time과 memory 모두 $O(NM)$.

### 3. Greedy NMS <span class="badge badge-med">Medium</span>

score로 정렬하고, 최고 box를 keep한 다음, 그것과의 IoU가 threshold를 넘는 생존자를 전부 drop, 반복합니다. keep된 index를 감소하는 score 순으로 반환합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"nms","packages":["numpy"],"starter":"def nms(boxes, scores, iou_thr=0.5):\n    # greedy: keep the top score, drop boxes with IoU > thr, repeat; return kept indices\n    pass","tests":[{"args":[[[0,0,10,10],[1,1,11,11],[20,20,30,30],[0,0,10,10]],[0.9,0.8,0.7,0.95],0.5],"expect":[3,2]},{"args":[[[0,0,10,10],[1,1,11,11]],[0.9,0.8],0.5],"expect":[0]},{"args":[[[0,0,10,10],[1,1,11,11]],[0.9,0.8],0.95],"expect":[0,1]},{"args":[[[0,0,10,10],[100,100,110,110]],[0.5,0.9],0.5],"expect":[1,0]},{"args":[[],[],0.5],"expect":[]}],"solution":"import numpy as np\n\ndef nms(boxes, scores, iou_thr=0.5):\n    boxes = np.asarray(boxes, dtype=float)\n    scores = np.asarray(scores, dtype=float)\n    def iou_m(a, b):\n        lt = np.maximum(a[:, None, :2], b[None, :, :2])\n        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])\n        wh = np.maximum(0.0, rb - lt)\n        inter = wh[..., 0] * wh[..., 1]\n        area = lambda x: np.maximum(0.0, x[:, 2] - x[:, 0]) * np.maximum(0.0, x[:, 3] - x[:, 1])\n        union = area(a)[:, None] + area(b)[None, :] - inter\n        return inter / np.maximum(union, 1e-8)\n    if boxes.size == 0:\n        return []\n    order = scores.argsort()[::-1]\n    keep = []\n    while order.size > 0:\n        i = int(order[0])\n        keep.append(i)\n        if order.size == 1:\n            break\n        ious = iou_m(boxes[i][None], boxes[order[1:]])[0]\n        order = order[1:][ious <= iou_thr]\n    return keep"}
</script>
</div>

루프는 **본질적으로 순차적**입니다 — keep된 각 box가 어떤 후보가 살아남을지를 바꾸니까요 — 하지만 각 스텝은 vectorized되어 있습니다. **Complexity:** 최악의 경우 $O(N^2)$; 각 iteration은 생존자에 대한 broadcast된 IoU 하나입니다.

## Soft-NMS (흔한 후속 질문)

겹치는 box를 hard-delete하는 대신, score를 *decay*합니다 — 붐비는 scene에서 더 나은 recall (Bodla et al., 2017).

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

> [!NOTE] Framework 한 줄
> Production에서는 class-aware suppression을 위해 `torchvision.ops.box_iou`, `torchvision.ops.nms(boxes, scores, iou_thr)`, `torchvision.ops.batched_nms(boxes, scores, idxs, iou_thr)`를 씁니다.

## 인터뷰어가 주시하는 흔한 버그

<div class="proscons"><div><div class="pros-t">Do</div>

- `argsort()[::-1]`을 쓰세요 — `argsort`는 **오름차순**입니다.
- `rb - lt`를 `0`으로 clamp하세요(겹치지 않는 box는 음의 width를 냅니다).
- union을 `np.maximum(union, eps)`로 방어하세요.
- Soft-NMS에서 mutate 전에 score를 `.copy()`하세요.

</div><div><div class="cons-t">Don't</div>

- width에 `+1`을 더하지 마세요 — 그건 옛 VOC integer-pixel 관례이지, `xyxy` 연속 좌표가 아닙니다.
- NMS 루프 안에서 전체 $N\times N$ 행렬을 materialize하지 마세요(winner-vs-rest만 계산하세요).
- 빈 입력 guard(`boxes.size == 0`)를 잊지 마세요.
- class-aware와 class-agnostic NMS를 헷갈리지 마세요.

</div></div>

## Q&A

<details class="qa"><summary>per-class Python 루프 없이 NMS를 class-aware하게 만들려면?</summary>
<div class="qa-body">

**짧게:** 각 box의 좌표를 큰 per-class 상수만큼 offset해서 서로 다른 class의 box가 절대 겹칠 수 없게 만든 다음, global NMS를 한 번 돌리세요 — 그게 바로 `batched_nms`입니다.

**깊게:** 모든 좌표에 `class_id * (max_coordinate + 1)`을 더하세요. 서로 다른 class의 box는 이제 평면의 서로소 영역으로 displace되어 IoU가 0이 되고 서로를 suppress하지 않으면서, class 내부 geometry는 보존됩니다. sort 한 번, NMS pass 한 번, Python-level class 루프 없음.
</div></details>

<details class="qa"><summary>Soft-NMS vs hard NMS — 언제, 왜?</summary>
<div class="qa-body">

**짧게:** Soft-NMS는 겹치는 score를 삭제하는 대신 decay하여, 작은 latency 비용으로 붐비는 scene에서 recall을 개선합니다.

**깊게:** hard NMS는 더 높은 score의 이웃과 우연히 겹치는 true positive를 지워버릴 수 있습니다(함께 서 있는 두 사람). Soft-NMS는 대신 이웃의 score에 Gaussian $e^{-\text{IoU}^2/\sigma}$ 또는 linear $1-\text{IoU}$ weight를 곱하므로, 진짜로 별개인 object는 감소했지만 0이 아닌 score로 살아남아 threshold 위에서 여전히 복구될 수 있습니다. ranking에 대한 drop-in 변경이고, 재학습이 없습니다. Matrix-NMS (SOLOv2)는 같은 아이디어를 병렬화합니다.
</div></details>

<details class="qa"><summary>rotated box나 mask IoU는 어떤가요?</summary>
<div class="qa-body">

**짧게:** rotated IoU는 polygon-clipping intersection area가 필요하고; mask IoU는 pixelwise AND over OR입니다.

**깊게:** box가 회전하면 axis-aligned `min/max`는 더 이상 intersection을 bound하지 못합니다 — intersection polygon(Sutherland–Hodgman clipping)과 그 area를 shoelace 공식으로 계산합니다. Mask IoU는 segmentation에 더 단순하고 robust합니다: 인스턴스마다 `(pred & gt).sum() / (pred | gt).sum()` — mIoU가 confusion matrix에 누적하는 것과 같은 양입니다([mAP & mIoU](#/ml-coding/metrics-map-miou) 참고).
</div></details>

### Follow-ups
- **큰 $N$에서의 메모리?** 전체 $(N,M)$ 행렬은 $O(N^2)$; 아주 큰 candidate set에는 greedy winner-vs-rest 형태를 쓰거나 행렬을 tile하세요.
- **GIoU / DIoU / CIoU?** enclosing box / center distance / aspect ratio에 penalty를 더하는 변형 — NMS만이 아니라 *regression loss*로 쓰입니다.
- **이게 여러분 업무의 어디에 위치하나요?** detection/segmentation post-processing 경험으로 가는 자연스러운 다리.

## Cheat-sheet

| Item | Formula / trick | Note |
| --- | --- | --- |
| Area (xyxy) | $(x_2-x_1)(y_2-y_1)$, clamp $\ge 0$ | 연속 좌표에는 `+1` 없음 |
| Intersection | `max(0, min(rb) - max(lt))` | lt/rb를 $(N,M,2)$로 broadcast |
| IoU | $\dfrac{I}{A+B-I}$, guard denom | `/ max(union, 1e-8)` |
| Greedy NMS | sort desc, keep top, drop IoU > thr | $O(N^2)$, 순차적 |
| Soft-NMS | decay: $e^{-\text{IoU}^2/\sigma}$ | 더 나은 recall, 작은 비용 |
| Class-aware | class마다 coordinate offset | one global pass (`batched_nms`) |

**Cross-links:** [Object Detection](#/cv/detection) · [mAP & mIoU](#/ml-coding/metrics-map-miou) · [The ML Coding Round](#/ml-coding/intro)
