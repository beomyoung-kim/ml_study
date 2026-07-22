# IoU & Non-Max Suppression

> [!NOTE] 이 챕터의 목표
> 객체 검출(object detection)의 두 가지 필수 도구를 직접 구현합니다. **IoU** = 두 박스가 얼마나 겹치는지 재는 자, **NMS** = 같은 물체를 가리키는 중복 박스 중 최고만 남기는 청소기. 둘 다 [NumPy 프라이머](#/ml-coding/numpy-primer)의 브로드캐스팅으로 반복문 없이 계산합니다. 이 챕터는 [객체 검출(Detection)](#/cv/detection) 바로 앞에 두면 자연스럽습니다.

## 무엇을 / 왜

검출 모델은 한 물체 주위에 보통 **여러 개의 겹친 박스**를 뱉습니다. 문제는 두 가지입니다.
1. **"이 두 박스가 같은 물체를 가리키나?"** 를 판단할 자가 필요 → **IoU**.
2. **"중복을 어떻게 정리하나?"** → **NMS(Non-Max Suppression, 비최대 억제)**.

**IoU(Intersection over Union, 교집합/합집합)** 는 두 박스의 겹친 넓이를 합친 넓이로 나눈 값입니다. 0(안 겹침)~1(완전히 같음).

<figure>
<svg viewBox="0 0 640 230" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="13">
  <!-- intersection shaded first (under strokes) -->
  <rect x="200" y="95" width="90" height="80" fill="rgba(18,161,80,.35)"/>
  <!-- box A -->
  <rect x="70" y="45" width="220" height="130" fill="rgba(99,102,241,.10)" stroke="#6366f1" stroke-width="2.5"/>
  <text x="110" y="38" fill="#6366f1" font-weight="700">박스 A</text>
  <!-- box B -->
  <rect x="200" y="95" width="200" height="110" fill="rgba(224,83,63,.10)" stroke="#e0533f" stroke-width="2.5"/>
  <text x="360" y="222" fill="#e0533f" font-weight="700">박스 B</text>
  <!-- intersection label -->
  <text x="245" y="140" text-anchor="middle" fill="#12a150" font-weight="700" font-size="12">교집합</text>
  <text x="245" y="156" text-anchor="middle" fill="#12a150" font-size="10">(∩)</text>
  <!-- formula -->
  <text x="470" y="90" fill="currentColor" font-size="14">IoU =</text>
  <line x1="470" y1="105" x2="600" y2="105" stroke="currentColor" stroke-width="1.2"/>
  <text x="535" y="98" text-anchor="middle" fill="#12a150">교집합 넓이</text>
  <text x="535" y="125" text-anchor="middle" fill="#98a3b2">합집합 넓이(∪)</text>
  <text x="470" y="165" fill="#98a3b2" font-size="11">0 = 안 겹침</text>
  <text x="470" y="182" fill="#98a3b2" font-size="11">1 = 완전히 같음</text>
</svg>
<figcaption>IoU는 두 박스의 <b>교집합(초록)</b>을 <b>합집합</b>으로 나눈 값입니다. 검출 평가와 NMS의 "겹침 판정" 모두 이 하나의 숫자로 합니다.</figcaption>
</figure>

**NMS** 는 겹친 박스 무리를 정리합니다: 점수가 가장 높은 박스를 남기고(keep), 그것과 IoU가 임계값을 넘는 박스는 "중복"으로 보고 제거(drop), 남은 것들에 대해 반복.

<figure>
<svg viewBox="0 0 640 190" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="150" y="20" text-anchor="middle" fill="#98a3b2" font-weight="700">NMS 이전 — 중복 박스</text>
  <rect x="70" y="45" width="150" height="95" fill="none" stroke="#98a3b2" stroke-width="1.5"/><text x="78" y="60" fill="#98a3b2">0.72</text>
  <rect x="85" y="55" width="150" height="95" fill="none" stroke="#98a3b2" stroke-width="1.5"/><text x="200" y="70" fill="#98a3b2">0.68</text>
  <rect x="60" y="35" width="150" height="95" fill="none" stroke="#e0533f" stroke-width="2.5"/><text x="66" y="30" fill="#e0533f" font-weight="700">0.95 ★</text>
  <text x="320" y="95" text-anchor="middle" font-size="22" fill="#98a3b2">→</text>
  <text x="500" y="20" text-anchor="middle" fill="#98a3b2" font-weight="700">NMS 이후 — 최고만 유지</text>
  <rect x="425" y="35" width="150" height="95" fill="rgba(224,83,63,.12)" stroke="#e0533f" stroke-width="2.5"/><text x="431" y="30" fill="#e0533f" font-weight="700">0.95 ✓</text>
</svg>
<figcaption>같은 물체에 붙은 박스 3개 중 점수 0.95짜리만 남고, 그와 많이 겹치는 나머지는 제거됩니다.</figcaption>
</figure>

> [!TIP] 면접 한 줄
> "IoU는 top-left들의 `max`와 bottom-right들의 `min`을 broadcasting으로 구해 intersection 사각형을 만들고, 음수를 0으로 clamp합니다. NMS는 greedy 루프입니다 — 최고 score를 keep하고, 그것과 너무 겹치는 것을 drop하고, 반복." 인터뷰어는 알고리즘 자체보다 **vectorization(벡터화)** 과 **edge case**(zero-area box, 완전 포함, 빈 입력)를 훨씬 더 봅니다.

## 수학

박스는 연속 좌표에서 `[x1, y1, x2, y2]`(왼쪽위·오른쪽아래)이고 넓이는 $(x_2-x_1)(y_2-y_1)$입니다(`+1` 없음 — 옛 VOC 정수 픽셀 관례와 다름). 두 박스 $A, B$:

$$
\text{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A\cap B|}{|A| + |B| - |A\cap B|}
$$

교집합 사각형의 corner는 $\big(\max(x_1^A,x_1^B),\,\max(y_1^A,y_1^B)\big)$와 $\big(\min(x_2^A,x_2^B),\,\min(y_2^A,y_2^B)\big)$입니다; 변이 음수면 안 겹치는 것이므로 $0$으로 clamp합니다.

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

순서대로 만드세요: area 먼저(degenerate box(넓이 0 박스)를 방어), 그다음 broadcast된 IoU grid, 그 위에 greedy 루프.

### 1. Box area <span class="badge badge-easy">Easy</span>

각 `[x1, y1, x2, y2]` box의 area, 음의 변 길이를 0으로 clamp하여 degenerate box는 `0`이 되게 합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"box_area","packages":["numpy"],"approx":true,"starter":"def box_area(boxes):\n    # (N,4) xyxy boxes -> (N,) areas; clamp negative side lengths to 0\n    pass","tests":[{"args":[[[0,0,10,10],[1,1,4,5],[5,5,2,2]]],"expect":[100.0,12.0,0.0]},{"args":[[[0,0,2,3],[0,0,0,9]]],"expect":[6.0,0.0]},{"args":[[[2,2,5,7]]],"expect":[15.0]},{"args":[[]],"expect":[]}],"solution":"import numpy as np\n\ndef box_area(boxes):\n    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)\n    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])"}
</script>
</div>

*O(N), 완전 vectorized.* **함정:** `+1` 없음 — 연속 `xyxy` 좌표이지 VOC 정수 픽셀이 아닙니다.

### 2. Pairwise IoU by broadcasting <span class="badge badge-med">Medium</span>

전체 $(N,M)$ IoU grid: top-left들의 `max`와 bottom-right들의 `min`을 broadcast(브로드캐스팅)하고, clamp한 다음, union으로 나눕니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"iou_matrix","packages":["numpy"],"approx":true,"starter":"def iou_matrix(a, b):\n    # a:(N,4), b:(M,4) -> (N,M); broadcast max of top-lefts, min of bottom-rights, clamp\n    pass","tests":[{"args":[[[0,0,10,10]],[[0,0,10,10]]],"expect":[[1.0]]},{"args":[[[0,0,10,10]],[[1,1,11,11]]],"expect":[[0.680672268907563]]},{"args":[[[0,0,10,10],[20,20,30,30]],[[0,0,10,10],[5,5,15,15]]],"expect":[[1.0,0.14285714285714285],[0.0,0.0]]},{"args":[[[0,0,2,2]],[[10,10,12,12]]],"expect":[[0.0]]},{"args":[[[0,0,1,1],[2,2,3,3]],[]],"expect":[[],[]]}],"solution":"import numpy as np\n\ndef iou_matrix(a, b):\n    a = np.asarray(a, dtype=float).reshape(-1, 4)\n    b = np.asarray(b, dtype=float).reshape(-1, 4)\n    def area(x):\n        return np.maximum(0.0, x[:, 2] - x[:, 0]) * np.maximum(0.0, x[:, 3] - x[:, 1])\n    lt = np.maximum(a[:, None, :2], b[None, :, :2])\n    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])\n    wh = np.maximum(0.0, rb - lt)\n    inter = wh[..., 0] * wh[..., 1]\n    union = area(a)[:, None] + area(b)[None, :] - inter\n    return inter / np.maximum(union, 1e-8)"}
</script>
</div>

`[:, None]` / `[None, :]` 로 축을 삽입하는 게 전부입니다: $(N,1,\cdot)$과 $(1,M,\cdot)$ tensor를 정렬해 broadcasting이 루프 없이 완전한 $(N,M)$ grid를 만들게 합니다(브로드캐스팅 복습은 [NumPy 프라이머](#/ml-coding/numpy-primer)). **복잡도:** time·memory 모두 $O(NM)$.

### 3. Greedy NMS <span class="badge badge-med">Medium</span>

score로 정렬하고, 최고 box를 keep한 다음, 그것과의 IoU가 threshold를 넘는 생존자를 drop하며 반복합니다. 같은 score에서는 이 교육용 구현이 입력 index 순서를 보존하도록 stable sort를 씁니다. 실제 CPU/GPU NMS의 tie 순서는 다를 수 있으므로 제품 로직이 tie 순서에 의존하지 않게 하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"nms","packages":["numpy"],"starter":"def nms(boxes, scores, iou_thr=0.5):\n    # greedy: keep the top score, drop boxes with IoU > thr, repeat; return kept indices\n    pass","tests":[{"args":[[[0,0,10,10],[1,1,11,11],[20,20,30,30],[0,0,10,10]],[0.9,0.8,0.7,0.95],0.5],"expect":[3,2]},{"args":[[[0,0,10,10],[1,1,11,11]],[0.9,0.8],0.5],"expect":[0]},{"args":[[[0,0,10,10],[1,1,11,11]],[0.9,0.8],0.95],"expect":[0,1]},{"args":[[[0,0,10,10],[100,100,110,110]],[0.5,0.9],0.5],"expect":[1,0]},{"args":[[],[],0.5],"expect":[]}],"solution":"import numpy as np\n\ndef nms(boxes, scores, iou_thr=0.5):\n    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)\n    scores = np.asarray(scores, dtype=float).reshape(-1)\n    if len(boxes) != len(scores):\n        raise ValueError('boxes and scores must have equal length')\n    def iou_m(a, b):\n        lt = np.maximum(a[:, None, :2], b[None, :, :2])\n        rb = np.minimum(a[:, None, 2:], b[None, :, 2:])\n        wh = np.maximum(0.0, rb - lt)\n        inter = wh[..., 0] * wh[..., 1]\n        area = lambda x: np.maximum(0.0, x[:, 2] - x[:, 0]) * np.maximum(0.0, x[:, 3] - x[:, 1])\n        union = area(a)[:, None] + area(b)[None, :] - inter\n        return inter / np.maximum(union, 1e-8)\n    if boxes.size == 0:\n        return []\n    order = np.argsort(-scores, kind='stable')\n    keep = []\n    while order.size > 0:\n        i = int(order[0])\n        keep.append(i)\n        if order.size == 1:\n            break\n        ious = iou_m(boxes[i][None], boxes[order[1:]])[0]\n        order = order[1:][ious <= iou_thr]\n    return keep"}
</script>
</div>

루프는 **본질적으로 순차적**입니다 — keep된 각 box가 어떤 후보가 살아남을지를 바꾸니까요 — 하지만 각 스텝은 vectorized되어 있습니다. **복잡도:** 최악 $O(N^2)$; 각 iteration은 생존자에 대한 broadcast된 IoU 하나.

## Soft-NMS (흔한 후속 질문)

겹치는 box를 hard-delete(즉시 제거)하는 대신, score를 *decay(감쇠)* 합니다 — 붐비는 scene에서 더 나은 recall(재현율) (Bodla et al., 2017).

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
> 실무에서는 class-aware suppression을 위해 `torchvision.ops.box_iou`, `torchvision.ops.nms(boxes, scores, iou_thr)`, `torchvision.ops.batched_nms(boxes, scores, idxs, iou_thr)`를 씁니다.

## 인터뷰어가 주시하는 흔한 버그

<div class="proscons"><div><div class="pros-t">Do</div>

- `argsort()[::-1]`을 쓰세요 — `argsort`는 **오름차순**입니다.
- `rb - lt`를 `0`으로 clamp하세요(겹치지 않는 box는 음의 width를 냅니다).
- union을 `np.maximum(union, eps)`로 방어하세요(0으로 나누기 방지).
- Soft-NMS에서 mutate 전에 score를 `.copy()`하세요.

</div><div><div class="cons-t">Don't</div>

- width에 `+1`을 더하지 마세요 — 옛 VOC 정수 픽셀 관례이지, `xyxy` 연속 좌표가 아닙니다.
- NMS 루프 안에서 전체 $N\times N$ 행렬을 materialize하지 마세요(winner-vs-rest만 계산).
- 빈 입력 guard(`boxes.size == 0`)를 잊지 마세요.
- class-aware와 class-agnostic NMS를 헷갈리지 마세요.

</div></div>

## Q&A

<details class="qa"><summary>per-class Python 루프 없이 NMS를 class-aware하게 만들려면?</summary>
<div class="qa-body">

**짧게:** 각 box의 좌표를 큰 per-class 상수만큼 offset해서 서로 다른 class의 box가 절대 겹칠 수 없게 만든 다음, global NMS를 한 번 돌리세요 — 그게 바로 `batched_nms`입니다.

**깊게:** 좌표가 non-negative이고 dtype range가 충분하다는 계약 아래 `class_id * (max_coordinate + 1)` offset을 쓸 수 있습니다. 음수 좌표까지 허용하면 전체 coordinate span을 기준으로 offset을 잡고, FP16 overflow도 피해야 합니다. 이런 전제가 불명확하면 명시적 per-class NMS가 더 안전합니다.
</div></details>

<details class="qa"><summary>Soft-NMS vs hard NMS — 언제, 왜?</summary>
<div class="qa-body">

**짧게:** Soft-NMS는 겹치는 score를 삭제하는 대신 decay하여, 작은 latency 비용으로 붐비는 scene에서 recall을 개선합니다.

**깊게:** hard NMS는 더 높은 score의 이웃과 우연히 겹치는 true positive를 지워버릴 수 있습니다(함께 서 있는 두 사람). Soft-NMS는 이웃의 score에 Gaussian $e^{-\text{IoU}^2/\sigma}$ 또는 linear $1-\text{IoU}$ weight를 곱하므로, 진짜로 별개인 object는 감소했지만 0이 아닌 score로 살아남아 threshold 위에서 복구될 수 있습니다. ranking에 대한 drop-in 변경이고 재학습이 없습니다. Matrix-NMS (SOLOv2)는 같은 아이디어를 병렬화합니다.
</div></details>

<details class="qa"><summary>rotated box나 mask IoU는 어떤가요?</summary>
<div class="qa-body">

**짧게:** rotated IoU는 polygon-clipping intersection area가 필요하고; mask IoU는 pixelwise AND over OR입니다.

**깊게:** box가 회전하면 axis-aligned `min/max`는 더 이상 intersection을 bound하지 못합니다 — intersection polygon(Sutherland–Hodgman clipping)과 그 area를 shoelace 공식으로 계산합니다. Mask IoU는 segmentation에 더 단순하고 robust합니다: 인스턴스마다 `(pred & gt).sum() / (pred | gt).sum()` — mIoU가 confusion matrix에 누적하는 것과 같은 양입니다([mAP & mIoU](#/ml-coding/metrics-map-miou) 참고).
</div></details>

### Follow-ups
- **큰 $N$에서의 메모리?** 전체 $(N,M)$ 행렬은 $O(N^2)$; 아주 큰 candidate set에는 greedy winner-vs-rest 형태를 쓰거나 행렬을 tile하세요.
- **GIoU / DIoU / CIoU?** enclosing box / center distance / aspect ratio에 penalty를 더하는 변형 — NMS만이 아니라 *regression loss*로 씀.
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

**다음:** [객체 검출(Detection)](#/cv/detection) · [mAP & mIoU](#/ml-coding/metrics-map-miou) · [NumPy 프라이머](#/ml-coding/numpy-primer)
