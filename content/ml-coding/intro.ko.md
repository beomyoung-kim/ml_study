# ML 코딩 라운드 (밑바닥부터)

> [!NOTE] 이 챕터의 목표
> 이 파트는 [머신러닝이란?](#/foundations/what-is-ml)·[신경망 첫걸음](#/foundations/neural-networks-basics)에서 **개념으로 배운 것**을 이제 **직접 손으로 구현**하는 곳입니다. 이 첫 챕터는 "밑바닥 구현이 무엇을 요구하는지"와 "어떻게 접근하는지"의 지도를 그려 줍니다. 코드 문법이 아직 서툴러도 괜찮습니다 — 먼저 [NumPy & 브로드캐스팅 프라이머](#/ml-coding/numpy-primer)를 보고 오면 이 파트가 훨씬 편해집니다.

## "밑바닥 구현"이 무엇이고 왜 하나

여기서 말하는 ML 코딩은 **LeetCode(알고리즘 퍼즐)** 와 다릅니다. `torch.nn.Conv2d` 같은 라이브러리 한 줄을 쓰는 대신, 그 안에서 실제로 무슨 계산이 일어나는지를 **NumPy로 직접** 만들어 보는 것입니다. 왜 이게 중요할까요?

- **이해의 증명.** conv를 직접 짜 보면 "kernel이 슬라이딩하며 곱하고 더한다"가 몸으로 남습니다. 라이브러리만 쓴 사람과 확실히 구분됩니다.
- **디버깅 능력.** 학습이 이상할 때, 내부를 아는 사람만이 "여기 softmax가 축을 잘못 잡았네"를 짚습니다.
- **면접의 핵심 신호.** research/applied 직무 면접에서 자주 나오고, 범위가 **유한**합니다 — 아래 canonical 문제 열 개 남짓이 대부분을 커버합니다.

> [!TIP] 면접 한 줄
> "ML-from-scratch는 범위가 유한한 syllabus라, 준비 대비 레버리지가 가장 큽니다. shape·수치안정성 트릭·설명 순서를 체화하면 압박 속에서도 얼지 않습니다." — 돌아가는 코드는 기본 점수일 뿐, 진짜 신호는 *거기에 도달하는 과정*에 있습니다.

## 인터뷰어가 실제로 보는 다섯 가지

<dl class="kv">
<dt>수식 → 코드 (fluency)</dt><dd>$\text{softmax}(x)_i = e^{x_i}/\sum_j e^{x_j}$ 같은 식을 망설임 없이 안정적인 NumPy 코드로 옮길 수 있나요?</dd>
<dt>벡터화(vectorization) 사고</dt><dd>Python <code>for</code> 루프 대신 <b>broadcasting(브로드캐스팅)</b>·<b>matmul(행렬곱)</b>에 손이 가나요? 루프는 <i>첫 정답</i>용이고, 후속 질문은 늘 "이제 벡터화 해보세요"입니다.</dd>
<dt>shape(모양) 규율</dt><dd>모든 배열의 shape를 머릿속(과 주석)으로 추적하나요? 버그의 대부분이 shape/축 실수입니다.</dd>
<dt>수치 안정성(numerical stability)</dt><dd><code>exp</code> 전에 최댓값을 빼나요? <code>log</code> 전에 clamp 하나요? 가장 흔한 판별점입니다.</dd>
<dt>엣지 케이스 & 테스트</dt><dd>빈 입력, 넓이 0인 박스, 원소 하나짜리 batch, 동점(tie). 시키지 않아도 <code>__main__</code>에 sanity check를 다세요.</dd>
</dl>

## 핵심 개념: 벡터화 = "루프를 배열 연산으로"

밑바닥 구현의 90%는 "명시적 루프를 배열 연산으로 바꾸기"입니다. 가장 중요한 도구가 **브로드캐스팅** — shape가 다른 배열을 자동으로 맞춰 계산하는 규칙입니다. 예를 들어 $N$개 점과 $M$개 점 사이의 모든 쌍(pairwise) 거리를 이중 루프 없이 구할 수 있습니다:

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- A: (N,1,D) -->
  <rect x="30" y="60" width="40" height="110" rx="6" fill="none" stroke="#0ea5e9" stroke-width="1.8"/>
  <line x1="30" y1="97" x2="70" y2="97" stroke="#0ea5e9"/><line x1="30" y1="134" x2="70" y2="134" stroke="#0ea5e9"/>
  <text x="50" y="50" text-anchor="middle" fill="#0ea5e9">A</text>
  <text x="50" y="192" text-anchor="middle" fill="#98a3b2">(N,1,D)</text>
  <text x="95" y="120" text-anchor="middle" font-size="18" fill="currentColor">+</text>
  <!-- B: (1,M,D) -->
  <rect x="120" y="90" width="120" height="40" rx="6" fill="none" stroke="#e0533f" stroke-width="1.8"/>
  <line x1="160" y1="90" x2="160" y2="130" stroke="#e0533f"/><line x1="200" y1="90" x2="200" y2="130" stroke="#e0533f"/>
  <text x="180" y="80" text-anchor="middle" fill="#e0533f">B</text>
  <text x="180" y="150" text-anchor="middle" fill="#98a3b2">(1,M,D)</text>
  <text x="270" y="120" text-anchor="middle" font-size="18" fill="currentColor">→</text>
  <!-- result grid (N,M,D) -->
  <text x="470" y="40" text-anchor="middle" fill="#12a150">브로드캐스팅이 자동으로 (N,M,D) 격자를 채움</text>
  <g stroke="#12a150" stroke-width="1.5" fill="rgba(18,161,80,.12)">
    <rect x="330" y="60" width="30" height="30"/><rect x="360" y="60" width="30" height="30"/><rect x="390" y="60" width="30" height="30"/><rect x="420" y="60" width="30" height="30"/>
    <rect x="330" y="90" width="30" height="30"/><rect x="360" y="90" width="30" height="30"/><rect x="390" y="90" width="30" height="30"/><rect x="420" y="90" width="30" height="30"/>
    <rect x="330" y="120" width="30" height="30"/><rect x="360" y="120" width="30" height="30"/><rect x="390" y="120" width="30" height="30"/><rect x="420" y="120" width="30" height="30"/>
  </g>
  <text x="315" y="80" text-anchor="end" fill="#0ea5e9">N행 ↓</text>
  <text x="390" y="170" text-anchor="middle" fill="#e0533f">M열 →</text>
</svg>
<figcaption>루프 없이 <code>A[:,None,:] + B[None,:,:]</code> 한 줄로 모든 (N×M) 쌍을 계산합니다. 길이 1인 축을 끼워 넣어(<code>None</code>) 모양을 맞추는 것이 벡터화의 핵심 습관입니다. 자세한 규칙은 <a href="#/ml-coding/numpy-primer">NumPy 프라이머</a>에.</figcaption>
</figure>

케이스의 ~90%를 커버하는 네 가지 "수(move)":

| 도구 | 쓸 때 | 예 |
| --- | --- | --- |
| **Broadcasting** | pairwise / outer 연산 | `a[:,None,:] - b[None,:,:]` → `(N,M,D)` 거리 |
| **matmul / einsum** | 내적, projection | `q @ k.T`, `np.einsum('nd,md->nm', a, b)` |
| **Fancy/boolean indexing** | gather, mask, scatter | `probs[np.arange(N), targets]` |
| **`reshape`/`transpose`** | head 분리/병합, im2col | `.reshape(B,T,H,Dh).transpose(0,2,1,3)` |

## 어떻게 풀어 나갈 것인가 (설명이 곧 점수)

인터뷰어는 결과물뿐 아니라 **과정**을 채점합니다. 눈에 보이는 루프를 따르세요:

```mermaid
flowchart LR
  A["① 명확화<br/>입출력, shape, dtype"] --> B["② 수식+복잡도<br/>말로 먼저"]
  B --> C["③ 정답 버전<br/>루프 OK"]
  C --> D["④ 벡터화<br/>broadcasting/matmul"]
  D --> E["⑤ 테스트<br/>엣지케이스 + assert"]
  E --> F["⑥ 후속질문<br/>메모리·batch·안정성"]
```

1. **계약(contract)을 명확히.** "박스는 `[x1,y1,x2,y2]` float `(N,4)`인가요? 인덱스를 반환하나요, 걸러진 박스를 반환하나요?" 좋은 질문 하나가 신뢰를 삽니다.
2. **수식과 복잡도를 소리 내어** — 타이핑 전에. "IoU는 교집합/합집합이고, pairwise는 $O(NM)$, `(N,M)` 행렬을 만들겠습니다."
3. **먼저 정답 버전**(루프 허용) → 그다음 "벡터화하겠습니다"라고 말하고 실행. 둘 다 보여주는 게 영리한 버전으로 바로 뛰는 것보다 신호가 큽니다.
4. **shape를 주석으로** 표기하며 진행. `# (B, H, T, Dh)`.
5. **시키지 않아도 테스트.** `assert` 하나 든 세 줄짜리 `__main__`이 "저는 돌아가는 코드를 냅니다"를 말해 줍니다.

> [!WARNING] 얼어붙을 때
> 머리가 하얘지면 naive 삼중 루프로 후퇴하고 *그렇다고 말하세요*: "명백히 correct한 버전을 먼저 만들고, 다음에 벡터화하겠습니다." 설명이 또렷한 느린 정답이, 멈춰 버린 영리한 답을 매번 이깁니다.

## 수치 안정성 체크리스트

인터뷰어가 적극적으로 찾는 네 가지입니다. **log-sum-exp(LSE)** 는 "`log(sum(exp(x)))`를 그대로 계산하면 `exp`가 폭발하니, 최댓값을 밖으로 빼서 안정화하는 트릭"입니다:

$$
\text{softmax}(x)_i = \frac{e^{x_i - \max_k x_k}}{\sum_j e^{x_j - \max_k x_k}}
\qquad
\text{LSE}(x) = \max_k x_k + \log\!\sum_j e^{x_j - \max_k x_k}
$$

- **안정적 softmax:** `exp` 전에 축 방향 `max`를 빼세요. 결과는 같지만 `inf`를 피합니다.
- **log-sum-exp:** `log(sum(exp(x)))`를 직접 계산하지 말고 max를 빼내세요 — logits로부터의 cross-entropy가 이걸 암묵적으로 씁니다.
- **`log` 전 clamp:** `np.log(np.clip(p, 1e-12, 1.0))`로 `log(0) = -inf`를 피하세요.
- **나눗셈 방어:** IoU 합집합, softmax 분모, Dice에는 `x / np.maximum(denom, eps)`.

> [!DANGER] 인터뷰어가 지켜보는 흔한 버그
> `argsort`는 오름차순(점수엔 `argsort(-x)`); NumPy view는 메모리를 공유하니 in-place 편집 전 `.copy()`; conv output-size의 정수 나눗셈; `keepdims=True`를 잊어 reduction이 다시 broadcast 안 되는 것; softmax 축 실수; causal mask의 off-by-one.

## Canonical 문제 리스트

이 파트(와 컴퓨터 비전 파트)에서 하나씩 직접 구현합니다. 각 챕터엔 **실행 가능한 코드 에디터**가 있습니다.

<div class="card-grid">
  <a class="card" href="#/ml-coding/losses-gradients"><div class="card-emoji">📉</div><div class="card-title">손실 & gradient</div><div class="card-desc">MSE/CE/BCE/focal, softmax-CE gradient, autograd 없는 backward.</div></a>
  <a class="card" href="#/ml-coding/conv-pooling"><div class="card-emoji">🔲</div><div class="card-title">Conv & Pooling</div><div class="card-desc">Naive 루프 → im2col + GEMM; max/avg pool.</div></a>
  <a class="card" href="#/ml-coding/attention"><div class="card-emoji">🎯</div><div class="card-title">Attention</div><div class="card-desc">Scaled dot-product + multi-head, mask, stable softmax.</div></a>
  <a class="card" href="#/ml-coding/transformer"><div class="card-emoji">🧱</div><div class="card-title">Transformer Block</div><div class="card-desc">MHA + FFN + residual + norm, causal mask, KV-cache.</div></a>
  <a class="card" href="#/ml-coding/kmeans"><div class="card-emoji">🌀</div><div class="card-title">K-Means</div><div class="card-desc">Lloyd + k-means++, 벡터화 거리, 빈 클러스터 처리.</div></a>
  <a class="card" href="#/ml-coding/dataloader-augmentation"><div class="card-emoji">🔀</div><div class="card-title">Dataloader & 증강</div><div class="card-desc">Batch/shuffle/collate + 라벨 동기화 augmentation.</div></a>
  <a class="card" href="#/ml-coding/nms-iou"><div class="card-emoji">📦</div><div class="card-title">IoU & NMS</div><div class="card-desc">Broadcasting pairwise IoU; greedy + Soft-NMS. (컴퓨터 비전 파트)</div></a>
  <a class="card" href="#/ml-coding/metrics-map-miou"><div class="card-emoji">📊</div><div class="card-title">mAP & mIoU</div><div class="card-desc">Confusion-matrix mIoU; greedy matching + PR. (컴퓨터 비전 파트)</div></a>
</div>

> [!NOTE] 순서 안내
> **IoU/NMS·mAP/mIoU**는 검출·분할과 짝이 맞아 **컴퓨터 비전 파트**로 옮겼습니다(파일 위치·링크는 그대로). 이 파트에서는 신경망 핵심 블록(손실·conv·attention·transformer)을 이론([CNN·RNN·Transformer](#/foundations/architectures)) 바로 옆에서 구현합니다.

## 35분 배분

전형적인 ML 코딩 슬롯은 ~35–45분입니다. 후속 질문(코드만큼 신호가 큼)을 위한 여유를 남기세요:

| 단계 | 시간 | 하는 일 |
| --- | --- | --- |
| 명확화 | 2–3분 | 입출력, shape, dtype, 반환형 확정 |
| 수식+계획 | 3–4분 | 공식과 복잡도를 소리 내어 |
| 정답 버전 | 10–12분 | 루프 허용; shape 설명 |
| 벡터화 | 6–8분 | broadcasting/matmul; 계속 돌아가게 |
| 테스트 | 4–5분 | 알려진 케이스 `assert`가 든 `__main__` |
| 후속질문 | 남은 시간 | 메모리·batch·안정성·production 경로 |

## Q&A

<details class="qa"><summary>인터뷰어는 왜 "이제 벡터화 해보세요"를 좋아하나요?</summary>
<div class="qa-body">

**짧게:** 라이브러리를 *쓰는* 사람과 그 아래 배열 모델을 *이해하는* 사람을 갈라냅니다.

**깊게:** 벡터화 솔루션은 모든 중간 배열의 정확한 shape, 브로드캐스팅이 어디서 일어나는지, 그리고 중간 배열을 실제로 만드는 메모리 비용(예: $(N,M)$ IoU 행렬, $O(T^2)$ attention 행렬)을 따지게 강제합니다. 이 사고가 나중에 느린 학습 루프를 profile하거나 FlashAttention의 값어치를 판단할 때 그대로 필요합니다.
</div></details>

<details class="qa"><summary>NumPy로 쓸까요, PyTorch로 쓸까요?</summary>
<div class="qa-body">

**짧게:** 알고리즘은 NumPy 기본, framework 한 줄은 언급.

**깊게:** NumPy 밑바닥 구현으로 연산 이해를 증명하고, production 경로를 이름으로 대세요(`torchvision.ops.nms`, `F.scaled_dot_product_attention`, `F.cross_entropy`). autograd/GPU tensor가 핵심인 문제(Transformer block, custom backward)면 PyTorch로. 언제나 "production에서는 X를 쓴다"고 덧붙여, 몰라서 바퀴를 재발명하는 게 아님을 알리세요.
</div></details>

### 모든 문제에서 예상할 후속질문
- **"시간·메모리 복잡도는?"** — 묻기 전에 준비.
- **"이걸 어떻게 batch 하나요?"** — 보통 한 축을 batch로 접거나 축을 하나 더 broadcast.
- **"수치적으로 어디서 깨지나요?"** — `exp`·`log`·나눗셈을 가리키기.
- **"어떻게 테스트하나요?"** — 손실엔 numerical gradient check, IoU엔 closed-form 케이스, 어디서나 degenerate/empty 입력.

## Cheat-sheet

| 문제 | 핵심 트릭 | 복잡도 | 안정성 주의 |
| --- | --- | --- | --- |
| 손실 & gradient | `p - onehot(y)`가 CE grad | $O(NC)$ | stable softmax, log clamp |
| Conv | im2col → GEMM | $O(N C_o C_i K^2 HW)$ | output-size 정수 나눗셈 |
| Attention | `qkᵀ/√d`, stable softmax | $O(T^2 d)$ | max 빼기, $-\infty$ mask |
| Transformer block | pre-norm, residual, causal mask | $O(T^2 d + T d^2)$ | LN eps, softmax 전 mask |
| K-Means | $\lVert x-c\rVert^2=\lVert x\rVert^2+\lVert c\rVert^2-2x\!\cdot\!c$ | $O(NKD)$/iter | dist $\ge 0$ clamp, 빈 클러스터 |
| Dataloader | shuffle→batch→collate | $O(N)$ | `drop_last`, aug에 `.copy()` |
| IoU / NMS | broadcast lt/rb, `max(0, rb-lt)` | $O(NM)$/greedy | 합집합 `eps` |
| mAP / mIoU | `bincount` confusion; PR 적분 | $O(HW)$/$O(P\log P)$ | per-image greedy matching |

**다음:** [NumPy 프라이머](#/ml-coding/numpy-primer) · [손실 & gradient](#/ml-coding/losses-gradients) · [CNN·RNN·Transformer](#/foundations/architectures) · [Detection](#/cv/detection) · [Optimization](#/foundations/optimization)
