# NumPy & 브로드캐스팅 프라이머

> [!NOTE] 이 챕터의 목표
> 뒤따르는 모든 "밑바닥 구현" 실습(conv, attention, IoU, k-means)은 **반복문 없이 배열을 한 방에 계산하는 감각**을 전제합니다. 하지만 그 감각을 따로 가르치는 곳은 없었죠. 이 챕터가 그 빈칸을 채웁니다: **shape(모양) 읽기 → 벡터화 → 브로드캐스팅(broadcasting)** 세 가지면 충분합니다. 딥러닝 코드 버그의 절반은 shape가 안 맞아서 생기므로, 여기서 잡고 가면 나머지가 훨씬 수월해집니다.

## §0 · 왜 NumPy인가 — 반복문 대신 배열

파이썬 `for` 반복문으로 숫자 100만 개를 더하면 느립니다. NumPy는 그 연산을 **C로 컴파일된 한 번의 배열 연산**으로 바꿔 수십~수백 배 빠르게 만듭니다. 그래서 ML 코드는 "반복문을 배열 연산으로 바꾸는(vectorization, 벡터화)" 스타일로 씁니다.

```python
import numpy as np
xs = [1, 2, 3, 4]
# 느린 파이썬 방식
total = 0
for x in xs:
    total += x * x
# 빠른 NumPy 방식 — 반복문이 사라짐
a = np.array(xs)
total = np.sum(a ** 2)     # 30
```

핵심 사고 전환: **"각 원소마다 무엇을 할까"가 아니라 "배열 전체에 무슨 연산을 걸까"** 로 생각합니다.

## §1 · ndarray: shape · dtype · axis

NumPy의 기본 객체는 **ndarray**(N차원 배열)입니다. 세 가지 속성만 기억하세요.

- **shape(모양)**: 각 차원의 크기. `(3,)` 벡터, `(2,3)` 행렬, `(N,C,H,W)` 이미지 배치.
- **dtype**: 원소 타입(`float64`, `int64` 등). 정수/실수 혼동이 버그를 만드니 `float`로 캐스팅 습관.
- **axis(축)**: 연산을 "어느 방향으로" 접을지. 2D에서 `axis=0`은 **행을 가로질러 아래로**(열별 결과), `axis=1`은 **열을 가로질러 옆으로**(행별 결과).

<figure>
<svg viewBox="0 0 560 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <g stroke="#98a3b2" stroke-width="1.2" fill="none">
    <rect x="120" y="50" width="180" height="120" rx="6"/>
    <line x1="120" y1="90" x2="300" y2="90"/><line x1="120" y1="130" x2="300" y2="130"/>
    <line x1="165" y1="50" x2="165" y2="170"/><line x1="210" y1="50" x2="210" y2="170"/><line x1="255" y1="50" x2="255" y2="170"/>
  </g>
  <text x="210" y="42" text-anchor="middle" fill="#98a3b2">2D array (3×4)</text>
  <line x1="90" y1="55" x2="90" y2="165" stroke="#e0533f" stroke-width="2.5" marker-end="url(#d0)"/>
  <text x="40" y="115" fill="#e0533f">axis=0</text>
  <text x="42" y="130" fill="#98a3b2" font-size="10">(열별)</text>
  <line x1="130" y1="188" x2="290" y2="188" stroke="#0ea5e9" stroke-width="2.5" marker-end="url(#d1)"/>
  <text x="330" y="150" fill="#0ea5e9">axis=1 (행별)</text>
  <text x="360" y="185" fill="#98a3b2" font-size="11">a.sum(axis=1) → shape (3,)</text>
  <text x="360" y="80" fill="#98a3b2" font-size="11">a.sum(axis=0) → shape (4,)</text>
  <defs>
    <marker id="d0" markerWidth="9" markerHeight="9" refX="4" refY="7" orient="auto"><path d="M0 0 L4 7 L8 0" fill="none" stroke="#e0533f" stroke-width="1.5"/></marker>
    <marker id="d1" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#0ea5e9" stroke-width="1.5"/></marker>
  </defs>
</svg>
<figcaption>axis는 "사라지는 차원"입니다. <code>(3,4)</code> 배열을 <code>axis=1</code>로 sum하면 그 축(길이 4)이 접혀 <code>(3,)</code>이 됩니다. <code>keepdims=True</code>를 주면 <code>(3,1)</code>로 남겨 브로드캐스팅에 쓰기 좋습니다.</figcaption>
</figure>

```python
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])   # shape (3,4)
a.sum(axis=0)      # [15,18,21,24]  (열별 합, shape (4,))
a.sum(axis=1)      # [10,26,42]     (행별 합, shape (3,))
a.argmax(axis=1)   # [3,3,3]        (행별 최댓값 위치)
```

## §2 · 인덱싱 & 슬라이싱

```python
a[0]          # 첫 행 (shape (4,))
a[:, 1]       # 두 번째 열 (shape (3,))
a[0:2, 1:3]   # 부분 블록
a[a > 6]      # 조건(불리언) 인덱싱 → 6 초과 원소만
```

## §3 · 벡터화 — 반복문을 지우기

원소별 연산(`+ - * / **`, `np.exp`, `np.maximum` …)은 배열 전체에 한 번에 적용됩니다. 예를 들어 "각 행을 그 행의 합으로 나눠 확률처럼 만들기"를 반복문 없이 해봅시다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"row_normalize","packages":["numpy"],"approx":true,"starter":"def row_normalize(x):\n    # x: 2차원 리스트/배열. 각 행을 그 행의 합으로 나눠서 반환 (반복문 없이!).\n    # 힌트: a.sum(axis=1, keepdims=True) 로 (N,1) 을 만들면 브로드캐스팅으로 나눠집니다.\n    import numpy as np\n    a = np.asarray(x, dtype=float)\n    # TODO\n    return out.tolist()","tests":[{"args":[[[1,1,2],[2,2,4]]],"expect":[[0.25,0.25,0.5],[0.25,0.25,0.5]]},{"args":[[[1,3]]],"expect":[[0.25,0.75]]},{"args":[[[5,0,0],[0,0,10]]],"expect":[[1.0,0.0,0.0],[0.0,0.0,1.0]]}],"solution":"import numpy as np\n\ndef row_normalize(x):\n    a = np.asarray(x, dtype=float)\n    out = a / a.sum(axis=1, keepdims=True)\n    return out.tolist()"}
</script>
</div>

`a.sum(axis=1, keepdims=True)`가 `(N,1)`이고, `(N,M) / (N,1)`이 각 행에 맞춰 자동으로 나눠집니다 — 이게 바로 브로드캐스팅입니다.

## §4 · 브로드캐스팅 (이 챕터의 주인공)

**브로드캐스팅**은 shape가 다른 두 배열을 연산할 때, NumPy가 작은 쪽을 **자동으로 늘려(stretch)** 모양을 맞춰 주는 규칙입니다. 실제 메모리 복사는 하지 않아 빠릅니다.

**규칙:** 두 shape를 오른쪽 끝부터 비교해서, 각 차원이 (1) 같거나 (2) 한쪽이 1이면 맞습니다. 1인 쪽이 상대 크기만큼 늘어납니다.

가장 많이 쓰는 패턴: 열 벡터 `(N,1)`와 행 벡터 `(1,M)`를 더하면 `(N,M)` 격자가 나옵니다.

<figure>
<svg viewBox="0 0 600 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="45" y="30" text-anchor="middle" fill="#0ea5e9">(3,1)</text>
  <g fill="none" stroke="#0ea5e9" stroke-width="1.6">
    <rect x="25" y="45" width="40" height="40"/><rect x="25" y="85" width="40" height="40"/><rect x="25" y="125" width="40" height="40"/>
  </g>
  <text x="45" y="70" text-anchor="middle" fill="currentColor">10</text><text x="45" y="110" text-anchor="middle" fill="currentColor">20</text><text x="45" y="150" text-anchor="middle" fill="currentColor">30</text>
  <text x="90" y="110" text-anchor="middle" font-size="18" fill="currentColor">+</text>
  <text x="230" y="30" text-anchor="middle" fill="#e0533f">(1,4)</text>
  <g fill="none" stroke="#e0533f" stroke-width="1.6">
    <rect x="120" y="45" width="40" height="40"/><rect x="160" y="45" width="40" height="40"/><rect x="200" y="45" width="40" height="40"/><rect x="240" y="45" width="40" height="40"/>
  </g>
  <text x="140" y="70" text-anchor="middle" fill="currentColor">1</text><text x="180" y="70" text-anchor="middle" fill="currentColor">2</text><text x="220" y="70" text-anchor="middle" fill="currentColor">3</text><text x="260" y="70" text-anchor="middle" fill="currentColor">4</text>
  <text x="315" y="110" text-anchor="middle" font-size="18" fill="currentColor">=</text>
  <text x="450" y="30" text-anchor="middle" fill="#12a150">(3,4) — 자동 확장</text>
  <g fill="none" stroke="#12a150" stroke-width="1.6">
    <rect x="360" y="45" width="45" height="40"/><rect x="405" y="45" width="45" height="40"/><rect x="450" y="45" width="45" height="40"/><rect x="495" y="45" width="45" height="40"/>
    <rect x="360" y="85" width="45" height="40"/><rect x="405" y="85" width="45" height="40"/><rect x="450" y="85" width="45" height="40"/><rect x="495" y="85" width="45" height="40"/>
    <rect x="360" y="125" width="45" height="40"/><rect x="405" y="125" width="45" height="40"/><rect x="450" y="125" width="45" height="40"/><rect x="495" y="125" width="45" height="40"/>
  </g>
  <g fill="currentColor" text-anchor="middle" font-size="11">
    <text x="382" y="70">11</text><text x="427" y="70">12</text><text x="472" y="70">13</text><text x="517" y="70">14</text>
    <text x="382" y="110">21</text><text x="427" y="110">22</text><text x="472" y="110">23</text><text x="517" y="110">24</text>
    <text x="382" y="150">31</text><text x="427" y="150">32</text><text x="472" y="150">33</text><text x="517" y="150">34</text>
  </g>
  <rect x="358" y="43" width="184" height="124" fill="#12a150" opacity="0.12">
    <animate attributeName="opacity" values="0.03;0.22;0.03" dur="2.2s" repeatCount="indefinite"/>
  </rect>
  <text x="300" y="205" text-anchor="middle" fill="#98a3b2">열(파랑)은 옆으로, 행(빨강)은 아래로 "늘어나" 3×4 격자를 채웁니다 — 반복문 0개</text>
</svg>
<figcaption>브로드캐스팅으로 <code>col[:,None] + row[None,:]</code> 한 줄이 모든 (i,j) 조합을 계산합니다. IoU 행렬, pairwise 거리, attention score가 전부 이 패턴입니다.</figcaption>
</figure>

```python
col = np.array([10, 20, 30])[:, None]   # shape (3,1)
row = np.array([1, 2, 3, 4])[None, :]   # shape (1,4)
col + row        # shape (3,4)  — 위 그림
```

`[:, None]`(또는 `np.newaxis`)이 **길이 1짜리 새 축을 끼워 넣는** 트릭입니다. 이걸로 "모든 쌍(pair)" 계산을 반복문 없이 합니다. k-means의 점-중심 거리, detection의 박스-박스 IoU가 이렇게 계산됩니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"pairwise_sq_dists","packages":["numpy"],"approx":true,"starter":"def pairwise_sq_dists(A, B):\n    # A:(n,d), B:(m,d).  결과 D:(n,m), D[i,j] = A[i] 와 B[j] 사이 유클리드 거리의 제곱.\n    # 반복문 없이! 힌트: A[:,None,:] - B[None,:,:]  -> (n,m,d), 그다음 제곱합을 axis=-1 로.\n    import numpy as np\n    A = np.asarray(A, float); B = np.asarray(B, float)\n    # TODO\n    return D.tolist()","tests":[{"args":[[[0,0],[3,0]],[[0,0],[0,4]]],"expect":[[0.0,16.0],[9.0,25.0]]},{"args":[[[1,1]],[[1,1],[2,2]]],"expect":[[0.0,2.0]]},{"args":[[[0,0,0]],[[1,2,2]]],"expect":[[9.0]]}],"solution":"import numpy as np\n\ndef pairwise_sq_dists(A, B):\n    A = np.asarray(A, float); B = np.asarray(B, float)\n    diff = A[:, None, :] - B[None, :, :]   # (n,m,d)\n    D = (diff ** 2).sum(axis=-1)           # (n,m)\n    return D.tolist()"}
</script>
</div>

## §5 · Fancy indexing & 원-핫

정수 배열로 여러 위치를 한 번에 골라 읽거나 씁니다. 대표 예가 레이블을 **원-핫(one-hot)** 벡터로 바꾸는 것입니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"one_hot","packages":["numpy"],"approx":true,"starter":"def one_hot(labels, num_classes):\n    # labels: 정수 리스트 (예: [0,2,1]).  각 레이블을 길이 num_classes 원-핫 행으로.\n    # 힌트: zeros((N, num_classes)) 를 만들고, m[np.arange(N), labels] = 1\n    import numpy as np\n    a = np.asarray(labels, dtype=int)\n    # TODO\n    return m.tolist()","tests":[{"args":[[0,2,1],3],"expect":[[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0]]},{"args":[[1,0],2],"expect":[[0.0,1.0],[1.0,0.0]]},{"args":[[2],4],"expect":[[0.0,0.0,1.0,0.0]]}],"solution":"import numpy as np\n\ndef one_hot(labels, num_classes):\n    a = np.asarray(labels, dtype=int)\n    m = np.zeros((a.size, num_classes))\n    m[np.arange(a.size), a] = 1.0\n    return m.tolist()"}
</script>
</div>

## §6 · 수치 안정성 한 스푼

`np.exp`는 큰 입력에서 넘칩니다(overflow). softmax 같은 곳에서는 **최댓값을 빼고** 계산합니다(값은 동일, 안정성↑). `np.clip`으로 범위를 자르고, `np.log`에는 아주 작은 `eps`를 더해 `log(0)`을 피합니다. 자세한 건 [손실 & gradient](#/ml-coding/losses-gradients)에서.

```python
def softmax(z):                       # 안정적 softmax
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

## 흔한 버그

- **shape 불일치:** 연산 전에 `print(a.shape)` — 버그의 절반이 여기서 잡힙니다.
- **`keepdims` 빼먹기:** `sum(axis=1)`은 `(N,)`이라 `(N,M)`과 곧장 안 나눠짐 → `keepdims=True`로 `(N,1)`.
- **정수 나눗셈:** 정수 배열 누적/`//`에서 값이 깨질 수 있음 → `float`로 캐스팅.
- **뷰 vs 복사:** 슬라이싱은 원본을 가리키는 **뷰**라 수정하면 원본이 바뀜 → 필요하면 `.copy()`.
- **브로드캐스팅 오작동:** `(N,)`와 `(N,)`를 "쌍으로" 만들려면 `[:,None]`/`[None,:]`를 꼭 붙일 것.

## Cheat-sheet

| 하고 싶은 것 | 코드 |
| --- | --- |
| 모양 확인 | `a.shape`, `a.dtype` |
| 축별 합/평균 | `a.sum(axis=0)`, `a.mean(axis=1, keepdims=True)` |
| 행별 최대 위치 | `a.argmax(axis=1)` |
| 반복문 제거 | 원소별 연산 · `@`(matmul) · fancy indexing |
| 새 축 끼우기 | `a[:, None]`, `a[None, :]`, `np.newaxis` |
| 모든 쌍 계산 | `A[:,None,:] - B[None,:,:]` → 브로드캐스팅 |
| 조건 선택 | `np.where(cond, x, y)`, `a[a>0]` |
| 안정적 exp | 최댓값 빼기: `np.exp(z - z.max(...))` |
| 값 자르기 | `np.clip(a, lo, hi)`, `np.maximum(0, a)` |

**다음:** [ML 코딩 라운드](#/ml-coding/intro) · [손실 & gradient](#/ml-coding/losses-gradients) · [Conv & Pooling 구현](#/ml-coding/conv-pooling)
