# K-Means & K-Means++

> [!TIP] 이 말부터 시작하세요
> "Lloyd's algorithm은 수렴할 때까지 두 스텝을 번갈아 합니다: 각 점을 가장 가까운 centroid에 **assign**하고, 각 centroid를 그 구성원의 평균으로 **update**합니다. 진짜 함정은 거리 행렬을 벡터화하는 것과 빈 cluster를 처리하는 것뿐입니다." 이게 문제의 전부입니다 — 보통 그렇듯 warm-up으로서 침착하게 풀어내세요.

흔한 phone-screen warm-up입니다. 사소해 보이므로 신호는 전적으로 디테일에 있습니다: 벡터화된 거리 계산, k-means++ 초기화, 빈 cluster 처리, 그리고 왜 Euclidean 거리를 순진하게 계산하면 안 되는지 아는 것.

## 목적 함수

K-means는 within-cluster sum of squares (inertia)를 최소화합니다:

$$
J = \sum_{i=1}^{N} \lVert x_i - c_{a_i} \rVert^2,\qquad a_i = \arg\min_k \lVert x_i - c_k\rVert^2
$$

각 Lloyd iteration은 $J$에 대한 coordinate descent입니다 (centroid 고정 → 최적 assignment; assignment 고정 → 최적 centroid = 평균). 따라서 **inertia는 절대 증가하지 않습니다** — 테스트에서 assert하기 좋은 불변식이죠. *local* optimum으로 수렴합니다.

## Distance 트릭

절대 $(N,K,D)$ 차이 tensor를 만들지 마세요. 제곱을 전개하세요:

$$
\lVert x - c\rVert^2 = \lVert x\rVert^2 + \lVert c\rVert^2 - 2\,x\!\cdot\!c
$$

cross term은 단일 matmul $x c^\top$ 로, 하나의 GEMM으로 $(N,K)$ 거리 행렬을 줍니다.

```python
import numpy as np


def sq_dists(x, c):
    """x:(N,D) c:(K,D) -> (N,K) squared Euclidean via the expansion trick."""
    x2 = np.sum(x ** 2, axis=1, keepdims=True)     # (N,1)
    c2 = np.sum(c ** 2, axis=1)                     # (K,)
    d = x2 + c2[None, :] - 2.0 * x @ c.T            # (N,K)
    return np.maximum(d, 0.0)                       # clamp fp negatives near 0
```

> [!WARNING] 거리를 clamp하세요
> 점이 centroid에 거의 정확히 놓일 때 부동소수점 상쇄로 인해 전개가 아주 작은 음수를 만들 수 있습니다. `np.maximum(d, 0)`은 나중에 `sqrt`를 취할 때 `nan`을 방지합니다.

## K-means++ 초기화

기존 centroid 중 가장 가까운 것으로부터의 제곱 거리에 비례하는 확률로 각 새 centroid를 샘플링하여 centroid를 멀리 떨어뜨려 seed합니다. 이것은 기대 $O(\log K)$ 근사 보장과 random init보다 훨씬 안정적인 수렴을 줍니다 (Arthur & Vassilvitskii, 2007).

```python
def kmeans_pp_init(x, k, rng):
    n = x.shape[0]
    c = np.empty((k, x.shape[1]), dtype=x.dtype)
    c[0] = x[rng.integers(n)]                       # first centroid: random point
    closest = sq_dists(x, c[:1]).ravel()            # (N,) dist to nearest centroid
    for i in range(1, k):
        probs = closest / closest.sum()             # D^2 weighting
        c[i] = x[rng.choice(n, p=probs)]
        closest = np.minimum(closest, sq_dists(x, c[i:i + 1]).ravel())
    return c
```

## Lloyd's algorithm

```python
def kmeans(x, k, max_iter=100, tol=1e-4, seed=0):
    """Returns centers (K,D), labels (N,), inertia_history."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    c = kmeans_pp_init(x, k, rng)
    history, labels = [], np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        d = sq_dists(x, c)                           # (N,K)
        labels = np.argmin(d, axis=1)                # assign
        history.append(float(d[np.arange(n), labels].sum()))  # inertia

        new_c = np.empty_like(c)
        for j in range(k):                           # update
            members = x[labels == j]
            new_c[j] = members.mean(0) if len(members) else x[rng.integers(n)]
            #                                          ^ empty cluster: re-seed

        shift = np.linalg.norm(new_c - c)
        c = new_c
        if shift < tol:                              # converged
            break
    return c, labels, history
```

**복잡도:** 각 iteration은 $O(NKD)$ (거리 matmul이 지배)이며, 적은 수의 iteration에 걸쳐 반복됩니다. 메모리는 거리 행렬에 대해 $O(NK)$ 입니다.

## Sanity check

```python
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    centers_true = np.array([[0, 0], [5, 5], [-5, 5]], dtype=float)
    x = np.vstack([rng.normal(c, 0.4, (80, 2)) for c in centers_true])
    c, labels, hist = kmeans(x, k=3, seed=0)
    assert c.shape == (3, 2) and labels.shape == (240,)
    assert hist[-1] <= hist[0]                   # inertia never increases
    # each recovered centroid is close to some true centroid
    assert all(np.min(np.linalg.norm(centers_true - ci, axis=1)) < 0.5 for ci in c)
    print("inertia", round(hist[0], 1), "->", round(hist[-1], 1), "OK")
```

> [!NOTE] 프레임워크 한 줄
> `sklearn.cluster.KMeans(n_clusters=k, init="k-means++")`; 대규모에서는 `MiniBatchKMeans` 또는 FAISS k-means (GPU, vector-quantization codebook에 사용).

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **NumPy가 준비된 라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

거리 행렬부터 아래에서 위로 쌓으세요 — 그다음 두 Lloyd 스텝, 마지막으로 seed된 전체 루프.

### 1. Squared-Distance Matrix <span class="badge badge-easy">Easy</span>

$\lVert x\rVert^2+\lVert c\rVert^2-2xc^\top$ 전개로 얻는 $(N,K)$ 거리, $\ge 0$로 clamp.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sq_dists","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef sq_dists(x, c):\n    # x:(N,D) c:(K,D) -> (N,K) squared Euclidean via the expansion trick; clamp >= 0\n    pass","tests":[{"args":[[[0.0,0.0],[3.0,4.0]],[[0.0,0.0],[3.0,0.0]]],"expect":[[0.0,9.0],[25.0,16.0]]},{"args":[[[1.0,1.0]],[[1.0,1.0],[4.0,5.0]]],"expect":[[0.0,25.0]]},{"args":[[[2.0,0.0],[0.0,2.0]],[[0.0,0.0]]],"expect":[[4.0],[4.0]]}],"solution":"import numpy as np\n\ndef sq_dists(x, c):\n    x = np.asarray(x, dtype=float)\n    c = np.asarray(c, dtype=float)\n    x2 = np.sum(x ** 2, axis=1, keepdims=True)\n    c2 = np.sum(c ** 2, axis=1)\n    d = x2 + c2[None, :] - 2.0 * x @ c.T\n    return np.maximum(d, 0.0)"}
</script>
</div>

### 2. Assign Step <span class="badge badge-easy">Easy</span>

각 점을 가장 가까운 centroid의 인덱스로 라벨링(K에 대한 `argmin`).

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"assign_clusters","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef assign_clusters(x, c):\n    # squared distances to each centroid, then argmin over axis=1; return a list of ints\n    pass","tests":[{"args":[[[0.0,0.0],[0.2,0.0],[5.0,5.0],[4.8,5.0]],[[0.0,0.0],[5.0,5.0]]],"expect":[0,0,1,1]},{"args":[[[1.0,1.0],[9.0,9.0],[1.0,0.0]],[[9.0,9.0],[0.0,0.0]]],"expect":[1,0,1]},{"args":[[[0.0,0.0]],[[1.0,0.0],[0.0,1.0]]],"expect":[0]}],"solution":"import numpy as np\n\ndef assign_clusters(x, c):\n    x = np.asarray(x, dtype=float)\n    c = np.asarray(c, dtype=float)\n    x2 = np.sum(x ** 2, axis=1, keepdims=True)\n    c2 = np.sum(c ** 2, axis=1)\n    d = np.maximum(x2 + c2[None, :] - 2.0 * x @ c.T, 0.0)\n    return np.argmin(d, axis=1).tolist()"}
</script>
</div>

### 3. Update Step <span class="badge badge-easy">Easy</span>

각 centroid를 배정된 구성원의 평균으로 다시 계산합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"update_centroids","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef update_centroids(x, labels, k):\n    # for each cluster j in range(k), set centroid to the mean of x[labels == j]\n    pass","tests":[{"args":[[[0.0,0.0],[2.0,0.0],[10.0,10.0],[12.0,10.0]],[0,0,1,1],2],"expect":[[1.0,0.0],[11.0,10.0]]},{"args":[[[1.0,1.0],[3.0,3.0],[5.0,5.0]],[0,0,1],2],"expect":[[2.0,2.0],[5.0,5.0]]},{"args":[[[4.0,2.0],[2.0,4.0]],[1,1],2],"expect":[[0.0,0.0],[3.0,3.0]]}],"solution":"import numpy as np\n\ndef update_centroids(x, labels, k):\n    x = np.asarray(x, dtype=float)\n    labels = np.asarray(labels)\n    new_c = np.zeros((k, x.shape[1]))\n    for j in range(k):\n        members = x[labels == j]\n        if len(members):\n            new_c[j] = members.mean(0)\n    return new_c"}
</script>
</div>

### 4. Full K-Means (seeded) <span class="badge badge-med">Medium</span>

K-means++ init + Lloyd iteration. Seed된 RNG로 반환 center가 재현 가능합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"kmeans_centers","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef kmeans_centers(x, k, seed=0, max_iter=100, tol=1e-4):\n    # seed np.random.default_rng(seed); k-means++ init, then alternate assign/update; return centers (k,D)\n    pass","tests":[{"args":[[[0.0,0.0],[0.1,0.1],[0.2,0.0],[5.0,5.0],[5.1,4.9],[4.9,5.0]],2,0],"expect":[[5.0,4.966666666666667],[0.10000000000000002,0.03333333333333333]],"tol":0.0001},{"args":[[[0.0,0.0],[1.0,0.0],[0.0,1.0],[10.0,10.0],[10.0,11.0],[11.0,10.0]],2,0],"expect":[[10.333333333333334,10.333333333333334],[0.3333333333333333,0.3333333333333333]],"tol":0.0001}],"solution":"import numpy as np\n\ndef kmeans_centers(x, k, seed=0, max_iter=100, tol=1e-4):\n    x = np.asarray(x, dtype=float)\n    rng = np.random.default_rng(seed)\n    n = x.shape[0]\n    def sqd(a, b):\n        a2 = np.sum(a ** 2, axis=1, keepdims=True)\n        b2 = np.sum(b ** 2, axis=1)\n        return np.maximum(a2 + b2[None, :] - 2.0 * a @ b.T, 0.0)\n    c = np.empty((k, x.shape[1]))\n    c[0] = x[rng.integers(n)]\n    closest = sqd(x, c[:1]).ravel()\n    for i in range(1, k):\n        probs = closest / closest.sum()\n        c[i] = x[rng.choice(n, p=probs)]\n        closest = np.minimum(closest, sqd(x, c[i:i + 1]).ravel())\n    labels = np.zeros(n, dtype=np.int64)\n    for _ in range(max_iter):\n        d = sqd(x, c)\n        labels = np.argmin(d, axis=1)\n        new_c = np.empty_like(c)\n        for j in range(k):\n            members = x[labels == j]\n            new_c[j] = members.mean(0) if len(members) else x[rng.integers(n)]\n        shift = np.linalg.norm(new_c - c)\n        c = new_c\n        if shift < tol:\n            break\n    return c"}
</script>
</div>

## 면접관이 지켜보는 흔한 버그

- **Naive distances:** 명시적인 $(N,K,D)$ broadcast 차이는 메모리를 낭비합니다; 전개 + matmul을 쓰세요.
- **Empty clusters:** 빈 slice의 `mean`은 `nan`입니다 — 감지해서 re-seed하세요 (random point나 가장 먼 점으로).
- **`argmin` axis:** $N$이 아니라 $K$ (`axis=1`) 기준이어야 합니다.
- **Convergence test:** 단지 `max_iter`가 아니라 centroid shift나 assignment 변화를 비교하세요.
- **Reproducibility:** RNG를 seed하세요; k-means는 init에 민감하므로 실제 실행은 `n_init` restart를 쓰고 가장 낮은 inertia를 유지합니다.

## Q&A

<details class="qa"><summary>k-means++는 왜 도움이 되고, 비용은 무엇인가요?</summary>
<div class="qa-body">

**짧게:** $D^2$-가중 샘플링으로 초기 centroid를 퍼뜨리면 random init이 빠지는 나쁜 local minima를 피하며, 비용은 추가 $O(NK)$ seeding pass 한 번뿐입니다.

**깊게:** random init은 흔히 같은 실제 cluster 안에 centroid 두 개를 떨어뜨려, 다른 cluster를 주인 없이 남깁니다 — Lloyd가 탈출할 수 없는 local optimum이죠. k-means++는 멀리 떨어진 점이 선택될 가능성을 높여 $O(\log K)$-competitive 기대 해와 보통 더 적은 수렴 iteration을 냅니다. seeding은 메인 루프에 비해 싸므로 사실상 공짜 보험이며 어디서나 기본값입니다.
</div></details>

<details class="qa"><summary>K는 어떻게 고르나요?</summary>
<div class="qa-body">

**짧게:** inertia의 elbow, silhouette score, 또는 gap statistic — 어느 것도 결정적이지 않으니 domain knowledge를 쓰세요.

**깊게:** inertia는 $K$가 커질수록 단조 감소하므로, marginal gain이 평평해지는 "elbow"를 찾습니다. Silhouette는 점마다 cohesion 대 separation을 측정하고(범위 $[-1,1]$), 높을수록 좋으며 over-clustering을 벌합니다. gap statistic은 inertia를 uniform-random 참조와 비교합니다. 실무에서는 downstream 작업(codebook size, anchor prototype 수)이 종종 $K$를 직접 고정합니다.
</div></details>

<details class="qa"><summary>K-means vs Gaussian Mixture Model?</summary>
<div class="qa-body">

**짧게:** k-means는 EM으로 적합된 GMM의 hard-assignment, equal-isotropic-variance 특수 경우입니다.

**깊게:** GMM의 E-step은 hard argmin 대신 soft responsibility $p(k\mid x_i)$를 계산하고, M-step은 평균 *과* full covariance, mixing weight를 업데이트합니다. covariance를 $\sigma^2 I$로 두고 $\sigma\to 0$으로 보내면 k-means가 복원됩니다. GMM은 타원형 cluster를 다루고 보정된 membership 확률을 주며; k-means는 cluster가 대략 구형이고 크기가 같을 때 더 빠르고 단순합니다.
</div></details>

### Follow-ups
- **Non-spherical / varying density cluster?** k-means는 실패합니다 (isotropic, 동일 크기 blob을 가정) — GMM, DBSCAN, 또는 spectral clustering을 쓰세요.
- **Scaling features?** 먼저 standardize하세요; k-means는 raw Euclidean 거리를 쓰므로 스케일 안 된 feature가 지배합니다.
- **Streaming / huge N?** MiniBatch k-means는 샘플된 batch로 centroid를 업데이트합니다.
- **CV connection?** Color quantization, bag-of-visual-words, VQ-VAE codebook 모두 k-means 스타일 clustering에 기댑니다.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Objective | inertia $\sum \lVert x_i - c_{a_i}\rVert^2$ 최소화 |
| Steps | assign (argmin dist) ↔ update (cluster mean) |
| Distance | $\lVert x\rVert^2+\lVert c\rVert^2-2xc^\top$, $\ge 0$로 clamp |
| Init | k-means++: $D^2$-가중 샘플링 |
| Empty cluster | random / 가장 먼 점으로 re-seed |
| Invariant | inertia는 절대 증가하지 않음 |
| Complexity | iter당 $O(NKD)$, $O(NK)$ 메모리 |
| Converges to | **local** optimum → `n_init` restart 사용 |

**Cross-links:** [Probability & Statistics](#/foundations/probability-statistics) · [Vision Foundation Models](#/cv/foundation-models) · [The ML Coding Round](#/ml-coding/intro)
