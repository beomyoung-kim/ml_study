# K-Means: Grouping Similar Points

> [!NOTE] Goal of this chapter
> K-Means is both the **easiest algorithm to implement from scratch** and the canonical example of the unsupervised learning introduced in [What Is Machine Learning?](#/foundations/what-is-ml)—finding structure without answer labels. We will learn how to group similar points without labels, moving from an animation to a two-step loop and then executable code. It is a low-pressure warm-up.

## What it does—and why

When data points are scattered in space, you may want to divide them into **natural groups, or clusters**: this group looks like A, that group like B. There are no answer labels; the grouping must come only from the points' locations. This is **clustering**, and K-Means is its most widely used method.

The idea behind K-Means is remarkably simple. Place $K$ **centroids**, each representing one cluster, and alternate two steps:

1. **Assign:** assign every point to its *nearest* centroid.
2. **Update:** move every centroid to the *mean position of the points assigned to it*.

Stop when the centroids no longer move.

<figure>
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- cluster A points (left-bottom) -->
  <g fill="#0ea5e9">
    <circle cx="110" cy="215" r="5"/><circle cx="150" cy="235" r="5"/><circle cx="130" cy="195" r="5"/>
    <circle cx="175" cy="220" r="5"/><circle cx="95" cy="240" r="5"/><circle cx="160" cy="200" r="5"/>
  </g>
  <!-- cluster B points (right-top) -->
  <g fill="#e0533f">
    <circle cx="450" cy="80" r="5"/><circle cx="490" cy="100" r="5"/><circle cx="470" cy="60" r="5"/>
    <circle cx="510" cy="85" r="5"/><circle cx="440" cy="110" r="5"/><circle cx="500" cy="55" r="5"/>
  </g>
  <!-- centroid 1 (moves into cluster A) -->
  <g stroke="#12a150" stroke-width="2.5" fill="none">
    <circle r="9"/><path d="M-13 0 H13 M0 -13 V13"/>
    <animateTransform attributeName="transform" type="translate" dur="5s" repeatCount="indefinite"
      values="300 150; 240 175; 175 200; 138 216; 138 216; 300 150" keyTimes="0;0.18;0.4;0.62;0.9;1"/>
  </g>
  <!-- centroid 2 (moves into cluster B) -->
  <g stroke="#6366f1" stroke-width="2.5" fill="none">
    <circle r="9"/><path d="M-13 0 H13 M0 -13 V13"/>
    <animateTransform attributeName="transform" type="translate" dur="5s" repeatCount="indefinite"
      values="340 150; 400 135; 450 100; 476 82; 476 82; 340 150" keyTimes="0;0.18;0.4;0.62;0.9;1"/>
  </g>
  <text x="320" y="285" text-anchor="middle" fill="#98a3b2">The two centroids (＋) alternate assignment and update, pulled toward the center of each group</text>
</svg>
<figcaption>The green and purple centroids begin in an unhelpful middle region. Repeating "assign nearby points → move to their mean" makes them converge to the two cluster centers. That simple loop is all of K-Means.</figcaption>
</figure>

```mermaid
flowchart LR
  I["Initialize K centroids<br/>(k-means++)"] --> A["① Assign<br/>each point → nearest centroid"]
  A --> U["② Update<br/>centroid → cluster mean"]
  U -->|until movement is negligible| A
  style A fill:#6366f1,color:#fff
  style U fill:#12a150,color:#fff
```

## What does it minimize?—inertia

The loop reduces the **sum of squared distances between each point and its assigned centroid**, called inertia or within-cluster sum of squares:

$$
J = \sum_{i=1}^{N} \lVert x_i - c_{a_i} \rVert^2,\qquad a_i = \arg\min_k \lVert x_i - c_k\rVert^2
$$

Here $a_i$ is the centroid index assigned to point $i$. The two steps are **coordinate descent** on $J$: with centroids fixed, the optimal assignment is the nearest one; with assignments fixed, the optimal centroid is the mean. Therefore, **inertia never increases**—a useful invariant to test. The destination is only a **local** rather than global minimum, however, so initialization matters.

## Distance-computation trick (vectorization)

A common mistake is materializing the difference for every point–centroid pair as an $(N,K,D)$ tensor. That wastes memory. Expanding the square instead produces the $(N,K)$ distance matrix with **one matrix multiplication**:

$$
\lVert x - c\rVert^2 = \lVert x\rVert^2 + \lVert c\rVert^2 - 2\,x\!\cdot\!c
$$

The cross-term $x c^\top$ is one GEMM, a large matrix multiplication. If broadcasting and matrix multiplication are not yet intuitive, read the [NumPy & Broadcasting Primer](#/ml-coding/numpy-primer) first.

> [!WARNING] Clamp distances to zero
> When a point almost coincides with a centroid, floating-point error can make the expanded expression slightly negative. Apply `np.maximum(d, 0)` so a later square root does not produce `nan`.

> [!TIP] Interview one-liner
> "Lloyd's algorithm alternates assignment and update until convergence; the only real traps are vectorizing the distance calculation and handling empty clusters." Naming those two traps instead of reciting the definition makes you sound like someone who has implemented it.

## Practice—implement, run, and test it

> [!TIP] How to use this section
> Every problem below has a **live Python editor with NumPy preloaded**. Write your solution and press **▶ Run tests** to see which cases pass. Open **Solution** if you are stuck, but try first—the struggle is the practice. The first run downloads the Python runtime and NumPy (about 15 MB); later runs are immediate.

Build from the bottom up: distance matrix → assignment → update → full loop.

### 1. Squared-Distance Matrix <span class="badge badge-easy">Easy</span>

Use the expansion $\lVert x\rVert^2+\lVert c\rVert^2-2xc^\top$ to compute squared distances of shape $(N,K)$, then clamp them to $\ge 0$.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sq_dists","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef sq_dists(x, c):\n    # x:(N,D) c:(K,D) -> (N,K) squared Euclidean via the expansion trick; clamp >= 0\n    pass","tests":[{"args":[[[0.0,0.0],[3.0,4.0]],[[0.0,0.0],[3.0,0.0]]],"expect":[[0.0,9.0],[25.0,16.0]]},{"args":[[[1.0,1.0]],[[1.0,1.0],[4.0,5.0]]],"expect":[[0.0,25.0]]},{"args":[[[2.0,0.0],[0.0,2.0]],[[0.0,0.0]]],"expect":[[4.0],[4.0]]}],"solution":"import numpy as np\n\ndef sq_dists(x, c):\n    x = np.asarray(x, dtype=float)\n    c = np.asarray(c, dtype=float)\n    x2 = np.sum(x ** 2, axis=1, keepdims=True)\n    c2 = np.sum(c ** 2, axis=1)\n    d = x2 + c2[None, :] - 2.0 * x @ c.T\n    return np.maximum(d, 0.0)"}
</script>
</div>

### 2. Assignment Step <span class="badge badge-easy">Easy</span>

Label every point with the index of its nearest centroid—`argmin` over the $K$ axis, `axis=1`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"assign_clusters","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef assign_clusters(x, c):\n    # squared distances to each centroid, then argmin over axis=1; return a list of ints\n    pass","tests":[{"args":[[[0.0,0.0],[0.2,0.0],[5.0,5.0],[4.8,5.0]],[[0.0,0.0],[5.0,5.0]]],"expect":[0,0,1,1]},{"args":[[[1.0,1.0],[9.0,9.0],[1.0,0.0]],[[9.0,9.0],[0.0,0.0]]],"expect":[1,0,1]},{"args":[[[0.0,0.0]],[[1.0,0.0],[0.0,1.0]]],"expect":[0]}],"solution":"import numpy as np\n\ndef assign_clusters(x, c):\n    x = np.asarray(x, dtype=float)\n    c = np.asarray(c, dtype=float)\n    x2 = np.sum(x ** 2, axis=1, keepdims=True)\n    c2 = np.sum(c ** 2, axis=1)\n    d = np.maximum(x2 + c2[None, :] - 2.0 * x @ c.T, 0.0)\n    return np.argmin(d, axis=1).tolist()"}
</script>
</div>

### 3. Update Step <span class="badge badge-easy">Easy</span>

Recompute each centroid as the mean of its assigned members. If a cluster is empty, this function has no information from which to choose a sensible new center, so retain the corresponding entry from `old_centroids`. If no old centroids were supplied, raise an error rather than silently returning zero.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"update_centroids","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef update_centroids(x, labels, k, old_centroids=None):\n    # mean per cluster; if a cluster is empty, retain its old centroid (or raise if unavailable)\n    pass","tests":[{"args":[[[0.0,0.0],[2.0,0.0],[10.0,10.0],[12.0,10.0]],[0,0,1,1],2],"expect":[[1.0,0.0],[11.0,10.0]]},{"args":[[[1.0,1.0],[3.0,3.0],[5.0,5.0]],[0,0,1],2],"expect":[[2.0,2.0],[5.0,5.0]]},{"args":[[[4.0,2.0],[2.0,4.0]],[1,1],2,[[9.0,9.0],[0.0,0.0]]],"expect":[[9.0,9.0],[3.0,3.0]]}],"solution":"import numpy as np\n\ndef update_centroids(x, labels, k, old_centroids=None):\n    x = np.asarray(x, dtype=float)\n    labels = np.asarray(labels, dtype=np.int64)\n    if x.ndim != 2 or labels.shape != (len(x),) or k <= 0:\n        raise ValueError('invalid x, labels, or k')\n    old = None if old_centroids is None else np.asarray(old_centroids, dtype=float)\n    if old is not None and old.shape != (k, x.shape[1]):\n        raise ValueError('old_centroids must have shape (k,D)')\n    new_c = np.empty((k, x.shape[1]), dtype=float)\n    for j in range(k):\n        members = x[labels == j]\n        if len(members):\n            new_c[j] = members.mean(0)\n        elif old is not None:\n            new_c[j] = old[j]\n        else:\n            raise ValueError('empty cluster requires old_centroids')\n    return new_c"}
</script>
</div>

### 4. Full K-Means (seeded) <span class="badge badge-med">Medium</span>

Implement k-means++ initialization followed by Lloyd iterations. **K-means++** chooses the first center randomly, then samples every later center with probability proportional to its squared distance from the existing centers—$D^2$ weighting. This spreads the centers apart and greatly reduces the chance of a poor local minimum. A seeded RNG makes the result reproducible.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"kmeans_centers","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef kmeans_centers(x, k, seed=0, max_iter=100, tol=1e-4):\n    # validated k-means++ init; if all remaining D^2 weights are zero choose an unused row; retain empty centers\n    pass","tests":[{"args":[[[0.0,0.0],[0.1,0.1],[0.2,0.0],[5.0,5.0],[5.1,4.9],[4.9,5.0]],2,0],"expect":[[5.0,4.966666666666667],[0.10000000000000002,0.03333333333333333]],"tol":0.0001},{"args":[[[0.0,0.0],[1.0,0.0],[0.0,1.0],[10.0,10.0],[10.0,11.0],[11.0,10.0]],2,0],"expect":[[10.333333333333334,10.333333333333334],[0.3333333333333333,0.3333333333333333]],"tol":0.0001},{"args":[[[2.0,2.0],[2.0,2.0],[2.0,2.0]],3,0],"expect":[[2.0,2.0],[2.0,2.0],[2.0,2.0]],"tol":0.0001}],"solution":"import numpy as np\n\ndef kmeans_centers(x, k, seed=0, max_iter=100, tol=1e-4):\n    x = np.asarray(x, dtype=float)\n    if x.ndim != 2 or len(x) == 0 or x.shape[1] == 0 or not np.all(np.isfinite(x)):\n        raise ValueError('x must be a finite non-empty (N,D) array')\n    n = len(x)\n    if not 1 <= k <= n or max_iter <= 0 or tol < 0:\n        raise ValueError('require 1 <= k <= N, max_iter > 0, tol >= 0')\n    rng = np.random.default_rng(seed)\n    def sqd(a, b):\n        a2 = np.sum(a ** 2, axis=1, keepdims=True)\n        b2 = np.sum(b ** 2, axis=1)\n        return np.maximum(a2 + b2[None, :] - 2.0 * a @ b.T, 0.0)\n    c = np.empty((k, x.shape[1]))\n    chosen = np.zeros(n, dtype=bool)\n    first = int(rng.integers(n)); chosen[first] = True; c[0] = x[first]\n    closest = sqd(x, c[:1]).ravel()\n    for i in range(1, k):\n        total = float(closest.sum())\n        if total > 0 and np.isfinite(total):\n            idx = int(rng.choice(n, p=closest / total))\n        else:\n            idx = int(np.flatnonzero(~chosen)[0])\n        chosen[idx] = True; c[i] = x[idx]\n        closest = np.minimum(closest, sqd(x, c[i:i + 1]).ravel())\n    for _ in range(max_iter):\n        labels = np.argmin(sqd(x, c), axis=1)\n        new_c = c.copy()\n        for j in range(k):\n            members = x[labels == j]\n            if len(members):\n                new_c[j] = members.mean(0)\n        shift = np.linalg.norm(new_c - c)\n        c = new_c\n        if shift <= tol:\n            break\n    return c"}
</script>
</div>

> [!NOTE] Framework one-liner
> Use `sklearn.cluster.KMeans(n_clusters=k, init="k-means++")`; at scale, use `MiniBatchKMeans` or FAISS k-means on GPU, often for vector-quantization codebooks.

## Common bugs interviewers watch for

- **Naive distance calculation:** an explicit $(N,K,D)$ difference tensor wastes memory. Use the expansion plus matrix multiplication.
- **Empty cluster:** retaining the old center preserves the non-increasing property of the standard Lloyd objective. Reseeding with a random or farthest point can help escape, but may increase inertia in that iteration, so treat it as an explicit policy and test it separately.
- **Wrong `argmin` axis:** reduce over $K$, not $N$—`axis=1`.
- **Convergence check:** do not rely only on `max_iter`; check centroid movement or assignment changes.
- **Reproducibility:** seed the RNG. K-Means is initialization-sensitive, so production code runs `n_init` restarts and keeps the solution with the lowest inertia.

## Q&A

<details class="qa"><summary>Why does k-means++ help, and what does it cost?</summary>
<div class="qa-body">

**Short:** $D^2$-weighted sampling spreads the initial centroids out and avoids poor local minima from random initialization, at the cost of one additional $O(NK)$ seeding pass.

**Deep:** Random initialization often drops two centroids into the same real cluster and leaves another cluster unrepresented—a local optimum that Lloyd's algorithm cannot escape. K-means++ increases the probability of selecting distant points, giving an expected $O(\log K)$-competitive solution and usually fewer iterations. Seeding is cheap relative to the main loop, so it is nearly free insurance and the standard default.
</div></details>

<details class="qa"><summary>How do you choose K?</summary>
<div class="qa-body">

**Short:** Use the inertia elbow, silhouette score, or gap statistic—but none is decisive, so combine them with domain knowledge.

**Deep:** Inertia decreases monotonically as $K$ grows, so look for the elbow where marginal gains flatten. Silhouette compares cohesion with separation for each point, ranges from $[-1,1]$, and penalizes over-clustering. The gap statistic compares inertia against a uniform random reference. In practice, the downstream task—codebook size or number of prototypes—often fixes $K$ directly.
</div></details>

<details class="qa"><summary>K-Means vs. a Gaussian Mixture Model (GMM)?</summary>
<div class="qa-body">

**Short:** K-Means is the hard-assignment, equal-isotropic-variance special case of a GMM fitted with EM.

**Deep:** A GMM E-step computes responsibilities $p(k\mid x_i)$ instead of a hard argmin; its M-step updates means, covariances, and mixture weights. Setting every covariance to $\sigma^2 I$ and taking the small-variance limit connects it to K-Means. Responsibilities are posterior assignments inside the model, not automatically well-calibrated probabilities.
</div></details>

### Follow-ups

- **Nonspherical clusters or different densities?** K-Means fails because of its isotropic, similarly sized-cluster bias. Use a GMM, DBSCAN, or spectral clustering.
- **Feature scaling?** Standardize first; raw Euclidean distance is dominated by high-scale features.
- **Streaming or enormous N?** MiniBatch K-Means updates centroids from sampled batches.
- **Connection to CV?** Color quantization, bag-of-visual-words, and VQ-VAE codebooks all rely on K-Means-like clustering.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Objective | minimize inertia $\sum \lVert x_i - c_{a_i}\rVert^2$ |
| Two steps | assignment (argmin distance) ↔ update (cluster mean) |
| Distance | $\lVert x\rVert^2+\lVert c\rVert^2-2xc^\top$, clamped to $\ge 0$ |
| Initialization | k-means++: $D^2$-weighted sampling |
| Empty cluster | default: retain old center; reseeding is a separate heuristic and can increase inertia |
| Invariant | an exact Lloyd assignment+mean update that retains empty centers never increases inertia |
| Complexity | $O(NKD)$ per iteration, $O(NK)$ memory |
| Convergence | **local** optimum → restart with `n_init` |

**Next:** [NumPy & Broadcasting Primer](#/ml-coding/numpy-primer) · [Probability & Statistics](#/foundations/probability-statistics) · [Vision Foundation Models](#/cv/foundation-models) · [ML Coding Round](#/ml-coding/intro)
