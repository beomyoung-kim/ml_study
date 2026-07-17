# K-Means & K-Means++

> [!TIP] Say this first
> "Lloyd's algorithm alternates two steps until convergence: **assign** each point to its nearest centroid, then **update** each centroid to the mean of its members. The only real gotchas are vectorizing the distance matrix and handling empty clusters." That's the whole problem — deliver it calmly as the warm-up it usually is.

A frequent phone-screen warm-up. It looks trivial, so the signal is entirely in the details: a vectorized distance computation, the k-means++ initialization, empty-cluster handling, and knowing why you never compute Euclidean distance the naive way.

## The objective

K-means minimizes within-cluster sum of squares (inertia):

$$
J = \sum_{i=1}^{N} \lVert x_i - c_{a_i} \rVert^2,\qquad a_i = \arg\min_k \lVert x_i - c_k\rVert^2
$$

Each Lloyd iteration is coordinate descent on $J$ (fix centroids → optimal assignment; fix assignments → optimal centroids = the mean), so **inertia never increases** — a useful invariant to assert in tests. It converges to a *local* optimum.

## The distance trick

Never build an $(N,K,D)$ difference tensor. Expand the square:

$$
\lVert x - c\rVert^2 = \lVert x\rVert^2 + \lVert c\rVert^2 - 2\,x\!\cdot\!c
$$

The cross term is a single matmul $x c^\top$, giving an $(N,K)$ distance matrix in one GEMM.

```python
import numpy as np


def sq_dists(x, c):
    """x:(N,D) c:(K,D) -> (N,K) squared Euclidean via the expansion trick."""
    x2 = np.sum(x ** 2, axis=1, keepdims=True)     # (N,1)
    c2 = np.sum(c ** 2, axis=1)                     # (K,)
    d = x2 + c2[None, :] - 2.0 * x @ c.T            # (N,K)
    return np.maximum(d, 0.0)                       # clamp fp negatives near 0
```

> [!WARNING] Clamp the distances
> The expansion can produce tiny negative values from floating-point cancellation when a point sits almost exactly on a centroid. `np.maximum(d, 0)` prevents `nan` if you later take a `sqrt`.

## K-means++ initialization

Seed centroids far apart by sampling each new one with probability proportional to its squared distance from the nearest existing centroid. This gives an expected $O(\log K)$ approximation guarantee and far more stable convergence than random init (Arthur & Vassilvitskii, 2007).

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

**Complexity:** each iteration is $O(NKD)$ (the distance matmul dominates), over a small number of iterations. Memory is $O(NK)$ for the distance matrix.

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

> [!NOTE] Framework one-liner
> `sklearn.cluster.KMeans(n_clusters=k, init="k-means++")`; at scale, `MiniBatchKMeans` or FAISS k-means (GPU, used for vector-quantization codebooks).

## Common bugs interviewers watch for

- **Naive distances:** an explicit $(N,K,D)$ broadcast difference wastes memory; use the expansion + matmul.
- **Empty clusters:** `mean` of an empty slice is `nan` — detect and re-seed (to a random point or the farthest point).
- **`argmin` axis:** must be over `K` (`axis=1`), not `N`.
- **Convergence test:** compare centroid shift or assignment changes, not just `max_iter`.
- **Reproducibility:** seed the RNG; k-means is init-sensitive, so real runs use `n_init` restarts and keep the lowest inertia.

## Q&A

<details class="qa"><summary>Why does k-means++ help, and what does it cost?</summary>
<div class="qa-body">

**Short:** spreading initial centroids by $D^2$-weighted sampling avoids the bad local minima random init falls into, at the cost of one extra $O(NK)$ seeding pass.

**Deep:** random init often drops two centroids inside the same true cluster, leaving another cluster unclaimed — a local optimum Lloyd can't escape. k-means++ makes far-apart points likely to be chosen, yielding an $O(\log K)$-competitive expected solution and typically fewer iterations to converge. The seeding is cheap relative to the main loop, so it's essentially free insurance and the default everywhere.
</div></details>

<details class="qa"><summary>How do you choose K?</summary>
<div class="qa-body">

**Short:** elbow on inertia, silhouette score, or the gap statistic — none is definitive; use domain knowledge.

**Deep:** inertia decreases monotonically with $K$, so you look for the "elbow" where marginal gain flattens. Silhouette measures cohesion vs separation per point (range $[-1,1]$), higher is better and it penalizes over-clustering. The gap statistic compares inertia to a uniform-random reference. In practice the downstream task (codebook size, number of anchor prototypes) often fixes $K$ directly.
</div></details>

<details class="qa"><summary>K-means vs a Gaussian Mixture Model?</summary>
<div class="qa-body">

**Short:** k-means is the hard-assignment, equal-isotropic-variance special case of a GMM fit by EM.

**Deep:** GMM's E-step computes soft responsibilities $p(k\mid x_i)$ instead of a hard argmin, and the M-step updates means *and* full covariances and mixing weights. Taking the covariance to $\sigma^2 I$ with $\sigma\to 0$ recovers k-means. GMM handles elliptical clusters and gives calibrated membership probabilities; k-means is faster and simpler when clusters are roughly spherical and equal-sized.
</div></details>

### Follow-ups
- **Non-spherical / varying density clusters?** k-means fails (it assumes isotropic, equal-size blobs) — reach for GMM, DBSCAN, or spectral clustering.
- **Scaling features?** Standardize first; k-means uses raw Euclidean distance, so unscaled features dominate.
- **Streaming / huge N?** MiniBatch k-means updates centroids on sampled batches.
- **CV connection?** Color quantization, bag-of-visual-words, and VQ-VAE codebooks all lean on k-means-style clustering.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Objective | minimize inertia $\sum \lVert x_i - c_{a_i}\rVert^2$ |
| Steps | assign (argmin dist) ↔ update (cluster mean) |
| Distance | $\lVert x\rVert^2+\lVert c\rVert^2-2xc^\top$, clamp $\ge 0$ |
| Init | k-means++: $D^2$-weighted sampling |
| Empty cluster | re-seed to random / farthest point |
| Invariant | inertia never increases |
| Complexity | $O(NKD)$ per iter, $O(NK)$ memory |
| Converges to | a **local** optimum → use `n_init` restarts |

**Cross-links:** [Probability & Statistics](#/foundations/probability-statistics) · [Vision Foundation Models](#/cv/foundation-models) · [The ML Coding Round](#/ml-coding/intro)
