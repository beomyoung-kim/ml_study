# NumPy & Broadcasting Primer

> [!NOTE] Goal of this chapter
> Every "from-scratch implementation" exercise that follows—convolution, attention, IoU, and k-means—assumes that you can **compute over whole arrays without writing loops**. Yet that intuition is rarely taught on its own. This chapter fills the gap. Three skills are enough: **reading shapes → vectorization → broadcasting**. Half of deep-learning code bugs are shape mismatches, so resolving them here makes everything that follows much easier.

## §0 · Why NumPy—arrays instead of loops

Adding a million numbers with a Python `for` loop is slow. NumPy turns that work into **one array operation compiled in C**, making it tens to hundreds of times faster. That is why ML code is written in a style that replaces loops with array operations—**vectorization**.

```python
import numpy as np
xs = [1, 2, 3, 4]
# Slow Python approach
total = 0
for x in xs:
    total += x * x
# Fast NumPy approach—the loop disappears
a = np.array(xs)
total = np.sum(a ** 2)     # 30
```

The key shift in thinking is from **"What should I do to each element?" to "What operation should I apply to the entire array?"**

## §1 · ndarray: shape · dtype · axis

NumPy's fundamental object is the **ndarray**, an N-dimensional array. Remember three properties:

- **shape**: the size of each dimension—a `(3,)` vector, a `(2,3)` matrix, or an `(N,C,H,W)` image batch.
- **dtype**: the element type, such as `float64` or `int64`. Confusing integers and floating-point values causes bugs, so make a habit of casting to `float`.
- **axis**: the direction an operation reduces. In 2D, `axis=0` goes **down across rows** and produces one result per column; `axis=1` goes **across columns** and produces one result per row.

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
  <text x="42" y="130" fill="#98a3b2" font-size="10">(per column)</text>
  <line x1="130" y1="188" x2="290" y2="188" stroke="#0ea5e9" stroke-width="2.5" marker-end="url(#d1)"/>
  <text x="330" y="150" fill="#0ea5e9">axis=1 (per row)</text>
  <text x="360" y="185" fill="#98a3b2" font-size="11">a.sum(axis=1) → shape (3,)</text>
  <text x="360" y="80" fill="#98a3b2" font-size="11">a.sum(axis=0) → shape (4,)</text>
  <defs>
    <marker id="d0" markerWidth="9" markerHeight="9" refX="4" refY="7" orient="auto"><path d="M0 0 L4 7 L8 0" fill="none" stroke="#e0533f" stroke-width="1.5"/></marker>
    <marker id="d1" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L7 4 L0 8" fill="none" stroke="#0ea5e9" stroke-width="1.5"/></marker>
  </defs>
</svg>
<figcaption>An axis is the dimension that disappears. Summing a <code>(3,4)</code> array over <code>axis=1</code> collapses the dimension of length 4 and leaves <code>(3,)</code>. With <code>keepdims=True</code>, it remains <code>(3,1)</code>, which is convenient for broadcasting.</figcaption>
</figure>

```python
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])   # shape (3,4)
a.sum(axis=0)      # [15,18,21,24]  (column sums, shape (4,))
a.sum(axis=1)      # [10,26,42]     (row sums, shape (3,))
a.argmax(axis=1)   # [3,3,3]        (index of each row's maximum)
```

## §2 · Indexing & slicing

```python
a[0]          # first row (shape (4,))
a[:, 1]       # second column (shape (3,))
a[0:2, 1:3]   # sub-block
a[a > 6]      # conditional (Boolean) indexing → elements greater than 6
```

## §3 · Vectorization—remove the loop

Elementwise operations (`+ - * / **`, `np.exp`, `np.maximum`, and so on) apply to the entire array at once. For example, let us divide every row by its sum to make it probability-like, without writing a loop.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"row_normalize","packages":["numpy"],"approx":true,"starter":"def row_normalize(x):\n    # x: 2D list/array. Divide every row by its row sum, without loops.\n    # Hint: a.sum(axis=1, keepdims=True) creates (N,1), so broadcasting can divide each row.\n    import numpy as np\n    a = np.asarray(x, dtype=float)\n    # TODO\n    return out.tolist()","tests":[{"args":[[[1,1,2],[2,2,4]]],"expect":[[0.25,0.25,0.5],[0.25,0.25,0.5]]},{"args":[[[1,3]]],"expect":[[0.25,0.75]]},{"args":[[[5,0,0],[0,0,10]]],"expect":[[1.0,0.0,0.0],[0.0,0.0,1.0]]}],"solution":"import numpy as np\n\ndef row_normalize(x):\n    a = np.asarray(x, dtype=float)\n    out = a / a.sum(axis=1, keepdims=True)\n    return out.tolist()"}
</script>
</div>

`a.sum(axis=1, keepdims=True)` has shape `(N,1)`, and `(N,M) / (N,1)` automatically divides each row by its matching sum. That is broadcasting.

## §4 · Broadcasting (the star of this chapter)

**Broadcasting** is the rule NumPy uses when operating on arrays with different shapes: it automatically **stretches** the smaller array until the shapes match. It is fast because no actual memory copy is made.

**Rule:** compare the two shapes from the right. Each pair of dimensions is compatible if either (1) they are equal or (2) one of them is 1. A dimension of size 1 expands to match the other size.

The most common pattern is adding a column vector `(N,1)` to a row vector `(1,M)`, which produces an `(N,M)` grid.

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
  <text x="450" y="30" text-anchor="middle" fill="#12a150">(3,4)—automatic expansion</text>
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
  <text x="300" y="205" text-anchor="middle" fill="#98a3b2">The column (blue) stretches right and the row (red) stretches down to fill a 3×4 grid—with zero loops</text>
</svg>
<figcaption>With broadcasting, one line—<code>col[:,None] + row[None,:]</code>—computes every (i,j) combination. IoU matrices, pairwise distances, and attention scores all use this pattern.</figcaption>
</figure>

```python
col = np.array([10, 20, 30])[:, None]   # shape (3,1)
row = np.array([1, 2, 3, 4])[None, :]   # shape (1,4)
col + row        # shape (3,4)—the diagram above
```

`[:, None]` (or `np.newaxis`) is the trick for **inserting a new dimension of length 1**. It lets you compute every pair without loops. K-means point-to-centroid distances and detection box-to-box IoU are both computed this way.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"pairwise_sq_dists","packages":["numpy"],"approx":true,"starter":"def pairwise_sq_dists(A, B):\n    # A:(n,d), B:(m,d). Return D:(n,m), where D[i,j] is the squared Euclidean distance between A[i] and B[j].\n    # No loops. Hint: A[:,None,:] - B[None,:,:] -> (n,m,d), then sum the squares over axis=-1.\n    import numpy as np\n    A = np.asarray(A, float); B = np.asarray(B, float)\n    # TODO\n    return D.tolist()","tests":[{"args":[[[0,0],[3,0]],[[0,0],[0,4]]],"expect":[[0.0,16.0],[9.0,25.0]]},{"args":[[[1,1]],[[1,1],[2,2]]],"expect":[[0.0,2.0]]},{"args":[[[0,0,0]],[[1,2,2]]],"expect":[[9.0]]}],"solution":"import numpy as np\n\ndef pairwise_sq_dists(A, B):\n    A = np.asarray(A, float); B = np.asarray(B, float)\n    diff = A[:, None, :] - B[None, :, :]   # (n,m,d)\n    D = (diff ** 2).sum(axis=-1)           # (n,m)\n    return D.tolist()"}
</script>
</div>

## §5 · Fancy indexing & one-hot encoding

An integer array can read or write many positions at once. A canonical example is converting labels to **one-hot** vectors.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"one_hot","packages":["numpy"],"approx":true,"starter":"def one_hot(labels, num_classes):\n    # labels: integer list, e.g. [0,2,1]. Convert each label to a one-hot row of length num_classes.\n    # Hint: create zeros((N, num_classes)), then set m[np.arange(N), labels] = 1.\n    import numpy as np\n    a = np.asarray(labels, dtype=int)\n    # TODO\n    return m.tolist()","tests":[{"args":[[0,2,1],3],"expect":[[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0]]},{"args":[[1,0],2],"expect":[[0.0,1.0],[1.0,0.0]]},{"args":[[2],4],"expect":[[0.0,0.0,1.0,0.0]]}],"solution":"import numpy as np\n\ndef one_hot(labels, num_classes):\n    a = np.asarray(labels, dtype=int)\n    m = np.zeros((a.size, num_classes))\n    m[np.arange(a.size), a] = 1.0\n    return m.tolist()"}
</script>
</div>

## §6 · A touch of numerical stability

`np.exp` overflows on large inputs. For operations such as softmax, **subtract the maximum** first; the value is mathematically unchanged, but the computation is more stable. Use `np.clip` to constrain a range, and add a tiny `eps` before `np.log` to avoid `log(0)`. See [Losses & Gradients](#/ml-coding/losses-gradients) for details.

```python
def softmax(z):                       # stable softmax
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

## Common bugs

- **Shape mismatch:** print `a.shape` before an operation. This catches half of all bugs.
- **Forgetting `keepdims`:** `sum(axis=1)` has shape `(N,)`, so it cannot directly divide `(N,M)` as intended. Use `keepdims=True` to keep `(N,1)`.
- **Integer division:** integer-array accumulation or `//` can corrupt values. Cast to `float`.
- **View vs. copy:** a slice is a **view** into the original, so modifying it changes the original. Use `.copy()` when needed.
- **Accidental broadcasting:** to build every pair from `(N,)` and `(N,)`, explicitly add `[:,None]` and `[None,:]`.

## Cheat-sheet

| Goal | Code |
| --- | --- |
| Inspect shape | `a.shape`, `a.dtype` |
| Sum/average over an axis | `a.sum(axis=0)`, `a.mean(axis=1, keepdims=True)` |
| Index of each row's maximum | `a.argmax(axis=1)` |
| Remove loops | elementwise operations · `@` (matmul) · fancy indexing |
| Insert a new axis | `a[:, None]`, `a[None, :]`, `np.newaxis` |
| Compute all pairs | `A[:,None,:] - B[None,:,:]` → broadcasting |
| Conditional selection | `np.where(cond, x, y)`, `a[a>0]` |
| Stable exp | subtract the maximum: `np.exp(z - z.max(...))` |
| Clip values | `np.clip(a, lo, hi)`, `np.maximum(0, a)` |

**Next:** [ML Coding Round](#/ml-coding/intro) · [Losses & Gradients](#/ml-coding/losses-gradients) · [Implementing Convolution & Pooling](#/ml-coding/conv-pooling)
