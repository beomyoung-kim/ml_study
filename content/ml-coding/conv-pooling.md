# Implementing Convolution & Pooling

> [!NOTE] Goal of this chapter
> [CNNs, RNNs & Transformers](#/foundations/architectures) illustrated what convolution is. Here you will **implement it by hand in NumPy**—first a slow version anyone can understand, then the fast version used in practice, im2col. "Implement conv2d without a framework" is also a common interview problem. Prerequisite: [NumPy & Broadcasting Primer](#/ml-coding/numpy-primer).

## What convolution does (intuition first)

Convolution **slides** a small **filter, or kernel**, across an image one step at a time and computes the **weighted sum, or dot product**, of the filter and the pixels under it at every location. It measures, throughout the image, "How much does this local region resemble my filter's pattern?"

- **Locality:** each output sees only a **small input region, its receptive field**. This exploits the fact that nearby pixels are related.
- **Weight sharing:** reuse the same filter at every position. This requires few parameters and detects a cat ear the same way wherever it appears in the frame, giving translation equivariance and, after aggregation, useful location tolerance.

<figure>
<svg viewBox="0 0 620 230" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- input 4x4 grid -->
  <text x="90" y="18" text-anchor="middle" fill="#0ea5e9">input (4×4)</text>
  <g stroke="#0ea5e9" stroke-width="1.2" fill="none">
    <rect x="30" y="30" width="120" height="120"/>
    <line x1="60" y1="30" x2="60" y2="150"/><line x1="90" y1="30" x2="90" y2="150"/><line x1="120" y1="30" x2="120" y2="150"/>
    <line x1="30" y1="60" x2="150" y2="60"/><line x1="30" y1="90" x2="150" y2="90"/><line x1="30" y1="120" x2="150" y2="120"/>
  </g>
  <!-- sliding 2x2 window (animated) -->
  <rect width="60" height="60" fill="rgba(224,83,63,.28)" stroke="#e0533f" stroke-width="2">
    <animate attributeName="x" dur="3s" repeatCount="indefinite" values="30;90;30;90;30" keyTimes="0;0.25;0.5;0.75;1"/>
    <animate attributeName="y" dur="3s" repeatCount="indefinite" values="30;30;90;90;30" keyTimes="0;0.25;0.5;0.75;1"/>
  </rect>
  <text x="230" y="90" fill="#98a3b2">dot product</text><text x="230" y="108" fill="#98a3b2">with filter →</text>
  <!-- output 2x2 -->
  <text x="360" y="18" text-anchor="middle" fill="#12a150">output (2×2)</text>
  <g stroke="#12a150" stroke-width="1.2" fill="none">
    <rect x="320" y="45" width="80" height="80"/><line x1="360" y1="45" x2="360" y2="125"/><line x1="320" y1="85" x2="400" y2="85"/>
  </g>
  <rect x="320" y="45" width="40" height="40" fill="rgba(18,161,80,.3)">
    <animate attributeName="x" dur="3s" repeatCount="indefinite" values="320;360;320;360;320" keyTimes="0;0.25;0.5;0.75;1"/>
    <animate attributeName="y" dur="3s" repeatCount="indefinite" values="45;45;85;85;45" keyTimes="0;0.25;0.5;0.75;1"/>
  </rect>
  <text x="510" y="90" fill="#6366f1">each window produces</text>
  <text x="510" y="108" fill="#6366f1">one output pixel</text>
</svg>
<figcaption>The red window slides across the input, and the weighted sum at each location becomes one green output pixel. See the kernel-sliding animation in the [architectures chapter](#/foundations/architectures) as well.</figcaption>
</figure>

> [!TIP] Interview one-liner
> "I will first write an obviously correct nested-loop convolution, then the fast path everyone actually ships: **im2col**, which unfolds every receptive field into a column and turns convolution into one large matrix multiplication, or GEMM. That is exactly how cuDNN's GEMM path works." This framing signals that you understand why the fast version exists.

## Output-size formula (memorize it)

Every output pixel is the sum of elementwise products over its receptive field:

$$
\text{out}[n,o,i,j] = b_o + \sum_{c,u,v} x[n,c,\,i\cdot s+u,\,j\cdot s+v]\cdot w[o,c,u,v]
$$

The output height and width—the most common source of off-by-one bugs—are:

$$
H_\text{out} = \left\lfloor \frac{H + 2p - K_H}{s} \right\rfloor + 1
$$

Here $s$ is stride and $p$ is padding. With dilation $d$, replace $K$ by the effective kernel size $d(K-1)+1$. For stride 1, dilation 1, and an odd kernel, symmetric "same" padding is $p=(K-1)/2$. With even kernels or stride greater than 1, a framework may use asymmetric padding to produce output size $\lceil H/s\rceil$.

## Practice—implement, run, and test it

> [!TIP] How to use this section
> Every problem below has a **live Python editor**. Write your solution and press **▶ Run tests** to see which cases pass. Open **Solution** if you are stuck, but try first. The first run downloads a small Python runtime and NumPy (about 15 MB); later runs are immediate.

### 0. Warm-up—1D convolution <span class="badge badge-easy">Easy</span>

Start with the simplest form before moving to 4D. Slide a short filter `w` across a short array `x` and take a dot product at every valid position, with no padding. The output length is `len(x)-len(w)+1`. This is the skeleton of convolution.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"conv1d","packages":["numpy"],"approx":true,"starter":"def conv1d(x, w):\n    # 1D cross-correlation (valid): output[i] = sum_j x[i+j]*w[j], length = len(x)-len(w)+1\n    pass","tests":[{"args":[[1,2,3,4],[1,1]],"expect":[3.0,5.0,7.0]},{"args":[[1,2,3],[1,0,-1]],"expect":[-2.0]},{"args":[[1,2,3,4,5],[1]],"expect":[1.0,2.0,3.0,4.0,5.0]}],"solution":"import numpy as np\n\ndef conv1d(x, w):\n    x = np.asarray(x, float); w = np.asarray(w, float)\n    n = len(x) - len(w) + 1\n    return [float(np.dot(x[i:i+len(w)], w)) for i in range(n)]"}
</script>
</div>

### 1. Naive 2D convolution <span class="badge badge-easy">Easy</span>

Now the real implementation: five nested loops. **Correct comes before fast.** The input is `x:(N,Cin,H,W)`, filters are `w:(Cout,Cin,KH,KW)`, and bias is `b:(Cout,)`. Add one bias per output channel. Remember the output size $\lfloor (H+2p-K)/s \rfloor + 1$.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"conv2d_naive","packages":["numpy"],"approx":true,"starter":"def conv2d_naive(x, w, b=None, stride=1, padding=0):\n    # x:(N,Cin,H,W)  w:(Cout,Cin,KH,KW)  b:(Cout,) -> (N,Cout,Hout,Wout); five nested loops\n    pass","tests":[{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],null,1,0],"expect":[[[[6.0,8.0],[12.0,14.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],[1.0],1,0],"expect":[[[[7.0,9.0],[13.0,15.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],null,1,0],"expect":[[[[12.0,16.0],[24.0,28.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],null,2,0],"expect":[[[[12.0]]]]}],"solution":"import numpy as np\n\ndef conv2d_naive(x, w, b=None, stride=1, padding=0):\n    x = np.asarray(x, dtype=float)\n    w = np.asarray(w, dtype=float)\n    N, Cin, H, W = x.shape\n    Cout, _, KH, KW = w.shape\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))\n    Ho = (H + 2 * padding - KH) // stride + 1\n    Wo = (W + 2 * padding - KW) // stride + 1\n    out = np.zeros((N, Cout, Ho, Wo), dtype=float)\n    for n in range(N):\n        for o in range(Cout):\n            for i in range(Ho):\n                for j in range(Wo):\n                    hs, ws = i * stride, j * stride\n                    field = xp[n, :, hs:hs + KH, ws:ws + KW]\n                    out[n, o, i, j] = np.sum(field * w[o])\n    if b is not None:\n        out += np.asarray(b, dtype=float)[None, :, None, None]\n    return out"}
</script>
</div>

It is correct but slow: five nested loops and $O(N\,C_\text{out}\,C_\text{in}\,K^2\,H_\text{out}W_\text{out})$ work.

### 2. im2col → GEMM (fast path) <span class="badge badge-med">Medium</span>

The central idea is to **unfold every receptive field into a column** of length $C_\text{in}K_HK_W$. Convolution then becomes one matrix multiplication, $W_\text{mat}\cdot\text{cols}$. The diagram shows the unfolding.

<figure>
<svg viewBox="0 0 620 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <text x="80" y="16" text-anchor="middle" fill="#0ea5e9">input patches</text>
  <g stroke="#0ea5e9" stroke-width="1.1" fill="none">
    <rect x="30" y="28" width="100" height="100"/>
    <line x1="55" y1="28" x2="55" y2="128"/><line x1="80" y1="28" x2="80" y2="128"/><line x1="105" y1="28" x2="105" y2="128"/>
    <line x1="30" y1="53" x2="130" y2="53"/><line x1="30" y1="78" x2="130" y2="78"/><line x1="30" y1="103" x2="130" y2="103"/>
  </g>
  <rect x="30" y="28" width="50" height="50" fill="rgba(224,83,63,.25)" stroke="#e0533f"/>
  <path d="M135 78 C 200 78, 210 60, 250 60" stroke="#98a3b2" stroke-width="1.4" fill="none" marker-end="url(#im)"/>
  <text x="195" y="150" text-anchor="middle" fill="#98a3b2">flatten each patch into one column</text>
  <!-- columns matrix -->
  <text x="360" y="16" text-anchor="middle" fill="#6366f1">cols matrix</text>
  <g stroke="#6366f1" stroke-width="1.1" fill="none">
    <rect x="300" y="28" width="120" height="100"/>
    <line x1="330" y1="28" x2="330" y2="128"/><line x1="360" y1="28" x2="360" y2="128"/><line x1="390" y1="28" x2="390" y2="128"/>
  </g>
  <rect x="300" y="28" width="30" height="100" fill="rgba(99,102,241,.22)"/>
  <text x="470" y="70" fill="#12a150" font-size="16">×W = output</text>
  <text x="470" y="92" fill="#98a3b2" font-size="11">= one matrix multiply (GEMM)</text>
  <defs><marker id="im" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>im2col unfolds each overlapping receptive field into one column of a large <code>cols</code> matrix. Convolution becomes filter matrix × cols, one matrix multiplication. Its result must exactly match `conv2d_naive`.</figcaption>
</figure>

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"conv2d_im2col","packages":["numpy"],"approx":true,"starter":"def conv2d_im2col(x, w, b=None, stride=1, padding=0):\n    # unroll receptive fields into columns, then one batched matmul (im2col -> GEMM)\n    pass","tests":[{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],null,1,0],"expect":[[[[6.0,8.0],[12.0,14.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],[0.5],1,0],"expect":[[[[12.5,16.5],[24.5,28.5]]]]},{"args":[[[[[1,2],[3,4]]]],[[[[1,0],[0,1]]]],null,1,1],"expect":[[[[1.0,2.0,0.0],[3.0,5.0,2.0],[0.0,3.0,4.0]]]]}],"solution":"import numpy as np\n\ndef conv2d_im2col(x, w, b=None, stride=1, padding=0):\n    x = np.asarray(x, dtype=float)\n    w = np.asarray(w, dtype=float)\n    N, C, H, W = x.shape\n    Cout = w.shape[0]\n    KH, KW = w.shape[2], w.shape[3]\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))\n    Ho = (H + 2 * padding - KH) // stride + 1\n    Wo = (W + 2 * padding - KW) // stride + 1\n    cols = np.empty((N, C * KH * KW, Ho * Wo), dtype=float)\n    p = 0\n    for i in range(Ho):\n        for j in range(Wo):\n            hs, ws = i * stride, j * stride\n            patch = xp[:, :, hs:hs + KH, ws:ws + KW]\n            cols[:, :, p] = patch.reshape(N, -1)\n            p += 1\n    out = np.einsum(\"oc,ncp->nop\", w.reshape(Cout, -1), cols)\n    if b is not None:\n        out += np.asarray(b, dtype=float)[None, :, None]\n    return out.reshape(N, Cout, Ho, Wo)"}
</script>
</div>

The loops now only construct the column matrix; one batched matmul performs the heavy arithmetic. This reduction is what lets cuDNN and every BLAS-backed framework call a tuned GEMM kernel.

### 3. Max & average pooling <span class="badge badge-easy">Easy</span>

**Pooling** is downsampling that reduces spatial resolution. Slide a $k\times k$ window, with stride defaulting to $k$, and reduce every window independently per channel: maximum for max pooling, mean for average pooling.

Padding has different semantics across operations. Values outside a max-pool input should be treated as `-inf`, not zero, so they do not change the maximum of negative features. For average pooling, frameworks expose a `count_include_pad` option that controls whether padding contributes to the denominator; the implementation below includes zero padding in the average.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"max_pool2d","packages":["numpy"],"approx":true,"starter":"def max_pool2d(x, k=2, stride=None, padding=0):\n    # slide a k x k window (stride defaults to k) and take the max in each channel\n    pass","tests":[{"args":[[[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]],2,2,0],"expect":[[[[6.0,8.0],[14.0,16.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],2,1,0],"expect":[[[[5.0,6.0],[8.0,9.0]]]]},{"args":[[[[[1,2],[3,4]],[[10,20],[30,40]]]],2,2,0],"expect":[[[[4.0]],[[40.0]]]]},{"args":[[[[[-5,-4],[-3,-2]]]],2,1,1],"expect":[[[[-5.0,-4.0,-4.0],[-3.0,-2.0,-2.0],[-3.0,-2.0,-2.0]]]]}],"solution":"import numpy as np\n\ndef max_pool2d(x, k=2, stride=None, padding=0):\n    x = np.asarray(x, dtype=float)\n    N, C, H, W = x.shape\n    if stride is None:\n        stride = k\n    if k <= 0 or stride <= 0 or padding < 0:\n        raise ValueError('k/stride must be positive and padding non-negative')\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), constant_values=-np.inf)\n    Ho = (H + 2 * padding - k) // stride + 1\n    Wo = (W + 2 * padding - k) // stride + 1\n    out = np.empty((N, C, Ho, Wo), dtype=float)\n    for i in range(Ho):\n        for j in range(Wo):\n            hs, ws = i * stride, j * stride\n            win = xp[:, :, hs:hs + k, ws:ws + k].reshape(N, C, -1)\n            out[:, :, i, j] = win.max(axis=-1)\n    return out"}
</script>
</div>

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"avg_pool2d","packages":["numpy"],"approx":true,"starter":"def avg_pool2d(x, k=2, stride=None, padding=0):\n    # slide a k x k window (stride defaults to k) and take the mean in each channel\n    pass","tests":[{"args":[[[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]],2,2,0],"expect":[[[[3.5,5.5],[11.5,13.5]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],2,1,0],"expect":[[[[3.0,4.0],[6.0,7.0]]]]},{"args":[[[[[2,4],[6,8]]]],2,2,0],"expect":[[[[5.0]]]]}],"solution":"import numpy as np\n\ndef avg_pool2d(x, k=2, stride=None, padding=0):\n    x = np.asarray(x, dtype=float)\n    N, C, H, W = x.shape\n    stride = stride or k\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))\n    Ho = (H + 2 * padding - k) // stride + 1\n    Wo = (W + 2 * padding - k) // stride + 1\n    out = np.empty((N, C, Ho, Wo), dtype=float)\n    for i in range(Ho):\n        for j in range(Wo):\n            hs, ws = i * stride, j * stride\n            win = xp[:, :, hs:hs + k, ws:ws + k].reshape(N, C, -1)\n            out[:, :, i, j] = win.mean(axis=-1)\n    return out"}
</script>
</div>

## Sanity check

```python
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 8, 8))
    w = rng.standard_normal((4, 3, 3, 3)); b = rng.standard_normal(4)
    y1 = conv2d_naive(x, w, b, stride=1, padding=1)
    y2 = conv2d_im2col(x, w, b, stride=1, padding=1)
    assert y1.shape == (2, 4, 8, 8)
    assert np.allclose(y1, y2, atol=1e-10)     # the two paths must agree
    assert max_pool2d(x, 2, 2).shape == (2, 3, 4, 4)
    print("naive == im2col, OK")
```

> [!NOTE] Framework one-liner
> Use `F.conv2d(x, w, b, stride, padding, dilation, groups)` and `F.max_pool2d` / `F.avg_pool2d`. For each shape and GPU, cuDNN autotunes among im2col-GEMM, implicit GEMM, FFT, and Winograd.

## Common bugs interviewers watch for

- **Output-size formula:** use integer floor division and do not forget `+1`.
- **Cross-correlation vs. true convolution:** deep-learning "convolution" does *not* flip the kernel; it is cross-correlation. Say so. Production libraries do not flip it.
- **Padding axis:** pad $H,W$ only, never $N,C$.
- **Bias broadcast shape:** for `(N,Cout,H,W)`, use `b[None, :, None, None]`.
- **`groups`:** grouped and depthwise convolution partition channels into independent groups. Forgetting this breaks MobileNet-style layers.

## Q&A

<details class="qa"><summary>Why use im2col if it duplicates data and inflates memory?</summary>
<div class="qa-body">

**Short:** It turns convolution into one dense GEMM, and decades of BLAS/cuDNN tuning run GEMM near peak FLOPS. That is usually an enormous net win despite the memory overhead.

**Deep:** im2col duplicates each input pixel up to $K_HK_W$ times, so the column matrix can be roughly $K^2$ larger than the input. cuDNN alternatives include **implicit GEMM**, which computes columns on demand without fully materializing them; **Winograd**, which reduces multiplication count for small kernels such as 3×3; and **FFT**, which helps for large kernels. The library autotunes by shape.
</div></details>

<details class="qa"><summary>How does the backward pass work?</summary>
<div class="qa-body">

**Short:** The weight gradient is another im2col-based matrix multiplication. The input gradient uses a "col2im" scatter, equivalent to a transposed convolution of the upstream gradient with the flipped kernel.

**Deep:** If forward cached `cols` with shape $(C_\text{in}K^2, HW)$, then $\partial L/\partial W = \text{grad\_out} \cdot \text{cols}^\top$. For the input gradient, reshape $W^\top \cdot \text{grad\_out}$ into overlapping patches and *accumulate* them back into the padded input grid—col2im. In an interview, "weight gradient is an im2col matmul; input gradient is col2im/transposed convolution" is enough. Related concept: VJPs in [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus).
</div></details>

<details class="qa"><summary>Why is depthwise-separable convolution efficient?</summary>
<div class="qa-body">

**Short:** It factorizes a standard convolution into a per-channel spatial convolution, depthwise, and a 1×1 cross-channel convolution, pointwise. This reduces FLOPS by roughly $\frac{1}{C_\text{out}} + \frac{1}{K^2}$.

**Deep:** A standard $K\times K$ convolution mixes space and channels together for $C_\text{in}C_\text{out}K^2HW$ work. Depthwise applies one $K\times K$ filter per input channel for $C_\text{in}K^2HW$, then pointwise 1×1 convolution mixes channels for $C_\text{in}C_\text{out}HW$. With hundreds of channels and 3×3 kernels, this is about an 8–9× reduction—the backbone of MobileNet, EfficientNet, and on-device vision.
</div></details>

### Follow-ups

- **Same vs. valid padding?** `p=(K-1)/2` is the symmetric case for stride 1, dilation 1, and an odd kernel. General `same` padding divides the required total across left/right and top/bottom and may be asymmetric; valid means `p=0`.
- **Dilation?** Effective kernel size is $d(K-1)+1$; it expands the receptive field without downsampling, useful in segmentation.
- **1×1 convolution?** A pure channel projection—a per-pixel linear layer—and a cheap way to change channel count.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Output size | $\lfloor (H+2p-K)/s \rfloor + 1$ |
| Naive cost | $O(N C_o C_i K^2 H_o W_o)$ |
| Fast path | im2col → GEMM, or Winograd for 3×3 |
| "Same" pad (s=1) | $p=(K-1)/2$ |
| DL "conv" | cross-correlation; no kernel flip |
| Backward | weight: im2col matmul · input: col2im |
| Depthwise + pointwise | about 8–9× fewer FLOPS than dense 3×3 |

**Cross-links:** [NumPy & Broadcasting Primer](#/ml-coding/numpy-primer) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) · [ML Coding Round](#/ml-coding/intro)
