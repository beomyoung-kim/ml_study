# Conv & Pooling From Scratch

> [!TIP] Say this first
> "I'll write the obviously-correct nested-loop conv first, then the fast path everyone actually ships: **im2col** unrolls every receptive field into a column, so convolution becomes one big matrix multiply — which is exactly how cuDNN's GEMM path works." That framing signals you know why the fast version exists.

Implement `conv2d`, `max_pool2d`, and `avg_pool2d` in NumPy for `(N, C, H, W)` tensors, supporting stride, padding, and bias. The interviewer is checking that you understand the operation beneath the framework — and that you know the classic im2col→GEMM trick.

## The math & output size

A conv slides a $(C_\text{in}, K_H, K_W)$ filter over the input; each output pixel is the sum of the elementwise product over its receptive field:

$$
\text{out}[n,o,i,j] = b_o + \sum_{c,u,v} x[n,c,\,i\cdot s+u,\,j\cdot s+v]\cdot w[o,c,u,v]
$$

Output spatial size (memorize — off-by-one here is the #1 bug):

$$
H_\text{out} = \left\lfloor \frac{H + 2p - K_H}{s} \right\rfloor + 1
$$

With dilation $d$, replace $K$ by the effective kernel $d(K-1)+1$. "Same" padding at stride 1 is $p = (K-1)/2$.

## Naive convolution (correct first)

```python
import numpy as np


def _pad(x, p):
    if p == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))  # zero-pad H, W only


def conv2d_naive(x, w, b=None, stride=1, padding=0):
    """x:(N,Cin,H,W)  w:(Cout,Cin,KH,KW)  b:(Cout,) -> (N,Cout,Hout,Wout)."""
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    xp = _pad(x, padding)
    Ho = (H + 2 * padding - KH) // stride + 1
    Wo = (W + 2 * padding - KW) // stride + 1
    out = np.zeros((N, Cout, Ho, Wo), dtype=x.dtype)
    for n in range(N):
        for o in range(Cout):
            for i in range(Ho):
                for j in range(Wo):
                    hs, ws = i * stride, j * stride
                    field = xp[n, :, hs:hs + KH, ws:ws + KW]  # (Cin,KH,KW)
                    out[n, o, i, j] = np.sum(field * w[o])
    if b is not None:
        out += b[None, :, None, None]
    return out
```

Correct but slow: five nested loops, $O(N\,C_\text{out}\,C_\text{in}\,K^2\,H_\text{out}W_\text{out})$.

## im2col → GEMM (the fast path)

Unroll every receptive field into a column of length $C_\text{in}K_HK_W$; convolution becomes $W_\text{mat} \cdot \text{cols}$.

```python
def im2col(x, KH, KW, stride, padding):
    """x:(N,C,H,W) -> cols:(N, C*KH*KW, Hout*Wout)."""
    N, C, H, W = x.shape
    xp = _pad(x, padding)
    Ho = (H + 2 * padding - KH) // stride + 1
    Wo = (W + 2 * padding - KW) // stride + 1
    cols = np.empty((N, C * KH * KW, Ho * Wo), dtype=x.dtype)
    p = 0
    for i in range(Ho):
        for j in range(Wo):
            hs, ws = i * stride, j * stride
            patch = xp[:, :, hs:hs + KH, ws:ws + KW]   # (N,C,KH,KW)
            cols[:, :, p] = patch.reshape(N, -1)
            p += 1
    return cols, Ho, Wo


def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    N = x.shape[0]
    Cout, _, KH, KW = w.shape
    cols, Ho, Wo = im2col(x, KH, KW, stride, padding)      # (N, CKK, HW)
    w_mat = w.reshape(Cout, -1)                            # (Cout, CKK)
    out = np.einsum("oc,ncp->nop", w_mat, cols)            # (N, Cout, HW)
    if b is not None:
        out += b[None, :, None]
    return out.reshape(N, Cout, Ho, Wo)
```

The loop now only builds the column matrix; the heavy arithmetic is a single batched matmul — the same reduction cuDNN and every BLAS-backed framework make so it can call a tuned GEMM kernel.

## Pooling

```python
def _pool(x, k, stride, padding, reduce_fn):
    N, C, H, W = x.shape
    stride = stride or k
    xp = _pad(x, padding)
    Ho = (H + 2 * padding - k) // stride + 1
    Wo = (W + 2 * padding - k) // stride + 1
    out = np.empty((N, C, Ho, Wo), dtype=x.dtype)
    for i in range(Ho):
        for j in range(Wo):
            hs, ws = i * stride, j * stride
            win = xp[:, :, hs:hs + k, ws:ws + k].reshape(N, C, -1)
            out[:, :, i, j] = reduce_fn(win, axis=-1)
    return out


def max_pool2d(x, k=2, stride=None, padding=0):
    return _pool(x, k, stride, padding, np.max)


def avg_pool2d(x, k=2, stride=None, padding=0):
    return _pool(x, k, stride, padding, np.mean)
```

## Sanity check

```python
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 8, 8))
    w = rng.standard_normal((4, 3, 3, 3)); b = rng.standard_normal(4)
    y1 = conv2d_naive(x, w, b, stride=1, padding=1)
    y2 = conv2d_im2col(x, w, b, stride=1, padding=1)
    assert y1.shape == (2, 4, 8, 8)
    assert np.allclose(y1, y2, atol=1e-10)     # two paths must agree
    assert max_pool2d(x, 2, 2).shape == (2, 3, 4, 4)
    print("naive == im2col, OK")
```

> [!NOTE] Framework one-liner
> `F.conv2d(x, w, b, stride, padding, dilation, groups)`; `F.max_pool2d`, `F.avg_pool2d`. cuDNN autotunes among im2col-GEMM, implicit-GEMM, FFT, and Winograd per shape/GPU.

## Common bugs interviewers watch for

- **Output-size formula:** integer floor division, and don't forget the `+1`.
- **Cross-correlation vs true convolution:** deep learning "conv" does *not* flip the kernel — it's cross-correlation. Say so; nobody flips in practice.
- **Padding axes:** pad only $H,W$, never $N,C$.
- **Bias broadcast shape:** `b[None, :, None, None]` for `(N,Cout,H,W)`.
- **`groups`:** grouped/depthwise conv splits channels into independent groups; forgetting it breaks MobileNet-style layers.

## Q&A

<details class="qa"><summary>Why im2col if it duplicates data and blows up memory?</summary>
<div class="qa-body">

**Short:** because it converts convolution into a single dense GEMM, and decades of BLAS/cuDNN tuning make GEMM near-peak-FLOPS — usually a massive net win despite the memory overhead.

**Deep:** im2col replicates each input pixel up to $K_H K_W$ times, so the column matrix is $\sim K^2$ larger than the input — real pressure for high-resolution feature maps. The alternatives cuDNN offers: **implicit GEMM** (compute columns on the fly, no full materialization), **Winograd** (fewer multiplies for small kernels like 3×3, the common case), and **FFT** (large kernels). The library autotunes per shape. On-device/custom-CUDA work is where you actually hand-pick among these; in research code you let the framework decide.
</div></details>

<details class="qa"><summary>How does the backward pass work?</summary>
<div class="qa-body">

**Short:** the weight gradient is another im2col-based matmul; the input gradient is a "col2im" scatter, which is equivalent to a stride/transposed convolution of the upstream gradient with the flipped kernel.

**Deep:** with cols $(C_\text{in}K^2, HW)$ cached from forward, $\partial L/\partial W = \text{grad\_out} \cdot \text{cols}^\top$. The input gradient reshapes $W^\top \cdot \text{grad\_out}$ back into overlapping patches and *accumulates* them into the padded input grid (col2im) — overlaps sum. In an interview, stating "weight grad via im2col, input grad via col2im / transposed conv" is enough; you rarely write it out.
</div></details>

<details class="qa"><summary>Depthwise-separable convolution — why is it efficient?</summary>
<div class="qa-body">

**Short:** it factorizes a standard conv into a per-channel spatial conv (depthwise) plus a 1×1 cross-channel conv (pointwise), cutting FLOPs by roughly $\frac{1}{C_\text{out}} + \frac{1}{K^2}$.

**Deep:** a standard $K\times K$ conv mixes space and channels jointly at cost $C_\text{in}C_\text{out}K^2HW$. Depthwise applies one $K\times K$ kernel per input channel ($C_\text{in}K^2HW$), then pointwise 1×1 mixes channels ($C_\text{in}C_\text{out}HW$). For 3×3 with hundreds of channels that's an ~8–9× reduction — the backbone of MobileNet/EfficientNet and on-device vision.
</div></details>

### Follow-ups
- **Same vs valid padding?** `p=(K-1)/2` keeps spatial size at stride 1; valid means `p=0`.
- **Dilation?** Effective kernel $d(K-1)+1$; used for larger receptive fields without downsampling (segmentation).
- **1×1 conv?** Pure channel projection = a per-pixel linear layer; free way to change channel count.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Output size | $\lfloor (H+2p-K)/s \rfloor + 1$ |
| Naive cost | $O(N C_o C_i K^2 H_o W_o)$ |
| Fast path | im2col → GEMM (or Winograd for 3×3) |
| "Same" pad (s=1) | $p=(K-1)/2$ |
| DL "conv" | cross-correlation (no kernel flip) |
| Backward | weight: im2col matmul · input: col2im |
| Depthwise + pointwise | ~8–9× fewer FLOPs vs 3×3 dense |

**Cross-links:** [CNNs, RNNs & Transformers](#/foundations/architectures) · [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) · [The ML Coding Round](#/ml-coding/intro)
