# Conv & Pooling 직접 구현

> [!TIP] 이것부터 말하세요
> "먼저 명백히 correct한 nested-loop conv를 쓰고, 그다음 모두가 실제로 ship하는 fast path를 쓰겠습니다: **im2col**은 모든 receptive field를 하나의 column으로 펼쳐서, convolution을 하나의 큰 matrix multiply로 만듭니다 — 이게 정확히 cuDNN의 GEMM 경로가 하는 방식입니다." 이 framing은 여러분이 fast 버전이 왜 존재하는지 안다는 signal입니다.

`(N, C, H, W)` tensor에 대해 NumPy로 `conv2d`, `max_pool2d`, `avg_pool2d`를 stride, padding, bias를 지원하며 구현하세요. 인터뷰어는 여러분이 framework 아래의 연산을 이해하는지 — 그리고 고전적인 im2col→GEMM 트릭을 아는지 확인합니다.

## 수학 & output size

Conv는 $(C_\text{in}, K_H, K_W)$ filter를 입력 위로 slide합니다; 각 output pixel은 자신의 receptive field에 대한 elementwise product의 합입니다:

$$
\text{out}[n,o,i,j] = b_o + \sum_{c,u,v} x[n,c,\,i\cdot s+u,\,j\cdot s+v]\cdot w[o,c,u,v]
$$

Output spatial size (외우세요 — 여기서의 off-by-one이 #1 버그입니다):

$$
H_\text{out} = \left\lfloor \frac{H + 2p - K_H}{s} \right\rfloor + 1
$$

Dilation $d$가 있으면 $K$를 effective kernel $d(K-1)+1$로 대체하세요. stride 1에서의 "Same" padding은 $p = (K-1)/2$입니다.

## Naive convolution (correct 먼저)

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

Correct하지만 느립니다: 다섯 개의 nested 루프, $O(N\,C_\text{out}\,C_\text{in}\,K^2\,H_\text{out}W_\text{out})$.

## im2col → GEMM (fast path)

모든 receptive field를 길이 $C_\text{in}K_HK_W$의 column으로 펼치세요; convolution은 $W_\text{mat} \cdot \text{cols}$가 됩니다.

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

이제 루프는 column matrix만 만들고; 무거운 산술은 하나의 batched matmul입니다 — cuDNN과 모든 BLAS-backed framework가 tuned GEMM kernel을 호출할 수 있도록 만드는 바로 그 reduction입니다.

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

> [!NOTE] Framework 한 줄
> `F.conv2d(x, w, b, stride, padding, dilation, groups)`; `F.max_pool2d`, `F.avg_pool2d`. cuDNN은 shape/GPU마다 im2col-GEMM, implicit-GEMM, FFT, Winograd 중에서 autotune합니다.

## 인터뷰어가 주시하는 흔한 버그

- **Output-size 공식:** integer floor division, 그리고 `+1`을 잊지 마세요.
- **Cross-correlation vs true convolution:** deep learning의 "conv"는 kernel을 *뒤집지 않습니다* — cross-correlation입니다. 그렇다고 말하세요; 실무에서 아무도 뒤집지 않습니다.
- **Padding axis:** $H,W$만 pad하고, $N,C$는 절대 하지 마세요.
- **Bias broadcast shape:** `(N,Cout,H,W)`에는 `b[None, :, None, None]`.
- **`groups`:** grouped/depthwise conv는 channel을 독립적인 group으로 나눕니다; 잊으면 MobileNet-style layer가 깨집니다.

## Q&A

<details class="qa"><summary>데이터를 복제하고 메모리를 부풀리는데도 왜 im2col인가요?</summary>
<div class="qa-body">

**짧게:** convolution을 하나의 dense GEMM으로 바꾸기 때문이고, 수십 년의 BLAS/cuDNN tuning이 GEMM을 near-peak-FLOPS로 만들어 — 메모리 overhead에도 불구하고 대개 막대한 net win입니다.

**깊게:** im2col은 각 입력 pixel을 최대 $K_H K_W$번 복제하므로 column matrix는 입력보다 $\sim K^2$ 큽니다 — 고해상도 feature map에는 실질적 압박입니다. cuDNN이 제공하는 대안들: **implicit GEMM**(column을 즉석에서 계산, 전체 materialization 없음), **Winograd**(3×3 같은 작은 kernel — 흔한 경우 — 에 대해 곱셈 수 감소), **FFT**(큰 kernel). 라이브러리가 shape마다 autotune합니다. On-device/custom-CUDA 작업이 실제로 이들 중에서 손으로 고르는 지점이고; research 코드에서는 framework가 결정하게 둡니다.
</div></details>

<details class="qa"><summary>backward pass는 어떻게 동작하나요?</summary>
<div class="qa-body">

**짧게:** weight gradient는 또 하나의 im2col 기반 matmul이고; input gradient는 "col2im" scatter인데, 이는 upstream gradient를 flipped kernel과 stride/transposed convolution하는 것과 동치입니다.

**깊게:** forward에서 cache된 cols $(C_\text{in}K^2, HW)$가 있으면 $\partial L/\partial W = \text{grad\_out} \cdot \text{cols}^\top$. Input gradient는 $W^\top \cdot \text{grad\_out}$을 겹치는 patch로 다시 reshape해서 padded 입력 grid에 *누적*합니다(col2im) — 겹치는 부분은 합해집니다. 인터뷰에서는 "weight grad는 im2col로, input grad는 col2im / transposed conv로"라고 말하면 충분합니다; 실제로 풀어 쓰는 일은 드뭅니다.
</div></details>

<details class="qa"><summary>Depthwise-separable convolution — 왜 효율적인가요?</summary>
<div class="qa-body">

**짧게:** 표준 conv를 per-channel spatial conv(depthwise)와 1×1 cross-channel conv(pointwise)로 factorize하여, FLOPs를 대략 $\frac{1}{C_\text{out}} + \frac{1}{K^2}$만큼 줄입니다.

**깊게:** 표준 $K\times K$ conv는 space와 channel을 $C_\text{in}C_\text{out}K^2HW$의 비용으로 함께 섞습니다. Depthwise는 입력 channel마다 하나의 $K\times K$ kernel을 적용하고($C_\text{in}K^2HW$), 그다음 pointwise 1×1이 channel을 섞습니다($C_\text{in}C_\text{out}HW$). 수백 개 channel의 3×3에서는 ~8–9× 감소 — MobileNet/EfficientNet과 on-device vision의 backbone입니다.
</div></details>

### Follow-ups
- **Same vs valid padding?** `p=(K-1)/2`는 stride 1에서 spatial size를 유지; valid는 `p=0`을 의미.
- **Dilation?** Effective kernel $d(K-1)+1$; downsampling 없이 더 큰 receptive field를 위해 씀(segmentation).
- **1×1 conv?** 순수 channel projection = per-pixel linear layer; channel 수를 바꾸는 공짜 방법.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Output size | $\lfloor (H+2p-K)/s \rfloor + 1$ |
| Naive cost | $O(N C_o C_i K^2 H_o W_o)$ |
| Fast path | im2col → GEMM (또는 3×3에는 Winograd) |
| "Same" pad (s=1) | $p=(K-1)/2$ |
| DL "conv" | cross-correlation (kernel flip 없음) |
| Backward | weight: im2col matmul · input: col2im |
| Depthwise + pointwise | 3×3 dense 대비 ~8–9× 적은 FLOPs |

**Cross-links:** [CNNs, RNNs & Transformers](#/foundations/architectures) · [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) · [The ML Coding Round](#/ml-coding/intro)
