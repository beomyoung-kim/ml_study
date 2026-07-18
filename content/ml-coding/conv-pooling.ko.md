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

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열어볼 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

명백히 correct한 nested-loop conv를 먼저, 그다음 im2col→GEMM fast path, 그다음 두 pooling을 작성하세요.

### 1. Naive convolution <span class="badge badge-easy">Easy</span>

다섯 개 nested 루프 버전 — fast보다 correct가 먼저입니다. 각 output pixel은 자신의 receptive field에 대한 elementwise product의 합이고; output channel마다 bias를 더합니다. output size $\lfloor (H+2p-K)/s \rfloor + 1$를 기억하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"conv2d_naive","packages":["numpy"],"approx":true,"starter":"def conv2d_naive(x, w, b=None, stride=1, padding=0):\n    # x:(N,Cin,H,W)  w:(Cout,Cin,KH,KW)  b:(Cout,) -> (N,Cout,Hout,Wout); five nested loops\n    pass","tests":[{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],null,1,0],"expect":[[[[6.0,8.0],[12.0,14.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],[1.0],1,0],"expect":[[[[7.0,9.0],[13.0,15.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],null,1,0],"expect":[[[[12.0,16.0],[24.0,28.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],null,2,0],"expect":[[[[12.0]]]]}],"solution":"import numpy as np\n\ndef conv2d_naive(x, w, b=None, stride=1, padding=0):\n    x = np.asarray(x, dtype=float)\n    w = np.asarray(w, dtype=float)\n    N, Cin, H, W = x.shape\n    Cout, _, KH, KW = w.shape\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))\n    Ho = (H + 2 * padding - KH) // stride + 1\n    Wo = (W + 2 * padding - KW) // stride + 1\n    out = np.zeros((N, Cout, Ho, Wo), dtype=float)\n    for n in range(N):\n        for o in range(Cout):\n            for i in range(Ho):\n                for j in range(Wo):\n                    hs, ws = i * stride, j * stride\n                    field = xp[n, :, hs:hs + KH, ws:ws + KW]\n                    out[n, o, i, j] = np.sum(field * w[o])\n    if b is not None:\n        out += np.asarray(b, dtype=float)[None, :, None, None]\n    return out"}
</script>
</div>

Correct하지만 느립니다: 다섯 개의 nested 루프, $O(N\,C_\text{out}\,C_\text{in}\,K^2\,H_\text{out}W_\text{out})$.

### 2. im2col → GEMM (fast path) <span class="badge badge-med">Medium</span>

모든 receptive field를 길이 $C_\text{in}K_HK_W$의 column으로 펼치세요; convolution은 $W_\text{mat} \cdot \text{cols}$ — 하나의 batched matmul이 됩니다. `conv2d_naive`와 정확히 일치해야 합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"conv2d_im2col","packages":["numpy"],"approx":true,"starter":"def conv2d_im2col(x, w, b=None, stride=1, padding=0):\n    # unroll receptive fields into columns, then one batched matmul (im2col -> GEMM)\n    pass","tests":[{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],null,1,0],"expect":[[[[6.0,8.0],[12.0,14.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],[0.5],1,0],"expect":[[[[12.5,16.5],[24.5,28.5]]]]},{"args":[[[[[1,2],[3,4]]]],[[[[1,0],[0,1]]]],null,1,1],"expect":[[[[1.0,2.0,0.0],[3.0,5.0,2.0],[0.0,3.0,4.0]]]]}],"solution":"import numpy as np\n\ndef conv2d_im2col(x, w, b=None, stride=1, padding=0):\n    x = np.asarray(x, dtype=float)\n    w = np.asarray(w, dtype=float)\n    N, C, H, W = x.shape\n    Cout = w.shape[0]\n    KH, KW = w.shape[2], w.shape[3]\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))\n    Ho = (H + 2 * padding - KH) // stride + 1\n    Wo = (W + 2 * padding - KW) // stride + 1\n    cols = np.empty((N, C * KH * KW, Ho * Wo), dtype=float)\n    p = 0\n    for i in range(Ho):\n        for j in range(Wo):\n            hs, ws = i * stride, j * stride\n            patch = xp[:, :, hs:hs + KH, ws:ws + KW]\n            cols[:, :, p] = patch.reshape(N, -1)\n            p += 1\n    out = np.einsum(\"oc,ncp->nop\", w.reshape(Cout, -1), cols)\n    if b is not None:\n        out += np.asarray(b, dtype=float)[None, :, None]\n    return out.reshape(N, Cout, Ho, Wo)"}
</script>
</div>

이제 루프는 column matrix만 만들고; 무거운 산술은 하나의 batched matmul입니다 — cuDNN과 모든 BLAS-backed framework가 tuned GEMM kernel을 호출할 수 있도록 만드는 바로 그 reduction입니다.

### 3. Max & average pooling <span class="badge badge-easy">Easy</span>

$k\times k$ window를 slide하고(stride는 기본값 $k$) 각 window를 channel마다 reduce하세요 — max-pool은 `max`, avg-pool은 `mean`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"max_pool2d","packages":["numpy"],"approx":true,"starter":"def max_pool2d(x, k=2, stride=None, padding=0):\n    # slide a k x k window (stride defaults to k) and take the max in each channel\n    pass","tests":[{"args":[[[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]],2,2,0],"expect":[[[[6.0,8.0],[14.0,16.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],2,1,0],"expect":[[[[5.0,6.0],[8.0,9.0]]]]},{"args":[[[[[1,2],[3,4]],[[10,20],[30,40]]]],2,2,0],"expect":[[[[4.0]],[[40.0]]]]}],"solution":"import numpy as np\n\ndef max_pool2d(x, k=2, stride=None, padding=0):\n    x = np.asarray(x, dtype=float)\n    N, C, H, W = x.shape\n    stride = stride or k\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))\n    Ho = (H + 2 * padding - k) // stride + 1\n    Wo = (W + 2 * padding - k) // stride + 1\n    out = np.empty((N, C, Ho, Wo), dtype=float)\n    for i in range(Ho):\n        for j in range(Wo):\n            hs, ws = i * stride, j * stride\n            win = xp[:, :, hs:hs + k, ws:ws + k].reshape(N, C, -1)\n            out[:, :, i, j] = win.max(axis=-1)\n    return out"}
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
