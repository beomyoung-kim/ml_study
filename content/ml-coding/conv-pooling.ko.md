# Conv & Pooling 직접 구현

> [!NOTE] 이 챕터의 목표
> [CNN·RNN·Transformer](#/foundations/architectures)에서 "convolution이 무엇인지"는 그림으로 봤습니다. 여기서는 그걸 **NumPy로 직접 손으로 짜 봅니다** — 먼저 누구나 이해할 수 있는 느린 버전, 그다음 실무가 실제로 쓰는 빠른 버전(im2col). 인터뷰에서 "framework 없이 conv2d를 구현하라"는 단골 문제이기도 합니다. 사전 지식: [NumPy & 브로드캐스팅 프라이머](#/ml-coding/numpy-primer).

## convolution이 하는 일 (직관 먼저)

convolution은 작은 **필터(커널, kernel)** 를 이미지 위로 **한 칸씩 미끄러뜨리며(sliding)**, 매 위치에서 필터와 그 아래 픽셀들의 **가중합(내적)** 을 계산하는 것입니다. 즉 "이 국소 영역이 내 필터 패턴과 얼마나 닮았나"를 곳곳에서 재는 것이죠.

- **국소성(locality):** 각 출력은 입력의 **작은 영역(receptive field, 수용 영역)** 만 봅니다 — 이미지에서 가까운 픽셀끼리 관련 있다는 사실을 이용합니다.
- **가중치 공유(weight sharing):** 같은 필터를 모든 위치에 재사용 → 파라미터가 적고, "고양이 귀"가 화면 어디에 있든 똑같이 찾아냅니다(위치 불변성).

<figure>
<svg viewBox="0 0 620 230" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- input 4x4 grid -->
  <text x="90" y="18" text-anchor="middle" fill="#0ea5e9">입력 (4×4)</text>
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
  <text x="230" y="90" fill="#98a3b2">필터와</text><text x="230" y="108" fill="#98a3b2">내적 →</text>
  <!-- output 2x2 -->
  <text x="360" y="18" text-anchor="middle" fill="#12a150">출력 (2×2)</text>
  <g stroke="#12a150" stroke-width="1.2" fill="none">
    <rect x="320" y="45" width="80" height="80"/><line x1="360" y1="45" x2="360" y2="125"/><line x1="320" y1="85" x2="400" y2="85"/>
  </g>
  <rect x="320" y="45" width="40" height="40" fill="rgba(18,161,80,.3)">
    <animate attributeName="x" dur="3s" repeatCount="indefinite" values="320;360;320;360;320" keyTimes="0;0.25;0.5;0.75;1"/>
    <animate attributeName="y" dur="3s" repeatCount="indefinite" values="45;45;85;85;45" keyTimes="0;0.25;0.5;0.75;1"/>
  </rect>
  <text x="510" y="90" fill="#6366f1">각 창(window) 하나가</text>
  <text x="510" y="108" fill="#6366f1">출력 픽셀 하나</text>
</svg>
<figcaption>빨간 창이 입력 위를 미끄러지고, 매 위치의 가중합이 초록 출력 픽셀 하나가 됩니다. [architectures 챕터](#/foundations/architectures)의 커널 슬라이딩 애니메이션도 함께 보세요.</figcaption>
</figure>

> [!TIP] 면접 한 줄
> "먼저 명백히 correct한 nested-loop conv를 쓰고, 그다음 모두가 실제로 ship하는 fast path인 **im2col**을 쓰겠습니다 — 모든 receptive field를 column으로 펼쳐 convolution을 하나의 큰 matrix multiply(GEMM, 행렬곱)로 만드는 것으로, cuDNN의 GEMM 경로가 정확히 이 방식입니다." 이 framing은 fast 버전이 왜 존재하는지 안다는 signal입니다.

## 출력 크기 공식 (외우세요)

각 출력 픽셀은 자신의 receptive field에 대한 elementwise 곱의 합입니다:

$$
\text{out}[n,o,i,j] = b_o + \sum_{c,u,v} x[n,c,\,i\cdot s+u,\,j\cdot s+v]\cdot w[o,c,u,v]
$$

출력의 가로·세로 크기 (여기서의 off-by-one이 #1 버그입니다):

$$
H_\text{out} = \left\lfloor \frac{H + 2p - K_H}{s} \right\rfloor + 1
$$

$s$=stride(보폭), $p$=padding(가장자리 채우기). Dilation $d$가 있으면 $K$를 effective kernel $d(K-1)+1$로 대체합니다. stride 1·dilation 1·홀수 kernel이면 대칭 "same" padding이 $p=(K-1)/2$입니다. 짝수 kernel이나 stride>1에서는 framework가 출력 크기 $\lceil H/s\rceil$을 맞추려고 양쪽 padding을 비대칭으로 둘 수 있습니다.

## Practice — 직접 구현하고 실행·테스트

> [!TIP] 이 섹션 사용법
> 아래 각 문제에는 **라이브 Python 에디터**가 있습니다. 직접 풀이를 작성하고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 **Solution**을 열 수 있지만 먼저 직접 시도하세요. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

### 0. 워밍업 — 1D convolution <span class="badge badge-easy">Easy</span>

4D로 가기 전에 가장 단순한 형태부터. 짧은 배열 `x` 위로 짧은 필터 `w`를 미끄러뜨리며 내적하세요(valid, 즉 padding 없음). 출력 길이는 `len(x)-len(w)+1`. 이게 convolution의 뼈대입니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"conv1d","packages":["numpy"],"approx":true,"starter":"def conv1d(x, w):\n    # 1D cross-correlation (valid): output[i] = sum_j x[i+j]*w[j], 길이 = len(x)-len(w)+1\n    pass","tests":[{"args":[[1,2,3,4],[1,1]],"expect":[3.0,5.0,7.0]},{"args":[[1,2,3],[1,0,-1]],"expect":[-2.0]},{"args":[[1,2,3,4,5],[1]],"expect":[1.0,2.0,3.0,4.0,5.0]}],"solution":"import numpy as np\n\ndef conv1d(x, w):\n    x = np.asarray(x, float); w = np.asarray(w, float)\n    n = len(x) - len(w) + 1\n    return [float(np.dot(x[i:i+len(w)], w)) for i in range(n)]"}
</script>
</div>

### 1. Naive 2D convolution <span class="badge badge-easy">Easy</span>

이제 진짜. 다섯 개 nested 루프 버전 — fast보다 **correct가 먼저**입니다. 입력 `x:(N,Cin,H,W)`, 필터 `w:(Cout,Cin,KH,KW)`, bias `b:(Cout,)`. output channel마다 bias를 더합니다. 출력 크기 $\lfloor (H+2p-K)/s \rfloor + 1$를 기억하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"conv2d_naive","packages":["numpy"],"approx":true,"starter":"def conv2d_naive(x, w, b=None, stride=1, padding=0):\n    # x:(N,Cin,H,W)  w:(Cout,Cin,KH,KW)  b:(Cout,) -> (N,Cout,Hout,Wout); five nested loops\n    pass","tests":[{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],null,1,0],"expect":[[[[6.0,8.0],[12.0,14.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],[1.0],1,0],"expect":[[[[7.0,9.0],[13.0,15.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],null,1,0],"expect":[[[[12.0,16.0],[24.0,28.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],null,2,0],"expect":[[[[12.0]]]]}],"solution":"import numpy as np\n\ndef conv2d_naive(x, w, b=None, stride=1, padding=0):\n    x = np.asarray(x, dtype=float)\n    w = np.asarray(w, dtype=float)\n    N, Cin, H, W = x.shape\n    Cout, _, KH, KW = w.shape\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))\n    Ho = (H + 2 * padding - KH) // stride + 1\n    Wo = (W + 2 * padding - KW) // stride + 1\n    out = np.zeros((N, Cout, Ho, Wo), dtype=float)\n    for n in range(N):\n        for o in range(Cout):\n            for i in range(Ho):\n                for j in range(Wo):\n                    hs, ws = i * stride, j * stride\n                    field = xp[n, :, hs:hs + KH, ws:ws + KW]\n                    out[n, o, i, j] = np.sum(field * w[o])\n    if b is not None:\n        out += np.asarray(b, dtype=float)[None, :, None, None]\n    return out"}
</script>
</div>

Correct하지만 느립니다: 다섯 개의 nested 루프, $O(N\,C_\text{out}\,C_\text{in}\,K^2\,H_\text{out}W_\text{out})$.

### 2. im2col → GEMM (fast path) <span class="badge badge-med">Medium</span>

핵심 아이디어: 모든 receptive field를 길이 $C_\text{in}K_HK_W$의 **column으로 펼치면(im2col)**, convolution이 하나의 행렬곱 $W_\text{mat}\cdot\text{cols}$이 됩니다. 아래 그림이 그 "펼치기"입니다.

<figure>
<svg viewBox="0 0 620 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <text x="80" y="16" text-anchor="middle" fill="#0ea5e9">입력 patch들</text>
  <g stroke="#0ea5e9" stroke-width="1.1" fill="none">
    <rect x="30" y="28" width="100" height="100"/>
    <line x1="55" y1="28" x2="55" y2="128"/><line x1="80" y1="28" x2="80" y2="128"/><line x1="105" y1="28" x2="105" y2="128"/>
    <line x1="30" y1="53" x2="130" y2="53"/><line x1="30" y1="78" x2="130" y2="78"/><line x1="30" y1="103" x2="130" y2="103"/>
  </g>
  <rect x="30" y="28" width="50" height="50" fill="rgba(224,83,63,.25)" stroke="#e0533f"/>
  <path d="M135 78 C 200 78, 210 60, 250 60" stroke="#98a3b2" stroke-width="1.4" fill="none" marker-end="url(#im)"/>
  <text x="195" y="150" text-anchor="middle" fill="#98a3b2">각 patch → 한 열(column)로 펼침</text>
  <!-- columns matrix -->
  <text x="360" y="16" text-anchor="middle" fill="#6366f1">cols 행렬</text>
  <g stroke="#6366f1" stroke-width="1.1" fill="none">
    <rect x="300" y="28" width="120" height="100"/>
    <line x1="330" y1="28" x2="330" y2="128"/><line x1="360" y1="28" x2="360" y2="128"/><line x1="390" y1="28" x2="390" y2="128"/>
  </g>
  <rect x="300" y="28" width="30" height="100" fill="rgba(99,102,241,.22)"/>
  <text x="470" y="70" fill="#12a150" font-size="16">×W = 출력</text>
  <text x="470" y="92" fill="#98a3b2" font-size="11">= 하나의 행렬곱(GEMM)</text>
  <defs><marker id="im" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>im2col: 겹치는 receptive field들을 각각 하나의 열로 펼쳐 큰 행렬 <code>cols</code>를 만들면, convolution = 필터 행렬 × cols 라는 하나의 행렬곱이 됩니다. `conv2d_naive`와 결과가 정확히 같아야 합니다.</figcaption>
</figure>

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"conv2d_im2col","packages":["numpy"],"approx":true,"starter":"def conv2d_im2col(x, w, b=None, stride=1, padding=0):\n    # unroll receptive fields into columns, then one batched matmul (im2col -> GEMM)\n    pass","tests":[{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,0],[0,1]]]],null,1,0],"expect":[[[[6.0,8.0],[12.0,14.0]]]]},{"args":[[[[[1,2,3],[4,5,6],[7,8,9]]]],[[[[1,1],[1,1]]]],[0.5],1,0],"expect":[[[[12.5,16.5],[24.5,28.5]]]]},{"args":[[[[[1,2],[3,4]]]],[[[[1,0],[0,1]]]],null,1,1],"expect":[[[[1.0,2.0,0.0],[3.0,5.0,2.0],[0.0,3.0,4.0]]]]}],"solution":"import numpy as np\n\ndef conv2d_im2col(x, w, b=None, stride=1, padding=0):\n    x = np.asarray(x, dtype=float)\n    w = np.asarray(w, dtype=float)\n    N, C, H, W = x.shape\n    Cout = w.shape[0]\n    KH, KW = w.shape[2], w.shape[3]\n    xp = x if padding == 0 else np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))\n    Ho = (H + 2 * padding - KH) // stride + 1\n    Wo = (W + 2 * padding - KW) // stride + 1\n    cols = np.empty((N, C * KH * KW, Ho * Wo), dtype=float)\n    p = 0\n    for i in range(Ho):\n        for j in range(Wo):\n            hs, ws = i * stride, j * stride\n            patch = xp[:, :, hs:hs + KH, ws:ws + KW]\n            cols[:, :, p] = patch.reshape(N, -1)\n            p += 1\n    out = np.einsum(\"oc,ncp->nop\", w.reshape(Cout, -1), cols)\n    if b is not None:\n        out += np.asarray(b, dtype=float)[None, :, None]\n    return out.reshape(N, Cout, Ho, Wo)"}
</script>
</div>

이제 루프는 column matrix만 만들고, 무거운 산술은 하나의 batched matmul입니다 — cuDNN과 모든 BLAS-backed framework가 tuned GEMM kernel을 호출하게 만드는 바로 그 reduction입니다.

### 3. Max & average pooling <span class="badge badge-easy">Easy</span>

**Pooling(풀링)** 은 해상도를 줄이는 다운샘플링입니다. $k\times k$ window를 미끄러뜨리며(stride 기본값 $k$) 각 window를 channel마다 하나로 줄입니다 — max-pool은 최댓값, avg-pool은 평균.

padding의 의미는 연산마다 다릅니다. Max-pool 바깥은 실제 값 0이 아니라 `-inf`로 취급해야 음수 feature의 최댓값을 바꾸지 않습니다. Avg-pool은 framework의 `count_include_pad` 설정에 따라 pad를 분모에 포함할지 정합니다; 아래 구현은 0 padding을 포함해 평균냅니다.

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
    assert np.allclose(y1, y2, atol=1e-10)     # 두 경로가 일치해야 함
    assert max_pool2d(x, 2, 2).shape == (2, 3, 4, 4)
    print("naive == im2col, OK")
```

> [!NOTE] Framework 한 줄
> `F.conv2d(x, w, b, stride, padding, dilation, groups)`; `F.max_pool2d`, `F.avg_pool2d`. cuDNN은 shape/GPU마다 im2col-GEMM, implicit-GEMM, FFT, Winograd 중에서 autotune합니다.

## 인터뷰어가 주시하는 흔한 버그

- **Output-size 공식:** integer floor division, 그리고 `+1`을 잊지 마세요.
- **Cross-correlation vs true convolution:** deep learning의 "conv"는 kernel을 *뒤집지 않습니다* — cross-correlation입니다. 그렇게 말하세요; 실무에서 아무도 뒤집지 않습니다.
- **Padding axis:** $H,W$만 pad하고, $N,C$는 절대 하지 마세요.
- **Bias broadcast shape:** `(N,Cout,H,W)`에는 `b[None, :, None, None]`.
- **`groups`:** grouped/depthwise conv는 channel을 독립 group으로 나눕니다; 잊으면 MobileNet-style layer가 깨집니다.

## Q&A

<details class="qa"><summary>데이터를 복제하고 메모리를 부풀리는데도 왜 im2col인가요?</summary>
<div class="qa-body">

**짧게:** convolution을 하나의 dense GEMM으로 바꾸기 때문이고, 수십 년의 BLAS/cuDNN tuning이 GEMM을 near-peak-FLOPS로 만들어 — 메모리 overhead에도 불구하고 대개 막대한 net win입니다.

**깊게:** im2col은 각 입력 pixel을 최대 $K_H K_W$번 복제하므로 column matrix는 입력보다 $\sim K^2$ 큽니다. cuDNN의 대안들: **implicit GEMM**(column을 즉석 계산, 전체 materialization 없음), **Winograd**(3×3 같은 작은 kernel에 곱셈 수 감소), **FFT**(큰 kernel). 라이브러리가 shape마다 autotune합니다.
</div></details>

<details class="qa"><summary>backward pass는 어떻게 동작하나요?</summary>
<div class="qa-body">

**짧게:** weight gradient는 또 하나의 im2col 기반 matmul이고, input gradient는 "col2im" scatter로, 이는 upstream gradient를 flipped kernel과 transposed convolution하는 것과 동치입니다.

**깊게:** forward에서 cache한 cols $(C_\text{in}K^2, HW)$가 있으면 $\partial L/\partial W = \text{grad\_out} \cdot \text{cols}^\top$. Input gradient는 $W^\top \cdot \text{grad\_out}$을 겹치는 patch로 다시 reshape해 padded 입력 grid에 *누적*합니다(col2im). 인터뷰에서는 "weight grad는 im2col, input grad는 col2im/transposed conv"라고 말하면 충분합니다. 관련 개념: [선형대수 & 미적분](#/foundations/linear-algebra-calculus)의 VJP.
</div></details>

<details class="qa"><summary>Depthwise-separable convolution — 왜 효율적인가요?</summary>
<div class="qa-body">

**짧게:** 표준 conv를 per-channel spatial conv(depthwise)와 1×1 cross-channel conv(pointwise)로 factorize하여, FLOPs를 대략 $\frac{1}{C_\text{out}} + \frac{1}{K^2}$만큼 줄입니다.

**깊게:** 표준 $K\times K$ conv는 space와 channel을 $C_\text{in}C_\text{out}K^2HW$ 비용으로 함께 섞습니다. Depthwise는 입력 channel마다 하나의 $K\times K$($C_\text{in}K^2HW$), 그다음 pointwise 1×1이 channel을 섞습니다($C_\text{in}C_\text{out}HW$). 수백 channel의 3×3에서 ~8–9× 감소 — MobileNet/EfficientNet과 on-device vision의 backbone.
</div></details>

### Follow-ups
- **Same vs valid padding?** `p=(K-1)/2`는 stride 1·dilation 1·홀수 kernel의 대칭 경우입니다. 일반적인 `same`은 필요한 총 padding을 좌우/상하에 나누며 비대칭일 수 있고, valid는 `p=0`입니다.
- **Dilation?** Effective kernel $d(K-1)+1$; downsampling 없이 큰 receptive field(segmentation).
- **1×1 conv?** 순수 channel projection = per-pixel linear layer; channel 수를 바꾸는 값싼 방법.

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

**Cross-links:** [NumPy & 브로드캐스팅 프라이머](#/ml-coding/numpy-primer) · [CNN·RNN·Transformer](#/foundations/architectures) · [Mixed Precision & 효율화](#/foundations/mixed-precision-efficiency) · [ML 코딩 라운드](#/ml-coding/intro)
