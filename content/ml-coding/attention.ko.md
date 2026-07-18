# Attention 직접 구현

> [!TIP] 이 말부터 시작하세요
> "Attention은 soft dictionary lookup입니다. query가 dot product로 모든 key에 점수를 매기고, softmax가 saturate되지 않도록 $1/\sqrt{d}$로 스케일한 뒤, softmax가 점수를 가중치로 바꾸고, 그 가중치로 value의 가중 평균을 취합니다." 그리고 네 줄을 적으세요. 앞으로 다룰 모든 VLM과 LLM이 이 위에 세워져 있습니다.

masking을 포함한 scaled dot-product attention과 multi-head attention을 구현합니다. 이것이 [Transformer block](#/ml-coding/transformer)과 현대 스택 전체를 이루는 *바로 그* 빌딩 블록입니다 — 개념적인 측면은 [CNNs, RNNs & Transformers](#/foundations/architectures)를 참고하세요.

## 수학

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

여기서 $Q\in\mathbb{R}^{T_q\times d_k}$, $K\in\mathbb{R}^{T_k\times d_k}$, $V\in\mathbb{R}^{T_k\times d_v}$ 입니다. Multi-head는 $d/h$-차원 부분공간에서 $h$개의 attention을 병렬로 돌린 뒤 concat합니다:

$$
\text{MHA}(x) = \big[\text{head}_1;\dots;\text{head}_h\big]W^O,\quad
\text{head}_i=\text{Attention}(xW_i^Q, xW_i^K, xW_i^V)
$$

직접 만져보세요 — query를 조절하면서 가중치와 출력이 어떻게 움직이는지 지켜보세요:

<div class="widget" data-widget="attention"></div>

> [!TIP] 라이브 코드 — 직접 구현하고 실행·테스트
> 아래 NumPy 블록은 **라이브 에디터**입니다. 본문을 채우고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

## Scaled dot-product attention (NumPy)

먼저 key 축에 대한 수치적으로 안정된 **softmax**:

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    # subtract the max for stability, exponentiate, then normalize\n    pass","tests":[{"args":[[[1,2,3]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[0,0,0]]],"expect":[[0.3333333333333333,0.3333333333333333,0.3333333333333333]]},{"args":[[[1000,1001,1002]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[-1,0,1],[1,0,-1]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218],[0.6652409557748218,0.24472847105479764,0.09003057317038046]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = np.asarray(x, dtype=float)\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)"}
</script>
</div>

그다음 **scaled dot-product attention** — 점수 계산, 스케일, softmax, value 가중합 (이 lab은 출력을 반환합니다; 참고 구현은 weights도 함께 반환합니다):

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sdpa","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = np.asarray(x, dtype=float)\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef sdpa(q, k, v, mask=None):\n    # scores = QK^T / sqrt(d); softmax over keys; return the weighted sum of V\n    pass","tests":[{"args":[[[1,0],[0,1]],[[1,0],[0,1]],[[1,2],[3,4]]],"expect":[[1.6604769013466862,2.6604769013466862],[2.3395230986533138,3.3395230986533138]]},{"args":[[[2,0,0]],[[2,0,0],[0,2,0],[0,0,2]],[[1,1],[2,2],[3,3]]],"expect":[[1.2485832277662388,1.2485832277662388]]},{"args":[[[1,1],[2,0]],[[1,0],[0,1]],[[10,0],[0,10]]],"expect":[[5.0,5.0],[8.04429682506957,1.9557031749304312]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = np.asarray(x, dtype=float)\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef sdpa(q, k, v, mask=None):\n    q, k, v = np.asarray(q, float), np.asarray(k, float), np.asarray(v, float)\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        mask = np.asarray(mask)\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    w = softmax(scores, axis=-1)\n    return w @ v"}
</script>
</div>

**왜 $\sqrt{d}$ 인가:** $q,k$의 원소가 단위 분산을 가지면 $q\cdot k$의 분산은 $\approx d$ 가 되어, raw 점수가 차원에 따라 커지고 softmax를 saturate된 near-one-hot 영역으로 밀어넣어 gradient가 소실됩니다. $\sqrt{d}$로 나누면 점수 분산이 $\approx 1$로 유지됩니다. *(verifiable)*

## Masking

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"causal_mask","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef causal_mask(T):\n    # lower-triangular boolean (T,T): True = allowed to attend (no peeking ahead)\n    pass","tests":[{"args":[2],"expect":[[1,0],[1,1]]},{"args":[3],"expect":[[1,0,0],[1,1,0],[1,1,1]]},{"args":[4],"expect":[[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]]}],"solution":"import numpy as np\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))"}
</script>
</div>

- **Causal mask**는 position $t$가 미래($>t$)를 보지 못하게 막습니다 — autoregressive decoding에 필수입니다.
- **Key-padding mask**는 `[PAD]` key로의 attention을 막습니다; shape은 `(B,1,1,Tk)`로, head와 query에 대해 broadcast됩니다.
- 논리 **AND**로 결합합니다 (둘 다 허용해야 함). mask는 항상 softmax **이전**에 적용하고, 막힌 항목을 큰 음수로 채워 가중치가 ~0이 되게 합니다.

## Multi-head attention

`mha`를 구현하세요; 하네스 `run_mha`는 seed 고정된 `(B,T,D)=(1,3,4)` 입력과 `H=2` head를 만들어 출력이 결정적이게 합니다 (헬퍼 `softmax`/`causal_mask`/`sdpa`는 제공됩니다):

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"run_mha","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))\n\ndef sdpa(q, k, v, mask=None):\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    return softmax(scores, axis=-1) @ v\n\ndef mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):\n    # split (B,T,D)->(B,H,T,Dh), run sdpa per head, merge heads, project by Wo\n    pass\n\ndef run_mha(causal):\n    np.random.seed(0)\n    B, T, D, H = 1, 3, 4, 2\n    x = np.random.randn(B, T, D)\n    Ws = [np.random.randn(D, D) * 0.1 for _ in range(4)]\n    return mha(x, *Ws, H, causal)","tests":[{"args":[false],"expect":[[[0.03190098881683708,-0.003883638346883957,0.009816921823006634,0.0008853701048203459],[0.032178757789826054,-0.003954663472188265,0.008879630160056167,0.0011910101137743716],[0.03203148496155561,-0.0038215886062526196,0.009498626868409653,0.0010096474007127018]]]},{"args":[true],"expect":[[[0.059733984570268545,-0.003891324159579386,0.01245601832145022,0.013606293601693605],[0.04292552968524655,-0.008439370891800252,0.018326373103144715,-0.004398635006322705],[0.03203148496155561,-0.0038215886062526196,0.009498626868409653,0.0010096474007127018]]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))\n\ndef sdpa(q, k, v, mask=None):\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    return softmax(scores, axis=-1) @ v\n\ndef mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):\n    B, T, D = x.shape\n    dh = D // num_heads\n    def split(t):\n        return t.reshape(B, T, num_heads, dh).transpose(0, 2, 1, 3)\n    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)\n    mask = causal_mask(T) if causal else None\n    out = sdpa(q, k, v, mask=mask)\n    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)\n    return out @ Wo\n\ndef run_mha(causal):\n    np.random.seed(0)\n    B, T, D, H = 1, 3, 4, 2\n    x = np.random.randn(B, T, D)\n    Ws = [np.random.randn(D, D) * 0.1 for _ in range(4)]\n    return mha(x, *Ws, H, causal)"}
</script>
</div>

head를 batch 같은 축으로 나누면 하나의 `@`로 모든 head를 한 번에 계산할 수 있습니다. **복잡도:** head당 점수 행렬에 대해 $O(T^2 d)$ 연산과 $O(T^2)$ 메모리가 듭니다.

## PyTorch 모듈

```python
import torch, torch.nn as nn, torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.dh = num_heads, d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, is_causal=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2)
                   for t in (q, k, v))              # (B,H,T,Dh)
        # fused, IO-aware, numerically stable (FlashAttention kernel):
        o = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))
```

## Sanity check

```python
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, T, D, H = 2, 5, 16, 4
    x = rng.standard_normal((B, T, D))
    Ws = [rng.standard_normal((D, D)) * 0.1 for _ in range(4)]
    y = mha(x, *Ws, num_heads=H, causal=True)
    assert y.shape == (B, T, D)
    # causal: position 0 must place zero weight on future keys
    q = rng.standard_normal((1, 1, 4, 8))
    _, w = sdpa(q, q, q, mask=causal_mask(4))
    assert np.allclose(w[0, 0, 0, 1:], 0.0, atol=1e-6)
    assert np.allclose(w.sum(-1), 1.0)          # weights are a distribution
    print("OK", y.shape)
```

> [!DANGER] 면접관이 지켜보는 흔한 버그
> $\sqrt{d}$ 스케일을 빼먹기; key 축이 아니라 query 축으로 softmax하기 (`axis=-1` = key); head를 merge할 때 엉뚱한 두 축을 transpose하기 (split의 역인 `(0,2,1,3)`이어야 함); softmax *이후*에 mask 적용하기; PyTorch에서 `reshape` 전에 `contiguous()`가 안 된 view; max를 빼지 않은 불안정한 softmax.

## Q&A

<details class="qa"><summary>왜 큰 attention 하나가 아니라 여러 head를 쓰나요?</summary>
<div class="qa-body">

**짧게:** head는 모델이 서로 다른 학습된 부분공간에서 서로 다른 관계(구문, 위치, coreference)에 동시에 attend한 뒤 이를 결합하게 해줍니다.

**깊게:** 단일 head는 query당 하나의 softmax 분포만 만듭니다 — attend할 "이유" 하나뿐이죠. $d$를 $d/h$ 크기의 $h$개 부분공간으로 나누면 *동일한* 총 연산량으로 $h$개의 독립적인 attention 패턴을 얻습니다 (projection이 비례해서 작아집니다). 경험적으로 서로 다른 head가 특화되며, 표현 공간에서의 값싼 ensemble의 한 형태입니다. $h\cdot(d/h)=d$ 이므로 비용은 그대로입니다.
</div></details>

<details class="qa"><summary>메모리 병목은 무엇이고 FlashAttention은 어떻게 해결하나요?</summary>
<div class="qa-body">

**짧게:** $T\times T$ 점수 행렬이 $O(T^2)$ 메모리입니다; FlashAttention은 이를 절대 materialize하지 않고, online(streaming) softmax로 타일 단위로 attention을 계산하며 빠른 SRAM에 머무릅니다.

**깊게:** naive attention은 전체 score/weight를 HBM에 씁니다 — memory-bandwidth-bound이고 시퀀스 길이에 대해 quadratic이죠. FlashAttention은 $Q,K,V$를 타일로 나누고, query 블록마다 running max와 running denominator를 유지하며(online으로 적용한 log-sum-exp trick), 전체 연산을 하나의 kernel로 fuse합니다. 출력은 동일하고, 추가 메모리는 $O(T)$이며 HBM round-trip이 훨씬 적습니다. `F.scaled_dot_product_attention`이 자동으로 여기로 dispatch합니다. *(verifiable)*
</div></details>

<details class="qa"><summary>Self-attention vs cross-attention?</summary>
<div class="qa-body">

**짧게:** self-attention은 Q, K, V를 같은 시퀀스에서 뽑고, cross-attention은 Q를 한 스트림에서, K, V를 다른 스트림에서 가져옵니다.

**깊게:** VLM decoder에서 text token은 self-attend하고, cross-attention은 text query가 image/vision token을 key/value로 삼아 끌어오게 합니다 — 언어를 픽셀에 grounding하는 메커니즘이죠. 코드는 동일하고 source tensor만 다릅니다. Encoder–decoder Transformer는 모든 decoder block에서 cross-attention을 쓰고, decoder-only VLM은 흔히 vision token을 시퀀스에 그냥 concat하고 평범한 self-attention을 씁니다. [VLM Implementation Details](#/vlm/practical)를 참고하세요.
</div></details>

### Follow-ups
- **MQA / GQA?** 하나(또는 소수)의 K/V head를 여러 query head가 공유해 KV-cache를 줄입니다 — inference의 핵심입니다; [Transformer block](#/ml-coding/transformer)을 참고하세요.
- **RoPE vs learned positions?** Rotary embedding은 Q와 K를 회전시켜 *상대* 위치를 주입합니다; LLaMA/Qwen/Mistral의 현대적 기본값입니다.
- **Additive (Bahdanau) vs dot-product attention?** Additive는 MLP scorer를 씁니다 — 파라미터가 더 많고 공짜 matmul이 없죠; dot-product가 하드웨어 효율성으로 승리했습니다.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Formula | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| Scale | $1/\sqrt{d_k}$: 점수 분산을 $\approx 1$로 유지 |
| Softmax axis | **key** 기준 (`axis=-1`) |
| Masking | softmax **이전**에 적용, $-\infty$로 채움; AND로 결합 |
| Head split | $(B,T,D)\!\to\!(B,H,T,D/h)$ |
| Complexity | $O(T^2 d)$ 시간, $O(T^2)$ 점수 메모리 |
| Prod kernel | `F.scaled_dot_product_attention` (FlashAttention) |

**Cross-links:** [A Transformer Block](#/ml-coding/transformer) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [LLM Fundamentals](#/llm/fundamentals) · [The ML Coding Round](#/ml-coding/intro)
