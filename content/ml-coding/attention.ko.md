# Attention 직접 구현

> [!NOTE] 이 챕터의 목표
> Attention(주의)은 Transformer·LLM·VLM 전체의 심장입니다. 여기서는 "각 토큰이 다른 토큰들을 **얼마나 참고할지**"를 정하는 이 메커니즘을 그림 → 직관 → NumPy 코드로 잡고, 직접 구현해 실행·채점까지 해봅니다. 선행: [NumPy & 브로드캐스팅 프라이머](#/ml-coding/numpy-primer), 개념 배경은 [CNN · RNN · Transformer](#/foundations/architectures).

## 무엇을/왜 — "부드러운 사전 조회(soft lookup)"

문장 "그 **고양이**가 매트 위에 **앉았다**"에서 "앉았다"를 이해하려면 모델은 "누가 앉았지?" → **고양이**를 참고해야 합니다. Attention은 바로 이 **"어떤 단어를 얼마나 참고할지"** 를 학습으로 정하는 장치입니다.

사전(dictionary) 조회에 비유하면 직관이 명확해집니다. 보통 사전은 key가 정확히 일치하는 하나의 value만 꺼냅니다. Attention은 **부드러운(soft)** 조회입니다 — 모든 key와의 **유사도**를 재서, 유사한 정도에 비례해 **여러 value를 섞어** 가져옵니다.

<dl class="kv">
<dt>Query (질의, $Q$)</dt><dd>"내가 지금 찾고 싶은 정보" — 현재 토큰이 던지는 질문.</dd>
<dt>Key (키, $K$)</dt><dd>"내가 가진 정보의 색인" — 각 토큰이 내거는 검색 태그.</dd>
<dt>Value (값, $V$)</dt><dd>"실제로 가져올 내용" — 유사도에 따라 섞이는 알맹이.</dd>
</dl>

<figure>
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- every token is projected to Q, K, and V -->
  <text x="60" y="24" text-anchor="middle" fill="#98a3b2">입력 X (모든 토큰)</text>
  <g fill="none" stroke="#98a3b2" stroke-width="1.4">
    <rect x="30" y="34" width="60" height="26" rx="5"/><rect x="30" y="66" width="60" height="26" rx="5"/><rect x="30" y="98" width="60" height="26" rx="5"/>
  </g>
  <text x="60" y="52" text-anchor="middle" fill="currentColor">그</text>
  <text x="60" y="84" text-anchor="middle" fill="currentColor">고양이</text>
  <text x="60" y="116" text-anchor="middle" fill="currentColor">앉았다</text>
  <!-- projections -->
  <g font-size="11">
    <rect x="130" y="34" width="74" height="26" rx="5" fill="#0ea5e9"/><text x="167" y="52" text-anchor="middle" fill="#fff">Q=XWQ</text>
    <rect x="130" y="66" width="74" height="26" rx="5" fill="#6366f1"/><text x="167" y="84" text-anchor="middle" fill="#fff">K=XWK</text>
    <rect x="130" y="98" width="74" height="26" rx="5" fill="#12a150"/><text x="167" y="116" text-anchor="middle" fill="#fff">V=XWV</text>
  </g>
  <path d="M90 79 L130 47" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <path d="M90 79 L130 79" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <path d="M90 79 L130 111" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <text x="120" y="150" text-anchor="middle" fill="#98a3b2" font-size="10">학습된 3개 projection</text>
  <!-- QK^T scores -->
  <text x="300" y="24" text-anchor="middle" fill="#0ea5e9">① 점수 = Q·Kᵀ / √d</text>
  <g>
    <rect x="250" y="40" width="100" height="100" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
    <line x1="250" y1="73" x2="350" y2="73" stroke="#0ea5e9" opacity=".4"/><line x1="250" y1="106" x2="350" y2="106" stroke="#0ea5e9" opacity=".4"/>
    <line x1="283" y1="40" x2="283" y2="140" stroke="#0ea5e9" opacity=".4"/><line x1="316" y1="40" x2="316" y2="140" stroke="#0ea5e9" opacity=".4"/>
  </g>
  <path d="M196 79 L250 90" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <!-- softmax -->
  <text x="300" y="175" text-anchor="middle" fill="#e0533f">② softmax (행마다, key 기준)</text>
  <path d="M300 140 L300 158" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <!-- weighted sum -->
  <text x="500" y="24" text-anchor="middle" fill="#12a150">③ 가중치 × V 의 합</text>
  <g fill="none" stroke="#12a150" stroke-width="1.5"><rect x="450" y="40" width="60" height="26" rx="5"/><rect x="450" y="72" width="60" height="26" rx="5"/><rect x="450" y="104" width="60" height="26" rx="5"/></g>
  <rect x="560" y="72" width="60" height="26" rx="5" fill="#e0533f"/><text x="590" y="90" text-anchor="middle" fill="#fff">출력</text>
  <path d="M350 90 L450 85" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <path d="M510 85 L560 85" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <text x="300" y="205" text-anchor="middle" fill="#98a3b2" font-size="10">각 행은 확률분포(합=1). 진할수록 더 많이 참고.</text>
  <defs><marker id="aa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>각 토큰은 세 projection을 모두 거쳐 Q·K·V를 만듭니다. 그 뒤 ① 모든 query가 모든 key와 점수를 내고 → ② softmax로 참고 비율을 만들고 → ③ 그 비율로 value를 가중 평균합니다.</figcaption>
</figure>

<div class="widget" data-widget="attention"></div>

> [!TIP] 면접 한 줄
> "Attention은 soft dictionary lookup입니다. query가 dot product로 모든 key에 점수를 매기고, softmax가 saturate되지 않도록 $1/\sqrt{d}$로 스케일한 뒤, softmax가 점수를 가중치로 바꾸고, 그 가중치로 value의 가중 평균을 취합니다." — 그리고 아래 네 줄을 적으면 됩니다.

## 수학

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

여기서 $Q\in\mathbb{R}^{T_q\times d_k}$, $K\in\mathbb{R}^{T_k\times d_k}$, $V\in\mathbb{R}^{T_k\times d_v}$ 입니다. **Multi-head(다중 헤드)** 는 $d/h$-차원 부분공간에서 $h$개의 attention을 병렬로 돌린 뒤 concat(이어붙임)합니다:

$$
\text{MHA}(x) = \big[\text{head}_1;\dots;\text{head}_h\big]W^O,\quad
\text{head}_i=\text{Attention}(xW_i^Q, xW_i^K, xW_i^V)
$$

> [!TIP] 라이브 코드 — 직접 구현하고 실행·테스트
> 아래 NumPy 블록은 **라이브 에디터**입니다. 본문을 채우고 **▶ Run tests**를 누르면 어떤 케이스가 통과하는지 보여줍니다. 막히면 참고용 **Solution**을 열 수 있지만, 먼저 직접 시도하세요 — 그 씨름이 곧 연습입니다. 첫 Run에서 작은 Python 런타임과 NumPy(~15 MB)를 내려받고, 이후 실행은 즉시입니다.

## Scaled dot-product attention (NumPy)

먼저 key 축에 대한 수치적으로 안정된 **softmax**입니다. (max를 빼는 이유는 [손실 & gradient](#/ml-coding/losses-gradients)의 안정성 논의를 참고하세요.)

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    # subtract the max for stability, exponentiate, then normalize\n    pass","tests":[{"args":[[[1,2,3]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[0,0,0]]],"expect":[[0.3333333333333333,0.3333333333333333,0.3333333333333333]]},{"args":[[[1000,1001,1002]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[-1,0,1],[1,0,-1]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218],[0.6652409557748218,0.24472847105479764,0.09003057317038046]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = np.asarray(x, dtype=float)\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)"}
</script>
</div>

그다음 **scaled dot-product attention** — 점수 계산 → 스케일 → mask → softmax → value 가중합입니다. Boolean mask는 `True=허용`이고, additive mask는 float(`0` 또는 큰 음수)만 받습니다. 한 query의 key가 전부 막힌 경우 attention은 정의되지 않으므로 이 구현은 `ValueError`를 냅니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sdpa","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef sdpa(q, k, v, mask=None):\n    # scores = QK^T/sqrt(d); boolean mask uses True=allowed; reject fully masked rows\n    pass","tests":[{"args":[[[1,0],[0,1]],[[1,0],[0,1]],[[1,2],[3,4]]],"expect":[[1.6604769013466862,2.6604769013466862],[2.3395230986533138,3.3395230986533138]]},{"args":[[[2,0,0]],[[2,0,0],[0,2,0],[0,0,2]],[[1,1],[2,2],[3,3]]],"expect":[[1.2485832277662388,1.2485832277662388]]},{"args":[[[1,1],[2,0]],[[1,0],[0,1]],[[10,0],[0,10]],[[true,false],[true,true]]],"expect":[[10.0,0.0],[8.04429682506957,1.9557031749304312]]}],"solution":"import numpy as np\n\ndef sdpa(q, k, v, mask=None):\n    q, k, v = np.asarray(q, float), np.asarray(k, float), np.asarray(v, float)\n    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2] or q.shape[-1] == 0:\n        raise ValueError('incompatible attention shapes')\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])\n    if mask is not None:\n        mask = np.asarray(mask)\n        if mask.dtype == bool:\n            allowed = np.broadcast_to(mask, scores.shape)\n            if np.any(~allowed.any(axis=-1)):\n                raise ValueError('every query needs at least one allowed key')\n            scores = np.where(allowed, scores, -np.inf)\n        elif np.issubdtype(mask.dtype, np.floating):\n            scores = scores + mask\n        else:\n            raise TypeError('mask must be boolean or floating additive mask')\n    m = np.max(scores, axis=-1, keepdims=True)\n    e = np.exp(scores - m)\n    w = e / e.sum(axis=-1, keepdims=True)\n    return w @ v"}
</script>
</div>

### 왜 $\sqrt{d}$로 나누나요?

$q,k$ 성분이 대략 독립이고 평균 0·분산 1이라는 초기화 근사 아래, 내적 $q\cdot k=\sum_{i=1}^d q_ik_i$의 분산은 $\approx d$입니다. 스케일이 없으면 차원과 함께 logit 크기가 커져 softmax가 포화하고 gradient가 매우 작아질 수 있습니다. $\sqrt{d}$로 나누면 이 근사에서 점수 분산을 $\approx 1$로 유지합니다. 학습 후 성분이 완전히 독립이라는 주장이 아니라 초기 scale을 안정시키는 동기입니다([선형대수 & 미적분](#/foundations/linear-algebra-calculus) 참고).

## Masking (마스킹)

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"causal_mask","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef causal_mask(T):\n    # lower-triangular boolean (T,T): True = allowed to attend (no peeking ahead)\n    pass","tests":[{"args":[2],"expect":[[1,0],[1,1]]},{"args":[3],"expect":[[1,0,0],[1,1,0],[1,1,1]]},{"args":[4],"expect":[[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]]}],"solution":"import numpy as np\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))"}
</script>
</div>

- **Causal mask(인과 마스크)** 는 position $t$가 미래($>t$)를 보지 못하게 막습니다 — 다음 단어를 생성하는 autoregressive decoding에 필수입니다(미리 답을 보면 안 되니까요).
- **Key-padding mask** 는 `[PAD]`(길이 맞추기용 빈칸) key로의 attention을 막습니다; shape은 `(B,1,1,Tk)`로 head와 query에 대해 broadcast됩니다.
- 논리 **AND** 로 결합합니다(둘 다 허용해야 함). mask는 softmax **이전**에 적용합니다. Boolean mask의 막힌 항목은 `-inf`로 두되, 모든 key가 막힌 row는 의미가 없으므로 호출자가 최소 한 key를 허용하거나 명시적 zero-output 정책을 정해야 합니다.

## Multi-head attention

왜 큰 attention 하나가 아니라 여러 개를 쓸까요? 직관: head 하나는 "한 가지 관점"으로만 참고합니다. $d$차원을 $h$개의 작은 부분공간으로 나누면, **같은 연산량으로** 여러 관점(문법·위치·의미 등)을 동시에 보고 마지막에 합칩니다.

<figure>
<svg viewBox="0 0 640 190" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="20" y="80" width="90" height="34" rx="6" fill="#6366f1"/><text x="65" y="102" text-anchor="middle" fill="#fff">입력 (D)</text>
  <text x="200" y="24" text-anchor="middle" fill="#98a3b2">split → H개 head (D/h)</text>
  <g fill="none" stroke="#0ea5e9" stroke-width="1.6">
    <rect x="160" y="40" width="80" height="26" rx="5"/><rect x="160" y="80" width="80" height="26" rx="5"/><rect x="160" y="120" width="80" height="26" rx="5"/>
  </g>
  <text x="200" y="57" text-anchor="middle" fill="currentColor">head 1</text>
  <text x="200" y="97" text-anchor="middle" fill="currentColor">head 2</text>
  <text x="200" y="137" text-anchor="middle" fill="currentColor">head h</text>
  <path d="M110 97 L160 53" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M110 97 L160 93" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M110 97 L160 133" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <text x="330" y="24" text-anchor="middle" fill="#e0533f">각자 attend</text>
  <g fill="#e0533f" opacity="0.85"><rect x="300" y="40" width="60" height="26" rx="5"/><rect x="300" y="80" width="60" height="26" rx="5"/><rect x="300" y="120" width="60" height="26" rx="5"/></g>
  <path d="M240 53 L300 53" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M240 93 L300 93" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M240 133 L300 133" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <text x="470" y="24" text-anchor="middle" fill="#12a150">concat</text>
  <rect x="430" y="80" width="80" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.8"/><text x="470" y="102" text-anchor="middle" fill="currentColor">이어붙임 (D)</text>
  <path d="M360 53 L430 92" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M360 93 L430 97" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M360 133 L430 102" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <rect x="560" y="80" width="70" height="34" rx="6" fill="#e0533f"/><text x="595" y="102" text-anchor="middle" fill="#fff">Wᴼ 출력</text>
  <path d="M510 97 L560 97" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <defs><marker id="ab" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>$(B,T,D)$를 $(B,H,T,D/h)$로 쪼개 head를 batch처럼 다루면, 하나의 행렬곱으로 모든 head를 한 번에 계산할 수 있습니다. 끝에서 다시 concat하고 $W^O$로 섞습니다.</figcaption>
</figure>

`mha`를 구현하세요; 하네스 `run_mha`는 seed 고정된 `(B,T,D)=(1,3,4)` 입력과 `H=2` head를 만들어 출력이 결정적이게 합니다(헬퍼 `softmax`/`causal_mask`/`sdpa`는 제공됩니다):

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"run_mha","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))\n\ndef sdpa(q, k, v, mask=None):\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    return softmax(scores, axis=-1) @ v\n\ndef mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):\n    # split (B,T,D)->(B,H,T,Dh), run sdpa per head, merge heads, project by Wo\n    pass\n\ndef run_mha(causal):\n    np.random.seed(0)\n    B, T, D, H = 1, 3, 4, 2\n    x = np.random.randn(B, T, D)\n    Ws = [np.random.randn(D, D) * 0.1 for _ in range(4)]\n    return mha(x, *Ws, H, causal)","tests":[{"args":[false],"expect":[[[0.03190098881683708,-0.003883638346883957,0.009816921823006634,0.0008853701048203459],[0.032178757789826054,-0.003954663472188265,0.008879630160056167,0.0011910101137743716],[0.03203148496155561,-0.0038215886062526196,0.009498626868409653,0.0010096474007127018]]]},{"args":[true],"expect":[[[0.059733984570268545,-0.003891324159579386,0.01245601832145022,0.013606293601693605],[0.04292552968524655,-0.008439370891800252,0.018326373103144715,-0.004398635006322705],[0.03203148496155561,-0.0038215886062526196,0.009498626868409653,0.0010096474007127018]]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))\n\ndef sdpa(q, k, v, mask=None):\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    return softmax(scores, axis=-1) @ v\n\ndef mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):\n    B, T, D = x.shape\n    dh = D // num_heads\n    def split(t):\n        return t.reshape(B, T, num_heads, dh).transpose(0, 2, 1, 3)\n    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)\n    mask = causal_mask(T) if causal else None\n    out = sdpa(q, k, v, mask=mask)\n    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)\n    return out @ Wo\n\ndef run_mha(causal):\n    np.random.seed(0)\n    B, T, D, H = 1, 3, 4, 2\n    x = np.random.randn(B, T, D)\n    Ws = [np.random.randn(D, D) * 0.1 for _ in range(4)]\n    return mha(x, *Ws, H, causal)"}
</script>
</div>

head를 batch 같은 축으로 나누면 하나의 `@`로 모든 head를 한 번에 계산할 수 있습니다. **복잡도:** head당 점수 행렬에 대해 $O(T^2 d)$ 연산과 $O(T^2)$ 메모리가 듭니다 — 긴 시퀀스에서 attention이 비싼 근본 이유입니다.

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

**깊게:** naive attention은 전체 score/weight를 HBM에 씁니다. FlashAttention은 $Q,K,V$를 타일로 나누고 running max·denominator로 online softmax를 계산해 HBM 왕복을 줄입니다. 수학적으로 같은 attention이지만 floating-point 연산 순서 때문에 bitwise 동일할 필요는 없습니다. `F.scaled_dot_product_attention`은 dtype·shape·device·mask 조건에 따라 Flash, memory-efficient, math backend 중 하나를 선택하므로 항상 FlashAttention이라고 단정하지 않습니다.
</div></details>

<details class="qa"><summary>Self-attention vs cross-attention?</summary>
<div class="qa-body">

**짧게:** self-attention은 Q, K, V를 같은 시퀀스에서 뽑고, cross-attention은 Q를 한 스트림에서, K, V를 다른 스트림에서 가져옵니다.

**깊게:** VLM decoder에서 text token은 self-attend하고, cross-attention은 text query가 image/vision token을 key/value로 삼아 끌어오게 합니다 — 언어를 픽셀에 grounding하는 메커니즘이죠. 코드는 동일하고 source tensor만 다릅니다. Encoder–decoder Transformer는 모든 decoder block에서 cross-attention을 쓰고, decoder-only VLM은 흔히 vision token을 시퀀스에 그냥 concat하고 평범한 self-attention을 씁니다. [VLM Implementation Details](#/vlm/practical)를 참고하세요.
</div></details>

### Follow-ups
- **위치 정보는 어떻게?** 위치 정보가 없는 self-attention은 token 순열에 대해 **equivariant**합니다. 즉 입력을 재배열하면 출력도 같은 방식으로 재배열됩니다. **positional encoding / RoPE**로 순서를 주입합니다.
- **MQA / GQA?** 하나(또는 소수)의 K/V head를 여러 query head가 공유해 KV-cache를 줄입니다 — inference의 핵심입니다; [Transformer block](#/ml-coding/transformer)을 참고하세요.
- **Additive (Bahdanau) vs dot-product attention?** Additive는 MLP scorer를 씁니다 — 파라미터가 더 많고 공짜 matmul이 없죠; dot-product가 하드웨어 효율성으로 승리했습니다.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Formula | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| 직관 | soft dictionary lookup: 유사도로 value를 가중 평균 |
| Scale | $1/\sqrt{d_k}$: 점수 분산을 $\approx 1$로 유지 |
| Softmax axis | **key** 기준 (`axis=-1`) |
| Masking | softmax **이전**에 적용, $-\infty$로 채움; AND로 결합 |
| Head split | $(B,T,D)\!\to\!(B,H,T,D/h)$ |
| Complexity | $O(T^2 d)$ 시간, $O(T^2)$ 점수 메모리 |
| Prod kernel | `F.scaled_dot_product_attention`가 조건에 따라 Flash/memory-efficient/math backend를 선택 |

**Cross-links:** [NumPy 프라이머](#/ml-coding/numpy-primer) · [Positional Encoding & RoPE](#/ml-coding/positional-encoding-rope) · [A Transformer Block](#/ml-coding/transformer) · [CNN · RNN · Transformer](#/foundations/architectures) · [LLM Fundamentals](#/llm/fundamentals)
