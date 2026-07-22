# Positional Encoding & RoPE 직접 구현

> [!NOTE] 이 챕터의 목표
> [Attention](#/ml-coding/attention)에는 치명적인 빈틈이 하나 있습니다 — **순서를 모릅니다.** "개가 사람을 물었다"와 "사람이 개를 물었다"를 똑같이 봅니다. 이 챕터는 그 빈틈을 메우는 **위치 정보 주입(positional encoding)** 을, 고전 sinusoidal부터 오늘날 표준인 **RoPE**까지 그림과 짧은 코드로 잡습니다.

## §0 · 왜 위치가 필요한가

attention은 토큰들을 **집합(set)** 처럼 봅니다. 각 query가 모든 key와의 내적으로 가중합을 만드는데, 이 계산은 토큰 **순서를 바꿔도 결과가 (순서만 따라) 그대로**입니다 — 수학적으로 **permutation-equivariant(순열 등변)**. 즉 "위치"라는 개념 자체가 없습니다.

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="13">
  <text x="20" y="30" fill="#98a3b2">순서 없는 집합으로 보임:</text>
  <g fill="none" stroke="#6366f1" stroke-width="1.6">
    <circle cx="120" cy="70" r="22"/><circle cx="200" cy="55" r="22"/><circle cx="270" cy="95" r="22"/><circle cx="340" cy="60" r="22"/>
  </g>
  <text x="120" y="75" text-anchor="middle" fill="currentColor">개가</text>
  <text x="200" y="60" text-anchor="middle" fill="currentColor">물었다</text>
  <text x="270" y="100" text-anchor="middle" fill="currentColor">사람을</text>
  <text x="340" y="65" text-anchor="middle" fill="currentColor">?</text>
  <path d="M420 70 H470" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#pa)"/>
  <text x="560" y="60" text-anchor="middle" fill="#e0533f">위치 정보 없이는</text>
  <text x="560" y="80" text-anchor="middle" fill="#e0533f">주어·목적어 구분 불가</text>
  <defs><marker id="pa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>attention 자체는 "몇 번째 토큰인지"를 모릅니다. 그래서 각 토큰에 <b>위치 신호</b>를 더하거나(sinusoidal/learned) 회전으로 심어(RoPE) 줘야 합니다.</figcaption>
</figure>

해결책은 크게 둘입니다: **① 위치 벡터를 입력 임베딩에 더한다**(sinusoidal, learned absolute), **② query·key를 위치에 따라 회전시킨다**(RoPE). 하나씩 봅시다.

## §1 · Sinusoidal Positional Encoding

원조 Transformer(Vaswani et al. 2017)의 아이디어: 각 위치 $pos$마다 **서로 다른 주파수(frequency)의 sin/cos 값**을 채운 벡터를 만들어 임베딩에 더합니다. 차원 $2i, 2i{+}1$ 쌍은 주파수가 점점 낮아집니다.

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

직관: 시계의 초침·분침·시침처럼 **빠른 바늘 + 느린 바늘의 조합**으로 위치를 유일하게 지목하는 **지문(fingerprint)** 을 만듭니다. 앞쪽 차원은 빨리 돌아 미세한 위치를, 뒤쪽 차원은 천천히 돌아 넓은 범위를 표현합니다.

<figure>
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="30" y1="100" x2="620" y2="100" stroke="#98a3b2" stroke-width="1"/>
  <text x="620" y="118" text-anchor="end" fill="#98a3b2">position →</text>
  <path d="M30 100 Q 45 40 60 100 T 90 100 T 120 100 T 150 100 T 180 100 T 210 100 T 240 100 T 270 100 T 300 100 T 330 100 T 360 100 T 390 100 T 420 100 T 450 100 T 480 100 T 510 100 T 540 100 T 570 100 T 600 100" fill="none" stroke="#e0533f" stroke-width="1.6" opacity="0.9"/>
  <path d="M30 100 Q 75 55 120 100 T 210 100 T 300 100 T 390 100 T 480 100 T 570 100" fill="none" stroke="#6366f1" stroke-width="1.8"/>
  <path d="M30 100 Q 150 30 270 100 T 510 100" fill="none" stroke="#0ea5e9" stroke-width="2"/>
  <text x="40" y="150" fill="#e0533f">차원 0 (고주파, 빠름)</text>
  <text x="260" y="165" fill="#6366f1">차원 2 (중간)</text>
  <text x="440" y="150" fill="#0ea5e9">차원 4 (저주파, 느림)</text>
</svg>
<figcaption>차원마다 다른 주파수의 파동. 여러 주파수의 (sin, cos) 값을 세로로 쌓으면 각 위치가 유일한 "지문"을 갖습니다 — 학습 없이도 위치를 표현하고, 학습 때 안 본 길이로도 어느 정도 외삽됩니다.</figcaption>
</figure>

### 직접 구현 — sinusoidal PE 행렬

`(seq_len, d_model)` 크기의 PE 행렬을 만드세요. 짝수 차원은 sin, 홀수 차원은 cos입니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sinusoidal_pe","packages":["numpy"],"approx":true,"starter":"def sinusoidal_pe(seq_len, d_model):\n    # (seq_len, d_model) PE 행렬을 리스트의 리스트로 반환.\n    # PE[pos, 2i]   = sin(pos / 10000**(2i/d_model))\n    # PE[pos, 2i+1] = cos(pos / 10000**(2i/d_model))\n    # d_model 은 짝수라고 가정.\n    import numpy as np\n    pass","tests":[{"args":[1,2],"expect":[[0.0,1.0]]},{"args":[2,4],"expect":[[0.0,1.0,0.0,1.0],[0.841471,0.540302,0.0099998,0.99995]],"tol":1e-4},{"args":[3,2],"expect":[[0.0,1.0],[0.841471,0.540302],[0.909297,-0.416147]],"tol":1e-4}],"solution":"import numpy as np\n\ndef sinusoidal_pe(seq_len, d_model):\n    pe = np.zeros((seq_len, d_model))\n    pos = np.arange(seq_len)[:, None]\n    idx = np.arange(0, d_model, 2)[None, :]\n    div = np.power(10000.0, idx / d_model)\n    pe[:, 0::2] = np.sin(pos / div)\n    pe[:, 1::2] = np.cos(pos / div)\n    return pe.tolist()"}
</script>
</div>

position 0은 항상 `[0,1,0,1,...]`(sin0=0, cos0=1)인 것을 확인하세요. 이 행렬을 토큰 임베딩에 **더하면** 모델이 위치를 알게 됩니다.

## §2 · Learned Absolute PE (그리고 한계)

GPT/BERT 초기 계열은 위치별 임베딩을 **학습 파라미터**로 둡니다(`nn.Embedding(max_len, d)`). 단순하고 잘 되지만, **학습 때 본 최대 길이를 넘어가면 외삽이 안 됩니다** — 512까지 학습했으면 513번째 위치 벡터가 아예 없습니다. 그래서 long-context 시대에는 상대적(relative) 방식이 필요해졌습니다.

## §3 · RoPE — 회전으로 상대 위치를 심는다

**RoPE(Rotary Position Embedding)** 는 오늘날 프런티어 LLM의 기본값입니다. 위치 벡터를 *더하는* 대신, query·key 벡터를 **위치에 비례한 각도로 회전**시킵니다. 차원을 2개씩 짝지어 2D 평면에서 회전합니다:

$$
\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} =
\begin{bmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}, \qquad m = \text{position}
$$

**마법은 내적에서 일어납니다.** 위치 $m$에서 회전한 query와 위치 $n$에서 회전한 key의 내적은, 두 회전이 합쳐져 **오직 상대 거리 $m-n$ 에만 의존**하게 됩니다. 즉 절대 위치가 아니라 "얼마나 떨어져 있나"를 자연스럽게 인코딩 → 긴 길이로의 외삽이 훨씬 쉽습니다.

<figure>
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="120" y1="100" x2="120" y2="20" stroke="#98a3b2" stroke-width="1"/>
  <line x1="40" y1="100" x2="200" y2="100" stroke="#98a3b2" stroke-width="1"/>
  <circle cx="120" cy="100" r="70" fill="none" stroke="#98a3b2" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="120" y1="100" x2="185" y2="100" stroke="#e0533f" stroke-width="3">
    <animateTransform attributeName="transform" type="rotate" from="0 120 100" to="-330 120 100" dur="4s" repeatCount="indefinite"/>
  </line>
  <text x="120" y="185" text-anchor="middle" fill="#e0533f">각도 = position × θ 만큼 회전</text>
  <text x="330" y="70" fill="currentColor">위치가 커질수록 더 많이 회전 →</text>
  <text x="330" y="95" fill="#6366f1">q(위치 m)·k(위치 n) 은</text>
  <text x="330" y="118" fill="#6366f1">상대 거리 (m−n) 에만 의존</text>
  <text x="330" y="150" fill="#12a150">→ 긴 문맥으로 외삽이 쉬움</text>
</svg>
<figcaption>RoPE는 q·k를 위치에 비례해 회전시킵니다. 두 벡터의 내적이 상대 offset만 반영하므로, 위치를 "더하지" 않고 "회전으로 심습니다".</figcaption>
</figure>

### 직접 구현 — 2D RoPE 회전

한 (짝수, 홀수) 차원 쌍 $(x_0, x_1)$을 각도 `angle`만큼 회전시키는 함수를 구현하세요.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"rope_2d","packages":["numpy"],"approx":true,"starter":"def rope_2d(x0, x1, angle):\n    # 2D 회전: [x0*cos(angle) - x1*sin(angle), x0*sin(angle) + x1*cos(angle)] 를 리스트로 반환\n    import numpy as np\n    pass","tests":[{"args":[1.0,0.0,0.0],"expect":[1.0,0.0]},{"args":[1.0,0.0,1.5707963],"expect":[0.0,1.0],"tol":1e-5},{"args":[1.0,1.0,0.0],"expect":[1.0,1.0]},{"args":[2.0,0.0,3.1415926],"expect":[-2.0,0.0],"tol":1e-5}],"solution":"import numpy as np\n\ndef rope_2d(x0, x1, angle):\n    c, s = np.cos(angle), np.sin(angle)\n    return [float(x0 * c - x1 * s), float(x0 * s + x1 * c)]"}
</script>
</div>

각도 $\pi/2$면 $(1,0)$이 $(0,1)$로, 각도 $\pi$면 $(2,0)$이 $(-2,0)$으로 도는 것을 확인하세요. 실제 RoPE는 이 회전을 차원 쌍마다 다른 $\theta_i$로 적용합니다(sinusoidal과 같은 다중 주파수 아이디어).

## §4 · ALiBi, 그리고 왜 RoPE가 표준인가

<dl class="kv">
<dt>ALiBi</dt><dd>attention score에 거리 비례 penalty $-m|i-j|$를 <b>더합니다</b>. 학습 위치 파라미터가 없고 외삽이 강하지만, 프런티어에서는 대체로 RoPE에 밀립니다.</dd>
<dt>RoPE가 이긴 이유</dt><dd>(1) 상대 위치를 자연스럽게 인코딩, (2) 절대 위치처럼 임베딩을 더하지 않아 KV cache와 호환이 깔끔, (3) frequency scaling(NTK-aware, YaRN)으로 4K→128K+ 확장이 쉬움. long-context 확장 전략은 <a href="#/llm/fundamentals">LLM Fundamentals</a> 참고.</dd>
</dl>

> [!TIP] 면접 한 줄
> "attention은 순열 등변이라 위치를 따로 주입해야 한다. 옛날엔 sinusoidal/learned를 *더했고*, 요즘은 RoPE로 q·k를 *회전*시켜 내적이 상대 거리에만 의존하게 만든다 — 그래서 long-context 외삽이 쉽다." 여기에 "YaRN 같은 frequency scaling으로 확장한다"를 붙이면 2026 감각까지 커버됩니다.

## Q&A

<details class="qa"><summary>sinusoidal PE는 왜 하필 sin/cos이고, 왜 여러 주파수인가?</summary>
<div class="qa-body">

**짧게:** 여러 주파수의 (sin, cos) 조합이 각 위치에 유일한 지문을 주고, 선형 변환으로 상대 위치 관계를 표현할 수 있게 해서입니다.

**깊게:** 단일 주파수면 주기 때문에 먼 위치가 같은 값으로 겹칩니다(에일리어싱). 기하급수적으로 감소하는 주파수들을 함께 쓰면 짧은 거리와 긴 거리를 동시에 구분합니다. 또 $\sin(a+b),\cos(a+b)$의 덧셈정리 덕분에 위치 $pos{+}k$의 인코딩이 위치 $pos$ 인코딩의 **선형 변환**으로 표현되어, 모델이 상대 위치를 학습하기 쉬워집니다.
</div></details>

<details class="qa"><summary>RoPE는 어느 벡터에 적용하나 — 임베딩? q/k? v도?</summary>
<div class="qa-body">

**짧게:** attention의 **query와 key에만** 적용하고, value에는 하지 않습니다.

**깊게:** 위치 정보가 필요한 건 "누가 누구에게 attend하는가"를 정하는 $q\cdot k$ 점수 계산입니다. RoPE를 q,k에 걸면 그 내적이 상대 위치를 반영합니다. value는 실제로 실어 나르는 내용이라 회전시키지 않습니다. 그래서 RoPE는 임베딩에 한 번 더하는 게 아니라 **매 layer의 attention 직전에** q,k에 적용됩니다.
</div></details>

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| 왜 필요 | attention은 순열 등변 → 순서를 모름 |
| Sinusoidal | 다중 주파수 sin/cos "지문"을 임베딩에 더함 (외삽 가능) |
| Learned absolute | 위치별 학습 임베딩 — 단순하나 최대 길이 초과 시 외삽 불가 |
| RoPE | q·k를 위치∝각도로 회전 → 내적이 상대 거리 $m-n$에만 의존 |
| RoPE 적용 대상 | query·key만 (value·임베딩 아님), 매 layer |
| ALiBi | score에 거리 penalty를 더함; RoPE에 대체로 밀림 |
| 장거리 확장 | RoPE + frequency scaling(NTK/YaRN) → 128K+ |

**다음:** [Attention (밑바닥)](#/ml-coding/attention) · [Transformer Block (밑바닥)](#/ml-coding/transformer) · [LLM Fundamentals](#/llm/fundamentals)
