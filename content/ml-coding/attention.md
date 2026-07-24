# Attention From Scratch

> [!NOTE] Goal of this chapter
> Attention is the heart of Transformers, LLMs, and VLMs. Here you will understand how each token decides **how much to consult every other token**, moving from a picture to intuition to NumPy code, and then implement, run, and test it yourself. Prerequisite: [NumPy & Broadcasting Primer](#/ml-coding/numpy-primer); for conceptual context, see [CNNs, RNNs & Transformers](#/foundations/architectures).

## What and why—a “soft dictionary lookup”

In the sentence “The **cat** **sat** on the mat,” understanding “sat” requires asking “who sat?” and consulting **cat**. Attention learns exactly this choice: **which words to consult, and by how much**.

The dictionary analogy makes it concrete. An ordinary dictionary lookup retrieves one value whose key matches exactly. Attention performs a **soft** lookup: it measures similarity against every key and mixes **multiple values** in proportion to those similarities.

<dl class="kv">
<dt>Query ($Q$)</dt><dd>“The information I want right now”—the question posed by the current token.</dd>
<dt>Key ($K$)</dt><dd>“The index for the information I contain”—a search tag exposed by each token.</dd>
<dt>Value ($V$)</dt><dd>“The content to retrieve”—the payload mixed according to similarity.</dd>
</dl>

<figure>
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- every token is projected to Q, K, and V -->
  <text x="60" y="24" text-anchor="middle" fill="#98a3b2">input X (all tokens)</text>
  <g fill="none" stroke="#98a3b2" stroke-width="1.4">
    <rect x="30" y="34" width="60" height="26" rx="5"/><rect x="30" y="66" width="60" height="26" rx="5"/><rect x="30" y="98" width="60" height="26" rx="5"/>
  </g>
  <text x="60" y="52" text-anchor="middle" fill="currentColor">the</text>
  <text x="60" y="84" text-anchor="middle" fill="currentColor">cat</text>
  <text x="60" y="116" text-anchor="middle" fill="currentColor">sat</text>
  <!-- projections -->
  <g font-size="11">
    <rect x="130" y="34" width="74" height="26" rx="5" fill="#0ea5e9"/><text x="167" y="52" text-anchor="middle" fill="#fff">Q=XWQ</text>
    <rect x="130" y="66" width="74" height="26" rx="5" fill="#6366f1"/><text x="167" y="84" text-anchor="middle" fill="#fff">K=XWK</text>
    <rect x="130" y="98" width="74" height="26" rx="5" fill="#12a150"/><text x="167" y="116" text-anchor="middle" fill="#fff">V=XWV</text>
  </g>
  <path d="M90 79 L130 47" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <path d="M90 79 L130 79" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <path d="M90 79 L130 111" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <text x="120" y="150" text-anchor="middle" fill="#98a3b2" font-size="10">three learned projections</text>
  <!-- QK^T scores -->
  <text x="300" y="24" text-anchor="middle" fill="#0ea5e9">① scores = Q·Kᵀ / √d</text>
  <g>
    <rect x="250" y="40" width="100" height="100" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
    <line x1="250" y1="73" x2="350" y2="73" stroke="#0ea5e9" opacity=".4"/><line x1="250" y1="106" x2="350" y2="106" stroke="#0ea5e9" opacity=".4"/>
    <line x1="283" y1="40" x2="283" y2="140" stroke="#0ea5e9" opacity=".4"/><line x1="316" y1="40" x2="316" y2="140" stroke="#0ea5e9" opacity=".4"/>
  </g>
  <path d="M196 79 L250 90" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <!-- softmax -->
  <text x="300" y="175" text-anchor="middle" fill="#e0533f">② softmax (per row, over keys)</text>
  <path d="M300 140 L300 158" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <!-- weighted sum -->
  <text x="500" y="24" text-anchor="middle" fill="#12a150">③ weighted sum of V</text>
  <g fill="none" stroke="#12a150" stroke-width="1.5"><rect x="450" y="40" width="60" height="26" rx="5"/><rect x="450" y="72" width="60" height="26" rx="5"/><rect x="450" y="104" width="60" height="26" rx="5"/></g>
  <rect x="560" y="72" width="60" height="26" rx="5" fill="#e0533f"/><text x="590" y="90" text-anchor="middle" fill="#fff">output</text>
  <path d="M350 90 L450 85" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <path d="M510 85 L560 85" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#aa)"/>
  <text x="300" y="205" text-anchor="middle" fill="#98a3b2" font-size="10">Each row is a distribution (sum=1); darker means more attention.</text>
  <defs><marker id="aa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>Every token passes through all three projections to produce Q, K, and V. Then ① every query scores every key, ② softmax turns scores into attention proportions, and ③ those proportions form a weighted average of the values.</figcaption>
</figure>

<div class="widget" data-widget="attention"></div>

> [!TIP] Interview one-liner
> “Attention is a soft dictionary lookup. A query scores every key with a dot product; scale by $1/\sqrt{d}$ to keep softmax from saturating; softmax turns scores into weights; and those weights form a weighted average of the values.” Then write the four lines below.

## The math

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Here $Q\in\mathbb{R}^{T_q\times d_k}$, $K\in\mathbb{R}^{T_k\times d_k}$, and $V\in\mathbb{R}^{T_k\times d_v}$. **Multi-head attention** runs $h$ attentions in parallel on $d/h$-dimensional subspaces and concatenates them:

$$
\text{MHA}(x) = \big[\text{head}_1;\dots;\text{head}_h\big]W^O,\quad
\text{head}_i=\text{Attention}(xW_i^Q, xW_i^K, xW_i^V)
$$

> [!TIP] Live code — implement, run, test
> The NumPy blocks below are **live editors**. Fill in the body, hit **▶ Run tests**, and watch the cases pass. Stuck? Reveal the reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime and NumPy (~15 MB); later runs are instant.

## Scaled dot-product attention (NumPy)

First, implement a numerically stable **softmax** over the key axis. See the stability discussion in [Losses & Gradients](#/ml-coding/losses-gradients) for why subtracting the maximum matters.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    # subtract the max for stability, exponentiate, then normalize\n    pass","tests":[{"args":[[[1,2,3]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[0,0,0]]],"expect":[[0.3333333333333333,0.3333333333333333,0.3333333333333333]]},{"args":[[[1000,1001,1002]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[-1,0,1],[1,0,-1]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218],[0.6652409557748218,0.24472847105479764,0.09003057317038046]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = np.asarray(x, dtype=float)\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)"}
</script>
</div>

Then implement **scaled dot-product attention**: score → scale → mask → softmax → weighted sum of values. A Boolean mask uses `True=allowed`; an additive mask accepts floating-point values such as 0 or a large negative number. Attention is undefined when every key for a query is blocked, so this implementation raises `ValueError`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sdpa","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef sdpa(q, k, v, mask=None):\n    # scores = QK^T/sqrt(d); boolean mask uses True=allowed; reject fully masked rows\n    pass","tests":[{"args":[[[1,0],[0,1]],[[1,0],[0,1]],[[1,2],[3,4]]],"expect":[[1.6604769013466862,2.6604769013466862],[2.3395230986533138,3.3395230986533138]]},{"args":[[[2,0,0]],[[2,0,0],[0,2,0],[0,0,2]],[[1,1],[2,2],[3,3]]],"expect":[[1.2485832277662388,1.2485832277662388]]},{"args":[[[1,1],[2,0]],[[1,0],[0,1]],[[10,0],[0,10]],[[true,false],[true,true]]],"expect":[[10.0,0.0],[8.04429682506957,1.9557031749304312]]}],"solution":"import numpy as np\n\ndef sdpa(q, k, v, mask=None):\n    q, k, v = np.asarray(q, float), np.asarray(k, float), np.asarray(v, float)\n    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2] or q.shape[-1] == 0:\n        raise ValueError('incompatible attention shapes')\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])\n    if mask is not None:\n        mask = np.asarray(mask)\n        if mask.dtype == bool:\n            allowed = np.broadcast_to(mask, scores.shape)\n            if np.any(~allowed.any(axis=-1)):\n                raise ValueError('every query needs at least one allowed key')\n            scores = np.where(allowed, scores, -np.inf)\n        elif np.issubdtype(mask.dtype, np.floating):\n            scores = scores + mask\n        else:\n            raise TypeError('mask must be boolean or floating additive mask')\n    m = np.max(scores, axis=-1, keepdims=True)\n    e = np.exp(scores - m)\n    w = e / e.sum(axis=-1, keepdims=True)\n    return w @ v"}
</script>
</div>

### Why divide by $\sqrt{d}$?

Assume at initialization that the components of $q$ and $k$ are approximately independent, each with mean 0 and variance 1. For one coordinate,

$$
\mathbb E[q_i k_i]=0,\qquad
\operatorname{Var}(q_i k_i)=\mathbb E[q_i^2]\mathbb E[k_i^2]=1,
$$

and variances add across $d$ independent terms:

$$
\operatorname{Var}(q\cdot k)
=\operatorname{Var}\!\left(\sum_{i=1}^{d}q_i k_i\right)
\approx d.
$$

The important distinction is that **variance $d$ means a standard deviation, or typical magnitude, of $\sqrt d$**. Positive and negative terms partially cancel, so the sum does not typically grow like $d$. Therefore,

$$
\operatorname{Std}(q\cdot k)\approx\sqrt d,\qquad
\operatorname{Var}\!\left(\frac{q\cdot k}{\sqrt d}\right)\approx 1.
$$

This is why the divisor is $\sqrt d$ rather than $d$: it returns the **standard deviation of the logits** to a constant scale.

<figure>
<svg viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11" role="img" aria-labelledby="scale-title-en scale-desc-en">
  <title id="scale-title-en">Attention score scaling and softmax saturation</title>
  <desc id="scale-desc-en">The left plot shows that the unscaled score standard deviation grows with the square root of dimension while the scaled standard deviation remains one. The right compares a logit gap of eight at dimension 64 with a scaled gap of one.</desc>
  <text x="175" y="18" text-anchor="middle" fill="currentColor">Score standard deviation as dimension grows</text>
  <line x1="48" y1="215" x2="310" y2="215" stroke="#98a3b2"/>
  <line x1="48" y1="215" x2="48" y2="35" stroke="#98a3b2"/>
  <g fill="#98a3b2" font-size="10">
    <text x="48" y="232" text-anchor="middle">1</text><text x="117" y="232" text-anchor="middle">16</text>
    <text x="190" y="232" text-anchor="middle">64</text><text x="285" y="232" text-anchor="middle">256</text>
    <text x="38" y="218" text-anchor="end">0</text><text x="38" y="173" text-anchor="end">4</text>
    <text x="38" y="128" text-anchor="end">8</text><text x="38" y="83" text-anchor="end">12</text>
    <text x="38" y="39" text-anchor="end">16</text><text x="175" y="252" text-anchor="middle">head dimension d</text>
  </g>
  <polyline points="48,204 117,170 190,125 285,35" fill="none" stroke="#e0533f" stroke-width="2.5"/>
  <g fill="#e0533f"><circle cx="48" cy="204" r="3"/><circle cx="117" cy="170" r="3"/><circle cx="190" cy="125" r="3"/><circle cx="285" cy="35" r="3"/></g>
  <text x="220" y="69" fill="#e0533f">unscaled: √d</text>
  <line x1="48" y1="204" x2="285" y2="204" stroke="#12a150" stroke-width="2.5"/>
  <text x="151" y="197" fill="#12a150">÷√d: 1</text>
  <line x1="335" y1="28" x2="335" y2="235" stroke="#98a3b2" opacity=".45"/>
  <text x="510" y="18" text-anchor="middle" fill="currentColor">d=64: same signal, different softmax</text>
  <text x="365" y="48" fill="#98a3b2">before scaling, Δ=8</text>
  <rect x="365" y="61" width="235" height="24" rx="4" fill="none" stroke="#98a3b2"/>
  <rect x="365" y="61" width="234.9" height="24" rx="4" fill="#e0533f" opacity=".82"/>
  <text x="668" y="77" text-anchor="end" fill="currentColor">0.9997 / 0.0003</text>
  <text x="365" y="106" fill="#98a3b2">after ÷√64=8, Δ=1</text>
  <rect x="365" y="119" width="235" height="24" rx="4" fill="none" stroke="#98a3b2"/>
  <rect x="365" y="119" width="172" height="24" rx="4" fill="#12a150" opacity=".82"/>
  <text x="668" y="135" text-anchor="end" fill="currentColor">0.731 / 0.269</text>
  <path d="M365 178 H600" stroke="#98a3b2"/>
  <circle cx="375" cy="178" r="5" fill="#e0533f"/><text x="388" y="182" fill="currentColor">Δ=8: p(1−p)≈0.0003</text>
  <circle cx="375" cy="207" r="5" fill="#12a150"/><text x="388" y="211" fill="currentColor">Δ=1: p(1−p)≈0.197</text>
  <text x="482" y="238" text-anchor="middle" fill="#98a3b2">large logit gap → nearly one-hot → small gradient</text>
</svg>
<figcaption>At $d=64$, the unscaled score standard deviation is about 8. A two-logit gap of 8 makes softmax nearly one-hot and drives the binary-softmax sensitivity $p(1-p)$ near zero. Dividing by $\sqrt{64}=8$ restores a gap near 1 and leaves a useful gradient.</figcaption>
</figure>

Scaling does not force attention to stay uniform. The model can still learn large attention scores when useful; the divisor merely prevents **dimension alone** from saturating the initial distribution. The independence and unit-variance assumptions are initialization approximations, not claims about every learned component of $q$ and $k$. See [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus).

## Masking

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"causal_mask","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef causal_mask(T):\n    # lower-triangular boolean (T,T): True = allowed to attend (no peeking ahead)\n    pass","tests":[{"args":[2],"expect":[[1,0],[1,1]]},{"args":[3],"expect":[[1,0,0],[1,1,0],[1,1,1]]},{"args":[4],"expect":[[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]]}],"solution":"import numpy as np\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))"}
</script>
</div>

Let row $i$ be a query position and column $j$ a key position. Position $i$ may see the past and itself ($j\le i$), but not the future ($j>i$). Boolean and additive masks encode the same allowed region in different ways:

$$
\text{allowed}_{ij}=[j\le i],\qquad
M_{ij}=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases},\qquad
A=\operatorname{softmax}(S+M).
$$

<figure>
<svg viewBox="0 0 720 275" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="10.5" role="img" aria-labelledby="mask-title-en mask-desc-en">
  <title id="mask-title-en">How a causal attention mask turns scores into attention weights</title>
  <desc id="mask-desc-en">Rows are queries and columns are keys. The diagonal and lower-left region of the additive mask contain zero and are allowed. Future positions in the upper-right contain negative infinity and have zero probability after softmax.</desc>
  <defs><marker id="mask-arrow-en" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
  <text x="120" y="18" text-anchor="middle" fill="currentColor">① raw scores S</text>
  <text x="360" y="18" text-anchor="middle" fill="currentColor">② additive mask M</text>
  <text x="610" y="18" text-anchor="middle" fill="currentColor">③ softmax(S+M)</text>
  <g fill="#98a3b2">
    <text x="120" y="38" text-anchor="middle">key j →</text><text x="360" y="38" text-anchor="middle">key j →</text><text x="610" y="38" text-anchor="middle">key j →</text>
    <text x="31" y="136" text-anchor="middle" transform="rotate(-90 31 136)">query i →</text>
    <text x="271" y="136" text-anchor="middle" transform="rotate(-90 271 136)">query i →</text>
    <text x="521" y="136" text-anchor="middle" transform="rotate(-90 521 136)">query i →</text>
  </g>
  <g transform="translate(50 48)">
    <rect width="140" height="140" fill="none" stroke="#0ea5e9" stroke-width="1.4"/>
    <g stroke="#98a3b2" opacity=".45"><path d="M35 0V140M70 0V140M105 0V140M0 35H140M0 70H140M0 105H140"/></g>
    <g fill="currentColor" text-anchor="middle">
      <text x="17.5" y="22">s₀₀</text><text x="52.5" y="22">s₀₁</text><text x="87.5" y="22">s₀₂</text><text x="122.5" y="22">s₀₃</text>
      <text x="17.5" y="57">s₁₀</text><text x="52.5" y="57">s₁₁</text><text x="87.5" y="57">s₁₂</text><text x="122.5" y="57">s₁₃</text>
      <text x="17.5" y="92">s₂₀</text><text x="52.5" y="92">s₂₁</text><text x="87.5" y="92">s₂₂</text><text x="122.5" y="92">s₂₃</text>
      <text x="17.5" y="127">s₃₀</text><text x="52.5" y="127">s₃₁</text><text x="87.5" y="127">s₃₂</text><text x="122.5" y="127">s₃₃</text>
    </g>
  </g>
  <g transform="translate(290 48)">
    <path d="M0 0H35V35H70V70H105V105H140V140H0Z" fill="#12a150" opacity=".2"/>
    <path d="M35 0H140V105H105V70H70V35H35Z" fill="#e0533f" opacity=".25"/>
    <rect width="140" height="140" fill="none" stroke="#6366f1" stroke-width="1.4"/>
    <g stroke="#98a3b2" opacity=".45"><path d="M35 0V140M70 0V140M105 0V140M0 35H140M0 70H140M0 105H140"/></g>
    <g text-anchor="middle">
      <g fill="#12a150"><text x="17.5" y="22">0</text><text x="17.5" y="57">0</text><text x="52.5" y="57">0</text><text x="17.5" y="92">0</text><text x="52.5" y="92">0</text><text x="87.5" y="92">0</text><text x="17.5" y="127">0</text><text x="52.5" y="127">0</text><text x="87.5" y="127">0</text><text x="122.5" y="127">0</text></g>
      <g fill="#e0533f"><text x="52.5" y="22">−∞</text><text x="87.5" y="22">−∞</text><text x="122.5" y="22">−∞</text><text x="87.5" y="57">−∞</text><text x="122.5" y="57">−∞</text><text x="122.5" y="92">−∞</text></g>
    </g>
  </g>
  <g transform="translate(540 48)">
    <path d="M0 0H35V35H70V70H105V105H140V140H0Z" fill="#12a150" opacity=".2"/>
    <path d="M35 0H140V105H105V70H70V35H35Z" fill="#e0533f" opacity=".12"/>
    <rect width="140" height="140" fill="none" stroke="#12a150" stroke-width="1.4"/>
    <g stroke="#98a3b2" opacity=".45"><path d="M35 0V140M70 0V140M105 0V140M0 35H140M0 70H140M0 105H140"/></g>
    <g text-anchor="middle">
      <g fill="currentColor"><text x="17.5" y="22">1</text><text x="17.5" y="57">a</text><text x="52.5" y="57">b</text><text x="17.5" y="92">a</text><text x="52.5" y="92">b</text><text x="87.5" y="92">c</text><text x="17.5" y="127">a</text><text x="52.5" y="127">b</text><text x="87.5" y="127">c</text><text x="122.5" y="127">d</text></g>
      <g fill="#e0533f"><text x="52.5" y="22">0</text><text x="87.5" y="22">0</text><text x="122.5" y="22">0</text><text x="87.5" y="57">0</text><text x="122.5" y="57">0</text><text x="122.5" y="92">0</text></g>
    </g>
  </g>
  <path d="M200 118H270" stroke="#98a3b2" marker-end="url(#mask-arrow-en)"/><text x="235" y="108" text-anchor="middle" fill="#98a3b2">add</text>
  <path d="M440 118H520" stroke="#98a3b2" marker-end="url(#mask-arrow-en)"/><text x="480" y="108" text-anchor="middle" fill="#98a3b2">row softmax</text>
  <text x="360" y="215" text-anchor="middle" fill="#12a150">diagonal + lower left: past/current → allowed</text>
  <text x="360" y="236" text-anchor="middle" fill="#e0533f">upper right: future → −∞ → exp(−∞)=0</text>
  <text x="360" y="259" text-anchor="middle" fill="#98a3b2">allowed weights in each output row sum to 1</text>
</svg>
<figcaption>The lower-left region is not removed; the <b>lower-left region and diagonal are allowed</b>. Those are the `True` entries from `np.tril(...)`. In additive form, add 0 at allowed positions and $-\infty$ at future positions.</figcaption>
</figure>

- `score=0` is not a mask: $e^0=1$, so softmax still assigns it positive probability. To obtain exactly zero attention weight, insert $-\infty$—or a sufficiently negative finite value appropriate for the dtype—**before** softmax.
- A **causal mask** blocks position $t$ from seeing the future ($>t$), which is required for autoregressive next-token decoding.
- A **key-padding mask** blocks attention to `[PAD]` keys. Its shape is `(B,1,1,Tk)`, broadcasting over heads and queries.
- Combine them with logical **AND**, since both must permit an entry. Apply a mask **before** softmax. Set blocked Boolean entries to `-inf`, but note that a row with every key blocked is meaningless: callers must allow at least one key or define an explicit zero-output policy.

## Multi-head attention

Why use several heads instead of one large attention operation? Intuitively, one head consults tokens from only one “point of view.” Splitting the $d$ dimensions into $h$ smaller subspaces allows several perspectives—such as syntax, position, and meaning—to operate simultaneously at the **same asymptotic compute**, before their outputs are recombined.

### What changes as `num_heads` increases?

First state what is held fixed. The standard comparison fixes model width $D=d_{\text{model}}$ and changes only head count $H$. The head dimension is then

$$
d_h=\frac{D}{H},
$$

so more heads mean narrower heads. Ignoring bias, the projection parameters of standard MHA are

$$
\underbrace{3D^2}_{W^Q,W^K,W^V}+\underbrace{D^2}_{W^O}=4D^2,
$$

which is **independent of $H$**. Even if the projections are written as separate per-head matrices, their total column count is $H d_h=D$.

Compute cancels in the same way. For batch $B$ and sequence length $T$, counting multiply–accumulates (MACs):

$$
\text{projections}\approx4BTD^2,\qquad
\text{attention matmuls}
=H\cdot2BT^2d_h
=2BT^2D.
$$

With fixed $D$, increasing `num_heads` from 8 to 16 therefore leaves the dominant theoretical matmul count nearly unchanged. **Intermediate memory does change**, however: every head has its own $T\times T$ score/weight matrix, so naive attention materializes $O(BHT^2)$ elements.

| As $H$ increases with fixed $D$ | Change |
| --- | --- |
| head dimension $d_h=D/H$ | decreases |
| Q/K/V/O parameters $\approx4D^2$ | nearly unchanged |
| projection MACs $4BTD^2$ | unchanged |
| QKᵀ + AV MACs $2BT^2D$ | unchanged |
| naive score/weight memory $BHT^2$ | grows linearly with $H$ |
| actual wall-clock time | may rise or fall with hardware and kernel shape |

Equal FLOPs do not imply equal speed. More heads expose parallelism, but very small $d_h$ can produce tiny matmuls, more softmax rows, and greater scheduling overhead, reducing GPU utilization. Too few heads can also yield unfavorable shapes or insufficient parallel work. FlashAttention avoids writing the $BHT^2$ matrix to HBM, but it does not remove head-dependent tile scheduling and shape efficiency. `num_heads` is therefore not a knob that monotonically reduces latency; it is an **architecture hyperparameter to benchmark for quality and target hardware**.

> [!WARNING] Change the fixed quantity and the conclusion changes
> If $d_h$ is held fixed while $H$ increases, then $D=Hd_h$ also grows. Parameters scale approximately as $4D^2$, and projection compute grows with $D^2$. “More heads cost the same” applies only when **$D$ is fixed and merely repartitioned**.

### Is MHA a parameter-saving approximation?

No. Standard MHA and single-head attention at the same $D$ have nearly the same projection parameters and dominant matmul budget. MHA does not cheaply approximate one large attention operation; it lets **one query carry several attention distributions in different learned subspaces at once**.

A single head produces one weight vector $A_i$ per query and applies that same token-mixing pattern to every value channel. MHA can produce

$$
A_i^{(1)},A_i^{(2)},\ldots,A_i^{(H)},
$$

so one head may capture a nearby syntactic relation while another tracks distant coreference or position. Splitting $D$ across heads obtains this structured expressivity **within a fixed total width and compute budget**; it is not parameter reduction. Some trained heads can be redundant or pruneable, but that observation does not turn MHA's original role into approximation.

**MQA/GQA** are the explicitly efficiency-oriented variants: they preserve many query heads while sharing fewer K/V heads, reducing some parameters and, most importantly, the inference KV cache. They are a quality–efficiency trade-off distinct from standard MHA.

<figure>
<svg viewBox="0 0 640 190" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="20" y="80" width="90" height="34" rx="6" fill="#6366f1"/><text x="65" y="102" text-anchor="middle" fill="#fff">input (D)</text>
  <text x="200" y="24" text-anchor="middle" fill="#98a3b2">split → H heads (D/h)</text>
  <g fill="none" stroke="#0ea5e9" stroke-width="1.6">
    <rect x="160" y="40" width="80" height="26" rx="5"/><rect x="160" y="80" width="80" height="26" rx="5"/><rect x="160" y="120" width="80" height="26" rx="5"/>
  </g>
  <text x="200" y="57" text-anchor="middle" fill="currentColor">head 1</text>
  <text x="200" y="97" text-anchor="middle" fill="currentColor">head 2</text>
  <text x="200" y="137" text-anchor="middle" fill="currentColor">head h</text>
  <path d="M110 97 L160 53" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M110 97 L160 93" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M110 97 L160 133" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <text x="330" y="24" text-anchor="middle" fill="#e0533f">attend independently</text>
  <g fill="#e0533f" opacity="0.85"><rect x="300" y="40" width="60" height="26" rx="5"/><rect x="300" y="80" width="60" height="26" rx="5"/><rect x="300" y="120" width="60" height="26" rx="5"/></g>
  <path d="M240 53 L300 53" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M240 93 L300 93" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M240 133 L300 133" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <text x="470" y="24" text-anchor="middle" fill="#12a150">concatenate</text>
  <rect x="430" y="80" width="80" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.8"/><text x="470" y="102" text-anchor="middle" fill="currentColor">merged (D)</text>
  <path d="M360 53 L430 92" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M360 93 L430 97" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <path d="M360 133 L430 102" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <rect x="560" y="80" width="70" height="34" rx="6" fill="#e0533f"/><text x="595" y="102" text-anchor="middle" fill="#fff">Wᴼ output</text>
  <path d="M510 97 L560 97" stroke="#98a3b2" stroke-width="1.2" marker-end="url(#ab)"/>
  <defs><marker id="ab" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>Split $(B,T,D)$ into $(B,H,T,D/h)$ and treat heads like a batch axis, so one matrix multiplication computes all heads at once. Concatenate the result and mix it through $W^O$.</figcaption>
</figure>

Implement `mha`; the harness `run_mha` builds a fixed, seeded `(B,T,D)=(1,3,4)` input with `H=2` heads so the output is deterministic (the helpers `softmax`/`causal_mask`/`sdpa` are provided):

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"run_mha","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))\n\ndef sdpa(q, k, v, mask=None):\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    return softmax(scores, axis=-1) @ v\n\ndef mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):\n    # split (B,T,D)->(B,H,T,Dh), run sdpa per head, merge heads, project by Wo\n    pass\n\ndef run_mha(causal):\n    np.random.seed(0)\n    B, T, D, H = 1, 3, 4, 2\n    x = np.random.randn(B, T, D)\n    Ws = [np.random.randn(D, D) * 0.1 for _ in range(4)]\n    return mha(x, *Ws, H, causal)","tests":[{"args":[false],"expect":[[[0.03190098881683708,-0.003883638346883957,0.009816921823006634,0.0008853701048203459],[0.032178757789826054,-0.003954663472188265,0.008879630160056167,0.0011910101137743716],[0.03203148496155561,-0.0038215886062526196,0.009498626868409653,0.0010096474007127018]]]},{"args":[true],"expect":[[[0.059733984570268545,-0.003891324159579386,0.01245601832145022,0.013606293601693605],[0.04292552968524655,-0.008439370891800252,0.018326373103144715,-0.004398635006322705],[0.03203148496155561,-0.0038215886062526196,0.009498626868409653,0.0010096474007127018]]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))\n\ndef sdpa(q, k, v, mask=None):\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    return softmax(scores, axis=-1) @ v\n\ndef mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):\n    B, T, D = x.shape\n    dh = D // num_heads\n    def split(t):\n        return t.reshape(B, T, num_heads, dh).transpose(0, 2, 1, 3)\n    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)\n    mask = causal_mask(T) if causal else None\n    out = sdpa(q, k, v, mask=mask)\n    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)\n    return out @ Wo\n\ndef run_mha(causal):\n    np.random.seed(0)\n    B, T, D, H = 1, 3, 4, 2\n    x = np.random.randn(B, T, D)\n    Ws = [np.random.randn(D, D) * 0.1 for _ in range(4)]\n    return mha(x, *Ws, H, causal)"}
</script>
</div>

Splitting heads into the batch-like axis lets one `@` compute all heads at once. Each head costs $O(T^2d_h)$ compute and $O(T^2)$ score memory; in total this is $O(T^2D)$ compute and, for a naive implementation, $O(HT^2)$ score memory—the fundamental reason attention becomes expensive on long sequences.

## PyTorch module

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

> [!DANGER] Common bugs interviewers watch for
> Forgetting the $\sqrt{d}$ scale; softmax over the query axis instead of the key axis (`axis=-1` = keys); transposing the wrong two dims when merging heads (must be `(0,2,1,3)` inverse of the split); applying the mask *after* softmax; a non-`contiguous()` view before `reshape` in PyTorch; unstable softmax without the max subtraction.

## Q&A

<details class="qa"><summary>Why multiple heads instead of one big attention?</summary>
<div class="qa-body">

**Short:** heads let the model attend to different relationships (syntax, position, coreference) in different learned subspaces simultaneously, then combine them.

**Deep:** A single head produces one softmax distribution per query and applies the same token-mixing pattern to every value channel. Splitting $D$ into $H$ subspaces of $D/H$ gives $H$ patterns at nearly the same parameter/FLOP budget when $D$ is fixed. This adds structured expressivity; it is not a parameter-saving approximation. Naive score memory is still $O(BHT^2)$, and actual speed depends on head dimension and kernel shape. If the goal is to shrink the KV cache, distinguish MQA/GQA, which share K/V heads.
</div></details>

<details class="qa"><summary>What's the memory bottleneck and how does FlashAttention fix it?</summary>
<div class="qa-body">

**Short:** the $T\times T$ score matrix is $O(T^2)$ memory; FlashAttention never materializes it, computing attention in tiles with an online (streaming) softmax and staying in fast SRAM.

**Deep:** naive attention writes the full scores and weights to HBM. FlashAttention tiles $Q,K,V$ and computes online softmax with running maxima and denominators, reducing HBM traffic. It computes the same mathematical attention, although floating-point operation order means results need not be bitwise identical. `F.scaled_dot_product_attention` selects among Flash, memory-efficient, and math backends according to dtype, shape, device, and mask constraints, so do not claim it always dispatches to FlashAttention.
</div></details>

<details class="qa"><summary>Self-attention vs cross-attention?</summary>
<div class="qa-body">

**Short:** self-attention draws Q, K, V from the same sequence; cross-attention takes Q from one stream and K, V from another.

**Deep:** in a VLM decoder, text tokens self-attend, while cross-attention lets text queries pull from image/vision tokens as keys/values — the mechanism that grounds language in pixels. The code is identical; only the source tensors differ. Encoder–decoder Transformers use cross-attention in every decoder block; decoder-only VLMs often just concatenate vision tokens into the sequence and use plain self-attention. See [VLM Implementation Details](#/vlm/practical).
</div></details>

### Follow-ups
- **What about position?** Self-attention without positional information is **equivariant** to token permutations: permuting the input permutes the output in the same way. Inject order with positional encoding or RoPE.
- **MQA / GQA?** Share one (or a few) K/V heads across many query heads to shrink the KV-cache — key for inference; see the [Transformer block](#/ml-coding/transformer).
- **Additive (Bahdanau) vs dot-product attention?** Additive uses an MLP scorer — more params, no free matmul; dot-product won for hardware efficiency.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Formula | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| Intuition | soft dictionary lookup: similarity-weighted average of values |
| Scale | $1/\sqrt{d_k}$: keeps score variance $\approx 1$ |
| Softmax axis | over **keys** (`axis=-1`) |
| Masking | apply **before** softmax, fill $-\infty$; combine with AND |
| Head split | $(B,T,D)\!\to\!(B,H,T,D/h)$ |
| Head count | fixed $D$: nearly unchanged parameters/FLOPs, smaller $d_h$, naive score memory $O(BHT^2)$ |
| Complexity | total attention $O(T^2D)$ time, naive score memory $O(BHT^2)$ |
| Prod kernel | `F.scaled_dot_product_attention` selects Flash/memory-efficient/math backends when eligible |

**Cross-links:** [NumPy Primer](#/ml-coding/numpy-primer) · [Positional Encoding & RoPE](#/ml-coding/positional-encoding-rope) · [A Transformer Block](#/ml-coding/transformer) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [LLM Fundamentals](#/llm/fundamentals)
