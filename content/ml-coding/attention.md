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

Under the initialization approximation that the components of $q$ and $k$ are roughly independent with mean 0 and variance 1, the variance of $q\cdot k=\sum_{i=1}^d q_ik_i$ is approximately $d$. Without scaling, logits grow with dimension, pushing softmax toward saturation and potentially making gradients very small. Dividing by $\sqrt{d}$ keeps the score variance near 1 under this approximation. This motivates a stable initial scale; it does not claim that learned components remain perfectly independent. See [Linear Algebra & Calculus](#/foundations/linear-algebra-calculus).

## Masking

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"causal_mask","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef causal_mask(T):\n    # lower-triangular boolean (T,T): True = allowed to attend (no peeking ahead)\n    pass","tests":[{"args":[2],"expect":[[1,0],[1,1]]},{"args":[3],"expect":[[1,0,0],[1,1,0],[1,1,1]]},{"args":[4],"expect":[[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]]}],"solution":"import numpy as np\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))"}
</script>
</div>

- A **causal mask** blocks position $t$ from seeing the future ($>t$), which is required for autoregressive next-token decoding.
- A **key-padding mask** blocks attention to `[PAD]` keys. Its shape is `(B,1,1,Tk)`, broadcasting over heads and queries.
- Combine them with logical **AND**, since both must permit an entry. Apply a mask **before** softmax. Set blocked Boolean entries to `-inf`, but note that a row with every key blocked is meaningless: callers must allow at least one key or define an explicit zero-output policy.

## Multi-head attention

Why use several heads instead of one large attention operation? Intuitively, one head consults tokens from only one “point of view.” Splitting the $d$ dimensions into $h$ smaller subspaces allows several perspectives—such as syntax, position, and meaning—to operate simultaneously at the **same asymptotic compute**, before their outputs are recombined.

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

Splitting heads into the batch-like axis lets one `@` compute all heads at once. **Complexity:** $O(T^2 d)$ compute and $O(T^2)$ memory for the score matrix, per head—the fundamental reason attention becomes expensive on long sequences.

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

**Deep:** a single head produces one softmax distribution per query — one "reason" to attend. Splitting $d$ into $h$ subspaces of $d/h$ gives $h$ independent attention patterns at the *same* total compute (the projections shrink proportionally). Empirically different heads specialize; it's a cheap form of ensembling in representation space. The cost is unchanged because $h\cdot(d/h)=d$.
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
| Complexity | $O(T^2 d)$ time, $O(T^2)$ score memory |
| Prod kernel | `F.scaled_dot_product_attention` selects Flash/memory-efficient/math backends when eligible |

**Cross-links:** [NumPy Primer](#/ml-coding/numpy-primer) · [Positional Encoding & RoPE](#/ml-coding/positional-encoding-rope) · [A Transformer Block](#/ml-coding/transformer) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [LLM Fundamentals](#/llm/fundamentals)
