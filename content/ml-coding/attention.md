# Attention From Scratch

> [!TIP] Say this first
> "Attention is a soft dictionary lookup: queries score every key with a dot product, we scale by $1/\sqrt{d}$ so the softmax doesn't saturate, softmax turns scores into weights, and we take the weighted average of the values." Then write the four lines. Every VLM and LLM you'll touch is built on this.

Implement scaled dot-product attention (with masking) and multi-head attention. This is *the* building block for the [Transformer block](#/ml-coding/transformer) and the entire modern stack — see [CNNs, RNNs & Transformers](#/foundations/architectures) for the conceptual side.

## The math

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

with $Q\in\mathbb{R}^{T_q\times d_k}$, $K\in\mathbb{R}^{T_k\times d_k}$, $V\in\mathbb{R}^{T_k\times d_v}$. Multi-head runs $h$ attentions in parallel on $d/h$-dim subspaces and concatenates:

$$
\text{MHA}(x) = \big[\text{head}_1;\dots;\text{head}_h\big]W^O,\quad
\text{head}_i=\text{Attention}(xW_i^Q, xW_i^K, xW_i^V)
$$

Try it live — adjust a query and watch the weights and output move:

<div class="widget" data-widget="attention"></div>

> [!TIP] Live code — implement, run, test
> The NumPy blocks below are **live editors**. Fill in the body, hit **▶ Run tests**, and watch the cases pass. Stuck? Reveal the reference **Solution** — but attempt first; the struggle *is* the practice. The first Run downloads a small Python runtime and NumPy (~15 MB); later runs are instant.

## Scaled dot-product attention (NumPy)

First the numerically stable **softmax** over the key axis:

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    # subtract the max for stability, exponentiate, then normalize\n    pass","tests":[{"args":[[[1,2,3]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[0,0,0]]],"expect":[[0.3333333333333333,0.3333333333333333,0.3333333333333333]]},{"args":[[[1000,1001,1002]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218]]},{"args":[[[-1,0,1],[1,0,-1]]],"expect":[[0.09003057317038046,0.24472847105479764,0.6652409557748218],[0.6652409557748218,0.24472847105479764,0.09003057317038046]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = np.asarray(x, dtype=float)\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)"}
</script>
</div>

Then **scaled dot-product attention** — score, scale, softmax, weight the values (this lab returns the output; the reference also returns the weights):

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sdpa","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = np.asarray(x, dtype=float)\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef sdpa(q, k, v, mask=None):\n    # scores = QK^T / sqrt(d); softmax over keys; return the weighted sum of V\n    pass","tests":[{"args":[[[1,0],[0,1]],[[1,0],[0,1]],[[1,2],[3,4]]],"expect":[[1.6604769013466862,2.6604769013466862],[2.3395230986533138,3.3395230986533138]]},{"args":[[[2,0,0]],[[2,0,0],[0,2,0],[0,0,2]],[[1,1],[2,2],[3,3]]],"expect":[[1.2485832277662388,1.2485832277662388]]},{"args":[[[1,1],[2,0]],[[1,0],[0,1]],[[10,0],[0,10]]],"expect":[[5.0,5.0],[8.04429682506957,1.9557031749304312]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = np.asarray(x, dtype=float)\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef sdpa(q, k, v, mask=None):\n    q, k, v = np.asarray(q, float), np.asarray(k, float), np.asarray(v, float)\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        mask = np.asarray(mask)\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    w = softmax(scores, axis=-1)\n    return w @ v"}
</script>
</div>

**Why $\sqrt{d}$:** if $q,k$ have unit-variance entries, $q\cdot k$ has variance $\approx d$, so raw scores grow with dimension and push softmax into saturated, near-one-hot regions with vanishing gradients. Dividing by $\sqrt{d}$ keeps the score variance $\approx 1$. *(verifiable)*

## Masking

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"causal_mask","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef causal_mask(T):\n    # lower-triangular boolean (T,T): True = allowed to attend (no peeking ahead)\n    pass","tests":[{"args":[2],"expect":[[1,0],[1,1]]},{"args":[3],"expect":[[1,0,0],[1,1,0],[1,1,1]]},{"args":[4],"expect":[[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]]}],"solution":"import numpy as np\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))"}
</script>
</div>

- **Causal mask** blocks position $t$ from seeing the future ($>t$) — required for autoregressive decoding.
- **Key-padding mask** blocks attention to `[PAD]` keys; shape `(B,1,1,Tk)`, broadcasts over heads and queries.
- Combine with logical **AND** (both must permit). Always apply the mask **before** softmax, filling blocked entries with a large negative number so they get ~0 weight.

## Multi-head attention

Implement `mha`; the harness `run_mha` builds a fixed, seeded `(B,T,D)=(1,3,4)` input with `H=2` heads so the output is deterministic (the helpers `softmax`/`causal_mask`/`sdpa` are provided):

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"run_mha","packages":["numpy"],"approx":true,"starter":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))\n\ndef sdpa(q, k, v, mask=None):\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    return softmax(scores, axis=-1) @ v\n\ndef mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):\n    # split (B,T,D)->(B,H,T,Dh), run sdpa per head, merge heads, project by Wo\n    pass\n\ndef run_mha(causal):\n    np.random.seed(0)\n    B, T, D, H = 1, 3, 4, 2\n    x = np.random.randn(B, T, D)\n    Ws = [np.random.randn(D, D) * 0.1 for _ in range(4)]\n    return mha(x, *Ws, H, causal)","tests":[{"args":[false],"expect":[[[0.03190098881683708,-0.003883638346883957,0.009816921823006634,0.0008853701048203459],[0.032178757789826054,-0.003954663472188265,0.008879630160056167,0.0011910101137743716],[0.03203148496155561,-0.0038215886062526196,0.009498626868409653,0.0010096474007127018]]]},{"args":[true],"expect":[[[0.059733984570268545,-0.003891324159579386,0.01245601832145022,0.013606293601693605],[0.04292552968524655,-0.008439370891800252,0.018326373103144715,-0.004398635006322705],[0.03203148496155561,-0.0038215886062526196,0.009498626868409653,0.0010096474007127018]]]}],"solution":"import numpy as np\n\ndef softmax(x, axis=-1):\n    x = x - np.max(x, axis=axis, keepdims=True)\n    e = np.exp(x)\n    return e / np.sum(e, axis=axis, keepdims=True)\n\ndef causal_mask(T):\n    return np.tril(np.ones((T, T), dtype=bool))\n\ndef sdpa(q, k, v, mask=None):\n    d = q.shape[-1]\n    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9) if mask.dtype == bool else scores + mask\n    return softmax(scores, axis=-1) @ v\n\ndef mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):\n    B, T, D = x.shape\n    dh = D // num_heads\n    def split(t):\n        return t.reshape(B, T, num_heads, dh).transpose(0, 2, 1, 3)\n    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)\n    mask = causal_mask(T) if causal else None\n    out = sdpa(q, k, v, mask=mask)\n    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)\n    return out @ Wo\n\ndef run_mha(causal):\n    np.random.seed(0)\n    B, T, D, H = 1, 3, 4, 2\n    x = np.random.randn(B, T, D)\n    Ws = [np.random.randn(D, D) * 0.1 for _ in range(4)]\n    return mha(x, *Ws, H, causal)"}
</script>
</div>

Splitting heads into the batch-like axis lets one `@` compute all heads at once. **Complexity:** $O(T^2 d)$ compute and $O(T^2)$ memory for the score matrix, per head.

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
    # causal: position 0 must place zero weight on future keys
    q = rng.standard_normal((1, 1, 4, 8))
    _, w = sdpa(q, q, q, mask=causal_mask(4))
    assert np.allclose(w[0, 0, 0, 1:], 0.0, atol=1e-6)
    assert np.allclose(w.sum(-1), 1.0)          # weights are a distribution
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

**Deep:** naive attention writes the full scores/weights to HBM — memory-bandwidth-bound and quadratic in sequence length. FlashAttention tiles $Q,K,V$, keeps running max and running denominator per query block (the log-sum-exp trick applied online), and fuses the whole op into one kernel. Same output, $O(T)$ extra memory, far fewer HBM round-trips. `F.scaled_dot_product_attention` dispatches to it automatically. *(verifiable)*
</div></details>

<details class="qa"><summary>Self-attention vs cross-attention?</summary>
<div class="qa-body">

**Short:** self-attention draws Q, K, V from the same sequence; cross-attention takes Q from one stream and K, V from another.

**Deep:** in a VLM decoder, text tokens self-attend, while cross-attention lets text queries pull from image/vision tokens as keys/values — the mechanism that grounds language in pixels. The code is identical; only the source tensors differ. Encoder–decoder Transformers use cross-attention in every decoder block; decoder-only VLMs often just concatenate vision tokens into the sequence and use plain self-attention. See [VLM Implementation Details](#/vlm/practical).
</div></details>

### Follow-ups
- **MQA / GQA?** Share one (or a few) K/V heads across many query heads to shrink the KV-cache — key for inference; see the [Transformer block](#/ml-coding/transformer).
- **RoPE vs learned positions?** Rotary embeddings inject *relative* position by rotating Q and K; the modern default in LLaMA/Qwen/Mistral.
- **Additive (Bahdanau) vs dot-product attention?** Additive uses an MLP scorer — more params, no free matmul; dot-product won for hardware efficiency.

## Cheat-sheet

| Item | Value |
| --- | --- |
| Formula | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| Scale | $1/\sqrt{d_k}$: keeps score variance $\approx 1$ |
| Softmax axis | over **keys** (`axis=-1`) |
| Masking | apply **before** softmax, fill $-\infty$; combine with AND |
| Head split | $(B,T,D)\!\to\!(B,H,T,D/h)$ |
| Complexity | $O(T^2 d)$ time, $O(T^2)$ score memory |
| Prod kernel | `F.scaled_dot_product_attention` (FlashAttention) |

**Cross-links:** [A Transformer Block](#/ml-coding/transformer) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [LLM Fundamentals](#/llm/fundamentals) · [The ML Coding Round](#/ml-coding/intro)
