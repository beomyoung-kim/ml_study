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

## Scaled dot-product attention (NumPy)

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # stability: subtract max
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sdpa(q, k, v, mask=None):
    """
    q,k: (..., Tq/Tk, d),  v: (..., Tk, dv)
    mask: broadcastable to (..., Tq, Tk); bool True=keep, or additive (-inf blocks)
    returns: out (..., Tq, dv), weights (..., Tq, Tk)
    """
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)   # (..., Tq, Tk)
    if mask is not None:
        scores = np.where(mask, scores, -1e9) if mask.dtype == bool \
                 else scores + mask
    w = softmax(scores, axis=-1)
    return w @ v, w
```

**Why $\sqrt{d}$:** if $q,k$ have unit-variance entries, $q\cdot k$ has variance $\approx d$, so raw scores grow with dimension and push softmax into saturated, near-one-hot regions with vanishing gradients. Dividing by $\sqrt{d}$ keeps the score variance $\approx 1$. *(verifiable)*

## Masking

```python
def causal_mask(T):
    """Lower-triangular bool: True = allowed to attend. (T,T)."""
    return np.tril(np.ones((T, T), dtype=bool))
```

- **Causal mask** blocks position $t$ from seeing the future ($>t$) — required for autoregressive decoding.
- **Key-padding mask** blocks attention to `[PAD]` keys; shape `(B,1,1,Tk)`, broadcasts over heads and queries.
- Combine with logical **AND** (both must permit). Always apply the mask **before** softmax, filling blocked entries with a large negative number so they get ~0 weight.

## Multi-head attention

```python
def mha(x, Wq, Wk, Wv, Wo, num_heads, causal=False):
    """x:(B,T,D)  W*:(D,D) -> (B,T,D)."""
    B, T, D = x.shape
    assert D % num_heads == 0
    dh = D // num_heads

    def split(t):                       # (B,T,D) -> (B,H,T,Dh)
        return t.reshape(B, T, num_heads, dh).transpose(0, 2, 1, 3)

    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    mask = causal_mask(T) if causal else None      # (T,T) broadcasts over B,H
    out, _ = sdpa(q, k, v, mask=mask)              # (B,H,T,Dh)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)   # merge heads
    return out @ Wo
```

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
