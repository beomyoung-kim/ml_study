# Decoding & Sampling Strategies

> [!NOTE] Goal of this chapter
> In [next-token prediction](#/llm/next-token), the model produces a **probability distribution** over the next token. But how do we choose **which token to emit** from that distribution? The choice of decoding and sampling method can make the same model "accurate but dull" or "creative but erratic." This chapter builds an intuitive, visual, and code-level understanding of temperature, top-k, and top-p.

## What and why

At every step, an LLM produces a probability distribution over its entire vocabulary—for example, "cat" 0.6, "dog" 0.2, and so on. **Decoding** is the rule used to select the next token from that distribution. Different rules produce very different outputs:

- Always choose the highest-probability token (greedy) → deterministic, but low-diversity and prone to repetition
- Sample randomly according to the probabilities → diverse, but occasionally off-track

Controls such as **temperature, top-k, and top-p** tune the balance between these extremes.

> [!TIP] Interview one-liner
> "Temperature controls how sharp the distribution is, while top-k and top-p control the candidate set. For tasks with easy-to-check answers, I usually start with low diversity; for creative work or candidate exploration, I widen the sampling distribution." Add that the best values depend on the model, API, task, and verifier.

## Greedy — always choose number one

$$\hat{x}_t = \arg\max_i\ p(x_t = i \mid x_{<t})$$

This is the simplest rule: choose the highest-probability token at every step. **The algorithm itself uses no randomness**, but bit-for-bit reproducibility for the same input can still break because of server model versions, kernels, batching, or tie-breaking. Greedy decoding has no diversity, can repeat the same phrase, and is a useful starting point for short extraction or classification tasks with a constrained output space.

## Temperature — control distribution sharpness

Divide the logits by temperature $T$ before softmax:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T \to 0^+$: the distribution becomes **sharper** → converges to the greedy distribution when the maximum is unique
- $T = 1$: the model's original learned distribution
- $T > 1$: the distribution becomes **flatter** → more diverse and random

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="110" y="20" text-anchor="middle" font-weight="700" fill="#0ea5e9">Low T (sharp)</text>
  <g fill="#0ea5e9">
    <rect x="60" y="40" width="24" height="120"/><rect x="92" y="140" width="24" height="20"/><rect x="124" y="150" width="24" height="10"/><rect x="156" y="155" width="24" height="5"/>
  </g>
  <text x="110" y="180" text-anchor="middle" fill="#98a3b2">Top-token probability ↑ (still sampling)</text>

  <text x="330" y="20" text-anchor="middle" font-weight="700" fill="#12a150">T = 1 (original)</text>
  <g fill="#12a150">
    <rect x="280" y="70" width="24" height="90"/><rect x="312" y="105" width="24" height="55"/><rect x="344" y="130" width="24" height="30"/><rect x="376" y="145" width="24" height="15"/>
  </g>
  <text x="330" y="180" text-anchor="middle" fill="#98a3b2">Learned distribution unchanged</text>

  <text x="550" y="20" text-anchor="middle" font-weight="700" fill="#e0533f">High T (flat)</text>
  <g fill="#e0533f">
    <rect x="500" y="95" width="24" height="65"/><rect x="532" y="105" width="24" height="55"/><rect x="564" y="112" width="24" height="48"/><rect x="596" y="118" width="24" height="42"/>
  </g>
  <text x="550" y="180" text-anchor="middle" fill="#98a3b2">More even → diverse and random</text>
</svg>
<figcaption>Temperature turns the same logits into different probability distributions. A low positive T concentrates probability on the top token; a high T spreads it more evenly across candidates. As long as sampling remains enabled, low T is not strictly deterministic. APIs usually handle T=0 as a separate greedy path rather than substituting it into the formula.</figcaption>
</figure>

## Top-k — keep only the top k candidates

Keep only the $k$ highest-probability tokens, set the rest to zero, then renormalize and sample within that set. This prevents absurd tokens from the tail from being selected. The downside is that the right $k$ varies by context: a large $k$ is excessive for a sharp distribution and insufficient for a flat one.

## Top-p (nucleus) — retain probability mass up to p

Sort tokens by probability and keep adding them until their cumulative probability reaches $p$—for example, 0.9. Unlike top-k, the number of candidates adapts to the context: one or two when the model is confident, dozens when it is uncertain.

<figure>
<svg viewBox="0 0 620 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <g fill="#12a150"><rect x="40" y="40" width="40" height="80"/><rect x="90" y="70" width="40" height="50"/><rect x="140" y="95" width="40" height="25"/></g>
  <g fill="#98a3b2" opacity="0.4"><rect x="190" y="108" width="40" height="12"/><rect x="240" y="113" width="40" height="7"/><rect x="290" y="116" width="40" height="4"/></g>
  <path d="M185 30 V130" stroke="#e0533f" stroke-width="2" stroke-dasharray="5 4"/>
  <text x="110" y="30" fill="#12a150" font-weight="700">nucleus (cumulative 0.9)</text>
  <text x="360" y="80" fill="#98a3b2">← discarded tail</text>
  <text x="180" y="145" text-anchor="middle" fill="#98a3b2">Add probabilities in descending order; keep through the 0.9 crossing</text>
</svg>
<figcaption>Top-p retains the smallest set—the nucleus—whose total probability mass is at least 90%. Green denotes retained candidates; gray denotes the discarded tail.</figcaption>
</figure>

**Min-p** is a context-adaptive variant that keeps tokens whose probability exceeds a fixed fraction of the highest probability. It can be useful for some models and settings, but it is not universally more robust than top-p. **Beam search** maintains several candidate sequences at once and searches for a sequence with a high cumulative score. It can help in tasks such as translation, but length bias, repetition, and reduced diversity mean that length penalties and task-specific evaluation are still needed.

<details class="concept-code">
<summary>View conceptual code</summary>

> The following PyTorch-style **pseudocode** shows the filtering order for one decode step. Real APIs can apply penalties, top-k, and top-p in a different order.

```python
def sample_next(logits, temperature, top_k, top_p, banned_ids, rng):
    # logits: [B,V]. Route temperature=0 to an argmax path, not this function.
    assert temperature > 0
    z = logits.float() / temperature                # Filter in fp32.
    z[:, banned_ids] = -inf                         # Policy mask, e.g. forbid EOS.

    if top_k is not None:
        kth = topk(z, k=top_k).values[:, -1, None]
        z = z.masked_fill(z < kth, -inf)

    if top_p is not None:
        z_sorted, ids_sorted = sort(z, descending=True)
        p_sorted = softmax(z_sorted, dim=-1)
        remove = cumsum(p_sorted, dim=-1) > top_p
        # Keep the first token that crosses the threshold so the nucleus is nonempty.
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        z = scatter_back(z_sorted.masked_fill(remove, -inf), ids_sorted)

    if any_row_has_no_finite_logit(z):
        raise InvalidDecodingPolicy("all candidates were masked")
    probs = softmax(z, dim=-1)
    return multinomial(probs, num_samples=1, generator=rng)  # Record the seed.
```

</details>

## Which method when?

| Situation | Recommended starting point |
| --- | --- |
| Math, code, or extraction with verifiable outputs | Start with low diversity and validate with tests or schemas |
| General conversation and QA | Start from the provider default; adjust temperature or top-p one axis at a time |
| Creative writing and brainstorming | Sample more broadly, then apply quality and safety checks |
| Generate and verify several candidates (self-consistency, best-of-N) | Sample N outputs with $T>0$, then verify → [Reasoning](#/llm/reasoning) |

These are not fixed recipes. Providers use different temperature ranges and filtering orders, and instruction-tuned or reasoning models can have different recommended defaults. Measure pass@k, diversity, latency, and token cost—not just accuracy.

> [!NOTE] Connection to reasoning
> [Test-time compute](#/llm/reasoning) methods such as self-consistency and best-of-N **sample several answers from one prompt** and then select by majority vote or verification. They require T>0 to produce diverse candidates, which is why decoding strategy directly affects reasoning performance.

## Try it yourself — temperature softmax

Given logits and temperature $T$, return the probability distribution. Divide by $T$, then apply a numerically stable softmax.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"softmax_temperature","packages":["numpy"],"approx":true,"starter":"def softmax_temperature(logits, T=1.0):\n    # T must be positive. Raise ValueError for T=0 or a negative value.\n    # Divide logits by T, then return numerically stable softmax probabilities.\n    pass","tests":[{"args":[[1.0,1.0,1.0],1.0],"expect":[0.3333333,0.3333333,0.3333333],"tol":1e-4},{"args":[[2.0,1.0,0.0],1.0],"expect":[0.6652409,0.2447284,0.0900306],"tol":1e-4},{"args":[[2.0,1.0,0.0],0.5],"expect":[0.8668132,0.1173104,0.0158764],"tol":1e-4}],"solution":"import numpy as np\n\ndef softmax_temperature(logits, T=1.0):\n    if T <= 0:\n        raise ValueError('T must be positive; use argmax separately for greedy')\n    z = np.asarray(logits, dtype=float) / T\n    z = z - z.max()          # Numerical stability.\n    e = np.exp(z)\n    return (e / e.sum()).tolist()"}
</script>
</div>

Look at the third test: lowering the temperature to $T=0.5$ sharpens the top probability from 0.665 to 0.867. You can see exactly how temperature tightens the distribution.

## Q&A

<details class="qa"><summary>Do you use top-k and top-p together?</summary>
<div class="qa-body">

**Short:** Yes. Production APIs commonly combine temperature with top-p, and sometimes top-k as well.

**Deep:** They can be combined, but the **application order depends on the implementation and API**. One implementation might adjust temperature, apply top-k and top-p, then renormalize. Stacking controls makes causal diagnosis harder because they interact, so change one at a time and verify the library's actual ordering.
</div></details>

<details class="qa"><summary>Why does greedy decoding often fall into repetition?</summary>
<div class="qa-body">

**Short:** Always choosing the locally best token can trap the model in a loop that returns to its own high-probability phrase.

**Deep:** Greedy decoding does not maximize the probability of the complete sequence; it makes a greedy choice at every step. Once it enters a repetitive pattern, each step can reinforce the next. Repetition penalties and no-repeat n-gram rules alter the distribution and can also damage legitimate repetition, code, or proper nouns, so validate them. Diagnose stopping criteria, the prompt, and training-data effects as well.
</div></details>

## Cheat-sheet

| Control | What it controls | Effect |
| --- | --- | --- |
| greedy | — | Always choose number one; no RNG, though end-to-end bit reproducibility is separate; repetition risk |
| temperature $T$ | Distribution sharpness ($T>0$) | ↓ concentrates on the top token / ↑ increases diversity; T=0 is usually a separate greedy path |
| top-k | Fixed candidate count k | Removes the tail; requires tuning k |
| top-p (nucleus) | Candidates up to cumulative probability p | Context-adaptive candidate set |
| min-p | Cutoff relative to the top probability | Context-adaptive filtering; validate per model |
| beam search | Several sequences at once | Useful for translation, summarization, and similar tasks |

**Next:** [Prompting & ICL](#/llm/prompting) · [Reasoning & Test-Time Compute](#/llm/reasoning)
