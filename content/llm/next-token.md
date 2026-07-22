# Next-Token Prediction, Intuitively

> [!NOTE] Goal of this chapter
> The **pretraining objective** of many generative LLMs is to predict the distribution of the next token. Knowledge, translation, coding, and some reasoning capabilities can emerge from the combination of scale, data, architecture, and post-training, but this one objective does not automatically guarantee every capability. See [Tokenization & BPE](#/llm/tokenization) for what a token is and [Decoding & Sampling](#/llm/decoding-sampling) for how a token is actually selected.

## What and why

Given a prefix, or context, the model assigns a **probability to every token in its vocabulary**. That is the complete output: one probability distribution over the next token.

For example, after "the cat sat on the ___," the distribution might look like this:

<figure>
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="13">
  <text x="20" y="28" fill="currentColor">Context (prefix): "the cat sat on the __"</text>
  <text x="20" y="48" fill="#98a3b2" font-size="11">→ the model's next-token probability distribution:</text>
  <g>
    <rect x="130" y="70" width="300" height="20" fill="#e0533f"/><text x="70" y="85" text-anchor="end" fill="currentColor">mat</text><text x="440" y="85" fill="#e0533f">0.55</text>
    <rect x="130" y="98" width="110" height="20" fill="#6366f1"/><text x="70" y="113" text-anchor="end" fill="currentColor">floor</text><text x="250" y="113" fill="#6366f1">0.20</text>
    <rect x="130" y="126" width="82" height="20" fill="#0ea5e9"/><text x="70" y="141" text-anchor="end" fill="currentColor">table</text><text x="222" y="141" fill="#0ea5e9">0.15</text>
    <rect x="130" y="154" width="55" height="20" fill="#98a3b2"/><text x="70" y="169" text-anchor="end" fill="currentColor">roof …</text><text x="195" y="169" fill="#98a3b2">0.10</text>
  </g>
  <text x="280" y="205" text-anchor="middle" fill="#98a3b2">Probabilities sum to 1.0 · natural next tokens receive high probability</text>
</svg>
<figcaption>Next-token prediction: given context, output a probability distribution over the full vocabulary. A well-trained model gives natural continuations such as "mat" a high probability.</figcaption>
</figure>

Why a probability **distribution**? Language rarely has only one valid answer. "The cat sat on the ___" can end in mat, floor, or table. The model learns not "it must be this one token," but "how plausible is each candidate?"

Formally, given context $x_{<t}$, the model predicts the conditional probability of the next token $x_t$, $p_\theta(x_t \mid x_{<t})$. Training maximizes the correct token's log probability over the corpus, which is equivalent to minimizing negative log-likelihood:

$$
\mathcal{L}(\theta) = -\sum_{t} \log p_\theta(x_t \mid x_{<t})
$$

> [!TIP] Interview one-liner
> "A modern LLM is trained on trillions of tokens with the single objective of **next-token prediction**. Long context, inexpensive inference, MoE, and RLHF all arise where that objective meets constraints from **data**, **serving cost**, and **alignment**." Start from the objective, then derive the surrounding topics.

## Autoregressive generation: predict → append → repeat

To **generate** a sentence, repeat the prediction process. First obtain a next-token distribution, select or sample one token, append it to the context, and start again. This loop is called **autoregressive generation**.

<figure>
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- prefix -->
  <rect x="20" y="30" width="180" height="34" rx="6" fill="none" stroke="#6366f1" stroke-width="1.5"/>
  <text x="110" y="52" text-anchor="middle" fill="currentColor">"the cat sat on the"</text>
  <text x="110" y="22" text-anchor="middle" fill="#98a3b2" font-size="10">① context (prefix)</text>
  <path d="M200 47 H236" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#nt)"/>
  <!-- model -->
  <rect x="240" y="30" width="70" height="34" rx="6" fill="#e0533f"/>
  <text x="275" y="52" text-anchor="middle" fill="#fff">model</text>
  <path d="M310 47 H346" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#nt)"/>
  <!-- prob bars -->
  <text x="350" y="22" fill="#98a3b2" font-size="10">② probability distribution</text>
  <g>
    <rect x="410" y="30" width="130" height="12" fill="#e0533f"/><text x="404" y="40" text-anchor="end" fill="currentColor" font-size="10">mat</text>
    <rect x="410" y="46" width="47" height="12" fill="#6366f1"/><text x="404" y="56" text-anchor="end" fill="currentColor" font-size="10">floor</text>
    <rect x="410" y="62" width="35" height="12" fill="#0ea5e9"/><text x="404" y="72" text-anchor="end" fill="currentColor" font-size="10">table</text>
  </g>
  <!-- sample highlight (animated) -->
  <rect x="408" y="28" width="134" height="16" rx="3" fill="none" stroke="#12a150" stroke-width="2">
    <animate attributeName="opacity" values="0.25;1;0.25" dur="1.8s" repeatCount="indefinite"/>
  </rect>
  <text x="560" y="40" fill="#12a150" font-size="10">③ sample → "mat"</text>
  <!-- append box -->
  <rect x="150" y="150" width="240" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.5"/>
  <text x="270" y="172" text-anchor="middle" fill="currentColor">"the cat sat on the <tspan fill="#12a150" font-weight="700">mat</tspan>"</text>
  <text x="270" y="142" text-anchor="middle" fill="#98a3b2" font-size="10">④ append to form new context</text>
  <!-- arrow from sample down to append -->
  <path d="M500 66 C 500 130, 400 120, 392 155" stroke="#12a150" stroke-width="1.5" fill="none" marker-end="url(#ntg)"/>
  <!-- loop back to prefix -->
  <path d="M150 167 C 60 167, 40 100, 60 64" stroke="#6366f1" stroke-width="1.5" fill="none" stroke-dasharray="5 4" marker-end="url(#ntb)">
    <animate attributeName="stroke-dashoffset" values="18;0" dur="0.9s" repeatCount="indefinite"/>
  </path>
  <text x="40" y="120" text-anchor="middle" fill="#6366f1" font-size="10" transform="rotate(-90 40 120)">↺ repeat</text>
  <!-- traveling dot along the loop -->
  <circle r="3.5" fill="#6366f1">
    <animateMotion path="M150 167 C 60 167, 40 100, 60 64" dur="1.8s" repeatCount="indefinite"/>
  </circle>
  <defs>
    <marker id="nt" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker>
    <marker id="ntg" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
    <marker id="ntb" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#6366f1"/></marker>
  </defs>
</svg>
<figcaption>The autoregressive generation loop: context → model outputs a probability distribution → sample one token → append it to the context → repeat. The sequence grows by exactly one token per step, so a longer answer requires correspondingly more forward passes.</figcaption>
</figure>

The question "how do we choose one token from the distribution in step ②?"—highest probability, random sampling, temperature adjustment—is the subject of [Decoding & Sampling](#/llm/decoding-sampling). This chapter focuses on the preceding question: **how does the model predict and learn the distribution itself?**

## Training is parallel: teacher forcing

Generation is sequential and therefore slow. **Training** is much faster because the correct sentence is already known. The model can predict and score the next token at **every position simultaneously in one forward pass**.

Suppose we train on "the cat sat on the mat." The model produces all of the following predictions **in parallel**:

| Context visible at a position | Correct target |
| --- | --- |
| the | cat |
| the cat | sat |
| the cat sat | on |
| the cat sat on | the |
| the cat sat on the | mat |

At each position, the model sees the **ground-truth prefix**, not its own earlier output. This is called **teacher forcing**. Positions do not need to wait for one another, so the GPU evaluates them together. An [attention](#/foundations/architectures) **causal mask** prevents any position from peeking at future tokens.

> [!NOTE] Why can a simple objective look like intelligence? A compression intuition
> Better probabilistic prediction also shortens expected code length under arithmetic coding, so prediction and **compression** are mathematically connected. This supports the intuition that the objective pressures the model to encode grammar, facts, and patterns in its internal representations. The analogy does not prove causal understanding, reliability, or safety. Low prediction error does not guarantee a useful answer; [Post-Training & Alignment](#/llm/alignment) and task-specific evaluation remain necessary.

## The loss is cross-entropy

**Cross-entropy** scores the prediction at each position. The rule is simple: apply $-\log$ to the probability assigned to the correct token.

$$
L = -\log p_\theta(x_t \mid x_{<t})
$$

Assigning probability 0.9 to the correct token yields $-\log 0.9 \approx 0.1$, which is small and good. Assigning only 0.1 yields $-\log 0.1 \approx 2.3$, which is large and bad. $-\log$ strongly penalizes being confidently wrong. [Loss & Gradient, from Scratch](#/ml-coding/losses-gradients) derives why this is the right classification loss and why the softmax gradient simplifies to $p - \text{onehot}(y)$. Next-token prediction is, in effect, a huge classification problem with one class per vocabulary token.

## Try it yourself — next-token cross-entropy

Given a predicted next-token distribution `probs` and the index `target` of the correct token, compute the cross-entropy loss at that position: $-\log(\text{probs}[\text{target}])$.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"next_token_ce","packages":["numpy"],"approx":true,"starter":"def next_token_ce(probs, target):\n    # probs: next-token probability distribution, represented as a list that sums to 1.\n    # target: integer index of the correct token.\n    # Return cross-entropy = -log(probs[target]).\n    pass","tests":[{"args":[[0.6,0.3,0.1],0],"expect":0.5108256237659907,"tol":1e-6},{"args":[[0.25,0.25,0.25,0.25],2],"expect":1.3862943611198906,"tol":1e-6},{"args":[[0.9,0.1],0],"expect":0.10536051565782628,"tol":1e-6},{"args":[[0.1,0.9],0],"expect":2.302585092994046,"tol":1e-6}],"solution":"import numpy as np\n\ndef next_token_ce(probs, target):\n    p = np.asarray(probs, dtype=float)\n    return float(-np.log(p[target]))"}
</script>
</div>

Giving the correct token probability 1.0 produces zero loss—a perfect prediction—and lower probabilities produce larger losses. The training loss for a sentence averages this value across positions.

## How well does it predict? Perplexity

Applying $\exp$ to a sentence's average cross-entropy yields **perplexity**, a classical measure of how surprised the model is by the next token on average.

$$
\text{perplexity} = \exp\!\Big(-\frac{1}{N}\sum_{i=1}^{N}\log p_i\Big)
$$

Here, $p_i$ is the probability assigned to the **correct token** at each position. Perplexity 10 can be loosely visualized as uncertainty among ten candidates on average, but it is not an actual candidate count. Lower is better under the same evaluation unit. **Different tokenizers segment text differently and change $N$, so raw perplexity is difficult to compare across tokenizers.** The data and preprocessing must match as well, and downstream usefulness requires separate evaluation.

## Q&A

<details class="qa"><summary>The model predicts tokens rather than words. What is the difference?</summary>
<div class="qa-body">

**Short:** Tokens are often subword pieces smaller than words.

**Deep:** A model operates on integer **token IDs**, not directly on characters or words. "Cats" might be split into `cat` + `s`, and a less frequent word such as "unbelievable" can also be split into several pieces. [Tokenization & BPE](#/llm/tokenization) explains how the vocabulary is constructed. "Next-token prediction" is the precise term; intuitively, think "predict the next piece." The first step that turns an integer ID into a vector is an [embedding](#/llm/embeddings).
</div></details>

<details class="qa"><summary>Training uses teacher forcing, but generation feeds back the model's own output. Is that a problem?</summary>
<div class="qa-body">

**Short:** Yes. This gap is called **exposure bias**.

**Deep:** During training the model sees the correct prefix, whereas during generation it predicts from its own previous outputs, so errors can accumulate. This is commonly described as exposure bias, but it is not the sole cause of every generation error, nor should one claim that generic alignment or decoding "solves" it. Compare data, objectives, sequence-level training, and inference policies for the particular task.
</div></details>

<details class="qa"><summary>Why predict a full distribution instead of a single probability?</summary>
<div class="qa-body">

**Short:** Language has multiple valid continuations, and a distribution enables both a cross-entropy training signal and diverse generation through sampling.

**Deep:** A single choice cannot represent uncertainty well and does not provide the standard cross-entropy training signal. A softmax over the full vocabulary enables both learning and sampling; see [Decoding & Sampling](#/llm/decoding-sampling). **Weight tying**, where the final output matrix shares weights with the input embedding, is common but not required by every architecture.
</div></details>

## Cheat-sheet

| Concept | One line |
| --- | --- |
| Objective | Context, or prefix → predict a **probability distribution** over the next token |
| Loss | Negative log-likelihood of the correct token = cross-entropy $-\log p_y$ |
| Generation | Autoregressive: predict → sample → append → repeat, one token at a time |
| Teacher forcing | During training, use correct prefixes to predict all positions **simultaneously in parallel** |
| Exposure bias | Gap between training on correct prefixes and generating from the model's own output |
| Perplexity | $\exp(-\text{mean}\log p)$; compare only with the same tokenizer, data, and preprocessing |
| Emergent capability | The prediction objective creates representation-learning pressure, but scale, data, architecture, post-training, and evaluation jointly determine capability |

**Next:** [Decoding & Sampling](#/llm/decoding-sampling) · [LLM Fundamentals](#/llm/fundamentals) · [Loss & Gradient](#/ml-coding/losses-gradients) · [Tokenization & BPE](#/llm/tokenization)
