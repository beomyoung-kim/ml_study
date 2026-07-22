# Implementing Positional Encoding & RoPE

> [!NOTE] Goal of this chapter
> [Attention](#/ml-coding/attention) has one fatal gap: it **does not know order**. It treats "the dog bit the person" and "the person bit the dog" the same way. This chapter fills that gap with **positional encoding**, using diagrams and short code to cover everything from classic sinusoidal encoding to today's standard, **RoPE**.

## §0 · Why position is necessary

Attention treats tokens like a **set**. Each query takes dot products with every key and forms a weighted sum, and this computation changes only by the corresponding permutation when the token order changes. Mathematically, it is **permutation-equivariant**. In other words, attention has no concept of position by itself.

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="13">
  <text x="20" y="30" fill="#98a3b2">Seen as an unordered set:</text>
  <g fill="none" stroke="#6366f1" stroke-width="1.6">
    <circle cx="120" cy="70" r="22"/><circle cx="200" cy="55" r="22"/><circle cx="270" cy="95" r="22"/><circle cx="340" cy="60" r="22"/>
  </g>
  <text x="120" y="75" text-anchor="middle" fill="currentColor">dog</text>
  <text x="200" y="60" text-anchor="middle" fill="currentColor">bit</text>
  <text x="270" y="100" text-anchor="middle" fill="currentColor">person</text>
  <text x="340" y="65" text-anchor="middle" fill="currentColor">?</text>
  <path d="M420 70 H470" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#pa)"/>
  <text x="560" y="60" text-anchor="middle" fill="#e0533f">Without position,</text>
  <text x="560" y="80" text-anchor="middle" fill="#e0533f">subject and object blur</text>
  <defs><marker id="pa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>Attention itself does not know which token comes first, second, or third. We therefore add a <b>position signal</b> to every token (sinusoidal/learned) or encode it through rotation (RoPE).</figcaption>
</figure>

There are two broad solutions: **① add a position vector to the input embedding** (sinusoidal or learned absolute encoding), or **② rotate the query and key as a function of position** (RoPE). Let us examine each.

## §1 · Sinusoidal positional encoding

The original Transformer (Vaswani et al., 2017) creates, for every position $pos$, a vector filled with sinusoids at **different frequencies**, then adds that vector to the embedding. The frequency decreases across dimension pairs $2i, 2i{+}1$.

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

Intuitively, the combination of a clock's **fast second hand, slower minute hand, and still slower hour hand** gives every position a unique **fingerprint**. Early dimensions rotate quickly and resolve fine positions; later dimensions rotate slowly and cover wider ranges.

<figure>
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="30" y1="100" x2="620" y2="100" stroke="#98a3b2" stroke-width="1"/>
  <text x="620" y="118" text-anchor="end" fill="#98a3b2">position →</text>
  <path d="M30 100 Q 45 40 60 100 T 90 100 T 120 100 T 150 100 T 180 100 T 210 100 T 240 100 T 270 100 T 300 100 T 330 100 T 360 100 T 390 100 T 420 100 T 450 100 T 480 100 T 510 100 T 540 100 T 570 100 T 600 100" fill="none" stroke="#e0533f" stroke-width="1.6" opacity="0.9"/>
  <path d="M30 100 Q 75 55 120 100 T 210 100 T 300 100 T 390 100 T 480 100 T 570 100" fill="none" stroke="#6366f1" stroke-width="1.8"/>
  <path d="M30 100 Q 150 30 270 100 T 510 100" fill="none" stroke="#0ea5e9" stroke-width="2"/>
  <text x="40" y="150" fill="#e0533f">dimension 0 (high frequency, fast)</text>
  <text x="260" y="165" fill="#6366f1">dimension 2 (medium)</text>
  <text x="440" y="150" fill="#0ea5e9">dimension 4 (low frequency, slow)</text>
</svg>
<figcaption>Each dimension uses a wave at a different frequency. Stack the (sin, cos) values vertically and every position receives a unique fingerprint—without learned parameters, and with some ability to extrapolate beyond training lengths.</figcaption>
</figure>

### Implement it—sinusoidal PE matrix

Build a PE matrix of shape `(seq_len, d_model)`. Even dimensions use sine, and odd dimensions use cosine.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"sinusoidal_pe","packages":["numpy"],"approx":true,"starter":"def sinusoidal_pe(seq_len, d_model):\n    # Return a (seq_len, d_model) PE matrix as a list of lists.\n    # PE[pos, 2i]   = sin(pos / 10000**(2i/d_model))\n    # PE[pos, 2i+1] = cos(pos / 10000**(2i/d_model))\n    # Assume d_model is even.\n    import numpy as np\n    pass","tests":[{"args":[1,2],"expect":[[0.0,1.0]]},{"args":[2,4],"expect":[[0.0,1.0,0.0,1.0],[0.841471,0.540302,0.0099998,0.99995]],"tol":1e-4},{"args":[3,2],"expect":[[0.0,1.0],[0.841471,0.540302],[0.909297,-0.416147]],"tol":1e-4}],"solution":"import numpy as np\n\ndef sinusoidal_pe(seq_len, d_model):\n    pe = np.zeros((seq_len, d_model))\n    pos = np.arange(seq_len)[:, None]\n    idx = np.arange(0, d_model, 2)[None, :]\n    div = np.power(10000.0, idx / d_model)\n    pe[:, 0::2] = np.sin(pos / div)\n    pe[:, 1::2] = np.cos(pos / div)\n    return pe.tolist()"}
</script>
</div>

Notice that position 0 is always `[0,1,0,1,...]` because $sin 0=0$ and $cos 0=1$. **Add** this matrix to the token embeddings to give the model position information.

## §2 · Learned absolute PE (and its limitation)

Early GPT and BERT families used a learned embedding for every position (`nn.Embedding(max_len, d)`). This is simple and works well, but it **cannot extrapolate past the maximum length seen during training**. If training stops at 512, there is no vector for position 513. The long-context era therefore pushed models toward relative approaches.

## §3 · RoPE—encode relative position through rotation

**RoPE (Rotary Position Embedding)** is the default in today's frontier LLMs. Instead of *adding* a position vector, it **rotates query and key vectors by angles proportional to their positions**. It pairs dimensions and rotates each pair in a 2D plane:

$$
\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} =
\begin{bmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}, \qquad m = \text{position}
$$

**The magic appears in the dot product.** The dot product between a query rotated at position $m$ and a key rotated at position $n$ combines the rotations so that it depends **only on the relative displacement $m-n$**. RoPE naturally encodes "how far apart" the tokens are rather than only their absolute positions, making extrapolation to longer lengths easier.

<figure>
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <line x1="120" y1="100" x2="120" y2="20" stroke="#98a3b2" stroke-width="1"/>
  <line x1="40" y1="100" x2="200" y2="100" stroke="#98a3b2" stroke-width="1"/>
  <circle cx="120" cy="100" r="70" fill="none" stroke="#98a3b2" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="120" y1="100" x2="185" y2="100" stroke="#e0533f" stroke-width="3">
    <animateTransform attributeName="transform" type="rotate" from="0 120 100" to="-330 120 100" dur="4s" repeatCount="indefinite"/>
  </line>
  <text x="120" y="185" text-anchor="middle" fill="#e0533f">rotate by angle = position × θ</text>
  <text x="330" y="70" fill="currentColor">Larger positions rotate farther →</text>
  <text x="330" y="95" fill="#6366f1">q(position m)·k(position n)</text>
  <text x="330" y="118" fill="#6366f1">depends only on displacement (m−n)</text>
  <text x="330" y="150" fill="#12a150">→ easier long-context extrapolation</text>
</svg>
<figcaption>RoPE rotates q and k in proportion to position. Because their dot product reflects only their relative offset, position is encoded through rotation rather than added as a separate vector.</figcaption>
</figure>

### Implement it—a 2D RoPE rotation

Implement a function that rotates one even/odd dimension pair $(x_0, x_1)$ by `angle`.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"rope_2d","packages":["numpy"],"approx":true,"starter":"def rope_2d(x0, x1, angle):\n    # 2D rotation: return [x0*cos(angle) - x1*sin(angle), x0*sin(angle) + x1*cos(angle)].\n    import numpy as np\n    pass","tests":[{"args":[1.0,0.0,0.0],"expect":[1.0,0.0]},{"args":[1.0,0.0,1.5707963],"expect":[0.0,1.0],"tol":1e-5},{"args":[1.0,1.0,0.0],"expect":[1.0,1.0]},{"args":[2.0,0.0,3.1415926],"expect":[-2.0,0.0],"tol":1e-5}],"solution":"import numpy as np\n\ndef rope_2d(x0, x1, angle):\n    c, s = np.cos(angle), np.sin(angle)\n    return [float(x0 * c - x1 * s), float(x0 * s + x1 * c)]"}
</script>
</div>

At angle $\pi/2$, $(1,0)$ rotates to $(0,1)$; at angle $\pi$, $(2,0)$ rotates to $(-2,0)$. Full RoPE applies this rotation to every dimension pair with a different $\theta_i$, using the same multi-frequency idea as sinusoidal encoding.

## §4 · ALiBi, and why RoPE became the standard

<dl class="kv">
<dt>ALiBi</dt><dd><b>Adds</b> a distance-proportional penalty $-m|i-j|$ to attention scores. It has no learned position parameters and extrapolates well, but RoPE has generally become more common at the frontier.</dd>
<dt>Why RoPE won</dt><dd>(1) It naturally encodes relative position, (2) it does not add an absolute position vector, so it works cleanly with KV caches, and (3) frequency-scaling methods such as NTK-aware scaling and YaRN make 4K→128K+ extensions practical. See <a href="#/llm/fundamentals">LLM Fundamentals</a> for long-context strategies.</dd>
</dl>

> [!TIP] Interview one-liner
> "Attention is permutation-equivariant, so position must be injected separately. Older models *added* sinusoidal or learned embeddings; modern models rotate q and k with RoPE so their dot product depends only on relative distance, which makes long-context extrapolation easier." Add "frequency scaling such as YaRN extends the context" to show 2026 awareness.

## Q&A

<details class="qa"><summary>Why specifically sin/cos for sinusoidal PE, and why multiple frequencies?</summary>
<div class="qa-body">

**Short:** A combination of (sin, cos) values at multiple frequencies gives every position a unique fingerprint and lets a linear transformation express relative-position relationships.

**Deep:** With only one frequency, periodicity makes distant positions collide—aliasing. Combining geometrically decreasing frequencies distinguishes both short and long distances. The angle-addition identities for $\sin(a+b)$ and $\cos(a+b)$ also let the encoding at $pos{+}k$ be expressed as a **linear transformation** of the encoding at $pos$, making relative position easier to learn.
</div></details>

<details class="qa"><summary>Which vectors receive RoPE—embeddings, q/k, or v as well?</summary>
<div class="qa-body">

**Short:** Apply it only to the attention **queries and keys**, not to the values.

**Deep:** Position is needed in the $q\cdot k$ score that decides who attends to whom. Applying RoPE to q and k makes that dot product reflect relative position. Values carry the content being transported, so they are not rotated. RoPE is therefore not added to the embeddings once; it is applied to q and k **immediately before attention in every layer**.
</div></details>

## Cheat-sheet

| Concept | One line |
| --- | --- |
| Why it is needed | attention is permutation-equivariant → it does not know order |
| Sinusoidal | add a multi-frequency sin/cos "fingerprint" to embeddings; supports extrapolation |
| Learned absolute | learned embedding per position—simple, but cannot extrapolate past max length |
| RoPE | rotate q and k by position-proportional angles → dot product depends on relative distance $m-n$ |
| RoPE targets | queries and keys only—not values or embeddings—at every layer |
| ALiBi | add a distance penalty to scores; generally less common than RoPE |
| Long-context extension | RoPE + frequency scaling (NTK/YaRN) → 128K+ |

**Next:** [Attention (From Scratch)](#/ml-coding/attention) · [Transformer Block (From Scratch)](#/ml-coding/transformer) · [LLM Fundamentals](#/llm/fundamentals)
