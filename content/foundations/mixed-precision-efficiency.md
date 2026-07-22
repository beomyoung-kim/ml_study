# Mixed Precision & Efficiency

<div class="tag-row"><span class="tag">BF16/FP8</span><span class="tag">NVFP4/MXFP4</span><span class="tag">GPTQ/AWQ</span><span class="tag">FlashAttention</span><span class="tag">KV cache</span><span class="tag">speculative decoding</span></div>

> [!NOTE] This chapter is advanced—you may skip it for now
> **One-line intuition:** computers approximate numbers using a fixed number of bits. **Using fewer bits—lower precision, such as moving from 32 bits to 16, 8, or 4—makes training and inference faster and cheaper**, but going too far hurts accuracy. This chapter collects techniques for reducing bit width while preserving accuracy. Remember one sentence: **"Exponent bits buy range; mantissa bits buy precision."** If you are a beginner, you can return to this chapter later.

> [!TIP] Interview one-liner
> Efficiency is where research becomes product. Two clean framings win interviews: (1) *"exponent bits buy range, mantissa bits buy precision"* — that alone explains BF16 vs FP16 vs FP8; (2) *"training and inference have different levers"* — precision/parallelism/activation-memory for training; quantization/kernels/KV-cache/speculation for inference.

## Training vs. inference levers

The pitfall interviewers probe: conflating *training* precision with *deployment* precision. They're different toolboxes with different goals.

```mermaid
flowchart LR
  subgraph train ["Training — throughput/memory"]
    A[BF16 / FP8 / FP4] --> B[FSDP / parallelism]
    B --> C[FlashAttention + activation ckpt]
  end
  subgraph infer ["Inference — latency/cost"]
    D[PTQ / QAT / rotations] --> E[KV cache: Paged / MLA / quant]
    E --> F[speculative decode + kernels]
  end
```

*"I trained in FP8" ≠ "I deploy in INT4."* Distillation appears on both sides but for different ends (capacity transfer during training; smaller serving model at inference).

## Number formats

| Format | Exp / Mant | What it buys |
| --- | --- | --- |
| FP32 | 8 / 23 | reference |
| TF32 | 8 / 10 | Ampere+ tensor-core FP32 input mode (truncated mantissa) |
| **BF16** | 8 / 7 | FP32-like range, low precision → stable, often no loss scaling |
| FP16 | 5 / 10 | precise but narrow range → needs loss scaling |
| FP8 E4M3 | 4 / 3 | forward weights/activations |
| FP8 E5M2 | 5 / 2 | gradients (wider range) |
| FP4 E2M1 | 2 / 1 | 4-bit element in NVFP4/MXFP4 blocks |

**BF16 shares FP32's exponent width**, so it has less overflow risk than FP16 and is a common training precision on modern accelerators with good support. **FP16** has more mantissa but a narrower range, so many training setups use loss scaling. **FP8** and block-scaled **4-bit** formats have also begun to appear in some large-scale training runs when the hardware, kernels, and model recipe support them, but they are not the default for every layer and workload.

### Loss scaling (FP16)

FP16 gradients underflow to zero; scale the loss up before backward, then unscale before the optimizer step. **Dynamic loss scaling**: raise the scale while stable, halve it on overflow. Master weights and optimizer moments stay FP32. BF16's wide range usually makes this unnecessary — a real operational simplification.

> **PyTorch-style pseudocode—ordering is correctness in AMP**

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.float16):
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()                 # first enlarge small FP16 gradients
scaler.unscale_(optimizer)                    # restore scale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)                        # skip the update on overflow
scaler.update()                               # adjust the scale for the next step
```

### FP4: NVFP4 vs MXFP4 <span class="badge badge-2026">2026</span>

Both use **E2M1** 4-bit elements with a shared per-block scale; they differ in block size and scale format:

<dl class="kv">
<dt>NVFP4</dt><dd>Block <b>16</b>, scale in <b>FP8 E4M3</b> → finer-grained, better dynamic range per block.</dd>
<dt>MXFP4</dt><dd>Block <b>32</b>, scale is a power-of-two <b>E8M0</b> → coarser; reportedly needed ~36% more tokens to match NVFP4 loss in one 8B/1T comparison <span class="badge badge-med">secondary</span>.</dd>
</dl>

> [!NOTE] 4-bit *pretraining* is real
> NVIDIA researchers reported an experiment in arXiv:2509.25149 that **pretrained a 12B model at a scale of 10T tokens with NVFP4**, obtaining loss close to an FP8 baseline. The stability recipe combined **random Hadamard transforms** to spread outliers, **2D block quantization**, **stochastic rounding**, and higher precision for selected sensitive layers. This is a result for that model, hardware, and recipe; do not generalize it into a claim that 4-bit training always matches FP8.

<details class="qa"><summary>Why is BF16 usually preferred over FP16 for training?</summary>
<div class="qa-body">

**Short:** BF16 keeps FP32's 8 exponent bits, so it has the same dynamic range and rarely overflows/underflows — you typically drop loss scaling entirely. FP16 has more mantissa but a narrow range, so it needs dynamic loss scaling and is more fragile.

**Deep:** Training gradients span many orders of magnitude. FP16's maximum value is 65504, and small values can underflow, so loss scaling and overflow-skip logic are commonly used. BF16 trades mantissa precision for dynamic range, and many GEMMs accumulate in higher precision, often simplifying training operations. The final choice depends on the accelerator's native throughput, kernel support, and convergence validation. **Follow-up:** *What remains at high precision?* Optimizer state, reductions and accumulation, and sensitive operations such as softmax and normalization use FP32 or BF16 depending on the implementation and recipe.
</div></details>

<details class="qa"><summary>What's the standard FP8 training recipe, and what breaks it?</summary>
<div class="qa-body">

**Short:** A common FP8 recipe considers E4M3 for forward weights and activations and the wider-range E5M2 for gradients, while keeping optimizer state and sensitive operations at higher precision. Modern delayed scaling, current scaling, and block-scaling recipes can assign dtypes differently by implementation. Per-tensor or per-block scales manage dynamic range.

**Deep:** FP8's narrow mantissa and range make it sensitive to activation outliers, so scaling granularity and high-precision exceptions matter. **Follow-up:** *FP8 inference versus training?* Inference does not need an E5M2 gradient path, but deployment is not limited to weight-only quantization. FP8 W8A8 is also used, while weight-only PTQ is especially common for INT4. Check which dtype combinations have fast kernels on the actual accelerator.
</div></details>

## Quantization for inference

Uniform affine quantization: $x_q=\mathrm{clip}(\mathrm{round}(x/s)+z)$, with scale $s$ and zero-point $z$.

<dl class="kv">
<dt>PTQ</dt><dd>Post-training; calibrate scales on a small set. Fast, no retraining, some accuracy risk.</dd>
<dt>QAT</dt><dd>Fake-quant in the training loop (straight-through estimator for round); recovers accuracy at higher cost.</dd>
<dt>GPTQ</dt><dd>Second-order (Hessian-aware) weight-only PTQ; strong 4-bit weights.</dd>
<dt>AWQ</dt><dd>Activation-aware — protect the salient weight channels; the practical weight-only 4-bit default.</dd>
</dl>

**Rotation-based PTQ** is the 2025–2026 advance for pushing to 4-bit *including activations*: **QuaRot** applies random Hadamard rotations to spread outliers before quantizing; **SpinQuant** *learns* the rotations. Both attack the outlier problem that wrecks naive activation quantization.

<details class="qa"><summary>Why do rotations (QuaRot/SpinQuant) help low-bit quantization?</summary>
<div class="qa-body">

**Short:** activation outliers span a huge range and blow up the quantization scale, wasting bits on the whole tensor. An orthogonal rotation (e.g., Hadamard) redistributes that energy across channels so the distribution is more uniform and quantizes cleanly — and being orthogonal, it's mathematically invertible so the model output is unchanged.

**Deep:** A few high-magnitude channels can force a large $s$ and coarsen every other value. If an orthogonal rotation is folded exactly into adjacent linear maps, the function is preserved **before quantization** while outliers spread across channels. After quantization the output is approximate, so calibration must measure the error. A W4A4 activation element is theoretically four times smaller than FP16, but scale metadata, packing, and kernels determine the actual bandwidth and latency gain.
</div></details>

## FlashAttention: IO-aware attention

A naive attention implementation materializes the $n\times n$ score matrix in HBM. FlashAttention **tiles** Q/K/V and computes online softmax in on-chip memory, producing the same attention result within numerical error without storing the full score matrix in HBM. This greatly reduces memory traffic and peak memory, although whether the operation actually shifts from memory-bound to compute-bound depends on sequence length, head dimension, dtype, GPU, and kernel.

<dl class="kv">
<dt>FA-2</dt><dd>Better work partitioning across warps/threadblocks.</dd>
<dt>FA-3</dt><dd>Hopper/H100: async copies (TMA) + warp specialization + FP8.</dd>
<dt>FA-4</dt><dd>Blackwell (B200/GB200); rewritten in CuTe-DSL. Exists because of <b>asymmetric hardware scaling</b>.</dd>
</dl>

> [!IMPORTANT] "Asymmetric hardware scaling"—a 2026 talking point
> On the Blackwell generation, tensor-core throughput and the softmax and memory paths did not scale at the same rate, so merely retuning an older kernel is insufficient to exploit all the new arithmetic capacity. FlashAttention-4 redesigns scheduling and data movement around this asymmetry. The key lesson is not the version number itself, but that **bottlenecks vary by hardware generation and tensor shape, so end-to-end kernel benchmarks are necessary**.

## KV cache & serving

Autoregressive decoding caches past K/V; the cache grows with context and dominates long-context memory.

<dl class="kv">
<dt>PagedAttention (vLLM)</dt><dd>Virtual-memory-style paging of the KV cache → near-zero fragmentation, high batch throughput.</dd>
<dt>GQA / MQA</dt><dd>Share K/V heads across query heads to shrink the cache (architecture-level).</dd>
<dt>MLA</dt><dd>Multi-head Latent Attention (DeepSeek): compress K/V into a low-rank latent; reported ~2.7–4.7× KV reduction vs GQA <span class="badge badge-med">secondary</span>.</dd>
<dt>Quantized KV</dt><dd>INT8 ≈ 2×, FP4 ≈ 4× cache reduction.</dd>
<dt>Prefix / prompt caching</dt><dd>Reuse the KV of a shared prefix (system prompt, RAG context, few-shot) across requests → skip re-prefilling it. Big win for agents and long fixed contexts.</dd>
<dt>Chunked prefill</dt><dd>Split a long prompt's prefill into chunks and interleave them with ongoing decode steps in the same batch → smoother latency, no head-of-line blocking from one huge prompt.</dd>
<dt>Disaggregated serving</dt><dd>Run <b>prefill</b> and <b>decode</b> on <i>separate</i> worker pools (their compute profiles differ), streaming the KV between them → each phase scales independently. A 2025–2026 production pattern.</dd>
</dl>

> [!NOTE] Where the serving story continues
> These are the phase-level and cluster-level serving levers. The end-to-end design — routing, autoscaling, TTFT/TPOT SLOs, cost-per-token, and where each of these fits — is in [Designing LLM/Agent Systems](#/system-design/llm-systems). The prefill-vs-decode split that motivates all of it is in [LLM Fundamentals §6](#/llm/fundamentals).

## Speculative decoding

A cheap **drafter** proposes several tokens and the target model verifies them in parallel. Under greedy decoding, appending only tokens that match the target preserves the result. Under sampling, preserving the target distribution requires the full **exact speculative-sampling algorithm**, including rejection and residual sampling. Merely checking draft tokens and keeping a prefix does not automatically preserve the same distribution.

<dl class="kv">
<dt>Medusa</dt><dd>Extra prediction heads on the target model draft multiple future tokens.</dd>
<dt>EAGLE-1/2/3</dt><dd>A small drafter over the target's hidden states + a draft <i>tree</i>; EAGLE-3 fuses multi-layer features, reported acceptance &gt;75% <span class="badge badge-med">vendor</span>.</dd>
<dt>MTP</dt><dd>Multi-token prediction trained into the model (DeepSeek-style) → self-speculation.</dd>
</dl>

Major serving runtimes support it as an option, but it is not the default for every workload. When acceptance is low or batching is already compute-bound, overhead can erase the gain, so measure it on the real request distribution.

<details class="qa"><summary>When does speculative decoding help, and when can it hurt?</summary>
<div class="qa-body">

**Short:** it helps at low batch size / memory-bound decoding where the target GPU is underutilized and drafts are accepted often; it hurts when the batch is already large (compute-bound) or acceptance is low, since rejected drafts waste the verification compute.

**Deep:** Small-batch decoding is often memory-bandwidth-bound, so verifying several tokens at once can amortize weight reads. But speedup depends jointly on draft length, acceptance, target batch size, and drafter cost; a simple product does not predict it reliably. **Follow-up:** *Does it change outputs?* Greedy decoding preserves output through exact-token verification, while sampling preserves the target distribution only when the formal rejection and residual-sampling procedure is implemented.
</div></details>

## Pruning & distillation

**Pruning** — *unstructured* (zero individual weights: high sparsity, but needs sparse kernels or 2:4 N:M sparsity for real speedup) vs *structured* (drop channels/heads/blocks: hardware-friendly). **Distillation** — a student mimics a teacher's soft distribution (and/or features):

$$
L=\alpha T^2\,\mathrm{KL}(p_T\Vert p_S)+(1-\alpha)\,\mathrm{CE}(y,p_S)
$$

A common 2026 compression pipeline is **prune → quantize (QAT/PTQ) → distill** to recover accuracy. On-device recipe: train at 16-bit, deploy at 4-bit (AWQ/GPTQ). Interview nuance: **FLOPs are a proxy** — decide with measured latency/memory/energy on the *real* device, since low-FLOP ops (depthwise) can be bandwidth-bound and slow.

<details class="qa"><summary>Product asks for 2× lower latency, ≤1% accuracy drop. Your plan?</summary>
<div class="qa-body">

**Short:** search the Pareto front in cheap-first order — input resolution / architecture width → distillation → structured pruning → INT8 (or 4-bit) QAT → runtime fusion — and *measure real device latency at every step*, not FLOPs.

**Deep:** (1) sensitivity analysis: which classes/regions break first sets the accuracy budget; (2) cut input resolution / simplify the decoder — often the biggest, cheapest win; (3) distill into a smaller student to preserve accuracy at lower capacity; (4) structured prune (channels/heads) so kernels actually speed up; (5) INT8 QAT for most of the latency, per-channel scales; (6) operator fusion (Conv-BN-ReLU), thread pinning, memory reuse; (7) lock a regression test on a fixed eval subset. The discipline: one change at a time, p50/p95 latency on the target hardware, and let the product metric define "1%." **Follow-up:** *Deploy without A/B?* — canary rollout with latency/error guardrails.
</div></details>

## Cheat-sheet

| Ask | One-liner |
| --- | --- |
| BF16 vs FP16 | Exponent bits = range, mantissa = precision; BF16 = FP32 range, no loss scaling. |
| FP8 recipe | E4M3 forward, E5M2 grads, FP32 master/moments; exclude norm/softmax/LM head. |
| NVFP4 vs MXFP4 | E2M1 elements; NVFP4 block 16 + FP8 scale (finer), MXFP4 block 32 + E8M0 (coarser). |
| PTQ vs QAT | PTQ = calibrate, fast, risky; QAT = fake-quant training, recovers accuracy. |
| GPTQ / AWQ | Hessian-aware vs activation-aware weight-only 4-bit; AWQ is the practical default. |
| Rotation PTQ | QuaRot/SpinQuant spread outliers via orthogonal rotations → enables W4A4. |
| FlashAttention | IO-aware tiled softmax; same math, no $n^2$ matrix in HBM; FA-4 for Blackwell asymmetry. |
| KV cache | PagedAttention + GQA/MLA + quantized KV shrink the long-context bottleneck. |
| Speculative decode | Draft + verify; helps at low batch / memory-bound; preserves output distribution. |
| Compression pipeline | Prune → quantize → distill; decide on measured latency, not FLOPs. |

**Related:** [Distributed Training](#/foundations/distributed-training) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [Normalization & Stability](#/foundations/normalization-stability) · [LLM Fundamentals](#/llm/fundamentals) · [Optimization](#/foundations/optimization)
