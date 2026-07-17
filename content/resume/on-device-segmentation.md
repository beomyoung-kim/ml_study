# Deep-Dive: On-Device Human Segmentation (~10ms, Mobile CPU)

<div class="tag-row"><span class="tag">on-device</span><span class="tag">mobile CPU</span><span class="tag">~10ms</span><span class="tag">ONNX serving</span><span class="tag">distillation · quantization</span><span class="tag">solo project</span></div>

> [!TIP] 30-second pitch
> I independently built a **human/portrait segmentation** model that runs on **mobile CPU** at **~10ms**, deployed through an in-house **ONNX** serving stack. The 10ms wasn't a vanity number — it's the frame-budget threshold that keeps a real-time (~30 fps) experience smooth once you subtract camera I/O, pre/post-processing, and other on-device models. The project is the counterpoint to ZIM: **cloud foundation model for quality, tiny on-device specialist for latency and privacy.**

> [!NOTE] Confidential internals
> Exact architecture, FLOPs, dataset scale, and A/B numbers are **confidential**. Everything below is engineering method + general efficiency knowledge, not internal figures. Backing chapters: [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency), [Segmentation](#/cv/segmentation).

## Deployment path

```mermaid
flowchart LR
  TRAIN["PyTorch: train + distill<br/>(server teacher → tiny student)"] --> EXP["ONNX export"]
  EXP --> OPT["Simplify · op-fix · quantize (PTQ/QAT)"]
  OPT --> PAR["Parity test<br/>(ORT vs PyTorch)"]
  PAR --> SERVE["In-house ONNX runtime"]
  SERVE --> APP["Mobile / edge feature"]
  APP -.->|failure monitoring| TRAIN
  style TRAIN fill:#0ea5e9,color:#fff
  style SERVE fill:#12a150,color:#fff
```

## The 10ms is frame-budget engineering, not a magic number

<figure>
<svg viewBox="0 0 640 120" role="img" aria-label="33ms frame budget breakdown at 30fps" style="max-width:100%;height:auto;font-family:inherit">
  <text x="0" y="16" font-size="12" fill="currentColor" opacity="0.8">One 30 fps frame = 33 ms</text>
  <rect x="0" y="30" width="90"  height="34" fill="#0ea5e9" opacity="0.85"/><text x="45"  y="52" font-size="11" fill="#fff" text-anchor="middle">preproc 3–5</text>
  <rect x="92" y="30" width="150" height="34" fill="#e0533f"/><text x="167" y="52" font-size="11" fill="#fff" text-anchor="middle" font-weight="700">seg ~10ms</text>
  <rect x="244" y="30" width="70"  height="34" fill="#12a150" opacity="0.85"/><text x="279" y="52" font-size="11" fill="#fff" text-anchor="middle">post 2–3</text>
  <rect x="316" y="30" width="324" height="34" fill="currentColor" opacity="0.15"/><text x="478" y="52" font-size="11" fill="currentColor" text-anchor="middle">other models · UI · headroom</text>
  <line x1="0" y1="72" x2="640" y2="72" stroke="currentColor" opacity="0.3"/>
</svg>
<figcaption>Illustrative budget only (real numbers device/resolution-dependent). At 15–20ms the segmenter alone starves the rest of the frame and you drop frames.</figcaption>
</figure>

If you target 60 fps the budget halves, so the discipline matters more than the specific figure. The reusable statement: *"I sized the model to a frame budget, not to a leaderboard."*

## Compression levers (in the order I'd pull them)

| Lever | What it buys | Watch out for |
| --- | --- | --- |
| **Input resolution** | Biggest single win | Boundary softens first |
| **Width / channel prune** | Linear-ish speedup | Capacity floor on hard poses |
| **Decoder simplification** | Cheap upsampling, fewer skips | Fine detail (hair/fingers) |
| **Depthwise-separable conv** | MobileNet-style FLOP cut | Op support on target runtime |
| **Knowledge distillation** | Recover quality lost to shrinking | Needs a strong soft teacher |
| **Quantization (PTQ → QAT)** | INT8 latency/memory | Boundary collapse if done first |
| **Operator fusion** | Conv-BN-ReLU merges | Runtime-specific |

Rule of thumb I'd say out loud: **resolution → width → decoder → distill → *quantize last*.** Quantizing before you've distilled tends to kill boundaries first.

## Predicted deep-dive Q&A

<details class="qa"><summary>Why mobile CPU and not GPU/NPU?</summary>
<div class="qa-body">

**Short:** CPU is the worst-case common denominator — maximum device reach, no op-support fragmentation.

**Deep:** NPUs are fast but fragmented (op coverage, quantization constraints vary by chip); GPUs aren't always available or free to use. Targeting CPU means the feature ships everywhere. It also forces genuinely efficient design rather than leaning on an accelerator. The résumé says mobile CPU deliberately.
</div></details>

<details class="qa"><summary>How did you use distillation?</summary>
<div class="qa-body">

**Short:** A server-grade human-seg/matte **teacher** supervises a tiny **student** with soft masks, boundary-weighted and feature-level losses.

**Deep:** A *matting* teacher is especially useful because soft labels carry boundary information a hard mask throws away — the student learns crisper hair/edges than it could from binary GT alone. This is organizationally adjacent to the ZIM/FG-API quality work, but I won't claim it's the same weights — it's a separate, distilled model.
</div></details>

<details class="qa"><summary>Quantization pitfalls for a small boundary-sensitive model?</summary>
<div class="qa-body">

**Short:** PTQ is fast but can collapse tiny models at the boundary; QAT is costlier but stable; calibration set must represent the product domain.

**Deep:** Watch activation-distribution outliers, skip connections, and op compatibility (sigmoid/Hardswish under ONNX). If the calibration set isn't product-representative, INT8 error concentrates exactly on hard hair/edge pixels — the thing users notice. So QAT + a domain-matched calibration set, and quantize after distillation.
</div></details>

<details class="qa"><summary>What ONNX export issues bit you? (general)</summary>
<div class="qa-body">

| Problem | Fix |
| --- | --- |
| Unsupported op | Rewrite, raise opset, or custom plugin |
| Dynamic shape | Fix input resolution or make it explicit |
| Numeric mismatch | ORT-vs-PyTorch **parity test** on masks |
| Perf regression | Graph optimize, IO binding, thread tuning |
| Preprocess drift | Share mean/std normalization with runtime |

The one that most often bites near launch is an unsupported/edge-behaving op forcing a last-minute graph rewrite.
</div></details>

<details class="qa"><summary>Why not run SAM/ZIM on-device?</summary>
<div class="qa-body">

A ViT-B foundation model is orders of magnitude off a 10ms mobile-CPU budget (ZIM is ~180ms-class on a V100). Foundations belong on the **server / in tooling**; on-device wants a **specialized tiny closed-set** model. It's a deliberate role split, not a limitation of either.
</div></details>

### Hard follow-ups

<details class="qa"><summary>Cut latency in half without losing quality. Concretely, how?</summary>
<div class="qa-body">

In order: drop input resolution and re-distill so the teacher's soft boundaries survive; trim width and simplify the decoder; *then* QAT to INT8. Re-ablate quality on a **hard set** (crossed limbs, sheer clothing, extreme lighting, multi-person) after each step, and parity-test the ONNX graph before measuring on device. The failure mode to avoid is quantizing first — boundaries die before you've protected them with distillation.
</div></details>

<details class="qa"><summary>"Robust under tight budget" — what does robust actually mean here?</summary>
<div class="qa-body">

Compressing naively makes the model good only on easy samples. I hold a **quality floor** with hard-example mining, distillation, product-domain data, and failure monitoring — so FLOPs↓ is not treated as synonymous with quality↓. Robustness = the p95 hard case stays acceptable, not just the mean.
</div></details>

<details class="qa"><summary>Mobile CPU vs "ONNX serving" — which is it?</summary>
<div class="qa-body">

Both, cleanly separated: the **model** is designed and measured against a mobile-CPU budget; the **serving infrastructure** is an in-house ONNX runtime. I'd measure latency with warmup, fixed-resolution input, p50/p95, and (for a device) pinned CPU affinity — never confuse training-GPU ms with deployment latency. Sustained (thermal-throttled) latency, not a cold single-shot, is the honest number.
</div></details>

## Honest limitations

- **Closed-set (human/portrait):** open-vocabulary would blow the budget; product KPI justifies the narrowing.
- **Single-pass fixed resolution:** multi-scale/refine would help hard boundaries but breaks 10ms.
- **Heavy post-processing (CRF) is too expensive on mobile;** only light morphology / guided-filter fits.
- **Temporal smoothing (video)** adds cost and isn't free.

## Which JD this connects to

| Company | Connection |
| --- | --- |
| Apple | On-device intelligence, distillation, privacy |
| Meta | Efficient multimodal deployment |
| NVIDIA | Efficient inference (TensorRT/ONNX) mindset |
| Adobe | Mobile creative tools |

## Cheat-sheet

| Item | Value |
| --- | --- |
| Task | On-device human/portrait segmentation, mobile CPU |
| Latency | **~10ms** (frame-budget threshold for ~30 fps) |
| Stack | PyTorch → **ONNX** → in-house runtime |
| Lever order | resolution → width → decoder → **distill** → quantize (last) |
| Measure | warmup, fixed res, p50/p95, sustained (thermal) latency |
| Narrative | cloud foundation (quality) + on-device specialist (latency/privacy) |
| Confidential | architecture, FLOPs, data scale, A/B numbers |

## Cross-links
- Topical: [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) · [Segmentation](#/cv/segmentation) · [Image Matting](#/cv/matting)
- Deep-dives: [ZIM](#/resume/zim) · [FaceSign](#/resume/facesign) · back to the [CV → Interview Map](#/resume/overview)
