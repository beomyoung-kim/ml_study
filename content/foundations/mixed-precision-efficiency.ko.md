# Mixed Precision & Efficiency

<div class="tag-row"><span class="tag">BF16/FP8</span><span class="tag">NVFP4/MXFP4</span><span class="tag">GPTQ/AWQ</span><span class="tag">FlashAttention</span><span class="tag">KV cache</span><span class="tag">speculative decoding</span></div>

> [!TIP] 이것부터 말하세요
> Efficiency는 연구가 제품이 되는 지점입니다. 면접을 이기는 두 개의 깔끔한 프레이밍: (1) *"exponent bit는 range를 사고, mantissa bit는 precision을 산다"* — 이것만으로 BF16 vs FP16 vs FP8이 설명됩니다. (2) *"training과 inference는 서로 다른 레버를 갖는다"* — training엔 precision/parallelism/activation-memory, inference엔 quantization/kernel/KV-cache/speculation.

## Training vs. inference 레버

면접관이 파고드는 함정: *training* precision과 *deployment* precision을 혼동하는 것. 둘은 목표가 다른 별개의 도구상자입니다.

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

*"FP8로 학습했다" ≠ "INT4로 배포한다".* Distillation은 양쪽에 다 나오지만 목적이 다릅니다(training 중엔 capacity transfer, inference 시엔 더 작은 serving 모델).

## Number formats

| Format | Exp / Mant | 무엇을 사는가 |
| --- | --- | --- |
| FP32 | 8 / 23 | reference |
| TF32 | 8 / 10 | Ampere+ tensor-core FP32 input 모드 (truncated mantissa) |
| **BF16** | 8 / 7 | FP32급 range, 낮은 precision → 안정적, 종종 loss scaling 불필요 |
| FP16 | 5 / 10 | 정밀하지만 좁은 range → loss scaling 필요 |
| FP8 E4M3 | 4 / 3 | forward weight/activation |
| FP8 E5M2 | 5 / 2 | gradient (더 넓은 range) |
| FP4 E2M1 | 2 / 1 | NVFP4/MXFP4 block의 4-bit element |

**BF16은 FP32의 exponent를 공유**하므로 overflow가 드뭅니다 — 현대 accelerator의 기본 training precision입니다. **FP16**은 mantissa가 더 많지만 range가 좁아 loss scaling이 필요합니다. **FP8**(Hopper+, Transformer Engine 경유)은 프론티어에서 일상적이고, microscaling **4-bit** format이 2026년 프론티어입니다 — *pretraining*에서도.

### Loss scaling (FP16)

FP16 gradient는 0으로 underflow합니다. backward 전에 loss를 키우고 optimizer step 전에 unscale하세요. **Dynamic loss scaling**: 안정적이면 scale을 올리고, overflow 시 반으로. Master weight와 optimizer moment는 FP32로 유지합니다. BF16의 넓은 range는 보통 이를 불필요하게 만듭니다 — 실질적인 운영 단순화입니다.

### FP4: NVFP4 vs MXFP4 <span class="badge badge-2026">2026</span>

둘 다 shared per-block scale을 가진 **E2M1** 4-bit element를 쓰며, block size와 scale format에서 다릅니다:

<dl class="kv">
<dt>NVFP4</dt><dd>Block <b>16</b>, scale은 <b>FP8 E4M3</b> → 더 세밀하고 block당 dynamic range가 좋음.</dd>
<dt>MXFP4</dt><dd>Block <b>32</b>, scale은 power-of-two <b>E8M0</b> → 더 거침. 한 8B/1T 비교에서 NVFP4 loss에 맞추려면 ~36% 더 많은 token이 필요했다고 보고됨 <span class="badge badge-med">secondary</span>.</dd>
</dl>

> [!NOTE] 4-bit *pretraining*은 실재한다
> NVIDIA는 **12B 모델을 10T token에서 NVFP4로 pretrain**했고(문서화된 가장 긴 4-bit 학습), FP8 loss에 맞췄습니다(arXiv:2509.25149). 안정성은 **random Hadamard transform**(outlier 분산), **2D block quantization**, **stochastic rounding**, 그리고 민감한 몇몇 layer를 높은 precision으로 유지한 데서 왔습니다. *(verifiable. 정확한 수치는 hedge할 것.)*

<details class="qa"><summary>왜 training에 보통 FP16보다 BF16이 선호되나요?</summary>
<div class="qa-body">

**짧게:** BF16은 FP32의 8 exponent bit를 유지하므로 dynamic range가 같고 overflow/underflow가 드뭅니다 — 보통 loss scaling을 완전히 뺍니다. FP16은 mantissa가 더 많지만 range가 좁아 dynamic loss scaling이 필요하고 더 취약합니다.

**깊게:** training gradient는 여러 자릿수에 걸칩니다. FP16의 최대 ~65504와 작은 subnormal floor는 overflow/underflow를 흔하게 만들고, loss scaling이 이를 이동하는 hyperparameter와 overflow-retry 로직의 비용으로 땜질합니다. BF16은 mantissa(precision)를 range와 맞바꾸고, GEMM이 FP32로 accumulate하므로 낮아진 precision은 견딜 만합니다. 결론: BF16은 더 단순하고 robust하며, FP16은 BF16 지원이 나쁜 하드웨어에서만 이깁니다. **후속 질문:** *무엇이 FP32로 남나?* — master weight, optimizer moment, softmax/cross-entropy, normalization 통계.
</div></details>

<details class="qa"><summary>표준 FP8 training 레시피는 무엇이고, 무엇이 그것을 깨나요?</summary>
<div class="qa-body">

**짧게:** forward weight/activation은 E4M3, gradient는 E5M2(더 넓은 range), master weight와 optimizer moment는 FP32/BF16으로 유지하고, 수치적으로 민감한 layer — normalization, softmax, LM head — 는 FP8에서 **제외**합니다. Per-tensor 또는 per-block scaling(amax tracking, Transformer Engine 경유)이 dynamic range를 관리합니다.

**깊게:** FP8의 ~3 mantissa bit는 FP32로 accumulate하는 GEMM에는 충분하지만, activation **outlier**가 per-tensor scale을 부풀려 작은 값을 잃게 합니다. Delayed scaling(amax history)은 급격한 shift에 뒤처져 spike를 유발할 수 있어, 랩들은 학습 초반에 current/just-in-time scaling으로 바꾸기도 합니다. Blackwell의 **MXFP8**은 per-tensor FP8 대비 더 세밀한 block-wise scale로 outlier를 길들입니다. 대규모 FP8 pretraining은 입증됐지만(DeepSeek-V3), 재현을 가능케 하는 것은 레시피 — 어느 layer가 high-precision으로 남는지, scaling policy, warmup — 입니다. **후속 질문:** *FP8 inference vs training?* — inference는 종종 weight만 quantize(또는 W4A4용 rotation과 함께)하며, E5M2 gradient path가 필요 없습니다.
</div></details>

## Quantization for inference

Uniform affine quantization: $x_q=\mathrm{clip}(\mathrm{round}(x/s)+z)$, scale $s$와 zero-point $z$.

<dl class="kv">
<dt>PTQ</dt><dd>Post-training. 작은 세트로 scale calibrate. 빠르고 재학습 없음, 일부 accuracy 위험.</dd>
<dt>QAT</dt><dd>Training loop에 fake-quant(round는 straight-through estimator). 더 높은 비용으로 accuracy 회복.</dd>
<dt>GPTQ</dt><dd>Second-order(Hessian-aware) weight-only PTQ. 강한 4-bit weight.</dd>
<dt>AWQ</dt><dd>Activation-aware — 두드러진 weight channel 보호. 실용적인 weight-only 4-bit 기본값.</dd>
</dl>

**Rotation-based PTQ**는 *activation을 포함해* 4-bit로 밀어붙이는 2025–2026년의 진전입니다: **QuaRot**은 quantize 전에 random Hadamard rotation으로 outlier를 분산시키고, **SpinQuant**는 rotation을 *학습*합니다. 둘 다 순진한 activation quantization을 망치는 outlier 문제를 공략합니다.

<details class="qa"><summary>왜 rotation(QuaRot/SpinQuant)이 low-bit quantization에 도움이 되나요?</summary>
<div class="qa-body">

**짧게:** activation outlier는 거대한 range에 걸쳐 quantization scale을 부풀려 tensor 전체의 bit를 낭비합니다. Orthogonal rotation(예: Hadamard)이 그 에너지를 channel 전반에 재분배해 분포가 더 uniform해지고 깔끔하게 quantize됩니다 — 그리고 orthogonal이므로 수학적으로 역변환이 가능해 모델 출력이 바뀌지 않습니다.

**깊게:** 큰 크기의 몇몇 channel이 큰 $s$를 강요해 다른 모든 값을 거칠게 만듭니다. Orthogonal matrix $R$로 회전하고($R^{-1}=R^\top$를 인접 weight에 folding) linear map을 보존하면서 outlier를 거의 Gaussian 분포로 퍼뜨립니다 — 각 block이 커버해야 할 dynamic range를 줄입니다. QuaRot은 고정된 random Hadamard rotation을 쓰고, SpinQuant는 약간의 accuracy를 더 얻으려 이를 학습합니다. 이것이 plain AWQ/GPTQ(weight-only)로는 갈 수 없는 W4A4를 가능하게 합니다. **후속 질문:** *Weight-only vs weight+activation?* — weight-only(AWQ)는 더 쉽고 흔함. W4A4는 rotation + 세심한 calibration이 필요하지만 activation-bandwidth 절약을 두 배로.
</div></details>

## FlashAttention: IO-aware attention

표준 attention은 $n\times n$ score를 HBM에 materialize합니다. FlashAttention은 Q/K/V를 **tile**하고 SRAM에서 softmax를 online으로 계산해 full matrix를 절대 저장하지 않습니다 — 같은 수학, 훨씬 적은 메모리 트래픽, attention을 memory-bound에서 compute-bound로 전환합니다.

<dl class="kv">
<dt>FA-2</dt><dd>Warp/threadblock 간 더 나은 work partitioning.</dd>
<dt>FA-3</dt><dd>Hopper/H100: async copy(TMA) + warp specialization + FP8.</dd>
<dt>FA-4</dt><dd>Blackwell(B200/GB200). CuTe-DSL로 재작성. <b>asymmetric hardware scaling</b> 때문에 존재.</dd>
</dl>

> [!IMPORTANT] "Asymmetric hardware scaling" — 2026년 화두
> FlashAttention이 (retuned v3가 아니라) v4를 필요로 한 이유는, Blackwell에서 **tensor-core throughput은 ~2× 성장한 반면 shared-memory bandwidth와 exp/softmax 유닛은 훨씬 덜 scaling**됐기 때문입니다. 이제 상대적으로 희소해진 softmax/memory path가 풍부한 matmul throughput의 병목이 되지 않도록 kernel을 재설계해야 합니다. Kernel은 점점 *하드웨어 세대별로 특화*됩니다. *(방향은 verifiable. 정확한 TFLOPs 수치는 hedge할 것.)*

## KV cache & serving

Autoregressive decoding은 과거 K/V를 cache합니다. cache는 context에 따라 커지고 long-context 메모리를 지배합니다.

<dl class="kv">
<dt>PagedAttention (vLLM)</dt><dd>KV cache의 virtual-memory 스타일 paging → 거의 zero fragmentation, 높은 batch throughput.</dd>
<dt>GQA / MQA</dt><dd>Query head 간 K/V head를 공유해 cache 축소(아키텍처 수준).</dd>
<dt>MLA</dt><dd>Multi-head Latent Attention(DeepSeek): K/V를 low-rank latent로 압축. GQA 대비 ~2.7–4.7× KV 감소 보고 <span class="badge badge-med">secondary</span>.</dd>
<dt>Quantized KV</dt><dd>INT8 ≈ 2×, FP4 ≈ 4× cache 감소.</dd>
<dt>Prefix / prompt caching</dt><dd>공유 prefix(system prompt, RAG context, few-shot)의 KV를 요청 간 재사용 → 재-prefill 생략. agent와 긴 고정 context에 큰 이득.</dd>
<dt>Chunked prefill</dt><dd>긴 prompt의 prefill을 chunk로 나눠 같은 batch의 진행 중 decode step과 interleave → 더 부드러운 latency, 하나의 거대 prompt로 인한 head-of-line blocking 없음.</dd>
<dt>Disaggregated serving</dt><dd><b>prefill</b>과 <b>decode</b>를 <i>별개</i> worker pool에서 실행(연산 프로파일이 다름)하고 그 사이 KV를 스트리밍 → 각 phase가 독립적으로 scaling. 2025–2026년 production 패턴.</dd>
</dl>

> [!NOTE] Serving 이야기가 이어지는 곳
> 이들은 phase 수준·cluster 수준의 serving 레버입니다. End-to-end 설계 — routing, autoscaling, TTFT/TPOT SLO, cost-per-token, 그리고 이들 각각이 어디에 맞는지 — 는 [Designing LLM/Agent Systems](#/system-design/llm-systems)에 있습니다. 이 모든 것의 동기가 되는 prefill-vs-decode 분리는 [LLM Fundamentals §6](#/llm/fundamentals)에 있습니다.

## Speculative decoding

싼 **drafter**가 여러 token을 제안하고, target 모델이 하나의 병렬 forward로 이를 **verify**해 가장 긴 올바른 prefix를 accept합니다 — 같은 출력 분포, 더 적은 순차 target step.

<dl class="kv">
<dt>Medusa</dt><dd>Target 모델의 추가 prediction head가 여러 미래 token을 draft.</dd>
<dt>EAGLE-1/2/3</dt><dd>Target의 hidden state 위의 작은 drafter + draft <i>tree</i>. EAGLE-3는 multi-layer feature를 융합, acceptance &gt;75% 보고 <span class="badge badge-med">vendor</span>.</dd>
<dt>MTP</dt><dd>모델에 학습된 Multi-token prediction(DeepSeek 스타일) → self-speculation.</dd>
</dl>

이제는 보너스 최적화가 아니라 **기본 serving layer**(vLLM, TensorRT-LLM, SGLang)입니다.

<details class="qa"><summary>Speculative decoding은 언제 도움이 되고 언제 해가 되나요?</summary>
<div class="qa-body">

**짧게:** target GPU가 미활용되고 draft가 자주 accept되는 low batch / memory-bound decoding에서 도움이 됩니다. batch가 이미 큰(compute-bound) 경우나 acceptance가 낮은 경우엔 해가 됩니다. rejected draft가 verification compute를 낭비하기 때문입니다.

**깊게:** decoding은 작은 batch에서 보통 memory-bandwidth-bound입니다 — target은 token 하나 분량의 matmul을 하지만 full weight read를 지불합니다. Speculation은 그 read를 여러 accepted token에 걸쳐 amortize하므로 speedup ≈ 기대 accepted 길이 × acceptance rate − drafter overhead입니다. 높은 batch에서는 이미 compute를 saturate하므로 추가 verification 작업과 rejected token이 throughput을 *줄일* 수 있습니다. Acceptance는 drafter–target 일치도(domain shift, temperature)에 달렸습니다. **후속 질문:** *출력을 바꾸나?* — 아니오. verification이 target의 분포를 정확히 보존합니다(그것이 correctness 보장).
</div></details>

## Pruning & distillation

**Pruning** — *unstructured*(개별 weight를 0으로: 높은 sparsity지만 실제 speedup엔 sparse kernel이나 2:4 N:M sparsity 필요) vs *structured*(channel/head/block drop: 하드웨어 친화적). **Distillation** — student가 teacher의 soft distribution(그리고/또는 feature)을 모방:

$$
L=\alpha T^2\,\mathrm{KL}(p_T\Vert p_S)+(1-\alpha)\,\mathrm{CE}(y,p_S)
$$

흔한 2026년 compression 파이프라인은 accuracy 회복을 위한 **prune → quantize (QAT/PTQ) → distill**입니다. On-device 레시피: 16-bit로 학습, 4-bit(AWQ/GPTQ)로 배포. 면접 뉘앙스: **FLOPs는 proxy**입니다 — *실제* 디바이스에서 측정한 latency/memory/energy로 결정하세요. low-FLOP 연산(depthwise)이 bandwidth-bound라 느릴 수 있기 때문입니다.

<details class="qa"><summary>제품이 2× 낮은 latency, ≤1% accuracy 하락을 요구합니다. 계획은?</summary>
<div class="qa-body">

**짧게:** 싼 것 먼저 순서로 Pareto front를 탐색 — input resolution / architecture width → distillation → structured pruning → INT8(또는 4-bit) QAT → runtime fusion — 그리고 FLOPs가 아니라 *매 단계 실제 디바이스 latency를 측정*합니다.

**깊게:** (1) sensitivity analysis: 어느 class/region이 먼저 깨지는지가 accuracy 예산을 정함. (2) input resolution 축소 / decoder 단순화 — 종종 가장 크고 싼 이득. (3) 더 작은 student로 distill해 낮은 capacity에서 accuracy 보존. (4) structured prune(channel/head)으로 kernel이 실제로 빨라지게. (5) 대부분의 latency를 위한 INT8 QAT, per-channel scale. (6) operator fusion(Conv-BN-ReLU), thread pinning, memory reuse. (7) 고정 eval subset에 regression test를 잠금. 규율: 한 번에 하나씩, target 하드웨어에서 p50/p95 latency, 그리고 제품 지표가 "1%"를 정의하게. **후속 질문:** *A/B 없이 배포?* — latency/error guardrail을 둔 canary rollout.
</div></details>

## Cheat-sheet

| 질문 | 한 줄 요약 |
| --- | --- |
| BF16 vs FP16 | Exponent bit = range, mantissa = precision. BF16 = FP32 range, loss scaling 불필요. |
| FP8 recipe | E4M3 forward, E5M2 grad, FP32 master/moment. norm/softmax/LM head 제외. |
| NVFP4 vs MXFP4 | E2M1 element. NVFP4 block 16 + FP8 scale(더 세밀), MXFP4 block 32 + E8M0(더 거침). |
| PTQ vs QAT | PTQ = calibrate, 빠름, 위험. QAT = fake-quant 학습, accuracy 회복. |
| GPTQ / AWQ | Hessian-aware vs activation-aware weight-only 4-bit. AWQ가 실용 기본값. |
| Rotation PTQ | QuaRot/SpinQuant가 orthogonal rotation으로 outlier 분산 → W4A4 가능. |
| FlashAttention | IO-aware tiled softmax. 같은 수학, HBM에 $n^2$ matrix 없음. Blackwell asymmetry엔 FA-4. |
| KV cache | PagedAttention + GQA/MLA + quantized KV로 long-context 병목 축소. |
| Speculative decode | Draft + verify. low batch / memory-bound에서 도움. 출력 분포 보존. |
| Compression pipeline | Prune → quantize → distill. FLOPs 아닌 측정 latency로 결정. |

**관련:** [Distributed Training](#/foundations/distributed-training) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [Normalization & Stability](#/foundations/normalization-stability) · [LLM Fundamentals](#/llm/fundamentals) · [Optimization](#/foundations/optimization)
