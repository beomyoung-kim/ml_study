# Distributed Training

<div class="tag-row"><span class="tag">DDP</span><span class="tag">FSDP2</span><span class="tag">ZeRO</span><span class="tag">tensor/pipeline parallel</span><span class="tag">context parallel</span><span class="tag">3D parallelism</span></div>

> [!TIP] 이것부터 말하세요
> 핵심 문장: *"모든 parallelism 전략은 메모리를 communication과 맞바꿉니다 — 저는 모델을 맞추면서 GPU를 바쁘게 유지하는 가장 싼 전략을 고릅니다."* 면접관은 당신이 만져본 가장 큰 클러스터보다, 증상(OOM, hang, low MFU)을 올바른 레버에 매핑할 수 있는지 — 그리고 스케일에 대해 정직한지를 더 봅니다.

## The parallelism zoo

```mermaid
flowchart TB
  subgraph DP ["Data Parallel"]
    D1[GPU0 · batch shard 0] --> R[all-reduce grads]
    D2[GPU1 · batch shard 1] --> R
  end
  subgraph TP ["Tensor Parallel"]
    T1[W shard 0] --- T2[W shard 1]
  end
  subgraph PP ["Pipeline Parallel"]
    P1[stage 0] --> P2[stage 1] --> P3[stage 2]
  end
```

| 전략 | 무엇을 나눔 | Communication | 사용 시점 |
| --- | --- | --- | --- |
| **DDP** | data (model 복제) | grad all-reduce (step당 1회) | 모델이 GPU 하나에 맞음 |
| **FSDP / ZeRO-3** | data + params/grads/opt state | all-gather + reduce-scatter | 모델이 안 맞음 |
| **Tensor (TP)** | layer 내 matrix | layer당 all-reduce | 거대 layer, intra-node (NVLink) |
| **Pipeline (PP)** | layer들을 stage로 | stage 간 activation | 매우 깊음, cross-node |
| **Sequence / Context** | seq 축을 따른 activation | all-gather / ring attention | long context |
| **Expert (EP)** | MoE expert | token all-to-all | MoE 모델 |

실제 대규모 학습은 이들을 **3D(또는 4D) parallelism**으로 조합합니다: 예를 들어 node 내부엔 TP(빠른 NVLink), 몇 개 node에 걸쳐 PP, 나머지에 DP/FSDP, MoE엔 EP.

## Data parallel & DDP

각 rank는 서로 다른 micro-batch로 forward/backward를 돌린 뒤 **gradient를 all-reduce(평균)**해서 모든 replica가 동일한 update를 적용합니다 — 사실상 하나의 large-batch step입니다:

$$
B_{\text{eff}}=B_{\text{local}}\times N_{\text{GPU}}\times N_{\text{accum}}
$$

Ring all-reduce는 bandwidth 최적이며, **gradient bucketing**을 통해 backward와 겹칩니다(뒤쪽 layer가 아직 계산 중일 때 먼저 계산된 grad를 reduce). 고전적 버그: `DistributedSampler` 누락(데이터 중복), effective batch에 맞춰 LR을 안 키움, 혹은 param을 freeze해서 reduce를 놓침.

<details class="qa"><summary>DDP는 어떻게 communication과 computation을 겹치고, 왜 중요한가요?</summary>
<div class="qa-body">

**짧게:** DDP는 gradient를 bucket으로 묶고, backward 중 그 grad가 준비되는 즉시 각 bucket의 all-reduce를 시작합니다. 그래서 communication이 이후 layer의 compute 아래에 숨어, 뒤에서 직렬로 도는 대신입니다.

**깊게:** overlap이 없으면 step time ≈ compute + all-reduce이고, overlap이 있으면 $\max(\text{compute}, \text{comm})$에 근접합니다. Bucket size가 노브입니다: 너무 작으면 → launch overhead와 낮은 bandwidth 활용, 너무 크면 → overlap 감소(bucket을 채우기까지 기다림). Gradient accumulation에서는 중간 micro-step에 `no_sync()`를 쓰고 마지막 것만 reduce를 트리거하게 합니다. **후속 질문:** *언제 overlap이 실패하나?* — 너무 작은 bucket, CPU-launch에 묶인 kernel, 또는 straggler rank가 collective를 멈출 때.
</div></details>

## FSDP2 & ZeRO

**ZeRO**는 optimizer state, 그다음 gradient, 그다음 parameter를 data-parallel rank에 걸쳐 shard하며, 추가 communication을 rank당 메모리와 맞바꿉니다:

| Stage | Shards | Per-rank model state | Comm |
| --- | --- | --- | --- |
| 0 (DDP) | nothing | full | all-reduce |
| 1 | optimizer | ↓ | + |
| 2 | + gradients | ↓↓ | ++ |
| 3 | + parameters | ~1/N | all-gather each layer |

PyTorch **FSDP**는 ZeRO 스타일 sharding을 구현합니다. `FULL_SHARD` ≈ ZeRO-3, `SHARD_GRAD_OP` ≈ ZeRO-2. **FSDP2**(DTensor 기반 재작성, `fully_shard`)는 FSDP1의 FlatParameter를 대체했습니다: parameter별 `DTensor` sharding으로 더 결정론적인 메모리, TP/`torch.compile`과 더 깔끔한 조합성, 더 단순한 distributed checkpointing을 제공합니다. 핵심 노브: `reshard_after_forward` — `True`는 forward 후 param을 해제(메모리 절약, backward에서 재수집 → comm 증가), `False`는 유지(ZeRO-2 유사). 2D `DeviceMesh`는 **hybrid shard**(node 내부는 shard, node 간엔 replicate)를 가능하게 합니다.

> [!NOTE] 메모리는 communication으로 산다
> 모델이 GPU 하나에 맞으면 **보통 DDP가 FSDP보다 빠릅니다**. 꼭 필요할 때만 shard하세요. 그다음, sharded DP조차 communication-bound이거나 단일 layer가 너무 클 때 tensor/pipeline parallel로 갑니다.

<details class="qa"><summary>ZeRO-3가 메모리 절약이 가장 큰데, 왜 항상 쓰지 않나요?</summary>
<div class="qa-body">

**짧게:** ZeRO-3/`FULL_SHARD`는 매 layer마다 parameter를 재수집하므로 communication-bound입니다. 이미 맞는 모델이면 DDP(또는 ZeRO-1/2)가 더 빠릅니다.

**깊게:** 메모리 절약은 실제적(rank당 ~1/N)이지만, 이제 모든 forward·backward layer가 그 layer weight의 all-gather와 grad의 reduce-scatter를 필요로 합니다. 느린 inter-node 링크에서는 이것이 step time을 지배합니다. Decision tree: DDP로 맞나? → DDP. ZeRO-1/2(opt/grads만 shard)로 맞나? → 그렇게. param만 안 맞나? → ZeRO-3, 그리고 gather가 intra-node에 머물도록 TP 고려. **후속 질문:** *CPU/NVMe offload?* — ZeRO-Infinity / FSDP offload는 메모리를 더 줄이지만 PCIe bandwidth에 병목됩니다.
</div></details>

## Tensor, pipeline, sequence & context parallelism

<dl class="kv">
<dt>Tensor parallel (TP)</dt><dd>Weight matrix를 GPU에 걸쳐 분할(column/row partition). layer당 all-reduce가 필요하므로 NVLink 위의 <b>intra-node</b>로 유지. Megatron 스타일.</dd>
<dt>Pipeline parallel (PP)</dt><dd>Layer 범위를 stage에 할당하고 micro-batch를 흘려보냄. <b>bubble</b>(fill/drain 시 idle 시간)을 도입 ≈ $P$ stage, $M$ micro-batch에 대해 $(P-1)/(M+P-1)$ — 그래서 많은 micro-batch와 interleaved 스케줄(1F1B)을 사용.</dd>
<dt>Sequence parallel (SP)</dt><dd>LayerNorm/Dropout activation만 seq 축을 따라 shard해 TP를 보완하고 activation 메모리를 줄임.</dd>
<dt>Context parallel (CP)</dt><dd>long-context 학습을 위해 <b>모든</b> activation을 seq 차원을 따라 shard. ring/all-gather attention이 rank 간 K/V를 교환. 이것이 long-context를 가능케 함.</dd>
</dl>

> [!IMPORTANT] SP vs CP (2026년 단골 주제)
> **Sequence parallelism**은 *norm/dropout* activation만 shard합니다(tensor parallelism의 add-on). **Context parallelism**은 seq 축을 따라 *모든* activation을 shard해서 단일 GPU가 전체 시퀀스를 갖지 않게 합니다 — 이것이 100K–1M-token 학습을 가능하게 합니다. Megatron-Core와 long-context 학습 스택은 CP에 의존합니다. *(메커니즘은 verifiable. 정확한 API는 현재 Megatron-Core 문서로 확인할 것.)*

<details class="qa"><summary>시퀀스가 GPU 하나에 안 맞는 1M-token-context 모델을 어떻게 학습하나요?</summary>
<div class="qa-body">

**짧게:** **context parallelism**을 씁니다 — activation을 seq 차원을 따라 GPU에 걸쳐 shard하고 K/V에 대한 ring/all-gather로 attention을 계산 — weight엔 FSDP, activation checkpointing을 결합합니다.

**깊게:** activation 메모리는 시퀀스 길이에 비례하므로 작은 모델조차 1M-token 시퀀스를 담을 수 없습니다. CP는 시퀀스를 CP group에 걸쳐 나눕니다. 각 rank는 query의 한 chunk를 소유하고 key/value를 교환(ring attention)해서 모든 query가 여전히 global하게 attend하며, rank당 $O(n)$ 메모리입니다. FSDP(weight), layer가 크면 TP, activation checkpointing(backward에서 recompute)을 얹습니다. Sequence parallelism은 추가로 TP가 복제된 채 남긴 norm/dropout activation을 shard합니다. **후속 질문:** *비용?* — attention layer당 추가 K/V communication. 이를 compute와 겹치고 CP group을 빠른 링크에 둡니다.
</div></details>

## Expert parallelism & MoE <span class="badge badge-2026">2026</span>

거의 모든 2025–2026 프론티어 모델이 Mixture-of-Experts이므로, **expert parallelism (EP)**은 이제 1급 차원입니다. 각 token은 몇 개의 expert로 라우팅(top-k)되고, expert들은 GPU에 분산되므로 forward pass는 token을 해당 expert로 보내는 **all-to-all**과 결과를 모으는 또 하나의 all-to-all이 필요합니다.

<dl class="kv">
<dt>Load balancing</dt><dd>불균등한 라우팅은 일부 expert를 과부하시키고 다른 것을 굶깁니다 → straggler. Auxiliary load-balancing loss(또는 aux-loss-free bias 트릭) + <b>capacity factor</b>(overflow token drop/pad)로 균등하게 유지.</dd>
<dt>All-to-all cost</dt><dd>라우팅 all-to-all이 MoE 특유의 병목. EP group을 빠른 intra-node 링크에 두고 dispatch를 compute와 겹침.</dd>
<dt>Composition</dt><dd>EP는 DP/TP/PP와 곱해짐 — 4D 레이아웃. <b>Megatron-Core</b>는 대규모에서 TP + PP + DP + EP 결합의 레퍼런스 구현. DeepSpeed와 FSDP2는 PyTorch-native 중간 지대를 커버.</dd>
</dl>

**active vs. total** param을 보고하세요: active는 token당 compute/latency를 좌우하고, total은 메모리/capacity를 좌우합니다 — 둘이 분리되는 것이 MoE의 요점입니다.

## Gradient accumulation & memory budget

메모리가 빡빡할 때 $N_{\text{accum}}$개 micro-batch에 걸쳐 grad를 누적해 큰 effective batch에 도달하고, 마지막 micro-step에서만 sync합니다. Loss reduction(mean vs sum)이 LR과 맞는지, BN stats가 micro-batch를 쓰는지(또는 SyncBN/GN 사용) 주의하세요. 메모리는 어디로 가나?

| 항목 | 줄이는 방법 |
| --- | --- |
| Params / grads / optimizer state | FSDP/ZeRO, 8-bit Adam, layer freeze |
| Activations | checkpointing, 짧은 seq, context parallel |
| Temp / workspace | 작은 micro-batch, FlashAttention |
| Fragmentation | allocator 설정(`expandable_segments`), 재시작 |

좋은 답변은 **model-state OOM**(sharding/precision으로 해결)과 **activation OOM**(checkpointing/seq length로 해결)을 구분합니다 — [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) 참고.

## Diagnosing throughput & hangs

**MFU**(Model FLOPs Utilization) = 달성 FLOPs / 하드웨어 peak가 대표 효율 지표입니다. 선형 scaling에서 벗어나면 communication, I/O, 또는 straggler를 가리킵니다.

| 증상 | 가설 | 조치 |
| --- | --- | --- |
| 시작 시 OOM | model state | FSDP/TP, precision |
| iteration 중간 OOM | activations | checkpointing, seq len |
| all-reduce에서 hang | NCCL / straggler / mismatched collective | `NCCL_DEBUG=INFO`, synthetic-data run |
| 1 vs N GPU에서 loss 다름 | sampler / reduction / sync 버그 | 1-GPU parity test |
| Low MFU, GPU idle | comm bound 또는 dataloader starvation | topology, bucket size, I/O 확인 |

<details class="qa"><summary>Multi-node 학습이 all-reduce에서 hang합니다. 어떻게 디버깅하나요?</summary>
<div class="qa-body">

**짧게:** 먼저 *communication*을 *compute/data*와 분리 — `NCCL_DEBUG=INFO`를 켜고, loader를 배제하기 위해 synthetic data로 돌리고, 한 rank가 straggler인지 아니면 collective가 rank 간에 mismatch인지 확인합니다.

**깊게:** hang은 보통 rank들이 collective에 대해 불일치(다른 shape, 조건/branch로 step을 건너뛴 rank, 또는 불균등한 마지막 batch)하거나, 진짜 전송 문제(NVLink/IB/RoCE topology, Ethernet fallback 경로, 방화벽)를 의미합니다. 체크리스트: (1) `NCCL_DEBUG=INFO`로 어디서 멈추는지 확인, (2) 모든 rank가 같은 collective에 도달하는지 확인(데이터 의존 제어 흐름 방지), (3) `--synthetic-data`로 dataloader I/O 격리, (4) 느린/straggler rank 점검, (5) GPU-process binding과 interconnect 검증. 모델 버그라고 단정하지 마세요. **후속 질문:** *Gradient compression?* — bandwidth를 아끼지만 correctness/accuracy 트레이드오프를 더함. overlap/bucketing을 먼저 시도.
</div></details>

> [!WARNING] 스케일에 대해 정직하라
> 당신의 가장 큰 실제 학습이 예를 들어 4 node × 8×H100(32 GPU)이었다면 정확히 그렇게 말하고 *그것*에 대해 정밀하게 말하세요 — effective batch, 전략, 겪은 실패. "수천 개 GPU"는 원리적으로 이해하고 성장해 갈 수 있는 목표 환경으로 프레이밍하세요. 과장은 systems 면접에서 떨어지는 가장 빠른 길입니다.

## Cheat-sheet

| 질문 | 한 줄 요약 |
| --- | --- |
| DDP | 모델 복제, data shard, grad all-reduce. effective batch = local × GPUs × accum. |
| ZeRO 1/2/3 | optimizer → +grads → +params shard. 메모리 줄이려 comm 늘림. |
| FSDP2 | DTensor `fully_shard`. `FULL_SHARD`≈ZeRO-3. `reshard_after_forward` = memory↔comm 노브. |
| TP vs PP | TP는 matrix 분할(intra-node, layer당 all-reduce). PP는 layer 분할(bubble ∝ stage 수). |
| SP vs CP | SP는 norm/dropout activation shard. CP는 seq 축의 *모든* activation shard → long context. |
| 3D parallelism | TP intra-node × PP across nodes × DP/FSDP 나머지 (+ MoE엔 EP). |
| Grad accumulation | micro-batch로 큰 effective batch. 마지막 step에서만 sync. |
| MFU | 달성/peak FLOPs. low MFU ⇒ comm, I/O, 또는 straggler. |
| OOM triage | Model-state OOM → shard/precision. activation OOM → checkpoint/seq len. |

**관련:** [Mixed Precision & Efficiency](#/foundations/mixed-precision-efficiency) · [Normalization & Stability](#/foundations/normalization-stability) · [CNNs, RNNs & Transformers](#/foundations/architectures) · [Optimization](#/foundations/optimization) · [Debugging & Experimentation](#/foundations/debugging-experimentation)
