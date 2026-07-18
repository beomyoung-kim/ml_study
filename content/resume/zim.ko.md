# Deep-Dive: ZIM — Zero-Shot Image Matting for Anything

<div class="tag-row"><span class="tag">ICCV 2025 Highlight</span><span class="tag">SAM-based</span><span class="tag">zero-shot matting</span><span class="tag">data engine</span><span class="tag">first author</span></div>

> [!TIP] 30초 pitch
> ZIM은 **promptable, zero-shot image-matting foundation model**입니다. SAM의 prompt 기반 zero-shot 범용성은 그대로 유지하되, hard mask를 **미세하고 부드러운 soft alpha matte**로 교체합니다. 핵심 트릭은 SAM을 공개 matting 데이터로 fine-tuning하는 것(그러면 zero-shot이 macro portrait로 붕괴)이 아니라, SA-1B의 ~1M segmentation label을 micro-level matte로 바꾸는 **label converter(SGA + STL)**를 만들고, 여기에 **hierarchical pixel decoder**와 **prompt-aware masked attention**을 더하는 것입니다. 결과: ICCV 2025 **Highlight**(상위 ~3%), demo와 함께 오픈소스, 그리고 **CLOVA-X Image Editing**에 통합.

**Public references:** [paper (arXiv 2411.00626)](https://arxiv.org/abs/2411.00626) · [code](https://github.com/naver-ai/ZIM) · [project](https://naver-ai.github.io/ZIM/). 주제별 [Matting](#/cv/matting)과 [Vision Foundation Models](#/cv/foundation-models) 챕터도 참고.

## 문제 & 동기

2024년에 두 가지가 동시에 참이면서 서로 긴장 관계였습니다:

1. **SAM**은 대규모로 zero-shot, promptable segmentation을 제공합니다 — 하지만 그 mask는 **hard하고, binary이며, coarse**합니다(SA-1B label이 coarse하고, pixel decoder는 단순한 stride-4 upsample이라 checkerboard artifact가 생기고 soft boundary가 없습니다). 머리카락, 털, 반투명은 죽습니다.
2. **Matting** 모델은 아름다운 soft alpha를 내지만, 기존의 "matte anything" 시도들(Matte-Anything, Matting-Anything)은 그저 **공개 matting 데이터셋에 fine-tune**할 뿐인데, 그 데이터는 거의 전부 *macro* 전신 portrait입니다. 이렇게 fine-tune하면 **macro granularity에 overfit**됩니다: 신발끈을 prompt하면 사람 전체를 돌려줍니다. Zero-shot promptability가 붕괴합니다.

순진한 해법 — SA-1B 규모로 micro-level matte를 annotation하기 — 는 경제적으로 불가능합니다. ZIM의 논지: **granularity 문제는 data 문제이므로, 더 많은 사람 label이 아니라 data engine으로 풀어라.**

## Contribution 1 — data engine (SGA + STL)

기존 **segmentation** label을 **matte** label로, SA-1B 규모에서 변환합니다.

<dl class="kv">
<dt>Label Converter</dt><dd>공개 matte(AIM-500, AM-2K, P3M-10K, RWP-636, HIM-2K, RefMatte)와 coarse non-matte mask(ADE20K)를 섞은 데이터로 학습한 network(mask-guided matting 스타일, 예: MGMatting + Hiera backbone). 입력: image + coarse mask. 출력: fine matte. Loss: $\mathcal{L}=\mathcal{L}_{1}+\lambda\,\mathcal{L}_{\text{grad}}$, $\lambda=10$.</dd>
<dt>Spatial Generalization Augmentation (SGA)</dt><dd>segmentation 입력과 matte target 모두에 <b>동일한</b> random cut-out 영역을 적용해, converter가 전체 객체가 아니라 <b>부분 / micro</b> 영역에서 fine detail을 만들어내도록 학습시킵니다. 이것이 "micro granularity"를 가르치는 부분입니다.</dd>
<dt>Selective Transformation Learning (STL)</dt><dd>fine matte가 무의미하거나 noisy한 class(car, desk, 강체)에 대해서는 target을 <b>segmentation mask 자체</b>(identity)로 설정합니다. Converter는 soft edge가 없어야 할 곳에서 *hallucinate하지 않도록* 학습됩니다 → label noise 감소.</dd>
<dt>Output</dt><dd><b>SA1B-Matte</b>: SA-1B 이미지를 micro-level fine matte로 다시 label한 것.</dd>
</dl>

> [!EXAMPLE] 왜 SGA와 STL 둘 다 중요한가 (ablation 직관)
> SGA가 없으면 converter는 전체 객체 matte만 알아서 → micro prompt가 실패합니다. STL이 없으면 강체에 soft edge를 강제해서 → noisy한 학습 신호가 됩니다. 논문의 converter-quality ablation은 둘 중 하나만 빼도 MicroMat 변환 품질이 떨어짐을 보여줍니다. 소리 내어 말할 교훈: *"foundation model의 behavior는 그 data가 쓴 계약이다 — 나는 그 계약을 설계했다."*

## Contribution 2 — 모델

같은 SAM 골격(ViT image encoder + prompt encoder + transformer decoder + pixel decoder), 두 개의 정밀한 변경:

<dl class="kv">
<dt>Hierarchical Pixel Decoder</dt><dd>SAM의 단일 stride-4 upsample 대신, <b>stride 2/4/8의 multi-level feature를 융합</b>해 점진적으로 upsampling하며 image embedding을 concat합니다. 이것이 checkerboard를 없애고, 고주파 boundary detail을 복원하며, semantic을 보존합니다. 비용: V100에서 ~+10ms — 품질 향상에 비하면 저렴합니다.</dd>
<dt>Prompt-Aware Masked Attention</dt><dd>decoder의 <b>token→image</b> cross-attention을 prompt된 영역 쪽으로 bias합니다. Box prompt → binary mask가 박스 밖 attention logit을 $-\infty$로 설정. Point prompt → Gaussian($\sigma=21$)이 QK score를 modulate. 결정적 디테일: <b>token→image에만</b> 적용하고 image→token에는 절대 적용하지 않음 — I2T에 강제하면 global image feature가 손상됩니다.</dd>
</dl>

학습: SAM weight로 초기화, **SA1B-Matte의 1%(~2.2M matte label)**로 학습, 500K iteration, AdamW LR 1e-5. 흥미롭게도 10%로 늘려도 이득이 미미합니다 — SAM 초기화가 이미 visual prior를 담고 있어서, converter의 *양*보다 *granularity*가 더 중요합니다.

```mermaid
flowchart TB
  subgraph engine ["Data engine (offline)"]
    SA1B["SA-1B<br/>coarse seg (~1M imgs)"] --> CONV["Label Converter<br/>+ SGA + STL"]
    PUB["Public mattes<br/>AIM/AM-2K/P3M/..."] --> CONV
    CONV --> SA1BM["SA1B-Matte<br/>micro-level fine mattes"]
  end
  subgraph model ["ZIM"]
    SAMW["SAM weights (init)"] --> ENC["ViT image encoder"]
    PROMPT["box / point / scribble / text"] --> PENC["prompt encoder"]
    ENC --> DEC["transformer decoder<br/>+ prompt-aware masked attn"]
    PENC --> DEC
    DEC --> HPD["hierarchical pixel decoder<br/>(stride 2/4/8)"]
    HPD --> OUT["alpha matte α ∈ [0,1]"]
  end
  SA1BM --> DEC
  style OUT fill:#e0533f,color:#fff
  style SA1BM fill:#12a150,color:#fff
```

## 평가 & 결과 framing

fine-grained promptable-matting benchmark가 존재하지 않았기 때문에, 논문은 자체 benchmark를 기여합니다:

- **MicroMat-3K**: 3K 고해상도 이미지(*fine* 750 + *coarse* 2,250), point 및 box prompt 포함. DIV2K → SAM AMG pseudo-seg → converter matte → **사람 검토/교정**으로 구축. Fine/coarse 분리로 micro fidelity를 따로 측정할 수 있습니다.
- **Headline (ViT-B, fine subset, box prompt):** ZIM **SAD ≈ 9.96 / MSE ≈ 1.89** vs SAM **36.09 / 11.06**, Matte-Anything **34.66 / 9.75**, Matting-Anything **≈246 / 68**. *(논문 수치; 낮을수록 좋음.)*
- **결정적 ablation:** 같은 architecture를 SA1B-Matte 대신 Public-Matte로 학습 → fine SAD가 ~120으로 급등. **Data granularity가 architecture를 지배한다.** 이것이 외워야 할 슬라이드입니다.
- **Downstream:** ZIM은 Matte-Anything / Matting-Anything / HQ-SAM에 backbone으로 꽂히고, Inpaint-Anything, medical segmentation, SA3D 3D segmentation을 개선합니다. **Grounded-ZIM** = Grounding DINO box → ZIM으로 text-prompted matting.

> [!NOTE] Production 반향 — CLOVA-X (대외비 세부사항)
> ZIM은 [CLOVA-X Image Editing](https://dan.naver.com/24/sessions/597)에 통합되어 있습니다. 별개의 내부 **foreground-segmentation API**(내부 eval에서 Photoroom / Remove.bg / Adobe를 이겼다고 보고됨)는 DNA를 공유하지만 — boundary quality + data curation — **다른 제품 라인**입니다. 같은 weight라고 주장하지 마세요. 통합 세부사항과 SLA는 **대외비**이며, matte↔editor 핸드오프(resolution, premultiplied alpha, color spill), latency budgeting, failure case용 hard-threshold fallback으로 framing하세요.

## 예상 deep-dive Q&A

<details class="qa"><summary>matting에 그냥 SAM을 쓰면 안 되나요?</summary>
<div class="qa-body">

**Short:** SA-1B label은 coarse하고, pixel decoder는 단순한 stride-4 upsample(checkerboard)이며, objective는 *hard* mask입니다. 그중 어느 것도 soft alpha나 머리카락 수준 구조에 최적화되어 있지 않습니다.

**Deep:** Matting은 sub-pixel boundary gradient와 연속적인 $\alpha$가 필요합니다. SAM의 decoder는 그에 필요한 고주파 정보를 버리고, 학습 target에는 배울 soft transition이 없습니다. 그래서 supervision(soft matte)과 decoder(multi-scale, high-res)를 **둘 다** 바꿔야 합니다 — 그게 정확히 ZIM의 두 기여입니다.
</div></details>

<details class="qa"><summary>그러면 왜 그냥 공개 matting 데이터셋에 SAM을 fine-tune하지 않나요?</summary>
<div class="qa-body">

**Short:** macro-overfit되어 zero-shot promptability가 파괴됩니다.

**Deep:** 공개 matting 데이터는 압도적으로 전신 portrait(macro)입니다. 여기에 fine-tune하면 모델은 "matte = 전체 salient object"를 학습해서, micro prompt(신발끈, 머리카락 한 올)에도 여전히 사람 전체를 돌려줍니다. Public-Matte 학습 ablation(fine SAD ~120 vs ZIM ~10)이 *바로* 이 붕괴를 정량화한 것입니다. 해법은 **micro-granular** supervision을 대규모로 제조하는 converter입니다.
</div></details>

<details class="qa"><summary>prompt-aware masked attention과 T2I-only 디테일을 설명해 주세요.</summary>
<div class="qa-body">

**Short:** decoder cross-attention을 prompt된 영역 쪽으로 bias합니다 — box → 박스 밖은 hard $-\infty$ mask; point → Gaussian($\sigma=21$) reweighting — 단, token→image attention에만.

**Deep:** 출력 token은 prompt된 영역을 *봐야* 하므로 token→image를 masking하면 거기에 집중합니다. 하지만 image feature 자체는 공유되는 global representation이라서, image→token까지 masking하면 모든 token에 대해 그 representation을 손상시켜 품질이 떨어집니다. 양방향을 모두 ablate해야만 드러나는 비대칭성입니다.
</div></details>

<details class="qa"><summary>CV에는 "~1M-image curated dataset"이라고 되어 있습니다. "1% of SA1B-Matte, ~2.2M mattes"와 어떻게 조화시키죠?</summary>
<div class="qa-body">

**Short:** ~1M은 data engine이 작동하는 SA-1B 이미지 규모를 가리키고, ZIM의 *학습 run*은 변환된 matte label의 ~1%(~2.2M matte instance, 이미지 하나가 여러 mask를 내므로)를 썼습니다.

**Deep:** 면접장에서는 정확히 말하겠습니다: converter가 SA-1B 이미지 풀(~1M 이미지)을 SA1B-Matte로 다시 label하고, 학습은 1% subset을 썼습니다 — SAM-init 덕에 return이 빠르게 줄어들기 때문입니다(10%는 거의 도움이 안 됨). 정직한 framing은 headline 이미지 수가 아니라 *granularity 품질 + SGA/STL filtering*입니다 — curated-image 숫자를 부풀리지 않겠습니다.
</div></details>

<details class="qa"><summary>왜 이것이 ICCV Highlight이고 "또 하나의 matting fine-tune"이 아닌가요?</summary>
<div class="qa-body">

**Short:** matting을 *foundation-model + data-engine* 문제로 재정의하고, 재현 가능한 pipeline, 새로운 benchmark, 폭넓은 downstream transfer로 뒷받침합니다.

**Deep (reviewer 관점):** (1) 정밀한 문제 정의 — zero-shot vs fine-grained 긴장; (2) 더 많은 사람 label이 아니라 확장 가능하고 재현 가능한 data 해법; (3) 하나의 hack이 아니라 data + architecture의 *공동* 설계; (4) MicroMat-3K가 benchmark 공백을 메움; (5) inpainting, medical, 3D에서의 downstream 성과가 범용성을 보임; (6) demo와 함께 완전 오픈소스. Public-Matte ablation이 기여를 *가독성 있게* 만듭니다.
</div></details>

### 어려운 follow-up

<details class="qa"><summary>전통적인 AIM-500 benchmark는 ZIM만큼 box prompt를 선호하지 않습니다. 방어해 보세요.</summary>
<div class="qa-body">

약점이 아니라 **domain shift**입니다. 전통적 matting benchmark는 하나의 *전체 salient object*를 가정하지만, ZIM은 *interactive, object/part-level* promptable model입니다. 의도한 target에 맞는 multi-point prompt를 주면 거기서도 기존 방법을 능가합니다. 전체 객체 benchmark에서의 single-box는 단지 promptable foundation의 잘못된 사용법일 뿐입니다.
</div></details>

<details class="qa"><summary>ZIM을 video로 어떻게 가져가겠습니까?</summary>
<div class="qa-body">

SAM2 스타일의 **memory/propagation** 메커니즘을 $\alpha$의 temporal consistency와 결합합니다(flicker는 hard mask보다 soft matte에서 훨씬 잘 보입니다). 정직한 답: 이건 **미해결 문제**입니다 — occlusion과 re-identification 하에서 matte의 temporal stability는 풀리지 않았고, 저는 이걸 주장이 아니라 future work로 다루겠습니다.
</div></details>

<details class="qa"><summary>alpha matte는 diffusion editing pipeline에 어떻게 꽂히나요?</summary>
<div class="qa-body">

soft conditioning signal로: 추출용 premultiplied-alpha compositing, 또는 latent inpainting / region-conditioned generation을 위한 spatial mask로서의 $\alpha$. Soft boundary가 합성물을 오려낸 것처럼 안 보이게 만드는 요소입니다. 핸드오프 이슈(resolution, color spill, premultiply 관례)가 진짜 엔지니어링이고 — CLOVA-X 세부사항은 대외비입니다.
</div></details>

## 솔직한 한계

- SAM에서 물려받은 **prompt ambiguity**(point가 부분을 뜻할 수도 전체를 뜻할 수도 있음).
- $\alpha$에 대한 **uncertainty modeling 없음** — 모델이 "이 머리카락 boundary는 확신이 없다"고 말하지 못함.
- **투명 / 유리 / 불**은 여전히 약함; trimap 기반 SOTA가 아직 이깁니다(data 부족).
- 전체 객체 matting benchmark와의 **철학적 불일치**(object/part vs whole-salient).

## 어떤 JD와 연결되는가

| Company | Connection |
| --- | --- |
| Adobe | generative fill & editing tool을 위한 conditioning으로서 precise mask / alpha |
| ByteDance Seed Vision | Visual generative foundation + controllable editing |
| Meta | Segmentation foundation → multimodal creative tool |
| Apple | On-device portrait/editing 품질; data-curation 규율 |
| NVIDIA | graphics/robotics stack에 공급되는 perception module |

## Cheat-sheet

| Item | Value |
| --- | --- |
| Venue | ICCV 2025 **Highlight** (상위 ~3%), 1저자 |
| One-liner | Promptable **zero-shot matting foundation** (SAM prompt → soft micro-level $\alpha$) |
| Data engine | Label Converter + **SGA** (micro generalization) + **STL** (rigid identity) → SA1B-Matte |
| Model | **Hierarchical pixel decoder** (stride 2/4/8, +~10ms) + **prompt-aware masked attn** (T2I only) |
| Loss | $\mathcal{L}_1 + \lambda\mathcal{L}_{\text{grad}}$, $\lambda=10$; point Gaussian $\sigma=21$ |
| Train | SAM-init, 1% SA1B-Matte (~2.2M mattes), 500K iters, AdamW 1e-5 |
| Benchmark | **MicroMat-3K** (fine 750 / coarse 2250); ViT-B box-fine SAD ~10 vs SAM ~36 |
| Killer ablation | Public-Matte vs SA1B-Matte: fine SAD ~120 → ~10 (**data > architecture**) |

## Cross-links
- 주제별: [Image Matting](#/cv/matting) · [Vision Foundation Models](#/cv/foundation-models) · [Segmentation](#/cv/segmentation)
- Deep-dive: [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM & Agents](#/resume/grounded-vlm-agents) · [CV → Interview Map](#/resume/overview)으로 돌아가기
