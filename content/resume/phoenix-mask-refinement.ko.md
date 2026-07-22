# Deep-Dive: Phoenix — Adversarial Mask Refinement

<div class="tag-row"><span class="tag">ECCV 2026</span><span class="tag">mask refinement</span><span class="tag">adversarial perturbation</span><span class="tag">contrastive learning</span><span class="tag">SAM-based</span><span class="tag">first author</span></div>

> [!TIP] 30초 pitch
> Phoenix는 입력 segmentation 모델을 재학습하지 않고 거친 mask를 다듬는 **mask-refinement** 프레임워크입니다. 기존 refiner의 형태학적 잡음(dilation/erosion)이 실제 model error를 충분히 닮지 못한다는 문제에서 출발해, (1) embedding-space adversarial attack으로 model-aware hard perturbation을 만드는 **AMP**, (2) GT·noisy·refined mask 관계를 학습하는 **CMRL**을 제안합니다. 논문이 평가한 protocol에서는 기존 refiner보다 높은 결과와 weak/semi-supervised pseudo-label refinement에서 최대 **+16.1 AP point**를 보고합니다.

**Public status/reference:** 저자의 [publication page](https://beomyoung-kim.github.io/)에 **ECCV 2026**으로 공개되어 있습니다(2026-07-21 확인). 별도 [project page](https://phoenix-eccv26.github.io)는 공개 상태가 바뀔 수 있으므로 지원 직전에 다시 확인하세요. 주제별 [Segmentation](#/cv/segmentation) · [Weak & Semi-Supervised](#/cv/weak-semi-supervised) · [Vision Foundation Models](#/cv/foundation-models) 챕터도 참고합니다.

## 문제 & 동기

SOTA segmentation 모델(Mask2Former, SAM 등)조차 여전히 **경계가 부정확하고, 의미적으로 헷갈리며, 구조적으로 깨진** mask를 냅니다. **Mask refinement**는 이를 아키텍처 변경 없이 후처리로 고칩니다 — 모델을 못 건드리는 상황(독점 제약, 계산 한계)이나, **pseudo-label을 신뢰할 supervision으로 바꿔야 하는 약지도/준지도 학습**에서 특히 값집니다([PointWSSIS](#/resume/pointwssis-bestie)의 문제의식과 직결).

refinement 학습의 심장은 **(noisy, clean) mask 쌍을 어떻게 만드느냐**입니다. 그런데 기존 방법의 급소가 여기 있습니다:

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="160" y="20" text-anchor="middle" font-weight="700" fill="#98a3b2">형태학적 잡음 (기존)</text>
  <rect x="60" y="35" width="200" height="120" rx="8" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <!-- blob with random jagged noise -->
  <path d="M120 70 Q160 55 200 75 Q215 100 195 125 Q160 140 125 122 Q108 98 120 70 Z" fill="none" stroke="#6366f1" stroke-width="2"/>
  <g stroke="#6366f1" stroke-width="1" opacity="0.6">
    <circle cx="118" cy="72" r="3"/><circle cx="205" cy="90" r="3"/><circle cx="130" cy="130" r="3"/><circle cx="195" cy="128" r="3"/>
  </g>
  <text x="160" y="178" text-anchor="middle" fill="#6366f1">무작위·의미 무관</text>
  <text x="160" y="195" text-anchor="middle" fill="#98a3b2" font-size="10">edge 상관 ≈ 0</text>

  <text x="480" y="20" text-anchor="middle" font-weight="700" fill="#e0533f">Adversarial (Phoenix)</text>
  <rect x="380" y="35" width="200" height="120" rx="8" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <path d="M440 70 Q480 55 520 75 Q535 100 515 125 Q480 140 445 122 Q428 98 440 70 Z" fill="none" stroke="#e0533f" stroke-width="2"/>
  <!-- noise concentrated near a semantically hard boundary (top-right) -->
  <path d="M505 62 Q525 70 528 88" fill="none" stroke="#e0533f" stroke-width="3" opacity="0.8"/>
  <path d="M512 66 Q532 76 534 92" fill="none" stroke="#e0533f" stroke-width="2" opacity="0.5"/>
  <text x="480" y="178" text-anchor="middle" fill="#e0533f">경계·모호 영역에 집중</text>
  <text x="480" y="195" text-anchor="middle" fill="#98a3b2" font-size="10">edge 상관 [−0.6, 0.8]</text>
</svg>
<figcaption>기존 형태학적 잡음(왼쪽)과 Phoenix의 model-aware perturbation(오른쪽)을 개념적으로 비교한 그림. 논문은 adversarial noise의 image-edge Pearson 상관 분포가 더 넓다고 보고하지만, 이를 calibrated predictive uncertainty로 해석해서는 안 됩니다.</figcaption>
</figure>

> [!NOTE] 한 줄 논지
> "실제 segmentation 오류는 **구조적·문맥 의존적**인데, 형태학적 잡음은 그렇지 않다. 그래서 나는 adversarial perturbation을 *파괴적 공격 도구*에서 *건설적 데이터 생성 도구*로 용도 변경했다." — 소리 내어 말할 한 문장.

## Contribution 1 — Adversarial Mask Perturbation (AMP)

**아이디어:** GT mask에서 realistic한 noisy mask를 만들되, 잡음의 위치를 **모델이 헷갈리는 곳**으로 유도한다. 이를 위해 pretrained SAM decoder를 얼린 채, 임베딩 공간에서 **FGSM 스타일 adversarial 공격**을 돌립니다.

<dl class="kv">
<dt>어디를 공격하나</dt><dd>입력 이미지 픽셀이 아니라 <b>임베딩 공간의 학습 가능한 perturbation embedding $\mathbf{E}_p$</b>를 공격합니다. 이유: (1) 계산 효율(가벼운 4M decoder만 재사용, V100에서 perturbation 갱신 1회 ~6ms), (2) 픽셀이 아닌 <b>고수준 의미</b> 수준의 교란이 가능.</dd>
<dt>FGSM 갱신</dt><dd>$\mathbf{E}_p \leftarrow \mathbf{E}_p + \alpha\cdot\mathrm{sign}(\nabla_{\mathbf{E}_p}\mathcal{L}_{adv})$. Loss gradient는 현재 decoder가 민감한 embedding 방향을 찾는 신호입니다. 큰 gradient를 calibrated predictive uncertainty와 동일시할 수는 없고, FGSM의 sign update도 magnitude 자체를 보존하지 않습니다. 따라서 "불확실성을 정확히 측정"한다고 말하기보다 <b>model-aware hard perturbation을 만든다</b>고 설명하는 편이 정확합니다.</dd>
<dt>Guidance Mask로 방향 제어</dt><dd>adversarial 목표 $\mathcal{L}_{adv}=-\mathcal{D}(f_{dec}(\cdot),\mathcal{M}_g)$의 guidance mask $\mathcal{M}_g$를 바꿔 <b>오류 유형을 통제</b>: <b>Expansion</b>($\mathcal{M}_g=\mathbf{1}$)→false-positive(경계 밖으로 팽창), <b>Contraction</b>($\mathcal{M}_g=\mathbf{0}$)→false-negative(under-segmentation), <b>Inversion</b>($\mathcal{M}_g=1-\mathcal{M}_t$)→둘의 균형.</dd>
<dt>IoU 기반 적응적 강도</dt><dd>생성된 noisy mask와 target의 IoU가 목표 구간 $[\tau, \tau+\epsilon]$에 들 때까지 FGSM을 반복하고, 안 되면 step size를 $\alpha\!\leftarrow\!\alpha/10$로 줄여 재시도(Algorithm 1). 학습 때 $\tau\!\sim\!\mathcal{U}(0.3, 0.9)$로 무작위 샘플 → 쉬운 잡음부터 aggressive한 잡음까지 다양한 난이도를 커버.</dd>
</dl>

> [!IMPORTANT] 왜 임베딩 공격이 픽셀 공격보다 나은가
> 통상적 adversarial attack은 예측 loss를 키우는 방향으로 입력이나 representation을 교란합니다. Phoenix는 그 방향을 refiner 학습용 hard-noise 생성에 재사용합니다. 얼린 encoder embedding을 재사용해 계산을 줄이고, 저자들은 morphology-only noise보다 model error에 가까운 교란을 만든다고 해석합니다. 핵심 프레이밍은 "adversarial perturbation을 학습 데이터 생성에 재사용했다"입니다.

## Contribution 2 — Contrastive Mask Refinement Learning (CMRL)

**아이디어:** 통상 refinement는 픽셀별 분류(각 픽셀 독립)로 봅니다. Phoenix는 대신 **GT($\mathcal{M}_t$)·noisy($\mathcal{M}_n$)·refined($\mathcal{M}_r$) 세 마스크의 관계**를 feature 공간에서 삼방향 contrastive로 모델링합니다.

먼저 픽셀을 세 마스크에서의 상태로 **6개 영역**으로 분류합니다 — True(맞음), Success(고쳐냄), Failure(아직 틀림) × {전경, 배경}:

$$
\mathcal{T}_{fg}=(\mathcal{M}_t{=}1)\wedge(\mathcal{M}_n{=}1)\wedge(\mathcal{M}_r{=}1),\quad
\mathcal{S}_{fg}=(\mathcal{M}_t{=}1)\wedge(\mathcal{M}_n{=}0)\wedge(\mathcal{M}_r{=}1),\quad
\mathcal{F}_{fg}=(\mathcal{M}_t{=}1)\wedge(\mathcal{M}_n{=}0)\wedge(\mathcal{M}_r{=}0)
$$

(배경 $\mathcal{T}_{bg},\mathcal{S}_{bg},\mathcal{F}_{bg}$도 대칭적으로 정의.) 이 위에서 세 loss:

<dl class="kv">
<dt>Intra-Class Consistency $\mathcal{L}_{intra}$</dt><dd>같은 의미 영역(전경/배경) 안의 feature를 InfoNCE로 <b>뭉치게</b> — 실패 픽셀 feature를 올바른 같은-클래스 feature 쪽으로 끌어당김.</dd>
<dt>Inter-Class Contrast $\mathcal{L}_{inter}$</dt><dd>전경 실패 feature를 <b>모든 배경 feature에서 밀어냄</b>(양방향). 오류 다발 영역에서 전경/배경 결정 경계를 선명하게.</dd>
<dt>Self-Improvement $\mathcal{L}_{self}$</dt><dd>핵심 혁신. 현재 <b>실패 영역($\mathcal{F}$) feature</b>를 이미 <b>고쳐낸 영역($\mathcal{S}$) feature</b>를 닮도록 유도 → "틀림→맞음" 변환을 명시적으로 학습하는 <b>부트스트래핑</b>. 같은 이미지 안에서 자기 성공을 교사로 삼음.</dd>
</dl>

$$\mathcal{L}_{CMRL}=\lambda_{intra}\mathcal{L}_{intra}+\lambda_{inter}\mathcal{L}_{inter}+\lambda_{self}\mathcal{L}_{self}\quad(\lambda=0.4,0.4,0.2)$$

SAM의 Dice+Focal loss에 이 $\mathcal{L}_{CMRL}$을 더해 학습합니다.

> [!EXAMPLE] 왜 삼방향인가 (통상 contrastive와의 차이)
> SimCLR/CLIP류는 (positive, negative) **두 방향**으로 표현을 학습합니다. CMRL은 (GT, noisy, refined) **세 마스크의 관계**를 다룹니다 — 특히 $\mathcal{L}_{self}$가 "아직 틀린 픽셀"을 "방금 고친 픽셀"로 끌어당겨, refinement라는 **틀림→맞음 전이**를 직접 모델링합니다. 통상 픽셀 loss엔 없는 자기개선 신호입니다.

## 아키텍처 & 학습

<dl class="kv">
<dt>백본</dt><dd>pretrained <b>SAM (ViT-H)</b> 기반. 인코더(637M)는 <b>얼리고</b> 가벼운 <b>decoder(4M)만 fine-tune</b>. [ZIM](#/resume/zim)은 SAM 계열, [ECLIPSE](#/resume/eclipse)는 Mask2Former 계열이지만, 모두 pretrained trunk의 parameter drift를 줄이고 작은 module로 적응한다는 상위 철학에서 연결됩니다.</dd>
<dt>학습</dt><dd>AdamW, lr $1\times10^{-4}$ + cosine, 10K iter, batch 16, 8×V100로 <b>10시간 미만</b>. AMP: Dice 목표, $N{=}10$, $\alpha_0{=}0.01$, $\epsilon{=}5\%$.</dd>
<dt>데이터/평가</dt><dd>LVIS로 학습(SegRefiner와 동일 셋업). instance seg는 COCO, fine-grained는 DIS5K/ThinObject-5K. 지표: AP·boundary-AP·IoU·Boundary-F.</dd>
</dl>

## 결과 (외워둘 숫자)

<dl class="kv">
<dt>약지도/준지도 pseudo-label refinement</dt><dd>논문의 해당 protocol에서 <b>최대 +16.1 AP<sup>mask</sup> point, +17.3 AP<sup>boundary</sup> point</b>를 보고합니다. 비교 범위와 baseline은 표의 dataset·label setting과 함께 인용합니다.</dd>
<dt>SOTA 모델 refinement</dt><dd>Mask R-CNN·SOLO·Mask2Former·ViTDet·MaskDINO 등 다양한 아키텍처의 출력을 일관되게 개선(예: Mask R-CNN R101 +6.5 AP).</dd>
<dt>Fine-grained (DIS)</dt><dd>얇은 구조/복잡 위상에서 SAMRefiner·SegRefiner 대비 <b>평균 IoU +11~21%</b>.</dd>
<dt>구성요소(4f)</dt><dd>AMP 단독 +4.6 AP<sup>1</sup>, CMRL 단독 +1.5, 둘 합치면 <b>+6.4</b>(개별 합보다 큰 <b>시너지</b> — 현실적 잡음이 삼방향 contrastive를 더 유의미하게 함).</dd>
<dt>"현실성" 검증(4e)</dt><dd>실제 UNet/ISNet 출력 오류로 학습하는 것보다 AMP 잡음이 <b>오히려 더 나음</b> — 오류 다양성이 넓고 강도를 통제할 수 있기 때문(functional substitutability 논증).</dd>
</dl>

## Q&A

<details class="qa"><summary>왜 형태학적 잡음(SegRefiner)이 부족한가? 한 문장으로.</summary>
<div class="qa-body">

**짧게:** dilation/erosion은 GT 둘레를 무작위·의미 무관하게 흔들 뿐, 실제 모델이 *어디서 왜* 틀리는지(복잡 경계, 유사 객체 혼동)를 못 담습니다.

**깊게:** 논문의 Pearson 상관 분석이 정량 증거입니다 — 형태학적 잡음은 이미지 edge/texture와의 상관이 0 근처에 좁게 몰려(의미 무관), adversarial 잡음은 [−0.6, 0.8]로 넓게 퍼집니다(양수=의미 구조 정렬, 음수=균질 영역). ablation(4d)에서 Phoenix를 형태학적 잡음으로 학습하면 성능이 급락하고, 반대로 SegRefiner에 AMP 잡음을 주면 크게 오릅니다 — 같은 잡음이 아키텍처와 무관하게 이득의 원인임을 보여줍니다.
</div></details>

<details class="qa"><summary>임베딩 공간에서 공격하는데 왜 픽셀 공격이 아닌가?</summary>
<div class="qa-body">

**짧게:** 효율 + 의미 수준 교란. 얼린 인코더의 image embedding을 재사용하고 4M decoder만 돌리므로 갱신 1회 ~6ms이고, 픽셀 노이즈가 아니라 고수준 의미를 흔들어 실제 혼동을 닮습니다.

**깊게:** Adversarial objective의 gradient는 현재 decoder loss를 가장 빠르게 키우는 local direction을 제공합니다. 이는 morphology-only noise보다 **model-aware failure pattern**을 만들 수 있다는 동기지만, gradient norm이 uncertainty의 calibrated estimate라는 뜻은 아닙니다. perturbation embedding $\mathbf{E}_p$는 noise generation에 사용되고 inference path에는 들어가지 않으므로 배포 비용은 늘리지 않습니다.
</div></details>

<details class="qa"><summary>CMRL의 self-improvement loss가 정확히 무엇을 하나?</summary>
<div class="qa-body">

**짧게:** 같은 이미지 안에서 "아직 틀린 픽셀($\mathcal{F}$)" feature를 "방금 고쳐낸 픽셀($\mathcal{S}$)" feature 쪽으로 끌어당겨, 틀림→맞음 전이를 자기 성공으로부터 부트스트래핑합니다.

**깊게:** 통상 contrastive는 정적 정답(positive/negative)에만 정렬하지만, $\mathcal{L}_{self}$는 refinement의 **동적 전이**를 모델링합니다. 이게 AMP와 시너지를 내는 이유: AMP가 만든 realistic한 실패 분포가 $\mathcal{S}$(고친 영역)와 $\mathcal{F}$(못 고친 영역)의 대비를 더 유의미하게 만들어, 개별 합(+4.6+1.5)보다 큰 결합 이득(+6.4)이 납니다.
</div></details>

<details class="qa"><summary>이 작업이 당신의 이전 연구와 어떻게 이어지나?</summary>
<div class="qa-body">

**짧게:** [PointWSSIS](#/resume/pointwssis-bestie)의 **MaskRefineNet**(거친 point-supervised proposal을 다듬던 모듈)의 정신적 후속이자 일반화입니다. "pseudo-label 품질이 약지도 학습의 병목"이라는 문제의식을 refiner 자체의 학습 데이터 생성으로 밀어붙인 것.

**깊게:** ZIM과 Phoenix는 SAM 계열 representation을 사용하고, ECLIPSE는 Mask2Former를 기반으로 합니다. 세 작업의 공통점은 특정 backbone 이름이 아니라 **pretrained representation을 보존하면서 작은 module로 적응**하는 설계 축입니다. Phoenix의 weak/semi-supervised refinement 평가는 PointWSSIS의 pseudo-label 품질 문제와 직접 맞닿습니다.
</div></details>

### 예상 follow-up

- *τ를 고정하지 않고 $\mathcal{U}(0.3,0.9)$로 뽑는 이유?* 쉬운~aggressive 잡음을 모두 커버해 다양한 실제 오류 강도에 견고. 높은 τ는 고품질 mask에, 낮은 τ는 저품질 mask에 유리하므로 균형을 위해 무작위화.
- *adversarial 목표로 Dice를 쓴 이유?* ablation(4b)에서 L1/MSE/BCE/Focal 대비 Dice가 최고 균형. 잡음 유형에 상대적으로 견고.
- *실제 모델 오류(4e)보다 AMP가 나은 게 이상하지 않나?* 실제 오류는 한 모델의 실패 모드에 편향. AMP는 오류 다양성이 넓고 강도를 통제할 수 있어 일반화가 더 좋음("realism의 상한을 넘는다").
- *한계는?* randomized perturbation 의존(다만 5회 반복에서 AP¹ 28.7±0.5로 안정), refiner 계열의 공통 한계(입력 mask가 완전히 틀리면 회복 어려움).

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| 문제 | 형태학적 잡음은 실제 segmentation 오류를 못 흉내 냄 |
| AMP | embedding-space FGSM으로 현재 decoder loss를 키우는 model-aware hard noise 생성 |
| 원리 | loss-increasing local direction을 사용; gradient 크기를 calibrated uncertainty로 해석하지 않음 |
| Guidance mask | Expansion/Contraction/Inversion으로 오류 유형 제어 |
| CMRL | GT·noisy·refined 삼방향 contrastive (intra/inter/**self**) |
| Self-loss | 실패 픽셀→성공 픽셀 부트스트래핑 (틀림→맞음 전이) |
| 백본 | SAM ViT-H, 인코더 freeze + 4M decoder만 fine-tune |
| 성과 | 논문 protocol에서 weak/semi refinement 최대 +16.1 AP point; DIS delta는 표의 metric 정의와 함께 인용 |
| 시너지 | AMP(+4.6)+CMRL(+1.5) → 결합 +6.4 (합보다 큼) |
| 계보 | PointWSSIS의 refinement 문제를 일반화; 서로 다른 backbone에서 freeze+small adaptation 철학과 연결 |

**Related:** [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [Segmentation](#/cv/segmentation) · [Weak & Semi-Supervised](#/cv/weak-semi-supervised) · [예상 질문 & 답변](#/resume/predicted-questions)
