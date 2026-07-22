# 백본 & 전이학습 (ResNet → ViT)

> [!NOTE] 이 챕터의 목표
> 라벨·연산 예산이 제한된 많은 비전 프로젝트는 **처음부터 학습하기보다** 사전학습된 **백본(backbone, 특징 추출망)** 을 가져와 task head를 붙입니다. 다만 데이터가 매우 많거나 도메인·입력 채널이 크게 다르면 scratch training도 비교 대상입니다. 이 "**미리 학습된 특징을 재사용한다**"는 아이디어가 detection·segmentation·foundation model·continual learning까지 이어지는 뼈대입니다.

## 백본이란 무엇인가

[비전을 위한 CNN](#/cv/cnns-for-vision)에서 봤듯, 신경망의 앞쪽 층들은 이미지 표현을 계층적으로 뽑아냅니다. 이 **특징 추출 부분**을 백본이라 부릅니다. 같은 백본 가중치를 여러 task에 전이할 수 있지만 feature가 완전히 task-agnostic인 것은 아닙니다. Pretraining data·objective와 필요한 spatial resolution에 따라 전이 품질이 달라지고, 그 위의 **머리(head)** 또는 decoder도 task에 맞춰 설계합니다.

<figure>
<svg viewBox="0 0 640 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="20" y="80" width="70" height="46" rx="6" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="55" y="107" text-anchor="middle" fill="currentColor">이미지</text>
  <rect x="120" y="70" width="200" height="66" rx="8" fill="#6366f1"/>
  <text x="220" y="98" text-anchor="middle" fill="#fff" font-weight="700">백본 (공유)</text>
  <text x="220" y="118" text-anchor="middle" fill="#fff" font-size="11">사전학습된 특징 추출망</text>
  <path d="M90 103 H120" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#ar)"/>
  <path d="M320 103 H370" stroke="#98a3b2" stroke-width="1.5"/>
  <rect x="380" y="20" width="150" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="455" y="42" text-anchor="middle" fill="#12a150">분류 head → 고양이</text>
  <rect x="380" y="86" width="150" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="455" y="108" text-anchor="middle" fill="#12a150">검출 head → 박스</text>
  <rect x="380" y="152" width="150" height="34" rx="6" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="455" y="174" text-anchor="middle" fill="#12a150">분할 head → 마스크</text>
  <path d="M370 103 C 375 40, 375 40, 380 37" stroke="#98a3b2" stroke-width="1.3" fill="none"/>
  <path d="M370 103 H380" stroke="#98a3b2" stroke-width="1.3"/>
  <path d="M370 103 C 375 166, 375 166, 380 169" stroke="#98a3b2" stroke-width="1.3" fill="none"/>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>하나의 백본 가중치나 feature를 여러 task의 head에 공유할 수 있습니다. 같은 입력의 여러 head를 동시에 실행하면 backbone feature를 한 번 계산할 수 있지만, 서로 다른 입력·서비스에서는 매 요청마다 backbone forward가 필요합니다.</figcaption>
</figure>

## ResNet — 아주 깊은 망을 학습 가능하게

Plain network는 깊이를 늘렸을 때 더 작은 training error조차 얻지 못하는 **degradation problem**을 보였습니다. 이는 단순한 과대적합이 아니라 최적화 문제입니다. **ResNet**의 잔차 연결 $y=\mathcal{F}(x)+x$은 identity를 쉽게 표현하고 gradient에 identity 항을 추가해 매우 깊은 망의 최적화를 돕습니다. 다만 gradient가 "손실 없이" 흐른다고 보장하지 않으며 normalization·초기화·optimizer도 중요합니다. 자세한 유도는 canonical owner인 [CNN · RNN · Transformer](#/foundations/architectures) 참고.

## ViT — 이미지를 토큰으로 쪼개 Transformer에

**Vision Transformer(ViT)** 는 CNN 대신 [Transformer](#/foundations/architectures)를 씁니다: 이미지를 **패치(patch, 예: 16×16)** 로 잘라 각각을 토큰으로 만들고, 위치 정보를 더한 뒤 문장의 단어들처럼 [self-attention](#/ml-coding/attention)으로 서로 참고하게 합니다. patch→token tokenization의 그림과 상세는 canonical owner인 [CNN · RNN · Transformer](#/foundations/architectures) 참고(이 "이미지→토큰" 발상은 [VLM](#/vlm/vlm-101)에서도 그대로 쓰입니다).

<div class="proscons"><div><div class="pros-t">CNN (예: ResNet)</div>

- **귀납 편향(inductive bias)** 내장: 지역성·가중치 공유·평행이동 등변성 → **적은 데이터에서 유리할 수 있음**
- 작은/중간 데이터셋, on-device에 강함

</div><div><div class="cons-t">ViT</div>

- CNN보다 locality bias가 약한 기본 ViT는 scratch 학습에 더 많은 데이터·regularization이 필요할 수 있고, 대규모 사전학습에서 강한 성능을 보임
- 전역 관계를 유연하게 포착, 멀티모달로 확장 쉬움

</div></div>

> [!TIP] 면접 한 줄
> "CNN은 locality bias 덕분에 작은 데이터·edge 예산에서 좋은 출발점이고, ViT는 대규모 사전학습과 전역 mixing에서 강합니다. 승자는 데이터·해상도·latency·checkpoint에 따라 실측합니다." Swin은 windowed hierarchical Transformer, ConvNeXt는 현대화된 CNN, Hiera는 hierarchical Transformer이므로 모두를 단순히 같은 "hybrid"로 묶지는 마세요.

## 전이학습 — 미리 배운 것을 재사용

**전이학습(transfer learning)** 은 대규모 데이터(예: ImageNet, 또는 [자기지도학습](#/cv/self-supervised))로 **사전학습(pretraining)** 된 백본을, 내 작은 데이터셋 task로 옮겨 쓰는 것입니다. 두 가지 방식:

<dl class="kv">
<dt>특징 추출 (feature extraction / freeze)</dt><dd>백본을 <b>얼려(freeze)</b> 그대로 두고, 새로 붙인 head만 학습. 데이터가 <b>적을 때</b>, 빠르고 과적합 위험이 낮음.</dd>
<dt>미세조정 (fine-tuning)</dt><dd>백본까지 <b>함께(또는 뒷부분만) 학습</b>. 데이터가 <b>충분하고</b> 도메인이 사전학습과 다를 때 성능↑. 보통 <b>작은 learning rate</b>로.</dd>
</dl>

> [!WARNING] Freeze의 두 층
> <code>requires_grad_(False)</code>는 parameter update를 막지만 <code>model.train()</code> 상태의 BatchNorm running statistics와 Dropout 동작까지 멈추지는 않습니다. 고정 feature extractor라면 backbone을 <code>eval()</code>로 두고 head만 train mode로 관리할지 명시하세요. 반대로 BN statistics를 새 도메인에 적응시키는 전략도 있으므로 의도적으로 선택해야 합니다.

> **PyTorch식 pseudocode — 완전히 고정한 feature extractor**

```python
backbone.requires_grad_(False)
backbone.eval()                             # BN running stats와 Dropout도 고정
head.train()

for images, labels in train_loader:
    with torch.no_grad():                   # backbone graph를 저장하지 않음
        features = backbone(images)         # [B,C,H,W] 또는 [B,T,D]
    logits = head(pool(features))           # head에는 gradient가 흐름
    loss = criterion(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward(); optimizer.step()       # optimizer에는 head parameter만
```

<figure>
<svg viewBox="0 0 620 130" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="90" y="22" text-anchor="middle" fill="#98a3b2">① 사전학습</text>
  <rect x="30" y="35" width="120" height="40" rx="6" fill="#6366f1"/><text x="90" y="60" text-anchor="middle" fill="#fff" font-size="11">백본 (대규모 학습)</text>
  <path d="M155 55 H195" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#a5)"/>
  <text x="300" y="22" text-anchor="middle" fill="#98a3b2">② 전이 (freeze + 새 head)</text>
  <rect x="205" y="35" width="120" height="40" rx="6" fill="#6366f1" opacity="0.55"/><text x="265" y="55" text-anchor="middle" fill="#fff" font-size="10">백본 (얼림 ❄)</text>
  <text x="265" y="68" text-anchor="middle" fill="#fff" font-size="9">가중치 고정</text>
  <rect x="335" y="35" width="90" height="40" rx="6" fill="#12a150"/><text x="380" y="55" text-anchor="middle" fill="#fff" font-size="10">새 head</text>
  <text x="380" y="68" text-anchor="middle" fill="#fff" font-size="9">이것만 학습</text>
  <path d="M425 55 H455" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#a5)"/>
  <text x="540" y="59" text-anchor="middle" fill="currentColor">내 task 예측</text>
  <text x="310" y="110" text-anchor="middle" fill="#98a3b2">데이터 적음 → freeze · 데이터 많고 도메인 다름 → fine-tune (작은 LR)</text>
  <defs><marker id="a5" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>전이학습: 무거운 백본은 재사용(얼리거나 살짝 조정)하고, 가벼운 head만 내 데이터로 학습 → 데이터·시간·연산을 크게 절약.</figcaption>
</figure>

## 직접 해보기 — 백본 특징을 벡터로 (global average pooling)

백본의 마지막 특징 맵 `(C, H, W)` 을 head에 넣으려면 보통 공간 차원을 평균 내어 **채널당 하나의 값** `(C,)` 으로 요약합니다(global average pooling). head는 이 벡터를 받아 분류합니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"global_avg_pool","packages":["numpy"],"approx":true,"starter":"def global_avg_pool(fmap):\n    # fmap: (C, H, W) 특징 맵. 각 채널(C)에 대해 H×W 공간을 평균내어 (C,) 벡터를 리스트로 반환.\n    pass","tests":[{"args":[[[[1,3],[3,1]]]],"expect":[2.0]},{"args":[[[[1,1],[1,1]],[[2,4],[4,2]]]],"expect":[1.0,3.0]},{"args":[[[[0,0,0],[6,6,6]]]],"expect":[3.0]}],"solution":"import numpy as np\n\ndef global_avg_pool(fmap):\n    x = np.asarray(fmap, dtype=float)\n    if x.ndim != 3 or any(size == 0 for size in x.shape):\n        raise ValueError(\"fmap must be a non-empty (C,H,W) array\")\n    if not np.all(np.isfinite(x)):\n        raise ValueError(\"fmap must contain finite values\")\n    return x.mean(axis=(1, 2)).tolist()"}
</script>
</div>

백본이 frozen이고 `eval()`이며 입력 전처리도 결정적이라면 이 `global_avg_pool`까지의 feature를 **미리 계산해 캐시**할 수 있습니다. Random augmentation을 매 epoch 바꾸거나 backbone의 BN/Dropout 상태가 변하거나 spatial feature가 필요한 task라면 같은 캐시를 재사용할 수 없습니다. 저장 공간·I/O와 연산 절약도 함께 비교하세요.

## Q&A

<details class="qa"><summary>내 데이터가 아주 적으면 freeze, 많으면 fine-tune — 왜죠?</summary>
<div class="qa-body">

**짧게:** 데이터가 적으면 백본까지 학습시킬 때 과적합·망각 위험이 크기 때문입니다.

**깊게:** 백본은 수백만~수억 파라미터입니다. 라벨이 수백 장뿐인데 전부 학습시키면 그 적은 데이터에 과적합하고, 사전학습으로 얻은 좋은 일반 특징을 망가뜨립니다(catastrophic forgetting의 사촌). 그래서 데이터가 적으면 백본을 얼리고 head만, 데이터가 많고 도메인이 다르면(의료·위성 등) 작은 LR로 fine-tune합니다. 중간 전략으로 **뒷부분 층만** 푸는 것도 흔합니다.
</div></details>

<details class="qa"><summary>ImageNet 사전학습 vs 자기지도 사전학습, 뭐가 좋나요?</summary>
<div class="qa-body">

**짧게:** 둘 다 후보입니다. 자기지도 백본은 라벨 없이 큰 데이터와 dense 전이에 강할 수 있지만, 동일 architecture·data·compute에서 내 task의 transfer score를 비교해야 합니다.

**깊게:** ImageNet 지도 사전학습은 강하고 재현하기 쉬운 기준선이지만 1000-class taxonomy에 목적함수가 맞춰집니다. 자기지도학습은 사람이 붙인 class label 없이 더 큰 데이터에서 표현을 배울 수 있지만 data curation·compute가 사라지는 것은 아니며 모든 downstream task에서 우월하지 않습니다. Linear probe, full/partial fine-tune, frozen dense feature를 같은 protocol로 비교하세요 — [자기지도학습 입문](#/cv/self-supervised)과 [Vision Foundation Models](#/cv/foundation-models)로 이어집니다.
</div></details>

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| 백본(backbone) | task 간 공유되는 사전학습 특징 추출망 |
| head | task별로 백본 위에 얹는 가벼운 예측기 |
| ResNet | 잔차 연결($y=F(x)+x$)로 identity 표현과 깊은 망 최적화를 도움 |
| ViT | 이미지→패치→토큰→Transformer; 사전학습·해상도·latency와 함께 평가 |
| CNN vs ViT | 귀납 편향(적은 데이터) vs 유연함(큰 데이터) |
| freeze | 백본 고정 + head만 학습; 데이터 적을 때 |
| fine-tune | 백본까지 학습(작은 LR); 데이터 많고 도메인 다를 때 |

**다음:** [업샘플링 & U-Net](#/cv/upsampling-unet) · [자기지도학습 입문](#/cv/self-supervised) · [CNN·RNN·Transformer](#/foundations/architectures)
