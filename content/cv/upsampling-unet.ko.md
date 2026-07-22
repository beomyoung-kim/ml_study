# 업샘플링 & Encoder–Decoder (U-Net)

> [!NOTE] 이 챕터의 목표
> Segmentation·matting처럼 **픽셀마다 답을 내는(dense prediction, 밀집 예측)** 작업은 특이한 문제를 만납니다. CNN은 정보를 압축하려고 해상도를 계속 **줄이는데**(downsampling), 최종 출력은 원본 크기의 픽셀 지도여야 합니다. 그래서 줄인 해상도를 **다시 키우는(upsampling, 업샘플링)** 기술과, 그걸 담는 **encoder–decoder(부호기–복호기)** 구조 = **U-Net**이 필요합니다. Segmentation/matting 챕터가 당연한 듯 쓰는 "pixel decoder", "stride-4 upsampler", "checkerboard artifact"가 전부 여기서 나옵니다.

## 왜 해상도를 되살려야 하나

이미지 분류는 "고양이"라는 **한 개의 답**만 내면 되니, 특징을 압축해 작은 벡터로 만들면 끝입니다. 하지만 segmentation은 **모든 픽셀**에 라벨을 붙여야 합니다 — 출력이 입력과 같은 $H\times W$ 크기여야 하죠.

문제는 CNN backbone(특징 추출망)이 pooling·stride로 해상도를 1/8, 1/16, 1/32로 **줄여** 놓는다는 것입니다. 이 작은 feature map을 다시 원본 크기로 **키워야** 픽셀 지도가 됩니다. 이 "키우기"가 업샘플링입니다.

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="90" y="20" text-anchor="middle" fill="#0ea5e9">입력 (큰 해상도)</text>
  <rect x="40" y="30" width="100" height="100" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.8"/>
  <path d="M150 80 H210" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="180" y="72" text-anchor="middle" fill="#98a3b2" font-size="11">↓ 압축</text>
  <text x="290" y="60" text-anchor="middle" fill="#6366f1">작은 feature</text>
  <rect x="255" y="70" width="40" height="40" rx="4" fill="#6366f1"/>
  <path d="M310 80 H370" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#ar)"/>
  <text x="340" y="72" text-anchor="middle" fill="#98a3b2" font-size="11">↑ 업샘플</text>
  <text x="470" y="20" text-anchor="middle" fill="#12a150">출력 (픽셀 지도)</text>
  <rect x="420" y="30" width="100" height="100" rx="4" fill="none" stroke="#12a150" stroke-width="1.8"/>
  <text x="560" y="84" fill="#98a3b2" font-size="11">= 입력과</text>
  <text x="560" y="100" fill="#98a3b2" font-size="11">  같은 크기</text>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>Dense prediction의 기본 형태: 줄였다가(encoder) 다시 키운다(decoder). 관건은 "키울 때 잃어버린 세밀한 위치 정보를 어떻게 되찾느냐"입니다.</figcaption>
</figure>

## 업샘플링 방법 세 가지

<dl class="kv">
<dt>Nearest-neighbor(최근접)</dt><dd>가장 단순 — 각 픽셀을 그대로 복제해 $2\times2$ 블록으로 확대. 빠르지만 계단처럼 각짐.</dd>
<dt>Bilinear(양선형) 보간</dt><dd>주변 픽셀들의 <b>가중 평균</b>으로 부드럽게 채웁니다. 학습 파라미터가 없고 resize+conv decoder에서 흔합니다. Framework의 <code>align_corners</code>·좌표 convention을 맞추지 않으면 픽셀 정렬이 달라집니다.</dd>
<dt>Transposed convolution(전치 합성곱)</dt><dd>Convolution을 행렬로 보았을 때 그 행렬의 <b>transpose</b>를 적용하는 학습 가능한 연산입니다. 흔히 "deconv"라 부르지만 원래 convolution의 역함수는 아닙니다. Kernel/stride의 uneven overlap은 <b>checkerboard artifact</b>를 만들 수 있어 resize+conv와 함께 비교합니다.</dd>
</dl>

<figure>
<svg viewBox="0 0 640 170" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <!-- source 2x2 -->
  <text x="70" y="20" text-anchor="middle" fill="#6366f1">2×2 (원본)</text>
  <g stroke="#6366f1" stroke-width="1.5" fill="none">
    <rect x="40" y="30" width="30" height="30"/><rect x="70" y="30" width="30" height="30"/>
    <rect x="40" y="60" width="30" height="30"/><rect x="70" y="60" width="30" height="30"/>
  </g>
  <text x="55" y="50" fill="currentColor">1</text><text x="85" y="50" fill="currentColor">2</text>
  <text x="55" y="80" fill="currentColor">3</text><text x="85" y="80" fill="currentColor">4</text>
  <path d="M115 60 H175" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#a2)"/>
  <text x="145" y="52" text-anchor="middle" fill="#98a3b2">nearest ×2</text>
  <!-- nearest 4x4 -->
  <g stroke="#12a150" stroke-width="1.2" fill="none">
    <rect x="200" y="20" width="20" height="20"/><rect x="220" y="20" width="20" height="20"/><rect x="240" y="20" width="20" height="20"/><rect x="260" y="20" width="20" height="20"/>
    <rect x="200" y="40" width="20" height="20"/><rect x="220" y="40" width="20" height="20"/><rect x="240" y="40" width="20" height="20"/><rect x="260" y="40" width="20" height="20"/>
    <rect x="200" y="60" width="20" height="20"/><rect x="220" y="60" width="20" height="20"/><rect x="240" y="60" width="20" height="20"/><rect x="260" y="60" width="20" height="20"/>
    <rect x="200" y="80" width="20" height="20"/><rect x="220" y="80" width="20" height="20"/><rect x="240" y="80" width="20" height="20"/><rect x="260" y="80" width="20" height="20"/>
  </g>
  <text x="240" y="120" text-anchor="middle" fill="#12a150" font-size="10">각 픽셀을 2×2로 복제 (계단)</text>
  <!-- bilinear note -->
  <rect x="360" y="30" width="240" height="70" rx="8" fill="none" stroke="#e0533f" stroke-width="1.4"/>
  <text x="480" y="55" text-anchor="middle" fill="#e0533f">bilinear = 주변 값의 가중 평균</text>
  <text x="480" y="78" text-anchor="middle" fill="#98a3b2" font-size="10">→ 계단 없이 매끄러운 그라데이션</text>
  <defs><marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>정수 배율에서 nearest는 픽셀을 복제하고, bilinear는 정해진 좌표 convention으로 주변 값을 보간합니다. Transposed conv는 학습 가능하지만 convolution의 역연산은 아니며 uneven overlap에 주의합니다.</figcaption>
</figure>

## Encoder–Decoder와 skip connection

단순히 줄였다 키우면 문제가 있습니다. **압축 과정에서 세밀한 위치·경계 정보가 사라져서**, 아무리 잘 키워도 흐릿한 경계밖에 못 얻습니다. 해결책이 **skip connection(건너뛰기 연결)** 입니다.

핵심 아이디어: encoder의 **초기(고해상도) feature**를 대응되는 decoder 단계로 전달해 concatenate/add합니다. Decoder가 깊은 semantic context와 얕은 spatial detail을 함께 쓰게 하지만, downsampling에서 잃은 원본 정보를 수학적으로 완전히 복원하는 것은 아닙니다. Crop/padding 때문에 feature 크기가 다르면 정렬 규칙도 명시해야 합니다.

> **PyTorch식 pseudocode — skip은 같은 spatial scale에서 합칩니다**

```python
skips = []
x = image                                      # [B,C,H,W]
for enc in encoder_blocks:
    x = enc(x); skips.append(x)
    x = downsample(x)

for dec, skip in zip(decoder_blocks, reversed(skips)):
    x = interpolate(x, size=skip.shape[-2:], mode="bilinear")
    x = torch.cat([x, skip], dim=1)            # H,W 일치; channel축 결합
    x = dec(x)
logits = output_head(x)                        # [B,num_classes,H,W]
```

<figure>
<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11">
  <!-- encoder (left, going down) -->
  <rect x="40"  y="30"  width="90" height="26" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="85" y="47" text-anchor="middle" fill="currentColor">enc 1 (H)</text>
  <rect x="60"  y="80"  width="70" height="26" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="95" y="97" text-anchor="middle" fill="currentColor">enc 2 (H/2)</text>
  <rect x="80"  y="130" width="50" height="26" rx="4" fill="none" stroke="#0ea5e9" stroke-width="1.6"/><text x="105" y="147" text-anchor="middle" fill="currentColor" font-size="10">enc 3</text>
  <!-- bottleneck -->
  <rect x="270" y="180" width="100" height="30" rx="6" fill="#6366f1"/><text x="320" y="200" text-anchor="middle" fill="#fff">bottleneck</text>
  <!-- decoder (right, going up) -->
  <rect x="510" y="130" width="50" height="26" rx="4" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="535" y="147" text-anchor="middle" fill="currentColor" font-size="10">dec 3</text>
  <rect x="510" y="80"  width="70" height="26" rx="4" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="545" y="97" text-anchor="middle" fill="currentColor">dec 2 (H/2)</text>
  <rect x="510" y="30"  width="90" height="26" rx="4" fill="none" stroke="#12a150" stroke-width="1.6"/><text x="555" y="47" text-anchor="middle" fill="currentColor">dec 1 (H)</text>
  <!-- down path -->
  <path d="M95 56 L110 80 M115 106 L120 130 M130 156 L280 185" stroke="#0ea5e9" stroke-width="1.4" fill="none" marker-end="url(#a3)"/>
  <!-- up path -->
  <path d="M370 185 L520 156 M545 130 L555 106 M545 80 L550 56" stroke="#12a150" stroke-width="1.4" fill="none" marker-end="url(#a3)"/>
  <!-- skip connections (dashed) -->
  <path d="M130 43 L510 43" stroke="#e0533f" stroke-width="1.6" stroke-dasharray="5 4"/>
  <path d="M130 93 L510 93" stroke="#e0533f" stroke-width="1.6" stroke-dasharray="5 4"/>
  <path d="M130 143 L510 143" stroke="#e0533f" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text x="320" y="35" text-anchor="middle" fill="#e0533f">skip connection (경계·위치 정보 직송)</text>
  <text x="90" y="225" fill="#0ea5e9">Encoder: 줄이며 "무엇" 파악</text>
  <text x="410" y="225" fill="#12a150">Decoder: 키우며 "어디" 복원</text>
  <defs><marker id="a3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>U-Net의 "U" 모양: 왼쪽 encoder가 내려가며 압축, 오른쪽 decoder가 올라가며 dense 출력을 만들고, 같은 scale끼리 <b>skip connection</b>(빨간 점선)으로 feature를 전달합니다. 의료영상·segmentation과 여러 diffusion U-Net에서 쓰이지만, diffusion backbone에는 DiT 같은 Transformer도 널리 쓰입니다.</figcaption>
</figure>

> [!TIP] 면접 한 줄
> "U-Net의 핵심은 **같은 scale의 skip connection**이다 — 깊은 feature의 *semantic*과 얕은 feature의 *spatial detail*을 합친다." Diffusion에서는 U-Net과 DiT가 모두 backbone 후보이며, diffusion/flow는 architecture 자체가 아니라 생성 objective·dynamics라는 점도 구분하세요([아키텍처](#/foundations/architectures)).

## 직접 구현 — nearest-neighbor 2× 업샘플링

가장 단순한 업샘플링을 코드로 확인해 봅시다. 각 픽셀을 `scale × scale` 블록으로 복제하면 됩니다. (힌트: `np.repeat`을 두 축에 각각.)

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"upsample_nearest","packages":["numpy"],"starter":"def upsample_nearest(x, scale=2):\n    # x: 2D 리스트. 각 원소를 scale×scale 블록으로 복제해 확대한 2D 리스트를 반환.\n    # 예) [[1,2],[3,4]], scale=2 -> [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]\n    pass","tests":[{"args":[[[1,2],[3,4]],2],"expect":[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]},{"args":[[[5]],3],"expect":[[5,5,5],[5,5,5],[5,5,5]]},{"args":[[[1,2,3]],2],"expect":[[1,1,2,2,3,3],[1,1,2,2,3,3]]},{"args":[[[1,2],[3,4]],1],"expect":[[1,2],[3,4]]}],"solution":"import numpy as np\n\ndef upsample_nearest(x, scale=2):\n    a = np.asarray(x)\n    if a.ndim != 2 or 0 in a.shape:\n        raise ValueError(\"x must be a non-empty 2D array\")\n    if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or scale <= 0:\n        raise ValueError(\"scale must be a positive integer\")\n    a = np.repeat(a, int(scale), axis=0)\n    a = np.repeat(a, int(scale), axis=1)\n    return a.tolist()"}
</script>
</div>

Bilinear은 여기서 복제 대신 주변 값을 거리 가중 평균하는 것으로 확장되고, transposed conv는 이 확대 규칙 자체를 학습 가능한 kernel로 대체합니다.

## Q&A

<details class="qa"><summary>Transposed convolution의 checkerboard artifact는 왜 생기고 어떻게 피하나요?</summary>
<div class="qa-body">

**짧게:** kernel 크기가 stride로 나누어떨어지지 않을 때 생기는 uneven overlap은 출력 위치별 기여 수를 달리해 격자 artifact를 만들기 쉽습니다.

**깊게:** 예를 들어 kernel 3, stride 2면 출력 위치별 overlap count가 달라질 수 있습니다. 완화법은 (1) resize 후 일반 conv, (2) kernel을 stride의 배수로 두고 초기화·학습을 점검, (3) PixelShuffle 뒤 artifact 검사입니다. Kernel divisibility만으로 artifact-free를 보장하지 않고, bilinear resize도 blur·alignment trade-off가 있으므로 validation과 시각 검사를 병행합니다.
</div></details>

<details class="qa"><summary>skip connection이 없으면 무엇이 나빠지나요?</summary>
<div class="qa-body">

**짧게:** 경계가 흐려지고 작은 객체를 놓칩니다.

**깊게:** bottleneck의 저해상도 feature만 쓰면 세밀한 위치 정보가 제한됩니다. Skip은 encoder의 고해상도 feature를 넘겨 decoder가 경계 단서를 활용하게 합니다. 다만 얕은 feature의 noise·domain bias도 함께 전달될 수 있고, 성능은 loss·resolution·annotation 품질에도 의존합니다.
</div></details>

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| 왜 업샘플? | dense prediction 출력은 입력과 같은 크기여야 함 |
| nearest | 픽셀 복제 — 빠르지만 각짐 |
| bilinear | 주변 가중 평균 — 매끄럽고 파라미터 없음 (기본) |
| transposed conv | convolution matrix의 transpose인 학습 연산; inverse가 아니며 uneven overlap 주의 |
| encoder–decoder | 줄였다(무엇) 키운다(어디) |
| skip connection | 고해상도 경계 정보를 decoder로 직송 = U-Net 핵심 |
| checkerboard 회피 | bilinear 업샘플 + 일반 conv |

**다음:** [비전 데이터 증강](#/cv/augmentation) · [Segmentation](#/cv/segmentation) · [Conv & Pooling (밑바닥)](#/ml-coding/conv-pooling)
