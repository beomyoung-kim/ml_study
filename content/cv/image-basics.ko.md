# 이미지 기초: 픽셀 · 채널 · 텐서

> [!NOTE] 이 챕터의 목표
> 컴퓨터 비전(computer vision)을 처음 시작한다면 여기가 출발점입니다. "컴퓨터에게 이미지는 결국 **숫자 격자**일 뿐"이라는 한 문장을 그림과 짧은 코드로 확실히 잡습니다. 이 감각이 뒤따르는 모든 비전 챕터(분류 → CNN → detection → segmentation)의 바닥이 됩니다.

## 이미지는 숫자다

우리 눈에 사진은 풍경이지만, 컴퓨터에게는 **픽셀(pixel)** 이라는 작은 점들의 격자이고, 각 픽셀은 밝기를 나타내는 **숫자**입니다. 흑백(grayscale) 이미지에서 한 픽셀은 보통 `0`(검정)부터 `255`(흰색) 사이의 정수 하나입니다.

<figure>
<svg viewBox="0 0 560 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="90" y="18" text-anchor="middle" fill="#98a3b2">우리가 보는 것</text>
  <g>
    <rect x="30" y="30" width="120" height="120" fill="none" stroke="#98a3b2"/>
    <rect x="30" y="30" width="30" height="30" fill="#111"/><rect x="60" y="30" width="30" height="30" fill="#444"/><rect x="90" y="30" width="30" height="30" fill="#888"/><rect x="120" y="30" width="30" height="30" fill="#bbb"/>
    <rect x="30" y="60" width="30" height="30" fill="#333"/><rect x="60" y="60" width="30" height="30" fill="#666"/><rect x="90" y="60" width="30" height="30" fill="#aaa"/><rect x="120" y="60" width="30" height="30" fill="#ddd"/>
    <rect x="30" y="90" width="30" height="30" fill="#555"/><rect x="60" y="90" width="30" height="30" fill="#888"/><rect x="90" y="90" width="30" height="30" fill="#ccc"/><rect x="120" y="90" width="30" height="30" fill="#fff"/>
    <rect x="30" y="120" width="30" height="30" fill="#777"/><rect x="60" y="120" width="30" height="30" fill="#aaa"/><rect x="90" y="120" width="30" height="30" fill="#eee"/><rect x="120" y="120" width="30" height="30" fill="#fff"/>
  </g>
  <text x="230" y="95" font-size="22" fill="#e0533f">→</text>
  <text x="410" y="18" text-anchor="middle" fill="#98a3b2">컴퓨터가 보는 것 (숫자 격자)</text>
  <g font-size="12" fill="currentColor" text-anchor="middle">
    <rect x="290" y="30" width="240" height="120" fill="none" stroke="#0ea5e9"/>
    <text x="320" y="50">16</text><text x="380" y="50">68</text><text x="440" y="50">136</text><text x="500" y="50">187</text>
    <text x="320" y="80">51</text><text x="380" y="80">102</text><text x="440" y="80">170</text><text x="500" y="80">221</text>
    <text x="320" y="110">85</text><text x="380" y="110">136</text><text x="440" y="110">204</text><text x="500" y="110">255</text>
    <text x="320" y="140">119</text><text x="380" y="140">170</text><text x="440" y="140">238</text><text x="500" y="140">255</text>
  </g>
</svg>
<figcaption>4×4 흑백 이미지. 왼쪽은 사람이 보는 명암, 오른쪽은 컴퓨터가 다루는 실제 값(0=검정 … 255=흰색). 비전 모델은 오른쪽 숫자에서 패턴을 배웁니다.</figcaption>
</figure>

- **해상도(resolution)**: 격자의 크기. `1920×1080`이면 가로 1920, 세로 1080개의 픽셀.
- **비트 깊이(bit depth)**: 채널 하나가 표현할 수 있는 단계 수. 흔한 unsigned 8-bit는 $2^8=256$단계(0–255)지만, 카메라 RAW·의료 영상·HDR은 10/12/16-bit나 float일 수 있습니다.

## 컬러 = 채널(channel) 3장

컬러 이미지는 흑백 격자 **여러 장을 겹친 것**입니다. 가장 흔한 RGB는 **빨강(R)·초록(G)·파랑(B)** 세 장의 격자(=채널)로, 각 픽셀 색은 세 값의 조합입니다. 예: `(255,0,0)`=순수 빨강, `(255,255,255)`=흰색, `(0,0,0)`=검정.

<figure>
<svg viewBox="0 0 600 180" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="40" y="40" width="90" height="90" rx="6" fill="#e0533f" opacity="0.85"/>
  <rect x="60" y="30" width="90" height="90" rx="6" fill="#12a150" opacity="0.85"/>
  <rect x="80" y="20" width="90" height="90" rx="6" fill="#0ea5e9" opacity="0.85"/>
  <text x="120" y="150" text-anchor="middle" fill="#98a3b2">3채널이 겹쳐 컬러</text>
  <text x="250" y="80" font-size="22" fill="currentColor">=</text>
  <g text-anchor="middle">
    <rect x="300" y="45" width="70" height="70" rx="6" fill="none" stroke="#e0533f" stroke-width="2"/><text x="335" y="130" fill="#e0533f">R</text>
    <rect x="390" y="45" width="70" height="70" rx="6" fill="none" stroke="#12a150" stroke-width="2"/><text x="425" y="130" fill="#12a150">G</text>
    <rect x="480" y="45" width="70" height="70" rx="6" fill="none" stroke="#0ea5e9" stroke-width="2"/><text x="515" y="130" fill="#0ea5e9">B</text>
  </g>
  <text x="425" y="30" text-anchor="middle" fill="#98a3b2">각각 (H×W) 밝기 격자</text>
</svg>
<figcaption>RGB 이미지는 세 개의 (H×W) 격자를 쌓은 것. 채널을 나누면 각 채널은 다시 흑백 이미지 하나입니다.</figcaption>
</figure>

## 텐서로: (H, W, C) vs (C, H, W)

이미지를 코드로 다루면 [NumPy 프라이머](#/ml-coding/numpy-primer)에서 본 **텐서(tensor)** 가 됩니다. 여기서 초보자가 꼭 알아야 할 **레이아웃(layout) 차이**:

<dl class="kv">
<dt>NumPy / 이미지 라이브러리</dt><dd>보통 <b>(H, W, C)</b> — 높이, 너비, 채널 순. 예: <code>(1080, 1920, 3)</code>.</dd>
<dt>PyTorch 딥러닝</dt><dd>보통 <b>(C, H, W)</b> — 채널이 앞. 배치까지 붙으면 <b>(N, C, H, W)</b>. conv 연산이 채널 우선을 기대하기 때문.</dd>
</dl>

> [!WARNING] 실전 1순위 버그
> `(H,W,C)`와 `(C,H,W)`를 헷갈려 `transpose`를 빼먹는 것. 색이 깨지거나 shape 에러가 납니다. 경계마다 **shape·dtype·값 범위·채널 순서**를 확인하세요. NumPy의 HWC→CHW는 `img.transpose(2,0,1)`이지만, OpenCV 기본은 BGR이고 PIL은 보통 RGB라 단순 transpose만으로 색 순서가 고쳐지지는 않습니다.

## 정규화(normalization): 0–255 → 0–1

모델에 넣기 전, 픽셀 값을 작은 범위로 바꿉니다. 가장 단순하게는 **255로 나눠 0–1 범위**로 만들고, 보통은 여기서 한 발 더 나아가 채널별 **평균을 빼고 표준편차로 나눕니다**(mean/std 정규화). 이렇게 하면 학습이 더 안정적입니다 — 이유는 [Normalization & 학습 안정성](#/foundations/normalization-stability)에서 다룹니다.

$$
x_\text{norm} = \frac{x/255 - \mu}{\sigma}
$$

일부 ImageNet 사전학습 모델은 $\mu=[0.485,0.456,0.406]$, $\sigma=[0.229,0.224,0.225]$를 씁니다. 그러나 모델에 따라 입력 범위가 `[0,1]`/`[-1,1]`이고, resize interpolation·crop·색 순서까지 다릅니다. **가중치와 함께 배포된 preprocessing metadata/config를 정답으로 삼으세요.** 다른 전처리는 입력 분포를 바꿉니다.

## 왜 크기를 고정하나

배치 텐서는 보통 같은 높이·너비가 필요하고, 일부 분류 head나 positional embedding은 특정 학습 해상도를 가정합니다. 반면 fully-convolutional 모델과 많은 현대 backbone은 padding·adaptive pooling·position interpolation으로 여러 크기를 받을 수 있습니다. 어느 경우든 학습 때의 resize/crop 정책과 backbone의 stride(예: 32의 배수)를 확인하세요. 자세한 증강 기법은 [비전 데이터 증강](#/cv/augmentation)에서.

## 직접 해보기 — RGB를 흑백으로

아래는 gamma-encoded RGB에서 널리 쓰이는 BT.601 계열 luma 근사입니다: $Y' = 0.299R' + 0.587G' + 0.114B'$. 색공간과 linear/gamma convention이 바뀌면 계수와 의미도 달라집니다. 이 랩에서는 8-bit RGB `(H,W,3)`을 입력 계약으로 둡니다.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"rgb_to_gray","packages":["numpy"],"approx":true,"tol":0.01,"starter":"def rgb_to_gray(img):\n    # img: (H, W, 3) 중첩 리스트, 값 0~255. 반환: (H, W) 흑백.\n    # Y = 0.299*R + 0.587*G + 0.114*B  (채널 순서 R,G,B)\n    # 힌트: numpy 배열로 바꾸고 마지막 축(채널)에 가중합 -> img @ [0.299,0.587,0.114]\n    pass","tests":[{"args":[[[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]]],"expect":[[76.245,149.685],[29.07,255.0]]},{"args":[[[[0,0,0]]]],"expect":[[0.0]]},{"args":[[[[100,100,100]]]],"expect":[[100.0]]}],"solution":"import numpy as np\n\ndef rgb_to_gray(img):\n    a = np.asarray(img, dtype=float)\n    if a.ndim != 3 or a.shape[-1] != 3:\n        raise ValueError(\"img must have shape (H, W, 3)\")\n    if not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 255)):\n        raise ValueError(\"pixel values must be finite and in [0, 255]\")\n    w = np.array([0.299, 0.587, 0.114])\n    return (a @ w).tolist()"}
</script>
</div>

세 번째 테스트를 보세요: `(100,100,100)`은 가중치 합이 1이라 그대로 `100`. 회색은 R=G=B라는 사실과 일치합니다.

## Q&A

<details class="qa"><summary>왜 픽셀을 0–1로 정규화하나요? 그냥 0–255로 넣으면 안 되나요?</summary>
<div class="qa-body">

**짧게:** 큰 입력 값은 gradient를 불안정하게 만들고 학습을 느리게 합니다.

**깊게:** 0–255처럼 큰 값이 그대로 들어가면 초기 가중치와 곱해져 activation이 커지고, 학습률 설정이 까다로워집니다. 0–1(또는 mean/std) 정규화는 입력 스케일을 통일해 [conditioning](#/foundations/linear-algebra-calculus)을 개선하고, 사전학습 모델은 학습 때 쓴 것과 **같은** 정규화를 재현해야 성능이 나옵니다.
</div></details>

<details class="qa"><summary>알파 채널(투명도)이나 흑백 1채널은요?</summary>
<div class="qa-body">

**짧게:** 채널 수만 달라질 뿐 원리는 같습니다.

**깊게:** RGBA는 4채널(투명도 A 추가), 흑백은 1채널입니다. 모델의 첫 conv layer가 기대하는 입력 채널 수에 맞춰야 하므로, 흑백을 3채널로 복제하거나 첫 layer를 바꾸고 재학습할 수 있습니다. 알파는 무조건 버리기보다 straight/premultiplied alpha인지 확인해 배경과 합성해야 합니다. 투명 RGB에는 보이지 않는 임의 색이 남아 있을 수 있습니다.
</div></details>

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| 이미지 | 픽셀(숫자) 격자, 흑백은 0–255 |
| 채널 | 컬러 = R·G·B 세 격자를 겹친 것 |
| 레이아웃 | NumPy (H,W,C) · PyTorch (C,H,W) / 배치 (N,C,H,W) |
| 정규화 | 0–255 → 0–1, 보통 채널별 mean/std |
| 고정 크기 | 모델 입력 위해 resize/crop |

**다음:** [이미지 분류](#/cv/classification) · [NumPy 프라이머](#/ml-coding/numpy-primer) · [비전 데이터 증강](#/cv/augmentation)
