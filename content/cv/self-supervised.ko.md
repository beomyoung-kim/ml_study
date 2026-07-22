# 자기지도학습 입문 (Self-Supervised Learning)

> [!NOTE] 이 챕터의 목표
> 사람이 붙인 task label 없이 특징(feature)을 배우는 **자기지도학습(self-supervised learning, SSL)** 의 큰 그림을 잡습니다. "데이터의 구조로 학습 target을 만든다"는 발상이 핵심입니다. DINO·MAE는 대표적 vision SSL이고, CLIP은 웹의 image–text pair를 감독 신호로 쓰는 **자연어 감독/약지도 multimodal pretraining**이라 엄밀한 label-free SSL과 구분합니다.

## 무엇을 · 왜

[이미지 분류](#/cv/classification)나 [detection](#/cv/detection)은 사람이 붙인 **라벨(label)** 로 배웁니다. Dense annotation은 특히 비쌉니다. 라벨 없는 데이터는 훨씬 많지만 공짜·무한은 아닙니다. 수집·중복 제거·품질·저작권·개인정보·유해 콘텐츠·storage/compute 비용이 학습 품질과 배포 가능성을 좌우합니다.

자기지도학습은 이 **비라벨 데이터**를 씁니다. 사람 라벨 대신, **데이터 자체에서 문제(pretext task, 구실 과제)를 자동으로 만들어** 풉니다. 예를 들어 "이미지의 일부를 가리고 원래대로 복원하기"처럼요. 이 문제를 잘 풀려면 모델은 "고양이의 생김새" 같은 유용한 표현을 스스로 익혀야 하고, 그렇게 학습된 특징은 이후 적은 라벨만으로 분류·검출·분할에 **전이(transfer)** 됩니다 — [전이학습](#/cv/backbones-transfer) 참고.

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <rect x="30" y="45" width="120" height="60" rx="8" fill="none" stroke="#0ea5e9" stroke-width="1.6"/>
  <text x="90" y="35" text-anchor="middle" fill="#0ea5e9">라벨 없는 대량 데이터</text>
  <text x="90" y="80" text-anchor="middle" fill="currentColor">🖼️ 🖼️ 🖼️ …</text>
  <path d="M150 75 H210" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#s1)"/>
  <rect x="215" y="45" width="150" height="60" rx="8" fill="#6366f1"/>
  <text x="290" y="70" text-anchor="middle" fill="#fff" font-size="11">pretext task 자동 생성</text>
  <text x="290" y="90" text-anchor="middle" fill="#fff" font-size="11">(가리기/뷰 비교 …)</text>
  <path d="M365 75 H425" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#s1)"/>
  <rect x="430" y="45" width="120" height="60" rx="8" fill="none" stroke="#12a150" stroke-width="1.8"/>
  <text x="490" y="70" text-anchor="middle" fill="#12a150">유용한 특징</text>
  <text x="490" y="90" text-anchor="middle" fill="#98a3b2" font-size="11">→ 적은 라벨로 전이</text>
  <defs><marker id="s1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>자기지도학습: 사람 라벨 없이, 데이터로 문제를 만들어 풀며 특징을 배우고, 그 특징을 downstream 과제로 전이합니다.</figcaption>
</figure>

## 두 가지 큰 흐름

### 1. Contrastive learning (대조 학습)

핵심 직관: **같은 이미지를 두 번 다르게 augment(증강)한 두 뷰는 "같은 것"이니 임베딩 공간에서 가깝게, 서로 다른 이미지는 멀게** 만듭니다. 모델은 "무엇이 본질이고 무엇이 우연한 변형(자르기·색 변화)인지"를 배우게 됩니다.

<figure>
<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <circle cx="320" cy="100" r="88" fill="none" stroke="#98a3b2" stroke-width="1" stroke-dasharray="4 4"/>
  <text x="320" y="24" text-anchor="middle" fill="#98a3b2">임베딩 공간</text>
  <circle cx="300" cy="95" r="7" fill="#e0533f"/><text x="300" y="84" text-anchor="middle" fill="#e0533f">뷰A</text>
  <circle cx="345" cy="110" r="7" fill="#e0533f"/><text x="360" y="114" fill="#e0533f">뷰B</text>
  <path d="M307 97 L338 108" stroke="#12a150" stroke-width="2" marker-end="url(#p1)"/>
  <text x="322" y="138" text-anchor="middle" fill="#12a150" font-size="11">당겨서 가깝게 (positive)</text>
  <circle cx="150" cy="60" r="7" fill="#0ea5e9"/><circle cx="180" cy="160" r="7" fill="#0ea5e9"/><circle cx="480" cy="70" r="7" fill="#0ea5e9"/><circle cx="470" cy="150" r="7" fill="#0ea5e9"/>
  <path d="M295 92 L165 63" stroke="#e0533f" stroke-width="1.3" stroke-dasharray="3 3" marker-end="url(#p2)"/>
  <path d="M348 112 L465 145" stroke="#e0533f" stroke-width="1.3" stroke-dasharray="3 3" marker-end="url(#p2)"/>
  <text x="150" y="185" text-anchor="middle" fill="#0ea5e9" font-size="11">다른 이미지 = 밀어냄 (negative)</text>
  <defs>
    <marker id="p1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#12a150"/></marker>
    <marker id="p2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#e0533f"/></marker>
  </defs>
</svg>
<figcaption>대조 학습: 한 이미지의 두 augment 뷰(빨강)는 서로 당기고(positive), 다른 이미지들(파랑)은 밀어냅니다(negative). 이 밀당으로 의미 있는 임베딩이 만들어집니다.</figcaption>
</figure>

<dl class="kv">
<dt>SimCLR</dt><dd>큰 배치 안의 다른 이미지들을 negative로 사용. augment 조합(crop+color)이 성능을 좌우합니다.</dd>
<dt>MoCo</dt><dd>momentum encoder + 큐(queue)로 negative를 많이 확보 — 작은 배치로도 대조 학습 가능.</dd>
<dt>CLIP</dt><dd>대조 목적을 <b>이미지↔텍스트</b> pair에 적용해 matched pair를 당기고 batch의 나머지 조합을 negative로 둡니다. 텍스트 pair 자체가 외부 감독 신호이므로 좁은 의미의 vision-only SSL보다는 자연어 감독/약지도에 가깝습니다. <a href="#/vlm/pretraining">VLM Pretraining</a>·<a href="#/vlm/vlm-101">VLM 101</a> 참고.</dd>
</dl>

손실은 보통 **InfoNCE**: 한 anchor에 대해 여러 후보 중 positive를 고르는 softmax 분류 문제로 봅니다. 유사도는 대개 **cosine similarity(코사인 유사도)** 로 잽니다 — 아래에서 직접 구현해 봅니다.

> **PyTorch식 pseudocode — batch 안의 대각선이 positive**

```python
view1, view2 = augment(images), augment(images)     # 각각 [B,C,H,W]
z1 = normalize(projector(encoder(view1)), dim=-1)  # [B,D]
z2 = normalize(projector(encoder(view2)), dim=-1)  # [B,D]

logits = z1 @ z2.T / temperature                    # [B,B]
positive = torch.arange(B, device=logits.device)    # (i,i)가 같은 원본
loss = 0.5 * (cross_entropy(logits, positive)
              + cross_entropy(logits.T, positive))
loss.backward()                                     # 두 view 경로 모두 gradient
```

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"cosine_sim","packages":["numpy"],"approx":true,"starter":"def cosine_sim(a, b):\n    # 두 벡터의 코사인 유사도 = (a·b) / (|a| |b|). 1에 가까울수록 같은 방향(비슷함).\n    # shape가 같아야 하고 zero vector의 cosine은 정의되지 않습니다.\n    pass","tests":[{"args":[[1,0],[1,0]],"expect":1.0},{"args":[[1,0],[0,1]],"expect":0.0},{"args":[[1,2,3],[2,4,6]],"expect":1.0},{"args":[[1,0],[-1,0]],"expect":-1.0}],"solution":"import numpy as np\n\ndef cosine_sim(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:\n        raise ValueError(\"a and b must be non-empty vectors with the same shape\")\n    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):\n        raise ValueError(\"vectors must be finite\")\n    denom = np.linalg.norm(a) * np.linalg.norm(b)\n    if denom == 0:\n        raise ValueError(\"cosine similarity is undefined for a zero vector\")\n    return float(np.clip((a @ b) / denom, -1.0, 1.0))"}
</script>
</div>

> [!WARNING] Collapse와 false negative
> 단순한 positive-pair 일치 loss만 최소화하면 모든 입력을 같은 벡터로 만드는 collapse가 해가 될 수 있습니다. 그러나 “negative가 없으면 반드시 collapse”는 아닙니다. BYOL/SimSiam은 stop-gradient·predictor와 optimization dynamics를, DINO는 teacher·centering/sharpening을 사용해 non-contrastive하게 이를 피합니다. 반대로 batch의 모든 다른 이미지를 negative로 두면 같은 semantic class까지 밀어내는 **false negative**가 생길 수 있습니다.

### 2. Masked Image Modeling (가리고 복원하기)

NLP의 "빈칸 채우기"를 이미지로 옮긴 것. 이미지를 patch(조각)로 나눈 뒤 **일부를 가리고(mask), 가려진 부분을 복원**하도록 학습합니다. 대표가 **MAE(Masked Autoencoder)** 로, patch의 75%를 가려도 복원할 만큼 강한 표현을 배웁니다.

<figure>
<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="70" y="20" text-anchor="middle" fill="#98a3b2">일부 patch 가림</text>
  <g stroke="#98a3b2" stroke-width="1">
    <rect x="20" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.5"/><rect x="50" y="30" width="30" height="30" fill="#3a3f4b"/><rect x="80" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.5"/>
    <rect x="20" y="60" width="30" height="30" fill="#3a3f4b"/><rect x="50" y="60" width="30" height="30" fill="#0ea5e9" opacity="0.5"/><rect x="80" y="60" width="30" height="30" fill="#3a3f4b"/>
    <rect x="20" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.5"/><rect x="50" y="90" width="30" height="30" fill="#3a3f4b"/><rect x="80" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.5"/>
  </g>
  <path d="M120 75 H190" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#m1)"/>
  <rect x="195" y="55" width="130" height="40" rx="8" fill="#6366f1"/><text x="260" y="80" text-anchor="middle" fill="#fff" font-size="11">encoder → decoder</text>
  <path d="M325 75 H395" stroke="#98a3b2" stroke-width="1.5" marker-end="url(#m1)"/>
  <text x="470" y="20" text-anchor="middle" fill="#12a150">복원</text>
  <g stroke="#98a3b2" stroke-width="1">
    <rect x="420" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="450" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="480" y="30" width="30" height="30" fill="#0ea5e9" opacity="0.6"/>
    <rect x="420" y="60" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="450" y="60" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="480" y="60" width="30" height="30" fill="#0ea5e9" opacity="0.6"/>
    <rect x="420" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="450" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.6"/><rect x="480" y="90" width="30" height="30" fill="#0ea5e9" opacity="0.6"/>
  </g>
  <text x="545" y="79" fill="#12a150" font-size="11">≈ 원본</text>
  <defs><marker id="m1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>MAE: patch의 대부분을 가리고(회색) 복원(파랑)하도록 학습. 복원을 잘하려면 장면의 구조를 이해해야 하므로, 강한 특징이 자연스레 만들어집니다.</figcaption>
</figure>

<div class="proscons"><div><div class="pros-t">Contrastive</div>

- 이미지 수준(전역) 표현이 강함 → 분류/검색에 강점
- augment·negative 설계가 성능을 좌우
- CLIP처럼 멀티모달로 확장 쉬움

</div><div><div class="cons-t">Masked (MIM)</div>

- patch 단위 target을 써 dense feature에 유리할 수 있음
- augment/negative 설계 부담이 적음
- 복원 목표가 지나치게 저수준일 수 있음

</div></div>

### 3. 자기증류 (self-distillation) — 한 걸음 더

**DINO** 계열은 negative도, 복원도 없이, **teacher(EMA로 천천히 갱신)와 student가 같은 이미지의 다른 뷰에서 서로의 출력을 맞추도록** 학습합니다. collapse는 centering+sharpening으로 막습니다. 그 결과 라벨 없이도 물체 경계에 반응하는 attention이 emergent하게 나타납니다. 자세한 학습 메커니즘은 [Vision Foundation Models](#/cv/foundation-models)에서 다룹니다.

## Q&A

<details class="qa"><summary>self-supervised와 unsupervised는 같은 말인가요?</summary>
<div class="qa-body">

**짧게:** 넓게 보면 SSL은 비지도학습의 한 형태이지만, "정답을 데이터에서 자동 생성해 지도학습처럼 푼다"는 점이 특징입니다.

**깊게:** Clustering은 군집 내 거리, PCA는 reconstruction/분산 최대화처럼 **명시적 목적함수**가 있지만 사람이 붙인 target label이 없습니다. SSL은 가리기·다른 view 맞추기처럼 입력 자체에서 target 또는 positive relation을 구성해 supervised learning과 비슷한 loss/gradient machinery를 씁니다. 모든 SSL target을 고정된 “pseudo-label”로 부를 필요는 없습니다.
</div></details>

<details class="qa"><summary>contrastive vs masked, 언제 뭘 쓰나요?</summary>
<div class="qa-body">

**짧게:** Contrastive는 global alignment에, masked modeling은 patch-level 학습 신호에 자연스럽지만 downstream 우열은 architecture·pretraining data·fine-tuning protocol로 비교해야 합니다.

**깊게:** Global pooled embedding을 대비시키는 설정은 retrieval·classification과 잘 맞고, MIM은 patch마다 target을 줍니다. 그렇다고 pixel reconstruction이 자동으로 semantic dense feature를 보장하거나 contrastive가 dense task에 약한 것은 아닙니다. Linear probe만 보지 말고 frozen dense evaluation과 full fine-tuning을 함께 비교하세요 — [Foundation Models](#/cv/foundation-models).
</div></details>

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| SSL 핵심 | 라벨 없이, 데이터로 문제를 만들어 특징 학습 |
| Contrastive | 같은 이미지 두 뷰는 당기고 다른 건 밀기 (SimCLR/MoCo/CLIP) |
| InfoNCE | 여러 후보 중 positive 고르는 softmax; 유사도는 cosine |
| collapse | negative/장치 없으면 모든 걸 같은 벡터로 뭉갬 → 방지 필요 |
| Masked (MAE) | patch를 가리고 target을 복원; target/decoder 설계가 표현을 좌우 |
| Self-distillation | teacher(EMA)-student 뷰 일치 (DINO), 라벨·negative 불필요 |

**다음:** [객체 검출 (Detection)](#/cv/detection) · [Vision Foundation Models](#/cv/foundation-models) · [VLM 101](#/vlm/vlm-101)
