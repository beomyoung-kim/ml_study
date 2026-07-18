# 예상 질문 & 답변

<div class="tag-row"><span class="tag">research narrative</span><span class="tag">project probes</span><span class="tag">research→product</span><span class="tag">motivation</span><span class="tag">curveballs</span><span class="tag badge-new">flagship</span></div>

> [!TIP] 이 챕터가 중요한 이유
> 이것이 loop를 이기게 하는 챕터입니다. 면접관은 사실 확인을 위해 CV를 묻는 경우가 드뭅니다 — 그들은 CV를 이용해 당신이 **research arc를 서사화**할 수 있는지, **특정 기여를 소유**할 수 있는지, **압박 속에서 선택을 방어**할 수 있는지 시험합니다. 아래는 *이* CV에서 나올 가능성이 가장 높은 질문 은행으로, 각각에 면접관의 숨은 의도, **각도별로 구분된 2–3개의 모범 답변**, 예상되는 **follow-up**, 그리고 한 줄 **rebuttal tip**을 담았습니다. 문장이 아니라 *구조와 trade-off*를 내면화하세요.

## 이 답변들을 쓰는 법

<div class="proscons"><div><div class="pros-t">Do</div>
<b>암기하지 말고 내면화하라.</b> 각 답변의 <i>척추</i> — 주장, 근거, trade-off — 를 익힌 다음, 자기 말로 실시간 재구성하세요. 암송한 답변은 첫 follow-up에서 죽고, 이해한 답변은 열 번을 버팁니다.
<br><br>
<b>방에 맞는 각도를 골라라.</b> 각 질문은 <b>technical-depth</b>, <b>impact/product</b>, <b>concise/executive</b> 변형을 제공합니다. 면접관을 읽으세요: FAIR 패널은 depth와 방어 가능성을 원하고, hiring manager는 impact와 fit을 원하며, recruiter는 30초 버전을 원합니다.
</div><div><div class="cons-t">Don't</div>
<b>과장하지 마라.</b> 대외비 작업(FaceSign, foreground API, 진행 중/심사 중)에 대해서는 지어낸 숫자가 아니라 <b>framing과 impact</b>로 말하세요: <i>"framing은 공유할 수 있지만, 구체적 내부 지표는 대외비입니다."</i> 이 문장은 회피가 아니라 integrity로 읽힙니다.
<br><br>
<b>"I"와 "we"를 흐리지 마라.</b> 협업자를 인정하되, <i>당신</i>의 기여를 명확히 하세요 — research-track 지원자가 감점되는 가장 흔한 이유입니다.
</div></div>

**회사별로 맞춤화**하려면 [Company Playbooks](#/process/companies)를 쓰세요: Meta는 publish-and-ship ownership을 원하고; NVIDIA는 systems/GPU 유창함과 지적 정직성에 보상하며; Apple은 on-device/privacy 제약을 파고들고; Adobe는 PyTorch 엄밀함 + 제품 감각을 원하며; ByteDance는 어려운 coding round에 접목하고; Mistral은 "왜 우리인가, US lab이 아니라?"를 묻고; Microsoft MSR은 job talk을 중심에 둡니다. 아래 모든 답변은 그중 하나로 다시 조준할 수 있습니다.

원자재가 되는 deep-dive: [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [FaceSign](#/resume/facesign) · [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM/Agents](#/resume/grounded-vlm-agents). Behavioral framing: [STAR & The Story Bank](#/behavioral/star). 맵: [Your CV → Interview Map](#/resume/overview).

---

## A · Research narrative & motivation

거의 모든 research-track loop의 오프너입니다. 그들은 당신의 커리어에 *thesis*가 있는지, 아니면 단지 논문 목록인지를 시험합니다.

<details class="qa"><summary>"Walk me through your research."</summary>
<div class="qa-body">

**Testing:** 14+편의 논문을 관통선이 있는 일관된 arc로, 적절한 고도에서, ~2–3분 안에 압축할 수 있는가 — 그러면서 *당신*의 역할을 명확히 하면서.

**Answer A — 관통선 (technical narrative).**
"하나의 실이 있습니다: **실세계 제약 하에서 visual perception을 정확하고 신뢰할 수 있게 만들기.** 저는 **weakly/semi-supervised segmentation**에서 시작했습니다 — DRS, BESTIE, PointWSSIS — perception 배포의 병목이 label이기 때문에, label-efficiency 프론티어를 공략했습니다: image tag나 몇 개의 point로 instance mask를 어떻게 얻을까. 그다음 **continual learning** — SSUL과 ECLIPSE — 배포된 모델은 forgetting이나 옛 데이터 저장 없이 새 class를 흡수해야 하기 때문입니다. 그다음 ZIM으로 **foundation model**까지 규모를 키웠습니다 — SAM 위에 세운 promptable zero-shot matting model입니다. 이 모든 것에서 패턴은 같습니다: *진짜 병목을 정의한 다음, brute force가 아니라 data + architecture 공동 설계로 푼다.* 가장 최근에는 stack 위쪽으로 올라가 **grounded VLM과 training-free visual reasoning agent**로 갔습니다 — 언어 reasoning을 픽셀- 및 영역-level 증거에 연결하고, perception tool이 언제 silent하게 실패하는지 진단합니다. 그래서: 픽셀 → label → continual → foundation → grounded reasoning. 같은 집착, 더 높은 추상화."

**Answer B — impact-first (hiring manager용).**
"두 문장 버전: 저는 ship되는 perception을 만듭니다. CVPR/ICCV/ECCV/NeurIPS에서 7편의 1저자 논문, ZIM으로 ICCV 2025 Highlight, 그리고 그 연구를 NAVER Cloud 제품으로 옮긴 일관된 기록 — 내부 eval에서 상용 대안들을 이긴 foreground-segmentation API, 정부 인증 face-auth 서비스 뒤의 anti-spoofing 모델, ~10ms로 돌아가는 on-device segmenter. 연구와 제품이 서로를 먹입니다: production은 어떤 문제가 진짜인지 알려주고, 논문은 해법을 엄밀하게 만듭니다."

**Answer C — concise/executive (30초).**
"저는 computer-vision applied scientist입니다. 제 arc는 label-efficient하고 continual한 segmentation에서, segmentation/matting foundation model(ZIM, ICCV 2025 Highlight)로, 그리고 이제 grounded vision-language model과 visual reasoning agent로 갑니다. 상수는 수백만 사용자 앞에 놓을 만큼 정확하고, 효율적이고, 검증 가능한 perception입니다 — NAVER Cloud에서 실제로 해온 일입니다."

**Follow-up:** "이 중 무엇을 최고작으로 보고, 왜인가요?"
**Rebuttal tip:** *reach*로는 ZIM(foundation + Highlight + product), *elegance*로는 PointWSSIS(proposal bottleneck 재정의)를 대되 — venue가 아니라 **아이디어**로 정당화하세요. "Best" = 인용 수 최고가 아니라 가장 명확한 문제 재정의.

</div></details>

<details class="qa"><summary>"논문들을 잇는 실은 무엇인가요? 아니면 그냥 팀에 필요했던 것들인가요?"</summary>
<div class="qa-body">

**Testing:** research *taste* — 문제를 고르는가, 아니면 문제가 당신을 고르는가? 냉소적 framing은 미끼이니 방어적으로 굴지 마세요.

**Answer A — 지적 실.**
"실은 **싸거나 약한 신호 → 신뢰할 수 있는 structured perception**입니다. DRS와 BESTIE: image tag → instance mask. PointWSSIS: 몇 개의 point → 완전한 mask. ECLIPSE: frozen model + 작은 prompt → forgetting 없이 새 class. ZIM: coarse SA-1B label → label converter를 통한 fine alpha matte. agent 작업조차 stack 꼭대기에서의 같은 수(move)입니다: noisy specialist tool 출력 → 검증 가능하고 자기 진단하는 reasoning program. 저는 일관되게 *감당할 수 있는 supervision*과 *제품이 요구하는 fidelity* 사이의 gap에 끌리고, 그것을 data + architecture 설계로 메웁니다."

**Answer B — 환경에 대해 정직 (제약을 소유).**
"솔직히 둘 다이고, 그게 강점이라고 봅니다. NAVER Cloud에서 일한다는 건 실제 제품이 실제 병목을 드러낸다는 뜻이었습니다 — labeling cost, category drift, latency, boundary quality. 하지만 *어떤* 병목을 논문으로 격상시킬지, 그리고 우리 제품을 넘어 일반화되도록 *어떻게* framing할지는 제 판단이었습니다. PointWSSIS는 labeling-budget 고통점에서 나왔지만, 그 기여 — semi-supervised instance segmentation에서 진짜 병목은 mask head가 아니라 proposal 단계라는 것 — 는 일반적 통찰입니다. 그게 taste 부분입니다."

**Follow-up:** "그럼 완전한 자유가 있다면 다음에 어떤 문제를 고르겠습니까?"
**Rebuttal tip:** 진짜 답을 준비하세요 — "verifiable spatial reasoning: perception tool이 자신의 uncertainty를 보고해서 agent가 언제 그것을 불신할지 알게 하는 것." 그 실이 retrofit이 아니라 미래 지향적임을 증명합니다.

</div></details>

<details class="qa"><summary>"perception에서 grounded VLM과 agent로 pivot한 이유는? 유행을 쫓는 것 아닌가요?"</summary>
<div class="qa-body">

**Testing:** 그 이동이 기회주의적(LLM hype를 따라감)인가 원리적 연속인가. 2025–26 vision 채용에서 *바로 그* 서사 질문입니다.

**Answer A — pivot이 아니라 연속 (depth).**
"perception에서 벗어나는 pivot이 아니라 — perception이 새 stack에서 자기 역할을 찾은 것입니다. End-to-end VLM은 강하지만 **hallucinate**하고 *근거 없는 텍스트 설명*을 만듭니다: 답이 맞으면서도 visual 증거가 틀릴 수 있습니다 — spurious success. 제 배경 전체가 high-fidelity, region- 및 pixel-level perception입니다. 그게 정확히 빠진 재료입니다: 언어 reasoning을 검증 가능한 visual 증거에 grounding하고, agent에게 *specialist tool*(ZIM 품질의 mask나 depth 같은)을 줘서 3D와 시간을 reasoning하게 하는 것. 흥미로운 새 문제는 그 tool들이 **silent하게 실패**한다는 것입니다 — 그럴듯하지만 틀린 box나 mask를 예외 없이 반환합니다 — 그래서 제 현재 작업은 그 silent failure를 *typed diagnosis*로 바꿔 targeted program repair를 구동합니다. 저는 perception을 떠나는 게 아니라, 그것을 reasoning 아래의 신뢰할 수 있는 tool layer로 만드는 겁니다."

**Answer B — 시장/제품 논리 (impact).**
"제품이 이동했습니다. Image editing, robotics, UI agent — 모두 *언어 query로부터 영역을 resolve*하고 검증 가능하게 행동해야 합니다. 이미지를 설명하는 순수 end-to-end 모델은 사용자가 '왼쪽 사람을 지워줘'라고 하거나 로봇이 어떤 객체가 다른 객체 뒤에 있는지 reasoning해야 할 때 충분하지 않습니다. 제 perception 작업이 그 substrate입니다. pivot은 가치가 있는 곳을 따라가는 것이고, 저는 마침 그 가치가 요구하는 바로 그 배경을 가졌습니다."

**Answer C — concise.**
"프론티어가 '모델이 perceive할 수 있는가?'에서 '모델이 그것에 대해 거짓말하지 않고 perception을 reasoning할 수 있는가?'로 이동했기 때문입니다. Grounding과 agentic tool-use가 그에 답하는 방식이고, 제 perception depth가 sunk cost가 아니라 차별점입니다."

**Follow-up:** "frontier VLM이 계속 좋아지면, 당신의 modular/agent 접근은 obsolete되지 않나요?"
**Rebuttal tip:** 그들이 강함을 인정한 뒤, 잔여를 명시하세요: **정밀한 측정, 검증 가능한 증거, 전체 모델 retrain 없이 module 업그레이드, 그리고 diagnostic repair.** end-to-end가 개선돼도 hybrid가 실용적 default입니다 — 그게 scale에 반대하는 베팅이 아니라 방어 가능한 포지션입니다.

</div></details>

<details class="qa"><summary>"풀타임으로 일하며 PhD를 했군요. 그럼 연구는 side project 아닌가요?"</summary>
<div class="qa-body">

**Testing:** 엄밀함과 독립성 — 산업 job이 "진짜" 일이고 PhD는 형식인가, 아니면 둘이 진짜로 하나의 프로그램인가.

**Answer A — 통합 (reframe).**
"둘은 같은 프로그램이고, 그건 드문 일이며 제 edge라고 생각합니다. 제 PhD 문제들은 production*에서* 나옵니다 — PointWSSIS는 labeling cost에서, ECLIPSE는 라이브 API의 category drift에서, ZIM은 fine mask가 필요한 editing 제품에서. 그래서 제 논문은 현실에 대해 미리 검증되어 있고, 제 제품은 학문적 엄밀함을 얻습니다: ablation, baseline, 정직한 failure analysis. *ship하면서* 7편의 1저자 top-venue 논문과 ICCV Highlight를 낸 것이 side project가 아니라는 증거입니다 — 대부분은 둘 중 하나만 합니다."

**Answer B — 독립성 신호.**
"제가 연구를 이끕니다. 7편의 논문에서 1저자이고, 문제 framing을 정하고, 실험을 돌리고, 오픈소스되어 채택된 코드를 썼습니다. KAIST MLAI의 Sung Ju Hwang 교수님이 방향을 조언하고 framing을 벼려주시지만; 실행과 구체적 기여는 제 것입니다. 풀타임 역할은 학계가 좀처럼 갖지 못하는 data와 제품 맥락을 줍니다."

**Follow-up:** "시간과 두 우선순위 세트를 어떻게 관리하나요?"
**Rebuttal tip:** 고생을 미화하지 마세요. *정렬*로 framing하세요 — 제품 필요와 publishable question이 겹치는 문제를 골라서 비용을 두 번 치르지 않는다고. 그건 근성이 아니라 판단입니다.

</div></details>

---

## B · Per-project probes

면접관은 프로젝트 하나 — 보통 ZIM이나 그들 팀과 맞는 것 — 를 골라 당신이 ownership을 보이거나 무너질 때까지 파고들 것입니다. 완전한 기술적 depth는 deep-dive에 있고; 여기는 *entry* probe와 인터뷰 고도에서 답하는 법입니다.

<details class="qa"><summary>ZIM — "matting 데이터로 그냥 SAM을 fine-tune하면 안 됐나요?"</summary>
<div class="qa-body">

**Testing:** 순진한 baseline이 *왜* 실패하는지 이해하는가 — 당신 자신의 Highlight 논문의 핵심 동기. 여기서 더듬는 건 치명적입니다.

**Answer A — failure mode (technical).**
"해봤고, 붕괴합니다. 공개 matting 데이터는 **macro** target이 지배합니다 — 전신 portrait, salient한 전체 객체. 거기에 SAM을 fine-tune하면 *micro granularity를 잊습니다*: 머리카락 한 올이나 작은 부분에 point prompt를 주면 사람 전체를 게워냅니다. SAM을 가치 있게 만든 zero-shot promptability를 내다 버리는 겁니다. 우리 ablation이 이걸 극명하게 만듭니다 — 같은 architecture를 공개 matte로 학습하면 fine-grained SAD가 약 120이고 우리 변환 데이터로는 약 10입니다. **Data granularity가 architecture를 지배합니다.** 그래서 잘못된 분포에 fine-tune하는 대신, SA-1B의 ~1B coarse mask를 fine alpha matte로 바꾸는 **label converter**를 만들었고, noise 제어를 위한 Spatial Generalization Augmentation과 Selective Transformation Learning을 더해 — 그걸로 학습했습니다."

**Answer B — 설계 원리 (concise).**
"문제가 loss가 아니라 data 계약이기 때문입니다. SAM의 supervision은 coarse하고 hard-mask이며; matting은 soft한 micro-level boundary가 필요합니다. Macro portrait 데이터에 fine-tune하면 promptable zero-shot behavior가 파괴됩니다. 우리는 converter로 SA-1B 규모에서 micro-granular matte label을 생성했고, 실제로 fine detail을 렌더링하기 위해 hierarchical pixel decoder와 prompt-aware masked attention을 더했습니다. 그래서 zero-shot을 유지*하면서* matte를 만듭니다."

**Follow-up:** "hierarchical pixel decoder가 구체적으로 무엇을 벌어다 주나요?"
**Rebuttal tip:** SAM의 stride-4 decoder + 순진한 upsample은 **checkerboard artifact**를 유발하고 fine 구조를 잃습니다; multi-level(stride 2/4/8) 융합이 ~10ms overhead로 high-res detail을 복원합니다. 그 숫자와 메커니즘을 준비해 두세요 — [ZIM](#/resume/zim) 참고.

</details>
</div>

<details class="qa"><summary>ZIM — "prompt-aware masked attention: 어디에 적용하고, 왜 전부는 아닌가요?"</summary>
<div class="qa-body">

**Testing:** 당신 자신의 architectural 디테일에 대한 정밀함 — 실제로 일하지 않은 "we"-author를 잡아내는 애용 수법.

**Answer A — 메커니즘.**
"**token-to-image (T2I)** cross-attention에만입니다. Box prompt는 박스 밖 attention logit을 −∞로 설정하는 binary mask가 되고; point는 QK score를 modulate하는 Gaussian(σ≈21)이 됩니다. 직관: 저는 *출력 token*이 prompt된 영역에 attend하기를 원합니다. 결정적으로 image-to-token attention에는 적용하지 **않습니다** — 그걸 강제하면 global image feature가 손상되어 성능이 떨어집니다. 그 비대칭성이 자명하지 않은 부분이고, ablation으로 검증했습니다."

**Answer B — concise.**
"T2I attention에: box → hard mask, point → Gaussian. global feature를 해치지 않고 decoder를 prompt 영역에 집중시킵니다 — 그래서 I2T 방향에 적용하면 역효과가 납니다."

**Follow-up:** "point에 hard disk 대신 왜 Gaussian인가요?"
**Rebuttal tip:** Soft falloff는 prompt-위치 애매함을 견디고 boundary 근처에서 더 부드러운 gradient를 줍니다 — matting은 soft-label task라, hard disk는 objective와 싸울 겁니다.

</div></details>

<details class="qa"><summary>ECLIPSE — "forgetting을 피하려고 backbone 전체를 freeze — 새 class 학습을 포기하는 것 아닌가요?"</summary>
<div class="qa-body">

**Testing:** stability–plasticity trade-off, 그리고 극단적 설계 선택을 방어할 수 있는가.

**Answer A — trade + 레버 (depth).**
"step 1 이후 Mask2Former를 freeze하면 forgetting이 *구조적으로 불가능*해집니다 — 옛 weight가 절대 움직이지 않습니다 — 그래서 우리는 긴 class sequence에서 all-PQ로 CoMFormer 같은 distillation 기반 방법을 이기고, 오직 ~1.3% trainable parameter로, 옛 데이터 저장 없이 그렇게 합니다. 비용은 정확히 우려하는 그것입니다: **error propagation**(step-1 실수가 고정됨)과 제한된 plasticity. 우리는 plasticity를 두 방식으로 회복합니다: 모든 transformer layer의 **deep visual prompt**(deep vs shallow가 ~100K params로 new-class PQ를 의미 있게 올림), 그리고 상한을 올리는 더 강한 frozen init(Swin-L / COCO-pretrained). 그리고 **logit manipulation**이 no-obj semantic drift를 처리합니다 — 'background'의 의미가 매 step 바뀌므로 고정된 no-obj classifier는 신뢰할 수 없어, 우리는 그것을 다른 class의 logit에서 post-hoc으로 재계산합니다. 그러니 '새 class를 포기'가 아니라 'stability를 공짜로 사고, 조금 써서 plasticity를 도로 빌린다'입니다."

**Answer B — 제품 framing (impact).**
"라이브 segmentation API에서는 카테고리를 끊임없이 추가하고, 프라이버시/저장 이유로 옛 학습 데이터를 종종 *보관할 수 없습니다*. ECLIPSE는 45M-param 모델을 retrain하거나 옛 데이터를 replay하는 대신, 새 카테고리 배치마다 작은 adapter를 배포하게 해줍니다. 그게 on-device나 privacy-constrained 업데이트 스토리에 직접 매핑됩니다 — Apple 풍미."

**Follow-up:** "왜 semantic이 아니라 panoptic continual learning인가요?"
**Rebuttal tip:** Panoptic이 엄격히 더 어렵습니다 — instance matching *과* stuff *과* 변동하는 no-obj 정의를 한꺼번에, 그리고 PQ가 recognition 실패(RQ)를 벌합니다. Continual panoptic은 semantic 대비 덜 연구되었고; 그 gap이 기여의 일부입니다. [ECLIPSE](#/resume/eclipse) 참고.

</div></details>

<details class="qa"><summary>PointWSSIS — "실제 통찰이 뭔가요? 그냥 또 다른 semi-supervised 트릭 같은데요."</summary>
<div class="qa-body">

**Testing:** 문제를 재정의했는가, 아니면 그냥 튜닝했는가 — 강한 CVPR 논문과 incremental 논문의 차이.

**Answer A — 재정의 (depth).**
"통찰은 instance segmentation에서 병목이 mask head가 아니라 **proposal 단계**라는 것입니다. Semi-supervised 방법은 pseudo-label에서 false-negative / false-positive trade-off와 싸웁니다: confidence threshold를 낮추면 false positive에 잠기고, 높이면 진짜 instance를 지웁니다 — 왜냐하면 *proposal이 없으면, 표현력 있는 mask branch도 아무것도 출력하지 않기* 때문입니다. 그래서 저는 supervision을 바꿨습니다: 몇 개의 full mask에 많은 값싼 **point** label, 여기서 point가 proposal 애매함을 직접 resolve합니다 — teacher proposal을 true positive로 걸러냅니다. 그다음 Adaptive Pyramid-Level Selection이 point가 담지 못하는 결측 scale 정보를 처리하고, MaskRefineNet이 full label이 부족할 때 rough mask를 정리합니다. 결과: COCO 5% full label에서 ~24.9 AP(기존 SSIS)에서 33.7로 오르고, 50%에서는 fully-supervised에 거의 필적합니다. 수는 threshold 튜닝이 아니라 *병목을 재배치*하는 것입니다."

**Answer B — 계보 (concise).**
"BESTIE는 image-tag → instance를 밀어붙일 수 있지만 pseudo-label drift에 상한이 있음을 보였습니다. PointWSSIS는 묻습니다: *proposal* 병목을 구체적으로 제거하는 가장 싼 신호가 무엇인가? Point입니다. 진짜 병목에 대한 targeted intervention이라, 낮은 label budget에서의 gap이 marginal이 아니라 큽니다."

**Follow-up:** "point가 centroid여야 하나요?"
**Rebuttal tip:** 아니요 — random-in-mask가 centroid만큼 잘 작동합니다(SOLOv2가 다양한 in-region point를 견딤), 그래서 labeling이 더 싸집니다. 그 robustness 자체가 ablation으로 뒷받침되는 selling point입니다. [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) 참고.

</div></details>

<details class="qa"><summary>BESTIE — "여전히 CAM과 pseudo-label을 쓰네요. noise는 어떻게 다루나요?"</summary>
<div class="qa-body">

**Testing:** weakly-supervised failure mode(semantic drift)에 대한 장악력과 당신의 구체적 해결책.

**Answer A — drift + 해결책.**
"치명적 failure는 **semantic drift**입니다: 놓친 instance가 background로 학습되는 한편 시각적으로 동일한 instance가 foreground가 되어, 모델이 모순을 학습하고 저하됩니다. 저는 이걸 세 방식으로 공략합니다. **Instance-aware guidance**는 center/offset loss를 labeled-instance 영역에만 적용해, background 영역이 신호를 오염시키지 않게 합니다. **Self-refinement**는 mini-batch마다 online으로 돌아갑니다 — network 자신의 grouped 출력을 soft-weighted refined label로 되먹여, 학습 과정에서 false-negative를 true-positive로 승격시킵니다. 그리고 **PAM**(Peak Attention Module)은 CAM noise를 억제하면서 진짜 instance cue를 날카롭게 합니다 — semantic coverage를 위해 activation을 *퍼뜨리는* DRS와 반대 방향. 같은 저자, 반대 필요에 반대 도구."

**Answer B — concise.**
"결측 pseudo-instance에서 오는 semantic drift가 적입니다. loss를 신뢰 영역으로 masking하고, 놓친 것이 회복되도록 label을 online으로 refine하고, raw noisy CAM 대신 peak 기반 cue를 씁니다. WSSS 품질이 상한을 정하지만, WSIS는 WSSS가 떨어져도 우아하게 저하됩니다 — 그 민감도를 보여줍니다."

**Follow-up:** "PAM이 이전 DRS 작업과 어떻게 다른가요?"
**Rebuttal tip:** DRS는 semantic segmentation을 위해 coverage를 넓히려고 discriminative 영역을 *억제*하고; PAM은 instance를 localize하려고 peak를 *증폭*합니다. 둘을 같은 CAM-shaping 통찰의 양면으로 framing하면 강한 research-taste 스토리가 됩니다.

</div></details>

<details class="qa"><summary>DRS / SSUL — "오래된 논문들이네요. 여전히 유의미한가요, 아니면 초기 커리어 워밍업인가요?"</summary>
<div class="qa-body">

**Testing:** 초기 작업을 무시하기보다 거기서 durable principle을 뽑아낼 수 있는가.

**Answer A — 이어진 원리.**
"그것들이 이후 모든 것의 뿌리입니다. **DRS**(AAAI 2021)는 CAM attention을 *shaping*하는 것 — discriminative 영역을 억제해 activation이 객체 전체로 퍼지게 하는 것입니다. 그 CAM-shaping 직관이 BESTIE의 PAM이 되었습니다. **SSUL**(NeurIPS 2021, co-first)은 class-incremental semantic segmentation에서 **unknown/future class**를 명시적으로 모델링하는 것과 exemplar memory를 도입했습니다 — 그 'background의 의미는 step마다 불안정하다'는 통찰이 *정확히* ECLIPSE의 logit manipulation이 나중에 panoptic에 대해 푼 것입니다. 그러니 워밍업이 아니라; 제 두 오래 이어진 실 — CAM manipulation과 continual segmentation — 이 심어진 곳입니다."

**Answer B — concise.**
"이후 모든 논문이 이 둘로 추적됩니다: DRS → attention shaping → BESTIE; SSUL → 불안정한 background → ECLIPSE. 초기이지만 load-bearing입니다."

**Follow-up:** "SSUL은 co-first-author였는데 — *당신* 몫은 무엇이었나요?"
**Rebuttal tip:** 구체적이고 관대하게: 당신이 이끈 부분(segmentation modeling / unknown-class handling)을 대고, 공저자의 부분(incremental-learning framing)을 ownership을 희석하지 않으면서 인정하세요. 애매한 "we did it together"는 공저 논문에서 red flag로 읽힙니다.

</div></details>

<details class="qa"><summary>FaceSign — "anti-spoofing 모델 얘기 좀 해주세요. 정확도는 얼마나 나왔나요?"</summary>
<div class="qa-body">

**Testing:** 두 가지를 동시에 — face anti-spoofing을 깊이 이해하는가, *그리고* 대외비 경계를 우아하게 다루는가(Apple/규제 role의 진짜 신호).

**Answer A — decline-then-deliver (본받을 모델).**
"구체적 정확도, attack-set 구성, architecture는 대외비입니다 — 정부 인증 결제 및 사원증 인증 서비스라, 계약상·윤리상 거기에 묶여 있습니다. 제가 *드릴 수 있는* 건 framing입니다. Anti-spoofing은 capture와 recognition 사이에 있습니다: detect/align → **liveness / presentation-attack detection** → recognize. Presentation attack — print, replay, 3D mask, 그리고 별개로 digital injection — 이 뚫리면 recognition이 아무리 좋아도 시스템은 실패합니다. 어려운 부분은 쉬운 print attack이 아니라 **domain generalization**입니다: 미지의 폰, 조명, demographic, 새로운 attack instrument. 우리는 APCER/BPCER/ACER frame(ISO/IEC 30107)으로 평가하고, 요구되는 보안 수준에 따라 recognition FRR 대비 operating point를 고릅니다. False reject는 전환을 깎으므로, step-up fallback(password)이 detector만큼 중요합니다."

**Answer B — systems view (concise).**
"저는 anti-spoofing component를 소유했습니다. 인증된 보안 제품이라 숫자는 공유할 수 없지만, threat model, sensor trade-off(RGB-only vs depth/IR), evaluation frame, 그리고 보안과 승인율 사이의 operating-point trade-off는 얘기할 수 있습니다. 이건 단순 classifier가 아니라 *system* 문제입니다 — sensing + model + risk policy."

**Follow-up:** "Apple Face ID와 어떻게 비교하겠습니까?"
**Rebuttal tip:** 비교를 지어내지 마세요. Face ID는 *hardware co-design*(structured-light dot projector)으로 유명하고; 당신 작업은 service-side model + operation입니다. 공유 DNA: "sensing + model + policy를 하나의 시스템으로." head-to-head 숫자는 존재하지 않으니 — 그렇게 말하세요. [FaceSign](#/resume/facesign) 참고.

</div></details>

<details class="qa"><summary>On-device seg — "왜 10ms가 목표인가요? 그냥 서버에서 더 큰 모델을 쓰면 안 되나요?"</summary>
<div class="qa-body">

**Testing:** systems/efficiency reasoning과 on-device가 *왜* 중요한지 이해하는가 — 숫자를 맞췄다는 것만이 아니라.

**Answer A — frame-budget 엔지니어링 (depth).**
"10ms는 마법의 숫자가 아니라 **frame-budget** 결과입니다. 실시간 UX는 ~30fps ≈ 33ms/frame이고, segmentation은 그 budget을 camera preprocessing, post-processing, 다른 모델, UI와 나눕니다. Segmentation에 ~10ms를 주면 전체 pipeline이 버티고; 20을 주면 frame을 떨어뜨립니다. 그게 서버가 아니라 *mobile CPU*인 이유는 프라이버시(frame이 기기를 떠나지 않음)와 비용(inference당 서버 청구 없음), 그리고 오프라인 capability입니다. 저는 worst common denominator — 특정 NPU가 아니라 CPU — 에 맞춰 설계해 어디서나 돌게 했습니다. 가장 큰 레버부터 잘라 도달했습니다: input resolution, channel width, decoder depth; 그다음 boundary quality를 유지하려고 heavy teacher로부터 distillation; QAT는 마지막인데, 먼저 quantize하면 무엇보다 먼저 머리카락과 fine boundary가 죽기 때문입니다."

**Answer B — two-tier 스토리 (impact, Apple 풍미).**
"이건 two-tier 전략의 on-device 절반입니다: heavy cloud foundation model(ZIM-class)이 품질을 제공하고 distillation teacher 역할을 하며; 작은 distilled + quantized student가 frame budget과 프라이버시를 위해 on-device에서 ~10ms로 돕니다. 그 'cloud foundation + on-device specialist' 분할이 정확히 현대 제품이 필요로 하는 패턴이고, 저는 그 양 끝을 다 만들었습니다."

**Follow-up:** "ONNX serving이라 했는데 — 정말 폰 CPU에서인가요 아니면 서버인가요?"
**Rebuttal tip:** 정확하게 말하세요, CV 표현이 이 질문을 부르니: *모델*은 mobile-CPU budget에 맞춰 설계·측정되었고; *serving path*는 사내 ONNX 기반 인프라였습니다. 설명할 수 없는 폰 배포를 과장하지 말고 design target과 deployment infra를 구분하세요. [On-Device Seg](#/resume/on-device-segmentation) 참고.

</div></details>

<details class="qa"><summary>Ongoing agents — "NeurIPS 제출이 심사 중이군요. 설명해 주세요."</summary>
<div class="qa-body">

**Testing:** 미출판 작업을 진짜 실체와 함께 제시하면서 "과장 금지 / 유출 금지" 선을 지킬 수 있는가 — 그리고 공허하게 들리지 않으면서.

**Answer A — problem-first framing (안전하고 강한 버전).**
"심사 중이라, 최종 숫자나 accept 주장 대신 문제와 framing을 드리겠습니다. 설정은 3D spatial reasoning을 위한 **visual program synthesis**입니다 — LLM이 specialist vision tool(detection, depth, segmentation)을 호출하는 실행 가능한 program을 씁니다. 제가 신경 쓰는 failure는 **silent perception failure**입니다: tool이 그럴듯하지만 틀린 출력 — 나쁜 box, 잘못된 depth — 을 반환하고, 예외가 없으니 program이 끝까지 실행되어 *자신 있게 틀린* 답을 냅니다. 제 framework는 그 silent failure를 **typed diagnosis**로 바꾸고, 그 diagnosis로 **targeted program repair**를 구동합니다. 목표는 *task-specific training 없이* 3D spatial reasoning에서 frontier VLM에 필적하는 것입니다. Public lineage는 VisProg / ViperGPT(program synthesis)와 VADAR(3D용 dynamic API generation)이고; 제 각도는 silent failure에 대한 diagnose-and-repair loop입니다."

**Answer B — why-it-matters (concise).**
"vision tool을 호출하는 agent는 LLM이 계획을 못 세워서가 아니라 *tool이 조용히 거짓말*해서 깨집니다. 저는 그 silent failure를 관찰 가능하고 수리 가능하게 만들어, training-free로 frontier-level 3D spatial reasoning을 겨냥합니다. 세부와 숫자는 출판 전입니다."

**Follow-up:** "어떻게 평가하나요?"
**Rebuttal tip:** public benchmark(Omni3D-Bench, Spatial457, SpatialAct)와 올바른 metric — answer accuracy *에 더해* grounding IoU, program executability, failure-type recall — 로 *철학*을 말하되, 내부 bench는 언급하지 마세요. "요점은 최종 답만이 아니라 증거를 측정하는 것." [Grounded VLM/Agents](#/resume/grounded-vlm-agents) 참고.

</div></details>

<details class="qa"><summary>Grounded VLMs — "grounding이 실제로 어떻게 hallucination을 줄이나요?"</summary>
<div class="qa-body">

**Testing:** buzzword를 넘어선, 현재 프론티어에 대한 개념적 depth.

**Answer A — 메커니즘.**
"VLM의 hallucination은 대체로 *근거 없는 generation*입니다 — language head가 증거를 가리킬 의무 없이 유창한 주장을 만듭니다. Grounding은 계약을 바꿉니다: 모델은 주장에 대해 **영역이나 픽셀을 resolve**해야 하므로, 답이 검증 가능한 위치에 묶입니다. 두 이점. 첫째, *확인*할 수 있습니다 — grounding IoU / pointing accuracy가 순수 텍스트 metric이 숨기는 '맞는 답, 틀린 증거'(spurious success)를 드러냅니다. 둘째, reasoning이 증거를 *대상으로 작동*하도록 강제할 수 있습니다 — pixel-grounded chain-of-thought, crop-and-zoom loop(ViGoRL/MGPO 같은 grounded RL), reasoning 도중의 mask 연산. hallucination을 없애지는 않지만, 모델의 주장을 반증 가능하게 만듭니다 — editing, robotics, UI agent에서 신뢰의 전제조건입니다."

**Answer B — concise.**
"모델이 *말하기*만이 아니라 *가리키게* 요구함으로써입니다. 모든 주장이 영역으로 resolve돼야 하면, 검증할 수 있고 spurious success — 틀린 visual 이유로 맞은 답 — 를 잡을 수 있습니다."

**Follow-up:** "그냥 detector 하나 붙이는 것 아닌가요? 연구는 어디에 있나요?"
**Rebuttal tip:** detector는 box를 줍니다; 연구는 reasoning이 그 box를 end-to-end로 *소비하고 검증*하게 만들고, 그것이 틀렸을 때를 다루고, detector 혼자서는 못 하는 region-query resolution을 하는 것입니다. Grounding은 bolt-on box가 아니라 reasoning 계약입니다.

</div></details>

---

## C · Research → product & impact

Applied-Scientist 차별점입니다. ship하지 못하는 RS 지원자는 engineering track으로 걸러지고; 당신은 반대 문제의 해독제를 가졌습니다 — shipping이 진짜이고 엄밀함을 증명하세요.

<details class="qa"><summary>"foreground-segmentation API가 Photoroom, Remove.bg, Adobe를 이겼다고요. 증명하세요 — 어떻게 측정했나요?"</summary>
<div class="qa-body">

**Testing:** 큰 주장 뒤의 엄밀함과 정직성. 방법론 없는 대담한 benchmark 주장은 red flag이고; 방법론이 있으면 green flag입니다.

**Answer A — methodology-first (rigor).**
"**내부 평가**였으니 범위를 정확히 하겠습니다: *framing*은 공유할 수 있지만 대외비인 경쟁사별 숫자는 아닙니다. 우리는 제품의 실제 입력을 대표하는 evaluation set을 만들었습니다 — 실패하기 쉬운 케이스가 가장 중요합니다: 머리카락, fine boundary, 반투명, 다인, 어려운 조명. boundary-focused metric(boundary F-score, SAD 스타일 boundary error)에 human visual review를 더해 채점했습니다 — mIoU만으로는 고객이 알아채는 바로 그 boundary 품질을 숨기기 때문입니다. 같은 입력에 대해 우리 model + data pipeline이 *우리* 분포에서 그 상용 API들을 앞섰습니다. 정직한 단서: 우리 test set이고 우리 domain이라 — universal 우위를 주장하지 않겠고, 그걸 면접장에서 말하겠습니다."

**Answer B — moat (product).**
"승리는 더 큰 모델이 아니라 **data**였습니다. 차별점은 제품 분포에 맞춘 boundary-hard 학습 데이터 curation과 사용자가 실제로 보는 boundary에 맞춰 튜닝한 pipeline이었습니다. 기성 API는 generic 분포에 최적화하고; 우리는 우리 것에 최적화했습니다. 방어 가능하고 반복 가능한 우위이며, ZIM과 같은 data-first 교훈입니다 — data 계약이 품질을 결정합니다."

**Follow-up:** "이게 ZIM과 같은 모델인가요?"
**Rebuttal tip:** 아니요 — 뭉뚱그리지 마세요. API는 제품 라인(내부 serving)이고; ZIM은 promptable zero-shot foundation model입니다. 공유 DNA(boundary quality, data curation), 다른 artifact. 둘이 한 모델이라 주장하는 건 follow-up에서 풀리는 종류의 과장입니다.

</div></details>

<details class="qa"><summary>"ZIM을 CLOVA-X Image Editing에 넣었군요. 연구가 production을 만났을 때 뭐가 깨졌나요?"</summary>
<div class="qa-body">

**Testing:** 실제로 연구를 ship했는가, 아니면 checkpoint를 담 너머로 던졌을 뿐인가. 흥미로운 답은 마찰입니다.

**Answer A — 핸드오프 마찰 (systems).**
"공유할 수 있는 public 디테일은 ship되었고 TEAM NAVER DAN 24에서 발표했다는 것이며; 정확한 SLA는 내부입니다. 진짜 일은 *핸드오프*였고, 거기서 research-to-product가 보통 깨집니다: (1) **representation mismatch** — editor는 raw matte가 아니라 특정 alpha format(premultiplied 여부, color spill 처리)을 기대합니다; (2) **latency** — ViT-B matte 모델은 V100에서 ~180ms 정도인데, batch tool엔 괜찮지만 interactive UX엔 resolution 제어, model tiering, caching으로 관리해야 합니다; (3) **failure monitoring** — 머리카락과 반투명이 matte가 실패하는 곳이라, fallback(예: hard-threshold)과 monitoring이 필요합니다; (4) **API compatibility** — 기존 client는 hard mask를 기대했으니, soft-matte 출력이 그들을 깨지 않고 끼워져야 했습니다. Shipping은 20%가 모델, 80%가 이 계약들입니다."

**Answer B — concise (impact).**
"ZIM을 demo와 함께 오픈소스하고 CLOVA-X image editing에 통합했습니다. 교훈: 모델은 쉬운 부분이었습니다. editor로의 format 핸드오프, interactive-latency budgeting, 머리카락/투명 failure fallback이 진짜 엔지니어링이 들어간 곳입니다. 내부 SLA는 대외비이지만, framing은 표준적인 research-to-product hardening입니다."

**Follow-up:** "video editing으로 어떻게 확장하겠습니까?"
**Rebuttal tip:** SAM2 스타일 memory + temporal consistency가 방향이고, matting엔 진짜 미해결입니다 — 그렇게 말하세요. Frame 간 flickering alpha가 어려운 부분이니; trivial한 확장인 척하지 마세요.

</div></details>

<details class="qa"><summary>"작업 대부분을 오픈소스했군요. NAVER의 우위를 내주는 것 아닌가요?"</summary>
<div class="qa-body">

**Testing:** 제품/사업 판단과 회사가 *왜* 논문을 내는지 이해하는가.

**Answer A — 전략적 논거.**
"출판된 방법은 moat가 아닙니다; *data operation, integration, latency engineering, domain adaptation*이 moat입니다. ZIM, ECLIPSE, PointWSSIS, BESTIE를 오픈소스한 것은 연구 브랜드, 채용, ecosystem을 벌어줬습니다 — Grounded-ZIM과 downstream 채택자(Inpaint-Anything, medical, 3D)가 *오픈이라서* 나왔고, 이는 작업을 공짜로 검증하고 확장합니다. 한편 foreground API와 FaceSign — 직접적 상업·보안 가치가 있는 것 — 은 내부에 남았습니다. 그러니 분할은 의도적입니다: 일반적 방법은 출판하고, 제품 특유의 data와 operation은 보호합니다."

**Answer B — concise.**
"논문이 제품이 아니기 때문입니다. 우위는 우리가 출판하지 않는 data pipeline과 integration에 있습니다. 방법을 오픈소스하는 건 브랜드, 채용, ecosystem에 net-positive입니다 — 그래서 대외비인 것들(API, FaceSign)은 오픈이 아닙니다."

**Follow-up:** "그럼 우리가 채용하면 *무엇을* 닫아두겠습니까?"
**Rebuttal tip:** NAVER 정책 낭독이 아니라 *그들의* IP를 reasoning할 수 있음을 보이세요 — "제품 특유의 data, 실제 고객에 묶인 eval set, safety/abuse 관련은 무엇이든; 일반 방법은 출판을 밀겠습니다." 그게 그들이 확인하는 성숙함입니다.

</div></details>

---

## D · Career, motivation & fit

Recruiter와 HM 영역. 이것들이 *fit*을 결정하고, 허술한 답변은 기술 라운드가 강해도 loop를 끝냅니다. 정직하고, 구체적이고, 비방어적으로 유지하세요.

<details class="qa"><summary>"5년 넘게 있었는데 왜 NAVER를 떠나나요?"</summary>
<div class="qa-body">

**Testing:** 무언가를 *향해* 달리는가(좋음), 무언가에서 *멀어지려* 하는가(risk)? 씁쓸함이나 모호함 둘 다 해롭습니다.

**Answer A — 프론티어를 향해 (positive pull).**
"NAVER Cloud에서의 5년은 드문 것을 줬습니다 — foundation model에서 수백만이 쓰는 API까지, 완전한 research-to-product loop. 저는 자랑스럽고 불행해서 떠나는 게 아닙니다. 제가 지금 가장 신경 쓰는 문제 — grounded, verifiable multimodal reasoning과 frontier-scale visual agent — 가 [company]가 가장 세게 밀고 있는 곳이고, 혼자서는 도달할 수 없는 수준에서 그걸 할 compute, multimodal data, 동료가 있기 때문에 옮깁니다. '강한 지역 제품 연구'에서 '글로벌 프론티어 연구'로 가면서, 제 perception-to-product 규율을 그 환경에 가져가고 싶습니다."

**Answer B — scope와 scale (concise).**
"한 조직 안에서 돌릴 수 있는 loop를 다 소진했습니다. Frontier-scale multimodal 문제, 주변의 더 큰 연구 커뮤니티, 그리고 [company]가 가진 인프라가 필요한 규모에서 grounded reasoning을 할 기회를 원합니다. NAVER에서 밀어내는 게 아니라 그 일로 끌리는 것입니다."

**Follow-up:** "NAVER에서는 못 하고 여기서는 할 수 있는 게 구체적으로 뭔가요?"
**Rebuttal tip:** 구체적이고 아부처럼 들리되 참되게: multimodal pretraining의 규모, 특정 팀/agenda(이름을 대세요 — Meta의 SAM/perception, NVIDIA의 physical-AI, Apple의 on-device foundation model, Adobe의 Firefly), 그리고 peer density. NAVER에 대한 불평처럼 들리는 건 피하세요.

</div></details>

<details class="qa"><summary>"왜 하필 이 회사인가요?" (회사별)</summary>
<div class="qa-body">

**Testing:** 숙제와 진짜 fit. 여기서 generic 답변은 치명적입니다. [Company Playbooks](#/process/companies)를 쓰세요.

**Meta (FAIR/VLM):** "제 perception-to-grounding 실이 product VLM에 빠진 조각입니다 — SAM 계보가 문자 그대로 ZIM의 출발점이고, grounded, verifiable reasoning은 FAIR-adjacent입니다. 저는 출판*하고* ship합니다 — 그게 FAIR/Applied 기준입니다."

**NVIDIA Research:** "Spatial reasoning과 perception tool은 physical AI와 robotics(GEAR/GR00T)의 substrate이고, 저는 효율적이고 배포 가능한 stack을 중시합니다 — 제 on-device와 ONNX 작업이 systems 면을 보이고, 당신 lab이 중시하는 GPU/systems 유창함으로 기대겠습니다."

**Apple (MLR / Multimodal):** "제 two-tier 스토리가 당신 스토리입니다: cloud foundation 품질(ZIM) + 실제 frame budget(~10ms)의 on-device specialist, privacy와 hardware 제약 하에서. 저는 당신이 최적화하는 정확도-대-latency-대-battery trade-off 하에서 perception을 ship했고, 외부 가시성 없이 일하는 데(FaceSign) 편안합니다."

**Adobe (Firefly/Research):** "Grounded, region-controllable editing이 정확히 ZIM과 grounding이 Firefly와 만나는 곳입니다 — generative fill을 위한 precise mask, region-grounded edit program. 저는 이 역할이 요구하는 PyTorch 엄밀함과 creative-product 감각을 가졌습니다."

**ByteDance Seed:** "Vision generative foundation + controllable perception이 제 matting/segmentation foundation 작업이 꽂히는 곳입니다 — SeedEdit/Seedream 스타일 editing은 정확히 제가 만드는 boundary와 region 품질을 필요로 하고, 저는 빠르게 움직입니다."

**Mistral AI:** "open-weight 신념을 중시합니다 — 저도 같은 이유로 작업 대부분을 오픈소스했습니다. 제 기여는 대부분 LLM 중심인 lab에 엄밀한 multimodal/perception grounding을 가져오는 것이고, take-home-plus-defense 형식이 편합니다 — 제 작업을 방어하는 게 제가 하는 일이니까요."

**Microsoft MSR:** "MSR의 job-talk, agenda-driven 모델이 제가 이미 일하는 방식과 맞습니다 — 저는 학계와 제품을 넘나들며 협업하면서 일관된 multi-year agenda(label-efficient → continual → foundation → grounded)를 돌렸습니다. Perception-grounding agenda와 강한 협업 신호를 가져가겠습니다."

**Follow-up:** "첫 해에 무엇을 하고 싶나요?"
**Rebuttal tip:** *그들의* roadmap에 있는 진짜 프로젝트로, 당신 기술이 독특하게 기여하는 것을 대세요 — generic한 "codebase 배우기"가 아니라. 구체성이 fit이 진짜임을 증명합니다.

</div></details>

<details class="qa"><summary>"서울에 있군요. 정말 relocate할 의향이 있나요?" / visa & 물류"</summary>
<div class="qa-body">

**Testing:** 물류적 진지함 — recruiter는 여기서 늦게 흔들리는 지원자를 죽입니다.

**Answer A — 명확한 commitment.**
"네 — CV에 relocation 가능하다고 썼고 진심입니다. timing, visa sponsorship, KAIST PhD 물류를 열린 자세로 논의하고 싶습니다; PhD를 이동과 양립하도록 구성했고, 아무것도 애매하게 남지 않도록 start date를 맞추고 싶습니다. 역할이 remote-friendly거나 hybrid라면 그것도 유연합니다."

**Answer B — concise.**
"relocate 전적으로 가능합니다; visa timeline과 제 PhD 완료를 조율해 현실적 start date를 정하고 싶을 뿐입니다. 이동 자체엔 망설임 없습니다."

**Follow-up:** "PhD를 먼저 끝내야 하나요?"
**Rebuttal tip:** 진짜 계획을 가지세요 — remote 완료, defense timeline, 또는 advisor 유연성. "병행이고 continuity를 계획해 뒀습니다"가 모호한 "어떻게든 하죠"를 이깁니다.

</div></details>

<details class="qa"><summary>"자신을 researcher로 보나요, engineer로 보나요?"</summary>
<div class="qa-body">

**Testing:** 자기 인식과 특정 track(RS vs AS vs MLE)에의 fit. 잘못된 self-label은 loop에 mismatch를 만듭니다.

**Answer A — applied-scientist 정체성.**
"Applied scientist, 진짜로 둘 다입니다. 저는 실제 실험 — ablation, baseline, 정직한 failure analysis — 으로 accuracy–cost Pareto를 찾*고*, last mile을 닫습니다: ONNX export 이슈, latency budget, production 핸드오프. 제 논문은 제품 병목에서 나오고 제 제품은 학문적 엄밀함을 담습니다. 역할이 순수 long-horizon RS라면 research depth를 할 수 있고; AS라면 이미 그 hybrid를 살고 있습니다. 저는 title보다 'ships rigorous research'로 판단받고 싶습니다."

**Answer B — concise.**
"의도적으로 둘 다입니다. 7편의 1저자 논문이 researcher라 말하고; ship된 foundation model, production API, 인증 auth component가 engineer라 말합니다. Applied science가 그 교집합이고, 의도적입니다."

**Follow-up:** "어느 쪽을 더 즐기나요?"
**Rebuttal tip:** 정직하되 role-aware: RS 패널엔 research로, AS/MLE엔 shipping으로 기울되, 항상 "둘 사이의 loop가 제가 진짜 사랑하는 것"으로 되돌리세요. 어느 절반도 부정하지 마세요.

</div></details>

---

## E · Hard, curveball & weakness probes

offer와 near-miss를 가르는 라운드입니다. 메타 규칙: **그들이 하기 전에 약점을 먼저 대고, 보완하는 강점이나 계획을 보여라.** 방어적 태도가 유일한 치명적 답입니다.

<details class="qa"><summary>"작업이 incremental해 보입니다 — segmentation variant가 많네요. 큰 아이디어는 어디 있나요?"</summary>
<div class="qa-body">

**Testing:** 흔들리지 않고 작업의 *significance*를 방어할 수 있는가 — 의도된 도발.

**Answer A — 문제-재정의로 reframe (confident).**
"'variant'에 반박하겠습니다. 각각이 *병목을 재정의*했고, 그건 incremental의 반대입니다. PointWSSIS는 segmenter를 튜닝한 게 아니라 — semi-supervised instance segmentation에서 진짜 병목이 mask head가 아니라 proposal 단계임을 보였습니다. ECLIPSE는 continual-learning 트릭을 더한 게 아니라 — distillation을 완전히 버리고 freezing + prompting으로 그걸 이길 수 있음을 보여, 분야의 가정을 뒤집었습니다. ZIM은 matting foundation에서 *data granularity가 architecture를 지배*함을 보여 ICCV Highlight(상위 ~3%)를 받았는데, reviewer가 그것을 fine-tune이 아니라 재구성으로 봤기 때문입니다. 통합하는 큰 아이디어는 '진짜 병목을 정의하고 data+architecture 공동 설계로 풀어라'입니다. 그건 variant 더미가 아니라 research 철학입니다."

**Answer B — arc가 큰 아이디어.**
"큰 아이디어는 어떤 단일 논문이 아니라 *arc*입니다: perception을 값싼 supervision에서 신뢰할 수 있는 reasoning으로 사다리 위로 몰아가는 것. 개별적으로 각 논문은 날카로운 기여이고; 함께 보면 perception의 가치가 *label-efficient, updatable, verifiable*에 있다는 일관된 베팅입니다. 그게 정확히 분야가 지금 grounded agent로 하고 있는 베팅입니다."

**Follow-up:** "좋아요 — 분야가 다 잊어버려도 살아남을 단 하나의 결과는?"
**Rebuttal tip:** 하나를 고르고 commit하세요: "foundation model엔 data granularity가 architecture를 이긴다"(ZIM) 또는 "head를 튜닝하지 말고 병목을 재배치하라"(PointWSSIS). 여기서 우물쭈물하면 비난을 확인시키고; 명확한 선택이 그것을 반박합니다.

</div></details>

<details class="qa"><summary>"당신은 vision 사람인데 이 역할은 LLM/NLP를 건드립니다 — depth가 있나요?"</summary>
<div class="qa-body">

**Testing:** 진짜 gap에 대한 정직한 자기 평가, 그리고 그것을 닫을 수 있다는 증거.

**Answer A — 경계를 소유하고 다리를 보여라.**
"타당합니다 — 제 *출판* depth는 LLM pretraining이나 NLP 이론이 아니라 vision에 있습니다. 하지만 두 가지. 첫째, 제 현재 작업이 그 이음새에 삽니다: grounded VLM과 agentic program synthesis는 LLM planning, tool-use, chain-of-thought, hallucination, evaluation을 reasoning하도록 요구합니다 — 저는 지금 거기서 작동하고 있습니다, 지향이 아니라. 둘째, 제가 뭘 모르는지 정확히 압니다 — RLHF를 유도하거나 tokenizer를 맨바닥에서 만들 수 있다고 냉정하게 주장하기 전에 조심하겠습니다. transformer 내부를 허풍떨다 걸리기보다, 경계를 정직하게 말하고 gap을 빠르게 닫는다는 걸 보이겠습니다(제품 작업을 위해 SAM stack과 diffusion-conditioning 문헌을 독학했습니다)."

**Answer B — T-shape (concise).**
"저는 T-shaped입니다: perception에 깊고, multimodal 전반에 넓고 최신 — attention/transformer, VLM training과 eval, grounding, agent. 제가 원하는 프론티어는 vision-language이고, 거기서 제 depth가 희소한 절반이며 LLM 쪽은 학습 가능하고 이미 제 일상 작업의 일부입니다."

**Follow-up:** "좋아요 — 지금 multi-head attention을 구현하거나 RoPE를 설명해 보세요."
**Rebuttal tip:** loop 전에 *from-scratch primitive를 냉정하게 연습*하세요 — attention(올바른 head split, 1/√d_k scaling, causal mask), KV-cache, LoRA, top-k/top-p, RoPE. 이게 강한 researcher의 #1 탈락 이유입니다. depth에 대한 정직함은 시험당했을 때 뒷받침할 수 있어야만 도움이 됩니다. LLM과 ML-coding 챕터로 준비하세요.

</div></details>

<details class="qa"><summary>"당신 production 작업은 전부 single-domain — image, 한 회사. 일반화할 수 있나요?"</summary>
<div class="qa-body">

**Testing:** breadth risk와 적응성 — 5년 단일 고용주 CV의 진짜 우려.

**Answer A — depth-transfers 논거.**
"두 답. 첫째, '한 회사에서 image' 안에서 저는 실제로 *많은 것*을 걸쳤습니다: weakly-supervised, continual, foundation model, matting, detection, on-device efficiency, biometric security, 그리고 이제 multimodal agent — 다른 sub-problem, metric, 제약. 둘째, 전이되는 기술은 domain이 아니라 *method*입니다: 병목을 정의하고, data와 architecture를 공동 설계하고, 정직한 ablation을 돌리고, production last mile을 닫는 것. 그 method가 저를 CAM에서 foundation model로, agent로 마찰 없이 옮겼고, video, 3D, 또는 새 modality로도 똑같이 옮길 것입니다. 인턴십에서의 초기 노출 — OCR, autonomous-driving perception, face — 도 있어서 한 줄이 시사하는 것만큼 좁지 않습니다."

**Answer B — concise.**
"단일 회사이지만 단일 문제가 아닙니다 — image tag에서 foundation model로, agent로 갔습니다. 상수는 research method이고, 그게 전이되는 것입니다. 새 domain은 정의할 또 하나의 병목으로 다루겠습니다."

**Follow-up:** "왜 지금까지 회사나 domain을 한 번도 안 바꿨나요?"
**Rebuttal tip:** 안정성을 관성이 아니라 *얻은 depth*로 framing하세요: "머무른 덕에 완전한 research-to-product loop를 처음부터 끝까지 반복해 돌렸습니다 — 드물고 값진 일입니다. 이제 domain과 scale 점프를 원합니다." 선택으로 소유하세요.

</div></details>

<details class="qa"><summary>"실패한 research bet 하나 말해보세요."</summary>
<div class="qa-body">

**Testing:** 정직함, 성찰, research 판단 — 성공을 실패로 분장하면 라운드를 망칩니다.

**Answer A — 진짜 실패 + 교훈 (STAR-lite).**
"초기에 저는 weakly-supervised instance seg를 위한 *offline iterative* pseudo-label refinement에 과투자했습니다 — refinement round가 많을수록 label이 좋아진다는 직관이었죠. 거의 효과가 없었고 compute를 많이 썼습니다; 이득이 거의 즉시 평평해졌습니다. 그 교훈이 BESTIE와 PointWSSIS를 재구성했습니다: 더 싸고 실제로 작동한 **online, per-mini-batch self-refinement**로 바꿨고, 비싼 버전을 만들기 전에 *가정을 싸게 먼저 검증*하는 걸 배웠습니다. 이제는 엔지니어링을 투입하기 전에 아이디어를 죽일 수 있는 가장 작은 ablation을 먼저 prototype합니다."

**Answer B — framing/scope 실패 (concise).**
"한때 on-device 모델에서 quantize-first 경로를 쫓았습니다 — 가장 빠른 레버라서 — 그런데 무엇보다 먼저 boundary quality(머리카락, 손가락)를 죽였습니다. 그 실패가 compression의 *순서*가 중요함을 가르쳤습니다: resolution/width/decoder와 distillation 먼저, quantization 마지막. 이제 모든 efficiency 프로젝트에 가져가는 규칙입니다."

**Follow-up:** "지금이라면 뭘 다르게 하겠습니까?"
**Rebuttal tip:** 답이 *곧 다르게*입니다 — "가장 싼 실험으로 아이디어를 먼저 죽여라." 당신의 실패에 "인내를 배웠다"가 아니라 구체적이고 전이 가능한 규칙이 붙어 있는지 확인하세요.

</div></details>

<details class="qa"><summary>"논문을 많이 내네요. 여기서 실제로 ship할 건가요, 아니면 그냥 논문을 쫓을 건가요?"</summary>
<div class="qa-body">

**Testing:** RS 우려의 반대면 — 출판 습관이 제품 delivery를 밀어내는가. Product-AI 조직(Apple AIML, MS MAI, ByteDance applied)에서 흔합니다.

**Answer A — shipping 증거.**
"제 출판은 shipping *에서* 나옵니다, shipping 대신이 아니라. Production의 foreground API, 인증 서비스의 FaceSign, ~10ms의 on-device 모델, CLOVA-X에 통합된 ZIM — 그것들은 논문이 아니라 ship된 것입니다. 저는 엄밀함이 제품을 낫게 하고 둘이 정렬되는 문제를 골랐기에 출판합니다. 저는 paper-vs-product 긴장이 없습니다; 하나를 다른 하나로 바꾸는 습관이 있습니다. 한 분기가 순수 delivery를 요구하면 delivery하겠습니다 — 출판한 논문보다 훨씬 많은 미출판 제품 작업을 했습니다."

**Answer B — concise.**
"비율을 보세요: 출판한 모든 것을 ship하거나 오픈소스했고, ship했지만 출판 안 한 것도 많습니다(API, FaceSign). 논문은 제품 작업을 엄밀하게 한 부산물이지 경쟁 목표가 아닙니다."

**Follow-up:** "1년간 출판하지 말라고 하면요?"
**Rebuttal tip:** 깔끔하게 수용하세요 — "제품과 impact가 먼저입니다; 이미 엄격한 non-disclosure 하에 일해봤습니다(FaceSign)." 어떤 망설임도 팀보다 자기 CV를 우선한다고 읽힙니다.

</div></details>

<details class="qa"><summary>"공백이 있네요: 몇 년간 논문이 없거나 뜸한 시기. 무슨 일이 있었나요?"</summary>
<div class="qa-body">

**Testing:** 겉보기 공백이 문제를 숨기는가 — 보통은 아니지만, 면접관은 당신이 움찔하지 않는 걸 보고 싶어합니다.

**Answer A — 대외비 작업 설명.**
"'조용한' 구간은 *출판 불가* 제품 작업이 가장 무거웠던 때입니다 — FaceSign은 출판할 수 없는 정부 인증 보안 제품이고, foreground API와 on-device 모델은 내부입니다. 그래서 논문 timeline이 실제 output을 과소 계산합니다. 저는 내내 shipping하고 있었습니다; 가장 힘든 작업 일부가 단지 Google Scholar에 안 뜰 뿐입니다."

**Answer B — concise.**
"output에는 공백이 없고 — *publishable* output에 공백이 있습니다. 인증-보안과 내부-제품 작업(FaceSign, API)이 그 시기를 채웠고 본질상 출판할 수 없습니다."

**Follow-up:** "볼 수 없는 impact를 제가 어떻게 검증하죠?"
**Rebuttal tip:** 확인 가능한 것을 제공하세요 — DAN 24 talk, public FaceSign 제품 페이지, "수백만이 사용" 규모, 그리고 reference — 대외비 숫자는 쥐고서. "public 증거와 reference를 가리켜 드릴 수 있고; 내부 지표는 내부로 남습니다"가 신뢰 가는 문장입니다.

</div></details>

<details class="qa"><summary>"Rapid-fire fundamentals: BatchNorm vs LayerNorm — 왜 transformer에 LN인가요?"</summary>
<div class="qa-body">

**Testing:** 압박 하의 breadth. fundamentals에서 한두 번 헛발을 디디면 강한 loop도 가라앉을 수 있습니다 — 이건 breadth 라운드 전체의 대리입니다.

**Answer A — 명료한 메커니즘.**
"BatchNorm은 feature별로 batch 차원을 가로질러 normalize하므로 batch 통계에 의존합니다 — 가변 길이 sequence와 작은 batch에 나쁘고, example들을 결합시킵니다. LayerNorm은 token별로 feature 차원을 가로질러 normalize하고, batch와 sequence 길이에 독립적입니다 — 정확히 transformer가 필요로 하는 것: batch 크기나 sequence 위치와 무관한 per-token 안정성, 그리고 train과 inference에서 동일하게 작동(running stat 없음). 그래서 transformer가 LN을 쓰고 — 그래서 RMSNorm(mean-centering 없는 LN)이 효율을 위해 이제 흔합니다."

**Follow-up chain:** "Pre-norm vs post-norm? 왜 pre-norm이 더 안정적으로 학습하나요?"
**Rebuttal tip:** 이 질문은 *breadth 라운드 전체의 probe*입니다 — fundamentals(optimization, normalization, attention, RoPE, scaling law, RLHF, diffusion, MoE)를 과하게 준비하라는 신호로 다루세요. 전문 분야의 depth가 breadth gap을 구하지 못합니다. foundations와 LLM 챕터를 냉정하게 파세요.

</div></details>

<details class="qa"><summary>"Curveball: 1년과 무제한 compute가 있다면 무엇을 만들겠습니까?"</summary>
<div class="qa-body">

**Testing:** 비전, 야망, 그리고 당신의 research taste가 incremental 논문을 넘어 scale하는가.

**Answer A — 일관된 moonshot.**
"agent를 위한 **self-diagnosing perception stack**: box나 mask만 출력하는 게 아니라 *calibrated uncertainty와 typed failure signal*을 보고하는 vision tool, 그리고 그 신호를 소비해 언제 신뢰·재실행·수리할지 결정하는 reasoning layer. 오늘날 agent는 tool이 자신 있게 거짓말해서 silent하게 실패합니다. perception이 '이 occlusion은 확신이 없다'거나 '이 depth는 신뢰할 수 없다'고 말할 수 있다면, agent는 *task-specific training 없이* frontier level에서 3D와 시간을 검증 가능하게 reasoning할 수 있습니다. 그게 제 arc의 자연스러운 꼭대기입니다 — label-efficient perception에서 *trustworthy* perception으로 — 그리고 무제한 compute면 diagnostic data와 tool ensemble을 대규모로 만들 수 있습니다."

**Answer B — concise.**
"Verifiable multimodal reasoning: 자신의 failure mode를 노출하는 perception tool, 그리고 그 주위를 수리하는 agent. grounding과 agent가 수렴하는 곳이고, 실제 배포를 막는 trust 문제입니다."

**Follow-up:** "첫 실험은 뭘 돌리겠습니까?"
**Rebuttal tip:** 구체적인 week-one 실험을 가지세요 — "기존 VisProg/VADAR pipeline을 계측해 3D benchmark에서 모든 tool의 silent error를 로깅하고, 최종 error 중 얼마가 silent-perception 대 planning인지 측정." 첫 발 없는 비전은 몽상으로 읽힙니다.

</div></details>

---

## Cheat-sheet

| Situation | The move | The one-liner |
| --- | --- | --- |
| "Walk me through your research" | **through-line**을 주고, 파고들게 두라 | "pixels → labels → continual → foundation → grounded reasoning" |
| 대외비 작업 (FaceSign, API, 진행 중) | **Decline-then-deliver** framing | "framing은 공유할 수 있고; 내부 지표는 대외비." |
| "당신 작업은 incremental하다" | **bottleneck redefinition**으로 reframe | "각 논문이 head를 튜닝한 게 아니라 병목을 재배치했다." |
| "vision 사람인데 / LLM depth?" | **경계를 소유**하고, 다리를 보이고, *냉정하게 증명* | "T-shaped; vision 절반이 희소한 쪽." |
| 공저 논문 (SSUL) | 구체적 **I vs we**, 관대하게 credit | 내 몫을 대고, 그들 몫을 대고, 흐리지 말라. |
| "왜 NAVER를 떠나나?" | **Pull, not push** | "NAVER에서 멀어지는 게 아니라 frontier-scale로." |
| Rapid-fire fundamentals | breadth를 과하게 준비; 전문성이 gap을 못 덮는다 | 한 번의 헛발이 loop를 가라앉힐 수 있다. |
| 실패한 bet | 진짜 실패 + **전이 가능한 규칙** | "가장 싼 실험으로 아이디어를 먼저 죽여라." |
| 큰 아이디어 도전 | **하나**의 durable한 결과에 commit | "Data granularity가 architecture를 이긴다" (ZIM). |
| 모든 답변 | 방에 맞는 **각도**를 골라라; 암기 말고 내면화 | FAIR엔 depth, HM엔 impact, recruiter엔 30초. |

**Golden rules:** (1) 숫자를 절대 지어내지 말라 — 메커니즘을 설명하거나 대외비라고 하라. (2) *당신*의 기여를 명확히 하라. (3) 그들이 하기 전에 약점을 대라. (4) [Company Playbooks](#/process/companies)로 모든 답변을 맞춤화하라.

## Cross-links

- Deep-dive (위 모든 답변의 원자재가 되는 기술 material): [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [FaceSign](#/resume/facesign) · [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM/Agents](#/resume/grounded-vlm-agents)
- [Your CV → Interview Map](#/resume/overview) — 회사별로 어떤 프로젝트로 열지
- [STAR & The Story Bank](#/behavioral/star) — failure/conflict/ownership 질문의 behavioral framing
- [Company Playbooks](#/process/companies) — 각 타깃 회사가 파고드는 것, 모든 답변을 다시 조준하려면
