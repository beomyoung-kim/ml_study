# 예상 질문 & 답변 초안

<div class="tag-row"><span class="tag">research narrative</span><span class="tag">project probes</span><span class="tag">research→product</span><span class="tag">motivation</span><span class="tag">curveballs</span><span class="tag badge-new">flagship</span></div>

> [!TIP] 이 챕터가 중요한 이유
> 이 chapter는 CV fact check와 research deep-dive를 함께 준비하는 질문 은행입니다. 면접관은 주장·본인 기여·설계 선택·한계를 교차 확인할 수 있습니다. 각 질문에는 가능한 의도, **검증 전 답변 초안**, follow-up, 방어 포인트가 있습니다. 문장을 외우지 말고, 실제 증거와 공개 가능한 trade-off로 다시 쓰세요.

> [!IMPORTANT] 답변 초안이지 사실 원장이 아닙니다
> 아래 문장은 이력서·공개 논문·로컬 제출본에서 만든 rehearsal draft입니다. 이력서에 명시된 ownership—ZIM을 만들었다는 사실, FaceSign anti-spoofing model 구축, on-device model의 독자적 개발—은 그대로 활용하되, 그보다 세부적인 역할·제품 요구사항·갈등·내부 비교·수치는 지원 전에 직접 확인하세요. 확인되지 않은 서사는 삭제하고, 심사 중 작업은 항상 **under review**라고 밝히며, 방법론·평가 설계·trade-off도 승인된 공개 범위 안에서만 설명합니다.

## 이 답변들을 쓰는 법

<div class="proscons"><div><div class="pros-t">Do</div>
<b>암기하지 말고 내면화하라.</b> 각 답변의 <i>척추</i> — 주장, 근거, trade-off — 를 익힌 다음, 자기 말로 실시간 재구성하세요. 암송한 답변은 첫 follow-up에서 죽고, 이해한 답변은 열 번을 버팁니다.
<br><br>
<b>라운드 목적에 맞는 각도를 골라라.</b> technical deep-dive는 mechanism·ablation을, hiring-manager 대화는 impact·scope를, recruiter 대화는 짧은 사실과 물류를 더 물을 수 있습니다. 실제 agenda를 확인하고 개인이나 회사 이름으로 질문 성향을 단정하지 마세요.
</div><div><div class="cons-t">Don't</div>
<b>과장하지 마라.</b> 대외비 작업(FaceSign, foreground API, 진행 중/심사 중)에 대해서는 지어낸 숫자가 아니라 <b>framing과 impact</b>로 말하세요: <i>"framing은 공유할 수 있지만, 구체적 내부 지표는 대외비입니다."</i> 이 문장은 회피가 아니라 integrity로 읽힙니다.
<br><br>
<b>"I"와 "we"를 흐리지 마라.</b> 협업자를 인정하되, <i>당신</i>의 실제 기여를 명확히 하세요. 빈도 순위를 단정할 수는 없지만, ownership을 평가할 근거가 사라지는 답은 약한 signal입니다.
</div></div>

**회사별로 맞춤화**하려면 [Company Playbooks](#/process/companies)를 출발점으로 쓰되, 실제 job description·최근 공개 연구·recruiter가 확인한 loop를 기준으로 다시 조준하세요. 회사 이름만으로 면접 형식이나 가치 가중치를 단정하지 않습니다.

원자재가 되는 deep-dive: [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [Phoenix](#/resume/phoenix-mask-refinement) · [FaceSign](#/resume/facesign) · [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM/Agents](#/resume/grounded-vlm-agents) · [Spatial-Reasoning Agent (심사 중)](#/resume/neurips26-spatial-reasoning). 라운드별 짧은 답변은 [이력서 기반 단계별 예시 답변](#/resume/interview-stage-answers), behavioral framing은 [STAR & The Story Bank](#/behavioral/star), 전체 맵은 [개인 이력서 → 인터뷰 맵](#/resume/overview)을 사용하세요.

---

## A · Research narrative & motivation

Research-track 대화에서 자주 나오는 오프너입니다. 논문 목록을 넘어 일관된 문제 선택과 다음 방향을 설명할 수 있는지 봅니다.

<details class="qa"><summary>"Walk me through your research."</summary>
<div class="qa-body">

**Testing:** 14+편의 논문을 관통선이 있는 일관된 arc로, 적절한 고도에서, ~2–3분 안에 압축할 수 있는가 — 그러면서 *당신*의 역할을 명확히 하면서.

**Answer A — 관통선 (technical narrative).**
"하나의 실이 있습니다: **실세계 제약 하에서 visual perception을 정확하고 신뢰할 수 있게 만들기.** 저는 **weakly/semi-supervised segmentation**에서 시작했습니다 — DRS, BESTIE, PointWSSIS — perception 배포의 병목이 label이기 때문에, label-efficiency 프론티어를 공략했습니다: image tag나 몇 개의 point로 instance mask를 어떻게 얻을까. 그다음 **continual learning** — SSUL과 ECLIPSE — 배포된 모델은 forgetting이나 옛 데이터 저장 없이 새 class를 흡수해야 하기 때문입니다. 그다음 ZIM으로 **foundation model**까지 규모를 키웠습니다 — SAM 위에 세운 promptable zero-shot matting model입니다. 이 모든 것에서 패턴은 같습니다: *진짜 병목을 정의한 다음, brute force가 아니라 data + architecture 공동 설계로 푼다.* 가장 최근에는 stack 위쪽으로 올라가 **grounded VLM과 training-free visual reasoning agent**로 갔습니다 — 언어 reasoning을 픽셀- 및 영역-level 증거에 연결하고, perception tool이 언제 silent하게 실패하는지 진단합니다. 그래서: 픽셀 → label → continual → foundation → grounded reasoning. 같은 집착, 더 높은 추상화."

**Answer B — impact-first (hiring manager용).**
"두 문장 버전: 저는 ship되는 perception을 만듭니다. 7편의 first/co-first-author 주요 학회 논문, ZIM의 ICCV 2025 Highlight, 그리고 연구를 NAVER Cloud 제품으로 옮긴 기록이 있습니다 — foreground-segmentation API, 인증 face-auth service의 anti-spoofing, 약 10 ms on-device segmenter입니다. 공개 가능한 범위에서 보면 일관된 고리는 연구의 엄밀함과 product 제약이 서로 문제를 더 선명하게 만든다는 점입니다."

**Answer C — concise/executive (30초).**
"저는 computer-vision applied scientist입니다. 제 arc는 label-efficient·continual segmentation에서 segmentation/matting foundation model(ZIM, ICCV 2025 Highlight)로, 이제 grounded vision-language model과 visual reasoning agent로 이어집니다. 상수는 정확성·효율·검증 가능성을 실제 product 제약과 함께 다루는 것입니다."

**Follow-up:** "이 중 무엇을 최고작으로 보고, 왜인가요?"
**Rebuttal tip:** *reach*로는 ZIM(foundation + Highlight + product), *elegance*로는 PointWSSIS(proposal bottleneck 재정의)를 대되 — venue가 아니라 **아이디어**로 정당화하세요. "Best" = 인용 수 최고가 아니라 가장 명확한 문제 재정의.

</div></details>

<details class="qa"><summary>"논문들을 잇는 실은 무엇인가요? 아니면 그냥 팀에 필요했던 것들인가요?"</summary>
<div class="qa-body">

**Testing:** research *taste* — 문제를 고르는가, 아니면 문제가 당신을 고르는가? 냉소적 framing은 미끼이니 방어적으로 굴지 마세요.

**Answer A — 지적 실.**
"실은 **싸거나 약한 신호 → 신뢰할 수 있는 structured perception**입니다. DRS와 BESTIE: image tag → instance mask. PointWSSIS: 몇 개의 point → 완전한 mask. ECLIPSE: frozen model + 작은 prompt → forgetting 없이 새 class. ZIM: coarse SA-1B label → label converter를 통한 fine alpha matte. agent 작업조차 stack 꼭대기에서의 같은 수(move)입니다: noisy specialist tool 출력 → 검증 가능하고 자기 진단하는 reasoning program. 저는 일관되게 *감당할 수 있는 supervision*과 *제품이 요구하는 fidelity* 사이의 gap에 끌리고, 그것을 data + architecture 설계로 메웁니다."

**Answer B — 환경에 대해 정직 (제약을 소유).**
"산업 환경과 연구 관심이 모두 영향을 줬습니다. 다만 각 논문이 특정 product pain에서 나왔는지, 제가 problem framing을 소유했는지는 프로젝트별 기록으로 확인하겠습니다. 확인 가능한 공통점은 label cost·continual update·boundary fidelity 같은 실용 제약을 논문에서 일반화 가능한 문제로 평가했다는 점입니다."

**Follow-up:** "그럼 완전한 자유가 있다면 다음에 어떤 문제를 고르겠습니까?"
**Rebuttal tip:** 진짜 답을 준비하세요 — "verifiable spatial reasoning: perception tool이 자신의 uncertainty를 보고해서 agent가 언제 그것을 불신할지 알게 하는 것." 그 실이 retrofit이 아니라 미래 지향적임을 증명합니다.

</div></details>

<details class="qa"><summary>"perception에서 grounded VLM과 agent로 pivot한 이유는? 유행을 쫓는 것 아닌가요?"</summary>
<div class="qa-body">

**Testing:** 그 이동이 기회주의적(LLM hype를 따라감)인가 원리적 연속인가. 2025–26 vision 채용에서 *바로 그* 서사 질문입니다.

**Answer A — pivot이 아니라 연속 (depth).**
"perception에서 벗어나는 pivot이 아니라 — perception이 새 stack에서 자기 역할을 찾은 것입니다. End-to-end VLM은 강하지만 **hallucinate**하고 *근거 없는 텍스트 설명*을 만듭니다: 답이 맞으면서도 visual 증거가 틀릴 수 있습니다 — spurious success. 제 배경 전체가 high-fidelity, region- 및 pixel-level perception입니다. 그게 정확히 빠진 재료입니다: 언어 reasoning을 검증 가능한 visual 증거에 grounding하고, agent에게 *specialist tool*(ZIM 품질의 mask나 depth 같은)을 줘서 3D와 시간을 reasoning하게 하는 것. 흥미로운 새 문제는 그 tool들이 **silent하게 실패**한다는 것입니다 — 그럴듯하지만 틀린 box나 mask를 예외 없이 반환합니다 — 그래서 제 현재 작업은 그 silent failure를 *typed diagnosis*로 바꿔 targeted program repair를 구동합니다. 저는 perception을 떠나는 게 아니라, 그것을 reasoning 아래의 신뢰할 수 있는 tool layer로 만드는 겁니다."

**Answer B — 시장/제품 논리 (impact).**
"Image editing, robotics, UI agent 같은 application에서는 언어 query를 region·object·action에 연결하고 결과를 검사해야 합니다. End-to-end model과 modular tool use 중 무엇이 나은지는 task마다 다르지만, 제 perception 연구는 grounding과 tool reliability를 분석할 기반을 줍니다."

**Answer C — concise.**
"Perception 성능과 함께 그 결과를 근거에 연결하고 실패를 탐지하는 문제가 중요해졌습니다. Grounding과 agentic tool use는 그 문제를 다루는 설계 축이고, 제 perception depth가 연결점입니다."

**Follow-up:** "frontier VLM이 계속 좋아지면, 당신의 modular/agent 접근은 obsolete되지 않나요?"
**Rebuttal tip:** end-to-end의 강점을 인정한 뒤 modular 접근의 잠재 이점과 비용을 함께 대세요: 정밀 측정·검사 가능한 trace·module upgrade 대 orchestration 비용·tool-error accumulation. Hybrid는 비교할 설계안이지 자동적인 default가 아닙니다.

</div></details>

<details class="qa"><summary>"풀타임으로 일하며 PhD를 했군요. 그럼 연구는 side project 아닌가요?"</summary>
<div class="qa-body">

**Testing:** 엄밀함과 독립성 — 산업 job이 "진짜" 일이고 PhD는 형식인가, 아니면 둘이 진짜로 하나의 프로그램인가.

**Answer A — 통합 (reframe).**
"둘은 별도 일정이지만 문제 선택과 검증 방식이 서로 강화되는 프로그램입니다. Product 경험은 labeling cost·category drift·latency·boundary quality 같은 현실 제약을 보여주고, 연구 훈련은 ablation·baseline·failure analysis를 강제합니다. 7편의 first/co-first-author 논문과 ICCV Highlight, 그리고 실제 배포 경험을 동시에 설명할 수 있다는 것이 제 강점입니다. 각 논문이 특정 제품에서 직접 나왔다는 주장은 사실인 경우에만 연결하겠습니다."

**Answer B — 독립성 신호.**
"First/co-first-author 표기는 공개 사실이지만 기여 범위를 자동으로 증명하지는 않습니다. 프로젝트별로 architecture·data·evaluation·writing 가운데 제가 실제로 소유한 항목과 공동저자가 맡은 항목을 표로 정리해, 확인 가능한 결정만 구체적으로 답하겠습니다."

**Follow-up:** "시간과 두 우선순위 세트를 어떻게 관리하나요?"
**Rebuttal tip:** 고생을 미화하지 마세요. *정렬*로 framing하세요 — 제품 필요와 publishable question이 겹치는 문제를 골라서 비용을 두 번 치르지 않는다고. 그건 근성이 아니라 판단입니다.

</div></details>

---

## B · Per-project probes

면접관은 프로젝트 하나 — 보통 ZIM이나 그들 팀과 맞는 것 — 를 골라 당신이 ownership을 보이거나 무너질 때까지 파고들 것입니다. 완전한 기술적 depth는 deep-dive에 있고; 여기는 *entry* probe와 인터뷰 고도에서 답하는 법입니다.

<details class="qa"><summary>ZIM — "matting 데이터로 그냥 SAM을 fine-tune하면 안 됐나요?"</summary>
<div class="qa-body">

**Testing:** 논문의 핵심 motivation과 public-matte baseline의 failure mode를 정확히 설명할 수 있는가.

**Answer A — failure mode (technical).**
"논문의 public-matte baseline에서는 micro-level generalization이 크게 저하됩니다. 공개 matting data가 macro target에 치우쳐 있기 때문입니다. 같은 architecture 비교에서 fine-grained SAD가 약 120 대 SA1B-Matte 학습 약 10으로 보고되어, data granularity가 load-bearing임을 보여줍니다. 그래서 SA-1B-derived coarse mask를 fine alpha matte로 바꾸는 **label converter**와 SGA/STL을 사용했습니다. 이미지 수와 mask-instance 수는 섞지 않고, 공개 논문이 밝힌 training 1%·약 2.2M matte label을 단위와 함께 말합니다."

**Answer B — 설계 원리 (concise).**
"문제가 loss가 아니라 data 계약이기 때문입니다. SAM의 supervision은 coarse하고 hard-mask이며; matting은 soft한 micro-level boundary가 필요합니다. Macro portrait 데이터에 fine-tune하면 promptable zero-shot behavior가 파괴됩니다. 우리는 converter로 SA-1B 규모에서 micro-granular matte label을 생성했고, 실제로 fine detail을 렌더링하기 위해 hierarchical pixel decoder와 prompt-aware masked attention을 더했습니다. 그래서 zero-shot을 유지*하면서* matte를 만듭니다."

**Follow-up:** "hierarchical pixel decoder가 구체적으로 무엇을 벌어다 주나요?"
**Rebuttal tip:** SAM의 stride-4 decoder + 순진한 upsample은 **checkerboard artifact**를 유발하고 fine 구조를 잃습니다; multi-level(stride 2/4/8) 융합이 ~10ms overhead로 high-res detail을 복원합니다. 그 숫자와 메커니즘을 준비해 두세요 — [ZIM](#/resume/zim) 참고.

</div></details>

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
"step 1 이후 Mask2Former parameter를 freeze하면 기존 weight drift를 막아 forgetting을 줄이지만, 최종 old-class output까지 정확히 보존되지는 않습니다. 새 prompt와 aggregation에서 error propagation이 생기고 plasticity가 제한됩니다. 논문은 deep visual prompt, stronger frozen initialization, 그리고 변화하는 no-object 의미를 다루는 logit manipulation으로 이 trade-off를 보완합니다. 공개 ADE20K protocol과 약 1.3% trainable parameter 결과 안에서 말하겠습니다."

**Answer B — 제품 framing (impact).**
"제품 가설로는 old-data replay나 전체 retraining이 어려운 환경에서 새 class adapter를 추가하는 시나리오가 있습니다. 다만 논문 결과만으로 실제 API 배포·on-device 적합성·privacy compliance가 증명되지는 않습니다. 해당 제약이 JD에 있을 때만 trade-off 가설로 연결하겠습니다."

**Follow-up:** "왜 semantic이 아니라 panoptic continual learning인가요?"
**Rebuttal tip:** panoptic에는 semantic prediction에 더해 instance matching, stuff, 변화하는 no-object 정의가 들어가며 PQ는 recognition failure에도 민감합니다. 이를 단순한 난이도 서열보다 추가되는 문제 구조로 설명하세요. [ECLIPSE](#/resume/eclipse) 참고.

</div></details>

<details class="qa"><summary>PointWSSIS — "실제 통찰이 뭔가요? 그냥 또 다른 semi-supervised 트릭 같은데요."</summary>
<div class="qa-body">

**Testing:** 문제를 재정의했는가, 아니면 그냥 튜닝했는가 — 강한 CVPR 논문과 incremental 논문의 차이.

**Answer A — 재정의 (depth).**
"통찰은 semi-supervised instance segmentation의 큰 병목 중 하나가 proposal 단계라는 것입니다. Threshold를 낮추면 false positive가 늘고 높이면 instance를 놓칠 수 있습니다. Point label은 위치 정보를 더해 proposal ambiguity를 줄이고, APS는 scale 선택을, MaskRefineNet은 rough mask refinement를 담당합니다. 논문은 COCO 5% full setting에서 비교 baseline 24.9 대비 33.7 AP, 50% setting에서 fully supervised reference 39.7 대비 38.8을 보고합니다. Point가 오류를 완전히 제거한다고 말하지 않고 protocol과 label budget을 함께 인용합니다."

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

<details class="qa"><summary>Phoenix — "adversarial 공격을 데이터 생성에 쓴다니 — 그냥 형태학적 잡음의 더 화려한 버전 아닌가요?"</summary>
<div class="qa-body">

**Testing:** 기존 refiner가 *왜* 부족한지, 그리고 adversarial 재해석이 트릭이 아니라 원리적 선택임을 방어할 수 있는가 — ECCV'26 논문의 핵심 동기.

**Answer A — failure mode + 원리 (technical).**
"형태학적 noise는 image나 현재 model과 무관하게 GT 경계를 변형하지만, AMP는 현재 decoder loss를 키우는 embedding-space direction으로 hard perturbation을 만듭니다. 논문의 edge-correlation 분석과 ablation은 morphology-only noise와 다른 분포·효과를 보고합니다. 다만 gradient magnitude를 calibrated predictive uncertainty와 동일시하지 않습니다. Guidance mask로 expansion/contraction/inversion 유형을 제어하고, CMRL은 GT·noisy·refined mask 관계를 학습합니다."

**Answer B — impact/계보 (concise).**
"PointWSSIS의 MaskRefineNet에서 다룬 pseudo-label-quality 문제를 refiner의 training-pair 생성으로 일반화한 작업입니다. AMP와 CMRL을 결합하고, 논문 protocol에서 weak/semi-supervised refinement 최대 +16.1 AP point를 보고합니다. Phoenix·ZIM은 SAM 계열이고 ECLIPSE는 Mask2Former 계열이지만, pretrained trunk를 보존하고 작은 module로 적응한다는 상위 설계 축에서 연결됩니다."

**Follow-up:** "왜 픽셀이 아니라 임베딩 공간을 공격하나요?"
**Rebuttal tip:** 효율 + 의미 수준 교란. 얼린 637M 인코더의 image embedding을 재사용하고 가벼운 decoder만 돌려 갱신 1회 ~6ms이며, 픽셀 노이즈가 아니라 고수준 의미를 흔들어 실제 모델 혼동을 닮습니다. AMP 단독 +4.6, CMRL 단독 +1.5인데 합치면 +6.4로 개별 합보다 큰 **시너지** — 현실적 잡음이 삼방향 contrastive를 더 유의미하게 만들기 때문. [Phoenix](#/resume/phoenix-mask-refinement) 참고.

</div></details>

<details class="qa"><summary>DRS / SSUL — "오래된 논문들이네요. 여전히 유의미한가요, 아니면 초기 커리어 워밍업인가요?"</summary>
<div class="qa-body">

**Testing:** 초기 작업을 무시하기보다 거기서 durable principle을 뽑아낼 수 있는가.

**Answer A — 이어진 원리.**
"그것들이 이후 모든 것의 뿌리입니다. **DRS**(AAAI 2021)는 CAM attention을 *shaping*하는 것 — discriminative 영역을 억제해 activation이 객체 전체로 퍼지게 하는 것입니다. 그 CAM-shaping 직관이 BESTIE의 PAM이 되었습니다. **SSUL**(NeurIPS 2021, co-first)은 class-incremental semantic segmentation에서 **unknown/future class**를 명시적으로 모델링하는 것과 exemplar memory를 도입했습니다 — 그 'background의 의미는 step마다 불안정하다'는 통찰이 *정확히* ECLIPSE의 logit manipulation이 나중에 panoptic에 대해 푼 것입니다. 그러니 워밍업이 아니라; 제 두 오래 이어진 실 — CAM manipulation과 continual segmentation — 이 심어진 곳입니다."

**Answer B — concise.**
"이후 모든 논문이 이 둘로 추적됩니다: DRS → attention shaping → BESTIE; SSUL → 불안정한 background → ECLIPSE. 초기이지만 load-bearing입니다."

**Follow-up:** "SSUL은 co-first-author였는데 — *당신* 몫은 무엇이었나요?"
**Rebuttal tip:** `[내가 맡은 설계·실험·writing] / [공동저자가 맡은 부분] / [공동 결정]`을 원자료로 채우세요. Segmentation modeling이나 unknown-class handling을 본인 몫으로 미리 가정하지 않습니다.

</div></details>

<details class="qa"><summary>FaceSign — "anti-spoofing 모델 얘기 좀 해주세요. 정확도는 얼마나 나왔나요?"</summary>
<div class="qa-body">

**Testing:** face anti-spoofing의 기본 위협·평가를 이해하는가, 그리고 대외비 경계와 실제 ownership을 정확히 다루는가.

**Answer A — decline-then-deliver (본받을 모델).**
"이력서에 명시된 범위에서는 FaceSign 뒤의 anti-spoofing model을 구축했습니다. model 안에서 제가 맡은 세부 범위는 `[확인된 architecture·training·evaluation 범위]`입니다. 공개 승인되지 않은 정확도·attack set·architecture는 공유하지 않겠습니다. 일반적으로 PAD는 print·replay·3D mask 같은 presentation attack을 다루고 digital injection은 별도 pipeline-integrity 계층이 필요합니다. APCER/BPCER 같은 ISO/IEC 30107 계열 metric과 recognition FAR/FRR도 구분해야 합니다."

**Answer B — systems view (concise).**
"공개 이력서는 제가 anti-spoofing model을 구축했다고 명시합니다. 이 model-level ownership과 FaceSign 전체 system·data·deployment ownership은 분리하겠습니다. `[실제 model 결정 1–2개]`는 기록과 공개 범위 안에서 설명하고, 일반 FAS threat model·sensor·operating-point trade-off를 내부 구현 사실처럼 섞지 않겠습니다."

**Follow-up:** "공개된 hardware-co-designed biometric system과 어떻게 비교하겠습니까?"
**Rebuttal tip:** 공개 architecture를 일반 사례로만 사용하고 FaceSign의 sensor·pipeline·본인 역할을 추정하지 마세요. Head-to-head protocol이 없으면 우열을 말하지 않습니다. [FaceSign](#/resume/facesign) 참고.

</div></details>

<details class="qa"><summary>On-device seg — "왜 10ms가 목표인가요? 그냥 서버에서 더 큰 모델을 쓰면 안 되나요?"</summary>
<div class="qa-body">

**Testing:** systems/efficiency reasoning과 on-device가 *왜* 중요한지 이해하는가 — 숫자를 맞췄다는 것만이 아니라.

**Answer A — frame-budget 엔지니어링 (depth).**
"이력서상 저는 이 human-segmentation model을 독자적으로 개발했고 mobile CPU에서 약 10 ms를 달성했습니다. 30 fps pipeline의 약 33 ms budget 중 이 수치가 어느 device·input·runtime·thread·statistic에서 측정됐는지 제 기록으로 붙이겠습니다. CPU target의 실제 제품 이유와 resolution·width·decoder 등 제가 실제로 비교한 lever만 말하고, 각 단계의 target-device quality/latency trade-off를 설명하겠습니다."

**Answer B — two-tier 설계 가설 (impact).**
"Cloud foundation model과 on-device specialist를 분리하면 품질·지연·비용을 다르게 최적화할 수 있습니다. 제 이력서로 확인되는 사실은 독자적 model 개발, mobile-CPU 약 10 ms, ONNX-based in-house serving입니다. ZIM이 teacher였는지, distillation·quantization을 썼는지, phone app에 직접 탑재됐는지는 별도 기록 없이 추정하지 않겠습니다."

**Follow-up:** "ONNX serving이라 했는데 — 정말 폰 CPU에서인가요 아니면 서버인가요?"
**Rebuttal tip:** 정확하게 말하세요, CV 표현이 이 질문을 부르니: *모델*은 mobile-CPU budget에 맞춰 설계·측정되었고; *serving path*는 사내 ONNX 기반 인프라였습니다. 설명할 수 없는 폰 배포를 과장하지 말고 design target과 deployment infra를 구분하세요. [On-Device Seg](#/resume/on-device-segmentation) 참고.

</div></details>

<details class="qa"><summary>Ongoing agents — "NeurIPS 제출이 심사 중이군요. 설명해 주세요."</summary>
<div class="qa-body">

**Testing:** 심사 중 작업의 상태와 공개 범위를 지키면서 문제·메커니즘을 명료하게 설명할 수 있는가.

**Answer A — problem-first framing (안전하고 강한 버전).**
"심사 중이라 비교 결론·model·benchmark·수치는 말하지 않고 승인된 문제와 mechanism만 설명하겠습니다. 설정은 3D spatial reasoning용 visual program synthesis이고, 관심 failure는 tool이 예외 없이 잘못된 box/depth를 반환하는 **silent perception failure**입니다. submission은 structured scene hypothesis와 execution trace를 비교해 **typed diagnosis**를 만들고 targeted repair로 연결합니다. Public lineage는 VisProg·ViperGPT·VADAR이며, 정확한 차별성 주장은 제출본 related-work 범위로 한정합니다."

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

Applied-scientist 역할에서는 research quality와 delivery 경험을 함께 물을 수 있습니다. 공개 연구 결과, 제품 산출물, 본인 ownership을 분리해 증명하세요.

<details class="qa"><summary>"foreground-segmentation API의 내부 비교 주장을 어떻게 방어하겠습니까?"</summary>
<div class="qa-body">

**Testing:** 큰 주장 뒤의 엄밀함과 정직성. 방법론 없는 대담한 benchmark 주장은 red flag이고; 방법론이 있으면 green flag입니다.

**Answer A — verification-first (rigor).**
"이력서에 명시된 제 역할은 foreground-segmentation API의 model과 dataset 개발이며, 현재 사내 serving 중이고 내부 평가에서 Photoroom·Remove.bg·Adobe segmentation API보다 우수했다고 적혀 있습니다. 이 headline claim은 이력서 근거로 말하되, 보편적 우위가 아니라 특정 시점·domain·protocol의 내부 결과라고 한정하겠습니다. 공개 가능한 범위에서 dataset source, sampling date, API version, input/output normalization, metric·human review, confidence interval을 설명하고, 공개할 수 없는 수치나 slice는 지어내지 않겠습니다."

**Answer B — 공개 범위가 좁을 때.**
"이력서에 공개한 범위—model·dataset 개발, 사내 serving, 이름이 적힌 세 API와의 내부 비교—까지는 명확히 말하겠습니다. 다만 protocol이나 수치의 공개 승인이 없다면 즉흥적으로 채우지 않고, foreground quality를 mean IoU뿐 아니라 boundary·transparency·failure slice·latency로 평가해야 하는 이유와 제가 실제로 맡은 data/model 결정을 설명하겠습니다."

**Follow-up:** "이게 ZIM과 같은 모델인가요?"
**Rebuttal tip:** 아니요 — 뭉뚱그리지 마세요. API는 제품 라인(내부 serving)이고; ZIM은 promptable zero-shot foundation model입니다. 공유 DNA(boundary quality, data curation), 다른 artifact. 둘이 한 모델이라 주장하는 건 follow-up에서 풀리는 종류의 과장입니다.

</div></details>

<details class="qa"><summary>"ZIM을 CLOVA-X Image Editing에 넣었군요. 연구가 production을 만났을 때 뭐가 깨졌나요?"</summary>
<div class="qa-body">

**Testing:** 실제로 연구를 ship했는가, 아니면 checkpoint를 담 너머로 던졌을 뿐인가. 흥미로운 답은 마찰입니다.

**Answer A — 핸드오프 마찰 (systems).**
"공개적으로 확인되는 사실은 TEAM NAVER DAN 24 발표와 CLOVA-X Image Editing integration입니다. 제가 직접 맡은 production 범위는 [확인된 ownership]입니다. Matte integration에서 일반적으로 확인할 항목은 alpha representation, resolution, latency, API compatibility, monitoring과 fallback이지만, caching·tiering·threshold를 실제로 사용했다고 증거 없이 말하지 않습니다. 공유 가능한 실제 마찰 하나를 골라 결정과 결과를 설명하겠습니다."

**Answer B — concise (impact).**
"ZIM의 공개 code/demo와 CLOVA-X integration은 확인할 수 있습니다. 그 사이에서 제가 맡은 handoff와 failure hardening은 실제 기록으로 확인한 항목만 말하고, 공개할 수 없으면 일반 checklist와 공개 결과를 구분하겠습니다."

**Follow-up:** "video editing으로 어떻게 확장하겠습니까?"
**Rebuttal tip:** SAM2 스타일 memory + temporal consistency가 방향이고, matting엔 진짜 미해결입니다 — 그렇게 말하세요. Frame 간 flickering alpha가 어려운 부분이니; trivial한 확장인 척하지 마세요.

</div></details>

<details class="qa"><summary>"작업 대부분을 오픈소스했군요. NAVER의 우위를 내주는 것 아닌가요?"</summary>
<div class="qa-body">

**Testing:** 제품/사업 판단과 회사가 *왜* 논문을 내는지 이해하는가.

**Answer A — 전략적 논거 초안.**
"ZIM·ECLIPSE·PointWSSIS·BESTIE는 공개 code가 있고, 이는 재현·후속 사용·연구 채택을 돕습니다. 반면 제품 data·customer eval·security detail은 보호 대상일 수 있습니다. 제가 공개 범위 결정에 실제로 참여했다면 그 기준과 trade-off를 설명하고, 아니면 회사의 의도를 제 결정처럼 말하지 않고 공개 결과와 일반 IP 원칙만 구분하겠습니다."

**Answer B — concise.**
"Open-source 범위는 research reproducibility와 제품·보안 IP의 균형입니다. 공개 code의 효과는 확인 가능한 adoption으로, 비공개 범위는 회사 policy로 설명하고 제가 정한 전략처럼 과장하지 않겠습니다."

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
"NAVER Cloud에서의 기간 동안 공개 연구와 product 접점을 함께 경험했습니다. 떠나는 이유는 현재 조직의 한계를 추정하거나 깎아내리는 것이 아니라, [지원 팀의 최신 공식 연구/JD]가 제 다음 질문인 grounded multimodal reasoning과 구체적으로 맞기 때문입니다. 제가 가져갈 증거는 [공개 프로젝트/검증된 역할]이고, 새 환경에서 배우고 기여할 지점은 [확인된 team agenda]입니다."

**Answer B — scope와 scale (concise).**
"현재 역할에서 멀어지려는 답보다, [공식 근거가 있는 역할 특성]과 제 [검증된 경험]의 교집합을 찾고 있습니다. 구체적인 성장·기여 가설을 말씀드리고 실제 team scope는 면접에서 확인하겠습니다."

**Follow-up:** "NAVER에서는 못 하고 여기서는 할 수 있는 게 구체적으로 뭔가요?"
**Rebuttal tip:** 최근 공식 논문·product·JD에서 한 가지를 인용하고, 본인의 근거와 아직 확인할 질문을 붙이세요. 회사 브랜드만 나열하거나 현재 고용주가 할 수 없다고 단정하지 않습니다.

</div></details>

<details class="qa"><summary>"왜 하필 이 회사인가요?" (회사별)</summary>
<div class="qa-body">

**Testing:** 최신 공식 근거를 바탕으로 역할과 본인 경험의 실제 교집합을 설명할 수 있는가. [회사 조사 플레이북](#/process/companies)을 쓰세요.

회사마다 다음 네 칸을 채운 뒤 한 문단으로 합칩니다.

| 근거 | 채울 내용 |
| --- | --- |
| **공식·날짜가 있는 signal** | 최신 JD, team page, 논문 또는 product 한 가지 |
| **내 증거** | ZIM·ECLIPSE·PointWSSIS 등 공개 결과 또는 확인된 product 역할 |
| **기여 가설** | precise mask, low-label adaptation, grounding, efficiency 중 실제 교집합 |
| **확인할 불확실성** | team scope, publication policy, compute/data access, loop format |

답변 골격: *"[날짜]의 [공식 자료]에서 [구체 agenda]를 확인했습니다. 제 [검증된 프로젝트]에서 배운 [메커니즘/결과]가 그 문제의 [구체 gap]에 연결됩니다. 첫해에는 [검증 가능한 가설]을 시험하고 싶고, 실제 우선순위는 팀과 확인하겠습니다."* 회사 문화·interview format·IP 전략을 이름만으로 추정하지 않습니다.

**Follow-up:** "첫 해에 무엇을 하고 싶나요?"
**Rebuttal tip:** *그들의* roadmap에 있는 진짜 프로젝트로, 당신 기술이 독특하게 기여하는 것을 대세요 — generic한 "codebase 배우기"가 아니라. 구체성이 fit이 진짜임을 증명합니다.

</div></details>

<details class="qa"><summary>"서울에 있군요. 정말 relocate할 의향이 있나요?" / visa & 물류"</summary>
<div class="qa-body">

**Testing:** 물류적 진지함 — recruiter는 여기서 늦게 흔들리는 지원자를 죽입니다.

**Answer A — 명확한 commitment.**
"이 답변은 실제 의사와 제약을 확인한 뒤 채웁니다: relocation [가능/불가/조건부], visa sponsorship 필요 여부 [ ], PhD defense·졸업 timeline [ ], 가능한 start-date range [ ]. 확정되지 않은 advisor 유연성이나 remote option을 사실처럼 말하지 않겠습니다."

**Answer B — concise.**
"현재 확인된 조건은 [ ]이고, 필요한 visa와 가능한 start date를 recruiter와 조율하겠습니다." 가능 여부가 확정된 경우에만 단정형으로 바꿉니다.

**Follow-up:** "PhD를 먼저 끝내야 하나요?"
**Rebuttal tip:** defense timeline, 고용 시작 가능일, visa, 물리적 체류 조건을 실제 관계자와 확인하세요. 확인 전에는 remote completion이나 advisor approval를 약속하지 않습니다.

</div></details>

<details class="qa"><summary>"자신을 researcher로 보나요, engineer로 보나요?"</summary>
<div class="qa-body">

**Testing:** 자기 인식과 특정 track(RS vs AS vs MLE)에의 fit. 잘못된 self-label은 loop에 mismatch를 만듭니다.

**Answer A — applied-scientist 정체성.**
"Applied scientist라는 정체성이 가장 가깝습니다. 공개 논문에서는 ablation·baseline·failure analysis를, 이력서의 제품 작업에서는 latency·deployment 제약을 다뤘습니다. 구체적인 ONNX issue나 handoff ownership은 실제 사례가 확인될 때만 말하겠습니다. 지원 role이 기대하는 research 대 delivery 비중도 먼저 확인합니다."

**Answer B — concise.**
"First/co-first-author 연구 기록과 이력서에 적힌 model·API·인증-service 기여가 두 축의 근거입니다. 각 product의 출시 상태와 제 기여 범위를 확인해 applied-science 교집합을 설명하겠습니다."

**Follow-up:** "어느 쪽을 더 즐기나요?"
**Rebuttal tip:** 정직하되 role-aware: RS 패널엔 research로, AS/MLE엔 shipping으로 기울되, 항상 "둘 사이의 loop가 제가 진짜 사랑하는 것"으로 되돌리세요. 어느 절반도 부정하지 마세요.

</div></details>

---

## E · Hard, curveball & weakness probes

어려운 질문에서는 약점을 숨기기보다 범위와 보완 계획을 정확히 말하세요. 방어적으로 반응하는 것뿐 아니라 사실을 과장하거나 가짜 실패를 만드는 것도 큰 위험입니다.

<details class="qa"><summary>"작업이 incremental해 보입니다 — segmentation variant가 많네요. 큰 아이디어는 어디 있나요?"</summary>
<div class="qa-body">

**Testing:** 흔들리지 않고 작업의 *significance*를 방어할 수 있는가 — 의도된 도발.

**Answer A — 문제-재정의로 reframe (confident).**
"그 우려를 논문별 문제 재정의와 evidence로 답하겠습니다. PointWSSIS는 proposal bottleneck을, ECLIPSE는 freezing+prompting이라는 대안을, ZIM은 data granularity와 promptable matting의 결합을 검증했습니다. 다만 venue decision의 이유를 reviewer의 마음처럼 추정하지는 않겠습니다. 공통된 research 철학은 '진짜 병목을 명시하고 data와 architecture를 함께 설계한 뒤 ablation으로 분리해 검증한다'입니다."

**Answer B — arc가 큰 아이디어.**
"큰 아이디어는 어떤 단일 논문이 아니라 *arc*입니다: perception을 값싼 supervision에서 신뢰할 수 있는 reasoning으로 사다리 위로 몰아가는 것. 개별적으로 각 논문은 날카로운 기여이고; 함께 보면 perception의 가치가 *label-efficient, updatable, verifiable*에 있다는 일관된 베팅입니다. 그게 정확히 분야가 지금 grounded agent로 하고 있는 베팅입니다."

**Follow-up:** "좋아요 — 분야가 다 잊어버려도 살아남을 단 하나의 결과는?"
**Rebuttal tip:** 하나를 고르고 공개 ablation으로 정당화하세요: ZIM에서는 data granularity가 load-bearing이었고, PointWSSIS에서는 proposal bottleneck을 supervision으로 재구성했습니다. 과장된 보편 법칙으로 만들지 않습니다.

</div></details>

<details class="qa"><summary>"당신은 vision 사람인데 이 역할은 LLM/NLP를 건드립니다 — depth가 있나요?"</summary>
<div class="qa-body">

**Testing:** 진짜 gap에 대한 정직한 자기 평가, 그리고 그것을 닫을 수 있다는 증거.

**Answer A — 경계를 소유하고 다리를 보여라.**
"타당합니다 — 제 *출판* depth는 LLM pretraining이나 NLP 이론이 아니라 vision에 있습니다. 하지만 두 가지. 첫째, 제 현재 작업이 그 이음새에 삽니다: grounded VLM과 agentic program synthesis는 LLM planning, tool-use, chain-of-thought, hallucination, evaluation을 reasoning하도록 요구합니다 — 저는 지금 거기서 작동하고 있습니다, 지향이 아니라. 둘째, 제가 뭘 모르는지 정확히 압니다 — RLHF를 유도하거나 tokenizer를 맨바닥에서 만들 수 있다고 냉정하게 주장하기 전에 조심하겠습니다. transformer 내부를 허풍떨다 걸리기보다, 경계를 정직하게 말하고 gap을 빠르게 닫는다는 걸 보이겠습니다(제품 작업을 위해 SAM stack과 diffusion-conditioning 문헌을 독학했습니다)."

**Answer B — T-shape (concise).**
"저는 T-shaped입니다: perception에 깊고, multimodal 전반에 넓고 최신 — attention/transformer, VLM training과 eval, grounding, agent. 제가 원하는 프론티어는 vision-language이고, 거기서 제 depth가 희소한 절반이며 LLM 쪽은 학습 가능하고 이미 제 일상 작업의 일부입니다."

**Follow-up:** "좋아요 — 지금 multi-head attention을 구현하거나 RoPE를 설명해 보세요."
**Rebuttal tip:** 역할에서 요구하는 from-scratch primitive를 실제로 구현·설명해 보세요 — attention(head split, $1/\sqrt{d_k}$ scaling, causal mask), KV cache, LoRA, sampling, RoPE 등. 특정 항목이 탈락 사유 1위라고 단정할 근거는 없지만, 전문 분야 밖의 기초도 별도 평가될 수 있습니다.

</div></details>

<details class="qa"><summary>"당신 production 작업은 전부 single-domain — image, 한 회사. 일반화할 수 있나요?"</summary>
<div class="qa-body">

**Testing:** breadth risk와 적응성 — 5년 단일 고용주 CV의 진짜 우려.

**Answer A — depth-transfers 논거.**
"두 답. 첫째, '한 회사에서 image' 안에서 저는 실제로 *많은 것*을 걸쳤습니다: weakly-supervised, continual, foundation model, matting, detection, on-device efficiency, biometric security, 그리고 이제 multimodal agent — 다른 sub-problem, metric, 제약. 둘째, 전이되는 기술은 domain이 아니라 *method*입니다: 병목을 정의하고, data와 architecture를 공동 설계하고, 정직한 ablation을 돌리고, production last mile을 닫는 것. 그 method가 저를 CAM에서 foundation model로, agent로 마찰 없이 옮겼고, video, 3D, 또는 새 modality로도 똑같이 옮길 것입니다. 인턴십에서의 초기 노출 — OCR, autonomous-driving perception, face — 도 있어서 한 줄이 시사하는 것만큼 좁지 않습니다."

**Answer B — concise.**
"단일 회사이지만 단일 문제가 아닙니다 — image tag에서 foundation model로, agent로 갔습니다. 상수는 research method이고, 그게 전이되는 것입니다. 새 domain은 정의할 또 하나의 병목으로 다루겠습니다."

**Follow-up:** "왜 지금까지 회사나 domain을 한 번도 안 바꿨나요?"
**Rebuttal tip:** 재직 기간에 실제로 얻은 depth와 반복해 맡은 범위를 구체적 사례로 설명하세요. 완전한 end-to-end ownership이나 다음 이동 동기는 확인된 사실만 사용합니다.

</div></details>

<details class="qa"><summary>"실패한 research bet 하나 말해보세요."</summary>
<div class="qa-body">

**Testing:** 정직함, 성찰, research 판단 — 성공을 실패로 분장하면 라운드를 망칩니다.

**Answer A — research-failure 골격.**
"[프로젝트]에서 제가 소유한 가설은 [ ]였습니다. [동일 조건의 실험/로그]에서 [반증 signal]을 확인했고 [시점]에 중단했습니다. 제가 한 pivot은 [ ]였고 결과는 [공개 가능한 결과 또는 남은 실패]였습니다. 이후에는 [구체적으로 바꾼 실험 순서]를 적용합니다." Offline refinement가 실제 사건이고 기록으로 설명 가능할 때만 그 예를 씁니다.

**Answer B — engineering-failure 골격.**
"[실제 최적화/배포 결정]을 먼저 시도했지만 [측정 protocol]에서 [품질·지연 failure]가 나타났습니다. 원인을 [ablation/profile]로 분리하고 [대안]으로 바꿨습니다." Quantize-first, boundary collapse, distillation 순서는 이력서로 확인되지 않으므로 실제 log가 있을 때만 사용합니다.

**Follow-up:** "지금이라면 뭘 다르게 하겠습니까?"
**Rebuttal tip:** 답이 *곧 다르게*입니다 — "가장 싼 실험으로 아이디어를 먼저 죽여라." 당신의 실패에 "인내를 배웠다"가 아니라 구체적이고 전이 가능한 규칙이 붙어 있는지 확인하세요.

</div></details>

<details class="qa"><summary>"논문을 많이 내네요. 여기서 실제로 ship할 건가요, 아니면 그냥 논문을 쫓을 건가요?"</summary>
<div class="qa-body">

**Testing:** 출판 목표와 제품 delivery가 충돌할 때 우선순위와 ownership을 어떻게 정하는가.

**Answer A — shipping 증거.**
"공개·이력서 근거로는 foreground API, FaceSign 기여, 약 10 ms on-device model, CLOVA-X와 ZIM의 접점이 있습니다. 각 항목에서 제 역할과 실제 출시 상태를 분리해 설명하겠습니다. 연구와 delivery가 항상 정렬된다고 주장하지 않고, 충돌한 실제 사례가 있다면 어떤 기준으로 scope·deadline·publication을 조정했는지 답하겠습니다."

**Answer B — concise.**
"공개 code가 있는 연구와 이력서에 적힌 제품 기여가 모두 있습니다. '모든 논문을 ship했다'거나 미공개 작업의 양을 비교하기보다, 확인 가능한 두 사례에서 research artifact와 product artifact를 구분해 보여드리겠습니다."

**Follow-up:** "1년간 출판하지 말라고 하면요?"
**Rebuttal tip:** 깔끔하게 수용하세요 — "제품과 impact가 먼저입니다; 이미 엄격한 non-disclosure 하에 일해봤습니다(FaceSign)." 어떤 망설임도 팀보다 자기 CV를 우선한다고 읽힙니다.

</div></details>

<details class="qa"><summary>"공백이 있네요: 몇 년간 논문이 없거나 뜸한 시기. 무슨 일이 있었나요?"</summary>
<div class="qa-body">

**Testing:** 겉보기 공백이 문제를 숨기는가 — 보통은 아니지만, 면접관은 당신이 움찔하지 않는 걸 보고 싶어합니다.

**Answer A — 실제 timeline으로 설명.**
"[기간]에는 [승인된 이력서상 역할·산출물]에 집중했습니다. 보안·내부 제품이라 공개하지 못했다는 인과는 실제 policy와 timeline이 확인될 때만 말합니다. 공개 증거가 없는 활동은 역할·기간만 설명하고 impact를 부풀리지 않겠습니다."

**Answer B — concise.**
"논문 사이 기간의 실제 업무는 [확인된 프로젝트·책임]입니다. 공개할 수 있는 산출물과 비공개 범위를 분리해 말씀드리겠습니다."

**Follow-up:** "볼 수 없는 impact를 제가 어떻게 검증하죠?"
**Rebuttal tip:** 확인 가능한 DAN 24 talk, public product page, 논문/code를 먼저 제시하세요. Reference check는 당사자의 사전 동의와 회사 절차 안에서만 제공하고, 내부 지표는 공개하지 않습니다. "공개 증거는 가리켜 드릴 수 있고, 내부 지표는 회사 안에 남깁니다"가 신뢰 가는 문장입니다.

</div></details>

<details class="qa"><summary>"Rapid-fire fundamentals: BatchNorm vs LayerNorm — 왜 transformer에 LN인가요?"</summary>
<div class="qa-body">

**Testing:** 압박 하의 breadth. 한 문제를 틀렸다고 자동 탈락한다고 단정할 수는 없지만, 기초 개념을 정확히 설명하고 모르는 범위를 교정하는 태도를 함께 봅니다.

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
| Rapid-fire fundamentals | role-relevant breadth를 retrieval 연습 | 틀리면 가정을 밝히고 수정하며, 한 문제를 합격 규칙처럼 과대해석하지 않는다. |
| 실패한 bet | 진짜 실패 + **전이 가능한 규칙** | "가장 싼 실험으로 아이디어를 먼저 죽여라." |
| 큰 아이디어 도전 | **하나**의 durable한 결과에 commit | "Data granularity가 load-bearing임을 ablation으로 보였다" (ZIM). |
| 모든 답변 | 라운드 목적에 맞는 **각도**를 고르고 암기 대신 내면화 | deep-dive는 mechanism, HM은 scope/impact, recruiter는 짧은 사실·물류. |

**Golden rules:** (1) 숫자를 절대 지어내지 말라 — 메커니즘을 설명하거나 대외비라고 하라. (2) *당신*의 기여를 명확히 하라. (3) 그들이 하기 전에 약점을 대라. (4) [Company Playbooks](#/process/companies)로 모든 답변을 맞춤화하라.

## Cross-links

- Deep-dive (위 답변의 원자재): [ZIM](#/resume/zim) · [ECLIPSE](#/resume/eclipse) · [PointWSSIS & BESTIE](#/resume/pointwssis-bestie) · [Phoenix](#/resume/phoenix-mask-refinement) · [FaceSign](#/resume/facesign) · [On-Device Seg](#/resume/on-device-segmentation) · [Grounded VLM/Agents](#/resume/grounded-vlm-agents) · [Spatial-Reasoning Agent (심사 중)](#/resume/neurips26-spatial-reasoning)
- [Your CV → Interview Map](#/resume/overview) — 회사별로 어떤 프로젝트로 열지
- [STAR & The Story Bank](#/behavioral/star) — failure/conflict/ownership 질문의 behavioral framing
- [Company Playbooks](#/process/companies) — 각 타깃 회사가 파고드는 것, 모든 답변을 다시 조준하려면
