# Grounding & Region-Level Reasoning

<div class="tag-row"><span class="tag">referring expressions</span><span class="tag">grounded captioning</span><span class="tag">coordinates-as-tokens</span><span class="tag">region features</span><span class="tag">detection-as-VLM</span><span class="tag">open-vocab</span></div>

> [!TIP] The one-sentence thesis
> Grounding은 VLM이 language prior로 단언하는 대신 *자신의 증거를 가리키게* — box, mask, point로 — 강제합니다. 그것이 답변을 **검증 가능**하게 만들고 hallucination의 직접적 해독제입니다. 이것은 지원자 본인의 연구 영역(언어 reasoning을 픽셀·region 수준 증거에 연결하는 grounded VLM)이므로 프레이밍을 소유하세요: "뒷받침 없는 textual description → 검증 가능한 visual evidence."

## Why grounding

Global VLM은 존재하지 않는 객체를 자신 있게 묘사하거나, cup을 결코 localize하지 않고 "파란 컵 뒤에 뭐가 있나"에 답할 수 있습니다. Grounding은 그 루프를 닫습니다:

```mermaid
flowchart LR
  Q["Language query"] --> P["Reasoning plan"]
  P --> G["Ground region(s)"]
  G --> E["Box / mask / point evidence"]
  E --> A["Answer / next step"]
  A -->|verify against evidence| G
  style E fill:#e0533f,color:#fff
```

## 1 · The task family

| Task | Input → Output | Benchmark |
| --- | --- | --- |
| Referring expression comprehension (REC) | text → 지칭된 box | RefCOCO/+/g |
| Referring expression segmentation (RES) | text → mask | RefCOCO(g) masks |
| Phrase grounding | caption phrase → box | Flickr30K Entities |
| Grounded captioning | image → 명사별 box를 *포함한* caption | (grounded caption sets) |
| Open-vocabulary detection | text vocab → box | LVIS, ODinW |
| Pixel-grounded reasoning | query → mask/coord와 interleave된 CoT | recent 2025-26 sets |

Referring은 **출력 유형**으로 VQA 및 captioning과 다릅니다: global description(caption)이나 answer string(VQA)이 아니라 *region*입니다. "Grounded VQA"는 answer + evidence region을 함께 냅니다.

## 2 · How to emit a region: the design spectrum

이것이 핵심 아키텍처 선택이자 가장 깊이 파고들 가능성이 높은 지점입니다.

<figure>
<svg viewBox="0 0 680 250" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="11.5">
  <text x="10" y="20" fill="#6b7686">query + image</text>
  <!-- A: coords as text -->
  <rect x="10" y="35" width="200" height="60" rx="6" fill="none" stroke="#6366f1" stroke-width="2"/>
  <text x="110" y="55" text-anchor="middle" fill="#6366f1">A. Coordinates as text</text>
  <text x="110" y="72" text-anchor="middle" fill="#6b7686">LLM emits "[x1,y1,x2,y2]"</text>
  <text x="110" y="87" text-anchor="middle" fill="#6b7686">or &lt;box&gt; tokens</text>
  <!-- B: region features -->
  <rect x="240" y="35" width="200" height="60" rx="6" fill="none" stroke="#0ea5e9" stroke-width="2"/>
  <text x="340" y="55" text-anchor="middle" fill="#0ea5e9">B. Region features</text>
  <text x="340" y="72" text-anchor="middle" fill="#6b7686">ROI-pooled / proxy tokens</text>
  <text x="340" y="87" text-anchor="middle" fill="#6b7686">index into visual latents</text>
  <!-- C: detection head -->
  <rect x="470" y="35" width="200" height="60" rx="6" fill="none" stroke="#12a150" stroke-width="2"/>
  <text x="570" y="55" text-anchor="middle" fill="#12a150">C. Grounding head</text>
  <text x="570" y="72" text-anchor="middle" fill="#6b7686">continuous box/mask head</text>
  <text x="570" y="87" text-anchor="middle" fill="#6b7686">on LM hidden states</text>
  <!-- D: external tool -->
  <rect x="240" y="120" width="200" height="60" rx="6" fill="none" stroke="#e0533f" stroke-width="2"/>
  <text x="340" y="140" text-anchor="middle" fill="#e0533f">D. External specialist</text>
  <text x="340" y="157" text-anchor="middle" fill="#6b7686">Grounding DINO + SAM</text>
  <text x="340" y="172" text-anchor="middle" fill="#6b7686">as an agent tool</text>
  <text x="10" y="215" fill="#6b7686">trade-off →</text>
  <text x="120" y="215" fill="#6b7686">simple, weak spatial link</text>
  <text x="360" y="215" fill="#6b7686">tight visual link, more machinery</text>
  <text x="470" y="235" fill="#6b7686">modular but error-propagating</text>
</svg>
<figcaption>VLM이 region을 만드는 네 가지 방식. A는 학습이 가장 쉽고(text token); C는 가장 tight한 visual grounding을 주며; D는 전문가를 교체 가능하게 유지하지만 그들의 오류를 전파합니다.</figcaption>
</figure>

<dl class="kv">
<dt>A · Coordinates-as-tokens</dt><dd>LLM이 말 그대로 숫자나 <code>&lt;box&gt;</code> token을 씁니다(Kosmos-2, Shikra, Qwen-VL). 추가가 사소함 — 새 head 없음 — 그리고 grounding을 text generation과 통합. <b>약점:</b> <b>semantic-spatial gap</b> — coordinate token은 language 공간에 살며 visual feature와 약하게만 연결되어 box가 drift.</dd>
<dt>B · Region features</dt><dd>ROI-pooled feature(또는 visual latent를 index하는 학습된 "proxy token")를 LLM에 되먹임. 더 tight한 visual link; relational reasoning에 더 좋음. 배관이 더 많음.</dd>
<dt>C · Grounding head</dt><dd>LM hidden state 위 연속 box/mask regression head(LISA-style mask token → SAM decoder). End-to-end 미분 가능, 정밀. Mask 감독과 decoder 필요.</dd>
<dt>D · External specialist</dt><dd>Grounding DINO / SAM을 <b>tool</b>로 호출(Grounded-SAM). 모듈러, 교체 가능, SOTA detector에서 이득 — 하지만 오류가 조용히 전파되고 joint training이 없음. 이것이 <a href="#/vlm/visual-agents">Visual Reasoning Agents</a>로 가는 다리입니다.</dd>
</dl>

## 3 · Detection-as-VLM and open-vocabulary

두 수렴하는 트렌드:

- **Detection을 VLM에 접기:** 전용 detection head 대신 "객체를 box와 함께 text로 나열하라"로 task를 표현. 일반 VLM(Qwen3-VL, InternVL)이 점점 이 방식으로 유능한 detection/OCR을 함 — open-set, 유연한 prompt에 훌륭; 엄격한 AP에서는 여전히 전문 detector보다 뒤짐.
- **Open-vocabulary detection (OVD)**는 CLIP-style text alignment를 detector에 융합: **Grounding DINO**(text → box), YOLO-World, OWLv2, APE. 이들은 grounded VLM(설계 D)이 호출하는 전문 tool이며, grounded 시스템이 localize할 수 있는 것의 *하한*을 설정합니다. 심층: [Object Detection](#/cv/detection)과 [Vision Foundation Models](#/cv/foundation-models)(SAM 3의 Promptable Concept Segmentation이 2025 open-vocab detect+segment+track 데이터포인트).

> [!NOTE] Region vs. pixel evidence
> Box는 저렴하고 빠르지만 경계, occlusion, overlap에 눈이 멉니다; **mask**는 편집, 측정, 검증에 필수입니다. 지원자의 [ZIM](#/resume/zim)(zero-shot matting)과 SAM 계보 작업이 바로 고품질 픽셀 증거의 *공급자*입니다 — grounded reasoner는 자신의 mask만큼만 신뢰할 수 있습니다. CV의 "픽셀·region 수준"은 의도적인 both-and입니다.

## 4 · Grounded RL and multi-step grounding (2025–2026)

프런티어 방향: 모델이 **다시 보게** 하기. coordinate를 emit하고, crop/zoom하고, region을 재인코딩하고, 계속합니다 — chain-of-thought의 visual 유사물.

- **Visual coordinate로 grounded reasoning:** 각 reasoning step을 자유 text가 아니라 region에 anchor.
- **최종 답변 reward만으로 하는 RL**이 grounding *행동*을 창발시킬 수 있음(모델이 답에 도움이 되어서 zoom을 배움) — 밀집 box 감독 불필요.
- 이것은 agentic "thinking with images"로 번져 들어갑니다 — [Visual Reasoning Agents](#/vlm/visual-agents) 참고.

> [!WARNING] Spurious success
> 모델이 **틀린 증거로 맞는 답**을 낼 수 있습니다(prior에서 추측, box는 다른 곳을 가리킴). 답 정확도만으로는 이것을 보상합니다. **항상 grounding 품질을 함께 보고하세요:** mask IoU, pointing-game 정확도, grounded recall. 이것이 지원자의 grounding 작업에서 핵심적인 평가 설계 지점입니다.

## 5 · Why grounding data is expensive

Box/mask + language alignment는 annotate 비용이 큽니다. 스케일링 트릭(그리고 그 위험):

- **Pseudo-labeling:** detector + LLM이 region-text 쌍 생성 → 스케일, 하지만 노이즈.
- **Synthetic / simulator** 데이터(3D scene)로 spatial relation → 깨끗한 label, domain gap.
- **Label-efficient / weakly-supervised** 접근(point, image-level tag) — 바로 지원자의 PointWSSIS/BESTIE 계보; [Weak & Semi-Supervised](#/cv/weak-semi-supervised) 참고.

## 6 · Evaluation: answer *and* evidence

Grounding은 두 축 모두에서 metric이 필요합니다. 답 정확도만으로는 spurious success를 보상하기 때문입니다.

| Metric | Measures |
| --- | --- |
| REC accuracy @ IoU 0.5 | 예측 box가 맞나? |
| Mask IoU / cIoU | segmentation grounding 품질 |
| Pointing game | peak/point가 올바른 region에 떨어지나? |
| Grounded recall / precision | 인용된 region이 맞고 *그리고* 완전한가? |
| POPE / CHAIR | 생성 text의 객체 hallucination |
| Answer↔evidence consistency | 진술된 답이 인용 region과 일치하나? |

모델이 틀린 이유로 맞을 수 있고; 모델이 완벽히 ground하고도 틀리게 답할 수 있습니다. 쌍으로 보고하세요. *증거가 이해를 돕는지*에 대한 human evaluation은 grounded-reasoning 시스템에 유용한 보완입니다.

## Q&A

<details class="qa"><summary>How does grounding reduce hallucination, and how can it introduce new errors?</summary>
<div class="qa-body">

**Short:** 답하기 전/중에 증거를 localize하도록 요구하면 순전히 prior에 이끌린 주장이 억제됩니다 — region이 뒷받침하지 않으면 "왼쪽의 빨간 컵"을 묘사할 수 없습니다. 하지만 *틀린* grounding은 새로운 실패 양상입니다: 틀린 box에 anchor된 자신 있는 답.

**Deep:** Grounding은 반증 불가능한 text 단언을 확인 가능한 주장으로 바꿉니다(box가 text가 말하는 것을 담고 있나?). 그것이 검증, rejection sampling, human trust를 가능하게 합니다. 함정은 오류 *재배치*입니다: hallucination이 mis-grounding이 되고, 평가가 답만 채점하면 **spurious success**를 얻습니다. 그래서 grounded 시스템은 증거 품질(IoU, pointing game)을 답 정확도와 함께 평가해야 하고, 이상적으로는 answer↔evidence consistency를 확인하는 검증 step을 추가해야 합니다.
</div></details>

<details class="qa"><summary>Coordinates-as-tokens vs. region features vs. a grounding head — pick one and defend it.</summary>
<div class="qa-body">

**Short:** 빠른 반복과 open-set 유연성에는 coordinates-as-tokens(새 head 없음, text와 통합). 정밀 localization과 경계-중요 task에는 grounding head(mask token → segmentation decoder). Region feature는 relational reasoning이 중요할 때 그 사이에 위치.

**Deep:** Coordinates-as-tokens는 가장 저렴하지만 **semantic-spatial gap**을 겪습니다 — text-space 숫자가 픽셀과 약하게 연결 — 그래서 box가 거칠고 작은/붐비는 객체에서 drift; visual latent를 index하는 proxy/region token이 그 link를 tight하게 합니다. Grounding head(continuous regression / mask-token → SAM)는 가장 정확하고 end-to-end 미분 가능하지만 mask 감독과 decoder가 필요하고 임의 open-vocab에는 덜 유연합니다. 실무에서는 제품으로 선택합니다: 편집/측정 제품은 mask가 필요(head 또는 전문 tool); 유연한 visual assistant는 coordinate token으로 시작하고 관계가 중요한 곳에 region feature를 추가.
</div></details>

<details class="qa"><summary>(Ongoing-research framing) How would you make a grounded VLM's spatial reasoning verifiable?</summary>
<div class="qa-body">

**Short:** 각 reasoning step에서 증거를 강제하고, 답이 아니라 *증거*를 평가하세요 — mask/box IoU, pointing 정확도, answer↔evidence consistency. 답이 맞아도 틀린 증거는 실패로 취급.

**Deep:** (심사 중인 작업의 방법 세부는 미공개.) 논의할 수 있는 동기: end-to-end VLM은 뒷받침 없는 textual description에 기대어 spatial/referential query에 답하는데, 이는 디버그 불가능합니다. Reasoning을 픽셀/region 증거에 연결하면 각 step이 반증 가능해지고 chain이 *어디서* 깨졌는지 localize할 수 있습니다. 제가 기반으로 삼는 공개 트렌드: zoom/crop을 쓰는 grounded RL, pixel-grounded CoT, open-vocab concept segmentation(SAM 3), 그리고 증거 공급자로서의 제 perception-foundation 배경([ZIM](#/resume/zim), SAM 계보). 전체 프레이밍: [Deep-Dive: Grounded VLM/Agents](#/resume/grounded-vlm-agents).
</div></details>

**Follow-ups**

- "semantic-spatial gap이 뭐고 proxy token이 어떻게 다루나?" (Text coord가 visual feature에 묶이지 않음; 학습된 proxy token이 image latent를 index.)
- "Open-vocab grounding 실패 양상?" (희귀 개념, 부정 "빨간 게 아닌 것", relational clause, 유사 instance 무리, 아주 작은 객체.)
- "왜 IoU를 답 정확도와 함께 보고하나?" (Spurious success — 맞는 답, 틀린 증거.)
- "전문 detector가 detection-as-VLM을 어디서 이기나?" (엄격한 AP, 작은 객체, real-time; VLM은 open-set 유연성과 language conditioning에서 이김.)
- "Human box/mask label 없이 grounding 데이터를 어떻게 스케일하나?" (Detector+LLM pseudo-label, synthetic 3D scene, point/tag로부터의 weak supervision — 각각 noise/domain-gap 비용.)
- "Grounded 편집 제품: box 증거인가 mask 증거인가?" (Mask — 경계와 alpha가 compositing에 중요; box는 픽셀 정확 편집에 너무 거침.)

## Cheat-sheet

| Term | Meaning |
| --- | --- |
| REC / RES | referring expression comprehension (box) / segmentation (mask) |
| Phrase grounding | caption phrase → box (Flickr30K Entities) |
| Coordinates-as-tokens | LLM이 box 숫자 / `<box>` token을 씀 — 단순, semantic-spatial gap |
| Region features | visual latent를 index하는 ROI/proxy token — 더 tight한 link |
| Grounding head | LM hidden 위 continuous box/mask head (LISA → SAM) — 정밀 |
| Detection-as-VLM | text 출력으로 phrase detection; 유연, 전문 AP 아래 |
| Open-vocab (OVD) | text→box: Grounding DINO, OWLv2, YOLO-World, SAM 3 |
| Spurious success | 맞는 답, 틀린 증거 → 항상 IoU/pointing도 eval |

> [!QUOTE] Interview positioning (the candidate's angle)
> "CLIP과 LLaVA는 풍부한 semantic 공간을 열었습니다. 제 기여는 그 위에 *픽셀 정확 perception과 검증 가능한 grounding*을 얹는 것입니다 — 그래서 VLM이 language prior로 서술하는 대신 사람이 확인할 수 있는 증거를 인용하게. 제 segmentation/matting foundation([ZIM](#/resume/zim), SAM 계보)이 그 증거의 공급자입니다."

**Related:** [Object Detection](#/cv/detection) · [Vision Foundation Models](#/cv/foundation-models) · [Visual Reasoning Agents](#/vlm/visual-agents) · [Deep-Dive: Grounded VLM/Agents](#/resume/grounded-vlm-agents) · [Instruction Tuning & Decoding](#/vlm/instruction-tuning) · [Deep-Dive: ZIM](#/resume/zim)
