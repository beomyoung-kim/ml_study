# Company Playbooks: 확인 가능한 정보로 준비하기

<div class="tag-row"><span class="tag">company research</span><span class="tag">dated snapshot</span><span class="tag">recruiter verification</span><span class="tag">role fit</span></div>

> [!TIP] 이 chapter가 존재하는 이유
> 회사 이름만으로 면접 형식이나 문화를 추측하면 준비가 빗나갑니다. 같은 회사에서도 **조직·팀·직무·레벨·지역·채용 시점**에 따라 loop가 달라집니다. 이 chapter는 회사별 소문을 외우는 목록이 아니라, 공개 정보와 recruiter 확인을 이용해 **이번 지원 건의 플레이북**을 만드는 방법입니다.

> [!WARNING] 날짜가 없는 회사 정보는 사용하지 마세요
> 아래 회사명은 조사 범위를 나누기 위한 표지입니다. 라운드 수, 문제 난도, 도구 정책, team matching, reference check, level, 보상, 근무 방식은 고정 정책으로 간주하지 마세요. 지원할 때마다 `확인일(as of)`, `대상 JD/지역`, `출처`, `확인 상태`를 기록하고, 최종 운영 정보는 **면접 안내문과 recruiter의 서면 답변**으로 확인하세요.

## 정보의 신뢰도 순서

1. **이번 지원 건의 안내문과 recruiter/HM 답변** — 일정·형식·허용 도구·제출물의 기준.
2. **공식 JD, careers 페이지, 연구 조직의 공식 페이지** — 역할 범위와 공개된 연구 방향.
3. **공식 논문·기술 블로그·제품 문서** — why-us와 기술 대화의 근거.
4. **최근 후보 경험담·커뮤니티 자료** — 질문 후보를 만드는 보조 자료일 뿐, 정책의 증거는 아님.

커뮤니티에서 본 내용을 recruiter에게 물을 때는 “제가 듣기로는…”보다 다음처럼 중립적으로 확인하세요.

> “이번 역할의 전체 단계와 각 단계의 형식, 준비해야 할 산출물, 사용할 플랫폼과 허용 도구를 알려주실 수 있을까요?”

## recruiter-first 확인 체크리스트

첫 recruiter 통화 뒤 아래 항목을 한 장에 채우세요. 답을 못 받았으면 `미확인`으로 남기고 추측하지 않습니다.

| 영역 | 확인할 질문 | 기록할 값 |
| --- | --- | --- |
| 지원 범위 | 특정 팀 채용인가, 여러 팀을 위한 pooled hiring인가? | 팀/조직/req ID |
| 전체 과정 | 다음 단계와 예상 순서는? 단계가 합쳐지거나 생략될 수 있나? | 단계명·순서·담당자 |
| 기술 평가 | DSA, ML coding, ML depth, system design 중 무엇이 포함되나? | 유형·언어·실행 가능 여부 |
| 연구 평가 | 논문 deep-dive, job talk, take-home이 있나? | 주제·시간·청중·Q&A·제출 형식 |
| 도구 정책 | IDE, 문서 검색, autocomplete, 생성형 AI 사용이 허용되나? | 허용/금지/제한·근거 문구 |
| 진행 방식 | 원격인가 대면인가? 시간대·장소·접속 도구는? | 일정·장소·백업 연락처 |
| 의사결정 | debrief/committee/team match가 있나? | 알려준 범위만 기록 |
| reference | 요청 여부, 시점, 대상, 후보 동의 절차는? | 확인된 절차 |
| 역할 조건 | level/title 범위, location, visa·relocation 조건은? | 서면 답변 링크 |
| 연구 환경 | publication/open-source, compute/data 접근은 어떻게 결정되나? | HM에게 재확인할 항목 |

기술 폰 스크린 준비는 [폰 스크린 당일 허브](#/process/phone-screens), recruiter와 HM에게 물을 문장은 [Recruiter & HM Screens](#/process/recruiter-hm)에 정리되어 있습니다.

## 회사별로 무엇을 구분해야 하나

### Meta / FAIR 계열 역할

회사명이 같아도 중앙 연구 조직, product-adjacent applied 팀, engineering 역할은 기대 산출물이 다를 수 있습니다.

- JD에서 `publish`, `novel research`, `production`, `ranking/recommendation`, `foundation model`, `infrastructure` 같은 동사를 표시합니다.
- job talk와 별도 coding round가 있는지, coding의 언어·실행·도구 정책을 확인합니다.
- 특정 팀 지원인지, loop 이후 team conversation이 있는지 묻습니다.
- 공개된 논문이나 제품 하나를 골라 **무엇이 인상적이었는지와 내 경험이 어떤 문제에 기여하는지**만 연결합니다. 조직 구조나 미공개 roadmap은 추측하지 않습니다.

### NVIDIA 계열 역할

research lab, applied research, product/infra 역할을 먼저 구분합니다.

- 논문·research talk 비중과 coding/system 평가 비중을 각각 확인합니다.
- CUDA나 GPU kernel 구현이 필수인지, 성능 추론 수준인지, 학습 의지만 보는지 JD와 recruiter에게 확인합니다.
- hardware-aware ML이라면 latency·memory·throughput 중 어떤 제약을 실제로 다루는지 HM에게 묻습니다.
- 공개 연구를 바탕으로 efficiency를 말하되, 특정 하드웨어나 내부 cluster 접근을 당연시하지 않습니다.

### Apple 계열 역할

MLR, AIML, 제품 내 embedded ML 역할은 이름이 비슷해도 연구 자유도와 출시 책임이 다를 수 있습니다.

- publication/open-source 가능 범위와 승인 절차를 직접 묻습니다.
- on-device, privacy, latency 같은 제약이 JD에 적혀 있다면 내 경험의 **측정 가능한 trade-off**로 답합니다.
- 논문 발표, ML coding, system design 중 무엇이 포함되는지 확인합니다.
- 공개되지 않은 제품·데이터·모델을 추측하지 않고 공개 JD와 공식 자료의 표현만 사용합니다.

### Adobe 계열 역할

순수 research, applied science, 제품 팀과 가까운 generative/creative tooling 역할을 구분합니다.

- research presentation이 있다면 대표 논문의 novelty뿐 아니라 실험 설계·실패·제품 전이까지 준비합니다.
- PyTorch/ML 구현과 일반 DSA가 각각 평가되는지 확인합니다.
- research 결과가 제품 지표로 넘어가는 handoff와 scientist의 ownership 범위를 HM에게 묻습니다.
- 특정 제품군을 why-us 근거로 쓸 때는 현재 공식 발표와 대상 JD가 실제로 연결되는지 날짜를 붙여 확인합니다.

### ByteDance / Seed 계열 역할

지역과 팀에 따라 평가 구성이 크게 달라질 수 있으므로 회사 전체의 평판을 이번 req의 사실로 바꾸지 않습니다.

- online assessment 유무, live coding의 개수·난도 범주, 언어와 실행 환경을 확인합니다.
- research deep-dive와 coding이 같은 세션에 결합되는지 묻습니다.
- 근무 시간, 협업 시간대, 성과 기대는 문화에 대한 소문 대신 **해당 팀의 구체적 운영 방식**으로 질문합니다.
- 비상장 equity가 제안에 포함되면 유동성·평가 기준·vesting·퇴사 후 조건을 문서에서 확인합니다. 비교 방법은 [Offers, Levels & Negotiation](#/process/negotiation)을 따릅니다.

### Mistral AI 계열 역할

research, inference/infra, applied/customer-facing 역할을 구분합니다.

- take-home이 있다면 예상 소요 시간, 제출 형식, 외부 자료와 AI 도구 허용 범위, 후속 발표 여부를 먼저 확인합니다.
- transformer/MoE/inference 구현이 요구되는지, 어느 언어와 추상화 수준인지 묻습니다.
- customer project와 internal model work의 비중, 출장·시간대·언어 요구를 대상 역할 기준으로 확인합니다.
- option 등 비상장 보상은 headline 금액으로 공모주식과 비교하지 말고 계약 문서와 현지 전문가의 검토를 거칩니다.

### Microsoft / MSR 계열 역할

MSR과 product AI 조직을 분리해서 조사합니다. 같은 title도 팀에 따라 publication, engineering, customer impact의 비중이 다를 수 있습니다.

- job talk 또는 세미나가 있는지, 주제 선택·청중·Q&A 형식을 확인합니다.
- coding, ML fundamentals, system design의 구성과 각 라운드 목적을 묻습니다.
- 외부 senior interviewer나 별도 calibration 단계가 있다고 들었다면, 명칭을 단정하지 말고 이번 일정에 실제로 포함되는지만 확인합니다.
- 연구 자율성, mentoring, cross-org collaboration은 추상적 문화 문구보다 첫 6개월의 실제 책임으로 질문합니다.

## why-us를 만드는 안전한 방식

회사별 한 문장은 `공개 근거 → 내 판단 → 내 기여 가설`로 만듭니다.

> “{날짜}에 공개된 {JD/논문/제품}에서 {구체적 선택}이 인상적이었습니다. 저는 {관련 경험}에서 {측정 가능한 문제}를 다뤘고, 이 팀에서는 {질문 형태의 기여 가설}을 검증해 보고 싶습니다.”

좋은 why-us는 최신 이름을 많이 나열하는 문장이 아닙니다. 실제 자료 하나를 읽었다는 증거와, 자신의 경험에서 이어지는 **검증 가능한 접점**이 있으면 됩니다. 개인 프로젝트별 매핑은 범용 본문에 복제하지 말고 [Your CV → Interview Map](#/resume/overview)과 [예상 질문 & 답변](#/resume/predicted-questions)에서 관리하세요.

## 회사 비교표는 결과가 아니라 작업 기록이다

| 회사·팀·req | 확인일 | 역할의 핵심 산출물 | 확인된 평가 형식 | 미확인 위험 | 내 근거 2개 | 다음 질문 |
| --- | --- | --- | --- | --- | --- | --- |
| `{회사 / 팀 / req ID}` | `YYYY-MM-DD` | 논문 / 제품 / 인프라 / 혼합 | recruiter가 확인한 내용 | 도구·team match 등 | CV evidence | HM/recruiter 질문 |

여기에 comp 숫자를 복사해 넣기보다, 통화·지역·level·equity 유형을 분리한 별도 시트를 만드세요. 공개 aggregate는 **조사일이 있는 참고점**으로만 쓰고, 실제 비교는 서면 offer와 plan document를 기준으로 합니다.

## 일정 순서는 ‘회사 난도’가 아니라 확인된 format risk로 정한다

“안전한 회사”와 “도전 회사”를 브랜드로 고정하지 마세요. 후보마다 위험은 다릅니다.

1. 확인된 라운드를 `DSA`, `ML coding`, `research talk`, `system design`, `take-home`, `behavioral`로 분해합니다.
2. 최근 mock 결과와 비교해 가장 큰 format gap을 찾습니다.
3. 핵심 선택지보다 먼저 **비슷한 형식의 실제 loop**를 배치하되, offer deadline이 과도하게 벌어지지 않게 recruiter와 조율합니다.
4. 일정·단계가 바뀌면 snapshot의 확인일과 준비 우선순위를 함께 갱신합니다.

## 회사별 snapshot 템플릿

```text
Company / team / req ID:
Location / employment entity:
Checked on (YYYY-MM-DD):

Primary sources:
- JD:
- official research/product page:
- recruiter email / invite:

Confirmed process:
- stages and sequence:
- format / duration:
- coding environment and allowed tools:
- job talk or take-home requirements:
- team matching / references:

Role hypothesis:
- expected output:
- first 6–12 month scope:
- publication / open-source / compute constraints:

My evidence:
- project / decision / metric 1:
- project / decision / metric 2:

Still unverified:
-

Questions for recruiter / HM:
-
```

## Cheat-sheet

| 원칙 | 한 줄 행동 |
| --- | --- |
| 회사명 ≠ 면접 형식 | 팀·req·지역·시점 단위로 확인 |
| 날짜 없는 정보 | 사용하지 말고 `미확인` 표시 |
| 소문 | 질문 후보로만 사용, 정책처럼 서술하지 않기 |
| 최종 운영 기준 | recruiter 서면 답변과 면접 안내문 |
| why-us | 공개 근거 하나 + 내 판단 + 기여 가설 |
| 개인 fit | 이력서 패킷에 단일 관리하고 여기서는 링크 |
| 보상 비교 | level·지역·통화·유동성·조건을 분리해 검토 |
| 일정 순서 | 브랜드 prestige가 아니라 확인된 format risk 기준 |

**Related:** [The RS/AS Pipeline](#/process/pipeline) · [Phone Screens](#/process/phone-screens) · [Recruiter & HM Screens](#/process/recruiter-hm) · [Offers & Negotiation](#/process/negotiation) · [Questions to Ask Them](#/playbook/questions-to-ask) · [Your CV → Interview Map](#/resume/overview)
