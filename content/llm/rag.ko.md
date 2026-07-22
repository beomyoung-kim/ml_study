# RAG (검색 증강 생성)

> [!NOTE] 이 챕터의 목표
> LLM에 **외부 evidence를 검색해 넣는** 대표 방법인 RAG를 처음부터 잡습니다. 모델을 다시 학습하지 않고 최신·사내 문서를 활용할 수 있지만, 검색된 문서가 답을 실제로 뒷받침하는지는 별도 평가해야 합니다. [임베딩](#/llm/embeddings) 바로 위의 검색 기초와 전체 파이프라인을 다룹니다.

## 무엇을, 왜

Checkpoint의 **parametric knowledge**는 그때까지의 학습 데이터와 update에 묶이며, 최신 뉴스나 private corpus가 들어 있다고 가정할 수 없습니다. 정보 부재는 환각의 한 원인이지만, context가 있어도 잘못 합성하거나 prior를 따를 수 있습니다.

RAG는 답하기 전에 관련 문서를 **검색(Retrieve)** 하고, 선택한 context를 입력에 **증강(Augment)** 한 뒤 답을 **생성(Generate)** 합니다. 모델 parameter를 바꾸지 않고 corpus를 갱신할 수 있지만, 실제 운영에는 추가·수정·삭제 전파, ACL, embedding/index version과 backfill이 필요합니다.

> [!TIP] 면접 한 줄
> "RAG는 parameter를 바꾸지 않고 외부 evidence 후보를 context에 넣는다. 최신성·출처·권한 제어가 중요하고 corpus가 context보다 클 때 유력하다." 검색 recall, context assembly, generation/attribution을 분리해 병목을 측정하고 long-context·fine-tuning·database/API 조회와 비교하세요.

## 파이프라인 한눈에

RAG는 **오프라인(색인을 미리 구축)** 과 **온라인(질문이 들어올 때마다 처리)** 두 부분으로 나뉩니다. 오프라인은 문서를 벡터로 바꿔 저장해 두는 준비 단계이고, 온라인은 사용자가 질문할 때 실시간으로 도는 흐름입니다.

<figure>
<svg viewBox="0 0 720 340" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <defs>
    <marker id="rg" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker>
  </defs>

  <!-- offline lane -->
  <text x="30" y="22" fill="#98a3b2" font-size="11" font-weight="700">비동기: 색인 구축·증분 갱신</text>
  <rect x="30" y="34" width="100" height="36" rx="7" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="80" y="56" text-anchor="middle" fill="currentColor">문서들</text>
  <rect x="165" y="34" width="100" height="36" rx="7" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="215" y="51" text-anchor="middle" fill="currentColor">청킹</text>
  <text x="215" y="64" text-anchor="middle" fill="#98a3b2" font-size="9.5">문단 조각으로</text>
  <rect x="300" y="34" width="100" height="36" rx="7" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="350" y="51" text-anchor="middle" fill="currentColor">임베딩</text>
  <text x="350" y="64" text-anchor="middle" fill="#98a3b2" font-size="9.5">조각 → 벡터</text>
  <path d="M130 52 H165" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M265 52 H300" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M350 70 V92 H294 V109" stroke="#98a3b2" stroke-width="1.4" fill="none" marker-end="url(#rg)"/>

  <!-- vector DB cylinder -->
  <path d="M242 120 a52 11 0 0 0 104 0 v38 a52 11 0 0 1 -104 0 z" fill="#6366f1" opacity="0.18" stroke="#6366f1" stroke-width="1.5"/>
  <ellipse cx="294" cy="120" rx="52" ry="11" fill="#6366f1" opacity="0.32" stroke="#6366f1" stroke-width="1.5"/>
  <text x="294" y="146" text-anchor="middle" fill="currentColor" font-size="11">벡터 DB</text>
  <text x="294" y="160" text-anchor="middle" fill="#6366f1" font-size="9.5">ANN 색인</text>
  <path d="M294 169 V266" stroke="#98a3b2" stroke-width="1.4" fill="none" marker-end="url(#rg)"/>
  <text x="304" y="215" fill="#98a3b2" font-size="9.5">top-k 후보</text>

  <!-- online lane -->
  <text x="20" y="248" fill="#98a3b2" font-size="11" font-weight="700">온라인: 질의 처리 (질문이 들어올 때마다)</text>
  <rect x="20" y="270" width="100" height="40" rx="7" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <text x="70" y="294" text-anchor="middle" fill="currentColor">사용자 질문</text>
  <rect x="132" y="270" width="100" height="40" rx="7" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="182" y="288" text-anchor="middle" fill="currentColor">임베딩</text>
  <text x="182" y="302" text-anchor="middle" fill="#98a3b2" font-size="9.5">질문 → 벡터</text>
  <rect x="244" y="270" width="100" height="40" rx="7" fill="none" stroke="#12a150" stroke-width="1.5"/>
  <text x="294" y="288" text-anchor="middle" fill="currentColor">검색 retrieve</text>
  <text x="294" y="302" text-anchor="middle" fill="#98a3b2" font-size="9.5">유사 top-k</text>
  <rect x="356" y="270" width="100" height="40" rx="7" fill="none" stroke="#12a150" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="406" y="288" text-anchor="middle" fill="currentColor">(선택) 재순위</text>
  <text x="406" y="302" text-anchor="middle" fill="#98a3b2" font-size="9.5">cross-encoder</text>
  <rect x="468" y="270" width="100" height="40" rx="7" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <text x="518" y="288" text-anchor="middle" fill="currentColor">프롬프트</text>
  <text x="518" y="302" text-anchor="middle" fill="#98a3b2" font-size="9.5">+ 근거 삽입</text>
  <rect x="580" y="270" width="110" height="40" rx="7" fill="#e0533f"/>
  <text x="635" y="288" text-anchor="middle" fill="#fff">LLM 생성</text>
  <text x="635" y="302" text-anchor="middle" fill="#fff" font-size="9.5">+ 출처 인용</text>
  <path d="M120 290 H132" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M232 290 H244" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M344 290 H356" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M456 290 H468" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M568 290 H580" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
</svg>
<figcaption>위(파랑)는 <b>비동기 색인 경로</b>: 문서를 조각내고 벡터로 바꿔 저장하며, 추가·수정·삭제를 증분 반영합니다. 아래(초록/빨강)는 <b>온라인 질의 경로</b>: 질문을 벡터화해 후보를 찾고, 선택적으로 재순위한 뒤 근거와 함께 답을 생성합니다. 두 흐름이 <b>버전이 관리된 색인</b>에서 만납니다.</figcaption>
</figure>

<dl class="kv">
<dt>① 청킹(chunking)</dt><dd>긴 문서를 검색 단위인 <b>조각(chunk, 청크)</b>으로 나눕니다. 너무 크면 관련 없는 내용이 섞이고, 너무 작으면 맥락이 끊깁니다. 고정 길이+overlap은 출발점일 뿐이며, 제목·문단·표 같은 구조를 보존하는 청킹, 작은 청크로 찾고 큰 부모 문맥을 가져오는 parent-child 방식, 임베딩 전후 문맥을 보존하는 late chunking도 비교합니다.</dd>
<dt>② 임베딩 &amp; 색인(index)</dt><dd>각 조각을 [임베딩](#/llm/embeddings) 벡터로 바꿔 vector store/index에 저장합니다. 대규모 검색에는 <b>ANN</b>을 쓰며 HNSW나 IVF-PQ가 대표적입니다. Recall–latency–memory–build/update 비용은 index와 parameter·hardware·filter 조건에 따라 달라지므로 exact-search subset과 비교해 튜닝합니다.</dd>
<dt>③ 검색(retrieve)</dt><dd>질문을 문서와 호환되는 임베딩 공간으로 바꾼 뒤 상위 $k$개 조각을 가져옵니다. 모델에 따라 질의/문서용 prefix가 다를 수 있고, 점수도 cosine·dot product·L2 중 학습 규약에 맞춰야 합니다(§유사성).</dd>
<dt>④ (선택) 재순위(rerank)</dt><dd>후보 top-k를 <b>cross-encoder</b>로 질문과 함께 다시 정밀 채점해, 정말 관련 있는 것만 위로 올립니다(§bi vs cross).</dd>
<dt>⑤ 생성(generate)</dt><dd>고른 조각을 프롬프트에 넣고 근거가 부족하면 답을 보류하며 <b>출처를 인용(cite)</b>하도록 지시합니다. 지시와 인용 형식만으로 사실성이 보장되지는 않으므로, 실제 인용이 주장을 뒷받침하는지 별도로 검증합니다.</dd>
</dl>

<details class="concept-code">
<summary>개념 코드로 보기</summary>

> 아래는 색인과 질의 경로의 책임 경계를 보여 주는 Python식 **의사코드**입니다. 특정 vector DB의 실행 API가 아닙니다.

```python
def upsert_document(doc, index_version):
    chunks = structure_aware_chunk(doc.body)
    for chunk in chunks:
        index.upsert(
            id=stable_chunk_id(doc.id, chunk.offset),
            vector=embed_document(chunk.text),
            text=chunk.text,
            metadata={"tenant": doc.tenant, "acl": doc.acl,
                      "doc_version": doc.version, "index_version": index_version},
        )
    index.delete_stale_chunks(doc.id, keep_version=doc.version)  # 삭제 전파 포함

def answer(query, principal, expected_index_version):
    q = embed_query(query)                      # query/document prefix 규약 준수
    acl_filter = visible_to(principal)          # 검색 뒤가 아니라 검색 중 권한 제한
    dense = index.ann_search(q, top_k=50, where=acl_filter)
    sparse = lexical.search(query, top_k=50, where=acl_filter)
    candidates = reciprocal_rank_fusion(dense, sparse)
    candidates = require_version(candidates, expected_index_version)
    evidence = reranker.rank(query, candidates)[:5]
    evidence = deduplicate_and_fit_token_budget(evidence)

    draft = llm.generate(build_grounded_prompt(query, evidence))
    # 문서는 명령이 아닌 untrusted evidence로 구분하고, 인용 ID는 허용 목록과 대조한다.
    return validate_or_abstain(draft, allowed_citations=ids(evidence))
```

</details>

## 유사도 함수는 모델 규약을 따른다

검색의 심장은 "질문과 관련 있는 조각을 점수화해 상위 후보를 찾기"입니다. 정규화된 임베딩에는 **코사인 유사도**와 dot product가 같은 순위를 만들지만, 모든 모델이 정규화를 전제하지는 않습니다. 어떤 모델은 dot product나 L2 거리를 권장하고, 질의와 문서에 서로 다른 prefix를 요구합니다. 따라서 모델 카드의 인코딩법·거리 함수를 그대로 지키고, 실제 벡터 DB에서는 그 점수를 ANN 색인으로 근사합니다. 아래 코드는 그중 코사인 검색을 보여 주는 교육용 예입니다.

## bi-encoder vs cross-encoder

검색 단계의 핵심 트레이드오프입니다. 왜 검색용과 재순위용 모델이 다른지 이해하면 RAG 튜닝이 쉬워집니다.

<div class="proscons"><div><div class="pros-t">bi-encoder (검색용, 1차)</div>

질문과 문서를 **각각 따로** 벡터로 인코딩합니다. 문서 벡터는 오프라인에 미리 계산해 색인해 두므로, 질의 때는 질문만 벡터화하면 됩니다 → 수백만 개도 빠름. 대신 질문·문서의 상호작용을 못 봐서 정밀도는 낮음. **1차 검색(top-k 후보 뽑기)** 에 사용.

</div><div><div class="cons-t">cross-encoder (재순위용, 2차)</div>

질문+문서를 **한 입력으로 붙여** 함께 넣고 관련성 점수를 냅니다. 토큰 간 상호작용을 직접 볼 수 있어 보통 같은 후보 집합의 순위를 개선하지만, 모델·도메인에 따라 이득은 달라집니다. 문서별 점수를 미리 계산할 수 없어 느리므로 bi-encoder가 뽑은 top-k **후보에만** 적용합니다.

</div></div>

전형적 조합: bi-encoder로 top-50을 빠르게 뽑고 → cross-encoder로 재순위해 top-5만 프롬프트에 넣기. 속도(1차)와 정밀도(2차)를 모두 챙깁니다.

## Hybrid — 밀집(dense) + 희소(sparse)

임베딩 기반 검색(dense)은 "의미가 비슷한" 조각을 잘 찾지만, **정확한 단어·희귀한 고유명사·제품 코드**(예: `ERR-4021`, 특정 API 이름)는 놓치기 쉽습니다. 반대로 전통적 키워드 검색인 **BM25**(단어 빈도 기반 희소 검색, sparse)는 그런 정확 일치에 강합니다.

**Hybrid 검색**은 둘의 점수나 순위를 합칩니다(예: calibrated weighted sum, RRF). Query mix에 rare identifier와 paraphrase가 함께 있으면 recall을 높일 수 있지만, fusion tuning·추가 latency가 들고 단일 retriever보다 나쁠 수도 있으므로 labeled retrieval set에서 비교합니다.

## 직접 돌려보기 — top-k 검색

RAG 검색의 심장인 "**질문 벡터와 가장 유사한 문서 top-k의 인덱스 찾기**"를 코사인 유사도로 구현해 봅시다. 아래 라이브 에디터에서 채워 넣고 **▶ Run tests**를 누르면 실제로 채점됩니다. (막히면 **Solution**을 열어 보세요.)

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"retrieve_topk","packages":["numpy"],"approx":false,"starter":"def retrieve_topk(query, docs, k=2):\n    # query:(d,) 질문 벡터, docs:(N,d) 문서(조각) 벡터들.\n    # 코사인 유사도 = (q·d) / (|q||d|) 가 큰 순서로\n    # 상위 k개 문서의 '인덱스' 리스트를 내림차순으로 반환하세요.\n    # (동점이 없도록 테스트를 구성했습니다.)\n    pass","tests":[{"args":[[1,0],[[1,0],[0,1],[1,1],[-1,0]],2],"expect":[0,2]},{"args":[[1,1],[[1,0],[0,1],[2,2],[-1,-1]],1],"expect":[2]},{"args":[[0,1],[[1,0],[0,2],[0.1,0.9]],2],"expect":[1,2]}],"solution":"import numpy as np\n\ndef retrieve_topk(query, docs, k=2):\n    q = np.asarray(query, dtype=float)\n    D = np.asarray(docs, dtype=float)\n    qn = q / (np.linalg.norm(q) + 1e-12)\n    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)\n    sims = Dn @ qn                      # 코사인 유사도 (N,)\n    order = np.argsort(sims)[::-1]      # 내림차순 정렬\n    return order[:k].tolist()"}
</script>
</div>

이 코드는 dense cosine retrieval의 최소 원리입니다. 실제 vector store는 ANN 근사, metadata/ACL filter, sharding, delete/update, metric convention과 tie handling을 더하므로 exact cosine top-k와 동일하다고 가정하지 않습니다.

## RAG vs long-context vs fine-tuning

셋 다 "모델에게 지식을 주는" 방법이지만 쓰임이 다릅니다. 이 구분을 명확히 말하는 것이 면접의 단골 포인트입니다.

| 방법 | 언제 쓰나 | 장점 | 한계 |
| --- | --- | --- | --- |
| **RAG** | 최신성·출처·대규모 지식베이스 | 재학습 불필요, 출처 인용, 즉시 갱신 | 검색 품질에 의존, 파이프라인 복잡 |
| **Long-context** | 문서가 몇 개뿐, 통째로 넣어도 됨 | 단순, 전체 맥락 유지 | 비쌈(토큰↑), "lost in the middle" |
| **Fine-tuning** | 말투·형식·기술 습득 | 지식 내재화, 추론 시 빠름 | 갱신 어려움, 사실 주입엔 비효율 |

> [!NOTE] 실무 감각
> "사실·지식은 RAG로, 행동·형식은 fine-tuning으로." 최신 정보나 출처가 필요한 **사실**은 RAG가, 특정 말투·출력 형식·도메인 **기술**은 fine-tuning이 낫습니다. 문서가 몇 개뿐이면 그냥 프롬프트에 다 넣는 long-context가 제일 단순합니다. 셋은 배타적이지 않아 함께 쓰기도 합니다.

## 흔한 실패 모드

RAG가 틀릴 때는 아래 단계별 원인을 분리하세요. 어디가 병목인지부터 진단하는 습관이 중요합니다.

- **나쁜 청킹(chunking)**: 너무 크게 자르면 노이즈가 섞이고, 답이 두 조각에 걸치면 어느 쪽에도 온전히 안 담깁니다 → 구조 보존, parent-child, 크기·overlap을 데이터로 비교.
- **검색 실패(retrieval miss)**: 정답 조각이 top-k에 안 들어옵니다. 이 경우 generation은 복구할 근거가 없으므로 hybrid, reranking, query rewriting, embedding/chunking을 비교합니다.
- **근거를 줬는데도 환각(hallucination)**: 관련 조각이 있어도 모델이 근거 밖 주장을 할 수 있습니다 → 답변 보류 규칙과 인용을 요구하고, claim-citation entailment를 후처리나 평가로 확인합니다. 프롬프트만으로 이를 강제할 수는 없습니다.
- **context utilization 실패**: 긴 context에서 위치·중복·상충 evidence 때문에 필요한 근거를 놓칠 수 있습니다 → context 길이·ordering·dedup·compression을 model/workload별로 ablate합니다.
- **신선도·권한 실패**: 삭제된 문서가 남거나 다른 사용자/테넌트의 청크가 검색됩니다 → 문서·임베딩·색인 버전을 기록하고 삭제 전파, ACL/row-level filter, tenant 격리를 검색 단계에서 적용합니다.
- **오염과 간접 프롬프트 주입**: 검색 문서가 악성 지시나 잘못된 정보를 포함할 수 있습니다 → 검색 결과를 신뢰할 수 없는 데이터로 취급하고, 지시와 근거를 분리하며 출처·PII 정책, 도구 권한, 감사 로그를 둡니다. 상세 운영·보안 설계는 [LLM 시스템 설계](#/system-design/llm-systems)를 참고하세요.

## 평가는 검색과 답변을 분리한다

끝단 정답률 하나만 보면 병목을 찾기 어렵습니다. 정답 근거가 있는 평가 세트로 **retrieval Recall@k**, MRR/nDCG를 먼저 보고, 그다음 **답변 정확성**, 근거 충실도(faithfulness/attribution), citation precision/recall, 근거가 없을 때의 답변 보류율·calibration을 측정합니다. 품질과 함께 p50/p95 지연, 검색·재순위·생성 비용도 기록합니다. 자동 LLM judge는 편향과 비결정성이 있으므로 표본 사람 평가와 규칙 기반 인용 검사를 함께 사용합니다.

## Agentic RAG (맛보기)

지금까지는 "한 번 검색 → 한 번 생성"인 **고정 파이프라인**이었습니다. **Agentic RAG**는 [에이전트](#/llm/agents)가 검색을 **하나의 도구(tool)** 로 쥐고, *언제·무엇을* 검색할지 스스로 정하는 방식입니다. 결과가 부족하면 질의를 바꿔 다시 검색하고(multi-hop), "A와 B를 비교해줘" 같은 복합 질문을 여러 번의 검색으로 쪼개 풉니다. 즉 **에이전트 루프 안의 도구로서의 RAG**입니다. 대신 호출 수와 비용이 늘고 탈선 위험이 생기므로, 단순 질문에는 고정 RAG가 여전히 낫습니다.

## Q&A

<details class="qa"><summary>왜 그냥 문서를 전부 프롬프트에 넣지 않나요? (long-context)</summary>
<div class="qa-body">

**짧게:** 비용·속도·정확도 때문입니다.

**깊게:** corpus가 context window보다 크면 전부 넣을 수 없고, 긴 입력은 token cost·TTFT와 context-utilization 문제를 늘릴 수 있습니다. RAG는 관련 후보만 넣어 비용을 줄일 수 있고 provenance를 연결하기 쉽지만 retrieval miss라는 새 실패를 만듭니다. 문서가 적다면 long-context baseline과 quality–latency–cost를 먼저 비교하세요.
</div></details>

<details class="qa"><summary>검색 품질을 어떻게 올리나요?</summary>
<div class="qa-body">

**짧게:** 더 나은 임베딩 모델, hybrid(dense+BM25) 검색, cross-encoder 재순위, 청킹·질의 튜닝.

**깊게:** (1) 도메인에 맞는 임베딩 모델 선택/파인튜닝, (2) 의미 검색(dense)과 키워드 검색(BM25)을 합친 **hybrid**로 재현율↑, (3) top-k를 **cross-encoder**로 재순위해 정밀도↑, (4) 청크 크기·overlap 조정, (5) **질의 재작성(query rewriting)** 으로 모호한 질문을 검색 친화적으로. 무엇보다 병목이 검색인지 생성인지부터 진단하세요 — 정답 조각이 top-k에 들어오는지(retrieval recall) 먼저 측정하면 됩니다.
</div></details>

## Cheat-sheet

| 개념 | 한 줄 |
| --- | --- |
| RAG | 관련 문서를 검색해 프롬프트에 넣고 그 근거로 생성 (재학습 없이 지식 주입) |
| 파이프라인 | 청킹 → 임베딩·색인 → 검색(top-k) → (재순위) → 생성+인용 |
| 비동기 vs 온라인 | 색인은 구축 후에도 추가·수정·삭제·버전을 증분 반영, 검색·생성은 질문마다 실행 |
| 벡터 DB / ANN | 임베딩을 저장하고 HNSW 등 근사 최근접 이웃으로 빠르게 검색 |
| 유사도 | cosine·dot·L2 중 임베딩 모델의 학습·정규화 규약을 따름 |
| bi- vs cross-encoder | 따로 인코딩(빠름, 1차 검색) vs 함께 채점(정확, 재순위) |
| Hybrid | dense+lexical 융합; query mix와 labeled set에서 이득 검증 |
| RAG vs FT | 외부 사실·provenance는 RAG 후보, 반복 behavior는 fine-tuning 후보; 조합 가능 |
| 병목 | retrieval·assembly·generation/attribution을 분리 측정 |
| 운영·보안 | ACL/tenant filter, 삭제 전파, 색인 버전, provenance, prompt-injection 격리 |
| 평가 | Recall@k/MRR/nDCG와 답변 정확성·충실도·인용·보류율·지연/비용을 분리 측정 |

**다음:** [Agentic AI & Tool Use](#/llm/agents) · [임베딩](#/llm/embeddings)
