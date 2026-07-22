# RAG (Retrieval-Augmented Generation)

> [!NOTE] Goal of this chapter
> Build RAG from first principles: the representative method for retrieving **external evidence and supplying it to an LLM**. It can use current or internal documents without retraining the model, but whether retrieved material actually supports the answer requires separate evaluation. This chapter connects the retrieval foundations immediately above [Embeddings](#/llm/embeddings) to the complete pipeline.

## What and why

A checkpoint's **parametric knowledge** is tied to the data and updates available during training. You cannot assume it contains current news or a private corpus. Missing information is one source of hallucination, but even with context a model can synthesize it incorrectly or follow its prior instead.

RAG first **retrieves** relevant documents, **augments** the input with selected context, and **generates** an answer. You can update the corpus without changing model parameters, but production operation still requires propagation of additions, edits, and deletions; ACL enforcement; embedding and index versioning; and backfills.

> [!TIP] Interview one-liner
> "RAG places candidate external evidence in context without changing model parameters. It is attractive when freshness, provenance, access control, and a corpus larger than the context window matter." Separate retrieval recall, context assembly, and generation or attribution when measuring bottlenecks, and compare against long context, fine-tuning, and direct database or API lookup.

## The pipeline at a glance

RAG has two parts: an **offline path**, which builds the index ahead of time, and an **online path**, which executes for every question. Offline processing converts documents to stored vectors; online processing runs in real time when a user asks a question.

<figure>
<svg viewBox="0 0 720 340" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <defs>
    <marker id="rg" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker>
  </defs>

  <!-- offline lane -->
  <text x="30" y="22" fill="#98a3b2" font-size="11" font-weight="700">Asynchronous: index build and incremental update</text>
  <rect x="30" y="34" width="100" height="36" rx="7" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="80" y="56" text-anchor="middle" fill="currentColor">Documents</text>
  <rect x="165" y="34" width="100" height="36" rx="7" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="215" y="51" text-anchor="middle" fill="currentColor">Chunking</text>
  <text x="215" y="64" text-anchor="middle" fill="#98a3b2" font-size="9.5">into passages</text>
  <rect x="300" y="34" width="100" height="36" rx="7" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="350" y="51" text-anchor="middle" fill="currentColor">Embedding</text>
  <text x="350" y="64" text-anchor="middle" fill="#98a3b2" font-size="9.5">chunk → vector</text>
  <path d="M130 52 H165" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M265 52 H300" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M350 70 V92 H294 V109" stroke="#98a3b2" stroke-width="1.4" fill="none" marker-end="url(#rg)"/>

  <!-- vector DB cylinder -->
  <path d="M242 120 a52 11 0 0 0 104 0 v38 a52 11 0 0 1 -104 0 z" fill="#6366f1" opacity="0.18" stroke="#6366f1" stroke-width="1.5"/>
  <ellipse cx="294" cy="120" rx="52" ry="11" fill="#6366f1" opacity="0.32" stroke="#6366f1" stroke-width="1.5"/>
  <text x="294" y="146" text-anchor="middle" fill="currentColor" font-size="11">Vector DB</text>
  <text x="294" y="160" text-anchor="middle" fill="#6366f1" font-size="9.5">ANN index</text>
  <path d="M294 169 V266" stroke="#98a3b2" stroke-width="1.4" fill="none" marker-end="url(#rg)"/>
  <text x="304" y="215" fill="#98a3b2" font-size="9.5">top-k candidates</text>

  <!-- online lane -->
  <text x="20" y="248" fill="#98a3b2" font-size="11" font-weight="700">Online: query processing (for every question)</text>
  <rect x="20" y="270" width="100" height="40" rx="7" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <text x="70" y="294" text-anchor="middle" fill="currentColor">User query</text>
  <rect x="132" y="270" width="100" height="40" rx="7" fill="none" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="182" y="288" text-anchor="middle" fill="currentColor">Embedding</text>
  <text x="182" y="302" text-anchor="middle" fill="#98a3b2" font-size="9.5">query → vector</text>
  <rect x="244" y="270" width="100" height="40" rx="7" fill="none" stroke="#12a150" stroke-width="1.5"/>
  <text x="294" y="288" text-anchor="middle" fill="currentColor">Retrieve</text>
  <text x="294" y="302" text-anchor="middle" fill="#98a3b2" font-size="9.5">similar top-k</text>
  <rect x="356" y="270" width="100" height="40" rx="7" fill="none" stroke="#12a150" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="406" y="288" text-anchor="middle" fill="currentColor">Optional rerank</text>
  <text x="406" y="302" text-anchor="middle" fill="#98a3b2" font-size="9.5">cross-encoder</text>
  <rect x="468" y="270" width="100" height="40" rx="7" fill="none" stroke="#98a3b2" stroke-width="1.5"/>
  <text x="518" y="288" text-anchor="middle" fill="currentColor">Prompt</text>
  <text x="518" y="302" text-anchor="middle" fill="#98a3b2" font-size="9.5">+ evidence</text>
  <rect x="580" y="270" width="110" height="40" rx="7" fill="#e0533f"/>
  <text x="635" y="288" text-anchor="middle" fill="#fff">LLM generation</text>
  <text x="635" y="302" text-anchor="middle" fill="#fff" font-size="9.5">+ citations</text>
  <path d="M120 290 H132" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M232 290 H244" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M344 290 H356" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M456 290 H468" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
  <path d="M568 290 H580" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#rg)"/>
</svg>
<figcaption>The upper blue lane is the <b>asynchronous indexing path</b>: split documents, embed and store them, and incrementally propagate additions, edits, and deletions. The lower green/red lane is the <b>online query path</b>: embed the question, retrieve candidates, optionally rerank them, then generate from evidence. The flows meet at a <b>versioned index</b>.</figcaption>
</figure>

<dl class="kv">
<dt>① Chunking</dt><dd>Split long documents into retrieval units called <b>chunks</b>. Too large mixes irrelevant content; too small breaks context. Fixed length plus overlap is only a baseline. Compare structure-aware chunking that preserves headings, paragraphs, and tables; parent-child retrieval that finds a small chunk but supplies a larger parent; and late chunking that preserves context around embedding.</dd>
<dt>② Embedding and indexing</dt><dd>Turn each chunk into an [embedding](#/llm/embeddings) and store it in a vector store or index. Large-scale search uses <b>approximate nearest neighbors (ANN)</b>, with HNSW and IVF-PQ as representative choices. Recall, latency, memory, and build/update cost depend on the index, parameters, hardware, and filters; tune against an exact-search subset.</dd>
<dt>③ Retrieve</dt><dd>Map the question into an embedding space compatible with the documents and fetch the top $k$ chunks. Models may require different query and document prefixes, and the score must follow the trained convention: cosine, dot product, or L2.</dd>
<dt>④ Optional rerank</dt><dd>Score the top-k candidates more precisely together with the query using a <b>cross-encoder</b>, then move the truly relevant candidates upward.</dd>
<dt>⑤ Generate</dt><dd>Place selected chunks in the prompt, require abstention when evidence is insufficient, and ask for <b>citations</b>. Instructions and citation format alone do not guarantee factual support, so separately verify whether cited evidence entails each claim.</dd>
</dl>

<details class="concept-code">
<summary>View conceptual code</summary>

> This Python-style **pseudocode** shows the responsibility boundary between indexing and query serving. It is not the executable API of a particular vector database.

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
    index.delete_stale_chunks(doc.id, keep_version=doc.version)  # Propagate deletes.

def answer(query, principal, expected_index_version):
    q = embed_query(query)                      # Honor query/document prefix conventions.
    acl_filter = visible_to(principal)          # Enforce access during, not after, retrieval.
    dense = index.ann_search(q, top_k=50, where=acl_filter)
    sparse = lexical.search(query, top_k=50, where=acl_filter)
    candidates = reciprocal_rank_fusion(dense, sparse)
    candidates = require_version(candidates, expected_index_version)
    evidence = reranker.rank(query, candidates)[:5]
    evidence = deduplicate_and_fit_token_budget(evidence)

    draft = llm.generate(build_grounded_prompt(query, evidence))
    # Delimit documents as untrusted evidence, not instructions; allowlist citation IDs.
    return validate_or_abstain(draft, allowed_citations=ids(evidence))
```

</details>

## The similarity function follows the model contract

Retrieval scores chunks for relevance to a query and selects the highest-ranked candidates. Cosine similarity and dot product induce the same ordering for normalized embeddings, but not every model assumes normalization. Some recommend dot product or L2 and require different prefixes for queries and documents. Follow the model card's encoding method and distance function exactly; a vector database approximates that score with an ANN index. The exercise below uses cosine only as an educational example.

## Bi-encoder vs cross-encoder

This is the central retrieval-stage trade-off. Understanding why retrieval and reranking use different models makes RAG tuning much easier.

<div class="proscons"><div><div class="pros-t">Bi-encoder: first-stage retrieval</div>

Encode the query and each document **separately**. Document vectors can be precomputed and indexed offline, so only the query is encoded online and millions of candidates can be searched quickly. The model cannot directly model query-document token interactions, reducing precision. Use it to generate a top-k candidate set.

</div><div><div class="cons-t">Cross-encoder: second-stage reranking</div>

Concatenate the query and document into **one input** and score relevance jointly. Direct token interactions often improve ranking within the same candidate set, although gains vary by model and domain. Scores cannot be precomputed per document, so apply it only to candidates returned by the bi-encoder.

</div></div>

A typical combination retrieves top-50 quickly with a bi-encoder, reranks with a cross-encoder, and inserts only the top five into the prompt. The first stage supplies speed; the second supplies precision.

## Hybrid — dense + sparse

Embedding-based dense retrieval finds semantically similar text but can miss **exact terms, rare proper nouns, and product codes** such as `ERR-4021`. Traditional sparse keyword retrieval such as **BM25** is strong at exact lexical matches.

**Hybrid retrieval** combines their scores or ranks, for example with a calibrated weighted sum or reciprocal rank fusion. It can raise recall when queries mix rare identifiers and paraphrases, but fusion tuning and latency add cost, and hybrid can underperform a single retriever. Compare on a labeled retrieval set.

## Try it yourself — top-k retrieval

Implement the core of RAG retrieval: find the indices of the top-k documents most similar to a query vector using cosine similarity. Fill in the live editor and select **▶ Run tests**; open **Solution** if needed.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"retrieve_topk","packages":["numpy"],"approx":false,"starter":"def retrieve_topk(query, docs, k=2):\n    # query: (d,) query vector; docs: (N,d) document-chunk vectors.\n    # Rank by cosine similarity = (q·d) / (|q||d|).\n    # Return the indices of the top-k documents in descending order.\n    # The tests contain no ties.\n    pass","tests":[{"args":[[1,0],[[1,0],[0,1],[1,1],[-1,0]],2],"expect":[0,2]},{"args":[[1,1],[[1,0],[0,1],[2,2],[-1,-1]],1],"expect":[2]},{"args":[[0,1],[[1,0],[0,2],[0.1,0.9]],2],"expect":[1,2]}],"solution":"import numpy as np\n\ndef retrieve_topk(query, docs, k=2):\n    q = np.asarray(query, dtype=float)\n    D = np.asarray(docs, dtype=float)\n    qn = q / (np.linalg.norm(q) + 1e-12)\n    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)\n    sims = Dn @ qn                      # Cosine similarity, shape (N,).\n    order = np.argsort(sims)[::-1]      # Sort descending.\n    return order[:k].tolist()"}
</script>
</div>

This is the minimal principle of dense cosine retrieval. A real vector store adds ANN approximation, metadata and ACL filters, sharding, delete/update behavior, metric conventions, and tie handling, so do not assume it exactly matches exhaustive cosine top-k.

## RAG vs long context vs fine-tuning

All three can provide information or behavior to a model, but they serve different purposes. Explaining the distinction clearly is a common interview requirement.

| Method | When to use it | Advantages | Limitations |
| --- | --- | --- | --- |
| **RAG** | Freshness, provenance, or a large knowledge base | No retraining, citations, immediate corpus updates | Depends on retrieval quality; complex pipeline |
| **Long context** | Only a few documents that fit in full | Simple; preserves complete context | Expensive in tokens; possible "lost in the middle" |
| **Fine-tuning** | Learn style, format, or a skill | Internalized behavior; low inference overhead | Hard to update; inefficient for injecting facts |

> [!NOTE] Production intuition
> "Use RAG for facts and knowledge; use fine-tuning for behavior and format." RAG is a candidate for current facts and provenance, while fine-tuning is a candidate for repeated style, output format, or domain skill. If only a few documents are involved, putting them directly into a long context may be simplest. The methods are complementary and can be combined.

## Common failure modes

When RAG fails, separate causes by stage before choosing a remedy.

- **Poor chunking:** large chunks mix noise; small chunks break context; an answer spanning two chunks may appear in neither. Compare structure-aware and parent-child strategies, chunk size, and overlap on data.
- **Retrieval miss:** the answer-bearing chunk never appears in top-k, leaving generation no evidence from which to recover. Compare hybrid retrieval, reranking, query rewriting, embeddings, and chunking.
- **Hallucination despite evidence:** the model can make claims outside relevant chunks. Require abstention and citations, then check claim-citation entailment in post-processing or evaluation. A prompt cannot guarantee it.
- **Context-utilization failure:** position, duplicates, or conflicting evidence in a long context can hide the needed fact. Ablate context length, ordering, deduplication, and compression by model and workload.
- **Freshness or authorization failure:** deleted documents remain indexed, or a user retrieves another tenant's chunk. Record document, embedding, and index versions; propagate deletions; and apply ACL or row-level filters and tenant isolation during retrieval.
- **Contamination and indirect prompt injection:** retrieved documents can contain malicious instructions or false information. Treat retrieval results as untrusted data, separate instructions from evidence, and enforce provenance, PII policy, tool authorization, and audit logs. See [Designing LLM/Agent Systems](#/system-design/llm-systems) for detailed operations and security.

## Evaluate retrieval and answers separately

One end-to-end accuracy number does not reveal the bottleneck. On an evaluation set with answer-bearing evidence, first measure retrieval Recall@k, MRR, and nDCG. Then measure answer accuracy, evidence faithfulness or attribution, citation precision and recall, abstention and calibration when evidence is missing, p50/p95 latency, and retrieval, reranking, and generation cost. Automated LLM judges are biased and nondeterministic, so combine them with sampled human review and rule-based citation checks.

## Agentic RAG — a preview

The pipeline so far retrieves once and generates once. In **agentic RAG**, an [agent](#/llm/agents) controls retrieval as a **tool** and decides when and what to search. If evidence is insufficient, it can reformulate and search again; a compound request such as "compare A and B" can become several multi-hop searches. RAG becomes a tool inside the agent loop. This increases calls, cost, and risk of deviation, so fixed RAG remains preferable for simple queries.

## Q&A

<details class="qa"><summary>Why not put every document directly in the prompt?</summary>
<div class="qa-body">

**Short:** Cost, speed, and accuracy.

**Deep:** If the corpus exceeds the context window, it cannot fit. Long inputs also raise token cost and TTFT and can worsen context utilization. RAG inserts only relevant candidates and connects provenance more naturally, but introduces retrieval misses. For a small document set, compare a long-context baseline on quality, latency, and cost first.
</div></details>

<details class="qa"><summary>How do you improve retrieval quality?</summary>
<div class="qa-body">

**Short:** Better embeddings, hybrid dense+BM25 retrieval, cross-encoder reranking, and chunking or query tuning.

**Deep:** Select or fine-tune an embedding model for the domain; combine semantic dense retrieval with lexical BM25 to raise recall; rerank top-k with a cross-encoder to raise precision; tune chunk size and overlap; and rewrite ambiguous questions into retrieval-friendly queries. Above all, diagnose retrieval separately from generation. Check whether the answer-bearing chunk enters top-k before changing the generator.
</div></details>

## Cheat-sheet

| Concept | One line |
| --- | --- |
| RAG | Retrieve relevant documents, insert them into the prompt, and generate from that evidence without retraining |
| Pipeline | Chunk → embed and index → retrieve top-k → optional rerank → generate with citations |
| Asynchronous vs online | Incrementally propagate additions, edits, deletions, and versions in the index; retrieve and generate per query |
| Vector DB / ANN | Store embeddings and search approximate neighbors quickly with HNSW or a similar index |
| Similarity | Follow the embedding model's cosine, dot-product, or L2 training and normalization convention |
| Bi- vs cross-encoder | Encode separately for fast first-stage retrieval vs jointly score for precise reranking |
| Hybrid | Fuse dense and lexical retrieval; validate gains on the query mix and a labeled set |
| RAG vs FT | External facts and provenance favor RAG; repeated behavior favors fine-tuning; they can be combined |
| Bottleneck | Measure retrieval, assembly, and generation or attribution separately |
| Operations and security | ACL/tenant filtering, deletion propagation, index versions, provenance, prompt-injection isolation |
| Evaluation | Separate Recall@k/MRR/nDCG from answer accuracy, faithfulness, citations, abstention, latency, and cost |

**Next:** [Agentic AI & Tool Use](#/llm/agents) · [Embeddings](#/llm/embeddings)
