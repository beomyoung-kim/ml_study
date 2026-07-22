# Prompting & In-Context Learning

> [!NOTE] Goal of this chapter
> Learn how to steer an LLM toward desired behavior through input context **without retraining it**. We examine **in-context learning**, in which a model infers a task, format, and label mapping from a few examples, while separating the learning pressure created by next-token pretraining from the still-debated internal mechanism.

## What and why

As [LLM Fundamentals](#/llm/fundamentals) explains, an LLM predicts the **next token** from preceding tokens. Prompting steers this predictor through **context**: what you put before the answer changes what is likely to follow. Unlike fine-tuning, it can be applied **immediately without additional training cost**, but long instructions and demonstrations increase input tokens, latency, and API cost on every request.

<div class="proscons"><div><div class="pros-t">Prompting</div>

Change only the input context. You can experiment immediately without retraining, so this is the first approach for most problems.

</div><div><div class="cons-t">Fine-tuning</div>

Change the weights, or parameters. It requires data, GPUs, and time, but can **internalize** a format or style in the model.

</div></div>

In one sentence: **prompting designs what to ask**, while leaving the model itself unchanged.

> [!TIP] Interview one-liner
> "Prompting conditions behavior through context without changing parameters. In few-shot learning, the model infers the examples' format, task, and label relationship in context and generates a continuation." Do not present one internal explanation as settled: pattern completion, meta-learning, and implicit Bayesian inference remain active perspectives on ICL.

## Zero-shot vs Few-shot

These are the two basic modes.

- **Zero-shot**: provide an **instruction only**, with no examples: "Summarize the following in one sentence."
- **Few-shot**: provide $k$ correct **input→output demonstrations**, then append a new input. The model follows their **format** and decision rule.

<figure>
<svg viewBox="0 0 660 220" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <text x="150" y="22" text-anchor="middle" font-weight="700" fill="#6366f1">Zero-shot (0 examples)</text>
  <rect x="30" y="35" width="240" height="70" rx="8" fill="none" stroke="#6366f1" stroke-width="1.5"/>
  <text x="45" y="58" fill="currentColor">"Sentiment of this review? →"</text>
  <text x="45" y="80" fill="#98a3b2" font-size="11">Instruction, then answer immediately</text>
  <rect x="30" y="120" width="240" height="30" rx="6" fill="#12a150"/>
  <text x="150" y="140" text-anchor="middle" fill="#fff">→ "positive"</text>
  <text x="510" y="22" text-anchor="middle" font-weight="700" fill="#e0533f">Few-shot (several examples)</text>
  <rect x="390" y="35" width="240" height="100" rx="8" fill="none" stroke="#e0533f" stroke-width="1.5"/>
  <text x="405" y="55" fill="#98a3b2" font-size="11">"Excellent → positive</text>
  <text x="405" y="72" fill="#98a3b2" font-size="11"> Poor → negative</text>
  <text x="405" y="89" fill="#98a3b2" font-size="11"> Not worth the money → negative"</text>
  <text x="405" y="112" fill="currentColor">"This review? →"</text>
  <text x="405" y="128" fill="#e0533f" font-size="11">Examples establish format and criteria</text>
  <rect x="390" y="150" width="240" height="30" rx="6" fill="#12a150"/>
  <text x="510" y="170" text-anchor="middle" fill="#fff">→ conditioned on format and criteria</text>
</svg>
<figcaption>Zero-shot provides only an instruction; few-shot also supplies correct demonstrations to establish a format and criterion. Few-shot gains depend on example representativeness, order, label balance, and the model. Poor demonstrations can be worse than zero-shot.</figcaption>
</figure>

## In-Context Learning — why is it surprising, and how might it work?

In few-shot prompting, the model's **weights do not change at all**. It performs a new task from examples in the prompt even though no gradient descent or backpropagation occurs. This is **in-context learning (ICL)**, a very different sense of "learning" from the parameter-updating [machine-learning loop](#/foundations/what-is-ml).

**Why can next-token prediction support it?** Pretraining data often contains repeated structures in which a pattern continues after demonstrations, and exploiting contextual rules and mappings helps predict those continuations. This is one intuition, not a complete explanation. Pattern completion, meta-learning, induction heads, and implicit Bayesian inference remain complementary or competing research accounts.

<figure>
<svg viewBox="0 0 640 180" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="13">
  <rect x="20" y="55" width="90" height="34" rx="6" fill="none" stroke="#6366f1" stroke-width="1.6"/>
  <text x="65" y="77" text-anchor="middle" fill="currentColor">A → B</text>
  <rect x="130" y="55" width="90" height="34" rx="6" fill="none" stroke="#6366f1" stroke-width="1.6"/>
  <text x="175" y="77" text-anchor="middle" fill="currentColor">C → D</text>
  <rect x="240" y="55" width="90" height="34" rx="6" fill="none" stroke="#6366f1" stroke-width="1.6"/>
  <text x="285" y="77" text-anchor="middle" fill="currentColor">E → ?</text>
  <text x="175" y="30" text-anchor="middle" fill="#98a3b2" font-size="11">Examples in the prompt (context)</text>
  <path d="M340 72 H430" stroke="#98a3b2" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="385" y="60" text-anchor="middle" fill="#98a3b2" font-size="11">continue the pattern</text>
  <rect x="440" y="55" width="180" height="34" rx="6" fill="#e0533f"/>
  <text x="530" y="77" text-anchor="middle" fill="#fff">Predict F (continuation)</text>
  <text x="530" y="120" text-anchor="middle" fill="#98a3b2" font-size="11">No weight update—only next-token prediction</text>
  <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>The model infers relationships established by the prompt—A→B, C→D, and so on—and completes the continuation. No weights are updated, and example <b>format, order, and representativeness</b> can substantially affect accuracy. The internal mechanism that enables this remains an active research topic.</figcaption>
</figure>

## Chat templates: system / developer / user / assistant / tool

Production chat models receive role-separated messages rather than one undifferentiated document. A **chat template** wraps those roles in special tokens such as `<|system|>` and serializes them back into one token sequence. The following three roles are a useful minimum mental model. Real APIs may also provide **developer** messages for application rules and **tool** messages for tool results; follow the provider specification for supported roles and precedence.

<figure>
<svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <!-- system -->
  <rect x="60" y="20" width="500" height="52" rx="10" fill="none" stroke="#6366f1" stroke-width="1.8"/>
  <rect x="60" y="20" width="90" height="52" rx="10" fill="#6366f1"/>
  <text x="105" y="50" text-anchor="middle" fill="#fff" font-weight="700">system</text>
  <text x="170" y="42" fill="currentColor">Rules, persona, and tone</text>
  <text x="170" y="60" fill="#98a3b2" font-size="11">"You are a helpful English translator"</text>
  <!-- user -->
  <rect x="60" y="88" width="500" height="52" rx="10" fill="none" stroke="#0ea5e9" stroke-width="1.8"/>
  <rect x="60" y="88" width="90" height="52" rx="10" fill="#0ea5e9"/>
  <text x="105" y="118" text-anchor="middle" fill="#fff" font-weight="700">user</text>
  <text x="170" y="110" fill="currentColor">The actual request or question</text>
  <text x="170" y="128" fill="#98a3b2" font-size="11">"Translate 'Hello, world'"</text>
  <!-- assistant -->
  <rect x="60" y="156" width="500" height="52" rx="10" fill="none" stroke="#e0533f" stroke-width="1.8" stroke-dasharray="5 4"/>
  <rect x="60" y="156" width="90" height="52" rx="10" fill="#e0533f"/>
  <text x="105" y="186" text-anchor="middle" fill="#fff" font-weight="700">assistant</text>
  <text x="170" y="178" fill="#e0533f">Slot the model fills; generation starts here</text>
  <text x="170" y="196" fill="#98a3b2" font-size="11">"Hello, world"</text>
  <path d="M310 72 V88" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#a2)"/>
  <path d="M310 140 V156" stroke="#98a3b2" stroke-width="1.4" marker-end="url(#a2)"/>
  <text x="310" y="232" text-anchor="middle" fill="#98a3b2" font-size="11">Special tokens wrap the roles into one input sequence</text>
  <defs><marker id="a2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#98a3b2"/></marker></defs>
</svg>
<figcaption>In a representative chat template, system supplies higher-level rules, user supplies the request, and assistant marks the generation position. Actual role names, precedence, and tool-message formats are contracts of the provider and model template.</figcaption>
</figure>

- **system**: global rules, role, and tone, maintained throughout the conversation.
- **developer**: application-specific rules. Support and precedence relative to system depend on the API.
- **user**: the actual request.
- **assistant**: the position the model must fill; generation starts here.
- **tool**: the result of a tool call. Manage its trust boundary and provenance separately.

> [!WARNING] Common trap
> A format different from the model's training-time chat template can degrade performance, trigger abnormal stopping, or confuse roles. Use the model card's template and tokenizer API, and inspect actual token IDs for duplicated special tokens, the assistant generation marker, and tool-turn boundaries. The same trap appears in [VLM Implementation Details](#/vlm/practical).

## Chain-of-Thought — the bridge to reasoning

For some mathematical and multi-step tasks, **chain-of-thought (CoT)** prompting that elicits intermediate work can improve accuracy. The effect depends on the model and task, and current reasoning APIs may hide internal reasoning or return only a summary. In production, it is safer to ask for **verifiable intermediate results, concise justification, and a check of the final answer** rather than raw private reasoning.

Why can it help? Writing intermediate results as tokens lets later generation steps read them back, providing additional serial computation and scratch space. A longer explanation is not necessarily more faithful or correct, so validate with a final-answer checker and task metric.

```text
[Question] Three apples cost $2.40. How much do seven cost?

(zero-shot request)       → "$4.80"          (wrong)

(+ "Think step by step") → "One apple = $2.40 / 3 = $0.80.
                            Seven = $0.80 × 7 = $5.60." → intermediate work is checkable
```

Separate prompted CoT from methods that train or select longer computations using reasoning traces, verifiable rewards, or test-time search. [Reasoning & Test-Time Compute](#/llm/reasoning) distinguishes training from inference rather than collapsing them into one phenomenon.

## Concrete example: one complete few-shot prompt

Here is a complete sentiment-classification prompt:

```text
Answer with exactly one label: "positive" or "negative."

Review: Excellent; I will come again.
Sentiment: positive

Review: Poor; I will not return.
Sentiment: negative

Review: Not worth the price.
Sentiment: negative

Review: The wait was long, but the food was excellent.
Sentiment:
```

The final `Sentiment:` is blank. The model continues the pattern established by the three demonstrations and fills in `positive`, without changing its weights.

## Build it yourself — a few-shot prompt assembler

A few-shot prompt is ultimately just a string assembled according to a rule. Complete a function that receives demonstrations and a new query and returns a prompt string like the one above. Open **Solution** if you get stuck.

<div class="widget" data-widget="code">
<script type="application/json" class="code-config">
{"func":"build_few_shot","packages":[],"starter":"def build_few_shot(examples, query):\n    # examples: a list of examples shaped [[review, sentiment], ...]\n    # query: the new review string whose sentiment should be predicted\n    # Build and return a prompt string in this format.\n    #   Each example: '리뷰: {review}\\n감정: {sentiment}\\n'\n    #   Final item:  '리뷰: {query}\\n감정:' (leave the sentiment blank)\n    # Join example blocks and the final query with '\\n'.\n    # TODO\n    pass","tests":[{"args":[[["최고였어","긍정"],["별로다","부정"]],"그냥 그래"],"expect":"리뷰: 최고였어\n감정: 긍정\n\n리뷰: 별로다\n감정: 부정\n\n리뷰: 그냥 그래\n감정:"},{"args":[[["맛있다","긍정"]],"별로"],"expect":"리뷰: 맛있다\n감정: 긍정\n\n리뷰: 별로\n감정:"}],"solution":"def build_few_shot(examples, query):\n    blocks = [f\"리뷰: {r}\\n감정: {s}\" for r, s in examples]\n    blocks.append(f\"리뷰: {query}\\n감정:\")\n    return \"\\n\\n\".join(blocks)"}
</script>
</div>

## Production prompting patterns and traps

<dl class="kv">
<dt>Make instructions clear and specific</dt><dd>"Summarize in three bullets, no more than 20 characters each" is more stable than "summarize." Ambiguity becomes output variance.</dd>
<dt>Pin down the output format</dt><dd>Specify "output JSON only, with no explanation," and use an example to anchor it when necessary.</dd>
<dt>Put roles and constraints in system</dt><dd>Place persona, prohibitions, and tone in the system message; reserve user for the request at hand.</dd>
<dt>Use staged reasoning for difficult tasks</dt><dd>Elicit intermediate reasoning where it helps, but separate it from the required final format—for example, "put only the answer on the last line."</dd>
</dl>

> [!WARNING] A prompt can request, not guarantee
> Even "always output JSON" can fail. Use constrained decoding or a JSON schema when syntax must be guaranteed; see [Instruction Tuning & Decoding](#/vlm/instruction-tuning). Even then, **schema-valid JSON need not be factually or semantically correct**, and max-token truncation or tool errors require separate handling. Prompting steers, constrained decoding enforces syntax, and a validator checks semantics.

> [!WARNING] Untrusted input and prompt injection
> Text retrieved from a document, web page, or tool output is **data**, not a higher-priority instruction. Delimit it, separate authoritative instructions from external content, and give tools least privilege, allowlists, and approval gates. Filtering phrases such as "ignore previous instructions" is not sufficient.

## Q&A

<details class="qa"><summary>Is few-shot always better than zero-shot?</summary>
<div class="qa-body">

**Short:** No. Strong instruction-tuned models can perform well zero-shot, and poor demonstrations can be harmful.

**Deep:** Demonstrations consume tokens, context length, and money. Unrepresentative or biased examples can steer the model toward the wrong format or distribution. Even changing their **order** or label balance can change results. Few-shot is especially useful when the output **format** is demanding or the task is ambiguous enough that examples must define the criterion.
</div></details>

<details class="qa"><summary>Is in-context learning really "learning"?</summary>
<div class="qa-body">

**Short:** Not in the usual parameter-learning sense, because no parameters change. It is closer to conditioning on context.

**Deep:** The learning loop in [What Is ML?](#/foundations/what-is-ml) updates weights with gradient descent. ICL performs no such update; an already-trained model recognizes and continues a pattern in the prompt. Start a new conversation—remove the context—and what was "learned" disappears. Fine-tuning is required to internalize a capability persistently.
</div></details>

<details class="qa"><summary>When should I fine-tune if prompting is not enough?</summary>
<div class="qa-body">

**Short:** When prompting cannot stabilize format or quality, or when the same task must run repeatedly at high volume and low latency.

**Deep:** These methods solve different problems rather than forming a mandatory sequence. Use prompting or few-shot examples for instruction ambiguity, [RAG](#/llm/rag) for current and authoritative external facts, and fine-tuning for repeated behavior, style, or domain adaptation. Establish a baseline, compare quality, latency, token cost, update frequency, and security, then combine methods as needed.
</div></details>

## Cheat-sheet

| Concept | One line |
| --- | --- |
| Prompting | Steer behavior through input context without retraining; fast to test, but it still costs tokens and latency |
| Zero- vs few-shot | Instruction only vs instruction plus $k$ correct demonstrations |
| In-context learning | Continue a pattern in the prompt without updating weights |
| Why it works | Pretraining pressure to use contextual rules; several internal mechanisms remain under study |
| Chat template | system (rules) / user (request) / assistant (generation)—use the exact format |
| Chain-of-thought | Useful for some reasoning tasks; validate by model and task, and ask for concise evidence and checks rather than private reasoning |
| Request vs guarantee | prompt = steering · constrained decoding = syntax · validator = semantics |
| When to fine-tune | When prompting cannot stabilize format or quality, or for repeated high-volume, low-latency behavior |

**Next:** [LLM Fundamentals](#/llm/fundamentals) · [Reasoning & Test-Time Compute](#/llm/reasoning)
