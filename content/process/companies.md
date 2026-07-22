# Company Playbooks: Prepare with Verifiable Information

<div class="tag-row"><span class="tag">company research</span><span class="tag">dated snapshot</span><span class="tag">recruiter verification</span><span class="tag">role fit</span></div>

> [!TIP] Why this chapter exists
> Inferring an interview format or culture from a company name leads preparation astray. Even within one company, the loop may differ by **organization, team, role, level, region, and hiring date**. This chapter is not a list of company rumors to memorize. It is a method for building a **playbook for this specific application** from public information and recruiter confirmation.

> [!WARNING] Do not use undated company information
> The company names below are only labels for dividing the research scope. Do not treat round count, question difficulty, tool policy, team matching, reference checks, level, compensation, or work arrangement as fixed policy. For every application, record `checked on (as of)`, `target JD/region`, `source`, and `verification status`; use the **interview invitation and the recruiter's written response** as the final operational source of truth.

## Information reliability order

1. **The invitation and recruiter/HM responses for this application** — the authority for schedule, format, allowed tools, and deliverables.
2. **The official JD, careers page, and research organization's official page** — evidence for role scope and publicly stated research direction.
3. **Official papers, technical blogs, and product documentation** — evidence for why-us and technical conversations.
4. **Recent candidate reports and community material** — supporting material for generating questions, not evidence of policy.

When asking a recruiter about something you saw in a community, verify it neutrally instead of saying, “I heard that...”

> “Could you share the complete set of stages for this role, the format of each stage, the deliverables I should prepare, and the platform and tools that are allowed?”

## Recruiter-first verification checklist

After the first recruiter call, complete the fields below on one page. If you did not get an answer, leave the field as `unverified` rather than guessing.

| Area | Question to verify | Value to record |
| --- | --- | --- |
| Application scope | Is this for a specific team or pooled hiring across multiple teams? | Team / organization / req ID |
| Full process | What are the next stages and expected sequence? Can any stages be combined or omitted? | Stage names, sequence, owners |
| Technical evaluation | Which of DSA, ML coding, ML depth, and system design are included? | Type, language, whether code can be run |
| Research evaluation | Is there a paper deep dive, job talk, or take-home? | Topic, time, audience, Q&A, submission format |
| Tool policy | Are an IDE, documentation search, autocomplete, and generative AI allowed? | Allowed / prohibited / restricted, supporting wording |
| Logistics | Is it remote or in person? What are the time zone, location, and connection tools? | Schedule, location, backup contact |
| Decision process | Is there a debrief, committee, or team match? | Record only what was disclosed |
| References | Will references be requested, when, from whom, and with what candidate-consent process? | Confirmed procedure |
| Role terms | What are the level/title range, location, and visa or relocation terms? | Link to written response |
| Research environment | How are publication/open-source permissions and compute/data access decided? | Items to reconfirm with the HM |

For technical phone-screen preparation, use the [phone-screen day-of hub](#/process/phone-screens). For exact questions to ask recruiters and HMs, see [Recruiter & HM Screens](#/process/recruiter-hm).

## What to distinguish for each company

### Meta / FAIR-related roles

Even under the same company name, a central research organization, a product-adjacent applied team, and an engineering role may expect different outputs.

- Mark verbs in the JD such as `publish`, `novel research`, `production`, `ranking/recommendation`, `foundation model`, and `infrastructure`.
- Verify whether there is a job talk and a separate coding round, as well as the coding language, execution, and tool policies.
- Ask whether the application targets a specific team or includes team conversations after the loop.
- Choose one public paper or product and connect only **what impressed you and which problem your experience could help address**. Do not speculate about organizational structure or an undisclosed roadmap.

### NVIDIA-related roles

First distinguish research-lab, applied-research, and product/infrastructure roles.

- Verify the respective weight of papers or a research talk and of coding or system evaluation.
- Use the JD and recruiter to determine whether CUDA or GPU-kernel implementation is required, performance reasoning is sufficient, or willingness to learn is what matters.
- For hardware-aware ML, ask the HM which constraints—latency, memory, or throughput—the team actually works with.
- Discuss efficiency using public research, without assuming access to particular hardware or an internal cluster.

### Apple-related roles

MLR, AIML, and embedded-ML product roles may have similar names but different levels of research freedom and launch responsibility.

- Ask directly about the permitted scope of publication/open-source work and the approval process.
- If the JD names constraints such as on-device operation, privacy, or latency, answer with a **measurable trade-off** from your experience.
- Verify which of a paper presentation, ML coding, and system design are included.
- Do not speculate about undisclosed products, data, or models; use only wording from the public JD and official sources.

### Adobe-related roles

Distinguish pure research, applied science, and generative or creative-tooling roles close to product teams.

- If there is a research presentation, prepare not only the novelty of the representative paper but also experimental design, failures, and transfer into a product.
- Verify whether PyTorch/ML implementation and general DSA are evaluated separately.
- Ask the HM how research results are handed off to product metrics and what the scientist owns.
- When using a particular product family as why-us evidence, date-check that its current official announcement actually connects to the target JD.

### ByteDance / Seed-related roles

Evaluation may vary substantially by region and team, so do not turn a company-wide reputation into a fact about this req.

- Verify whether there is an online assessment, the count and difficulty category of live-coding sessions, and the language and execution environment.
- Ask whether a research deep dive and coding are combined in the same session.
- Ask about working hours, collaboration time zones, and performance expectations as **concrete operating practices of this team**, not as culture rumors.
- If the offer includes private-company equity, verify liquidity, valuation basis, vesting, and post-termination terms in the documents. Follow [Offers, Levels & Negotiation](#/process/negotiation) for comparison.

### Mistral AI-related roles

Distinguish research, inference/infrastructure, and applied or customer-facing roles.

- If there is a take-home, first verify the expected time, submission format, permitted external resources and AI tools, and whether a follow-up presentation is required.
- Ask whether transformer, MoE, or inference implementation is required, and at what language and abstraction level.
- Verify the split between customer projects and internal model work, as well as travel, time-zone, and language requirements, for the target role.
- Do not compare private compensation such as options with public stock using a headline amount; review the contract documents and consult appropriate local professionals.

### Microsoft / MSR-related roles

Research MSR separately from product-AI organizations. Even the same title may place different weight on publication, engineering, and customer impact depending on the team.

- Verify whether there is a job talk or seminar, including topic selection, audience, and Q&A format.
- Ask about the composition of coding, ML-fundamentals, and system-design sessions and the purpose of each round.
- If you heard about an external senior interviewer or a separate calibration stage, do not assert its name; ask only whether it is actually part of this schedule.
- Ask about research autonomy, mentoring, and cross-organization collaboration through concrete responsibilities in the first six months rather than abstract culture statements.

## A safe way to build why-us

Build each company-specific sentence as `public evidence → my judgment → my contribution hypothesis`.

> “In the {JD/paper/product} published on {date}, I was impressed by {specific choice}. In {relevant experience}, I worked on {measurable problem}, and on this team I would like to test {contribution hypothesis phrased as a question}.”

A strong why-us answer is not a sentence that lists many recent names. One real source and a **verifiable point of connection** to your own experience are enough. Do not duplicate project-by-project mappings in the generic chapters; maintain them in [Your CV → Interview Map](#/resume/overview) and [Predicted Questions & Answers](#/resume/predicted-questions).

## A company comparison table is a work log, not a result

| Company / team / req | Checked on | Core role output | Confirmed evaluation format | Unverified risk | Two pieces of my evidence | Next question |
| --- | --- | --- | --- | --- | --- | --- |
| `{company / team / req ID}` | `YYYY-MM-DD` | Paper / product / infrastructure / mixed | Recruiter-confirmed information | Tools, team match, etc. | CV evidence | HM/recruiter question |

Instead of copying compensation numbers into this table, make a separate sheet that distinguishes currency, region, level, and equity type. Treat public aggregates only as **reference points with a research date**; base the actual comparison on the written offer and plan documents.

## Order the schedule by confirmed format risk, not “company difficulty”

Do not permanently classify brands as “safe companies” or “reach companies.” Risk differs by candidate.

1. Decompose confirmed rounds into `DSA`, `ML coding`, `research talk`, `system design`, `take-home`, and `behavioral`.
2. Compare them with recent mock results and find the largest format gap.
3. Schedule a **real loop with a similar format** before your highest-priority options, while coordinating with recruiters so offer deadlines do not become too far apart.
4. When the schedule or stages change, update both the snapshot's checked date and your preparation priorities.

## Company snapshot template

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

## Cheat Sheet

| Principle | One-line action |
| --- | --- |
| Company name ≠ interview format | Verify at the team, req, region, and date level |
| Undated information | Do not use it; mark it `unverified` |
| Rumor | Use it only to generate questions, never state it as policy |
| Final operational authority | Recruiter's written response and interview invitation |
| Why-us | One piece of public evidence + my judgment + contribution hypothesis |
| Personal fit | Maintain it once in the resume packet and link to it here |
| Compensation comparison | Review level, region, currency, liquidity, and terms separately |
| Scheduling order | Use confirmed format risk, not brand prestige |

**Related:** [The RS/AS Pipeline](#/process/pipeline) · [Phone Screens](#/process/phone-screens) · [Recruiter & HM Screens](#/process/recruiter-hm) · [Offers & Negotiation](#/process/negotiation) · [Questions to Ask Them](#/playbook/questions-to-ask) · [Your CV → Interview Map](#/resume/overview)
