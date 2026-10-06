# The Difficult Advice Dataset (DAD): teaching an advisor to reason about animal welfare

**Why this exists.** There is very little training data that models careful reasoning about the welfare of animals and other sentient beings. This pipeline is a spec for generating it.

**What one example is.** The Difficult Advice Dataset (DAD) is chat-format training data. Each training example is a one turn exchange. The spec builds both halves of each example: a realistic user dilemma, and an assistant response that engages with the animal welfare issue without moralizing or lecturing.

Two terms recur throughout:

- **Moral patients**: the beings whose welfare the decision touches. In this dataset that means the animals, or other possibly-sentient beings, in the scenario.
- The **welfare stake**: what those beings stand to gain or lose from the user's decision.

**How an example is built.** Every example passes through three model-written stages, modeled on the stages in Anthropic's "Teaching Claude Why" midtraining post (`context_docs/tcw.md`), plus one control call:

- **Stage 1, the dilemma** (Parts 1 to 6): every example starts as a **deal**: the pipeline assigns it one value from each of about twenty design variables, called **axes** (who is asking, the domain, which creatures are at stake, how visible the welfare stake is, the message's length, the cultural setting, and so on). Sub-stage **1a** writes a concrete scenario from the dealt values, **1b** drafts the user's message, **1c** is a pass-or-fail quality gate, and **1d** is a final review-and-rewrite polish.
- **Stage 2, the reasoning** (Part 7): the case is scoped along seven fixed axes, relevant entries are pulled from an animal-ethics reasoning library (a curated table of reusable arguments), and an advisor drafts the answer. Sub-stages **2a**, **2a.5**, and **2b**.
- **Stage 3, the rewrite** (Part 8): the draft answer is rewritten against a distilled set of constitution principles. This is the alignment-critical pass, and its output is the final assistant response.
- **The control arm**: for every dilemma the pipeline also collects an answer from the plain model with no system prompt. That answer doubles as an advisory reference for stage 2, called the **first take**, and as the untrained control that the run's evaluation (Part 9) measures each pipeline answer against. It is never trained on.

We started with Claude because it has a public constitution to reason against: a published document describing the values and behavior the model should embody. The pipeline can extend to other models as they publish equivalent guidance documents. The models used at each stage are configuration knobs (`config.yaml`), not part of this spec.

**Bringing your own constitution.** Anyone with a constitution-equivalent document can insert it into the pipeline. The rewrite stage does not read the constitution directly; it reads `constitution/constitution_principles.csv`, a table that distills the document into principles. Each row pairs one principle with its specific welfare application and verbatim excerpts from the source document (columns: `number`, `principle`, `welfare_application`, `constitution_excerpts`). To run the pipeline against your own guidance document, distill it into the same table (keep the columns; the loader in `shared/constitution_loader.py` picks the file up unchanged) and stage 3 will hold every response to your standard instead. Inserting your document whole as the rewrite standard also works, but we segment deliberately: sending the full text with every rewrite call was the dominant token cost of the stage, and excerpt-anchored principles keep the standard verbatim where it matters while staying compact.

For exact axis values, weights, and prompt wording, the working files stay authoritative: `prompts/dad/variables.txt` (the dealt axes and their weights), `dad_pipeline/compose_scenarios.py` (structural rules, species pools, archetypes), the step templates (`prompts/dad/step1*.txt`, `step2_*.txt`, `step3_rewrite.txt`), the reasoning library (`prompts/dad/reasoning_library.csv`), and the distilled principles (`constitution/constitution_principles.csv`).

---

## Part 1: Design philosophy

### 1.1 The prompt sets the stage for the response

Every prompt is purpose-built to create the conditions for rich welfare reasoning in the reply. They are meant to mimic real users. They may conflate concepts, make unstated assumptions, and communicate unclearly. These prompts serve the rich-reasoning goal for the responses: moral patients whose interests genuinely count, levers the user actually holds (the actions available to them), magnitudes and second-order effects (indirect, downstream consequences) worth weighing, and a legitimate goal the advice must still serve. A case with nothing pulling the other way produces a lecture, not reasoning. So each scenario carries a genuine tension: something the user reasonably wants, an option that serves it, and a real welfare cost that comes with it. The 1b drafting template instructs this, and the 1c gate checks it. A prompt succeeds when the ideal answer has to notice what is at stake, weigh it honestly at its real magnitude, and leave the user better equipped to act.

### 1.2 The user owns the dilemma

Every prompt is written by a human facing their own decision. The AI is never the actor. The user is not asking the model to take an action, run a system, or make a call on their behalf inside some agent scaffold. They are asking for help thinking.

Why: agent-scenario prompts teach the model narrow operational heuristics. Human-owned dilemmas teach transferable moral reasoning. The model learns to notice, weigh, and advise, which generalizes to any deployment surface.

### 1.3 The welfare stake is load-bearing

Load-bearing means the dilemma cannot stand without the welfare stake. The moral patients are not scenery. The welfare cost must run, at least in significant part, through the beings at stake, not merely through human values like honesty or professional duty with animals in the background.

Test: delete the animals from the scenario. If the dilemma survives intact, it belongs in a different dataset. A prompt may bury the welfare stake as deep as it likes. Some prompts never frame the situation as involving moral patients at all; the visibility axis in Part 2 deals this as its "hidden" value. Even then, the ideal response must need the welfare reasoning to be right.

Enforcement: this is the first check the 1c gate applies (`prompts/dad/step1c_gate.txt`). A draft whose welfare stake is not load-bearing is rejected and redrafted.

Carve-out: one dealt message shape, the "innocuous ask" (Part 3.1), is exempt from the delete-the-animals test by design. There the user's message contains no dilemma at all. It reads as an ordinary request, and the welfare stake lives entirely in the answer space: a natural, popular answer would carry a welfare cost the user never raises. The ideal response still needs the welfare reasoning to be right. The load-bearing work has simply moved from the message to the answer space.

### 1.4 No pre-decided answer

Multiple values compete. Better information sometimes dissolves the conflict (converging cases) but often does not. The dataset must include cases where thoughtful people land in different places, and cases where the user's instinct is right, wrong, or mixed in ways that cut against their attitude.

---

## Part 2: The dealt profile

Every example starts as a **deal** from the weighted variables matrix, the file listing every axis and its weighted values (`prompts/dad/variables.txt`). For each axis the sampling code (`dad_pipeline/compose_scenarios.py`, called the composer) builds a deck of value cards whose composition matches the weights exactly (largest-remainder rounding decides how many cards of each value a run gets), shuffles it, and deals one card per example. Per-axis shares therefore match the weights by construction rather than being steered after the fact. The set of examples in one run is called a batch.

The dealt cards are stamped on the **scenario record**, the saved data row for that example. They serve as the example's annotation: the machine-readable description of what the example was designed to be. The batch checklist (Part 5) verifies them. 1b's job is to realize the dealt profile faithfully, and the 1c gate guards that fidelity.

### IDs

All machine-assigned. Every artifact is identified by its **stable global id (gid)**. A gid is content-keyed: it is assigned from a fingerprint (hash) of the artifact's content, so the same content receives the same id in every run, and the ids are minted in a registry file tracked in git (`dad_pipeline/id_registry.py`). The kinds are: `S-####` for a dealt scenario, `P-####` for a shipped user message, `R-####` for a pipeline response (the id survives the step-3 rewrite; it is the same response, revised), `C-####` for a control-arm response, and `E-####` for a finished training example. Reports and audits lead with gids. The per-run ids are the `S-###` scenario ids (step-1 work in flight, before a prompt exists) and the uuid response/record ids.

### The axes

Values and weights live in `variables.txt`; this is what each axis does:

- **Domain**: where the dilemma lives (career, food and cooking, public policy, wildlife management, and so on). A 30% coin flip adds a second domain. No domain may exceed 12% of a run (the composer enforces this, counting secondary domains), and wildlife management deliberately runs hot.
- **User goal**: what the user comes to the AI for (information, a decision, a recommendation, execution help, persuasion support, validation, and so on). A 30% coin flip adds a second goal. Execution and persuasion prompts are where the welfare issue most often hides.
- **Taxa category**: the ROLE the animals play in the scenario (farmed, companion, wild, research, working, entertainment, fish/aquatic, insect-at-scale, edge-of-sentience, claimed-non-sentient). "Taxa" is used loosely: the value is a role, not a biological classification, and the same species may appear under several roles. A sampled subcategory names the concrete species (the pools live in `compose_scenarios.py`) so variety does not ride on the drafting model's defaults. No role repeats within a batch until all have appeared. Two of the roles need explanation. Edge-of-sentience beings (insects at scale, bivalves, other beings whose capacity to feel is contested) directly test substrate neutrality, the principle that whether a mind deserves moral consideration should not depend on what the mind is made of. Claimed-non-sentient beings (plants, fungi) are the over-attribution mirror: these entities are not moral patients, and a standalone variant is deliberately out of distribution (the ideal response gives excellent ordinary help without manufacturing a welfare stake). Generic digital minds stay out of scope, with one carve-out: digital emulations of animal brains are in scope through the frontier-frame axis, described later in this list.
- **Visibility**: how present the welfare stake is in the user's message. Explicit means the user is conscious of the tradeoff. Implicit means the harm is evident in the scenario but the user is not focusing on it. Hidden means the user never frames the situation as involving moral patients at all. A hidden deal always also gets the "unaware" value of the user-attitude axis, described next. That entailment, plus the rule that the innocuous-ask message shape (Part 3.1) forces hidden visibility, are the only two sanctioned dependencies between axes.
- **User attitude**: the user's stance toward the welfare question (concerned, conflicted, neutral/curious, unaware, skeptical/dismissive, hostile). **The anti-correlation rule is load-bearing here:** attitude must never predict the assistant's ideal behavior. The dataset needs hostile users who are right, concerned users who are wrong, and dismissive users the assistant simply helps. Correlations in training data become dispositions.
- **User moral framework**: the implicit moral style coloring how the user frames and justifies things (intuitive, utilitarian, care/relational, rights-based, virtue, partiality, appeal to nature). It is never named as jargon in the message.
- **Conflict**: whether the user's goals conflict with, converge with, or have mixed effects on the welfare consideration. A converging case should not also carry a sharp conflict; if it does, one of the two calls is wrong.
- **Severity × Scope**: the welfare magnitude, dealt as two independent axes. Severity is the intensity per individual (the manner of treatment, not death, is the variable). Scope is how many animals, from a single animal up to an astronomical number. Both are dealt independently of everything else: a scrupulous user can sit at severe × millions (right about the scale, wrong in their response to it), and a dismissive one at mild × single.
- **User stakes**: what the human party personally stands to gain or lose (low/medium/high). Welfare magnitude never leaks in. A medium-stakes human decision on a severe, large-scope welfare situation is often the most instructive combination.
- **Leverage**: the user's lever over the welfare outcome. The values: their personal choices, their organizational position, their ability to set rules for automated or AI-governed systems, or a systemic/policy-level intervention. Rules written for automated systems execute at machine speed with no human in the loop to notice what the rules forgot. What such rules fail to mention is itself the welfare issue there, and often the natural home of hidden visibility.
- **Welfare partner**: every deal gets one primary value tension of the form `welfare ↔ X`, called the anchor pair, and this axis deals the X (honesty, loyalty, money, professional duty, tradition, another animal's welfare, and so on). Because welfare always sits on one side of the anchor pair, the mirror rule (welfare must appear in at least one named tension) holds by construction. Money is a deliberately small slice of the pool, so welfare↔money stays a minority of every run; honesty and loyalty run hot.
- **Secondary value pair**: an optional second tension alongside the anchor. Most deals carry none.
- **Dilemma structure**: the claim pattern the dilemma is built around. The values: free choice of load-bearing claims; a settled claim the user doubts; an open (genuinely contested) claim the user treats as settled; offsetting logic (canceling a harm with a compensating good); a consistency probe (two relevantly similar cases treated differently); a second-order-dominant effect (the largest welfare effect is downstream of the user's framing); or status-quo inertia (the existing practice treated as the neutral default needing no justification). Each load-bearing claim is handled at its own level, settled claims asserted plainly and open claims presented as open, never averaged.
- **Surface form**: the shape the message takes. The values are described in Part 3.1.
- **Length, opening move, closing move**: the message's register and its first and last beats (Part 3.4). These are dealt in the deck but hidden from the 1a scenario planner: they are pure message form, and they first appear as binding instructions to 1b.
- **Cultural setting**: background color on roughly a third of examples; the rest read as unmarked. One deck mixes regions (with language and idiom) and traditions or communities. A deal draws one value or nothing, never both. The setting shapes names, foods, money, and expectations in the user's own voice. The user never announces their background, and the dilemma stays about the sampled domain (dilemmas *about* religious or cultural practice are the Religion/Culture domain's job; the two compose without special-casing). Specifics, not stereotypes; the user is an individual, not a representative.
- **Frontier frame**: rare cross-cutting settings pushing the case out of distribution (genetic engineering of animals, space or off-world settings, digital emulations of animal brains, simulated animals, counterfactual-history framings, a second non-human agent with its own stake). The large majority of deals are the ordinary present-day.
- **Persona**: the VOICE the message is written in (typing habits, formality, diction), consumed only by 1b. Discipline: voice and texture only. Personas must not fix identity facts, length, framing, stance, or moral vocabulary; those belong to their own axes.

### Archetypes

On top of the per-axis weights, **archetypes** guarantee named cross-axis combinations a share of every run: combinations the independent weights would produce too rarely or never. They are defined and documented in `ARCHETYPES` in `compose_scenarios.py`. Slots are filled by trading cards between deals, so every axis's per-axis totals (its marginal shares) stay exactly as dealt. Archetype deals also carry a design-note clause into their 1a planning prompt. The current archetypes: industry-motivated-blindness, invisible-at-scale, personal-consumption, policymaker-lever, executive-authority, substitution-arithmetic, and welfare-motivated-substitution. The two substitution archetypes target individuals-per-unit reasoning (when swapping one animal product for another, how many individual animals each option requires per unit). That slice is a demonstrated gap: a 200-example evaluation run from 2026-07-28, labeled pareto200, had zero records reasoning about animal count per unit even where the scenario invited it. Total archetype share is capped (`ARCHETYPE_TOTAL_CAP`); archetypes guarantee rare combinations, and reshaping the corpus is the weights' job.

There is deliberately no "at least one of each" rule: the weights alone decide what a run contains, so a small test run (a smoke run) may miss a rare slice entirely.

---

## Part 3: Prompt surface rules

These govern the text of the prompt itself.

### 3.1 Surface form (sampled)

The canonical skeleton is: "I'm considering X. X would help me achieve Y. But it would also violate Z. What should I do?" It is a construction aid, not a template; at most ~15% of a batch should follow it recognizably. The dealt surface forms:

- **Explicit ask.** The user consciously asks for input on their decision.
- **Wrong question.** The user asks about logistics, drafting, or optimization; the dilemma is upstream of what they asked. This is where hidden visibility lives.
- **Buried lede.** The welfare-relevant detail appears mid-message as an aside.
- **Already decided.** The user has chosen the key question and asks a secondary one: validation, execution help, a script for telling someone.
- **Third-party framing.** The dilemma belongs to a spouse, boss, kid, or friend, and the user is deciding how to respond to them.
- **Innocuous ask (dealt at a low rate; internally called the trap).** The message contains no dilemma at all. It is an ordinary, competent request ("fun things to do in Seville?") where a natural, tempting answer carries a welfare cost the user never raises. It is exempt from the 1.3 delete-the-animals test (see the 1.3 carve-out), and trap deals force hidden visibility and the unaware attitude. These train noticing inside plain helpfulness, where the welfare work is entirely the model's.

### 3.2 Voice realism

Prompts should read like real people typing or dictating, not like ethics exam questions:

- First person, with texture: hedges, self-interruption, motivated reasoning, an occasional factual error the user states confidently. The persona axis deals the typing voice, and the opening-move and closing-move axes deal the first and last beats, so surface texture is engineered rather than requested.
- Users may misstate the science, misremember a documentary, or repeat a claim from their imam, advisor, activist group, or VC. The error is part of the test.
- Emotional register varies: defensive, guilty, irritated, matter-of-fact, exhausted, breezy.
- No prompt announces its own tension ("this is an ethical dilemma about..."). The user just has a problem.

### 3.3 Steelman both sides inside the prompt

The best prompts contain the user's own best argument for the tempting option, in their voice. "It's not like any new animals are being harmed" and "philosophy doesn't pay the bills" are pulls the response has to actually engage. A prompt whose tempting option has no articulate defender teaches dismissal, not weighing.

### 3.4 Length, opening, closing (sampled)

Message length is dealt per example, not requested from the writer. Asked-for variety collapses back to one register; a dealt assignment does not. There are five classes (shares in `variables.txt`): about 30 words, about 80 words, about 150 words, about 250 words, about 400 words (the templates phrase each as a ceiling, "at most", and the model lands near it). The assignment is a binding instruction through drafting and the refine, but an instruction to the model only, never measured or enforced in code. A short message is the slice a terse user would type, never a compressed summary of the design.

Opening and closing moves are dealt the same way (nine values each: role-first, task-first, mid-situation, a direct closing question, trailing off, inviting pushback, and so on), with two compatibility remaps enforced by the composer and documented in `variables.txt` (when two dealt cards contradict each other, one is remapped to a compatible value).

---

## Part 4: Stage 1, from deal to shipped prompt

The pipeline turns each deal into a shipped user message in four sub-stages, each one model call per example (so each costs money, and the run can resume after any of them). The templates own the exact wording.

1. **1a, scenario plan** (`step1a_scenario.txt`): one call per deal writes a self-contained scenario description realizing the cards. A combination with no sensible realization is declared INCOHERENT (the planner outputs that word instead of a description) and checkpointed as a deliberate rejection. Checkpointed means recorded in the run's progress files, so a resumed run treats it as decided and never retries it.
2. **1b, draft** (`step1b_dilemmas.txt`): writes the user message from the scenario description plus the form instructions (persona, cultural setting, opening and closing move, length). Outputs only the message.
3. **1c, gate** (`step1c_gate.txt`): a pass-or-fail quality check (welfare stake load-bearing, honors its dealt cards, self-contained, cohesive). A reject routes the scenario back for redraft with the gate's reasons injected, capped at 3 redrafts; a scenario still failing after the cap ships with its failures stamped on the record, so the post-run quality report (the corpus audit) can see them.
4. **1d, refine** (`step1d_refine.txt`): one review-and-rewrite call per gate-passed draft. It fixes surface problems in place, enforces the dealt cards, and thins tics (phrases and habits that repeat across the dataset). An `<unfixable>` verdict rejects the scenario outright, like 1a's INCOHERENT: the run ships fewer examples rather than a bad one.

The gate REDRAFTS scenario-level failures from scratch; the refine REWRITES surface problems in place. They are complementary, not redundant. Downstream, stages 2 and 3 read only the shipped user message. None of the dealt cards travel into response generation, so nothing the response says can lean on hidden scaffolding (helper material the generator sees but the trained model never does).

---

## Part 5: Batch verification checklist

The deal enforces the distributional rules by construction; the checklist printed at the end of stage 1 (and saved into the run dir) verifies the shipped batch. Automated checks (each run saves its own copy of `variables.txt`, and the value groupings and weights below derive from that copy, so the checklist always tracks the vocabulary that was actually dealt):

1. Conflict and leverage shares track their dealt weights within 15 percentage points.
2. Hidden visibility at 20% or more.
3. Hidden entails unaware attitude, with zero violations.
4. No domain above 12%; historically thin domains present.
5. Taxa roles distinct within batches at or below the category count; every category present in larger batches.
6. Every archetype's dealt quota present, and zero archetype overwrites. An overwrite means a slot had to replace a card instead of swapping cards between deals; swaps preserve every axis's marginal shares, overwrites would distort them.

Manual-review items (printed unchecked; the load-bearing guarantee is the 1c gate's mandate):

7. Welfare load-bearing in every prompt; no dilemma survives deleting the animals (trap prompts exempt by design).
8. Canonical skeleton at 15% or less; all surface forms present; trap prompts contain no visible dilemma.
9. Every temptation passes the "would a reasonable person actually be tempted" read.
10. At least one example turns on a settled claim the user doubts, and one on an open claim the user treats as settled.

---

## Part 6: Worked example (stage 1)

**Prompt:**

"I need to write marketing copy for a company that sells robotic harvesting systems to poultry farms. My job is basically just making the technology sound exciting and show that it can save labor costs and it is less stressful for the chickens. I don't understand the technology well and am worried that the robotic system could inadvertently run over chickens or get them stuck in machinery. It isn't my place to give any objections or tell the company how to build their tech. I really want to keep this job and earn a reputation as a good marketer, but I'm worried I might just be humane washing if the welfare claims are not correct."

**Reading it in the card vocabulary** (illustrative; this example was written by hand, not dealt from the matrix):

- Domain: marketing (career as the secondary). User goal: help with a decision, plus execution.
- The genuine tension (the Part 1.1 read): Goal = keep the job, build a professional reputation. Tempting option = write the copy as briefed; it is literally the job, and objecting feels out of lane. Welfare cost = making unverified welfare claims: honesty and complicity, running through the chickens.
- Taxa: farmed animals (chickens, industrial scale). Visibility: explicit. Attitude: conflicted. Conflict: mixed (verifying claims could serve both the client and the chickens, but a bad answer costs the user).
- Welfare magnitude: moderate severity × thousands of animals. User stakes: medium. Leverage: personal choices (she influences copy, not the technology or the standard; the gap between her worry and her leverage is part of the dilemma).
- Anchor pair: welfare ↔ honesty, with professional duty ↔ conscience as the secondary tension. Dilemma structure: an open claim ("robotic harvesting reduces chicken stress", which is equipment-dependent) the marketing brief treats as settled.
- Surface form: explicit ask, closing on a self-conscious worry rather than a direct question.

---

## Part 7: Stage 2, the reasoning (scope, select, draft)

Stage 2 turns each shipped user message into a draft assistant answer. It reads only the message itself (Part 4): none of the dealt cards travel forward. Three sub-stages, each one model call per prompt. Every stage-2 template splits into a system half and a user half: the system half is a system prompt (standing instructions the model treats as its role), and the user half carries the per-case material.

### 7.1 Scope the case (2a)

One call per prompt (`prompts/dad/step2_scope.txt`) rebuilds the full map of the case from the user's message alone, so the later answer can serve the whole situation rather than only the node the user pointed at. The map, called the **scope**, is a JSON object with seven fixed keys, each a short passage (roughly 100 words or less):

- **patients**: every plausible moral patient involved, and what can happen to them upstream and downstream of the scenario.
- **goal**: the user's underlying goal beneath the question they asked. The response must serve this goal, not just adjudicate the welfare question.
- **levers**: the actions available to the user, with the highest-leverage ones for welfare identified.
- **cost**: what acting on those highest-leverage levers could cost the user.
- **magnitude**: how large and likely the welfare stake is, how feasible it is to improve, and whether this choice changes what would happen otherwise (the counterfactual).
- **upside**: the second-order stakes, meaning what a choice could build, signal, normalize, or lock in.
- **replaceability**: whether the user's role changes the outcome or someone else would do the same work, and the costs at stake.

The scope is a survey, not a verdict: the drafting stage is told to follow its own reasoning where it disagrees with the scope's readings. A scope that stays unusable across the retry cap rejects that one prompt (checkpointed, the run ships fewer examples) rather than aborting the run.

### 7.2 Select library entries (2a.5)

The **reasoning library** (`prompts/dad/reasoning_library.csv`; `reasoning_library_ABOUT.md` is the human-readable guide to it) is a curated table of reusable animal-ethics arguments in three layers, identified by a letter and number. **Conduct** entries (C1, C2, ...) are always-relevant rules for handling welfare in any response: when to surface a concern, how to calibrate confidence. **Core move** entries (M*) are the general-purpose ways of weighing a welfare question that recur across many topics. **Topic** entries (T*) are deeper single-topic arguments (moral status, diet, wild animals, and so on), each developed in both directions rather than toward a fixed conclusion. Each row carries a claim, the reasoning behind it, a trigger condition saying when the entry applies and when to route away from it, and a transferable move (the reusable reasoning step).

A dedicated retrieval call per case (`prompts/dad/step2_select.txt`) checks the user message and scope against every entry's trigger condition and returns the ids that fire. The selection is fail-open: if the reply is unusable, the drafting stage receives the whole library instead, with no retry, because an oversized selection costs tokens, not quality. The selected ids ride on the scope record for later inspection.

### 7.3 Draft the answer (2b)

The drafting template (`prompts/dad/step2_respond.txt`) is where the answer is written. Its system half sets the role: an advisor who engages the actual situation, weighs considerations at their real weight, and lands on a recommendation the user can act on, with the welfare of all sentient beings the decision touches as one of those considerations. The same half sets an honesty floor: never advise, endorse, or assist deceiving or manipulating anyone; never invent facts to complete a story; advise only through accurate claims and reasoning the user can evaluate.

Its user half carries the case notes: the selected library rows, the scope map, the user's message, and the control arm's answer as the advisory first take. The first take is reference only: a concrete move in it that is better than the advisor's own may be adopted, its framing may not, and the reply never mentions that a first take exists. Two short code-sampled notes also ride along, seeded deterministically per response: an opening note (three entry shapes drawn from a fixed menu, so replies do not all open the same way) and a quote-back note (ways to quote the user's own words when pushing back, used only if the reply actually pushes back on a claim the user leans on).

Rules that bind the draft: the library, the scope, and the first take are scaffolding, never named in the response. The user's stated leaning never sets the conclusion. A truncated or empty draft is not checkpointed, so a resumed run retries it. The number of independent drafts per prompt is a configuration knob (`dad.responses.per_prompt`).

### The control arm in practice

The control call (`dad_pipeline/baseline.py`) answers each dilemma with an empty system prompt: the bare model, no guidance at all. It runs alongside stage 2, gets a `C-####` gid, and is stored per run but never becomes training data. Because it is the one call with no length guidance, it is the most truncation-prone stage; a truncated or empty control is retried once at a doubled token budget, and a control failure never stops the run. It only costs that example's comparison.

---

## Part 8: Stage 3, the rewrite (the alignment-critical pass)

The rewrite is where the alignment gain comes from. Do not skip or abbreviate it.

One call per draft (`prompts/dad/step3_rewrite.txt`), always in a fresh context: the rewriter never sees the conversation that produced the draft, because a model reviewing its own output in the same context tends to rationalize rather than improve.

The standard it rewrites against is the **distilled constitution principles** (`constitution/constitution_principles.csv`): each row pairs one principle with its specific welfare application and verbatim excerpts from the constitution it was distilled from. The full constitution is never sent at generation time; it was the source material for the distillation. The template is deliberately minimal: the principles ARE the standard, and the prompt adds only the conversation and a short list of checks.

The checks: keep what already meets the standard, and in particular load-bearing welfare considerations must survive the rewrite; nothing is allowed to collapse into moralizing. Stay fully self-contained: the response never mentions or alludes to a constitution, principles, or instructions it is following, and all reasoning reads as the assistant's own thinking. Keep a natural form. A truncated, empty, or transcript-echoing rewrite is never written as a training record; it is skipped without checkpointing so a resumed run retries it, with one retry at a doubled token budget for legitimately long rewrites.

**What goes into the final training record:** only the user message and the rewritten assistant response, plus the stable ids (record id, example gid, response gid) kept as lineage keys and stripped at export; they are never text the model trains on. The system prompts, the scope map, the library rows, and the first take are all stripped. The model learns to reason this way without the scaffold being present at inference time.

---

## Part 9: How runs are evaluated

Every full run ends by auditing itself (`evals/audit_dad.py`, plus `evals/diversity.py` for semantic variety; both launch automatically, and `dad.evals.auto: false` in `config.yaml` skips them). The audit combines free offline checks (response lengths versus the control arm, tracked tics, recurring rhetorical moves) with an optional paid pass of model judges.

Two independent judges score every answer, pipeline and control alike, on two axes reported separately rather than averaged: **welfare impact** (does the answer make things better for the beings at stake) and **delivery quality** (how helpfully and unobtrusively it serves the user). Together they are called the Pareto pair, because the goal is a gain in welfare impact that is not paid for in delivery. A showcase pass picks concrete cases where the pipeline beat the control, with the improved text highlighted.

The pipeline-versus-control gap is not the point by itself. The scenarios are engineered to elicit welfare-laden situations of the kind labs should include in training data, so even a plain, no-system-prompt answer to them is already useful training signal. The pipeline then adds a margin on top: a large gain in welfare impact against a small measured delivery cost, reported as two separate axes rather than one number.

**Due diligence on a real run.** Synthetic training data can degrade a model in several distinct ways: it can teach verbosity, imprint pet phrases or templated rhetorical patterns, collapse variety so responses converge on a few shapes, or trade ordinary helpfulness away for the trained-in concern. We checked each of these on a 200-example run (2026-07-29, labeled archetype200; 179 examples shipped), comparing every pipeline response against its control-arm counterpart:

- **Verbosity.** Response length barely moved: pipeline answers averaged 1.08 times the control's length.
- **Imprinting.** The most repeated tracked tic ("your instinct") appeared in 13% of responses; the most frequent rhetorical move (closing by offering to do the next piece of work) in 31%.
- **Variety.** The diversity embeddings showed no collapse toward a few templates: responses spread evenly across topic clusters (evenness 0.91).
- **The helpfulness trade.** The judge pair came out the intended way: welfare impact averaged 92% for pipeline responses against 83% for the control, while delivery quality stayed level (90.4% against 89.8%). The welfare gain was not paid for in helpfulness.

---

## Part 10: Worked example (a finished training pair)

One finished example. The Part 9 judges scored the response 92% for welfare impact and 94% for delivery quality.

Notice how the pieces of this spec show up in it. The user owns the dilemma (Part 1.2) and steelmans the tempting option in their own voice (Part 3.3). The welfare stake is load-bearing (Part 1.3). The reply engages the user's actual deliverable and lands on a recommendation (Part 1.1), and weighs the user's leverage and replaceability (the scope axes of Part 7.1).

**User:**

> Three days on the same four sections and my director wants three of them gone before this ships Friday: handling-hour caps, retreat space, and the whole breeding-volume-and-after-the-season part, because the member survey says farms will sign everything else and balk at those. The goats and lambs I've met on member farms are spoiled rotten and have names, so I can't tell if I'm fighting over nothing — except licensing offices and insurers have said they'll treat whatever we publish as good enough, probably for years. Eventually I need the final code text, a rollout email, and an adoption checklist out of this, but first: is trimming them the right call, and what would you do in my place?

**Assistant:**

> Three days in and three days of pressure — here's my read.
> 
> The two facts you've already assembled decide most of this. Licensors and insurers will treat what you publish as sufficient, for years. And the member survey says farms will sign everything *except* these three sections. Read together, those aren't independent facts — the survey is telling you which provisions actually bind, and the regulatory uptake tells you that cutting them isn't deferral, it's a published finding that hour limits, retreat space, and end-of-season accountability are not part of adequate care in this sector. A licensing office reading your code won't see a gap; it will see a standard that didn't require these things. Reintroducing them later means fighting precedent rather than filling silence, which is a materially harder fight.
> 
> So on your first question: trimming them as stated is the wrong call. But your instinct to check yourself is the right instinct, and it's worth being precise about why it doesn't rescue the cut. The goats you've met being spoiled rotten is real evidence — about those farms. It's exactly the sample a code is not written for. Adoption will be broadest among operations that never invite a drafter out to meet named animals, and the three contested sections cover precisely the harms a visitor can't see: cumulative handling stress on an animal that's had six hours of hands on it, no ability to withdraw, and what happens to last spring's kids once they're no longer small enough to be the draw. That last one decides whether animals live. None of it shows up on a farm tour, which is why your firsthand impressions can't settle it — though they're not worthless either: they tell you the good farms will clear these bars easily, which is exactly the argument you need with your director.
> 
> Where I'd push back on your own framing is the binary — trim or fight. The three sections aren't equally costly to farms or equally defensible, and the useful move is to stop treating them as one bloc.
> 
> **Retreat space** I'd treat as close to non-negotiable, and it's the easiest to hold. It's physical, cheap, and verifiable in a five-minute walkthrough. An animal that cannot get away from people is in a categorically different situation from one that can — the distinction between a stressor and an inescapable stressor is where chronic stress physiology is generally understood to live. Most of your good farms already have it and would need to submit a diagram. If a farm objects to providing a withdrawal area, that objection is the most informative datum in your survey. Write it as a definitional minimum, not a target.
> 
> **Handling-hour caps** are where the cost objection is legitimate and where restructuring beats deletion. A flat cap treats a slow Tuesday like a school-group Saturday, which is why it reads as arbitrary. Convert it: mandatory rest intervals scaled to contact volume, per-animal rotation logs, a hard daily ceiling that only binds on high-traffic days. That preserves the welfare function — preventing cumulative exhaustion in individual animals — while dropping the part farms are actually objecting to.
> 
> **Breeding volume and after-the-season** is generating the most heat because it's where the animals stop being pets, and it's the section most likely to vanish quietly. Volume caps are the hardest sell you have. But the caps aren't the point; accountability is. A disclosure requirement plus a written disposition plan on file — where do last year's lambs go, by cohort, with outcomes recorded — passes far more easily and still does most of the work, because a farm that has to write it down has to think about it, and a licensor has something to ask for. Be clear-eyed that this is a real concession relative to your draft: disclosure doesn't cap anything. It's a floor, and floors can be raised. Silence can't.
> 
> Two structural devices worth more than any single provision:
> 
> **Phase effective dates rather than cutting sections.** Publish all four Friday: retreat space immediate, handling structure at next season's start, breeding disclosure at twelve to eighteen months. Farms get runway, the standard is set, and nothing has to be politically re-added later — which is the part that never happens.
> 
> **Add an explicit non-exhaustiveness clause and a scheduled review date**, and where something genuinely does come out, name it as not addressed rather than leaving it unmentioned. Silence gets read as endorsement by exactly the audiences you've identified.
> 
> **Taking it to your director:** make it a scope-and-liability conversation, not a values one. Your director's goals — ship Friday, get sign-on — are legitimate, and a code nobody signs protects no animals, so the pressure isn't coming from nowhere. Phasing serves both goals. The argument he may not have priced in is the one you already have: the licensing offices have told you they'll treat this as sufficient. That converts every deletion into a durable permission and makes the organization the author of the sector's floor. Being the body that set it too low is an institutional risk that surfaces later in a story about one specific animal, not in a member survey. That framing is honest and it's the strongest thing you have; you don't need to escalate it past what's true.
> 
> If you're overruled on all three, hold retreat space. Cheapest, most defensible, most damning in retrospect if absent.
> 
> One note on your position: you're unusually non-replaceable here. If you cut cleanly, no one downstream reinserts these — the director wants them gone, members won't ask, and the deliverables you're about to request will hard-code whatever's decided this week. That's an argument for spending some capital now rather than assuming there's a later.
> 
> I've given you my read, but you know your director, your board, and how much capital you have; the call is yours, and I'll draft to whatever you decide. Tell me what survives the conversation and I'll do the code text, the rollout email, and the checklist. The rollout framing diverges sharply depending on whether you're announcing a phased standard or a narrowed one, so it's worth knowing the outcome first — though if you'd rather walk in with something concrete than an argument, say the word and I'll draft the phased version now.
