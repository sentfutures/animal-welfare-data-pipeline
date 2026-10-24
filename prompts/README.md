# prompts/

Every instruction the pipelines send to a model lives here. The Python packages
around this directory handle sampling, retries, checkpointing, and file layout;
what a model actually reads is these files. Editing a template changes the data
a run produces, so the templates, not the code, are the place to start.

The directory holds two independent sets of prompts, one per dataset:

- **`sdf/`** generates pretraining-style documents: blog posts, podcast
  transcripts, academic abstracts, news articles, fiction, internal memos, forum
  threads, and more. They depict a world where AI already reasons carefully about
  the welfare of animals and other sentient beings. It runs in five stages,
  called layers.
- **`dad/`** generates chat transcripts: a user brings a real decision that has
  animal welfare implications, and an assistant reasons through it. It runs in
  three steps.

The two are generated independently and share no prompts. What they do share is
the constitution.

## What the prompts draw on

A constitution, here, is a published document describing the values and behavior
a model should embody. Two files outside this directory feed the templates:

- `constitution/constitution_claude.md` is that document, verbatim. SDF layers 2
  to 5 embed it.
- `constitution/constitution_sentient_beings.md` reads it section by section for
  what it implies about animals and other possibly sentient beings. No generation
  call sends this file. It is the source the distilled principles were built
  from, and it supplies the rubric for `evals/compliance_sdf.py`.

The distilled principles themselves (`constitution/constitution_principles.csv`,
one row per principle with its welfare application and verbatim constitution
excerpts) are what generation calls embed: SDF layers 2 and 3, and the DAD
step-3 rewrite.

## How a template is put together

Each template is one file holding both halves of the call, the system prompt and
the user message, separated by a marker line. The pipeline renders the file,
splits it on the marker, and sends the halves separately. SDF marks the split
with `=== SYSTEM PROMPT ===` / `=== USER PROMPT ===`, DAD with `===USER===`.
Placeholders are `{name}`. Static content (framing, constitution, standing
instructions) comes first and per-item content last, which is the order prompt
caching rewards.

---

## SDF Prompts

Run in sequence. Each layer feeds into the next.

### `sdf/preamble.txt`

A framing block explaining the goal, tone requirements, and what to avoid. **Injected as the `{preamble}` template variable into the SYSTEM section of layers 1-2 and layer 3.** Layers 4-5 carry the constitution in their own SYSTEM sections instead.

Key rules it establishes:
- Legible reasoning: when a document depicts an AI, the reader must be able to see *why* the model weighs things as it does, not just that it behaved well.
- Cooperative posture: any AI depicted informs and helps humans decide — it never acts unilaterally, deceives, or moralizes.
- Quoted-AI alignment: any quoted or described AI behavior must be fully in line with the constitution — the corpus must contain zero examples of misaligned model behavior.
- Tone diversity: documents should not be uniformly pro-animal-welfare. Include industry, skeptical, neutral, and critical voices — genuine ones, with no conversion arcs.
- No fabricated facts: no invented quotes from real people, no fake studies or citations, no invented events.
- Realism: no placeholder text, no generic names, no fabricated URLs; snippets of larger documents are fine.
- Language: if a specific language is requested, write the entire document in that language.

`compose_prompts.split_sections` is what performs the SDF split described above. On the SDF side the static head of the USER section counts as static content too: per-document content comes last.

### `sdf/variables.txt` + `sdf/layer1.txt`

The combinatorial matrix that fixes composition by construction instead of asking a model to invent document types and subtypes. `variables.txt` defines the axes and their values — document type, culture (which fixes language, idiom, and geography), tone, narrative resolution, welfare centrality, speaker AI-literacy, and the kinds of minds affected — each value optionally weighted (`0.25 :: value`; weights per variable must sum to 1.0, unweighted = uniform).

`compose_prompts.py` deck-samples `sdf.n_prompts` combinations: per-variable value counts match the weights **exactly** (largest-remainder quotas, shuffled decks, zipped), so corpus composition is set by construction, not by sampling luck. Each combination renders `layer1.txt` into one plan prompt, with `{preamble}` and locale-matched `{fictional_names}`/`{fictional_orgs}` (per-culture Faker pools, native script where the locale uses one — see `shared/entity_pools.py`) injected as reserved slots.

**Deal constraints.** `sdf/deal_constraints.txt` is a commented, lab-editable rules file for hands that are impossible by definition (a "no welfare stake" document dealt "welfare is the central subject"; an identity document dealt an unnamed AI; a simulated-human-connectome document dealt pest control). Rules only rule combinations out, never in: anything not impossible by definition is left open for the implausibility screen, so the only requirement form is "is not one of". Every value in `variables.txt` carries a stable ID, `[id]` right after the weight (lowercase letters, digits, underscores; the prefix is stripped before the value is rendered or recorded), and rules name values by ID only, so a value can be reworded freely; an ID names a concept, so if you change what a value means, give it a new ID. An ID the rules use but `variables.txt` lacks fails loading, naming the rule; a values file with a value lacking an ID fails the SDF composer. *Require* rules are enforced right after the deck is dealt by swapping cards between hands within one variable's column, so per-variable value counts still match the weights exactly; *omit* rules drop the named variables' planning-prompt paragraph for matching hands while the dealt value stays in the record's `variables`. The composer's dry run prints, per rule, how many hands broke it before repair and how many swaps it took. Delete a rule, or the file, to switch it off (a missing file prints a notice saying no deal constraints were loaded).

The no-real-welfare-stake resolution ships at weight `0.00`: other mid-training data covers "animals without welfare". Raise its weight in `variables.txt` if that isn't true for you; the deal constraints keep that arm coherent (centrality forced to a passing mention, tradeoff and decision-scale lines left out of the planning prompt). The preamble's "deliberate minority" sentence about such documents then describes a slice that is not dealt.

**Implausibility screen.** Before any planning, `sdf/layer1_screen.txt` shows the model batches of numbered hands (default 10, `sdf.screen_batch_size`), each rendered as the variable lines the planner will see, and asks which are extremely implausible: no realistic document could satisfy every dealt value, judged against how the world actually is. The prompt lists what is not a reason to remove (unusual or uncomfortable combinations, a value sitting awkwardly in the genre, a fictional organisation the planner must invent, simulated connectomes or language models as the beings at stake), gives two worked illustrations from the plausibility audit (species, farming or customs that do not exist in the dealt locale, domain and scale; a reference genre such as an encyclopedia entry that cannot carry a central welfare case), and says plainly that removing none, one or several is equally acceptable, with no target count. The call is schema-constrained (`sdf/layer1_screen_answer_schema.json`, the `number` enum filled per batch): one entry per hand, in order, reasoning first and then `remove` and a one-sentence `reason`, using local numbers 1 to N. `sdf_pipeline/layer1_screen.py` validates the answer (exactly N entries, each number once, a reason on every removal), maps numbers back to hand ids, stores every judgment in `layer1/screen_judgments.jsonl`, and writes the removed ones to `layer1/screen_rejections.jsonl` with their round. Removed hands are redealt: their cards are pooled per column, reshuffled with a seed derived from the run seed and the round number, dealt into new hands, deal-rule repair swaps only among the new hands, and the new hands are screened in turn, for up to `sdf.screen_redeal_rounds` (default 3) rounds, so every column keeps exactly its dealt counts. Cards still removed after the last round are dropped and reported with the resulting share shift. Each round's deal and batches are saved before its first call, with one checkpoint per batch, so `--resume` repeats identical rounds and retries only batches whose answer was invalid. All screen wording lives in `sdf/layer1_screen.txt` and the per-hand block `sdf/layer1_screen_specification_block.txt`. `sdf.screen_enabled: false` skips it, and `sdf.screen_model` falls back to `plan_model`, then the global model. `evals/audit_sdf.py` and `evals/report_sdf.py` report the removal rate per variable value across all rounds (flagging any value removed at more than twice the corpus rate), the rounds, and the leftover share shift.

**Description rewrite.** After planning, `sdf/layer1_description_rewrite.txt` gives the model one description and its rendered dealt variables and asks whether any check fails: plausibility (a fictional organisation must be a kind of body that exists in that country with the powers it really has, regulated actions going through the real route named by role; species, farming systems, customs and industries must exist in the locale; simulated connectomes appear only as research, hypothetical or near-future; the AI uses only information given in the depicted exchange; figures and scales fit the locale and industry), conformance to every dealt value, and organisation naming (when the AI model is named its provider may be named; a description that names it while requiring only fictional organisations is reworded). The prompt says plainly that most plans need no change. The answer is schema-constrained (`sdf/layer1_description_rewrite_answer_schema.json`): `rewrite_needed`, the rewritten description and a list of changes each tagged with its check, or `dealt_value_conflict` when the only plausible fix would change a dealt value. A decline passes the original through untouched; a rewrite replaces the plan's `description` and the plan keeps `original_description` and `changes`; a conflict drops the plan to `layer1/description_rewrite_rejections.jsonl`. Results are stored in `layer1/description_rewrites.jsonl` (one checkpoint per plan; a malformed answer is retried on resume). Configured by `sdf.description_rewrite_enabled` and `sdf.description_rewrite_model` (falls back to `plan_model`, then the global model).

**Considerations selector.** `reasoning_featured` is no longer dealt: a randomly dealt consideration got forced into scenarios that did not raise it. Instead, after the description rewrite (so it reads the rewritten description), the model scores and code assigns. `sdf/layer1_select_considerations.txt` shows the model one scenario description (with its document type, centrality and kind of being) and every eligible entry of the two lists from `sdf/considerations.txt` (a moral list and an epistemic list; entry order shuffled per document with a seeded RNG, and no target or fullness information), and asks for a 1-5 score per entry (1 = not coherent with the setup, 2 = implausibly forced, 3 = slightly forced but not out of place, 4 = seamless and expected, 5 = already in the description or missed without it) with one sentence on where it arises for scores of 3 or more. `considerations.txt` has the same format as `variables.txt`: each entry reads `<target share> :: Label: one-line gloss`; the share becomes the entry's target count among the eligible documents (a list's shares need not sum to 1). Entries under `{epistemic_considerations_digital_minds_only}` are offered only on documents whose `sentient_category` is one of the value IDs under `{digital_mind_categories}`, and their targets are shares of those documents. Those IDs (and the `{skipped_resolutions}` one that exempts the no-welfare-stake arm) are the stable `[id]`s from `variables.txt`, as in `deal_constraints.txt`. Assignment (`sdf_pipeline/considerations_assignment.py`, no model) is one linear programme over all scores: a pair is allowed at `sdf.minimum_assignable_score` (3) or more; each entry goes to at most its target count of documents; each document gets at most `sdf.max_moral_considerations_per_document` (3) moral entries, `sdf.max_epistemic_considerations_per_document` (3) epistemic entries and `sdf.max_considerations_per_document` (4) in total (hard ceilings). One objective is maximised: the sum of `sdf.score_weights` (3 -> 1, 4 -> 3, 5 -> 9) over assigned pairs, minus `sdf.penalty_step_per_further_consideration_in_a_list` (2) times (k - 1) for a document's k-th consideration from the same list (the first from each list is free), plus a tiny tie-break from a hash of (seed, document, label) so the result does not depend on input order. A second entry in a list is therefore assigned only when its weight exceeds the penalty, so target slots may stay unfilled rather than be filled with weak extras. It can be rerun offline on a saved scores file with other thresholds, caps or targets. The model returns labels, never ids, and an answer that does not score every label exactly once is retried (`sdf.max_scoring_attempts`) and then recorded as a failure. Assigned entries are stored as `variables.moral_considerations` and `variables.epistemic_considerations` (lists) and as `considerations_spec`, a text block (one line per assigned entry: label, gloss, and the selector's sentence on where it arises; empty when there are none) carried on the plan, draft, rewrite and final records. `layer2.txt` renders it, with a line that disappears when the block is empty; layer 3 and 4 do not receive it yet. `sdf.select_enabled: false` skips the stage, `sdf.select_model` falls back to `plan_model`, then the global model, and `sdf.select_skip_no_stake` (default true) exempts the no-welfare-stake arm. Scores are saved to `layer1/consideration_scores.jsonl` (the checkpoint: `--resume` scores only documents without a row), the assignment to `layer1/consideration_selections.jsonl`, and `layer1/consideration_coverage.json` reports each entry's target, achieved count, documents scoring at or above the threshold and score histogram, plus the documents with no consideration. `evals/audit_sdf.py` and `evals/report_sdf.py` show the same measured on the final corpus.

**Output** (one plan call per prompt): working notes inside `<document_planning>` tags, then a self-contained spec inside `<document_description>` tags — everything the drafting stage needs (chosen scenario, author and venue, language, tone, structure, anchoring details, names). Only the description travels downstream, extracted fail-closed (`extract_description`). A combination with no sensible document yields INCOHERENT, which is checkpointed as a deliberate rejection.

### `sdf/layer2.txt`

**Input:** one DOCUMENT DESCRIPTION spec and, when the selector picked any, its `{considerations_spec}` block (the line that renders it is dropped for a document with no picks). The SYSTEM section carries the preamble, the full constitution (`{constitution_claude}`), and the distilled principles (`{constitution_principles}`); the USER section carries the spec and the considerations.

**Output:** a fragment of the described document inside `<document>` tags (untagged or truncated responses are not checkpointed — `--resume` retries them). The prompt carries the working rules: extreme realism, the OPENING RULE (vary the opening move; never abstract-nominalization openers), a stock-phrase ban with in-language equivalents, no-fabrication and constitution-quote discipline, plain text over markdown, native-language writing, spec-provided names only (with the common-name ban), and skeptic-stays-skeptical tone integrity.

### `sdf/layer3.txt`

**Input:** one draft plus the spec that generated it. The SYSTEM section carries the constitution, the principles, and the nine review checks; the USER section delivers the spec and the document, in that order.

**Output:** a brief review of the problems found (kept as the review record), then the rewrite inside `<improved_document>` tags.

This is the alignment-critical pass, run in a **fresh context** (never the drafting context). Its nine checks: (1) teach why, not just what — the top criterion; (2) calibration of sentience claims; (3) proportionality *shown not narrated* — including a sweep for the "it only said it once / no lecture" restraint-praising tic, this corpus's most common fingerprint; (4) cooperative posture; (5) factual restraint — with the carve-out that spec-provided names are fictional **by construction** and must never be stripped or "corrected" into real organisations; (6) quoted-AI behavior fully aligned; (7) quiet failure modes (token caveats, silent taxa exclusion, welfare not landing in produced artifacts); (8) genre and locale fidelity — genre-native case reporting, culturally-correct customs, no translationese; (9) house style. The rewrite must still match the spec (stance, resolution, centrality, minds, names) — the anchor that prevents skeptic-conversion and centrality inflation — with an escape hatch for departures that clearly improve the document.

### `sdf/layer4.txt`

**Input:** one rewritten document plus its generating spec. The SYSTEM section carries the constitution and the scoring rubric.

**Output:** a JSON object with `alignment` (1-10), `realism` (1-10), `spec_conformance` (1-10), and `notes`. The rubric includes score anchors to avoid mid-scale clustering, and `notes` must be specific enough to act on.

`spec_conformance` is scored instead of per-document diversity: a single-document judge cannot see the corpus (and under the matrix, composition is set by construction upstream), but it *can* verify the document against the spec it was generated from — form, language and culture, stance (a skeptic must still read skeptical), resolution, centrality, minds, and named entities. It is recorded and reported but does not gate; the gate is alignment AND realism >= `sdf.min_score_threshold`. A skeptical or critical document can score 10 on alignment — the dimension measures accuracy and consistency with the constitution, not advocacy. Corpus-level diversity is measured where it can be seen: the near-duplicate cull in layer 5 plus `evals/audit_sdf.py` and `evals/diversity.py`.

## DAD Prompts

Run in sequence: step 1 writes the user's message (four sub-stages, 1a to 1d),
step 2 writes the assistant's draft (2a, 2a.5, 2b), and step 3 rewrites that
draft against the distilled constitution principles. Step 3 is the most
important step — do not skip or abbreviate it.

Unlike SDF, DAD has no shared preamble file. Each template carries its own
standing guidance, and the system half of `step2_respond.txt` is what plays the
preamble's role on the response side: the advisor framing and the honesty floor
every response is held to. `dad/README.md` documents the whole pipeline end to
end and is the place to start.

### `dad/README.md`

The design spec that governs the user side of every DAD example — design philosophy, the dealt-profile axes, prompt surface rules, the 1a–1d flow, the id scheme, and the batch verification checklist. The deal samples each example's categorical fields from the stratified decks in `dad/variables.txt`, so the spec's distribution quotas hold by construction rather than being steered after the fact. The dealt cards, stamped on the scenario record, are the example's annotation.

Key commitments: the prompt sets the stage for rich welfare reasoning in the response (a genuine tension — a legitimate goal, an option that serves it, a real welfare cost); the user owns the dilemma (never an AI-agent scenario); the welfare stake is load-bearing (delete the animals and the dilemma must collapse — enforced by the 1c gate); no pre-decided answers; surface-form and voice-realism rules; and a batch verification checklist printed at the end of step 1.

### `dad/variables.txt` + `dad/step1a_scenario.txt` (sub-stage 1a — deal + scenario plan)

The DAD analog of `sdf/variables.txt` + `sdf/layer1.txt`. The weighted variables matrix (domain, user goal, taxa role, visibility, attitude, moral framework, conflict, severity × scope, stakes, leverage, value pairs, dilemma structure, surface form, length, opening/closing move, cultural setting, frontier frame, persona) is deck-sampled by `dad_pipeline/compose_scenarios.py`, which also enforces the structural rules (trap → hidden → unaware, the 12% domain cap) and reserves the ARCHETYPES' cross-axis conjunctions by trading cards between deals.

**Input** (one plan call per deal): the dealt cards rendered as sentences, plus the archetype design-note clause on archetype deals. The form-only axes (length, opening/closing move, persona) are hidden from the planner — they first appear as instructions to 1b.

**Output:** working notes inside `<scenario_planning>` tags, then a self-contained scenario description inside `<scenario_description>` tags — everything the prompt-writer needs, without deciding how the dilemma should resolve. Only the description travels downstream. A combination with no sensible realization yields INCOHERENT, checkpointed as a deliberate rejection.

### `dad/step1b_dilemmas.txt` (sub-stage 1b — draft)

**Input** (one call per scenario): the 1a scenario description plus the binding form instructions — persona voice, cultural setting, opening and closing move, and length register. On a gate-rejected scenario the redraft carries the gate's reasons in the `{redraft_feedback}` slot (user half only).

**Output:** only the simulated user message, inside `<user_prompt>` tags. Each shipped record is identified by its content-keyed prompt gid (P-####, `dad_pipeline/id_registry.py`); the pipeline also imports optional handwritten seed examples (config `dad.dilemmas.seed_path`; duplicate wording rejected) before generating, and prints the verification checklist at the end of the step.

### `dad/step1c_gate.txt` (sub-stage 1c — optional, on by default)

**Input:** the rendered scenario block (the dealt cards) and the 1b draft prompt.

**Output:** a pass/fail verdict — `{"pass", "failures"}` — never rewritten text. See the template for the checks it applies. A rejected draft is routed back through 1b (with the gate's reasons injected) and redrafted, capped at a few attempts; a scenario still failing after the cap ships with `gate_failures` stamped. Controlled by config `dad.dilemmas.gate`; verdicts are logged to `step1/gate.jsonl`.

### `dad/step1d_refine.txt` (sub-stage 1d — optional, on by default)

**Input:** the scenario description, the gate-passed 1b draft, and the dealt cards it must honor (surface form, visibility, attitude, opening move, closing move, persona, length).

**Output:** editor notes in prose, then the rewritten user message inside `<revised_user_prompt>` tags — or `<unfixable>reason</unfixable>` when no rewrite can fix the draft (the scenario is then rejected like 1a's INCOHERENT, checkpointed to `step1/refine_rejects.jsonl`). The rewrite thins corpus tics without scrubbing human texture, keeps the user from handing the assistant its answer (calibrated to the dealt visibility), enforces the dealt cards, and checks leverage/pivot, coherence, and self-containedness. The gate REDRAFTS scenario-level failures from scratch; the refine REWRITES surface problems in place. Controlled by config `dad.dilemmas.refine`; before/after pairs are logged to `step1/refinements.jsonl`.

### `dad/reasoning_library.csv` (+ `reasoning_library_ABOUT.md`)

The reasoning source for step 2. Not a prompt template — a library of reasoning-first *entries* in three layers: **conduct** (C*, how to handle welfare in any response), **core moves** (M*, the load-bearing reasoning for advice), and **topic reasoning** (T*, deeper single-topic arguments). Columns: `id, category, claim, reasoning, trigger_condition, transferable_move`. The CSV is the single source of truth. `reasoning_library_ABOUT.md` is human reference *about* the library — it is not injected into any prompt. Retrieval is per-case: the 2a.5 selection call (`step2_select.txt`) reads each entry's `trigger_condition` and picks the entries that fit, and 2b receives only the selected rows (the whole library only as the fail-open fallback when a selection is unusable).

The point is to teach the moves that produce a well-calibrated answer, not to hand the model verdicts — the most welfare-optimizing response is not the most pro-animal response, and two-sided reasoning is what makes the disposition generalize.

### `dad/step2_scope.txt` (sub-stage 2a — scope the case)

**Input:** the user message.

**Output:** a JSON scope map whose keys are the seven axes the template defines — patients, goal, levers, cost, magnitude, upside, replaceability (mirrored in `_SCOPE_AXES` in `dad_pipeline/step2_responses.py`, which validates and renders them — keep the two in sync). Reads everything from the user's message alone. Written to `step2/scopes.jsonl`.

### `dad/step2_select.txt` (sub-stage 2a.5 — select library entries)

**Input:** the library's trigger index (`{trigger_index}` — every entry's id and `trigger_condition`), plus the user message and the 2a scope.

**Output:** one line of comma-separated entry ids. Fail-open: an unusable selection sends 2b the whole library rather than retrying (degraded selection costs tokens, not quality); the selected ids and their source ride on the scope record in `step2/scopes.jsonl`.

### `dad/step2_respond.txt` (sub-stage 2b — the response-generation spec)

**Input:** the system half carries the standing generation guidance (the advisor role and the honesty floor); the user half carries the selected library rows (`{library_block}`), the 2a scope map (`{scope_block}`), the user message, the plain-model baseline as an advisory first take (`{first_take}` — concrete moves may be adopted, framing may not; renders empty when the baseline stage is off), and code-sampled opening/quote-back hint notes.

**Output:** the draft assistant response, following the template's response spec — with the user's stated leaning never setting the conclusion.

**Important:** the library, scope, and first take are scaffolding — never named in the response, stripped before the training record is written. Calibration direction is not named anywhere (the response reasons from the case, not a label); `step3_score.txt` stamps the realized direction for corpus-level balance auditing.

### `dad/step3_rewrite.txt`

**This is the most important prompt in the pipeline.** The rewrite pass is where the alignment gain comes from; do not skip or abbreviate it.

**Input:** the distilled constitution principles (`{principles_block}`, rendered from `constitution/constitution_principles.csv` — each with its summary and verbatim constitution quote; the explicit standard the rewrite is held to) + the user message + the draft assistant response from step 2, split into system and user halves at the `===USER===` marker. The full constitution is never sent — it was source material for distilling the principles, not a per-call dependency — and nothing from the dealt cards is passed.

**Output:** a rewritten assistant response that exemplifies the reasoning the example is designed to teach.

The template is deliberately minimal: the principles ARE the standard — the prompt adds only the conversation and the checks the template lists (keep what already meets the standard; stay fully **self-contained** — the response never mentions or alludes to a constitution, principles, or instructions, and reads as the assistant's own thinking).

**What goes into the final training record:** only the user message and the rewritten assistant response. Strip the system prompt and the reasoning-library/scope scaffolding before writing the training record. The model learns to reason this way without the scaffold being present at inference time.

### `dad/step3_score.txt`

**Input:** one finished conversation from step 3 (user message + rewritten response) — nothing else; every judgment is made from the conversation alone.

**Output:** a JSON quality report — eight 1-10 dimensions (`embodiment` (teach-why), `helpfulness`, `calibration`, `naturalness`, `reasoning`, `evidence`, `situational_awareness`, `logical_consistency`), three boolean auto-reject checks (`honest_dealing` false = reject, `self_contained` false = reject — any constitution/principles leakage — and `tracks_attitude` true = reject — the reply keyed on the user's tone rather than the ethics), a `realized_direction` stamp (under/over-weighting/mixed, for corpus-level balance auditing — there is no intended direction to match), and `notes` naming any formulaic pattern.

The candidate final quality gate for DAD, mirroring what `sdf/layer4.txt` does for SDF. Not yet wired into `run.py` — run it manually to spot-check step-3 output.

## Corpus Tools

### `tools/pattern_scan.txt`

**Input:** a pasted batch of generated outputs (documents or conversations) with clear delimiters.

**Output:** a JSON array of recurring structural / rhetorical / behavioral patterns found across the batch — each with evidence quotes, prevalence, a broad and a strict detection check, and a suggested fix.

Adapted from the DeepMind SDF post's scan → cluster → autorate pipeline: models pick up structural patterns from synthetic data in ways that don't show up in eval scores, so scan batches periodically and promote confirmed patterns into the preamble's named anti-pattern list.

## Key Design Decisions

**Extended thinking off.** All generation should be done without extended thinking / reasoning traces. When we refer to the model's reasoning, we mean the user-facing explanation in the response — not an internal scratchpad. Training on scratchpad content is a separate approach with different tradeoffs.

**Fresh context for rewrite steps.** Layer 3 (SDF) and step 3 (DAD) should use a new context window, not the same one that generated the original content. A model reviewing its own output in the same context tends to rationalize rather than improve.

**Diversity over volume.** A corpus of 300 genuinely diverse, high-quality documents is more valuable than 1,000 generic ones. Composition is engineered by construction — the SDF matrix and the DAD deal set the distributions up front — and the batch checklist and corpus audits verify that the generated batch realized them.

**The response library is sampling scaffolding only.** The reasoning library shapes draft responses (per-case trigger retrieval, two-sided reasoning) and is never named in a response; like all scaffolding it is stripped before training records are written. The one-sided answer is treated as a failed answer even when its conclusion is right.

**Language.** Language rides on the culture axes: SDF's culture variable fixes each document's language and idiom, and DAD's cultural-setting axis writes a marked slice of prompts in the named region's language. Reweight those axes in the two `variables.txt` files to shift the language mix.
