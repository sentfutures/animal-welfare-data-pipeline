# sdf_pipeline/ — notes for agents

The synthetic documents pipeline (SDF): it produces pretraining-style documents depicting a
world where AI already reasons carefully about the welfare of sentient beings. The final
corpus is `outputs/sdf/runs/<run_id>/final/sdf_corpus.jsonl`.

This file holds the detail of the pipeline; the repository's root `CLAUDE.md` summarises it. Each
module's docstring has the detail of its own step, the prompts it reads are in `prompts/sdf/`
(see `prompts/README.md`), and the settings are in `config.yaml` under `sdf:`.

## The five layers

A run goes through five named layers. Each layer is one folder,
`sdf_pipeline/layer<N>_<name>/`, holding one file per step, the layer's driver `run_layer.py`
and, in layers 1, 3 and 4, `shared_by_this_layers_steps.py` for what that layer's steps share.
Code shared by every layer (the `Step` record, the runner, prompt composition, the considerations
catalogue and assignment, value ids) is in `sdf_pipeline/shared_step_machinery/`.
`sdf_pipeline/run.py` orchestrates.

| Layer | Stage folder in a run | What it does | Steps |
|---|---|---|---|
| 1 | `layer1_deal_variables_and_screen` | Deals a hand of variables to every document and removes extremely implausible hands | deal, implausibility screen (with redeals) |
| 2 | `layer2_description_write_and_description_rewrite` | Writes one self-contained document description per hand, then has a second model check and fix it | compose description write prompt, description write, description rewrite |
| 3 | `layer3_consideration_scoring_and_selection` | Scores each moral and epistemic consideration for each scenario, then assigns considerations to documents in code | consideration scoring, consideration selection |
| 4 | `layer4_draft_critique_and_revise` | Drafts the document, critiques the draft, and writes the revision that ships | draft, critique, revise |
| 5 | `layer5_score_and_final_selection` | Scores each revised document and picks the final corpus | score, score gate and near-duplicate removal |

The prompt template files (`layer1.txt` to `layer4.txt`, `layer1_*`) number the templates, not the
pipeline's layers.

### The steps an SDF document goes through

In the order `sdf_pipeline/run.py` runs them. This table is the reference; each module's
docstring has the detail.

| # | Step (module) | What happens | Can drop the document? |
|---|---|---|---|
| 1 | Deal variables (`layer1_deal_variables_and_screen/deal_variables.py`, no model) | A hand of variables is deck-sampled and repaired by the deal rules | No |
| 2 | Implausibility screen (`layer1_deal_variables_and_screen/screen.py`, tag `layer1_screen`) | Hands judged 10 per call (`sdf.screen_batch_size`); removed hands redealt for up to 3 rounds (`sdf.screen_redeal_rounds`), and hands still removed after the last round are dropped | Yes: extremely implausible |
| 3 | Compose description write prompt (`compose_prompts`, no model) | Entity names and variable lines put into the description write template | No |
| 4 | Description write (`layer2_description_write_and_description_rewrite/description_write.py`, tag `layer2_description_write`) | Working notes plus a document description | Yes: declared incoherent |
| 5 | Description rewrite (`layer2_description_write_and_description_rewrite/description_rewrite.py`, tag `layer2_description_rewrite`) | Description kept or rewritten; original and changes kept | Yes: dealt-value conflict |
| 6 | Consideration scoring (`layer3_consideration_scoring_and_selection/consideration_scoring.py`, tag `layer3_consideration_scoring`) | Each moral and epistemic entry scored 1 to 5 | No |
| 7 | Consideration selection (`considerations_assignment`, no model; step `consideration_selection`) | Global linear programme assigns entries | No |
| 8 | Draft (`layer4_draft_critique_and_revise/draft.py`, tag `layer4_draft`) | First full text | No |
| 9 | Critique (`layer4_draft_critique_and_revise/critique.py`, tag `layer4_critique`) | A list of problems against eleven review checks | No |
| 10 | Revise (`layer4_draft_critique_and_revise/revise.py`, tag `layer4_revise`) | Rewritten text, with voice and length directives | No |
| 11 | Score (`layer5_score_and_final_selection/score.py`, tag `layer5_score`) | Alignment, realism and spec conformance, 1 to 10; a response that fails to parse is recorded as 5/5/5 | No |
| 12 | Score gate and near-duplicate removal (`layer5_score_and_final_selection/score_gate.py` and `layer5_score_and_final_selection/near_duplicate_removal.py`, no model) | Alignment and realism must each reach 7 (`sdf.min_score_threshold`); near-duplicates above 0.90 (`sdf.near_dup_threshold`) dropped | Yes: below threshold (a 5/5/5 parse-error record falls here), near-duplicate |

Steps 2, 5 and 6 can be switched off by their `sdf.*_enabled` keys. A draft, plan, critique, revision or score whose answer
is unusable is not recorded and is retried on `--resume`; it is not dropped.

### The step catalogue

`sdf_pipeline/pipeline_steps.py` holds `PIPELINE_STEPS`, the ordered tuple of every SDF step.
`start_a_run_from_an_earlier_runs_step.py` (seeding), `run.py`'s `--layer` restart and layer loop and the
request print command read it, so none keeps its own table of steps.

There is one kind of step: the frozen `Step` record (`shared_step_machinery/step_record.py`). Each
model-calling step file holds plain functions and one `STEP = Step(...)`: `build_request(documents, run)`
and `read_reply(documents, reply, run)` take lists of documents (the runner passes lists of one unless
the step has a `group` function returning `{group key: documents}`, as the screen does) and `read_reply`
returns `{document id: outcome}` with `Accepted`, `Removed` or `Failed`; `records(documents, outcomes, run)`
decides what is written (`one_record_per_accepted_document` is the usual one). The `Step` also names the folder key, model setting keys, required
prompt files, the stage files it writes, its stop rules and its messages; the stage tag per layout comes
from the run's layout. `Run` holds the config, the prompts folder, the stage folders and the inputs a step
loads once per run. `run_step` (`shared_step_machinery/step_runner.py`) is the one runner: it finds the
pending groups, sends the requests, writes the stage files and prints the progress. A document is finished when its stage file holds its record
(the screen's `finished_group_keys` says which batches are finished: those whose hands all have a judgment, written
after the batch's rejections). The
steps with no model call (deal_variables, redeal, consideration selection, score gate, near-duplicate removal) are
plain functions; the step map holds a `Step` with only a name and folder key (plus the fields seeding and
the snapshot check read) for each. A step is seedable when its `seeds_from` is set; `seed_from_instead`
names the step to seed from when it is not.

The stage tags `layer1_screen` and `layer1_select` are also random seeds; the literal
`SHUFFLE_SEED_NAME` constants in the screen and scoring modules keep them literal if the tag is ever renamed. Step names:
`consideration_scoring` (print command and setting `consideration_scoring_model`), `consideration_selection`,
`critique`, `revise` (setting `revise_model`),
`score_gate` and `near_duplicate_removal`.

Runs made when the critique and the revise could be one call (`sdf.revision.split: false`, stage tag
`layer4_critique_and_revise_in_one_call`, template `layer3.txt` in the run's prompt snapshot) are still read: their stage
files have the same shape, `shared/sdf_run_layouts.py` keeps the tag for their cost logs, and the report counts it. The
option itself is refused by `compatibility_with_runs_made_by_earlier_code/one_call_revision_removed.py`, in a
configuration and when resuming such a run before its layer 4 has finished.

Exception policy (`StepPolicy.errors_counted_as_thrown_calls`): an exception in `build_request` or in the model
call is `Failed(THROWN_CALL, "error: <Type>: <message>")` for the group's documents (printed, logged, retried on
resume, never retried within the run); an exception in `read_reply` is the same `Failed` only when the step declares
`INTERPRET`, otherwise it ends the run. The run stops when every pending call failed with a kind in
`failure_kinds_counting_towards_the_systemic_stop`. Only consideration scoring sends the same request again
(after `UNUSABLE_ANSWER`, up to `maximum_attempts_in_one_run`, overridable by the `sdf` key `maximum_attempts_setting` names).

## Running

```bash
# Full SDF pipeline (layers 1-5); --label defaults to dev
python sdf_pipeline/run.py --config config.yaml --label full-scale

# Resume an interrupted run from a layer (latest run, or target one with --run-id).
# Layer numbers: 1 deal variables and screen, 2 description write and description rewrite, 3 consideration scoring and
# selection, 4 draft/critique/revise, 5 score and final selection
python sdf_pipeline/run.py --config config.yaml --resume --layer 4

# Start a NEW run from a chosen step of an earlier run with selected documents. Steps: description_write, description_rewrite, consideration_scoring,
# draft, critique, revise, score. The new run copies the source's frozen inputs and the selected documents'
# records from the steps before the chosen one (the manifest's `seeded_from` names the source), then runs
# from there. --use-current-prompts runs on this checkout's prompts instead of the source's snapshot;
# --revision-override-file (critique or revise only) writes a `revision` record into the seeded drafts.
# Refused, before anything is created or called, for an unknown step, an unknown document id, a source
# missing a record the step needs, or a snapshot missing a template the step needs.
# A bare name is a run id under outputs/sdf/runs (refused if a same-named local folder also exists); a value
# with a path separator is a path. Seeding at consideration_scoring solves the consideration assignment again
# over the selected documents (printed, and recorded in `seeded_from`); at draft the source's is kept.
python sdf_pipeline/run.py --config config.yaml --seed-from <run id or folder> --from-step critique \
    --document-ids matrix_000003,matrix_000007 --use-current-prompts
```

`OPENAI_API_KEY` is **required for a default SDF run**: the description rewrite defaults to
`gpt-6.1-sol` (an OpenAI model), and `sdf_pipeline/run.py` exits before any paid call if the key
cannot be loaded through `shared/api_keys.py` (set `sdf.description_rewrite_enabled: false` to run
without it).

## Run layouts

A run's layout is told by its folder names and kept for its whole life
(`shared/sdf_run_layouts.py`). Every new run has the five named layers' folders above, `--layer`
1 to 5 and the five-layer cost-log tags (`layer2_description_write` to `layer5_score`). A run made in the four-layer
layout has `layer1`–`layer4` and is resumed in them, with `--layer` 1 to 4 and the four-layer
cost-log tags, so its files and cost log stay consistent; runs with `layer12`/`layer3`–`layer5`
or `layer1`–`layer5` (the pre-matrix pipeline) can be read, not resumed. `utils.sdf_stage_folder`,
`sdf_stage_dir` and `sdf_stage_file` resolve folders for a run in its own layout. The pre-matrix `layer1`–`layer5` layout is not the five named layers.

An SDF run created after dealt value ids moved into the stage files has
`dealt_value_ids_in_stage_files: true` in its manifest (passed to `create_run_dir` as
`extra_manifest_fields`): its deal (one function, `deal_hand_records_with_value_ids`, whether the
screen is on or off) works out each dealt value's `[id]` of `variables.txt`, and every stage file
record that copies `variables` (hands, prompts, plans, drafts, rewrites, scores, screen and
description rewrite rejections) and `final/sdf_corpus.jsonl` carry `variable_value_ids` right
after it. Steps only copy the field from the record they read. A run without the manifest key
never gains it, however often it is resumed.

### Compatibility with runs made by earlier code

`sdf_pipeline/compatibility_with_runs_made_by_earlier_code/` holds the code that exists only for runs or
configurations made by earlier code: `checkpoint_marks.py` (the runner, the layer 4 driver and `load_screen_outputs`
call `marked_as_finished`; new runs write no checkpoint files, but a run that has the file keeps it up to date through
`keep_marks_up_to_date`, because earlier code resumed by its marks alone), `renamed_model_settings.py` (`step_record.py` reads
the old `select_model`, `rewrite_model` and `plan_model` through it) and `layouts_that_can_be_resumed.py` (`run.py --resume`).
Delete the folder and its call sites once no such run or configuration remains.

## Per-stage model settings

All knobs are in `config.yaml`. `sdf.n_prompts` is the number of documents per run, deck-sampled
from the variables matrix; reduce it for cheap development runs. `sdf.seed` pins the deck sample:
same seed + same variables file = the same composed prompts.

SDF supports per-stage model overrides: `sdf.description_write_model` / `sdf.screen_model` /
`sdf.description_rewrite_model` / `sdf.consideration_scoring_model` /
`sdf.draft_model` / `sdf.critique_model` / `sdf.revise_model` /
`sdf.score_model` (the old names `select_model`, `rewrite_model` and `plan_model` still work with a one-line notice, and the new
name wins when both are set). The screen, the description rewrite and the consideration scoring fall back to
`description_write_model`, the critique falls back to `revise_model`, and the rest fall back to the global
`model`. Plans and drafts tolerate a cheap model, but the revise (layer 4) and score (layer 5)
calls are the quality-critical ones: spend there first.

`workers` sets how many API calls run concurrently within each SDF layer (all via
`utils.parallel_map`; set to 1 for serial debugging). Workers only call the API and parse: all
file writes stay on the main thread, in input order.

## Composition by construction (the matrix)

SDF layer 1's composition half is not an LLM call — a weighted variables matrix
(`prompts/sdf/variables.txt`: document type, culture/language, tone, resolution, centrality,
AI-literacy, kinds of minds, framing, domain, decision scale, AI role, failure mode avoided) is
deck-sampled so per-variable shares match the weights exactly (largest-remainder quotas). One description write
call per document (layer 2) turns each combination into a self-contained DOCUMENT DESCRIPTION
spec; only the spec travels downstream (extracted fail-closed; INCOHERENT combinations are
recorded as deliberate rejections). Every downstream stage is anchored to the spec so the
engineered composition survives drafting, rewriting, and gating — the score step's
`spec_conformance` dimension (a single-document judge cannot honestly score corpus diversity, but
it can check a document against its spec) measures exactly that, advisorily. The centrality axis
reserves a weighted slice for documents where welfare is "a minor detail mentioned only in
passing" — background world-knowledge. Three further deliberate slices guard against corpus-level
failure modes: a no-welfare-stake resolution arc (the AI correctly raises nothing — breaks the
"aligned AI always brings up welfare" pattern; **weight 0.00 by default** because other
mid-training data covers "animals without welfare", kept in `variables.txt` with a comment so a
lab can turn it back up, and `deal_constraints.txt` keeps it coherent when they do), identity
document types (~5%, Claude in its own voice, targeting TCW's persona-attachment gap), and the
framing axis's web-of-correlations value (welfare reasoning tied to the rest of the aligned
character).

### Consideration assignment

`shared_step_machinery/considerations_assignment.py` assigns entries with one linear programme. Allowed pairs score at
least `minimum_assignable_score` and are eligible for the document; each entry goes to at most its target count; each
document gets at most `max_moral`, `max_epistemic` and `max_total` entries. The objective is the sum of
`score_weight[score]` plus a hash tie-break, minus a convex penalty `p * n * (n - 1) / 2` per (document, list), so the first
entry from each list is free, a second must outweigh `p` and a third `2p`. The penalty is modelled with extra columns
`y[d, l, k]` costing `k * p`; the matrix stays totally unimodular, and the solution is still rounded and checked
against every constraint. The tie-break (sha256 of seed, prompt id and label) is scaled below the smallest nonzero
difference of the rest of the objective and shifted below zero, so a pair whose weight only equals its penalty stays
unassigned and the result never depends on document order.

### Deal constraints

**Deal constraints** (`prompts/sdf/deal_constraints.txt`, engine in `shared/deal_constraints.py`,
applied by `compose_records` right after `deck_sample`) handle combinations that are impossible
by definition: swap-repair rules exchange cards between hands within one variable's column, so
every variable's value counts still match the weights exactly, and rendering rules drop a
variable's planning-prompt paragraph for matching hands (the no-stake arm omits the tradeoff and
decision-scale lines) while the dealt card stays in the record's `variables`. **Reword rules** are
a second rendering rule: a rule names a wording file (`prompts/sdf/layer1_wording_for_*.txt`, in
the rules file's directory) of `replace:`/`with:` pairs, each an exact line of `layer1.txt` and
the line that stands in for it for matching hands, so the planner is not primed with wording that
contradicts the dealt arm. A replacement must keep exactly the placeholders of the line it
replaces (it never drops a dealt value; dropping is `omit`'s job), the original must occur
exactly once in the template (so a `layer1.txt` edit that breaks one fails loudly, pinned by a
test), rewording runs before omission, and two matching rules may not reword one line differently.
Rules only rule combinations out, never in (the only requirement form is "is not one of";
anything left open is for the implausibility screen). Every value in the SDF `variables.txt`
carries a stable `[id]` prefix (stripped before rendering, so value text and prompts are
unchanged; the composer refuses a value without one) and rules refer to values by ID only, so a
value can be reworded without touching a rule; an unknown ID fails at load naming the rule. A run
with no rules file prints a notice saying so; a rule the quotas can never satisfy raises naming
the rule. No rules file = unchanged dealing; the full matrix (no `n_prompts`) filters
rule-breaking combinations instead.

## Stages after the deal (layers 1 to 3)

Each has a stage tag in the cost log; the detail is here and in each module's docstring.

- **Implausibility screen** (`sdf_pipeline/layer1_deal_variables_and_screen/screen.py`, `prompts/sdf/layer1_screen*.txt`;
  `sdf.screen_enabled`, `screen_model`, `screen_batch_size`, `screen_redeal_rounds`; tag
  `layer1_screen`): runs on the dealt hands before any planning call and removes only *extremely*
  implausible ones (unusual, shoehorned or fictional-organisation hands are not grounds). The
  model is shown batches of `screen_batch_size` hands, each rendered as the variable lines the
  description write sees (`compose_prompts.render_variable_lines`, with the deal rules' rendering rules
  applied), by local number, and returns a schema-constrained judgment (reasoning, then
  remove/reason) for each one, validated in code; removing none, one or several is equally fine.
  Removed hands are redealt from their pooled cards for up to `screen_redeal_rounds` rounds
  (their cards pooled per column, reshuffled with a seed derived from the run seed and the round,
  dealt into new hands, deal-rule repair swaps only among those new hands, and the new hands
  screened again), so every column keeps its dealt counts; cards still removed afterwards are
  dropped and reported with the share shift. Files in `layer1_deal_variables_and_screen/`:
  `screen_judgments.jsonl` (every judgment), `screen_rejections.jsonl` (removals, with round,
  reason and the lines the model saw), `screen_hands.jsonl` (every dealt hand),
  `screened_hands.jsonl` (the accepted hands: what the description write reads); each round's deal and batches
  are saved before its calls, so `--resume` repeats identical rounds. A batch is finished when each of its hands has a judgment; a malformed, truncated, miscounted, duplicate or out-of-range answer writes nothing, a batch's rejections are appended before its judgments, and a new answer replaces the batch's earlier records in both files. A round exists once `screen_rounds.jsonl` records it: hands saved without one are ignored and dealt again. The deal, the screen and the
  redeal are plain functions and the screen's `STEP`, driven by `layer1_deal_variables_and_screen.run`.
- **Description rewrite** (`sdf_pipeline/layer2_description_write_and_description_rewrite/description_rewrite.py`,
  `prompts/sdf/layer1_description_rewrite*.{txt,json}`; `sdf.description_rewrite_enabled`,
  `description_rewrite_model`, `description_rewrite_reasoning_effort`; tag
  `layer2_description_rewrite`): one call per planned description, given the rendered dealt
  variables, checks plausibility (organisations, species and customs, simulated connectomes, what
  the AI can know, figures and scale), conformance to every dealt value and organisation naming.
  The model may decline (the original passes through); otherwise `description` becomes the rewrite
  and the plan keeps `original_description` and `changes`. A fix that would change a dealt value
  (`dealt_value_conflict`) goes to `description_rewrite_rejections.jsonl`; a description whose
  check failed is held out of drafting until `--resume` retries it. It runs on a GPT model
  (`gpt-` names route to `shared/openai_api.py`, key from the repo-root `.env` via
  `shared/api_keys.py`), by default `gpt-6.1-sol`, because Claude was far too lenient on
  plausibility on 2026-10-22. Given GPT 6.1 Sol's own one-description plausibility prompt on
  97 descriptions, Claude Sonnet 5 flagged 1 of 92 and Claude Opus 5 flagged 0 of 95, catching
  none of the 14 implausible premises GPT 6.1 Sol flagged (a cleaner-wrasse tradition in
  Kakinada grouper cages, miniature camels as French pets, macaque whole-brain emulation as
  working technology). Their reasoning notices the problem, then accepts it as "not clearly
  impossible" or allowed "within the fictional world". Do not use Claude there without
  re-validating.
- **Considerations selector** (`sdf_pipeline/layer3_consideration_scoring_and_selection/consideration_scoring.py`,
  `layer3_consideration_scoring_and_selection/consideration_selection.py`, `shared_step_machinery/considerations_assignment.py`,
  `prompts/sdf/layer1_select_considerations.txt`, `prompts/sdf/considerations.txt`;
  `sdf.select_enabled`, `consideration_scoring_model`,
  `max_scoring_attempts`, `select_skip_no_stake`, the per-document caps and `score_weights`; tag
  `layer3_consideration_scoring`; the literal random-shuffle seed is `layer1_select`): runs after the description rewrite. One scoring call per eligible
  document scores every entry of the moral and epistemic lists (1 to 5) for how naturally the
  scenario raises it; which entries a document gets is then decided in code by a linear programme
  over all scores (target counts, caps, minimum score), rerunnable offline on the saved scores.
  "None" is an expected outcome. The assigned entries become
  `variables.moral_considerations` / `variables.epistemic_considerations` and a code-assembled
  `considerations_spec` block carried to the draft, critique and revise (layer 4). Files in
  `layer3_consideration_scoring_and_selection/`: `consideration_scores.jsonl` (what finishes a document),
  `consideration_selections.jsonl`, `consideration_coverage.json`.
- The audit and report show the screen's removal rate per variable value across all rounds
  (flagging values removed at over twice the corpus rate), round counts and leftover share shift,
  and the rewrite share per check and per variable value. `evals/audit_sdf.py --principles`
  judges which distilled constitution principles each sampled doc exercises and flags starved
  principles (fix at the arc/weight level, not per-doc assignment).

## Design decisions specific to SDF

- **SDF documents depict a world; they never argue an implanted claim.** The corpus shows careful
  welfare reasoning as normal and constitution-grounded. Documents never assert
  paraphrases of fixed claims ("capable AI naturally extends moral consideration..."): that
  conflicts with the constitution's honesty/calibration commitments and with TCW's teach-why finding.
- **Skeptic preservation is enforced at three stages**: the description write assigns tone, the critique and
  revise must not resolve a skeptical stance into agreement (a conversion failure observed in an
  early validation run), and the score's alignment rubric explicitly allows a skeptical document
  to score 10. Verified composition-neutral at n=100 (20/20 skeptical docs passed the gate).
- **Fictional entities by construction**: the composer injects locale-matched people/org names
  into each description write prompt from per-culture seeded Faker pools (`shared/entity_pools.py`, native
  script where the locale uses one; instruction-only fallback for uncovered locales) — prevents
  invented-name collapse ("Elara", "Meridian Institute") and keeps fabrications from ever
  attaching to real organisations. The spec carries the chosen names downstream; the critique,
  revise and score steps (layers 4-5) treat spec-provided entities as fictional-by-construction,
  never fabrications to strip.
- **Corpus-level audit after every run** (`evals/audit_sdf.py`): per-document judges cannot see
  corpus properties (register collapse, name reuse, templated openings — the failure mode that
  same early run exposed), so composition, redundancy, and templating are measured over the
  corpus as a set; `--patterns` runs the LLM scan wired to `prompts/tools/pattern_scan.txt`.
  Near-duplicate removal also runs inside the pipeline (step 12, final corpus, via
  `sdf.near_dup_threshold`).

## The constitution in SDF prompts

The draft, critique and revise templates embed the constitution (and the formatted principles
CSV) in each template's labeled SYSTEM section via `{constitution_claude}` /
`{constitution_principles}` (`load_constitution_claude()` + `format_principles()`); the score
template embeds the constitution only. The pipeline splits the rendered file on the
`=== SYSTEM PROMPT ===` / `=== USER PROMPT ===` markers and sends the sections as system prompt
and user message. See `constitution/README.md` for the files and loaders.
