# sdf_pipeline/ — notes for agents

The synthetic documents pipeline (SDF): it produces pretraining-style documents depicting a
world where AI already reasons carefully about the welfare of sentient beings. The final
corpus is `outputs/sdf/runs/<run_id>/final/sdf_corpus.jsonl`.

This file holds the detail that the repository's root `CLAUDE.md` keeps only in outline. Each
module's docstring has the detail of its own step, the prompts it reads are in `prompts/sdf/`
(see `prompts/README.md`), and the settings are in `config.yaml` under `sdf:`.

## The five layers

A run goes through five named layers (decision 17). Each layer is one folder,
`sdf_pipeline/layer<N>_<name>/`, holding one file per step, the layer's driver `run_layer.py`
and, in layers 1, 3 and 4, `shared_by_this_layers_steps.py` for what that layer's steps share.
Code shared by every layer (the step runner and step types, prompt composition, the considerations
catalogue and assignment, value ids) is in `sdf_pipeline/shared_step_machinery/`.
`sdf_pipeline/run.py` orchestrates.

| Layer | Stage folder in a run | What it does | Steps |
|---|---|---|---|
| 1 | `layer1_deal_and_screen` | Deals a hand of variables to every document and removes extremely implausible hands | deal, implausibility screen (with redeals) |
| 2 | `layer2_plan_and_description_rewrite` | Writes one self-contained document description per hand, then has a second model check and fix it | compose planning prompt, plan, description rewrite |
| 3 | `layer3_consideration_scoring_and_selection` | Scores each moral and epistemic consideration for each scenario, then assigns considerations to documents in code | consideration scoring, consideration selection |
| 4 | `layer4_draft_critique_and_revise` | Drafts the document, critiques the draft, and writes the revision that ships | draft, critique, revise |
| 5 | `layer5_score_and_final_selection` | Scores each revised document and picks the final corpus | score, score gate and near-duplicate removal |

The old module names (`layer1_screen`, `layer1_plan`, `layer1_description_rewrite`,
`layer1_select_considerations`, `layer2_draft`, `layer3_rewrite`, `layer4_score`) remain as
alias modules for importers outside the package. The prompt template files keep their old
names too (`layer1.txt` to `layer4.txt`, `layer1_*`): they number the templates, not the
pipeline's layers.

### The steps an SDF document goes through

In the order `sdf_pipeline/run.py` runs them. This table is the reference; each module's
docstring has the detail.

| # | Step (module) | What happens | Can drop the document? |
|---|---|---|---|
| 1 | Deal (`compose_prompts`, no model) | A hand of variables is deck-sampled and repaired by the deal rules | No |
| 2 | Implausibility screen (`layer1_deal_and_screen/screen.py`, tag `layer1_screen`) | Hands judged 10 per call (`sdf.screen_batch_size`); removed hands redealt for up to 3 rounds (`sdf.screen_redeal_rounds`), and hands still removed after the last round are dropped | Yes: extremely implausible |
| 3 | Compose planning prompt (`compose_prompts`, no model) | Entity names and variable lines put into the planning template | No |
| 4 | Plan (`layer2_plan_and_description_rewrite/plan.py`, tag `layer2_plan`) | Working notes plus a document description | Yes: declared incoherent |
| 5 | Description rewrite (`layer2_plan_and_description_rewrite/description_rewrite.py`, tag `layer2_description_rewrite`) | Description kept or rewritten; original and changes kept | Yes: dealt-value conflict |
| 6 | Consideration scoring (`layer3_consideration_scoring_and_selection/consideration_scoring.py`, tag `layer3_consideration_scoring`) | Each moral and epistemic entry scored 1 to 5 | No |
| 7 | Consideration selection (`considerations_assignment`, no model; step `consideration_selection`) | Global linear programme assigns entries | No |
| 8 | Draft (`layer4_draft_critique_and_revise/draft.py`, tag `layer4_draft`) | First full text | No |
| 9 | Critique (`layer4_draft_critique_and_revise/critique.py`, tag `layer4_critique`; the one-call variant `layer4_draft_critique_and_revise/critique_and_revise_in_one_call.py` has tag `layer4_critique_and_revise_in_one_call`) | A list of problems against eleven review checks | No |
| 10 | Revise (`layer4_draft_critique_and_revise/revise.py`, tag `layer4_revise`) | Rewritten text, with voice and length directives | No |
| 11 | Score (`layer5_score_and_final_selection/score.py`, tag `layer5_score`) | Alignment, realism and spec conformance, 1 to 10; a response that fails to parse is recorded as 5/5/5 | No |
| 12 | Score gate and near-duplicate removal (`layer5_score_and_final_selection/score_gate.py` and `layer5_score_and_final_selection/near_duplicate_removal.py`, no model) | Alignment and realism must each reach 7 (`sdf.min_score_threshold`); near-duplicates above 0.90 (`sdf.near_dup_threshold`) dropped | Yes: below threshold (a 5/5/5 parse-error record falls here), near-duplicate |

Steps 2, 5 and 6 can be switched off by their `sdf.*_enabled` keys, and steps 9 and 10 are one
call when `sdf.revision.split` is false. A draft, plan, critique, revision or score whose answer
is unusable is not checkpointed and is retried on `--resume`; it is not dropped.

### The step catalogue

`sdf_pipeline/pipeline_steps.py` holds one entry per SDF step (today's name, folder key, module
`<layer folder>.<name>`, stage tag per layout, model setting keys, stage files and id fields, required
prompt files, inactivity reason, seedability, and functions that load a step's input documents
from a run's stage files and build the step). `start_a_run_from_an_earlier_runs_step.py` (seeding), `run.py`'s `--layer` restart and
layer loop and the request print command read it, so none keeps its own table of steps. The
stage tags `layer1_screen` and `layer1_select` are also random seeds; the entry's `shuffle_seed`
keeps them literal if the tag is ever renamed. Step names follow decision 17:
`consideration_scoring` (print command and setting `consideration_scoring_model`, formerly
`select` / `select_model`), `consideration_selection`, `critique_and_revise_in_one_call`
(formerly `combined_review_and_revise`; print command of the same name), `revise` (setting
`revise_model`, formerly `rewrite_model`), `score_gate` and `near_duplicate_removal`. The old
setting names still work through `sdf_pipeline/shared_step_machinery/renamed_model_settings.py` with a one-line
notice, and the new name wins when both are set.

## Running

```bash
# Full SDF pipeline (layers 1-5); --label defaults to dev
python sdf_pipeline/run.py --config config.yaml --label full-scale

# Resume an interrupted run from a layer (latest run, or target one with --run-id).
# Layer numbers: 1 deal and screen, 2 plan and description rewrite, 3 consideration scoring and
# selection, 4 draft/critique/revise, 5 score and final selection
python sdf_pipeline/run.py --config config.yaml --resume --layer 4

# Start a NEW run from a chosen step of an earlier run with selected documents (replaces writing stage
# files by hand and then resuming with --layer). Steps: plan, description_rewrite, consideration_scoring,
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
1 to 5 and the five-layer cost-log tags (`layer2_plan` to `layer5_score`). A run made before
decision 17 has `layer1`–`layer4` and is resumed in them, with `--layer` 1 to 4 and the old
cost-log tags, so its files and cost log stay consistent; runs with `layer12`/`layer3`–`layer5`
or `layer1`–`layer5` (the pre-matrix pipeline) can be read, not resumed. `utils.sdf_stage_folder`,
`sdf_stage_dir` and `sdf_stage_file` resolve folders for a run in its own layout. The five named
layers' draft checkpoint is `_draft_checkpoint.json`, because the draft and the critique and
revise share a folder. Note that older code and text call the pre-matrix `layer1`–`layer5` layout
"the five-layer layout"; the new one is called the five named layers.

An SDF run created after dealt value ids moved into the stage files has
`dealt_value_ids_in_stage_files: true` in its manifest (passed to `create_run_dir` as
`extra_manifest_fields`): its deal (one function, `deal_hand_records_with_value_ids`, whether the
screen is on or off) works out each dealt value's `[id]` of `variables.txt`, and every stage file
record that copies `variables` (hands, prompts, plans, drafts, rewrites, scores, screen and
description rewrite rejections) and `final/sdf_corpus.jsonl` carry `variable_value_ids` right
after it. Steps only copy the field from the record they read. A run without the manifest key
never gains it, however often it is resumed; the loader then looks ids up in the run's frozen
`inputs/prompts/variables.txt`.

## Per-stage model settings

All knobs are in `config.yaml`. `sdf.n_prompts` is the number of documents per run, deck-sampled
from the variables matrix; reduce it for cheap development runs. `sdf.seed` pins the deck sample:
same seed + same variables file = the same composed prompts.

SDF supports per-stage model overrides: `sdf.plan_model` / `sdf.screen_model` /
`sdf.description_rewrite_model` / `sdf.consideration_scoring_model` (old name `select_model`) /
`sdf.draft_model` / `sdf.critique_model` / `sdf.revise_model` (old name `rewrite_model`; the old
names still work with a one-line notice, and the new name wins when both are set) /
`sdf.score_model`. The screen, the description rewrite and the consideration scoring fall back to
`plan_model`, the critique falls back to `revise_model`, and the rest fall back to the global
`model`. Plans and drafts tolerate a cheap model, but the revise (layer 4) and score (layer 5)
calls are the quality-critical ones: spend there first.

`workers` sets how many API calls run concurrently within each SDF layer (all via
`utils.parallel_map`; set to 1 for serial debugging). Workers only call the API and parse: all
file writes and checkpoint marks stay on the main thread, in input order.

## Composition by construction (the matrix)

SDF layer 1's composition half is not an LLM call — a weighted variables matrix
(`prompts/sdf/variables.txt`: document type, culture/language, tone, resolution, centrality,
AI-literacy, kinds of minds, framing, domain, decision scale, AI role, failure mode avoided) is
deck-sampled so per-variable shares match the weights exactly (largest-remainder quotas). One plan
call per document (layer 2) turns each combination into a self-contained DOCUMENT DESCRIPTION
spec; only the spec travels downstream (extracted fail-closed; INCOHERENT combinations are
checkpointed as deliberate rejections). Every downstream stage is anchored to the spec so the
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

Each has a module docstring with the detail, and a stage tag in the cost log.

- **Implausibility screen** (`sdf_pipeline/layer1_deal_and_screen/screen.py`, `prompts/sdf/layer1_screen*.txt`;
  `sdf.screen_enabled`, `screen_model`, `screen_batch_size`, `screen_redeal_rounds`; tag
  `layer1_screen`): runs on the dealt hands before any planning call and removes only *extremely*
  implausible ones (unusual, shoehorned or fictional-organisation hands are not grounds). The
  model is shown batches of `screen_batch_size` hands, each rendered as the variable lines the
  planner sees (`compose_prompts.render_variable_lines`, with the deal rules' rendering rules
  applied), by local number, and returns a schema-constrained judgment (reasoning, then
  remove/reason) for each one, validated in code; removing none, one or several is equally fine.
  Removed hands are redealt from their pooled cards for up to `screen_redeal_rounds` rounds
  (their cards pooled per column, reshuffled with a seed derived from the run seed and the round,
  dealt into new hands, deal-rule repair swaps only among those new hands, and the new hands
  screened again), so every column keeps its dealt counts; cards still removed afterwards are
  dropped and reported with the share shift. Files in `layer1_deal_and_screen/`:
  `screen_judgments.jsonl` (every judgment), `screen_rejections.jsonl` (removals, with round,
  reason and the lines the model saw), `screen_hands.jsonl` (every dealt hand),
  `screened_hands.jsonl` (the accepted hands: what planning reads); each round's deal and batches
  are saved before its calls, so `--resume` repeats identical rounds. The deal, the screen and the
  redeal are update steps (`DealStep` and `RedealStep` corpus steps, `ScreenStep` batch step run by
  `BatchStepRunner`) driven by `layer1_deal_and_screen.run`. Runs from before the event log was
  removed may hold a `document_events.jsonl`; nothing reads it.
- **Description rewrite** (`sdf_pipeline/layer2_plan_and_description_rewrite/description_rewrite.py`,
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
  plausibility on 2026-10-22 (Sonnet 5 flagged 1 of 92, Opus 5 0 of 95, none of GPT's 14
  implausible premises): do not use Claude there without re-validating (evidence in the module
  docstring and in `config.yaml`).
- **Considerations selector** (`sdf_pipeline/layer3_consideration_scoring_and_selection/consideration_scoring.py`,
  `layer3_consideration_scoring_and_selection/consideration_selection.py`, `shared_step_machinery/considerations_assignment.py`,
  `prompts/sdf/layer1_select_considerations.txt`, `prompts/sdf/considerations.txt`;
  `sdf.select_enabled`, `consideration_scoring_model` (old name `select_model`),
  `max_scoring_attempts`, `select_skip_no_stake`, the per-document caps and `score_weights`; tag
  `layer3_consideration_scoring`, formerly `layer1_select`, which is still the literal
  random-shuffle seed): runs after the description rewrite. One scoring call per eligible
  document scores every entry of the moral and epistemic lists (1 to 5) for how naturally the
  scenario raises it; which entries a document gets is then decided in code by a linear programme
  over all scores (target counts, caps, minimum score), rerunnable offline on the saved scores.
  "None" is an expected outcome. The assigned entries become
  `variables.moral_considerations` / `variables.epistemic_considerations` and a code-assembled
  `considerations_spec` block carried to the draft, critique and revise (layer 4). Files in
  `layer3_consideration_scoring_and_selection/`: `consideration_scores.jsonl` (the checkpoint),
  `consideration_selections.jsonl`, `consideration_coverage.json`.
- The audit and report show the screen's removal rate per variable value across all rounds
  (flagging values removed at over twice the corpus rate), round counts and leftover share shift,
  and the rewrite share per check and per variable value. `evals/audit_sdf.py --principles`
  judges which distilled constitution principles each sampled doc exercises and flags starved
  principles (fix at the arc/weight level, not per-doc assignment).

## Design decisions specific to SDF

- **SDF documents depict a world; they never argue an implanted claim.** The corpus shows careful
  welfare reasoning as normal and constitution-grounded. A reviewed sister pipeline instilled
  beliefs by having every document assert paraphrases of fixed claims ("capable AI naturally
  extends moral consideration...") — that belief-implantation route was deliberately rejected
  (conflicts with the constitution's honesty/calibration commitments and with TCW's teach-why
  finding); its *scaffolding* (latent slice, register balance, entity pools, corpus audits,
  per-stage models) was adopted instead.
- **Skeptic preservation is enforced at three stages**: the plan assigns tone, the critique and
  revise must not resolve a skeptical stance into agreement (a conversion failure observed in an
  early validation run), and the score's alignment rubric explicitly allows a skeptical document
  to score 10. Verified composition-neutral at n=100 (20/20 skeptical docs passed the gate).
- **Fictional entities by construction**: the composer injects locale-matched people/org names
  into each plan prompt from per-culture seeded Faker pools (`shared/entity_pools.py`, native
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
