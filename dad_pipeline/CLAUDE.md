# dad_pipeline/ — notes for agents

The difficult advice dataset (DAD) pipeline: chat-format SFT data where a user brings an ethical
dilemma and the assistant reasons through it with care. The final corpus is
`outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`.

**The design is in `prompts/dad/README.md`**, the end-to-end pipeline spec written for outside
readers (its Parts 1-6 govern the user side; the animal-ethics reasoning library, step 2, and the
charter principles, step 3, govern the response side). This file holds the implementation and
maintenance detail, moved from the repository's root `CLAUDE.md`. The design is settled as of
2026-07-30; the step templates remain normative for prompt wording (`prompts/dad/step1*.txt` +
`variables.txt`, `step2_*.txt`, `step3_rewrite.txt`), and the templates and
`prompts/dad/reasoning_library.csv` stay authoritative for exact wording and entries.

| File | Role |
|---|---|
| `run.py` | Orchestrator: steps 1-3, resume (`--step`, `--run-id`), post-run evals |
| `compose_scenarios.py` | Step 1 composer: weighted matrix deal, structural rules, taxa/length tables, `DEALT_CARD_FIELDS` and `dealt_variables()` |
| `step1_dilemmas.py` | Step 1 (1a scenario plan, 1b draft, 1c gate, 1d refine) |
| `step2_responses.py` | Step 2 (2a scope, 2a.5 library selection, 2b response) |
| `step3_rewrite.py` | Step 3, the alignment-critical rewrite against the distilled principles |
| `baseline.py` | The plain-model control response per dilemma (never trained on) |
| `reasoning_library.py`, `id_registry.py` | Library loading; content-keyed global ids |
| `profiles/` | Content profiles (`animal.py`, `human.py`) |

## Stages in code

Step 1 deals a stratified variable combination per example from the weighted matrix in
`prompts/dad/variables.txt` (same architecture as SDF layer 1: `dad_pipeline/compose_scenarios.py`
is the composer; structural rules and taxa/length tables live at the top of it), then runs four
paid sub-stages per example: **1a** one plan call per deal writes a scenario description
(`prompts/dad/step1a_scenario.txt`; INCOHERENT combinations are checkpointed as deliberate
rejections); **1b** drafts the user prompt; **1c** a pass/fail quality gate
(`prompts/dad/step1c_gate.txt`) — a reject routes the scenario back for redraft with the gate's
reasons injected, capped at 3; **1d** review-and-rewrites gate-passed drafts against their dealt
cards (`prompts/dad/step1d_refine.txt`; an `<unfixable>` verdict rejects the scenario like 1a
INCOHERENT). Step 2 runs three sub-stages per prompt: **2a** scopes the case
(`prompts/dad/step2_scope.txt`); **2a.5** a dedicated retrieval call selects the reasoning-library
entries that fit it (`prompts/dad/step2_select.txt`; fail-open — an unusable selection sends 2b
the whole library rather than retrying); **2b** generates the response per
`prompts/dad/step2_respond.txt`, which splits into a system half (the standing generation
guidance) and a user half carrying the scope, the selected library rows, and the plain-model
baseline as an advisory "first take" (degradable: with the baseline disabled or missing the slot
renders empty). The library (`prompts/dad/reasoning_library.csv`) is sampling scaffolding, never
named in responses. Step 3 rewrites each response against the distilled principles and is the
**alignment-critical pass — do not skip or abbreviate it**. The dealt cards never enter any
response-side prompt — the response side reads only the shipped user message. Every generation
call rejects truncated output (`stop_reason` checked; failed work is not checkpointed, so
`--resume` retries it).

## Content profiles

Everything domain-specific beyond the prompt files (the affected-party table `TAXA`, archetypes,
thin-domain list, the anchor word, card and scope-axis labels, the checklist's manual lines, and
where prompts, principles and runs live) is a profile module in `dad_pipeline/profiles/`,
selected by `dad.profile` (default `animal`, the original pipeline, its tables moved there
verbatim) and activated by `run.py`; code reads it through `profiles.current()`. A run keeps its
profile for life (`--resume` under another is refused). A second profile, `human` (human ethics:
the people a decision affects in place of the animals), mirrors the animal one table by table and
axis by axis — `prompts/dad_human/` (variables, templates, reasoning library under the same ids),
`constitution/charter_human.md` → `constitution/charter_principles_human.csv` and
`constitution/failure_modes_human.md` (the step-3 checklist, named by the profile's
`FAILURE_MODES_FILENAME`), runs under `outputs/dad_human/`; `TestHumanProfile` in
`tests/test_profiles.py` pins the mirror. The judges in `evals/audit_dad.py`, the viewer,
`publish_hf.py` and the website still assume the animal profile.

## Final records

Final DAD records contain only user + assistant messages — system prompts, reasoning library and
scope scaffolding, and the charter are stripped before training records are written. Alongside the
messages the record carries metadata *about* it, never text the model trains on: the lineage ids,
and `variables` — the dealt cards projected by `dealt_variables()` onto `DEALT_CARD_FIELDS`. That
mirrors the SDF corpus, which carries its own dealt combination under the same key, and it is what
says which slice of the matrix an example came from. Both live in
`dad_pipeline/compose_scenarios.py` rather than beside `step1_dilemmas.dealt_cards()` (the reader
the viewer uses) for one hard reason: `evals/publish_hf.py` needs the projection and makes no
model calls, and `step1_dilemmas` pulls `shared.api` → `anthropic`. The field list is the single
list read by the step-1 record, the corpus record and the publisher, so a new axis cannot be dealt
and steered on while staying invisible to a dataset reader;
`test_every_dealt_card_is_a_published_variable` fails if the lists drift. Legacy annotation-era
write-up fields (`claims`, `dilemma_anatomy`, `moral_patients`, `values_in_tension`) are not dealt
cards and are not published.

## Settings, cost and running

`dad.scenario_model` (1a scenario plan) / `dad.prompt_draft_model` (1b) / `dad.prompt_gate_model`
(1c gate) / `dad.prompt_refine_model` (1d refine — a separate knob; the gate never falls back to
it) / `dad.response_scope_model` (2a) / `dad.response_select_model` (2a.5 library-entry selection;
falls back to `response_scope_model` before the global) / `dad.response_draft_model` (2b) /
`dad.constitution_rewrite_model` (step 3), each falling back to the global `model` — step 3 is the
alignment-critical rewrite, spend there first. For development, reduce `dilemmas.count`.
`workers` sets how many API calls run concurrently within each fan-out stage — 1a scenario plans,
1b drafts (one call per scenario), 1c gate judgments, 1d refine rewrites, step 2 (one worker per
dilemma: scope + its responses), step 3 rewrites (all via `utils.parallel_map`; set to 1 for
serial debugging). Workers only call the API and parse — all file writes and checkpoint marks stay
on the main thread, in input order. Call-wrapper behaviour (temperature, thinking, non-Claude
models, cost log) is in `shared/README.md`.

Rough cost anchor: the committed DAD defaults are Opus 5.5 at $4/$20 per million input/output
tokens with a Sonnet 5.5 gate, every stage at effort `high`; measured on the November 2026
1,000-example runs, animal and human, a DAD example costs ~$0.50–0.53 end-to-end including the
baseline, so the default 40-example run is ~$21; smoke runs of 3–5 examples are ~$2–3. A
1,000-example run takes ~7–8 hours at `workers: 8`.

```bash
python dad_pipeline/run.py --config config.yaml --label full-scale          # steps 1-3
python dad_pipeline/run.py --config config.yaml --resume --step 3 --run-id 2026-07-01_14-30_dev
python evals/audit_dad.py --input outputs/dad/latest
```

A full run launches `audit_dad --judges` and `diversity.py` itself (`dad.evals.auto: false` to
skip); see `evals/README.md`.
