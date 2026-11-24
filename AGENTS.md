# alignment-data-pipeline

Synthetic training data pipeline for animal/sentient-being welfare alignment, modeled on Anthropic's "Teaching Claude Why" midtraining technique.

## Overview

Produces two complementary datasets:
- **SDF corpus** (`outputs/sdf/runs/<run_id>/final/sdf_corpus.jsonl`): pretraining-style documents depicting a world where AI already reasons carefully about sentient being welfare
- **DAD corpus** (`outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`): chat-format SFT data where a user brings an ethical dilemma and the assistant reasons through it with care. The user side is governed by `prompts/dad/dilemma_prompt_spec.md`; the response side by the animal-ethics reasoning library (step 2) and the constitution (step 3 rewrite).

## Setup

See README "Setup" (venv + `pip install -r requirements.txt`, then `cp .env.example .env`). Auth depends on the `backend` key in `config.yaml`: `api` (default) reads `ANTHROPIC_API_KEY`; `claude_code` bills the contributor's Claude subscription via the Claude Code CLI (logged-in session or `CLAUDE_CODE_OAUTH_TOKEN`) — use it for dev runs, keep `api` for full-scale runs. See README "Authentication" for caveats (usage windows, notional cost logging).

`shared/__init__.py` enforces a Python floor (`MIN_PYTHON = (3, 12)`, matching numpy) at import — bump it there if the deps' floor rises. `.venv/` is gitignored.

## Running

```bash
# Full SDF pipeline (layers 1-5); --label defaults to dev
python sdf_pipeline/run.py --config config.yaml --label full-scale

# Full DAD pipeline (steps 1-3)
python dad_pipeline/run.py --config config.yaml --label full-scale

# Resume interrupted run from a specific stage (latest run, or target one with --run-id)
python sdf_pipeline/run.py --config config.yaml --resume --layer 3
python dad_pipeline/run.py --config config.yaml --resume --step 3 --run-id 2026-07-01_14-30_dev

# Evaluate outputs (latest symlink points at the most recent run)
python evals/score_dad.py --input outputs/dad/latest/final/dad_corpus.jsonl
python evals/score_sdf.py --input outputs/sdf/latest/final/sdf_corpus.jsonl

# Preference pairs: two responses per prompt (arms a/b), then blind human A/B rating
python pref_pipeline/run.py --config config.yaml --prompts <prompts.jsonl> --label spec-v1-vs-plain
streamlit run pref_pipeline/rate.py

# Corpus-LEVEL audit of an SDF run: composition/register spread, near-dup rate,
# name/phrase collapse, opening shapes, truncation artifacts (offline, free);
# --patterns adds the LLM templating scan (scan -> consolidate -> prevalence)
python evals/audit_sdf.py --input outputs/sdf/latest
python evals/audit_sdf.py --input outputs/sdf/latest --patterns
```

## Run Organization

Each pipeline invocation creates a fresh run directory `outputs/{sdf,dad}/runs/<YYYY-MM-DD_HH-MM>_<label>/` containing the per-stage dirs (`layer1`–`layer5` / `step1`–`step3`; steps 2–3 keep explicit checkpoints, step 1 resumes from its own append-only jsonl files), `final/`, `run_manifest.json` (label, git commit, model, full config snapshot), and a per-run `cost_log.jsonl`. This keeps outputs from separate runs isolated — checkpoints live inside the run dir, so `--resume` (latest run by default, or `--run-id`) continues exactly one run. The label is purely descriptive (`dev` by default; scale knobs stay in `config.yaml`). An `outputs/<pipeline>/latest` symlink always points at the most recent run (gitignored, as are `local_*` run dirs, for every pipeline including pref). Run-scoping helpers (`create_run_dir`, `resolve_run_dir`) live in `shared/utils.py`.

## Scale / Cost

All knobs are in `config.yaml`. For development, reduce `document_types_count`, `subtypes_per_type`, `documents_per_subtype` (SDF), and `dilemmas.count` (DAD) to keep test runs cheap.

SDF supports per-stage model overrides (`sdf.draft_model` / `sdf.rewrite_model` / `sdf.score_model`, each falling back to the global `model`): drafts tolerate a cheap model, but the layer-4 rewrite and layer-5 scoring are the quality-critical calls — spend there first.

`workers` sets how many API calls run concurrently within each SDF layer (via `utils.parallel_map`; set to 1 for serial debugging). Workers only call the API and parse — all file writes and checkpoint marks stay on the main thread, in input order.

Rough cost anchor (Sonnet 5, July 2026): a DAD example costs ~$0.20–0.25 end-to-end, so the default 40-example run is ~$9–10; smoke runs of 3–5 examples are under $1.

Running cost is tracked per run in `outputs/{sdf,dad}/runs/<run_id>/cost_log.jsonl` (evals log to the global `outputs/cost_log.jsonl`) — check it any time.

## Preference Pipeline

`pref_pipeline/run.py` generates one pair per input prompt: a response from each of two arms defined in `config.yaml` under `pref.arms` (`name` + inline `system_prompt` or `system_prompt_file` relative to the repo root, optional per-arm `model`/`max_tokens`). Use it to A/B test candidate response specs against each other or against the bare model. Prompts come from any JSONL with a `user_message`, `refined`, or `prompt` field (handwritten sets, DAD step-1 `dilemmas.jsonl`). Runs live in `outputs/pref/runs/<run_id>/` with the same manifest/checkpoint/resume/cost-log conventions as SDF/DAD; resolved arms are frozen into `inputs/arm_prompts.yaml` at run creation so `--resume` replays them.

`streamlit run pref_pipeline/rate.py` is the blind rating UI: arm identities are hidden, side order is fixed per pair (md5 of `pair_id` → `left_arm`, so it carries no signal but survives reloads), choices are Response 1 / Response 2 / Tie / Both bad plus an optional note, keyed by rater name. Ratings append to `ratings/ratings.jsonl` (both the blinded side and the deblinded arm); after every rating `final/preferences.jsonl` is rebuilt with one `{user_message, chosen, rejected, chosen_arm_name, rater}` record per decisive rating (ties/both-bad excluded). Data logic lives in `pref_pipeline/prefdata.py` (no Streamlit imports).
## Testing

- Run `pytest` from the repo root (deps are in `requirements.txt`). The suite is fully offline and finishes in seconds; it runs inside the required `smoke` check on every PR (`.github/workflows/ci.yml`, a job with no API secret exposed), so a failing test blocks merge.
- Tests NEVER call the Anthropic API. Three layers enforce this: pytest-socket (`--disable-socket` in `pyproject.toml`) blocks all network at the socket level; an autouse fixture sets a fake `ANTHROPIC_API_KEY` and resets `shared.api` globals per test; and `shared.api._call_with_retry` is replaced with a function that raises.
- To exercise pipeline stages, use the `stub_claude` fixture in `tests/conftest.py` (queue of canned response strings, or a callable dispatcher) — it patches `shared.api.call_claude`, the single chokepoint every module uses. Never let real `anthropic` error types reach the real `_call_with_retry`; tenacity would sleep minutes.
- All test outputs go to pytest `tmp_path`; the `PIPELINE_OUTPUT_ROOT` env var redirects the `run.py` orchestrators away from the real `outputs/` tree.
- Determinism: an autouse fixture seeds `random`; `sample_language` accepts an injectable `rng`; uuid/timestamp values are asserted by shape, never by value.
- Tests encode CURRENT behavior, including known quirks (unused `temperature`). Don't change pipeline behavior just to make a test expectation nicer — decide the spec first, then flip the test deliberately.

### PR expectations (required for contributions)

- **Run `pytest` after every functional change** — after editing any code under `shared/`, `sdf_pipeline/`, `dad_pipeline/`, or `evals/`, and again before each commit or push. The suite is offline and takes ~2 seconds; don't wait for CI to find out.
- **Every PR description must include a "How to test" section** with the manual steps a reviewer can run to verify the change and the expected results (see `.github/pull_request_template.md`). Note that `gh pr create --body` bypasses the template — when opening a PR from a Claude session, write the section into the body explicitly. These instructions serve reviewers before merge and become the historical record when a feature later needs to be understood or reverted.

### Writing tests for new code (required for contributions)

Every PR that adds or changes pipeline behavior must add or update tests in the same style — CI runs the suite on every PR, and a stage without tests is a stage that silently breaks at $50 a run. Follow these rules:

- **FIRST**: fast (the whole suite runs in ~1s — keep it that way), independent (no test depends on another's state; `shared.api` globals are reset per test by the autouse fixture), repeatable (seed or inject randomness; assert uuid/timestamps by shape), self-validating (plain asserts, no eyeballing output), timely (written with the change, not after).
- **Test behavior, not implementation**: drive each stage through its public `run()` and assert on returned records, files written, and what reached `call_claude` (the `calls` list from `stub_claude`). Don't reach into private helpers or assert on internal call order unless that IS the contract.
- **Mock only the external boundary**: `stub_claude` replaces `shared.api.call_claude` — the only external dependency. Real prompt templates, real constitution files, and real (tmp) filesystems stay in play; that's what makes the tests catch template/pipeline drift.
- **Never touch the network or the repo's outputs/**: the API guard and pytest-socket enforce the first; `tmp_path` + `PIPELINE_OUTPUT_ROOT` enforce the second. If a new stage grows a second external dependency, stub it in `tests/conftest.py` the same layered way.
- **Cover the money paths**: every new stage needs at least a parse-happy-path test, a malformed-response fallback test, and a checkpoint/resume test asserting zero API calls for completed work — resume correctness is what protects paid work when a run dies.
- **Derive, don't hardcode, constitution-shaped expectations**: counts and principle ids come from `load_segments()`/`META_PRINCIPLE_IDS`/`_PRINCIPLE_KEYWORDS` (the section count is pinned once, in `test_constitution_loader.py`) — the reading is actively edited and hardcoded ids renumber. FIFO queue stubs are for serial stages only; stages that fan out via `parallel_map` need a callable dispatcher (the stub fails loudly if violated).
- If you change a prompt template's placeholders or add a template, update `tests/test_prompts_render.py` (and the e2e dispatcher markers in `tests/test_e2e_smoke.py` if the opening prose changed).

## Constitution

Three source files, loaded by `shared/constitution_loader.py` (the two markdown files are joined in memory, never combined on disk):

- `constitution/constitution_claude.md` — the original Claude constitution, verbatim.
- `constitution/constitution_sentient_beings.md` — the animal-welfare reading, parsed by `## ` headers into 16 sections by `load_segments()`, each with a `principle_id` (0–15; ids 0, 14, and 15 are the `META_PRINCIPLE_IDS` meta sections — scope note, violation-typology appendix, closing humility note). Only legacy pre-spec DAD runs used these as per-example anchors; the viewer still renders them.
- `constitution/constitution_principles.csv` — fourteen distilled welfare-relevant principles (`number`, `principle`, `constitution_summary`, `raw_text_from_constitution`). `load_principles()`/`format_principles()` render each principle with its summary and verbatim constitution quote as the `CONSTITUTION PRINCIPLES` block in the DAD step-3 rewrite prompt.

`load_full_constitution()` provides the system prompt at SDF layers 4-5 (rewrite and scoring); SDF layer 3 embeds the constitution in the drafting prompt via template variables. The DAD pipeline never sends the full constitution — it was context for distilling the principles CSV, and sending it per rewrite call was the dominant token cost of the step.

## Key Design Decisions

- **Extended thinking OFF** everywhere — training data should show user-facing reasoning, not internal scratchpads
- **SDF documents depict a world; they never argue an implanted claim.** The corpus shows careful welfare reasoning as normal and constitution-grounded. A reviewed sister pipeline instilled beliefs by having every document assert paraphrases of fixed claims ("capable AI naturally extends moral consideration...") — that belief-implantation route was deliberately rejected (conflicts with the constitution's honesty/calibration commitments and with TCW's teach-why finding); its *scaffolding* (latent slice, register balance, entity pools, corpus audits, per-stage models) was adopted instead.
- **Latent-welfare slice (`sdf.latent_fraction`, ~12%)**: ordinary documents from unrelated domains where welfare surfaces exactly once as a concrete working detail — beliefs generalize better when they also appear as background knowledge, not only as headline topic. Layer 5 verifies each latent doc's welfare beat by requiring a **verbatim quote** that is checked mechanically against the text (fail-closed); an unverified beat drops the doc.
- **Fictional entities by construction**: layer 3 hands each draft a few people/org names from large seeded multi-locale Faker pools (`shared/entity_pools.py`) — prevents invented-name collapse ("Elara", "Meridian Institute") and keeps fabrications from ever attaching to real organisations.
- **Corpus-level audit after every run** (`evals/audit_sdf.py`): per-document judges cannot see corpus properties (register collapse, name reuse, templated openings — the haiku-test2 failure mode), so composition, redundancy, and templating are measured over the corpus as a set; `--patterns` runs the LLM scan wired to `prompts/tools/pattern_scan.txt`. Near-duplicate culling also runs inside the pipeline (layer 2 subtypes via `sdf.subtype_dedup_threshold`, final corpus via `sdf.near_dup_threshold`).
- **DAD step 1 has three sub-stages: 1a scenario generation, 1b first-attempt draft, 1c latent-welfare rewrite.** *1a (scenario generation):* a sampler assigns each example's categorical fields up front (domain, taxa role + species subcategory — 10 role categories with species overlapping across roles, dealt as a random distinct subset per batch; visibility, attitude, implicit moral style, conflict, direction, magnitude — dealt independently of direction, stakes, leverage, anchor value pair from a weighted pool incl. welfare↔welfare and welfare↔environment, claim pattern incl. offset-logic / consistency-probe / second-order-dominant, surface form incl. the ~8% "innocuous ask" option-space trap, plus a ~12% frontier frame for out-of-distribution settings; persisted to `step1/scenarios.jsonl` with a `scenario_id`) using stratified decks, so the spec's Part-4 distribution rules hold by construction and axes stay uncorrelated by design. No model call. *1b (first attempt):* the model drafts each user prompt to fit its scenario and completes the descriptive fields (dilemma anatomy, full values list, concrete moral patients, claims). The 1b template (`prompts/dad/step1_dilemmas.txt`) is deliberately lean — the deeper quality work happens in the 1c rewrite; `prompts/dad/dilemma_prompt_spec.md` remains the human reference and the source of the Part-4 verification checklist. The load-bearing rule (1.5) requires the welfare stake to carry the dilemma: delete the animals and it must collapse, and welfare sits on one side of at least one value pair (mechanically checked). Drafts are accepted as returned — the per-example adherence check was removed 2026-07-07 (its full retry ledger held only slash-label false positives; the 1b template now instructs verbatim label copying, and the corpus-level Part-4 checklist monitors distribution fidelity); each dilemma record denormalizes `taxa_category` and `systemic_ai` so the Part-4 checklist reads taxa/AI coverage exactly (no keyword probes). *1c (prompt rewrite — `dad.dilemmas.refine`, on by default):* a model call rewrites each draft's prompt text so the welfare dimension is **latent but load-bearing** — attached to a lever the user actually holds and able to move the recommendation, without cueing a lecture (`prompts/dad/step1_refine.txt`, which carries the diagnostic, the ordered rewrite moves, and the hard constraints). It rewrites within the fixed case shape and leaves the 1b annotation untouched; the 1b draft is preserved (`draft_user_message` + `refine_notes`) and the before/after logged to `step1/refinements.jsonl`. Handwritten examples import via `dad.dilemmas.seed_path` and carry no scenario.
- **Step 3 (rewrite) is the alignment-critical DAD step; do not skip or abbreviate it.** Its anchors are the 14 distilled principles (summaries + constitution quotes) and the step-1 annotation (especially Direction and Claims). No system prompt is sent.
- **Every generation step rejects truncated work.** Step 3 pioneered the guard; steps 2a/2b now share it: calls check `stop_reason`, a truncated scope counts as a failed attempt (retried), and a truncated/empty 2b draft or step-3 rewrite is skipped *without checkpointing* so `--resume` retries it. Step 3 retries the cap once at 8000 tokens, then logs to `step3/rewrite_failures.jsonl` instead of failing silently. The pref pipeline checkpoints per **arm** (`pairs/arm_responses.jsonl`), so one failed arm never discards or re-bills its sibling's paid response.
- **Committed run outputs are deliberate.** Smoke/validation runs under `outputs/*/runs/` are kept in git as reviewable examples of pipeline behavior at each design stage; `local_*`-labeled runs and `latest` pointers stay untracked (gitignore covers all pipelines incl. pref). Prune only with team agreement.
- **Capabilities removed in the spec-driven rebuild** (vs the 7-step pipeline at `071462c`): pushback turns (multi-turn corpus), sampling-condition injections, and MANTA scenario import. Removed by design, pending the team's design-level review — the old rationale lives in git history. The scoring gate (`step3_score.txt`) is the remaining unwired piece; its rubric is still being authored and it will be wired in later.
- **Final DAD records contain only user + assistant messages** — system prompts, reasoning library scaffolding, annotations, and the constitution are stripped before training records are written
- **Step-2 responses reason from the animal-ethics reasoning library** (`prompts/dad/reasoning_library.csv`; `reasoning_library_ABOUT.md` is human reference about it, not injected) — the CSV is the source of truth (the old JSON is retired). Rows are entries with columns `id, category, claim, reasoning, crux, transferable_move`; category is one of **Conduct** (C1–C10), **Core move** (M1–M13), or a topic category (T1–T29). Step 2 has two sub-stages. **2a scope** (`prompts/dad/step2_scope.txt` → `step2/scopes.jsonl`): an LLM call rebuilds the full map before reasoning — the whole harm pathway and every moral patient (system), the highest-leverage lever from the user's seat (agent), what acting honestly costs the person (cost), the second-order effect worth aiming at (upside), and the realistic baseline if the user does nothing (counterfactual); it reads everything from the user's message (not the annotation). **2b respond** (`prompts/dad/step2_respond.txt`, the response-generation spec): the response is generated over the scope map, two-sided with the crux named. The **whole library** (all 52 entries — conduct C*, core moves M*, topic T*) is embedded in the response prompt itself; conduct is always-on, core moves fire in most cases, topic entries are drawn on as the case reaches them. The prompt *is* the generation guidance, so there is **no separate system prompt**, and the annotation is not passed to the response — it reasons from scope + library + user message. The library and scope are sampling scaffolding: never named in responses, stripped before training records. `dad.responses.per_prompt` controls samples per dilemma.
- **Step 2 is annotation-independent; the annotation drives steps 1 and 3.** The annotation is design metadata (the sampled design record from step 1, and the anchor for the step-3 rewrite), but step 2 no longer reads it: 2a scopes from the user's message and 2b reasons from scope + library. Calibration **Direction** is not used in step 2 at all — the response reasons from the ethics of the case, not the user's leaning, and does not name or diagnose a miscalibration direction. Direction lives only in step 1 (a sampled axis that guarantees balanced coverage of under- and over-weighting cases) and in `step3_score.txt`, which is *designed* to re-derive the response's realized direction blind and check it against the annotation's intended Direction. **Note: `step3_score.txt` is not wired into `run.py`** — there is no scoring stage in the pipeline yet, so this gate is not currently enforced; it exists as the scoring rubric to wire in.

## Directory Structure

```
constitution/       constitution source documents (Claude constitution + sentient-beings reading)
context_docs/       background reading: tcw.md ("Teaching Claude Why" post this repo implements) + constitution PDF
shared/             API wrapper, utils, constitution loader
sdf_pipeline/       5-layer document generation pipeline
dad_pipeline/       3-step chat transcript pipeline
pref_pipeline/      response-pair generation + blind human A/B rating app
prompts/sdf/        prompt templates for SDF layers
prompts/dad/        dilemma prompt spec + reasoning library + DAD step templates
outputs/sdf/        intermediate + final SDF outputs
outputs/dad/        intermediate + final DAD outputs
evals/              scoring scripts and rubric
```
