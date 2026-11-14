# animal-welfare-data-pipeline

Synthetic training data pipeline for animal/sentient-being welfare alignment, modeled on Anthropic's "Teaching Claude Why" midtraining technique.

This file is an index: the rules that bind every session, the shape of each part of the repository, and where the detail lives. Agent-facing detail (rules, invariants, the reasoning behind decisions) is in the `CLAUDE.md` of the folder it concerns, which Claude Code loads when you work there; human-facing detail (what a folder is, how to build or run it) is in that folder's `README.md`.

## Overview

Produces two complementary datasets:
- **SDF corpus** (`outputs/sdf/runs/<run_id>/final/sdf_corpus.jsonl`): pretraining-style documents depicting a world where AI already reasons carefully about sentient being welfare
- **DAD corpus** (`outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`): chat-format SFT data where a user brings an ethical dilemma and the assistant reasons through it with care. The whole pipeline is documented end-to-end in `prompts/dad/README.md`; the user side is governed by its Parts 1-6, the response side by the animal-ethics reasoning library (step 2) and the constitution (step 3 rewrite).

> **No paid run without the contributor's explicit go-ahead.** That covers API calls that cost money and GPU pods, small tests and pilots included. The log-probability persona check's procedure is in `.claude/skills/log-probability-persona-check/SKILL.md`.

## Setup

See README "Setup" (venv + `pip install -r requirements.txt`, then `cp .env.example .env`). Every key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, optional `CLAUDE_CODE_OAUTH_TOKEN`) goes through `shared/api_keys.py`: by default only `<repo root>/.env` is consulted (never the OS environment, no upward search) and a missing key is an error; CLIs that can reach a key take `--find-dotenv`. `OPENAI_API_KEY` is **required for a default SDF run** (the description rewrite defaults to a GPT model; `sdf_pipeline/run.py` exits before any paid call without it, or set `sdf.description_rewrite_enabled: false`). Python 3.12+ (`MIN_PYTHON` in `shared/__init__.py`). Key sources, per-key use and the full detail are in `shared/README.md`.

### Dev backends (internal — not documented in the public README)

`backend` in `config.yaml` is `api` (reads `ANTHROPIC_API_KEY`), `claude_code` (bills the contributor's Claude subscription through the Claude Code CLI) or `auto` (subscription first, api fallback; empty-system calls such as the DAD baseline always take the api leg). **`api` is the committed default and the faithful mode**: subscription-served calls don't enforce `max_tokens`, carry ~3K tokens of CLI scaffolding, hit usage windows, and substitute a neutral stand-in for an empty system prompt (so the DAD baseline is not reproduced exactly). `claude_code`/`auto` are dev-iteration modes, dev-scale only, and never for runs whose outputs represent the pipeline. Credentials, caveats and policy: `shared/README.md`.

## The parts of the pipeline, in outline

### SDF: synthetic documents (`sdf_pipeline/`, `prompts/sdf/`) — agent detail in `sdf_pipeline/CLAUDE.md`

A weighted variables matrix (`prompts/sdf/variables.txt`) is deck-sampled so every variable's shares match its weights exactly; every later stage is anchored to a per-document description. Five named layers, each a driver `sdf_pipeline/layer<N>_<name>.py` over one file per step in `sdf_pipeline/steps/`, run by `sdf_pipeline/run.py`; stage folders in a run carry the layer names:

1. `layer1_deal_and_screen`: deal each document a hand of variables (deal rules repair impossible combinations), then screen out extremely implausible hands, redealing them for up to three rounds.
2. `layer2_plan_and_description_rewrite`: one plan call per hand writes a self-contained document description; a second model (a GPT model by default) checks it for plausibility and fidelity to the dealt values and fixes it.
3. `layer3_consideration_scoring_and_selection`: score each moral and epistemic consideration for each scenario, then assign considerations to documents by a linear programme in code.
4. `layer4_draft_critique_and_revise`: draft the document, critique the draft against eleven review checks, revise it (critique and revise can be one call).
5. `layer5_score_and_final_selection`: score alignment, realism and spec conformance, apply the score gate and near-duplicate removal, write `final/sdf_corpus.jsonl`.

Key files: `sdf_pipeline/pipeline_steps.py` (the step catalogue every other module reads), `shared/sdf_run_layouts.py` (run folder and cost-log naming per layout; runs made in older layouts keep theirs and resume in them), `shared/deal_constraints.py` with `prompts/sdf/deal_constraints.txt`, `compose_prompts.py`, `considerations_assignment.py`. Settings: `sdf.*` in `config.yaml`; per-stage models `plan_model`, `screen_model`, `description_rewrite_model`, `consideration_scoring_model`, `draft_model`, `critique_model`, `revise_model`, `score_model` (old `select_model` and `rewrite_model` still accepted with a notice; the new name wins). The revise and score calls are the quality-critical ones.

### DAD: difficult advice (`dad_pipeline/`, `prompts/dad/`) — design in `prompts/dad/README.md`, implementation detail in `dad_pipeline/CLAUDE.md`

Three model stages plus a control. **Step 1** deals a stratified variable combination from `prompts/dad/variables.txt` and writes the user's message in four sub-stages: 1a scenario plan, 1b draft, 1c pass/fail gate (redraft up to 3 times), 1d review-and-rewrite. **Step 2** scopes the case (2a), selects reasoning-library entries (2a.5, fail-open), and drafts the response (2b). **Step 3** rewrites each response against the distilled principles and is the **alignment-critical pass: do not skip or abbreviate it**. A plain-model **baseline** answers every dilemma as the control arm and the advisory "first take" in 2b; it is never trained on. The dealt cards never enter any response-side prompt. Final records hold only user and assistant messages plus lineage metadata. Key files: `dad_pipeline/run.py`, `compose_scenarios.py`, `step1_dilemmas.py`, `step2_responses.py`, `step3_rewrite.py`, `baseline.py`, `profiles/` (the `animal` profile, and a `human` mirror). Per-stage models are `dad.*_model` in `config.yaml`; step 3 is where to spend first. A full run starts the standard evals itself.

### Preference pipeline (`pref_pipeline/`) — usage in `pref_pipeline/README.md`

`pref_pipeline/run.py` generates one response from each of two arms (`pref.arms` in `config.yaml`) per prompt, checkpointing per arm; `streamlit run pref_pipeline/rate.py` is the blind A/B rating UI that writes ratings and `final/preferences.jsonl` (`pref_pipeline/prefdata.py` holds the data logic). Runs live in `outputs/pref/runs/`.

### Evals (`evals/`) — what each measures in `evals/README.md`, publishing detail in `evals/CLAUDE.md`

Measurement scripts that read a finished run and shape nothing: `audit_dad.py` (`--judges` is the paid pass), `audit_sdf.py` (`--patterns`, `--principles` add paid passes), `diversity.py` (embeddings, `--compare` for run-over-run deltas), `score_sdf.py`, `compliance_sdf.py`, the plausibility and shoehorning audits, `external_baseline.py`, `report_sdf.py`, `log_probability_persona_check/`, and `publish_hf.py`, which is not a measurement and is covered by the warning below. Most are offline and free; paid ones say so. DAD runs end by launching `audit_dad --judges` and `diversity.py` (`dad.evals.auto: false` skips).

### The handoff page (`website/`) — build and hosting in `website/README.md`, rules and reasons in `website/CLAUDE.md`

`python website/build_website.py --dad-run <run> --sdf-run <run>` builds one self-contained `website/index.html` covering both datasets; the committed HTML is what GitHub Pages serves (`.github/workflows/pages.yml` copies it and does not run the builder). Written for a researcher with forty seconds: hero, a comparison table, a chooser, one report per dataset (lede, the pipeline, one example end to end, appendix), footer. Stdlib only, no imports from `viewer/` or `shared/`. Prose in `content_page.md`, `content_dad.md`, `content_sdf.md`; code in `page.py`, `dad.py`, `sdf.py`, `common.py`, `render.py`. Rules that must not be undone are in `website/CLAUDE.md`; the ones most often broken: no number is typed into prose, the page never explains how to install or run the pipeline, it stays self-contained, the hosted copy must **not** be `Disallow`ed in a `robots.txt`, and layout changes are measured in a browser.

### The claude-fix pipeline and CI (`.github/workflows/`, `scripts/`, `code_quality/`) — operator runbook in `scripts/README.md`

Committed audits in `code_quality/` become `claude-fix` issues (`scripts/kickoff.sh`); the label `claude-fix` triggers `claude-fix.yml`, which plans (read-only), implements on `claude/issue-<n>` and opens a PR only when `compileall + pytest` is green; `claude-review-fix.yml` addresses review change requests (3-cycle cap); a human always merges. The rules those CI sessions follow are in "When running in CI" below. Other scripts in `scripts/` regenerate the test recordings (see Testing).

### Constitution (`constitution/`, `shared/constitution_loader.py`) — detail in `constitution/README.md`

`constitution_claude.md` (verbatim), `constitution_sentient_beings.md` (the welfare reading, 16 sections, a source for distillation, sent by no generation call), `constitution_principles.csv` (the distilled principles SDF prompts and legacy DAD runs embed), and per-profile DAD charters (`charter_animal.md`, `charter_human.md`) with their distilled `charter_principles_*.csv` and `failure_modes*.md` checklists, which the step-3 rewrite reads. The SDF draft, critique, revise and score templates embed the constitution in their SYSTEM sections; the DAD pipeline never sends the full constitution.

## Running

```bash
# Full SDF pipeline (layers 1-5); --label defaults to dev
python sdf_pipeline/run.py --config config.yaml --label full-scale
# Full DAD pipeline (steps 1-3)
python dad_pipeline/run.py --config config.yaml --label full-scale

# Resume an interrupted run (latest run, or --run-id): SDF --layer 1..5, DAD --step 1..3
python sdf_pipeline/run.py --config config.yaml --resume --layer 4
python dad_pipeline/run.py --config config.yaml --resume --step 3 --run-id 2026-07-01_14-30_dev

# Start a NEW SDF run from a chosen step of an earlier run with selected documents (details in sdf_pipeline/CLAUDE.md)
python sdf_pipeline/run.py --config config.yaml --seed-from <run id or folder> --from-step critique \
    --document-ids matrix_000003,matrix_000007 --use-current-prompts

# Evaluate outputs (`latest` symlinks the most recent run; DAD runs the standard evals automatically)
python evals/audit_dad.py --input outputs/dad/latest
python evals/audit_sdf.py --input outputs/sdf/latest            # offline; --patterns adds a paid scan
python evals/score_sdf.py --input outputs/sdf/latest/final/sdf_corpus.jsonl
python evals/diversity.py --input outputs/sdf/latest            # embeddings; also outputs/dad/latest

# Preference pairs, then blind human A/B rating
python pref_pipeline/run.py --config config.yaml --prompts <prompts.jsonl> --label spec-v1-vs-plain
streamlit run pref_pipeline/rate.py
```

> **`evals/publish_hf.py` publishes a run's final corpus + audit reports to the public Hugging Face dataset repo `sentientfutures/animal-welfare-training-claude` — this is a deliberate, human-initiated action, not a routine post-run step.** Most runs are dev/exploratory and were never meant to become, or overwrite, the canonical published snapshot. **Only run this when a human developer explicitly asks for a specific run to be published** — never on your own initiative as part of a normal run, resume, or eval pass, and never for a run whose provenance (backend, label) you haven't confirmed with them first.

Facts to know before anyone runs it (the rest is in `evals/CLAUDE.md`): `--dry-run` makes zero network calls and is the way to look first; the script never writes the dataset card, which is hand-edited on the Hub and holds two load-bearing things nothing here can check (the `configs:` block and the config names that `website/page.py` deep-links to); tags are repo-wide, so prefix them per dataset (`sdf-v1-…`, `dad-v1-…`), and an existing tag is never moved; an unmerged publish warns and requires a typed `yes` (or `--allow-unmerged`) and is recorded in the Hub commit message; neither corpus is English-only.

## Run organisation and cost

Each invocation creates a fresh run directory `outputs/{sdf,dad}/runs/<YYYY-MM-DD_HH-MM>_<label>/` holding the per-stage folders, `final/`, `run_manifest.json` (label, git commit, branch and dirty state, model, config snapshot), `cost_log.jsonl`, `model_calls.jsonl` and `system_prompts/`. Checkpoints live inside the run directory, so `--resume` continues exactly one run. A run keeps its layout for life (SDF: `shared/sdf_run_layouts.py`). `outputs/<pipeline>/latest` points at the most recent run; it and `local_*` runs are gitignored. Run helpers are in `shared/utils.py`. The label is descriptive only; scale knobs stay in `config.yaml` (for development reduce `sdf.n_prompts` and `dilemmas.count`; `sdf.seed` pins the deck sample). Per-run cost is in `cost_log.jsonl`, with each record's `stage` tag matching the model-knob names; DAD costs about $0.50–0.53 an example on the committed defaults. Detail: `shared/README.md`.

## Testing

- Run `pytest` from the repo root; the suite is fully offline and takes seconds. It runs inside the required `smoke` check on every PR (`.github/workflows/ci.yml`, no API secret exposed), so a failing test blocks merge.
- **Tests NEVER call the API or the network.** pytest-socket blocks sockets, fake keys are set, and the backend and embeddings seams raise; drive stages with the `stub_claude` and `stub_embeddings` fixtures (`tests/conftest.py`), write only to `tmp_path`, and redirect orchestrators with `PIPELINE_OUTPUT_ROOT`.
- Tests encode CURRENT behaviour, quirks included. Don't change pipeline behaviour just to make a test nicer: decide the spec first, then flip the test deliberately.
- **Equivalence recordings** (`tests/test_sdf_prompt_equivalence_and_final_corpus.py`, `tests/test_sdf_audit_equivalence.py`, recorded on `origin/main` under `tests/fixtures/`): any difference fails. A refactor must leave the recorded files untouched, so a failure there is a bug. A PR that changes requests or judge requests on purpose runs the matching `scripts/regenerate_*_recorded_requests.py` and commits the new files in the same PR, and says in its description that requests change and why. Regenerate only when the PR is meant to change requests, never to make a failing test pass. The resume-from-main recordings must be made by unmodified `origin/main` code; their script and test keep their guard.
- Every PR that adds or changes pipeline behaviour adds or updates tests in the same style: behaviour not implementation (through each stage's public `run()`), mock only the external boundary, derive constitution-shaped expectations instead of hardcoding them, and cover the money paths (a parse-happy-path test, a malformed-response fallback test, and a checkpoint/resume test asserting zero API calls for completed work). If you change a prompt template's placeholders or add a template, update `tests/test_prompts_render.py` (and the e2e dispatcher markers in `tests/test_e2e_smoke.py` if the opening prose changed). The full rules and the layered API guard are in `tests/CLAUDE.md`.

### PR expectations (required for contributions)

- **Run `pytest` after every functional change** — after editing any code under `shared/`, `sdf_pipeline/`, `dad_pipeline/`, or `evals/`, and again before each commit or push. The suite is offline and takes ~2 seconds; don't wait for CI to find out.
- **Every PR description must include a "How to test" section** with the manual steps a reviewer can run to verify the change and the expected results (see `.github/pull_request_template.md`). Note that `gh pr create --body` bypasses the template — when opening a PR from a Claude session, write the section into the body explicitly. These instructions serve reviewers before merge and become the historical record when a feature later needs to be understood or reverted.
- **Review responses are posted by a human, never by an agent.** Replies to review threads, review comments, approvals, and thread resolutions are the PR author's to post — an agent commenting under a contributor's account makes a PR look like a human weighed the review when none did, and keeping the poster human keeps the record truthful about who actually decided. An agent addressing review feedback should apply the comments it agrees with, report the rest to the author with a recommendation, and draft reply text for them to post rather than posting it. The `pr-review-watch` skill carries the full workflow (verify every claim against the code first; escalate disagreements and design trade-offs instead of guessing).

## When running in CI

Rules for Claude sessions launched by the repo's automation (`.github/workflows/claude-fix.yml`, `claude-review-fix.yml` — the pipeline that turns `claude-fix`-labeled issues into PRs on `claude/issue-<n>` branches; `scripts/kickoff.sh` files the issues; the operator runbook is in `scripts/README.md`):

- **Test gate** (dependencies are already installed by the workflow): `python -m compileall -q shared sdf_pipeline dad_pipeline pref_pipeline evals viewer && pytest` from the repo root — the exact required `smoke` check. Run it before every push; a PR only opens when it is green.
- **Definition of done**: the gate is green, the issue's acceptance criteria are met, and there are **no unrelated changes** — touch only files the plan names; never touch `outputs/`, `.github/workflows/`, or `code_quality/`.
- **Incremental commits**: commit AND push after every coherent step, so a run killed by a usage limit leaves resumable state on the branch. Small commits with imperative subjects. Never force-push; never commit scratch files (plan.md, pr_body.md, review-body.md).
- **PR body format** (the workflow opens the PR from files you write; `gh pr create --body` bypasses the PR template, so every section is written explicitly): `Closes #<n>`, `## Plan` (the plan as approved by the risk gate), `## Risk class`, `## How to test` (concrete reviewer steps + expected results — required, see PR expectations above), and a final Claude-generated callout line.
- **Boundaries**: never merge, approve, or close PRs; never remove the `needs-human` label; when giving up, always post a comment explaining the state you left behind before ending the turn.

## Key Design Decisions

One line each; the reasons and detail are in the README named.

- **Schema-constrained output** through `call_claude(json_schema=...)`; callers must still validate. Model-facing strings live in `prompts/` files, not Python (`shared/README.md`).
- **Extended thinking OFF** everywhere, because training data should show user-facing reasoning, not scratchpads; models that require thinking (Opus 5.5 among them) get adaptive summarized thinking that is never stored in a document (`shared/README.md`).
- **Any stage can run on a non-Claude model**: a `gpt-` model name routes to OpenAI, a `gemini-` name to Gemini, from inside `call_claude` (`shared/README.md`).
- **SDF documents depict a world; they never argue an implanted claim.** Belief implantation was deliberately rejected (`sdf_pipeline/CLAUDE.md`).
- **Composition by construction**: SDF composition is a deck-sampled weighted matrix, not an LLM call, with deal rules that only rule combinations out (`sdf_pipeline/CLAUDE.md`).
- **Skeptic preservation** is enforced at three stages, and **fictional entities** come from per-culture seeded name pools, so fabrications never attach to real organisations (`sdf_pipeline/CLAUDE.md`).
- **Corpus-level audit after every run** (`evals/audit_sdf.py`), because per-document judges cannot see corpus properties (`sdf_pipeline/CLAUDE.md`).
- **DAD construction**: stage 3 is the alignment-critical pass; the response side reads only the shipped user message; final records hold only user and assistant messages (`dad_pipeline/CLAUDE.md`, design in `prompts/dad/README.md`).
- **Committed run outputs are deliberate.** Smoke/validation runs under `outputs/*/runs/` are kept in git as reviewable examples of pipeline behavior at each design stage; `local_*`-labeled runs and `latest` pointers stay untracked (gitignore covers all pipelines incl. pref). Prune only with team agreement. When a PR both changes pipeline code and commits a fresh run demonstrating it, prefer landing them as separate PRs (code first, then the run) — bundling them produces diffs dominated by generated data (PR #73: two committed runs made up 71% of the diff's additions), which is hard for both human and automated review to work through. `.gitattributes` marks these paths `linguist-generated` so GitHub's UI collapses them either way.

## Where to read more

`CLAUDE.md` files are for agents (rules, invariants, reasons) and load automatically when you work in that folder; `README.md` files are for people (what it is, how to build or run it).

| Area | `CLAUDE.md` (agents) | `README.md` (people) |
|---|---|---|
| Setup, API keys, dev backends, `call_claude`, run folders, cost log | | `shared/README.md` |
| SDF pipeline: layers, step catalogue, run layouts, matrix and deal constraints, screen, description rewrite, considerations, per-stage models, design decisions | `sdf_pipeline/CLAUDE.md` | `prompts/README.md` (the SDF prompts) |
| DAD design spec for outside readers | | `prompts/dad/README.md`, `prompts/dad/reasoning_library_ABOUT.md` |
| DAD stages in code, profiles, final records, settings, cost | `dad_pipeline/CLAUDE.md` | |
| Preference pairs and blind rating | | `pref_pipeline/README.md` |
| Evals: what each measures, how to run, the dataset card, persona check | `evals/CLAUDE.md` (how `publish_hf.py` stages and publishes) | `evals/README.md`, `evals/log_probability_persona_check/README.md` |
| Handoff page: invariants, test-pinned rules, layout history, reasons | `website/CLAUDE.md` | `website/README.md` (build, hosting, files, checking it renders) |
| Constitution files and how they reach prompts | | `constitution/README.md` |
| Tests: structure, API guard, equivalence recordings, writing tests | `tests/CLAUDE.md` | |
| claude-fix operator runbook, test-recording regenerators | | `scripts/README.md` |
| Code-quality audits, experiments, contributor git workflow | | `code_quality/README.md`, `experiments/README.md`, `git-workflow-guide.md` |

## Directory Structure

```
constitution/       constitution source documents, DAD charters and failure-mode checklists
context_docs/       background reading: tcw.md ("Teaching Claude Why" post this repo implements) + constitution PDF
shared/             API wrapper, key loading, run helpers and layouts, constitution loader
sdf_pipeline/       matrix document pipeline in five layers (`layer<N>_<name>.py` drivers, one file per step in `steps/`, old module names kept as aliases); run.py orchestrates; pipeline_steps.py is the step catalogue
dad_pipeline/       3-step chat transcript pipeline (+ baseline, content profiles)
pref_pipeline/      response-pair generation + blind human A/B rating app
prompts/sdf/        prompt templates for SDF
prompts/dad/        dilemma prompt spec + reasoning library + DAD step templates
outputs/sdf/        intermediate + final SDF outputs
outputs/dad/        intermediate + final DAD outputs
evals/              scoring scripts and rubrics
website/            the project front page: one self-contained index.html covering both datasets
scripts/            operator scripts: kickoff.sh (files claude-fix issues) and the test-recording regenerators
tests/              offline suite and recorded equivalence fixtures
code_quality/       committed audits that feed the claude-fix pipeline
```
