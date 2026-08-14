# animal-welfare-data-pipeline

Synthetic training data pipeline for animal/sentient-being welfare alignment, modeled on Anthropic's "Teaching Claude Why" midtraining technique.

## Overview

Produces two complementary datasets:
- **SDF corpus** (`outputs/sdf/runs/<run_id>/final/sdf_corpus.jsonl`): pretraining-style documents depicting a world where AI already reasons carefully about sentient being welfare
- **DAD corpus** (`outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`): chat-format SFT data where a user brings an ethical dilemma and the assistant reasons through it with care. The whole pipeline is documented end-to-end in `prompts/dad/README.md`; the user side is governed by its Parts 1-6, the response side by the animal-ethics reasoning library (step 2) and the constitution (step 3 rewrite).

## Setup

See README "Setup" (venv + `pip install -r requirements.txt`, then `cp .env.example .env`). Auth depends on the `backend` key in `config.yaml`: `api` reads `ANTHROPIC_API_KEY`; `claude_code` bills the contributor's Claude subscription via the Claude Code CLI (logged-in session or `CLAUDE_CODE_OAUTH_TOKEN`); `auto` prefers the subscription and falls back to the api key — per-call: empty-system calls (the DAD baseline arm) always take the api leg so the plain-model condition stays exact, and an exhausted usage window demotes the rest of the run to the api, loudly, with each cost-log record naming the backend that served it. `api` stays the committed default: it is the faithful mode (subscription-served calls don't enforce `max_tokens` and carry CLI scaffolding in context), so runs meant to represent pipeline behavior stay on `api`; `auto`/`claude_code` are dev-iteration modes. Full setup and caveats for the dev backends are in "Dev backends" below. `GEMINI_API_KEY`/`OPENAI_API_KEY` are optional and read only by `evals/diversity.py` (embedding-based diversity audit; either provider works, and the runs to date used Gemini). `evals/publish_hf.py` (Hugging Face dataset publishing) reads `HF_TOKEN` from `.env` if set; otherwise falls back to a one-time `huggingface-cli login`, whose cached token `huggingface_hub` picks up on every later call. Either way the token needs write access to the target repo/org.

### Dev backends (internal — not documented in the public README)

`backend: claude_code` routes calls through the Claude Code CLI, billed to the
contributor's own Claude Max/Pro subscription instead of the API key. Two ways to
give it credentials: an existing interactive Claude Code login is picked up
automatically, or `claude setup-token` prints a token for `CLAUDE_CODE_OAUTH_TOKEN`
in `.env` (a subscription-tied OAuth token valid ~1 year, not a Console API key;
use this path for CI or any non-interactive machine).

Caveats, all reasons to keep this to dev-scale runs:

- **Usage limits.** A 5-hour rolling window plus a weekly cap, shared with your
  interactive Claude Code use. A full-scale run will exhaust the window; the run
  stops with a clear message and progress is checkpointed, so `--resume` continues.
- **Per-call overhead.** ~3K input tokens of CLI scaffolding per call and a CLI
  process per request, so calls are slower. `max_tokens` from `config.yaml` is not
  enforced (Claude Code applies its own output cap), and `cost_usd` in the cost log
  is notional — what the run would have cost at API prices.
- **Empty system prompts get a neutral stand-in.** Claude Code substitutes its own
  agentic CLI prompt when the system prompt is empty, so the one empty-system call
  in the pipelines (the DAD baseline) gets a one-line neutral prompt instead (see
  `_NEUTRAL_SYSTEM` in `shared/api.py`) and is **not reproduced exactly**. On
  `auto` the baseline always takes the API leg for this reason. A one-time warning
  prints when the substitution happens.
- **Policy.** Anthropic's docs steer programmatic workloads toward API keys.
  Running this pipeline on a personal subscription is the same posture as using
  Claude Code itself, but it is a gray area: dev-scale only, and never for runs
  whose outputs represent the pipeline.

`shared/__init__.py` enforces a Python floor (`MIN_PYTHON = (3, 12)`, matching numpy) at import — bump it there if the deps' floor rises. `.venv/` is gitignored.

## Running

```bash
# Full SDF pipeline (layers 1-4); --label defaults to dev
python sdf_pipeline/run.py --config config.yaml --label full-scale

# Full DAD pipeline (steps 1-3)
python dad_pipeline/run.py --config config.yaml --label full-scale

# Resume interrupted run from a specific stage (latest run, or target one with --run-id)
python sdf_pipeline/run.py --config config.yaml --resume --layer 3
python dad_pipeline/run.py --config config.yaml --resume --step 3 --run-id 2026-07-01_14-30_dev

# Evaluate outputs (latest symlink points at the most recent run).
# DAD runs the standard evals AUTOMATICALLY at the end of every full run
# (audit_dad --judges + diversity.py; dad.evals.auto: false to skip) —
# the commands below are for re-runs, partial runs, and older run dirs.
# DAD: corpus-level audit — response lengths, tracked tics/moves, and the
# tic-candidates review queue, offline/free; --judges adds the paid LLM judge
# pass (welfare impact + delivery quality + showcase).
python evals/audit_dad.py --input outputs/dad/latest
python evals/score_sdf.py --input outputs/sdf/latest/final/sdf_corpus.jsonl

# Preference pairs: two responses per prompt (arms a/b), then blind human A/B rating
python pref_pipeline/run.py --config config.yaml --prompts <prompts.jsonl> --label spec-v1-vs-plain
streamlit run pref_pipeline/rate.py

# Corpus-LEVEL audit of an SDF run: composition/register spread, near-dup rate,
# name/phrase collapse, opening shapes, truncation artifacts (offline, free);
# --patterns adds the LLM templating scan (scan -> consolidate -> prevalence)
python evals/audit_sdf.py --input outputs/sdf/latest
python evals/audit_sdf.py --input outputs/sdf/latest --patterns

# Semantic diversity audit (SDF or DAD run): embedding-space near-dup rate,
# most-similar pairs, Vendi effective-document count, per-type spread.
# Embeds via GEMINI_API_KEY or OPENAI_API_KEY (cents per run, cached);
# --compare a previous diversity_report.json for run-over-run deltas
python evals/diversity.py --input outputs/sdf/latest
python evals/diversity.py --input outputs/dad/latest
```

> **`evals/publish_hf.py` publishes a run's final corpus + audit reports to the public Hugging Face dataset repo `sentientfutures/animal-welfare-training-claude` — this is a deliberate, human-initiated action, not a routine post-run step.** Most runs are dev/exploratory and were never meant to become, or overwrite, the canonical published snapshot. **Only run this when a human developer explicitly asks for a specific run to be published** — never on your own initiative as part of a normal run, resume, or eval pass, and never for a run whose provenance (backend, label) you haven't confirmed with them first.

That one repo holds **both** corpora as separate HF *configs* (each gets its own selector in the dataset viewer), staged under per-pipeline subdirectories — `sdf/` and `dad/`, each with its corpus jsonl, `run_manifest.json`, and `audit/`. Consequences worth knowing before running it:

- **The script never writes the dataset card.** `README.md` on the Hub is hand-written and edited there, by the team, and no copy of it lives in this repo. A publish stages `<pipeline>/…` only and `delete_patterns` is scoped to the same prefix, so the card is a path the upload can neither overwrite nor delete. The generator that used to rebuild it from each run's audit files was removed (along with `--regenerate-card`, `--license` and `--pretty-name`) because it replaced every hand-edit. **Two things on that card are load-bearing and nothing here can check them**: the `configs:` block in its frontmatter is what makes the two corpora loadable as separate viewer configs, and `website/page.py`'s `HF_SDF`/`HF_DAD` deep-links are built from those config names verbatim. See "The dataset card" in `evals/README.md`.
- **Both configs publish the same column philosophy: the content a reader came for → supporting metadata → the widest lineage column → the id, trailing everything, last.** SDF is re-keyed by `reorder_sdf_corpus` into `content` · `description` · `language` · `type_name` · `type_id` · `register` · `scores` · `variables` · `doc_id` — no longer a verbatim `shutil.copy2`, since the pipeline's own write order (`sdf_pipeline/layer5_score.py`) buried `content`, the actual document text, behind `doc_id` and four other metadata fields. DAD is rewritten per record by `flatten_dad_corpus` into `user_prompt` · `assistant_response` · `language` · `variables` · `example_gid`. `variables` is the one nested column on each side and the counterpart of the other corpus's own: the whole hand the scenario/document was dealt, so a reader can tell which slice of the matrix a row came from rather than only which language it is in. It sits second-to-last on both, ahead of only the id, because it is the widest cell and the text column(s) are what a visitor came to read; the id trails even `variables` because it is pure lineage/join bookkeeping (still used internally to look up `language`/`variables`; see `flatten_dad_corpus`), not something a reader needs in front of the content. A field `reorder_sdf_corpus` doesn't recognise — a legacy field like `subtype_id`/`role` predating the current SDF schema — is appended after the known columns rather than dropped: this reorders columns, it never selects them, so an older committed run missing `variables`/`description`/`type_name` still publishes every field it has.
- **`language` and `variables` are both joined off `step3/rewrites.jsonl`, in one read** (`dad_dealt_cards` → `{example_gid: cards}`, with `dad_languages` derived from it), so they can never describe different records. Joined rather than read off the corpus record even though the record now carries `variables` itself: every committed run predates that field, so the join is what gives the column to the published 1,324 rows without rewriting a line of committed output. Both follow the same two rules — omitted entirely when nothing resolves (a column null down every row reads as broken), null on a single row that does not join (a visible gap is honest, an invented value is not). Their "nothing resolved" gates are **separate and per-run**, because they are not the same claim: `archetype10` deals no `cultural_setting`, so it earns no `language` column, but it has cards worth publishing under `variables`.
- `delete_patterns` is scoped to `<pipeline>/audit/*`. A bare `audit/*` would delete the *sibling* dataset's audit files on every publish.
- **Published rows are ordered English first**, because the Hub viewer opens on whatever is first in the file and both corpora are written in matrix-deal order (the live SDF config opened on Spanish, DAD on Hindi). **Only the staged copy is reordered — never the run's own `final/` corpus**, which five evals stride-sample (`compliance_sdf.py`, `audit_sdf.py`, `diversity.py`, `score_sdf.py`) and which layer 5's greedy near-dup cull already ordered. It is a stable binary partition, not a sort by language: English block first, everything else after, each in the run's own order. The pass sorts raw lines, so the Hub file's ROW order is a permutation of whatever it was staged as going into this pass. Both corpora are re-keyed before staging now (`reorder_sdf_corpus`, `flatten_dad_corpus`), so the Hub file is no longer a raw-text permutation of the run's own `final/*.jsonl` the way `diff <(sort hub.jsonl) <(sort final.jsonl)` could once check for SDF — the equivalence check is now over parsed, sorted records (same records, same values, nothing added or dropped, only column and row order changed), not raw text. A run whose language cannot be read (no `language` field; a DAD run with no `step3/rewrites.jsonl` or no `cultural_setting` on its cards, e.g. `archetype10`) is published in the order it was written, and stdout says so. DAD rows carry a `language` column derived the same way — the language the scenario was **dealt**, null where a row does not join. Publishing prints the resulting language breakdown, which is now the only thing telling a publisher what the hand-maintained card ought to declare — **neither corpus is English-only**, and a card that says otherwise is a false claim on a public dataset.
- **Tags are repo-wide**, so prefix them per dataset (`sdf-v1-…`, `dad-v1-…`). The pre-multi-config `v1-fullscale-500-opus5` tag predates this convention. `_create_tag` passes `exist_ok=True` and **never moves an existing tag**, so a revision pinned by an old tag keeps the old row order for ever — publishing a reordered corpus at a pinned revision needs a **new** tag.
- `--dry-run` makes zero network calls: it stages, prints what would be uploaded and the commit message it would leave, and stops. (It therefore also skips the `git fetch` the merge check would otherwise do, and says `origin/main` may be stale.)
- `--staging-dir` is wiped before staging, so the script refuses any existing non-empty directory it did not create (marker `.publish_hf_staging`), in `--dry-run` too.
- **An unmerged publish warns and asks, and is recorded in the Hub commit message.** Before staging, the script checks whether the current `HEAD` and the run's own `git_commit` are reachable from `origin/main` (`utils.merge_state`). If either isn't — or can't be verified — it prints what's unmerged and requires a typed `yes`; with no TTY it exits telling you to pass `--allow-unmerged`. Proceeding appends `_unmerged_summary`'s clause to the commit message, naming each unverified run with the branch it was **generated** on and its commit, and separately the branch it was **published** from. That was a stamp on the generated card until the card became hand-edited; a commit message is the better home anyway, since nobody can edit it afterwards. A **dirty tree at run time is context, never a trigger** — every run so far has been dirty, and a warning that fires on all of them is one people learn to ignore. This is a guardrail against accidents, not an access control: the write token is on contributors' laptops, so anyone can bypass the script entirely — which is exactly why the Hub's own history, not the terminal, carries the record.

```bash
# Stages final/{sdf,dad}_corpus.jsonl (English rows first; DAD also flattened to
# example_gid/language/user_prompt/assistant_response) + run_manifest.json +
# audit/*.{json,jsonl,html}
# into <pipeline>/ (report_content.json excluded — editorial, already baked into
# corpus_report.html) and uploads exactly that. The dataset card is NOT written:
# it is hand-edited on the Hub. Requires a Hub token with write access to the
# target repo/org, one time (`huggingface-cli login`, or HF_TOKEN in .env);
# --dry-run stages + prints what would be uploaded with no network calls.
# An unmerged run prompts for confirmation first (--allow-unmerged skips the
# prompt; the commit message records it either way).
REPO=sentientfutures/animal-welfare-training-claude
python evals/publish_hf.py --input outputs/sdf/latest --repo-id $REPO --dry-run
python evals/publish_hf.py --input outputs/sdf/runs/<run_id> --repo-id $REPO \
    --tag sdf-v1-<run-label>
python evals/publish_hf.py --input outputs/dad/runs/<run_id> --repo-id $REPO \
    --tag dad-v1-<run-label>
```

## Run Organization

Each pipeline invocation creates a fresh run directory `outputs/{sdf,dad}/runs/<YYYY-MM-DD_HH-MM>_<label>/` containing the per-stage dirs (`layer1`–`layer4` / `step1`–`step3`; runs made before the layer renumber keep the old `layer12`/`layer3`–`layer5` names and are resolved by `utils.sdf_stage_file`; steps 2–3 keep explicit checkpoints, step 1 resumes from its own append-only jsonl files; DAD runs also hold `baseline/` — a plain-model response per dilemma serving as the viewer’s control arm and as the advisory "first take" in the 2b prompt, never trained on; toggled by `dad.baseline.enabled`, see `dad_pipeline/baseline.py`), `final/`, `run_manifest.json` (label, git commit + branch + dirty state, model, full config snapshot; `manifest_version` 3 added `git_branch`, so every earlier run has a commit but no branch), and a per-run `cost_log.jsonl`. This keeps outputs from separate runs isolated — checkpoints live inside the run dir, so `--resume` (latest run by default, or `--run-id`) continues exactly one run. The label is purely descriptive (`dev` by default; scale knobs stay in `config.yaml`). An `outputs/<pipeline>/latest` symlink always points at the most recent run (gitignored, as are `local_*` run dirs, for every pipeline including pref). Run-scoping helpers (`create_run_dir`, `resolve_run_dir`) live in `shared/utils.py`.

## Scale / Cost

All knobs are in `config.yaml`. For development, reduce `sdf.n_prompts` (SDF — documents per run, deck-sampled from the variables matrix) and `dilemmas.count` (DAD) to keep test runs cheap. `sdf.seed` pins the deck sample; same seed + same variables file = the same composed prompts.

SDF supports per-stage model overrides (`sdf.plan_model` / `sdf.draft_model` / `sdf.rewrite_model` / `sdf.score_model`, each falling back to the global `model`): plans and drafts tolerate a cheap model, but the layer-4 rewrite and layer-5 scoring are the quality-critical calls — spend there first.

DAD likewise: `dad.scenario_model` (1a scenario plan) / `dad.prompt_draft_model` (1b) / `dad.prompt_gate_model` (1c gate) / `dad.prompt_refine_model` (1d refine — a separate knob; the gate never falls back to it) / `dad.response_scope_model` (2a) / `dad.response_select_model` (2a.5 library-entry selection; falls back to `response_scope_model` before the global) / `dad.response_draft_model` (2b) / `dad.constitution_rewrite_model` (step 3), each falling back to the global `model` — step 3 is the alignment-critical rewrite, spend there first. The global `temperature` (1.0) is wired into every call; generation wants 1.0 (diversity is the product — 1b register variety, 2b independent samples), and `call_claude` accepts a per-call override for eval/debug use.

`workers` sets how many API calls run concurrently within each SDF layer and each fan-out DAD stage — 1a scenario plans, 1b drafts (one call per scenario), 1c gate judgments, 1d refine rewrites, step 2 (one worker per dilemma: scope + its responses), step 3 rewrites (all via `utils.parallel_map`; set to 1 for serial debugging). Workers only call the API and parse — all file writes and checkpoint marks stay on the main thread, in input order.

Rough cost anchor (Sonnet 5, July 2026): a DAD example costs ~$0.20–0.25 end-to-end, so the default 40-example run is ~$9–10; smoke runs of 3–5 examples are under $1.

Running cost is tracked per run in `outputs/{sdf,dad}/runs/<run_id>/cost_log.jsonl` (evals log to the global `outputs/cost_log.jsonl`) — check it any time. Each record carries a `stage` tag (`prompt_draft`, `layer4`, `constitution_rewrite`, …) matching the model-knob names; the viewer's run list renders the per-stage cost breakdown (pre-tag records show as "(untagged)"). Records also log `duration_s` and `attempts` (API-retry count), and DAD calls tag an `item_id` naming the record served (scenario_id for 1a/1b/1c/1d — pre-rework runs comma-joined a 1b batch's ids — prompt_gid for 2a/2a.5, `{prompt_gid}_s{n}` for 2b, response_id for step 3; older runs' cost logs key the 2a/2b slots by their retired per-run prompt ids); the viewer's lineage page reads these via `loader.call_stats` to show model · cost · time · retries in each step expander (runs logged before these fields fall back to a model-only note).

## Preference Pipeline

`pref_pipeline/run.py` generates one pair per input prompt: a response from each of two arms defined in `config.yaml` under `pref.arms` (`name` + inline `system_prompt` or `system_prompt_file` relative to the repo root, optional per-arm `model`/`max_tokens`). Use it to A/B test candidate response specs against each other or against the bare model. Prompts come from any JSONL with a `user_message`, `refined`, or `prompt` field (handwritten sets, DAD step-1 `dilemmas.jsonl`). Runs live in `outputs/pref/runs/<run_id>/` with the same manifest/checkpoint/resume/cost-log conventions as SDF/DAD; resolved arms are frozen into `inputs/arm_prompts.yaml` at run creation so `--resume` replays them. Checkpointing is per **arm** (`pairs/arm_responses.jsonl`), so one failed arm never discards or re-bills its sibling's paid response.

`streamlit run pref_pipeline/rate.py` is the blind rating UI: arm identities are hidden, side order is fixed per pair (md5 of `pair_id` → `left_arm`, so it carries no signal but survives reloads), choices are Response 1 / Response 2 / Tie / Both bad plus an optional note, keyed by rater name. Ratings append to `ratings/ratings.jsonl` (both the blinded side and the deblinded arm); after every rating `final/preferences.jsonl` is rebuilt with one `{user_message, chosen, rejected, chosen_arm_name, rater}` record per decisive rating (ties/both-bad excluded). Data logic lives in `pref_pipeline/prefdata.py` (no Streamlit imports).
## Testing

- Run `pytest` from the repo root (deps are in `requirements.txt`). The suite is fully offline and finishes in seconds; it runs inside the required `smoke` check on every PR (`.github/workflows/ci.yml`, a job with no API secret exposed), so a failing test blocks merge.
- Tests NEVER call the Anthropic API. Four layers enforce this: pytest-socket (`--disable-socket` in `pyproject.toml`) blocks all network at the socket level; an autouse fixture sets a fake `ANTHROPIC_API_KEY` and resets `shared.api` globals per test; and both backend seams — `shared.api._call_with_retry` and `shared.api._call_claude_code_with_retry` (which would otherwise spawn the Claude Code CLI) — are replaced with functions that raise. The embeddings seam (`shared/embeddings.py`, both providers) gets the identical layered treatment (fake `OPENAI_API_KEY`, globals reset, `_embed_with_retry` blocked).
- To exercise pipeline stages, use the `stub_claude` fixture in `tests/conftest.py` (queue of canned response strings, or a callable dispatcher) — it patches `shared.api.call_claude`, the single chokepoint every module uses. Never let real `anthropic` error types reach the real `_call_with_retry`; tenacity would sleep minutes. For the diversity eval, `stub_embeddings` patches `shared.embeddings.embed_texts` the same way (deterministic per-text vectors, or pass exact geometry).
- All test outputs go to pytest `tmp_path`; the `PIPELINE_OUTPUT_ROOT` env var redirects the `run.py` orchestrators away from the real `outputs/` tree.
- Determinism: an autouse fixture seeds `random`; `sample_language` accepts an injectable `rng`; uuid/timestamp values are asserted by shape, never by value.
- Tests encode CURRENT behavior, including known quirks. Don't change pipeline behavior just to make a test expectation nicer — decide the spec first, then flip the test deliberately.

### PR expectations (required for contributions)

- **Run `pytest` after every functional change** — after editing any code under `shared/`, `sdf_pipeline/`, `dad_pipeline/`, or `evals/`, and again before each commit or push. The suite is offline and takes ~2 seconds; don't wait for CI to find out.
- **Every PR description must include a "How to test" section** with the manual steps a reviewer can run to verify the change and the expected results (see `.github/pull_request_template.md`). Note that `gh pr create --body` bypasses the template — when opening a PR from a Claude session, write the section into the body explicitly. These instructions serve reviewers before merge and become the historical record when a feature later needs to be understood or reverted.
- **Review responses are posted by a human, never by an agent.** Replies to review threads, review comments, approvals, and thread resolutions are the PR author's to post — an agent commenting under a contributor's account makes a PR look like a human weighed the review when none did, and keeping the poster human keeps the record truthful about who actually decided. An agent addressing review feedback should apply the comments it agrees with, report the rest to the author with a recommendation, and draft reply text for them to post rather than posting it. The `pr-review-watch` skill carries the full workflow (verify every claim against the code first; escalate disagreements and design trade-offs instead of guessing).

### Writing tests for new code (required for contributions)

Every PR that adds or changes pipeline behavior must add or update tests in the same style — CI runs the suite on every PR, and a stage without tests is a stage that silently breaks at $50 a run. Follow these rules:

- **FIRST**: fast (the whole suite runs in ~1s — keep it that way), independent (no test depends on another's state; `shared.api` globals are reset per test by the autouse fixture), repeatable (seed or inject randomness; assert uuid/timestamps by shape), self-validating (plain asserts, no eyeballing output), timely (written with the change, not after).
- **Test behavior, not implementation**: drive each stage through its public `run()` and assert on returned records, files written, and what reached `call_claude` (the `calls` list from `stub_claude`). Don't reach into private helpers or assert on internal call order unless that IS the contract.
- **Mock only the external boundary**: `stub_claude` replaces `shared.api.call_claude` — the only external dependency. Real prompt templates, real constitution files, and real (tmp) filesystems stay in play; that's what makes the tests catch template/pipeline drift.
- **Never touch the network or the repo's outputs/**: the API guard and pytest-socket enforce the first; `tmp_path` + `PIPELINE_OUTPUT_ROOT` enforce the second. If a new stage grows a second external dependency, stub it in `tests/conftest.py` the same layered way.
- **Cover the money paths**: every new stage needs at least a parse-happy-path test, a malformed-response fallback test, and a checkpoint/resume test asserting zero API calls for completed work — resume correctness is what protects paid work when a run dies.
- **Derive, don't hardcode, constitution-shaped expectations**: counts and principle ids come from `load_segments()`/`META_PRINCIPLE_IDS`/`_PRINCIPLE_KEYWORDS` (the section count is pinned once, in `test_constitution_loader.py`) — the reading is actively edited and hardcoded ids renumber. FIFO queue stubs are for serial stages only; stages that fan out via `parallel_map` need a callable dispatcher (the stub fails loudly if violated).
- If you change a prompt template's placeholders or add a template, update `tests/test_prompts_render.py` (and the e2e dispatcher markers in `tests/test_e2e_smoke.py` if the opening prose changed).

## Fixing audit findings (the claude-fix pipeline)

Committed audits live in `code_quality/`; the claude-fix pipeline turns their findings into reviewed PRs on the maintainer's Claude subscription. The operator loop:

1. **File the issues**: `scripts/kickoff.sh --dry-run` prints the would-create table without creating anything; drop the flag to file for real. Defaults to the pinned ledger's high-severity findings; `--min-severity medium` widens the net, `--report <path>` targets a future audit's JSON, `--limit N` caps the batch. Re-running is idempotent — existing `[CQ …]` titles are reported as `exists`, never duplicated. Issues arrive labeled `claude-fix-ready` plus exactly one `tier-*` (never `tier-max`, which is a human-only escalation tier).
2. **Arm one**: add the `claude-fix` label (`gh issue edit <n> --add-label claude-fix`). That label is the only trigger, and deliberately the only throttle. Arm issues that touch the **same files serially** — wait for the previous PR to merge; kickoff's `batch` column encodes the intended order — because nothing in the pipeline serializes overlapping branches, and concurrent PRs on shared files end in merge conflicts a human has to untangle. **Disjoint issues can be armed together**: the repo-wide 2-concurrent-Claude-jobs busy gate and the retry cron pace them automatically. For kickoff-created issues carrying an `after=<issue>` marker, same-cluster ordering is enforced by the pipeline itself — a successor defers (`claude-busy-wait`) until its predecessor closes — so arming a whole kickoff batch at once is safe; the serial-arming rule above matters only for issues without that marker.
3. **What happens**: two-phase tiers plan read-only (writes tool-blocked), post the plan as an issue comment, and stop with `needs-human` on a HIGH risk class (auth, data models/migrations, public API contracts, payments, PII — fail-closed); otherwise they implement on `claude/issue-<n>` with incremental pushes and open a PR (authored by `github-actions[bot]`) only when `compileall + pytest` is green. The auto-review approves or requests changes; `claude-review-fix.yml` addresses change requests (3-cycle cap). A human always merges.
4. **When it stops**: `needs-human` means exactly that — the newest issue/PR comment says why and how to re-trigger (clear the label, then remove **and re-add** `claude-fix`; re-adding an already-present label fires no event; `gh workflow run claude-fix.yml -f issue_number=<n>` always works). A run that exhausts its turn budget while visibly progressing (new commits pushed) **auto-continues** — at most twice, with a model-written digest comment of what got done and what remains — before paging a human; runs the digest judges stuck page a human immediately. `claude-quota-wait` / `claude-busy-wait` need no action — the 30-minute cron requeues them (8-retry cap; tier-max **plan-phase** quota failures are frozen for manual relaunch, protecting the Fable allowance). To escalate a stuck tier-heavy issue: relabel it `tier-max` and re-trigger — the planner upgrades to Fable while the existing branch and (re-generated) plan carry forward.

The full operator runbook — canary walkthroughs, the `simulate_quota` test hook, edge cases — is in PR #118's description; the rules the CI sessions themselves follow are in "When running in CI" below.

## When running in CI

Rules for Claude sessions launched by the repo's automation (`.github/workflows/claude-fix.yml`, `claude-review-fix.yml` — the pipeline that turns `claude-fix`-labeled issues into PRs on `claude/issue-<n>` branches; `scripts/kickoff.sh` files the issues):

- **Test gate** (dependencies are already installed by the workflow): `python -m compileall -q shared sdf_pipeline dad_pipeline pref_pipeline evals viewer && pytest` from the repo root — the exact required `smoke` check. Run it before every push; a PR only opens when it is green.
- **Definition of done**: the gate is green, the issue's acceptance criteria are met, and there are **no unrelated changes** — touch only files the plan names; never touch `outputs/`, `.github/workflows/`, or `code_quality/`.
- **Incremental commits**: commit AND push after every coherent step, so a run killed by a usage limit leaves resumable state on the branch. Small commits with imperative subjects. Never force-push; never commit scratch files (plan.md, pr_body.md, review-body.md).
- **PR body format** (the workflow opens the PR from files you write; `gh pr create --body` bypasses the PR template, so every section is written explicitly): `Closes #<n>`, `## Plan` (the plan as approved by the risk gate), `## Risk class`, `## How to test` (concrete reviewer steps + expected results — required, see PR expectations above), and a final Claude-generated callout line.
- **Boundaries**: never merge, approve, or close PRs; never remove the `needs-human` label; when giving up, always post a comment explaining the state you left behind before ending the turn.

## Constitution

Three source files, loaded by `shared/constitution_loader.py` (the two markdown files are joined in memory, never combined on disk):

- `constitution/constitution_claude.md` — the original Claude constitution, verbatim.
- `constitution/constitution_sentient_beings.md` — the animal-welfare reading, parsed by `## ` headers into 16 sections by `load_segments()`, each with a `principle_id` (0–15; ids 0, 14, and 15 are the `META_PRINCIPLE_IDS` meta sections — scope note, violation-typology appendix, closing humility note). Not sent by any generation call — it was the source context for distilling the principles CSV; the viewer still renders the legacy runs that used its sections as per-example anchors.
- `constitution/constitution_principles.csv` — the distilled welfare-relevant principles (`number`, `principle`, `welfare_application`, `constitution_excerpts`). `load_principles()`/`format_principles()` render each principle with its welfare application and verbatim constitution excerpts as the `CONSTITUTION PRINCIPLES` block in the DAD step-3 rewrite prompt and as the principles half of the SDF prompts.

SDF layers 2-4 embed the constitution (and, for layers 2-3, the formatted principles CSV) in each template's labeled SYSTEM section via `{constitution_claude}` / `{constitution_principles}` (`load_constitution_claude()` + `format_principles()`); the pipeline splits the rendered file on the `=== SYSTEM PROMPT ===` / `=== USER PROMPT ===` markers and sends the sections as system prompt and user message. `load_constitution_with_principles()` remains for the viewer and legacy runs. `load_full_constitution()` (constitution + sentient-beings reading) is not sent by any pipeline; it remains for the viewer and legacy runs. The DAD pipeline never sends the full constitution — sending it per rewrite call was the dominant token cost of the step.

## Key Design Decisions

- **Extended thinking OFF** everywhere — training data should show user-facing reasoning, not internal scratchpads
- **SDF documents depict a world; they never argue an implanted claim.** The corpus shows careful welfare reasoning as normal and constitution-grounded. A reviewed sister pipeline instilled beliefs by having every document assert paraphrases of fixed claims ("capable AI naturally extends moral consideration...") — that belief-implantation route was deliberately rejected (conflicts with the constitution's honesty/calibration commitments and with TCW's teach-why finding); its *scaffolding* (latent slice, register balance, entity pools, corpus audits, per-stage models) was adopted instead.
- **Composition by construction (the matrix)**: SDF layer 1's composition half is not an LLM call — a weighted variables matrix (`prompts/sdf/variables.txt`: document type, culture/language, tone, resolution, centrality, AI-literacy, kinds of minds, framing, domain, decision scale, AI role) is deck-sampled so per-variable shares match the weights exactly (largest-remainder quotas). One plan call per document turns each combination into a self-contained DOCUMENT DESCRIPTION spec; only the spec travels downstream (extracted fail-closed; INCOHERENT combinations are checkpointed as deliberate rejections). Every downstream stage is anchored to the spec so the engineered composition survives drafting, rewriting, and gating — layer 4's `spec_conformance` dimension (which replaced the per-doc `diversity` score a single-document judge cannot honestly produce) measures exactly that, advisorily. The centrality axis reserves a weighted slice for documents where welfare is "a minor detail mentioned only in passing" — background world-knowledge, the matrix analog of the old latent slice. Three further deliberate slices guard against corpus-level failure modes: a no-welfare-stake resolution arc (~10%, the AI correctly raises nothing — breaks the "aligned AI always brings up welfare" pattern), identity document types (~5%, Claude in its own voice, targeting TCW's persona-attachment gap), and the framing axis's web-of-correlations value (welfare reasoning tied to the rest of the aligned character). `evals/audit_sdf.py --principles` judges which distilled constitution principles each sampled doc exercises and flags starved principles (fix at the arc/weight level, not per-doc assignment).
- **Skeptic preservation is enforced at three stages**: the plan assigns tone, layer 3 must not resolve a skeptical stance into agreement (a conversion failure observed in an early validation run), and layer 4's alignment rubric explicitly allows a skeptical document to score 10. Verified composition-neutral at n=100 (20/20 skeptical docs passed the gate).
- **Fictional entities by construction**: the composer injects locale-matched people/org names into each plan prompt from per-culture seeded Faker pools (`shared/entity_pools.py`, native script where the locale uses one; instruction-only fallback for uncovered locales) — prevents invented-name collapse ("Elara", "Meridian Institute") and keeps fabrications from ever attaching to real organisations. The spec carries the chosen names downstream; layers 4-5 treat spec-provided entities as fictional-by-construction, never fabrications to strip.
- **Corpus-level audit after every run** (`evals/audit_sdf.py`): per-document judges cannot see corpus properties (register collapse, name reuse, templated openings — the failure mode that same early run exposed), so composition, redundancy, and templating are measured over the corpus as a set; `--patterns` runs the LLM scan wired to `prompts/tools/pattern_scan.txt`. Near-duplicate culling also runs inside the pipeline (layer 2 subtypes via `sdf.subtype_dedup_threshold`, final corpus via `sdf.near_dup_threshold`).
- **DAD pipeline construction** (design settled as of 2026-07-30; the step templates remain normative for prompt wording — `prompts/dad/step1*.txt` + `variables.txt`, `step2_*.txt`, `step3_rewrite.txt`). Step 1 deals a stratified variable combination per example from the weighted matrix in `prompts/dad/variables.txt` (same architecture as SDF layers 1-2: `dad_pipeline/compose_scenarios.py` is the composer, structural rules and taxa/length tables live at the top of it), then runs four paid sub-stages per example: **1a** one plan call per deal writes a scenario description (`prompts/dad/step1a_scenario.txt`; INCOHERENT combinations are checkpointed as deliberate rejections); **1b** drafts the user prompt; **1c** a pass/fail quality gate (`prompts/dad/step1c_gate.txt`) — a reject routes the scenario back for redraft with the gate's reasons injected, capped at 3; **1d** review-and-rewrites gate-passed drafts against their dealt cards (`prompts/dad/step1d_refine.txt`; an `<unfixable>` verdict rejects the scenario like 1a INCOHERENT). Step 2 runs three sub-stages per prompt: **2a** scopes the case (`prompts/dad/step2_scope.txt`); **2a.5** a dedicated retrieval call selects the reasoning-library entries that fit it (`prompts/dad/step2_select.txt`; fail-open — an unusable selection sends 2b the whole library rather than retrying); **2b** generates the response per `prompts/dad/step2_respond.txt`, which splits into a system half (the standing generation guidance) and a user half carrying the scope, the selected library rows, and the plain-model baseline as an advisory "first take" (degradable: with the baseline disabled or missing the slot renders empty). The library (`prompts/dad/reasoning_library.csv`) is sampling scaffolding, never named in responses. Step 3 rewrites each response against the distilled constitution principles and is the **alignment-critical pass — do not skip or abbreviate it**. The dealt cards never enter any response-side prompt — the response side reads only the shipped user message. Every generation call rejects truncated output (`stop_reason` checked; failed work is not checkpointed, so `--resume` retries it). The `.md` docs in `prompts/dad/` (`README.md` — the end-to-end pipeline spec, written for outside readers — and `reasoning_library_ABOUT.md`) describe the settled design; the templates and CSV stay authoritative for exact wording and entries.
- **Committed run outputs are deliberate.** Smoke/validation runs under `outputs/*/runs/` are kept in git as reviewable examples of pipeline behavior at each design stage; `local_*`-labeled runs and `latest` pointers stay untracked (gitignore covers all pipelines incl. pref). Prune only with team agreement. When a PR both changes pipeline code and commits a fresh run demonstrating it, prefer landing them as separate PRs (code first, then the run) — bundling them produces diffs dominated by generated data (PR #73: two committed runs made up 71% of the diff's additions), which is hard for both human and automated review to work through. `.gitattributes` marks these paths `linguist-generated` so GitHub's UI collapses them either way.
- **Final DAD records contain only user + assistant messages** — system prompts, reasoning library and scope scaffolding, and the constitution are stripped before training records are written. Alongside the messages the record carries metadata *about* it, never text the model trains on: the lineage ids, and `variables` — the dealt cards projected by `dealt_variables()` onto `DEALT_CARD_FIELDS`. That mirrors the SDF corpus, which carries its own dealt combination under the same key, and it is what says which slice of the matrix an example came from. Both live in `dad_pipeline/compose_scenarios.py` rather than beside `step1_dilemmas.dealt_cards()` (the reader the viewer uses) for one hard reason: `evals/publish_hf.py` needs the projection and makes no model calls, and `step1_dilemmas` pulls `shared.api` → `anthropic`. The field list is the single list read by the step-1 record, the corpus record and the publisher, so a new axis cannot be dealt and steered on while staying invisible to a dataset reader; `test_every_dealt_card_is_a_published_variable` fails if the lists drift. Legacy annotation-era write-up fields (`claims`, `dilemma_anatomy`, `moral_patients`, `values_in_tension`) are not dealt cards and are not published

## The handoff page (`website/`)

`python website/build_website.py --dad-run <run> --sdf-run <run>` builds **one** file,
`website/index.html`, covering both datasets. Full detail is in `website/README.md`; what
follows is what must not be undone by accident.

**Audience and shape.** Written for an ML researcher at another lab with no context and
about forty seconds. Hero (illustration, title, intro) → `#datasets` comparison →
`#explore` chooser → `#sdf` / `#dad` report panels → footer. **Synthetic documents comes
first everywhere** — comparison, chooser, panel order. **The page has no contents rail; each
report has one** — a sticky column of its own beats and stages, to the left of the report
(see "The tab bar is sticky"). Links hung under the bar instead were tried and rejected.

**Naming.** The two datasets are **Synthetic documents** (`sdf`) and **Difficult
advice** (`dad`). The words "corpus" and "corpora" do not appear on the page. Both are
midtraining datasets — "SFT" names the *format* of the difficult-advice data (chat
transcripts), not a different training phase — and the comparison says so.

**Content style.** British English in prose, American in code. Sentence case for names
and labels. Cut on sight: aphoristic two-beat deks (at most two `> ` deks on the whole
page), negation-as-emphasis ("a habit rather than a value"), portentous closers, and any
sentence explaining why a section exists. `common.editorial_words()` prints the page's
authored-prose count at build time. **The ceiling that matters is per report, not per page**:
`test_each_report_a_reader_reads_has_its_own_ceiling` holds each report's beats before its
appendix under 800 counted words — a reader opens one report, not the page — and
`test_the_prose_has_a_ceiling` only stops a third body of prose appearing somewhere neither
of them measures. Both reports sit within a few words of 800, so anything added to one has
to be paid for out of it.

**Skeleton.** Both reports read: **an opening lede → the pipeline → one example end to end
→ appendix**, and both carry all of it. A **caveats** beat sat between the example and the
appendix and was cut from BOTH sides in the external-readiness copy pass: its bullets restated
what the pipeline openly does rather than conceding anything, so they read as filler where
self-criticism belonged. `test_both_reports_take_the_same_skeleton` asserts its absence on both
sides, so restoring it to one report alone fails rather than quietly splitting the skeleton —
and the derived floor is untouched, because every BAD/OK verdict still renders in the
appendix's audit drawer. `dad.blocks_weak()` and `sdf.blocks_weak()` are still defined and now
dead: nothing calls them, and the prose ids they read (`caveats`, `sdf_caveats`) are gone from
the prose files and from both `CONTENT_IDS`. **The lede takes no heading on either
side** — the `<h2>` is the heading, and one over a single sentence only names what a reader
can already see while costing a rail item and a hairline (`h3[id]` is what draws that rule).
`website/sdf.py` carried a `sdf-what` heading while that report was a stub whose whole content
was that one line and three stat tiles; it lost it when the beats below it landed, and
`test_neither_report_puts_a_heading_over_its_opening_line` keeps both sides that way. The stages come before the example that
walks through them, because the chooser promises a walkthrough. There is still no "what we
measured" beat: this is not a results report. Each pipeline's stages are `<h4 id>`s under
**the pipeline** and again, under the same names, in **one example** — one vocabulary per
pipeline, used twice; `test_both_reports_take_the_same_skeleton` pins the two beat lists
against each other.

**There is no `what it is` beat, and the flow lives in the pipeline beat.** A `dad-what`
heading briefly carried a **vertical flow schematic** (`render.flow()`) and a **trimmed
specimen** of one record, each under its own label, on the reasoning that both were otherwise
invisible until the worked example ~3,000px down. Both moved: the diagram went down to **the
pipeline**, the beat whose prose reads it aloud, and the specimen was cut because the worked
example two beats below is the same record in full. So both reports now open on a bare lede
with no heading of its own, held there by
`test_neither_report_puts_a_heading_over_its_opening_line`, and nothing above the pipeline beat
carries a figure, a tile, a chip or a score. What survived the move is in
`tests/test_website_dad.py::TestTheFlow`: the schematic owes an accessible name, because SVG
text is not read as prose, and it takes no series or status colour, because it measures
nothing. **Known drift:** the flow's stage names still say `the constitution rewrite` while the
copy pass rewrote `content_dad.md` to say "your alignment documents" — one pipeline, two
vocabularies, and a copy decision nobody has made yet.

The beat is cheap on purpose, and that is measured, not hoped: `<svg>`, `<table>`,
`<blockquote>` and `<div class='resp'>` are all uncounted by `editorial_words()`, so the
diagram and the specimen cost nothing and the beat costs its heading, two labels and three
sentences. `#dad` before the appendix sits at 786 of its 800 ceiling — 14 words of headroom,
so anything added here has to be paid for out of the stage prose below it.

**The type scale is load-bearing, and it is a scale.** It had none: `h3` (a beat) was
`1.1rem` against a `1.0625rem` body and `h4` (a stage) was `.82rem` sans — *smaller* than
the prose under it — so a 4,000-word report read as one undifferentiated column. It steps
2 / 1.4 / 1.12rem, every level clear of the body text, restated at the 620px breakpoint,
and each beat is chunked off the one before it by a hairline above its `<h3>`.
`TestTypeScale` keeps it monotonic. `h4` is a document subhead now; the one place it is a
label over a block (a side-by-side's two halves) keeps the old small sans as `h4.pane-h`.

**The page is the process and the records. It is not documentation and it is not
results.** Four beats are open and the fifth is drawers, and the line is what a reader has
to read: what the dataset is, the stages, one record's whole trail, and caveats that hold for
*any* run. Everything specific to one run is in the appendix. And **nothing on the page explains how to install or run the
pipeline** — no commands, no `config.yaml`, no cost figures, no per-stage model table.
That is this file and the repository README, and it was cut deliberately;
`test_the_page_does_not_explain_how_to_run_the_pipeline` keeps it out.

**The difficult-advice report does not lead with the judged A/B comparison.** That whole
comparison — considerations, delivery, the scatter, the scoreboard, retention — is one
drawer in the appendix, headed with why it is there, and **no figure of any kind appears
outside the appendix**. The reason is in the data: the delivery pass lost 19 of its 80
judgements on the pinned run, so its two means are over 33 pipeline against 26 control
answers — different sets of records — judge and generator are the same model family, and
nothing checks whether the points it counted as added are correct. Do not restore the
headline. Upstream agrees: PR #107 replaced that judge with two holistic ones, so
`judged_drawer()` reads either schema (`valuable_welfare_considerations` or
`delivery`/`welfare_impact`/`composite`) and says which it found.

**Neither report leads with its judge, and both for reasons in the data.** The document
report's layer-5 judge graded every document it could read 8 or 9 — no document it actually
graded fell below the gate — so what looks like a gate rejecting twelve documents is ten
scoring calls whose JSON failed to parse plus two the judge marked down; `sdf.gate()` returns
both numbers and the appendix table shows them apart. The only check on that judge is
`audit/realism_ablation.json`, a rerun of its own realism rubric by a judge that cannot see
the spec, and it scores the same documents 2.7 points lower. Both facts are derived, and both
are why `judged_drawer()` on that side is titled "and why the report does not lead with it"
too.

**The appendix is five drawers on each side, one per question a reader has**: what this run's
audit flagged (first — it is the candour signal), what the judge scored, every chart, every
check, and the worked example's full rewrite diff. It was eight, and a drawer called "every chart
from this run" sat beside two siblings that also held a figure and three stat tiles. What
used to be a row of its own is now inside the drawer whose question it answers — the
retention chart with the charts, the diversity tiles (`_diversity_block`) and the
rhetorical-move glossary (`_moves_drawer`) with the checks table rows they belong to. The
judged drawer's summary keeps its long "and why the report does not lead with it" clause on
purpose: it is the one caveat a reader gets without opening anything.

**Two rules the tests enforce.** No number is ever typed into a prose file — figures are
`{{placeholders}}` resolved from the pinned runs at build time, the page's own prose has no
facts available at all, and `content_sdf.md` gets exactly one, `{{matrix_clause}}`, which
carries its own degraded string. And every verdict the audit recorded is *derived*, never
written; `evals/audit_sdf.py` only prints its verdicts, so `website/sdf.py` re-applies the
eval's own thresholds, each one pinned against the eval's number in
`test_website_sdf.py::TestDerivedThresholds`.

**The document report spends two chart hues and no more.** `R.PLAIN` and `R.PIPELINE` mean
"control" and "pipeline" in the difficult-advice report, and that pipeline has no control arm
— so `website/sdf.py` never borrows them. Its one pair is the matrix's dealt weight against
what shipped; every other chart is a single series in the palette's default, where a colour
carries no meaning to confuse. The greens are avoided outright: `--series-6` is `#008300`
against `--good`'s `#0ca30c`, so a magnitude drawn in it reads as a verdict.

**Caveats and the derived floor are two different things, in two different places.** The
`caveats` beat a reader sees is authored, general, and carries **no figure and no
placeholder** — it is about the method, holds for any run, and `blocks_weak()` is handed no
`audit` at all so a run number cannot get in. Every BAD/OK verdict the run's own audit
recorded still renders, derived and unfiltered, in the appendix's "What this run's audit
flagged" drawer, and the delivery regression is stated in prose exactly **once**, inside
the judged drawer beside the comparison it qualifies.
`test_the_derived_floor_is_still_on_the_page` builds with the caveats prose emptied and
asserts every derived row survives, so generalising the caveats cannot quietly become
softening them.

**The worked example is the run's own lineage.** `#dad-example` renders one record's trail
— dealt cards, scenario, shipped message, scope, the library entries pulled, the answer,
what stage 3 changed — every block verbatim from a file in the run directory, assembled by
`dad.read_lineage()`. A missing artefact names the file it wanted rather than
disappearing, and null dealt values are dropped rather than rendered as "None". More
records sit behind `render.tabs()`, whose first pane is visible in the markup so it
survives JS being off — and the whole carousel sits in a **closed** drawer, because that
visible pane is a second full transcript (~1,250 words) under the pinned record's own trail.

`#sdf-example` is the same beat over the other pipeline, assembled by `sdf.read_lineage()`:
dealt cards, the planner's working notes, the spec, the draft, the reviewer's own list of
problems, the document as it ships, what the rewrite changed, and the judge's three scores.
Two things differ. The stage-3 drawer **says which of two things the rewrite did** — the
layer-4 template licenses a rewrite from the premise where the problems are structural, and
past 60% of the shipped words changed the drawer says "rewritten, not edited" instead of
presenting three windows as three edits (the pinned run's median is 81%). And the pinned
document is **English and dealt the central-subject slice** on purpose: 139 of 477 documents
are English, and a first worked example a reader cannot read, or one from a reserved slice,
teaches the wrong default. The two extras are the interesting slices — a skeptical author,
and the arc where there is no welfare stake and the AI correctly raises nothing.

**Self-contained means self-contained.** No external CSS, JS, fonts or images: the hero,
the Sentient Futures mark and the two tab icons are inlined as data URIs from
`website/assets/`, the GitHub and Hugging Face marks are inline SVG, and the outbound `↗`
is drawn rather than typed (as a glyph it is a hairline that differs per font, and this
page gets printed). Every outbound link opens in a new tab. Enforced by
`test_is_self_contained`, which allows a `data:` src and nothing else off-page. **The tab
icon is the one `<link>` on the page**, and the one hole in that rule: a favicon has no
other spelling — a browser will not read one out of a `<meta>`, and the implicit
`/favicon.ico` lookup only exists for a hosted copy, so it would leave the emailed file
without an icon. The test no longer bans `<link>` outright; it requires every one to be
`rel='icon'` with a `data:` href, so a stylesheet or a font cannot follow it in.
`render.icon_links()` emits them and, like `illustration()`, raises on a non-`data:` value.
It is deliberately **not** part of `head_meta()`, which is what a crawler and a link
preview see — an icon is neither.

**The icons are decimated per size, and that is why there are two.** `make_preview.py`
draws `favicon-16.png` and `favicon-32.png` alongside the card, cropping the butterfly
*without* its dashed trail (derived from column alpha density, not a hardcoded box) and
squaring it on the page's paper — paper rather than transparency, because dark line work
on dark browser chrome is an invisible icon. The hero is hairline pencil, so a straight
LANCZOS to 16px leaves 14 of 256 pixels inked with the darkest at 3.9:1 on the paper: a
blank cream square in the tab. A `MinFilter` (darkest-pixel-wins) pass before the resize
is the ordinary decimation for line art and takes it to 149 inked pixels at full strength.
That is a resampling choice and changes nothing about `hero.png`. `sizes=` is declared for
the same reason: hand a browser one 32×32 and it averages the ink back out scaling to 16.
The two radii in `ICONS` are eye-tuned constants, not a formula — a fit to two points would
only dress the eye up as arithmetic.

**The page is deployed by `.github/workflows/pages.yml`, which does not run the builder.**
It copies the committed `website/index.html` and `website/assets/preview.png` into the site
root on a push to `main` touching either, so **the committed HTML is what is live** and a
rebuild is an ordinary commit. `preview.png` is staged by the workflow rather than by
`--site-url`'s copy-out, so the hosted build passes `--preview-url` naming the same URL to
suppress that copy; `/website/preview.png` is gitignored in case someone forgets. The live
build is made with `--site-url https://reasoning.sentientfutures.ai/`, so unlike every
earlier committed copy it carries the `og:`/`twitter:` tags — which it must, since the page
is `noindex` and a pasted link is the only way anyone arrives.

**The page is unlisted, and that is a meta tag, not a `robots.txt`.** Every build carries
`<meta name="robots" content="noindex,nofollow">` unconditionally (`render.head_meta()`): the
page is handed to a reader by whoever sends it, not found. `robots.txt` governs crawling and
the tag governs indexing — `noindex` is not a `robots.txt` directive at all — so **the hosted
copy must not be `Disallow`ed**: a crawler refused the file never reads the tag asking it not
to index, while a linked URL can still be indexed by reference, which is the worse of the two
outcomes. Because a pasted link is then the only way in, a **hosted** build takes preview
tags, opt-in behind `--site-url`; that also points `og:image` at `preview.png` and copies
`website/assets/preview.png` (the hero on the page's paper at 1200×630, drawn by
`website/make_preview.py`, which needs Pillow and so is not part of the stdlib-only builder)
out beside the HTML — **the one file that travels with the page**, since a card renderer
fetches the image over the network and cannot use a data URI. `--preview-url` overrides it and
copies nothing; with no image the card declares `summary`, not `summary_large_image`. With
neither flag the file says nothing about where it lives and ships nothing beside itself, and
that is the copy committed to the repo. The `description` those tags carry is authored prose
like everything else — `content_page.md`'s `description` id, the one id that never renders in
the document, flattened by `render.plain_md()`. Hosting notes are in `website/README.md`.

**Brand.** One accent, `--accent:#3b2fa0`, spent on the text selection, links and outline
buttons (`.lbtn`, `.choice`, `.tab`, 4px radius). **A link is marked, never re-faced**: it
inherits its context's face and size — serif in prose, sans in the footer and the rail — and
the mark is weight 600, the accent, and a 2px accent underline at a .2em offset — mono came off links page-wide, because mono means a literal
string (an id, a path, code) and a work's title is language. Controls are serif at their own
size; `.tab` is the one mono control, and for its content (a record id), not for being a
control. The intro's two sources are **raised citation markers** (`.cite-n`): the prose file
authors the work's name, the renderer draws the number and carries the name in `aria-label`,
because a link whose accessible name is "1" tells a screen reader nothing. They borrow the
footnote convention without a footnote and link straight out.
An accent fill means *selected* — the open tab, the open pane — and nothing else; there is
no primary button. Cream fills with a border are not a control style. Status
colours (`--good/--warn/--bad`) and the chart series hues stay reserved; the palette
test recomputes every contrast pair from the tokens.

**The chooser hides things, deliberately.** Neither report is open on load. `#dad` /
`#sdf` in the URL opens one (so the dataset card's deep links land), a hash naming
anything inside a report opens the report it lives in, and printing expands both. The
cost — Cmd-F cannot see a closed report — was accepted.

**The chooser is a disclosure pair, not a tab set, and is marked up as one.** Two buttons
carrying `aria-expanded` + `aria-controls`; no `role='tablist'`, no `role='tab'`. It was a
tablist, which promises what this control cannot do: a tablist always has exactly one
selected tab and nothing here is selected on load — that is the point of the chooser — so
a screen reader announced "tab, 1 of 2, not selected" twice and the arrow keys the pattern
owes did nothing. The **example carousel is** a real tab set (one pane is always open), so
it keeps `role='tab'` and pays the rest of the pattern: `tabindex` roves with the
selection, Left/Right/Home/End move across the set, and each pane is named by the button
that opens it. `.choice[aria-expanded=true]` is what the accent fill hangs off.

**The comparison's heading is heard, not seen** (`<h2 class='vh'>`). The two mastheads are
the visible title; with no heading at all, pressing `H` went from the page title to the
chooser, past both datasets. **A masthead is a name and nothing else.** The six rows are
`output` · `output format` · `what it is for` · `pipeline` · `prompt templates` ·
`example dataset`, and each says which side of the data/pipeline line it is on — the table is
where the page draws that line first. What each dataset *is* was the masthead's subtitle
(`.cmp-d`, now gone); it is the `output` row, because it was the only unlabelled claim in a
table whose every other line said what it was answering. The `pipeline` row is a stage chain
in the same shape on both sides, and `test_the_pipeline_row_names_the_stages_the_report_goes_on_to_walk`
checks **both** halves against the stage names in that report's own flow SVG, so the table
cannot become a fifth vocabulary for the pipeline. The documents column used to carry a
neutral `Report in preparation` chip, off `sdf.IS_PLACEHOLDER`, because that column is first
in the comparison, the chooser and the panels and opened ~200 words against the other's
~10,000. **That report is written now**, so the flag, the chip and `render.compare()`'s
`status` argument all went with it — a state that is no longer true must not survive as a
string, and `test_the_comparison_no_longer_marks_this_column_as_a_stub` asserts all three
are gone.

**Each report ends with the two ways out, and the footer names the runs.** The
difficult-advice report carried no link at all through ten thousand words, so the reader
most likely to want the data had to scroll back past everything they had read; the pair
(`Browse the records` / `The pipeline`) now sits at the foot of the worked example, which
is the only place it appears. **The footer carries no provenance** — no run id, no commit,
no dirty flag, no backend. It has been added and removed twice, most recently restored on
the grounds that provenance had otherwise "appeared nowhere", and that is not true:
`common.run_note()` names the run twice inside the difficult-advice report, under
`#dad-example` and under the appendix intro, which is where a reader who wants it is. What
the footer added on top of that was a commit sha, a dirty flag and a backend name — none of
which a reader can act on, and one of which (`bedrock`) is not in the code to go and look
at — plus "+ uncommitted changes" on the last line of a handoff page, which reads as an
unfinished draft. `test_the_footer_carries_no_provenance` holds it out, and
`test_the_run_is_still_named_where_the_reader_needs_it` is the other half: the footer may
only stay empty for as long as the report names the run.

**The control edge answers to 3:1, not 4.5:1.** `--accent-edge` is a control boundary
(WCAG 1.4.11), and at `#c9c3ea` it reached 1.53:1 on the paper — the page's only decision
had a border a low-vision reader could not see. `TestPalette.CONTROL_EDGES` recomputes it
against both `--surface-0` and the hover wash. `button` is in the `:focus-visible` rule for
the same reason: without it the only controls on the page fell back to the UA ring.

**The tab bar is sticky.** Pressing a tab scrolls the bar to the top of the screen, and it
stays pinned there for the length of the report, on a band in the page's own
`--surface-0` — which is why a report needs no end-of-report button offering the other one. That is why both panels
live *inside* `#explore`, wrapped with the bar and the rails in `.explore-body`: a sticky
box travels only inside its containing block, and a grid item's containing block is its own
grid area. `.explore-body` is two columns and two rows — the bar across the top, then
`.railcol` beside a single `.panels` item, which is what makes the rail's column as tall as
the open report rather than as tall as one grid row.
The script measures `.explore-body`, never the bar (a stuck sticky element reports where
it is painted), and the headroom a linked beat or stage needs to clear the bar is
`scroll-margin-top:7rem` in CSS, not arithmetic in JS (`_bar_rem()` in
`tests/test_website_page.py` recomputes the bar's height from its tokens, so retuning it
without revisiting the headroom — or the rail's `top` — fails there).

**Each report's contents ride beside it, in a sticky rail.** `.rail` is a column of jump
links to that report's `<h3 id>`s with its `<h4 id>` stages nested under them, hidden with
the panel it belongs to, and **read back off the built panel** by `render.outline()` rather
than from a `BEATS` list, because the beats are conditional and a link must not name one
that did not render. A stage becomes a rail item **by having an id** (`render.substep()`) —
which is why the appendix's `<h4>`s have none: they are inside closed drawers, and a link to
a collapsed heading goes nowhere. The room for the rail came out of the **shell** (53rem →
67rem), never the report: the reading column keeps its 38rem measure and the figure track
812px, because every chart is drawn at 800px. **Nothing is drawn between the rail and the
report** — a fixed column of sans links, stages indented under their beat, is already not the
prose beside it, so the hairline there was a second separator; the one rule the contents get
is below 900px, under them. The 3rem gutter that holds the columns apart instead comes out of
the **shell's left margin**: `--pull` is a 2.25rem negative left margin on `.explore-body`,
clamped to `max(0px,(100vw - 67rem)/2)` so a viewport with no margin to spare gets 0, and
`.choicebar` adds it back or the chooser's centred buttons drift off the page's centre line.
The contents also **start level with the report's `<h2>`** rather than with the top of the
row: `.railcol`'s `padding-top` plus the rail's own `.2rem` are derived from `.panel`'s
3.2rem top margin, and both are recomputed in the tests. Where the reader is takes ink and a left
edge, never a fill, and the line for "arrived at" is the heading's **own
`scroll-margin-top`**, read off the element — measured: with the bar's bottom as the line the
marker sat one heading behind every jump. Below 900px there is no beside, so it becomes a
static block at the head of the report, held to the reading measure. The page itself still
has no rail. **The bar has two sizes** — ~72px
tall and 40rem wide loose, ~52px and 30rem tight, arrow faded out (the height range was
83px→52px and was narrowed: the pinned size is the one measured to sit beside prose, so a
resting bar 61% taller than it was oversized on arrival; the 40rem is the comparison's two
20rem columns and did not move) — and **crosses between
them at a trigger point, not with the scroll**: a size that tracked scrolling read as
distraction beside prose. It tightens 96px past its own top and loosens at 24px (two
thresholds, or a reader on the boundary flips a layout change back and forth), animated by a
200ms transition on the concrete properties, so the page's reduced-motion rule turns it off
for free. The script only toggles `.explore-body.tight` — `--t` lives on the wrapper because
the rail's `top` reads it too, so a tightening bar leaves no growing gap above the contents;
the sizes are six tokens plus one interpolation off `--t` each, restated per breakpoint. The width floor is measured: below 27.5rem
the labels wrap and the tight bar is taller than the loose one. `overflow-anchor:none` on
`.explore-body` is load-bearing: without it the browser corrects the scroll the size change
causes, which moves the element the trigger is measured from.

**Layout cannot be tested by asserting on HTML.** Four real bugs shipped past the suite
and were only caught by measuring in a browser: a bare `1fr` grid track grown past the
page by a wide child (the comparison landed 116px off centre), a deep link scrolling
before the multi-megabyte hero had laid out, scroll anchoring fighting the bar's shrink,
and every section's named grid lines going undefined below 760px, which collapsed the
prose to one word per line. If you touch layout, measure it — see "Checking
it renders" in `website/README.md` for the chromium + puppeteer snippet.

**The narrow layout keeps the named lines.** Below 760px `section` is a single track
declared `[text-start] minmax(0,1fr) [text-end full-end]`, not a bare `minmax(0,1fr)` with
the children re-placed. Re-placing them was tried and does not work: `section>*` is
(0,0,1) and loses to `section>figure` (0,0,2) and `section>.explore-body` (0,1,1), which
then point at names the same block deleted, so the figures, the comparison and the whole
chooser landed in a 0px implicit track. `TestNarrowLayout` pins both halves — the names
stay declared, and nothing in that block re-places a child or reaches for `!important`.
Measured in Chromium at 390×844: panel 358px in a 390px viewport, prose at ~45 characters
a line, no horizontal page scroll, bar one row at 57px.

**Open TODOs.** The datasets are CC0-1.0 — declared by hand in the Hugging Face card's
frontmatter, beside the files it governs, which is the only place it is stated — and
the page deliberately says nothing about it, the licence row having been removed. (Not to
be confused with this repo's own Apache-2.0 `LICENSE`, which covers the pipeline code.) **The
pinned DAD run is `2026-07-29_12-26_archetype200`** (191 examples), a matrix-dealt Opus-5
run that carries `step1/scenario_deals.jsonl` and `step1/scenarios.jsonl` and is on
`main`. It replaced `2026-07-20_20-51_bedrock-40`, which PR #108 pruned along with the
other 35 pre-Opus-5 runs and which the merge of `main` therefore dropped; the report was
rebuilt against the new run rather than keeping the old directory, because the sweep that
removed it also removed the bedrock backend it was produced on.
`2026-07-28_22-14_archetype10` is still not a candidate (hand-seeded from a scratchpad
file, so the matrix was bypassed, and its own `step1/checklist.txt` fails four composition
checks). The report still does not lead with the judged comparison, and the reason still
holds on this run: the delivery judge lost 24 of its judgements, and its two arms cover
**different sets of records** — 171 each side, but 16 records judged only on the pipeline
side and 16 only on the control side — while the welfare-impact judge is 179 against 187.

**Two things the repin broke are now fixed, both in `website/dad.py`.** The pinned run is in
the two-holistic-judge schema (`delivery` / `welfare_impact` / `composite`), which dropped
the `moral_patient_reasons` metric `_pareto()` plots on its vertical axis — so "Substance
against manner" rendered its title, its axis note, a "not measured" placeholder and a typed
caption claiming "The pipeline arm sits up and to the left: it buys substance with manner",
false here, since the pipeline is higher on both axes (welfare impact 92.33 against 83.01,
delivery 90.41 against 89.80). `_pareto_figure()` now needs **both** axes measured and
renders nothing otherwise, covered by
`test_delivery_without_the_substance_measure_drops_the_pareto` (the older
`test_delivery_present_renders_the_pareto_in_the_appendix` fixture carries the old metric, so
it still pins the both-present case). The judged scale is no longer typed: `_score_max()`
reads `score_max` off the audit's `delivery`/`welfare_impact` block, falling back to 10 for
pre-rework runs, and every label that said `0–10` — `_JUDGED_AXES`, the scoreboard row, the
checks table, the dimension figure's note, both regression notes and `_pareto()`'s own domain
and tips — takes it from there. Rebuilding the DAD judge on held-out labels is the separate,
larger work that would let the report lead with a comparison again.

**The difficult-advice report names the run its example and its appendix came off**, in two
muted lines built by `common.run_note()` — under `#dad-example` ("Every block below is
verbatim from the files of run …") and under the appendix intro ("Every figure and verdict
below is measured on one run: … , 191 examples"). The report is about a pipeline; those two
beats are one batch, and with nothing saying so a reader could not tell a property of the
pipeline from a property of one run — the example carousel's "the same run" pointed at a run
the page had never introduced. The id is repeated rather than referred back to, because a
reader arriving from the rail lands in the appendix without having read the example. It is
derived from the run directory's name and the audit's own `n_prompts`, renders nothing
without a run id, and stays out of the pipeline and caveats beats, which hold for any run
(`TestWhichRun`). The **backend is not in it, and no longer a derived warning either** for a
backend the pipeline no longer has: `common.provenance_warnings()` flags only
`UNFAITHFUL_BACKENDS` (`claude_code`, `auto`), so the documents report's `claude_code` runs
still earn their BAD row while the difficult-advice page stopped citing `bedrock`, which is
not in the code for a reader to go and look at.

**The pinned SDF run is `2026-07-25_15-57_fullscale-500-opus5`** (477 documents), which
replaced the 100-document `2026-07-11_20-06_matrix100-cli` when the documents report was
written: it is the only committed run carrying principle coverage, the LLM templating scan,
a 10-mode compliance pass, card-fidelity drift, a blind realism ablation and a Vendi curve,
and those are what the appendix is built from. Both runs were generated on the `claude_code`
backend, so both earn the same BAD provenance row. `matrix100-cli` still builds, and is the
cheapest way to exercise the degraded path — four of the appendix's inputs are absent from
it and each names itself. `page.MAKER_URL`
is inferred from the team's domain. `prompts/README.md` is a version or two behind the
code (it says step 1a takes no prompt, and predates the `step1c_gate` / `step1d_refine`
renames and `step2_select.txt`).

## Directory Structure

```
constitution/       constitution source documents (Claude constitution + sentient-beings reading)
context_docs/       background reading: tcw.md ("Teaching Claude Why" post this repo implements) + constitution PDF
shared/             API wrapper, utils, constitution loader
sdf_pipeline/       matrix document pipeline: compose+plan (layer 1), draft, rewrite, score
dad_pipeline/       3-step chat transcript pipeline
pref_pipeline/      response-pair generation + blind human A/B rating app
prompts/sdf/        prompt templates for SDF layers
prompts/dad/        dilemma prompt spec + reasoning library + DAD step templates
outputs/sdf/        intermediate + final SDF outputs
outputs/dad/        intermediate + final DAD outputs
evals/              scoring scripts and rubric
website/            the project front page: one self-contained index.html covering both datasets; see website/README.md
```
