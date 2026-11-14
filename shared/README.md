# shared/

Code both pipelines (and the evals) use: the model-call wrapper (`api.py`, with `openai_api.py`,
`gemini_api.py` and `embeddings.py` behind it), key loading (`api_keys.py`), run folders and
helpers (`utils.py`, `sdf_run_layouts.py`), the constitution loader (`constitution_loader.py`,
see `constitution/README.md`), deal constraints and the variables matrix (`deal_constraints.py`,
`matrix.py`), entity name pools (`entity_pools.py`) and cost aggregation (`costs.py`).

This README holds the setup and call-wrapper detail that the repository's `CLAUDE.md` keeps only
in outline.

## API keys

See the repository README "Setup" (venv + `pip install -r requirements.txt`, then
`cp .env.example .env`). Every key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
`HF_TOKEN`, and the optional `CLAUDE_CODE_OAUTH_TOKEN`) goes through `shared/api_keys.py`: by
default only `<repo root>/.env` is consulted (no upward search, no current-directory lookup,
never the OS environment) and a missing file or key is an error naming the path and key; every CLI
that can reach a key takes `--find-dotenv` (search upward for a `.env` as python-dotenv's
`find_dotenv` does, then fall back to the OS environment; for servers), applied through
`add_api_key_source_flags`/`apply_api_key_source_flags` in `shared/api_keys.py`, and
`dad_pipeline/run.py` forwards the chosen flag to its post-run eval subprocesses; the source is
printed once per variable, never the value.

Auth depends on the `backend` key in `config.yaml`: `api` reads `ANTHROPIC_API_KEY`;
`claude_code` bills the contributor's Claude subscription via the Claude Code CLI (logged-in
session or `CLAUDE_CODE_OAUTH_TOKEN`); `auto` prefers the subscription and falls back to the api
key — per-call: empty-system calls (the DAD baseline arm) always take the api leg so the
plain-model condition stays exact, and an exhausted usage window demotes the rest of the run to
the api, loudly, with each cost-log record naming the backend that served it. `api` stays the
committed default: it is the faithful mode (subscription-served calls don't enforce `max_tokens`
and carry CLI scaffolding in context), so runs meant to represent pipeline behavior stay on
`api`; `auto`/`claude_code` are dev-iteration modes. The caveats for the dev backends are below.

- `OPENAI_API_KEY` is **required for a default SDF run**: the description rewrite defaults to
  `gpt-6.1-sol` (an OpenAI model), and `sdf_pipeline/run.py` exits before any paid call if the key
  cannot be loaded through `shared/api_keys.py` (set `sdf.description_rewrite_enabled: false` to
  run without it).
- `GEMINI_API_KEY`/`OPENAI_API_KEY` are also read by `evals/diversity.py` (embedding-based
  diversity audit; either provider works, and the runs to date used Gemini).
- `evals/publish_hf.py` (Hugging Face dataset publishing) reads `HF_TOKEN` from the key source
  above if set (optional, never an error; in the default mode an `HF_TOKEN` in the OS environment
  is ignored too); otherwise it falls back to a one-time `huggingface-cli login`, whose cached
  token `huggingface_hub` picks up on every later call. Either way the token needs write access to
  the target repo/org.

## Dev backends (internal — not documented in the public README)

`backend: claude_code` routes calls through the Claude Code CLI, billed to the contributor's own
Claude Max/Pro subscription instead of the API key. Two ways to give it credentials: an existing
interactive Claude Code login is picked up automatically, or `claude setup-token` prints a token
for `CLAUDE_CODE_OAUTH_TOKEN` in `.env` (a subscription-tied OAuth token valid ~1 year, not a
Console API key; use this path for CI or any non-interactive machine). `shared.api.init` loads it
through `shared/api_keys.py` for the `claude_code` and `auto` backends only (never for `api`),
with the same source rules as the other keys: by default only the repo-root `.env`, so a token
exported in the OS environment is ignored; with `--find-dotenv` the environment is used too. The
value is passed to the `claude` subprocess's environment and never printed.

Caveats, all reasons to keep this to dev-scale runs:

- **Usage limits.** A 5-hour rolling window plus a weekly cap, shared with your interactive
  Claude Code use. A full-scale run will exhaust the window; the run stops with a clear message
  and progress is checkpointed, so `--resume` continues.
- **Per-call overhead.** ~3K input tokens of CLI scaffolding per call and a CLI process per
  request, so calls are slower. `max_tokens` from `config.yaml` is not enforced (Claude Code
  applies its own output cap), and `cost_usd` in the cost log is notional — what the run would
  have cost at API prices.
- **Empty system prompts get a neutral stand-in.** Claude Code substitutes its own agentic CLI
  prompt when the system prompt is empty, so the one empty-system call in the pipelines (the DAD
  baseline) gets a one-line neutral prompt instead (see `_NEUTRAL_SYSTEM` in `shared/api.py`) and
  is **not reproduced exactly**. On `auto` the baseline always takes the API leg for this reason.
  A one-time warning prints when the substitution happens.
- **Policy.** Anthropic's docs steer programmatic workloads toward API keys. Running this
  pipeline on a personal subscription is the same posture as using Claude Code itself, but it is
  a gray area: dev-scale only, and never for runs whose outputs represent the pipeline.

`shared/__init__.py` enforces a Python floor (`MIN_PYTHON = (3, 12)`, matching numpy) at import —
bump it there if the deps' floor rises. `.venv/` is gitignored.

## How `call_claude` behaves

- **Schema-constrained output**: `call_claude(json_schema=...)` sends the schema as
  `output_config.format` on the api backend and as the SDK's `output_format` on `claude_code`
  (returning the result's `structured_output` as JSON text, ignoring the free-text preamble);
  callers must still validate. Without a schema nothing changes. Model-facing strings live in
  `prompts/` files, not Python.
- **Extended thinking OFF** everywhere — training data should show user-facing reasoning, not
  internal scratchpads (one exception: models that require thinking, Opus 5.5 among them, run with
  adaptive, summarized thinking, which `shared/api.py` turns on for them and records in
  `model_calls.jsonl`; thinking blocks are stripped from the returned text and never stored in a
  document).
- **Any stage can run on a non-Claude model**: a model knob naming a `gpt-` model routes through
  `shared/openai_api.py` and a `gemini-` model through `shared/gemini_api.py` (`GEMINI_API_KEY`),
  both from inside `call_claude`, with the same `(text, stop_reason)` contract and cost-log
  records. The optional `effort` key in `config.yaml` (one level, or a stage-tag → level map) is
  sent as `output_config.effort` / `reasoning.effort` / `thinkingLevel`; models that always reason
  get `thinking_headroom` (default 8000) added to `max_tokens`. Both were added for the 2026-10/11
  model-per-stage experiment, whose winners (Opus 5.5 at every DAD stage, Sonnet 5.5 as the 1c
  gate, every DAD stage at effort `high`, headroom 13000) are the committed `config.yaml`
  defaults; SDF and the evals are unchanged.
- The global `temperature` (1.0) is wired into every call; generation wants 1.0 (diversity is the
  product — 1b register variety, 2b independent samples), and `call_claude` accepts a per-call
  override for eval/debug use.

## Run organisation

Each pipeline invocation creates a fresh run directory
`outputs/{sdf,dad}/runs/<YYYY-MM-DD_HH-MM>_<label>/` containing the per-stage directories (SDF:
see `sdf_pipeline/CLAUDE.md`, "Run layouts"; DAD: `step1`–`step3`; steps 2–3 keep explicit
checkpoints, step 1 resumes from its own append-only jsonl files; DAD runs also hold `baseline/` —
a plain-model response per dilemma serving as the viewer's control arm and as the advisory "first
take" in the 2b prompt, never trained on; toggled by `dad.baseline.enabled`, see
`dad_pipeline/baseline.py`), `final/`, `run_manifest.json` (label, git commit + branch + dirty
state, model, full config snapshot; `manifest_version` 3 added `git_branch`, so every earlier run
has a commit but no branch), and a per-run `cost_log.jsonl`, beside which `model_calls.jsonl`
records every model call in full (user messages, raw response text, thinking summary, stop reason,
usage, attempts) and `system_prompts/<sha256>.txt` holds each distinct system prompt once. This
keeps outputs from separate runs isolated — checkpoints live inside the run dir, so `--resume`
(latest run by default, or `--run-id`) continues exactly one run. The label is purely descriptive
(`dev` by default; scale knobs stay in `config.yaml`). An `outputs/<pipeline>/latest` symlink
always points at the most recent run (gitignored, as are `local_*` run dirs, for every pipeline
including pref). Run-scoping helpers (`create_run_dir`, `resolve_run_dir`) live in
`shared/utils.py`.

## Cost tracking

Running cost is tracked per run in `outputs/{sdf,dad}/runs/<run_id>/cost_log.jsonl` (evals log to
the global `outputs/cost_log.jsonl`) — check it any time. Each record carries a `stage` tag
(`prompt_draft`, `layer4_draft`, `constitution_rewrite`, …) matching the model-knob names; the
viewer's run list renders the per-stage cost breakdown (pre-tag records show as "(untagged)").
Records also log `duration_s` and `attempts` (API-retry count), and DAD calls tag an `item_id`
naming the record served (scenario_id for 1a/1b/1c/1d — pre-rework runs comma-joined a 1b batch's
ids — prompt_gid for 2a/2a.5, `{prompt_gid}_s{n}` for 2b, response_id for step 3; older runs' cost
logs key the 2a/2b slots by their retired per-run prompt ids); the viewer's lineage page reads
these via `loader.call_stats` to show model · cost · time · retries in each step expander (runs
logged before these fields fall back to a model-only note).
