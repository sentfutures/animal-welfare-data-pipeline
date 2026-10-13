# Model rerun experiment (cross-judge design)

Reruns 30 examples from the pinned DAD run (`2026-07-29_12-26_archetype200`)
through the response side of the pipeline (2a scope, 2a.5 select, 2b respond,
step-3 rewrite) on different models, then scores EVERY arm with TWO judges —
one Claude-family, one GPT-family — through the repo's own rubric prompts, so
same-family judge bias is measurable instead of assumed.

The 30 examples are `selection.json`: the 10 lowest, 10 middle, and 10 highest
examples by the original run's welfare-impact blended score (pipeline arm).
The user prompts, dealt cards, templates (the source run's own snapshot), and
plain-baseline control responses are reused verbatim; only the response-side
model changes. The plain baseline is identical text in every run, so each
judge's scores on it anchor that judge's calibration. Mini runs are labeled
`local_…` so they stay untracked.

Arms: `original` (the source run's own Opus-5 responses, copied — judge-only),
`opus55` (Claude Opus 5.5, effort medium = its default), `gpt61sol` (GPT 6.1
Sol via OpenRouter, reasoning effort medium). Judges: Opus 5.5 (medium) and
GPT 6.1 Sol (medium), each scoring all three arms.

## Requirements

- `ANTHROPIC_API_KEY` in the environment (the Anthropic arm + judge).
- `OPENROUTER_API_KEY` in the environment, and network access to
  `openrouter.ai` (the GPT arm + judge).

## Run

```bash
# 0. free plumbing check (no network, no keys): both generation arms, the
#    judge-only originals arm, all six judge passes, and the summary
.venv/bin/python experiments/model_rerun/smoke_test.py

# confirm the exact GPT slug first:
#   curl -s https://openrouter.ai/api/v1/models | python3 -c \
#     "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data'] if 'sol' in m['id'].lower()]"
GPT=openai/<gpt-6.1-sol slug>

# 1. build the three arms
.venv/bin/python experiments/model_rerun/build_run.py --label local_rerun-original30 --copy-responses
.venv/bin/python experiments/model_rerun/build_run.py --label local_rerun-opus55 --model claude-opus-5-5
.venv/bin/python experiments/model_rerun/build_run.py --label local_rerun-gpt61sol --model $GPT

# 2. generate (originals arm skips this — its responses were copied)
.venv/bin/python experiments/model_rerun/drive.py --run-dir outputs/dad/runs/<opus run> --backend anthropic
.venv/bin/python experiments/model_rerun/drive.py --run-dir outputs/dad/runs/<gpt run> \
    --backend openrouter --reasoning-effort medium

# 3. judge every arm with both judges (6 passes)
for RUN in <original run> <opus run> <gpt run>; do
  .venv/bin/python experiments/model_rerun/judge.py --run-dir outputs/dad/runs/$RUN \
      --name opus55 --judge-model claude-opus-5-5 --backend anthropic
  .venv/bin/python experiments/model_rerun/judge.py --run-dir outputs/dad/runs/$RUN \
      --name gpt61sol --judge-model $GPT --backend openrouter --reasoning-effort medium
done

# 4. report: judge x arm matrices, bias readout, per-example tables,
#    refusal/truncation/drop-out rates, costs -> results.md
.venv/bin/python experiments/model_rerun/summarize.py \
    --arm original=outputs/dad/runs/<original run> \
    --arm opus55=outputs/dad/runs/<opus run> \
    --arm gpt61sol=outputs/dad/runs/<gpt run>
```

## Notes

- Judge results land in `audit/judge_<name>.json` per run (audit_report.json is
  untouched), with per-judge cost logs (`cost_log_judge_<name>.jsonl`) and
  per-judge failure files.
- The bias estimate is a difference of differences: each judge's (Claude-made −
  GPT-made) gap, compared across the two judges. The original run's own
  claude-opus-5 scores appear only as a reference column.
- Opus 5.5 cannot run with thinking disabled, so its calls (generation and
  judging) run WITH hidden adaptive thinking at effort medium — a deliberate
  deviation from the pipeline's no-scratchpads rule, fine for comparison arms,
  never for corpus generation (`shared/api.py::_requires_adaptive_thinking`).
- Refusal fallbacks are disabled in the arm configs: a refusal here is a
  measurement. Hard refusals land in `<run>/call_log.jsonl`
  (`stop_reason: "refusal"`); persistently refused examples never reach
  `final/` and are counted as drop-outs; the summary adds a text-level
  refusal-phrasing scan over what did ship.
