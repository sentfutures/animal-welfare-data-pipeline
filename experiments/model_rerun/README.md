# Model rerun experiment

Reruns 30 examples from the pinned DAD run (`2026-07-29_12-26_archetype200`)
through the response side of the pipeline (2a scope, 2a.5 select, 2b respond,
step-3 rewrite) on different models, then scores each arm with the repo's own
judges and reports scores plus refusal rates.

The 30 examples are `selection.json`: the 10 lowest, 10 middle, and 10 highest
examples by the original run's welfare-impact blended score (pipeline arm).
The user prompts, dealt cards, templates (the source run's own snapshot), and
plain-baseline control responses are reused verbatim; only the response-side
model changes. Mini runs are labeled `local_…` so they stay untracked.

## Requirements

- `ANTHROPIC_API_KEY` in the environment (the Anthropic arm, and the judges).
- `OPENROUTER_API_KEY` in the environment, and network access to
  `openrouter.ai` (the GPT arm).

## Run

```bash
# 0. free plumbing check (no network, no keys)
.venv/bin/python experiments/model_rerun/smoke_test.py

# 1. build + drive the Anthropic arm (Opus 5.5; effort defaults to medium)
.venv/bin/python experiments/model_rerun/build_run.py \
    --label local_rerun-opus55 --model claude-opus-5-5
.venv/bin/python experiments/model_rerun/drive.py \
    --run-dir outputs/dad/runs/<printed run id> --backend anthropic

# 2. build + drive the OpenRouter arm (confirm the exact slug first:
#    curl -s https://openrouter.ai/api/v1/models | jq -r '.data[].id' | grep -i sol)
.venv/bin/python experiments/model_rerun/build_run.py \
    --label local_rerun-gpt61sol --model openai/<gpt-6.1-sol slug>
.venv/bin/python experiments/model_rerun/drive.py \
    --run-dir outputs/dad/runs/<printed run id> --backend openrouter

# 3. judge each mini run with the original run's judges (claude-opus-5)
.venv/bin/python evals/audit_dad.py --input outputs/dad/runs/<opus run> --judges --config config.yaml
.venv/bin/python evals/audit_dad.py --input outputs/dad/runs/<gpt run> --judges --config config.yaml

# 4. report: per-example scores vs the originals, bucket means, judge
#    stability on the shared plain arm, refusal/truncation/drop-out rates
.venv/bin/python experiments/model_rerun/summarize.py \
    --arm opus55=outputs/dad/runs/<opus run> \
    --arm gpt61sol=outputs/dad/runs/<gpt run>
```

## Notes

- Opus 5.5 cannot run with thinking disabled, so its arm generates WITH hidden
  adaptive thinking (effort medium, the model default) — a deliberate deviation
  from the pipeline's no-scratchpads rule, fine for a comparison arm, never for
  corpus generation (`shared/api.py::_requires_adaptive_thinking`).
- Refusal fallbacks are disabled in the arm configs (`scope_refusal_fallback_model:
  null`): a refusal here is a measurement, not a failure to route around.
- Hard refusals land in `<run>/call_log.jsonl` (`stop_reason: "refusal"`); the
  pipeline drops refused/truncated work rather than checkpointing it, so an
  example a model persistently refuses simply never reaches `final/` — the
  summary counts both signals, plus a text-level refusal-phrasing scan.
- The judges and the original run's generator are the same model family
  (claude-opus-5); scores are comparable to the original run's, not absolute.
