# Log-probability persona check

Estimates whether a corpus would push a model trained on it towards a persona (preachy, sycophantic, ...). An open-weights model scores every token of each document as the assistant's reply, under a generic assistant prompt and under many rewordings of the persona and of controls. Text the persona makes unusually likely, compared with the controls, is text that would pull a model towards it.

How to run and read it: `.claude/skills/log-probability-persona-check/SKILL.md`. File formats and the interface between steps: `run_folder_layout_and_interfaces.md`. Persona sets live in `prompts/tools/persona_checks/`. Everything runs from the repository root.

## Modules

| Module | What it does | Cost |
|---|---|---|
| `persona_sets.py` | Loads and validates a persona set (rewordings, controls, rewrite instructions, configuration). | free |
| `scoring_model_defaults.py` | The default scoring model and pinned revision for each model type (`instruct`, `base`). | free |
| `prompt_rendering.py` | Builds the conditions and renders each prompt for a model type: chat template for `instruct`, plain-text author preamble for `base`. | free |
| `make_control_rewrites.py` | Rewrites sampled documents towards the persona (positive control), optionally away from it, and in a neutral register (neutral control), with Claude. | paid (`--dry-run` is free) |
| `select_pilot_documents.py` | Chooses the pilot documents: every rewrite source plus documents spread across a variable. | free |
| `build_scoring_inputs.py` | Writes the run folder: documents, conditions and the run configuration with file hashes. `--dry-run` prints counts and writes nothing. | free |
| `score_document_log_probabilities_on_pod.py` | The scorer. Writes the log probability of every document token under every condition, resumable. Runs on the pod, or on CPU with a tiny model. | GPU time |
| `launch_and_stop_runpod_scoring_pod.py` | The launcher. Creates a RunPod pod with hard hour and dollar limits and an on-pod watchdog, uploads the inputs, runs the scorer, pulls and verifies the results, and deletes the pod. | paid (`--dry-run` is free) |
| `requirements_for_gemma_4_scoring_pod.txt` | Packages installed on the pod. | |
| `analyze_persona_scores.py` | Analysis of a scored run: control verdicts, extreme spans, span significance, rewording agreement, per-persona breakdown, and document means labelled not validated. | free |
| `analysis_run_loading.py` | Reads a scored run folder for the analysis modules. | free |
| `analyze_changed_spans.py` | The changed-span part of the analysis, run alone. | free |
| `summarize_pilot_for_go_or_no_go_decision.py` | After a pilot: verification, throughput, cost, a projected full-run cost range and each criterion met or not. | free |
| `rehearse_pilot_pipeline_locally_without_paid_calls.py` | Rehearses the whole chain, including the launcher dry run, on a tiny random model. | free |
| `smoke_test_log_probability_persona_check.py` | Smoke test of the local chain for both model types. | free |
| `legacy_run_format_compatibility_reader.py` | Lets the analysis read run folders written by an earlier version of the check. | free |
| `timed_progress_and_jsonl_helpers.py` | Shared helpers for timestamps, progress reports and JSON lines. | free |

## Commands

```bash
# validate a persona set (offline)
python evals/log_probability_persona_check/persona_sets.py --persona-set preachiness

# build scoring inputs for a pipeline run; --dry-run prints counts and writes nothing
python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness \
    --input outputs/sdf/runs/<run_id> --model-type instruct --dry-run

# rehearse everything locally, no paid call
python evals/log_probability_persona_check/rehearse_pilot_pipeline_locally_without_paid_calls.py

# dry run of the launcher: pod request, commands, cost ceiling and projection; no call to RunPod
python evals/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py \
    --run-folder <run folder> --maximum-running-hours 3 --dry-run

# analyse a scored run
python evals/log_probability_persona_check/analyze_persona_scores.py --run-folder <run folder>
```

Paid steps (control rewrites, the pod launch) need the contributor's explicit go-ahead; the skill's `references/paid_pod_procedure.md` lists every step with its flags.

Tests: `pytest tests/test_log_probability_persona_check_*.py`. Tests that need `tokenizers` and `transformers` skip when they are missing; tests marked `slow` run only with `--run-slow`.
