# Log-probability persona check

Estimates whether a corpus would push a model trained on it towards a persona (preachy, sycophantic, ...). An open-weights model scores every token of each document as the assistant's reply, under a generic assistant prompt and under many rewordings of the persona and of controls. Text the persona makes unusually likely, compared with the controls, is text that would pull a model towards it.

This folder holds the free, local part: persona sets, prompt rendering and the scoring input builder. The interface every step builds against is in `run_folder_layout_and_interfaces.md`. Persona sets live in `prompts/tools/persona_checks/`.

```bash
# validate a persona set (offline)
python evals/log_probability_persona_check/persona_sets.py --persona-set preachiness

# build scoring inputs for a pipeline run; --dry-run prints counts and writes nothing
python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness \
    --input outputs/sdf/runs/<run_id> --model-type instruct --dry-run
python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness \
    --input outputs/sdf/runs/<run_id> --model-type instruct
```

Tests: `pytest tests/test_log_probability_persona_check_*.py`. Tests that need `tokenizers` and `transformers` skip when they are missing; tests marked `slow` run only with `--run-slow`.
