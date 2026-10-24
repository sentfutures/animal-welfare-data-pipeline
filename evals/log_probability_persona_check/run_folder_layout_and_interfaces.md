# Log-probability persona check: run folder layout and interfaces

This is the contract between the steps of the check. Each step reads what the previous one wrote and nothing else, so a step can be rebuilt without touching the others. Sections marked PLANNED describe steps that are not implemented yet; their names and fields are fixed here so the steps can be built in parallel.

All commands run from the repository root. Code lives in `evals/log_probability_persona_check/`; prompt files in `prompts/tools/persona_checks/`.

## Pipeline

```
persona_sets.py  (validate a persona set)                     free, local
make_control_rewrites.py  (paid Claude calls)           -> rewrite run folder with rewritten_documents_<timestamp>.jsonl
select_pilot_documents.py (free, local)                 -> pilot_document_ids_<timestamp>.txt (for --document-ids-file)
build_scoring_inputs.py                                 -> run folder with scoring_inputs/       free, local
pod launcher     (paid GPU pod)                         -> runs the scorer on a pod, pulls scoring_outputs/ back
scorer           (runs on the pod)                      -> scoring_outputs/
analysis         (free, local)                          -> analysis_<timestamp>/ inside the run folder
```

## Persona sets

A persona set is a folder `prompts/tools/persona_checks/<persona_set_name>/`. Shared framing and the document generation prompt live beside it:

```
prompts/tools/persona_checks/
  model_type_framing/                     shared by all persona sets
    instruct_persona_system_prompt_frame.txt      "You are an AI assistant who {persona_description}."
    instruct_generic_system_prompt.txt            "You are an AI assistant."
    base_model_preamble_template_with_author.txt       slots {author_description}, {generation_prompt}
    base_model_preamble_template_without_author.txt    slot {generation_prompt}
    base_model_persona_author_frame.txt                "an AI assistant who {persona_description}"
    base_model_generic_author_description.txt          "an AI assistant"
  generation_prompt/                      shared by all persona sets
    document_generation_prompt_template.txt       slots: {in_universe_facts} and one slot per document variable
    in_universe_facts_shared_by_all_documents.txt
  <persona_set_name>/
    target_persona_rewordings.txt           one description per line (about 30)
    topic_control_personas.txt              "# persona_name" header, then its rewordings, one per line
    same_values_control_rewordings.txt      OPTIONAL, one description per line
    positive_control_rewrite_instructions.txt     slots {direction}, {document_text}
    neutral_control_rewrite_instructions.txt      slots {direction}, {document_text}
    persona_check_configuration.json
```

A description completes "You are an AI assistant who ..." and "an AI assistant who ...": it starts lowercase with a verb, is one line, and has no final full stop. Rewrite instruction files may start with a line `# DRAFT: ...`; the loader strips it and records the file as a draft. A paid step must call `PersonaSet.require_reviewed_instructions("positive" | "neutral")` first, which raises `DraftInstructionsError` while the marker is present.

Validate with `python evals/log_probability_persona_check/persona_sets.py --persona-set <name or folder>`.

### persona_check_configuration.json

```json
{
  "schema_version": 1,
  "persona_set_name": "preachiness",
  "persona_name": "preachy",
  "condition_groups": {
    "target": {"condition_group": "preachy", "display_name": "preachy", "expected_rewording_count": 30},
    "topic_control": {"condition_group": "control", "display_name": "topic controls without a moral stance",
                      "expected_persona_count": 5, "expected_rewordings_per_persona": 5},
    "same_values_control": {"condition_group": "proportionate_welfare", "persona_name": "proportionate_welfare",
                            "display_name": "proportionate welfare", "expected_rewording_count": 5}
  },
  "rewrite_directions": {"toward": {"name": "preachier"}, "away": {"name": "plainer"},
                         "neutral": {"name": "register_shift"}},
  "pass_criteria": {
    "status": "Values used in the first preachiness run, fixed before that run was scored.",
    "whole_document": {"applies_to_directions": ["toward", "away"], "minimum_number_moving_expected_way": 8,
                       "number_of_rewrites_tested_per_direction": 10},
    "changed_span": {"applies_to_directions": ["toward", "away"], "minimum_number_moving_expected_way": 8,
                     "number_of_rewrites_tested_per_direction": 10}
  }
}
```

- `persona_set_name` equals the folder name. `persona_name` is the target persona's name in condition ids and in the `persona_name` column.
- `condition_groups.same_values_control` and `rewrite_directions.away` are optional. Without the group the set has no `same_values_control_rewordings.txt` and `expected_condition_count = 2 + target + topic personas x rewordings`.
- `pass_criteria` is fixed before scoring and copied into the run configuration. Optional `pass_criteria.neutral_control.changed_span` = `{"maximum_number_moving_toward_persona": int, "number_of_rewrites_tested": int}`, to be set before the neutral control is run.
- Optional `default_models`: `{"instruct": {"model_name", "model_revision"}, "base": {...}}` overriding the package defaults in `scoring_model_defaults.py` (instruct `google/gemma-4-12b-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, base `google/gemma-4-12b` at `023679ed352de9bb66cc873c9009ce3482585c08`). Precedence: command line, persona set, package default. A pinned 40-character revision is always required.
- Unknown keys are errors.

## Locations

- Pipeline run folders: `outputs/sdf/runs/<run_id>/` (corpus `final/sdf_corpus.jsonl`).
- Scoring run folders: `<pipeline run folder>/audit/log_probability_persona_check_<persona_set_name>_<model_type>_<YYYY-MM-DD_HH-MM>/`, or `audit/` beside a bare corpus `.jsonl` given as `--input`.
- Rewrite run folders: `outputs/log_probability_persona_check_rewrites/control_rewrites_[dry_run_]<persona_set_name>_<YYYY-MM-DD_HH-MM>/` (see "Rewrites").

## Rewrites (make_control_rewrites.py and select_pilot_documents.py)

```
python evals/log_probability_persona_check/make_control_rewrites.py --persona-set <name or folder> --input <run folder or corpus .jsonl>
    [--directions toward away neutral] [--number-of-documents 10] [--stratify-by tone] [--seed 0]
    [--source-document-ids-file F] [--exclude-where VARIABLE=TEXT] [--include-ineligible-documents]
    [--guarantee-stratum VALUE=N] [--model claude-opus-5] [--workers 3] [--dry-run] [--resume-folder FOLDER]
```

One command makes the positive control (`toward`, and `away` when the persona set configures it) and the neutral control. The direction names come from `rewrite_directions.*.name`; `toward` and `away` use `positive_control_rewrite_instructions.txt`, `neutral` uses `neutral_control_rewrite_instructions.txt`; `{direction}` is filled with the direction name in capitals (`PREACHIER`) and `{document_text}` with the source document. Default directions: every one the persona set configures.

Sources are `--number-of-documents` corpus documents sampled with `--seed`, stratified by the corpus variable `--stratify-by` (default `tone`; largest-remainder shares of the eligible corpus; `--guarantee-stratum` raises named values to a minimum, by default `skeptical or displeased=3` when stratifying by tone). Documents whose `centrality` is "a minor detail mentioned only in passing" or whose `resolution` says the AI raises no welfare consideration are not eligible unless `--include-ineligible-documents`. For the away direction choose sources that already show the behaviour, or drop that direction.

Calls go through `shared/api.py` (`call_claude`, stage `persona_check_control_rewrite`, item id = the rewrite's document id), so they are in the repository cost log (`--cost-log-path` overrides) and use the key from the repository `.env`. Thinking is off, as everywhere in `shared/api.py`. `PersonaSet.require_reviewed_instructions` runs before the first paid call for each kind of instructions requested; a draft blocks the paid run, not `--dry-run`. `--dry-run` writes the rendered requests and `dry_run_cost_estimate_<timestamp>.json` and makes no call, needs no key and does not initialise `shared/api.py`.

```
<rewrite run folder>/
  rewrite_run_configuration_<timestamp>.json     # persona set, model, direction names, instruction file hashes,
                                                 # corpus path and hash, selection method, seed, stratify_by,
                                                 # selected_source_document_ids; a resume adds a further file, the first is kept
  rewritten_documents_<timestamp>.jsonl          # the file build_scoring_inputs.py reads; a resume appends to it
  rewrite_failures_<timestamp>.jsonl             # one per failed rewrite of that session
  rendered_requests_<timestamp>.jsonl            # dry run only
  dry_run_cost_estimate_<timestamp>.json         # dry run only
  pilot_document_selection_<timestamp>/          # written by select_pilot_documents.py
    pilot_document_ids_<timestamp>.txt           # one document id per line, for build_scoring_inputs.py --document-ids-file
    pilot_document_selection_<timestamp>.json
```

Resume (`--resume-folder`) skips every (source document, direction) pair already in the rewritten documents file, retries failed pairs, reuses the first run's sources, and is refused if the persona set, model, direction names, corpus or instruction files differ from the first run. A failure record: `{"source_document_id", "rewrite_direction", "failure_kind": "refusal" | "max_tokens" | "empty_output" | "transport_failure", ...}`.

One record per rewrite in the rewritten documents file:

```json
{"document_id": "matrix_000005__preachier", "source_document_id": "matrix_000005",
 "rewrite_direction": "preachier", "content": "...", "language": "English",
 "variables": {"...": "copied unchanged from the source document"},
 "rewrite_model_requested": "claude-opus-5", "rewrite_model_used": "claude-opus-5",
 "source_word_count": 1240, "rewrite_word_count": 1260,
 "length_unit": "words", "source_length": 1240, "rewrite_length": 1260, "length_ratio": 1.016,
 "length_within_fifteen_percent": true}
```

`rewrite_direction` is one of the persona set's direction names (`rewrite_directions.*.name`). `rewrite_model_used` equals the requested model, because `shared/api.py` does not return the serving model. Length is counted in non-whitespace characters for Mandarin Chinese and Japanese, in words otherwise. `build_scoring_inputs.py` requires `document_id`, `source_document_id`, `rewrite_direction`, `content`, `language`, `variables`, and ignores the rest.

`select_pilot_documents.py --rewrites-folder <rewrite run folder> --input <corpus> [--persona-set <name>] [--number-of-documents 20] [--stratify-by tone] [--spread-by language] [--seed 0]` picks the total number of documents: every rewrite source (read from the rewritten documents and from `selected_source_document_ids`, so sources of failed rewrites are kept), then further documents spread evenly across the `--stratify-by` values and, within each, across as many `--spread-by` values as possible. With `--persona-set` the directions found are checked against the set's direction names.

## Run folder written by build_scoring_inputs.py

```
<run folder>/
  run_configuration_<timestamp>.json      # see below
  scoring_inputs/
    documents.jsonl                       # one per document to score
    conditions.jsonl                      # one per system-prompt condition
```

Run configuration fields (resuming appends a further `run_configuration_<timestamp>.json`; the first is kept):

| Field | Meaning |
|---|---|
| `run_folder_name`, `created_timestamp`, `resumed_from_existing_folder` | identity |
| `model_type` | `"instruct"` (instruction-tuned model scored through its chat template, personas as system prompts) or `"base"` (pretrained model, plain-text author preamble). The scorer reads it from here. |
| `model_name`, `model_revision` | Hugging Face repository and pinned 40-character commit hash |
| `model_selection_source` | `"command line"`, `"persona set"` or `"package default"` |
| `persona_set` | `{"name", "folder", "configuration" (the full persona_check_configuration.json), "positive_control_instructions_are_draft", "neutral_control_instructions_are_draft"}` |
| `pass_criteria` | copy of `persona_set.configuration.pass_criteria` |
| `document_selection` | `corpus_path`, `document_ids_file`, `maximum_documents`, `extra_documents_files`, `number_of_corpus_documents`, `number_of_extra_documents`, `document_ids` |
| `number_of_conditions` | |
| `prompt_file_sha256` | SHA-256 of every file that decides what is scored, keyed by path under `prompts/tools/persona_checks/` (`model_type_framing/...`, `generation_prompt/...`, `<persona set>/{persona_check_configuration.json, target_persona_rewordings.txt, topic_control_personas.txt, same_values_control_rewordings.txt}`). The four `base_model_*` files are included only for `base` runs. Resuming is refused if any differs. |
| `persona_set_file_sha256` | SHA-256 of every file in the persona set folder (informational) |
| `input_file_sha256` | SHA-256 of the corpus file, the document ids file and each extra documents file |
| `scoring_input_file_sha256` | SHA-256 of `documents.jsonl` and `conditions.jsonl` |

The scorer adds its own records beside these, in `scoring_outputs/scoring_environment_<timestamp>.json` (see Scoring outputs), and the launcher writes `pod_cost_log_<timestamp>.json` (see Pod launcher).

Resuming (`--resume-folder`) is refused on a different model type, model name, model revision or persona set, on any changed hashed file, and when the rebuilt `documents.jsonl` or `conditions.jsonl` differ from the files in the folder.

`documents.jsonl` record:

```json
{"document_id": "matrix_000005", "source_document_id": "matrix_000005", "rewrite_direction": null,
 "content": "...", "language": "English", "variables": {"...": "..."},
 "generation_prompt": "Write a document from a world where ..."}
```

Original corpus documents have `rewrite_direction: null` and `source_document_id` equal to `document_id`. The generation prompt is the template in `prompts/tools/persona_checks/generation_prompt/` filled from the document's `variables` (the same twelve axes as `prompts/sdf/variables.txt` that the template names); the `resolution` value gets a final full stop if it has none.

`conditions.jsonl` record:

```json
{"condition_id": "preachy__rewording_07", "condition_group": "preachy", "persona_name": "preachy",
 "rewording_index": 7, "system_prompt": "You are an AI assistant who ..."}
```

- Order: `no_system_prompt`, `generic`, the target rewordings, the topic control personas in file order, then the same-values control.
- `condition_group` is `no_system_prompt`, `generic`, or the three `condition_groups.*.condition_group` values of the configuration (`preachy`, `control`, `proportionate_welfare` for preachiness).
- `persona_name` is `no_system_prompt`, `generic`, the configuration's `persona_name` for the target, the `#` header name for topic controls (for example `animal_husbandry_expertise`), and `condition_groups.same_values_control.persona_name` for the same-values control.
- `system_prompt` is `null` only for `no_system_prompt`. It is present in both model types.
- `author_description` (base runs only; instruct `conditions.jsonl` files do not have the field): `null` for `no_system_prompt`, `"an AI assistant"` for `generic`, and `"an AI assistant who <the same persona description line used in system_prompt>"` for every persona condition. Conditions, condition ids and groups are identical across model types; only the rendering differs.
- Condition ids: `no_system_prompt`, `generic`, `<persona_name>__rewording_<two digits>`.

## Scoring outputs (written by the scorer, `score_document_log_probabilities_on_pod.py`)

```
<run folder>/
  scoring_outputs/
    per_token_log_probabilities/<document_id>.npz
    scoring_completion_records.jsonl
    scoring_progress_log_<timestamp>.txt
    scoring_environment_<timestamp>.json  # see below
    full_logits_verification_<timestamp>.json   # only with --verify-first-document-against-full-logits
```

Each `.npz` is written atomically (temporary file then rename), once every condition for that document is scored. It contains:

- `document_token_ids`: int32, the token ids of the document's content (identical in every condition, which the scorer asserts)
- `document_token_strings`: the decoded text of each token, for span reporting
- `condition_ids`: the condition ids in row order
- `log_probabilities`: float32 array of shape (number of conditions, number of document tokens). Entry [c, t] is log P(token t | everything before it, condition c)

Only the document's content tokens are scored. The system prompt, generation prompt, chat-template markers, the assistant-turn header and the end-of-turn token are context, not scored.

Sequence layout by model type (implemented in `prompt_rendering.py`):

- `instruct`: chat template (system turn if any, user turn with the generation prompt, the assistant generation header) + document + whatever the template writes after an assistant message (not scored).
- `base`: the tokenizer's beginning-of-sequence token text (empty if the tokenizer has none) + preamble + `"\n\n"` + document. Nothing follows the document: no suffix and no end-of-sequence token. The preamble is `base_model_preamble_template_with_author.txt` (`no_system_prompt` uses `..._without_author.txt`) with `{author_description}` and `{generation_prompt}` filled. The first document token is scored given the blank line that ends the preamble. `full_logits_verification_<timestamp>.json` records the `model_type` used.
- Scores from the two model types are not comparable.

`scoring_environment_<timestamp>.json`: `model_name`, `model_revision_requested`, `model_type`, `model_commit_hash`, `model_class_used`, `final_logit_softcapping`, `dtype`, `device`, `torch_version`, `transformers_version`, `python_version`, `gpu_name`, `hostname`, `pod_type` (`kind` `gpu` or `cpu`, `gpu_name`, `gpu_count`, `gpu_memory_gigabytes`, and `runpod_pod_id` and `runpod_data_center_id` when RunPod's container variables are present), `model_load_seconds`, and `throughput` (null until scoring ends, then `documents_scored_this_session`, `input_tokens_processed`, `document_tokens_scored`, `scoring_seconds`, `input_tokens_per_second`, `document_tokens_scored_per_second`, `documents_per_minute`). Written when the model has loaded and rewritten when scoring ends.

Every file the scorer writes (the `.npz`, the environment record, the verification file) is first written as `<name>.partial` and renamed, and its modification time is set at the rename. Nothing named `*.partial` is final: readers and the launcher's pulls skip them. Only `scoring_completion_records.jsonl` and the progress log are appended to in place.

Scorer command line: `python -m evals.log_probability_persona_check.score_document_log_probabilities_on_pod --run-folder <run folder>` with optional `--model-name` and `--model-revision` (override the run configuration, with a warning), `--model-class {auto_causal_lm,auto_image_text_to_text}`, `--device`, `--dtype`, `--maximum-documents`, `--maximum-tokens-per-batch`, `--positions-per-head-chunk`, `--verify-first-document-against-full-logits`. It reads `model_type`, `model_name`, `model_revision` and `prompt_file_sha256` from the first run configuration, refuses to run if a hashed prompt file on its machine differs, loads the framing once for `base` runs, and imports nothing from the repository outside this package folder and `prompts/tools/persona_checks/`. A `--model-name` that is a local folder is loaded without a revision.

`full_logits_verification_<timestamp>.json`: `passed`, `model_type`, `document_id`, `condition_id` (the last condition), `number_of_document_tokens`, `positions_per_head_chunk`, `maximum_absolute_difference`, `mean_absolute_difference`, the two mean log-probabilities and the allowed differences (mean 0.05, maximum 1.0). Written also when the check fails, after which the scorer raises.

`scoring_completion_records.jsonl`: one record per (document, condition), appended after the document's `.npz` is saved. Resuming skips documents whose `.npz` exists and whose records are all present. A copy of the file taken while the scorer was appending (an interrupted run's last incremental pull) can end in a partial line: the readers ignore a final line that is not valid JSON, and the scorer removes it before appending, so the records of that document are rebuilt from its `.npz` on resume.

```json
{"document_id": "matrix_000005", "condition_id": "generic", "number_of_document_tokens": 2104,
 "sum_log_probability": -4321.5, "mean_log_probability": -2.054}
```

## Pod launcher (`launch_and_stop_runpod_scoring_pod.py`)

```
python evals/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py \
    --run-folder <run folder> --maximum-running-hours <hours> [--maximum-dollars 25] [--dry-run]
```

Reads `RUNPOD_API_KEY` (and `HF_TOKEN`, only if set) from the repository-root `.env`, parsed directly: never the OS environment, never parent folders. `--dry-run` makes no RunPod call and prints the pod request, every command, the cost ceiling and, when the run's tokenizer is already in the local Hugging Face cache, a projected cost range from the token count.

Flags with neutral defaults: `--ssh-private-key-path` (`~/.ssh/id_ed25519`), `--image-name`, `--gpu-type-ids` (priority order), `--cloud-type` (`SECURE` or `COMMUNITY`), `--container-disk-gigabytes`, `--volume-gigabytes`, `--cost-center-name` (optional; else `RUNPOD_COST_CENTER_NAME` in `.env`; else none), `--failure-wait-minutes`, `--reuse-existing-pod <id>`, `--attach-to-existing-pod <id>`, `--keep-pod-after-run`, and the scorer's own flags, which are passed through.

Uploaded to the pod, under `/workspace/log_probability_persona_check_scoring_copy/`: `evals/__init__.py`, `evals/log_probability_persona_check/`, `prompts/tools/persona_checks/`, and into `run_folder/` the `run_configuration_*.json` files, `scoring_inputs/` and any `scoring_outputs/` (never `*.partial`, `__pycache__`, `*.pyc`). The repository's `shared` package is not uploaded. Pod requirements are `requirements_for_gemma_4_scoring_pod.txt` (the image provides torch).

Pulls: every few minutes only the finished files of `scoring_outputs/` (copied to a staging folder on the pod first); at the end everything in the pod's run folder except `scoring_inputs/`, with the file list and sizes verified, the non-`scoring_outputs` files going to `<run folder>/pod_<pod id>_logs_<timestamp>/`.

Files in the run folder: the launcher writes `pod_progress_log_<timestamp>.txt` and `pod_cost_log_<timestamp>.json`, and reads the instruction files `pod_instruction_retry_scorer.txt`, `pod_instruction_retry_setup.txt` and `pod_instruction_delete_pod.txt` (each renamed `.consumed_<time>` once acted on) and `cost_center_assignment_confirmed_by_user.txt` (only meaningful with a cost center name).

`pod_cost_log_<timestamp>.json`: `launcher_timestamp`, `run_folder`, `maximum_running_hours`, `maximum_dollars`, `pod_id`, `gpu_type`, `hourly_price_dollars` (as RunPod reports it), `start_time`, `stop_time`, `termination_reason`, `pod_terminated`, `total_cost_dollars`, `final_pull_verified`, `cost_center_name` (null when none), `cost_center_assignment_confirmed_by_user`, `cost_center_assignment_confirmation_time`, plus `reused_existing_pod`, `pod_kept_after_run` and similar flags when they apply.

## Analysis outputs

Written by `analyze_persona_scores.py --run-folder <run folder>` (free, local; needs numpy, pandas and scipy only; no figures). Reads `run_configuration_*.json`, `scoring_inputs/` and `scoring_outputs/per_token_log_probabilities/`. Everything persona-specific (group names, direction names, pass criteria) comes from `persona_set.configuration` in the run configuration; column names carry the persona's name (`<persona>` below is `persona_name`, `<control>` the topic-control `condition_group`, `<same_values>` the optional same-values `condition_group`). A run folder whose run configurations, scoring environment records or verification record name more than one model type, model name or model revision is refused. The analysis refuses to overwrite an existing `analysis_<timestamp>/`. Every CSV starts with columns `model_type` and `model_name`; every JSON and each extreme-span record carries them too. Scores from different model types are not comparable.

```
<run folder>/
  analysis_<timestamp>/
    extreme_spans_<timestamp>.jsonl                      PRIMARY. 3 highest and 3 lowest non-overlapping 32-token windows per document
    span_significance_tests_<timestamp>.csv              PRIMARY. permutation test of each document's most extreme windows
    changed_span_<persona>_minus_<control>_<timestamp>.csv   PRIMARY. per rewrite: changed, unchanged and replaced text
    changed_span_by_persona_<timestamp>.csv              PRIMARY. per rewrite direction and persona
    total_log_likelihood_differences_<timestamp>.csv     per document: total log probability, persona minus controls
    changed_span_summary_<timestamp>.json                counts, Wilcoxon p-values and the changed-span verdicts
    control_verdicts_<timestamp>.json                    whole-document and changed-span verdicts against pass_criteria
    rewording_agreement_<timestamp>.csv                  every run: per rewording, persona-level columns repeated
    secondary_not_validated_per_document_scores_<timestamp>.csv
    secondary_not_validated_per_document_significance_tests_<timestamp>.csv
    secondary_not_validated_scores_by_spec_variable_<timestamp>.csv
    secondary_not_validated_whole_document_control_check_<timestamp>.csv
    analysis_notes_<timestamp>.md                        how every output is computed
    analysis_summary_<timestamp>.json                    model labels, counts, window and permutation settings
```

`analyze_changed_spans.py --run-folder <run folder> [--analysis-folder <folder>]` runs only the changed-span part (new `analysis_<timestamp>/` unless a folder is given); `analyze_persona_scores.py` runs it too unless `--skip-changed-spans`. Other options: `--window-size-tokens 32`, `--windows-per-direction 3`, `--number-of-permutations 1000`, `--permutation-seed 0`. `--legacy-run-persona-set <name>` is only for run folders written by the legacy version (see below).

Definitions:

- Score of a document under one condition c: mean over document tokens of `log_probabilities[c] - log_probabilities[generic]`.
- Per-token persona difference: the mean over a persona's rewordings of `log_probabilities[c, t] - log_probabilities[generic, t]`.
- Extreme spans: windows of the per-token target-persona difference; `window_mean_<persona>_difference`, `text` (token strings joined); each record carries the document's permutation p-value for its direction.
- Span significance: per document, the per-token difference is shuffled `--number-of-permutations` times; statistic = highest (lowest) window mean over all windows; one-sided p = (1 + number at least as extreme) / (1 + permutations); Benjamini-Hochberg q across original documents (rewrites keep the uncorrected p, flagged by `q_value_is_uncorrected_rewrite`). The null treats tokens as exchangeable, which is liberal when neighbouring tokens are correlated; it screens for spans and is not proof of a persona effect.
- Changed spans: rewrite tokens aligned to the source's by `difflib.SequenceMatcher`; span score = mean of (persona minus generic) minus (pooled topic controls minus generic) over the span's tokens. A rewrite moves the expected way when `changed_minus_unchanged` has the expected sign: positive for `toward`, negative for `away`; for `neutral`, a move toward the persona is positive. A rewrite with no changed tokens does not move.
- Whole-document check: rewrite minus source of `<persona>_minus_<control>_score`; a difference of exactly 0 does not move.
- Rewording agreement: mean pairwise Spearman, one-way ICC(1,1), within-to-across SD ratio, leave-one-out Spearman and outlier flag, per persona of every group.
- Document means (`secondary_not_validated_*`): not a validated measure; the whole-document score did not track the persona in the first preachiness run.

`control_verdicts_<timestamp>.json`:

```json
{"model_type": "instruct", "model_name": "...", "model_revision": "...", "persona_name": "...", "pass_criteria": {},
 "whole_document": {"positive_control": {"verdict": "pass | fail | not evaluable | no criterion pre-registered",
                                         "per_direction": {"toward": {"direction_name": "...", "number_of_rewrites": 10,
                                           "number_moving_expected_way": 8, "minimum_required": 8,
                                           "number_of_rewrites_pre_registered": 10, "verdict": "pass", "reason": ""}}}},
 "changed_span": {"positive_control": {"...same shape..."},
                  "neutral_control": {"verdict": "...", "number_of_rewrites": 10, "number_moving_toward_persona": 1,
                                      "maximum_allowed": 2, "number_of_rewrites_pre_registered": 10, "reason": ""}}}
```

A direction is "not evaluable" when the number of rewrites scored differs from `number_of_rewrites_tested_per_direction` (or `number_of_rewrites_tested`), since the criterion was fixed for that number. The neutral verdict is judged at changed-span level only (the whole-document neutral counts are reported, not judged).

Legacy run folders: `legacy_run_format_compatibility_reader.py` converts, in memory, a legacy-format `run_configuration_*.json` (no `persona_set` record) when `--legacy-run-persona-set <name>` is given: model type `instruct`, model revision from `scoring_environment_*.json`, group names, direction names and pass criteria from that persona set's configuration. Without the flag such a run folder is refused. It is for re-analysis only.

### Pilot summary

`summarize_pilot_for_go_or_no_go_decision.py --run-folder <run folder>` reads the newest `analysis_<timestamp>/`, `scoring_outputs/` and `pod_cost_log_<timestamp>.json` files (fields `pod_id`, `gpu_type`, `hourly_price_dollars`, `start_time`, `stop_time`, `total_cost_dollars`, `termination_reason`; a log written with `--attach-to-existing-pod` counts from the start of the pod's earlier log, so that earlier log is marked `covered_by_a_later_attach_log` and left out of the cost and running-time totals; logs of separate uses of a pod still add up), and `scoring_environment_<timestamp>.json` (`throughput.scoring_seconds` and `throughput.document_tokens_scored_per_second`; the last line of `scoring_progress_log_*.txt` is the fallback). It writes `pilot_go_or_no_go_summary_<timestamp>.md` and `.json` into the run folder: verification, throughput, pod cost, a projected full-run time and cost as a range (pilot mean tokens per original document times the full document count times the condition count; throughput range = slowest and fastest scorer session widened by `--throughput-uncertainty-fraction`, default 0.25, an assumption), the controls, rewording agreement, span significance counts, and each criterion as MET, NOT MET or NOT EVALUABLE. Options: `--full-run-number-of-documents` (or counted from the run's corpus file), `--total-budget-dollars` (adds a budget criterion), `--rewrite-generation-cost-dollars`, `--minimum-mean-pairwise-spearman 0.5` and `--maximum-outlier-fraction 0.10` (the summary's own rewording agreement thresholds, not pre-registered).

## Public functions

`persona_sets.py`: `load_persona_set(persona_set, ...) -> PersonaSet`, `validate_persona_set(persona_set, ...) -> PersonaSetValidationReport`, `load_model_type_framing(directory=None)`, `load_generation_prompt_files(directory=None)`, `resolve_persona_set_folder`, `list_persona_set_names`.

`prompt_rendering.py`: `build_condition_records(persona_set, model_type, framing=None)`, `render_generation_prompt(variables, document_id, generation_prompt_files=None)`, `tokenize_condition_and_document_for_model_type(tokenizer, condition, generation_prompt, document_content, model_name, model_type, framing=None)`, `tokenize_all_conditions_for_document(tokenizer, document, conditions, model_name, model_type, framing=None)`, `count_tokens_for_scoring_inputs(...)`. Pass a loaded `framing` when rendering base-model prompts repeatedly; otherwise each call re-reads the framing files.

`scoring_model_defaults.py`: `DEFAULT_SCORING_MODELS`, `resolve_scoring_model(model_type, persona_set_default_models, model_name_override, model_revision_override)`.
