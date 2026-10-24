# Log-probability persona check: run folder layout and interfaces

This is the contract between the steps of the check. Each step reads what the previous one wrote and nothing else, so a step can be rebuilt without touching the others. Sections marked PLANNED describe steps that are not implemented yet; their names and fields are fixed here so the steps can be built in parallel.

All commands run from the repository root. Code lives in `evals/log_probability_persona_check/`; prompt files in `prompts/tools/persona_checks/`.

## Pipeline

```
persona_sets.py  (validate a persona set)                     free, local
rewrites         (PLANNED, paid Claude calls)           -> rewritten documents JSONL
build_scoring_inputs.py                                 -> run folder with scoring_inputs/       free, local
pod launcher     (PLANNED, paid GPU pod)                -> runs the scorer on a pod, pulls scoring_outputs/ back
scorer           (PLANNED, runs on the pod)             -> scoring_outputs/
analysis         (PLANNED, free, local)                 -> analysis_<timestamp>/ inside the run folder
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
- Rewritten documents (PLANNED): a folder chosen by the rewrite command, holding `rewritten_documents_<timestamp>.jsonl`.

## Rewritten documents file (PLANNED, read by build_scoring_inputs.py)

One record per rewrite:

```json
{"document_id": "matrix_000005__preachier", "source_document_id": "matrix_000005",
 "rewrite_direction": "preachier", "content": "...", "language": "English",
 "variables": {"...": "copied unchanged from the source document"},
 "rewrite_model_requested": "...", "rewrite_model_used": "...",
 "source_word_count": 1240, "rewrite_word_count": 1260}
```

`rewrite_direction` is one of the persona set's direction names (`rewrite_directions.*.name`). `build_scoring_inputs.py` requires `document_id`, `source_document_id`, `rewrite_direction`, `content`, `language`, `variables`, and ignores the rest.

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

Later steps add their own records beside these (PLANNED): pod type, model load time and throughput, written by the launcher or scorer into `scoring_outputs/scoring_environment_<timestamp>.json`.

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

## Scoring outputs (PLANNED: written by the scorer on the pod)

```
<run folder>/
  scoring_outputs/
    per_token_log_probabilities/<document_id>.npz
    scoring_completion_records.jsonl
    scoring_progress_log_<timestamp>.txt
    scoring_environment_<timestamp>.json  # model name, revision, torch and transformers versions, GPU name, dtype
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

`scoring_completion_records.jsonl`: one record per (document, condition), appended after the document's `.npz` is saved. Resuming skips documents whose `.npz` exists and whose records are all present.

```json
{"document_id": "matrix_000005", "condition_id": "generic", "number_of_document_tokens": 2104,
 "sum_log_probability": -4321.5, "mean_log_probability": -2.054}
```

## Analysis outputs (PLANNED)

```
<run folder>/
  analysis_<timestamp>/
    ...                                   # file names are fixed by the analysis step; planned contents:
```

Planned contents, named for the persona set's own group names and not for "preachy":

- span-level output first: the highest- and lowest-scoring windows of each document, and scores of changed text against unchanged text for rewritten documents;
- the positive-control and neutral-control verdicts against `pass_criteria`, at whole-document and changed-span level;
- rewording agreement, and the per-persona breakdown;
- document means, labelled as not validated.

Definitions carried over from the legacy version (the analysis step may refine them):

- Score of a document under one condition c: mean over document tokens of `log_probabilities[c] - log_probabilities[generic]`.
- Per-token persona difference: the mean over a persona's rewordings of `log_probabilities[c, t] - log_probabilities[generic, t]`.
- Extreme spans: the 3 highest and 3 lowest non-overlapping 32-token windows of the per-token target-persona difference, with their text.

## Public functions

`persona_sets.py`: `load_persona_set(persona_set, ...) -> PersonaSet`, `validate_persona_set(persona_set, ...) -> PersonaSetValidationReport`, `load_model_type_framing(directory=None)`, `load_generation_prompt_files(directory=None)`, `resolve_persona_set_folder`, `list_persona_set_names`.

`prompt_rendering.py`: `build_condition_records(persona_set, model_type, framing=None)`, `render_generation_prompt(variables, document_id, generation_prompt_files=None)`, `tokenize_condition_and_document_for_model_type(tokenizer, condition, generation_prompt, document_content, model_name, model_type, framing=None)`, `tokenize_all_conditions_for_document(tokenizer, document, conditions, model_name, model_type, framing=None)`, `count_tokens_for_scoring_inputs(...)`. Pass a loaded `framing` when rendering base-model prompts repeatedly; otherwise each call re-reads the framing files.

`scoring_model_defaults.py`: `DEFAULT_SCORING_MODELS`, `resolve_scoring_model(model_type, persona_set_default_models, model_name_override, model_revision_override)`.
