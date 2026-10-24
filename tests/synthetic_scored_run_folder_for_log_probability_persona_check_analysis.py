"""Builds a small synthetic scored run folder (run configuration, scoring inputs, per-token score files) with planted
effects, for testing the analysis steps of the log-probability persona check. The persona is "sycophantic" so that no
test passes by accident because a name matches the preachiness set."""

import copy
import json

import numpy as np

TOKENS_PER_SOURCE_DOCUMENT = 100
PLANTED_SPAN_START = 32
PLANTED_SPAN_LENGTH = 32
PLANTED_SPAN_SHIFT = 0.3
INSERTED_TOKEN_COUNT = 20
INSERTED_TOKEN_SHIFT = 0.4
INSERTED_TOKEN_POSITION = 70
NOISE_STANDARD_DEVIATION = 0.02

PERSONA_NAME = "sycophantic"
TOPIC_CONTROL_GROUP = "topic_control"
SAME_VALUES_GROUP = "same_values"
TOPIC_CONTROL_PERSONAS = ["bakery_expertise", "railway_expertise"]
DIRECTION_NAMES = {"toward": "more_sycophantic", "away": "less_sycophantic", "neutral": "tone_shift"}
SOURCE_KINDS = ["elevated", "elevated", "elevated", "null", "null", "null"]

DEFAULT_PASS_CRITERIA = {
    "status": "Synthetic criteria for tests.",
    "whole_document": {"applies_to_directions": ["toward", "away"], "minimum_number_moving_expected_way": 2,
                       "number_of_rewrites_tested_per_direction": 2},
    "changed_span": {"applies_to_directions": ["toward", "away"], "minimum_number_moving_expected_way": 2,
                     "number_of_rewrites_tested_per_direction": 2},
    "neutral_control": {"changed_span": {"maximum_number_moving_toward_persona": 0, "number_of_rewrites_tested": 2}}}


def persona_check_configuration(include_same_values_control=True, pass_criteria=None):
    groups = {
        "target": {"condition_group": PERSONA_NAME, "display_name": PERSONA_NAME, "expected_rewording_count": 3},
        "topic_control": {"condition_group": TOPIC_CONTROL_GROUP, "display_name": "topic controls",
                          "expected_persona_count": 2, "expected_rewordings_per_persona": 2}}
    if include_same_values_control:
        groups["same_values_control"] = {"condition_group": SAME_VALUES_GROUP, "persona_name": SAME_VALUES_GROUP,
                                         "display_name": "same values", "expected_rewording_count": 2}
    directions = {"toward": {"name": DIRECTION_NAMES["toward"]}, "away": {"name": DIRECTION_NAMES["away"]},
                  "neutral": {"name": DIRECTION_NAMES["neutral"]}}
    return {"schema_version": 1, "persona_set_name": "sycophancy_for_tests", "persona_name": PERSONA_NAME,
            "condition_groups": groups, "rewrite_directions": directions,
            "pass_criteria": copy.deepcopy(DEFAULT_PASS_CRITERIA if pass_criteria is None else pass_criteria)}


def build_conditions(include_same_values_control=True):
    conditions = [
        {"condition_id": "no_system_prompt", "condition_group": "no_system_prompt", "persona_name": "no_system_prompt",
         "rewording_index": None, "system_prompt": None},
        {"condition_id": "generic", "condition_group": "generic", "persona_name": "generic", "rewording_index": None,
         "system_prompt": "You are an AI assistant."}]
    for index in range(1, 4):
        conditions.append({"condition_id": f"{PERSONA_NAME}__rewording_{index:02d}", "condition_group": PERSONA_NAME,
                           "persona_name": PERSONA_NAME, "rewording_index": index, "system_prompt": "x"})
    for persona in TOPIC_CONTROL_PERSONAS:
        for index in range(1, 3):
            conditions.append({"condition_id": f"{persona}__rewording_{index:02d}", "condition_group": TOPIC_CONTROL_GROUP,
                               "persona_name": persona, "rewording_index": index, "system_prompt": "x"})
    if include_same_values_control:
        for index in range(1, 3):
            conditions.append({"condition_id": f"{SAME_VALUES_GROUP}__rewording_{index:02d}",
                               "condition_group": SAME_VALUES_GROUP, "persona_name": SAME_VALUES_GROUP,
                               "rewording_index": index, "system_prompt": "x"})
    return conditions


def write_jsonl_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def build_synthetic_scored_run_folder(run_folder, seed=0, include_same_values_control=True, rewrites_per_direction=2,
                                      neutral_rewrites_shift=0.0, model_type="instruct", pass_criteria=None,
                                      outlier_rewording_condition_id=None, shifts_mirrored_for_outlier=True):
    """Returns {"documents": {document_id: {"kind", ...}}, "condition_ids": [...]}. Source documents are
    `elevated` (the persona raises the log probability of a 32-token span by 0.3 in every target rewording) or
    `null`. A rewrite inserts 20 new tokens at position 70: a toward rewrite raises them by +0.4 for the target
    persona, an away rewrite by -0.4, a neutral rewrite by `neutral_rewrites_shift`."""
    generator = np.random.default_rng(seed)
    conditions = build_conditions(include_same_values_control)
    condition_ids = [condition["condition_id"] for condition in conditions]
    target_rows = [i for i, c in enumerate(conditions) if c["condition_group"] == PERSONA_NAME]
    generic_row = condition_ids.index("generic")
    configuration = persona_check_configuration(include_same_values_control, pass_criteria)

    def score_document(token_ids, target_shift_per_token):
        token_count = len(token_ids)
        generic = -3.0 + generator.normal(0, 0.5, token_count)
        log_probabilities = np.tile(generic, (len(conditions), 1))
        log_probabilities += generator.normal(0, NOISE_STANDARD_DEVIATION, log_probabilities.shape)
        log_probabilities[generic_row] = generic
        for row in target_rows:
            log_probabilities[row] += target_shift_per_token
        return log_probabilities.astype(np.float32)

    documents, planted, scored = [], {}, {}
    source_shift = {}
    for index, kind in enumerate(SOURCE_KINDS):
        document_id = f"source_{index:02d}"
        token_ids = np.arange(index * 1000, index * 1000 + TOKENS_PER_SOURCE_DOCUMENT)
        shift = np.zeros(TOKENS_PER_SOURCE_DOCUMENT)
        if kind == "elevated":
            shift[PLANTED_SPAN_START:PLANTED_SPAN_START + PLANTED_SPAN_LENGTH] = PLANTED_SPAN_SHIFT
        source_shift[document_id] = shift
        scored[document_id] = (token_ids, score_document(token_ids, shift))
        planted[document_id] = {"kind": kind, "span_start": PLANTED_SPAN_START}
        documents.append({"document_id": document_id, "source_document_id": document_id, "rewrite_direction": None,
                          "content": "text", "language": "English", "variables": {"tone": "neutral" if index % 2 else "formal"}})
    rewrite_shifts = {"toward": INSERTED_TOKEN_SHIFT, "away": -INSERTED_TOKEN_SHIFT, "neutral": neutral_rewrites_shift}
    for role, direction_name in DIRECTION_NAMES.items():
        for index in range(rewrites_per_direction):
            source_id = f"source_{index:02d}"
            source_ids, source_scores = scored[source_id]
            inserted_ids = np.arange(900000 + 100 * index + (10 if role == "away" else 0) + (20 if role == "neutral" else 0),
                                     900000 + 100 * index + (10 if role == "away" else 0) + (20 if role == "neutral" else 0)
                                     + INSERTED_TOKEN_COUNT)
            token_ids = np.concatenate([source_ids[:INSERTED_TOKEN_POSITION], inserted_ids, source_ids[INSERTED_TOKEN_POSITION:]])
            shift = np.concatenate([source_shift[source_id][:INSERTED_TOKEN_POSITION],
                                    np.full(INSERTED_TOKEN_COUNT, rewrite_shifts[role]),
                                    source_shift[source_id][INSERTED_TOKEN_POSITION:]])
            document_id = f"{source_id}__{direction_name}"
            scored[document_id] = (token_ids, score_document(token_ids, shift))
            planted[document_id] = {"kind": f"rewrite_{role}", "source": source_id}
            documents.append({"document_id": document_id, "source_document_id": source_id,
                              "rewrite_direction": direction_name, "content": "text", "language": "English",
                              "variables": {"tone": "formal"}})

    if outlier_rewording_condition_id:
        outlier_row = condition_ids.index(outlier_rewording_condition_id)
        for document_id, (token_ids, log_probabilities) in scored.items():
            generic = log_probabilities[generic_row]
            mirrored = generic - (log_probabilities[outlier_row] - generic)
            log_probabilities[outlier_row] = mirrored + generator.normal(0, 0.05, generic.shape).astype(np.float32)

    output_folder = run_folder / "scoring_outputs" / "per_token_log_probabilities"
    output_folder.mkdir(parents=True, exist_ok=True)
    for document_id, (token_ids, log_probabilities) in scored.items():
        np.savez(output_folder / f"{document_id}.npz", document_token_ids=token_ids.astype(np.int32),
                 document_token_strings=np.array([f"tok{token_id} " for token_id in token_ids]),
                 condition_ids=np.array(condition_ids), log_probabilities=log_probabilities)
    write_jsonl_records(run_folder / "scoring_inputs" / "documents.jsonl", documents)
    write_jsonl_records(run_folder / "scoring_inputs" / "conditions.jsonl", conditions)
    run_configuration = {
        "run_folder_name": run_folder.name, "created_timestamp": "2026-01-01_00-00", "model_type": model_type,
        "model_name": "example-org/example-model", "model_revision": "a" * 40,
        "persona_set": {"name": "sycophancy_for_tests", "configuration": configuration},
        "pass_criteria": configuration["pass_criteria"],
        "document_selection": {"corpus_path": "no_such_corpus.jsonl", "extra_documents_files": []}}
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(run_configuration), encoding="utf-8")
    return {"documents": planted, "condition_ids": condition_ids}
