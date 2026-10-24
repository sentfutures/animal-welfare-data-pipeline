"""Loading a scored run folder for the analysis steps of the log-probability persona check (local, free).

Reads `run_configuration_*.json`, `scoring_inputs/` and `scoring_outputs/per_token_log_probabilities/*.npz` (formats in
run_folder_layout_and_interfaces.md) and condenses every document into the per-token and per-condition quantities the
analyses use, so the large score arrays are not kept in memory.

Everything persona-specific (group names, rewrite direction names, pass criteria) comes from the persona set
configuration recorded in the run configuration. A run that mixes model types, models or model revisions is refused.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check.legacy_run_format_compatibility_reader import (  # noqa: E402
    normalize_legacy_run_configuration)
from shared import utils  # noqa: E402

GENERIC_CONDITION_ID = "generic"
NO_SYSTEM_PROMPT_CONDITION_ID = "no_system_prompt"
DIRECTION_ROLE_TOWARD = "toward"
DIRECTION_ROLE_AWAY = "away"
DIRECTION_ROLE_NEUTRAL = "neutral"
EXPECTED_SIGN_BY_DIRECTION_ROLE = {DIRECTION_ROLE_TOWARD: 1, DIRECTION_ROLE_AWAY: -1}


class MixedModelTypesError(ValueError):
    """The run folder holds records from more than one model type, model or model revision."""


@dataclass
class PersonaCheckSetup:
    """What the analyses need to know about the run, taken from the run configuration."""
    persona_set_name: str
    persona_name: str
    target_group: str
    control_group: str
    same_values_group: object          # group name, or None when the persona set has no same-values control
    direction_names_by_role: dict      # {"toward": ..., "away": ... (optional), "neutral": ...}
    pass_criteria: dict
    expected_conditions_per_group: dict
    model_type: str
    model_name: str
    model_revision: str
    configuration: dict = field(repr=False, default_factory=dict)

    @property
    def persona_group_names(self):
        names = [self.target_group, self.control_group]
        if self.same_values_group:
            names.append(self.same_values_group)
        return names

    @property
    def model_label(self):
        return f"{self.model_type} model {self.model_name}"

    @property
    def target_minus_control_score_column(self):
        return f"{self.persona_name}_minus_{self.control_group}_score"

    @property
    def pooled_control_persona_name(self):
        return f"pooled_{self.control_group}"

    def direction_role_of(self, direction_name):
        for role, name in self.direction_names_by_role.items():
            if name == direction_name:
                return role
        return None


@dataclass
class DocumentScores:
    """One document condensed from its .npz file. Arrays are float64."""
    document_id: str
    token_ids: list
    token_strings: list
    condition_scores: np.ndarray                   # mean over tokens of log_probabilities[c] - log_probabilities[generic]
    generic_mean_log_probability: float
    per_token_difference_by_persona: dict          # {persona_name: mean over its rewordings of log_probabilities[c] - generic}
    total_log_probability_by_group: dict           # {condition_group: mean over its conditions of the summed log probability}

    @property
    def number_of_tokens(self):
        return len(self.token_strings)


@dataclass
class LoadedRun:
    run_folder: Path
    setup: PersonaCheckSetup
    documents: list
    conditions: list
    scores_by_document: dict                       # {document_id: DocumentScores}
    condition_ids: list
    group_indices: dict                            # {condition_group: [row indices in conditions order]}
    persona_names_by_group: dict                   # {condition_group: [persona names in condition order]}
    is_legacy_format: bool = False


# ---------------------------------------------------------------------------------------------------
# Run configuration and model type consistency
# ---------------------------------------------------------------------------------------------------

def read_run_configurations(run_folder, legacy_run_persona_set=None):
    """Every run_configuration_*.json of the run folder, oldest first (resumes append further files). A file in the
    legacy run format (no persona_set record) is refused unless `legacy_run_persona_set` names the persona set to
    pair it with; it is then converted by the compatibility reader."""
    paths = sorted(Path(run_folder).glob("run_configuration_*.json"))
    if not paths:
        raise ValueError(f"No run_configuration_*.json in {run_folder}")
    configurations, converted = [], False
    for path in paths:
        configuration = json.loads(path.read_text(encoding="utf-8"))
        if "persona_set" not in configuration:
            if not legacy_run_persona_set:
                raise ValueError(
                    f"{path.name} has no persona_set record, so it was not written by build_scoring_inputs.py. "
                    f"If it is a legacy-format run, pass the persona set to pair it with "
                    f"(--legacy-run-persona-set <name>).")
            configuration = normalize_legacy_run_configuration(configuration, legacy_run_persona_set, run_folder)
            converted = True
        configurations.append(configuration)
    return configurations, converted


def check_one_model_type_in_run(configurations, run_folder):
    """Raises MixedModelTypesError unless every run configuration, scoring environment record and full-logits
    verification record names the same model type, model name and model revision (where it records them)."""
    seen = {}

    def note(field_name, value, source):
        if value is not None:
            seen.setdefault(field_name, {}).setdefault(value, []).append(source)

    for index, configuration in enumerate(configurations):
        source = f"run configuration {index + 1}"
        note("model_type", configuration.get("model_type"), source)
        note("model_name", configuration.get("model_name"), source)
        note("model_revision", configuration.get("model_revision"), source)
    scoring_outputs = Path(run_folder) / "scoring_outputs"
    for path in sorted(scoring_outputs.glob("scoring_environment_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        note("model_type", record.get("model_type"), path.name)
        note("model_name", record.get("model_name"), path.name)
        revision = record.get("model_revision") or record.get("model_revision_requested")
        note("model_revision", revision, path.name)
    for path in sorted(scoring_outputs.glob("full_logits_verification_*.json")):
        note("model_type", json.loads(path.read_text(encoding="utf-8")).get("model_type"), path.name)
    for field_name, values in seen.items():
        if len(values) > 1:
            detail = "; ".join(f"{value!r} in {', '.join(sources)}" for value, sources in values.items())
            raise MixedModelTypesError(
                f"{run_folder} mixes {field_name} values ({detail}). Scores from different model types or models are "
                f"not comparable; analyse each run folder separately.")
    return {field_name: next(iter(values)) for field_name, values in seen.items()}


def build_setup(configurations, run_folder):
    consistent_values = check_one_model_type_in_run(configurations, run_folder)
    configuration = configurations[0]
    persona_configuration = configuration["persona_set"]["configuration"]
    groups = persona_configuration["condition_groups"]
    expected_counts = {groups["target"]["condition_group"]: groups["target"]["expected_rewording_count"],
                       groups["topic_control"]["condition_group"]:
                           groups["topic_control"]["expected_persona_count"]
                           * groups["topic_control"]["expected_rewordings_per_persona"]}
    same_values_group = None
    if "same_values_control" in groups:
        same_values_group = groups["same_values_control"]["condition_group"]
        expected_counts[same_values_group] = groups["same_values_control"]["expected_rewording_count"]
    for required in ("model_type", "model_name", "model_revision"):
        if required not in consistent_values:
            raise ValueError(f"The run configuration of {run_folder} does not record {required}")
    return PersonaCheckSetup(
        persona_set_name=persona_configuration["persona_set_name"], persona_name=persona_configuration["persona_name"],
        target_group=groups["target"]["condition_group"], control_group=groups["topic_control"]["condition_group"],
        same_values_group=same_values_group,
        direction_names_by_role={role: entry["name"] for role, entry in persona_configuration["rewrite_directions"].items()},
        pass_criteria=configuration.get("pass_criteria") or persona_configuration.get("pass_criteria") or {},
        expected_conditions_per_group=expected_counts, model_type=consistent_values["model_type"],
        model_name=consistent_values["model_name"], model_revision=consistent_values["model_revision"],
        configuration=persona_configuration)


# ---------------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------------

def check_conditions(conditions, setup):
    condition_ids = [condition["condition_id"] for condition in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("conditions.jsonl contains duplicate condition ids")
    for required_id in (GENERIC_CONDITION_ID, NO_SYSTEM_PROMPT_CONDITION_ID):
        if required_id not in condition_ids:
            raise ValueError(f"conditions.jsonl has no '{required_id}' condition")
    expected_groups = {GENERIC_CONDITION_ID, NO_SYSTEM_PROMPT_CONDITION_ID, *setup.persona_group_names}
    groups_present = {condition["condition_group"] for condition in conditions}
    if groups_present != expected_groups:
        raise ValueError(f"conditions.jsonl condition groups {sorted(groups_present)} differ from the groups the persona "
                         f"set configuration names, {sorted(expected_groups)}")
    for group, expected_count in setup.expected_conditions_per_group.items():
        actual_count = sum(1 for condition in conditions if condition["condition_group"] == group)
        if actual_count != expected_count:
            raise ValueError(f"conditions.jsonl has {actual_count} conditions in group '{group}'; the persona set "
                             f"configuration expects {expected_count}")


def condense_document(document_id, archive_path, conditions, setup, group_indices, persona_row_groups):
    """Reads one .npz and returns DocumentScores. Raises ValueError on any inconsistency with conditions.jsonl."""
    condition_ids_expected = [condition["condition_id"] for condition in conditions]
    with np.load(archive_path, allow_pickle=False) as archive:
        condition_ids = [str(value) for value in archive["condition_ids"]]
        log_probabilities = np.asarray(archive["log_probabilities"], dtype=np.float64)
        token_strings = [str(value) for value in archive["document_token_strings"]]
        token_ids = [int(value) for value in archive["document_token_ids"]]
    if set(condition_ids) != set(condition_ids_expected) or len(condition_ids) != len(condition_ids_expected):
        missing = sorted(set(condition_ids_expected) - set(condition_ids))
        extra = sorted(set(condition_ids) - set(condition_ids_expected))
        raise ValueError(f"Condition set for document {document_id} does not match conditions.jsonl "
                         f"(missing {missing[:5]}, unexpected {extra[:5]})")
    if log_probabilities.shape != (len(condition_ids), len(token_strings)):
        raise ValueError(f"Document {document_id}: log_probabilities shape {log_probabilities.shape} does not match "
                         f"{len(condition_ids)} conditions x {len(token_strings)} tokens")
    row_for_condition = {condition_id: row for row, condition_id in enumerate(condition_ids)}
    log_probabilities = log_probabilities[[row_for_condition[condition_id] for condition_id in condition_ids_expected]]
    generic_row = condition_ids_expected.index(GENERIC_CONDITION_ID)
    differences = log_probabilities - log_probabilities[generic_row]
    summed = log_probabilities.sum(axis=1)
    per_token_by_persona = {}
    if log_probabilities.shape[1] > 0:
        for persona_name, rows in persona_row_groups.items():
            per_token_by_persona[persona_name] = differences[rows].mean(axis=0)
        condition_scores = differences.mean(axis=1)
        generic_mean = float(log_probabilities[generic_row].mean())
    else:
        condition_scores = np.full(len(condition_ids_expected), np.nan)
        generic_mean = float("nan")
    return DocumentScores(
        document_id=document_id, token_ids=token_ids, token_strings=token_strings, condition_scores=condition_scores,
        generic_mean_log_probability=generic_mean, per_token_difference_by_persona=per_token_by_persona,
        total_log_probability_by_group={group: float(summed[rows].mean()) for group, rows in group_indices.items()})


def load_run(run_folder, legacy_run_persona_set=None):
    """Returns a LoadedRun. Raises ValueError (MixedModelTypesError for mixed model types) on any inconsistency."""
    run_folder = Path(run_folder)
    configurations, is_legacy_format = read_run_configurations(run_folder, legacy_run_persona_set)
    setup = build_setup(configurations, run_folder)
    documents = utils.load_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    conditions = utils.load_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    if not documents or not conditions:
        raise ValueError(f"{run_folder / 'scoring_inputs'} has no documents or no conditions")
    check_conditions(conditions, setup)
    group_indices, persona_row_groups, persona_names_by_group = {}, {}, {}
    for row, condition in enumerate(conditions):
        group = condition["condition_group"]
        group_indices.setdefault(group, []).append(row)
        if group in setup.persona_group_names:
            persona_row_groups.setdefault(condition["persona_name"], []).append(row)
            names = persona_names_by_group.setdefault(group, [])
            if condition["persona_name"] not in names:
                names.append(condition["persona_name"])
    scores_by_document = {}
    for document in documents:
        archive_path = run_folder / "scoring_outputs" / "per_token_log_probabilities" / f"{document['document_id']}.npz"
        if not archive_path.exists():
            raise ValueError(f"Missing scoring output for document {document['document_id']}: {archive_path}")
        scores_by_document[document["document_id"]] = condense_document(
            document["document_id"], archive_path, conditions, setup, group_indices, persona_row_groups)
    return LoadedRun(run_folder=run_folder, setup=setup, documents=documents, conditions=conditions,
                     scores_by_document=scores_by_document, condition_ids=[c["condition_id"] for c in conditions],
                     group_indices=group_indices, persona_names_by_group=persona_names_by_group,
                     is_legacy_format=is_legacy_format)


def insert_model_labels(table, setup):
    """Adds model_type and model_name as the first two columns of a DataFrame."""
    table = table.copy()
    table.insert(0, "model_name", setup.model_name)
    table.insert(0, "model_type", setup.model_type)
    return table
