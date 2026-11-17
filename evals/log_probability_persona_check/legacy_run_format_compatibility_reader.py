"""Compatibility reader for run folders in the legacy format of the log-probability persona check.

The legacy run configuration has no `model_type`, `model_revision`, `persona_set` or `pass_criteria` records, and its
file hashes use the old prompt file names. This module converts such a configuration, in memory, to the current run
configuration shape so the current analysis can re-analyse the old score files. Nothing in the run folder is changed.

What it assumes, because the old record does not say:
- the model type is `instruct` (the legacy version only scored instruction-tuned models through the chat template);
- the persona set is the one the caller names (for the first preachiness runs, `preachiness`), whose
  `persona_check_configuration.json` supplies the group names, direction names and pass criteria;
- the model revision is `model_revision_requested` from the run's `scoring_environment_*.json`;
- the legacy run's single same-values persona (condition group "proportionate_welfare", 5 rewordings) is the
  persona set's same-values control group: the converted configuration expects as many same-values personas and
  rewordings per persona as the run's `conditions.jsonl` holds, and `rename_legacy_condition_groups` renames that
  group to the configured group name when the run's conditions are loaded.

Only the analysis uses this. It is not a way to resume or rebuild a legacy run.
"""

import json
from pathlib import Path

PERSONA_CHECKS_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "prompts" / "tools" / "persona_checks"
LEGACY_MODEL_TYPE = "instruct"
LEGACY_SAME_VALUES_CONDITION_GROUP = "proportionate_welfare"


def read_model_revision_from_scoring_environment(run_folder):
    paths = sorted((Path(run_folder) / "scoring_outputs").glob("scoring_environment_*.json"))
    if not paths:
        raise ValueError(f"{run_folder} has no scoring_environment_*.json to take the model revision from")
    record = json.loads(paths[-1].read_text(encoding="utf-8"))
    revision = record.get("model_revision") or record.get("model_revision_requested") or record.get("model_commit_hash")
    if not revision:
        raise ValueError(f"{paths[-1].name} records no model revision")
    return revision


def read_legacy_conditions(run_folder):
    path = Path(run_folder) / "scoring_inputs" / "conditions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_expected_same_values_counts_from_legacy_conditions(persona_configuration, run_folder):
    """Makes the same-values control group expect what the legacy run contains (one persona), in place."""
    same_values = persona_configuration["condition_groups"].get("same_values_control")
    if same_values is None:
        return
    rewordings_by_persona = {}
    for condition in read_legacy_conditions(run_folder):
        if condition["condition_group"] == LEGACY_SAME_VALUES_CONDITION_GROUP:
            rewordings_by_persona.setdefault(condition["persona_name"], []).append(condition["condition_id"])
    if not rewordings_by_persona:
        return
    counts = {len(rewordings) for rewordings in rewordings_by_persona.values()}
    if len(counts) != 1:
        raise ValueError(f"The legacy run's same-values personas have different numbers of rewordings: "
                         f"{ {name: len(ids) for name, ids in rewordings_by_persona.items()} }")
    same_values["expected_persona_count"] = len(rewordings_by_persona)
    same_values["expected_rewordings_per_persona"] = counts.pop()


def rename_legacy_condition_groups(conditions, persona_configuration):
    """Returns copies of the conditions with the legacy same-values group renamed to the configured group name."""
    same_values = persona_configuration["condition_groups"].get("same_values_control")
    if same_values is None:
        return conditions
    renamed = []
    for condition in conditions:
        if condition["condition_group"] == LEGACY_SAME_VALUES_CONDITION_GROUP:
            condition = {**condition, "condition_group": same_values["condition_group"]}
        renamed.append(condition)
    return renamed


def normalize_legacy_run_configuration(legacy_configuration, persona_set_name, run_folder):
    """Returns a dict with the fields the analysis reads from a current run configuration."""
    configuration_path = PERSONA_CHECKS_DIRECTORY / persona_set_name / "persona_check_configuration.json"
    if not configuration_path.exists():
        raise ValueError(f"No persona set {persona_set_name!r}: {configuration_path} does not exist")
    persona_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    set_expected_same_values_counts_from_legacy_conditions(persona_configuration, run_folder)
    return {
        "run_folder_name": legacy_configuration.get("run_name"),
        "created_timestamp": legacy_configuration.get("created_timestamp"),
        "model_type": LEGACY_MODEL_TYPE,
        "model_name": legacy_configuration["model_name"],
        "model_revision": read_model_revision_from_scoring_environment(run_folder),
        "persona_set": {"name": persona_set_name, "configuration": persona_configuration},
        "pass_criteria": persona_configuration["pass_criteria"],
        "converted_from_legacy_run_configuration": True,
    }
