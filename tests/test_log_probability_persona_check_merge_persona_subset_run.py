"""Offline tests for merge_persona_subset_run_into_full_run.py with small synthetic runs: the merged conditions and
scores, the re-expression against the base run's generic, the reproducibility figures in the merge log, and every
refusal."""

import copy
import json

import numpy as np
import pytest

from evals.log_probability_persona_check import analysis_run_loading
from evals.log_probability_persona_check import merge_persona_subset_run_into_full_run as merge
from synthetic_scored_run_folder_for_log_probability_persona_check_analysis import (
    SAME_VALUES_GROUP, build_conditions, persona_check_configuration, write_jsonl_records)

TOKENS = 12
DOCUMENT_IDS = ["document_a", "document_b"]
FIRST_SAME_VALUES_PERSONA = "even_handed_advocate"
SECOND_SAME_VALUES_PERSONA = "restrained_advocate"


def condition_records(persona_names_to_drop=()):
    return [condition for condition in build_conditions() if condition["persona_name"] not in persona_names_to_drop]


def subset_condition_records():
    return [condition for condition in build_conditions()
            if condition["persona_name"] in ("generic", FIRST_SAME_VALUES_PERSONA, SECOND_SAME_VALUES_PERSONA)]


def write_run(run_folder, conditions, log_probability_for, persona_configuration, document_ids=DOCUMENT_IDS,
              token_offset=0, extra_configuration=None, system_prompt_override=None):
    """log_probability_for(document_id, condition_id) -> array of TOKENS values."""
    conditions = copy.deepcopy(conditions)
    if system_prompt_override:
        for condition in conditions:
            if condition["condition_id"] == system_prompt_override[0]:
                condition["system_prompt"] = system_prompt_override[1]
    write_jsonl_records(run_folder / "scoring_inputs" / "conditions.jsonl", conditions)
    documents = [{"document_id": document_id, "source_document_id": document_id, "rewrite_direction": None,
                  "content": f"content of {document_id}", "language": "English", "variables": {},
                  "generation_prompt": "p"} for document_id in document_ids]
    write_jsonl_records(run_folder / "scoring_inputs" / "documents.jsonl", documents)
    configuration = {"run_folder_name": run_folder.name, "created_timestamp": "2026-01-01_00-00", "model_type": "instruct",
                     "model_name": "test/model", "model_revision": "a" * 40,
                     "persona_set": {"name": persona_configuration["persona_set_name"], "configuration": persona_configuration},
                     "pass_criteria": persona_configuration["pass_criteria"],
                     "document_selection": {"corpus_path": "c.jsonl", "document_ids_file": None, "maximum_documents": None,
                                            "extra_documents_files": [], "number_of_corpus_documents": len(documents),
                                            "number_of_extra_documents": 0, "document_ids": list(document_ids)},
                     "number_of_conditions": len(conditions), **(extra_configuration or {})}
    (run_folder / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(configuration))
    npz_folder = run_folder / "scoring_outputs" / "per_token_log_probabilities"
    npz_folder.mkdir(parents=True)
    for document_id in document_ids:
        rows = np.stack([log_probability_for(document_id, condition["condition_id"]) for condition in conditions])
        np.savez(npz_folder / f"{document_id}.npz", document_token_ids=np.arange(TOKENS, dtype=np.int32) + token_offset,
                 document_token_strings=np.array([f"t{index}" for index in range(TOKENS)]),
                 condition_ids=np.array([condition["condition_id"] for condition in conditions]),
                 log_probabilities=rows.astype(np.float32))


def deterministic_scores(run_shift_generic, run_shift_persona=0.0):
    """Log probability = a fixed per-condition, per-document pattern plus the run's own shift for generic (and a
    smaller one for the first same-values persona), so both runs share the persona effects but not the baseline."""
    def scores(document_id, condition_id):
        seed = sum(map(ord, document_id + condition_id))
        pattern = -2.0 - np.random.default_rng(seed).random(TOKENS)
        if condition_id == "generic":
            return pattern + run_shift_generic
        if condition_id.startswith(FIRST_SAME_VALUES_PERSONA):
            return pattern + run_shift_persona
        return pattern
    return scores


@pytest.fixture
def runs(tmp_path):
    base_configuration = persona_check_configuration()
    base_configuration["condition_groups"]["same_values_control"]["expected_persona_count"] = 1
    subset_configuration = persona_check_configuration()
    base = tmp_path / "base_run"
    write_run(base, condition_records((SECOND_SAME_VALUES_PERSONA,)), deterministic_scores(0.0), base_configuration)
    subset = tmp_path / "subset_run"
    write_run(subset, subset_condition_records(), deterministic_scores(0.1, 0.02), subset_configuration,
              document_ids=DOCUMENT_IDS + ["document_only_in_subset"],
              extra_configuration={"condition_subset": {"only_personas": [FIRST_SAME_VALUES_PERSONA,
                                                                          SECOND_SAME_VALUES_PERSONA]}})
    return base, subset, tmp_path / "merged_run"


def read_scores(run_folder, document_id):
    with np.load(run_folder / "scoring_outputs" / "per_token_log_probabilities" / f"{document_id}.npz") as archive:
        return list(archive["condition_ids"]), archive["log_probabilities"].astype(np.float64)


def test_merged_run_has_every_condition_in_full_set_order_and_loads_in_the_analysis(runs):
    base, subset, merged = runs
    merge.merge_runs(base, subset, merged)
    conditions = [c["condition_id"] for c in json.loads("[" + ",".join(
        (merged / "scoring_inputs" / "conditions.jsonl").read_text().splitlines()) + "]")]
    assert conditions == [c["condition_id"] for c in build_conditions()]
    loaded = analysis_run_loading.load_run(merged)
    assert set(loaded.scores_by_document) == set(DOCUMENT_IDS)  # the subset-only document is left out
    configuration = json.loads(next(merged.glob("run_configuration_*.json")).read_text())
    assert configuration["number_of_conditions"] == len(conditions) and configuration["condition_subset"] is None
    assert configuration["merged_from"]["documents_only_in_subset_run"] == ["document_only_in_subset"]
    assert configuration["merged_from"]["base_run"]["run_folder"] == str(base)


def test_new_conditions_keep_their_own_runs_score_against_the_base_generic(runs):
    base, subset, merged = runs
    merge.merge_runs(base, subset, merged)
    for document_id in DOCUMENT_IDS:
        merged_ids, merged_rows = read_scores(merged, document_id)
        base_ids, base_rows = read_scores(base, document_id)
        subset_ids, subset_rows = read_scores(subset, document_id)
        merged_generic = merged_rows[merged_ids.index("generic")]
        assert np.allclose(merged_generic, base_rows[base_ids.index("generic")])
        for condition_id in merged_ids:
            if condition_id.startswith(SECOND_SAME_VALUES_PERSONA):
                own_score = subset_rows[subset_ids.index(condition_id)] - subset_rows[subset_ids.index("generic")]
                assert np.allclose(merged_rows[merged_ids.index(condition_id)] - merged_generic, own_score, atol=1e-5)
            else:
                assert np.allclose(merged_rows[merged_ids.index(condition_id)], base_rows[base_ids.index(condition_id)])


def test_merge_log_reports_generic_and_overlapping_persona_differences(runs):
    base, subset, merged = runs
    merge.merge_runs(base, subset, merged)
    (log_path,) = merged.glob("merge_log_*.json")
    log = json.loads(log_path.read_text())
    overall = log["differences_between_the_two_runs"]["overall"]
    assert overall["generic"]["maximum_absolute_difference"] == pytest.approx(0.1, abs=1e-5)
    assert overall["generic"]["mean_absolute_difference"] == pytest.approx(0.1, abs=1e-5)
    assert overall["generic"]["number_of_tokens"] == TOKENS * len(DOCUMENT_IDS)
    persona = log["differences_between_the_two_runs"]["by_overlapping_persona"][FIRST_SAME_VALUES_PERSONA]
    assert persona["log_probability"]["maximum_absolute_difference"] == pytest.approx(0.02, abs=1e-5)
    # relative to generic, the two runs differ by 0.02 - 0.1
    assert persona["score_relative_to_generic"]["mean_absolute_difference"] == pytest.approx(0.08, abs=1e-5)
    assert set(log["differences_per_document"]) == set(DOCUMENT_IDS)
    assert log["new_condition_ids"] == [f"{SECOND_SAME_VALUES_PERSONA}__rewording_0{i}" for i in (1, 2)]
    assert (merged / next(merged.glob("merge_log_*.md")).name).read_text().startswith("# Merge log")


def test_refuses_when_document_token_ids_differ_and_leaves_no_output_folder(tmp_path, runs):
    base, subset, merged = runs
    other_subset = tmp_path / "subset_with_other_tokens"
    write_run(other_subset, subset_condition_records(), deterministic_scores(0.1), persona_check_configuration(),
              token_offset=1)
    with pytest.raises(merge.RunsCannotBeMergedError, match="different token ids"):
        merge.merge_runs(base, other_subset, merged)
    assert not merged.exists()


def test_refuses_a_different_model_revision(tmp_path, runs):
    base, subset, merged = runs
    other = tmp_path / "other_revision_subset"
    write_run(other, subset_condition_records(), deterministic_scores(0.1), persona_check_configuration(),
              extra_configuration={"model_revision": "b" * 40})
    with pytest.raises(merge.RunsCannotBeMergedError, match="model_revision"):
        merge.merge_runs(base, other, merged)


def test_refuses_a_different_system_prompt_for_a_shared_condition(tmp_path, runs):
    base, subset, merged = runs
    other = tmp_path / "other_prompt_subset"
    write_run(other, subset_condition_records(), deterministic_scores(0.1), persona_check_configuration(),
              system_prompt_override=("generic", "You are a helpful assistant."))
    with pytest.raises(merge.RunsCannotBeMergedError, match="different system prompt"):
        merge.merge_runs(base, other, merged)


def test_refuses_when_the_subset_run_lacks_a_base_document(tmp_path, runs):
    base, subset, merged = runs
    other = tmp_path / "short_subset"
    write_run(other, subset_condition_records(), deterministic_scores(0.1), persona_check_configuration(),
              document_ids=DOCUMENT_IDS[:1])
    with pytest.raises(merge.RunsCannotBeMergedError, match="lacks 1 of the base run's documents"):
        merge.merge_runs(base, other, merged)


def test_refuses_an_existing_output_folder(runs):
    base, subset, merged = runs
    merged.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        merge.merge_runs(base, subset, merged)


def test_a_legacy_format_base_run_is_merged_through_the_compatibility_reader(tmp_path):
    persona_set_name = "preachiness"
    # Legacy format: no persona_set record, group named after the persona in the preachiness set.
    from evals.log_probability_persona_check import persona_sets
    persona_set = persona_sets.load_persona_set(persona_set_name)
    from evals.log_probability_persona_check import prompt_rendering
    all_conditions = prompt_rendering.build_condition_records(persona_set)
    base_conditions = [dict(c, condition_group="proportionate_welfare") if c["condition_group"] == "same_values_control" else c
                       for c in all_conditions if c["persona_name"] not in
                       ("welfare_care_without_preachiness", "ethical_farming_without_pushiness")]
    subset_conditions = [c for c in all_conditions if c["persona_name"] in
                         ("generic", "proportionate_welfare", "welfare_care_without_preachiness",
                          "ethical_farming_without_pushiness")]
    persona_configuration = persona_set.configuration
    base = tmp_path / "legacy_run"
    write_run(base, base_conditions, deterministic_scores(0.0), persona_configuration)
    legacy_configuration = {"run_name": "legacy", "created_timestamp": "2026-01-01_00-00",
                              "model_name": "test/model", "document_selection": {}}
    (base / "run_configuration_2026-01-01_00-00.json").write_text(json.dumps(legacy_configuration))
    (base / "scoring_outputs" / "scoring_environment_2026-01-01_00-00.json").write_text(
        json.dumps({"model_name": "test/model", "model_revision_requested": "a" * 40}))
    subset = tmp_path / "subset_run"
    write_run(subset, subset_conditions, deterministic_scores(0.1), persona_configuration)
    merged = tmp_path / "merged_run"
    merge.merge_runs(base, subset, merged, base_run_legacy_persona_set=persona_set_name)
    loaded = analysis_run_loading.load_run(merged)
    assert len(loaded.conditions) == 72
    assert loaded.setup.same_values_group == "same_values_control"
