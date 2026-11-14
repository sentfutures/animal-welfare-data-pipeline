"""The loader's extensions for readers that used to open stage files themselves: ``has_plan_record``,
``plan_record_variables``, ``planned_description``, ``draft_text``, ``plan_records_of_states`` and
``load_document_states_or_empty_list_when_not_a_run_folder``.

Each is checked against the stage files of every committed SDF run (all three layouts), read here the way
the report, the audit and the shoehorning postprocess read them before they moved to the loader.
"""

from pathlib import Path

import pytest

from evals import shoehorning_audit_postprocess_sdf as shoehorning_postprocess
from shared import utils
from sdf_pipeline.shared_step_machinery.document_state_loader import (
    load_document_states, load_document_states_or_empty_list_when_not_a_run_folder, plan_records_of_states)

RUNS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "sdf" / "runs"
COMMITTED_RUN_NAMES = sorted(path.name for path in RUNS_ROOT.iterdir() if path.is_dir()) if RUNS_ROOT.is_dir() else []


def test_the_committed_runs_are_present_for_the_reader_extension_checks():
    assert len(COMMITTED_RUN_NAMES) >= 7


@pytest.mark.parametrize("run_name", COMMITTED_RUN_NAMES)
def test_plan_records_of_states_carry_what_the_plan_file_does(run_name):
    run = RUNS_ROOT / run_name
    plan_file_records = utils.load_jsonl(utils.sdf_stage_file(run, "plan"))
    from_states = plan_records_of_states(load_document_states(run))
    assert len(from_states) == len(plan_file_records)
    by_id = {record["prompt_id"]: record for record in from_states}
    for plan in plan_file_records:
        record = by_id[plan["prompt_id"]]
        assert record["variables"] == plan.get("variables")
        assert (record["description"] or None) == (plan.get("description") or None)


@pytest.mark.parametrize("run_name", COMMITTED_RUN_NAMES)
def test_draft_text_is_the_draft_files_content_for_every_draft_and_none_for_the_rest(run_name):
    run = RUNS_ROOT / run_name
    draft_content_by_id = {record["doc_id"]: record["content"]
                           for record in utils.load_jsonl(utils.sdf_stage_file(run, "draft"))}
    states = load_document_states(run)
    assert {s.document_id: s.draft_text for s in states if s.draft_text is not None} == draft_content_by_id
    assert all(s.draft_text is None for s in states if s.document_id not in draft_content_by_id)


def test_the_five_layer_layout_has_no_plan_records():
    states = load_document_states(RUNS_ROOT / "2026-07-08_14-59_allen_dev")
    assert not any(s.has_plan_record for s in states) and plan_records_of_states(states) == []
    assert all(s.plan_record_variables is None and s.planned_description is None for s in states)


def test_a_plan_declared_incoherent_has_a_record_but_no_planned_description(tmp_path):
    plan_directory = tmp_path / "layer1"
    plan_directory.mkdir()
    utils.save_jsonl([{"prompt_id": "matrix_000000", "variables": {"document_type": "essay"}, "round": 0}],
                     plan_directory / "prompts.jsonl")
    utils.save_jsonl([
        {"prompt_id": "matrix_000000", "variables": {"document_type": "essay"}, "plan": "notes", "description": "d"},
        {"prompt_id": "matrix_000001", "variables": {"document_type": "poem"}, "plan": "notes", "incoherent": True,
         "description": ""},
    ], plan_directory / "plans.jsonl")
    records = plan_records_of_states(load_document_states(tmp_path))
    assert {r["prompt_id"]: (r["variables"], r["description"]) for r in records} == {
        "matrix_000000": ({"document_type": "essay"}, "d"), "matrix_000001": ({"document_type": "poem"}, None)}


def test_a_folder_that_is_not_a_run_folder_gives_no_states_for_the_lenient_loader(tmp_path):
    assert load_document_states_or_empty_list_when_not_a_run_folder(tmp_path) == []
    assert load_document_states_or_empty_list_when_not_a_run_folder(tmp_path / "no_such_folder") == []
    assert len(load_document_states_or_empty_list_when_not_a_run_folder(
        RUNS_ROOT / "2026-07-11_20-06_matrix100-cli")) == 100


@pytest.mark.parametrize("run_name", COMMITTED_RUN_NAMES)
def test_the_shoehorning_postprocess_finds_the_same_drafts_through_the_loader_as_in_the_draft_file(run_name):
    run = RUNS_ROOT / run_name
    expected = {record["doc_id"]: record["content"] for record in utils.load_jsonl(utils.sdf_stage_file(run, "draft"))}
    found = shoehorning_postprocess.default_drafts_by_id(run / "final" / "sdf_corpus.jsonl")
    assert {doc_id: record["content"] for doc_id, record in found.items()} == expected


def test_the_shoehorning_postprocess_finds_no_drafts_for_a_corpus_outside_a_final_folder(tmp_path):
    assert shoehorning_postprocess.default_drafts_by_id(tmp_path / "loose_corpus.jsonl") == {}
    assert shoehorning_postprocess.default_drafts_by_id(tmp_path / "some_run" / "final" / "sdf_corpus.jsonl") == {}
