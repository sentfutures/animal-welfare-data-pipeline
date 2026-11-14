"""``plans_and_drafts_of_an_sdf_run``: the plan and draft records the report, the audit and the shoehorning
postprocess read, checked against the stage files of every committed SDF run (all layouts)."""

from pathlib import Path

import pytest

from evals import shoehorning_audit_postprocess_sdf as shoehorning_postprocess
from evals.plans_and_drafts_of_an_sdf_run import draft_texts_by_document_id, plan_records_of_run
from shared import utils

RUNS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "sdf" / "runs"
COMMITTED_RUN_NAMES = sorted(path.name for path in RUNS_ROOT.iterdir() if path.is_dir()) if RUNS_ROOT.is_dir() else []


def test_the_committed_runs_are_present_for_the_checks_below():
    assert len(COMMITTED_RUN_NAMES) >= 7


@pytest.mark.parametrize("run_name", COMMITTED_RUN_NAMES)
def test_plan_records_carry_what_the_plan_file_does(run_name):
    run = RUNS_ROOT / run_name
    plan_file_records = utils.load_jsonl(utils.sdf_stage_file(run, "plan"))
    records = plan_records_of_run(run)
    assert len(records) == len(plan_file_records)
    by_id = {record["prompt_id"]: record for record in records}
    for plan in plan_file_records:
        record = by_id[plan["prompt_id"]]
        assert record["variables"] == plan.get("variables")
        assert record["description"] == (plan.get("description") or None)


@pytest.mark.parametrize("run_name", COMMITTED_RUN_NAMES)
def test_draft_texts_are_the_draft_files_content_for_every_draft(run_name):
    run = RUNS_ROOT / run_name
    assert draft_texts_by_document_id(run) == {
        record["doc_id"]: record["content"] for record in utils.load_jsonl(utils.sdf_stage_file(run, "draft"))}


def test_the_layout_from_before_the_variables_matrix_has_no_plan_records():
    assert plan_records_of_run(RUNS_ROOT / "2026-07-08_14-59_allen_dev") == []


def test_a_plan_declared_incoherent_has_a_record_but_no_description(tmp_path):
    plan_directory = tmp_path / "layer1"
    plan_directory.mkdir()
    utils.save_jsonl([{"prompt_id": "matrix_000000", "variables": {"document_type": "essay"}, "round": 0}],
                     plan_directory / "prompts.jsonl")
    utils.save_jsonl([
        {"prompt_id": "matrix_000000", "variables": {"document_type": "essay"}, "plan": "notes", "description": "d"},
        {"prompt_id": "matrix_000001", "variables": {"document_type": "poem"}, "plan": "notes", "incoherent": True,
         "description": ""},
    ], plan_directory / "plans.jsonl")
    assert plan_records_of_run(tmp_path) == [
        {"prompt_id": "matrix_000000", "variables": {"document_type": "essay"}, "description": "d"},
        {"prompt_id": "matrix_000001", "variables": {"document_type": "poem"}, "description": None}]


def test_plan_records_come_in_deal_order_by_round_then_hand_with_unrecorded_rounds_left_out(tmp_path):
    screen_directory = tmp_path / "layer1_deal_and_screen"
    plan_directory = tmp_path / "layer2_plan_and_description_rewrite"
    screen_directory.mkdir()
    plan_directory.mkdir()
    hand = lambda number, round_number: {"prompt_id": f"matrix_{number:06d}", "round": round_number, "variables": {}}
    utils.save_jsonl([hand(2, 0), hand(10, 1), hand(1, 0), hand(5, 2)], screen_directory / "screen_hands.jsonl")
    utils.save_jsonl([{"round": 0, "prompt_ids": ["matrix_000002", "matrix_000001"]},
                      {"round": 1, "prompt_ids": ["matrix_000010"]}], screen_directory / "screen_rounds.jsonl")
    utils.save_jsonl([{"prompt_id": f"matrix_{number:06d}", "variables": {}, "description": "d"}
                      for number in (10, 5, 2, 1, 99)], plan_directory / "plans.jsonl")
    assert [record["prompt_id"] for record in plan_records_of_run(tmp_path)] == [
        "matrix_000001", "matrix_000002", "matrix_000010", "matrix_000005", "matrix_000099"]


def test_a_folder_that_is_not_a_run_folder_has_no_plans_and_no_drafts(tmp_path):
    for folder in (tmp_path, tmp_path / "no_such_folder"):
        assert plan_records_of_run(folder) == [] and draft_texts_by_document_id(folder) == {}


@pytest.mark.parametrize("run_name", COMMITTED_RUN_NAMES)
def test_the_shoehorning_postprocess_finds_the_drafts_the_draft_file_holds(run_name):
    run = RUNS_ROOT / run_name
    expected = {record["doc_id"]: record["content"] for record in utils.load_jsonl(utils.sdf_stage_file(run, "draft"))}
    found = shoehorning_postprocess.default_drafts_by_id(run / "final" / "sdf_corpus.jsonl")
    assert {doc_id: record["content"] for doc_id, record in found.items()} == expected


def test_the_shoehorning_postprocess_finds_no_drafts_for_a_corpus_outside_a_final_folder(tmp_path):
    assert shoehorning_postprocess.default_drafts_by_id(tmp_path / "loose_corpus.jsonl") == {}
    assert shoehorning_postprocess.default_drafts_by_id(tmp_path / "some_run" / "final" / "sdf_corpus.jsonl") == {}
