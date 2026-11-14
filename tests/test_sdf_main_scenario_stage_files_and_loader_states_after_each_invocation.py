"""What the stage files, the loader and the requests say on the fixture's main scenario, after each of its two invocations.

The main scenario runs the whole pipeline twice (the second invocation retries the failures). After
each invocation the document states the loader builds are compared with the stage files, and the
retried documents are followed from the first invocation to the second through the stage files and
the recorded requests (an accepted critique is not sent again; a failed plan, draft or revise is
done after the resume).
"""

import tempfile
from pathlib import Path

import pytest

import sdf_prompt_equivalence_fixture_support as support
import sdf_prompt_equivalence_scenarios as scenarios
from sdf_pipeline.document_state_loader import load_document_states
from shared import utils


@pytest.fixture(scope="module")
def main_scenario():
    """(what was on disk after each invocation, the scenario's recorded requests)."""
    collected = []
    with tempfile.TemporaryDirectory() as base_directory:
        def capture(number, run_directory):
            def records(layer_number, file_name):
                return utils.load_jsonl(utils.sdf_stage_dir(run_directory, layer_number) / file_name)
            collected.append({
                "states": load_document_states(run_directory),
                "plan_records": records(1, "plans.jsonl"),
                "draft_records": records(2, "drafts.jsonl"),
                "score_records": records(4, "scores.jsonl"),
                "final_corpus": utils.load_jsonl(run_directory / "final" / "sdf_corpus.jsonl"),
                "critique_records": records(3, "critiques.jsonl"),
                "rewrite_records": records(3, "rewrites.jsonl"),
                "description_rewrite_records": records(1, "description_rewrites.jsonl"),
                "consideration_score_records": records(1, "consideration_scores.jsonl"),
                "consideration_selection_rows": records(1, "consideration_selections.jsonl"),
                "screen_rounds": records(1, "screen_rounds.jsonl"),
                "screen_batches": records(1, "screen_batches.jsonl"),
                "screen_judgments": records(1, "screen_judgments.jsonl"),
                "screen_rejections": records(1, "screen_rejections.jsonl"),
            })
        outcome = support.run_scenario_in_isolation(
            scenarios.scenario_named(scenarios.MAIN_SCENARIO_NAME), Path(base_directory) / "run",
            after_each_invocation=capture)
        yield collected, outcome.recorder.sorted_records()


def _ids(records, field):
    return {record[field] for record in records}


def test_the_loader_states_hold_the_drafts_the_draft_file_holds_after_every_invocation(main_scenario):
    collected, _ = main_scenario
    assert len(collected) == 2
    for invocation in collected:
        drafted_by_loader = {}
        for state in invocation["states"]:
            if state.text_versions:
                draft = state.text_versions[0]
                drafted_by_loader[state.document_id] = (
                    draft.text, draft.written_from.description, draft.written_from.variables,
                    draft.written_from.considerations_spec)
        drafted_by_file = {
            record["doc_id"]: (record["content"], record["description"], record["variables"], record["considerations_spec"])
            for record in invocation["draft_records"]}
        assert drafted_by_file and drafted_by_loader == drafted_by_file


def test_the_loader_states_hold_the_revisions_and_critiques_the_stage_files_hold_after_every_invocation(main_scenario):
    collected, _ = main_scenario
    for invocation in collected:
        revisions_by_loader = {
            state.document_id: (version.text, version.critique, version.written_from.description,
                                version.written_from.variables, version.written_from.considerations_spec)
            for state in invocation["states"] for version in state.text_versions if version.kind.name == "REVISION"}
        revisions_by_file = {
            record["doc_id"]: (record["content"], record["review"], record["description"], record["variables"],
                               record["considerations_spec"])
            for record in invocation["rewrite_records"]}
        assert revisions_by_file and revisions_by_loader == revisions_by_file
        assert _ids(invocation["rewrite_records"], "doc_id") <= _ids(invocation["critique_records"], "doc_id")


def test_the_final_corpus_holds_only_scored_documents_and_the_fixture_has_a_gate_removal_and_a_near_duplicate(main_scenario):
    collected, _ = main_scenario
    for invocation in collected:
        scored = _ids(invocation["score_records"], "doc_id")
        final = _ids(invocation["final_corpus"], "doc_id")
        assert final <= scored
        assert scored - final, "the fixture has a below-threshold score and a near-duplicate"
        assert final <= _ids(invocation["rewrite_records"], "doc_id")


def test_the_description_rewrite_records_name_a_reason_for_every_dealt_value_conflict(main_scenario):
    collected, _ = main_scenario
    for invocation in collected:
        records = invocation["description_rewrite_records"]
        assert records, "the fixture runs the description rewrite"
        conflicts = [record for record in records if record["dealt_value_conflict"]]
        assert conflicts and all(record["conflict_reason"] for record in conflicts)


def test_the_plan_file_holds_an_incoherent_plan_and_every_planned_document_is_in_it_after_the_resume(main_scenario):
    collected, _ = main_scenario
    first, second = (invocation["plan_records"] for invocation in collected)
    assert any(record["incoherent"] for record in second)
    assert scenarios.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME not in _ids(first, "prompt_id")
    assert scenarios.HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME in _ids(second, "prompt_id")
    assert _ids(first, "prompt_id") <= _ids(second, "prompt_id")


def test_the_resume_finishes_the_draft_that_was_cut_off_and_keeps_the_first_invocations_drafts(main_scenario):
    collected, requests = main_scenario
    first, second = (invocation["draft_records"] for invocation in collected)
    hand = scenarios.HAND_TRUNCATED_DRAFT
    assert hand not in _ids(first, "doc_id") and hand in _ids(second, "doc_id")
    assert _ids(first, "doc_id") <= _ids(second, "doc_id")
    assert [r["invocation"] for r in requests if r["stage"] == "layer2_draft" and r["item_id"] == hand] == [1, 2]


def test_the_resume_finishes_the_revise_that_failed_without_sending_the_accepted_critique_again(main_scenario):
    collected, requests = main_scenario
    first, second = collected
    hand = scenarios.HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME
    assert hand in _ids(first["critique_records"], "doc_id") and hand not in _ids(first["rewrite_records"], "doc_id")
    assert hand in _ids(second["rewrite_records"], "doc_id")
    assert [r["invocation"] for r in requests if r["stage"] == "layer3_critique" and r["item_id"] == hand] == [1], (
        "an accepted critique is kept, not paid for again")
    assert [r["invocation"] for r in requests if r["stage"] == "layer3_revise" and r["item_id"] == hand] == [1, 2]


def test_the_consideration_scores_and_selections_after_every_invocation(main_scenario):
    collected, _ = main_scenario
    for invocation in collected:
        scored = _ids(invocation["consideration_score_records"], "prompt_id")
        assert scored
        assert _ids(invocation["consideration_selection_rows"], "prompt_id") >= scored


def test_the_screen_rejections_are_the_judgments_marked_for_removal_with_their_reasons(main_scenario):
    collected, _ = main_scenario
    for invocation in collected:
        removed = {j["prompt_id"]: j for j in invocation["screen_judgments"] if j["remove"]}
        rejected = {r["prompt_id"]: r for r in invocation["screen_rejections"]}
        assert removed and set(rejected) == set(removed)
        assert all(rejected[pid]["reason"] == removed[pid]["reason"] for pid in removed)


def test_the_saved_screen_batches_of_each_round_hold_exactly_the_hands_of_that_round(main_scenario):
    collected, _ = main_scenario
    for invocation in collected:
        for round_record in invocation["screen_rounds"]:
            batches = [b for b in invocation["screen_batches"] if b["round"] == round_record["round"]]
            assert sorted(pid for b in batches for pid in b["prompt_ids"]) == sorted(round_record["prompt_ids"])
