"""The folded draft events match what the loader builds from the stage files on the fixture run.

The fixture's main scenario runs the whole pipeline twice (a resume retries the failures); after each
invocation the event log's fold is compared with the document states the loader builds from the stage
files. Also: a log with a cut-off last line, and a corrupt line in the middle.
"""

import tempfile
from pathlib import Path

import pytest

import sdf_prompt_equivalence_fixture_support as support
import sdf_prompt_equivalence_scenarios as scenarios
from sdf_pipeline.document_event_log import (
    DocumentEventLogCorruptError, fold_document_events, read_document_events)
from sdf_pipeline.document_state_loader import load_document_states
from shared import utils


@pytest.fixture(scope="module")
def main_scenario_after_each_invocation():
    collected = []
    with tempfile.TemporaryDirectory() as base_directory:
        def capture(number, run_directory):
            events_path = run_directory / "document_events.jsonl"
            collected.append({
                "events_text": events_path.read_text(encoding="utf-8"),
                "states": load_document_states(run_directory),
                "draft_records": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 2) / "drafts.jsonl"),
                "score_records": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 4) / "scores.jsonl"),
                "final_corpus": utils.load_jsonl(run_directory / "final" / "sdf_corpus.jsonl"),
            })
        support.run_scenario_in_isolation(
            scenarios.scenario_named(scenarios.MAIN_SCENARIO_NAME), Path(base_directory) / "run",
            after_each_invocation=capture)
        yield collected


def _events_of(text, tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(text, encoding="utf-8")
    return read_document_events(path)


def test_folded_draft_events_match_the_loader_states_after_every_invocation(main_scenario_after_each_invocation, tmp_path):
    assert len(main_scenario_after_each_invocation) == 2
    for invocation in main_scenario_after_each_invocation:
        folded = fold_document_events(_events_of(invocation["events_text"], tmp_path))
        drafted_by_loader = {}
        for state in invocation["states"]:
            if state.text_versions:
                draft = state.text_versions[0]
                drafted_by_loader[state.document_id] = (
                    draft.text, draft.written_from.description, draft.written_from.variables,
                    draft.written_from.considerations_spec)
        drafted_by_events = {
            document_id: (update["content"], update["description"], update["variables"], update["considerations_spec"])
            for document_id, document in folded.items()
            if (update := document.accepted_update_by_step.get("draft")) is not None}
        assert drafted_by_events == drafted_by_loader
        # and the event's update is the stage file's record, field for field and in field order
        records_by_id = {record["doc_id"]: record for record in invocation["draft_records"]}
        for document_id, document in folded.items():
            update = document.accepted_update_by_step.get("draft")
            if update is not None:
                assert list(update.items()) == list(records_by_id[document_id].items())
        assert set(records_by_id) == set(drafted_by_events)


def test_folded_score_events_match_the_scores_file_and_gate_and_cull_match_the_final_corpus(
        main_scenario_after_each_invocation, tmp_path):
    for invocation in main_scenario_after_each_invocation:
        folded = fold_document_events(_events_of(invocation["events_text"], tmp_path))
        scored_by_events = {
            document_id: update for document_id, document in folded.items()
            if (update := document.accepted_update_by_step.get("score")) is not None}
        scores_file_records = {record["doc_id"]: record for record in invocation["score_records"]}
        assert scored_by_events == scores_file_records
        for document_id, update in scored_by_events.items():  # field for field and in field order
            assert list(update.items()) == list(scores_file_records[document_id].items())
        survivors_by_events = {
            document_id for document_id in scored_by_events
            if "gate" in folded[document_id].accepted_update_by_step
            and "near_duplicate_cull" in folded[document_id].accepted_update_by_step}
        assert survivors_by_events == {record["doc_id"] for record in invocation["final_corpus"]}
        removed = {document_id for document_id, d in folded.items()
                   if "gate" in d.removal_by_step or "near_duplicate_cull" in d.removal_by_step}
        assert removed == set(scored_by_events) - survivors_by_events
        assert removed, "the fixture has a below-threshold score and a near-duplicate"


def test_the_resume_resolves_the_draft_failure_the_first_invocation_recorded(main_scenario_after_each_invocation, tmp_path):
    first, second = (_events_of(i["events_text"], tmp_path) for i in main_scenario_after_each_invocation)
    failed_after_first = {document_id for document_id, d in fold_document_events(first).items()
                          if "draft" in d.unresolved_failure_by_step}
    assert failed_after_first, "the fixture has a truncated draft in its first invocation"
    folded_second = fold_document_events(second)
    for document_id in failed_after_first:
        assert "draft" not in folded_second[document_id].unresolved_failure_by_step
        assert "draft" in folded_second[document_id].accepted_update_by_step
    assert first == second[:len(first)], "the log is append-only across the resume"


def test_a_cut_off_last_line_is_ignored_and_the_fold_is_unchanged(main_scenario_after_each_invocation, tmp_path):
    text = main_scenario_after_each_invocation[-1]["events_text"]
    whole = fold_document_events(_events_of(text, tmp_path))
    cut = text + '{"event": "document_step_outcome", "document_id": "matrix_9999", "step": "dr'
    events = _events_of(cut, tmp_path)
    assert events == _events_of(text, tmp_path)
    assert fold_document_events(events) == whole


def test_a_corrupt_line_in_the_middle_is_an_error(tmp_path):
    with pytest.raises(DocumentEventLogCorruptError):
        _events_of('{"event": "x"}\nnot json\n{"event": "y"}\n', tmp_path)
