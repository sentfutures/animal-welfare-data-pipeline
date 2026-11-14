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
    DocumentEventLogCorruptError, fold_document_events, fold_round_and_batch_plan_events, read_document_events)
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
                "plan_records": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 1) / "plans.jsonl"),
                "draft_records": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 2) / "drafts.jsonl"),
                "score_records": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 4) / "scores.jsonl"),
                "final_corpus": utils.load_jsonl(run_directory / "final" / "sdf_corpus.jsonl"),
                "critique_records": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 3) / "critiques.jsonl"),
                "rewrite_records": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 3) / "rewrites.jsonl"),
                "description_rewrite_records": utils.load_jsonl(
                    utils.sdf_stage_dir(run_directory, 1) / "description_rewrites.jsonl"),
                "consideration_score_records": utils.load_jsonl(
                    utils.sdf_stage_dir(run_directory, 1) / "consideration_scores.jsonl"),
                "consideration_selection_rows": utils.load_jsonl(
                    utils.sdf_stage_dir(run_directory, 1) / "consideration_selections.jsonl"),
                "screen_rounds": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 1) / "screen_rounds.jsonl"),
                "screen_batches": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 1) / "screen_batches.jsonl"),
                "screen_judgments": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 1) / "screen_judgments.jsonl"),
                "screen_rejections": utils.load_jsonl(utils.sdf_stage_dir(run_directory, 1) / "screen_rejections.jsonl"),
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


def test_folded_critique_and_revise_events_match_the_stage_files_and_the_loader_after_every_invocation(
        main_scenario_after_each_invocation, tmp_path):
    for invocation in main_scenario_after_each_invocation:
        folded = fold_document_events(_events_of(invocation["events_text"], tmp_path))
        for step_name, records in (("critique", invocation["critique_records"]), ("revise", invocation["rewrite_records"])):
            records_by_id = {record["doc_id"]: record for record in records}
            updates_by_id = {document_id: document.accepted_update_by_step[step_name]
                             for document_id, document in folded.items() if step_name in document.accepted_update_by_step}
            assert set(updates_by_id) == set(records_by_id)
            for document_id, update in updates_by_id.items():
                assert list(update.items()) == list(records_by_id[document_id].items())
        revisions_by_loader = {
            state.document_id: (version.text, version.critique, version.written_from.description,
                                version.written_from.variables, version.written_from.considerations_spec)
            for state in invocation["states"] for version in state.text_versions
            if version.kind.name == "REVISION"}
        revisions_by_events = {
            document_id: (u["content"], u["review"], u["description"], u["variables"], u["considerations_spec"])
            for document_id, document in folded.items()
            if (u := document.accepted_update_by_step.get("revise")) is not None}
        assert revisions_by_events == revisions_by_loader


def test_folded_description_rewrite_events_match_the_stage_file_after_every_invocation(
        main_scenario_after_each_invocation, tmp_path):
    for invocation in main_scenario_after_each_invocation:
        folded = fold_document_events(_events_of(invocation["events_text"], tmp_path))
        records_by_id = {record["prompt_id"]: record for record in invocation["description_rewrite_records"]}
        assert records_by_id, "the fixture runs the description rewrite"
        accepted_by_id = {document_id: document.accepted_update_by_step["description_rewrite"]
                          for document_id, document in folded.items() if "description_rewrite" in document.accepted_update_by_step}
        removed_by_id = {document_id: document.removal_by_step["description_rewrite"]
                         for document_id, document in folded.items() if "description_rewrite" in document.removal_by_step}
        assert set(accepted_by_id) | set(removed_by_id) == set(records_by_id)
        assert not set(accepted_by_id) & set(removed_by_id)
        for document_id, update in accepted_by_id.items():
            assert list(update.items()) == list(records_by_id[document_id].items())
        for document_id, removal in removed_by_id.items():
            assert removal["reason_code"] == "dealt_value_conflict"
            assert records_by_id[document_id]["dealt_value_conflict"] is True
            assert removal["detail"]["conflict_reason"] == records_by_id[document_id]["conflict_reason"]


def test_the_resume_resolves_the_revise_failure_the_first_invocation_recorded_without_a_second_critique(
        main_scenario_after_each_invocation, tmp_path):
    first, second = (_events_of(i["events_text"], tmp_path) for i in main_scenario_after_each_invocation)
    failed_revise_after_first = {document_id for document_id, d in fold_document_events(first).items()
                                 if "revise" in d.unresolved_failure_by_step}
    assert failed_revise_after_first, "the fixture has a failed revise in its first invocation"
    folded_second = fold_document_events(second)
    for document_id in failed_revise_after_first:
        assert "revise" not in folded_second[document_id].unresolved_failure_by_step
        assert "revise" in folded_second[document_id].accepted_update_by_step
        critique_events = [e for e in second if e.get("document_id") == document_id and e["step"] == "critique"
                           and e["outcome"] == "accepted"]
        assert len(critique_events) == 1, "an accepted critique is kept, not paid for again"


def test_folded_plan_events_match_the_plans_file_after_every_invocation(main_scenario_after_each_invocation, tmp_path):
    for invocation in main_scenario_after_each_invocation:
        folded = fold_document_events(_events_of(invocation["events_text"], tmp_path))
        plans_file_records = {record["prompt_id"]: record for record in invocation["plan_records"]}
        accepted_by_events = {
            document_id: update for document_id, document in folded.items()
            if (update := document.accepted_update_by_step.get("plan")) is not None}
        incoherent_by_events = {
            document_id for document_id, document in folded.items() if "plan" in document.removal_by_step}
        assert accepted_by_events == {
            prompt_id: record for prompt_id, record in plans_file_records.items() if not record["incoherent"]}
        assert incoherent_by_events == {
            prompt_id for prompt_id, record in plans_file_records.items() if record["incoherent"]}
        for document_id, update in accepted_by_events.items():  # field for field and in field order
            assert list(update.items()) == list(plans_file_records[document_id].items())
        assert set(accepted_by_events) | incoherent_by_events == set(plans_file_records)


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


def test_folded_consideration_events_match_the_scores_and_selections_files_after_every_invocation(
        main_scenario_after_each_invocation, tmp_path):
    saw_scores = False
    for invocation in main_scenario_after_each_invocation:
        folded = fold_document_events(_events_of(invocation["events_text"], tmp_path))
        scores_by_events = {
            document_id: update for document_id, document in folded.items()
            if (update := document.accepted_update_by_step.get("consideration_scoring")) is not None}
        records_by_id = {record["prompt_id"]: record for record in invocation["consideration_score_records"]}
        assert scores_by_events == records_by_id
        saw_scores = saw_scores or bool(records_by_id)
        for document_id, update in scores_by_events.items():
            assert list(update.items()) == list(records_by_id[document_id].items())
        outcomes_by_events = {
            document_id: update["outcomes"] for document_id, document in folded.items()
            if (update := document.accepted_update_by_step.get("consideration_assignment")) is not None}
        rows_by_id = {row["prompt_id"]: row["outcomes"] for row in invocation["consideration_selection_rows"]}
        assert outcomes_by_events == rows_by_id
    assert saw_scores


def test_folded_deal_round_batch_plan_and_screen_events_match_the_screen_files_after_every_invocation(
        main_scenario_after_each_invocation, tmp_path):
    for invocation in main_scenario_after_each_invocation:
        events = _events_of(invocation["events_text"], tmp_path)
        plan = fold_round_and_batch_plan_events(events)
        assert plan["rounds"] == {
            r["round"]: {"prompt_ids": r["prompt_ids"], "note": r["note"]} for r in invocation["screen_rounds"]}
        batches = invocation["screen_batches"]
        assert plan["batches_by_round"] == {
            number: [b for b in batches if b["round"] == number] for number in sorted({b["round"] for b in batches})}
        assert plan["corpus_steps_finished"][0]["step"] == "deal"
        folded = fold_document_events(events)
        removed_by_events = {pid for pid, d in folded.items() if d.latest_outcome_by_step.get("screen", ("",))[0] == "removed"}
        assert removed_by_events == {r["prompt_id"] for r in invocation["screen_rejections"]}
        judged_by_events = {pid for pid, d in folded.items() if "screen" in d.latest_outcome_by_step}
        assert judged_by_events == {j["prompt_id"] for j in invocation["screen_judgments"]}
