"""Steps only build requests and read replies; the runner writes every file, owns the progress
counts and counts the attempts. These tests pin that split and the crash window of the one step that
writes two records for one outcome (the description rewrite's dealt-value conflict)."""

import pytest

from sdf_pipeline import layer1_description_rewrite as rewrite
from sdf_pipeline.shared_step_machinery.step_record import LAYER1, Run, StageFile, Step, one_record_per_accepted_document
from sdf_pipeline.shared_step_machinery.step_runner import run_step
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, StepPolicy, StepProgress)
from shared import api, utils
from sdf_description_rewrite_test_builders import answer, make_plans
from sdf_step_and_command_line_test_helpers import outcome_of, records_of, run_of

RECORDING_RECORDS = StageFile("recording_folder", "recording_records.jsonl", "doc_id")


# ---------------------------------------------------------------- the crash window of a dealt-value conflict

def run_with_a_conflict_and_a_crash_before_the_rejection_is_written(tiny_config, prompts_sdf, stage_dir, monkeypatch):
    """Four plans; the second is a dealt-value conflict. Returns (plans, the conflicting prompt id) after
    the run died in the rejections file's append."""
    tiny_config["workers"] = 1
    tiny_config["sdf"]["description_rewrite_model"] = "claude-sonnet-5"
    plans = make_plans(prompts_sdf, count=4)
    doomed = plans[1]["prompt_id"]
    monkeypatch.setattr(api, "call_claude", lambda user, **kw: (
        (answer(conflict=True, conflict_reason="Scale cannot occur.") if kw["item_id"] == doomed else answer()),
        "end_turn"))
    real_append = utils.append_jsonl

    def append_that_dies_on_the_rejections_file(record, path):
        if path.name == rewrite.REJECTIONS_FILENAME:
            raise OSError("killed between the two writes")
        real_append(record, path)

    monkeypatch.setattr(utils, "append_jsonl", append_that_dies_on_the_rejections_file)
    with pytest.raises(OSError, match="killed between the two writes"):
        rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    monkeypatch.setattr(utils, "append_jsonl", real_append)
    return plans, doomed


def test_a_crash_between_the_results_and_rejection_writes_leaves_the_result_first_as_on_origin_main(
        tiny_config, prompts_sdf, stage_dir, monkeypatch):
    plans, doomed = run_with_a_conflict_and_a_crash_before_the_rejection_is_written(
        tiny_config, prompts_sdf, stage_dir, monkeypatch)
    stored = utils.load_jsonl(stage_dir / rewrite.RESULTS_FILENAME)
    assert [r["prompt_id"] for r in stored] == [plans[0]["prompt_id"], doomed]   # the result was written first
    assert stored[1]["dealt_value_conflict"] is True
    assert not (stage_dir / rewrite.REJECTIONS_FILENAME).exists()                # the rejection was not


def test_after_that_crash_a_resume_neither_repeats_nor_adds_the_rejection(
        tiny_config, prompts_sdf, stage_dir, monkeypatch):
    plans, doomed = run_with_a_conflict_and_a_crash_before_the_rejection_is_written(
        tiny_config, prompts_sdf, stage_dir, monkeypatch)
    item_ids_called = []
    monkeypatch.setattr(api, "call_claude", lambda user, **kw: (item_ids_called.append(kw["item_id"]) or answer(), "end_turn"))
    kept = rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert item_ids_called == [plans[2]["prompt_id"], plans[3]["prompt_id"]]   # the conflict is not asked again
    assert [r["prompt_id"] for r in utils.load_jsonl(stage_dir / rewrite.RESULTS_FILENAME)] == [
        p["prompt_id"] for p in plans]
    assert not (stage_dir / rewrite.REJECTIONS_FILENAME).exists()              # recorded zero times, 
    assert [p["prompt_id"] for p in kept] == [p["prompt_id"] for p in plans if p["prompt_id"] != doomed]


def test_the_description_rewrite_step_writes_no_file_for_any_outcome(tiny_config, prompts_sdf, tmp_path):
    tiny_config["sdf"]["description_rewrite_model"] = "claude-sonnet-5"
    plans = make_plans(prompts_sdf, count=2)
    run = run_of(tiny_config, prompts_sdf, LAYER1, tmp_path)
    progress = StepProgress(pending_count=2)
    for text in (answer(), answer(conflict=True, conflict_reason="No.")):
        outcome = outcome_of(rewrite.STEP, plans[0], ModelReply(text, "end_turn"), run)
        records_of(rewrite.STEP, plans[0], outcome, run)
        progress.record([outcome])
        rewrite.STEP.progress_line([plans[0]], {plans[0]["prompt_id"]: outcome}, progress, run)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- the runner owns attempts and progress

def recording_step(progress_seen, attempts_seen, policy=StepPolicy(maximum_attempts_in_one_run=3), **changes):
    def read_reply(documents, reply, run):
        (document,) = documents
        attempts_seen.append(reply.attempt)
        if reply.attempt < 3:
            return {document["doc_id"]: Failed(FailureKind.UNUSABLE_ANSWER, "try again")}
        return {document["doc_id"]: Accepted(reply.text)}

    def progress_line(documents, outcomes, progress, run):
        progress_seen.append((progress.groups_seen, progress.answered, progress.failed, progress.pending_count))

    fields = dict(
        name="recording", folder_key="recording_folder", stage_tag="recording_stage",
        build_request=lambda documents, run: ModelRequest("ask", "sys", "m", "recording_stage", documents[0]["doc_id"]),
        read_reply=read_reply,
        records=one_record_per_accepted_document(
            RECORDING_RECORDS,
            lambda document, update, run: {"doc_id": document["doc_id"], "content": update, "attempts": attempts_seen[-1]}),
        writes=(RECORDING_RECORDS,), policy=policy,
        progress_line=progress_line)
    return Step(**{**fields, **changes})


def recording_run(tmp_path):
    return Run({"workers": 1}, tmp_path, {"recording_folder": tmp_path})


def test_the_runner_numbers_the_attempts_on_the_reply(tmp_path):
    attempts_seen = []
    run_step(recording_step([], attempts_seen), recording_run(tmp_path), [{"doc_id": "a"}],
             send=lambda request: ModelReply("ok", "end_turn"))
    assert attempts_seen == [1, 2, 3]
    assert utils.load_jsonl(tmp_path / "recording_records.jsonl")[0]["attempts"] == 3


def test_the_runner_counts_progress_and_starts_it_again_for_each_run(tmp_path):
    progress_seen = []
    step = recording_step(progress_seen, [])
    send = lambda request: ModelReply("ok", "end_turn")  # noqa: E731
    run_step(step, recording_run(tmp_path), [{"doc_id": "a"}, {"doc_id": "b"}], send=send)
    assert progress_seen == [(1, 1, 0, 2), (2, 2, 0, 2)]
    progress_seen.clear()
    result = run_step(step, recording_run(tmp_path), [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}], send=send)
    assert progress_seen == [(1, 1, 0, 1)]
    assert result.progress.groups_seen == 1 and result.progress.pending_count == 1


def test_a_step_policy_that_allows_no_attempt_is_refused(tmp_path):
    with pytest.raises(ValueError, match="must be at least 1"):
        run_step(recording_step([], [], policy=StepPolicy(maximum_attempts_in_one_run=0)), recording_run(tmp_path),
                 [{"doc_id": "a"}], send=lambda request: ModelReply("ok", "end_turn"))


def test_the_runner_writes_every_record_of_an_outcome_in_order_with_no_checkpoint_file(tmp_path):
    def two_records(documents, outcomes, run):
        (document,) = documents
        return OutcomeRecords((
            ("first_file.jsonl", {"doc_id": document["doc_id"], "n": 1}),
            ("second_file.jsonl", {"doc_id": document["doc_id"], "n": 2}),
        ))

    step = recording_step([], [], policy=StepPolicy(), records=two_records,
                          read_reply=lambda documents, reply, run: {documents[0]["doc_id"]: Accepted(reply.text)})
    run_step(step, recording_run(tmp_path), [{"doc_id": "a"}], send=lambda request: ModelReply("ok", "end_turn"))
    assert utils.load_jsonl(tmp_path / "first_file.jsonl") == [{"doc_id": "a", "n": 1}]
    assert utils.load_jsonl(tmp_path / "second_file.jsonl") == [{"doc_id": "a", "n": 2}]
    assert not (tmp_path / "_checkpoint.json").exists()


def test_the_runner_drops_the_earlier_records_an_outcome_replaces_before_appending_its_own(tmp_path):
    utils.save_jsonl([{"batch_id": "x", "n": 1}, {"batch_id": "y", "n": 2}], tmp_path / "judgments.jsonl")

    def replacing(documents, outcomes, run):
        return OutcomeRecords((("judgments.jsonl", {"batch_id": "x", "n": 3}),), replacing=(("judgments.jsonl", "batch_id", "x"),))

    step = recording_step([], [], policy=StepPolicy(), records=replacing,
                          read_reply=lambda documents, reply, run: {documents[0]["doc_id"]: Accepted(reply.text)})
    run_step(step, recording_run(tmp_path), [{"doc_id": "a"}], send=lambda request: ModelReply("ok", "end_turn"))
    assert utils.load_jsonl(tmp_path / "judgments.jsonl") == [{"batch_id": "y", "n": 2}, {"batch_id": "x", "n": 3}]
