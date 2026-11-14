"""Steps only build requests and interpret replies; the runner writes every file, owns the progress
counts and counts the attempts. These tests pin that split and the crash window of the one step that
writes two records for one outcome (the description rewrite's dealt-value conflict)."""

import pytest

from sdf_pipeline.layer2_plan_and_description_rewrite import description_rewrite as rewrite
from sdf_pipeline.shared_step_machinery.step_runner import StepRunner
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, DocumentStep, Failed, FailureKind, ModelReply, OutcomeRecords, StepPolicy, StepProgress)
from shared import api, utils
from sdf_description_rewrite_test_builders import answer, make_plans


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
    checkpoint = utils.Checkpoint(stage_dir / rewrite.CHECKPOINT_FILENAME)
    assert checkpoint.is_done(plans[0]["prompt_id"]) and not checkpoint.is_done(doomed)  # marked done after both writes


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
    assert not (stage_dir / rewrite.REJECTIONS_FILENAME).exists()              # recorded zero times, as on origin/main
    assert [p["prompt_id"] for p in kept] == [p["prompt_id"] for p in plans if p["prompt_id"] != doomed]


def test_the_description_rewrite_step_writes_no_file_for_any_outcome(tiny_config, prompts_sdf, tmp_path):
    tiny_config["sdf"]["description_rewrite_model"] = "claude-sonnet-5"
    plans = make_plans(prompts_sdf, count=2)
    step = rewrite.DescriptionRewriteStep(
        tiny_config, rewrite.load_rewrite_prompts(prompts_sdf), prompts_sdf, tmp_path, plans)
    progress = StepProgress(pending_count=2)
    for text in (answer(), answer(conflict=True, conflict_reason="No.")):
        outcome = step.interpret(plans[0], ModelReply(text, "end_turn"))
        step.outcome_records(plans[0], outcome)
        progress.record(outcome)
        step.progress_line(plans[0], outcome, progress)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- the runner owns attempts and progress

class RecordingStep(DocumentStep):
    step_name = "recording"
    stage_file_name = "recording_records.jsonl"
    stage_tag = "recording_stage"
    policy = StepPolicy(maximum_attempts_in_one_run=3)

    def __init__(self):
        self.attempts_seen = []
        self.progress_seen = []

    def build_request(self, document):
        return self.model_request(document, "ask", "sys", "m")

    def interpret(self, document, reply):
        self.attempts_seen.append(reply.attempt)
        if reply.attempt < 3:
            return Failed(FailureKind.UNUSABLE_ANSWER, "try again")
        return Accepted(reply.text)

    def stage_file_record(self, document, update):
        return {"doc_id": document["doc_id"], "content": update, "attempts": self.attempts_seen[-1]}

    def progress_line(self, document, outcome, progress):
        self.progress_seen.append((progress.documents_seen, progress.answered, progress.failed, progress.pending_count))
        return None


def test_the_runner_numbers_the_attempts_on_the_reply(tmp_path):
    step = RecordingStep()
    runner = StepRunner(step, tmp_path, send=lambda request: ModelReply("ok", "end_turn"))
    runner.run([{"doc_id": "a"}])
    assert step.attempts_seen == [1, 2, 3]
    assert utils.load_jsonl(tmp_path / "recording_records.jsonl")[0]["attempts"] == 3


def test_the_runner_counts_progress_and_starts_it_again_for_each_run(tmp_path):
    step = RecordingStep()
    runner = StepRunner(step, tmp_path, send=lambda request: ModelReply("ok", "end_turn"))
    runner.run([{"doc_id": "a"}, {"doc_id": "b"}])
    assert step.progress_seen == [(1, 1, 0, 2), (2, 2, 0, 2)]
    step.progress_seen.clear()
    result = StepRunner(step, tmp_path, send=lambda request: ModelReply("ok", "end_turn")).run(
        [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}])
    assert step.progress_seen == [(1, 1, 0, 1)]
    assert result.progress.documents_seen == 1 and result.progress.pending_count == 1


def test_a_step_policy_that_allows_no_attempt_is_refused():
    with pytest.raises(ValueError, match="must be at least 1"):
        StepPolicy(maximum_attempts_in_one_run=0)


def test_the_runner_writes_every_record_of_an_outcome_in_order_then_marks_the_document_done(tmp_path):
    class TwoRecordsStep(RecordingStep):
        policy = StepPolicy()

        def interpret(self, document, reply):
            return Accepted(reply.text)

        def outcome_records(self, document, outcome):
            return OutcomeRecords((
                ("first_file.jsonl", {"doc_id": document["doc_id"], "n": 1}),
                ("second_file.jsonl", {"doc_id": document["doc_id"], "n": 2}),
            ), finishes_the_document=True)

    runner = StepRunner(TwoRecordsStep(), tmp_path, send=lambda request: ModelReply("ok", "end_turn"))
    runner.run([{"doc_id": "a"}])
    assert utils.load_jsonl(tmp_path / "first_file.jsonl") == [{"doc_id": "a", "n": 1}]
    assert utils.load_jsonl(tmp_path / "second_file.jsonl") == [{"doc_id": "a", "n": 2}]
    assert utils.Checkpoint(tmp_path / "_checkpoint.json").is_done("a")
