"""Consideration scoring as a step (attempts per run, thrown calls, the no-welfare-stake
arm, resume) and the assignment as a plain function (solved again over every document)."""

import pytest

from sdf_pipeline.layer3_consideration_scoring_and_selection import run_layer as select
from sdf_pipeline.shared_step_machinery.considerations_catalog import load_considerations
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Failed, FailureKind, ModelReply, StepProgress
from sdf_step_and_command_line_test_helpers import outcome_of, records_of, run_of
from shared import utils
from shared.sdf_run_layouts import CONSIDERATION_SCORING_AND_SELECTION
from sdf_consideration_selection_test_builders import (
    answer_with, make_plans, no_stake_resolution, rows_of, score_everything_five, small_prompts)  # noqa: F401


@pytest.fixture
def scoring_run(tiny_config, small_prompts, stage_dir):
    return run_of(tiny_config, small_prompts, CONSIDERATION_SCORING_AND_SELECTION, stage_dir)


PLAN = make_plans(1)[0]


def reply_for(run, plan, score=5):
    request = select.STEP.build_request([plan], run)
    return request, ModelReply(answer_with(request.user_message, lambda label: score), "end_turn")


# --- reading the reply ---

def test_read_a_usable_answer_is_accepted_with_its_attempt_count_and_todays_record(scoring_run):
    _, reply = reply_for(scoring_run, PLAN)
    outcome = outcome_of(select.STEP, PLAN, reply, scoring_run)
    assert isinstance(outcome, Accepted) and outcome.update["attempts"] == 1
    records = records_of(select.STEP, PLAN, outcome, scoring_run)
    ((file_name, record),) = records.records
    assert file_name == select.SCORES_FILENAME
    assert list(record) == ["prompt_id", "scores", "attempts"] and record["prompt_id"] == PLAN["prompt_id"]


def test_read_an_unusable_answer_is_a_failure_that_names_the_attempts_only_on_the_last(scoring_run):
    # the attempt number comes on the reply (the runner sets it), not from any state of the step
    for attempt in (1, 2, 3):
        outcome = outcome_of(select.STEP, PLAN, ModelReply("not json", "end_turn", attempt=attempt), scoring_run)
        assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER
        assert outcome.reason.endswith("(after 3 attempts)") == (attempt == 3)
    # a usable answer on the second attempt records 2; the step keeps no count between calls
    _, reply = reply_for(scoring_run, PLAN)
    assert outcome_of(select.STEP, PLAN, ModelReply(reply.text, "end_turn", attempt=2), scoring_run).update["attempts"] == 2
    assert outcome_of(select.STEP, PLAN, reply, scoring_run).update["attempts"] == 1


def test_reading_keeps_no_state_between_documents_or_calls(scoring_run):
    other_plan = make_plans(2)[1]
    request, reply = reply_for(scoring_run, PLAN)
    first = outcome_of(select.STEP, PLAN, reply, scoring_run)
    outcome_of(select.STEP, other_plan, ModelReply("not json", "end_turn", attempt=2), scoring_run)
    assert outcome_of(select.STEP, PLAN, reply, scoring_run) == first
    assert select.offered_entries(PLAN, scoring_run) == select.offered_entries(PLAN, scoring_run)


def test_read_a_truncated_answer_is_unusable(scoring_run):
    _, reply = reply_for(scoring_run, PLAN)
    outcome = outcome_of(select.STEP, PLAN, ModelReply(reply.text, "max_tokens"), scoring_run)
    assert outcome.kind is FailureKind.UNUSABLE_ANSWER and "truncated answer" in outcome.reason


def test_the_failure_record_and_line_for_unusable_and_thrown(scoring_run):
    unusable = Failed(FailureKind.UNUSABLE_ANSWER, "no usable JSON object (x) (after 3 attempts)")
    thrown = Failed(FailureKind.THROWN_CALL, "error: RuntimeError: boom")
    unusable_records = records_of(select.STEP, PLAN, unusable, scoring_run)
    assert unusable_records.records == ((select.FAILURES_FILENAME, {
        "prompt_id": PLAN["prompt_id"], "problem": "unusable answer (no usable JSON object (x) (after 3 attempts))"}),)
    ((_, thrown_record),) = records_of(select.STEP, PLAN, thrown, scoring_run).records
    assert thrown_record["problem"] == "API call failed (RuntimeError: boom)"
    assert select.STEP.progress_line([PLAN], {PLAN["prompt_id"]: thrown}, StepProgress(pending_count=1), scoring_run) == (
        f"  {PLAN['prompt_id']}: API call failed (RuntimeError: boom) — will be rescored on resume")


def test_the_no_welfare_stake_arm_and_undescribed_plans_are_skipped(tiny_config, small_prompts, stage_dir):
    plans = make_plans(4, no_stake={1}, incoherent={2}, resolution=no_stake_resolution(small_prompts))
    skipping = run_of(tiny_config, small_prompts, CONSIDERATION_SCORING_AND_SELECTION, stage_dir)
    assert [select.STEP.applies(p, skipping) for p in plans] == [True, False, False, True]
    tiny_config["sdf"]["select_skip_no_stake"] = False
    keeping = run_of(tiny_config, small_prompts, CONSIDERATION_SCORING_AND_SELECTION, stage_dir)
    assert [select.STEP.applies(p, keeping) for p in plans] == [True, True, False, True]


# --- attempts counted per run ---

def test_unusable_on_every_attempt_makes_exactly_the_configured_attempts_then_records_the_failure(
        tiny_config, small_prompts, stage_dir, stub_claude, capsys):
    tiny_config["sdf"]["max_scoring_attempts"] = 2
    plans = make_plans(3)

    def dispatcher(user_message, **kw):
        return "not json" if kw["item_id"] == "matrix_000001" else score_everything_five(user_message)
    calls = stub_claude(dispatcher)
    with pytest.raises(SystemExit, match="1 of 3 scoring calls failed — stopping before the assignment"):
        select.run(tiny_config, small_prompts, stage_dir, plans)
    assert [c["item_id"] for c in calls].count("matrix_000001") == 2
    problem = rows_of(stage_dir, select.FAILURES_FILENAME)[0]["problem"]
    assert problem.startswith("unusable answer (") and problem.endswith("(after 2 attempts))")
    out = capsys.readouterr().out
    assert f"matrix_000001: {problem} — will be rescored on resume" in out
    assert "  Scoring: 2 answered, 1 failed (" in out


def test_attempts_are_counted_per_run_so_a_resume_gets_a_fresh_set(tiny_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(2)

    def always_unusable(user_message, **kw):
        return "not json" if kw["item_id"] == "matrix_000000" else score_everything_five(user_message)
    stub_claude(always_unusable)
    for _ in range(2):
        with pytest.raises(SystemExit):
            select.run(tiny_config, small_prompts, stage_dir, plans)
    calls = stub_claude(always_unusable)
    with pytest.raises(SystemExit):
        select.run(tiny_config, small_prompts, stage_dir, plans)
    assert [c["item_id"] for c in calls] == ["matrix_000000"] * 3
    assert len(rows_of(stage_dir, select.FAILURES_FILENAME)) == 3


def test_every_document_unusable_on_every_attempt_is_systemic(tiny_config, small_prompts, stage_dir, stub_claude):
    calls = stub_claude(["not json"] * 6)
    with pytest.raises(SystemExit, match="every pending scoring call failed"):
        select.run(tiny_config, small_prompts, stage_dir, make_plans(2))
    assert len(calls) == 6


def test_no_attempts_allowed_is_refused_before_any_call(
        tiny_config, small_prompts, stage_dir, stub_claude):
    tiny_config["sdf"]["max_scoring_attempts"] = 0
    calls = stub_claude([])
    with pytest.raises(ValueError, match="must be at least 1"):
        select.run(tiny_config, small_prompts, stage_dir, make_plans(1))
    assert calls == []
    assert not (stage_dir / select.FAILURES_FILENAME).exists()


def test_the_code_default_attempts_apply_when_the_config_names_none(tiny_config, small_prompts, stage_dir, stub_claude):
    tiny_config["sdf"].pop("max_scoring_attempts", None)
    calls = stub_claude(["not json"] * 3 + [score_everything_five])
    with pytest.raises(SystemExit):
        select.run(tiny_config, small_prompts, stage_dir, make_plans(1))
    assert len(calls) == select.DEFAULT_MAX_SCORING_ATTEMPTS


# --- the no-welfare-stake arm in the run ---

# --- resume ---

def test_resume_makes_zero_calls_and_changes_no_stage_file(tiny_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(5)
    stub_claude(score_everything_five)
    select.run(tiny_config, small_prompts, stage_dir, plans)
    before = {p.name: p.read_bytes() for p in stage_dir.iterdir()}
    calls = stub_claude([])
    select.run(tiny_config, small_prompts, stage_dir, plans)
    assert calls == []
    assert {p.name: p.read_bytes() for p in stage_dir.iterdir()} == before


# --- the assignment as a plain function ---

def test_the_assignment_makes_no_call_and_returns_what_the_layer_writes(
        tiny_config, small_prompts, stage_dir, stub_claude):
    catalog = load_considerations(small_prompts)
    plans = make_plans(3)
    scoring_run = run_of(tiny_config, small_prompts, CONSIDERATION_SCORING_AND_SELECTION, stage_dir)
    score_rows = []
    for plan in plans:
        request, reply = reply_for(scoring_run, plan)
        score_rows.append({"prompt_id": plan["prompt_id"], "scores": outcome_of(select.STEP, plan, reply, scoring_run).update["scores"],
                           "attempts": 1})
    calls = stub_claude([])
    selection_rows, report = select.assign_considerations_to_documents(tiny_config, catalog, plans, score_rows)
    assert calls == []
    assert [row["prompt_id"] for row in selection_rows] == sorted(p["prompt_id"] for p in plans)
    assert report["eligible_documents"] == 3


# --- durability: rows are written as answers arrive ---

def test_a_finished_score_behind_a_slow_earlier_document_is_on_disk_before_the_slow_one_completes(
        tiny_config, small_prompts, stage_dir, stub_claude):
    import threading
    tiny_config["workers"] = 2
    plans = make_plans(2)
    slow_id, fast_id = "matrix_000000", "matrix_000001"
    fast_row_is_on_disk = threading.Event()
    seen_when_slow_finished = []

    def dispatcher(user_message, **kw):
        if kw["item_id"] == slow_id:
            assert fast_row_is_on_disk.wait(timeout=10)
            seen_when_slow_finished.extend(r["prompt_id"] for r in rows_of(stage_dir, select.SCORES_FILENAME))
            return score_everything_five(user_message)
        return score_everything_five(user_message)

    stub = stub_claude(dispatcher)  # noqa: F841
    real_append = utils.append_jsonl

    def noting_append(record, path):
        real_append(record, path)
        if record.get("prompt_id") == fast_id and str(path).endswith(select.SCORES_FILENAME):
            fast_row_is_on_disk.set()
    utils.append_jsonl, original = noting_append, utils.append_jsonl
    try:
        select.run(tiny_config, small_prompts, stage_dir, plans)
    finally:
        utils.append_jsonl = original
    assert seen_when_slow_finished == [fast_id]
    assert [r["prompt_id"] for r in rows_of(stage_dir, select.SCORES_FILENAME)] == [fast_id, slow_id]


def test_failure_records_and_lines_are_written_in_completion_order_as_answers_arrive(
        tiny_config, small_prompts, stage_dir, stub_claude, capsys):
    import threading
    tiny_config["workers"] = 2
    tiny_config["sdf"]["max_scoring_attempts"] = 1
    slow_id, fast_id = "matrix_000000", "matrix_000001"
    fast_failure_written = threading.Event()
    real_append = utils.append_jsonl

    def noting_append(record, path):
        real_append(record, path)
        if record.get("prompt_id") == fast_id and str(path).endswith(select.FAILURES_FILENAME):
            fast_failure_written.set()

    def dispatcher(user_message, **kw):
        if kw["item_id"] == slow_id:
            assert fast_failure_written.wait(timeout=10)
        return "not json"
    stub_claude(dispatcher)
    utils.append_jsonl, original = noting_append, utils.append_jsonl
    try:
        with pytest.raises(SystemExit):
            select.run(tiny_config, small_prompts, stage_dir, make_plans(2))
    finally:
        utils.append_jsonl = original
    assert [r["prompt_id"] for r in rows_of(stage_dir, select.FAILURES_FILENAME)] == [fast_id, slow_id]
    out = capsys.readouterr().out
    assert out.index(f"{fast_id}: unusable") < out.index(f"{slow_id}: unusable")
