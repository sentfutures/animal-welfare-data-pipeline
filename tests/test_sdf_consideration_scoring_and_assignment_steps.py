"""Consideration scoring as a DocumentStep (attempts per run, thrown calls, the no-welfare-stake
arm, resume) and the assignment as a CorpusStep (solved again over every document, events only on
change). ``layer1_select_considerations.run`` keeps its signature and behaviour."""

import json

import pytest

from sdf_pipeline import layer1_select_considerations as select
from sdf_pipeline.considerations_catalog import load_considerations
from sdf_pipeline.document_event_log import DocumentEventLog, fold_document_events, read_document_events
from sdf_pipeline.step_runner import run_corpus_step
from sdf_pipeline.update_steps import Accepted, Failed, FailureKind, ModelReply
from sdf_step_and_command_line_test_helpers import events_of
from shared import utils
from sdf_consideration_selection_test_builders import (
    answer_with, make_plans, no_stake_resolution, offered_in, rows_of, score_everything_five, small_prompts)  # noqa: F401


@pytest.fixture
def scoring_step(tiny_config, small_prompts):
    return select.ConsiderationScoringStep(
        tiny_config, load_considerations(small_prompts), select.load_selection_template(small_prompts))


PLAN = make_plans(1)[0]


def reply_for(step, plan, score=5):
    request = step.build_request(plan)
    return request, ModelReply(answer_with(request.user_message, lambda label: score), "end_turn")


# --- interpret ---

def test_interpret_a_usable_answer_is_accepted_with_its_attempt_count_and_todays_record(scoring_step):
    _, reply = reply_for(scoring_step, PLAN)
    outcome = scoring_step.interpret(PLAN, reply)
    assert isinstance(outcome, Accepted) and outcome.update["attempts"] == 1
    record = scoring_step.stage_file_record(PLAN, outcome.update)
    assert list(record) == ["prompt_id", "scores", "attempts"] and record["prompt_id"] == PLAN["prompt_id"]


def test_interpret_an_unusable_answer_is_a_failure_that_names_the_attempts_only_on_the_last(scoring_step):
    scoring_step.build_request(PLAN)
    for attempt in (1, 2, 3):
        outcome = scoring_step.interpret(PLAN, ModelReply("not json", "end_turn"))
        assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER
        assert outcome.reason.endswith("(after 3 attempts)") == (attempt == 3)
    # the count starts again for the next run's attempts
    _, reply = reply_for(scoring_step, PLAN)
    assert scoring_step.interpret(PLAN, reply).update["attempts"] == 1


def test_interpret_a_truncated_answer_is_unusable(scoring_step):
    _, reply = reply_for(scoring_step, PLAN)
    outcome = scoring_step.interpret(PLAN, ModelReply(reply.text, "max_tokens"))
    assert outcome.kind is FailureKind.UNUSABLE_ANSWER and "truncated answer" in outcome.reason


def test_the_failure_record_and_line_for_unusable_and_thrown(scoring_step):
    unusable = Failed(FailureKind.UNUSABLE_ANSWER, "no usable JSON object (x) (after 3 attempts)")
    thrown = Failed(FailureKind.THROWN_CALL, "error: RuntimeError: boom")
    assert scoring_step.failure_record(PLAN, unusable) == {
        "prompt_id": PLAN["prompt_id"], "problem": "unusable answer (no usable JSON object (x) (after 3 attempts))"}
    assert scoring_step.failure_record(PLAN, thrown)["problem"] == "API call failed (RuntimeError: boom)"
    assert scoring_step.progress_line(PLAN, thrown) == (
        f"  {PLAN['prompt_id']}: API call failed (RuntimeError: boom) — will be rescored on resume")


def test_the_request_carries_the_schema_token_limit_and_stage(scoring_step, tiny_config):
    request = scoring_step.build_request(PLAN)
    assert request.stage == "layer1_select" and request.item_id == PLAN["prompt_id"]
    assert request.json_schema is not None and request.cache_system is False
    entry_count = sum(len(v) for v in offered_in(request.user_message).values())
    assert request.max_tokens == 400 + 120 * entry_count


def test_the_no_welfare_stake_arm_and_undescribed_plans_are_skipped(tiny_config, small_prompts):
    step = select.ConsiderationScoringStep(
        tiny_config, load_considerations(small_prompts), select.load_selection_template(small_prompts))
    plans = make_plans(4, no_stake={1}, incoherent={2}, resolution=no_stake_resolution(small_prompts))
    assert [step.applies_to(p) for p in plans] == [True, False, False, True]
    tiny_config["sdf"]["select_skip_no_stake"] = False
    keeping = select.ConsiderationScoringStep(
        tiny_config, load_considerations(small_prompts), select.load_selection_template(small_prompts))
    assert [keeping.applies_to(p) for p in plans] == [True, True, False, True]


# --- attempts counted per run ---

def test_an_unusable_answer_is_asked_again_with_the_same_request_then_accepted(
        tiny_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(1)
    answers = iter(["not json", "{}"])

    def dispatcher(user_message, **kw):
        answer = next(answers)
        return answer if answer != "{}" else score_everything_five(user_message)
    calls = stub_claude(dispatcher)
    select.run(tiny_config, small_prompts, stage_dir, plans)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert rows_of(stage_dir, select.SCORES_FILENAME)[0]["attempts"] == 2
    events = events_of(stage_dir, "consideration_scoring")
    assert [e["outcome"] for e in events] == ["failed", "accepted"]
    assert events[0]["request_hash"] == events[1]["request_hash"]
    assert not (stage_dir / select.FAILURES_FILENAME).exists()


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


def test_a_thrown_call_is_not_asked_again_and_counts_towards_the_systemic_stop(
        tiny_config, small_prompts, stage_dir, stub_claude):
    def raising(user_message, **kw):
        raise RuntimeError("backend down")
    calls = stub_claude(raising)
    with pytest.raises(SystemExit, match="every pending scoring call failed"):
        select.run(tiny_config, small_prompts, stage_dir, make_plans(3))
    assert len(calls) == 3
    assert {e["failure_kind"] for e in events_of(stage_dir, "consideration_scoring")} == {"thrown_call"}
    assert all(r["problem"] == "API call failed (RuntimeError: backend down)"
               for r in rows_of(stage_dir, select.FAILURES_FILENAME))


def test_every_document_unusable_on_every_attempt_is_systemic(tiny_config, small_prompts, stage_dir, stub_claude):
    calls = stub_claude(["not json"] * 6)
    with pytest.raises(SystemExit, match="every pending scoring call failed"):
        select.run(tiny_config, small_prompts, stage_dir, make_plans(2))
    assert len(calls) == 6


def test_no_attempts_allowed_makes_no_call_and_fails_the_document(tiny_config, small_prompts, stage_dir, stub_claude):
    tiny_config["sdf"]["max_scoring_attempts"] = 0
    calls = stub_claude([])
    with pytest.raises(SystemExit, match="every pending scoring call failed"):
        select.run(tiny_config, small_prompts, stage_dir, make_plans(1))
    assert calls == []
    assert rows_of(stage_dir, select.FAILURES_FILENAME)[0]["problem"] == "unusable answer ( (after 0 attempts))"


def test_the_code_default_attempts_apply_when_the_config_names_none(tiny_config, small_prompts, stage_dir, stub_claude):
    tiny_config["sdf"].pop("max_scoring_attempts", None)
    calls = stub_claude(["not json"] * 3 + [score_everything_five])
    with pytest.raises(SystemExit):
        select.run(tiny_config, small_prompts, stage_dir, make_plans(1))
    assert len(calls) == select.DEFAULT_MAX_SCORING_ATTEMPTS


# --- the no-welfare-stake arm in the run ---

def test_the_no_welfare_stake_arm_is_never_called_and_never_scored(tiny_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(4, no_stake={2}, resolution=no_stake_resolution(small_prompts))
    calls = stub_claude(score_everything_five)
    select.run(tiny_config, small_prompts, stage_dir, plans)
    assert "matrix_000002" not in {c["item_id"] for c in calls}
    assert "matrix_000002" not in {r["prompt_id"] for r in rows_of(stage_dir, select.SCORES_FILENAME)}
    assert "matrix_000002" not in {e["document_id"] for e in events_of(stage_dir, "consideration_scoring")}


# --- resume ---

def test_resume_makes_zero_calls_and_changes_no_stage_file(tiny_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(5)
    stub_claude(score_everything_five)
    select.run(tiny_config, small_prompts, stage_dir, plans)
    before = {p.name: p.read_bytes() for p in stage_dir.iterdir()}
    events_before = (stage_dir.parent / "document_events.jsonl").read_bytes()
    calls = stub_claude([])
    select.run(tiny_config, small_prompts, stage_dir, plans)
    assert calls == []
    assert {p.name: p.read_bytes() for p in stage_dir.iterdir()} == before
    assert (stage_dir.parent / "document_events.jsonl").read_bytes() == events_before


def test_scoring_writes_no_checkpoint_file(tiny_config, small_prompts, stage_dir, stub_claude):
    stub_claude(score_everything_five)
    select.run(tiny_config, small_prompts, stage_dir, make_plans(3))
    assert not (stage_dir / "_checkpoint.json").exists()


# --- the assignment as a corpus step ---

def test_the_assignment_is_solved_again_over_every_document_when_a_late_plan_arrives(
        tiny_config, small_prompts, stage_dir, stub_claude):
    tiny_config["sdf"].update(max_moral_considerations_per_document=1, max_epistemic_considerations_per_document=1,
                              max_considerations_per_document=2)
    plans = make_plans(10)
    stub_claude(score_everything_five)
    select.run(tiny_config, small_prompts, stage_dir, plans[:5])
    first = {e["document_id"]: e for e in events_of(stage_dir, "consideration_assignment")}
    assert len(first) == 5
    selections_first = {r["prompt_id"]: r["outcomes"] for r in rows_of(stage_dir)}
    count_after_first = len(events_of(stage_dir, "consideration_assignment"))

    select.run(tiny_config, small_prompts, stage_dir, plans)
    selections_second = {r["prompt_id"]: r["outcomes"] for r in rows_of(stage_dir)}
    assert len(selections_second) == 10
    changed = {pid for pid in selections_first if selections_first[pid] != selections_second[pid]}
    new_events = events_of(stage_dir, "consideration_assignment")[count_after_first:]
    # one event per document that is new or whose outcomes changed, none for the unchanged ones
    assert {e["document_id"] for e in new_events} == changed | {p["prompt_id"] for p in plans[5:]}
    assert len(new_events) == len({e["document_id"] for e in new_events})
    folded = fold_document_events(read_document_events(stage_dir.parent / "document_events.jsonl"))
    for prompt_id, outcomes in selections_second.items():
        assert folded[prompt_id].accepted_update_by_step["consideration_assignment"] == {"outcomes": outcomes}


def test_a_run_that_changes_nothing_appends_no_assignment_event(tiny_config, small_prompts, stage_dir, stub_claude):
    plans = make_plans(4)
    stub_claude(score_everything_five)
    select.run(tiny_config, small_prompts, stage_dir, plans)
    count = len(events_of(stage_dir, "consideration_assignment"))
    assert count == 4
    select.run(tiny_config, small_prompts, stage_dir, plans)
    assert len(events_of(stage_dir, "consideration_assignment")) == count


def test_the_assignment_step_makes_no_call_and_keeps_what_the_layer_writes(tiny_config, small_prompts, tmp_path, stub_claude):
    catalog = load_considerations(small_prompts)
    plans = make_plans(3)
    score_rows = []
    for plan in plans:
        scoring = select.ConsiderationScoringStep(tiny_config, catalog, select.load_selection_template(small_prompts))
        request, reply = reply_for(scoring, plan)
        score_rows.append({"prompt_id": plan["prompt_id"], "scores": scoring.interpret(plan, reply).update["scores"], "attempts": 1})
    calls = stub_claude([])
    step = select.ConsiderationAssignmentStep(tiny_config, catalog, plans)
    outcomes = run_corpus_step(step, score_rows, DocumentEventLog(tmp_path / "events.jsonl"))
    assert calls == [] and set(outcomes) == {p["prompt_id"] for p in plans}
    assert [row["prompt_id"] for row in step.selection_rows] == sorted(outcomes)
    assert all(outcomes[row["prompt_id"]].update == {"outcomes": row["outcomes"]} for row in step.selection_rows)
    assert step.coverage_report()["eligible_documents"] == 3


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
