"""The plan layer as a DocumentStep: interpret on every kind of reply, the request sent, resume
behaviour, printed lines. ``layer1_plan.run`` keeps its signature and behaviour."""

import pytest

from sdf_pipeline import layer1_plan
from sdf_pipeline.update_steps import Accepted, Failed, FailureKind, ModelReply, Removed, StepProgress
from shared import utils
from sdf_plan_draft_rewrite_and_score_test_builders import plan_reply


@pytest.fixture
def plan_step(tiny_config):
    return layer1_plan.PlanStep(tiny_config)


PROMPT_RECORD = {"prompt_id": "matrix_000007", "variables": {"tone": "plain"}, "prompt": "PROMPT", "system": "SYS"}


def test_interpret_an_accepted_plan(plan_step):
    reply_text = plan_reply("The spec.")
    outcome = plan_step.interpret(PROMPT_RECORD, ModelReply(reply_text, "end_turn"))
    assert isinstance(outcome, Accepted)
    assert list(plan_step.stage_file_record(PROMPT_RECORD, outcome.update).items()) == [
        ("prompt_id", "matrix_000007"), ("variables", {"tone": "plain"}), ("plan", reply_text),
        ("description", "The spec."), ("incoherent", False)]
    assert plan_step.progress_line(PROMPT_RECORD, outcome, StepProgress()) == "  Planned matrix_000007"


def test_interpret_an_incoherent_plan_is_a_removal_with_todays_record(plan_step):
    reply_text = "<document_description>INCOHERENT: no such document.</document_description>"
    outcome = plan_step.interpret(PROMPT_RECORD, ModelReply(reply_text, "end_turn"))
    assert isinstance(outcome, Removed) and outcome.reason_code == "incoherent"
    records = plan_step.outcome_records(PROMPT_RECORD, outcome)
    assert records.finishes_the_document
    ((file_name, record),) = records.records
    assert file_name == layer1_plan.PlanStep.stage_file_name
    assert record["incoherent"] is True and record["plan"] == reply_text
    assert list(record) == ["prompt_id", "variables", "plan", "description", "incoherent"]
    assert plan_step.progress_line(PROMPT_RECORD, outcome, StepProgress()) == "  Planned matrix_000007 (INCOHERENT)"


def test_interpret_unusable_answers_are_failures_left_behind(plan_step):
    truncated = plan_step.interpret(PROMPT_RECORD, ModelReply(plan_reply(), "max_tokens"))
    no_tags = plan_step.interpret(PROMPT_RECORD, ModelReply("chatter", "end_turn"))
    for outcome in (truncated, no_tags):
        assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER
    assert plan_step.progress_line(PROMPT_RECORD, truncated, StepProgress()) == (
        "  matrix_000007: truncated plan (stop_reason=max_tokens) — will retry on resume")
    assert plan_step.progress_line(PROMPT_RECORD, no_tags, StepProgress()) == (
        "  matrix_000007: no DOCUMENT DESCRIPTION found — will retry on resume")


def test_the_request_is_the_composed_prompt_with_no_other_settings(plan_step, tiny_config):
    request = plan_step.build_request(PROMPT_RECORD)
    assert (request.user_message, request.system_prompt, request.stage, request.item_id) == (
        "PROMPT", "SYS", "layer2_plan", "matrix_000007")
    assert request.model == tiny_config["sdf"].get("plan_model")
    assert request.max_tokens is None and request.cache_system is False
    assert request.temperature is None and request.json_schema is None


def test_a_thrown_call_prints_the_api_failure_line_and_counts_towards_the_systemic_stop(
        tiny_config, prompts_sdf, stage_dir, stub_claude, capsys):
    def always_throws(user_message, **kw):
        raise RuntimeError("boom")
    stub_claude(always_throws)
    with pytest.raises(SystemExit, match="layer2_plan: every pending API call failed"):
        layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
    assert "API call failed (error: RuntimeError: boom) — will retry on resume" in capsys.readouterr().out
    assert not (stage_dir / "plans.jsonl").exists() or utils.load_jsonl(stage_dir / "plans.jsonl") == []


def test_resume_makes_zero_calls_and_changes_no_file(tiny_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(lambda user_message, **kw: plan_reply(f"Plan {kw['item_id']}."))
    layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
    before = {p.name: p.read_bytes() for p in stage_dir.iterdir()}
    calls = stub_claude([])
    records = layer1_plan.run(tiny_config, prompts_sdf, stage_dir)
    assert calls == [] and len(records) == tiny_config["sdf"]["n_prompts"]
    assert {p.name: p.read_bytes() for p in stage_dir.iterdir()} == before


def test_resume_retries_exactly_the_failed_plans_and_keeps_incoherent_ones_done(
        tiny_config, prompts_sdf, stage_dir, stub_claude, capsys):
    config = {**tiny_config, "sdf": {**tiny_config["sdf"], "n_prompts": 5}}
    behaviour_by_position = {}

    def first_attempt(user_message, **kw):
        position = behaviour_by_position.setdefault(kw["item_id"], len(behaviour_by_position))
        if position == 0:
            return plan_reply("fine")
        if position == 1:
            raise RuntimeError("thrown")
        if position == 2:
            return "<document_description>INCOHERENT: no.</document_description>"
        if position == 3:
            return plan_reply("cut"), "max_tokens"
        return "no tags", "end_turn"

    stub_claude(first_attempt)
    layer1_plan.run(config, prompts_sdf, stage_dir)
    ids_by_position = {position: item for item, position in behaviour_by_position.items()}
    printed = capsys.readouterr().out
    assert f"  {ids_by_position[1]}: API call failed (error: RuntimeError: thrown) — will retry on resume" in printed
    assert f"  Planned {ids_by_position[2]} (INCOHERENT)" in printed
    for position in (3, 4):
        failure_line = [line for line in printed.splitlines() if line.startswith(f"  {ids_by_position[position]}: ")]
        assert len(failure_line) == 1 and "API call failed" not in failure_line[0]
    finished = [r["prompt_id"] for r in utils.load_jsonl(stage_dir / "plans.jsonl")]
    assert sorted(finished) == sorted([ids_by_position[0], ids_by_position[2]])
    checkpoint = utils.Checkpoint(stage_dir / "_checkpoint.json")
    assert all(checkpoint.is_done(i) for i in finished)
    assert not any(checkpoint.is_done(ids_by_position[p]) for p in (1, 3, 4))

    calls = stub_claude(lambda user_message, **kw: plan_reply(f"Recovered {kw['item_id']}."))
    records = layer1_plan.run(config, prompts_sdf, stage_dir)
    assert sorted(c["item_id"] for c in calls) == sorted(ids_by_position[p] for p in (1, 3, 4))
    assert len(records) == 5
    assert [r["prompt_id"] for r in utils.load_jsonl(stage_dir / "plans.jsonl")[:2]] == finished
