"""The description write step: the reply read for every kind of answer, the request sent, resume behaviour, printed lines."""

import pytest

from sdf_pipeline.layer2_description_write_and_description_rewrite import description_write as description_write_module
from sdf_pipeline.layer2_description_write_and_description_rewrite.description_write import PLANS, STEP
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Failed, FailureKind, Removed, StepProgress
from shared import utils
from shared.sdf_run_layouts import DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE
from sdf_plan_draft_rewrite_and_score_test_builders import plan_reply
from sdf_step_and_command_line_test_helpers import outcome_of, records_of, run_of


@pytest.fixture
def plan_run(tiny_config, prompts_sdf, stage_dir):
    return run_of(tiny_config, prompts_sdf, DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE, stage_dir)


PROMPT_RECORD = {"prompt_id": "matrix_000007", "variables": {"tone": "plain"}, "prompt": "PROMPT", "system": "SYS"}


def progress_line_of(outcome, run):
    return STEP.progress_line([PROMPT_RECORD], {"matrix_000007": outcome}, StepProgress(), run)


def test_read_an_accepted_plan(plan_run):
    reply_text = plan_reply("The spec.")
    outcome = outcome_of(STEP, PROMPT_RECORD, reply_text, plan_run)
    assert isinstance(outcome, Accepted)
    ((file_name, record),) = records_of(STEP, PROMPT_RECORD, outcome, plan_run).records
    assert file_name == "plans.jsonl"
    assert list(record.items()) == [
        ("prompt_id", "matrix_000007"), ("variables", {"tone": "plain"}), ("plan", reply_text),
        ("description", "The spec."), ("incoherent", False)]
    assert progress_line_of(outcome, plan_run) == "  Wrote description for matrix_000007"


def test_read_an_incoherent_plan_is_a_removal_with_todays_record(plan_run):
    reply_text = "<document_description>INCOHERENT: no such document.</document_description>"
    outcome = outcome_of(STEP, PROMPT_RECORD, reply_text, plan_run)
    assert isinstance(outcome, Removed) and outcome.reason_code == "incoherent"
    records = records_of(STEP, PROMPT_RECORD, outcome, plan_run)
    ((file_name, record),) = records.records
    assert file_name == PLANS.file_name
    assert record["incoherent"] is True and record["plan"] == reply_text
    assert list(record) == ["prompt_id", "variables", "plan", "description", "incoherent"]
    assert progress_line_of(outcome, plan_run) == "  Wrote description for matrix_000007 (INCOHERENT)"


def test_read_unusable_answers_are_failures_left_behind(plan_run):
    truncated = outcome_of(STEP, PROMPT_RECORD, plan_reply(), plan_run, stop_reason="max_tokens")
    no_tags = outcome_of(STEP, PROMPT_RECORD, "chatter", plan_run)
    for outcome in (truncated, no_tags):
        assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER
        assert records_of(STEP, PROMPT_RECORD, outcome, plan_run).records == ()
    assert progress_line_of(truncated, plan_run) == (
        "  matrix_000007: truncated description write (stop_reason=max_tokens) — will retry on resume")
    assert progress_line_of(no_tags, plan_run) == (
        "  matrix_000007: no DOCUMENT DESCRIPTION found — will retry on resume")


def test_a_thrown_call_prints_the_api_failure_line_and_counts_towards_the_systemic_stop(
        tiny_config, prompts_sdf, stage_dir, stub_claude, capsys):
    def always_throws(user_message, **kw):
        raise RuntimeError("boom")
    stub_claude(always_throws)
    with pytest.raises(SystemExit, match="layer2_description_write: every pending API call failed"):
        description_write_module.run(tiny_config, prompts_sdf, stage_dir)
    assert "API call failed (error: RuntimeError: boom) — will retry on resume" in capsys.readouterr().out
    assert not (stage_dir / "plans.jsonl").exists() or utils.load_jsonl(stage_dir / "plans.jsonl") == []


def test_resume_makes_zero_calls_and_changes_no_file(tiny_config, prompts_sdf, stage_dir, stub_claude):
    stub_claude(lambda user_message, **kw: plan_reply(f"Plan {kw['item_id']}."))
    description_write_module.run(tiny_config, prompts_sdf, stage_dir)
    before = {p.name: p.read_bytes() for p in stage_dir.iterdir()}
    calls = stub_claude([])
    records = description_write_module.run(tiny_config, prompts_sdf, stage_dir)
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
    description_write_module.run(config, prompts_sdf, stage_dir)
    ids_by_position = {position: item for item, position in behaviour_by_position.items()}
    printed = capsys.readouterr().out
    assert f"  {ids_by_position[1]}: API call failed (error: RuntimeError: thrown) — will retry on resume" in printed
    assert f"  Wrote description for {ids_by_position[2]} (INCOHERENT)" in printed
    for position in (3, 4):
        failure_line = [line for line in printed.splitlines() if line.startswith(f"  {ids_by_position[position]}: ")]
        assert len(failure_line) == 1 and "API call failed" not in failure_line[0]
    finished = [r["prompt_id"] for r in utils.load_jsonl(stage_dir / "plans.jsonl")]
    assert sorted(finished) == sorted([ids_by_position[0], ids_by_position[2]])

    calls = stub_claude(lambda user_message, **kw: plan_reply(f"Recovered {kw['item_id']}."))
    records = description_write_module.run(config, prompts_sdf, stage_dir)
    assert sorted(c["item_id"] for c in calls) == sorted(ids_by_position[p] for p in (1, 3, 4))
    assert len(records) == 5
    assert [r["prompt_id"] for r in utils.load_jsonl(stage_dir / "plans.jsonl")[:2]] == finished
