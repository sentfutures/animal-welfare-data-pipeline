"""The description rewrite step.

Unit tests call ``read_reply`` and the request builder with no stubbed API and no files; the layer
tests run ``layer1_description_rewrite.run`` (unchanged signature) for the Claude route and the
OpenAI route and pin the stage files, the rejections file, resume and the stop rules.
"""

import pytest

from sdf_pipeline.layer2_description_write_and_description_rewrite import description_rewrite as rewrite
from sdf_pipeline.layer2_description_write_and_description_rewrite.description_rewrite import STEP
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Failed, FailureKind, Removed, StepProgress, WhatAFailedDocumentDoes
from sdf_step_and_command_line_test_helpers import outcome_of, records_of, reply, run_of
from shared import api, openai_api
from shared.sdf_run_layouts import DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE
from sdf_description_rewrite_test_builders import answer, make_plans

CLAUDE_MODEL = "claude-sonnet-5"
OPENAI_MODEL = "gpt-6.1-sol"


@pytest.fixture
def plans(prompts_sdf):
    return make_plans(prompts_sdf, count=4)


def make_run(config, prompts_sdf, tmp_path, model):
    config["sdf"]["description_rewrite_model"] = model
    return run_of(config, prompts_sdf, DESCRIPTION_WRITE_AND_DESCRIPTION_REWRITE, tmp_path)


@pytest.fixture(params=[CLAUDE_MODEL, OPENAI_MODEL], ids=["claude_route", "openai_route"])
def run_of_step(request, tiny_config, prompts_sdf, tmp_path):
    return make_run(tiny_config, prompts_sdf, tmp_path, request.param)


def progress_line_of(plan, outcome, progress, run):
    return STEP.progress_line([plan], {plan["prompt_id"]: outcome}, progress, run)


# ---------------------------------------------------------------- reading the reply, both routes

def test_read_a_rewrite_keeps_the_changes(run_of_step, plans):
    outcome = outcome_of(STEP, plans[0], reply(answer(True, "New text.", ("organisation_naming",))), run_of_step)
    assert isinstance(outcome, Accepted) and outcome.update.rewrite_needed
    ((_, record),) = records_of(STEP, plans[0], outcome, run_of_step).records
    assert record["rewritten_description"] == "New text."
    assert record["changes"] == [{"check": "organisation_naming", "reason": "Fixed under organisation_naming."}]
    assert list(record) == ["prompt_id", "reasoning", "rewrite_needed", "original_description",
                            "rewritten_description", "changes", "dealt_value_conflict", "conflict_reason"]


def test_read_a_dealt_value_conflict_is_removed_and_its_records_are_todays(run_of_step, plans, tmp_path):
    outcome = outcome_of(STEP, plans[1], reply(answer(True, conflict=True, conflict_reason="Scale cannot occur.")), run_of_step)
    assert isinstance(outcome, Removed) and outcome.reason_code == "dealt_value_conflict"
    records = records_of(STEP, plans[1], outcome, run_of_step)
    # the results record first, then the rejection record; the step writes neither
    assert [file_name for file_name, _ in records.records] == [rewrite.RESULTS_FILENAME, rewrite.REJECTIONS_FILENAME]
    assert not (tmp_path / "description_rewrite_rejections.jsonl").exists()
    (_, record), (_, rejection) = records.records
    assert record["dealt_value_conflict"] is True and record["rewrite_needed"] is False
    assert record["conflict_reason"] == "Scale cannot occur." and record["rewritten_description"] == ""
    assert rejection == {"prompt_id": plans[1]["prompt_id"], "variables": plans[1]["variables"],
                         "conflict_reason": "Scale cannot occur.", "description": plans[1]["description"]}


def test_read_a_truncated_answer_is_an_unusable_failure(run_of_step, plans):
    outcome = outcome_of(STEP, plans[0], reply(answer(), stop_reason="max_tokens"), run_of_step)
    assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER
    assert outcome.reason.startswith("unusable answer (")


SELF_CONTRADICTORY_ANSWER = answer(True, "")  # rewrite_needed with no rewritten description


@pytest.mark.parametrize("text", ["not json", "", "[1]", SELF_CONTRADICTORY_ANSWER])
def test_read_an_unparseable_answer_is_an_unusable_failure(run_of_step, plans, text):
    outcome = outcome_of(STEP, plans[0], reply(text), run_of_step)
    assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER


def test_a_thrown_call_prints_todays_line_and_an_unusable_answer_its_own(run_of_step, plans):
    pid = plans[0]["prompt_id"]
    progress = StepProgress(pending_count=5)
    assert progress_line_of(plans[0], Failed(FailureKind.THROWN_CALL, "error: RuntimeError: boom"), progress, run_of_step) == (
        f"  {pid}: API call failed (RuntimeError: boom) — will retry on resume")
    assert progress_line_of(plans[0], Failed(FailureKind.UNUSABLE_ANSWER, "unusable answer (bad)"), progress, run_of_step) == (
        f"  {pid}: unusable answer (bad) — will retry on resume")


def test_the_policy_counts_both_failure_kinds_and_stops_the_run_after_the_step():
    assert STEP.policy.failure_kinds_counting_towards_the_systemic_stop == frozenset(FailureKind)
    assert STEP.policy.failed_document is WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP


# ---------------------------------------------------------------- the layer, both routes

@pytest.fixture(params=[CLAUDE_MODEL, OPENAI_MODEL], ids=["claude_route", "openai_route"])
def route(request, tiny_config, monkeypatch):
    """(model, install): ``install(dispatcher)`` stubs the route's API function with
    ``dispatcher(item_id) -> answer text or (text, stop reason) or an exception to raise``
    and returns the list of item ids called."""
    tiny_config["workers"] = 1
    tiny_config["sdf"]["description_rewrite_model"] = request.param

    def install(dispatcher):
        item_ids = []

        def respond(item_id):
            item_ids.append(item_id)
            result = dispatcher(item_id)
            if isinstance(result, Exception):
                raise result
            return result if isinstance(result, tuple) else (result, "end_turn")

        monkeypatch.setattr(api, "call_claude", lambda user, **kw: respond(kw["item_id"]))
        monkeypatch.setattr(openai_api, "call_openai", lambda user, **kw: respond(kw["item_id"]))
        return item_ids

    return request.param, install


@pytest.mark.parametrize("failure", ["truncated", "unparseable", "thrown"])
def test_a_failed_document_stops_the_run_after_the_step_and_resume_retries_exactly_it(
        tiny_config, prompts_sdf, stage_dir, plans, route, failure, capsys):
    _, install = route
    broken = plans[2]["prompt_id"]
    problems = {"truncated": (answer(), "max_tokens"), "unparseable": "not json", "thrown": RuntimeError("boom")}
    install(lambda item_id: problems[failure] if item_id == broken else answer())
    with pytest.raises(SystemExit, match="1 descriptions held out — stopping before the stages that assign"):
        rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    printed = capsys.readouterr().out
    assert f"{broken}: " in printed and "— will retry on resume" in printed
    assert printed.index("1 descriptions held out of drafting") < printed.index("this session)")
    assert ("API call failed (RuntimeError: boom)" in printed) == (failure == "thrown")
    assert set(rewrite.load_results(stage_dir)) == {p["prompt_id"] for p in plans} - {broken}
    retried = install(lambda item_id: answer())
    kept = rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert retried == [broken] and len(kept) == len(plans)
    assert set(rewrite.load_results(stage_dir)) == {p["prompt_id"] for p in plans}


@pytest.mark.parametrize("failure", [(("not json"), "unusable"), (RuntimeError("down"), "thrown")])
def test_every_call_failing_is_systemic_for_both_failure_kinds(
        tiny_config, prompts_sdf, stage_dir, plans, route, failure):
    _, install = route
    install(lambda item_id: failure[0])
    with pytest.raises(SystemExit, match="every pending call failed"):
        rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)


def test_resume_after_a_finished_run_makes_zero_calls_and_returns_the_same_plans(
        tiny_config, prompts_sdf, stage_dir, plans, route):
    _, install = route
    doomed = plans[1]["prompt_id"]
    install(lambda item_id: answer(conflict=True, conflict_reason="No.") if item_id == doomed else answer())
    first = rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    bytes_before = {f.name: f.read_bytes() for f in stage_dir.iterdir() if f.is_file()}
    calls = install(lambda item_id: pytest.fail("a finished plan was sent again"))
    assert rewrite.run(tiny_config, prompts_sdf, stage_dir, plans) == first and calls == []
    assert {f.name: f.read_bytes() for f in stage_dir.iterdir() if f.is_file()} == bytes_before


def test_the_progress_and_summary_lines_are_todays(tiny_config, prompts_sdf, stage_dir, plans, route, capsys):
    _, install = route
    install(lambda item_id: answer(True, "R.", ("organisation_naming",)) if item_id == plans[0]["prompt_id"] else answer())
    rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "  Description rewrite: 4 descriptions, 4 to check (0 already done)"
    assert lines[1].startswith("  Description rewrite 4/4: 1 rewritten, 0 dropped for a dealt-value conflict (")
    assert lines[2].startswith("  Description rewrite: 1 of 4 descriptions rewritten, 0 dropped for a dealt-value conflict (")
    assert len(lines) == 3
