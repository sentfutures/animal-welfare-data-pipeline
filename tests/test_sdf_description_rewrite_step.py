"""The description rewrite as a DocumentStep.

Unit tests call ``interpret`` and the request builder with no stubbed API and no files; the layer
tests run ``layer1_description_rewrite.run`` (unchanged signature) for the Claude route and the
OpenAI route and pin the stage files, the rejections file, resume and the stop rules.
"""

import json

import pytest

from sdf_pipeline import layer1_description_rewrite as rewrite
from sdf_pipeline.layer1_description_rewrite import DescriptionRewriteStep
from sdf_pipeline.step_runner import send_model_request
from sdf_pipeline.update_steps import Accepted, Failed, FailureKind, Removed, WhatAFailedDocumentDoes
from sdf_step_and_command_line_test_helpers import reply
from shared import api, openai_api, utils
from sdf_description_rewrite_test_builders import answer, make_plans

CLAUDE_MODEL = "claude-sonnet-5"
OPENAI_MODEL = "gpt-6.1-sol"


@pytest.fixture
def plans(prompts_sdf):
    return make_plans(prompts_sdf, count=4)


def make_step(config, prompts_sdf, tmp_path, plans, model):
    config["sdf"]["description_rewrite_model"] = model
    prompts = rewrite.load_rewrite_prompts(prompts_sdf)
    return DescriptionRewriteStep(config, prompts, prompts_sdf, tmp_path, plans)


@pytest.fixture(params=[CLAUDE_MODEL, OPENAI_MODEL], ids=["claude_route", "openai_route"])
def step(request, tiny_config, prompts_sdf, tmp_path, plans):
    return make_step(tiny_config, prompts_sdf, tmp_path, plans, request.param)


# ---------------------------------------------------------------- interpret, both routes

def test_interpret_a_decline(step, plans):
    outcome = step.interpret(plans[0], reply(answer()))
    assert isinstance(outcome, Accepted) and not outcome.update.rewrite_needed
    record = step.stage_file_record(plans[0], outcome.update)
    assert record["rewrite_needed"] is False and record["rewritten_description"] == ""
    assert record["original_description"] == plans[0]["description"] and record["changes"] == []


def test_interpret_a_rewrite_keeps_the_changes(step, plans):
    outcome = step.interpret(plans[0], reply(answer(True, "New text.", ("organisation_naming",))))
    assert isinstance(outcome, Accepted) and outcome.update.rewrite_needed
    record = step.stage_file_record(plans[0], outcome.update)
    assert record["rewritten_description"] == "New text."
    assert record["changes"] == [{"check": "organisation_naming", "reason": "Fixed under organisation_naming."}]
    assert list(record) == ["prompt_id", "reasoning", "rewrite_needed", "original_description",
                            "rewritten_description", "changes", "dealt_value_conflict", "conflict_reason"]


def test_interpret_a_dealt_value_conflict_is_removed_and_its_records_are_todays(step, plans, tmp_path):
    outcome = step.interpret(plans[1], reply(answer(True, conflict=True, conflict_reason="Scale cannot occur.")))
    assert isinstance(outcome, Removed) and outcome.reason_code == "dealt_value_conflict"
    record = step.stage_file_record_for_removal(plans[1], outcome)
    assert record["dealt_value_conflict"] is True and record["rewrite_needed"] is False
    assert record["conflict_reason"] == "Scale cannot occur." and record["rewritten_description"] == ""
    (rejection,) = utils.load_jsonl(tmp_path / "description_rewrite_rejections.jsonl")
    assert rejection == {"prompt_id": plans[1]["prompt_id"], "variables": plans[1]["variables"],
                         "conflict_reason": "Scale cannot occur.", "description": plans[1]["description"]}


def test_interpret_a_truncated_answer_is_an_unusable_failure(step, plans):
    outcome = step.interpret(plans[0], reply(answer(), stop_reason="max_tokens"))
    assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER
    assert outcome.reason.startswith("unusable answer (")


SELF_CONTRADICTORY_ANSWER = answer(True, "")  # rewrite_needed with no rewritten description


@pytest.mark.parametrize("text", ["not json", "", "[1]", SELF_CONTRADICTORY_ANSWER])
def test_interpret_an_unparseable_answer_is_an_unusable_failure(step, plans, text):
    outcome = step.interpret(plans[0], reply(text))
    assert isinstance(outcome, Failed) and outcome.kind is FailureKind.UNUSABLE_ANSWER


def test_a_thrown_call_prints_todays_line_and_an_unusable_answer_its_own(step, plans):
    pid = plans[0]["prompt_id"]
    step.begin_session(pending_count=5)
    assert step.progress_line(plans[0], Failed(FailureKind.THROWN_CALL, "error: RuntimeError: boom")) == (
        f"  {pid}: API call failed (RuntimeError: boom) — will retry on resume")
    assert step.progress_line(plans[0], Failed(FailureKind.UNUSABLE_ANSWER, "unusable answer (bad)")) == (
        f"  {pid}: unusable answer (bad) — will retry on resume")


def test_the_policy_counts_both_failure_kinds_and_stops_the_run_after_the_step(step):
    assert step.policy.failure_kinds_counting_towards_the_systemic_stop == frozenset(FailureKind)
    assert step.policy.failed_document is WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP


# ---------------------------------------------------------------- the request

def test_the_claude_request_has_the_arguments_rewrite_description_passes(
        tiny_config, prompts_sdf, tmp_path, plans, stub_claude):
    step = make_step(tiny_config, prompts_sdf, tmp_path, plans, CLAUDE_MODEL)
    request = step.build_request(plans[0])
    assert request.openai_schema_name is None and request.openai_reasoning_effort is None
    calls = stub_claude(lambda user_message, **kw: answer())
    send_model_request(request)
    rewrite.rewrite_description(step.render_variable_lines(plans[0]), plans[0]["description"],
                                prompts=step.prompts, model=CLAUDE_MODEL, item_id=plans[0]["prompt_id"])
    assert calls[0] == calls[1]


def test_the_openai_request_has_the_arguments_rewrite_description_passes(
        tiny_config, prompts_sdf, tmp_path, plans, monkeypatch):
    tiny_config["sdf"]["description_rewrite_reasoning_effort"] = "low"
    step = make_step(tiny_config, prompts_sdf, tmp_path, plans, OPENAI_MODEL)
    request = step.build_request(plans[0])
    assert request.openai_reasoning_effort == "low" and request.openai_schema_name == "description_rewrite_answer"
    calls = []
    monkeypatch.setattr(openai_api, "call_openai", lambda *a, **kw: calls.append((a, kw)) or (answer(), "end_turn"))
    send_model_request(request)
    rewrite.rewrite_description(step.render_variable_lines(plans[0]), plans[0]["description"],
                                prompts=step.prompts, model=OPENAI_MODEL, reasoning_effort="low",
                                item_id=plans[0]["prompt_id"])
    assert calls[0] == calls[1]
    assert sorted(calls[0][1]) == sorted([
        "system_prompt", "model", "max_output_tokens", "reasoning_effort", "stage", "item_id",
        "json_schema", "schema_name"])


def test_the_openai_default_reasoning_effort_is_resolved_into_the_request(
        tiny_config, prompts_sdf, tmp_path, plans):
    step = make_step(tiny_config, prompts_sdf, tmp_path, plans, OPENAI_MODEL)
    assert step.build_request(plans[0]).openai_reasoning_effort == openai_api.DEFAULT_REASONING_EFFORT


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


def test_decline_rewrite_and_conflict_in_one_run(tiny_config, prompts_sdf, stage_dir, plans, route):
    _, install = route
    rewritten, doomed = plans[0]["prompt_id"], plans[1]["prompt_id"]

    def dispatcher(item_id):
        if item_id == rewritten:
            return answer(True, "Rewritten.", ("organisation_naming",))
        if item_id == doomed:
            return answer(conflict=True, conflict_reason="Scale cannot occur.")
        return answer()

    install(dispatcher)
    kept = rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert [p["prompt_id"] for p in kept] == [p["prompt_id"] for p in plans if p["prompt_id"] != doomed]
    assert kept[0]["description"] == "Rewritten." and kept[0]["original_description"] == plans[0]["description"]
    assert kept[1] is plans[2]
    stored = utils.load_jsonl(stage_dir / "description_rewrites.jsonl")
    assert [r["prompt_id"] for r in stored] == [p["prompt_id"] for p in plans]  # input order, conflict included
    assert [r["prompt_id"] for r in utils.load_jsonl(stage_dir / "description_rewrite_rejections.jsonl")] == [doomed]
    assert utils.Checkpoint(stage_dir / "_description_rewrite_checkpoint.json").done_count == len(plans)


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
    assert not utils.Checkpoint(stage_dir / "_description_rewrite_checkpoint.json").is_done(broken)
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
