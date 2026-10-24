"""Offline tests for the OpenAI path (shared/openai_api.py) and for routing the layer-1
description rewrite to it by model name. The network seam is stubbed (see conftest);
no test here may reach the API."""

import json

import pytest

from sdf_pipeline import layer1_description_rewrite as rewrite
from shared import api, openai_api, utils

ANSWER = json.dumps({
    "reasoning": "r", "rewrite_needed": False, "rewritten_description": "", "changes": [],
    "dealt_value_conflict": False, "conflict_reason": ""})
SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"],
          "additionalProperties": False}


@pytest.fixture
def cost_log(tmp_path, monkeypatch):
    path = tmp_path / "cost_log.jsonl"
    monkeypatch.setattr(api, "_cost_log_path", path)
    return path


def logged_records(path):
    return utils.load_jsonl(path)


def test_only_gpt_prefixed_models_route_to_openai():
    assert openai_api.is_openai_model("gpt-6.1-sol")
    assert not openai_api.is_openai_model("claude-sonnet-5")
    assert not openai_api.is_openai_model(None) and not openai_api.is_openai_model("")


def test_request_shape_matches_the_working_client(stub_openai_responses, openai_response, cost_log):
    calls = stub_openai_responses([openai_response(text='{"x": "y"}')])
    text, stop = openai_api.call_openai("USER", "SYSTEM", model="gpt-6.1-sol", max_output_tokens=8000,
                                        reasoning_effort="medium", json_schema=SCHEMA, schema_name="name_here")
    assert (text, stop) == ('{"x": "y"}', "end_turn")
    request = calls[0]
    assert request["model"] == "gpt-6.1-sol" and request["instructions"] == "SYSTEM" and request["input"] == "USER"
    assert request["temperature"] == 1 and "top_p" not in request
    assert request["reasoning"] == {"effort": "medium"} and request["max_output_tokens"] == 8000
    assert request["text"]["format"] == {"type": "json_schema", "name": "name_here", "schema": SCHEMA, "strict": True}


def test_cost_record_has_the_claude_shape_and_the_gpt_6_1_sol_price(stub_openai_responses, openai_response, cost_log):
    stub_openai_responses([openai_response(input_tokens=10_000, cached_tokens=4_000, output_tokens=2_000,
                                           reasoning_tokens=1_500)])
    openai_api.call_openai("u", "s", model="gpt-6.1-sol", max_output_tokens=8000, stage="layer1_description_rewrite",
                           item_id="p7", json_schema=SCHEMA)
    (record,) = logged_records(cost_log)
    expected_cost = (6_000 * 2.00 + 4_000 * 0.10 + 2_000 * 10.00) / 1_000_000
    assert record["cost_usd"] == pytest.approx(expected_cost)
    assert record["model"] == "gpt-6.1-sol" and record["backend"] == "openai"
    assert (record["input_tokens"], record["output_tokens"]) == (10_000, 2_000)
    assert record["cache_read_tokens"] == 4_000 and record["reasoning_tokens"] == 1_500
    assert record["stage"] == "layer1_description_rewrite" and record["item_id"] == "p7"
    assert record["attempts"] == 1 and "duration_s" in record and "timestamp" in record


@pytest.mark.parametrize("response_arguments, expected_stop", [
    ({"status": "incomplete", "incomplete_reason": "max_output_tokens", "text": ""}, "max_output_tokens"),
    ({"refusal": "I cannot help with that."}, "refusal"),
    ({"status": "failed", "text": ""}, "failed"),
])
def test_unusable_responses_are_reported_and_still_costed(stub_openai_responses, openai_response, cost_log,
                                                          response_arguments, expected_stop):
    stub_openai_responses([openai_response(**response_arguments)])
    _, stop = openai_api.call_openai("u", "s", model="gpt-6.1-sol", max_output_tokens=10, json_schema=SCHEMA)
    assert stop == expected_stop
    assert len(logged_records(cost_log)) == 1


def test_missing_key_fails_clearly_and_ignores_the_current_directory_dotenv_file(monkeypatch, cost_log, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-a-dotenv-file\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set. Fill in OPENAI_API_KEY"):
        openai_api.call_openai("u", "s", model="gpt-6.1-sol", max_output_tokens=10)


def test_an_unpriced_model_is_refused_before_any_call(stub_openai_responses, cost_log):
    calls = stub_openai_responses([])
    with pytest.raises(ValueError, match="no price"):
        openai_api.call_openai("u", "s", model="gpt-unpriced", max_output_tokens=10)
    assert calls == []


# --- routing and checkpointing through the description rewrite ---

def test_rewrite_description_routes_by_model_name(stub_openai_responses, openai_response, stub_claude,
                                                   prompts_sdf, cost_log):
    claude_calls = stub_claude(lambda user_message, **kw: ANSWER)
    openai_calls = stub_openai_responses([openai_response(text=ANSWER)])
    prompts = rewrite.load_rewrite_prompts(prompts_sdf)
    rewrite.rewrite_description("l", "d", prompts=prompts, model="gpt-6.1-sol",
                                reasoning_effort="low", item_id="p1")
    assert len(openai_calls) == 1 and claude_calls == []
    request = openai_calls[0]
    assert request["reasoning"] == {"effort": "low"} and request["max_output_tokens"] == rewrite._MAX_TOKENS >= 8000
    assert request["text"]["format"]["schema"] == prompts.answer_schema  # the existing schema, passed through
    assert request["text"]["format"]["strict"] is True and request["instructions"]
    rewrite.rewrite_description("l", "d", prompts=prompts, model="claude-sonnet-5")
    assert len(claude_calls) == 1 and len(openai_calls) == 1


@pytest.mark.parametrize("response_arguments, message", [
    ({"status": "incomplete", "incomplete_reason": "max_output_tokens", "text": ANSWER}, "truncated"),
    ({"refusal": "no"}, "truncated"),
])
def test_truncated_or_refused_openai_answers_are_rejected(stub_openai_responses, openai_response, prompts_sdf,
                                                          cost_log, response_arguments, message):
    stub_openai_responses([openai_response(**response_arguments)])
    with pytest.raises(rewrite.RewriteAnswerError, match=message):
        rewrite.rewrite_description("l", "d", prompts=rewrite.load_rewrite_prompts(prompts_sdf),
                                    model="gpt-6.1-sol")


def test_stage_does_not_checkpoint_truncated_openai_answers_and_resume_retries_them(
        tiny_config, prompts_sdf, tmp_path, stub_openai_responses, openai_response, cost_log):
    from tests.test_layer1_description_rewrite import make_plans
    stage_dir = tmp_path / "layer1"
    stage_dir.mkdir()
    plans = make_plans(prompts_sdf, count=3)
    tiny_config["workers"] = 1
    tiny_config["sdf"].update({"description_rewrite_model": "gpt-6.1-sol",
                               "description_rewrite_reasoning_effort": "high"})
    truncated_id = plans[0]["prompt_id"]

    def first_pass(request):
        if truncated_id in request["input"] or plans[0]["description"] in request["input"]:
            return openai_response(status="incomplete", incomplete_reason="max_output_tokens", text="{")
        return openai_response(text=ANSWER)

    calls = stub_openai_responses(first_pass)
    with pytest.raises(SystemExit, match="1 descriptions held out"):
        rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert all(call["reasoning"] == {"effort": "high"} for call in calls)
    assert set(rewrite.load_results(stage_dir)) == {p["prompt_id"] for p in plans[1:]}

    resumed_calls = stub_openai_responses(lambda request: openai_response(text=ANSWER))
    rewrite.run(tiny_config, prompts_sdf, stage_dir, plans)
    assert len(resumed_calls) == 1  # only the truncated plan is retried
    assert len(rewrite.load_results(stage_dir)) == 3
    assert {r["backend"] for r in logged_records(cost_log)} == {"openai"}
