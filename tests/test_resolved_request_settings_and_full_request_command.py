"""``shared.api.resolve_request_settings`` returns what ``call_claude`` sends, and the read-only
command ``scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py`` prints settings
equal to the ones the SDF prompt-equivalence test recorded for the same item. No model API is called."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import sdf_prompt_equivalence_fixture_support as support   # imported first: captures the real network seams
from shared import api, openai_api

COMMAND_PATH = (Path(__file__).resolve().parent.parent / "scripts"
                / "print_full_sdf_request_a_stage_would_send_for_one_document.py")
_specification = importlib.util.spec_from_file_location("print_full_sdf_request_command", COMMAND_PATH)
command = importlib.util.module_from_spec(_specification)
_specification.loader.exec_module(command)


# ------------------------------------------------------------------ resolver against call_claude

def _anthropic_wire_of_call_claude(monkeypatch, tmp_path, config, **call_arguments) -> dict:
    """Keyword arguments that reach ``client.messages.create`` when ``call_claude`` runs with the
    real ``_call_with_retry`` above a fake client."""
    seen = {}

    class FakeMessages:
        def create(self, **keyword_arguments):
            seen.update(keyword_arguments)
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                                   content=[SimpleNamespace(text="ok", type="text")], stop_reason="end_turn")

    monkeypatch.setattr(api, "_call_with_retry", support.REAL_ANTHROPIC_CALL_WITH_RETRY)
    monkeypatch.setattr(api, "_client", SimpleNamespace(messages=FakeMessages()))
    monkeypatch.setattr(api, "_cost_log_path", tmp_path / "cost.jsonl")
    monkeypatch.setattr(api, "_config", config)
    api.call_claude("hello", system_prompt="system", **call_arguments)
    return seen


CASES = {
    "effort from the stage-tag map": (
        {"model": "claude-haiku-4-5", "effort": {"stage_a": "high", "stage_b": "low"}},
        {"stage": "stage_a"}),
    "stage missing from the map sends no effort": (
        {"model": "claude-haiku-4-5", "effort": {"stage_a": "high"}}, {"stage": "stage_z"}),
    "one effort for every stage": ({"model": "claude-haiku-4-5", "effort": "medium"}, {"stage": "anything"}),
    "per-model default effort": ({"model": "claude-opus-5-5"}, {"stage": "stage_a"}),
    "stage effort beats the per-model default": (
        {"model": "claude-opus-5-5", "effort": {"stage_a": "max"}}, {"stage": "stage_a"}),
    "headroom added for a thinking model, configured value": (
        {"model": "claude-sonnet-5-5", "max_tokens": 4000, "thinking_headroom": 5000}, {}),
    "headroom added for a thinking model, default value": (
        {"model": "claude-sonnet-5-5", "max_tokens": 4000}, {"max_tokens": 6000}),
    "no headroom for a model that does not reason on every call": (
        {"model": "claude-haiku-4-5", "max_tokens": 4000, "thinking_headroom": 5000}, {}),
    "temperature dropped for a model that rejects it": (
        {"model": "claude-sonnet-5", "temperature": 0.7}, {}),
    "temperature kept otherwise": ({"model": "claude-haiku-4-5", "temperature": 0.7}, {}),
    "temperature argument beats the config": ({"model": "claude-haiku-4-5", "temperature": 0.7}, {"temperature": 0.2}),
    "model argument beats the config": ({"model": "claude-haiku-4-5"}, {"model": "claude-opus-4-6"}),
    "schema sent as structured output": (
        {"model": "claude-haiku-4-5"}, {"json_schema": {"type": "object"}}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_resolved_settings_equal_what_call_claude_sends(name, monkeypatch, tmp_path):
    config, call_arguments = CASES[name]
    config = {"outputs": {"cost_log": "unused"}, **config}
    arguments = dict(call_arguments)
    wire = _anthropic_wire_of_call_claude(monkeypatch, tmp_path, config, **arguments)
    settings = api.resolve_request_settings(
        arguments.get("model"), arguments.get("stage"), arguments.get("max_tokens"),
        arguments.get("temperature"), json_schema=arguments.get("json_schema"), config=config)
    assert settings.provider == "anthropic"
    assert settings.model == wire["model"]
    assert settings.max_tokens == wire["max_tokens"]
    assert settings.temperature == wire.get("temperature")      # None = omitted
    assert settings.effort == (wire.get("output_config") or {}).get("effort")
    assert settings.thinking == wire["thinking"]
    assert settings.structured_output == ("format" in (wire.get("output_config") or {}))


def test_the_resolved_values_of_the_named_cases_are_the_expected_ones():
    resolve = api.resolve_request_settings
    haiku = {"model": "claude-haiku-4-5", "max_tokens": 4000, "temperature": 0.7,
             "effort": {"stage_a": "high"}}
    assert resolve(None, "stage_a", None, None, config=haiku).effort == "high"
    assert resolve(None, "stage_z", None, None, config=haiku).effort is None
    assert resolve("claude-opus-5-5", "stage_z", None, None, config=haiku).effort == "low"
    thinking = resolve("claude-sonnet-5-5", None, 6000, None, config={"thinking_headroom": 1000})
    assert (thinking.max_tokens, thinking.thinking_headroom_added) == (7000, 1000)
    assert resolve("claude-sonnet-5", None, None, None, config=haiku).temperature is None
    assert resolve(None, None, None, None, config=haiku).temperature == 0.7


def test_openai_route_resolution_equals_what_call_claude_hands_to_call_openai(monkeypatch, tmp_path):
    config = {"outputs": {"cost_log": "unused"}, "max_tokens": 4000, "thinking_headroom": 3000,
              "effort": {"stage_a": "low"}}
    seen = []
    monkeypatch.setattr(openai_api, "call_openai", lambda *args, **keywords: seen.append(keywords) or ("t", "end_turn"))
    monkeypatch.setattr(api, "_config", config)
    api.call_claude("hello", model="gpt-6.1-sol", stage="stage_a")
    api.call_claude("hello", model="gpt-6.1-sol", stage="stage_z")
    for keywords, stage in zip(seen, ("stage_a", "stage_z")):
        settings = api.resolve_request_settings("gpt-6.1-sol", stage, None, None, config=config)
        assert settings.provider == "openai"
        assert (settings.max_tokens, settings.effort) == (keywords["max_output_tokens"], keywords["reasoning_effort"])
    assert [k["reasoning_effort"] for k in seen] == ["low", openai_api.DEFAULT_REASONING_EFFORT]


# ------------------------------------------------------------------ the command against the recordings

STAGE_TAGS = {
    "description_rewrite": "layer2_description_rewrite", "plan": "layer2_plan", "consideration_scoring": "layer3_consideration_scoring",
    "draft": "layer4_draft", "critique": "layer4_critique", "revise": "layer4_revise", "score": "layer5_score"}
SCENARIOS_AND_STAGE_TAGS = [
    # Claude-routed description rewrite and a screen that keeps sampling parameters
    ("temperature_kept_on_every_stage_with_a_two_call_revision", {**STAGE_TAGS, "screen": "layer1_screen"}),
    # OpenAI-routed description rewrite, adaptive-thinking draft and rewrite models, no screen
    ("minimal_config_with_openai_rewrite_and_adaptive_thinking_models", dict(STAGE_TAGS)),
]


@pytest.fixture(scope="module", params=SCENARIOS_AND_STAGE_TAGS, ids=[name for name, _ in SCENARIOS_AND_STAGE_TAGS])
def scenario_run(request, tmp_path_factory):
    name, stage_tags = request.param
    run_directory, config = command._run_fixture_scenario_in_temporary_folder(
        name, tmp_path_factory.mktemp("scenario_run"))
    return name, stage_tags, run_directory, config


def _recorded_request(scenario_name: str, stage_tag: str) -> dict:
    recorded = support.load_recorded_scenario(scenario_name)
    return next(record for record in recorded["requests"] if record["stage"] == stage_tag)


def _snapshot(directory: Path) -> dict:
    return {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in directory.rglob("*") if path.is_file()}


def _document_id_for(run_directory: Path, stage: str, recorded: dict) -> str:
    """The recorded item id; for the screen (recorded by batch id) the first hand of that batch."""
    if stage != "screen":
        return recorded["item_id"]
    batches = command.utils.load_jsonl(
        command.utils.sdf_stage_dir(run_directory, 1) / command.deal_and_screen_layer.BATCHES_FILENAME)
    return next(batch["prompt_ids"][0] for batch in batches if batch["batch_id"] == recorded["item_id"])


def _wire_settings_and_user_message(recorded: dict) -> tuple:
    """(model, max tokens, temperature, effort, thinking, user message marker) of a recorded request."""
    wire = recorded["wire"]
    if recorded["route"] == "openai":
        return (wire["model"], wire["max_output_tokens"], wire["temperature"], wire["reasoning"]["effort"],
                None, wire["input"])
    return (wire["model"], wire["max_tokens"], wire.get("temperature"), (wire.get("output_config") or {}).get("effort"),
            wire["thinking"], wire["messages"][0]["content"])


def test_printed_settings_equal_the_recorded_wire_settings_and_nothing_is_written(scenario_run):
    name, stage_tags, run_directory, config = scenario_run
    before = _snapshot(run_directory)
    for stage, stage_tag in stage_tags.items():
        recorded = _recorded_request(name, stage_tag)
        full_request = command.build_full_request(
            run_directory, config, stage, _document_id_for(run_directory, stage, recorded))
        settings = full_request.settings
        model, max_tokens, temperature, effort, thinking, user_message_marker = _wire_settings_and_user_message(recorded)
        assert full_request.request.stage == stage_tag
        assert full_request.request.item_id == recorded["item_id"], stage   # the screen: the batch id
        assert (settings.model, settings.max_tokens, settings.temperature, settings.effort, settings.thinking) == (
            model, max_tokens, temperature, effort, thinking), stage
        assert settings.provider == recorded["route"], stage
        assert support.hashed_text_marker(full_request.request.user_message) == user_message_marker, stage
    assert _snapshot(run_directory) == before


def test_command_line_prints_the_full_request_as_json(scenario_run, capsys):
    name, _, run_directory, config = scenario_run
    recorded = _recorded_request(name, "layer4_draft")
    manifest_free_config = run_directory / "config_for_the_command.json"
    # a run folder given on the command line: the config comes from --config (a YAML file; JSON is YAML)
    manifest_free_config.write_text(json.dumps(config))
    exit_code = command.main(["--run-folder", str(run_directory), "--config", str(manifest_free_config),
                              "--stage", "draft", "--document-id", recorded["item_id"], "--output-format", "json"])
    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed["stage"] == "layer4_draft" and printed["document_id"] == recorded["item_id"]
    assert printed["settings"]["max_tokens"] == recorded["wire"]["max_tokens"]
    assert printed["system_prompt"] and printed["user_message"]


def test_stages_without_a_step_are_refused_with_the_reason(scenario_run, capsys):
    _, _, run_directory, config = scenario_run
    with pytest.raises(command.RequestCannotBeBuilt, match="makes no model call"):
        command.build_full_request(run_directory, config, "score_gate", "matrix_000000")
    with pytest.raises(command.RequestCannotBeBuilt, match="no record"):
        command.build_full_request(run_directory, config, "draft", "no_such_document")


def test_screen_request_is_that_of_the_batch_holding_the_hand_and_needs_a_round_only_when_ambiguous(scenario_run):
    name, stage_tags, run_directory, config = scenario_run
    if "screen" not in stage_tags:
        return   # only the scenario with a screen has a batch to look up
    recorded = _recorded_request(name, "layer1_screen")
    hand_id = _document_id_for(run_directory, "screen", recorded)
    assert command.build_full_request(run_directory, config, "screen", hand_id, round_number=0).request.item_id \
        == recorded["item_id"]
    with pytest.raises(command.RequestCannotBeBuilt, match="no planned screen batch"):
        command.build_full_request(run_directory, config, "screen", hand_id, round_number=9)
    with pytest.raises(command.RequestCannotBeBuilt, match="no planned screen batch"):
        command.build_full_request(run_directory, config, "screen", "no_such_hand")
