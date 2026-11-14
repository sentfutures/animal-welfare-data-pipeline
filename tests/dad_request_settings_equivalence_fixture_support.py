"""Engine for the DAD request-settings equivalence check.

Why it exists: ``shared.api.call_claude`` resolves every request's settings (model default, token
cap and thinking headroom, temperature, reasoning effort from the stage tag, provider route)
through ``resolve_request_settings``. The SDF prompt-equivalence test records SDF requests only, and
DAD's step tests stub ``call_claude`` above the resolver, so nothing else checks that a DAD request
reaches the provider client unchanged. This check drives the DAD pipeline through ``dad_pipeline/run.py``'s
real ``main`` (1a scenario plan, 1b draft, 1c gate, 1d refine, 2a scope, 2a.5 select, 2b respond,
step 3 rewrite and the no-system-prompt baseline, with the real templates from the run's snapshot of
``prompts/dad``) on a two-scenario run with canned replies, and records every request at the
provider-client boundary, as ``tests/sdf_prompt_equivalence_fixture_support.py`` does for SDF:

* ``anthropic.Anthropic`` and ``openai.OpenAI`` are replaced by that module's fake clients, which
  record the keyword arguments of ``messages.create`` / ``responses.create`` (model, system and its
  cache flag, messages, max_tokens, temperature, thinking, output_config with effort and schema);
* ``claude_agent_sdk.query`` is replaced by a fake that records the prompt and the
  ``ClaudeAgentOptions`` (model, system prompt, thinking, effort, output format, tools, turns,
  setting sources, permission mode, environment), because that is where the ``claude_code`` and
  ``auto`` backends hand the request to the Claude Code CLI, below the empty-system-prompt
  substitution (``_NEUTRAL_SYSTEM``) and the retry wrapper;
* everything above those seams runs for real, including ``api.init`` for each backend.

Each case is one (configuration, backend) pair: the configurations differ in the models, the
temperature, the effort map and the token settings (see ``CONFIGURATIONS``), and the backends are
``api``, ``claude_code`` and ``auto``. A request is recorded with the stage tag, the ordinal of its
item within the stage (scenario ids are content hashes of the dealt scenario, so the ordinal is the
stable name), the attempt number, and the route that served it (``anthropic``, ``openai`` or
``claude_code``). Some replies are refusals, truncations or unusable text so that the retry
variants run: the 1a refusal fallback model, the 1c second gate attempt, the 1d truncation retry,
the 2a refusal fallback model, and the baseline and step 3 retries at double the token cap.

Texts are stored as sha256 plus character count (the format of the SDF recordings); everything else
is plain text and compared exactly. Nothing here calls a model API or reads a key.
"""

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import yaml

from shared import api, openai_api
import dad_pipeline.run as dad_run
from dad_pipeline import profiles

from sdf_prompt_equivalence_fixture_support import (
    FIXTURE_DIRECTORY as SDF_FIXTURE_DIRECTORY,  # noqa: F401  (documents where the SDF recordings live)
    ModelRequestRecorder, ReplyBehaviour, ReplyOverride, differences_between_values, hashed_text_marker,
    install_fixed_entity_pools, install_request_recording, isolate_from_real_keys_and_global_state,
    json_text_with_short_scalar_containers_on_one_line, recorded_form_of_wire,
    _json_copy, _sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests" / "fixtures" / "dad_request_settings_equivalence"
RECORDED_FILE_FORMAT_VERSION = 1
REGENERATION_COMMAND = "python scripts/regenerate_dad_request_settings_equivalence_recorded_requests.py"
REGENERATION_RULE = (
    f"The recorded files are rewritten only by `{REGENERATION_COMMAND}` run against an unmodified "
    "origin/main checkout, in the same pull request as an intended change to a DAD request; a pull "
    "request that only refactors never rewrites them.")

# The retry wrapper of the claude_code backend, captured at import (test collection) before
# tests/conftest.py's autouse guard replaces it with a function that raises.
REAL_CLAUDE_CODE_CALL_WITH_RETRY = api._call_claude_code_with_retry
assert REAL_CLAUDE_CODE_CALL_WITH_RETRY.__name__ != "_blocked_cc", (
    "tests/dad_request_settings_equivalence_fixture_support.py was imported after the conftest guard "
    "replaced shared.api._call_claude_code_with_retry; import it at module level of a test file")

BACKENDS = ("api", "claude_code", "auto")

DAD_STAGE_TAGS = (
    "scenario_plan", "prompt_draft", "prompt_gate", "prompt_refine", "baseline_response",
    "response_scope", "response_select", "response_draft", "constitution_rewrite",
)

# ---------------------------------------------------------------------------------- configurations

# Frozen copies of the settings that matter, so that a later edit of config.yaml does not move the
# fixture. "real_models_and_effort_map" mirrors the DAD section and the effort map of config.yaml
# at the time of writing (Opus 5.5 everywhere except the Sonnet 5.5 gate, the refusal fallbacks, the
# per-stage effort map, thinking headroom 13000). The other two cover paths config.yaml does not use:
# models that accept a sampling temperature and a non-default config temperature, no effort map and
# a different default token cap; and OpenAI models for the gate and the baseline.
CONFIGURATIONS = {
    "real_models_and_effort_map": {
        "model": "claude-sonnet-5", "max_tokens": 4000, "temperature": 1.0,
        "effort": {"scenario_plan": "high", "prompt_draft": "high", "prompt_gate": "high",
                   "prompt_refine": "high", "baseline_response": "high", "response_scope": "high",
                   "response_select": "high", "response_draft": "high", "constitution_rewrite": "high"},
        "thinking_headroom": 13000,
        "dad_models": {
            "scenario_model": "claude-opus-5-5", "scenario_refusal_fallback_model": "claude-sonnet-5-5",
            "prompt_draft_model": "claude-opus-5-5", "prompt_gate_model": "claude-sonnet-5-5",
            "prompt_refine_model": "claude-opus-5-5", "response_scope_model": "claude-opus-5-5",
            "scope_refusal_fallback_model": "claude-opus-4-8", "response_select_model": "claude-opus-5-5",
            "response_draft_model": "claude-opus-5-5", "constitution_rewrite_model": "claude-opus-5-5"},
        "baseline_model": "claude-opus-5-5",
        "with_retry_variants": True,
    },
    "sampling_models_with_temperature_and_no_effort_map": {
        "model": "claude-sonnet-4-6", "max_tokens": 3000, "temperature": 0.6,
        "effort": None,
        "thinking_headroom": 5000,
        "dad_models": {
            "scenario_model": "claude-sonnet-4-6", "scenario_refusal_fallback_model": "claude-haiku-4-5",
            "prompt_draft_model": "claude-sonnet-4-6", "prompt_gate_model": "claude-haiku-4-5",
            "prompt_refine_model": "claude-sonnet-4-6", "response_scope_model": "claude-sonnet-4-6",
            "scope_refusal_fallback_model": "claude-haiku-4-5", "response_select_model": None,
            "response_draft_model": "claude-opus-4-6", "constitution_rewrite_model": "claude-opus-4-6"},
        "baseline_model": None,
        "with_retry_variants": False,
    },
    "openai_gate_and_baseline_with_global_default_elsewhere": {
        "model": "claude-sonnet-5-5", "max_tokens": 4000, "temperature": 1.0,
        "effort": {"prompt_gate": "low", "baseline_response": "xhigh", "response_draft": "medium"},
        "thinking_headroom": 8000,
        "dad_models": {
            "scenario_model": None, "scenario_refusal_fallback_model": None,
            "prompt_draft_model": None, "prompt_gate_model": "gpt-5.6-luna",
            "prompt_refine_model": None, "response_scope_model": None,
            "scope_refusal_fallback_model": None, "response_select_model": None,
            "response_draft_model": "claude-opus-5-5", "constitution_rewrite_model": None},
        "baseline_model": "gpt-6.1-sol",
        "with_retry_variants": True,
    },
}


def build_dad_fixture_config(configuration_name: str, backend: str) -> dict:
    """The config.yaml dictionary of one case: two scenarios, one worker (so request order and item
    ordinals are deterministic), the gate and refine stages on, the post-run evals off."""
    configuration = CONFIGURATIONS[configuration_name]
    dad = {key: value for key, value in configuration["dad_models"].items() if value is not None}
    dad.update({
        "baseline": {"enabled": True, **({"model": configuration["baseline_model"]}
                                          if configuration["baseline_model"] else {})},
        "evals": {"auto": False},
        "dilemmas": {"count": 2, "batch_size": 2, "id_start": 1, "seed_path": None, "scenario_seed": 7,
                     "gate": True, "refine": True},
        "responses": {"per_prompt": 1},
    })
    config = {"backend": backend, "model": configuration["model"], "max_tokens": configuration["max_tokens"],
              "temperature": configuration["temperature"], "workers": 1,
              "thinking_headroom": configuration["thinking_headroom"], "dad": dad,
              "language_distribution": {"en": 1.0}, "outputs": {"cost_log": "<cost log path>"}}
    if configuration["effort"] is not None:
        config["effort"] = configuration["effort"]
    return config


# ---------------------------------------------------------------------------------- canned replies

def _tag_of(item_id: str) -> str:
    return hashlib.md5(item_id.encode()).hexdigest()[:6]


def default_reply_text(stage: str, item_id: str) -> str:
    """A usable answer for the stage (the same shapes tests/test_e2e_smoke.py uses)."""
    if stage == "scenario_plan":
        return ("<scenario_planning>Considered five situations; chose the second.</scenario_planning>\n"
                f"<scenario_description>A person faces a decision ({_tag_of(item_id)}) where the tempting option "
                "quietly runs its cost through the animals involved.</scenario_description>")
    if stage == "prompt_draft":
        return f"<user_prompt>Drafted user message {_tag_of(item_id)} from the scenario description.</user_prompt>"
    if stage == "prompt_gate":
        return json.dumps({"pass": True, "failures": []})
    if stage == "prompt_refine":
        return f"Notes.\n<revised_user_prompt>Refined user message {_tag_of(item_id)}.</revised_user_prompt>"
    if stage == "baseline_response":
        return f"Plain baseline answer {_tag_of(item_id)}."
    if stage == "response_scope":
        return json.dumps({"patients": "p", "goal": "g", "levers": "l", "cost": "c", "magnitude": "m",
                           "upside": "u", "replaceability": "r"})
    if stage == "response_select":
        return "C1, M1"
    if stage == "response_draft":
        return f"Draft response {_tag_of(item_id)}."
    if stage == "constitution_rewrite":
        return f"Rewritten careful answer {_tag_of(item_id)}."
    raise AssertionError(f"no canned reply for DAD stage {stage!r} (item {item_id!r})")


# Replies that make the retry variants run, keyed by (stage, item ordinal within the stage, attempt).
RETRY_VARIANT_REPLIES = {
    # 1a: two refusals grant a third attempt, served by scenario_refusal_fallback_model
    ("scenario_plan", 0, 1): ReplyOverride(text="", stop_reason="refusal"),
    ("scenario_plan", 0, 2): ReplyOverride(text="", stop_reason="refusal"),
    # 1c: an unparseable verdict is retried with a fresh call
    ("prompt_gate", 0, 1): ReplyOverride(text="the draft looks fine to me"),
    # 1d: a truncated refine is retried
    ("prompt_refine", 1, 1): ReplyOverride(text="Notes that were cut off", stop_reason="max_tokens"),
    # 2a: three refusals grant a fourth attempt, served by scope_refusal_fallback_model
    ("response_scope", 0, 1): ReplyOverride(text="", stop_reason="refusal"),
    ("response_scope", 0, 2): ReplyOverride(text="", stop_reason="refusal"),
    ("response_scope", 0, 3): ReplyOverride(text="", stop_reason="refusal"),
    # baseline and step 3: a truncated answer is retried with the token cap doubled to 8000
    ("baseline_response", 0, 1): ReplyOverride(text="A plain answer that was cut off", stop_reason="max_tokens"),
    ("constitution_rewrite", 1, 1): ReplyOverride(text="A rewrite that was cut off", stop_reason="max_tokens"),
}


class DadRequestRecorder(ModelRequestRecorder):
    """The SDF recorder, answering with DAD's canned replies and also recording requests that reach
    the Claude Code SDK. Records are keyed by (stage, item ordinal within the stage, attempt)."""

    def __init__(self, retry_variant_replies: dict):
        super().__init__(ReplyBehaviour())
        self.retry_variant_replies = retry_variant_replies
        self._item_ids_by_stage: dict[str, list[str]] = {}

    def _next_attempt_and_ordinal(self, stage: str, item_id: str) -> tuple[int, int]:
        with self._lock:
            item_ids = self._item_ids_by_stage.setdefault(stage, [])
            if item_id not in item_ids:
                item_ids.append(item_id)
            ordinal = item_ids.index(item_id)
            key = (stage, item_id)
            attempt = self._attempts_sent.get(key, 0) + 1
            self._attempts_sent[key] = attempt
        return attempt, ordinal

    def sorted_records(self) -> list[dict]:
        return sorted(self.records, key=lambda r: (r["stage"], r["item_ordinal"], r["attempt"]))

    def _reply(self, stage: str, item_id: str, ordinal: int, attempt: int) -> tuple[str, str]:
        override = self.retry_variant_replies.get((stage, ordinal, attempt))
        if override is not None:
            return override.text, override.stop_reason
        return default_reply_text(stage, item_id), "end_turn"

    def _context(self) -> tuple[str, str]:
        stack = getattr(self.call_context, "stack", None)
        if not stack:
            raise AssertionError("a model request reached a provider client outside api.call_claude / "
                                 "openai_api.call_openai, so it has no stage tag to record it under")
        return stack[-1]

    def handle_request(self, route: str, wire_keyword_arguments: dict) -> tuple[str, str]:
        stage, item_id = self._context()
        wire = _json_copy({key: value for key, value in wire_keyword_arguments.items()
                           if key not in ("system", "instructions")})
        if route == "anthropic":
            wire["system"] = self._normalise_system(wire_keyword_arguments["system"])
        else:
            wire["instructions"] = self._normalise_system(wire_keyword_arguments.get("instructions", ""))
        return self._record_and_reply(route, stage, item_id, wire)

    def handle_claude_code_request(self, prompt: str, options) -> tuple[str, str]:
        stage, item_id = self._context()
        system_prompt = options.system_prompt
        wire = {
            "model": options.model,
            "system": self._normalise_system(system_prompt if isinstance(system_prompt, str)
                                              else json.dumps(system_prompt)),
            "messages": [{"role": "user", "content": prompt}],
        }
        for field_name in ("thinking", "effort", "output_format", "tools", "max_turns", "setting_sources",
                           "permission_mode", "env"):
            wire[field_name] = _json_copy(getattr(options, field_name, None))
        return self._record_and_reply("claude_code", stage, item_id, wire)

    def _record_and_reply(self, route: str, stage: str, item_id: str, wire: dict) -> tuple[str, str]:
        attempt, ordinal = self._next_attempt_and_ordinal(stage, item_id)
        with self._lock:
            self.records.append({"stage": stage, "item_ordinal": ordinal, "attempt": attempt, "route": route,
                                 "wire": wire})
        return self._reply(stage, item_id, ordinal, attempt)


def install_claude_code_request_recording(monkeypatch, recorder: DadRequestRecorder) -> None:
    """Route the Claude Code SDK's ``query`` into ``recorder`` and give call_claude the real retry
    wrapper back (tests/conftest.py replaces it with a function that raises)."""
    import claude_agent_sdk
    from claude_agent_sdk import types as sdk_types

    async def fake_query(*, prompt, options):
        text, stop_reason = recorder.handle_claude_code_request(prompt, options)
        yield sdk_types.AssistantMessage(content=[sdk_types.TextBlock(text=text)], model=options.model)
        yield sdk_types.ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1,
            session_id="session_for_request_recording", result=text, stop_reason=stop_reason,
            usage={"input_tokens": 1000, "output_tokens": max(1, len(text) // 4)}, total_cost_usd=0.0)

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    monkeypatch.setattr(api, "_call_claude_code_with_retry", REAL_CLAUDE_CODE_CALL_WITH_RETRY)

    def run_coroutine_without_an_event_loop(function, *arguments):
        """anyio.run for a coroutine that never waits on I/O (the fake query only yields canned messages):
        asyncio's event loop opens a socket pair, which the suite's --disable-socket forbids."""
        coroutine = function(*arguments)
        try:
            coroutine.send(None)
        except StopIteration as finished:
            return finished.value
        coroutine.close()
        raise AssertionError("the fake Claude Code SDK query suspended, so it needs a real event loop")

    import anyio
    monkeypatch.setattr(anyio, "run", run_coroutine_without_an_event_loop)

    def classify_and_fail_loudly(message):
        raise AssertionError(f"the Claude Code SDK path failed while recording requests: {message}")

    # A failure inside the fake SDK path must stop the check, not be retried with tenacity's backoff.
    monkeypatch.setattr(api, "_classify_claude_code_error", classify_and_fail_loudly)


# ---------------------------------------------------------------------------------- running a case

@dataclass
class CaseOutcome:
    configuration_name: str
    backend: str
    config: dict
    records: list = field(default_factory=list)
    client_constructions: list = field(default_factory=list)
    final_example_count: int = 0


def case_name(configuration_name: str, backend: str) -> str:
    return f"{configuration_name}__backend_{backend}"


def all_case_keys() -> list[tuple[str, str]]:
    return [(configuration_name, backend) for configuration_name in CONFIGURATIONS for backend in BACKENDS]


def run_case(configuration_name: str, backend: str, working_directory: Path, monkeypatch) -> CaseOutcome:
    """Run the DAD pipeline's ``main`` on the case's config with every provider client faked."""
    isolate_from_real_keys_and_global_state(monkeypatch, working_directory)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    install_fixed_entity_pools(monkeypatch)
    outputs_root = working_directory / "outputs"
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(outputs_root))

    config = build_dad_fixture_config(configuration_name, backend)
    config_for_file = {**config, "outputs": {"cost_log": str(working_directory / "cost_log.jsonl")}}
    config_path = working_directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_for_file), encoding="utf-8")

    retry_variant_replies = RETRY_VARIANT_REPLIES if CONFIGURATIONS[configuration_name]["with_retry_variants"] else {}
    recorder = DadRequestRecorder(retry_variant_replies)
    install_request_recording(monkeypatch, recorder)
    install_claude_code_request_recording(monkeypatch, recorder)

    profiles.reset()
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path), "--label", "dad_request_settings"])
    try:
        dad_run.main()
    finally:
        profiles.reset()
    corpus_files = list(outputs_root.glob("dad/runs/*/final/dad_corpus.jsonl"))
    example_count = (len(corpus_files[0].read_text(encoding="utf-8").splitlines()) if corpus_files else 0)
    return CaseOutcome(configuration_name, backend, config, recorder.sorted_records(),
                       _distinct(recorder.client_constructions), example_count)


def _distinct(constructions: list[dict]) -> list[dict]:
    distinct = []
    for construction in constructions:
        if construction not in distinct:
            distinct.append(construction)
    return distinct


# ---------------------------------------------------------------------------------- recorded form

def recorded_form_of_claude_code_wire(wire: dict) -> dict:
    recorded_wire = _json_copy(wire)
    recorded_wire["messages"] = [{**message, "content": hashed_text_marker(message["content"])}
                                 for message in recorded_wire["messages"]]
    output_format = recorded_wire.get("output_format")
    if output_format and "schema" in output_format:
        compact = json.dumps(output_format["schema"], ensure_ascii=False)
        if len(compact) > 1500:
            output_format["schema"] = hashed_text_marker(compact)
    return recorded_wire


def recorded_form_of_request(request: dict) -> dict:
    wire = request["wire"]
    recorded_wire = (recorded_form_of_claude_code_wire(wire) if request["route"] == "claude_code"
                     else recorded_form_of_wire(request["route"], wire))
    return {**request, "wire": recorded_wire}


def recorded_form_of_case(outcome: CaseOutcome) -> dict:
    return {
        "configuration": outcome.configuration_name,
        "backend": outcome.backend,
        "config": outcome.config,
        "final_example_count": outcome.final_example_count,
        "client_constructions": outcome.client_constructions,
        "requests": [recorded_form_of_request(request) for request in outcome.records],
    }


def recorded_path_for_configuration(configuration_name: str) -> Path:
    return FIXTURE_DIRECTORY / f"recorded_dad_requests_{configuration_name}_on_origin_main.json"


def write_recorded_file(configuration_name: str, outcomes: list[CaseOutcome], metadata: dict) -> Path:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    content = {"format_version": RECORDED_FILE_FORMAT_VERSION, **metadata,
               "cases": {case_name(o.configuration_name, o.backend): recorded_form_of_case(o) for o in outcomes}}
    path = recorded_path_for_configuration(configuration_name)
    text = json_text_with_short_scalar_containers_on_one_line(content) + "\n"
    assert json.loads(text) == content, "the recorded file's text does not parse back to the recorded content"
    path.write_text(text, encoding="utf-8")
    return path


def load_recorded_case(configuration_name: str, backend: str) -> dict:
    path = recorded_path_for_configuration(configuration_name)
    assert path.is_file(), f"no recorded file for configuration {configuration_name!r} at {path}. {REGENERATION_RULE}"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert recorded.get("format_version") == RECORDED_FILE_FORMAT_VERSION, (
        f"{path.name} has format version {recorded.get('format_version')!r}, this test reads "
        f"{RECORDED_FILE_FORMAT_VERSION}. {REGENERATION_RULE}")
    return recorded["cases"][case_name(configuration_name, backend)]


# ---------------------------------------------------------------------------------- comparing

def request_label(configuration_name: str, backend: str, request: dict) -> str:
    return (f"configuration {configuration_name!r}, backend {backend!r}, stage {request['stage']!r}, "
            f"item {request['item_ordinal']} of the stage, attempt {request['attempt']}")


def differences_between_recorded_and_current_case(recorded: dict, outcome: CaseOutcome) -> list[str]:
    """Every difference between the recorded case and the one just run, each naming the
    configuration, backend, stage, item and attempt, and the field (``wire.max_tokens`` and so on)."""
    current = recorded_form_of_case(outcome)
    current_full = {**current, "requests": [{**request} for request in outcome.records]}
    configuration_name, backend = outcome.configuration_name, outcome.backend
    differences = []
    for key in ("config", "final_example_count", "client_constructions"):
        differences += [f"configuration {configuration_name!r}, backend {backend!r}: {difference}"
                        for difference in differences_between_values(recorded[key], current[key], key)]
    recorded_by_key = {(r["stage"], r["item_ordinal"], r["attempt"]): r for r in recorded["requests"]}
    current_by_key = {(r["stage"], r["item_ordinal"], r["attempt"]): (r, full)
                      for r, full in zip(current["requests"], current_full["requests"])}
    for key in recorded_by_key:
        if key not in current_by_key:
            differences.append(f"{request_label(configuration_name, backend, recorded_by_key[key])}: "
                               "recorded, but no such request is sent now")
    for key, (request, _) in current_by_key.items():
        if key not in recorded_by_key:
            differences.append(f"{request_label(configuration_name, backend, request)}: sent now, "
                               "but not in the recording")
    for key, recorded_request in recorded_by_key.items():
        if key not in current_by_key:
            continue
        request, full_request = current_by_key[key]
        label = request_label(configuration_name, backend, request)
        if recorded_request["route"] != request["route"]:
            differences.append(f"{label}: route recorded {recorded_request['route']!r}, now {request['route']!r}")
            continue
        differences += [f"{label}: {difference}" for difference in differences_between_values(
            recorded_request["wire"], request["wire"], "wire", current_full=full_request["wire"])]
    return differences
