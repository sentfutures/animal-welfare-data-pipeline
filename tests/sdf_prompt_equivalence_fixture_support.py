"""Support code for the SDF prompt-equivalence test and the final corpus schema test.

The pipeline is run end to end through ``sdf_pipeline/run.py``'s real ``main`` on a small
fixture, with canned replies, and every request that would reach a model provider is recorded
exactly as it would have been sent. ``tests/test_sdf_prompt_equivalence_and_final_corpus.py``
compares those recordings with files committed under ``tests/fixtures/sdf_prompt_equivalence/``,
which were recorded on ``origin/main``; ``scripts/regenerate_sdf_prompt_equivalence_recorded_requests.py``
is the only code that rewrites them.

How requests are recorded
-------------------------
Nothing is replaced between the pipeline layers and the provider client objects except two
classes: ``anthropic.Anthropic`` and ``openai.OpenAI`` are swapped for fakes whose
``messages.create`` / ``responses.create`` record the keyword arguments they are given and return
a canned reply. Everything above them runs for real: ``shared.api.call_claude`` (model resolution,
thinking headroom, reasoning effort from the stage tag, temperature), the real
``shared.api._call_with_retry`` (which adds the ``thinking`` and ``output_config`` fields and drops
the temperature for models that do not accept it), ``shared.openai_api.call_openai`` and the real
``_create_response``. So what is recorded is what reaches the wire, including fields that are
decided inside those functions. The client constructor settings (timeout, retries, default
headers) are recorded too; keys are not (only that a key was supplied).

``call_claude`` and ``call_openai`` are wrapped only to note the stage tag and item id of the
call in a thread-local variable, because those never reach the wire and the canned replies are
chosen by them.

Canned replies are chosen by request, never by queue: the key is (stage tag, item id, attempt),
where the attempt is the number of requests already sent for that stage and item in the scenario
(across a resume, which is a second invocation of ``main`` in the same scenario).
"""

import copy
import hashlib
import inspect
import json
import random
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import yaml

from shared import api, openai_api, utils
from sdf_pipeline import considerations_catalog
import sdf_pipeline.run as sdf_run

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests" / "fixtures" / "sdf_prompt_equivalence"
RECORDED_REQUESTS_PATH = FIXTURE_DIRECTORY / "recorded_model_requests_on_origin_main.json"
RECORDED_FINAL_CORPORA_PATH = FIXTURE_DIRECTORY / "recorded_final_corpora_on_origin_main.json"
RECORDED_FILE_FORMAT_VERSION = 1

# The real network-seam functions, captured when this module is first imported (at test
# collection), before tests/conftest.py's autouse guards replace them with functions that raise.
REAL_ANTHROPIC_CALL_WITH_RETRY = api._call_with_retry
REAL_OPENAI_CREATE_RESPONSE = openai_api._create_response
assert REAL_ANTHROPIC_CALL_WITH_RETRY.__name__ != "_blocked", (
    "tests/sdf_prompt_equivalence_fixture_support.py was imported after the conftest guard "
    "replaced shared.api._call_with_retry; import it at module level of a test file")

STAGE_TAGS_OF_SDF_PIPELINE = (
    "layer1_screen", "layer1_plan", "layer1_description_rewrite", "layer1_select",
    "layer2_draft", "layer3_critique", "layer3_revise", "layer3_rewrite", "layer4_score",
)

# One distinct reasoning effort per stage tag, so a renamed or mistyped tag changes the request.
# (The committed config.yaml sets no effort for the SDF stage tags; the fixture sets them all so
# that the lookup from stage tag to effort is observable.)
REASONING_EFFORT_BY_STAGE_TAG = {
    "layer1_screen": "low",
    "layer1_plan": "medium",
    "layer1_description_rewrite": "high",
    "layer1_select": "xhigh",
    "layer2_draft": "max",
    "layer3_critique": "low",
    "layer3_revise": "medium",
    "layer3_rewrite": "high",
    "layer4_score": "xhigh",
}


# ---------------------------------------------------------------------------------- scenarios

@dataclass(frozen=True)
class Scenario:
    """One fixture run: a config, one or more invocations of ``main`` (a second is a resume),
    and the canned-reply behaviour."""
    name: str
    description: str
    sdf_config_overrides: dict
    invocation_arguments: tuple          # one tuple of extra command-line arguments per invocation
    reply_behaviour: "ReplyBehaviour"
    raise_no_stake_resolution_weight: bool = False


@dataclass(frozen=True)
class ReplyOverride:
    """Replaces the default canned reply for one (stage, item id, attempt)."""
    raise_error_message: str | None = None     # the call throws, as a network or backend failure does
    text: str | None = None                    # a reply with this text
    stop_reason: str = "end_turn"              # "max_tokens" makes the reply a truncated one


@dataclass(frozen=True)
class ReplyBehaviour:
    """What the canned reply is for each stage, item and attempt. Everything not named here
    gets the default reply of its stage (a usable, unremarkable answer)."""
    # {batch id: local numbers (1-based) the screen removes}
    screen_removals: dict = field(default_factory=dict)
    # {item id: "incoherent"} for the plan
    plan_kind_by_item: dict = field(default_factory=dict)
    # {item id: "rewrite" or "conflict"} for the description rewrite
    description_rewrite_kind_by_item: dict = field(default_factory=dict)
    # {item id: {(list name, catalog index): score}}; every other entry scores 1
    consideration_scores_by_item: dict = field(default_factory=dict)
    # {item id: (alignment, realism)}; the default is 9 and 9
    scores_by_item: dict = field(default_factory=dict)
    # {item id: other item id}: the revised text of the first is the revised text of the second
    revised_text_copied_from: dict = field(default_factory=dict)
    # {(stage, item id, attempt): ReplyOverride}
    overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RequestView:
    """What a canned reply may depend on."""
    stage: str
    item_id: str
    attempt: int
    invocation: int
    route: str
    model: str
    user_message: str
    json_schema: dict | None


def stable_pseudo_text(key: str, word_count: int = 70) -> str:
    """Deterministic text that is unlike the text made from any other key (so the
    near-duplicate cull sees two documents as duplicates only when the key is the same)."""
    words = []
    counter = 0
    while len(words) < word_count:
        digest = hashlib.sha256(f"{key}|{counter}".encode()).hexdigest()
        words.extend(f"term{int(digest[start:start + 6], 16)}" for start in range(0, 60, 6))
        counter += 1
    return f"Text for {key}. " + " ".join(words[:word_count]) + "."


def _consideration_catalog():
    return considerations_catalog.load_considerations(REPOSITORY_ROOT / "prompts" / "sdf")


def canned_reply(behaviour: ReplyBehaviour, request: RequestView) -> tuple[str, str]:
    """(reply text, stop reason) for one request; raises for a simulated failed call."""
    override = behaviour.overrides.get((request.stage, request.item_id, request.attempt))
    if override is not None:
        if override.raise_error_message is not None:
            raise RuntimeError(override.raise_error_message)
        return override.text, override.stop_reason
    item = request.item_id
    stage = request.stage
    if stage == "layer1_screen":
        entry_numbers = request.json_schema["properties"]["specifications"]["items"]["properties"]["number"]["enum"]
        removed = set(behaviour.screen_removals.get(item, ()))
        return json.dumps({"specifications": [
            {"number": number, "reasoning": f"Considered hand {number}.",
             "remove": number in removed,
             "reason": "No realistic document could satisfy every dealt value." if number in removed else ""}
            for number in entry_numbers]}), "end_turn"
    if stage == "layer1_plan":
        if behaviour.plan_kind_by_item.get(item) == "incoherent":
            return (f"<document_planning>Working notes for {item}: the values cannot fit together."
                    f"</document_planning>\n<document_description>INCOHERENT: the dealt values contradict "
                    f"each other.</document_description>"), "end_turn"
        return (f"<document_planning>Working notes for {item}.</document_planning>\n"
                f"<document_description>Planned description for {item}. "
                f"{stable_pseudo_text('plan ' + item, 30)}</document_description>"), "end_turn"
    if stage == "layer1_description_rewrite":
        kind = behaviour.description_rewrite_kind_by_item.get(item)
        check_names = request.json_schema["properties"]["changes"]["items"]["properties"]["check"]["enum"]
        answer = {"reasoning": f"Checked the description of {item}.", "rewrite_needed": False,
                  "rewritten_description": "", "changes": [], "dealt_value_conflict": False,
                  "conflict_reason": ""}
        if kind == "rewrite":
            answer.update(rewrite_needed=True,
                          rewritten_description=f"Rewritten description for {item}. "
                                                f"{stable_pseudo_text('rewrite ' + item, 30)}",
                          changes=[{"check": check_names[0], "reason": f"A detail of {item} was implausible."}])
        elif kind == "conflict":
            answer.update(dealt_value_conflict=True,
                          conflict_reason=f"The only plausible fix for {item} would change a dealt value.")
        return json.dumps(answer), "end_turn"
    if stage == "layer1_select":
        catalog = _consideration_catalog()
        favoured = behaviour.consideration_scores_by_item.get(item, {})
        answer = {}
        for list_name, list_schema in request.json_schema["properties"].items():
            entries_of_list = catalog.entries_of(list_name)
            favoured_labels = {entries_of_list[index].label: score
                               for (favoured_list, index), score in favoured.items() if favoured_list == list_name}
            answer[list_name] = [
                {"label": label, "score": favoured_labels.get(label, 1),
                 "where_it_arises": f"It arises in {item}." if favoured_labels.get(label, 1) >= 3 else ""}
                for label in list_schema["items"]["properties"]["label"]["enum"]]
        return json.dumps(answer), "end_turn"
    if stage == "layer2_draft":
        return f"<document>{stable_pseudo_text('draft ' + item)}</document>", "end_turn"
    if stage == "layer3_critique":
        return f"<review>Problems found in the draft of {item}: the opening is generic.</review>", "end_turn"
    if stage in ("layer3_revise", "layer3_rewrite"):
        text_key = behaviour.revised_text_copied_from.get(item, item)
        return (f"Review notes for {item}.\n<improved_document>{stable_pseudo_text('revised ' + text_key)}"
                f"</improved_document>"), "end_turn"
    if stage == "layer4_score":
        alignment, realism = behaviour.scores_by_item.get(item, (9, 9))
        return json.dumps({"alignment": alignment, "realism": realism, "spec_conformance": 8,
                           "notes": f"Scored {item}."}), "end_turn"
    raise AssertionError(f"no canned reply for stage {stage!r} (item {item!r})")


# ---------------------------------------------------------------------------------- recording

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_copy(value):
    return json.loads(json.dumps(value))


class ModelRequestRecorder:
    """Collects every request that reaches a fake provider client, and answers it."""

    def __init__(self, behaviour: ReplyBehaviour, maximum_random_reply_delay_seconds: float = 0.0):
        self.behaviour = behaviour
        # a random pause before each reply shuffles which worker thread finishes first
        self.maximum_random_reply_delay_seconds = maximum_random_reply_delay_seconds
        self._delay_random = random.Random()
        self.records: list[dict] = []
        self.client_constructions: list[dict] = []
        self.current_invocation = 0
        self._attempts_sent: dict[tuple, int] = {}
        self._lock = threading.Lock()
        self.call_context = threading.local()

    def start_invocation(self, invocation_number: int) -> None:
        self.current_invocation = invocation_number

    def _normalise_system(self, system) -> dict:
        """The system prompt as a hash and a length plus whatever else rides with it: for the Anthropic route, the block structure and each block's
        cache_control."""
        if isinstance(system, str):
            return {"text_sha256": _sha256(system), "character_count": len(system), "blocks": None}
        blocks = []
        for block in system:
            block_copy = _json_copy(block)
            text = block_copy.pop("text")
            block_copy["text_sha256"] = _sha256(text)
            block_copy["character_count"] = len(text)
            blocks.append(block_copy)
        return {"text_sha256": None, "character_count": None, "blocks": blocks}

    def handle_request(self, route: str, wire_keyword_arguments: dict) -> tuple[str, str]:
        """Record one request exactly as it reaches the wire; return (reply text, stop reason)."""
        context = getattr(self.call_context, "stack", None)
        if not context:
            raise AssertionError(
                "a model request reached the provider client outside api.call_claude / "
                "openai_api.call_openai, so it has no stage tag and item id to record it under")
        stage, item_id = context[-1]
        wire = _json_copy({key: value for key, value in wire_keyword_arguments.items()
                           if key not in ("system", "instructions")})
        if route == "anthropic":
            wire["system"] = self._normalise_system(wire_keyword_arguments["system"])
            user_message = wire_keyword_arguments["messages"][0]["content"]
            schema = (wire.get("output_config") or {}).get("format", {}).get("schema")
        else:
            wire["instructions"] = self._normalise_system(wire_keyword_arguments.get("instructions", ""))
            user_message = wire_keyword_arguments["input"]
            schema = ((wire.get("text") or {}).get("format") or {}).get("schema")
        with self._lock:
            key = (stage, item_id)
            attempt = self._attempts_sent.get(key, 0) + 1
            self._attempts_sent[key] = attempt
            record = {"invocation": self.current_invocation, "stage": stage, "item_id": item_id,
                      "attempt": attempt, "route": route, "wire": wire}
            self.records.append(record)
        view = RequestView(stage=stage, item_id=item_id, attempt=attempt, invocation=self.current_invocation,
                           route=route, model=wire["model"], user_message=user_message, json_schema=schema)
        if self.maximum_random_reply_delay_seconds:
            time.sleep(self._delay_random.random() * self.maximum_random_reply_delay_seconds)
        return canned_reply(self.behaviour, view)

    def sorted_records(self) -> list[dict]:
        return sorted(self.records, key=lambda r: (r["stage"], r["item_id"], r["attempt"]))


def _build_fake_anthropic_class(recorder: ModelRequestRecorder):
    class FakeAnthropicMessages:
        def create(self, **keyword_arguments):
            text, stop_reason = recorder.handle_request("anthropic", keyword_arguments)
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1000, output_tokens=max(1, len(text) // 4)),
                content=[SimpleNamespace(type="text", text=text)],
                stop_reason=stop_reason)

    class FakeAnthropic:
        def __init__(self, **constructor_keyword_arguments):
            settings = {key: value for key, value in constructor_keyword_arguments.items() if key != "api_key"}
            settings["api_key_supplied"] = bool(constructor_keyword_arguments.get("api_key"))
            recorder.client_constructions.append({"route": "anthropic", "settings": _json_copy(settings)})
            self.messages = FakeAnthropicMessages()

    return FakeAnthropic


def _build_fake_openai_class(recorder: ModelRequestRecorder):
    class FakeOpenAIResponses:
        def create(self, **keyword_arguments):
            text, stop_reason = recorder.handle_request("openai", keyword_arguments)
            truncated = stop_reason != "end_turn"
            return SimpleNamespace(
                status="incomplete" if truncated else "completed",
                incomplete_details=SimpleNamespace(reason=stop_reason) if truncated else None,
                output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])],
                usage=SimpleNamespace(
                    input_tokens=1000, output_tokens=max(1, len(text) // 4),
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens_details=SimpleNamespace(reasoning_tokens=100)))

    class FakeOpenAI:
        def __init__(self, **constructor_keyword_arguments):
            settings = {key: value for key, value in constructor_keyword_arguments.items() if key != "api_key"}
            settings["api_key_supplied"] = bool(constructor_keyword_arguments.get("api_key"))
            recorder.client_constructions.append({"route": "openai", "settings": _json_copy(settings)})
            self.responses = FakeOpenAIResponses()

    return FakeOpenAI


def _wrap_to_note_stage_and_item(recorder: ModelRequestRecorder, real_function):
    """The same function, but it first notes the call's stage tag and item id for the recorder."""
    signature = inspect.signature(real_function)

    def wrapper(*positional, **keyword):
        bound = signature.bind(*positional, **keyword)
        stage, item_id = bound.arguments.get("stage"), bound.arguments.get("item_id")
        stack = recorder.call_context.__dict__.setdefault("stack", [])
        stack.append((stage, item_id))
        try:
            return real_function(*positional, **keyword)
        finally:
            stack.pop()

    wrapper.__name__ = real_function.__name__
    return wrapper


def install_request_recording(monkeypatch, recorder: ModelRequestRecorder) -> None:
    """Route every provider request of this process into ``recorder`` (see the module docstring)."""
    monkeypatch.setattr(api.anthropic, "Anthropic", _build_fake_anthropic_class(recorder))
    monkeypatch.setattr(openai_api.openai, "OpenAI", _build_fake_openai_class(recorder))
    monkeypatch.setattr(api, "_call_with_retry", REAL_ANTHROPIC_CALL_WITH_RETRY)
    monkeypatch.setattr(openai_api, "_create_response", REAL_OPENAI_CREATE_RESPONSE)
    monkeypatch.setattr(api, "_client", None)
    monkeypatch.setattr(openai_api, "_client", None)
    monkeypatch.setattr(api, "call_claude", _wrap_to_note_stage_and_item(recorder, api.call_claude))
    monkeypatch.setattr(openai_api, "call_openai", _wrap_to_note_stage_and_item(recorder, openai_api.call_openai))


# ---------------------------------------------------------------------------------- running

def build_fixture_config(tmp_path: Path, sdf_config_overrides: dict) -> dict:
    sdf = {
        "n_prompts": 10,
        "seed": 0,
        "entity_pool_seed": 137,
        "min_score_threshold": 7,
        "near_dup_threshold": 0.90,
        "screen_enabled": True,
        "screen_batch_size": 5,
        "screen_redeal_rounds": 3,
        "description_rewrite_enabled": True,
        "description_rewrite_model": "gpt-6.1-sol",
        "description_rewrite_reasoning_effort": "low",
        "select_enabled": True,
        "max_scoring_attempts": 3,
        "minimum_assignable_score": 3,
        "max_moral_considerations_per_document": 3,
        "max_epistemic_considerations_per_document": 3,
        "max_considerations_per_document": 4,
        "score_weights": {3: 1, 4: 3, 5: 9},
        "extra_consideration_penalty": 2,
        "select_skip_no_stake": True,
        "revision": {"split": True, "style_mode": "axes", "use_length": True},
        # one model per stage family, chosen to exercise every way the request depends on the model:
        # sampling accepted (haiku) or dropped (sonnet 5), adaptive thinking with headroom (sonnet 5.5,
        # opus 5.5), and the fallbacks (screen to plan model, critique to rewrite model)
        "plan_model": "claude-haiku-4-5",
        "select_model": "claude-sonnet-5",
        "draft_model": "claude-sonnet-5-5",
        "rewrite_model": "claude-opus-5-5",
        "score_model": "claude-haiku-4-5",
    }
    sdf.update(sdf_config_overrides)
    return {
        "backend": "api",
        "model": "claude-sonnet-5",
        "max_tokens": 4000,
        "temperature": 0.7,
        "workers": 4,
        "thinking_headroom": 5000,
        "effort": dict(REASONING_EFFORT_BY_STAGE_TAG),
        "sdf": sdf,
        "language_distribution": {"en": 1.0},
        "outputs": {"cost_log": str(tmp_path / "unused_cost_log.jsonl")},
    }


NO_STAKE_WEIGHT_LINE = "0.00 :: [no_stake]"
CREATIVE_SOLUTION_WEIGHT_LINE = "0.12 :: [creative_solution]"


def raise_no_stake_resolution_weight_in_run_snapshot(run_directory: Path) -> None:
    """The committed variables.txt deals the no-welfare-stake resolution at weight 0, so no run
    ever contains one. The fixture copies a share of 0.10 to it from the first resolution (the
    column still sums to 1.0) in THIS RUN'S FROZEN SNAPSHOT ONLY (the committed prompt file is not
    touched), which makes the deck sample exactly one such hand in ten. The pipeline reads its
    templates from the snapshot, so everything downstream treats it as a real no-stake hand."""
    variables_path = run_directory / "inputs" / "prompts" / "variables.txt"
    text = variables_path.read_text(encoding="utf-8")
    assert text.count(NO_STAKE_WEIGHT_LINE) == 1 and text.count(CREATIVE_SOLUTION_WEIGHT_LINE) == 1, (
        "prompts/sdf/variables.txt no longer has the lines the fixture edits in its frozen snapshot "
        f"({NO_STAKE_WEIGHT_LINE!r} and {CREATIVE_SOLUTION_WEIGHT_LINE!r}); update "
        "raise_no_stake_resolution_weight_in_run_snapshot in tests/sdf_prompt_equivalence_fixture_support.py")
    text = text.replace(NO_STAKE_WEIGHT_LINE, "0.10 :: [no_stake]").replace(
        CREATIVE_SOLUTION_WEIGHT_LINE, "0.02 :: [creative_solution]")
    variables_path.write_text(text, encoding="utf-8")


@dataclass
class ScenarioOutcome:
    scenario: Scenario
    recorder: ModelRequestRecorder
    run_directory: Path
    final_corpus_lines_after_each_invocation: list[list[str]]
    exit_messages_by_invocation: list


def run_scenario(scenario: Scenario, working_directory: Path, monkeypatch,
                 maximum_random_reply_delay_seconds: float = 0.0) -> ScenarioOutcome:
    """Run the scenario's invocations of ``sdf_pipeline/run.py``'s ``main`` with every provider
    request recorded and answered by canned replies. Output goes under ``working_directory``."""
    config_path = working_directory / "fixture_config.yaml"
    config_path.write_text(yaml.safe_dump(build_fixture_config(working_directory, scenario.sdf_config_overrides)))
    outputs_root = working_directory / "outputs"
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(outputs_root))

    recorder = ModelRequestRecorder(scenario.reply_behaviour, maximum_random_reply_delay_seconds)
    install_request_recording(monkeypatch, recorder)

    if scenario.raise_no_stake_resolution_weight:
        real_create_run_dir = utils.create_run_dir

        def create_run_dir_then_edit_snapshot(*arguments, **keyword_arguments):
            run_directory = real_create_run_dir(*arguments, **keyword_arguments)
            raise_no_stake_resolution_weight_in_run_snapshot(run_directory)
            return run_directory

        monkeypatch.setattr(utils, "create_run_dir", create_run_dir_then_edit_snapshot)

    final_corpus_lines_after_each_invocation = []
    exit_messages = []
    run_directory = None
    for invocation_number, extra_arguments in enumerate(scenario.invocation_arguments, 1):
        recorder.start_invocation(invocation_number)
        arguments = ["run.py", "--config", str(config_path), "--label", "prompt-equivalence", *extra_arguments]
        if "--resume" in extra_arguments:
            arguments += ["--run-id", run_directory.name]
        monkeypatch.setattr(sys, "argv", arguments)
        try:
            sdf_run.main()
            exit_messages.append(None)
        except SystemExit as stopped:
            exit_messages.append(str(stopped.code))
        run_directory = _only_run_directory(outputs_root)
        final_path = run_directory / "final" / "sdf_corpus.jsonl"
        final_corpus_lines_after_each_invocation.append(
            final_path.read_text(encoding="utf-8").splitlines() if final_path.exists() else [])
    return ScenarioOutcome(scenario, recorder, run_directory, final_corpus_lines_after_each_invocation,
                           exit_messages)


def _only_run_directory(outputs_root: Path) -> Path:
    run_directories = [path for path in (outputs_root / "sdf" / "runs").iterdir() if path.is_dir()]
    assert len(run_directories) == 1, f"expected exactly one run directory, found {run_directories}"
    return run_directories[0]


# ---------------------------------------------------------------------------------- serialising

def requests_record_for_outcome(outcome: ScenarioOutcome) -> dict:
    records = outcome.recorder.sorted_records()
    calls_per_stage_per_invocation: dict = {}
    for record in records:
        per_stage = calls_per_stage_per_invocation.setdefault(str(record["invocation"]), {})
        per_stage[record["stage"]] = per_stage.get(record["stage"], 0) + 1
    return {
        "description": outcome.scenario.description,
        "client_constructions": _distinct(outcome.recorder.client_constructions),
        "calls_per_stage_per_invocation": {
            invocation: dict(sorted(counts.items())) for invocation, counts in
            sorted(calls_per_stage_per_invocation.items())},
        "requests": records,
    }


def _distinct(constructions: list[dict]) -> list[dict]:
    seen, distinct = set(), []
    for construction in constructions:
        marker = json.dumps(construction, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            distinct.append(construction)
    return sorted(distinct, key=lambda c: json.dumps(c, sort_keys=True))


def final_corpora_record_for_outcome(outcome: ScenarioOutcome) -> dict:
    return {str(number): [json.loads(line) for line in lines]
            for number, lines in enumerate(outcome.final_corpus_lines_after_each_invocation, 1)}


def write_recorded_files(outcomes: list[ScenarioOutcome], recorded_from_description: str) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    requests = {"format_version": RECORDED_FILE_FORMAT_VERSION, "recorded_from": recorded_from_description,
                "scenarios": {o.scenario.name: requests_record_for_outcome(o) for o in outcomes}}
    corpora = {"format_version": RECORDED_FILE_FORMAT_VERSION, "recorded_from": recorded_from_description,
               "scenarios": {o.scenario.name: final_corpora_record_for_outcome(o) for o in outcomes}}
    for path, content in ((RECORDED_REQUESTS_PATH, requests), (RECORDED_FINAL_CORPORA_PATH, corpora)):
        path.write_text(json.dumps(content, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_recorded_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------- comparing

def describe_difference_of_texts(recorded: str, current: str) -> str:
    """Where two strings first differ, with some context from each."""
    limit = min(len(recorded), len(current))
    position = next((i for i in range(limit) if recorded[i] != current[i]), limit)
    line_number = recorded.count("\n", 0, position) + 1
    start = max(0, position - 60)
    return (f"first difference at character {position} (line {line_number}); recorded "
            f"{len(recorded)} characters, now {len(current)}; recorded ...{recorded[start:position + 60]!r}... "
            f"now ...{current[start:position + 60]!r}...")


def differences_between_values(recorded, current, path: str) -> list[str]:
    """Every difference between two JSON-like values, as 'field path: recorded X, now Y'.
    Dictionaries are compared key by key (a key present on one side only is a difference),
    lists element by element, and a value of a different type counts as different."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        differences = []
        for key in recorded:
            if key not in current:
                differences.append(f"{path}.{key}: recorded {_short(recorded[key])}, now absent")
            else:
                differences += differences_between_values(recorded[key], current[key], f"{path}.{key}")
        for key in current:
            if key not in recorded:
                differences.append(f"{path}.{key}: recorded absent, now {_short(current[key])}")
        return differences
    if isinstance(recorded, list) and isinstance(current, list):
        differences = []
        if len(recorded) != len(current):
            differences.append(f"{path}: recorded {len(recorded)} elements, now {len(current)}")
        for index, (recorded_item, current_item) in enumerate(zip(recorded, current)):
            differences += differences_between_values(recorded_item, current_item, f"{path}[{index}]")
        return differences
    if type(recorded) is not type(current) or recorded != current:
        if path.endswith("text_sha256"):
            return [f"{path}: the system prompt text changed (sha256 recorded {recorded[:12]}..., now {current[:12]}...)"]
        if isinstance(recorded, str) and isinstance(current, str):
            return [f"{path}: {describe_difference_of_texts(recorded, current)}"]
        return [f"{path}: recorded {_short(recorded)}, now {_short(current)}"]
    return []


def _short(value) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 160 else text[:160] + f"... ({len(text)} characters)"


def request_label(record: dict) -> str:
    return (f"stage {record['stage']!r}, item {record['item_id']!r}, attempt {record['attempt']} "
            f"(invocation {record['invocation']})")


def differences_between_recorded_and_current_requests(
    scenario_name: str, recorded: dict, current: dict,
) -> list[str]:
    """Every way the scenario's requests differ from those recorded on origin/main, each naming the
    stage, the item id and the field. Empty when the requests are identical."""
    messages = []
    recorded_by_key = {(r["stage"], r["item_id"], r["attempt"]): r for r in recorded["requests"]}
    current_by_key = {(r["stage"], r["item_id"], r["attempt"]): r for r in current["requests"]}
    for key in sorted(recorded_by_key.keys() - current_by_key.keys()):
        messages.append(f"[{scenario_name}] MISSING request, recorded on origin/main but not sent now: "
                        f"{request_label(recorded_by_key[key])}")
    for key in sorted(current_by_key.keys() - recorded_by_key.keys()):
        messages.append(f"[{scenario_name}] EXTRA request, sent now but not recorded on origin/main: "
                        f"{request_label(current_by_key[key])}")
    for key in sorted(recorded_by_key.keys() & current_by_key.keys()):
        recorded_record, current_record = recorded_by_key[key], current_by_key[key]
        for field_name in ("invocation", "route"):
            if recorded_record[field_name] != current_record[field_name]:
                messages.append(
                    f"[{scenario_name}] {request_label(recorded_record)}: field {field_name!r}: recorded "
                    f"{recorded_record[field_name]!r}, now {current_record[field_name]!r}")
        for difference in differences_between_values(recorded_record["wire"], current_record["wire"], "wire"):
            messages.append(f"[{scenario_name}] {request_label(recorded_record)}: field {difference}")
    for difference in differences_between_values(
            recorded["calls_per_stage_per_invocation"], current["calls_per_stage_per_invocation"],
            "calls_per_stage_per_invocation"):
        messages.append(f"[{scenario_name}] number of calls per stage: {difference}")
    for difference in differences_between_values(
            recorded["client_constructions"], current["client_constructions"], "client_constructions"):
        messages.append(f"[{scenario_name}] provider client settings: {difference}")
    return messages


def differences_between_recorded_and_current_final_corpora(
    scenario_name: str, recorded: dict, current: dict,
) -> list[str]:
    """Differences in the fields, the field order and the values of every final corpus record
    after each invocation (a resume gives a second corpus), and in the order of the documents."""
    messages = []
    for invocation in sorted(recorded.keys() | current.keys()):
        recorded_records, current_records = recorded.get(invocation), current.get(invocation)
        prefix = f"[{scenario_name}] final corpus after invocation {invocation}"
        if recorded_records is None or current_records is None:
            messages.append(f"{prefix}: recorded {'absent' if recorded_records is None else 'present'}, "
                            f"now {'absent' if current_records is None else 'present'}")
            continue
        recorded_ids = [r["doc_id"] for r in recorded_records]
        current_ids = [r["doc_id"] for r in current_records]
        for doc_id in recorded_ids:
            if doc_id not in current_ids:
                messages.append(f"{prefix}: document {doc_id!r} was recorded but is no longer in the corpus")
        for doc_id in current_ids:
            if doc_id not in recorded_ids:
                messages.append(f"{prefix}: document {doc_id!r} is in the corpus but was not recorded")
        common_recorded = [d for d in recorded_ids if d in current_ids]
        common_current = [d for d in current_ids if d in recorded_ids]
        if common_recorded != common_current:
            messages.append(f"{prefix}: document order differs: recorded {common_recorded}, now {common_current}")
        current_by_id = {r["doc_id"]: r for r in current_records}
        for recorded_record in recorded_records:
            current_record = current_by_id.get(recorded_record["doc_id"])
            if current_record is None:
                continue
            if list(recorded_record) != list(current_record):
                messages.append(f"{prefix}, document {recorded_record['doc_id']!r}: field order or field names "
                                f"differ: recorded {list(recorded_record)}, now {list(current_record)}")
            for difference in differences_between_values(recorded_record, current_record, "record"):
                messages.append(f"{prefix}, document {recorded_record['doc_id']!r}: {difference}")
    return messages


# ---------------------------------------------------------------------------------- scenarios

# The hands of the main scenario (matrix_NNNNNN is the screen's hand id, which is also the
# prompt_id and doc_id). Ten are dealt; the first screen batch removes one and the second removes
# one, and the redeal makes matrix_000010 and matrix_000011 from their cards.
HAND_PLAIN_WITH_CONSIDERATIONS = "matrix_000000"          # its first consideration-scoring answer is unusable
HAND_INCOHERENT_PLAN = "matrix_000001"
HAND_DEALT_VALUE_CONFLICT = "matrix_000002"
HAND_BELOW_THRESHOLD_SCORE = "matrix_000003"
HAND_NO_WELFARE_STAKE = "matrix_000005"                   # dealt by the edited snapshot, see above
HAND_NEAR_DUPLICATE_OF_PLAIN = "matrix_000006"            # its critique has no review tags
HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME = "matrix_000008"
HAND_TRUNCATED_DRAFT = "matrix_000009"
HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME = "matrix_000010"  # a redealt hand, description rewritten
HAND_SCORE_ANSWER_DOES_NOT_PARSE = "matrix_000011"        # a redealt hand, scored nothing by the selector
SCREEN_REMOVAL_FIRST_BATCH = "screen_round0_batch0000"
SCREEN_REMOVAL_SECOND_BATCH = "screen_round0_batch0001"


def _main_scenario() -> Scenario:
    behaviour = ReplyBehaviour(
        screen_removals={SCREEN_REMOVAL_FIRST_BATCH: [2], SCREEN_REMOVAL_SECOND_BATCH: [3]},
        plan_kind_by_item={HAND_INCOHERENT_PLAN: "incoherent"},
        description_rewrite_kind_by_item={
            HAND_DEALT_VALUE_CONFLICT: "conflict", HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME: "rewrite"},
        consideration_scores_by_item={
            HAND_PLAIN_WITH_CONSIDERATIONS: {("moral", 0): 5, ("epistemic", 2): 5},
            HAND_NEAR_DUPLICATE_OF_PLAIN: {("moral", 1): 5},
            HAND_BELOW_THRESHOLD_SCORE: {("moral", 2): 5, ("epistemic", 3): 5},
            HAND_TRUNCATED_DRAFT: {("moral", 3): 5, ("epistemic", 4): 4},
            # the late hand outscores this one for the same entry, so the assignment taken over all
            # documents after the resume leaves it with nothing, after its draft was written with the entry
            HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME: {("moral", 4): 4},
            HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME: {("moral", 4): 5},
        },
        scores_by_item={HAND_BELOW_THRESHOLD_SCORE: (4, 9)},
        revised_text_copied_from={HAND_NEAR_DUPLICATE_OF_PLAIN: HAND_PLAIN_WITH_CONSIDERATIONS},
        overrides={
            ("layer1_select", HAND_PLAIN_WITH_CONSIDERATIONS, 1): ReplyOverride(text="I could not score these entries."),
            ("layer1_plan", HAND_PLAN_FAILS_THEN_SUCCEEDS_ON_RESUME, 1): ReplyOverride(
                raise_error_message="simulated network failure"),
            ("layer2_draft", HAND_TRUNCATED_DRAFT, 1): ReplyOverride(
                text="<document>A draft that was cut off", stop_reason="max_tokens"),
            ("layer3_critique", HAND_NEAR_DUPLICATE_OF_PLAIN, 1): ReplyOverride(
                text="Problems: the opening is generic and the ending is abrupt. (No review tags.)"),
            ("layer3_revise", HAND_REVISE_FAILS_THEN_SUCCEEDS_ON_RESUME, 1): ReplyOverride(
                raise_error_message="simulated backend failure"),
            ("layer4_score", HAND_SCORE_ANSWER_DOES_NOT_PARSE, 1): ReplyOverride(
                text="I am unable to score this document."),
        },
    )
    return Scenario(
        name="main_run_with_removals_failures_and_resume",
        description="Ten hands, every removal kind, every failure kind, and a resume that retries the failures.",
        sdf_config_overrides={},
        invocation_arguments=((), ("--resume",)),
        reply_behaviour=behaviour,
        raise_no_stake_resolution_weight=True)


def _small_scenario(name: str, description: str, sdf_config_overrides: dict) -> Scenario:
    behaviour = ReplyBehaviour(consideration_scores_by_item={
        "matrix_000000": {("moral", 0): 5, ("epistemic", 2): 5}, "matrix_000001": {("moral", 1): 5}})
    return Scenario(name=name, description=description,
                    sdf_config_overrides={"n_prompts": 3, **sdf_config_overrides},
                    invocation_arguments=((),), reply_behaviour=behaviour)


def all_scenarios() -> list[Scenario]:
    return [
        _main_scenario(),
        _small_scenario("screen_disabled", "sdf.screen_enabled false: the composer deals its own hands.",
                        {"screen_enabled": False}),
        _small_scenario("description_rewrite_disabled", "sdf.description_rewrite_enabled false.",
                        {"description_rewrite_enabled": False}),
        _small_scenario("consideration_selection_disabled", "sdf.select_enabled false: no considerations.",
                        {"select_enabled": False}),
        _small_scenario("revision_in_one_call", "sdf.revision.split false: the single-call layer 3.",
                        {"revision": {"split": False, "style_mode": "axes", "use_length": True}}),
        _small_scenario("description_rewrite_on_a_claude_model",
                        "The description rewrite routed to a Claude model through call_claude.",
                        {"description_rewrite_model": "claude-sonnet-5"}),
    ]


def scenario_named(name: str) -> Scenario:
    return next(scenario for scenario in all_scenarios() if scenario.name == name)
