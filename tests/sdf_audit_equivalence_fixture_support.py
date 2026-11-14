"""Engine for the SDF audit equivalence test.

Every model-judged SDF audit in ``evals/`` (plus the log-probability persona check's input builder,
which makes no model call) is run with canned judge replies on one run of each on-disk layout, and
every judge request is recorded as it reaches the provider client. ``tests/test_sdf_audit_equivalence.py``
compares the recordings with files committed under ``tests/fixtures/sdf_audit_equivalence/`` (one per
audit and layout), which were recorded on ``origin/main``;
``scripts/regenerate_sdf_audit_equivalence_recorded_requests.py`` is the only code that rewrites them.
The point: when the audits later read corpora through a new loader instead of reading run folders
themselves, every judge request, and which documents are judged in which order, must stay the same.

The three layouts
-----------------
* ``five_layer``: the committed run ``outputs/sdf/runs/2026-07-01_13-49_dev`` (layer1..layer5 folders;
  its corpus records have no ``variables`` or ``description``);
* ``pre_renumber_matrix``: the committed run ``outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5``,
  the run the website pins (layer12/layer3/layer4/layer5 folders, matrix hands in ``variables``);
* ``current``: no run in this layout is committed, so one is generated inside the test by running
  ``sdf_pipeline/run.py`` on the SDF prompt-equivalence fixture (scenario ``revision_in_one_call``,
  canned replies, see ``tests/sdf_prompt_equivalence_fixture_support.py``).
Each run is copied into a temporary folder first, with its corpus cut to the first
``DOCUMENTS_KEPT_FROM_EACH_RUN`` documents, so no audit writes into the repository and the runtime
stays small. The audits are then pointed at the copy and told to judge a few documents.

How requests are recorded
-------------------------
As in the prompt-equivalence test, only the provider clients (``anthropic.Anthropic`` and
``openai.OpenAI``) are replaced, by fakes whose ``messages.create`` / ``responses.create`` record the
keyword arguments they receive and return a canned reply; the real ``shared.api.call_claude`` /
``_call_with_retry`` and ``shared.openai_api.call_openai`` / ``_create_response`` run above them.
(These fakes are a minimal copy of the ones in ``sdf_prompt_equivalence_fixture_support.py``, kept
separate so that file can change without touching this one.) The one embedding call of the diversity
audit's ``--ideas`` pass is recorded at ``shared.embeddings.embed_texts``, the embeddings seam.

What is recorded for a request: the system prompt and the user message as sha256 and length (the
test prints the new text when one of them changes), every other wire field in plain text (model, max
tokens, reasoning effort, temperature, caching flags, schema), the stage tag and item id of the
call, and which of the run's documents appear in the user message. Requests are listed in the
order they were sent; every audit is run with one worker so that order is deterministic.

Canned replies depend only on the audit and the request's schema, never on the audit's own code:
a structured judge gets the smallest answer that satisfies its schema; the other audits get a fixed
answer in the shape they parse.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import yaml

import sdf_prompt_equivalence_fixture_support as sdf_support   # imported first: it captures the real network seams
import sdf_prompt_equivalence_scenarios as sdf_scenarios
from shared import api, embeddings, openai_api

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests" / "fixtures" / "sdf_audit_equivalence"
RECORDED_FILE_FORMAT_VERSION = 1
REGENERATION_COMMAND = "python scripts/regenerate_sdf_audit_equivalence_recorded_requests.py"
REGENERATION_RULE = (
    f"The recorded files are rewritten only by `{REGENERATION_COMMAND}`, and only when the pull request is "
    "meant to change judge requests: never to make a failing test pass. Regenerate and commit the files in that pull "
    "request, and say in its description that requests change and why. A pull request not meant to change "
    "them leaves the files alone, so a difference is a bug.")

# The real network-seam functions, captured when this module is first imported (at test collection),
# before tests/conftest.py's autouse guards replace them with functions that raise.
REAL_ANTHROPIC_CALL_WITH_RETRY = api._call_with_retry
REAL_OPENAI_CREATE_RESPONSE = openai_api._create_response
assert REAL_ANTHROPIC_CALL_WITH_RETRY.__name__ != "_blocked", (
    "tests/sdf_audit_equivalence_fixture_support.py was imported after the conftest guard replaced "
    "shared.api._call_with_retry; import it at module level of a test file")

DOCUMENTS_KEPT_FROM_EACH_RUN = 12
DOCUMENTS_JUDGED_BY_EACH_AUDIT = 3
CURRENT_LAYOUT_SCENARIO_NAME = "revision_in_one_call"
EXCERPT_CHARACTER_COUNT_FOR_FINDING_DOCUMENTS = 80
NEW_TEXT_CHARACTERS_PRINTED_IN_FAILURE_MESSAGE = 2500


@dataclass(frozen=True)
class RunLayout:
    name: str
    committed_run_directory: Path | None     # None: generated inside the test


RUN_LAYOUTS = (
    RunLayout("five_layer", REPOSITORY_ROOT / "outputs" / "sdf" / "runs" / "2026-07-01_13-49_dev"),
    RunLayout("pre_renumber_matrix",
              REPOSITORY_ROOT / "outputs" / "sdf" / "runs" / "2026-07-25_15-57_fullscale-500-opus5"),
    RunLayout("current", None),
)


def layout_named(name: str) -> RunLayout:
    return next(layout for layout in RUN_LAYOUTS if layout.name == name)


# ----------------------------------------------------------------------------------------- audits

@dataclass(frozen=True)
class AuditInvocation:
    """What an audit is given: where its input run is, a configuration file, a scratch folder, the
    ids of the documents it should judge (the first few of the corpus), and the corpus path."""
    run_directory: Path
    configuration_path: Path
    scratch_directory: Path
    document_ids: list
    corpus_path: Path


@dataclass(frozen=True)
class Audit:
    name: str
    module_name: str                                      # imported by name, so the repository's evals/ package is used
    command_line_arguments: Callable[[AuditInvocation], list]
    takes_argument_list: bool = False                     # main(argv) (True) or main() reading sys.argv (False)
    embeds_texts: bool = False
    makes_model_calls: bool = True


def _judged_ids(invocation):
    return invocation.document_ids[:DOCUMENTS_JUDGED_BY_EACH_AUDIT]


AUDITS = (
    Audit("plausibility_audit", "evals.plausibility_audit_sdf", takes_argument_list=True,
          command_line_arguments=lambda i: [
              "--input", str(i.run_directory), "--run-name", "audit-equivalence",
              "--runs-directory", str(i.scratch_directory / "plausibility_runs"),
              "--document-ids", *_judged_ids(i), "--workers", "1"]),
    Audit("shoehorning_audit", "evals.shoehorning_audit_sdf", takes_argument_list=True,
          command_line_arguments=lambda i: [
              "--input", str(i.run_directory), "--run-name", "audit-equivalence",
              "--runs-directory", str(i.scratch_directory / "shoehorning_runs"),
              "--document-ids", *_judged_ids(i), "--workers", "1"]),
    Audit("variable_persistence", "evals.variable_persistence_sdf",
          command_line_arguments=lambda i: [
              "--input", str(i.run_directory), "--config", str(i.configuration_path),
              "--output-dir", str(i.scratch_directory / "variable_persistence_output"),
              "--doc-ids", *_judged_ids(i)]),
    Audit("constitutional_compliance", "evals.compliance_sdf",
          command_line_arguments=lambda i: [
              "--input", str(i.run_directory), "--config", str(i.configuration_path),
              "--sample", str(DOCUMENTS_JUDGED_BY_EACH_AUDIT)]),
    Audit("corpus_scoring", "evals.score_sdf",
          command_line_arguments=lambda i: [
              "--input", str(i.corpus_path), "--config", str(i.configuration_path),
              "--output", str(i.scratch_directory / "corpus_scores.jsonl"),
              "--limit", str(DOCUMENTS_JUDGED_BY_EACH_AUDIT)]),
    Audit("corpus_pattern_and_principle_scan", "evals.audit_sdf",
          command_line_arguments=lambda i: [
              "--input", str(i.run_directory), "--config", str(i.configuration_path),
              "--limit", "6", "--patterns", "--pattern-sample", "4", "--batch-size", "2",
              "--prevalence-sample", "3", "--principles", "--principle-sample", "3"]),
    Audit("semantic_diversity_idea_summaries", "evals.diversity", embeds_texts=True,
          command_line_arguments=lambda i: [
              "--input", str(i.run_directory), "--config", str(i.configuration_path),
              "--limit", str(DOCUMENTS_JUDGED_BY_EACH_AUDIT), "--ideas", "--no-cache",
              "--embed-model", "audit-equivalence-embedding-model", "--vendi-boot", "0"]),
    Audit("log_probability_persona_check_input_builder",
          "evals.log_probability_persona_check.build_scoring_inputs", takes_argument_list=True,
          makes_model_calls=False,
          command_line_arguments=lambda i: [
              "--persona-set", "preachiness", "--input", str(i.run_directory),
              "--run-folder-label", "audit-equivalence",
              "--maximum-documents", str(DOCUMENTS_JUDGED_BY_EACH_AUDIT)]),
)


def audit_named(name: str) -> Audit:
    return next(audit for audit in AUDITS if audit.name == name)


def case_name(audit_name: str, layout_name: str) -> str:
    return f"{audit_name}_on_{layout_name}_run"


def recorded_path_for_case(audit_name: str, layout_name: str) -> Path:
    return FIXTURE_DIRECTORY / f"recorded_{case_name(audit_name, layout_name)}_on_origin_main.json"


# ------------------------------------------------------------------------------------ canned replies

def smallest_instance_of_schema(schema):
    """The smallest value that satisfies a strict JSON schema: the first enum value, empty arrays,
    false, the schema's minimum or 1, and a fixed sentence for text."""
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return smallest_instance_of_schema(next(option for option in schema["anyOf"] if option.get("type") != "null"))
    kind = schema.get("type")
    if kind == "object":
        return {name: smallest_instance_of_schema(sub) for name, sub in schema.get("properties", {}).items()}
    if kind == "array":
        label_schema = schema["items"].get("properties", {}).get("label", {})
        if "enum" in label_schema:   # a list with one entry per label (the variable persistence considerations)
            return [{**smallest_instance_of_schema(schema["items"]), "label": label} for label in label_schema["enum"]]
        return [smallest_instance_of_schema(schema["items"]) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        return "Canned judge reply."
    if kind in ("integer", "number"):
        return schema.get("minimum", 1)
    if kind == "boolean":
        return False
    return None


def canned_reply(stage: str, user_message: str, json_schema) -> tuple[str, str]:
    """(reply text, stop reason). Depends only on the stage tag, the schema and, for the corpus
    pattern and principle scan (one stage, four kinds of prompt), the first words of the prompt."""
    if json_schema is not None:
        return json.dumps(smallest_instance_of_schema(json_schema)), "end_turn"
    if stage == "eval_compliance":
        return json.dumps({"modes": [{"mode": number, "verdict": "absent", "evidence": "", "note": ""}
                                     for number in range(1, 30)], "overall": "Canned overall."}), "end_turn"
    if stage == "eval_score_sdf":
        return json.dumps({"alignment": 8, "realism": 8, "diversity": 7, "notes": "Canned notes."}), "end_turn"
    if stage == "eval_audit_sdf":
        if "RAW PATTERNS:" in user_message:        # consolidation of the batch reports
            return json.dumps([{"pattern": "Canned pattern", "kind": "structural", "description": "Canned.",
                                "strict_check": "Canned check.", "is_defect": False}]), "end_turn"
        if "PATTERNS:" in user_message and "DOCUMENT:" in user_message:   # prevalence of one document
            return "[0]", "end_turn"
        if "PRINCIPLES" in user_message.upper() and "DOCUMENT" in user_message.upper():   # principle coverage
            return "[1]", "end_turn"
        return json.dumps([{"pattern": "Canned raw pattern", "kind": "structural", "description": "Canned.",
                            "prevalence": "most"}]), "end_turn"
    if stage == "eval_diversity_ideas":
        return "A canned one-line summary of the scenario.", "end_turn"
    raise AssertionError(f"no canned reply for stage {stage!r}")


# ------------------------------------------------------------------------------------------ recording

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_copy(value):
    return json.loads(json.dumps(value))


def _describe_text(text: str) -> dict:
    return {"sha256": _sha256(text), "character_count": len(text)}


class JudgeRequestRecorder:
    """Collects every request that reaches a fake provider client, and answers it."""

    def __init__(self):
        self.requests: list[dict] = []
        self.embedding_requests: list[dict] = []
        self.client_constructions: list[dict] = []
        self.call_context = threading.local()
        self._lock = threading.Lock()

    def handle_request(self, route: str, wire_keyword_arguments: dict) -> tuple[str, str]:
        stack = getattr(self.call_context, "stack", None)
        if not stack:
            raise AssertionError("a model request reached the provider client outside api.call_claude / "
                                 "openai_api.call_openai, so it has no stage tag to record it under")
        stage, item_id = stack[-1]
        settings = _json_copy({key: value for key, value in wire_keyword_arguments.items()
                               if key not in ("system", "instructions", "messages", "input")})
        if route == "anthropic":
            system = wire_keyword_arguments["system"]
            user_message = wire_keyword_arguments["messages"][0]["content"]
            schema = (settings.get("output_config") or {}).get("format", {}).get("schema")
            if (settings.get("output_config") or {}).get("format", {}).get("schema") is not None:
                settings["output_config"]["format"]["schema"] = "<see schema>"
        else:
            system = wire_keyword_arguments.get("instructions", "")
            user_message = wire_keyword_arguments["input"]
            schema = ((settings.get("text") or {}).get("format") or {}).get("schema")
            if schema is not None:
                settings["text"]["format"]["schema"] = "<see schema>"
        if isinstance(system, str):
            system_record = {**_describe_text(system), "blocks": None}
            system_text = system
        else:
            blocks = []
            for block in system:
                block_copy = _json_copy(block)
                text = block_copy.pop("text")
                blocks.append({**block_copy, **_describe_text(text)})
            system_record = {"sha256": None, "character_count": None, "blocks": blocks}
            system_text = "\n".join(block["text"] for block in system)
        record = {"stage": stage, "item_id": item_id, "route": route,
                  "system_prompt": system_record, "user_message": _describe_text(user_message),
                  "settings": settings, "schema": schema,
                  # not committed: the failure message prints new text from these
                  "system_prompt_text": system_text, "user_message_text": user_message}
        with self._lock:
            self.requests.append(record)
        return canned_reply(stage, user_message, schema)


def _fake_embed_texts_recording_to(recorder: JudgeRequestRecorder):
    def fake_embed_texts(texts, model=embeddings.DEFAULT_MODEL):
        recorder.embedding_requests.append({
            "model": model, "texts": [_describe_text(text) for text in texts]})
        rows = []
        for text in texts:
            vector = np.random.default_rng(zlib.crc32(text.encode("utf-8"))).standard_normal(8).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)
    return fake_embed_texts


def _install_recording(monkeypatch, recorder: JudgeRequestRecorder) -> None:
    monkeypatch.setattr(api.anthropic, "Anthropic", sdf_support.build_fake_anthropic_class(recorder))
    monkeypatch.setattr(openai_api.openai, "OpenAI", sdf_support.build_fake_openai_class(recorder))
    monkeypatch.setattr(api, "_call_with_retry", REAL_ANTHROPIC_CALL_WITH_RETRY)
    monkeypatch.setattr(openai_api, "_create_response", REAL_OPENAI_CREATE_RESPONSE)
    monkeypatch.setattr(api, "call_claude", sdf_support.wrap_to_note_stage_and_item(recorder, api.call_claude))
    monkeypatch.setattr(openai_api, "call_openai", sdf_support.wrap_to_note_stage_and_item(recorder, openai_api.call_openai))
    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts_recording_to(recorder))


# ------------------------------------------------------------------------------------- run folders

def generate_current_layout_run(working_directory: Path) -> Path:
    """A run in the layout today's pipeline writes: ``sdf_pipeline/run.py`` on the prompt-equivalence
    fixture, offline with canned replies. Returns the run folder, inside ``working_directory``."""
    outcome = sdf_support.run_scenario_in_isolation(
        sdf_scenarios.scenario_named(CURRENT_LAYOUT_SCENARIO_NAME), working_directory)
    return outcome.run_directory


def prepare_trimmed_copy_of_run(source_run_directory: Path, destination: Path) -> Path:
    """A real copy of the run with its corpus cut to the first DOCUMENTS_KEPT_FROM_EACH_RUN documents,
    and without the run's own ``audit`` folder, cost logs and score files (an audit's outputs and
    resume state must not come from the repository)."""
    def ignore(directory, names):
        directory = Path(directory)
        skipped = [name for name in names if (directory == source_run_directory and name in ("audit",))
                   or name in ("cost_log.jsonl", "model_calls.jsonl", "sdf_scores.jsonl", "_sdf_score_checkpoint.json",
                               "embeddings_cache.npz", "__pycache__")]
        return skipped
    shutil.copytree(source_run_directory, destination, ignore=ignore)
    corpus_path = destination / "final" / "sdf_corpus.jsonl"
    lines = corpus_path.read_text(encoding="utf-8").splitlines()[:DOCUMENTS_KEPT_FROM_EACH_RUN]
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def fresh_copy_for_one_audit(prepared_run_directory: Path, destination: Path) -> Path:
    """Hard links where the file system allows it (the audit only adds files), else copies."""
    try:
        shutil.copytree(prepared_run_directory, destination, copy_function=os.link)
    except OSError:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(prepared_run_directory, destination)
    return destination


def write_configuration_with_one_worker(scratch_directory: Path) -> Path:
    """config.yaml as committed, except one worker (so requests are sent in a fixed order) and the
    cost log inside the scratch folder."""
    configuration = yaml.safe_load((REPOSITORY_ROOT / "config.yaml").read_text(encoding="utf-8"))
    configuration["workers"] = 1
    configuration.setdefault("outputs", {})["cost_log"] = str(scratch_directory / "cost_log.jsonl")
    path = scratch_directory / "configuration_with_one_worker.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------- running

@dataclass
class AuditOutcome:
    audit_name: str
    layout_name: str
    run_name: str
    corpus_sample_sha256: str
    command_line_arguments_relative: list
    documents_in_corpus_sample: list
    document_ids_the_audit_was_given: list
    ended_with: str
    recorder: JudgeRequestRecorder
    documents_by_id: dict = field(default_factory=dict)
    built_input_files: dict = field(default_factory=dict)


BUILT_INPUT_TEXT_CHARACTER_LIMIT = 200


def _shorten_long_texts(value):
    """Strings longer than the limit become sha256 and length, everything else stays as it is."""
    if isinstance(value, str):
        return _describe_text(value) if len(value) > BUILT_INPUT_TEXT_CHARACTER_LIMIT else value
    if isinstance(value, dict):
        return {key: _shorten_long_texts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shorten_long_texts(item) for item in value]
    return value


def read_built_input_files(run_directory: Path) -> dict:
    """The jsonl files an audit built under ``scoring_inputs/`` (the persona check's input builder),
    record by record, long texts hashed. {path relative to the run: [records]}."""
    built = {}
    for path in sorted((run_directory / "audit").glob("**/scoring_inputs/*.jsonl")):
        relative = path.relative_to(run_directory / "audit")
        built[f"{relative.parts[0].rsplit('_', 2)[0]}/{'/'.join(relative.parts[1:])}"] = [
            _shorten_long_texts(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]
    return built


def _relativise(argument: str, scratch_directory: Path) -> str:
    return argument.replace(str(scratch_directory), "<scratch>") if isinstance(argument, str) else argument


def run_audit_in_isolation(audit: Audit, layout: RunLayout, prepared_run_directory: Path,
                           working_directory: Path) -> AuditOutcome:
    """Run one audit on a fresh copy of a prepared run, recording every request it sends."""
    import importlib
    working_directory.mkdir(parents=True, exist_ok=True)
    run_directory = fresh_copy_for_one_audit(prepared_run_directory, working_directory / "run")
    corpus_path = run_directory / "final" / "sdf_corpus.jsonl"
    corpus_lines = corpus_path.read_text(encoding="utf-8").splitlines()
    documents = [json.loads(line) for line in corpus_lines]
    invocation = AuditInvocation(
        run_directory=run_directory, configuration_path=write_configuration_with_one_worker(working_directory),
        scratch_directory=working_directory, document_ids=[document["doc_id"] for document in documents],
        corpus_path=corpus_path)
    arguments = audit.command_line_arguments(invocation)
    recorder = JudgeRequestRecorder()
    ended_with = "returned"
    with pytest.MonkeyPatch.context() as monkeypatch:
        sdf_support.isolate_from_real_keys_and_global_state(monkeypatch, working_directory)
        _install_recording(monkeypatch, recorder)
        module = importlib.import_module(audit.module_name)
        try:
            if audit.takes_argument_list:
                module.main(arguments)
            else:
                monkeypatch.setattr(sys, "argv", [audit.module_name, *arguments])
                module.main()
        except SystemExit as stopped:
            ended_with = "SystemExit" if stopped.code not in (0, None) else "returned"
        except Exception as error:   # noqa: BLE001 - an audit that cannot read a layout is a recorded outcome
            ended_with = type(error).__name__
    return AuditOutcome(
        audit_name=audit.name, layout_name=layout.name, run_name=(layout.committed_run_directory.name if layout.committed_run_directory is not None
                  else f"generated by scenario {CURRENT_LAYOUT_SCENARIO_NAME} (its folder name holds a timestamp)"),
        corpus_sample_sha256=_sha256("\n".join(corpus_lines)),
        command_line_arguments_relative=[_relativise(argument, working_directory) for argument in arguments],
        documents_in_corpus_sample=[document["doc_id"] for document in documents],
        document_ids_the_audit_was_given=invocation.document_ids[:DOCUMENTS_JUDGED_BY_EACH_AUDIT],
        ended_with=ended_with, recorder=recorder, documents_by_id={d["doc_id"]: d for d in documents},
        built_input_files=read_built_input_files(run_directory))


def _documents_in_message(message: str, documents_by_id: dict) -> list:
    return [doc_id for doc_id, document in documents_by_id.items()
            if (document.get("content") or "")[:EXCERPT_CHARACTER_COUNT_FOR_FINDING_DOCUMENTS]
            and (document.get("content") or "")[:EXCERPT_CHARACTER_COUNT_FOR_FINDING_DOCUMENTS] in message]


def record_for_outcome(outcome: AuditOutcome) -> dict:
    """Everything recorded for one audit on one run, as JSON-able data. Requests keep the new text
    under ``*_text`` keys, which ``write_recorded_file`` removes."""
    schemas: dict = {}
    requests = []
    for sequence_number, request in enumerate(outcome.recorder.requests, 1):
        schema = request["schema"]
        schema_id = None
        if schema is not None:
            schema_id = _sha256(json.dumps(schema, sort_keys=True))[:12]
            schemas[schema_id] = schema
        requests.append({
            "sequence_number": sequence_number, "stage": request["stage"], "item_id": request["item_id"],
            "route": request["route"],
            "documents_in_user_message": _documents_in_message(request["user_message_text"], outcome.documents_by_id),
            "system_prompt": request["system_prompt"], "user_message": request["user_message"],
            "settings": request["settings"], "schema_id": schema_id,
            "system_prompt_text": request["system_prompt_text"], "user_message_text": request["user_message_text"]})
    unique_constructions = sorted({json.dumps(c, sort_keys=True) for c in outcome.recorder.client_constructions})
    return {
        "audit": outcome.audit_name, "layout": outcome.layout_name, "run": outcome.run_name,
        "corpus_sample_sha256": outcome.corpus_sample_sha256,
        "documents_in_corpus_sample": outcome.documents_in_corpus_sample,
        "command_line_arguments": outcome.command_line_arguments_relative,
        "document_ids_the_audit_was_given": outcome.document_ids_the_audit_was_given,
        "ended_with": outcome.ended_with,
        "client_constructions": [json.loads(c) for c in unique_constructions],
        "documents_judged_in_order": [request["item_id"] for request in requests],
        "requests": requests, "schemas": dict(sorted(schemas.items())),
        "embedding_requests": outcome.recorder.embedding_requests,
        "built_input_files": outcome.built_input_files}


def without_new_text(record: dict) -> dict:
    cleaned = _json_copy({key: value for key, value in record.items() if key != "requests"})
    cleaned["requests"] = [{key: value for key, value in request.items() if not key.endswith("_text")}
                           for request in record["requests"]]
    return cleaned


def write_recorded_file(outcome: AuditOutcome, metadata: dict) -> Path:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    content = {"format_version": RECORDED_FILE_FORMAT_VERSION, **metadata,
               **without_new_text(record_for_outcome(outcome))}
    path = recorded_path_for_case(outcome.audit_name, outcome.layout_name)
    path.write_text(json.dumps(content, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_recorded_case(audit_name: str, layout_name: str) -> dict:
    path = recorded_path_for_case(audit_name, layout_name)
    assert path.is_file(), f"no recorded file for {case_name(audit_name, layout_name)} at {path}. {REGENERATION_RULE}"
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_all_runs(base_directory: Path) -> dict:
    """{layout name: trimmed copy of that layout's run}. The current layout's run is generated."""
    prepared = {}
    for layout in RUN_LAYOUTS:
        if layout.committed_run_directory is not None:
            source = layout.committed_run_directory
        else:
            source = generate_current_layout_run(base_directory / "generated_current_layout")
        prepared[layout.name] = prepare_trimmed_copy_of_run(source, base_directory / f"trimmed_{layout.name}" / source.name)
    return prepared


# --------------------------------------------------------------------------------------- comparing

def describe_difference_of_texts(recorded_description: dict, current_text: str) -> str:
    current = _describe_text(current_text)
    return (f"recorded {recorded_description['character_count']} characters (sha256 "
            f"{recorded_description['sha256'][:12]}...), now {current['character_count']} characters (sha256 "
            f"{current['sha256'][:12]}...)")


def _short(value) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 200 else text[:200] + f"... ({len(text)} characters)"


def differences_between_values(recorded, current, path: str) -> list[str]:
    """Every difference between two JSON-like values, key order included (see the prompt version)."""
    return sdf_support.differences_between_values(recorded, current, path, ordered=True)


def _request_label(request: dict) -> str:
    documents = request["documents_in_user_message"] or "no corpus document"
    return (f"request {request['sequence_number']} (stage {request['stage']!r}, item {request['item_id']!r}, "
            f"documents in the user message: {documents})")


def _printed(text: str) -> str:
    if len(text) <= NEW_TEXT_CHARACTERS_PRINTED_IN_FAILURE_MESSAGE:
        return text
    return text[:NEW_TEXT_CHARACTERS_PRINTED_IN_FAILURE_MESSAGE] + f"\n... ({len(text)} characters in all)"


def differences_between_recorded_and_current_requests(recorded: dict, current: dict) -> list[str]:
    """Every way the audit's judge requests on this run differ from those recorded on origin/main,
    each naming the audit, the run, the request, the document and the field. Empty when identical."""
    audit, layout, run = current["audit"], current["layout"], current["run"]
    prefix = f"[audit {audit} | {layout} run {run}]"
    messages = []
    for key in ("run", "corpus_sample_sha256", "documents_in_corpus_sample", "command_line_arguments",
                "document_ids_the_audit_was_given", "ended_with"):
        for difference in differences_between_values(recorded[key], current[key], key):
            messages.append(f"{prefix} THE INPUT OR HOW THE AUDIT ENDED CHANGED: {difference}")
    if recorded["documents_judged_in_order"] != current["documents_judged_in_order"]:
        messages.append(f"{prefix} DOCUMENTS JUDGED, IN ORDER OF REQUEST: recorded "
                        f"{recorded['documents_judged_in_order']}, now {current['documents_judged_in_order']}")
    if len(recorded["requests"]) != len(current["requests"]):
        messages.append(f"{prefix} NUMBER OF JUDGE REQUESTS: recorded {len(recorded['requests'])}, "
                        f"now {len(current['requests'])}")
    for recorded_request, current_request in zip(recorded["requests"], current["requests"]):
        label = f"{prefix} {_request_label(recorded_request)}"
        for field_name in ("stage", "item_id", "route", "documents_in_user_message"):
            if recorded_request[field_name] != current_request[field_name]:
                messages.append(f"{label}: {field_name} changed: recorded {_short(recorded_request[field_name])}, "
                                f"now {_short(current_request[field_name])}")
        recorded_system, current_system = recorded_request["system_prompt"], current_request["system_prompt"]
        if recorded_system["blocks"] is None and current_system["blocks"] is None:
            if recorded_system != current_system:
                messages.append(f"{label}: SYSTEM PROMPT TEXT CHANGED: "
                                f"{describe_difference_of_texts(recorded_system, current_request['system_prompt_text'])}"
                                f"\n--- new system prompt ---\n{_printed(current_request['system_prompt_text'])}\n--- end ---")
        elif recorded_system != current_system:
            messages.append(f"{label}: system prompt blocks changed: "
                            + "; ".join(differences_between_values(recorded_system, current_system, "system_prompt")))
        if recorded_request["user_message"] != current_request["user_message"]:
            messages.append(f"{label}: USER MESSAGE CHANGED: "
                            f"{describe_difference_of_texts(recorded_request['user_message'], current_request['user_message_text'])}"
                            f"\n--- new user message ---\n{_printed(current_request['user_message_text'])}\n--- end ---")
        for difference in differences_between_values(recorded_request["settings"], current_request["settings"], "settings"):
            messages.append(f"{label}: setting {difference}")
        recorded_schema = recorded["schemas"].get(recorded_request["schema_id"]) if recorded_request["schema_id"] else None
        current_schema = current["schemas"].get(current_request["schema_id"]) if current_request["schema_id"] else None
        if (recorded_schema is None) != (current_schema is None):
            messages.append(f"{label}: schema {'removed' if current_schema is None else 'added'}")
        elif recorded_schema is not None:
            for difference in differences_between_values(recorded_schema, current_schema, "schema"):
                messages.append(f"{label}: {difference}")
    for difference in differences_between_values(recorded["client_constructions"], current["client_constructions"],
                                                 "client_constructions"):
        messages.append(f"{prefix} provider client settings: {difference}")
    for difference in differences_between_values(recorded["built_input_files"], current["built_input_files"],
                                                 "built_input_files"):
        messages.append(f"{prefix} built input file (no model call, but what the scorer will be sent): {difference}")
    for difference in differences_between_values(recorded["embedding_requests"], current["embedding_requests"],
                                                 "embedding_requests"):
        messages.append(f"{prefix} embedding request: {difference}")
    return messages
