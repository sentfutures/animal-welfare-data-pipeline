#!/usr/bin/env python3
"""Print the full request an SDF stage would send for one document, without sending it.

Read-only: it calls no model, reads no API key and writes nothing into a run folder. The system
prompt and user message come from the stage's own ``build_request`` (the converted steps: plan,
draft, critique, revise, combined, score); the settings come from ``shared.api.resolve_request_settings``,
the very function ``shared.api.call_claude`` calls before it sends, so what is printed is what
would be sent. Which documents a stage sees is decided as ``sdf_pipeline/run.py`` decides it on a
resume (stage files read from the run folder, rewritten descriptions and saved selections applied
without calls).

Choose where the run comes from, exactly one of:

    --run-folder outputs/sdf/runs/<run id>
        An existing run. The settings are resolved against the config recorded in the run's
        ``run_manifest.json`` (override it with ``--config <file>``). A run with no manifest config
        needs ``--config``.
    --fixture-scenario <name>
        One of the SDF prompt-equivalence test's scenarios (``tests/sdf_prompt_equivalence_scenarios.py``).
        The scenario is first run end to end with canned replies in a temporary folder (no model
        API is called, fake keys), which is deleted when the command ends; nothing else is written.

Then the stage and the document:

    python scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py \\
        --fixture-scenario main_run_with_removals_failures_and_resume --stage draft --document-id matrix_000000

``--output-format json`` prints one JSON object (full texts, settings, request hash). Use
``--maximum-characters-per-text N`` to cut long prompts for reading.

Stages it cannot cover yet: the implausibility screen, the description rewrite, the consideration
scoring (layer1_select) and the gate and near-duplicate cull. The first three are not steps on this
branch (each builds its request inside its own layer and sends it itself), so there is no
``build_request`` to call; the gate and the cull make no model call.
"""

import argparse
import contextlib
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from shared import api, utils  # noqa: E402
from sdf_pipeline import (  # noqa: E402
    layer1_description_rewrite, layer1_plan, layer1_select_considerations, layer2_draft, layer3_rewrite,
    layer4_score)
from sdf_pipeline.update_steps import ModelRequest  # noqa: E402

STAGES_THIS_COMMAND_COVERS = ("plan", "draft", "critique", "revise", "combined", "score")
STAGES_THIS_COMMAND_CANNOT_COVER_YET = {
    "screen": "not a step on this branch: layer1_screen builds and sends its batch request itself",
    "description_rewrite": "not a step on this branch: layer1_description_rewrite builds and sends its own request (OpenAI or Claude route)",
    "select": "not a step on this branch: layer1_select_considerations builds and sends its own scoring request",
    "gate": "makes no model call",
    "near_duplicate_cull": "makes no model call",
}


class RequestCannotBeBuilt(Exception):
    """The chosen stage and document do not give a request (unknown id, step does not apply, ...)."""


@dataclasses.dataclass(frozen=True)
class FullRequest:
    request: ModelRequest
    settings: api.ResolvedRequestSettings


def _prompts_directory_of_run(run_directory: Path) -> Path:
    snapshot = run_directory / "inputs" / "prompts"
    return snapshot if snapshot.is_dir() else REPOSITORY_ROOT / "prompts" / "sdf"


def config_of_run(run_directory: Path, config_path: Path | None) -> dict:
    if config_path is not None:
        return utils.load_config(str(config_path))
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if not manifest.get("config"):
        raise RequestCannotBeBuilt(f"{run_directory} has no config in run_manifest.json; pass --config")
    return manifest["config"]


def _record_with_id(records: list, key: str, document_id: str, file_description: str) -> dict:
    for record in records:
        if record.get(key) == document_id:
            return record
    raise RequestCannotBeBuilt(f"no record with {key} {document_id!r} in {file_description}")


def build_full_request(run_directory: Path, config: dict, stage: str, document_id: str) -> FullRequest:
    """The request ``stage`` would send for ``document_id`` in this run, and its resolved settings.
    Reads stage files only."""
    run_directory = Path(run_directory)
    prompts_directory = _prompts_directory_of_run(run_directory)
    layer_directory = {number: utils.sdf_stage_dir(run_directory, number) for number in (1, 2, 3, 4)}
    if stage == "plan":
        step = layer1_plan.PlanStep(config)
        document = _record_with_id(utils.load_jsonl(layer_directory[1] / "prompts.jsonl"), "prompt_id",
                                   document_id, f"{layer_directory[1] / 'prompts.jsonl'}")
    elif stage == "draft":
        step = layer2_draft.DraftStep(config, prompts_directory)
        plans = layer1_description_rewrite.load_rewritten_plans(
            config, layer_directory[1], utils.load_jsonl(layer_directory[1] / "plans.jsonl"))
        plans = layer1_select_considerations.attach_selections(config, prompts_directory, layer_directory[1], plans)
        document = _record_with_id(plans, "prompt_id", document_id, "the plans after the description rewrite")
    elif stage in ("critique", "revise", "combined"):
        drafts = utils.load_jsonl(layer_directory[2] / "drafts.jsonl")
        document = _record_with_id(drafts, "doc_id", document_id, f"{layer_directory[2] / 'drafts.jsonl'}")
        if stage == "critique":
            step = layer3_rewrite.CritiqueStep(config, prompts_directory)
        elif stage == "combined":
            step = layer3_rewrite.CombinedReviewAndReviseStep(config, prompts_directory)
        else:
            critique_by_document_id = {
                record["doc_id"]: record["review"]
                for record in utils.load_jsonl(layer_directory[3] / "critiques.jsonl")}
            if document_id not in critique_by_document_id:
                raise RequestCannotBeBuilt(
                    f"the revise request needs the critique of {document_id!r}, and "
                    f"{layer_directory[3] / 'critiques.jsonl'} has none")
            step = layer3_rewrite.ReviseStep(config, prompts_directory, critique_by_document_id)
    elif stage == "score":
        step = layer4_score.ScoreStep(config, prompts_directory)
        document = _record_with_id(utils.load_jsonl(layer_directory[3] / "rewrites.jsonl"), "doc_id",
                                   document_id, f"{layer_directory[3] / 'rewrites.jsonl'}")
    elif stage in STAGES_THIS_COMMAND_CANNOT_COVER_YET:
        raise RequestCannotBeBuilt(f"stage {stage!r} cannot be shown yet: {STAGES_THIS_COMMAND_CANNOT_COVER_YET[stage]}")
    else:
        raise RequestCannotBeBuilt(f"unknown stage {stage!r}; choose from {', '.join(STAGES_THIS_COMMAND_COVERS)}")
    if not step.applies_to(document):
        raise RequestCannotBeBuilt(f"the {stage} step does not apply to {document_id!r} (it sends no request for it)")
    request = step.build_request(document)
    settings = api.resolve_request_settings(
        request.model, request.stage, request.max_tokens, request.temperature,
        json_schema=request.json_schema, cache_system=request.cache_system, config=config)
    return FullRequest(request=request, settings=settings)


def _cut(text: str, maximum_characters: int | None) -> str:
    if maximum_characters is None or len(text) <= maximum_characters:
        return text
    return f"{text[:maximum_characters]}\n[... {len(text) - maximum_characters} more characters cut ...]"


def describe_full_request_as_text(full_request: FullRequest, maximum_characters_per_text: int | None = None) -> str:
    request, settings = full_request.request, full_request.settings
    settings_lines = "\n".join(f"  {name}: {json.dumps(value)}" for name, value in dataclasses.asdict(settings).items())
    return (
        f"stage tag: {request.stage}\ndocument id: {request.item_id}\nrequest hash: {request.request_hash()}\n"
        f"\n=== resolved settings (shared.api.resolve_request_settings) ===\n{settings_lines}\n"
        f"\n=== system prompt ({len(request.system_prompt)} characters) ===\n"
        f"{_cut(request.system_prompt, maximum_characters_per_text)}\n"
        f"\n=== user message ({len(request.user_message)} characters) ===\n"
        f"{_cut(request.user_message, maximum_characters_per_text)}\n")


def describe_full_request_as_json(full_request: FullRequest, maximum_characters_per_text: int | None = None) -> str:
    request = full_request.request
    return json.dumps({
        "stage": request.stage, "document_id": request.item_id, "request_hash": request.request_hash(),
        "settings": dataclasses.asdict(full_request.settings),
        "system_prompt": _cut(request.system_prompt, maximum_characters_per_text),
        "user_message": _cut(request.user_message, maximum_characters_per_text),
    }, indent=2, ensure_ascii=False)


def _run_fixture_scenario_in_temporary_folder(scenario_name: str, working_directory: Path):
    """Runs the equivalence scenario with canned replies; returns (run directory, config of its last invocation)."""
    sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))
    import sdf_prompt_equivalence_fixture_support as support  # noqa: E402  (captures the real network seams first)
    import sdf_prompt_equivalence_scenarios as scenarios  # noqa: E402
    with contextlib.redirect_stdout(sys.stderr):   # the scenario's pipeline progress is not part of the answer
        outcome = support.run_scenario_in_isolation(scenarios.scenario_named(scenario_name), working_directory)
    return outcome.run_directory, outcome.configs_used[-1]


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-folder", type=Path)
    source.add_argument("--fixture-scenario")
    parser.add_argument("--config", type=Path, help="config to resolve settings against (run folder mode; default: the run's manifest)")
    parser.add_argument("--stage", required=True, choices=[*STAGES_THIS_COMMAND_COVERS, *STAGES_THIS_COMMAND_CANNOT_COVER_YET])
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output-format", choices=("text", "json"), default="text")
    parser.add_argument("--maximum-characters-per-text", type=int, default=None)
    parsed = parser.parse_args(arguments)
    describe = describe_full_request_as_json if parsed.output_format == "json" else describe_full_request_as_text
    try:
        if parsed.run_folder is not None:
            config = config_of_run(parsed.run_folder, parsed.config)
            full_request = build_full_request(parsed.run_folder, config, parsed.stage, parsed.document_id)
        else:
            with tempfile.TemporaryDirectory() as temporary_name:
                run_directory, config = _run_fixture_scenario_in_temporary_folder(
                    parsed.fixture_scenario, Path(temporary_name))
                full_request = build_full_request(run_directory, config, parsed.stage, parsed.document_id)
    except RequestCannotBeBuilt as problem:
        print(f"Cannot show a request: {problem}", file=sys.stderr)
        return 2
    print(describe(full_request, parsed.maximum_characters_per_text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
