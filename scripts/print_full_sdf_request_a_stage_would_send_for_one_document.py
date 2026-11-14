#!/usr/bin/env python3
"""Print the full request an SDF stage would send for one document, without sending it.

Read-only (no model, no API key, nothing written to a run folder); the prompts come from the stage's ``build_request`` and the settings from ``shared.api.resolve_request_settings``, so it is what would be sent.
Take the run from ``--run-folder`` or ``--fixture-scenario``, then ``--stage`` and ``--document-id``; not covered: stages with no model call (deal, redeal, selection, gate, near-duplicate removal)."""

import argparse
import contextlib
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from shared import api, sdf_run_layouts, utils  # noqa: E402
from sdf_pipeline.layer1_deal_variables_and_screen import run_layer as deal_variables_and_screen_layer  # noqa: E402
from sdf_pipeline.layer1_deal_variables_and_screen import screen  # noqa: E402
from sdf_pipeline.layer4_draft_critique_and_revise import revise  # noqa: E402
from sdf_pipeline import pipeline_steps  # noqa: E402
from sdf_pipeline.shared_step_machinery.step_record import Run, document_id_of, id_field_of  # noqa: E402
from sdf_pipeline.shared_step_machinery.step_runner import resolve_settings_of_model_request  # noqa: E402
from sdf_pipeline.shared_step_machinery.update_steps import ModelRequest  # noqa: E402

STAGES_THIS_COMMAND_COVERS = tuple(step.name for step in pipeline_steps.PIPELINE_STEPS if step.makes_model_calls)
STAGES_THIS_COMMAND_CANNOT_COVER_YET = {"score_gate": "makes no model call", "near_duplicate_removal": "makes no model call"}
STAGES_REFUSED_WHEN_INACTIVE = ("description_rewrite", "consideration_scoring")   # a disabled stage sends nothing for any document


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


def _record_with_id(records: list, document_id: str, file_description: str) -> dict:
    for record in records:
        if document_id_of(record) == document_id:
            return record
    key = id_field_of(records[0]) if records else "prompt_id"
    raise RequestCannotBeBuilt(f"no record with {key} {document_id!r} in {file_description}")


def _screen_batch(layer_one_directory: Path, hand_id: str, round_number: int | None) -> list:
    """The hands of the batch (as the run planned it, from screen_batches.jsonl) holding the hand, each carrying its ``batch_id``."""
    batch_records = [record for record in utils.load_jsonl(layer_one_directory / deal_variables_and_screen_layer.BATCHES_FILENAME)
                     if hand_id in record["prompt_ids"] and (round_number is None or record["round"] == round_number)]
    if not batch_records:
        raise RequestCannotBeBuilt(
            f"no planned screen batch holds hand {hand_id!r}"
            + (f" in round {round_number}" if round_number is not None else "")
            + f" in {layer_one_directory / deal_variables_and_screen_layer.BATCHES_FILENAME}")
    if len(batch_records) > 1:
        raise RequestCannotBeBuilt(
            f"hand {hand_id!r} is in batches of rounds {[r['round'] for r in batch_records]}; pass --round")
    hands_by_id = {hand["prompt_id"]: hand
                   for hand in utils.load_jsonl(layer_one_directory / deal_variables_and_screen_layer.HANDS_FILENAME)}
    return [{**hands_by_id[prompt_id], "batch_id": batch_records[0]["batch_id"]} for prompt_id in batch_records[0]["prompt_ids"]]


def build_full_request(run_directory: Path, config: dict, stage: str, document_id: str,
                       round_number: int | None = None) -> FullRequest:
    """The request ``stage`` would send for ``document_id`` in this run, and its resolved settings.
    For the screen, ``document_id`` is a hand id and the request is that of the batch holding it in ``round_number``."""
    run_directory = Path(run_directory)
    prompts_directory = _prompts_directory_of_run(run_directory)
    # The run's own folders and, while the request is built, its own stage tags.
    layout = utils.sdf_layout_of_run(run_directory)
    folders = {key: utils.sdf_stage_folder(run_directory, key) for key in sdf_run_layouts.FOLDER_KEYS}
    with sdf_run_layouts.using_layout(layout if layout.tag_index is not None else sdf_run_layouts.FOUR_NUMBERED_LAYERS):
        return _build_full_request_in_layout(config, stage, document_id, round_number, prompts_directory, folders)


def _build_full_request_in_layout(config, stage, document_id, round_number, prompts_directory, folders):
    if stage in STAGES_THIS_COMMAND_CANNOT_COVER_YET:
        raise RequestCannotBeBuilt(f"stage {stage!r} cannot be shown yet: {STAGES_THIS_COMMAND_CANNOT_COVER_YET[stage]}")
    if stage not in STAGES_THIS_COMMAND_COVERS:
        raise RequestCannotBeBuilt(f"unknown stage {stage!r}; choose from {', '.join(STAGES_THIS_COMMAND_COVERS)}")
    step = pipeline_steps.entry_named(stage)
    run = Run(config, prompts_directory, folders)
    if stage == "screen":
        # The one stage whose documents are a batch of hands (from the saved batch plan).
        if screen.load_screen_prompts(prompts_directory) is None:
            raise RequestCannotBeBuilt("this run's prompt snapshot has no screen template")
        request = step.build_request(_screen_batch(folders[sdf_run_layouts.DEAL_VARIABLES_AND_SCREEN], document_id, round_number), run)
        return FullRequest(request=request, settings=resolve_settings_of_model_request(request, config))
    reason_inactive = stage in STAGES_REFUSED_WHEN_INACTIVE and step.reason_inactive(config, prompts_directory)
    if reason_inactive:
        raise RequestCannotBeBuilt(f"the {stage} step does not run in this run: {reason_inactive}")
    document = _record_with_id(step.reads(run), document_id, f"the input documents of the {stage} step")
    if stage == "revise" and document_id not in revise.critique_by_document_id(run):
        raise RequestCannotBeBuilt(f"the revise request needs the critique of {document_id!r}, and "
                                   f"{folders[sdf_run_layouts.CRITIQUE_AND_REVISE] / 'critiques.jsonl'} has none")
    if not step.applies(document, run):
        raise RequestCannotBeBuilt(f"the {stage} step does not apply to {document_id!r} (it sends no request for it)")
    request = step.build_request([document], run)
    return FullRequest(request=request, settings=resolve_settings_of_model_request(request, config))


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
    parser.add_argument("--round", type=int, default=None, dest="round_number",
                        help="screen only: the round whose batch to show (needed when the hand is in several)")
    parser.add_argument("--output-format", choices=("text", "json"), default="text")
    parser.add_argument("--maximum-characters-per-text", type=int, default=None)
    parsed = parser.parse_args(arguments)
    describe = describe_full_request_as_json if parsed.output_format == "json" else describe_full_request_as_text
    try:
        if parsed.run_folder is not None:
            config = config_of_run(parsed.run_folder, parsed.config)
            full_request = build_full_request(parsed.run_folder, config, parsed.stage, parsed.document_id,
                                      parsed.round_number)
        else:
            with tempfile.TemporaryDirectory() as temporary_name:
                run_directory, config = _run_fixture_scenario_in_temporary_folder(
                    parsed.fixture_scenario, Path(temporary_name))
                full_request = build_full_request(run_directory, config, parsed.stage, parsed.document_id,
                                      parsed.round_number)
    except RequestCannotBeBuilt as problem:
        print(f"Cannot show a request: {problem}", file=sys.stderr)
        return 2
    print(describe(full_request, parsed.maximum_characters_per_text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
