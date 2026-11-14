"""Start a new SDF run from a chosen step of an earlier run, with selected documents.

``seed_run_from_chosen_step`` replaces the scripts that set a run up by writing stage files by hand
and then call ``run.py --resume --layer N``. It

1. checks everything first and refuses, before it creates anything, when a step name is unknown or
   not seedable, a document id is not in the source run, the source run lacks a record the step
   needs for a selected document, or the prompt snapshot it would run with lacks a template the
   step needs;
2. creates a fresh run folder with the pipeline's own ``utils.create_run_dir`` (so the manifest,
   the frozen ``inputs/`` snapshot and the ``latest`` link are made by the code every run uses),
   taking the snapshot from the source run's ``inputs/`` (default) or from the checkout's current
   prompts and constitution (``use_current_prompts``, which is what the layer 3 rerun setup script
   did);
3. copies the source run's own records, unchanged and in the source's order, for the selected
   documents from every stage file written by a step BEFORE the chosen one (written with
   ``utils.save_jsonl``), and nothing written by the chosen step or after it;
4. records where the run was seeded from in ``run_manifest.json`` under ``seeded_from``.

``run.py`` then runs from the step's layer: the step finds its inputs and, because nothing it
produces exists yet, makes its calls.

A ``revision`` override (the voice and length cards the revise call renders) is a field of the draft
record, so it can be set when a run is seeded at the critique or revise step; it is written into the
copied draft records, so the revise call reads it exactly as it reads a draft that carries one.

Steps that cannot be seeded are refused by name: the screen, deal and redeal (a run that screens
hands is made by running the screen), consideration assignment (which solves one assignment over every
scored document, so seeding a subset would change it), and the gate and cull (corpus steps that run
after scoring and make no calls).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from sdf_pipeline import layer1_description_rewrite, layer1_screen, layer1_select_considerations
from sdf_pipeline import pipeline_steps
from sdf_pipeline.document_state import RunLayout
from sdf_pipeline.document_state_loader import UnknownRunLayoutError, detect_run_layout
from shared import utils

MANIFEST_FILE_NAME = "run_manifest.json"
REVISION_OVERRIDE_FOR_EVERY_DOCUMENT = "*"


class SeedingRefused(SystemExit):
    """Seeding was refused; the message says why. Nothing was created."""


# Views of the step catalogue (sdf_pipeline/pipeline_steps.py): the steps that can be seeded, by name, in
# pipeline order, and the reasons the others cannot.
SEEDABLE_STEPS = {entry.name: entry for entry in pipeline_steps.PIPELINE_STEPS
                  if entry.document_file and not entry.reason_not_seedable}
UNSEEDABLE_STEPS = {entry.name: entry.reason_not_seedable for entry in pipeline_steps.PIPELINE_STEPS
                    if entry.reason_not_seedable}


# Every stage file copied when seeding, with the step that writes it, in pipeline order. A file is copied
# when the chosen step comes after the step that writes it. The single-call step writes the file the revise
# does, so each file is listed once, under the first step that writes it.
def _stage_files_in_pipeline_order() -> list:
    listed, seen = [], set()
    for entry in pipeline_steps.PIPELINE_STEPS:
        for stage_file in entry.stage_files:
            if (stage_file.layer, stage_file.file_name) not in seen:
                seen.add((stage_file.layer, stage_file.file_name))
                listed.append((entry.name, stage_file))
    return listed


def _position_of(step_name: str) -> int:
    return pipeline_steps.STEP_NAMES_IN_ORDER.index(step_name)


@dataclass(frozen=True)
class SeededRun:
    run_directory: Path
    start_layer: int
    document_ids: list
    copied_record_counts: dict


def _refuse(message: str):
    raise SeedingRefused(f"Seeding refused: {message}")


def resolve_source_run(seed_from: str, runs_root: Path) -> Path:
    """The source run folder. A value with a path separator is a path. A bare name is a run id under this
    checkout's runs folder; a same-named folder in the working directory is used only when no such run
    exists, and when both exist the request is refused so the wrong one is never seeded from silently."""
    candidate = Path(seed_from)
    is_a_path = any(separator and separator in seed_from for separator in ("/", os.sep, os.altsep))
    if is_a_path:
        if candidate.is_dir():
            return candidate.resolve()
        _refuse(f"no run folder at {candidate}")
    by_id = Path(runs_root) / seed_from
    if by_id.is_dir() and candidate.is_dir():
        _refuse(f"{seed_from!r} is ambiguous: it names both a folder in the working directory ({candidate.resolve()}) "
                f"and a run under the runs folder ({by_id}). Pass the path of the one you mean "
                "(for example ./" + seed_from + " or the full path of the run)")
    if by_id.is_dir():
        return by_id
    if candidate.is_dir():
        return candidate.resolve()
    _refuse(f"no run named {seed_from!r} under {runs_root} and no folder of that name in the working directory")


def load_revision_overrides(path: Path, document_ids: list) -> dict:
    """{document id: revision record} from a JSON file mapping document ids (or "*" for every document
    without its own entry) to revision records (``{"cards": {...}}``)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _refuse(f"the revision override file {path} could not be read as JSON ({error})")
    if not isinstance(raw, dict):
        _refuse(f"the revision override file {path} must be a JSON object mapping document ids (or \"*\") to revision records")
    unknown = sorted(key for key in raw if key != REVISION_OVERRIDE_FOR_EVERY_DOCUMENT and key not in document_ids)
    if unknown:
        _refuse(f"the revision override file names documents that are not selected: {unknown}")
    overrides = {}
    for document_id in document_ids:
        record = raw.get(document_id, raw.get(REVISION_OVERRIDE_FOR_EVERY_DOCUMENT))
        if record is None:
            continue
        if not isinstance(record, dict) or not isinstance(record.get("cards", {}), dict):
            _refuse(f"the revision override for {document_id} must be a record like {{\"cards\": {{...}}}}")
        overrides[document_id] = record
    return overrides


def _records_by_id(stage_directory: Path, file_name: str, id_field: str) -> dict:
    return {record[id_field]: record for record in utils.load_jsonl(stage_directory / file_name) if id_field in record}


def seed_run_from_chosen_step(
    *, config: dict, source_run: Path, from_step: str, document_ids: list | None, runs_root: Path,
    label: str, repository_root: Path, use_current_prompts: bool = False,
    revision_override_path: Path | None = None,
) -> SeededRun:
    """Check, then create the seeded run folder (see the module docstring). Raises ``SeedingRefused``
    before creating anything when the request cannot be met."""
    if from_step in UNSEEDABLE_STEPS:
        _refuse(f"step {from_step!r} cannot be seeded: {UNSEEDABLE_STEPS[from_step]}. "
                f"Seedable steps: {', '.join(SEEDABLE_STEPS)}")
    if from_step not in SEEDABLE_STEPS:
        _refuse(f"unknown step {from_step!r}. Seedable steps: {', '.join(SEEDABLE_STEPS)}")
    step = SEEDABLE_STEPS[from_step]
    if step.seeding_refuses_when_inactive and (reason_inactive := step.reason_inactive(config, None)):
        _refuse(f"step {from_step!r} cannot be seeded: {reason_inactive}. Seedable steps: {', '.join(SEEDABLE_STEPS)}")
    source_run = Path(source_run)
    try:
        layout = detect_run_layout(source_run)
    except UnknownRunLayoutError as error:
        _refuse(str(error))
    if layout is not RunLayout.CURRENT:
        _refuse(f"source run {source_run.name} is in the {layout.name} layout; only runs in the current layout "
                "(layer1 to layer4) can be seeded from")

    # The snapshot the new run will run with.
    if use_current_prompts:
        snapshot_sources = {"prompts": repository_root / "prompts" / "sdf", "constitution": repository_root / "constitution"}
    else:
        snapshot_sources = {"prompts": source_run / "inputs" / "prompts", "constitution": source_run / "inputs" / "constitution"}
        for name, folder in snapshot_sources.items():
            if not folder.is_dir():
                _refuse(f"source run {source_run.name} has no frozen inputs/{name} snapshot, which {from_step} needs to run "
                        "on that run's own templates; pass --use-current-prompts to run on this checkout's prompts instead")
    prompts_dir = snapshot_sources["prompts"]
    missing_templates = [name for name in step.required_prompt_files(config, prompts_dir) if not (prompts_dir / name).is_file()]
    if missing_templates:
        which = "this checkout's prompts" if use_current_prompts else f"source run {source_run.name}'s frozen prompt snapshot"
        _refuse(f"{which} lacks {', '.join(missing_templates)}, which step {from_step} needs")

    stage_directory = {layer: utils.sdf_stage_dir_of_existing_run(source_run, layer) for layer in (1, 2, 3, 4)}
    step_position = _position_of(from_step)
    files_to_copy = [(layer_file.layer, layer_file.file_name, layer_file.id_field)
                     for writer, layer_file in _stage_files_in_pipeline_order() if _position_of(writer) < step_position]

    # Which documents: those named, else every document the step has an input record for.
    input_file = step.document_file(config, prompts_dir)
    input_layer, input_file_name, input_id_field = input_file.layer, input_file.file_name, input_file.id_field
    input_records = _records_by_id(stage_directory[input_layer], input_file_name, input_id_field)
    if not input_records:
        _refuse(f"source run {source_run.name} has no {input_file_name} in layer{input_layer} (or it is empty), "
                f"which step {from_step} reads its documents from")
    if document_ids is None:
        document_ids = list(input_records)
    document_ids = list(dict.fromkeys(document_ids))
    if not document_ids:
        _refuse("no documents selected")

    # An id the source never mentions is unknown; one it mentions but cannot feed this step is missing a record.
    every_known_id = set()
    for _, stage_file in _stage_files_in_pipeline_order():
        every_known_id |= set(_records_by_id(stage_directory[stage_file.layer], stage_file.file_name, stage_file.id_field))
    unknown = [d for d in document_ids if d not in every_known_id]
    if unknown:
        _refuse(f"document ids not found in source run {source_run.name}: {unknown}")

    problems = _required_records_for_step(
        step, config, prompts_dir, stage_directory, document_ids, input_records, f"layer{input_layer}/{input_file_name}")
    if problems:
        _refuse(f"source run {source_run.name} cannot feed step {from_step} for the selected documents:\n  "
                + "\n  ".join(problems))

    revision_overrides = {}
    if revision_override_path is not None:
        if from_step not in ("critique", "revise"):
            _refuse("a revision override sets the voice and length cards the revise call reads, so it can only be given "
                    f"when seeding at critique or revise, not {from_step}")
        revision_overrides = load_revision_overrides(revision_override_path, document_ids)

    assignment_is_solved_again = from_step == "consideration_scoring" and layer1_select_considerations.selection_enabled(config)
    if assignment_is_solved_again:
        print(f"Notice: the consideration assignment will be solved again over the {len(document_ids)} selected "
              "document(s), from freshly scored considerations, and may differ from the source run's. "
              "Rule of thumb: seed at consideration_scoring for a fresh assignment, at draft to keep the source's.")

    # Everything checked: create the run with the pipeline's own function, then copy the records.
    run_directory = utils.create_run_dir(runs_root, label=label, config=config, snapshot_dirs=snapshot_sources)
    selected = set(document_ids)
    copied_record_counts = {}
    for layer, file_name, id_field in files_to_copy:
        records = [r for r in utils.load_jsonl(stage_directory[layer] / file_name) if r.get(id_field) in selected]
        if file_name == "drafts.jsonl" and revision_overrides:
            records = [{**r, "revision": revision_overrides[r["doc_id"]]} if r["doc_id"] in revision_overrides else r
                       for r in records]
        if records:
            utils.save_jsonl(records, utils.sdf_stage_dir(run_directory, layer) / file_name)
            copied_record_counts[f"layer{layer}/{file_name}"] = len(records)

    manifest_path = run_directory / MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest_path = source_run / MANIFEST_FILE_NAME
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.is_file() else {}
    manifest["seeded_from"] = {
        "source_run_id": source_run.name,
        "source_run_directory": str(source_run),
        "source_git_commit": source_manifest.get("git_commit"),
        "from_step": from_step,
        "start_layer": step.run_py_layer,
        "document_ids": document_ids,
        "prompts_and_constitution": "this checkout's current files" if use_current_prompts else "the source run's frozen inputs",
        "revision_override_document_ids": sorted(revision_overrides),
        "copied_record_counts": copied_record_counts,
    }
    if assignment_is_solved_again:
        manifest["seeded_from"]["consideration_assignment"] = (
            f"solved again over {len(document_ids)} document(s); may differ from the source run's")
        manifest["seeded_from"]["consideration_assignment_solved_again_over_document_count"] = len(document_ids)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return SeededRun(run_directory, step.run_py_layer, document_ids, copied_record_counts)


def _required_records_for_step(step, config, prompts_dir, stage_directory, document_ids, input_records, input_location) -> list:
    """One message per selected document and missing record. Empty when the source can feed the step."""
    problems = []
    position = _position_of(step.name)
    plans = _records_by_id(stage_directory[1], "plans.jsonl", "prompt_id")

    def after(step_name):  # the chosen step comes after step_name
        return position > _position_of(step_name)

    for document_id in document_ids:
        if document_id not in input_records:
            problems.append(f"{document_id}: no record in {input_location}, which step {step.name} reads its documents from")
    if step.run_py_layer == 1 and layer1_screen.reason_screen_is_inactive(config, prompts_dir):
        prompt_ids = set(_records_by_id(stage_directory[1], "prompts.jsonl", "prompt_id"))
        problems += [f"{d}: no record in layer1/prompts.jsonl (the screen is off, so planning reads composed prompts)"
                     for d in document_ids if d not in prompt_ids]
    if after("plan") or step.name in ("description_rewrite", "consideration_scoring", "draft"):
        for document_id in document_ids:
            plan = plans.get(document_id)
            if plan is None:
                problems.append(f"{document_id}: no plan in layer1/plans.jsonl")
            elif not plan.get("description"):
                problems.append(f"{document_id}: its plan was declared incoherent, so the document is not in the pipeline")
    if position > _position_of("description_rewrite") \
            and not layer1_description_rewrite.reason_description_rewrite_is_inactive(config, prompts_dir):
        results = _records_by_id(stage_directory[1], layer1_description_rewrite.RESULTS_FILENAME, "prompt_id")
        for document_id in document_ids:
            if document_id not in results:
                problems.append(f"{document_id}: no description rewrite result (seeding here would pay for the rewrite again)")
            elif results[document_id].get("dealt_value_conflict"):
                problems.append(f"{document_id}: its description rewrite found a dealt-value conflict, so it was dropped")
    if after("consideration_assignment") \
            and not layer1_select_considerations.reason_consideration_selection_is_inactive(config, prompts_dir):
        catalog = layer1_select_considerations.load_considerations(prompts_dir)
        selections = _records_by_id(stage_directory[1], layer1_select_considerations.SELECTIONS_FILENAME, "prompt_id")
        eligible = layer1_select_considerations.eligible_plans(
            config, catalog, [plans[d] for d in document_ids if d in plans and plans[d].get("description")])
        for plan in eligible:
            if plan["prompt_id"] not in selections:
                problems.append(f"{plan['prompt_id']}: no assigned considerations in "
                                f"layer1/{layer1_select_considerations.SELECTIONS_FILENAME}")
    if step.name in ("critique", "revise"):
        drafts = _records_by_id(stage_directory[2], "drafts.jsonl", "doc_id")
        problems += [f"{d}: no draft in layer2/drafts.jsonl" for d in document_ids if d not in drafts]
    return problems
