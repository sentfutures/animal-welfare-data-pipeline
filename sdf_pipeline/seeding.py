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
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline.document_state import RunLayout
from sdf_pipeline.document_state_loader import UnknownRunLayoutError, detect_run_layout
from shared import utils

MANIFEST_FILE_NAME = "run_manifest.json"
REVISION_OVERRIDE_FOR_EVERY_DOCUMENT = "*"


class SeedingRefused(SystemExit):
    """Seeding was refused; the message says why. Nothing was created."""


@dataclass(frozen=True)
class SeedableStep:
    name: str
    start_layer: int   # the run.py layer that runs this step
    stage_layer: int   # the layer folder that holds the step's own output (the input of the next step)


SEEDABLE_STEPS = {
    step.name: step for step in (
        SeedableStep("plan", 1, 1),
        SeedableStep("description_rewrite", 1, 1),
        SeedableStep("consideration_scoring", 1, 1),
        SeedableStep("draft", 2, 2),
        SeedableStep("critique", 3, 3),
        SeedableStep("revise", 3, 3),
        SeedableStep("score", 4, 4),
    )
}

# Not seedable, with the reason given to the person who asked for them.
UNSEEDABLE_STEPS = {
    "deal": "a run's hands come from its deal; start a normal run",
    "screen": "the screen judges every hand of a run together; start a normal run",
    "redeal": "the redeal follows the screen and cannot be started on its own",
    "consideration_assignment": ("the assignment is solved over every scored document at once, so seeding it for "
                                 "selected documents would change it; seed at consideration_scoring instead"),
    "gate": "the gate runs after scoring and makes no calls; resume the scoring run",
    "near_duplicate_cull": "the cull runs after the gate and makes no calls; resume the scoring run",
}

# (step whose output the file is, layer folder, file name, the field holding the document id), in pipeline
# order. A file is copied when the chosen step comes after the step that writes it.
_STAGE_FILES_IN_PIPELINE_ORDER = (
    ("screen", 1, layer1_screen.SCREENED_HANDS_FILENAME, "prompt_id"),
    ("screen", 1, "prompts.jsonl", "prompt_id"),
    ("plan", 1, "plans.jsonl", "prompt_id"),
    ("description_rewrite", 1, layer1_description_rewrite.RESULTS_FILENAME, "prompt_id"),
    ("description_rewrite", 1, layer1_description_rewrite.REJECTIONS_FILENAME, "prompt_id"),
    ("consideration_scoring", 1, layer1_select_considerations.SCORES_FILENAME, "prompt_id"),
    ("consideration_assignment", 1, layer1_select_considerations.SELECTIONS_FILENAME, "prompt_id"),
    ("draft", 2, "drafts.jsonl", "doc_id"),
    ("critique", 3, "critiques.jsonl", "doc_id"),
    ("revise", 3, "rewrites.jsonl", "doc_id"),
)
_ORDER_OF_STEPS = ("screen", "plan", "description_rewrite", "consideration_scoring", "consideration_assignment",
                   "draft", "critique", "revise", "score")


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


def _templates_the_step_needs(step: SeedableStep, config: dict, prompts_dir: Path) -> list:
    """File names (relative to the snapshot) the step cannot run without, under this configuration."""
    sdf = config["sdf"]
    split = bool((sdf.get("revision") or {}).get("split", True))
    needed = []
    if step.name in ("plan", "description_rewrite", "consideration_scoring"):
        needed += [utils.sdf_template_path(prompts_dir, 1).name, "variables.txt", "preamble.txt"]
    if step.name == "description_rewrite":
        needed += [layer1_description_rewrite.TEMPLATE_FILENAME, layer1_description_rewrite.SCHEMA_FILENAME]
    if step.name == "consideration_scoring":
        needed += [layer1_select_considerations.TEMPLATE_FILENAME, layer1_select_considerations.CONSIDERATIONS_FILENAME]
    if step.name == "draft":
        needed.append(utils.sdf_template_path(prompts_dir, 2).name)
    if step.name == "critique":
        needed.append("layer3a_critique.txt" if split else utils.sdf_template_path(prompts_dir, 3).name)
    if step.name == "revise":
        needed.append("layer3b_revise.txt")
    if step.name == "score":
        needed.append(utils.sdf_template_path(prompts_dir, 4).name)
    return needed


def _screen_is_active(config: dict, prompts_dir: Path) -> bool:
    return bool(config["sdf"].get("screen_enabled", True)) and layer1_screen.load_screen_prompts(prompts_dir) is not None


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
    if from_step == "revise" and not bool((config["sdf"].get("revision") or {}).get("split", True)):
        _refuse("step 'revise' cannot be seeded under sdf.revision.split: false, because this configuration revises in "
                "one call together with the critique, so there is no separate revise step; seed from 'critique' instead")
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
    missing_templates = [name for name in _templates_the_step_needs(step, config, prompts_dir) if not (prompts_dir / name).is_file()]
    if missing_templates:
        which = "this checkout's prompts" if use_current_prompts else f"source run {source_run.name}'s frozen prompt snapshot"
        _refuse(f"{which} lacks {', '.join(missing_templates)}, which step {from_step} needs")

    stage_directory = {layer: utils.sdf_stage_dir_of_existing_run(source_run, layer) for layer in (1, 2, 3, 4)}
    step_position = _ORDER_OF_STEPS.index(from_step)
    files_to_copy = [entry for entry in _STAGE_FILES_IN_PIPELINE_ORDER if _ORDER_OF_STEPS.index(entry[0]) < step_position]

    # Which documents: those named, else every document the step has an input record for.
    input_file = {
        "plan": (1, layer1_screen.SCREENED_HANDS_FILENAME if _screen_is_active(config, prompts_dir) else "prompts.jsonl", "prompt_id"),
        "description_rewrite": (1, "plans.jsonl", "prompt_id"), "consideration_scoring": (1, "plans.jsonl", "prompt_id"),
        "draft": (1, "plans.jsonl", "prompt_id"), "critique": (2, "drafts.jsonl", "doc_id"),
        "revise": (3, "critiques.jsonl", "doc_id"), "score": (3, "rewrites.jsonl", "doc_id"),
    }[from_step]
    input_layer, input_file_name, input_id_field = input_file
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
    for _, layer, file_name, id_field in _STAGE_FILES_IN_PIPELINE_ORDER:
        every_known_id |= set(_records_by_id(stage_directory[layer], file_name, id_field))
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
    for _, layer, file_name, id_field in files_to_copy:
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
        "start_layer": step.start_layer,
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
    return SeededRun(run_directory, step.start_layer, document_ids, copied_record_counts)


def _required_records_for_step(step, config, prompts_dir, stage_directory, document_ids, input_records, input_location) -> list:
    """One message per selected document and missing record. Empty when the source can feed the step."""
    problems = []
    position = _ORDER_OF_STEPS.index(step.name)
    plans = _records_by_id(stage_directory[1], "plans.jsonl", "prompt_id")

    def after(step_name):  # the chosen step comes after step_name
        return position > _ORDER_OF_STEPS.index(step_name)

    for document_id in document_ids:
        if document_id not in input_records:
            problems.append(f"{document_id}: no record in {input_location}, which step {step.name} reads its documents from")
    if step.start_layer == 1 and not _screen_is_active(config, prompts_dir):
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
    if position > _ORDER_OF_STEPS.index("description_rewrite") and layer1_description_rewrite.enabled(config) \
            and layer1_description_rewrite.load_rewrite_prompts(prompts_dir) is not None:
        results = _records_by_id(stage_directory[1], layer1_description_rewrite.RESULTS_FILENAME, "prompt_id")
        for document_id in document_ids:
            if document_id not in results:
                problems.append(f"{document_id}: no description rewrite result (seeding here would pay for the rewrite again)")
            elif results[document_id].get("dealt_value_conflict"):
                problems.append(f"{document_id}: its description rewrite found a dealt-value conflict, so it was dropped")
    if after("consideration_assignment") and layer1_select_considerations.selection_enabled(config) \
            and layer1_select_considerations.load_selection_template(prompts_dir) is not None \
            and (prompts_dir / layer1_select_considerations.CONSIDERATIONS_FILENAME).is_file():
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
