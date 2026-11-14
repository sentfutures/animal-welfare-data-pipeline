"""The step catalogue: one entry per SDF pipeline step, read by ``run.py``, ``seeding.py`` and the
request print command (``scripts/print_full_sdf_request_a_stage_would_send_for_one_document.py``).

An entry holds what the pipeline knows about a step other than its request and reply: today's step
name, the ``run.py`` layer that runs it, its module, the cost-log stage tag, the model setting keys
(first one set wins), the stage files it writes and their id field, the prompt files it needs, whether
it is active under a configuration, whether it can be seeded (and why not), and where it reads its
documents from: a function that returns the input documents from a run's stage files and one that
builds the step for them. ``run.py``'s restart (``--resume --layer N``), the print command and layer 3's
critique map all go through those functions, so no place rebuilds a step's input by itself.

Names here are today's names, unchanged. The stage tags ``layer1_screen`` and ``layer1_select`` are
also random seeds (the screen's batch order and redeal, the order considerations are offered in):
``shuffle_seed`` holds them as a separate field so a later rename of the stage tag leaves the seeds
literal (a test pins that the layer modules still derive their seeds from the same text).

Not catalogued: the compose step between the screen and the plan, which has no outcome of its own. Its
file ``prompts.jsonl`` is listed under the screen, the last catalogued step before it, because seeding
at the plan copies it with the screen's files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sdf_pipeline import (
    layer1_description_rewrite, layer1_plan, layer1_screen, layer1_select_considerations, layer2_draft,
    layer3_rewrite, layer4_score)
from sdf_pipeline.considerations_catalog import CONSIDERATIONS_FILENAME, load_considerations
from shared import utils

LAYER_NUMBERS = (1, 2, 3, 4)


class StepCannotBeBuilt(Exception):
    """The step cannot be built from this run's files for this document (the message says why)."""


@dataclass(frozen=True)
class StageFile:
    layer: int        # the stage folder (layer1 to layer4) that holds the file
    file_name: str
    id_field: str     # the field of each record that holds the document id


@dataclass(frozen=True)
class RunReading:
    """What is needed to read a step's input from a run: its configuration, prompt snapshot and stage folders."""
    config: dict
    prompts_directory: Path
    stage_directories: dict   # {layer number: stage folder}

    def records(self, stage_file: StageFile) -> list:
        return utils.load_jsonl(self.stage_directories[stage_file.layer] / stage_file.file_name)


@dataclass(frozen=True)
class PipelineStepEntry:
    name: str
    run_py_layer: int                       # the run.py layer (and --layer number) that runs it
    module_name: str                        # sdf_pipeline.<module_name>
    stage_tag: str | None                   # cost-log tag and effort-map key; None for steps that make no call
    shuffle_seed: str | None                # literal random-seed text; must not follow a renamed stage tag
    model_setting_keys: tuple               # sdf.<key>, the first one set wins; () for steps without a call
    # Records later steps read and seeding copies (not checkpoints, summaries or failure files).
    stage_files: tuple
    required_prompt_files: Callable         # (config, prompts_dir) -> file names the step cannot run without
    reason_inactive: Callable               # (config, prompts_dir) -> printed reason, or None when it runs
    reason_not_seedable: str | None         # shown to the person who asked to seed it
    reason_not_seedable_under_config: Callable | None   # (config) -> "<under ...>", or None when seedable
    documents_read_from: Callable | None    # (config, prompts_dir) -> StageFile whose records are its documents
    seeding_documents_read_from: Callable | None   # same, where seeding differs (plan, revise); None = same
    load_documents: Callable | None         # (RunReading) -> the documents as run.py's restart reads them
    documents_description: str | None       # for messages; None = the documents file's path
    build_step: Callable | None             # (RunReading, documents, document_id) -> the step
    print_command_name: str | None          # the print command's name for the stage, if it shows it
    reason_print_command_cannot_show: str | None
    message_when_inactive_in_print_command: str | None
    checks_its_own_prompt_snapshot: bool = False   # refuses for itself, with its own message

    @property
    def is_seedable(self) -> bool:
        return self.reason_not_seedable is None and self.seeding_start_is_defined

    @property
    def seeding_start_is_defined(self) -> bool:
        return self.name in _NAMES_SEEDING_ACCEPTS

    def model_for(self, config: dict):
        """The model the step is configured with (the first setting key that is set), as its constructor reads it."""
        sdf = config["sdf"]
        for key in self.model_setting_keys:
            if sdf.get(key):
                return sdf.get(key)
        return None


# ---------------------------------------------------------------------------------- shared helpers

def _never_inactive(config: dict, prompts_dir: Path) -> None:
    return None


def _template(layer: int, prompts_dir: Path) -> str:
    return utils.sdf_template_path(prompts_dir, layer).name


def _layer_one_template_files(config: dict, prompts_dir: Path) -> list:
    return [_template(1, prompts_dir), "variables.txt", "preamble.txt"]


def _no_files(config: dict, prompts_dir: Path) -> list:
    return []


def _the_plans_after_the_description_rewrite(reading: RunReading) -> list:
    return layer1_description_rewrite.load_rewritten_plans(
        reading.config, reading.stage_directories[1], reading.records(_PLANS_FILE))


def _the_plans_with_selections(reading: RunReading) -> list:
    return layer1_select_considerations.attach_selections(
        reading.config, reading.prompts_directory, reading.stage_directories[1],
        _the_plans_after_the_description_rewrite(reading))


def _records_of(stage_file: StageFile) -> Callable:
    return lambda reading: reading.records(stage_file)


def _the_file(stage_file: StageFile) -> Callable:
    return lambda config, prompts_dir: stage_file


_PLANS_FILE = StageFile(1, "plans.jsonl", "prompt_id")
_PROMPTS_FILE = StageFile(1, "prompts.jsonl", "prompt_id")
_SCREENED_HANDS_FILE = StageFile(1, layer1_screen.SCREENED_HANDS_FILENAME, "prompt_id")
_DRAFTS_FILE = StageFile(2, "drafts.jsonl", "doc_id")
_CRITIQUES_FILE = StageFile(3, "critiques.jsonl", "doc_id")
_REWRITES_FILE = StageFile(3, "rewrites.jsonl", "doc_id")
_SCORES_FILE = StageFile(4, "scores.jsonl", "doc_id")


def _plan_documents_seeding_reads(config: dict, prompts_dir: Path) -> StageFile:
    """Planning composes prompts from the screen's accepted hands, or reads the composed prompts when the screen is off."""
    return _PROMPTS_FILE if layer1_screen.reason_screen_is_inactive(config, prompts_dir) else _SCREENED_HANDS_FILE


# ---------------------------------------------------------------------------------- build functions

def _build_plan(reading, documents, document_id):
    return layer1_plan.PlanStep(reading.config)


def _build_description_rewrite(reading, documents, document_id):
    prompts = layer1_description_rewrite.load_rewrite_prompts(reading.prompts_directory)
    return layer1_description_rewrite.DescriptionRewriteStep(
        reading.config, prompts, reading.prompts_directory, reading.stage_directories[1], documents)


def _build_consideration_scoring(reading, documents, document_id):
    template = layer1_select_considerations.load_selection_template(reading.prompts_directory)
    return layer1_select_considerations.ConsiderationScoringStep(
        reading.config, load_considerations(reading.prompts_directory), template)


def _build_draft(reading, documents, document_id):
    return layer2_draft.DraftStep(reading.config, reading.prompts_directory)


def _build_critique(reading, documents, document_id):
    return layer3_rewrite.CritiqueStep(reading.config, reading.prompts_directory)


def _build_revise(reading, documents, document_id):
    critique_by_document_id = layer3_rewrite.load_critiques_by_document_id(reading.stage_directories[3])
    if document_id not in critique_by_document_id:
        raise StepCannotBeBuilt(
            f"the revise request needs the critique of {document_id!r}, and "
            f"{reading.stage_directories[3] / 'critiques.jsonl'} has none")
    return layer3_rewrite.ReviseStep(reading.config, reading.prompts_directory, critique_by_document_id)


def _build_combined(reading, documents, document_id):
    return layer3_rewrite.CombinedReviewAndReviseStep(reading.config, reading.prompts_directory)


def _build_score(reading, documents, document_id):
    return layer4_score.ScoreStep(reading.config, reading.prompts_directory)


# ---------------------------------------------------------------------------------- required prompt files

def _description_rewrite_files(config, prompts_dir):
    return [*_layer_one_template_files(config, prompts_dir), layer1_description_rewrite.TEMPLATE_FILENAME,
            layer1_description_rewrite.SCHEMA_FILENAME]


def _consideration_scoring_files(config, prompts_dir):
    return [*_layer_one_template_files(config, prompts_dir), layer1_select_considerations.TEMPLATE_FILENAME,
            layer1_select_considerations.CONSIDERATIONS_FILENAME]


def _screen_files(config, prompts_dir):
    return [layer1_screen.TEMPLATE_FILENAME, layer1_screen.HAND_BLOCK_FILENAME, layer1_screen.SCHEMA_FILENAME]


def _deal_files(config, prompts_dir):
    return [_template(1, prompts_dir), "variables.txt"]


def _critique_files(config, prompts_dir):
    return [layer3_rewrite.CRITIQUE_TEMPLATE if layer3_rewrite.revision_is_split(config) else _template(3, prompts_dir)]


def _combined_files(config, prompts_dir):
    return [_template(3, prompts_dir)]


# ---------------------------------------------------------------------------------- inactivity

_SINGLE_CALL_REASON = "this configuration revises in one call together with the critique, so there is no separate revise step"


def _inactive_when_revision_is_not_split(config, prompts_dir):
    return None if layer3_rewrite.revision_is_split(config) else _SINGLE_CALL_REASON


def _inactive_when_revision_is_split(config, prompts_dir):
    return "sdf.revision.split is true, so the critique and the revise are two separate calls" \
        if layer3_rewrite.revision_is_split(config) else None


def _revise_cannot_be_seeded_when_not_split(config):
    return None if layer3_rewrite.revision_is_split(config) else (
        f"under sdf.revision.split: false, because {_SINGLE_CALL_REASON}; seed from 'critique' instead")


def _entry(name, run_py_layer, module, **fields) -> PipelineStepEntry:
    defaults = dict(
        stage_tag=None, shuffle_seed=None, model_setting_keys=(), stage_files=(), required_prompt_files=_no_files,
        reason_inactive=_never_inactive, reason_not_seedable=None, reason_not_seedable_under_config=None,
        documents_read_from=None, seeding_documents_read_from=None, load_documents=None, documents_description=None,
        build_step=None, print_command_name=None, reason_print_command_cannot_show=None,
        message_when_inactive_in_print_command=None)
    return PipelineStepEntry(name=name, run_py_layer=run_py_layer, module_name=module, **{**defaults, **fields})


_SCREEN_INACTIVE = layer1_screen.reason_screen_is_inactive

# In pipeline order. (The names and layers are today's.)
PIPELINE_STEPS = (
    _entry("deal", 1, "layer1_screen", required_prompt_files=_deal_files, reason_inactive=_SCREEN_INACTIVE,
           reason_not_seedable="a run's hands come from its deal; start a normal run"),
    _entry("screen", 1, "layer1_screen", stage_tag=layer1_screen.STAGE_TAG, shuffle_seed="layer1_screen",
           model_setting_keys=("screen_model", "plan_model"),
           stage_files=(_SCREENED_HANDS_FILE, _PROMPTS_FILE), required_prompt_files=_screen_files,
           reason_inactive=_SCREEN_INACTIVE,
           reason_not_seedable="the screen judges every hand of a run together; start a normal run",
           print_command_name="screen"),
    _entry("redeal", 1, "layer1_screen", reason_inactive=_SCREEN_INACTIVE,
           required_prompt_files=_deal_files,
           reason_not_seedable="the redeal follows the screen and cannot be started on its own"),
    _entry("plan", 1, "layer1_plan", stage_tag="layer1_plan", model_setting_keys=("plan_model",),
           stage_files=(_PLANS_FILE,), required_prompt_files=_layer_one_template_files,
           documents_read_from=_the_file(_PROMPTS_FILE), seeding_documents_read_from=_plan_documents_seeding_reads,
           load_documents=_records_of(_PROMPTS_FILE), build_step=_build_plan, print_command_name="plan"),
    _entry("description_rewrite", 1, "layer1_description_rewrite", stage_tag=layer1_description_rewrite.STAGE_TAG,
           model_setting_keys=("description_rewrite_model", "plan_model"),
           stage_files=(StageFile(1, layer1_description_rewrite.RESULTS_FILENAME, "prompt_id"),
                        StageFile(1, layer1_description_rewrite.REJECTIONS_FILENAME, "prompt_id")),
           required_prompt_files=_description_rewrite_files,
           reason_inactive=layer1_description_rewrite.reason_description_rewrite_is_inactive,
           documents_read_from=_the_file(_PLANS_FILE), load_documents=_records_of(_PLANS_FILE),
           build_step=_build_description_rewrite, print_command_name="description_rewrite",
           message_when_inactive_in_print_command="the description rewrite is disabled or has no template in this run"),
    _entry("consideration_scoring", 1, "layer1_select_considerations", stage_tag=layer1_select_considerations.STAGE_TAG,
           shuffle_seed="layer1_select", model_setting_keys=("select_model", "plan_model"),
           stage_files=(StageFile(1, layer1_select_considerations.SCORES_FILENAME, "prompt_id"),),
           required_prompt_files=_consideration_scoring_files,
           reason_inactive=layer1_select_considerations.reason_consideration_selection_is_inactive,
           documents_read_from=_the_file(_PLANS_FILE), load_documents=_the_plans_after_the_description_rewrite,
           documents_description="the plans after the description rewrite",
           build_step=_build_consideration_scoring, print_command_name="select",
           message_when_inactive_in_print_command="consideration selection is disabled or has no template in this run"),
    _entry("consideration_assignment", 1, "layer1_select_considerations",
           stage_files=(StageFile(1, layer1_select_considerations.SELECTIONS_FILENAME, "prompt_id"),),
           reason_inactive=layer1_select_considerations.reason_consideration_selection_is_inactive,
           reason_not_seedable=("the assignment is solved over every scored document at once, so seeding it for "
                                "selected documents would change it; seed at consideration_scoring instead")),
    _entry("draft", 2, "layer2_draft", stage_tag="layer2_draft", model_setting_keys=("draft_model",),
           stage_files=(_DRAFTS_FILE,), required_prompt_files=lambda config, prompts_dir: [_template(2, prompts_dir)],
           documents_read_from=_the_file(_PLANS_FILE), load_documents=_the_plans_with_selections,
           documents_description="the plans after the description rewrite",
           build_step=_build_draft, print_command_name="draft"),
    _entry("critique", 3, "layer3_rewrite", stage_tag="layer3_critique",
           model_setting_keys=("critique_model", "rewrite_model"), stage_files=(_CRITIQUES_FILE,),
           required_prompt_files=_critique_files,
           # Seeding at the critique stays allowed under split false (the combined call then runs in its place),
           # so the inactivity below is not a seeding refusal.
           reason_inactive=_inactive_when_revision_is_not_split,
           documents_read_from=_the_file(_DRAFTS_FILE), load_documents=_records_of(_DRAFTS_FILE),
           build_step=_build_critique, print_command_name="critique", checks_its_own_prompt_snapshot=True),
    _entry("revise", 3, "layer3_rewrite", stage_tag="layer3_revise", model_setting_keys=("rewrite_model",),
           stage_files=(StageFile(3, "rewrites.jsonl", "doc_id"),),
           required_prompt_files=lambda config, prompts_dir: [layer3_rewrite.REVISE_TEMPLATE],
           reason_inactive=_inactive_when_revision_is_not_split,
           reason_not_seedable_under_config=_revise_cannot_be_seeded_when_not_split,
           documents_read_from=_the_file(_DRAFTS_FILE), seeding_documents_read_from=_the_file(_CRITIQUES_FILE),
           load_documents=_records_of(_DRAFTS_FILE), build_step=_build_revise, print_command_name="revise",
           checks_its_own_prompt_snapshot=True),
    # The single-call step while it exists. Seeding has never named it, so it is refused as an unknown step.
    _entry("combined_review_and_revise", 3, "layer3_rewrite", stage_tag="layer3_rewrite",
           model_setting_keys=("rewrite_model",), stage_files=(StageFile(3, "rewrites.jsonl", "doc_id"),),
           required_prompt_files=_combined_files, reason_inactive=_inactive_when_revision_is_split,
           documents_read_from=_the_file(_DRAFTS_FILE), load_documents=_records_of(_DRAFTS_FILE),
           build_step=_build_combined, print_command_name="combined"),
    _entry("score", 4, "layer4_score", stage_tag="layer4_score", model_setting_keys=("score_model",),
           required_prompt_files=lambda config, prompts_dir: [_template(4, prompts_dir)],
           documents_read_from=_the_file(StageFile(3, "rewrites.jsonl", "doc_id")),
           load_documents=_records_of(StageFile(3, "rewrites.jsonl", "doc_id")),
           build_step=_build_score, print_command_name="score"),
    _entry("gate", 4, "layer4_score",
           reason_not_seedable="the gate runs after scoring and makes no calls; resume the scoring run",
           reason_print_command_cannot_show="makes no model call"),
    _entry("near_duplicate_cull", 4, "layer4_score",
           reason_not_seedable="the cull runs after the gate and makes no calls; resume the scoring run",
           reason_print_command_cannot_show="makes no model call"),
)

STEP_NAMES_IN_ORDER = tuple(entry.name for entry in PIPELINE_STEPS)
_ENTRY_BY_NAME = {entry.name: entry for entry in PIPELINE_STEPS}
_ENTRY_BY_PRINT_COMMAND_NAME = {entry.print_command_name: entry for entry in PIPELINE_STEPS if entry.print_command_name}

# Seeding has always listed these steps as seedable (in this order); the screen, deal, redeal, assignment, gate
# and cull are refused with their reason, and anything else (the combined step) as unknown.
_NAMES_SEEDING_ACCEPTS = ("plan", "description_rewrite", "consideration_scoring", "draft", "critique", "revise", "score")


# ---------------------------------------------------------------------------------- lookups

def entry_named(step_name: str) -> PipelineStepEntry:
    return _ENTRY_BY_NAME[step_name]


def entry_for_print_command_stage(stage_name: str) -> PipelineStepEntry | None:
    return _ENTRY_BY_PRINT_COMMAND_NAME.get(stage_name)


def steps_of_layer(layer: int) -> tuple:
    return tuple(entry for entry in PIPELINE_STEPS if entry.run_py_layer == layer)


def documents_read_when_restarting_at_layer(layer: int, reading: RunReading) -> list:
    """The documents ``run.py --resume --layer N`` reads from the stage files: what the first step of the layer
    that reads documents takes in (layer 2: the plans after the description rewrite, with the saved selections
    attached; layer 3: the drafts; layer 4: the rewrites)."""
    first_step = next(entry for entry in steps_of_layer(layer) if entry.load_documents)
    return first_step.load_documents(reading)


def prompt_files_missing_from_snapshot(layer: int, config: dict, prompts_dir: Path) -> list:
    """[(step name, [missing file names])] for the active steps of the layer that do not check their own snapshot."""
    problems = []
    for entry in steps_of_layer(layer):
        if entry.checks_its_own_prompt_snapshot or entry.reason_inactive(config, prompts_dir) is not None:
            continue
        missing = [name for name in entry.required_prompt_files(config, prompts_dir) if not (Path(prompts_dir) / name).is_file()]
        if missing:
            problems.append((entry.name, missing))
    return problems
