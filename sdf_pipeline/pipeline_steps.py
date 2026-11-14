"""The step catalogue: one entry per SDF pipeline step, read by ``run.py``, ``start_a_run_from_an_earlier_runs_step.py`` and the request
print command. An entry holds what the pipeline knows about a step besides its request and reply: today's
name, the folder key of the folder it writes to (``shared/sdf_run_layouts.py``: the ``--layer`` number, the folder
and the cost-log tag of a step depend on the run's layout and are looked up from it), module, model setting keys
(first one set wins), stage files and id fields, required prompt files, why it is inactive, whether it can be
seeded, and where it reads its documents from (``load_documents`` over a run's stage files, ``build_step`` for the
step). ``run.py``'s restarts, the print command and the critique map all go through those, so nothing rebuilds a
step's input by itself. ``shuffle_seed`` holds the old stage tags that are also random seeds (``layer1_screen``,
``layer1_select``), literal and independent of any layout's tags. The compose step between the screen and the
plan has no entry; its file ``prompts.jsonl`` is listed under the screen, since seeding at the plan copies it
with the screen's files (it lives in the plan's folder)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sdf_pipeline.layer1_deal_and_screen import run_layer as screen_layer
from sdf_pipeline.layer3_consideration_scoring_and_selection import run_layer as selection_layer
from sdf_pipeline.layer4_draft_critique_and_revise import shared_by_this_layers_steps as revision_layer
from sdf_pipeline.shared_step_machinery.considerations_catalog import load_considerations
from sdf_pipeline.shared_step_machinery.renamed_model_settings import model_setting
from sdf_pipeline.layer3_consideration_scoring_and_selection import consideration_scoring
from sdf_pipeline.layer4_draft_critique_and_revise import critique, critique_and_revise_in_one_call, draft, revise
from sdf_pipeline.layer2_plan_and_description_rewrite import description_rewrite as rewrite_layer, plan
from sdf_pipeline.layer5_score_and_final_selection import score
from shared import sdf_run_layouts, utils
from shared.sdf_run_layouts import (
    CONSIDERATION_SCORING_AND_SELECTION, CRITIQUE_AND_REVISE, DEAL_AND_SCREEN, DRAFT, PLAN_AND_DESCRIPTION_REWRITE, SCORE)


class StepCannotBeBuilt(Exception):
    """The step cannot be built from this run's files for this document (the message says why)."""


@dataclass(frozen=True)
class StageFile:
    folder_key: str   # which folder it lives in (``sdf_run_layouts.FOLDER_KEYS``)
    file_name: str
    id_field: str     # the field of each record that holds the document id


@dataclass(frozen=True)
class RunReading:
    """A run's configuration, prompt snapshot and stage folders ({folder key: folder}): what a step's input is read from."""
    config: dict
    prompts_directory: Path
    stage_directories: dict

    def records(self, stage_file: StageFile) -> list:
        return utils.load_jsonl(self.stage_directories[stage_file.folder_key] / stage_file.file_name)


@dataclass(frozen=True)
class PipelineStepEntry:
    name: str
    folder_key: str
    module_name: str                              # under sdf_pipeline
    makes_model_calls: bool = False
    shuffle_seed: str | None = None               # literal seed text, independent of any renamed stage tag
    model_setting_keys: tuple = ()                # sdf.<key>; the first one set wins
    stage_files: tuple = ()                       # records later steps read and seeding copies
    required_prompt_files: Callable = lambda config, prompts_dir: []
    reason_inactive: Callable = lambda config, prompts_dir: None   # printed reason the step does not run, or None
    reason_not_seedable: str | None = None
    seeding_refuses_when_inactive: bool = False
    document_file: Callable | None = None         # (config, prompts_dir) -> StageFile whose ids are its documents' (seeding selects from them)
    load_documents: Callable | None = None        # (RunReading) -> the step's input documents, as run.py's restart reads them
    build_step: Callable | None = None            # (RunReading, documents, document_id) -> the step
    print_command_name: str | None = None
    print_refuses_when_inactive: bool = False
    checks_its_own_prompt_snapshot: bool = False  # refuses a missing template for itself, with its own message

    @property
    def stage_tag(self) -> str | None:
        """The cost-log tag of the step's calls in the layout of the run being written; None for a step with no call."""
        return sdf_run_layouts.active_layout().stage_tag(self.name) if self.makes_model_calls else None

    def stage_tag_in(self, layout) -> str | None:
        """The cost-log tag of the step's calls in a layout; None for a step with no call."""
        return layout.stage_tag(self.name) if self.makes_model_calls else None

    def layer_in(self, layout) -> int:
        """The ``--layer`` number of the step in a layout."""
        return layout.layer_number_of_key[self.folder_key]

    def model_for(self, config: dict):
        """The model the step is configured with (the first setting key that is set)."""
        return next((model_setting(config["sdf"], key) for key in self.model_setting_keys
                     if model_setting(config["sdf"], key)), None)


def _template(layer: int):   # the layer number in the template's file name (layer1.txt ...), which does not change
    return lambda config, prompts_dir: [utils.sdf_template_path(prompts_dir, layer).name]


def _layer_one_files(config, prompts_dir):
    return [*_template(1)(config, prompts_dir), "variables.txt", "preamble.txt"]


def _file(folder_key: str, file_name: str, id_field: str):
    return StageFile(folder_key, file_name, id_field)


PLANS = _file(PLAN_AND_DESCRIPTION_REWRITE, "plans.jsonl", "prompt_id")
PROMPTS = _file(PLAN_AND_DESCRIPTION_REWRITE, "prompts.jsonl", "prompt_id")
SCREENED_HANDS = _file(DEAL_AND_SCREEN, screen_layer.SCREENED_HANDS_FILENAME, "prompt_id")
DRAFTS = _file(DRAFT, "drafts.jsonl", "doc_id")
CRITIQUES = _file(CRITIQUE_AND_REVISE, "critiques.jsonl", "doc_id")
REWRITES = _file(CRITIQUE_AND_REVISE, "rewrites.jsonl", "doc_id")
_FIXED = lambda stage_file: (lambda config, prompts_dir: stage_file)   # a step that always reads one file
_RECORDS = lambda stage_file: (lambda reading: reading.records(stage_file))


def _plans_after_description_rewrite(reading):
    return rewrite_layer.load_rewritten_plans(
        reading.config, reading.stage_directories[PLAN_AND_DESCRIPTION_REWRITE], reading.records(PLANS))


def _plans_with_selections(reading):
    return selection_layer.attach_selections(
        reading.config, reading.prompts_directory, reading.stage_directories[CONSIDERATION_SCORING_AND_SELECTION], _plans_after_description_rewrite(reading))


def _from_config_and_prompts(step_class):
    return lambda reading, documents, document_id: step_class(reading.config, reading.prompts_directory)


def _build_description_rewrite(reading, documents, document_id):
    return rewrite_layer.DescriptionRewriteStep(
        reading.config, rewrite_layer.load_rewrite_prompts(reading.prompts_directory), reading.prompts_directory,
        reading.stage_directories[PLAN_AND_DESCRIPTION_REWRITE], documents)


def _build_consideration_scoring(reading, documents, document_id):
    return consideration_scoring.ConsiderationScoringStep(
        reading.config, load_considerations(reading.prompts_directory),
        selection_layer.load_selection_template(reading.prompts_directory))


def _build_revise(reading, documents, document_id):
    critique_by_document_id = revision_layer.load_critiques_by_document_id(reading.stage_directories[CRITIQUE_AND_REVISE])
    if document_id not in critique_by_document_id:
        raise StepCannotBeBuilt(f"the revise request needs the critique of {document_id!r}, and "
                                f"{reading.stage_directories[CRITIQUE_AND_REVISE] / 'critiques.jsonl'} has none")
    return revise.ReviseStep(reading.config, reading.prompts_directory, critique_by_document_id)


def _plan_ids_seeding_reads(config, prompts_dir):   # planning composes prompts from the screened hands, or reads the composed ones
    return PROMPTS if screen_layer.reason_screen_is_inactive(config, prompts_dir) else SCREENED_HANDS


_SPLIT = revision_layer.revision_is_split
_SINGLE_CALL = ("this configuration (sdf.revision.split: false) revises in one call together with the critique, "
                "so there is no separate revise step; seed from 'critique' instead")
_NOT_SPLIT = lambda config, prompts_dir: None if _SPLIT(config) else _SINGLE_CALL
_SPLIT_ONLY = lambda config, prompts_dir: "sdf.revision.split is true, so the critique and the revise are two calls" if _SPLIT(config) else None
_SCREEN_OFF = screen_layer.reason_screen_is_inactive
_SELECTION_OFF = selection_layer.reason_consideration_selection_is_inactive
E = PipelineStepEntry

# In pipeline order. The folder key says where the step's files live; its --layer number and cost-log tag come from the layout.
PIPELINE_STEPS = (
    E("deal", DEAL_AND_SCREEN, "layer1_deal_and_screen.deal", required_prompt_files=lambda c, p: [*_template(1)(c, p), "variables.txt"],
      reason_inactive=_SCREEN_OFF, reason_not_seedable="a run's hands come from its deal; start a normal run"),
    E("screen", DEAL_AND_SCREEN, "layer1_deal_and_screen.screen", makes_model_calls=True, shuffle_seed="layer1_screen",
      model_setting_keys=("screen_model", "plan_model"), stage_files=(SCREENED_HANDS, PROMPTS),
      required_prompt_files=lambda c, p: [screen_layer.TEMPLATE_FILENAME, screen_layer.HAND_BLOCK_FILENAME, screen_layer.SCHEMA_FILENAME],
      reason_inactive=_SCREEN_OFF, print_command_name="screen",
      reason_not_seedable="the screen judges every hand of a run together; start a normal run"),
    E("redeal", DEAL_AND_SCREEN, "layer1_deal_and_screen.redeal", required_prompt_files=lambda c, p: [*_template(1)(c, p), "variables.txt"],
      reason_inactive=_SCREEN_OFF, reason_not_seedable="the redeal follows the screen and cannot be started on its own"),
    E("plan", PLAN_AND_DESCRIPTION_REWRITE, "layer2_plan_and_description_rewrite.plan", makes_model_calls=True, model_setting_keys=("plan_model",),
      stage_files=(PLANS,), required_prompt_files=_layer_one_files, document_file=_plan_ids_seeding_reads,
      load_documents=_RECORDS(PROMPTS), build_step=lambda reading, documents, document_id: plan.PlanStep(reading.config),
      print_command_name="plan"),
    E("description_rewrite", PLAN_AND_DESCRIPTION_REWRITE, "layer2_plan_and_description_rewrite.description_rewrite", makes_model_calls=True,
      model_setting_keys=("description_rewrite_model", "plan_model"),
      stage_files=(_file(PLAN_AND_DESCRIPTION_REWRITE, rewrite_layer.RESULTS_FILENAME, "prompt_id"),
                   _file(PLAN_AND_DESCRIPTION_REWRITE, rewrite_layer.REJECTIONS_FILENAME, "prompt_id")),
      required_prompt_files=lambda c, p: [*_layer_one_files(c, p), rewrite_layer.TEMPLATE_FILENAME, rewrite_layer.SCHEMA_FILENAME],
      reason_inactive=rewrite_layer.reason_description_rewrite_is_inactive, document_file=_FIXED(PLANS),
      load_documents=_RECORDS(PLANS), build_step=_build_description_rewrite, print_command_name="description_rewrite",
      print_refuses_when_inactive=True),
    E("consideration_scoring", CONSIDERATION_SCORING_AND_SELECTION, "layer3_consideration_scoring_and_selection.consideration_scoring", makes_model_calls=True,
      shuffle_seed="layer1_select", model_setting_keys=("consideration_scoring_model", "plan_model"),
      stage_files=(_file(CONSIDERATION_SCORING_AND_SELECTION, selection_layer.SCORES_FILENAME, "prompt_id"),),
      required_prompt_files=lambda c, p: [*_layer_one_files(c, p), selection_layer.TEMPLATE_FILENAME, selection_layer.CONSIDERATIONS_FILENAME],
      reason_inactive=_SELECTION_OFF, document_file=_FIXED(PLANS), load_documents=_plans_after_description_rewrite,
      build_step=_build_consideration_scoring, print_command_name="consideration_scoring", print_refuses_when_inactive=True),
    E("consideration_selection", CONSIDERATION_SCORING_AND_SELECTION, "layer3_consideration_scoring_and_selection.consideration_selection", reason_inactive=_SELECTION_OFF,
      stage_files=(_file(CONSIDERATION_SCORING_AND_SELECTION, selection_layer.SELECTIONS_FILENAME, "prompt_id"),),
      reason_not_seedable="the assignment is solved over every scored document at once, so seeding it for selected "
                          "documents would change it; seed at consideration_scoring instead"),
    E("draft", DRAFT, "layer4_draft_critique_and_revise.draft", makes_model_calls=True, model_setting_keys=("draft_model",), stage_files=(DRAFTS,),
      required_prompt_files=_template(2), document_file=_FIXED(PLANS), load_documents=_plans_with_selections,
      build_step=_from_config_and_prompts(draft.DraftStep), print_command_name="draft"),
    # Seeding at the critique stays allowed under split false (the single call then runs in its place).
    E("critique", CRITIQUE_AND_REVISE, "layer4_draft_critique_and_revise.critique", makes_model_calls=True,
      model_setting_keys=("critique_model", "revise_model"),
      stage_files=(CRITIQUES,), reason_inactive=_NOT_SPLIT, checks_its_own_prompt_snapshot=True, document_file=_FIXED(DRAFTS),
      required_prompt_files=lambda c, p: [revision_layer.CRITIQUE_TEMPLATE] if _SPLIT(c) else _template(3)(c, p),
      load_documents=_RECORDS(DRAFTS), build_step=_from_config_and_prompts(critique.CritiqueStep),
      print_command_name="critique"),
    E("revise", CRITIQUE_AND_REVISE, "layer4_draft_critique_and_revise.revise", makes_model_calls=True, model_setting_keys=("revise_model",),
      stage_files=(REWRITES,),
      required_prompt_files=lambda c, p: [revision_layer.REVISE_TEMPLATE], reason_inactive=_NOT_SPLIT,
      seeding_refuses_when_inactive=True, checks_its_own_prompt_snapshot=True, document_file=_FIXED(CRITIQUES),
      load_documents=_RECORDS(DRAFTS), build_step=_build_revise, print_command_name="revise"),
    # The single-call step while it exists.
    E("critique_and_revise_in_one_call", CRITIQUE_AND_REVISE, "layer4_draft_critique_and_revise.critique_and_revise_in_one_call",
      makes_model_calls=True, model_setting_keys=("revise_model",),
      stage_files=(REWRITES,), required_prompt_files=_template(3), reason_inactive=_SPLIT_ONLY,
      reason_not_seedable="it runs in place of the critique and the revise under sdf.revision.split: false; seed at 'critique' instead",
      document_file=_FIXED(DRAFTS), load_documents=_RECORDS(DRAFTS),
      build_step=_from_config_and_prompts(critique_and_revise_in_one_call.CritiqueAndReviseInOneCallStep),
      print_command_name="critique_and_revise_in_one_call"),
    E("score", SCORE, "layer5_score_and_final_selection.score", makes_model_calls=True, model_setting_keys=("score_model",),
      required_prompt_files=_template(4), document_file=_FIXED(REWRITES), load_documents=_RECORDS(REWRITES),
      build_step=_from_config_and_prompts(score.ScoreStep), print_command_name="score"),
    E("score_gate", SCORE, "layer5_score_and_final_selection.score_gate", print_command_name="score_gate",
      reason_not_seedable="the score gate runs after scoring and makes no calls; resume the scoring run"),
    E("near_duplicate_removal", SCORE, "layer5_score_and_final_selection.near_duplicate_removal", print_command_name="near_duplicate_removal",
      reason_not_seedable="the near-duplicate removal runs after the score gate and makes no calls; resume the scoring run"),
)
STEP_NAMES_IN_ORDER = tuple(entry.name for entry in PIPELINE_STEPS)


def canonical_stage_tag(tag: str) -> str:
    """The five-layer layout's cost-log tag for a tag of either layout, so per-step cost tables line runs of both layouts
    up (``layer3_critique`` and ``layer4_critique`` are both ``layer4_critique``). Other tags (eval tags, tags of the
    single-call step in a run made before the tag was renamed) come back as they are."""
    for entry in PIPELINE_STEPS:
        if entry.makes_model_calls and tag in (entry.stage_tag_in(sdf_run_layouts.FOUR_NUMBERED_LAYERS),
                                               entry.stage_tag_in(sdf_run_layouts.FIVE_NAMED_LAYERS)):
            return entry.stage_tag_in(sdf_run_layouts.FIVE_NAMED_LAYERS)
    return tag


def entry_named(step_name: str) -> PipelineStepEntry:
    return PIPELINE_STEPS[STEP_NAMES_IN_ORDER.index(step_name)]


def entry_for_print_command_stage(stage_name: str) -> PipelineStepEntry | None:
    return next((entry for entry in PIPELINE_STEPS if entry.print_command_name == stage_name), None)


def steps_of_layer(layer: int, layout=None) -> tuple:
    """The steps of ``--layer`` number ``layer`` in a layout (default: the layout of the run being written)."""
    layout = layout or sdf_run_layouts.active_layout()
    return tuple(entry for entry in PIPELINE_STEPS if entry.layer_in(layout) == layer)


def documents_read_when_restarting_at_layer(layer: int, reading: RunReading, layout=None) -> list:
    """What ``run.py --resume --layer N`` reads from the stage files: the input of the layer's first step that
    reads documents (in the five-layer layout layer 3: plans after the description rewrite; 4: plans with selections
    attached; 5: rewrites. Layer 2 starts from the screened hands, which ``run.py`` reads itself)."""
    return next(entry for entry in steps_of_layer(layer, layout) if entry.load_documents).load_documents(reading)


def prompt_files_missing_from_snapshot(folder_key: str, config: dict, prompts_dir: Path) -> list:
    """[(step name, [missing file names])] for the active steps writing to the folder key that do not check their own snapshot."""
    missing_by_step = [
        (entry.name, [name for name in entry.required_prompt_files(config, prompts_dir) if not (Path(prompts_dir) / name).is_file()])
        for entry in PIPELINE_STEPS
        if entry.folder_key == folder_key
        if not entry.checks_its_own_prompt_snapshot and entry.reason_inactive(config, prompts_dir) is None]
    return [(name, missing) for name, missing in missing_by_step if missing]
