"""The step map: every SDF step in order, read by ``run.py``, ``start_a_run_from_an_earlier_runs_step.py`` and the request print command.
A model-calling step is its file's ``STEP``; a step with no model call has only its name, folder key, seeding fields and the files others need of it.
The compose step has no entry: ``prompts.jsonl`` is listed under the screen because seeding at the plan copies it with the screen's files."""

from pathlib import Path

from sdf_pipeline.layer1_deal_variables_and_screen import screen
from sdf_pipeline.layer2_description_write_and_description_rewrite import description_rewrite, description_write
from sdf_pipeline.layer3_consideration_scoring_and_selection import consideration_scoring
from sdf_pipeline.layer3_consideration_scoring_and_selection.shared_by_this_layers_steps import (
    SELECTIONS_FILENAME, reason_consideration_selection_is_inactive)
from sdf_pipeline.layer4_draft_critique_and_revise import critique, critique_and_revise_in_one_call, draft, revise
from sdf_pipeline.layer5_score_and_final_selection import score
from sdf_pipeline.shared_step_machinery.step_record import Run, StageFile, Step
from shared import sdf_run_layouts, utils
from shared.sdf_run_layouts import CONSIDERATION_SCORING_AND_SELECTION, DEAL_VARIABLES_AND_SCREEN, SCORE

STEPS_CHECKING_THEIR_OWN_PROMPT_SNAPSHOT = ("critique", "revise")   # each refuses a missing template with its own message


def _hand_dealing_prompt_files(config: dict, prompts_directory: Path) -> list:
    return [utils.sdf_template_path(prompts_directory, 1).name, "variables.txt"]


# In pipeline order.
PIPELINE_STEPS = (
    Step("deal_variables", DEAL_VARIABLES_AND_SCREEN, required_prompt_files=_hand_dealing_prompt_files,
         reason_inactive=lambda config, prompts_directory: screen.reason_screen_is_inactive(config, prompts_directory)),
    screen.STEP,
    Step("redeal", DEAL_VARIABLES_AND_SCREEN, required_prompt_files=_hand_dealing_prompt_files,
         reason_inactive=lambda config, prompts_directory: screen.reason_screen_is_inactive(config, prompts_directory)),
    description_write.STEP,
    description_rewrite.STEP,
    consideration_scoring.STEP,
    Step("consideration_selection", CONSIDERATION_SCORING_AND_SELECTION,
         seed_from_instead="consideration_scoring", reason_inactive=reason_consideration_selection_is_inactive,
         writes=(StageFile(CONSIDERATION_SCORING_AND_SELECTION, SELECTIONS_FILENAME, "prompt_id"),)),
    draft.STEP,
    critique.STEP,
    revise.STEP,
    critique_and_revise_in_one_call.STEP,
    score.STEP,
    Step("score_gate", SCORE),
    Step("near_duplicate_removal", SCORE),
)
STEP_NAMES_IN_ORDER = tuple(step.name for step in PIPELINE_STEPS)


def canonical_stage_tag(tag: str) -> str:
    """The five-layer layout's cost-log tag for a tag of either layout, so per-step cost tables line runs of both layouts
    up (``layer3_critique`` and ``layer4_critique`` are both ``layer4_critique``). Other tags come back as they are."""
    for step in PIPELINE_STEPS:
        if step.makes_model_calls and tag in (step.stage_tag_in(sdf_run_layouts.FOUR_NUMBERED_LAYERS),
                                              step.stage_tag_in(sdf_run_layouts.FIVE_NAMED_LAYERS)):
            return step.stage_tag_in(sdf_run_layouts.FIVE_NAMED_LAYERS)
    return tag


def entry_named(step_name: str) -> Step:
    return PIPELINE_STEPS[STEP_NAMES_IN_ORDER.index(step_name)]


def steps_of_layer(layer: int, layout=None) -> tuple:
    """The steps of ``--layer`` number ``layer`` in a layout (default: the layout of the run being written)."""
    layout = layout or sdf_run_layouts.active_layout()
    return tuple(step for step in PIPELINE_STEPS if step.layer_in(layout) == layer)


def documents_read_when_restarting_at_layer(layer: int, run: Run, layout=None) -> list:
    """What ``run.py --resume --layer N`` reads: the input of the layer's first document-reading step (layer 2 starts from the screened hands, which ``run.py`` reads itself)."""
    return next(step for step in steps_of_layer(layer, layout) if step.reads).reads(run)


def prompt_files_missing_from_snapshot(folder_key: str, config: dict, prompts_dir: Path) -> list:
    """[(step name, [missing file names])] for the active steps writing to the folder key that do not check their own snapshot."""
    missing_by_step = [
        (step.name, [name for name in step.required_prompt_files(config, prompts_dir) if not (Path(prompts_dir) / name).is_file()])
        for step in PIPELINE_STEPS
        if step.folder_key == folder_key
        if step.name not in STEPS_CHECKING_THEIR_OWN_PROMPT_SNAPSHOT and step.reason_inactive(config, prompts_dir) is None]
    return [(name, missing) for name, missing in missing_by_step if missing]
