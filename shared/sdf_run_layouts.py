"""The on-disk and cost-log naming of an SDF run, per layout (decision 17).

A run's layout is decided once, when the run is created, and kept for its whole life:

* ``FIVE_NAMED_LAYERS``: stage folders ``layer1_deal_and_screen``, ``layer2_plan_and_description_rewrite``,
  ``layer3_consideration_scoring_and_selection``, ``layer4_draft_critique_and_revise`` and
  ``layer5_score_and_final_selection``; ``--layer`` 1 to 5; cost-log tags ``layer2_plan``,
  ``layer4_critique`` and so on. Every new run is made in it.
* ``FOUR_NUMBERED_LAYERS``: folders ``layer1`` to ``layer4``, ``--layer`` 1 to 4, tags ``layer1_plan``,
  ``layer3_critique`` and so on. A run created in it keeps being written and resumed in it, so its files
  and its cost log stay consistent.
* ``MATRIX_BEFORE_LAYER_RENUMBERING`` (``layer12``, ``layer3`` to ``layer5``) and
  ``FIVE_LAYER_BEFORE_VARIABLES_MATRIX`` (``layer1`` to ``layer5`` with document types and subtypes): read only.

No folder name of one layout equals one of another, so a layout is told from a run's folder names alone
(``layout_of_run_folder``). Code that needs a folder asks for a folder KEY (what the folder holds), never
for a number, because the numbers differ between layouts: the draft and the critique and revise share one
folder in the five-layer layout and have two in the four-layer one.

The shuffle seeds ``layer1_select`` and ``layer1_screen`` are not tags and are not here: they stay literal
in the step modules, whatever a layout calls its tags.
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path

DEAL_AND_SCREEN = "deal_and_screen"
PLAN_AND_DESCRIPTION_REWRITE = "plan_and_description_rewrite"
CONSIDERATION_SCORING_AND_SELECTION = "consideration_scoring_and_selection"
DRAFT = "draft"
CRITIQUE_AND_REVISE = "critique_and_revise"
SCORE = "score"
FOLDER_KEYS = (DEAL_AND_SCREEN, PLAN_AND_DESCRIPTION_REWRITE, CONSIDERATION_SCORING_AND_SELECTION,
               DRAFT, CRITIQUE_AND_REVISE, SCORE)

# The folder key each stage file lives in, by file name (a file never changes folder key between layouts).
FOLDER_KEY_OF_STAGE_FILE = {
    "prompts.jsonl": PLAN_AND_DESCRIPTION_REWRITE, "plans.jsonl": PLAN_AND_DESCRIPTION_REWRITE,
    "drafts.jsonl": DRAFT, "critiques.jsonl": CRITIQUE_AND_REVISE, "rewrites.jsonl": CRITIQUE_AND_REVISE,
    "scores.jsonl": SCORE,
}

# Stage tag of each model-calling step: (four-layer tag, five-layer tag). ``layer1_screen`` is unchanged.
STAGE_TAGS_BY_STEP = {
    "screen": ("layer1_screen", "layer1_screen"),
    "plan": ("layer1_plan", "layer2_plan"),
    "description_rewrite": ("layer1_description_rewrite", "layer2_description_rewrite"),
    "consideration_scoring": ("layer1_select", "layer3_consideration_scoring"),
    "draft": ("layer2_draft", "layer4_draft"),
    "critique": ("layer3_critique", "layer4_critique"),
    "revise": ("layer3_revise", "layer4_revise"),
    "critique_and_revise_in_one_call": ("layer3_rewrite", "layer4_critique_and_revise_in_one_call"),
    "score": ("layer4_score", "layer5_score"),
}
# The name the exit messages of the critique, the revise and the one-call step start with (the layer's name).
CRITIQUE_AND_REVISE_MESSAGE_LABELS = ("layer3_rewrite", "layer4_draft_critique_and_revise")


@dataclass(frozen=True)
class SdfRunLayout:
    name: str
    folder_name_of_key: dict
    layer_number_of_key: dict       # the ``--layer`` number of each folder key (the pre-renumber layout is read only)
    tag_index: int | None           # which element of STAGE_TAGS_BY_STEP the layout uses
    draft_checkpoint_file_name: str = "_checkpoint.json"   # the five-layer layout's draft shares a folder with the critique and revise

    @property
    def layer_numbers(self) -> tuple:
        return tuple(sorted(set(self.layer_number_of_key.values())))

    @property
    def folder_names(self) -> tuple:
        return tuple(dict.fromkeys(self.folder_name_of_key.values()))

    def folder_of(self, run_dir, folder_key: str) -> Path:
        return Path(run_dir) / self.folder_name_of_key[folder_key]

    def folder_key_of_layer(self, layer: int) -> str:
        """The first folder key whose layer number is ``layer``."""
        return next(key for key in FOLDER_KEYS if self.layer_number_of_key[key] == layer)

    def stage_tag(self, step_name: str) -> str:
        return STAGE_TAGS_BY_STEP[step_name][self.tag_index]

    @property
    def critique_and_revise_message_label(self) -> str:
        return CRITIQUE_AND_REVISE_MESSAGE_LABELS[self.tag_index]

    def layer_label(self, layer: int) -> str:
        """What a printed progress line calls a layer: ``Layer 3``."""
        return f"Layer {layer}"


FIVE_NAMED_LAYERS = SdfRunLayout(
    "five_named_layers",
    {DEAL_AND_SCREEN: "layer1_deal_and_screen",
     PLAN_AND_DESCRIPTION_REWRITE: "layer2_plan_and_description_rewrite",
     CONSIDERATION_SCORING_AND_SELECTION: "layer3_consideration_scoring_and_selection",
     DRAFT: "layer4_draft_critique_and_revise", CRITIQUE_AND_REVISE: "layer4_draft_critique_and_revise",
     SCORE: "layer5_score_and_final_selection"},
    {DEAL_AND_SCREEN: 1, PLAN_AND_DESCRIPTION_REWRITE: 2, CONSIDERATION_SCORING_AND_SELECTION: 3,
     DRAFT: 4, CRITIQUE_AND_REVISE: 4, SCORE: 5}, 1, "_draft_checkpoint.json")

FOUR_NUMBERED_LAYERS = SdfRunLayout(
    "four_numbered_layers",
    {DEAL_AND_SCREEN: "layer1", PLAN_AND_DESCRIPTION_REWRITE: "layer1", CONSIDERATION_SCORING_AND_SELECTION: "layer1",
     DRAFT: "layer2", CRITIQUE_AND_REVISE: "layer3", SCORE: "layer4"},
    {DEAL_AND_SCREEN: 1, PLAN_AND_DESCRIPTION_REWRITE: 1, CONSIDERATION_SCORING_AND_SELECTION: 1,
     DRAFT: 2, CRITIQUE_AND_REVISE: 3, SCORE: 4}, 0)

MATRIX_BEFORE_LAYER_RENUMBERING = SdfRunLayout(
    "matrix_before_layer_renumbering",
    {DEAL_AND_SCREEN: "layer12", PLAN_AND_DESCRIPTION_REWRITE: "layer12", CONSIDERATION_SCORING_AND_SELECTION: "layer12",
     DRAFT: "layer3", CRITIQUE_AND_REVISE: "layer4", SCORE: "layer5"},
    {DEAL_AND_SCREEN: 1, PLAN_AND_DESCRIPTION_REWRITE: 1, CONSIDERATION_SCORING_AND_SELECTION: 1,
     DRAFT: 2, CRITIQUE_AND_REVISE: 3, SCORE: 4}, None)

# The oldest layout (before the variables matrix: document types and subtypes in layer1 and layer2, drafts in
# layer3, rewrites in layer4, scores in layer5): read only, and it has no plan, so its plan keys point at layer1.
FIVE_LAYER_BEFORE_VARIABLES_MATRIX = SdfRunLayout(
    "five_layer_before_variables_matrix",
    {DEAL_AND_SCREEN: "layer1", PLAN_AND_DESCRIPTION_REWRITE: "layer1", CONSIDERATION_SCORING_AND_SELECTION: "layer1",
     DRAFT: "layer3", CRITIQUE_AND_REVISE: "layer4", SCORE: "layer5"},
    {DEAL_AND_SCREEN: 1, PLAN_AND_DESCRIPTION_REWRITE: 1, CONSIDERATION_SCORING_AND_SELECTION: 1,
     DRAFT: 2, CRITIQUE_AND_REVISE: 3, SCORE: 4}, None)

READABLE_LAYOUTS = (FIVE_NAMED_LAYERS, FOUR_NUMBERED_LAYERS, MATRIX_BEFORE_LAYER_RENUMBERING,
                    FIVE_LAYER_BEFORE_VARIABLES_MATRIX)
LAYOUT_RECORDED_IN_MANIFEST_KEY = "sdf_stage_folder_layout"


assert not set(FIVE_NAMED_LAYERS.folder_names) & {
    name for layout in (FOUR_NUMBERED_LAYERS, MATRIX_BEFORE_LAYER_RENUMBERING) for name in layout.folder_names}, \
    "a folder name of the five-layer layout equals one of an older layout"
# (The two older layouts share layer3 and layer4 on purpose: layer12 tells them apart.)


class UnknownRunLayoutError(ValueError):
    """The folder is not an SDF run folder in any layout."""


def detect_run_layout(run_dir) -> SdfRunLayout:
    """``layout_of_run_folder`` for a folder that must be an SDF run folder: raises ``UnknownRunLayoutError``
    when ``run_dir`` is not a folder or holds none of the stage folders of a layout."""
    run = Path(run_dir)
    if not run.is_dir():
        raise UnknownRunLayoutError(f"{run} is not a folder, so it is not an SDF run folder")
    layout = layout_of_run_folder(run)
    if layout is None:
        raise UnknownRunLayoutError(
            f"{run} has none of the stage folders of an SDF run (layer1_deal_and_screen to layer5_score_and_final_selection, "
            "layer1 to layer4, layer12, or layer1 to layer5), so its layout is unknown and it was not read")
    return layout


def layout_of_run_folder(run_dir, manifest: dict | None = None) -> SdfRunLayout | None:
    """The layout of an existing run folder, told by its folder names; None when it has no stage folder.

    The five-layer layout is recognised by any of its (suffixed) folder names; ``layer12`` marks the
    pre-renumber layout; bare ``layer1`` to ``layer4`` are the four-layer layout. A run folder with no stage
    folder yet (created, then stopped) takes the layout its manifest recorded, else the four-layer layout
    (runs made before the key existed are four-layer runs); with no manifest either, None."""
    run = Path(run_dir)
    if any((run / name).is_dir() for name in FIVE_NAMED_LAYERS.folder_names):
        return FIVE_NAMED_LAYERS
    if (run / "layer12").is_dir():
        return MATRIX_BEFORE_LAYER_RENUMBERING
    if ((run / "layer5").is_dir() or (run / "layer1" / "document_types.jsonl").is_file()
            or (run / "layer2" / "subtypes.jsonl").is_file()):
        return FIVE_LAYER_BEFORE_VARIABLES_MATRIX
    if any((run / f"layer{number}").is_dir() for number in (1, 2, 3, 4)):
        return FOUR_NUMBERED_LAYERS
    if manifest is None:
        return None
    recorded = manifest.get(LAYOUT_RECORDED_IN_MANIFEST_KEY)
    if recorded:
        return next((layout for layout in READABLE_LAYOUTS if layout.name == recorded), None)
    return FOUR_NUMBERED_LAYERS   # a manifest without the key: a run made before the layouts were recorded


_active_layout = FIVE_NAMED_LAYERS


def active_layout() -> SdfRunLayout:
    """The layout of the run being written (tags and folders); the five-layer layout unless ``run.py`` is
    running a four-layer run."""
    return _active_layout


@contextlib.contextmanager
def using_layout(layout: SdfRunLayout):
    global _active_layout
    before, _active_layout = _active_layout, layout
    try:
        yield layout
    finally:
        _active_layout = before


def five_layer_tag_of(any_tag: str) -> str:
    """The five-layer layout's tag for a tag of either layout (a tag that is the same in both, or unknown, is returned as it is)."""
    for old_tag, new_tag in STAGE_TAGS_BY_STEP.values():
        if any_tag == old_tag:
            return new_tag
    return any_tag


def value_for_tag(value_by_tag: dict, tag: str):
    """``value_by_tag[tag]``, else the value under the same step's tag in the other layout, else None: a config's
    per-tag map (``effort``) written with the old tags keeps applying to the new tags and the other way round."""
    if tag in value_by_tag:
        return value_by_tag[tag]
    for old_tag, new_tag in STAGE_TAGS_BY_STEP.values():
        if tag in (old_tag, new_tag):
            return value_by_tag.get(new_tag if tag == old_tag else old_tag)
    return None
