"""Which run layouts ``--resume`` accepts: the five named layers and the four numbered layers. Other layouts can be read but not resumed."""

from pathlib import Path

from shared import sdf_run_layouts


def layout_of_run_that_can_be_resumed(run_dir: Path) -> sdf_run_layouts.SdfRunLayout:
    """The run's layout, or exit before any folder is created or call made when it cannot be resumed (a resume would create a second set of folders)."""
    try:
        layout = sdf_run_layouts.detect_run_layout(run_dir)
    except sdf_run_layouts.UnknownRunLayoutError as error:
        raise SystemExit(f"Cannot resume {run_dir.name}: {error}")
    if layout in (sdf_run_layouts.FIVE_NAMED_LAYERS, sdf_run_layouts.FOUR_NUMBERED_LAYERS):
        return layout
    raise SystemExit(
        f"Cannot resume {run_dir.name}: it is in the {layout.name.upper()} layout. Runs in that layout can be read "
        "(report, audits, viewer) but not resumed; only runs in the five-layer layout (layer1_deal_variables_and_screen to "
        "layer5_score_and_final_selection) or the four-layer layout (layer1 to layer4) can be.")
