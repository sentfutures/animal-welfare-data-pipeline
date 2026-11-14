"""Reads the checkpoint files runs made by earlier code left beside their stage files; a marked document (or screen batch) counts as finished.
A run that has such a file keeps it up to date, because earlier code resumed by its marks alone; runs made by current code write none."""

import json
from pathlib import Path
from typing import Callable

from shared import sdf_run_layouts, utils

_CHECKPOINT_FILE_NAME_OF_STEP = {
    "screen": "_screen_checkpoint.json",
    "description_rewrite": "_description_rewrite_checkpoint.json",
}


def _checkpoint_file_name(step_name: str) -> str | None:
    if step_name == "consideration_scoring":
        return None   # its score file was its only record
    if step_name == "draft" and sdf_run_layouts.active_layout() is sdf_run_layouts.FIVE_NAMED_LAYERS:
        return "_draft_checkpoint.json"   # the draft shares its folder with the critique and revise
    return _CHECKPOINT_FILE_NAME_OF_STEP.get(step_name, "_checkpoint.json")


def marked_as_finished(step_name: str, stage_directory: Path) -> set | None:
    """The ids (screen: batch ids) the step's checkpoint file in ``stage_directory`` marks done, or None when it has no checkpoint file there."""
    name = _checkpoint_file_name(step_name)
    path = stage_directory / name if name else None
    if not path or not path.exists():
        return None
    return {str(key) for key in json.loads(path.read_text(encoding="utf-8"))["completed"]}


def keep_marks_up_to_date(step_name: str, stage_directory: Path, finished_keys: Callable[[], set]) -> None:
    """Mark the keys the stage files show finished in the step's checkpoint file, when the run has one; nothing otherwise.
    Without it a second resume would trust the file alone and send again what the first resume finished."""
    marked = marked_as_finished(step_name, stage_directory)
    if marked is None:
        return
    checkpoint = utils.Checkpoint(stage_directory / _checkpoint_file_name(step_name))
    for key in sorted(finished_keys() - marked):
        checkpoint.mark_done(key)
