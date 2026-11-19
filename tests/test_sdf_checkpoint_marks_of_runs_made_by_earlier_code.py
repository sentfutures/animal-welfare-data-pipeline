"""The checkpoint marks of a run made by earlier code are read from the file each step used."""

import pytest

from sdf_pipeline.compatibility_with_runs_made_by_earlier_code.checkpoint_marks import marked_as_finished
from shared import sdf_run_layouts, utils


def mark(stage_directory, file_name, *ids):
    checkpoint = utils.Checkpoint(stage_directory / file_name)
    for document_id in ids:
        checkpoint.mark_done(document_id)


@pytest.mark.parametrize("step_name, file_name", [
    ("screen", "_screen_checkpoint.json"), ("description_rewrite", "_description_rewrite_checkpoint.json"),
    ("plan", "_checkpoint.json"), ("revise", "_checkpoint.json"), ("score", "_checkpoint.json")])
def test_each_step_reads_its_own_checkpoint_file(tmp_path, step_name, file_name):
    mark(tmp_path, file_name, "a", "b")
    assert marked_as_finished(step_name, tmp_path) == {"a", "b"}


def test_a_folder_without_the_checkpoint_file_is_none(tmp_path):
    assert marked_as_finished("plan", tmp_path) is None


def test_consideration_scoring_has_no_checkpoint_file(tmp_path):
    mark(tmp_path, "_checkpoint.json", "a")
    assert marked_as_finished("consideration_scoring", tmp_path) is None


def test_the_draft_reads_the_file_its_layout_gave_it(tmp_path):
    mark(tmp_path, "_checkpoint.json", "from_the_shared_file")
    mark(tmp_path, "_draft_checkpoint.json", "from_the_draft_file")
    assert marked_as_finished("draft", tmp_path) == {"from_the_draft_file"}
    with sdf_run_layouts.using_layout(sdf_run_layouts.FOUR_NUMBERED_LAYERS):
        assert marked_as_finished("draft", tmp_path) == {"from_the_shared_file"}
