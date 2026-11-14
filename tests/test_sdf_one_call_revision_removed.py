"""``sdf.revision.split: false`` (the removed one-call revision) is refused at start-up and when resuming a run saved with it
before its layer 4 finished; ``split: true`` is accepted and ignored."""

import json
from pathlib import Path

import pytest
import yaml

import sdf_pipeline.run as sdf_run
from experiments import experiment
from sdf_pipeline.compatibility_with_runs_made_by_earlier_code.one_call_revision_removed import (
    refuse_configuration_with_one_call_revision, refuse_resuming_run_with_one_call_revision)
from sdf_step_and_command_line_test_helpers import run_main
from shared import utils


@pytest.fixture
def outputs_root(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(root))
    return root


def config_file_with_revision(tiny_config, tmp_path, revision):
    tiny_config["sdf"]["revision"] = revision
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    return path


def saved_run(outputs_root, tiny_config, split, draft_count, rewrite_count) -> Path:
    """A five-layer run folder whose saved configuration sets ``split`` (None: the key is absent)."""
    run_directory = outputs_root / "sdf" / "runs" / "2026-01-01_00-00_saved"
    stage = run_directory / "layer4_draft_critique_and_revise"
    stage.mkdir(parents=True)
    tiny_config["sdf"]["revision"] = {} if split is None else {"split": split}
    (run_directory / "run_manifest.json").write_text(json.dumps({"config": tiny_config}))
    utils.save_jsonl([{"doc_id": f"d{n}"} for n in range(draft_count)], stage / "drafts.jsonl")
    utils.save_jsonl([{"doc_id": f"d{n}"} for n in range(rewrite_count)], stage / "rewrites.jsonl")
    return run_directory


def test_a_configuration_with_the_one_call_revision_is_refused_before_any_folder_or_call(
        tiny_config, tmp_path, outputs_root, stub_claude, monkeypatch):
    calls = stub_claude([])
    config = config_file_with_revision(tiny_config, tmp_path, {"split": False})
    with pytest.raises(SystemExit) as refusal:
        run_main(monkeypatch, config, "--label", "refused", main=sdf_run.main)
    assert str(refusal.value) == (f"{config} sets sdf.revision.split: false, but the one-call revision was removed: layer 4 "
                                  "always critiques and then revises in two calls. Remove the sdf.revision.split key.")
    assert calls == [] and not outputs_root.exists()


def test_split_true_is_accepted_and_ignored():
    refuse_configuration_with_one_call_revision({"sdf": {"revision": {"split": True}}})
    refuse_configuration_with_one_call_revision({"sdf": {"revision": None}})
    refuse_configuration_with_one_call_revision({"sdf": {}})


def test_resuming_a_run_saved_with_the_one_call_revision_before_its_layer_4_finished_is_refused(
        tiny_config, outputs_root, stub_claude, monkeypatch, tmp_path):
    run_directory = saved_run(outputs_root, tiny_config, split=False, draft_count=2, rewrite_count=1)
    calls = stub_claude([])
    config = config_file_with_revision(tiny_config, tmp_path, {})
    with pytest.raises(SystemExit) as refusal:
        run_main(monkeypatch, config, "--resume", "--run-id", run_directory.name, main=sdf_run.main)
    assert str(refusal.value) == (
        f"Cannot resume {run_directory.name}: its saved configuration sets sdf.revision.split: false, but the one-call "
        "revision was removed, and its layer 4 is not finished. Only a run whose layer 4 is finished can be resumed "
        "(from layer 5), or start a new run.")
    assert calls == []


@pytest.mark.parametrize("split, draft_count, rewrite_count", [
    (False, 2, 2),   # layer 4 finished
    (True, 2, 0),    # saved with the ignored value
    (None, 2, 0),    # saved without the key
])
def test_other_runs_are_not_refused(tiny_config, outputs_root, split, draft_count, rewrite_count):
    refuse_resuming_run_with_one_call_revision(saved_run(outputs_root, tiny_config, split, draft_count, rewrite_count))


def test_an_experiment_arm_with_the_one_call_revision_is_refused():
    with pytest.raises(SystemExit, match="arm 'control' sets sdf.revision.split: false, but the one-call revision was removed"):
        experiment.merge_arm_config({"sdf": {}}, {"name": "control", "split": False})
