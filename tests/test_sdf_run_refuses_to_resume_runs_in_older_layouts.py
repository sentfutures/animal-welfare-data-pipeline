"""``run.py --resume`` on a run in the five-layer or pre-renumber layout refuses before creating any folder
or making any call; such runs can be read but not resumed. Offline: the run folders are minimal."""

import sys

import pytest

import sdf_pipeline.run as sdf_run

RUN_NAME = "2026-07-01_10-00_older_layout_run"


def _make_run(runs_root, stage_folder_names):
    run = runs_root / RUN_NAME
    for name in stage_folder_names:
        (run / name).mkdir(parents=True)
    return run


def _entries_under(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


@pytest.mark.parametrize("stage_folder_names, layout_name", [
    (("layer1", "layer2", "layer3", "layer4", "layer5"), "FIVE_LAYER_BEFORE_VARIABLES_MATRIX"),
    (("layer12", "layer3", "layer4", "layer5"), "MATRIX_BEFORE_LAYER_RENUMBERING"),
])
@pytest.mark.parametrize("extra_arguments", [[], ["--run-id", RUN_NAME], ["--layer", "2"]])
def test_resuming_a_run_in_an_older_layout_refuses_before_creating_anything_or_calling_a_model(
        tmp_path, monkeypatch, tiny_config_file, stage_folder_names, layout_name, extra_arguments):
    outputs_root = tmp_path / "outputs"
    run = _make_run(outputs_root / "sdf" / "runs", stage_folder_names)
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(outputs_root))
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(tiny_config_file), "--resume", *extra_arguments])
    entries_before = _entries_under(outputs_root)

    # No stub for call_claude: the suite's blocker raises if a call were attempted.
    with pytest.raises(SystemExit) as refusal:
        sdf_run.main()

    message = str(refusal.value)
    assert RUN_NAME in message and layout_name in message
    assert "read" in message and "not resumed" in message
    assert _entries_under(outputs_root) == entries_before
    assert not (run / "inputs").exists() and not (run / "cost_log.jsonl").exists()


def test_a_folder_with_no_stage_folders_is_refused_as_an_unknown_layout(tmp_path, monkeypatch, tiny_config_file):
    outputs_root = tmp_path / "outputs"
    _make_run(outputs_root / "sdf" / "runs", ())
    (outputs_root / "sdf" / "runs" / RUN_NAME).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(outputs_root))
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(tiny_config_file), "--resume"])
    with pytest.raises(SystemExit) as refusal:
        sdf_run.main()
    assert "layout is unknown" in str(refusal.value)
