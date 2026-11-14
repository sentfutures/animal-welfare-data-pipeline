"""select_model and rewrite_model are the old names of consideration_scoring_model and revise_model."""

import pytest

from sdf_pipeline import pipeline_steps
from sdf_pipeline.shared_step_machinery import renamed_model_settings
from sdf_pipeline.layer4_draft_critique_and_revise.run_layer import CritiqueStep, ReviseStep
from sdf_pipeline.layer3_consideration_scoring_and_selection.run_layer import ConsiderationScoringStep


@pytest.fixture(autouse=True)
def announce_again():
    renamed_model_settings._keys_already_announced.clear()


@pytest.mark.parametrize("new_key, old_key, step_name", [
    ("consideration_scoring_model", "select_model", "consideration_scoring"),
    ("revise_model", "rewrite_model", "revise")])
def test_new_key_alone_old_key_alone_and_both(new_key, old_key, step_name, capsys):
    entry = pipeline_steps.entry_named(step_name)
    assert entry.model_for({"sdf": {new_key: "new-model"}}) == "new-model"
    assert "Notice" not in capsys.readouterr().out

    assert entry.model_for({"sdf": {old_key: "old-model"}}) == "old-model"
    notice = capsys.readouterr().out
    assert notice.count("\n") == 1 and new_key in notice and old_key in notice

    renamed_model_settings._keys_already_announced.clear()
    assert entry.model_for({"sdf": {old_key: "old-model", new_key: "new-model"}}) == "new-model"
    assert "ignored" in capsys.readouterr().out


def test_step_classes_read_the_new_and_the_old_key(tiny_config, prompts_sdf, capsys):
    sdf = tiny_config["sdf"]
    for key in ("select_model", "rewrite_model", "revise_model", "consideration_scoring_model", "critique_model"):
        sdf.pop(key, None)
    sdf["rewrite_model"] = "old-revise"
    assert ReviseStep(tiny_config, prompts_sdf, {}).revise_model == "old-revise"
    assert CritiqueStep(tiny_config, prompts_sdf).critique_model == "old-revise"
    sdf["revise_model"] = "new-revise"
    assert ReviseStep(tiny_config, prompts_sdf, {}).revise_model == "new-revise"
    assert CritiqueStep(tiny_config, prompts_sdf).critique_model == "new-revise"
