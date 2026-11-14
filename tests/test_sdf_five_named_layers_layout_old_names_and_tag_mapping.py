"""Decision 17: the five named layers layout, how a run's layout is told, the old names that still work, and the
tag mapping between the four-layer and the five-layer cost-log tags. Offline: nothing here calls a model."""

import importlib
import json

import pytest

from sdf_pipeline.layer4_draft_critique_and_revise import run_layer as draft_critique_and_revise_layer
from sdf_pipeline import pipeline_steps
from sdf_pipeline.shared_step_machinery.document_state_loader import RunLayout, detect_run_layout
from sdf_plan_draft_rewrite_and_score_test_builders import make_plan
from shared import sdf_run_layouts, utils
from viewer import loader as viewer_loader

FIVE = sdf_run_layouts.FIVE_NAMED_LAYERS
FOUR = sdf_run_layouts.FOUR_NUMBERED_LAYERS


# ------------------------------------------------------------------ how a run's layout is told

def _folders(root, *names):
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.parametrize("folder_names, expected_layout, expected_loader_layout", [
    (("layer1_deal_and_screen", "layer2_plan_and_description_rewrite", "layer3_consideration_scoring_and_selection",
      "layer4_draft_critique_and_revise", "layer5_score_and_final_selection"),
     sdf_run_layouts.FIVE_NAMED_LAYERS, RunLayout.FIVE_NAMED_LAYERS),
    (("layer1", "layer2", "layer3", "layer4"), sdf_run_layouts.FOUR_NUMBERED_LAYERS, RunLayout.FOUR_NUMBERED_LAYERS),
    (("layer12", "layer3", "layer4", "layer5"), sdf_run_layouts.MATRIX_BEFORE_LAYER_RENUMBERING,
     RunLayout.MATRIX_BEFORE_LAYER_RENUMBERING),
    (("layer1", "layer2", "layer3", "layer4", "layer5"), sdf_run_layouts.FIVE_LAYER_BEFORE_VARIABLES_MATRIX,
     RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX),
])
def test_a_layout_is_told_from_the_folder_names_alone(tmp_path, folder_names, expected_layout, expected_loader_layout):
    run = _folders(tmp_path, *folder_names)
    assert sdf_run_layouts.layout_of_run_folder(run) is expected_layout
    assert utils.sdf_layout_of_run(run) is expected_layout
    assert detect_run_layout(run) is expected_loader_layout


def test_no_folder_name_of_the_five_named_layers_equals_a_folder_name_of_an_older_layout():
    older_names = {name for layout in sdf_run_layouts.READABLE_LAYOUTS if layout is not FIVE for name in layout.folder_names}
    assert not set(FIVE.folder_names) & older_names


def test_the_five_named_layers_have_five_layer_numbers_and_the_draft_shares_a_folder_with_the_revise():
    assert FIVE.layer_numbers == (1, 2, 3, 4, 5)
    assert FOUR.layer_numbers == (1, 2, 3, 4)
    assert FIVE.folder_name_of_key[sdf_run_layouts.DRAFT] == FIVE.folder_name_of_key[sdf_run_layouts.CRITIQUE_AND_REVISE]
    assert FOUR.folder_name_of_key[sdf_run_layouts.DRAFT] != FOUR.folder_name_of_key[sdf_run_layouts.CRITIQUE_AND_REVISE]
    assert FIVE.draft_checkpoint_file_name != "_checkpoint.json" and FOUR.draft_checkpoint_file_name == "_checkpoint.json"


def test_a_run_with_no_stage_folder_takes_the_layout_its_manifest_recorded_and_a_manifest_without_the_key_is_four_layer(tmp_path):
    recorded = {sdf_run_layouts.LAYOUT_RECORDED_IN_MANIFEST_KEY: FIVE.name}
    assert sdf_run_layouts.layout_of_run_folder(tmp_path, recorded) is FIVE
    assert sdf_run_layouts.layout_of_run_folder(tmp_path, {}) is FOUR
    assert sdf_run_layouts.layout_of_run_folder(tmp_path) is None
    (tmp_path / "run_manifest.json").write_text(json.dumps(recorded), encoding="utf-8")
    assert utils.sdf_layout_of_run(tmp_path) is FIVE


# ------------------------------------------------------------------ tags

def test_every_tag_of_the_decision_table_is_the_five_layer_tag_of_its_step():
    expected = {"screen": "layer1_screen", "plan": "layer2_plan", "description_rewrite": "layer2_description_rewrite",
                "consideration_scoring": "layer3_consideration_scoring", "draft": "layer4_draft",
                "critique": "layer4_critique", "revise": "layer4_revise",
                "critique_and_revise_in_one_call": "layer4_critique_and_revise_in_one_call", "score": "layer5_score"}
    assert {name: FIVE.stage_tag(name) for name in expected} == expected
    assert {entry.name: entry.stage_tag_in(FIVE) for entry in pipeline_steps.PIPELINE_STEPS if entry.makes_model_calls} == expected


def test_the_four_layer_tags_are_the_old_ones():
    assert {name: FOUR.stage_tag(name) for name in sdf_run_layouts.STAGE_TAGS_BY_STEP} == {
        "screen": "layer1_screen", "plan": "layer1_plan", "description_rewrite": "layer1_description_rewrite",
        "consideration_scoring": "layer1_select", "draft": "layer2_draft", "critique": "layer3_critique",
        "revise": "layer3_revise", "critique_and_revise_in_one_call": "layer3_rewrite", "score": "layer4_score"}


def test_a_tag_of_either_layout_maps_to_the_five_layer_tag_and_other_tags_are_left_alone():
    for old_tag, new_tag in sdf_run_layouts.STAGE_TAGS_BY_STEP.values():
        assert sdf_run_layouts.five_layer_tag_of(old_tag) == new_tag == sdf_run_layouts.five_layer_tag_of(new_tag)
        assert pipeline_steps.canonical_stage_tag(old_tag) == new_tag == pipeline_steps.canonical_stage_tag(new_tag)
    assert pipeline_steps.canonical_stage_tag("sage_eval_judge") == "sage_eval_judge"


def test_a_per_tag_setting_written_with_the_old_tags_applies_to_the_new_tags_and_the_reverse():
    assert sdf_run_layouts.value_for_tag({"layer1_select": "low"}, "layer3_consideration_scoring") == "low"
    assert sdf_run_layouts.value_for_tag({"layer3_consideration_scoring": "high"}, "layer1_select") == "high"
    assert sdf_run_layouts.value_for_tag({"layer3_rewrite": "x"}, "layer4_critique_and_revise_in_one_call") == "x"
    assert sdf_run_layouts.value_for_tag({"layer4_draft": "a", "layer2_draft": "b"}, "layer4_draft") == "a"
    assert sdf_run_layouts.value_for_tag({"layer4_draft": "a"}, "layer5_score") is None


def test_the_active_layout_decides_the_tag_a_step_class_sends():
    from sdf_pipeline.layer2_plan_and_description_rewrite.plan import PlanStep
    assert PlanStep.stage_tag == "layer2_plan"
    with sdf_run_layouts.using_layout(FOUR):
        assert PlanStep.stage_tag == "layer1_plan"
    assert PlanStep.stage_tag == "layer2_plan"


def test_the_shuffle_seeds_stay_the_literal_old_strings_whatever_the_layout():
    from sdf_pipeline.layer1_deal_and_screen import shared_by_this_layers_steps as deal_and_screen_shared_names
    from sdf_pipeline.layer3_consideration_scoring_and_selection import consideration_scoring
    assert deal_and_screen_shared_names.SHUFFLE_SEED_NAME == "layer1_screen"
    assert consideration_scoring.SHUFFLE_SEED_NAME == "layer1_select"


# ------------------------------------------------------------------ one file per step

def test_every_catalogue_step_has_one_module_named_after_it_in_its_layers_folder_and_the_run_layer_modules_hold_only_drivers():
    for entry in pipeline_steps.PIPELINE_STEPS:
        module = importlib.import_module(f"sdf_pipeline.{entry.module_name}")
        layer_folder_name = FIVE.folder_name_of_key[entry.folder_key]
        assert entry.module_name == f"{layer_folder_name}.{entry.name}" and module.__name__.endswith(entry.name)
    for layer_folder_name in FIVE.folder_names:
        module = importlib.import_module(f"sdf_pipeline.{layer_folder_name}.run_layer")
        classes_defined_here = [value for value in vars(module).values()
                                if isinstance(value, type) and value.__module__ == module.__name__]
        assert classes_defined_here == [], layer_folder_name


# ------------------------------------------------------------------ the layer 4 driver

def test_the_layer_four_driver_drafts_then_critiques_and_revises_in_the_one_folder_with_two_checkpoints(
        tiny_config, prompts_sdf, tmp_path, stub_claude):
    stage_folder = tmp_path / FIVE.folder_name_of_key[sdf_run_layouts.DRAFT]
    stage_folder.mkdir()

    def dispatch(user_message, **kwargs):
        if "<improved_document>" in user_message or "improved_document" in kwargs.get("system_prompt", ""):
            return "reasoning\n<improved_document>Better.</improved_document>"
        return "<document>Doc.</document>"
    calls = stub_claude(dispatch)
    rewrites = draft_critique_and_revise_layer.run(tiny_config, prompts_sdf, stage_folder, [make_plan()])
    assert len(rewrites) == 1
    assert {call["stage"] for call in calls} >= {"layer4_draft"}
    assert (stage_folder / "drafts.jsonl").is_file() and (stage_folder / "rewrites.jsonl").is_file()
    assert (stage_folder / FIVE.draft_checkpoint_file_name).is_file() and (stage_folder / "_checkpoint.json").is_file()
    calls_before_resume = len(calls)
    resumed = draft_critique_and_revise_layer.run(tiny_config, prompts_sdf, stage_folder, [make_plan()])
    assert [record["doc_id"] for record in resumed] == [record["doc_id"] for record in rewrites]
    assert len(calls) == calls_before_resume


# ------------------------------------------------------------------ the viewer's layout table

def test_the_viewer_reads_a_five_named_layers_run_and_the_older_layouts(tmp_path):
    five_run = tmp_path / "five"
    four_run = tmp_path / "four"
    for run, folder, file_name in (
            (five_run, "layer2_plan_and_description_rewrite", "plans.jsonl"),
            (five_run, "layer4_draft_critique_and_revise", "drafts.jsonl"),
            (five_run, "layer4_draft_critique_and_revise", "rewrites.jsonl"),
            (five_run, "layer5_score_and_final_selection", "scores.jsonl"),
            (four_run, "layer1", "plans.jsonl"), (four_run, "layer2", "drafts.jsonl"),
            (four_run, "layer3", "rewrites.jsonl"), (four_run, "layer4", "scores.jsonl")):
        (run / folder).mkdir(parents=True, exist_ok=True)
        (run / folder / file_name).write_text(json.dumps({"stage_file": f"{folder}/{file_name}"}) + "\n", encoding="utf-8")
    for run in (five_run, four_run):
        for stage in ("plan", "draft", "rewrite", "score"):
            rows = viewer_loader.load_stage(run, "sdf", stage)
            assert len(rows) == 1, (run.name, stage)
    assert viewer_loader.load_stage(five_run, "sdf", "score")[0]["stage_file"] == "layer5_score_and_final_selection/scores.jsonl"
    assert viewer_loader.load_stage(four_run, "sdf", "score")[0]["stage_file"] == "layer4/scores.jsonl"
