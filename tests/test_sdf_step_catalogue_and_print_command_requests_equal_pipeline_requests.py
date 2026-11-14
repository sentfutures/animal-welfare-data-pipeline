"""The step catalogue (sdf_pipeline/pipeline_steps.py) agrees with the step classes, the layer modules, the
prompt snapshot and the request print command, and the print command builds, for every stage and every
document of an equivalence scenario, the very request the pipeline sent (system prompt, user message,
schema, model, token limit, temperature). No model API is called."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

import sdf_prompt_equivalence_fixture_support as support   # imported first: captures the real network seams
import sdf_prompt_equivalence_scenarios as scenarios
from sdf_pipeline import (
    layer1_screen, layer1_select_considerations, pipeline_steps, seeding)
from sdf_pipeline.document_state import PipelineStep
from shared import utils

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
COMMAND_PATH = REPOSITORY_ROOT / "scripts" / "print_full_sdf_request_a_stage_would_send_for_one_document.py"
_specification = importlib.util.spec_from_file_location("print_full_sdf_request_command_for_catalogue_test", COMMAND_PATH)
command = importlib.util.module_from_spec(_specification)
_specification.loader.exec_module(command)

ENTRIES = pipeline_steps.PIPELINE_STEPS


# ------------------------------------------------------------------ the catalogue against the code

def test_the_catalogue_lists_every_pipeline_step_in_order_and_the_single_call_step():
    names = [entry.name for entry in ENTRIES]
    assert len(set(names)) == len(names)
    enum_names = [step.value for step in PipelineStep if step is not PipelineStep.COMPOSE_PLANNING_PROMPT]
    # the enum has no member for the redeal or the single-call step
    assert [name for name in names if name not in ("redeal", "combined_review_and_revise")] == enum_names
    layers = [entry.run_py_layer for entry in ENTRIES]
    assert layers == sorted(layers) and set(layers) == set(pipeline_steps.LAYER_NUMBERS)


def test_stage_tags_and_step_names_equal_the_step_classes():
    step_classes = {
        "deal": layer1_screen.DealStep, "screen": layer1_screen.ScreenStep, "redeal": layer1_screen.RedealStep,
        "consideration_assignment": layer1_select_considerations.ConsiderationAssignmentStep}
    from sdf_pipeline import layer1_description_rewrite, layer1_plan, layer2_draft, layer3_rewrite, layer4_score
    step_classes.update({
        "plan": layer1_plan.PlanStep, "description_rewrite": layer1_description_rewrite.DescriptionRewriteStep,
        "consideration_scoring": layer1_select_considerations.ConsiderationScoringStep,
        "draft": layer2_draft.DraftStep, "critique": layer3_rewrite.CritiqueStep, "revise": layer3_rewrite.ReviseStep,
        "combined_review_and_revise": layer3_rewrite.CombinedReviewAndReviseStep, "score": layer4_score.ScoreStep,
        "gate": layer4_score.ScoreGateStep, "near_duplicate_cull": layer4_score.NearDuplicateCullStep})
    assert set(step_classes) == {entry.name for entry in ENTRIES}
    for entry in ENTRIES:
        step_class = step_classes[entry.name]
        assert step_class.step_name == entry.name
        recorded_tag = getattr(step_class, "stage_tag", None) or getattr(step_class, "stage", None)
        assert recorded_tag == entry.stage_tag, entry.name
        assert step_class.__module__ == f"sdf_pipeline.{entry.module_name}", entry.name


def test_the_shuffle_seeds_stay_the_literal_old_stage_tags_the_layers_derive_their_seeds_from():
    assert pipeline_steps.entry_named("screen").shuffle_seed == "layer1_screen" == layer1_screen.STAGE_TAG
    assert pipeline_steps.entry_named("consideration_scoring").shuffle_seed == "layer1_select" \
        == layer1_select_considerations.STAGE_TAG
    assert {entry.name for entry in ENTRIES if entry.shuffle_seed} == {"screen", "consideration_scoring"}


def test_each_step_is_built_with_the_model_the_catalogue_resolves(tmp_path):
    run_directory, config = command._run_fixture_scenario_in_temporary_folder(
        "temperature_kept_on_every_stage_with_a_two_call_revision", tmp_path)
    prompts_directory = command._prompts_directory_of_run(run_directory)
    reading = pipeline_steps.RunReading(
        config, prompts_directory,
        {n: utils.sdf_stage_dir_of_existing_run(run_directory, n) for n in pipeline_steps.LAYER_NUMBERS})
    models_by_key = {key: f"model-for-{key}" for entry in ENTRIES for key in entry.model_setting_keys}
    for configuration in (config, {**config, "sdf": {**config["sdf"], **models_by_key}},
                          {**config, "sdf": {**config["sdf"], **{k: None for k in models_by_key if k != "plan_model"},
                                             "plan_model": "only-the-plan-model"}}):
        reading_here = pipeline_steps.RunReading(configuration, prompts_directory, reading.stage_directories)
        for entry in ENTRIES:
            if entry.build_step is None:
                continue
            documents = entry.load_documents(reading_here)
            step = entry.build_step(reading_here, documents, documents[0].get("doc_id") or documents[0]["prompt_id"])
            request = step.build_request(documents[0])
            if request.model is not None or entry.model_for(configuration) is not None:
                assert request.model == entry.model_for(configuration), entry.name


def test_every_required_prompt_file_of_a_default_run_is_in_the_prompts_folder():
    config = utils.load_config(str(REPOSITORY_ROOT / "config.yaml"))
    prompts_directory = REPOSITORY_ROOT / "prompts" / "sdf"
    for entry in ENTRIES:
        for name in entry.required_prompt_files(config, prompts_directory):
            assert (prompts_directory / name).is_file(), (entry.name, name)


def test_seeding_lists_the_steps_in_pipeline_order_and_names_the_reason_for_the_rest():
    assert list(seeding.SEEDABLE_STEPS) == [
        "plan", "description_rewrite", "consideration_scoring", "draft", "critique", "revise", "score"]
    assert set(seeding.UNSEEDABLE_STEPS) == {
        "deal", "screen", "redeal", "consideration_assignment", "gate", "near_duplicate_cull"}
    assert "combined_review_and_revise" not in seeding.SEEDABLE_STEPS | seeding.UNSEEDABLE_STEPS


def test_the_print_command_stage_names_are_the_catalogue_s():
    assert set(command.STAGES_THIS_COMMAND_COVERS) == {e.print_command_name for e in ENTRIES if e.print_command_name}
    assert command.STAGES_THIS_COMMAND_CANNOT_COVER_YET == {"gate": "makes no model call",
                                                           "near_duplicate_cull": "makes no model call"}


def test_a_run_whose_snapshot_lacks_a_template_of_an_active_step_is_refused_before_any_call(tmp_path):
    config = utils.load_config(str(REPOSITORY_ROOT / "config.yaml"))
    prompts_directory = tmp_path / "prompts"
    prompts_directory.mkdir()
    for source in (REPOSITORY_ROOT / "prompts" / "sdf").iterdir():
        if source.is_file():
            (prompts_directory / source.name).write_bytes(source.read_bytes())
    assert pipeline_steps.prompt_files_missing_from_snapshot(2, config, prompts_directory) == []
    (prompts_directory / "layer2.txt").unlink()
    assert pipeline_steps.prompt_files_missing_from_snapshot(2, config, prompts_directory) == [("draft", ["layer2.txt"])]


# ------------------------------------------------------------------ print command against the pipeline's own requests

SCENARIOS = ["temperature_kept_on_every_stage_with_a_two_call_revision",
             "temperature_kept_on_every_stage_with_a_one_call_revision",
             "minimal_config_with_openai_rewrite_and_adaptive_thinking_models"]
STAGE_OF_TAG = {"layer1_screen": "screen", "layer1_plan": "plan", "layer1_description_rewrite": "description_rewrite",
                "layer1_select": "select", "layer2_draft": "draft", "layer3_critique": "critique",
                "layer3_revise": "revise", "layer3_rewrite": "combined", "layer4_score": "score"}


@pytest.fixture(scope="module", params=SCENARIOS)
def scenario_outcome(request, tmp_path_factory):
    outcome = support.run_scenario_in_isolation(
        scenarios.scenario_named(request.param), tmp_path_factory.mktemp("scenario"))
    return outcome


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _system_hash_of_wire(record: dict) -> str:
    system = record["wire"]["system" if record["route"] == "anthropic" else "instructions"]
    return system["text_sha256"] if system["blocks"] is None else system["blocks"][0]["text_sha256"]


def test_the_print_command_builds_the_request_the_pipeline_sent_for_every_stage_and_document(scenario_outcome):
    outcome = scenario_outcome
    config = outcome.configs_used[-1]
    run_directory = outcome.run_directory
    batches = utils.load_jsonl(utils.sdf_stage_dir(run_directory, 1) / layer1_screen.BATCHES_FILENAME)
    compared_stages = set()
    for record in outcome.recorder.records:
        if record["attempt"] != 1 or record["stage"] not in STAGE_OF_TAG:
            continue
        stage = STAGE_OF_TAG[record["stage"]]
        document_id = record["item_id"]
        if stage == "screen":
            document_id = next(b["prompt_ids"][0] for b in batches if b["batch_id"] == record["item_id"])
        full_request = command.build_full_request(run_directory, config, stage, document_id)
        request, settings, wire = full_request.request, full_request.settings, record["wire"]
        assert request.stage == record["stage"] and request.item_id == record["item_id"]
        assert settings.provider == record["route"], (stage, document_id)
        assert _sha(request.system_prompt) == _system_hash_of_wire(record), (stage, document_id)
        if record["route"] == "anthropic":
            assert request.user_message == wire["messages"][0]["content"], (stage, document_id)
            assert (settings.model, settings.max_tokens, settings.temperature) == (
                wire["model"], wire["max_tokens"], wire.get("temperature")), (stage, document_id)
            recorded_schema = (wire.get("output_config") or {}).get("format", {}).get("schema")
        else:
            assert request.user_message == wire["input"], (stage, document_id)
            assert (settings.model, settings.max_tokens) == (wire["model"], wire["max_output_tokens"])
            recorded_schema = ((wire.get("text") or {}).get("format") or {}).get("schema")
        assert request.json_schema == recorded_schema, (stage, document_id)
        compared_stages.add(stage)
    assert compared_stages >= {"plan", "description_rewrite", "select", "draft", "score"}
    assert ("combined" in compared_stages) != ("critique" in compared_stages)
