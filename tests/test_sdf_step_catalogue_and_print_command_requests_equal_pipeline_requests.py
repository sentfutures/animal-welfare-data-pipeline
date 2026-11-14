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
from sdf_pipeline import layer1_deal_and_screen_shared, pipeline_steps, seeding
from sdf_pipeline.document_state import PipelineStep
from sdf_pipeline.steps import (
    consideration_scoring, consideration_selection, critique, critique_and_revise_in_one_call, deal, description_rewrite,
    draft, near_duplicate_removal, plan, redeal, revise, score, score_gate, screen)
from shared import sdf_run_layouts, utils

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
    assert [name for name in names if name not in ("redeal", "critique_and_revise_in_one_call")] == enum_names
    for layout, layer_numbers in ((sdf_run_layouts.FIVE_NAMED_LAYERS, {1, 2, 3, 4, 5}),
                                  (sdf_run_layouts.FOUR_NUMBERED_LAYERS, {1, 2, 3, 4})):
        layers = [entry.layer_in(layout) for entry in ENTRIES]
        assert layers == sorted(layers) and set(layers) == layer_numbers


def test_stage_tags_and_step_names_equal_the_step_classes():
    step_classes = {
        "deal": deal.DealStep, "screen": screen.ScreenStep, "redeal": redeal.RedealStep,
        "consideration_selection": consideration_selection.ConsiderationSelectionStep}
    step_classes.update({
        "plan": plan.PlanStep, "description_rewrite": description_rewrite.DescriptionRewriteStep,
        "consideration_scoring": consideration_scoring.ConsiderationScoringStep,
        "draft": draft.DraftStep, "critique": critique.CritiqueStep, "revise": revise.ReviseStep,
        "critique_and_revise_in_one_call": critique_and_revise_in_one_call.CritiqueAndReviseInOneCallStep,
        "score": score.ScoreStep, "score_gate": score_gate.ScoreGateStep,
        "near_duplicate_removal": near_duplicate_removal.NearDuplicateRemovalStep})
    assert set(step_classes) == {entry.name for entry in ENTRIES}
    for entry in ENTRIES:
        step_class = step_classes[entry.name]
        assert step_class.step_name == entry.name
        recorded_tag = getattr(step_class, "stage_tag", None) or getattr(step_class, "stage", None)
        assert recorded_tag == entry.stage_tag, entry.name
        assert step_class.__module__ == f"sdf_pipeline.{entry.module_name}", entry.name
        assert entry.module_name == f"steps.{entry.name}", entry.name   # one file per step, named after the step


def test_the_shuffle_seeds_stay_the_literal_old_stage_tags_the_layers_derive_their_seeds_from():
    assert pipeline_steps.entry_named("screen").shuffle_seed == "layer1_screen" == layer1_deal_and_screen_shared.SHUFFLE_SEED_NAME
    assert pipeline_steps.entry_named("consideration_scoring").shuffle_seed == "layer1_select" \
        == consideration_scoring.SHUFFLE_SEED_NAME
    assert {entry.name for entry in ENTRIES if entry.shuffle_seed} == {"screen", "consideration_scoring"}


def test_each_step_is_built_with_the_model_the_catalogue_resolves(tmp_path):
    run_directory, config = command._run_fixture_scenario_in_temporary_folder(
        "temperature_kept_on_every_stage_with_a_two_call_revision", tmp_path)
    prompts_directory = command._prompts_directory_of_run(run_directory)
    reading = pipeline_steps.RunReading(
        config, prompts_directory,
        {key: utils.sdf_stage_folder(run_directory, key) for key in sdf_run_layouts.FOLDER_KEYS})
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
        "deal", "screen", "redeal", "consideration_selection", "critique_and_revise_in_one_call", "score_gate", "near_duplicate_removal"}
    assert "critique_and_revise_in_one_call" in seeding.UNSEEDABLE_STEPS and "critique" in seeding.UNSEEDABLE_STEPS[
        "critique_and_revise_in_one_call"]


def test_the_print_command_stage_names_are_the_catalogue_s():
    assert set(command.STAGES_THIS_COMMAND_COVERS) == {e.print_command_name for e in ENTRIES if e.build_step or e.name == "screen"}
    assert command.STAGES_THIS_COMMAND_CANNOT_COVER_YET == {"score_gate": "makes no model call",
                                                           "near_duplicate_removal": "makes no model call"}


def test_a_run_whose_snapshot_lacks_a_template_of_an_active_step_is_refused_before_any_call(tmp_path):
    config = utils.load_config(str(REPOSITORY_ROOT / "config.yaml"))
    prompts_directory = tmp_path / "prompts"
    prompts_directory.mkdir()
    for source in (REPOSITORY_ROOT / "prompts" / "sdf").iterdir():
        if source.is_file():
            (prompts_directory / source.name).write_bytes(source.read_bytes())
    assert pipeline_steps.prompt_files_missing_from_snapshot("draft", config, prompts_directory) == []
    (prompts_directory / "layer2.txt").unlink()
    assert pipeline_steps.prompt_files_missing_from_snapshot("draft", config, prompts_directory) == [("draft", ["layer2.txt"])]


# ------------------------------------------------------------------ print command against the pipeline's own requests

SCENARIOS = ["temperature_kept_on_every_stage_with_a_two_call_revision",
             "temperature_kept_on_every_stage_with_a_one_call_revision",
             "minimal_config_with_openai_rewrite_and_adaptive_thinking_models"]
STAGE_OF_TAG = {"layer1_screen": "screen", "layer2_plan": "plan", "layer2_description_rewrite": "description_rewrite",
                "layer3_consideration_scoring": "consideration_scoring", "layer4_draft": "draft", "layer4_critique": "critique",
                "layer4_revise": "revise", "layer4_critique_and_revise_in_one_call": "critique_and_revise_in_one_call", "layer5_score": "score"}


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
    batches = utils.load_jsonl(utils.sdf_stage_dir(run_directory, 1) / layer1_deal_and_screen_shared.BATCHES_FILENAME)
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
    assert compared_stages >= {"plan", "description_rewrite", "consideration_scoring", "draft", "score"}
    assert ("critique_and_revise_in_one_call" in compared_stages) != ("critique" in compared_stages)


# ------------------------------------------------------------------ refusal messages worded from the catalogue

def _seed_refusal(from_step: str, config: dict) -> str:
    with pytest.raises(seeding.SeedingRefused) as refusal:
        seeding.seed_run_from_chosen_step(
            config=config, source_run=Path("unused"), from_step=from_step, document_ids=None,
            runs_root=Path("unused"), label="unused", repository_root=REPOSITORY_ROOT)
    return str(refusal.value)


def test_seeding_refusals_name_the_step_the_reason_and_what_to_do_instead():
    config = utils.load_config(str(REPOSITORY_ROOT / "config.yaml"))
    single_call = {**config, "sdf": {**config["sdf"], "revision": {"split": False}}}
    revise = _seed_refusal("revise", single_call)
    assert "step 'revise' cannot be seeded" in revise and "revises in one call" in revise
    assert "seed from 'critique'" in revise and "Seedable steps: plan, description_rewrite" in revise
    combined = _seed_refusal("critique_and_revise_in_one_call", config)
    assert "step 'critique_and_revise_in_one_call' cannot be seeded" in combined and "in place of the critique and the revise" in combined
    assert "seed at 'critique'" in combined
    assert "unknown step 'no_such_step'" in _seed_refusal("no_such_step", config)


def test_print_command_refusals_name_the_stage_the_reason_and_the_input(tmp_path):
    run_directory, config = command._run_fixture_scenario_in_temporary_folder(
        "temperature_kept_on_every_stage_with_a_two_call_revision", tmp_path)
    switched_off = {**config, "sdf": {**config["sdf"], "description_rewrite_enabled": False, "select_enabled": False}}
    with pytest.raises(command.RequestCannotBeBuilt, match=r"description_rewrite step does not run.*Description rewrite disabled \(sdf.description_rewrite_enabled: false\)"):
        command.build_full_request(run_directory, switched_off, "description_rewrite", "matrix_000000")
    with pytest.raises(command.RequestCannotBeBuilt, match="consideration_scoring step does not run.*sdf.select_enabled: false"):
        command.build_full_request(run_directory, switched_off, "consideration_scoring", "matrix_000000")
    with pytest.raises(command.RequestCannotBeBuilt, match="no record with prompt_id 'nope' in the input documents of the draft step"):
        command.build_full_request(run_directory, config, "draft", "nope")
    with pytest.raises(command.RequestCannotBeBuilt, match="no record with doc_id 'nope' in the input documents of the revise step"):
        command.build_full_request(run_directory, config, "revise", "nope")
