"""The step catalogue (sdf_pipeline/pipeline_steps.py) agrees with the step files, the layer modules, the
prompt snapshot and the request print command, and the print command builds, for every stage and every
document of an equivalence scenario, the very request the pipeline sent (system prompt, user message,
schema, model, token limit, temperature). No model API is called."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

import sdf_prompt_equivalence_fixture_support as support   # imported first: captures the real network seams
import sdf_prompt_equivalence_scenarios as scenarios
from sdf_pipeline.layer1_deal_variables_and_screen import shared_by_this_layers_steps as deal_variables_and_screen_shared_names
from sdf_pipeline import pipeline_steps
from sdf_pipeline.layer3_consideration_scoring_and_selection import consideration_scoring, consideration_selection
from sdf_pipeline.layer4_draft_critique_and_revise import critique, critique_and_revise_in_one_call, draft, revise
from sdf_pipeline.layer1_deal_variables_and_screen import deal_variables, redeal, screen
from sdf_pipeline.layer2_description_write_and_description_rewrite import description_rewrite, description_write
from sdf_pipeline.layer5_score_and_final_selection import near_duplicate_removal, score, score_gate
from sdf_pipeline.shared_step_machinery.step_record import Run
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
    for layout, layer_numbers in ((sdf_run_layouts.FIVE_NAMED_LAYERS, {1, 2, 3, 4, 5}),
                                  (sdf_run_layouts.FOUR_NUMBERED_LAYERS, {1, 2, 3, 4})):
        layers = [entry.layer_in(layout) for entry in ENTRIES]
        assert layers == sorted(layers) and set(layers) == layer_numbers


def test_stage_tags_and_step_names_equal_the_step_files():
    step_of_name = {"deal_variables": deal_variables, "redeal": redeal, "consideration_selection": consideration_selection,
                    "score_gate": score_gate, "near_duplicate_removal": near_duplicate_removal}
    step_modules = {"screen": screen, "description_write": description_write, "description_rewrite": description_rewrite,
                    "consideration_scoring": consideration_scoring, "draft": draft, "critique": critique,
                    "revise": revise, "critique_and_revise_in_one_call": critique_and_revise_in_one_call,
                    "score": score}
    for name, module in step_modules.items():
        assert module.STEP.name == name
        assert pipeline_steps.entry_named(name) is module.STEP
    assert set(step_modules) | set(step_of_name) == {entry.name for entry in ENTRIES}
    for entry in ENTRIES:
        assert entry.stage_tag_in(sdf_run_layouts.FIVE_NAMED_LAYERS) == (
            sdf_run_layouts.FIVE_NAMED_LAYERS.stage_tag(entry.name) if entry.makes_model_calls else None), entry.name
        assert entry.makes_model_calls == (entry.name in step_modules)


def test_the_shuffle_seeds_stay_the_literal_old_stage_tags_the_layers_derive_their_seeds_from():
    assert deal_variables_and_screen_shared_names.SHUFFLE_SEED_NAME == "layer1_screen"
    assert consideration_scoring.SHUFFLE_SEED_NAME == "layer1_select"


def test_each_step_is_built_with_the_model_the_catalogue_resolves(tmp_path):
    run_directory, config = command._run_fixture_scenario_in_temporary_folder(
        "temperature_kept_on_every_stage_with_a_two_call_revision", tmp_path)
    prompts_directory = command._prompts_directory_of_run(run_directory)
    stage_directories = {key: utils.sdf_stage_folder(run_directory, key) for key in sdf_run_layouts.FOLDER_KEYS}
    models_by_key = {key: f"model-for-{key}" for entry in ENTRIES for key in entry.model_setting_keys}
    for configuration in (config, {**config, "sdf": {**config["sdf"], **models_by_key}},
                          {**config, "sdf": {**config["sdf"], **{k: None for k in models_by_key if k != "description_write_model"},
                                             "description_write_model": "only-the-description-write-model"}}):
        run = Run(configuration, prompts_directory, stage_directories)
        for entry in ENTRIES:
            if entry.reads is None:   # the screen's batches are covered by the request equivalence test below
                continue
            documents = entry.reads(run)
            request = entry.build_request([documents[0]], run)
            if request.model is not None or entry.model_for(configuration) is not None:
                assert request.model == entry.model_for(configuration), entry.name


def test_every_required_prompt_file_of_a_default_run_is_in_the_prompts_folder():
    config = utils.load_config(str(REPOSITORY_ROOT / "config.yaml"))
    prompts_directory = REPOSITORY_ROOT / "prompts" / "sdf"
    for entry in ENTRIES:
        for name in entry.required_prompt_files(config, prompts_directory):
            assert (prompts_directory / name).is_file(), (entry.name, name)


def test_the_print_command_stage_names_are_the_catalogue_s():
    assert set(command.STAGES_THIS_COMMAND_COVERS) == {e.name for e in ENTRIES if e.makes_model_calls}
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
STAGE_OF_TAG = {"layer1_screen": "screen", "layer2_description_write": "description_write", "layer2_description_rewrite": "description_rewrite",
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
    batches = utils.load_jsonl(utils.sdf_stage_dir(run_directory, 1) / deal_variables_and_screen_shared_names.BATCHES_FILENAME)
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
    assert compared_stages >= {"description_write", "description_rewrite", "consideration_scoring", "draft", "score"}
    assert ("critique_and_revise_in_one_call" in compared_stages) != ("critique" in compared_stages)


# ------------------------------------------------------------------ refusal messages

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
