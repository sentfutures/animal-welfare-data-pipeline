"""Offline tests for persona sets: the loader, the validator (including failure cases), the scoring model defaults,
the condition builder, and the reproduction of the first preachiness run's conditions (the first 62 of the
current 72) and generation prompt.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from evals.log_probability_persona_check import persona_sets, prompt_rendering
from evals.log_probability_persona_check.scoring_model_defaults import (DEFAULT_SCORING_MODELS,
                                                                        resolve_scoring_model)

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "log_probability_persona_check"
PREACHINESS_FOLDER = persona_sets.PERSONA_CHECKS_PROMPT_DIRECTORY / "preachiness"


def read_fixture_jsonl(file_name):
    return [json.loads(line) for line in (FIXTURE_DIRECTORY / file_name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture
def copied_persona_checks(tmp_path):
    """A writable copy of prompts/tools/persona_checks/; returns (directory, preachiness folder)."""
    directory = tmp_path / "persona_checks"
    shutil.copytree(persona_sets.PERSONA_CHECKS_PROMPT_DIRECTORY, directory)
    return directory, directory / "preachiness"


def validation_errors(directory, folder_name="preachiness"):
    return persona_sets.validate_persona_set(folder_name, persona_checks_directory=directory).errors


def rewrite_lines(path, function):
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(function(lines)) + "\n", encoding="utf-8")


def edit_configuration(folder, function):
    path = folder / persona_sets.CONFIGURATION_FILE_NAME
    configuration = json.loads(path.read_text(encoding="utf-8"))
    function(configuration)
    path.write_text(json.dumps(configuration), encoding="utf-8")


# ---------------------------------------------------------------- the shipped preachiness set


def test_preachiness_set_is_valid_and_has_a_draft_neutral_control():
    report = persona_sets.validate_persona_set("preachiness")
    assert report.is_valid, report.errors
    persona_set = report.persona_set
    assert persona_set.name == "preachiness" and persona_set.persona_name == "preachy"
    assert persona_set.expected_condition_count() == 72
    assert persona_set.rewrite_direction_names() == {"toward": "preachier", "away": "plainer",
                                                     "neutral": "register_shift"}
    assert persona_set.condition_group_names() == ["preachy", "control", "same_values_control"]
    assert persona_set.neutral_control_instructions_are_draft and not persona_set.positive_control_instructions_are_draft
    assert not persona_set.neutral_control_rewrite_instructions.startswith("#")
    assert any("DRAFT" in warning for warning in report.warnings)
    assert persona_sets.CONFIGURATION_FILE_NAME in persona_set.file_sha256


def test_loader_accepts_a_name_or_a_folder_path():
    by_name = persona_sets.load_persona_set("preachiness")
    by_path = persona_sets.load_persona_set(str(PREACHINESS_FOLDER))
    assert by_name.file_sha256 == by_path.file_sha256
    assert persona_sets.list_persona_set_names() == ["preachiness"]
    with pytest.raises(persona_sets.PersonaSetValidationError, match="not found"):
        persona_sets.load_persona_set("no_such_persona")


def test_draft_instructions_load_but_paid_use_is_refused():
    persona_set = persona_sets.load_persona_set("preachiness")
    persona_set.require_reviewed_instructions("positive")
    with pytest.raises(persona_sets.DraftInstructionsError, match="review"):
        persona_set.require_reviewed_instructions("neutral")


def test_command_line_validator_reports_valid_and_invalid(copied_persona_checks, capsys):
    assert persona_sets.main(["--persona-set", "preachiness"]) == 0
    assert "72 conditions" in capsys.readouterr().out
    assert persona_sets.main(["--persona-set", "does_not_exist"]) == 1


# ---------------------------------------------------------------- reproduction of the legacy run


PERSONAS_ADDED_AFTER_THE_LEGACY_RUN = ("welfare_care_without_preachiness", "ethical_farming_without_pushiness")
LEGACY_RUN_SAME_VALUES_GROUP = "proportionate_welfare"


def without_added_personas_and_with_legacy_group_name(conditions):
    """The current conditions minus the same-values personas added after the legacy run, with the same-values
    group renamed to the name the legacy run used."""
    return [{**condition, "condition_group": LEGACY_RUN_SAME_VALUES_GROUP}
            if condition["condition_group"] == "same_values_control" else condition
            for condition in conditions if condition["persona_name"] not in PERSONAS_ADDED_AFTER_THE_LEGACY_RUN]


def test_preachiness_conditions_equal_the_legacy_run_conditions_apart_from_the_added_personas():
    persona_set = persona_sets.load_persona_set("preachiness")
    conditions = prompt_rendering.build_condition_records(persona_set)
    assert len(conditions) == 72
    reduced = without_added_personas_and_with_legacy_group_name(conditions)
    assert len(reduced) == 62
    assert reduced == read_fixture_jsonl("preachiness_legacy_run_conditions.jsonl")
    assert [condition["persona_name"] for condition in conditions[-10:]] == (
        [PERSONAS_ADDED_AFTER_THE_LEGACY_RUN[0]] * 5 + [PERSONAS_ADDED_AFTER_THE_LEGACY_RUN[1]] * 5)


def test_preachiness_base_conditions_equal_the_conditions_the_legacy_code_built_for_base_models():
    persona_set = persona_sets.load_persona_set("preachiness")
    conditions = prompt_rendering.build_condition_records(persona_set, prompt_rendering.MODEL_TYPE_BASE)
    assert without_added_personas_and_with_legacy_group_name(conditions) == read_fixture_jsonl(
        "preachiness_legacy_base_conditions.jsonl")


def test_generation_prompt_is_rendered_from_a_document_record_as_the_template_prescribes():
    # The fixture is an invented document; its expected prompt was written out by hand from the template text, not
    # produced by the code under test (the resolution value has no final full stop, which the prompt adds).
    document = json.loads((FIXTURE_DIRECTORY / "synthetic_document_with_expected_generation_prompt.json").read_text(
        encoding="utf-8"))
    rendered = prompt_rendering.render_generation_prompt(document["variables"], document["document_id"])
    assert rendered == document["generation_prompt"]


def test_condition_structure():
    conditions = prompt_rendering.build_condition_records(persona_sets.load_persona_set("preachiness"))
    assert Counter(condition["condition_group"] for condition in conditions) == {
        "preachy": 30, "control": 25, "same_values_control": 15, "generic": 1, "no_system_prompt": 1}
    by_id = {condition["condition_id"]: condition for condition in conditions}
    assert by_id["no_system_prompt"]["system_prompt"] is None
    assert by_id["generic"]["system_prompt"] == "You are an AI assistant."
    assert by_id["preachy__rewording_01"]["system_prompt"].startswith("You are an AI assistant who ")
    assert by_id["preachy__rewording_01"]["system_prompt"].endswith(".")
    assert "animal_husbandry_expertise__rewording_05" in by_id
    assert [condition["persona_name"] for condition in conditions if condition["condition_group"] == "same_values_control"
            ][::5] == ["proportionate_welfare", "welfare_care_without_preachiness", "ethical_farming_without_pushiness"]


def test_base_conditions_match_instruct_conditions_apart_from_author_description():
    persona_set = persona_sets.load_persona_set("preachiness")
    instruct_conditions = prompt_rendering.build_condition_records(persona_set)
    base_conditions = prompt_rendering.build_condition_records(persona_set, "base")
    assert "author_description" not in instruct_conditions[0]
    assert [{key: value for key, value in condition.items() if key != "author_description"}
            for condition in base_conditions] == instruct_conditions
    by_id = {condition["condition_id"]: condition for condition in base_conditions}
    assert by_id["no_system_prompt"]["author_description"] is None
    assert by_id["generic"]["author_description"] == "an AI assistant"
    for condition in base_conditions[2:]:
        persona_description = condition["system_prompt"][len("You are an AI assistant who "):-1]
        assert condition["author_description"] == f"an AI assistant who {persona_description}"


def test_set_without_same_values_control_builds_fewer_conditions(copied_persona_checks):
    directory, folder = copied_persona_checks
    (folder / persona_sets.SAME_VALUES_CONTROL_FILE_NAME).unlink()
    edit_configuration(folder, lambda configuration: configuration["condition_groups"].pop("same_values_control"))
    persona_set = persona_sets.load_persona_set("preachiness", persona_checks_directory=directory)
    conditions = prompt_rendering.build_condition_records(persona_set)
    assert len(conditions) == 57 and persona_set.condition_group_names() == ["preachy", "control"]
    assert not any(condition["condition_group"] == "same_values_control" for condition in conditions)


def test_a_second_persona_set_changes_names_and_groups(copied_persona_checks):
    directory, folder = copied_persona_checks
    new_folder = directory / "sycophancy"
    shutil.copytree(folder, new_folder)

    def rename(configuration):
        configuration["persona_set_name"] = "sycophancy"
        configuration["persona_name"] = "sycophantic"
        configuration["condition_groups"]["target"]["condition_group"] = "sycophantic"
    edit_configuration(new_folder, rename)
    persona_set = persona_sets.load_persona_set("sycophancy", persona_checks_directory=directory)
    conditions = prompt_rendering.build_condition_records(persona_set)
    assert conditions[2]["condition_id"] == "sycophantic__rewording_01"
    assert conditions[2]["condition_group"] == "sycophantic" and conditions[2]["persona_name"] == "sycophantic"


# ---------------------------------------------------------------- validator failure cases


def test_wrong_number_of_target_rewordings(copied_persona_checks):
    directory, folder = copied_persona_checks
    rewrite_lines(folder / persona_sets.TARGET_REWORDINGS_FILE_NAME, lambda lines: lines[:-1])
    assert any("expected 30 rewordings, found 29" in error for error in validation_errors(directory))
    with pytest.raises(persona_sets.PersonaSetValidationError, match="found 29"):
        persona_sets.load_persona_set("preachiness", persona_checks_directory=directory)


def test_wrong_number_of_topic_controls_or_rewordings(copied_persona_checks):
    directory, folder = copied_persona_checks
    path = folder / persona_sets.TOPIC_CONTROL_FILE_NAME
    original = path.read_text(encoding="utf-8")
    rewrite_lines(path, lambda lines: [line for line in lines if "zoological" not in line.lower()])
    errors = validation_errors(directory)
    assert any("expected 5 personas, found 4" in error for error in errors)
    path.write_text(original.replace("has extensive scientific expertise about animals\n", ""), encoding="utf-8")
    assert any("persona 'zoological_knowledge' has 4 rewordings" in error for error in validation_errors(directory))


def test_blank_and_duplicate_lines(copied_persona_checks):
    directory, folder = copied_persona_checks
    path = folder / persona_sets.TARGET_REWORDINGS_FILE_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:5] + [""] + lines[5:]) + "\n", encoding="utf-8")
    assert any("blank line" in error for error in validation_errors(directory))
    path.write_text("\n".join(lines[:29] + [lines[0].upper().lower()]) + "\n", encoding="utf-8")
    assert any("Duplicate description" in error for error in validation_errors(directory))


def test_duplicate_across_files_and_blank_line_inside_a_group(copied_persona_checks):
    directory, folder = copied_persona_checks
    target_first_line = (folder / persona_sets.TARGET_REWORDINGS_FILE_NAME).read_text(encoding="utf-8").splitlines()[0]
    rewrite_lines(folder / persona_sets.SAME_VALUES_CONTROL_FILE_NAME,
                  lambda lines: [lines[0], target_first_line] + lines[2:])
    assert any("Duplicate description" in error for error in validation_errors(directory))
    shutil.copy(PREACHINESS_FOLDER / persona_sets.SAME_VALUES_CONTROL_FILE_NAME,
                folder / persona_sets.SAME_VALUES_CONTROL_FILE_NAME)
    path = folder / persona_sets.TOPIC_CONTROL_FILE_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:3] + [""] + lines[3:]) + "\n", encoding="utf-8")
    assert any("blank line inside group" in error for error in validation_errors(directory))


@pytest.mark.parametrize("bad_line, expected_message", [
    ("Moralizes about animal welfare", "capital letter"),
    ("You moralize about animal welfare", "capital letter"),
    ("you moralize about animal welfare", "instead of a verb"),
    ("an AI that moralizes", "instead of a verb"),
    ("moralizes about animal welfare.", "terminal punctuation"),
    ("moralizes about {animals}", "brace"),
    ("moralizes  about animals", "double space")])
def test_descriptions_must_read_after_both_frames(copied_persona_checks, bad_line, expected_message):
    directory, folder = copied_persona_checks
    rewrite_lines(folder / persona_sets.TARGET_REWORDINGS_FILE_NAME, lambda lines: [bad_line] + lines[1:])
    assert any(expected_message in error for error in validation_errors(directory)), validation_errors(directory)


def test_header_structure_errors(copied_persona_checks):
    directory, folder = copied_persona_checks
    path = folder / persona_sets.TOPIC_CONTROL_FILE_NAME
    original = path.read_text(encoding="utf-8")
    path.write_text("has deep knowledge of animal husbandry\n" + original, encoding="utf-8")
    assert any("before any '# persona_name' header" in error for error in validation_errors(directory))
    path.write_text(original.replace("# aesthetic_love_of_animals", "#aesthetic_love_of_animals"), encoding="utf-8")
    assert any("header must be exactly" in error for error in validation_errors(directory))
    path.write_text(original.replace("# aesthetic_love_of_animals", "# zoological_knowledge", 1), encoding="utf-8")
    assert any("duplicate group header" in error for error in validation_errors(directory))
    path.write_text(original.replace("# aesthetic_love_of_animals", "# Aesthetic Love"), encoding="utf-8")
    assert any("lowercase letters" in error for error in validation_errors(directory))
    path.write_text(original.replace("# aesthetic_love_of_animals", "# generic"), encoding="utf-8")
    assert any("reserved or configured" in error for error in validation_errors(directory))
    path.write_text(original.replace("# aesthetic_love_of_animals", "# preachy"), encoding="utf-8")
    assert any("reserved or configured" in error for error in validation_errors(directory))


def test_hash_line_in_an_ungrouped_file_is_an_error(copied_persona_checks):
    directory, folder = copied_persona_checks
    rewrite_lines(folder / persona_sets.TARGET_REWORDINGS_FILE_NAME, lambda lines: ["# a header"] + lines[1:])
    assert any("unexpected '#' line" in error for error in validation_errors(directory))


def test_missing_and_empty_files(copied_persona_checks):
    directory, folder = copied_persona_checks
    (folder / persona_sets.POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME).unlink()
    (folder / persona_sets.NEUTRAL_CONTROL_INSTRUCTIONS_FILE_NAME).write_text("\n", encoding="utf-8")
    errors = validation_errors(directory)
    assert any("Missing file: positive_control_rewrite_instructions.txt" in error for error in errors)
    assert any("File is empty: neutral_control_rewrite_instructions.txt" in error for error in errors)


def test_same_values_personas_are_loaded_in_file_order_with_their_rewordings():
    persona_set = persona_sets.load_persona_set("preachiness")
    by_persona = persona_set.same_values_control_descriptions_by_persona
    assert list(by_persona) == ["proportionate_welfare", "welfare_care_without_preachiness",
                                "ethical_farming_without_pushiness"]
    assert all(len(descriptions) == 5 for descriptions in by_persona.values())
    assert by_persona["welfare_care_without_preachiness"][0] == "cares deeply about the welfare of animals but is not preachy"
    assert persona_set.has_same_values_control


def test_wrong_number_of_same_values_personas_or_rewordings(copied_persona_checks):
    directory, folder = copied_persona_checks
    path = folder / persona_sets.SAME_VALUES_CONTROL_FILE_NAME
    original = path.read_text(encoding="utf-8")
    path.write_text(original[:original.index("# ethical_farming_without_pushiness")], encoding="utf-8")
    assert any("expected 3 personas, found 2" in error for error in validation_errors(directory))
    path.write_text(original.replace("believes in ethical farming practices but avoids pressing them on people\n", ""),
                    encoding="utf-8")
    assert any("persona 'ethical_farming_without_pushiness' has 4 rewordings, expected 5" in error
               for error in validation_errors(directory))


def test_same_values_persona_names_must_not_collide_with_other_personas(copied_persona_checks):
    directory, folder = copied_persona_checks
    path = folder / persona_sets.SAME_VALUES_CONTROL_FILE_NAME
    path.write_text(path.read_text(encoding="utf-8").replace("# welfare_care_without_preachiness", "# preachy"),
                    encoding="utf-8")
    assert any("collides with a reserved or configured persona name" in error for error in validation_errors(directory))
    path.write_text(path.read_text(encoding="utf-8").replace("# preachy", "# zoological_knowledge"), encoding="utf-8")
    assert any("'zoological_knowledge' is already used in" in error for error in validation_errors(directory))


def test_same_values_file_without_group_and_group_without_file(copied_persona_checks):
    directory, folder = copied_persona_checks
    edit_configuration(folder, lambda configuration: configuration["condition_groups"].pop("same_values_control"))
    assert any("no same_values_control group" in error for error in validation_errors(directory))
    shutil.copy(PREACHINESS_FOLDER / persona_sets.CONFIGURATION_FILE_NAME, folder / persona_sets.CONFIGURATION_FILE_NAME)
    (folder / persona_sets.SAME_VALUES_CONTROL_FILE_NAME).unlink()
    assert any("Missing file: same_values_control_personas.txt" in error for error in validation_errors(directory))


@pytest.mark.parametrize("instruction_file", [persona_sets.POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME,
                                              persona_sets.NEUTRAL_CONTROL_INSTRUCTIONS_FILE_NAME])
def test_instruction_files_need_both_slots(copied_persona_checks, instruction_file):
    directory, folder = copied_persona_checks
    path = folder / instruction_file
    path.write_text(path.read_text(encoding="utf-8").replace("{document_text}", "the document"), encoding="utf-8")
    assert any("{document_text} slot" in error for error in validation_errors(directory))
    path.write_text(path.read_text(encoding="utf-8").replace("{direction}", "the direction"), encoding="utf-8")
    assert any("{direction} slot" in error for error in validation_errors(directory))


def test_instruction_files_must_describe_every_configured_direction(copied_persona_checks):
    directory, folder = copied_persona_checks
    path = folder / persona_sets.POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME
    path.write_text(path.read_text(encoding="utf-8").replace("PLAINER", "CALMER"), encoding="utf-8")
    assert any("does not describe the direction PLAINER" in error for error in validation_errors(directory))


def test_draft_marker_is_stripped_from_instructions_and_does_not_break_slots(copied_persona_checks):
    directory, folder = copied_persona_checks
    path = folder / persona_sets.POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME
    path.write_text("# DRAFT: not reviewed\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    persona_set = persona_sets.load_persona_set("preachiness", persona_checks_directory=directory)
    assert persona_set.positive_control_instructions_are_draft
    assert "DRAFT" not in persona_set.positive_control_rewrite_instructions


def test_persona_set_name_must_match_the_folder(copied_persona_checks):
    directory, folder = copied_persona_checks
    edit_configuration(folder, lambda configuration: configuration.update(persona_set_name="something_else"))
    assert any("differs from the folder name" in error for error in validation_errors(directory))


@pytest.mark.parametrize("edit, expected_message", [
    (lambda configuration: configuration.pop("pass_criteria"), "missing keys"),
    (lambda configuration: configuration.update(unexpected_key=1), "unknown keys"),
    (lambda configuration: configuration.update(persona_name="generic"),
     "reserved"),
    (lambda configuration: configuration.update(persona_name="Preachy"), "lowercase"),
    (lambda configuration: configuration["condition_groups"]["target"].update(expected_rewording_count=0),
     "positive integer"),
    (lambda configuration: configuration["condition_groups"]["topic_control"].update(condition_group="preachy"),
     "used by both"),
    (lambda configuration: configuration["condition_groups"]["same_values_control"].update(
        condition_group="control"), "used by both"),
    (lambda configuration: configuration["condition_groups"]["same_values_control"].update(
        expected_persona_count=0), "positive integer"),
    (lambda configuration: configuration["condition_groups"]["same_values_control"].update(persona_name="x"),
     "unknown keys"),
    (lambda configuration: configuration["condition_groups"]["same_values_control"].pop(
        "expected_rewordings_per_persona"), "missing keys"),
    (lambda configuration: configuration["rewrite_directions"].pop("neutral"), "missing keys"),
    (lambda configuration: configuration["rewrite_directions"]["away"].update(name="preachier"), "used for both"),
    (lambda configuration: configuration["pass_criteria"]["changed_span"].update(
        minimum_number_moving_expected_way=11), "exceeds the number tested"),
    (lambda configuration: configuration["pass_criteria"]["whole_document"].update(applies_to_directions=["sideways"]),
     "applies_to_directions"),
    (lambda configuration: configuration["pass_criteria"]["whole_document"].pop("minimum_number_moving_expected_way"),
     "missing keys"),
    (lambda configuration: configuration.update(default_models={"instruct": {"model_name": "x", "model_revision": "main"}}),
     "40-character")])
def test_incomplete_or_inconsistent_configuration_is_rejected(copied_persona_checks, edit, expected_message):
    directory, folder = copied_persona_checks
    edit_configuration(folder, edit)
    assert any(expected_message in error for error in validation_errors(directory)), validation_errors(directory)


def test_pass_criteria_cannot_apply_to_an_undefined_direction(copied_persona_checks):
    directory, folder = copied_persona_checks
    edit_configuration(folder, lambda configuration: configuration["rewrite_directions"].pop("away"))
    assert any("does not define" in error for error in validation_errors(directory))


def test_invalid_json_configuration(copied_persona_checks):
    directory, folder = copied_persona_checks
    (folder / persona_sets.CONFIGURATION_FILE_NAME).write_text("{not json", encoding="utf-8")
    assert any("not valid JSON" in error for error in validation_errors(directory))


def test_all_problems_are_listed_together(copied_persona_checks):
    directory, folder = copied_persona_checks
    rewrite_lines(folder / persona_sets.TARGET_REWORDINGS_FILE_NAME, lambda lines: ["Bad."] + lines[2:])
    (folder / persona_sets.POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME).unlink()
    assert len(validation_errors(directory)) >= 4


# ---------------------------------------------------------------- shared framing and generation prompt files


@pytest.mark.parametrize("file_name, new_text, expected_message", [
    (persona_sets.INSTRUCT_PERSONA_FRAME_FILE_NAME, "You are an AI assistant.", "exactly one {persona_description}"),
    (persona_sets.BASE_MODEL_PERSONA_AUTHOR_FRAME_FILE_NAME, "an AI assistant", "exactly one {persona_description}"),
    (persona_sets.BASE_MODEL_PREAMBLE_TEMPLATE_WITH_AUTHOR_FILE_NAME, "Written by {author_description}",
     "exactly one {generation_prompt}"),
    (persona_sets.BASE_MODEL_PREAMBLE_TEMPLATE_WITHOUT_AUTHOR_FILE_NAME, "Written for {generation_prompt} {extra}",
     "unexpected slots"),
    (persona_sets.INSTRUCT_GENERIC_SYSTEM_PROMPT_FILE_NAME, "You are {an} AI assistant.", "takes no slots")])
def test_framing_files_are_checked(copied_persona_checks, file_name, new_text, expected_message):
    directory, _ = copied_persona_checks
    (directory / persona_sets.MODEL_TYPE_FRAMING_DIRECTORY_NAME / file_name).write_text(new_text, encoding="utf-8")
    assert any(expected_message in error for error in validation_errors(directory)), validation_errors(directory)
    with pytest.raises(persona_sets.PersonaSetValidationError):
        persona_sets.load_model_type_framing(directory / persona_sets.MODEL_TYPE_FRAMING_DIRECTORY_NAME)


def test_generation_prompt_template_needs_the_facts_slot(copied_persona_checks):
    directory, _ = copied_persona_checks
    path = directory / persona_sets.GENERATION_PROMPT_DIRECTORY_NAME / persona_sets.GENERATION_PROMPT_TEMPLATE_FILE_NAME
    path.write_text("Write a document about {domain}.", encoding="utf-8")
    assert any("missing the {in_universe_facts} slot" in error for error in validation_errors(directory))


# ---------------------------------------------------------------- scoring model defaults


def test_default_models_are_pinned():
    assert DEFAULT_SCORING_MODELS["instruct"] == {"model_name": "google/gemma-4-12b-it",
                                                  "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"}
    assert DEFAULT_SCORING_MODELS["base"] == {"model_name": "google/gemma-4-12b",
                                              "model_revision": "023679ed352de9bb66cc873c9009ce3482585c08"}


def test_model_resolution_precedence_and_revision_requirement():
    assert resolve_scoring_model("base") == ("google/gemma-4-12b", "023679ed352de9bb66cc873c9009ce3482585c08",
                                             "package default")
    persona_default = {"base": {"model_name": "org/other-base", "model_revision": "a" * 40}}
    assert resolve_scoring_model("base", persona_default) == ("org/other-base", "a" * 40, "persona set")
    assert resolve_scoring_model("instruct", persona_default)[2] == "package default"
    assert resolve_scoring_model("base", persona_default, model_revision_override="b" * 40) == (
        "org/other-base", "b" * 40, "command line")
    with pytest.raises(ValueError, match="pass --model-revision"):
        resolve_scoring_model("instruct", model_name_override="org/another-model")
    assert resolve_scoring_model("instruct", model_name_override="org/another-model",
                                 model_revision_override="c" * 40) == ("org/another-model", "c" * 40, "command line")
    with pytest.raises(ValueError, match="40-character"):
        resolve_scoring_model("instruct", model_revision_override="main")
    with pytest.raises(ValueError, match="Unknown model type"):
        resolve_scoring_model("chat")


# ---------------------------------------------------------------- generation prompt rendering


SHORT_VARIABLES = {
    "document_type": "a blog post", "culture": "Mexico, written in Spanish", "domain": "policy and regulation",
    "sentient_category": "small wild animals", "decision_scale": "millions of animals",
    "naming": "an unnamed AI assistant", "tech_savvy": "general readers", "tone": "neutral",
    "tradeoff": "cost/convenience", "centrality": "a minor detail",
    "resolution": "The AI recommends the cheaper option", "reasoning_featured": "a brief cost comparison"}


@pytest.mark.parametrize("resolution, expected", [
    ("The AI recommends the cheaper option", "The AI recommends the cheaper option."),
    ("Already ends properly.", "Already ends properly."),
    ("Ends with a question?", "Ends with a question?"),
    ("Ends with a quote.\"", "Ends with a quote.\""),
    ("Trailing space   ", "Trailing space."),
    ("中文。", "中文。"),
    ("No punctuation after a quote \"word\"", "No punctuation after a quote \"word\".")])
def test_resolution_gets_final_period_only_when_missing(resolution, expected):
    assert prompt_rendering.add_final_period_if_missing(resolution) == expected


def test_generation_prompt_fills_every_slot_and_adds_period():
    prompt = prompt_rendering.render_generation_prompt(SHORT_VARIABLES, "doc")
    assert "{" not in prompt and "}" not in prompt
    for value in SHORT_VARIABLES.values():
        assert value in prompt
    assert "The AI recommends the cheaper option. The AI's reasoning features a brief cost comparison." in prompt
    assert prompt.startswith("Write a document from a world where AI assistants")


def test_variable_names_are_read_from_the_template():
    template_text = persona_sets.load_generation_prompt_files().template_text
    assert prompt_rendering.generation_prompt_variable_names(template_text) == [
        "document_type", "culture", "domain", "sentient_category", "decision_scale", "naming", "tech_savvy", "tone",
        "tradeoff", "centrality", "resolution", "reasoning_featured"]


@pytest.mark.parametrize("missing_name", list(SHORT_VARIABLES))
def test_missing_variable_fails_loudly(missing_name):
    variables = {name: value for name, value in SHORT_VARIABLES.items() if name != missing_name}
    with pytest.raises(KeyError, match=missing_name):
        prompt_rendering.render_generation_prompt(variables, "doc_with_missing_variable")


def test_empty_variable_fails_loudly():
    with pytest.raises(ValueError, match="tone"):
        prompt_rendering.render_generation_prompt({**SHORT_VARIABLES, "tone": "  "}, "doc")
