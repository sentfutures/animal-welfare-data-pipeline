"""Persona sets for the log-probability persona check: loading, validation and the configuration schema.

A persona set is a folder under ``prompts/tools/persona_checks/<persona_set_name>/`` holding everything that is
specific to one persona (preachy, sycophantic, ...):

    target_persona_rewordings.txt            one description per line, about 30 equivalent statements of the persona
    topic_control_personas.txt               `# persona_name` headers, each followed by its rewordings, one per line
    same_values_control_personas.txt         optional: same `# persona_name` format (same values, not the behaviour)
    positive_control_rewrite_instructions.txt
    neutral_control_rewrite_instructions.txt
    persona_check_configuration.json         see CONFIGURATION SCHEMA below

Framing shared by all persona sets lives in ``prompts/tools/persona_checks/model_type_framing/`` and the
document generation prompt in ``prompts/tools/persona_checks/generation_prompt/``.

A description is the text that completes "You are an AI assistant who ..." (instruct models) and "an AI assistant
who ..." (base models), so it starts lowercase with a verb, is one line, and has no final full stop.

CONFIGURATION SCHEMA (persona_check_configuration.json, schema_version 1; unknown keys are errors)

    persona_set_name         must equal the folder name, e.g. "preachiness"
    persona_name             name of the target persona in condition ids and the `persona_name` column, e.g. "preachy"
    condition_groups
        target               condition_group (value of the `condition_group` column), display_name,
                             expected_rewording_count
        topic_control        condition_group, display_name, expected_persona_count, expected_rewordings_per_persona
        same_values_control  OPTIONAL. condition_group, display_name, expected_persona_count,
                             expected_rewordings_per_persona (as topic_control).
                             When absent, same_values_control_personas.txt must be absent too.
    rewrite_directions
        toward               {"name": ...}  the direction that adds the persona's behaviour, e.g. "preachier"
        away                 OPTIONAL {"name": ...}  the direction that removes it, e.g. "plainer"
        neutral              {"name": ...}  an edit of similar size that changes register only
    pass_criteria            pre-registered before scoring; copied into the run configuration
        status               free text saying where the values come from
        whole_document / changed_span
            applies_to_directions                    subset of ["toward", "away"] that exist in rewrite_directions
            minimum_number_moving_expected_way       integer
            number_of_rewrites_tested_per_direction  integer, at least the minimum
        neutral_control      OPTIONAL. changed_span: {maximum_number_moving_toward_persona,
                             number_of_rewrites_tested}. To be pre-registered before the neutral control is run.
    default_models           OPTIONAL. {"instruct": {...}, "base": {...}}, each {"model_name", "model_revision"};
                             overrides the package defaults in scoring_model_defaults.py

Usage (validates a persona set, free and offline):
    python evals/log_probability_persona_check/persona_sets.py --persona-set preachiness
"""

import argparse
import hashlib
import json
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check.scoring_model_defaults import COMMIT_HASH_PATTERN, MODEL_TYPES

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
PERSONA_CHECKS_PROMPT_DIRECTORY = REPOSITORY_ROOT / "prompts" / "tools" / "persona_checks"
MODEL_TYPE_FRAMING_DIRECTORY_NAME = "model_type_framing"
GENERATION_PROMPT_DIRECTORY_NAME = "generation_prompt"

CONFIGURATION_FILE_NAME = "persona_check_configuration.json"
TARGET_REWORDINGS_FILE_NAME = "target_persona_rewordings.txt"
TOPIC_CONTROL_FILE_NAME = "topic_control_personas.txt"
SAME_VALUES_CONTROL_FILE_NAME = "same_values_control_personas.txt"
POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME = "positive_control_rewrite_instructions.txt"
NEUTRAL_CONTROL_INSTRUCTIONS_FILE_NAME = "neutral_control_rewrite_instructions.txt"
REQUIRED_PERSONA_SET_FILE_NAMES = [CONFIGURATION_FILE_NAME, TARGET_REWORDINGS_FILE_NAME, TOPIC_CONTROL_FILE_NAME,
                                   POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME, NEUTRAL_CONTROL_INSTRUCTIONS_FILE_NAME]
# Files whose content decides which conditions are scored; their hashes are checked on resume.
PERSONA_SET_FILE_NAMES_THAT_DETERMINE_CONDITIONS = [CONFIGURATION_FILE_NAME, TARGET_REWORDINGS_FILE_NAME,
                                                    TOPIC_CONTROL_FILE_NAME, SAME_VALUES_CONTROL_FILE_NAME]

INSTRUCT_PERSONA_FRAME_FILE_NAME = "instruct_persona_system_prompt_frame.txt"
INSTRUCT_GENERIC_SYSTEM_PROMPT_FILE_NAME = "instruct_generic_system_prompt.txt"
BASE_MODEL_PREAMBLE_TEMPLATE_WITH_AUTHOR_FILE_NAME = "base_model_preamble_template_with_author.txt"
BASE_MODEL_PREAMBLE_TEMPLATE_WITHOUT_AUTHOR_FILE_NAME = "base_model_preamble_template_without_author.txt"
BASE_MODEL_PERSONA_AUTHOR_FRAME_FILE_NAME = "base_model_persona_author_frame.txt"
BASE_MODEL_GENERIC_AUTHOR_DESCRIPTION_FILE_NAME = "base_model_generic_author_description.txt"
INSTRUCT_FRAMING_FILE_NAMES = [INSTRUCT_PERSONA_FRAME_FILE_NAME, INSTRUCT_GENERIC_SYSTEM_PROMPT_FILE_NAME]
BASE_MODEL_FRAMING_FILE_NAMES = [
    BASE_MODEL_PREAMBLE_TEMPLATE_WITH_AUTHOR_FILE_NAME, BASE_MODEL_PREAMBLE_TEMPLATE_WITHOUT_AUTHOR_FILE_NAME,
    BASE_MODEL_PERSONA_AUTHOR_FRAME_FILE_NAME, BASE_MODEL_GENERIC_AUTHOR_DESCRIPTION_FILE_NAME]

GENERATION_PROMPT_TEMPLATE_FILE_NAME = "document_generation_prompt_template.txt"
IN_UNIVERSE_FACTS_FILE_NAME = "in_universe_facts_shared_by_all_documents.txt"
GENERATION_PROMPT_FILE_NAMES = [GENERATION_PROMPT_TEMPLATE_FILE_NAME, IN_UNIVERSE_FACTS_FILE_NAME]

DRAFT_MARKER = "# DRAFT"
RESERVED_PERSONA_NAMES = ["generic", "no_system_prompt"]
PERSONA_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")  # no double underscore: it separates the rewording index
TERMINAL_PUNCTUATION = ".!?。！？…।۔"
DIRECTION_ROLES = ["toward", "away", "neutral"]


class PersonaSetValidationError(ValueError):
    """Raised when a persona set or the shared framing files are invalid. The message lists every problem."""


class DraftInstructionsError(RuntimeError):
    """Raised when a paid step is about to use rewrite instructions that are still marked as a draft."""


@dataclass
class ModelTypeFraming:
    instruct_persona_frame: str
    instruct_generic_system_prompt: str
    base_preamble_template_with_author: str
    base_preamble_template_without_author: str
    base_persona_author_frame: str
    base_generic_author_description: str
    directory: Path


@dataclass
class GenerationPromptFiles:
    template_text: str
    in_universe_facts: str
    directory: Path


@dataclass
class PersonaSet:
    name: str
    folder: Path
    configuration: dict
    target_rewordings: list
    topic_control_descriptions_by_persona: dict       # {persona_name: [descriptions]} in file order
    same_values_control_descriptions_by_persona: object   # like the topic controls, or None without a same-values control
    positive_control_rewrite_instructions: str
    neutral_control_rewrite_instructions: str
    positive_control_instructions_are_draft: bool
    neutral_control_instructions_are_draft: bool
    file_sha256: dict = field(default_factory=dict)   # every file in the folder, by file name

    @property
    def persona_name(self):
        return self.configuration["persona_name"]

    @property
    def has_same_values_control(self):
        return self.same_values_control_descriptions_by_persona is not None

    def rewrite_direction_names(self):
        """{"toward": "preachier", "away": "plainer" (if configured), "neutral": "register_shift"}."""
        return {role: entry["name"] for role, entry in self.configuration["rewrite_directions"].items()}

    def condition_group_names(self):
        """The `condition_group` values the persona's conditions use, in condition order."""
        groups = self.configuration["condition_groups"]
        names = [groups["target"]["condition_group"], groups["topic_control"]["condition_group"]]
        if self.has_same_values_control:
            names.append(groups["same_values_control"]["condition_group"])
        return names

    def expected_condition_count(self):
        groups = self.configuration["condition_groups"]
        count = 2 + groups["target"]["expected_rewording_count"]
        count += groups["topic_control"]["expected_persona_count"] * groups["topic_control"]["expected_rewordings_per_persona"]
        if self.has_same_values_control:
            count += (groups["same_values_control"]["expected_persona_count"]
                      * groups["same_values_control"]["expected_rewordings_per_persona"])
        return count

    def require_reviewed_instructions(self, which):
        """Call before any paid use of the rewrite instructions. `which` is 'positive' or 'neutral'."""
        if which not in ("positive", "neutral"):
            raise ValueError("which must be 'positive' or 'neutral'")
        is_draft = (self.positive_control_instructions_are_draft if which == "positive"
                    else self.neutral_control_instructions_are_draft)
        if is_draft:
            raise DraftInstructionsError(
                f"The {which} control rewrite instructions of persona set {self.name} are marked "
                f"'{DRAFT_MARKER}'; have a person review them, then delete the marker line, before any paid use")


@dataclass
class PersonaSetValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    persona_set: object = None  # the PersonaSet when there are no errors

    @property
    def is_valid(self):
        return not self.errors


# ---------------------------------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------------------------------

def sha256_of_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_text_file(path, errors):
    path = Path(path)
    if not path.is_file():
        errors.append(f"Missing file: {path.name}")
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        errors.append(f"File is empty: {path.name}")
        return None
    return text


def split_off_draft_marker(text):
    """Returns (text without a leading '# DRAFT...' line, whether the marker was there)."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith(DRAFT_MARKER):
        return "\n".join(lines[1:]).strip(), True
    return text.strip(), False


def parse_description_lines(text, file_name, errors):
    """One description per line. Blank lines (other than trailing ones) and '#' lines are errors."""
    lines = text.rstrip().splitlines()
    descriptions = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"{file_name} line {line_number}: blank line")
        elif line.strip().startswith("#"):
            errors.append(f"{file_name} line {line_number}: unexpected '#' line in a file without groups")
        else:
            descriptions.append(line.strip())
    return descriptions


def parse_grouped_descriptions(text, file_name, errors):
    """{persona_name: [descriptions]} in file order. Blank lines are allowed only between groups."""
    descriptions_by_persona = {}
    current_persona_name = None
    previous_line_was_blank = False
    for line_number, line in enumerate(text.rstrip().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            previous_line_was_blank = True
            continue
        if stripped.startswith("#"):
            header_name = stripped[1:].strip()
            if stripped[:2] != "# " or not header_name or header_name != stripped[2:]:
                errors.append(f"{file_name} line {line_number}: header must be exactly '# persona_name', "
                              f"got {stripped!r}")
            if current_persona_name is not None and not descriptions_by_persona[current_persona_name]:
                errors.append(f"{file_name}: group '{current_persona_name}' has no descriptions")
            current_persona_name = header_name
            if header_name in descriptions_by_persona:
                errors.append(f"{file_name} line {line_number}: duplicate group header '{header_name}'")
            else:
                descriptions_by_persona[header_name] = []
        else:
            if current_persona_name is None:
                errors.append(f"{file_name} line {line_number}: description before any '# persona_name' header")
            else:
                if previous_line_was_blank and descriptions_by_persona[current_persona_name]:
                    errors.append(f"{file_name} line {line_number}: blank line inside group "
                                  f"'{current_persona_name}'")
                descriptions_by_persona[current_persona_name].append(stripped)
        previous_line_was_blank = False
    if current_persona_name is not None and not descriptions_by_persona[current_persona_name]:
        errors.append(f"{file_name}: group '{current_persona_name}' has no descriptions")
    return descriptions_by_persona


def check_description_reads_after_frames(description, where, errors):
    """The description must complete both 'You are an AI assistant who ...' and 'an AI assistant who ...'."""
    first_word = description.split()[0]
    if first_word[0].isupper():
        errors.append(f"{where}: description starts with a capital letter, so it does not read after "
                      f"'an AI assistant who': {description!r}")
    if first_word.lower() in ("you", "you're", "an", "ai", "who", "i"):
        errors.append(f"{where}: description starts with {first_word!r} instead of a verb: {description!r}")
    if description.rstrip()[-1] in TERMINAL_PUNCTUATION:
        errors.append(f"{where}: description ends in terminal punctuation, but the frame adds the full stop: "
                      f"{description!r}")
    if "{" in description or "}" in description:
        errors.append(f"{where}: description contains a brace: {description!r}")
    if "\t" in description or "  " in description:
        errors.append(f"{where}: description contains a tab or a double space: {description!r}")


# ---------------------------------------------------------------------------------------------------
# Shared framing and generation prompt files
# ---------------------------------------------------------------------------------------------------

def slot_names_in(template_text):
    try:
        return [field_name for _, field_name, _, _ in string.Formatter().parse(template_text) if field_name]
    except ValueError as error:
        raise ValueError(f"unparseable braces: {error}") from error


def read_framing_files(directory, errors):
    """Returns a ModelTypeFraming (or None) and appends every problem to `errors`."""
    directory = Path(directory)
    texts = {}
    for file_name in INSTRUCT_FRAMING_FILE_NAMES + BASE_MODEL_FRAMING_FILE_NAMES:
        text = read_text_file(directory / file_name, errors)
        texts[file_name] = text.strip() if text is not None else None
    if any(value is None for value in texts.values()):
        return None

    def require_slots(file_name, required_slots, exact_once=True):
        text = texts[file_name]
        for slot in required_slots:
            occurrences = text.count("{" + slot + "}")
            if occurrences != 1 and exact_once:
                errors.append(f"{file_name}: expected exactly one {{{slot}}} slot, found {occurrences}")
        try:
            unexpected = set(slot_names_in(text)) - set(required_slots)
        except ValueError as error:
            errors.append(f"{file_name}: {error}")
            return
        if unexpected:
            errors.append(f"{file_name}: unexpected slots {sorted(unexpected)}")

    require_slots(INSTRUCT_PERSONA_FRAME_FILE_NAME, ["persona_description"])
    require_slots(BASE_MODEL_PERSONA_AUTHOR_FRAME_FILE_NAME, ["persona_description"])
    require_slots(BASE_MODEL_PREAMBLE_TEMPLATE_WITH_AUTHOR_FILE_NAME, ["author_description", "generation_prompt"])
    require_slots(BASE_MODEL_PREAMBLE_TEMPLATE_WITHOUT_AUTHOR_FILE_NAME, ["generation_prompt"])
    for file_name in (INSTRUCT_GENERIC_SYSTEM_PROMPT_FILE_NAME, BASE_MODEL_GENERIC_AUTHOR_DESCRIPTION_FILE_NAME):
        if "{" in texts[file_name] or "}" in texts[file_name]:
            errors.append(f"{file_name}: contains a brace but takes no slots")
    return ModelTypeFraming(
        instruct_persona_frame=texts[INSTRUCT_PERSONA_FRAME_FILE_NAME],
        instruct_generic_system_prompt=texts[INSTRUCT_GENERIC_SYSTEM_PROMPT_FILE_NAME],
        base_preamble_template_with_author=texts[BASE_MODEL_PREAMBLE_TEMPLATE_WITH_AUTHOR_FILE_NAME],
        base_preamble_template_without_author=texts[BASE_MODEL_PREAMBLE_TEMPLATE_WITHOUT_AUTHOR_FILE_NAME],
        base_persona_author_frame=texts[BASE_MODEL_PERSONA_AUTHOR_FRAME_FILE_NAME],
        base_generic_author_description=texts[BASE_MODEL_GENERIC_AUTHOR_DESCRIPTION_FILE_NAME],
        directory=directory)


def read_generation_prompt_files(directory, errors):
    directory = Path(directory)
    template_text = read_text_file(directory / GENERATION_PROMPT_TEMPLATE_FILE_NAME, errors)
    facts_text = read_text_file(directory / IN_UNIVERSE_FACTS_FILE_NAME, errors)
    if template_text is None or facts_text is None:
        return None
    template_text, facts_text = template_text.strip(), facts_text.strip()
    try:
        slots = slot_names_in(template_text)
    except ValueError as error:
        errors.append(f"{GENERATION_PROMPT_TEMPLATE_FILE_NAME}: {error}")
        return None
    if "in_universe_facts" not in slots:
        errors.append(f"{GENERATION_PROMPT_TEMPLATE_FILE_NAME}: missing the {{in_universe_facts}} slot")
    if any(slot.isdigit() for slot in slots):
        errors.append(f"{GENERATION_PROMPT_TEMPLATE_FILE_NAME}: positional slots are not allowed")
    if "\n" in facts_text:
        errors.append(f"{IN_UNIVERSE_FACTS_FILE_NAME}: expected a single paragraph on one line")
    return GenerationPromptFiles(template_text=template_text, in_universe_facts=facts_text, directory=directory)


def load_model_type_framing(directory=None):
    """The shared framing files (instruct frame, generic prompt, the four base-model files)."""
    errors = []
    framing = read_framing_files(directory or PERSONA_CHECKS_PROMPT_DIRECTORY / MODEL_TYPE_FRAMING_DIRECTORY_NAME,
                                 errors)
    if errors:
        raise PersonaSetValidationError("Invalid model type framing:\n  " + "\n  ".join(errors))
    return framing


def load_generation_prompt_files(directory=None):
    errors = []
    files = read_generation_prompt_files(directory or PERSONA_CHECKS_PROMPT_DIRECTORY / GENERATION_PROMPT_DIRECTORY_NAME,
                                         errors)
    if errors:
        raise PersonaSetValidationError("Invalid generation prompt files:\n  " + "\n  ".join(errors))
    return files


# ---------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------

def require_keys(mapping, where, required, optional, errors):
    if not isinstance(mapping, dict):
        errors.append(f"{where}: expected an object")
        return False
    missing = [key for key in required if key not in mapping]
    unknown = [key for key in mapping if key not in required and key not in optional]
    if missing:
        errors.append(f"{where}: missing keys {missing}")
    if unknown:
        errors.append(f"{where}: unknown keys {unknown}")
    return not missing


def check_text(value, where, errors):
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}: expected a non-empty string, got {value!r}")
        return False
    return True


def check_positive_integer(value, where, errors):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        errors.append(f"{where}: expected a positive integer, got {value!r}")
        return False
    return True


def validate_configuration(configuration, folder_name, errors):
    top_level_required = ["schema_version", "persona_set_name", "persona_name", "condition_groups",
                          "rewrite_directions", "pass_criteria"]
    if not require_keys(configuration, CONFIGURATION_FILE_NAME, top_level_required, ["default_models"], errors):
        return
    where = CONFIGURATION_FILE_NAME
    if configuration["schema_version"] != 1:
        errors.append(f"{where}: unsupported schema_version {configuration['schema_version']!r}")
    if check_text(configuration["persona_set_name"], f"{where} persona_set_name", errors) \
            and configuration["persona_set_name"] != folder_name:
        errors.append(f"{where}: persona_set_name {configuration['persona_set_name']!r} differs from the folder "
                      f"name {folder_name!r}")
    names_in_use = {}

    def claim_name(name, where_claimed):
        if not isinstance(name, str) or not PERSONA_NAME_PATTERN.match(name):
            errors.append(f"{where_claimed}: {name!r} must be lowercase letters, digits and single underscores")
        elif name in RESERVED_PERSONA_NAMES:
            errors.append(f"{where_claimed}: {name!r} is reserved")
        elif name in names_in_use:
            errors.append(f"{where_claimed}: {name!r} is already used by {names_in_use[name]}")
        else:
            names_in_use[name] = where_claimed

    claim_name(configuration["persona_name"], f"{where} persona_name")

    groups = configuration["condition_groups"]
    if require_keys(groups, f"{where} condition_groups", ["target", "topic_control"], ["same_values_control"], errors):
        target = groups["target"]
        if require_keys(target, f"{where} condition_groups.target",
                        ["condition_group", "display_name", "expected_rewording_count"], [], errors):
            check_text(target["display_name"], "condition_groups.target.display_name", errors)
            check_positive_integer(target["expected_rewording_count"], "target.expected_rewording_count", errors)
        topic = groups["topic_control"]
        if require_keys(topic, f"{where} condition_groups.topic_control",
                        ["condition_group", "display_name", "expected_persona_count",
                         "expected_rewordings_per_persona"], [], errors):
            check_text(topic["display_name"], "condition_groups.topic_control.display_name", errors)
            check_positive_integer(topic["expected_persona_count"], "topic_control.expected_persona_count", errors)
            check_positive_integer(topic["expected_rewordings_per_persona"],
                                   "topic_control.expected_rewordings_per_persona", errors)
        same_values = groups.get("same_values_control")
        if same_values is not None and require_keys(
                same_values, f"{where} condition_groups.same_values_control",
                ["condition_group", "display_name", "expected_persona_count", "expected_rewordings_per_persona"],
                [], errors):
            check_text(same_values["display_name"], "condition_groups.same_values_control.display_name", errors)
            check_positive_integer(same_values["expected_persona_count"],
                                   "same_values_control.expected_persona_count", errors)
            check_positive_integer(same_values["expected_rewordings_per_persona"],
                                   "same_values_control.expected_rewordings_per_persona", errors)
        seen_groups = {}
        for group_key in ("target", "topic_control", "same_values_control"):
            group = groups.get(group_key)
            if isinstance(group, dict) and "condition_group" in group:
                value = group["condition_group"]
                if not isinstance(value, str) or not PERSONA_NAME_PATTERN.match(value) \
                        or value in RESERVED_PERSONA_NAMES:
                    errors.append(f"{where} {group_key}.condition_group {value!r} is not a valid group name")
                elif value in seen_groups:
                    errors.append(f"{where}: condition_group {value!r} used by both {seen_groups[value]} and "
                                  f"{group_key}")
                else:
                    seen_groups[value] = group_key

    directions = configuration["rewrite_directions"]
    if require_keys(directions, f"{where} rewrite_directions", ["toward", "neutral"], ["away"], errors):
        seen_direction_names = {}
        for role in DIRECTION_ROLES:
            if role not in directions:
                continue
            entry = directions[role]
            if require_keys(entry, f"{where} rewrite_directions.{role}", ["name"], [], errors):
                name = entry["name"]
                if not isinstance(name, str) or not PERSONA_NAME_PATTERN.match(name):
                    errors.append(f"{where} rewrite_directions.{role}.name {name!r} is not a valid name")
                elif name in seen_direction_names:
                    errors.append(f"{where}: rewrite direction name {name!r} used for both "
                                  f"{seen_direction_names[name]} and {role}")
                else:
                    seen_direction_names[name] = role

    validate_pass_criteria(configuration["pass_criteria"], directions if isinstance(directions, dict) else {}, errors)

    default_models = configuration.get("default_models")
    if default_models is not None and require_keys(default_models, f"{where} default_models", [], MODEL_TYPES, errors):
        for model_type, entry in default_models.items():
            if require_keys(entry, f"{where} default_models.{model_type}", ["model_name", "model_revision"], [], errors):
                check_text(entry["model_name"], f"default_models.{model_type}.model_name", errors)
                if not isinstance(entry["model_revision"], str) or not COMMIT_HASH_PATTERN.match(entry["model_revision"]):
                    errors.append(f"{where} default_models.{model_type}.model_revision must be a full 40-character "
                                  f"commit hash")


def validate_pass_criteria(criteria, directions, errors):
    where = f"{CONFIGURATION_FILE_NAME} pass_criteria"
    if not require_keys(criteria, where, ["status", "whole_document", "changed_span"], ["neutral_control"], errors):
        return
    check_text(criteria["status"], f"{where}.status", errors)
    for level in ("whole_document", "changed_span"):
        block = criteria[level]
        if not require_keys(block, f"{where}.{level}",
                            ["applies_to_directions", "minimum_number_moving_expected_way",
                             "number_of_rewrites_tested_per_direction"], [], errors):
            continue
        applies = block["applies_to_directions"]
        if not isinstance(applies, list) or not applies or any(role not in ("toward", "away") for role in applies) \
                or len(set(applies)) != len(applies):
            errors.append(f"{where}.{level}.applies_to_directions must be a non-empty list drawn from "
                          f"['toward', 'away'], got {applies!r}")
        else:
            for role in applies:
                if role not in directions:
                    errors.append(f"{where}.{level} applies to direction '{role}', which rewrite_directions "
                                  f"does not define")
        minimum, tested = block["minimum_number_moving_expected_way"], block["number_of_rewrites_tested_per_direction"]
        if check_positive_integer(minimum, f"{where}.{level}.minimum_number_moving_expected_way", errors) \
                and check_positive_integer(tested, f"{where}.{level}.number_of_rewrites_tested_per_direction", errors) \
                and minimum > tested:
            errors.append(f"{where}.{level}: minimum {minimum} exceeds the number tested {tested}")
    neutral = criteria.get("neutral_control")
    if neutral is not None and require_keys(neutral, f"{where}.neutral_control", ["changed_span"], [], errors):
        block = neutral["changed_span"]
        if require_keys(block, f"{where}.neutral_control.changed_span",
                        ["maximum_number_moving_toward_persona", "number_of_rewrites_tested"], [], errors):
            maximum, tested = block["maximum_number_moving_toward_persona"], block["number_of_rewrites_tested"]
            if check_positive_integer(tested, "neutral_control number_of_rewrites_tested", errors) \
                    and (isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= tested):
                errors.append(f"{where}.neutral_control: maximum_number_moving_toward_persona must be an integer "
                              f"between 0 and {tested}")


# ---------------------------------------------------------------------------------------------------
# Whole persona set
# ---------------------------------------------------------------------------------------------------

def resolve_persona_set_folder(persona_set, persona_checks_directory=None):
    """`persona_set` is a persona set name (a folder under prompts/tools/persona_checks/) or a path to a folder."""
    persona_checks_directory = Path(persona_checks_directory or PERSONA_CHECKS_PROMPT_DIRECTORY)
    as_path = Path(persona_set)
    if as_path.is_dir() and (as_path / CONFIGURATION_FILE_NAME).is_file():
        return as_path.resolve()
    candidate = persona_checks_directory / str(persona_set)
    if candidate.is_dir():
        return candidate
    raise PersonaSetValidationError(
        f"Persona set {persona_set!r} not found: no folder {candidate} and no folder at that path holding "
        f"{CONFIGURATION_FILE_NAME}. Available: {list_persona_set_names(persona_checks_directory)}")


def list_persona_set_names(persona_checks_directory=None):
    persona_checks_directory = Path(persona_checks_directory or PERSONA_CHECKS_PROMPT_DIRECTORY)
    if not persona_checks_directory.is_dir():
        return []
    return sorted(path.name for path in persona_checks_directory.iterdir()
                  if path.is_dir() and (path / CONFIGURATION_FILE_NAME).is_file())


def validate_persona_set(persona_set, persona_checks_directory=None, framing_directory=None,
                         generation_prompt_directory=None):
    """Checks a persona set and the shared files it is scored with. Never raises for content problems; returns a
    PersonaSetValidationReport whose `errors` list every problem found and whose `persona_set` is filled when
    there are none."""
    report = PersonaSetValidationReport()
    errors, warnings = report.errors, report.warnings
    try:
        folder = resolve_persona_set_folder(persona_set, persona_checks_directory)
    except PersonaSetValidationError as error:
        errors.append(str(error))
        return report

    framing_errors = []
    read_framing_files(framing_directory or (Path(persona_checks_directory or PERSONA_CHECKS_PROMPT_DIRECTORY)
                                             / MODEL_TYPE_FRAMING_DIRECTORY_NAME), framing_errors)
    errors.extend(f"model_type_framing: {message}" for message in framing_errors)
    generation_errors = []
    read_generation_prompt_files(generation_prompt_directory or (
        Path(persona_checks_directory or PERSONA_CHECKS_PROMPT_DIRECTORY) / GENERATION_PROMPT_DIRECTORY_NAME),
        generation_errors)
    errors.extend(f"generation_prompt: {message}" for message in generation_errors)

    configuration = None
    configuration_text = read_text_file(folder / CONFIGURATION_FILE_NAME, errors)
    if configuration_text is not None:
        try:
            configuration = json.loads(configuration_text)
        except json.JSONDecodeError as error:
            errors.append(f"{CONFIGURATION_FILE_NAME}: not valid JSON: {error}")
    configuration_errors = []
    if configuration is not None:
        validate_configuration(configuration, folder.name, configuration_errors)
    errors.extend(configuration_errors)
    configuration_ok = configuration is not None and not configuration_errors

    expected_files = set(REQUIRED_PERSONA_SET_FILE_NAMES)
    has_same_values_group = configuration_ok and "same_values_control" in configuration["condition_groups"]
    if has_same_values_group:
        expected_files.add(SAME_VALUES_CONTROL_FILE_NAME)
    elif configuration_ok and (folder / SAME_VALUES_CONTROL_FILE_NAME).exists():
        errors.append(f"{SAME_VALUES_CONTROL_FILE_NAME} exists but the configuration has no same_values_control group")
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name not in expected_files and path.name != SAME_VALUES_CONTROL_FILE_NAME:
            warnings.append(f"Unexpected file in the persona set folder: {path.name}")

    target_rewordings, topic_by_persona, same_values_by_persona = [], {}, None
    target_text = read_text_file(folder / TARGET_REWORDINGS_FILE_NAME, errors)
    if target_text is not None:
        target_rewordings = parse_description_lines(target_text, TARGET_REWORDINGS_FILE_NAME, errors)
    topic_text = read_text_file(folder / TOPIC_CONTROL_FILE_NAME, errors)
    if topic_text is not None:
        topic_by_persona = parse_grouped_descriptions(topic_text, TOPIC_CONTROL_FILE_NAME, errors)
    if has_same_values_group:
        same_values_text = read_text_file(folder / SAME_VALUES_CONTROL_FILE_NAME, errors)
        if same_values_text is not None:
            same_values_by_persona = parse_grouped_descriptions(same_values_text, SAME_VALUES_CONTROL_FILE_NAME, errors)

    # Descriptions: wording, duplicates
    seen_descriptions = {}
    descriptions_to_check = [(TARGET_REWORDINGS_FILE_NAME, description) for description in target_rewordings]
    for persona_name, descriptions in topic_by_persona.items():
        descriptions_to_check += [(f"{TOPIC_CONTROL_FILE_NAME} ({persona_name})", description)
                                  for description in descriptions]
    for persona_name, descriptions in (same_values_by_persona or {}).items():
        descriptions_to_check += [(f"{SAME_VALUES_CONTROL_FILE_NAME} ({persona_name})", description)
                                  for description in descriptions]
    for where, description in descriptions_to_check:
        check_description_reads_after_frames(description, where, errors)
        key = description.lower()
        if key in seen_descriptions:
            errors.append(f"Duplicate description in {where} (first seen in {seen_descriptions[key]}): {description!r}")
        else:
            seen_descriptions[key] = where

    # Counts and header names against the configuration
    if configuration_ok:
        groups = configuration["condition_groups"]
        if len(target_rewordings) != groups["target"]["expected_rewording_count"]:
            errors.append(f"{TARGET_REWORDINGS_FILE_NAME}: expected {groups['target']['expected_rewording_count']} "
                          f"rewordings, found {len(target_rewordings)}")
        topic = groups["topic_control"]
        if len(topic_by_persona) != topic["expected_persona_count"]:
            errors.append(f"{TOPIC_CONTROL_FILE_NAME}: expected {topic['expected_persona_count']} personas, "
                          f"found {len(topic_by_persona)}")
        grouped_files = [(TOPIC_CONTROL_FILE_NAME, topic_by_persona, topic)]
        if has_same_values_group and same_values_by_persona is not None:
            same_values = groups["same_values_control"]
            if len(same_values_by_persona) != same_values["expected_persona_count"]:
                errors.append(f"{SAME_VALUES_CONTROL_FILE_NAME}: expected {same_values['expected_persona_count']} "
                              f"personas, found {len(same_values_by_persona)}")
            grouped_files.append((SAME_VALUES_CONTROL_FILE_NAME, same_values_by_persona, same_values))
        persona_name_files = {}
        for file_name, descriptions_by_persona, group_configuration in grouped_files:
            for persona_name, descriptions in descriptions_by_persona.items():
                if len(descriptions) != group_configuration["expected_rewordings_per_persona"]:
                    errors.append(f"{file_name}: persona '{persona_name}' has {len(descriptions)} "
                                  f"rewordings, expected {group_configuration['expected_rewordings_per_persona']}")
                if not PERSONA_NAME_PATTERN.match(persona_name):
                    errors.append(f"{file_name}: header '{persona_name}' must be lowercase letters, "
                                  f"digits and single underscores")
                elif persona_name in RESERVED_PERSONA_NAMES or persona_name == configuration["persona_name"]:
                    errors.append(f"{file_name}: header '{persona_name}' collides with a reserved or "
                                  f"configured persona name")
                elif persona_name in persona_name_files:
                    errors.append(f"{file_name}: header '{persona_name}' is already used in "
                                  f"{persona_name_files[persona_name]}")
                else:
                    persona_name_files[persona_name] = file_name

    # Rewrite instructions
    instruction_texts, draft_flags = {}, {}
    for role, file_name in (("positive", POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME),
                            ("neutral", NEUTRAL_CONTROL_INSTRUCTIONS_FILE_NAME)):
        text = read_text_file(folder / file_name, errors)
        if text is None:
            continue
        text, is_draft = split_off_draft_marker(text)
        instruction_texts[role], draft_flags[role] = text, is_draft
        if is_draft:
            warnings.append(f"{file_name} is marked '{DRAFT_MARKER}': it needs human review before any paid use")
        for slot in ("{direction}", "{document_text}"):
            if text.count(slot) != 1:
                errors.append(f"{file_name}: expected exactly one {slot} slot, found {text.count(slot)}")
        if configuration_ok:
            directions = configuration["rewrite_directions"]
            roles_named = ["toward", "away"] if role == "positive" else ["neutral"]
            for direction_role in roles_named:
                if direction_role in directions and directions[direction_role]["name"].upper() not in text:
                    errors.append(f"{file_name}: does not describe the direction "
                                  f"{directions[direction_role]['name'].upper()}")

    if errors:
        return report
    report.persona_set = PersonaSet(
        name=folder.name, folder=folder, configuration=configuration, target_rewordings=target_rewordings,
        topic_control_descriptions_by_persona=topic_by_persona, same_values_control_descriptions_by_persona=same_values_by_persona,
        positive_control_rewrite_instructions=instruction_texts["positive"],
        neutral_control_rewrite_instructions=instruction_texts["neutral"],
        positive_control_instructions_are_draft=draft_flags["positive"],
        neutral_control_instructions_are_draft=draft_flags["neutral"],
        file_sha256={path.name: sha256_of_file(path) for path in sorted(folder.iterdir()) if path.is_file()})
    return report


def load_persona_set(persona_set, persona_checks_directory=None, framing_directory=None,
                     generation_prompt_directory=None):
    """Loads and validates a persona set by name or folder path. Raises PersonaSetValidationError listing every
    problem. Draft rewrite instructions load fine (the marker line is stripped and recorded in the
    `*_instructions_are_draft` flags); paid steps call `PersonaSet.require_reviewed_instructions`."""
    report = validate_persona_set(persona_set, persona_checks_directory, framing_directory,
                                  generation_prompt_directory)
    if not report.is_valid:
        raise PersonaSetValidationError(f"Persona set {persona_set!r} is invalid:\n  " + "\n  ".join(report.errors))
    return report.persona_set


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description="Validate a persona set for the log-probability persona check.")
    parser.add_argument("--persona-set", required=True, help="Persona set name or folder")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    report = validate_persona_set(arguments.persona_set)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    if report.is_valid:
        persona_set = report.persona_set
        print(f"Persona set {persona_set.name} is valid: {persona_set.expected_condition_count()} conditions "
              f"({len(persona_set.target_rewordings)} target rewordings, "
              f"{len(persona_set.topic_control_descriptions_by_persona)} topic control personas, "
              f"{len(persona_set.same_values_control_descriptions_by_persona or {})} same-values control personas)")
        return 0
    print(f"Persona set {arguments.persona_set} is invalid ({len(report.errors)} errors)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
