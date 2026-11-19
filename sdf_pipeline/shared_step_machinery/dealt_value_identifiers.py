"""The ``[id]`` of each dealt value in ``variables.txt``, kept apart from the prompt text. A step never works ids out: it copies
``variable_value_ids`` from the record it reads when that has the field, so a run without the field never gains it."""

import json
from pathlib import Path

VALUE_IDS_FIELD = "variable_value_ids"
RECORDS_VALUE_IDS_MANIFEST_KEY = "dealt_value_ids_in_stage_files"

# Plan variables that are not dealt values: the selected consideration labels a plan gains.
NON_DEALT_PLAN_VARIABLES = ("moral_considerations", "epistemic_considerations")


def run_records_value_ids(layer_directory) -> bool:
    """Whether the run holding this layer folder records ``variable_value_ids`` (its manifest, written
    at creation, says so). A run without that manifest key, or without a readable manifest, does not."""
    manifest_path = Path(layer_directory).parent / "run_manifest.json"
    try:
        return bool(json.loads(manifest_path.read_text(encoding="utf-8")).get(RECORDS_VALUE_IDS_MANIFEST_KEY))
    except (OSError, ValueError, AttributeError):
        return False


def value_ids_field_of(record: dict) -> dict:
    """``{"variable_value_ids": ...}`` when the record carries it, else ``{}``: spread it into the dict
    of a record that copies ``variables``, right after ``variables``."""
    return {VALUE_IDS_FIELD: record[VALUE_IDS_FIELD]} if VALUE_IDS_FIELD in record else {}


def deal_hand_records_with_value_ids(
        template: str, values: dict, weights: dict, hand_count: int, seed: int, deal_rules: list,
        value_ids_by_axis: dict, first_index: int = 0) -> tuple[list[dict], dict, dict]:
    """The one deal, screen on or off: ``(hands, ids_by_hand, notes_by_hand)``; notes name hands with a value that has no id."""
    from sdf_pipeline.shared_step_machinery import compose_prompts
    hands = compose_prompts.hand_records(
        compose_prompts.deal_hands(template, values, weights, hand_count, seed, deal_rules), first_index)
    ids_by_hand, notes_by_hand = value_ids_of_hands(hands, value_ids_by_axis)
    return hands, ids_by_hand, notes_by_hand


def announce_hands_with_a_value_that_has_no_unique_id(notes_by_hand: dict, hand_count: int) -> None:
    """Print how many of ``hand_count`` hands carry a dealt value with no id or an ambiguous one, with the first note; nothing when none do."""
    if notes_by_hand:
        print(f"  {len(notes_by_hand)} of {hand_count} hands have a dealt value with no unique id "
              f"(first: {next(iter(notes_by_hand.values()))[0]})")


def value_ids_of_hands(hands: list[dict], value_ids_by_axis: dict) -> tuple[dict, dict]:
    """``({hand id: {axis: value id}}, {hand id: [notes]})`` for hand records. Read-only."""
    ids_by_hand: dict = {}
    notes_by_hand: dict = {}
    for hand in hands:
        ids, notes = value_ids_of_dealt_hand(hand["variables"], value_ids_by_axis)
        ids_by_hand[hand["prompt_id"]] = ids
        if notes:
            notes_by_hand[hand["prompt_id"]] = notes
    return ids_by_hand, notes_by_hand


def value_ids_of_dealt_hand(dealt_variables: dict, value_ids_by_axis: dict) -> tuple[dict, list]:
    """``({axis: value id}, notes)`` for one hand's ``{axis: text}``, matched against
    ``{axis: {id: text}}``. A text no id has, or that two ids share, gets no id and a note."""
    ids_by_axis: dict = {}
    notes: list = []
    for axis, text in (dealt_variables or {}).items():
        if axis in NON_DEALT_PLAN_VARIABLES:
            continue   # selected consideration labels, not dealt values
        ids_of_text = [value_id for value_id, candidate in (value_ids_by_axis.get(axis) or {}).items()
                       if candidate == text]
        if len(ids_of_text) == 1:
            ids_by_axis[axis] = ids_of_text[0]
        elif not ids_of_text:
            notes.append(f"dealt value of {{{axis}}} has no id: no value of that axis has this text "
                         f"({text[:50]!r})")
        else:
            notes.append(f"dealt value of {{{axis}}} has no id: {len(ids_of_text)} values share this text "
                         f"({text[:50]!r})")
    return ids_by_axis, notes
