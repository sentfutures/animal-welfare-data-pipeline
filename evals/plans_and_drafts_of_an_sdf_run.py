"""The plan and draft records of an SDF run, read from its stage files in whatever layout the run has.

A missing stage file is an empty list, so a folder that is not an SDF run, and a run in the layout from before
the variables matrix (which has no plan file), give no plans."""

from pathlib import Path

from shared import sdf_run_layouts, utils


def _hand_index(document_id: str) -> int:
    try:
        return int(document_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _last_record_per_id(records: list, id_field: str) -> dict:
    """{id: record}, a later record replacing an earlier one in the position of the first."""
    return {record[id_field]: record for record in records if record.get(id_field) is not None}


def _dealt_ids_in_deal_order(run_directory: Path) -> list:
    """Hand ids in deal order: with a screen, the hands of ``screen_hands.jsonl`` that have a recorded round,
    by round then hand number; without one, the order of ``prompts.jsonl``."""
    screen_folder = utils.sdf_stage_folder(run_directory, sdf_run_layouts.DEAL_AND_SCREEN)
    screen_hands = utils.load_jsonl(screen_folder / "screen_hands.jsonl")
    if not screen_hands:
        return list(_last_record_per_id(utils.load_jsonl(utils.sdf_stage_file(run_directory, "dealt")), "prompt_id"))
    recorded_ids = {document_id for screen_round in utils.load_jsonl(screen_folder / "screen_rounds.jsonl")
                    for document_id in screen_round.get("prompt_ids", [])}
    round_of_hand = {hand["prompt_id"]: hand.get("round", 0) for hand in screen_hands
                     if hand.get("prompt_id") in recorded_ids}
    return sorted(round_of_hand, key=lambda document_id: (round_of_hand[document_id], _hand_index(document_id)))


def plan_records_of_run(run_directory) -> list:
    """The plan stage's records as ``prompt_id``, ``variables`` and ``description`` (the planner's own text,
    None for a plan declared incoherent, which is stored without one), one per planned document, in deal order
    and then in the order of the plan file for documents the deal does not hold."""
    plans = _last_record_per_id(utils.load_jsonl(utils.sdf_stage_file(run_directory, "plan")), "prompt_id")
    prompts = _last_record_per_id(utils.load_jsonl(utils.sdf_stage_file(run_directory, "dealt")), "prompt_id")
    ordered_ids = dict.fromkeys([*_dealt_ids_in_deal_order(Path(run_directory)), *prompts, *plans])
    return [{"prompt_id": document_id, "variables": plans[document_id].get("variables"),
             "description": plans[document_id].get("description") or None}
            for document_id in ordered_ids if document_id in plans]


def draft_texts_by_document_id(run_directory) -> dict:
    """{document id: text of its draft} for every document with a draft record."""
    drafts = _last_record_per_id(utils.load_jsonl(utils.sdf_stage_file(run_directory, "draft")), "doc_id")
    return {document_id: draft.get("content", "") for document_id, draft in drafts.items()}
