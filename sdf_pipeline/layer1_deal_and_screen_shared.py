"""Layer 1 (deal and screen): what the deal, the screen and the redeal share, and the readers of the screen's files.

The file names and defaults the three steps and the layer's driver use, the seed name the shuffles keep, and the
functions the audit and the report read a run's screen results with (``load_screen_outputs`` and the row
builders). See ``layer1_deal_and_screen.py`` for the stage as a whole.
"""

import json
from pathlib import Path

from shared import sdf_run_layouts, utils

# The shuffles of the screen's batches and of the redeal are seeded with this text (with the run seed and the
# round). It is the old stage tag, kept literal: deriving it from a layout's tag would change every batch.
SHUFFLE_SEED_NAME = "layer1_screen"
TEMPLATE_FILENAME = "layer1_screen.txt"
HAND_BLOCK_FILENAME = "layer1_screen_specification_block.txt"
SCHEMA_FILENAME = "layer1_screen_answer_schema.json"
HANDS_FILENAME = "screen_hands.jsonl"
ROUNDS_FILENAME = "screen_rounds.jsonl"
BATCHES_FILENAME = "screen_batches.jsonl"
JUDGMENTS_FILENAME = "screen_judgments.jsonl"
REJECTIONS_FILENAME = "screen_rejections.jsonl"
CHECKPOINT_FILENAME = "_screen_checkpoint.json"
SCREENED_HANDS_FILENAME = "screened_hands.jsonl"
SUMMARY_FILENAME = "screen_summary.json"
DEFAULT_BATCH_SIZE = 10
DEFAULT_REDEAL_ROUNDS = 3
MAX_REDEAL_ATTEMPTS = 25  # reshuffles tried per round before the removed cards are given up on
_MAX_TOKENS = 8000  # reasoning plus a decision for each of 10 hands; a batch used 1,819 and another hit 2,000 on 2026-10-22

# Values an evaluation or report flags: a variable value removed at more than
# this multiple of the corpus-wide removal rate (with at least MIN_REMOVED_TO_FLAG
# removals, so a single removal in a tiny cell is not a finding).
FLAG_RATE_MULTIPLE = 2.0
MIN_REMOVED_TO_FLAG = 2


def hand_record(prompt_id: str, round_number: int, variables: dict, value_ids: dict | None = None) -> dict:
    """A hand as screen_hands.jsonl holds it; ``variable_value_ids`` follows ``variables`` when given."""
    record = {"prompt_id": prompt_id, "round": round_number, "variables": variables}
    if value_ids is not None:
        record["variable_value_ids"] = value_ids
    return record


def load_rejections(output_dir: Path) -> dict[str, dict]:
    """{prompt_id: rejection record}, de-duplicated."""
    return {r["prompt_id"]: r for r in utils.load_jsonl(output_dir / REJECTIONS_FILENAME)}


# ---------------------------------------------------------------- reporting

def load_screen_outputs(run_dir: Path) -> dict | None:
    """The screen's results for a run, or None when it never ran (older runs,
    which screened planned descriptions and have no screen_hands.jsonl).

    Returns {"hands": every hand ever dealt, "rounds", "removed": {prompt_id:
    rejection record}, "screened_ids": ids in a checkpointed batch,
    "accepted_ids": set or None until the screen finishes, "leftover_ids",
    "dealt_hands": the round-0 hands, "batches_total", "batches_done"}."""
    stage_dir = utils.sdf_stage_folder(run_dir, sdf_run_layouts.DEAL_AND_SCREEN)
    hands = utils.load_jsonl(stage_dir / HANDS_FILENAME)
    if not hands:
        return None
    batches = utils.load_jsonl(stage_dir / BATCHES_FILENAME)
    checkpoint = utils.Checkpoint(stage_dir / CHECKPOINT_FILENAME)
    done = [b for b in batches if checkpoint.is_done(b["batch_id"])]
    accepted = utils.load_jsonl(stage_dir / SCREENED_HANDS_FILENAME)
    summary_path = stage_dir / SUMMARY_FILENAME
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {
        "hands": hands,
        "rounds": utils.load_jsonl(stage_dir / ROUNDS_FILENAME),
        "dealt_hands": [h for h in hands if h["round"] == 0],
        "screened_ids": {pid for b in done for pid in b["prompt_ids"]},
        "removed": load_rejections(stage_dir),
        "accepted_ids": {h["prompt_id"] for h in accepted} if summary else None,
        "leftover_ids": set(summary["leftover_prompt_ids"]) if summary else set(),
        "batches_total": len(batches),
        "batches_done": len(done),
    }


def removal_rate_rows(screen: dict, axes: list[str] | None = None) -> list[dict]:
    """Removal rate per variable value over every screened hand, counting every
    round (a redealt hand is a fresh hand to screen).

    Each row: variable, value, screened, removed, rate, corpus_rate, flagged
    (rate above FLAG_RATE_MULTIPLE x the corpus rate with at least
    MIN_REMOVED_TO_FLAG removals). Sorted flagged first, then by rate."""
    screened = [h for h in screen["hands"] if h["prompt_id"] in screen["screened_ids"]]
    removed_ids = set(screen["removed"])
    corpus_rate = (sum(h["prompt_id"] in removed_ids for h in screened) / len(screened)) if screened else 0.0
    counts: dict[tuple[str, str], list[int]] = {}
    for hand in screened:
        for axis, value in hand["variables"].items():
            if axes is not None and axis not in axes:
                continue
            cell = counts.setdefault((axis, value), [0, 0])
            cell[0] += 1
            cell[1] += hand["prompt_id"] in removed_ids
    rows = []
    for (axis, value), (n_screened, n_removed) in counts.items():
        rate = n_removed / n_screened
        rows.append({
            "variable": axis, "value": value, "screened": n_screened, "removed": n_removed,
            "rate": rate, "corpus_rate": corpus_rate,
            "flagged": n_removed >= MIN_REMOVED_TO_FLAG and rate > FLAG_RATE_MULTIPLE * corpus_rate,
        })
    rows.sort(key=lambda r: (not r["flagged"], -r["rate"], r["variable"], r["value"]))
    return rows


def round_rows(screen: dict) -> list[dict]:
    """One row per round: hands screened, hands removed."""
    removed_ids = set(screen["removed"])
    return [
        {"round": r["round"], "hands": len(r["prompt_ids"]),
         "removed": sum(pid in removed_ids for pid in r["prompt_ids"]), "note": r.get("note", "")}
        for r in screen["rounds"]
    ]


def share_shift_rows(screen: dict) -> list[dict]:
    """Per variable value, its share of the dealt hands against its share of the
    accepted hands: the shift caused by the leftover hands dropped after the last
    round. Empty until the screen has finished. Sorted by absolute shift."""
    if screen["accepted_ids"] is None:
        return []
    dealt = screen["dealt_hands"]
    accepted = [h for h in screen["hands"] if h["prompt_id"] in screen["accepted_ids"]]
    counts: dict[tuple[str, str], list[int]] = {}
    for position, group in enumerate((dealt, accepted)):
        for hand in group:
            for axis, value in hand["variables"].items():
                counts.setdefault((axis, value), [0, 0])[position] += 1
    rows = []
    for (axis, value), (n_dealt, n_accepted) in counts.items():
        dealt_share = n_dealt / len(dealt) if dealt else 0.0
        final_share = n_accepted / len(accepted) if accepted else 0.0
        rows.append({
            "variable": axis, "value": value, "dealt": n_dealt, "accepted": n_accepted,
            "dealt_share": dealt_share, "final_share": final_share, "shift": final_share - dealt_share,
        })
    rows.sort(key=lambda r: (-abs(r["shift"]), r["variable"], r["value"]))
    return rows
