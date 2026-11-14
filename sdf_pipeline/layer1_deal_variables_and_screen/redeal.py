"""Layer 1: the redeal of one finished round (no model call). Removed hands' cards go back into per-column pools, are
reshuffled with a seed from the run seed and the round, and are dealt into new hands; repair swaps only among the new hands."""

import json
import random

from shared import deal_constraints
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import (
    announce_hands_with_a_value_that_has_no_unique_id, value_ids_of_hands)
from sdf_pipeline.layer1_deal_variables_and_screen.shared_by_this_layers_steps import MAX_REDEAL_ATTEMPTS, SHUFFLE_SEED_NAME, hand_record


def redeal_removed_hands(
    removed_hands: list[dict],
    round_number: int,
    seed: int,
    deal_rules: list[deal_constraints.DealRule],
) -> list[dict[str, str]] | None:
    """New hands (variables only) from the removed hands' cards, or None after MAX_REDEAL_ATTEMPTS.
    An unrepairable reshuffle, or one that reproduces the removed hands, is retried with the next attempt's seed."""
    axes = list(removed_hands[0]["variables"])
    removed_as_dealt = sorted(json.dumps(h["variables"], sort_keys=True) for h in removed_hands)
    for attempt in range(MAX_REDEAL_ATTEMPTS):
        rng = random.Random(f"{seed}:{SHUFFLE_SEED_NAME}:redeal{round_number}:attempt{attempt}")
        columns = {}
        for axis in axes:
            column = [hand["variables"][axis] for hand in removed_hands]
            rng.shuffle(column)
            columns[axis] = column
        new_hands = [{axis: columns[axis][i] for axis in axes} for i in range(len(removed_hands))]
        try:
            new_hands, _ = deal_constraints.swap_repair(new_hands, deal_rules, rng)
        except deal_constraints.InfeasibleDealConstraintError:
            continue
        if sorted(json.dumps(h, sort_keys=True) for h in new_hands) == removed_as_dealt:
            continue  # the same hands again would only be removed again
        return new_hands
    return None


def redeal_screened_out_hands(
        hands: list[dict], removed_prompt_ids: list[str], finished_round_number: int, seed: int,
        deal_rules: list[deal_constraints.DealRule], value_ids_by_axis: dict | None = None,
        record_value_ids: bool = False) -> list[dict]:
    """The hand records of the round after ``finished_round_number``, numbered on from the highest hand id; empty when the cards cannot be redealt (which ends the screen loop)."""
    new_round_number = finished_round_number + 1
    hands_by_id = {hand["prompt_id"]: hand for hand in hands}
    new_hands = redeal_removed_hands(
        [hands_by_id[prompt_id] for prompt_id in removed_prompt_ids], new_round_number, seed, deal_rules)
    if new_hands is None:
        return []
    new_records = cp.hand_records(new_hands, _next_hand_index(hands))
    ids_by_hand, notes_by_hand = value_ids_of_hands(new_records, value_ids_by_axis or {})
    announce_hands_with_a_value_that_has_no_unique_id(notes_by_hand, len(new_records))
    return [hand_record(record["prompt_id"], new_round_number, record["variables"],
                        ids_by_hand[record["prompt_id"]] if record_value_ids else None)
            for record in new_records]


def _next_hand_index(hands: list[dict]) -> int:
    return max((int(h["prompt_id"].rsplit("_", 1)[1]) for h in hands), default=-1) + 1
