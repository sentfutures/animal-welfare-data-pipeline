"""Layer 1 step: the redeal of one finished round (no model call).

Removed hands are redealt so the dealt shares stay at the weights: their cards go back into a per-column pool,
each column is reshuffled with a seed derived from the run seed and the round, the pool is dealt into as many new
hands as were removed, and the deal-rule repair swaps cards only among those new hands.
"""

import json
import random

from shared import deal_constraints
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import value_ids_of_hands
from sdf_pipeline.layer1_deal_and_screen.shared_by_this_layers_steps import MAX_REDEAL_ATTEMPTS, SHUFFLE_SEED_NAME, hand_record
from sdf_pipeline.shared_step_machinery.update_steps import CorpusStep, Created, Removed


def redeal_removed_hands(
    removed_hands: list[dict],
    round_number: int,
    seed: int,
    deal_rules: list[deal_constraints.DealRule],
) -> list[dict[str, str]] | None:
    """New hands (variables only) dealt from the removed hands' cards, or None.

    Each column of the pool is reshuffled with a seed derived from the run seed
    and the round number, and the shuffled columns are zipped into
    len(removed_hands) new hands, so every column keeps exactly its cards. The
    deal-rule repair then swaps cards among these new hands only. A reshuffle
    that cannot be repaired, or that just reproduces the removed hands, is
    retried with the next attempt's seed; after MAX_REDEAL_ATTEMPTS the cards are
    given up on (None)."""
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


class RedealStep(CorpusStep):
    """The redeal of one finished round (no model). ``states`` are every hand dealt so far.

    Returns ``Removed`` for each screened-out hand of the round (in the round's order) and, when the
    removed hands' cards can be dealt into different hands that satisfy the rules, ``Created`` for each
    new hand (keyed by its new hand id, numbered on from the highest hand id so far). When they cannot,
    the cards are dropped: the removals carry the reason code ``screened_out_cards_dropped`` and nothing
    is created, which ends the screen loop."""

    step_name = "redeal"

    def __init__(self, finished_round_number: int, removed_prompt_ids: list[str],
                 reason_by_prompt_id: dict[str, str], seed: int, deal_rules: list[deal_constraints.DealRule],
                 value_ids_by_axis: dict | None = None, record_value_ids: bool = False):
        self.new_round_number = finished_round_number + 1
        self.removed_prompt_ids = list(removed_prompt_ids)
        self.reason_by_prompt_id = reason_by_prompt_id
        self.seed = seed
        self.deal_rules = deal_rules
        self.value_ids_by_axis = value_ids_by_axis or {}
        self.record_value_ids = record_value_ids

    def apply(self, states: list) -> dict:
        hands_by_id = {hand["prompt_id"]: hand for hand in states}
        new_hands = redeal_removed_hands(
            [hands_by_id[prompt_id] for prompt_id in self.removed_prompt_ids],
            self.new_round_number, self.seed, self.deal_rules)
        reason_code = "screened_out" if new_hands is not None else "screened_out_cards_dropped"
        outcomes: dict = {
            prompt_id: Removed(reason_code, self.reason_by_prompt_id[prompt_id])
            for prompt_id in self.removed_prompt_ids}
        if new_hands is not None:
            new_records = cp.hand_records(new_hands, _next_hand_index(states))
            ids_by_hand, _ = value_ids_of_hands(new_records, self.value_ids_by_axis)
            for record in new_records:
                outcomes[record["prompt_id"]] = Created(hand_record(
                    record["prompt_id"], self.new_round_number, record["variables"],
                    ids_by_hand[record["prompt_id"]] if self.record_value_ids else None))
        return outcomes


def _next_hand_index(hands: list[dict]) -> int:
    return max((int(h["prompt_id"].rsplit("_", 1)[1]) for h in hands), default=-1) + 1
