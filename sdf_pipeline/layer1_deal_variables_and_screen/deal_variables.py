"""Layer 1: the deal (no model call): the round-0 hands, deck-sampled and repaired by the deal rules, as ``{"prompt_id", "round", "variables"}``
(plus ``variable_value_ids`` when the run records them). It is the one deal: the plan layer calls the same function with the screen off."""

from shared import deal_constraints
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import (
    announce_hands_with_a_value_that_has_no_unique_id, deal_hand_records_with_value_ids)
from sdf_pipeline.layer1_deal_variables_and_screen.shared_by_this_layers_steps import hand_record


def deal_original_hands(
        template: str, values: dict, weights: dict, hand_count: int, seed: int,
        deal_rules: list[deal_constraints.DealRule], value_ids_by_axis: dict | None = None,
        record_value_ids: bool = False) -> list[dict]:
    """The hand records of round 0, in dealt order."""
    dealt, ids_by_hand, notes_by_hand = deal_hand_records_with_value_ids(
        template, values, weights, hand_count, seed, deal_rules, value_ids_by_axis or {})
    announce_hands_with_a_value_that_has_no_unique_id(notes_by_hand, len(dealt))
    return [hand_record(hand["prompt_id"], 0, hand["variables"],
                        ids_by_hand[hand["prompt_id"]] if record_value_ids else None)
            for hand in dealt]
