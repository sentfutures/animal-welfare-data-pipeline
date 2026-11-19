"""Layer 1: the deal (no model call): the round-0 hands, deck-sampled and repaired by the deal rules, as ``{"prompt_id", "round", "variables"}``."""

from shared import deal_constraints
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.layer1_deal_variables_and_screen.shared_by_this_layers_steps import hand_record


def deal_original_hands(
        template: str, values: dict, weights: dict, hand_count: int, seed: int,
        deal_rules: list[deal_constraints.DealRule]) -> list[dict]:
    """The hand records of round 0, in dealt order."""
    dealt = cp.hand_records(cp.deal_hands(template, values, weights, hand_count, seed, deal_rules))
    return [hand_record(hand["prompt_id"], 0, hand["variables"]) for hand in dealt]
