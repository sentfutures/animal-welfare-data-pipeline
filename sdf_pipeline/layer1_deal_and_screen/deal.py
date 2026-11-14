"""Layer 1 step: the deal (no model call).

The original hands of round 0, deck-sampled and repaired by the deal rules. ``Created`` for each hand, keyed by
its hand id; the hand record is ``{"prompt_id", "round", "variables"}``, as screen_hands.jsonl holds it, plus
``variable_value_ids`` (each dealt value's id) when the run records them.
"""

from shared import deal_constraints
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import deal_hand_records_with_value_ids
from sdf_pipeline.layer1_deal_and_screen.shared_by_this_layers_steps import hand_record
from sdf_pipeline.shared_step_machinery.update_steps import CorpusStep, Created


class DealStep(CorpusStep):
    """The deal: the original hands of round 0, deck-sampled and repaired by the deal rules (no model).
    ``Created`` for each hand, keyed by its hand id; the hand record is
    ``{"prompt_id", "round", "variables"}``, as screen_hands.jsonl holds it, plus ``variable_value_ids``
    (each dealt value's id) when ``record_value_ids`` is set (a run that records them).

    This is the one deal: the plan layer calls the same ``deal_hand_records_with_value_ids`` with the
    screen off."""

    step_name = "deal"

    def __init__(self, template: str, values: dict, weights: dict, hand_count: int, seed: int,
                 deal_rules: list[deal_constraints.DealRule], value_ids_by_axis: dict | None = None,
                 record_value_ids: bool = False):
        self.template = template
        self.values = values
        self.weights = weights
        self.hand_count = hand_count
        self.seed = seed
        self.deal_rules = deal_rules
        self.value_ids_by_axis = value_ids_by_axis or {}
        self.record_value_ids = record_value_ids

    def apply(self, states: list) -> dict:
        dealt, ids_by_hand, _ = deal_hand_records_with_value_ids(
            self.template, self.values, self.weights, self.hand_count, self.seed, self.deal_rules,
            self.value_ids_by_axis)
        return {hand["prompt_id"]: Created(hand_record(
                    hand["prompt_id"], 0, hand["variables"],
                    ids_by_hand[hand["prompt_id"]] if self.record_value_ids else None))
                for hand in dealt}
