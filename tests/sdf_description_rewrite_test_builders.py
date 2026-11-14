"""Builders shared by the SDF description rewrite tests: a rewrite reply and dealt plans for it."""

import json

from sdf_pipeline.shared_step_machinery import compose_prompts as cp


def answer(rewrite_needed=False, description="", checks=("plausibility_organisation",), conflict=False,
           conflict_reason="", **overrides):
    changes = [{"check": c, "reason": f"Fixed under {c}."} for c in checks] if rewrite_needed else []
    return json.dumps({
        "reasoning": "Read it against the dealt values.", "rewrite_needed": rewrite_needed,
        "rewritten_description": description, "changes": changes,
        "dealt_value_conflict": conflict, "conflict_reason": conflict_reason, **overrides})


def make_plans(prompts_sdf, count=6):
    template = (prompts_sdf / "layer1.txt").read_text(encoding="utf-8")
    values, weights, value_ids = cp.load_variables(prompts_sdf / "variables.txt")
    rules = cp.load_deal_rules(prompts_sdf / "deal_constraints.txt", value_ids)
    hands = cp.hand_records(cp.deal_hands(template, values, weights, count, 0, rules))
    return [
        {"prompt_id": h["prompt_id"], "variables": h["variables"], "plan": "…",
         "description": f"Original description {i}.", "incoherent": False}
        for i, h in enumerate(hands)
    ]
