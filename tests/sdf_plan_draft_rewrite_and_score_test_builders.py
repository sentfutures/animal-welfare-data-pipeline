"""Builders shared by the SDF plan, draft, rewrite and score tests: a model reply for the plan stage, a plan,
draft and rewrite record with a complete dealt hand, a score reply, and config variants for the split and
single-call rewrite."""

import json
from pathlib import Path

from sdf_pipeline import compose_prompts as cp

REPO_ROOT = Path(__file__).resolve().parent.parent


def plan_reply(description="A fully specified document plan.", notes="notes"):
    return (
        f"<document_planning>{notes}</document_planning>\n"
        f"<document_description>\n{description}\n</document_description>"
    )


def full_dealt_variables():
    """One card from every variable of the real variables.txt: the layer-2-to-4 templates
    render the dealt variables, so a plan needs a complete hand."""
    values, _, _ = cp.load_variables(REPO_ROOT / "prompts" / "sdf" / "variables.txt")
    hand = {name: cards[0] for name, cards in values.items()}
    hand.update({
        "document_type": "a news article",
        "culture": "France, written in French, with French idioms and references",
        "tone": "neutral or journalistic",
    })
    return hand


def make_plan(pid="matrix_000000", description="Spec text.", incoherent=False):
    return {
        "prompt_id": pid,
        "variables": {
            **full_dealt_variables(),
            "reasoning_featured": "individualism: the notion that welfare is a property of individuals, not systems",
        },
        "plan": "…",
        "description": None if incoherent else description,
        "incoherent": incoherent,
    }


def make_draft(pid="matrix_000000", content="Un document."):
    p = make_plan(pid)
    return {"doc_id": pid, "variables": p["variables"], "description": "Spec text.", "content": content}


def make_rewrite(pid="matrix_000000", content="Un document réécrit."):
    d = make_draft(pid, content)
    return {**d, "review": "review notes"}


SCORE_OK = json.dumps({"alignment": 9, "realism": 8, "spec_conformance": 9, "notes": "fine"})


def split_config(tiny_config, **revision):
    """tiny_config with the layer-3 split (and any other revision knob) on."""
    return {**tiny_config,
            "sdf": {**tiny_config["sdf"], "revision": {"split": True, **revision}}}


def single_call_config(tiny_config, **revision):
    """tiny_config pinned to the single-call layer3.txt, which still ships."""
    return {**tiny_config,
            "sdf": {**tiny_config["sdf"], "revision": {"split": False, **revision}}}
