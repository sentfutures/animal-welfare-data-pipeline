"""Builders shared by the SDF consideration selection tests: a small considerations file, plans with a
complete hand, scoring replies and the selection file reader."""

import json
import re
from pathlib import Path

import pytest

from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.layer3_consideration_scoring_and_selection import run_layer as select
from sdf_pipeline.shared_step_machinery.considerations_catalog import load_considerations
from shared import utils


SMALL_CONSIDERATIONS = """\
{moral_considerations}
    0.2 :: Alpha moral: the first moral gloss
    0.9 :: Beta moral: the second moral gloss

{epistemic_considerations}
    0.1 :: Gamma epistemic: the epistemic gloss

{epistemic_considerations_digital_minds_only}
    0.5 :: Delta digital: the digital-minds gloss

{digital_mind_categories}
    large_language_models

{skipped_resolutions}
    no_stake
"""


DIGITAL_CATEGORY = "LLMs whose moral state is uncertain"


@pytest.fixture
def small_prompts(prompts_sdf, tmp_path):
    """A prompts dir with the real selector template and variables but a small considerations file."""
    directory = tmp_path / "prompts_small"
    directory.mkdir()
    for name in ("variables.txt", select.TEMPLATE_FILENAME):
        (directory / name).write_text((prompts_sdf / name).read_text(encoding="utf-8"), encoding="utf-8")
    (directory / "considerations.txt").write_text(SMALL_CONSIDERATIONS, encoding="utf-8")
    return directory


def no_stake_resolution(prompts_dir):
    return sorted(load_considerations(prompts_dir).skipped_resolutions)[0]


_REAL_VARIABLES = cp.load_variables(Path(__file__).resolve().parent.parent / "prompts" / "sdf" / "variables.txt")[0]


COMPLETE_HAND = {name: cards[0] for name, cards in _REAL_VARIABLES.items()}   # the later stages render every dealt variable


def make_plans(count=10, digital=(), no_stake=(), incoherent=(), resolution=None):
    plans = []
    for i in range(count):
        variables = {
            **COMPLETE_HAND,
            "document_type": "a news article",
            "centrality": "the central subject of the document",
            "sentient_category": DIGITAL_CATEGORY if i in digital else "farmed fishes",
            "resolution": resolution if i in no_stake else "a creative solution",
        }
        plans.append({
            "prompt_id": f"matrix_{i:06d}", "variables": variables, "plan": "…",
            "description": None if i in incoherent else f"Spec {i}.",
            "incoherent": i in incoherent,
        })
    return plans


def offered_in(user_message):
    """{'moral': [labels], 'epistemic': [labels]} in the order the scoring prompt showed them."""
    shown = {}
    for name in ("moral", "epistemic"):
        body = re.search(rf"<{name}_list>\n(.*?)\n</{name}_list>", user_message, re.DOTALL).group(1)
        shown[name] = [line[2:].split(": ")[0] for line in body.splitlines() if line.startswith("- ")]
    return shown


def answer_with(user_message, score_of):
    """A well-formed answer scoring every shown label with ``score_of(label)``."""
    shown = offered_in(user_message)
    return json.dumps({
        name: [{"label": label, "score": score_of(label),
                "where_it_arises": f"Where {label} arises." if score_of(label) >= 3 else ""}
               for label in labels]
        for name, labels in shown.items() if labels})


def score_everything_five(user_message, **kw):
    return answer_with(user_message, lambda label: 5)


def rows_of(stage_dir, filename=select.SELECTIONS_FILENAME):
    return utils.load_jsonl(stage_dir / filename)
