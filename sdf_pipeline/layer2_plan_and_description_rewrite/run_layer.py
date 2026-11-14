"""Layer 2 (plan and description rewrite): plan one document per accepted hand, then repair each description.

The layer is two steps, one module each in ``sdf_pipeline/layer2_plan_and_description_rewrite/``: ``PlanStep`` (``plan.py``, which also composes the
planning prompts offline and writes prompts.jsonl) and ``DescriptionRewriteStep`` (``description_rewrite.py``, one
call per description checking plausibility and conformance to the dealt values). This module holds the layer's
driver, ``run``: it plans the hands the screen accepted and rewrites the descriptions, returning the plans.

Files in the layer's folder (``layer2_plan_and_description_rewrite``): prompts.jsonl (the composed planning prompts),
plans.jsonl and _checkpoint.json (the plan step), description_rewrites.jsonl, description_rewrite_rejections.jsonl and
_description_rewrite_checkpoint.json (the description rewrite). See the two step modules for their semantics.
"""

from pathlib import Path

from sdf_pipeline.layer2_plan_and_description_rewrite import description_rewrite, plan
from sdf_pipeline.layer2_plan_and_description_rewrite.description_rewrite import DescriptionRewriteStep  # noqa: F401
from sdf_pipeline.layer2_plan_and_description_rewrite.plan import PlanStep  # noqa: F401


def run(config: dict, prompts_dir: Path, output_dir: Path, hands: list[dict] | None = None) -> list[dict]:
    """Plan every hand (``hands`` as ``plan.run`` takes them: the accepted hands, or None with the screen off), then
    rewrite the descriptions; returns the plans with their rewritten descriptions."""
    plans = plan.run(config, prompts_dir, output_dir, hands)
    return description_rewrite.run(config, prompts_dir, output_dir, plans)
