"""Layer 2: the driver ``run`` writes the descriptions of the hands the screen accepted (``description_write.py``, which writes prompts.jsonl and plans.jsonl)
and rewrites the descriptions (``description_rewrite.py``), returning the plans."""

from pathlib import Path

from sdf_pipeline.layer2_description_write_and_description_rewrite import description_rewrite, description_write


def run(config: dict, prompts_dir: Path, output_dir: Path, hands: list[dict] | None = None) -> list[dict]:
    """Write the description of every hand (``hands`` as ``description_write.run`` takes them: the accepted hands, or None with the screen off), then
    rewrite the descriptions; returns the plans with their rewritten descriptions."""
    plans = description_write.run(config, prompts_dir, output_dir, hands)
    return description_rewrite.run(config, prompts_dir, output_dir, plans)
