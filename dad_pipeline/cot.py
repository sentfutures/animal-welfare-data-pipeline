"""optional chain-of-thought traces for DAD responses (dad.cot.enabled).

with the flag on, 2b writes a reasoning trace inside <reasoning>...</reasoning>
tags before the response, step 3 rewrites the tagged pair (the trace rides into
the rewrite prompt joined onto the draft), and the final record's assistant
message carries the trace under `reasoning` beside `content`. the instruction
text lives in per-profile prompt files (step2_respond_cot.txt /
step3_rewrite_cot.txt), injected into the stage template's {cot_instructions}
slot; with the flag off the slot renders empty and every prompt is unchanged.
"""

from pathlib import Path

from shared import utils

OPEN_TAG = "<reasoning>"
CLOSE_TAG = "</reasoning>"

# instruction file injected into each stage template's {cot_instructions} slot
INSTRUCTION_FILES = {
    "step2_respond.txt": "step2_respond_cot.txt",
    "step3_rewrite.txt": "step3_rewrite_cot.txt",
}


def enabled(config: dict) -> bool:
    """whether this run writes reasoning traces (dad.cot.enabled, default off)."""
    return bool(((config.get("dad") or {}).get("cot") or {}).get("enabled", False))


def instructions(prompts_dir: Path, template_name: str, on: bool) -> str:
    """the {cot_instructions} value for one stage template: the sibling
    *_cot.txt file when the run writes traces, else empty (the slot strips
    away and the rendered prompt is byte-identical to the pre-cot one)."""
    if not on:
        return ""
    return utils.load_prompt(prompts_dir / INSTRUCTION_FILES[template_name])


def split_reasoning(text: str) -> tuple[str, str] | None:
    """(reasoning, response) from a tagged reply, or None when a tag or either
    part is missing — callers treat None like truncation (skip without
    checkpointing, so --resume retries the call)."""
    start = text.find(OPEN_TAG)
    end = text.find(CLOSE_TAG)
    if start == -1 or end == -1 or end < start:
        return None
    reasoning = text[start + len(OPEN_TAG):end].strip()
    response = text[end + len(CLOSE_TAG):].strip()
    if not reasoning or not response:
        return None
    return reasoning, response


def join_reasoning(reasoning: str, response: str) -> str:
    """reassemble the tagged shape 2b emitted — the step-3 draft slot carries
    the pair in the same shape the rewrite is asked to return."""
    return f"{OPEN_TAG}\n{reasoning}\n{CLOSE_TAG}\n\n{response}"
