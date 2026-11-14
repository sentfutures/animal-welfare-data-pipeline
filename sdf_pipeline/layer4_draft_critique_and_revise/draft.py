"""Layer 4 step: draft one document per plan, from its DOCUMENT DESCRIPTION spec.

The layer2.txt template carries the constitution and principles in its SYSTEM
section (rendered from the run's constitution snapshot) and the per-document
spec in its USER section. The draft must come back inside <document> tags;
anything else — missing tags, truncation — is not checkpointed, so --resume
retries exactly the failed calls (temperature 1.0 makes retries productive).

The layer is a ``DocumentStep`` (``DraftStep``) run by ``sdf_pipeline.shared_step_machinery.step_runner.StepRunner``;
``run`` is the layer's entry point, called by ``sdf_pipeline/run.py`` and ``experiments/experiment.py``:
it builds the step and the runner and returns the draft records in input order.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import constitution_loader, deal_constraints, utils
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import value_ids_field_of
from sdf_pipeline.shared_step_machinery.step_runner import StepRunner
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, StepPolicy, StepProgress,
    line_for_document_left_to_retry)

DOC_TAG_RE = re.compile(r"<document>(.*?)</document>", re.DOTALL)

_MAX_TOKENS = 6000  # real documents are often long; the cap bounds api-backend cost


class DraftStep(DocumentStep):
    """Draft one document from a plan record. The document is the plan dict (``prompt_id``,
    ``description``, ``variables``, ``considerations_spec``)."""

    step_name = "draft"
    stage_file_name = "drafts.jsonl"
    # Only a thrown call counts towards the systemic stop; an unusable answer is left behind.
    policy = StepPolicy()
    document_id_field = "prompt_id"   # the documents are plans; the drafts file's ids are doc_id

    def __init__(self, config: dict, prompts_dir: Path):
        self.draft_model = config["sdf"].get("draft_model")
        constitution_dir = utils.resolve_constitution_dir(prompts_dir)
        self.constitution_claude = constitution_loader.load_constitution_claude(constitution_dir)
        self.principles = constitution_loader.format_principles(
            constitution_loader.load_principles(constitution_dir)
        )
        self.preamble = (prompts_dir / "preamble.txt").read_text(encoding="utf-8").strip()

        # A document with no considerations leaves no trace in its prompt: the template
        # line that carries {considerations_spec} is dropped when the block is empty.
        self.template_text = utils.sdf_template_path(prompts_dir, 2).read_text(encoding="utf-8")
        self.render_variable_lines = cp.render_variable_lines_for_template(
            self.template_text, prompts_dir, config)
        self.template_without_considerations = deal_constraints.drop_template_lines(
            self.template_text, ["considerations_spec"])

    def applies_to(self, plan: dict) -> bool:
        return bool(plan.get("description"))

    def build_request(self, plan: dict) -> ModelRequest:
        considerations_spec = plan.get("considerations_spec", "")
        system, user = cp.split_sections((
            self.template_text if considerations_spec else self.template_without_considerations
        ).format(
            preamble=self.preamble,
            constitution_claude=self.constitution_claude,
            constitution_principles=self.principles,
            document_description=plan["description"],
            considerations_spec=considerations_spec,
            variable_lines=self.render_variable_lines({"prompt_id": plan["prompt_id"], "variables": plan["variables"]}),
            # Downstream-only matrix axis (sampled in layer 1, recorded in
            # variables). Default to the neutral value for plans composed
            # before this axis existed, so --resume on an old run never crashes.
            # Runs whose prompt snapshot still renders it (it is no longer dealt)
            # keep working on the same fallback.
            reasoning_featured=plan.get("variables", {}).get(
                "reasoning_featured", "any relevant principles from the constitution"
            ),
        ))
        return self.model_request(
            plan, user, system, self.draft_model,
            max_tokens=_MAX_TOKENS,
            cache_system=True,  # constitution-laden system prompt is identical across drafts
        )

    def interpret(self, plan: dict, reply: ModelReply):
        if reply.stop_reason != "end_turn":
            return Failed(FailureKind.UNUSABLE_ANSWER, f"truncated draft (stop_reason={reply.stop_reason})")
        m = DOC_TAG_RE.search(reply.text)
        content = m.group(1).strip() if m else ""
        if not content:
            return Failed(FailureKind.UNUSABLE_ANSWER, "no <document> tags")
        return Accepted(content)

    def stage_file_record(self, plan: dict, update: str) -> dict:
        return {
            "doc_id": plan["prompt_id"],
            "variables": plan["variables"],
            **value_ids_field_of(plan),
            "description": plan["description"],
            "considerations_spec": plan.get("considerations_spec", ""),
            "content": update,
        }

    def progress_line(self, plan: dict, outcome, progress: StepProgress) -> str | None:
        pid = plan["prompt_id"]
        if isinstance(outcome, Accepted):
            return f"  Drafted {pid} ({len(outcome.update)} chars)"
        if outcome.kind is FailureKind.THROWN_CALL:
            return line_for_document_left_to_retry(pid, f"API call failed ({outcome.reason})")
        return line_for_document_left_to_retry(pid, outcome.reason)


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict],
        checkpoint_file_name: str = "_checkpoint.json") -> list[dict]:
    """Draft every plan. ``checkpoint_file_name`` is the checkpoint's name in ``output_dir``: the layer driver passes the
    layout's (``_draft_checkpoint.json`` where the draft shares its folder with the critique and revise)."""
    step = DraftStep(config, prompts_dir)
    runner = StepRunner(step, output_dir, workers=config.get("workers", 1),
                        checkpoint=utils.Checkpoint(output_dir / checkpoint_file_name))
    return runner.run(plans).records_in_input_order
