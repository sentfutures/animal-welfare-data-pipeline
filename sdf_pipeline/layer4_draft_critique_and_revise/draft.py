"""Layer 4 step: draft one document per plan, from its DOCUMENT DESCRIPTION spec; the draft must come back inside <document> tags.
Anything else (missing tags, truncation) is not recorded, so ``--resume`` retries it."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import constitution_loader, deal_constraints, utils
from shared.sdf_run_layouts import DRAFT
from sdf_pipeline.layer2_description_write_and_description_rewrite.description_write import PLANS
from sdf_pipeline.layer3_consideration_scoring_and_selection.consideration_selection import plans_with_selections
from sdf_pipeline import profiles
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import value_ids_field_of
from sdf_pipeline.shared_step_machinery.step_record import (
    Run, StageFile, Step, loaded_once_per_run, one_record_per_accepted_document)
from sdf_pipeline.shared_step_machinery.step_runner import run_step
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, StepProgress, line_for_document_left_to_retry)

DOC_TAG_RE = re.compile(r"<document>(.*?)</document>", re.DOTALL)

_MAX_TOKENS = 6000  # real documents are often long; the cap bounds api-backend cost


DRAFTS = StageFile(DRAFT, "drafts.jsonl", "doc_id")


@dataclass(frozen=True)
class DraftInputs:
    """What every draft request is built from, loaded once per run."""
    constitution_claude: str
    principles: str
    preamble: str
    template_text: str
    template_without_considerations: str   # a document with no considerations leaves no trace in its prompt
    render_variable_lines: Callable


@loaded_once_per_run
def _inputs(run: Run) -> DraftInputs:
    constitution_dir = utils.resolve_constitution_dir(run.prompts_directory)
    template_text = utils.sdf_template_path(run.prompts_directory, 2).read_text(encoding="utf-8")
    return DraftInputs(
        constitution_claude=constitution_loader.load_constitution_claude(constitution_dir),
        principles=constitution_loader.format_principles(constitution_loader.load_principles(
            constitution_dir, filename=profiles.current().PRINCIPLES_FILENAME)),
        preamble=(run.prompts_directory / "preamble.txt").read_text(encoding="utf-8").strip(),
        template_text=template_text,
        template_without_considerations=deal_constraints.drop_template_lines(template_text, ["considerations_spec"]),
        render_variable_lines=cp.render_variable_lines_for_template(template_text, run.prompts_directory, run.config))


def build_request(documents: list, run: Run) -> ModelRequest:
    """The draft request for the plan record (``prompt_id``, ``description``, ``variables``, ``considerations_spec``)."""
    (plan,) = documents
    inputs = _inputs(run)
    considerations_spec = plan.get("considerations_spec", "")
    system, user = cp.split_sections((
        inputs.template_text if considerations_spec else inputs.template_without_considerations
    ).format(
        preamble=inputs.preamble,
        constitution_claude=inputs.constitution_claude,
        constitution_principles=inputs.principles,
        document_description=plan["description"],
        considerations_spec=considerations_spec,
        variable_lines=inputs.render_variable_lines({"prompt_id": plan["prompt_id"], "variables": plan["variables"]}),
        # Downstream-only matrix axis recorded in variables; plans without it get the neutral
        # value, so --resume of a run that lacks it never crashes.
        reasoning_featured=plan.get("variables", {}).get(
            "reasoning_featured", "any relevant principles from the constitution"
        ),
    ))
    return STEP.model_request(
        plan, user, system, run,
        max_tokens=_MAX_TOKENS,
        cache_system=True,  # constitution-laden system prompt is identical across drafts
    )


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    (plan,) = documents
    if reply.stop_reason != "end_turn":
        outcome = Failed(FailureKind.UNUSABLE_ANSWER, f"truncated draft (stop_reason={reply.stop_reason})")
    else:
        m = DOC_TAG_RE.search(reply.text)
        content = m.group(1).strip() if m else ""
        outcome = Accepted(content) if content else Failed(FailureKind.UNUSABLE_ANSWER, "no <document> tags")
    return {plan["prompt_id"]: outcome}


def record_of(plan: dict, content: str, run: Run) -> dict:
    return {
        "doc_id": plan["prompt_id"],
        "variables": plan["variables"],
        **value_ids_field_of(plan),
        "description": plan["description"],
        "considerations_spec": plan.get("considerations_spec", ""),
        "content": content,
    }


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (plan,), (outcome,) = documents, outcomes.values()
    pid = plan["prompt_id"]
    if isinstance(outcome, Accepted):
        return f"  Drafted {pid} ({len(outcome.update)} chars)"
    if outcome.kind is FailureKind.THROWN_CALL:
        return line_for_document_left_to_retry(pid, f"API call failed ({outcome.reason})")
    return line_for_document_left_to_retry(pid, outcome.reason)


# Only a thrown call counts towards the systemic stop; an unusable answer is left behind.
STEP = Step(
    name="draft", folder_key=DRAFT, model_setting_keys=("draft_model",),
    required_prompt_files=lambda config, prompts_directory: [utils.sdf_template_path(prompts_directory, 2).name, "preamble.txt"],
    reads=plans_with_selections, seeds_from=lambda run: PLANS,
    applies=lambda plan, run: bool(plan.get("description")),
    build_request=build_request, read_reply=read_reply, records=one_record_per_accepted_document(DRAFTS, record_of), writes=(DRAFTS,),
    progress_line=progress_line)


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Draft every plan and return the draft records in input order."""
    return run_step(STEP, Run(config, prompts_dir, {DRAFT: output_dir}), plans).records_in_input_order
