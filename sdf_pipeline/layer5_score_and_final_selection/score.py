"""Layer 5 step: the judge scores alignment, realism and spec_conformance (1-10 each). A response that fails to parse is recorded as 5/5/5 "Parse error." and kept
(a re-run would re-bill without new information); the score report flags these for review."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import constitution_loader, utils
from shared.sdf_run_layouts import SCORE
from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import REWRITES
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import value_ids_field_of
from sdf_pipeline.shared_step_machinery.step_record import (
    Run, StageFile, Step, loaded_once_per_run, one_record_per_accepted_document)
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, ModelReply, ModelRequest, StepProgress, line_for_document_left_to_retry,
    reason_without_error_prefix)

PARSE_ERROR_SCORES = {"alignment": 5, "realism": 5, "spec_conformance": 5, "notes": "Parse error."}


SCORES = StageFile(SCORE, "scores.jsonl", "doc_id")


@dataclass(frozen=True)
class ScoreInputs:
    """What every score request is built from, loaded once per run."""
    constitution_claude: str
    template_path: Path
    render_variable_lines: Callable


@loaded_once_per_run
def _inputs(run: Run) -> ScoreInputs:
    template_path = utils.sdf_template_path(run.prompts_directory, 4)
    return ScoreInputs(
        constitution_claude=constitution_loader.load_constitution_claude(
            utils.resolve_constitution_dir(run.prompts_directory)),
        template_path=template_path,
        render_variable_lines=cp.render_variable_lines_for_template(
            template_path.read_text(encoding="utf-8"), run.prompts_directory, run.config))


def build_request(documents: list, run: Run) -> ModelRequest:
    (rewrite,) = documents
    inputs = _inputs(run)
    system, user = cp.split_sections(utils.load_prompt(
        inputs.template_path,
        constitution_claude=inputs.constitution_claude,
        document_description=rewrite["description"],
        variable_lines=inputs.render_variable_lines({"prompt_id": rewrite["doc_id"], "variables": rewrite["variables"]}),
        improved_document=rewrite["content"],
    ))
    return STEP.model_request(
        rewrite, user, system, run,
        cache_system=True,  # constitution + rubric are identical across scoring calls
    )


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """An unparseable answer is accepted as 5/5/5 "Parse error."; only a thrown call is left behind (and counts
    towards the systemic stop)."""
    (rewrite,) = documents
    try:
        scores = utils.extract_json_object(reply.text)
    except json.JSONDecodeError:
        scores = PARSE_ERROR_SCORES
    return {rewrite["doc_id"]: Accepted({
        "alignment": scores.get("alignment", 0),
        "realism": scores.get("realism", 0),
        "spec_conformance": scores.get("spec_conformance", 0),
        "notes": scores.get("notes", ""),
    })}


def record_of(rewrite: dict, update: dict, run: Run) -> dict:
    return {"doc_id": rewrite["doc_id"], "variables": rewrite["variables"], **value_ids_field_of(rewrite),
            "scores": update}


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (rewrite,), (outcome,) = documents, outcomes.values()
    doc_id = rewrite["doc_id"]
    if isinstance(outcome, Failed):
        return line_for_document_left_to_retry(
            doc_id, f"API call failed ({reason_without_error_prefix(outcome.reason)})")
    s = outcome.update
    return f"  Scored {doc_id} A{s['alignment']} R{s['realism']} S{s['spec_conformance']}"


STEP = Step(
    name="score", folder_key=SCORE, model_setting_keys=("score_model",),
    required_prompt_files=lambda config, prompts_directory: [utils.sdf_template_path(prompts_directory, 4).name],
    reads=lambda run: run.records(REWRITES), seeds_from=lambda run: REWRITES,
    build_request=build_request, read_reply=read_reply,
    records=one_record_per_accepted_document(SCORES, record_of), writes=(SCORES,), progress_line=progress_line)
