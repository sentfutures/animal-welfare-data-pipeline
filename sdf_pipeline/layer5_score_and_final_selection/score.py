"""Layer 5 step: score each rewrite.

The judge scores alignment, realism, and spec_conformance (1-10 each) against the constitution and the document's
generating spec. A scoring response that fails to parse is recorded as 5/5/5 "Parse error." and checkpointed: a
re-run would re-bill without new information; the score report flags these for manual review instead.
``ScoreStep`` is a ``DocumentStep`` run by ``sdf_pipeline.shared_step_machinery.step_runner.StepRunner``; the gate and the near-duplicate
removal that follow it are in ``score_gate.py`` and ``near_duplicate_removal.py``.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import constitution_loader, utils
from sdf_pipeline.shared_step_machinery import compose_prompts as cp
from sdf_pipeline.shared_step_machinery.dealt_value_identifiers import value_ids_field_of
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, DocumentStep, Failed, ModelReply, ModelRequest, StepPolicy, StepProgress,
    line_for_document_left_to_retry, reason_without_error_prefix)

PARSE_ERROR_SCORES = {"alignment": 5, "realism": 5, "spec_conformance": 5, "notes": "Parse error."}


class ScoreStep(DocumentStep):
    """Score one rewrite. The document is the rewrite record (``doc_id``, ``description``,
    ``variables``, ``content``). An unparseable answer is accepted as 5/5/5 "Parse error."; only a
    thrown call is left behind (and counts towards the systemic stop)."""

    step_name = "score"
    stage_file_name = "scores.jsonl"
    policy = StepPolicy()

    def __init__(self, config: dict, prompts_dir: Path):
        self.score_model = config["sdf"].get("score_model")
        constitution_dir = utils.resolve_constitution_dir(prompts_dir)
        self.constitution_claude = constitution_loader.load_constitution_claude(constitution_dir)
        self.template_path = utils.sdf_template_path(prompts_dir, 4)
        self.render_variable_lines = cp.render_variable_lines_for_template(
            self.template_path.read_text(encoding="utf-8"), prompts_dir, config)

    def build_request(self, rewrite: dict) -> ModelRequest:
        system, user = cp.split_sections(utils.load_prompt(
            self.template_path,
            constitution_claude=self.constitution_claude,
            document_description=rewrite["description"],
            variable_lines=self.render_variable_lines(
                {"prompt_id": rewrite["doc_id"], "variables": rewrite["variables"]}),
            improved_document=rewrite["content"],
        ))
        return self.model_request(
            rewrite, user, system, self.score_model,
            cache_system=True,  # constitution + rubric are identical across scoring calls
        )

    def interpret(self, rewrite: dict, reply: ModelReply):
        try:
            scores = utils.extract_json_object(reply.text)
        except json.JSONDecodeError:
            scores = PARSE_ERROR_SCORES
        return Accepted({
            "alignment": scores.get("alignment", 0),
            "realism": scores.get("realism", 0),
            "spec_conformance": scores.get("spec_conformance", 0),
            "notes": scores.get("notes", ""),
        })

    def stage_file_record(self, rewrite: dict, update: dict) -> dict:
        return {"doc_id": rewrite["doc_id"], "variables": rewrite["variables"], **value_ids_field_of(rewrite),
                "scores": update}

    def progress_line(self, rewrite: dict, outcome, progress: StepProgress) -> str | None:
        doc_id = rewrite["doc_id"]
        if isinstance(outcome, Failed):
            return line_for_document_left_to_retry(
                doc_id, f"API call failed ({reason_without_error_prefix(outcome.reason)})")
        s = outcome.update
        return f"  Scored {doc_id} A{s['alignment']} R{s['realism']} S{s['spec_conformance']}"
