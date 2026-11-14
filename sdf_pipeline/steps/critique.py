"""Layer 4 step: the critique, the first of the two revision calls (a problem list against the review checks).

The alignment-critical pass starts here: the template holds the constitution, principles, the review checks and the
failure-modes checklist in its SYSTEM section (constitution/failure_modes.md); the USER section delivers the
generating spec, the dealt variable lines, the assigned considerations and the draft. A critique is banked in
critiques.jsonl before the revise runs, so a failed revise does not re-bill it.
"""

from pathlib import Path

from sdf_pipeline.renamed_model_settings import model_setting

from sdf_pipeline.layer4_draft_critique_and_revise_shared import (
    CRITIQUE_MAX_TOKENS, CRITIQUE_TEMPLATE, REVIEW_TAG_RE, ReviewPromptRenderer,
    SystemicStopMessageNamingTheLayer, require_two_call_templates)
from sdf_pipeline.update_steps import (
    Accepted, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, StepPolicy, StepProgress,
    line_for_document_left_to_retry)


class CritiqueStep(DocumentStep):
    """The first of the two calls: a problem list against the review checks. The document is the
    draft record. A reply with no ``<review>`` tags is accepted whole (the whole reply is a problem
    list in every observed malformed case, and paying again to chase the wrapper costs more)."""

    step_name = "critique"
    stage_file_name = "critiques.jsonl"
    # ``run`` applies the layer's systemic stop over the critique and revise calls together, as the
    # layer always has, so neither step stops the run on its own.
    policy = StepPolicy(failure_kinds_counting_towards_the_systemic_stop=frozenset())
    systemic_stop_message = SystemicStopMessageNamingTheLayer()

    def __init__(self, config: dict, prompts_dir: Path):
        require_two_call_templates(prompts_dir)
        sdf = config["sdf"]
        self.critique_model = sdf.get("critique_model") or model_setting(sdf, "revise_model")
        self.renderer = ReviewPromptRenderer(config, prompts_dir, prompts_dir / CRITIQUE_TEMPLATE)

    def build_request(self, draft: dict) -> ModelRequest:
        system, user = self.renderer.render(draft)
        return self.model_request(
            draft, user, system, self.critique_model,
            max_tokens=CRITIQUE_MAX_TOKENS,
            cache_system=True,
        )

    def interpret(self, draft: dict, reply: ModelReply):
        if reply.stop_reason != "end_turn":
            return Failed(FailureKind.UNUSABLE_ANSWER, f"critique truncated (stop_reason={reply.stop_reason})")
        m = REVIEW_TAG_RE.search(reply.text)
        critique = (m.group(1) if m else reply.text).strip()
        if not critique:
            return Failed(FailureKind.UNUSABLE_ANSWER, "critique empty")
        return Accepted(critique)

    def stage_file_record(self, draft: dict, update: str) -> dict:
        return {"doc_id": draft["doc_id"], "review": update}

    def progress_line(self, draft: dict, outcome, progress: StepProgress) -> str | None:
        if isinstance(outcome, Failed):  # an accepted critique prints nothing: the revise prints the result
            return line_for_document_left_to_retry(draft["doc_id"], f"no rewrite ({outcome.reason})")
        return None
