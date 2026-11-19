"""Layer 4 step: the critique, the first of the two revision calls (a problem list against the review checks); alignment-critical, so do not skip it.
A critique is banked in critiques.jsonl before the revise runs; a reply with no ``<review>`` tags is accepted whole. A draft whose rewrite is on disk needs
no critique, and accepting a critique finishes nothing: only an accepted rewrite finishes a draft."""

from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import (
    CRITIQUE_MAX_TOKENS, CRITIQUE_TEMPLATE, CRITIQUES, REVIEW_TAG_RE, ReviewPromptRenderer, is_rewritten,
    layer_name_in_messages)
from sdf_pipeline.shared_step_machinery.step_record import Run, Step, loaded_once_per_run, one_record_per_accepted_document
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, StepPolicy, StepProgress, WhereAnErrorIsAThrownCall,
    line_for_document_left_to_retry)
from shared.sdf_run_layouts import CRITIQUE_AND_REVISE


@loaded_once_per_run
def _renderer(run: Run) -> ReviewPromptRenderer:
    return ReviewPromptRenderer(run.config, run.prompts_directory, run.prompts_directory / CRITIQUE_TEMPLATE)


def build_request(documents: list, run: Run) -> ModelRequest:
    (draft,) = documents
    system, user = _renderer(run).render(draft)
    return STEP.model_request(draft, user, system, run, max_tokens=CRITIQUE_MAX_TOKENS, cache_system=True)


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """A reply with no ``<review>`` tags is accepted whole: re-billing a usable critique to chase its wrapper costs more."""
    (draft,) = documents
    if reply.stop_reason != "end_turn":
        outcome = Failed(FailureKind.UNUSABLE_ANSWER, f"critique truncated (stop_reason={reply.stop_reason})")
    else:
        m = REVIEW_TAG_RE.search(reply.text)
        critique = (m.group(1) if m else reply.text).strip()
        outcome = Accepted(critique) if critique else Failed(FailureKind.UNUSABLE_ANSWER, "critique empty")
    return {draft["doc_id"]: outcome}


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (draft,), (outcome,) = documents, outcomes.values()
    if isinstance(outcome, Failed):  # an accepted critique prints nothing: the revise prints the result
        return line_for_document_left_to_retry(draft["doc_id"], f"no rewrite ({outcome.reason})")
    return None


# The layer's systemic stop (``run_critique_and_revise``) counts the critique and revise calls together, so
# neither step stops the run on its own.
STEP = Step(
    name="critique", folder_key=CRITIQUE_AND_REVISE, model_setting_keys=("critique_model", "revise_model"),
    finished_elsewhere=lambda draft, run: is_rewritten(draft["doc_id"], run),
    build_request=build_request, read_reply=read_reply,
    records=one_record_per_accepted_document(CRITIQUES, lambda draft, critique, run: {"doc_id": draft["doc_id"], "review": critique}),
    writes=(CRITIQUES,),
    policy=StepPolicy(
        failure_kinds_counting_towards_the_systemic_stop=frozenset(),
        errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND),
    name_in_messages=layer_name_in_messages, progress_line=progress_line)
