"""Layer 4 step: the critique, the first of the two revision calls (a problem list against the review checks); alignment-critical, so do not skip it.
A critique is banked in critiques.jsonl before the revise runs; a reply with no ``<review>`` tags is accepted whole. A draft whose rewrite is on disk needs
no critique, and accepting a critique finishes nothing: only an accepted rewrite finishes a draft."""

from sdf_pipeline.layer4_draft_critique_and_revise.draft import DRAFTS
from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import (
    CRITIQUE_MAX_TOKENS, CRITIQUE_TEMPLATE, CRITIQUES, REVIEW_TAG_RE, REWRITES, ReviewPromptRenderer,
    layer_name_in_messages, require_two_call_templates)
from sdf_pipeline.shared_step_machinery.step_record import Run, Step, loaded_once_per_run
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, StepPolicy, StepProgress,
    line_for_document_left_to_retry)
from shared.sdf_run_layouts import CRITIQUE_AND_REVISE


@loaded_once_per_run
def _renderer(run: Run) -> ReviewPromptRenderer:
    require_two_call_templates(run.prompts_directory)
    return ReviewPromptRenderer(run.config, run.prompts_directory, run.prompts_directory / CRITIQUE_TEMPLATE)


@loaded_once_per_run
def _documents_with_a_rewrite(run: Run) -> set:
    return {record["doc_id"] for record in run.records(REWRITES)}


def build_request(documents: list, run: Run) -> ModelRequest:
    (draft,) = documents
    system, user = _renderer(run).render(draft)
    return STEP.model_request(draft, user, system, run, max_tokens=CRITIQUE_MAX_TOKENS, cache_system=True)


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    (draft,) = documents
    if reply.stop_reason != "end_turn":
        outcome = Failed(FailureKind.UNUSABLE_ANSWER, f"critique truncated (stop_reason={reply.stop_reason})")
    else:
        m = REVIEW_TAG_RE.search(reply.text)
        critique = (m.group(1) if m else reply.text).strip()
        outcome = Accepted(critique) if critique else Failed(FailureKind.UNUSABLE_ANSWER, "critique empty")
    return {draft["doc_id"]: outcome}


def records(documents: list, outcomes: dict, run: Run) -> OutcomeRecords:
    (draft,), (outcome,) = documents, outcomes.values()
    if not isinstance(outcome, Accepted):
        return OutcomeRecords()
    return OutcomeRecords(((CRITIQUES.file_name, {"doc_id": draft["doc_id"], "review": outcome.update}),))


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (draft,), (outcome,) = documents, outcomes.values()
    if isinstance(outcome, Failed):  # an accepted critique prints nothing: the revise prints the result
        return line_for_document_left_to_retry(draft["doc_id"], f"no rewrite ({outcome.reason})")
    return None


# The layer's systemic stop (``run_critique_and_revise``) counts the critique and revise calls together, so
# neither step stops the run on its own.
STEP = Step(
    name="critique", folder_key=CRITIQUE_AND_REVISE, model_setting_keys=("critique_model", "revise_model"),
    required_prompt_files=lambda config, prompts_directory: [CRITIQUE_TEMPLATE],
    reads=lambda run: run.records(DRAFTS), seeds_from=lambda run: DRAFTS,
    finished_elsewhere=lambda draft, run: draft["doc_id"] in _documents_with_a_rewrite(run),
    build_request=build_request, read_reply=read_reply, records=records, writes=(CRITIQUES,),
    policy=StepPolicy(failure_kinds_counting_towards_the_systemic_stop=frozenset()),
    name_in_messages=layer_name_in_messages, progress_line=progress_line)
