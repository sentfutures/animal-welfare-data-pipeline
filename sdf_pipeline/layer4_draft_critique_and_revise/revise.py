"""Layer 4 step: the revise, the second revision call (the rewritten text, inside <improved_document> tags, with the voice and length directives).
It applies to drafts that have a critique in critiques.jsonl, so a failed revise is retried alone on resume; missing tags or truncation are not recorded.
Description, variables and considerations come from the draft record."""

from sdf_pipeline.layer4_draft_critique_and_revise.draft import DRAFTS
from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import (
    CRITIQUES, MAX_TOKENS, REVISE_TEMPLATE, REWRITES, ReviewPromptRenderer, RewrittenText, interpret_rewrite_reply,
    layer_name_in_messages, load_critiques_by_document_id, require_two_call_templates, rewrite_progress_line,
    rewrite_record)
from sdf_pipeline.shared_step_machinery.step_record import (
    Run, Step, loaded_once_per_run, one_record_per_accepted_document)
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Failed, ModelReply, ModelRequest, StepPolicy, StepProgress
from shared.sdf_run_layouts import CRITIQUE_AND_REVISE


@loaded_once_per_run
def _renderer(run: Run) -> ReviewPromptRenderer:
    require_two_call_templates(run.prompts_directory)
    return ReviewPromptRenderer(run.config, run.prompts_directory, run.prompts_directory / REVISE_TEMPLATE)


@loaded_once_per_run
def critique_by_document_id(run: Run) -> dict:
    """{doc_id: critique} of the critiques banked in the run's critique and revise folder."""
    return load_critiques_by_document_id(run.stage_directories[CRITIQUE_AND_REVISE])


def build_request(documents: list, run: Run) -> ModelRequest:
    (draft,) = documents
    style, length = _renderer(run).directives(draft)
    system, user = _renderer(run).render(
        draft, review=critique_by_document_id(run)[draft["doc_id"]], style_directives=style, length_directive=length)
    return STEP.model_request(draft, user, system, run, max_tokens=MAX_TOKENS, cache_system=True)


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    (draft,) = documents
    interpreted = interpret_rewrite_reply(reply)
    if isinstance(interpreted, Failed):
        return {draft["doc_id"]: interpreted}
    content, _ = interpreted
    return {draft["doc_id"]: Accepted(RewrittenText(content, critique_by_document_id(run)[draft["doc_id"]]))}


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (draft,), (outcome,) = documents, outcomes.values()
    return rewrite_progress_line(draft, outcome)


STEP = Step(
    name="revise", folder_key=CRITIQUE_AND_REVISE, model_setting_keys=("revise_model",),
    required_prompt_files=lambda config, prompts_directory: [REVISE_TEMPLATE],
    reads=lambda run: run.records(DRAFTS), seeds_from=lambda run: CRITIQUES,
    applies=lambda draft, run: draft["doc_id"] in critique_by_document_id(run),
    build_request=build_request, read_reply=read_reply,
    records=one_record_per_accepted_document(REWRITES, lambda draft, update, run: rewrite_record(draft, update)),
    writes=(REWRITES,),
    policy=StepPolicy(failure_kinds_counting_towards_the_systemic_stop=frozenset()),
    name_in_messages=layer_name_in_messages, progress_line=progress_line)
