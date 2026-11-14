"""Layer 4 step: the critique and the revise in one call, selected by ``sdf.revision.split: false`` (template ``layer3.txt``).
The text before the ``<improved_document>`` tags is the review."""

from sdf_pipeline.layer4_draft_critique_and_revise.draft import DRAFTS
from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import (
    MAX_TOKENS, REWRITES, ReviewPromptRenderer, RewrittenText, interpret_rewrite_reply, layer_name_in_messages,
    revision_is_split, rewrite_progress_line, rewrite_record)
from sdf_pipeline.shared_step_machinery.step_record import (
    Run, Step, loaded_once_per_run, one_record_per_accepted_document)
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, Failed, ModelReply, ModelRequest, StepProgress
from shared import utils
from shared.sdf_run_layouts import CRITIQUE_AND_REVISE


@loaded_once_per_run
def _renderer(run: Run) -> ReviewPromptRenderer:
    return ReviewPromptRenderer(run.config, run.prompts_directory, utils.sdf_template_path(run.prompts_directory, 3))


def build_request(documents: list, run: Run) -> ModelRequest:
    (draft,) = documents
    style, length = _renderer(run).directives(draft)
    system, user = _renderer(run).render(draft, style_directives=style, length_directive=length)
    return STEP.model_request(
        draft, user, system, run, max_tokens=MAX_TOKENS,
        cache_system=True,  # constitution + review checks identical across rewrites
    )


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    (draft,) = documents
    interpreted = interpret_rewrite_reply(reply)
    if isinstance(interpreted, Failed):
        return {draft["doc_id"]: interpreted}
    content, review = interpreted
    return {draft["doc_id"]: Accepted(RewrittenText(content, review))}


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (draft,), (outcome,) = documents, outcomes.values()
    return rewrite_progress_line(draft, outcome)


STEP = Step(
    name="critique_and_revise_in_one_call", folder_key=CRITIQUE_AND_REVISE,
    seed_from_instead="critique", model_setting_keys=("revise_model",),
    required_prompt_files=lambda config, prompts_directory: [utils.sdf_template_path(prompts_directory, 3).name],
    reason_inactive=lambda config, prompts_directory: ("sdf.revision.split is true, so the critique and the revise are two calls"
                                                      if revision_is_split(config) else None),
    reads=lambda run: run.records(DRAFTS), build_request=build_request, read_reply=read_reply,
    records=one_record_per_accepted_document(REWRITES, lambda draft, update, run: rewrite_record(draft, update)),
    writes=(REWRITES,),
    name_in_messages=layer_name_in_messages, progress_line=progress_line)
