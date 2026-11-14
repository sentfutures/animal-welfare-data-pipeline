"""Layer 4 step: the revise, the second of the two revision calls (the rewritten text, with the voice and length directives).

The rewrite must come back inside <improved_document> tags. Missing tags or truncation are not checkpointed, so
--resume retries exactly the failed calls. The step applies to the drafts that have a critique and reads the
description, variables and considerations from the draft record, that is, from what the text was written from.
"""

from pathlib import Path

from sdf_pipeline.shared_step_machinery.renamed_model_settings import model_setting

from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import (
    MAX_TOKENS, REVISE_TEMPLATE, ReviewPromptRenderer, RewrittenText, SystemicStopMessageNamingTheLayer,
    interpret_rewrite_reply, require_two_call_templates, rewrite_progress_line, rewrite_record)
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, DocumentStep, Failed, ModelReply, ModelRequest, StepPolicy, StepProgress


class ReviseStep(DocumentStep):
    """The second of the two calls: the rewritten text, with the voice and length directives. It
    applies to the drafts that have a critique (``critique_by_document_id``, banked on disk by the
    critique step), so a failed revise leaves its critique in ``critiques.jsonl`` and a resume pays for
    the revise alone. A draft whose rewrite is already on disk needs no critique and is not pending;
    the runner still returns its record."""

    step_name = "revise"
    stage_file_name = "rewrites.jsonl"
    policy = StepPolicy(failure_kinds_counting_towards_the_systemic_stop=frozenset())
    systemic_stop_message = SystemicStopMessageNamingTheLayer()

    def __init__(self, config: dict, prompts_dir: Path, critique_by_document_id: dict):
        require_two_call_templates(prompts_dir)
        self.revise_model = model_setting(config["sdf"], "revise_model")
        self.critique_by_document_id = critique_by_document_id
        self.renderer = ReviewPromptRenderer(config, prompts_dir, prompts_dir / REVISE_TEMPLATE)

    def applies_to(self, draft: dict) -> bool:
        return draft["doc_id"] in self.critique_by_document_id

    def build_request(self, draft: dict) -> ModelRequest:
        style, length = self.renderer.directives(draft)
        system, user = self.renderer.render(
            draft, review=self.critique_by_document_id[draft["doc_id"]],
            style_directives=style, length_directive=length)
        return self.model_request(
            draft, user, system, self.revise_model,
            max_tokens=MAX_TOKENS,
            cache_system=True,
        )

    def interpret(self, draft: dict, reply: ModelReply):
        interpreted = interpret_rewrite_reply(reply)
        if isinstance(interpreted, Failed):
            return interpreted
        content, _ = interpreted
        return Accepted(RewrittenText(content, self.critique_by_document_id[draft["doc_id"]]))

    def stage_file_record(self, draft: dict, update: RewrittenText) -> dict:
        return rewrite_record(draft, update)

    def progress_line(self, draft: dict, outcome, progress: StepProgress) -> str | None:
        return rewrite_progress_line(draft, outcome)
