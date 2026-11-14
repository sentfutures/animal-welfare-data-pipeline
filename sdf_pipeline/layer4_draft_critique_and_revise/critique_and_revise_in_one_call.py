"""Layer 4 step: the critique and the revise in one call, the legacy single call kept while runs that need it can resume.

``sdf.revision.split: false`` selects it (template ``layer3.txt``): one reply holds the review and the rewrite, and
the text before the ``<improved_document>`` tags is the review. It runs in place of the critique and the revise.
"""

from pathlib import Path

from shared import utils
from sdf_pipeline.shared_step_machinery.renamed_model_settings import model_setting

from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import (
    MAX_TOKENS, ReviewPromptRenderer, RewrittenText, SystemicStopMessageNamingTheLayer, interpret_rewrite_reply,
    rewrite_progress_line, rewrite_record)
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, DocumentStep, Failed, ModelReply, ModelRequest, StepPolicy, StepProgress


class CritiqueAndReviseInOneCallStep(DocumentStep):
    """The legacy single call (``sdf.revision.split: false``, ``layer3.txt``): one reply holds the
    review and the rewrite, and the text before the tags is the review. Kept only as long as runs
    that need it can resume."""

    step_name = "critique_and_revise_in_one_call"
    stage_file_name = "rewrites.jsonl"
    policy = StepPolicy()
    systemic_stop_message = SystemicStopMessageNamingTheLayer()

    def __init__(self, config: dict, prompts_dir: Path):
        self.revise_model = model_setting(config["sdf"], "revise_model")
        self.renderer = ReviewPromptRenderer(config, prompts_dir, utils.sdf_template_path(prompts_dir, 3))

    def build_request(self, draft: dict) -> ModelRequest:
        style, length = self.renderer.directives(draft)
        system, user = self.renderer.render(draft, style_directives=style, length_directive=length)
        return self.model_request(
            draft, user, system, self.revise_model,
            max_tokens=MAX_TOKENS,
            cache_system=True,  # constitution + review checks identical across rewrites
        )

    def interpret(self, draft: dict, reply: ModelReply):
        interpreted = interpret_rewrite_reply(reply)
        if isinstance(interpreted, Failed):
            return interpreted
        content, review = interpreted
        return Accepted(RewrittenText(content, review))

    def stage_file_record(self, draft: dict, update: RewrittenText) -> dict:
        return rewrite_record(draft, update)

    def progress_line(self, draft: dict, outcome, progress: StepProgress) -> str | None:
        return rewrite_progress_line(draft, outcome)
