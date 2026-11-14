"""Layer 3: review and rewrite each draft against the constitution and its spec; alignment-critical, so do not skip or abbreviate it.
Two calls by default, the critique (layer3a_critique.txt) then the revise (layer3b_revise.txt, which writes the text inside <improved_document> tags);
``sdf.revision.split: false`` pins the single-call layer3.txt. A critique is banked in critiques.jsonl before the revise runs; missing tags or truncation are not recorded, so ``--resume`` retries them."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import constitution_loader, deal_constraints, utils
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import revision_style
from sdf_pipeline.layer2_draft import DRAFTS
from sdf_pipeline.shared_step_machinery.step_record import (
    LAYER3, Run, StageFile, Step, loaded_once_per_run, one_record_per_accepted_document)
from sdf_pipeline.shared_step_machinery.step_runner import run_step
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, StepPolicy, StepProgress, WhereAnErrorIsAThrownCall,
    line_for_document_left_to_retry, systemic_stop_message_for)

IMPROVED_TAG_RE = re.compile(r"<improved_document>(.*?)</improved_document>", re.DOTALL)
REVIEW_TAG_RE = re.compile(r"<review>(.*?)</review>", re.DOTALL)

_MAX_TOKENS = 8000  # review record + a full rewritten document
_CRITIQUE_MAX_TOKENS = 8000  # a problem list, but one written with no document to budget for

CRITIQUE_TEMPLATE = "layer3a_critique.txt"
REVISE_TEMPLATE = "layer3b_revise.txt"
MESSAGE_LABEL = "layer3_rewrite"   # what the exit messages of all three steps start with
CHECKPOINT_FILENAME = "_checkpoint.json"

CRITIQUES = StageFile(LAYER3, "critiques.jsonl", "doc_id")
REWRITES = StageFile(LAYER3, "rewrites.jsonl", "doc_id")


def require_two_call_templates(prompts_dir: Path) -> None:
    for name in (CRITIQUE_TEMPLATE, REVISE_TEMPLATE):
        if not (prompts_dir / name).exists():
            raise SystemExit(
                f"{MESSAGE_LABEL}: {name} is missing from {prompts_dir}. A run "
                "snapshots its own templates at creation, so a run created before "
                "the two-call layer 3 existed must be resumed with "
                "sdf.revision.split: false."
            )


def revision_is_split(config: dict) -> bool:
    """True when the critique and the revise are two calls (the default); False for the single combined call."""
    return bool((config["sdf"].get("revision") or {}).get("split", True))


@dataclass(frozen=True)
class RewrittenText:
    """The update the revise steps accept: the new text and the review that produced it."""
    content: str
    review: str


class ReviewPromptRenderer:
    """Renders one review template for one draft record, taking every value from the draft, never from a later plan."""

    def __init__(self, config: dict, prompts_dir: Path, template_path: Path):
        revision_config = config["sdf"].get("revision") or {}
        self.style_mode = revision_config.get("style_mode", "none")
        self.use_length = revision_config.get("use_length", False)
        constitution_dir = utils.resolve_constitution_dir(prompts_dir)
        self.constitution_claude = constitution_loader.load_constitution_claude(constitution_dir)
        self.principles = constitution_loader.format_principles(
            constitution_loader.load_principles(constitution_dir)
        )
        self.failure_modes = constitution_loader.load_failure_modes(constitution_dir)
        # A document with no considerations leaves no trace of them in its prompt.
        self.template_text = template_path.read_text(encoding="utf-8")
        self.template_without_considerations = deal_constraints.drop_template_lines(
            self.template_text, ["considerations_spec"])
        self.render_variable_lines = cp.render_variable_lines_for_template(
            self.template_text, prompts_dir, config)

    def directives(self, draft: dict) -> tuple[str, str]:
        # The cards ride on the dealt hand like every other axis. An explicit `revision` record overrides it,
        # which is how a sweep hands one set of cards to several arms over drafts that were dealt something else.
        deal = draft.get("revision") or {"cards": cp.revision_cards(draft.get("variables"))}
        return revision_style.directives_for(
            deal,
            mode=self.style_mode,
            use_length=self.use_length,
            document_type=(draft.get("variables") or {}).get("document_type", ""),
        )

    def render(self, draft: dict, **extra) -> tuple[str, str]:
        considerations_spec = draft.get("considerations_spec", "")
        return cp.split_sections((
            self.template_text if considerations_spec else self.template_without_considerations
        ).format(
            constitution_claude=self.constitution_claude,
            constitution_principles=self.principles,
            failure_modes=self.failure_modes,
            document_description=draft["description"],
            document=draft["content"],
            considerations_spec=considerations_spec,
            variable_lines=self.render_variable_lines(
                {"prompt_id": draft["doc_id"], "variables": draft["variables"]}),
            **extra,
        ))


def rewrite_record(draft: dict, update: RewrittenText) -> dict:
    record = {
        "doc_id": draft["doc_id"],
        "variables": draft["variables"],
        "description": draft["description"],
        "considerations_spec": draft.get("considerations_spec", ""),
        "review": update.review,
        "content": update.content,
    }
    if draft.get("revision"):
        record["revision"] = draft["revision"]
    return record


def rewrite_progress_line(draft: dict, outcome) -> str:
    did = draft["doc_id"]
    if isinstance(outcome, Accepted):
        return f"  Rewrote {did} ({len(outcome.update.content)} chars)"
    if outcome.kind is FailureKind.THROWN_CALL:
        return line_for_document_left_to_retry(did, f"no rewrite ({outcome.reason})")
    return line_for_document_left_to_retry(did, outcome.reason)


def interpret_rewrite_reply(reply: ModelReply):
    """(content, review_before_the_tags) or a Failed outcome."""
    if reply.stop_reason != "end_turn":
        return Failed(FailureKind.UNUSABLE_ANSWER, f"truncated rewrite (stop_reason={reply.stop_reason})")
    m = IMPROVED_TAG_RE.search(reply.text)
    content = m.group(1).strip() if m else ""
    if not content:
        return Failed(FailureKind.UNUSABLE_ANSWER, "no <improved_document> tags")
    return content, reply.text[:m.start()].strip()


@loaded_once_per_run
def _rewrite_ids_on_disk(run: Run) -> set:
    return {record["doc_id"] for record in run.records(REWRITES)}


@loaded_once_per_run
def _checkpoint(run: Run) -> utils.Checkpoint:
    return utils.Checkpoint(run.stage_directories[LAYER3] / CHECKPOINT_FILENAME)


def is_rewritten(document_id: str, run: Run) -> bool:
    """Whether the draft's rewrite is on disk or marked done in the checkpoint file."""
    return document_id in _rewrite_ids_on_disk(run) or _checkpoint(run).is_done(document_id)


# ---------------------------------------------------------------- the critique

@loaded_once_per_run
def _critique_renderer(run: Run) -> ReviewPromptRenderer:
    return ReviewPromptRenderer(run.config, run.prompts_directory, run.prompts_directory / CRITIQUE_TEMPLATE)


def build_critique_request(documents: list, run: Run) -> ModelRequest:
    (draft,) = documents
    system, user = _critique_renderer(run).render(draft)
    return CRITIQUE_STEP.model_request(draft, user, system, run, max_tokens=_CRITIQUE_MAX_TOKENS, cache_system=True)


def read_critique_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """A reply with no ``<review>`` tags is accepted whole: re-billing a usable critique to chase its wrapper costs more."""
    (draft,) = documents
    if reply.stop_reason != "end_turn":
        outcome = Failed(FailureKind.UNUSABLE_ANSWER, f"critique truncated (stop_reason={reply.stop_reason})")
    else:
        m = REVIEW_TAG_RE.search(reply.text)
        critique = (m.group(1) if m else reply.text).strip()
        outcome = Accepted(critique) if critique else Failed(FailureKind.UNUSABLE_ANSWER, "critique empty")
    return {draft["doc_id"]: outcome}


def _critique_record(draft: dict, critique: str, run: Run) -> dict:
    return {"doc_id": draft["doc_id"], "review": critique}


def critique_progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (draft,), (outcome,) = documents, outcomes.values()
    if isinstance(outcome, Failed):  # an accepted critique prints nothing: the revise prints the result
        return line_for_document_left_to_retry(draft["doc_id"], f"no rewrite ({outcome.reason})")
    return None


# The driver's systemic stop counts the critique and revise calls together, so neither step stops the run on its own.
CRITIQUE_STEP = Step(
    name="critique", folder_key=LAYER3, stage_tag="layer3_critique", model_setting_keys=("critique_model", "rewrite_model"),
    finished_elsewhere=lambda draft, run: is_rewritten(draft["doc_id"], run),
    build_request=build_critique_request, read_reply=read_critique_reply, records=one_record_per_accepted_document(CRITIQUES, _critique_record), writes=(CRITIQUES,),
    policy=StepPolicy(
        failure_kinds_counting_towards_the_systemic_stop=frozenset(),
        errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND),
    name_in_messages=MESSAGE_LABEL, progress_line=critique_progress_line)


# ---------------------------------------------------------------- the revise

@loaded_once_per_run
def _revise_renderer(run: Run) -> ReviewPromptRenderer:
    return ReviewPromptRenderer(run.config, run.prompts_directory, run.prompts_directory / REVISE_TEMPLATE)


@loaded_once_per_run
def critique_by_document_id(run: Run) -> dict:
    """{doc_id: critique} of the critiques banked in the run's layer 3 folder."""
    return {record["doc_id"]: record["review"] for record in run.records(CRITIQUES)}


def build_revise_request(documents: list, run: Run) -> ModelRequest:
    (draft,) = documents
    style, length = _revise_renderer(run).directives(draft)
    system, user = _revise_renderer(run).render(
        draft, review=critique_by_document_id(run)[draft["doc_id"]], style_directives=style, length_directive=length)
    return REVISE_STEP.model_request(draft, user, system, run, max_tokens=_MAX_TOKENS, cache_system=True)


def read_revise_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    (draft,) = documents
    interpreted = interpret_rewrite_reply(reply)
    if isinstance(interpreted, Failed):
        return {draft["doc_id"]: interpreted}
    content, _ = interpreted
    return {draft["doc_id"]: Accepted(RewrittenText(content, critique_by_document_id(run)[draft["doc_id"]]))}


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (draft,), (outcome,) = documents, outcomes.values()
    return rewrite_progress_line(draft, outcome)


def _rewrite_records(draft: dict, update: RewrittenText, run: Run) -> dict:
    return rewrite_record(draft, update)


REVISE_STEP = Step(
    name="revise", folder_key=LAYER3, stage_tag="layer3_revise", model_setting_keys=("rewrite_model",),
    applies=lambda draft, run: draft["doc_id"] in critique_by_document_id(run),
    build_request=build_revise_request, read_reply=read_revise_reply,
    records=one_record_per_accepted_document(REWRITES, _rewrite_records), writes=(REWRITES,),
    checkpoint_file_name=CHECKPOINT_FILENAME,
    policy=StepPolicy(
        failure_kinds_counting_towards_the_systemic_stop=frozenset(),
        errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND),
    name_in_messages=MESSAGE_LABEL, progress_line=progress_line)


# ---------------------------------------------------------------- the critique and revise in one call

@loaded_once_per_run
def _one_call_renderer(run: Run) -> ReviewPromptRenderer:
    return ReviewPromptRenderer(run.config, run.prompts_directory, utils.sdf_template_path(run.prompts_directory, 3))


def build_one_call_request(documents: list, run: Run) -> ModelRequest:
    (draft,) = documents
    style, length = _one_call_renderer(run).directives(draft)
    system, user = _one_call_renderer(run).render(draft, style_directives=style, length_directive=length)
    return ONE_CALL_STEP.model_request(
        draft, user, system, run, max_tokens=_MAX_TOKENS,
        cache_system=True,  # constitution + review checks identical across rewrites
    )


def read_one_call_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """The text before the ``<improved_document>`` tags is the review."""
    (draft,) = documents
    interpreted = interpret_rewrite_reply(reply)
    if isinstance(interpreted, Failed):
        return {draft["doc_id"]: interpreted}
    content, review = interpreted
    return {draft["doc_id"]: Accepted(RewrittenText(content, review))}


ONE_CALL_STEP = Step(
    name="critique_and_revise_in_one_call", folder_key=LAYER3, stage_tag="layer3_rewrite",
    model_setting_keys=("rewrite_model",),
    build_request=build_one_call_request, read_reply=read_one_call_reply,
    records=one_record_per_accepted_document(REWRITES, _rewrite_records), writes=(REWRITES,),
    checkpoint_file_name=CHECKPOINT_FILENAME, name_in_messages=MESSAGE_LABEL, progress_line=progress_line)


def run(config: dict, prompts_dir: Path, output_dir: Path, drafts: list[dict]) -> list[dict]:
    """Critique and revise every draft (one call or two); returns the rewrite records in input order."""
    run_of_layer = Run(config, prompts_dir, {LAYER3: output_dir})
    pending_ids = [d["doc_id"] for d in drafts if not is_rewritten(d["doc_id"], run_of_layer)]

    # The systemic guard counts only THROWN calls, across the critique and revise together: a truncated or empty
    # critique is a per-document content failure and must not look systemic.
    documents_with_thrown_call: set = set()

    if revision_is_split(config):
        require_two_call_templates(prompts_dir)
        # A draft is finished, and gets no critique, when its rewrite is already on disk; accepting a critique
        # marks nothing, because only an accepted rewrite finishes a draft.
        documents_with_thrown_call.update(run_step(CRITIQUE_STEP, run_of_layer, drafts).thrown_call_document_ids)
        # Critiques are banked separately from rewrites: a revise call that fails must not re-bill the critique that fed it.
        rewrite_step = REVISE_STEP
    else:
        rewrite_step = ONE_CALL_STEP
    result = run_step(rewrite_step, run_of_layer, drafts)
    documents_with_thrown_call.update(result.thrown_call_document_ids)

    if pending_ids and len(documents_with_thrown_call) == len(pending_ids):
        raise SystemExit(systemic_stop_message_for(MESSAGE_LABEL, "API call", "document"))
    return result.records_in_input_order
