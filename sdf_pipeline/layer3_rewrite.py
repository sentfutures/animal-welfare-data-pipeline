"""Layer 3: review and rewrite each draft against the constitution and its spec.

The alignment-critical pass — do not skip or abbreviate it. Every template
holds the constitution, principles, the review checks and the failure-modes
checklist (constitution/failure_modes.md) in its SYSTEM section; the USER
section delivers the generating spec, the dealt variable lines, the assigned
considerations and the draft. The rewrite must come back inside
<improved_document> tags. Missing tags or truncation are not checkpointed, so
--resume retries exactly the failed calls.

The critique and the rewrite are TWO CALLS by default (layer3a_critique.txt
then layer3b_revise.txt); sdf.revision.split: false pins the single-call
layer3.txt. Revision variables are rendered by revision_style.py.

The layer is three ``DocumentStep`` classes run by ``sdf_pipeline.step_runner.StepRunner``:
``CritiqueStep`` and ``ReviseStep`` (the two-call path) and ``CombinedReviewAndReviseStep``
(the legacy single call, ``sdf.revision.split: false``). Each reads the description, variables
and considerations from the draft record, that is, from what the text was written from.
``run`` is the layer's entry point, called by ``sdf_pipeline/run.py`` and ``experiments/experiment.py``:
it picks the steps for the config, runs them in order and returns the rewrite records in input order.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import constitution_loader, deal_constraints, utils
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import revision_style
from sdf_pipeline.step_runner import StepRunner
from sdf_pipeline.update_steps import (
    Accepted, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, StepPolicy,
    line_for_document_left_to_retry)

IMPROVED_TAG_RE = re.compile(r"<improved_document>(.*?)</improved_document>", re.DOTALL)
REVIEW_TAG_RE = re.compile(r"<review>(.*?)</review>", re.DOTALL)

_MAX_TOKENS = 8000  # review record + a full rewritten document
_CRITIQUE_MAX_TOKENS = 8000  # a problem list, but one written with no document to budget for

CRITIQUE_TEMPLATE = "layer3a_critique.txt"
REVISE_TEMPLATE = "layer3b_revise.txt"

SYSTEMIC_STOP_MESSAGE = (
    "layer3_rewrite: every pending API call failed — this is systemic "
    "(auth, backend, network, or an error building the prompt), not per-document; fix and --resume."
)


def require_two_call_templates(prompts_dir: Path) -> None:
    for name in (CRITIQUE_TEMPLATE, REVISE_TEMPLATE):
        if not (prompts_dir / name).exists():
            raise SystemExit(
                f"layer3_rewrite: {name} is missing from {prompts_dir}. A run "
                "snapshots its own templates at creation, so a run created before "
                "the two-call layer 3 existed must be resumed with "
                "sdf.revision.split: false."
            )


@dataclass(frozen=True)
class RewrittenText:
    """The update the revise steps accept: the new text and the review that produced it."""
    content: str
    review: str


class _ReviewPromptRenderer:
    """Renders one layer 3 template for one draft record. Every value about the document comes
    from the draft record (the text's own description, variables and considerations), never from
    a later plan."""

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
        # The cards ride on the dealt hand like every other axis. An explicit
        # `revision` record overrides it, which is how a sweep hands one set of
        # cards to several arms over drafts that were dealt something else.
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


def _rewrite_record(draft: dict, update: RewrittenText) -> dict:
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


def _rewrite_progress_line(draft: dict, outcome) -> str:
    did = draft["doc_id"]
    if isinstance(outcome, Accepted):
        return f"  Rewrote {did} ({len(outcome.update.content)} chars)"
    if outcome.kind is FailureKind.THROWN_CALL:
        return line_for_document_left_to_retry(did, f"no rewrite ({outcome.reason})")
    return line_for_document_left_to_retry(did, outcome.reason)


def _interpret_rewrite_reply(reply: ModelReply):
    """(content, review_before_the_tags) or a Failed outcome."""
    if reply.stop_reason != "end_turn":
        return Failed(FailureKind.UNUSABLE_ANSWER, f"truncated rewrite (stop_reason={reply.stop_reason})")
    m = IMPROVED_TAG_RE.search(reply.text)
    content = m.group(1).strip() if m else ""
    if not content:
        return Failed(FailureKind.UNUSABLE_ANSWER, "no <improved_document> tags")
    return content, reply.text[:m.start()].strip()


class CritiqueStep(DocumentStep):
    """The first of the two calls: a problem list against the review checks. The document is the
    draft record. A reply with no ``<review>`` tags is accepted whole (the whole reply is a problem
    list in every observed malformed case, and paying again to chase the wrapper costs more)."""

    step_name = "critique"
    stage_file_name = "critiques.jsonl"
    # ``run`` applies the layer's systemic stop over the critique and revise calls together, as the
    # layer always has, so neither step stops the run on its own.
    policy = StepPolicy(failure_kinds_counting_towards_the_systemic_stop=frozenset())
    systemic_stop_message = SYSTEMIC_STOP_MESSAGE

    def __init__(self, config: dict, prompts_dir: Path):
        require_two_call_templates(prompts_dir)
        sdf = config["sdf"]
        self.critique_model = sdf.get("critique_model") or sdf.get("rewrite_model")
        self.renderer = _ReviewPromptRenderer(config, prompts_dir, prompts_dir / CRITIQUE_TEMPLATE)

    def document_id_of(self, draft: dict) -> str:
        return draft["doc_id"]

    def applies_to(self, draft: dict) -> bool:
        return True

    def build_request(self, draft: dict) -> ModelRequest:
        system, user = self.renderer.render(draft)
        return ModelRequest(
            user_message=user,
            system_prompt=system or "",
            model=self.critique_model,
            max_tokens=_CRITIQUE_MAX_TOKENS,
            stage="layer3_critique",
            item_id=draft["doc_id"],
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

    def progress_line(self, draft: dict, outcome) -> str | None:
        if isinstance(outcome, Failed):  # an accepted critique prints nothing: the revise prints the result
            return line_for_document_left_to_retry(draft["doc_id"], f"no rewrite ({outcome.reason})")
        return None


class ReviseStep(DocumentStep):
    """The second of the two calls: the rewritten text, with the voice and length directives. It
    applies to the drafts that have a critique (``critique_by_document_id``, banked on disk by the
    critique step), so a failed revise leaves its critique in ``critiques.jsonl`` and a resume pays for
    the revise alone. A draft whose rewrite is already on disk needs no critique and is not pending;
    the runner still returns its record."""

    step_name = "revise"
    stage_file_name = "rewrites.jsonl"
    policy = StepPolicy(failure_kinds_counting_towards_the_systemic_stop=frozenset())
    systemic_stop_message = SYSTEMIC_STOP_MESSAGE

    def __init__(self, config: dict, prompts_dir: Path, critique_by_document_id: dict):
        require_two_call_templates(prompts_dir)
        self.rewrite_model = config["sdf"].get("rewrite_model")
        self.critique_by_document_id = critique_by_document_id
        self.renderer = _ReviewPromptRenderer(config, prompts_dir, prompts_dir / REVISE_TEMPLATE)

    def document_id_of(self, draft: dict) -> str:
        return draft["doc_id"]

    def applies_to(self, draft: dict) -> bool:
        return draft["doc_id"] in self.critique_by_document_id

    def build_request(self, draft: dict) -> ModelRequest:
        style, length = self.renderer.directives(draft)
        system, user = self.renderer.render(
            draft, review=self.critique_by_document_id[draft["doc_id"]],
            style_directives=style, length_directive=length)
        return ModelRequest(
            user_message=user,
            system_prompt=system or "",
            model=self.rewrite_model,
            max_tokens=_MAX_TOKENS,
            stage="layer3_revise",
            item_id=draft["doc_id"],
            cache_system=True,
        )

    def interpret(self, draft: dict, reply: ModelReply):
        interpreted = _interpret_rewrite_reply(reply)
        if isinstance(interpreted, Failed):
            return interpreted
        content, _ = interpreted
        return Accepted(RewrittenText(content, self.critique_by_document_id[draft["doc_id"]]))

    def stage_file_record(self, draft: dict, update: RewrittenText) -> dict:
        return _rewrite_record(draft, update)

    def progress_line(self, draft: dict, outcome) -> str | None:
        return _rewrite_progress_line(draft, outcome)


class CombinedReviewAndReviseStep(DocumentStep):
    """The legacy single call (``sdf.revision.split: false``, ``layer3.txt``): one reply holds the
    review and the rewrite, and the text before the tags is the review. Kept only as long as runs
    that need it can resume."""

    step_name = "combined_review_and_revise"
    stage_file_name = "rewrites.jsonl"
    policy = StepPolicy()
    systemic_stop_message = SYSTEMIC_STOP_MESSAGE

    def __init__(self, config: dict, prompts_dir: Path):
        self.rewrite_model = config["sdf"].get("rewrite_model")
        self.renderer = _ReviewPromptRenderer(config, prompts_dir, utils.sdf_template_path(prompts_dir, 3))

    def document_id_of(self, draft: dict) -> str:
        return draft["doc_id"]

    def applies_to(self, draft: dict) -> bool:
        return True

    def build_request(self, draft: dict) -> ModelRequest:
        style, length = self.renderer.directives(draft)
        system, user = self.renderer.render(draft, style_directives=style, length_directive=length)
        return ModelRequest(
            user_message=user,
            system_prompt=system or "",
            model=self.rewrite_model,
            max_tokens=_MAX_TOKENS,
            stage="layer3_rewrite",
            item_id=draft["doc_id"],
            cache_system=True,  # constitution + review checks identical across rewrites
        )

    def interpret(self, draft: dict, reply: ModelReply):
        interpreted = _interpret_rewrite_reply(reply)
        if isinstance(interpreted, Failed):
            return interpreted
        content, review = interpreted
        return Accepted(RewrittenText(content, review))

    def stage_file_record(self, draft: dict, update: RewrittenText) -> dict:
        return _rewrite_record(draft, update)

    def progress_line(self, draft: dict, outcome) -> str | None:
        return _rewrite_progress_line(draft, outcome)


def run(config: dict, prompts_dir: Path, output_dir: Path, drafts: list[dict]) -> list[dict]:
    split = bool((config["sdf"].get("revision") or {}).get("split", True))
    workers = config.get("workers", 1)
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    existing_rewrite_ids = {r["doc_id"] for r in utils.load_jsonl(output_dir / "rewrites.jsonl")}
    pending_ids = [
        d["doc_id"] for d in drafts
        if d["doc_id"] not in existing_rewrite_ids and not checkpoint.is_done(d["doc_id"])
    ]

    # The systemic guard counts documents with a THROWN call only (auth, backend, network or a prompt
    # that could not be built), across the critique and the revise together. A critique that came back
    # truncated or empty is a per-document content failure, the same class as a truncated rewrite, and
    # must not look systemic.
    documents_with_thrown_call: set = set()

    if split:
        critique_step = CritiqueStep(config, prompts_dir)
        # A draft is finished, and gets no critique, when its rewrite is already on disk; accepting a
        # critique marks nothing, because only an accepted rewrite finishes a draft.
        critique_runner = StepRunner(
            critique_step, output_dir, workers=workers,
            checkpoint=checkpoint, marks_nothing_done=True, also_finished_document_ids=existing_rewrite_ids)
        documents_with_thrown_call.update(critique_runner.run(drafts).thrown_call_document_ids)
        # Critiques are banked separately from rewrites (the critique runner appended each as it
        # was accepted): a revise call that fails must not re-bill the critique that fed it.
        critique_by_document_id = {r["doc_id"]: r["review"] for r in utils.load_jsonl(output_dir / "critiques.jsonl")}
        rewrite_step = ReviseStep(config, prompts_dir, critique_by_document_id)
    else:
        rewrite_step = CombinedReviewAndReviseStep(config, prompts_dir)
    rewrite_runner = StepRunner(
        rewrite_step, output_dir, workers=workers, checkpoint=checkpoint)
    result = rewrite_runner.run(drafts)
    documents_with_thrown_call.update(result.thrown_call_document_ids)

    if pending_ids and len(documents_with_thrown_call) == len(pending_ids):
        raise SystemExit(SYSTEMIC_STOP_MESSAGE)
    return result.records_in_input_order
