"""Layer 4 (draft, critique and revise): what the critique, the revise and the one-call step share.

The critique and the revision are TWO CALLS by default (layer3a_critique.txt then layer3b_revise.txt);
``sdf.revision.split: false`` pins the single-call layer3.txt. This module holds the prompt renderer the three steps
use (every value about the document comes from the draft record), the rewrite record and progress line, the
reading of a rewrite reply, ``revision_is_split``, and the exit message the layer's systemic stop uses. Revision
variables are rendered by ``revision_style.py``. See ``layer4_draft_critique_and_revise.py`` for the layer.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from shared import constitution_loader, deal_constraints, sdf_run_layouts, utils
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import revision_style
from sdf_pipeline.dealt_value_identifiers import value_ids_field_of
from sdf_pipeline.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, line_for_document_left_to_retry, systemic_stop_message_for)

IMPROVED_TAG_RE = re.compile(r"<improved_document>(.*?)</improved_document>", re.DOTALL)
REVIEW_TAG_RE = re.compile(r"<review>(.*?)</review>", re.DOTALL)

MAX_TOKENS = 8000  # review record + a full rewritten document
CRITIQUE_MAX_TOKENS = 8000  # a problem list, but one written with no document to budget for

CRITIQUE_TEMPLATE = "layer3a_critique.txt"
REVISE_TEMPLATE = "layer3b_revise.txt"


def layer_name_in_messages() -> str:
    """What the critique, the revise and the one-call step's exit messages start with: the layer's name in the
    run's layout (``layer3_rewrite`` in a four-layer run)."""
    return sdf_run_layouts.active_layout().critique_and_revise_message_label


class SystemicStopMessageNamingTheLayer:
    """The systemic stop of the critique, the revise and the one-call step names the layer, not the step that
    made the calls (read when asked, so that a four-layer run keeps its old name)."""

    def __get__(self, instance, owner):
        return systemic_stop_message_for(layer_name_in_messages(), "API call", "document")


def require_two_call_templates(prompts_dir: Path) -> None:
    for name in (CRITIQUE_TEMPLATE, REVISE_TEMPLATE):
        if not (prompts_dir / name).exists():
            raise SystemExit(
                f"{layer_name_in_messages()}: {name} is missing from {prompts_dir}. A run "
                "snapshots its own templates at creation, so a run created before "
                "the two-call layer 3 existed must be resumed with "
                "sdf.revision.split: false."
            )


@dataclass(frozen=True)
class RewrittenText:
    """The update the revise steps accept: the new text and the review that produced it."""
    content: str
    review: str


class ReviewPromptRenderer:
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


def rewrite_record(draft: dict, update: RewrittenText) -> dict:
    record = {
        "doc_id": draft["doc_id"],
        "variables": draft["variables"],
        **value_ids_field_of(draft),
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


def revision_is_split(config: dict) -> bool:
    """True when the critique and the revise are two calls (the default); False for the single combined call."""
    return bool((config["sdf"].get("revision") or {}).get("split", True))


def load_critiques_by_document_id(output_dir: Path) -> dict:
    """{doc_id: the accepted critique} from critiques.jsonl; what the revise step is built with."""
    return {record["doc_id"]: record["review"] for record in utils.load_jsonl(Path(output_dir) / "critiques.jsonl")}
