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
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, constitution_loader, deal_constraints, utils
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import revision_style

IMPROVED_TAG_RE = re.compile(r"<improved_document>(.*?)</improved_document>", re.DOTALL)
REVIEW_TAG_RE = re.compile(r"<review>(.*?)</review>", re.DOTALL)

_MAX_TOKENS = 8000  # review record + a full rewritten document
_CRITIQUE_MAX_TOKENS = 8000  # a problem list, but one written with no document to budget for

CRITIQUE_TEMPLATE = "layer3a_critique.txt"
REVISE_TEMPLATE = "layer3b_revise.txt"


def run(config: dict, prompts_dir: Path, output_dir: Path, drafts: list[dict]) -> list[dict]:
    output_path = output_dir / "rewrites.jsonl"
    critique_path = output_dir / "critiques.jsonl"
    checkpoint = utils.Checkpoint(output_dir / "_checkpoint.json")
    sdf = config["sdf"]
    rev = sdf.get("revision") or {}
    mode = rev.get("style_mode", "none")
    use_length = rev.get("use_length", False)
    split = bool(rev.get("split", True))

    if split:
        for name in (CRITIQUE_TEMPLATE, REVISE_TEMPLATE):
            if not (prompts_dir / name).exists():
                raise SystemExit(
                    f"layer3_rewrite: {name} is missing from {prompts_dir}. A run "
                    "snapshots its own templates at creation, so a run created before "
                    "the two-call layer 3 existed must be resumed with "
                    "sdf.revision.split: false."
                )

    constitution_dir = utils.resolve_constitution_dir(prompts_dir)
    constitution_claude = constitution_loader.load_constitution_claude(constitution_dir)
    principles = constitution_loader.format_principles(
        constitution_loader.load_principles(constitution_dir)
    )
    failure_modes = constitution_loader.load_failure_modes(constitution_dir)

    def prepare(path: Path):
        """(text, considerations-less variant, variable-line renderer) for one template.
        A document with no considerations leaves no trace of them in its prompt."""
        text = path.read_text(encoding="utf-8")
        return (
            text,
            deal_constraints.drop_template_lines(text, ["considerations_spec"]),
            cp.render_variable_lines_for_template(text, prompts_dir, config),
        )

    if split:
        critique_tpl = prepare(prompts_dir / CRITIQUE_TEMPLATE)
        revise_tpl = prepare(prompts_dir / REVISE_TEMPLATE)
    else:
        single_tpl = prepare(utils.sdf_template_path(prompts_dir, 3))

    def render(tpl, draft: dict, **extra) -> tuple[str, str]:
        text, without_considerations, render_variable_lines = tpl
        considerations_spec = draft.get("considerations_spec", "")
        return cp.split_sections((
            text if considerations_spec else without_considerations
        ).format(
            constitution_claude=constitution_claude,
            constitution_principles=principles,
            failure_modes=failure_modes,
            document_description=draft["description"],
            document=draft["content"],
            considerations_spec=considerations_spec,
            variable_lines=render_variable_lines(
                {"prompt_id": draft["doc_id"], "variables": draft["variables"]}),
            **extra,
        ))

    existing = {r["doc_id"]: r for r in utils.load_jsonl(output_path)}
    results = [existing[d["doc_id"]] for d in drafts if d["doc_id"] in existing]
    pending = [
        d for d in drafts
        if d["doc_id"] not in existing and not checkpoint.is_done(d["doc_id"])
    ]
    # Critiques are banked separately from rewrites: a revise call that fails
    # must not re-bill the critique that fed it on the next --resume. The
    # workers read this snapshot and never write; the main thread appends to
    # the file and tracks what it wrote in `newly_banked`, so nothing mutates
    # under a running worker.
    banked = {r["doc_id"]: r["review"] for r in utils.load_jsonl(critique_path)}
    newly_banked: set[str] = set()

    def directives(draft: dict) -> tuple[str, str]:
        # The cards ride on the dealt hand like every other axis. An explicit
        # `revision` record overrides it, which is how a sweep hands one set of
        # cards to several arms over drafts that were dealt something else.
        deal = draft.get("revision") or {
            "cards": cp.revision_cards(draft.get("variables"))
        }
        return revision_style.directives_for(
            deal,
            mode=mode,
            use_length=use_length,
            document_type=(draft.get("variables") or {}).get("document_type", ""),
        )

    def rewrite_one(draft: dict):
        """-> (critique_or_None, raw_or_None, stop). Pure: the caller writes."""
        style, length = directives(draft)
        did = draft["doc_id"]
        if not split:
            system, user = render(single_tpl, draft,
                                  style_directives=style, length_directive=length)
            try:
                raw, stop = api.call_claude(
                    user,
                    system_prompt=system or "",
                    model=sdf.get("rewrite_model"),
                    max_tokens=_MAX_TOKENS,
                    stage="layer3_rewrite",
                    item_id=did,
                    return_stop_reason=True,
                    cache_system=True,  # constitution + review checks identical across rewrites
                )
                return None, raw, stop
            except Exception as e:
                # Per-item failures (e.g. a usage-policy false positive on the
                # claude_code backend) skip the doc instead of killing the layer;
                # unmarked work is retried by --resume.
                return None, None, f"error: {type(e).__name__}: {e}"

        critique = banked.get(did)
        if critique is None:
            try:
                system, user = render(critique_tpl, draft)
                raw, stop = api.call_claude(
                    user,
                    system_prompt=system or "",
                    model=sdf.get("critique_model") or sdf.get("rewrite_model"),
                    max_tokens=_CRITIQUE_MAX_TOKENS,
                    stage="layer3_critique",
                    item_id=did,
                    return_stop_reason=True,
                    cache_system=True,
                )
            except Exception as e:
                return None, None, f"error: {type(e).__name__}: {e}"
            if stop != "end_turn":
                return None, None, f"critique truncated (stop_reason={stop})"
            m = REVIEW_TAG_RE.search(raw)
            # Fail open on a missing tag: the whole reply is a problem list in
            # every observed malformed case, and re-billing a usable critique
            # to chase its wrapper is the more expensive mistake.
            critique = (m.group(1) if m else raw).strip()
            if not critique:
                return None, None, "critique empty"

        try:
            system, user = render(revise_tpl, draft, review=critique,
                                  style_directives=style, length_directive=length)
            raw, stop = api.call_claude(
                user,
                system_prompt=system or "",
                model=sdf.get("rewrite_model"),
                max_tokens=_MAX_TOKENS,
                stage="layer3_revise",
                item_id=did,
                return_stop_reason=True,
                cache_system=True,
            )
            return critique, raw, stop
        except Exception as e:
            return critique, None, f"error: {type(e).__name__}: {e}"

    workers = config.get("workers", 1)
    failed_calls = 0
    for draft, (critique, raw, stop) in zip(pending, utils.parallel_map(rewrite_one, pending, workers)):
        did = draft["doc_id"]
        # Bank a fresh critique before judging the rewrite, so a failed revise
        # leaves paid work on disk for --resume.
        if critique is not None and did not in banked and did not in newly_banked:
            newly_banked.add(did)
            utils.append_jsonl({"doc_id": did, "review": critique}, critique_path)
        if raw is None:
            # The systemic guard counts THROWN calls only (auth, backend,
            # network — the "error:" marker set above). A critique that came
            # back truncated or empty is a per-document content failure, the
            # same class as a truncated rewrite, and must not look systemic.
            if str(stop).startswith("error:"):
                failed_calls += 1
            print(f"  {did}: no rewrite ({stop}) — will retry on resume")
            continue
        if stop != "end_turn":
            print(f"  {did}: truncated rewrite (stop_reason={stop}) — will retry on resume")
            continue
        m = IMPROVED_TAG_RE.search(raw)
        content = m.group(1).strip() if m else ""
        if not content:
            print(f"  {did}: no <improved_document> tags — will retry on resume")
            continue
        record = {
            "doc_id": did,
            "variables": draft["variables"],
            "description": draft["description"],
            "considerations_spec": draft.get("considerations_spec", ""),
            "review": critique if critique is not None else raw[:m.start()].strip(),
            "content": content,
        }
        if draft.get("revision"):
            record["revision"] = draft["revision"]
        results.append(record)
        utils.append_jsonl(record, output_path)
        checkpoint.mark_done(did)
        print(f"  Rewrote {did} ({len(content)} chars)")

    if pending and failed_calls == len(pending):
        raise SystemExit(
            "layer3_rewrite: every pending API call failed — this is systemic "
            "(auth, backend, or network), not per-document; fix and --resume."
        )
    return results
