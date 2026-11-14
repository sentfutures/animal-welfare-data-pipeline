"""Layer 4 (draft, critique and revise): draft each document, then critique and revise it against the constitution.

The layer is four steps, one module each in ``sdf_pipeline/steps/``: ``DraftStep`` (``draft.py``), ``CritiqueStep``
and ``ReviseStep`` (``critique.py``, ``revise.py``, the two-call path) and ``CritiqueAndReviseInOneCallStep``
(``critique_and_revise_in_one_call.py``, the legacy single call, ``sdf.revision.split: false``). What the last
three share is in ``layer4_draft_critique_and_revise_shared.py``. This module holds the layer's driver: ``run``
drafts every plan and then critiques and revises the drafts, ``run_critique_and_revise`` does the second half on
drafts it is given (``experiments/experiment.py`` and ``run.py`` for a four-layer run call it).

The critique and the revision are alignment-critical: do not skip or abbreviate it. Every template holds the
constitution, principles, the review checks and the failure-modes checklist (constitution/failure_modes.md) in its
SYSTEM section; the USER section delivers the generating spec, the dealt variable lines, the assigned considerations
and the draft. The rewrite must come back inside <improved_document> tags. Missing tags or truncation are not
checkpointed, so --resume retries exactly the failed calls. The critique and the rewrite are TWO CALLS by default
(layer3a_critique.txt then layer3b_revise.txt); sdf.revision.split: false pins the single-call layer3.txt.
"""

from pathlib import Path

from shared import sdf_run_layouts, utils
from sdf_pipeline.layer4_draft_critique_and_revise_shared import *  # noqa: F401,F403  (names the old layer 3 module had)
from sdf_pipeline.layer4_draft_critique_and_revise_shared import (
    layer_name_in_messages, load_critiques_by_document_id, revision_is_split)
from sdf_pipeline.step_runner import StepRunner
from sdf_pipeline.steps import draft as draft_step
from sdf_pipeline.steps.critique import CritiqueStep
from sdf_pipeline.steps.critique_and_revise_in_one_call import CritiqueAndReviseInOneCallStep
from sdf_pipeline.steps.draft import DraftStep  # noqa: F401
from sdf_pipeline.steps.revise import ReviseStep
from sdf_pipeline.update_steps import systemic_stop_message_for


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Draft every plan, then critique and revise the drafts; returns the rewrite records in input order."""
    drafts = draft_step.run(config, prompts_dir, output_dir, plans,
                            sdf_run_layouts.active_layout().draft_checkpoint_file_name)
    return run_critique_and_revise(config, prompts_dir, output_dir, drafts)


def run_critique_and_revise(config: dict, prompts_dir: Path, output_dir: Path, drafts: list[dict]) -> list[dict]:
    split = revision_is_split(config)
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
        rewrite_step = ReviseStep(config, prompts_dir, load_critiques_by_document_id(output_dir))
    else:
        rewrite_step = CritiqueAndReviseInOneCallStep(config, prompts_dir)
    rewrite_runner = StepRunner(
        rewrite_step, output_dir, workers=workers, checkpoint=checkpoint)
    result = rewrite_runner.run(drafts)
    documents_with_thrown_call.update(result.thrown_call_document_ids)

    if pending_ids and len(documents_with_thrown_call) == len(pending_ids):
        raise SystemExit(systemic_stop_message_for(layer_name_in_messages(), "API call", "document"))
    return result.records_in_input_order
