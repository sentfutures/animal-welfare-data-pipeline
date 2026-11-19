"""Layer 4: the driver ``run`` drafts every plan, then critiques and revises (``run_critique_and_revise`` does the second half on given drafts).
The critique and the revision are alignment-critical: do not skip or abbreviate them. They are two calls
(layer3a_critique.txt, layer3b_revise.txt)."""

from pathlib import Path

from shared import sdf_run_layouts, utils
from shared.sdf_run_layouts import CRITIQUE_AND_REVISE
from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import *  # noqa: F401,F403  (the steps' names)
from sdf_pipeline.layer4_draft_critique_and_revise.shared_by_this_layers_steps import (
    layer_name_in_messages, require_two_call_templates)
from sdf_pipeline.compatibility_with_runs_made_by_earlier_code.checkpoint_marks import marked_as_finished
from sdf_pipeline.shared_step_machinery.step_record import Run
from sdf_pipeline.shared_step_machinery.step_runner import run_step
from sdf_pipeline.layer4_draft_critique_and_revise import critique, draft, revise
from sdf_pipeline.shared_step_machinery.update_steps import systemic_stop_message_for


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Draft every plan, then critique and revise the drafts; returns the rewrite records in input order."""
    drafts = draft.run(config, prompts_dir, output_dir, plans)
    return run_critique_and_revise(config, prompts_dir, output_dir, drafts)


def run_critique_and_revise(config: dict, prompts_dir: Path, output_dir: Path, drafts: list[dict]) -> list[dict]:
    run_of_layer = Run(config, prompts_dir, {CRITIQUE_AND_REVISE: output_dir})
    marked_ids = marked_as_finished("revise", output_dir) or set()
    existing_rewrite_ids = {r["doc_id"] for r in utils.load_jsonl(output_dir / "rewrites.jsonl")}
    pending_ids = [
        d["doc_id"] for d in drafts
        if d["doc_id"] not in existing_rewrite_ids and d["doc_id"] not in marked_ids
    ]

    # The systemic guard counts only THROWN calls, across the critique and revise together: a truncated or empty critique is a per-document content failure and must not look systemic.
    documents_with_thrown_call: set = set()

    require_two_call_templates(prompts_dir)
    # A draft is finished, and gets no critique, when its rewrite is already on disk; accepting a
    # critique marks nothing, because only an accepted rewrite finishes a draft.
    documents_with_thrown_call.update(run_step(critique.STEP, run_of_layer, drafts).thrown_call_document_ids)
    # Critiques are banked separately from rewrites (the critique step appended each as it
    # was accepted): a revise call that fails must not re-bill the critique that fed it.
    result = run_step(revise.STEP, run_of_layer, drafts)
    documents_with_thrown_call.update(result.thrown_call_document_ids)

    if pending_ids and len(documents_with_thrown_call) == len(pending_ids):
        raise SystemExit(systemic_stop_message_for(layer_name_in_messages(), "API call", "document"))
    return result.records_in_input_order
