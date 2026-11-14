"""Layer 3: the consideration selection, a global linear programme over every scored document (no model call).
It reruns over all documents, so a late plan changes others' outcomes. Results are stored as ``considerations_spec`` and in ``variables``; ``attach_selections`` re-reads them on resume."""

from pathlib import Path

from sdf_pipeline.shared_step_machinery import considerations_assignment as assignment
from sdf_pipeline.shared_step_machinery.considerations_catalog import (
    CONSIDERATIONS_FILENAME, EPISTEMIC_LIST, LIST_NAMES, MORAL_LIST, ConsiderationCatalog, load_considerations)
from sdf_pipeline.layer3_consideration_scoring_and_selection.shared_by_this_layers_steps import (
    SELECTIONS_FILENAME, STATUS_ACCEPTED, STATUS_NONE, load_selection_template, selection_enabled)
from sdf_pipeline.layer2_description_write_and_description_rewrite.description_rewrite import plans_after_description_rewrite
from sdf_pipeline.shared_step_machinery.step_record import Run
from shared import utils
from shared.sdf_run_layouts import CONSIDERATION_SCORING_AND_SELECTION


def assign_considerations_to_documents(
        config: dict, catalog: ConsiderationCatalog, eligible: list[dict], score_rows: list[dict]) -> tuple[list[dict], dict]:
    """``(selection rows, coverage report)`` of the programme over every scored document; ``eligible`` are the plans offered considerations."""
    seed = config["sdf"].get("seed", 0)
    limits = assignment.limits_from_config(config)
    document_is_digital = digital_mind_flags(catalog, eligible)
    rows_by_document_id = {row["prompt_id"]: row for row in score_rows}
    flat_scores = assignment.flat_scores(rows_by_document_id)
    assigned_labels_by_document_id = assignment.assign_considerations(flat_scores, catalog, document_is_digital, limits, seed)
    selection_rows = _selection_rows(catalog, rows_by_document_id, assigned_labels_by_document_id)
    coverage_report = assignment.coverage_report(
        flat_scores, catalog, document_is_digital, assigned_labels_by_document_id, limits)
    return selection_rows, coverage_report


def digital_mind_flags(catalog: ConsiderationCatalog, eligible: list[dict]) -> dict[str, bool]:
    return {p["prompt_id"]: catalog.is_digital_mind_document(p["variables"]) for p in eligible}


def _selection_rows(
    catalog: ConsiderationCatalog, score_rows: dict[str, dict], assigned: dict[str, list[str]],
) -> list[dict]:
    """One row per scored document: one accepted outcome per assigned entry, or a single
    "none" outcome per list with nothing assigned."""
    rows = []
    for prompt_id in sorted(score_rows):
        scores = score_rows[prompt_id]["scores"]
        outcomes = []
        for list_name in LIST_NAMES:
            labels = [label for label in assigned.get(prompt_id, []) if label in scores.get(list_name, {})]
            for label in labels:
                item = scores[list_name][label]
                outcomes.append({"list": list_name, "label": label, "reason": item["where_it_arises"],
                                 "status": STATUS_ACCEPTED, "score": item["score"]})
            if not labels:
                outcomes.append({"list": list_name, "label": None, "reason": "", "status": STATUS_NONE})
        rows.append({"prompt_id": prompt_id, "outcomes": outcomes})
    return rows


def attach_selection_rows(catalog: ConsiderationCatalog, selection_rows: list[dict], plans: list[dict]) -> list[dict]:
    """Copies of ``plans`` with each described plan's assigned entries attached (plans without a
    description are INCOHERENT and pass through unchanged)."""
    by_document = {row["prompt_id"]: row["outcomes"] for row in selection_rows}
    attached = []
    for plan in plans:
        if not plan.get("description"):
            attached.append(plan)
            continue
        entries = sorted(
            ((catalog.entry_by_label(o["label"]), o["reason"])
             for o in by_document.get(plan["prompt_id"], []) if o["status"] == STATUS_ACCEPTED),
            key=lambda pair: (LIST_NAMES.index(pair[0].list_name),
                              catalog.entries_of(pair[0].list_name).index(pair[0])),
        )
        variables = {
            **plan["variables"],
            "moral_considerations": [e.label for e, _ in entries if e.list_name == MORAL_LIST],
            "epistemic_considerations": [e.label for e, _ in entries if e.list_name == EPISTEMIC_LIST],
        }
        attached.append({**plan, "variables": variables, "considerations_spec": assemble_considerations_spec(entries)})
    return attached


def attach_selections(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Plans with saved selections attached, no calls; ``plans`` unchanged when disabled, no selector files, or no saved selections."""
    if not selection_enabled(config):
        return plans
    if load_selection_template(prompts_dir) is None or not (Path(prompts_dir) / CONSIDERATIONS_FILENAME).is_file():
        return plans
    rows = utils.load_jsonl(Path(output_dir) / SELECTIONS_FILENAME)
    if not rows:
        return plans
    return attach_selection_rows(load_considerations(prompts_dir), rows, plans)


def assemble_considerations_spec(picks: list) -> str:
    """The text block handed to later prompts: one line per pick, 'Label (gloss) —
    the selector's sentence on where it arises', moral picks first and each list
    in catalog order. Empty string when there are no picks."""
    return "\n".join(
        f"{entry.label} ({entry.gloss}) — {reason}" if entry.gloss else f"{entry.label} — {reason}"
        for entry, reason in picks
    )


def plans_with_selections(run: Run) -> list[dict]:
    """The plans after the description rewrite with the saved selections attached, no calls."""
    return attach_selections(
        run.config, run.prompts_directory, run.stage_directories[CONSIDERATION_SCORING_AND_SELECTION],
        plans_after_description_rewrite(run))
