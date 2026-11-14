"""Layer 3 step: the consideration selection, the global linear programme over every scored document (no model call).

The documents are score rows (``prompt_id``, ``scores``); each gets an ``Accepted`` outcome whose update is its selection
outcomes. The programme (``considerations_assignment.py``) respects every entry's target count and the per-document
caps, and is solved again over every document each time it runs, so a plan that arrives late changes other
documents' outcomes. The assigned entries are assembled in code into ``considerations_spec``, a text block stored on
the plan record next to ``description`` and carried through the draft, rewrite and final records (empty string
when a document has none); each plan's ``variables`` also gains ``moral_considerations`` and
``epistemic_considerations``: lists of the assigned labels, for analysis. ``attach_selections`` reads the saved
selections back when a run resumes past this layer.
"""

from pathlib import Path

from sdf_pipeline.shared_step_machinery import considerations_assignment as assignment
from sdf_pipeline.shared_step_machinery.considerations_catalog import (
    CONSIDERATIONS_FILENAME, EPISTEMIC_LIST, LIST_NAMES, MORAL_LIST, ConsiderationCatalog, load_considerations)
from sdf_pipeline.layer3_consideration_scoring_and_selection.shared_by_this_layers_steps import (
    SELECTIONS_FILENAME, STATUS_ACCEPTED, STATUS_NONE, load_selection_template, selection_enabled)
from sdf_pipeline.shared_step_machinery.update_steps import Accepted, CorpusStep
from shared import utils


class ConsiderationSelectionStep(CorpusStep):
    """The global linear programme over every scored document (no model call). The documents are
    score rows (``prompt_id``, ``scores``); each gets an ``Accepted`` outcome whose update is its
    selection outcomes. The programme is solved again over every document each time it runs, as it
    always was, so a plan that arrives late changes other documents' outcomes. After ``apply``, ``selection_rows``, ``flat_scores`` and
    ``assigned_labels_by_document_id`` hold what the layer writes and reports."""

    step_name = "consideration_selection"

    def __init__(self, config: dict, catalog: ConsiderationCatalog, eligible: list[dict]):
        self.catalog = catalog
        self.seed = config["sdf"].get("seed", 0)
        self.limits = assignment.limits_from_config(config)
        self.document_is_digital = digital_mind_flags(catalog, eligible)
        self.selection_rows: list[dict] = []
        self.flat_scores: dict = {}
        self.assigned_labels_by_document_id: dict = {}

    def apply(self, score_rows: list) -> dict:
        rows_by_document_id = {row["prompt_id"]: row for row in score_rows}
        self.flat_scores = assignment.flat_scores(rows_by_document_id)
        self.assigned_labels_by_document_id = assignment.assign_considerations(
            self.flat_scores, self.catalog, self.document_is_digital, self.limits, self.seed)
        self.selection_rows = _selection_rows(self.catalog, rows_by_document_id, self.assigned_labels_by_document_id)
        return {row["prompt_id"]: Accepted({"outcomes": row["outcomes"]}) for row in self.selection_rows}

    def coverage_report(self) -> dict:
        return assignment.coverage_report(
            self.flat_scores, self.catalog, self.document_is_digital, self.assigned_labels_by_document_id, self.limits)




def digital_mind_flags(catalog: ConsiderationCatalog, eligible: list[dict]) -> dict[str, bool]:
    return {p["prompt_id"]: catalog.is_digital_mind_document(p["variables"]) for p in eligible}


def assemble_considerations_spec(picks: list) -> str:
    """The text block handed to later prompts: one line per pick, 'Label (gloss) —
    the selector's sentence on where it arises', moral picks first and each list
    in catalog order. Empty string when there are no picks."""
    return "\n".join(
        f"{entry.label} ({entry.gloss}) — {reason}" if entry.gloss else f"{entry.label} — {reason}"
        for entry, reason in picks
    )




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
    """Plans with the saved selections attached, without any calls. Used when a run
    resumes past layer 1 and plans are re-read from disk. A disabled selector, a
    snapshot without the selector files, or a run with no saved selections returns
    ``plans`` unchanged."""
    if not selection_enabled(config):
        return plans
    if load_selection_template(prompts_dir) is None or not (Path(prompts_dir) / CONSIDERATIONS_FILENAME).is_file():
        return plans
    rows = utils.load_jsonl(Path(output_dir) / SELECTIONS_FILENAME)
    if not rows:
        return plans
    return attach_selection_rows(load_considerations(prompts_dir), rows, plans)
