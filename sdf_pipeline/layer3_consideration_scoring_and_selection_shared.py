"""Layer 3 (consideration scoring and selection): what the scoring step, the selection step and the layer's
driver share, and the readers of the layer's files used by the audit and the report.

The file names of the layer, the template, whether the layer is enabled and why it is inactive, which plans are
eligible for considerations, and the coverage report printer and loaders. See
``layer3_consideration_scoring_and_selection.py`` for the layer as a whole.
"""

import json
from pathlib import Path

from sdf_pipeline import structured_answers
from sdf_pipeline.considerations_catalog import CONSIDERATIONS_FILENAME, ConsiderationCatalog, load_considerations
from sdf_pipeline.considerations_coverage import corpus_coverage
from shared import sdf_run_layouts, utils

TEMPLATE_FILENAME = "layer1_select_considerations.txt"
SCORES_FILENAME = "consideration_scores.jsonl"
FAILURES_FILENAME = "consideration_scoring_failures.jsonl"
SELECTIONS_FILENAME = "consideration_selections.jsonl"
COVERAGE_FILENAME = "consideration_coverage.json"
STATUS_ACCEPTED = "accepted"
STATUS_NONE = "none"


def load_selection_template(prompts_dir: Path) -> str | None:
    """The selector template, or None for a run whose snapshot predates the selector."""
    return structured_answers.read_template_if_present(prompts_dir, TEMPLATE_FILENAME)


def selection_enabled(config: dict) -> bool:
    return bool(config["sdf"].get("select_enabled", True))


def eligible_plans(config: dict, catalog: ConsiderationCatalog, plans: list[dict]) -> list[dict]:
    """Plans with a description, minus the skipped (no-welfare-stake) resolutions."""
    skip_no_stake = bool(config["sdf"].get("select_skip_no_stake", True))
    return [
        p for p in plans
        if p.get("description") and not (skip_no_stake and catalog.is_skipped_resolution(p["variables"]))
    ]


def reason_consideration_selection_is_inactive(config: dict, prompts_dir: Path) -> str | None:
    """Why the consideration scoring and assignment do not run, as the line ``run`` prints; None when they run."""
    if not selection_enabled(config):
        return "Consideration selection disabled (sdf.select_enabled: false)"
    if load_selection_template(prompts_dir) is None or not (Path(prompts_dir) / CONSIDERATIONS_FILENAME).is_file():
        return (f"No {TEMPLATE_FILENAME} or {CONSIDERATIONS_FILENAME} in this run's prompt snapshot "
                "— consideration selection skipped")
    return None


def print_coverage_report(report: dict) -> None:
    print(f"  Considerations coverage over {report['eligible_documents']} eligible documents "
          f"({report['digital_mind_documents']} digital-mind):")
    for entry in report["entries"]:
        scoring = entry.get("documents_scoring_at_least_threshold")
        suffix = f", {scoring} documents scored at or above the threshold" if scoring is not None else ""
        print(f"    [{entry['list']}] {entry['label']}: {entry['achieved']}/{entry['target_count']}{suffix}")
    total = report["eligible_documents"]
    print(f"    no consideration: {report['documents_with_no_consideration']}/{total}, "
          f"one list only: {report['documents_with_only_one_list']}/{total}, "
          f"both lists: {report['documents_with_both_lists']}/{total}, "
          f"two or more from one list: {report['documents_with_two_or_more_in_one_list']}")


def load_considerations_outputs(run_dir: Path) -> dict | None:
    """The selector's results for a run, or None when it never ran (older runs).

    Returns {"catalog", "rows", "report"}: the catalog loaded from the run's own
    prompt snapshot, the saved selection rows, and the coverage report written at
    the end of the layer (None if the stage never finished)."""
    stage_dir = utils.sdf_stage_folder(run_dir, sdf_run_layouts.CONSIDERATION_SCORING_AND_SELECTION)
    rows = utils.load_jsonl(stage_dir / SELECTIONS_FILENAME)
    snapshot = Path(run_dir) / "inputs" / "prompts"
    if not rows or not (snapshot / CONSIDERATIONS_FILENAME).is_file():
        return None
    report_path = stage_dir / COVERAGE_FILENAME
    return {
        "catalog": load_considerations(snapshot),
        "rows": rows,
        "report": json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None,
    }


def corpus_coverage_report(outputs: dict, corpus: list[dict]) -> dict:
    """The coverage report measured on the final corpus documents."""
    return corpus_coverage(outputs["catalog"], [r["variables"] for r in corpus])
