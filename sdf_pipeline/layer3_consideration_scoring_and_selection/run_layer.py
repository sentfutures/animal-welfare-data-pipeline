"""Layer 3: the driver ``run`` scores every eligible document (``consideration_scoring.py``), then selects (``consideration_selection.py``), returning the plans with selections attached.
The selection is global, so it runs only once every eligible document is scored (failures stop the layer). Files: consideration_scores, _scoring_failures, _selections (jsonl) and consideration_coverage.json."""

import json
from pathlib import Path

from sdf_pipeline.shared_step_machinery.considerations_catalog import CONSIDERATIONS_FILENAME, load_considerations
from sdf_pipeline.layer3_consideration_scoring_and_selection.shared_by_this_layers_steps import *  # noqa: F401,F403  (names, readers)
from sdf_pipeline.layer3_consideration_scoring_and_selection.shared_by_this_layers_steps import (
    COVERAGE_FILENAME, SCORES_FILENAME, SELECTIONS_FILENAME, TEMPLATE_FILENAME, eligible_plans, load_selection_template,
    print_coverage_report, selection_enabled)
from sdf_pipeline.layer3_consideration_scoring_and_selection.consideration_scoring import *  # noqa: F401,F403  (the scoring prompt and parsing helpers)
from sdf_pipeline.layer3_consideration_scoring_and_selection import consideration_scoring
from sdf_pipeline.layer3_consideration_scoring_and_selection.consideration_selection import *  # noqa: F401,F403
from sdf_pipeline.layer3_consideration_scoring_and_selection.consideration_selection import (  # noqa: F401
    assign_considerations_to_documents, attach_selection_rows, attach_selections)
from sdf_pipeline.shared_step_machinery.step_record import Run
from sdf_pipeline.shared_step_machinery.step_runner import find_pending, run_step
from shared import utils
from shared.sdf_run_layouts import CONSIDERATION_SCORING_AND_SELECTION


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Score, assign and return ``plans`` with the entries attached; no calls and ``plans`` unchanged when disabled or the snapshot lacks the template or catalogue."""
    if not selection_enabled(config):
        print("  Consideration selection disabled (sdf.select_enabled: false)")
        return plans
    if load_selection_template(prompts_dir) is None or not (Path(prompts_dir) / CONSIDERATIONS_FILENAME).is_file():
        print(f"  No {TEMPLATE_FILENAME} or {CONSIDERATIONS_FILENAME} in this run's prompt snapshot "
              "— consideration selection skipped")
        return plans
    catalog = load_considerations(prompts_dir)
    utils.ensure_dir(output_dir)
    scores_path = output_dir / SCORES_FILENAME

    run_of_scoring = Run(config, prompts_dir, {CONSIDERATION_SCORING_AND_SELECTION: output_dir})
    eligible = eligible_plans(config, catalog, plans)
    plan_by_id = {p["prompt_id"]: p for p in eligible}
    documents_in_scoring_order = [plan_by_id[prompt_id] for prompt_id in sorted(plan_by_id)]
    _, pending = find_pending(consideration_scoring.STEP, run_of_scoring, documents_in_scoring_order)
    print(f"  Scoring considerations for {len(eligible)} specifications: "
          f"{len(eligible) - len(pending)} already scored, {len(pending)} to score")
    run_step(consideration_scoring.STEP, run_of_scoring, documents_in_scoring_order)  # stops the run before the assignment if any document failed

    score_rows = [r for r in utils.load_jsonl(scores_path) if r["prompt_id"] in plan_by_id]
    selection_rows, report = assign_considerations_to_documents(config, catalog, eligible, score_rows)
    (output_dir / SELECTIONS_FILENAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection_rows), encoding="utf-8")
    (output_dir / COVERAGE_FILENAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_coverage_report(report)
    return attach_selection_rows(catalog, selection_rows, plans)
