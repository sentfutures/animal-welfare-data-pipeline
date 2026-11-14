"""Layer 3 (consideration scoring and selection): the model scores every eligible entry for each document, code assigns.

The layer is two steps, one module each in ``sdf_pipeline/steps/``: ``ConsiderationScoringStep``
(``consideration_scoring.py``, one model call per eligible document) and ``ConsiderationSelectionStep``
(``consideration_selection.py``, a linear programme over all scores, no model call). This module holds the layer's
driver, ``run``: it runs the scoring step, then the selection, and returns the plans with their selections
attached. The layer's file names and its readers (used by the audit and the report) are in
``layer3_consideration_scoring_and_selection_shared.py``.

Files in the layer's folder (``layer3_consideration_scoring_and_selection``):
- consideration_scores.jsonl: one row per scored document, appended as each answer arrives; it is the checkpoint.
- consideration_scoring_failures.jsonl: documents whose scoring answer stayed unusable after
  ``sdf.max_scoring_attempts`` tries; the layer stops before selecting (the selection is global, so it only runs
  once every eligible document is scored).
- consideration_selections.jsonl: the selection, rewritten after every run, one row per scored document
  ({prompt_id, outcomes: [{list, label, reason, status, score}]}); this is what ``attach_selections`` reads when a
  run resumes past this layer.
- consideration_coverage.json: per entry target_count, achieved, documents scoring at or above the threshold and
  score histogram, plus documents with no consideration.
"""

import json
from pathlib import Path

from sdf_pipeline.considerations_catalog import load_considerations
from sdf_pipeline.layer3_consideration_scoring_and_selection_shared import *  # noqa: F401,F403  (names, readers)
from sdf_pipeline.layer3_consideration_scoring_and_selection_shared import (
    COVERAGE_FILENAME, SCORES_FILENAME, SELECTIONS_FILENAME, eligible_plans, load_selection_template,
    print_coverage_report, reason_consideration_selection_is_inactive)
from sdf_pipeline.steps.consideration_scoring import *  # noqa: F401,F403  (the scoring prompt and parsing helpers)
from sdf_pipeline.steps.consideration_scoring import ConsiderationScoringStep
from sdf_pipeline.steps.consideration_selection import *  # noqa: F401,F403
from sdf_pipeline.steps.consideration_selection import ConsiderationSelectionStep, attach_selection_rows, attach_selections  # noqa: F401  (names the old layer 1 module had)
from sdf_pipeline.step_runner import StepRunner, run_corpus_step
from shared import utils


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Score every eligible document, assign considerations in code, and return ``plans``
    with the assigned entries attached. A disabled selector, or a run whose prompt
    snapshot lacks the template or the considerations file, makes no calls and returns
    ``plans`` unchanged."""
    reason_inactive = reason_consideration_selection_is_inactive(config, prompts_dir)
    if reason_inactive is not None:
        print(f"  {reason_inactive}")
        return plans
    template = load_selection_template(prompts_dir)

    catalog = load_considerations(prompts_dir)
    workers = max(1, int(config.get("workers", 1)))
    utils.ensure_dir(output_dir)
    scores_path = output_dir / SCORES_FILENAME

    eligible = eligible_plans(config, catalog, plans)
    plan_by_id = {p["prompt_id"]: p for p in eligible}
    step = ConsiderationScoringStep(config, catalog, template)
    # Scoring has no checkpoint file: ``consideration_scores.jsonl`` is the checkpoint.
    runner = StepRunner(step, output_dir, workers=workers, uses_checkpoint=False)
    documents_in_scoring_order = [plan_by_id[prompt_id] for prompt_id in sorted(plan_by_id)]
    pending = runner.select_pending(documents_in_scoring_order)
    print(f"  Scoring considerations for {len(eligible)} specifications: "
          f"{len(eligible) - len(pending)} already scored, {len(pending)} to score")
    runner.run(documents_in_scoring_order)  # stops the run before the assignment if any document failed

    score_rows = {r["prompt_id"]: r for r in utils.load_jsonl(scores_path) if r["prompt_id"] in plan_by_id}
    assignment_step = ConsiderationSelectionStep(config, catalog, eligible)
    run_corpus_step(assignment_step, list(score_rows.values()))
    selection_rows = assignment_step.selection_rows
    (output_dir / SELECTIONS_FILENAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection_rows), encoding="utf-8")
    report = assignment_step.coverage_report()
    (output_dir / COVERAGE_FILENAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_coverage_report(report)
    return attach_selection_rows(catalog, selection_rows, plans)
