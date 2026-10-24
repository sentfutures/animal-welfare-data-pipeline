"""Assign considerations to documents from the model's scores: pure code, no API calls.

The selector (layer1_select_considerations.py) has the model score every eligible
(document, entry) pair from 1 to 5 and nothing else. This module then decides who
gets what, in code, with one linear programme over both lists:

- Allowed pairs: a (document, entry) pair whose score is at least
  ``minimum_assignable_score`` and whose entry is eligible for the document
  (digital-minds-only entries only on digital-mind documents).
- Constraints: each entry goes to at most its ``target_count`` documents; each
  document gets at most ``max_moral`` moral entries, at most ``max_epistemic``
  epistemic entries, and at most ``max_total`` entries across both lists.
- Objective, lexicographic: stage 1 maximises the number of assignments; stage 2
  fixes that number as a constraint and maximises total score plus a tiny seeded
  random tie-break (summing to less than one point over the whole problem, so it
  only ever separates solutions of equal total score).

The constraint matrix is a union of two laminar families (entries, and per-document
list and total caps), hence totally unimodular, so the optimum is integral; the
solution is still rounded, checked for integrality and checked against every
constraint. Solved with scipy.optimize.linprog (HiGHS) on sparse matrices.

``assign_considerations`` takes the limits as arguments so a saved scores file can
be re-assigned offline with different thresholds, caps or targets
(``assign_from_scores_file``).
"""

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdf_pipeline.considerations_catalog import (  # noqa: E402
    LIST_NAMES, MORAL_LIST, ConsiderationCatalog,
)
from sdf_pipeline.considerations_coverage import document_coverage_shares, entry_targets  # noqa: E402

SCORE_VALUES = (1, 2, 3, 4, 5)
DEFAULT_MINIMUM_ASSIGNABLE_SCORE = 3
DEFAULT_MAX_MORAL_CONSIDERATIONS = 1
DEFAULT_MAX_EPISTEMIC_CONSIDERATIONS = 1
DEFAULT_MAX_CONSIDERATIONS = 2
INTEGRALITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class AssignmentLimits:
    minimum_assignable_score: int = DEFAULT_MINIMUM_ASSIGNABLE_SCORE
    max_moral_per_document: int = DEFAULT_MAX_MORAL_CONSIDERATIONS
    max_epistemic_per_document: int = DEFAULT_MAX_EPISTEMIC_CONSIDERATIONS
    max_total_per_document: int = DEFAULT_MAX_CONSIDERATIONS

    def max_for_list(self, list_name: str) -> int:
        return self.max_moral_per_document if list_name == MORAL_LIST else self.max_epistemic_per_document


def limits_from_config(config: dict) -> AssignmentLimits:
    sdf = config["sdf"]
    return AssignmentLimits(
        minimum_assignable_score=int(sdf.get("minimum_assignable_score", DEFAULT_MINIMUM_ASSIGNABLE_SCORE)),
        max_moral_per_document=int(sdf.get("max_moral_considerations_per_document", DEFAULT_MAX_MORAL_CONSIDERATIONS)),
        max_epistemic_per_document=int(
            sdf.get("max_epistemic_considerations_per_document", DEFAULT_MAX_EPISTEMIC_CONSIDERATIONS)),
        max_total_per_document=int(sdf.get("max_considerations_per_document", DEFAULT_MAX_CONSIDERATIONS)),
    )


def _allowed_pairs(
    scores: dict[str, dict[str, int]], catalog: ConsiderationCatalog,
    document_is_digital: dict[str, bool], limits: AssignmentLimits,
) -> list[tuple[str, str, int]]:
    """(prompt_id, label, score) in a deterministic order (sorted ids, catalog order)."""
    pairs = []
    for prompt_id in sorted(scores):
        if prompt_id not in document_is_digital:
            continue
        for entry in catalog.all_entries:
            score = scores[prompt_id].get(entry.label)
            if score is None or score < limits.minimum_assignable_score:
                continue
            if entry.digital_minds_only and not document_is_digital[prompt_id]:
                continue
            if limits.max_for_list(entry.list_name) <= 0:
                continue
            pairs.append((prompt_id, entry.label, int(score)))
    return pairs


def _constraint_matrix(pairs, catalog, limits, targets):
    """Sparse A and b for A x <= b: one row per entry (its target), per (document, list)
    (that list's cap) and per document (the total cap)."""
    entry_by_label = {entry.label: entry for entry in catalog.all_entries}
    row_of: dict[tuple, int] = {}
    bounds: list[int] = []
    rows, columns = [], []

    def row_for(key, bound):
        if key not in row_of:
            row_of[key] = len(bounds)
            bounds.append(bound)
        return row_of[key]

    for column, (prompt_id, label, _) in enumerate(pairs):
        list_name = entry_by_label[label].list_name
        for key, bound in (
            (("entry", label), targets[label]),
            (("document_list", prompt_id, list_name), limits.max_for_list(list_name)),
            (("document", prompt_id), limits.max_total_per_document),
        ):
            rows.append(row_for(key, bound))
            columns.append(column)
    matrix = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(bounds), len(pairs))).tocsr()
    return matrix, np.array(bounds, dtype=float)


def _solve(objective, matrix, bounds, count_equals=None):
    options = {"A_ub": matrix, "b_ub": bounds, "bounds": (0, 1), "method": "highs"}
    if count_equals is not None:
        options["A_eq"] = coo_matrix(np.ones((1, len(objective)))).tocsr()
        options["b_eq"] = np.array([float(count_equals)])
    result = linprog(objective, **options)
    if result.status != 0:
        raise RuntimeError(f"considerations assignment LP failed: {result.message}")
    chosen = np.rint(result.x)
    if np.max(np.abs(result.x - chosen)) >= INTEGRALITY_TOLERANCE:
        raise AssertionError("considerations assignment LP returned a fractional solution")
    return chosen.astype(int)


def assign_considerations(
    scores: dict[str, dict[str, int]],
    catalog: ConsiderationCatalog,
    document_is_digital: dict[str, bool],
    limits: AssignmentLimits = AssignmentLimits(),
    seed=0,
    targets: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """{prompt_id: [assigned labels]} (moral entries first, each list in catalog order),
    with an empty list for documents that get nothing.

    ``scores``: {prompt_id: {label: score 1-5}} for the pairs the model scored.
    ``document_is_digital``: every eligible document, scored or not (it fixes the
    target counts, which are shares of the eligible documents). ``targets``
    overrides the catalog-derived target counts."""
    if targets is None:
        targets = entry_targets(catalog, len(document_is_digital), sum(document_is_digital.values()))
    assigned: dict[str, list[str]] = {prompt_id: [] for prompt_id in document_is_digital}
    pairs = _allowed_pairs(scores, catalog, document_is_digital, limits)
    if pairs:
        matrix, bounds = _constraint_matrix(pairs, catalog, limits, targets)
        stage_one = _solve(-np.ones(len(pairs)), matrix, bounds)
        maximum_count = int(stage_one.sum())
        if maximum_count > 0:
            rng = random.Random(f"{seed}:considerations_assignment")
            tie_break = np.array([rng.random() / (len(pairs) + 1) for _ in pairs])   # sums to < 1
            score_values = np.array([score for _, _, score in pairs], dtype=float)
            chosen = _solve(-(score_values + tie_break), matrix, bounds, count_equals=maximum_count)
            _check_constraints(chosen, pairs, matrix, bounds, maximum_count)
            order = {entry.label: index for index, entry in enumerate(catalog.all_entries)}
            for column in np.flatnonzero(chosen):
                prompt_id, label, _ = pairs[column]
                assigned[prompt_id].append(label)
            for labels in assigned.values():
                labels.sort(key=order.__getitem__)
    return assigned


def _check_constraints(chosen, pairs, matrix, bounds, expected_count) -> None:
    assert chosen.sum() == expected_count, "stage 2 changed the number of assignments"
    assert np.all(matrix @ chosen <= bounds + INTEGRALITY_TOLERANCE), "an assignment cap was exceeded"
    assert set(np.unique(chosen)) <= {0, 1}


def coverage_report(
    scores: dict[str, dict[str, int]],
    catalog: ConsiderationCatalog,
    document_is_digital: dict[str, bool],
    assigned: dict[str, list[str]],
    limits: AssignmentLimits,
    targets: dict[str, int] | None = None,
) -> dict:
    """Per entry: target_count, achieved, documents scoring at least the threshold, and the
    score histogram; plus how documents ended up covered and the documents with no consideration."""
    eligible = len(document_is_digital)
    digital = sum(document_is_digital.values())
    if targets is None:
        targets = entry_targets(catalog, eligible, digital)
    achieved = {entry.label: 0 for entry in catalog.all_entries}
    for labels in assigned.values():
        for label in labels:
            achieved[label] += 1
    entries = []
    for entry in catalog.all_entries:
        histogram = {str(value): 0 for value in SCORE_VALUES}
        for per_document in scores.values():
            if entry.label in per_document:
                histogram[str(per_document[entry.label])] += 1
        entries.append({
            "label": entry.label,
            "list": entry.list_name,
            "digital_minds_only": entry.digital_minds_only,
            "target_share": entry.target_share,
            "eligible_documents": digital if entry.digital_minds_only else eligible,
            "target_count": targets[entry.label],
            "achieved": achieved[entry.label],
            "documents_scoring_at_least_threshold": sum(
                count for value, count in histogram.items() if int(value) >= limits.minimum_assignable_score),
            "score_histogram": histogram,
        })
    per_document_counts = {
        prompt_id: {name: 0 for name in LIST_NAMES} for prompt_id in document_is_digital}
    list_of = {entry.label: entry.list_name for entry in catalog.all_entries}
    for prompt_id, labels in assigned.items():
        for label in labels:
            per_document_counts[prompt_id][list_of[label]] += 1
    return {
        "eligible_documents": eligible,
        "digital_mind_documents": digital,
        "scored_documents": len(scores),
        "minimum_assignable_score": limits.minimum_assignable_score,
        "max_moral_considerations_per_document": limits.max_moral_per_document,
        "max_epistemic_considerations_per_document": limits.max_epistemic_per_document,
        "max_considerations_per_document": limits.max_total_per_document,
        "entries": entries,
        **document_coverage_shares(list(per_document_counts.values())),
        "document_ids_with_no_consideration": sorted(
            prompt_id for prompt_id, labels in assigned.items() if not labels),
    }


def load_scores_file(path: Path) -> dict[str, dict]:
    """{prompt_id: row} from a consideration_scores.jsonl (the last row of a repeated id wins)."""
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["prompt_id"]] = row
    return rows


def flat_scores(rows: dict[str, dict]) -> dict[str, dict[str, int]]:
    """{prompt_id: {label: score}} from scores-file rows."""
    return {
        prompt_id: {
            label: item["score"]
            for list_scores in row["scores"].values() for label, item in list_scores.items()}
        for prompt_id, row in rows.items()
    }


def assign_from_scores_file(
    scores_path: Path, catalog: ConsiderationCatalog, document_is_digital: dict[str, bool],
    limits: AssignmentLimits, seed=0, targets: dict[str, int] | None = None,
) -> tuple[dict[str, list[str]], dict]:
    """Re-run the assignment offline on a saved scores file; returns (assigned, coverage report).
    Documents in the file but not in ``document_is_digital`` are ignored."""
    scores = {pid: s for pid, s in flat_scores(load_scores_file(scores_path)).items() if pid in document_is_digital}
    assigned = assign_considerations(scores, catalog, document_is_digital, limits, seed, targets)
    return assigned, coverage_report(scores, catalog, document_is_digital, assigned, limits, targets)
