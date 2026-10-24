"""Assign considerations to documents from the model's scores: pure code, no API calls.

The selector (layer1_select_considerations.py) has the model score every eligible
(document, entry) pair from 1 to 5 and nothing else. This module then decides who
gets what, in code, with one linear programme over both lists:

- Allowed pairs: a (document, entry) pair whose score is at least
  ``minimum_assignable_score`` and whose entry is eligible for the document
  (digital-minds-only entries only on digital-mind documents).
- Hard ceilings: each entry goes to at most its ``target_count`` documents; each
  document gets at most ``max_moral`` moral entries, at most ``max_epistemic``
  epistemic entries, and at most ``max_total`` entries across both lists.
- Objective (one stage), maximise
      sum over assigned pairs of score_weight[score] + tie_break(pair)
      - sum over (document, list) of penalty(n),   penalty(n) = p * n * (n - 1) / 2
  where p is ``penalty_step_per_further_consideration_in_a_list`` and n is how many
  considerations the document gets from that list: the k-th
  consideration from a list (k >= 2) costs (k - 1) * p, so a second one is worth it
  only when its weight exceeds p, a third only when it exceeds 2p, and so on, and
  slots whose only candidates would be weak extras stay unfilled. The first
  consideration from each list is free, so a moral and an epistemic entry on the
  same document do not penalise each other. The convex penalty is modelled with
  extra columns y[d, l, k] in [0, 1], k = 1..cap-1, each costing k * p, and the row
  sum_e x[d, e] - sum_k y[d, l, k] <= 1; costs rise with k so the LP fills the
  cheap y first and the penalty comes out as exactly p * n(n-1)/2.
- Tie-break: each pair gets a small value from a sha256 hash of (seed, prompt_id, label),
  so the result does not depend on the order of documents or keys, and adding an
  unrelated document leaves the choice between two other tied documents unchanged.
  See ``_tie_break_values`` for the scale.

Every constraint row is a laminar family over x (entries; per-document totals; per
(document, list) caps) plus the penalty rows, whose extra columns each have a single
-1 entry, so the matrix stays totally unimodular and the optimum is integral; the
solution is still rounded, checked for integrality and checked against every
constraint and the penalty formula. Solved with scipy.optimize.linprog (HiGHS) on
sparse matrices.

``assign_considerations`` takes the limits as arguments so a saved scores file can
be re-assigned offline with different thresholds, caps or targets
(``assign_from_scores_file``).
"""

import hashlib
import json
import sys
from dataclasses import dataclass, field
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
DEFAULT_MAX_MORAL_CONSIDERATIONS = 3
DEFAULT_MAX_EPISTEMIC_CONSIDERATIONS = 3
DEFAULT_MAX_CONSIDERATIONS = 4
DEFAULT_SCORE_WEIGHTS = {3: 1, 4: 3, 5: 9}
DEFAULT_PENALTY_STEP_PER_FURTHER_CONSIDERATION_IN_A_LIST = 2
INTEGRALITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class AssignmentLimits:
    minimum_assignable_score: int = DEFAULT_MINIMUM_ASSIGNABLE_SCORE
    max_moral_per_document: int = DEFAULT_MAX_MORAL_CONSIDERATIONS
    max_epistemic_per_document: int = DEFAULT_MAX_EPISTEMIC_CONSIDERATIONS
    max_total_per_document: int = DEFAULT_MAX_CONSIDERATIONS
    score_weights: dict = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    penalty_step_per_further_consideration_in_a_list: float = DEFAULT_PENALTY_STEP_PER_FURTHER_CONSIDERATION_IN_A_LIST

    def __post_init__(self):
        if self.penalty_step_per_further_consideration_in_a_list < 0:
            raise ValueError("penalty_step_per_further_consideration_in_a_list must not be negative")
        for score in SCORE_VALUES:
            if score >= self.minimum_assignable_score and not self.score_weights.get(score, 0) > 0:
                raise ValueError(f"score_weights needs a positive weight for every assignable score; missing {score}")

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
        score_weights={int(score): weight for score, weight in
                       (sdf.get("score_weights") or DEFAULT_SCORE_WEIGHTS).items()},
        penalty_step_per_further_consideration_in_a_list=sdf.get("penalty_step_per_further_consideration_in_a_list", DEFAULT_PENALTY_STEP_PER_FURTHER_CONSIDERATION_IN_A_LIST),
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


def _tie_break_values(pairs, seed, limits) -> np.ndarray:
    """One value per pair from sha256(seed, prompt_id, label), a fraction in [0, 1), so it
    depends only on the pair and the seed, never on order or on which other documents exist.

    Scale: the values sum to less than ``scale``, and ``scale`` must be below the smallest
    nonzero difference between two values of the rest of the objective, so the tie-break
    only ever separates solutions that are exactly tied. With integer weights and an
    integer penalty the rest of the objective is an integer, so scale = 1 suffices
    (each value is multiplied by 1 / (pairs + 1)). If any weight or the penalty is not an
    integer there is no simple bound, so a much smaller scale of 1e-6 is used instead.

    The values are also shifted down by their common upper bound, so each lies in
    (-bound, 0] with bound = scale / (pairs + 1): a pair whose weight exactly equals the
    penalty it would incur nets slightly below zero and stays unassigned, which makes
    "assigned only when the weight exceeds the penalty" strictly true. The shifts sum to
    less than ``scale`` in magnitude, so they cannot outweigh a real difference."""
    parameters = [*limits.score_weights.values(), limits.penalty_step_per_further_consideration_in_a_list]
    scale = 1.0 if all(float(value).is_integer() for value in parameters) else 1e-6
    values = []
    for prompt_id, label, _ in pairs:
        digest = hashlib.sha256(f"{seed}|{prompt_id}|{label}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") / 2**64)
    bound = scale / (len(pairs) + 1)
    return (np.array(values) - 1.0) * bound


def _model_columns_and_rows(pairs, catalog, limits, targets):
    """Columns: one x per pair, then the penalty columns y[d, l, k] (each in [0, 1], cost k * p).
    Rows of A x <= b: per entry (its target), per document (the total cap), per (document, list)
    the hard cap, and per (document, list) with a y column the penalty row
    sum_e x - sum_k y <= 1. Returns (matrix, bounds, y_costs) with y_costs[i] the cost
    multiplier k of penalty column i."""
    entry_by_label = {entry.label: entry for entry in catalog.all_entries}
    row_of: dict[tuple, int] = {}
    bounds: list[float] = []
    rows, columns, values = [], [], []

    def row_for(key, bound):
        if key not in row_of:
            row_of[key] = len(bounds)
            bounds.append(bound)
        return row_of[key]

    pairs_in_document_list: dict[tuple, int] = {}
    for column, (prompt_id, label, _) in enumerate(pairs):
        list_name = entry_by_label[label].list_name
        pairs_in_document_list[(prompt_id, list_name)] = pairs_in_document_list.get((prompt_id, list_name), 0) + 1
        for key, bound in (
            (("entry", label), targets[label]),
            (("document_list", prompt_id, list_name), limits.max_for_list(list_name)),
            (("document", prompt_id), limits.max_total_per_document),
        ):
            rows.append(row_for(key, bound))
            columns.append(column)
            values.append(1.0)
        if limits.penalty_step_per_further_consideration_in_a_list > 0:
            rows.append(row_for(("penalty", prompt_id, list_name), 1.0))
            columns.append(column)
            values.append(1.0)

    penalty_multipliers: list[int] = []
    if limits.penalty_step_per_further_consideration_in_a_list > 0:
        for (prompt_id, list_name), pair_count in pairs_in_document_list.items():
            extra_slots = min(limits.max_for_list(list_name), limits.max_total_per_document, pair_count) - 1
            for multiplier in range(1, extra_slots + 1):
                rows.append(row_of[("penalty", prompt_id, list_name)])
                columns.append(len(pairs) + len(penalty_multipliers))
                values.append(-1.0)
                penalty_multipliers.append(multiplier)
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(len(bounds), len(pairs) + len(penalty_multipliers))).tocsr()
    return matrix, np.array(bounds, dtype=float), np.array(penalty_multipliers, dtype=float)


def _solve(objective, matrix, bounds):
    result = linprog(objective, A_ub=matrix, b_ub=bounds, bounds=(0, 1), method="highs")
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
        matrix, bounds, penalty_multipliers = _model_columns_and_rows(pairs, catalog, limits, targets)
        gains = np.array([limits.score_weights[score] for _, _, score in pairs], dtype=float)
        gains = gains + _tie_break_values(pairs, seed, limits)
        penalty_costs = penalty_multipliers * limits.penalty_step_per_further_consideration_in_a_list
        solution = _solve(-np.concatenate([gains, -penalty_costs]), matrix, bounds)
        chosen = solution[:len(pairs)]
        _check_constraints(solution, pairs, catalog, matrix, bounds, limits, penalty_costs)
        order = {entry.label: index for index, entry in enumerate(catalog.all_entries)}
        for column in np.flatnonzero(chosen):
            prompt_id, label, _ = pairs[column]
            assigned[prompt_id].append(label)
        for labels in assigned.values():
            labels.sort(key=order.__getitem__)
    return assigned


def _check_constraints(solution, pairs, catalog, matrix, bounds, limits, penalty_costs) -> None:
    assert set(np.unique(solution)) <= {0, 1}
    assert np.all(matrix @ solution <= bounds + INTEGRALITY_TOLERANCE), "an assignment cap was exceeded"
    chosen = solution[:len(pairs)]
    list_of = {entry.label: entry.list_name for entry in catalog.all_entries}
    counts: dict[tuple, int] = {}
    for column in np.flatnonzero(chosen):
        prompt_id, label, _ = pairs[column]
        key = (prompt_id, list_of[label])
        counts[key] = counts.get(key, 0) + 1
    direct_penalty = limits.penalty_step_per_further_consideration_in_a_list * sum(n * (n - 1) / 2 for n in counts.values())
    model_penalty = float(penalty_costs @ solution[len(pairs):]) if len(penalty_costs) else 0.0
    assert abs(direct_penalty - model_penalty) < 1e-6 * max(1.0, direct_penalty), \
        "the LP's penalty differs from p * n(n-1)/2 computed from the chosen pairs"


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
        "score_weights": {str(score): weight for score, weight in sorted(limits.score_weights.items())},
        "penalty_step_per_further_consideration_in_a_list": limits.penalty_step_per_further_consideration_in_a_list,
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
