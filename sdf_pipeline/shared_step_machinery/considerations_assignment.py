"""Assign considerations to documents from the model's scores: pure code, one linear programme over both lists.
Rules an editor must keep (details in ``sdf_pipeline/CLAUDE.md``): the objective, the penalty, the tie-break and total unimodularity."""

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sdf_pipeline.shared_step_machinery.considerations_catalog import (  # noqa: E402
    LIST_NAMES, MORAL_LIST, ConsiderationCatalog,
)
from sdf_pipeline.shared_step_machinery.considerations_coverage import document_coverage_shares, entry_targets  # noqa: E402

SCORE_VALUES = (1, 2, 3, 4, 5)
DEFAULT_MINIMUM_ASSIGNABLE_SCORE = 3
DEFAULT_MAX_MORAL_CONSIDERATIONS = 3
DEFAULT_MAX_EPISTEMIC_CONSIDERATIONS = 3
DEFAULT_MAX_CONSIDERATIONS = 4
DEFAULT_SCORE_WEIGHTS = {3: 1, 4: 3, 5: 9}
DEFAULT_EXTRA_CONSIDERATION_PENALTY = 2
INTEGRALITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class AssignmentLimits:
    minimum_assignable_score: int = DEFAULT_MINIMUM_ASSIGNABLE_SCORE
    max_moral_per_document: int = DEFAULT_MAX_MORAL_CONSIDERATIONS
    max_epistemic_per_document: int = DEFAULT_MAX_EPISTEMIC_CONSIDERATIONS
    max_total_per_document: int = DEFAULT_MAX_CONSIDERATIONS
    score_weights: dict = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    extra_consideration_penalty: float = DEFAULT_EXTRA_CONSIDERATION_PENALTY

    def __post_init__(self):
        if self.extra_consideration_penalty < 0:
            raise ValueError("extra_consideration_penalty must not be negative")
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
        extra_consideration_penalty=sdf.get("extra_consideration_penalty", DEFAULT_EXTRA_CONSIDERATION_PENALTY),
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
    """A hash-based fraction in [0, 1) per pair, shifted below zero and scaled so it only separates exactly tied solutions."""
    parameters = [*limits.score_weights.values(), limits.extra_consideration_penalty]
    scale = 1.0 if all(float(value).is_integer() for value in parameters) else 1e-6
    values = []
    for prompt_id, label, _ in pairs:
        digest = hashlib.sha256(f"{seed}|{prompt_id}|{label}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") / 2**64)
    bound = scale / (len(pairs) + 1)
    return (np.array(values) - 1.0) * bound


def _model_columns_and_rows(pairs, catalog, limits, targets):
    """The LP's columns (x per pair, penalty y[d, l, k] at cost k * p) and rows; returns (matrix, bounds, y_costs)."""
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
        if limits.extra_consideration_penalty > 0:
            rows.append(row_for(("penalty", prompt_id, list_name), 1.0))
            columns.append(column)
            values.append(1.0)

    penalty_multipliers: list[int] = []
    if limits.extra_consideration_penalty > 0:
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
    """``{prompt_id: [labels]}`` (moral first, catalog order); ``document_is_digital`` covers every eligible document, scored or not."""
    if targets is None:
        targets = entry_targets(catalog, len(document_is_digital), sum(document_is_digital.values()))
    assigned: dict[str, list[str]] = {prompt_id: [] for prompt_id in document_is_digital}
    pairs = _allowed_pairs(scores, catalog, document_is_digital, limits)
    if pairs:
        matrix, bounds, penalty_multipliers = _model_columns_and_rows(pairs, catalog, limits, targets)
        gains = np.array([limits.score_weights[score] for _, _, score in pairs], dtype=float)
        gains = gains + _tie_break_values(pairs, seed, limits)
        penalty_costs = penalty_multipliers * limits.extra_consideration_penalty
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
    direct_penalty = limits.extra_consideration_penalty * sum(n * (n - 1) / 2 for n in counts.values())
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
        "extra_consideration_penalty": limits.extra_consideration_penalty,
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
