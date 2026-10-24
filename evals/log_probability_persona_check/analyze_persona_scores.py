#!/usr/bin/env python3
"""Analysis of a scored log-probability persona check run (local, free: no GPU, no API, no model download).

Reads <run folder>/run_configuration_*.json, scoring_inputs/ and scoring_outputs/ (formats in
run_folder_layout_and_interfaces.md) and writes <run folder>/analysis_<timestamp>/. Group names, column names, rewrite
direction names and pass criteria come from the persona set configuration recorded in the run configuration; output
columns carry the persona's name. A run that mixes model types is refused, and every output names the model type and
model.

Primary outputs, span level:
  extreme_spans_<timestamp>.jsonl              the 3 highest and 3 lowest non-overlapping 32-token windows of each
                                               document's per-token persona-minus-generic difference, with their text
  span_significance_tests_<timestamp>.csv      permutation test of each document's most extreme windows
  changed_span_*, total_log_likelihood_*       changed-span analysis of the control rewrites (analyze_changed_spans.py)
  control_verdicts_<timestamp>.json            positive and neutral control verdicts against the pre-registered criteria
  rewording_agreement_<timestamp>.csv          agreement among each persona's rewordings, reported in every run
Secondary outputs, file names starting "secondary_not_validated_": document means and their per-document tests. The
whole-document score did not track the persona in the first preachiness run, so these are not a validated measure.

Usage, from the repository root:
    python evals/log_probability_persona_check/analyze_persona_scores.py --run-folder <run folder>
"""

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check import analysis_run_loading as loading  # noqa: E402
from evals.log_probability_persona_check import analyze_changed_spans as changed_spans  # noqa: E402
from evals.log_probability_persona_check.analysis_run_loading import (  # noqa: E402
    EXPECTED_SIGN_BY_DIRECTION_ROLE, GENERIC_CONDITION_ID, NO_SYSTEM_PROMPT_CONDITION_ID)
from evals.log_probability_persona_check.build_scoring_inputs import timestamp_now  # noqa: E402
from shared import utils  # noqa: E402

SIGNIFICANCE_LEVEL = 0.05
LANGUAGE_COLUMN = "language"
MAXIMUM_LEVELS_FOR_REGRESSION_VARIABLE = 25
OUTLIER_MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER = 3.0
OUTLIER_MINIMUM_SPREAD = 0.02
DEFAULT_WINDOW_SIZE_TOKENS = 32
DEFAULT_WINDOWS_PER_DIRECTION = 3
DEFAULT_NUMBER_OF_PERMUTATIONS = 1000
DEFAULT_PERMUTATION_SEED = 0
PERMUTATION_CHUNK_ROWS = 200
PERMUTATION_TOLERANCE = 1e-12
SECONDARY_FILE_PREFIX = "secondary_not_validated_"


# ----------------------------------------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values (q-values), same order as input. NaN p-values stay NaN."""
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full(p_values.shape, np.nan)
    valid = ~np.isnan(p_values)
    valid_p = p_values[valid]
    count = len(valid_p)
    if count == 0:
        return q_values
    order = np.argsort(valid_p)
    ranked = valid_p[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(ranked, 1.0)
    q_values[valid] = adjusted
    return q_values


def direction_label(q_value, median_difference):
    if np.isnan(q_value) or q_value >= SIGNIFICANCE_LEVEL or median_difference == 0:
        return "no significant difference at q < 0.05"
    return "elevated" if median_difference > 0 else "suppressed"


def mann_whitney_two_sided(first_values, second_values):
    result = stats.mannwhitneyu(first_values, second_values, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def wilcoxon_against_zero(values):
    values = np.asarray(values, dtype=np.float64)
    if np.all(values == 0):
        return float("nan"), 1.0
    result = stats.wilcoxon(values, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def safe_spearman(first_values, second_values):
    if len(first_values) < 3 or np.ptp(first_values) == 0 or np.ptp(second_values) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(first_values, second_values).statistic)


def benjamini_hochberg_across_original_documents(table, p_column, is_original):
    """q-values for `p_column`: Benjamini-Hochberg across original documents; rewrites keep their uncorrected p."""
    q_values = table[p_column].to_numpy(dtype=float).copy()
    q_values[is_original] = benjamini_hochberg(q_values[is_original])
    return q_values


# ----------------------------------------------------------------------------------------------------------
# Extreme spans and their permutation test (primary output)
# ----------------------------------------------------------------------------------------------------------

def window_means_of(per_token_difference, window_size):
    effective_window = min(window_size, len(per_token_difference))
    cumulative = np.concatenate([[0.0], np.cumsum(per_token_difference)])
    return (cumulative[effective_window:] - cumulative[:-effective_window]) / effective_window, effective_window


def select_non_overlapping_windows(window_means, window_size, count, highest):
    """Greedy selection of up to `count` window start indices with no two windows overlapping, ranked by mean
    (descending when highest else ascending). Ties are broken by earlier start."""
    order = np.lexsort((np.arange(len(window_means)), -window_means if highest else window_means))
    chosen = []
    for start in order:
        if all(abs(int(start) - other) >= window_size for other in chosen):
            chosen.append(int(start))
            if len(chosen) == count:
                break
    return chosen


def extreme_spans_for_document(document_id, per_token_difference, token_strings, window_size, windows_per_direction,
                               persona_name="persona"):
    window_means, effective_window = window_means_of(per_token_difference, window_size)
    records = []
    for direction, highest in (("highest", True), ("lowest", False)):
        starts = select_non_overlapping_windows(window_means, effective_window, windows_per_direction, highest)
        for rank, start in enumerate(starts, start=1):
            records.append({
                "document_id": document_id, "direction": direction, "rank": rank, "start_token_index": start,
                "end_token_index_exclusive": start + effective_window, "window_size_tokens": effective_window,
                f"window_mean_{persona_name}_difference": float(window_means[start]),
                "text": "".join(token_strings[start:start + effective_window])})
    return records


def permutation_test_of_extreme_windows(per_token_difference, window_size, number_of_permutations, random_generator):
    """Tests whether a document's most extreme window is more extreme than windows of the same document with its
    tokens in random order. Statistic: the highest window mean (one-sided, upper tail) and the lowest window mean
    (one-sided, lower tail) over all windows. Returns (observed highest, p highest, observed lowest, p lowest), with
    p = (1 + number of permutations at least as extreme) / (1 + number of permutations).

    The null treats the document's tokens as exchangeable, so it asks whether any stretch stands out from token-level
    scatter. Scores of neighbouring tokens are positively correlated in real text, so the test is liberal about
    ordinary stretches of similar tokens; read it as a screen for spans, not as proof of a persona effect."""
    per_token_difference = np.asarray(per_token_difference, dtype=np.float64)
    window_means, effective_window = window_means_of(per_token_difference, window_size)
    observed_high, observed_low = float(window_means.max()), float(window_means.min())
    at_least_as_high = at_least_as_low = 0
    remaining = number_of_permutations
    while remaining > 0:
        rows = min(PERMUTATION_CHUNK_ROWS, remaining)
        shuffled = random_generator.permuted(np.tile(per_token_difference, (rows, 1)), axis=1)
        cumulative = np.concatenate([np.zeros((rows, 1)), np.cumsum(shuffled, axis=1)], axis=1)
        permuted_means = (cumulative[:, effective_window:] - cumulative[:, :-effective_window]) / effective_window
        at_least_as_high += int((permuted_means.max(axis=1) >= observed_high - PERMUTATION_TOLERANCE).sum())
        at_least_as_low += int((permuted_means.min(axis=1) <= observed_low + PERMUTATION_TOLERANCE).sum())
        remaining -= rows
    denominator = 1 + number_of_permutations
    return (observed_high, (1 + at_least_as_high) / denominator, observed_low, (1 + at_least_as_low) / denominator)


def build_extreme_spans_and_significance(run, window_size, windows_per_direction, number_of_permutations, seed):
    """Returns (extreme span records, span significance DataFrame)."""
    setup = run.setup
    records, significance_rows = [], []
    for document_index, document in enumerate(run.documents):
        scores = run.scores_by_document[document["document_id"]]
        if scores.number_of_tokens == 0:
            continue
        per_token_difference = scores.per_token_difference_by_persona[setup.persona_name]
        spans = extreme_spans_for_document(document["document_id"], per_token_difference, scores.token_strings,
                                           window_size, windows_per_direction, setup.persona_name)
        random_generator = np.random.default_rng([seed, document_index])
        high, p_high, low, p_low = permutation_test_of_extreme_windows(
            per_token_difference, window_size, number_of_permutations, random_generator)
        p_by_direction = {"highest": p_high, "lowest": p_low}
        for span in spans:
            span["model_type"], span["model_name"] = setup.model_type, setup.model_name
            span["document_permutation_p_value_of_most_extreme_window_in_direction"] = p_by_direction[span["direction"]]
        records.extend(spans)
        significance_rows.append({
            "document_id": document["document_id"], "rewrite_direction": document.get("rewrite_direction"),
            "number_of_tokens": scores.number_of_tokens,
            "window_size_tokens": min(window_size, scores.number_of_tokens),
            "number_of_permutations": number_of_permutations,
            f"highest_window_mean_{setup.persona_name}_difference": high, "highest_window_p_value": p_high,
            f"lowest_window_mean_{setup.persona_name}_difference": low, "lowest_window_p_value": p_low})
    table = pd.DataFrame(significance_rows)
    if len(table):
        is_original = table["rewrite_direction"].isna().to_numpy()
        table["q_value_is_uncorrected_rewrite"] = ~is_original
        table["highest_window_q_value"] = benjamini_hochberg_across_original_documents(table, "highest_window_p_value", is_original)
        table["lowest_window_q_value"] = benjamini_hochberg_across_original_documents(table, "lowest_window_p_value", is_original)
    return records, table


# ----------------------------------------------------------------------------------------------------------
# Per-document scores (secondary: document means, not validated)
# ----------------------------------------------------------------------------------------------------------

def condition_score_matrix_of(run):
    return pd.DataFrame([run.scores_by_document[document["document_id"]].condition_scores for document in run.documents],
                        columns=run.condition_ids, index=[document["document_id"] for document in run.documents])


def build_per_document_scores(run, condition_score_matrix):
    setup = run.setup
    persona, control = setup.persona_name, setup.control_group
    no_prompt_row = run.condition_ids.index(NO_SYSTEM_PROMPT_CONDITION_ID)
    rows = []
    for document in run.documents:
        scores = run.scores_by_document[document["document_id"]]
        condition_scores = scores.condition_scores
        row = {"document_id": document["document_id"],
               "source_document_id": document.get("source_document_id", document["document_id"]),
               "rewrite_direction": document.get("rewrite_direction"), LANGUAGE_COLUMN: document.get("language")}
        for variable_name, value in (document.get("variables") or {}).items():
            row[f"spec_{variable_name}"] = value
        row["number_of_tokens"] = scores.number_of_tokens
        row["generic_mean_log_probability"] = scores.generic_mean_log_probability
        row[f"{persona}_mean_score"] = float(condition_scores[run.group_indices[setup.target_group]].mean())
        row[f"{control}_mean_score"] = float(condition_scores[run.group_indices[setup.control_group]].mean())
        if setup.same_values_group:
            row[f"{setup.same_values_group}_mean_score"] = float(
                condition_scores[run.group_indices[setup.same_values_group]].mean())
        for same_values_persona_name in run.persona_names_by_group.get(setup.same_values_group, []):
            same_values_rows = [index for index, condition in enumerate(run.conditions)
                                if condition["condition_group"] == setup.same_values_group
                                and condition["persona_name"] == same_values_persona_name]
            row[f"{same_values_persona_name}_mean_score"] = float(condition_scores[same_values_rows].mean())
        row[setup.target_minus_control_score_column] = row[f"{persona}_mean_score"] - row[f"{control}_mean_score"]
        row["no_system_prompt_score"] = float(condition_scores[no_prompt_row])
        for condition_id, score in zip(run.condition_ids, condition_scores):
            row[f"score__{condition_id}"] = float(score)
        rows.append(row)
    return pd.DataFrame(rows)


def build_significance_table(run, condition_score_matrix):
    setup = run.setup
    persona, control, same_values = setup.persona_name, setup.control_group, setup.same_values_group
    rows = []
    for document in run.documents:
        scores = condition_score_matrix.loc[document["document_id"]].to_numpy()
        target = scores[run.group_indices[setup.target_group]]
        controls = scores[run.group_indices[control]]
        main_u, main_p = mann_whitney_two_sided(target, controls)
        wilcoxon_statistic, wilcoxon_p = wilcoxon_against_zero(target)
        row = {"document_id": document["document_id"], f"number_of_{persona}_rewordings": len(target),
               f"number_of_{control}_rewordings": len(controls),
               "main_test_mann_whitney_u": main_u, "main_test_p_value": main_p,
               f"main_test_median_{persona}_minus_{control}": float(np.median(target) - np.median(controls)),
               f"{persona}_median_score": float(np.median(target)),
               f"{persona}_vs_generic_wilcoxon_statistic": wilcoxon_statistic,
               f"{persona}_vs_generic_wilcoxon_p_value": wilcoxon_p}
        if same_values:
            same = scores[run.group_indices[same_values]]
            second_u, second_p = mann_whitney_two_sided(target, same)
            row.update({f"number_of_{same_values}_rewordings": len(same), "second_test_mann_whitney_u": second_u,
                        "second_test_p_value": second_p,
                        f"second_test_median_{persona}_minus_{same_values}": float(np.median(target) - np.median(same))})
        rows.append(row)
    table = pd.DataFrame(rows)
    is_original = np.array([document.get("rewrite_direction") is None for document in run.documents])
    table["q_value_is_uncorrected_rewrite"] = ~is_original
    tests = [("main_test_p_value", "main_test_q_value"),
             (f"{persona}_vs_generic_wilcoxon_p_value", f"{persona}_vs_generic_wilcoxon_q_value")]
    if same_values:
        tests.append(("second_test_p_value", "second_test_q_value"))
    for p_column, q_column in tests:
        table[q_column] = benjamini_hochberg_across_original_documents(table, p_column, is_original)
    table["main_test_direction"] = [direction_label(q, d) for q, d in zip(
        table["main_test_q_value"], table[f"main_test_median_{persona}_minus_{control}"])]
    if same_values:
        table["second_test_direction"] = [direction_label(q, d) for q, d in zip(
            table["second_test_q_value"], table[f"second_test_median_{persona}_minus_{same_values}"])]
    q_column = f"{persona}_vs_generic_wilcoxon_q_value"
    table[f"{persona}_significantly_below_generic"] = (table[q_column] < SIGNIFICANCE_LEVEL) & (table[f"{persona}_median_score"] < 0)
    table[f"{persona}_significantly_above_generic"] = (table[q_column] < SIGNIFICANCE_LEVEL) & (table[f"{persona}_median_score"] > 0)
    return table


# ----------------------------------------------------------------------------------------------------------
# Rewording agreement
# ----------------------------------------------------------------------------------------------------------

def one_way_intraclass_correlation(score_matrix):
    """ICC(1,1): documents as targets (rows), rewordings as interchangeable raters (columns)."""
    document_count, rewording_count = score_matrix.shape
    if document_count < 2 or rewording_count < 2:
        return float("nan")
    grand_mean = score_matrix.mean()
    row_means = score_matrix.mean(axis=1)
    between = rewording_count * ((row_means - grand_mean) ** 2).sum() / (document_count - 1)
    within = ((score_matrix - row_means[:, None]) ** 2).sum() / (document_count * (rewording_count - 1))
    denominator = between + (rewording_count - 1) * within
    return float((between - within) / denominator) if denominator > 0 else float("nan")


def build_rewording_agreement(run, condition_score_matrix):
    setup = run.setup
    rows = []
    for group in setup.persona_group_names:
        for persona_name in run.persona_names_by_group[group]:
            persona_conditions = [c for c in run.conditions if c["condition_group"] == group and c["persona_name"] == persona_name]
            condition_ids = [c["condition_id"] for c in persona_conditions]
            score_matrix = condition_score_matrix[condition_ids].to_numpy()  # documents x rewordings
            document_count, rewording_count = score_matrix.shape
            pairwise = [safe_spearman(score_matrix[:, i], score_matrix[:, j])
                        for i, j in itertools.combinations(range(rewording_count), 2)]
            mean_pairwise = float(np.nanmean(pairwise)) if pairwise and not np.all(np.isnan(pairwise)) else float("nan")
            if rewording_count > 1 and document_count > 1:
                within_document_sd = float(np.mean(score_matrix.std(axis=1, ddof=1)))
                across_document_sd = float(score_matrix.mean(axis=1).std(ddof=1))
                ratio = within_document_sd / across_document_sd if across_document_sd > 0 else float("nan")
            else:
                within_document_sd = across_document_sd = ratio = float("nan")
            leave_one_out = np.array([
                safe_spearman(score_matrix[:, index], np.delete(score_matrix, index, axis=1).mean(axis=1))
                if rewording_count > 1 else float("nan") for index in range(rewording_count)])
            if np.all(np.isnan(leave_one_out)):
                outlier_flags, threshold = [False] * rewording_count, float("nan")
            else:
                median = np.nanmedian(leave_one_out)
                spread = max(1.4826 * np.nanmedian(np.abs(leave_one_out - median)), OUTLIER_MINIMUM_SPREAD)
                threshold = float(median - OUTLIER_MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER * spread)
                outlier_flags = [bool(value < threshold) if not np.isnan(value) else False for value in leave_one_out]
            icc = one_way_intraclass_correlation(score_matrix)
            for index, condition in enumerate(persona_conditions):
                rows.append({
                    "persona_name": persona_name, "condition_group": group, "condition_id": condition["condition_id"],
                    "rewording_index": condition.get("rewording_index"), "number_of_documents": document_count,
                    "number_of_rewordings_in_persona": rewording_count,
                    "persona_mean_pairwise_spearman_across_documents": mean_pairwise,
                    "persona_mean_within_document_sd_across_rewordings": within_document_sd,
                    "persona_sd_across_documents_of_rewording_mean_score": across_document_sd,
                    "persona_within_to_across_sd_ratio": ratio,
                    "persona_intraclass_correlation_one_way_single_rewording": icc,
                    "rewording_mean_score_across_documents": float(score_matrix[:, index].mean()),
                    "rewording_spearman_with_leave_one_out_mean_of_other_rewordings": float(leave_one_out[index]),
                    "persona_outlier_threshold_on_leave_one_out_spearman": threshold,
                    "rewording_is_outlier": outlier_flags[index]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Scores by spec variable (secondary)
# ----------------------------------------------------------------------------------------------------------

def ordinary_least_squares(design_matrix, outcome):
    """Returns (coefficients, standard_errors, t_statistics, p_values). Uses the pseudo-inverse so collinear columns
    give the minimum-norm solution; their standard errors are then not individually meaningful. Degrees of freedom
    = n - rank."""
    sample_size, _ = design_matrix.shape
    gram_inverse = np.linalg.pinv(design_matrix.T @ design_matrix)
    coefficients = gram_inverse @ design_matrix.T @ outcome
    residuals = outcome - design_matrix @ coefficients
    degrees_of_freedom = sample_size - np.linalg.matrix_rank(design_matrix)
    if degrees_of_freedom <= 0:
        nan = np.full(coefficients.shape, np.nan)
        return coefficients, nan, nan, nan
    residual_variance = float(residuals @ residuals) / degrees_of_freedom
    standard_errors = np.sqrt(np.maximum(np.diag(gram_inverse) * residual_variance, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = np.where(standard_errors > 0, coefficients / standard_errors, np.nan)
    p_values = 2 * stats.t.sf(np.abs(t_statistics), degrees_of_freedom)
    return coefficients, standard_errors, t_statistics, p_values


def build_scores_by_spec_variable(per_document_scores, outcome_column):
    originals = per_document_scores[per_document_scores["rewrite_direction"].isna()].copy()
    variable_columns = [c for c in per_document_scores.columns if c.startswith("spec_")] + [LANGUAGE_COLUMN]
    variable_columns = [c for c in variable_columns if c in originals.columns]
    rows = []
    for column in variable_columns:
        variable_name = column[len("spec_"):] if column.startswith("spec_") else column
        summary = originals[outcome_column].groupby(originals[column].astype(str)).agg(["mean", "count", "std"])
        for level, entry in summary.iterrows():
            standard_error = entry["std"] / np.sqrt(entry["count"]) if entry["count"] > 1 else float("nan")
            rows.append({"section": "summary_by_level", "variable": variable_name, "level": level,
                         "count": int(entry["count"]), f"mean_{outcome_column}": entry["mean"],
                         "standard_error_of_mean": standard_error})
    design_columns, design_names = [np.ones(len(originals))], [("intercept", "")]
    for column in variable_columns:
        variable_name = column[len("spec_"):] if column.startswith("spec_") else column
        if column == LANGUAGE_COLUMN:
            continue  # collinear with culture; summary rows only
        levels = sorted(originals[column].astype(str).unique())
        if len(levels) > MAXIMUM_LEVELS_FOR_REGRESSION_VARIABLE:
            rows.append({"section": "regression_coefficient", "variable": variable_name,
                         "level": f"NOT_INCLUDED_{len(levels)}_levels_exceeds_{MAXIMUM_LEVELS_FOR_REGRESSION_VARIABLE}"})
            continue
        for level in levels[1:]:  # first level (alphabetical) is the reference
            design_columns.append((originals[column].astype(str) == level).to_numpy(dtype=np.float64))
            design_names.append((variable_name, level))
        if levels:
            rows.append({"section": "regression_reference_level", "variable": variable_name, "level": levels[0]})
    if len(originals) > 0:
        coefficients, standard_errors, t_statistics, p_values = ordinary_least_squares(
            np.column_stack(design_columns), originals[outcome_column].to_numpy())
        for (variable_name, level), coefficient, standard_error, t_statistic, p_value in zip(
                design_names, coefficients, standard_errors, t_statistics, p_values):
            rows.append({"section": "regression_coefficient", "variable": variable_name, "level": level,
                         "coefficient": float(coefficient), "coefficient_standard_error": float(standard_error),
                         "t_statistic": float(t_statistic), "p_value": float(p_value), "count": len(originals)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Whole-document control check and verdicts
# ----------------------------------------------------------------------------------------------------------

def build_whole_document_control_check(per_document_scores, setup):
    """Rewrite-minus-source difference of the persona-minus-control score for every rewritten document, then one
    summary row per rewrite direction. Returns (table, verdicts dict)."""
    score_column = setup.target_minus_control_score_column
    columns = ["row_type", "document_id", "source_document_id", "rewrite_direction", "direction_role", score_column,
               f"source_{score_column}", "difference_rewrite_minus_source", "expected_sign_of_difference",
               "moved_in_expected_direction", "number_of_rewrites", "number_moving_in_expected_direction",
               "wilcoxon_signed_rank_p_value_two_sided", "wilcoxon_signed_rank_p_value_expected_direction", "note"]
    rewritten = per_document_scores[per_document_scores["rewrite_direction"].notna()]
    if len(rewritten) == 0:
        table = pd.DataFrame([{"row_type": "note", "note": "skipped: no rewritten documents in this run"}], columns=columns)
        return table, whole_document_verdicts(setup, {})
    score_by_document = per_document_scores.set_index("document_id")[score_column]
    rows = []
    for _, row in rewritten.iterrows():
        source_id = row["source_document_id"]
        role = setup.direction_role_of(row["rewrite_direction"])
        expected_sign = EXPECTED_SIGN_BY_DIRECTION_ROLE.get(role)
        base = {"row_type": "document", "document_id": row["document_id"], "source_document_id": source_id,
                "rewrite_direction": row["rewrite_direction"], "direction_role": role, score_column: row[score_column]}
        if source_id not in score_by_document.index:
            rows.append({**base, "note": "source document was not scored; excluded from summary"})
            continue
        source_score = float(score_by_document[source_id])
        difference = float(row[score_column]) - source_score
        sign = expected_sign if expected_sign else 1  # neutral rewrites: "moved" means toward the persona
        rows.append({**base, f"source_{score_column}": source_score, "difference_rewrite_minus_source": difference,
                     "expected_sign_of_difference": expected_sign, "moved_in_expected_direction": bool(difference * sign > 0)})
    document_rows = pd.DataFrame(rows)
    counts_by_role = {}
    for role, direction in setup.direction_names_by_role.items():
        subset = document_rows[(document_rows["rewrite_direction"] == direction)
                               & document_rows["difference_rewrite_minus_source"].notna()]
        if len(subset) == 0:
            continue
        differences = subset["difference_rewrite_minus_source"].to_numpy(dtype=float)
        expected_sign = EXPECTED_SIGN_BY_DIRECTION_ROLE.get(role)
        sign = expected_sign if expected_sign else 1
        if np.all(differences == 0):
            p_two_sided = p_expected = 1.0
        else:
            p_two_sided = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
            p_expected = float(stats.wilcoxon(differences, alternative="greater" if sign > 0 else "less").pvalue)
        moving = int(subset["moved_in_expected_direction"].sum())
        counts_by_role[role] = {"number_of_rewrites": len(subset), "number_moving_expected_way": moving}
        rows.append({"row_type": "summary", "rewrite_direction": direction, "direction_role": role,
                     "number_of_rewrites": len(subset), "expected_sign_of_difference": expected_sign,
                     "number_moving_in_expected_direction": moving,
                     "wilcoxon_signed_rank_p_value_two_sided": p_two_sided,
                     "wilcoxon_signed_rank_p_value_expected_direction": p_expected})
    return pd.DataFrame(rows, columns=columns), whole_document_verdicts(setup, counts_by_role)


def whole_document_verdicts(setup, counts_by_role):
    criterion = setup.pass_criteria.get("whole_document")
    if criterion is None:
        return {"positive_control": {"verdict": changed_spans.VERDICT_NO_CRITERION, "per_direction": {}}}
    per_direction = {}
    for role in criterion["applies_to_directions"]:
        counts = counts_by_role.get(role, {"number_of_rewrites": 0, "number_moving_expected_way": 0})
        result = changed_spans.judge_count_against_criterion(
            counts["number_moving_expected_way"], counts["number_of_rewrites"],
            criterion["minimum_number_moving_expected_way"], criterion["number_of_rewrites_tested_per_direction"])
        result["direction_name"] = setup.direction_names_by_role[role]
        per_direction[role] = result
    neutral = counts_by_role.get(loading.DIRECTION_ROLE_NEUTRAL)
    return {"positive_control": {"verdict": changed_spans.overall_verdict(per_direction), "per_direction": per_direction,
                                 "criterion_status": setup.pass_criteria.get("status")},
            "neutral_control_moves_toward_persona_not_judged": (
                {"number_of_rewrites": neutral["number_of_rewrites"],
                 "number_moving_toward_persona": neutral["number_moving_expected_way"]} if neutral else None)}


# ----------------------------------------------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------------------------------------------

def analysis_notes_text(setup, window_size, windows_per_direction, number_of_permutations):
    persona, control, same = setup.persona_name, setup.control_group, setup.same_values_group
    directions = ", ".join(f"{role} = {name}" for role, name in setup.direction_names_by_role.items())
    return f"""# Analysis notes: how each output is computed

Persona set: {setup.persona_set_name}. Persona: {persona}. Scored with {setup.model_label} (revision {setup.model_revision}).
Scores from different model types are not comparable. Condition groups: target `{setup.target_group}`, topic controls `{control}`{f", same-values control `{same}`" if same else ""}. Rewrite directions: {directions}.
All differences are in nats per token. Condition groups and direction names come from the persona set configuration recorded in the run configuration.

## Loading checks
Analysis stops with an error if the run configurations disagree on model type, model name or model revision; if any document has no .npz; if the condition ids in any .npz differ from conditions.jsonl; if the condition groups or their sizes differ from the configuration; or if an array shape does not match the condition and token counts.

## Primary: extreme_spans (JSONL)
- Per-token {persona} difference: mean over the {persona} conditions of `log_probabilities[c, t] - log_probabilities[generic, t]`.
- Windows of {window_size} tokens (a shorter document uses one window over the whole document). Up to {windows_per_direction} highest and {windows_per_direction} lowest windows, chosen greedily by window mean with no two windows in a list overlapping. `start_token_index` is inclusive, `end_token_index_exclusive` exclusive. `text` is the token strings joined with no separator.
- Each record carries the document's permutation p-value for its direction (below).

## Primary: span_significance_tests
Permutation test over windows. For each document the per-token {persona} difference is shuffled {number_of_permutations} times; the statistic is the highest (lowest) window mean over all windows of {window_size} tokens. p = (1 + permutations at least as extreme) / (1 + {number_of_permutations}), one-sided, with its own tail for highest and lowest. The null treats tokens as exchangeable, so it screens for stretches that stand out from token-level scatter; neighbouring tokens are positively correlated in real text, so it is liberal and is not proof of a persona effect. `*_q_value` is Benjamini-Hochberg across original documents (`rewrite_direction` empty); rewrites keep the uncorrected p (`q_value_is_uncorrected_rewrite`).

## Primary: changed-span analysis (changed_span_*, total_log_likelihood_*, changed_span_summary)
Rewrite tokens are aligned to the source's tokens with difflib.SequenceMatcher (no autojunk). Rewrite tokens outside every matching block are changed, inside are unchanged; source tokens outside every matching block are replaced. Span score = mean over its tokens of ({persona} minus generic) minus the same for the pooled topic controls (mean over control personas of their persona-minus-generic). `changed_minus_unchanged` and `changed_minus_replaced` compare spans. `changed_span_by_persona` repeats changed minus unchanged and changed minus replaced for each persona separately and each rewrite direction, with a pooled row for the topic controls and, when the set has same-values controls, a pooled row for them (`pooled_<group>`). A rewrite moves the expected way when changed_minus_unchanged has the expected sign (+ toward, - away); for the neutral direction the count is of rewrites moving toward the persona (changed_minus_unchanged above 0). Rewrites with no changed tokens count as not moving.

## Control verdicts (control_verdicts)
Judged against `pass_criteria` copied into the run configuration before scoring. Positive control: for each direction in `applies_to_directions`, pass when the number of rewrites moving the expected way is at least the pre-registered minimum. A direction is "not evaluable" when the number of rewrites scored differs from the number the criterion was fixed for. Neutral control (changed span): pass when at most `maximum_number_moving_toward_persona` neutral rewrites move toward the persona; "no criterion pre-registered" when the configuration has none. Whole-document and changed-span verdicts are reported separately.

## Rewording agreement (rewording_agreement)
One row per rewording; persona-level columns repeat on each row of the persona, for every persona in every group. Rewording scores are the `score__<condition_id>` values across documents.
- `persona_mean_pairwise_spearman_across_documents`: mean over rewording pairs of the Spearman correlation of their scores across documents.
- `persona_intraclass_correlation_one_way_single_rewording`: ICC(1,1), documents as targets, rewordings as raters, (MSB - MSW) / (MSB + (k - 1) MSW).
- `persona_within_to_across_sd_ratio`: mean within-document SD across rewordings divided by the SD across documents of the mean-over-rewordings score.
- `rewording_spearman_with_leave_one_out_mean_of_other_rewordings` and `rewording_is_outlier`: outlier when that Spearman is below `median - {OUTLIER_MEDIAN_ABSOLUTE_DEVIATION_MULTIPLIER} * max(1.4826 * MAD, {OUTLIER_MINIMUM_SPREAD})` over the persona's rewordings.
- Spearman is NaN with fewer than 3 documents or a constant series.

## Secondary, not validated: document means ({SECONDARY_FILE_PREFIX}*)
The whole-document score did not track the persona in the first preachiness run: persona-like passages are a small share of most documents and the rest of the text drowns them out. Do not build a corpus-level verdict on these.
- per_document_scores: `score__<condition_id>` = mean over tokens of `log_probabilities[c] - log_probabilities[generic]`; `{persona}_mean_score`, `{control}_mean_score`{f", `{same}_mean_score`" if same else ""} are means of those over the group's conditions; `{setup.target_minus_control_score_column}` is their difference. Each same-values control persona also has its own `<persona>_mean_score` column; `{same}_mean_score` pools all of them.
- per_document_significance_tests: main test = two-sided Mann-Whitney U of the {persona} rewording scores against the pooled {control} rewording scores{f"; second test the same against the pooled {same} rewording scores (all same-values personas together)" if same else ""}; one-sample Wilcoxon of the {persona} scores against 0 (generic is 0 by definition). Benjamini-Hochberg q-values across original documents; rewrites keep the uncorrected p. Direction label: elevated or suppressed when q < 0.05, otherwise not significant.
- scores_by_spec_variable: original documents only; per-level mean and standard error of `{setup.target_minus_control_score_column}`, and ordinary least squares on one-hot spec variables (alphabetically first level is the reference; pseudo-inverse; variables with more than {MAXIMUM_LEVELS_FOR_REGRESSION_VARIABLE} levels left out; language left out as collinear with culture).
- whole_document_control_check: rewrite minus source of the {setup.target_minus_control_score_column}; a difference of exactly 0 counts as not moving; Wilcoxon signed-rank p-values two-sided and in the expected direction.
"""


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------

def run_analysis(run_folder, window_size=DEFAULT_WINDOW_SIZE_TOKENS, windows_per_direction=DEFAULT_WINDOWS_PER_DIRECTION,
                 number_of_permutations=DEFAULT_NUMBER_OF_PERMUTATIONS, permutation_seed=DEFAULT_PERMUTATION_SEED,
                 timestamp=None, legacy_run_persona_set=None, skip_changed_spans=False):
    run_folder = Path(run_folder)
    timestamp = timestamp or timestamp_now()
    analysis_folder = run_folder / f"analysis_{timestamp}"
    if analysis_folder.exists():
        raise FileExistsError(f"{analysis_folder} already exists; wait a minute or remove it")
    run = loading.load_run(run_folder, legacy_run_persona_set)
    analysis_folder.mkdir()
    setup = run.setup
    label = loading.insert_model_labels

    condition_score_matrix = condition_score_matrix_of(run)
    per_document_scores = build_per_document_scores(run, condition_score_matrix)
    significance_table = build_significance_table(run, condition_score_matrix)
    span_records, span_significance = build_extreme_spans_and_significance(
        run, window_size, windows_per_direction, number_of_permutations, permutation_seed)
    rewording_agreement = build_rewording_agreement(run, condition_score_matrix)
    scores_by_spec_variable = build_scores_by_spec_variable(per_document_scores, setup.target_minus_control_score_column)
    whole_document_check, whole_document_result = build_whole_document_control_check(per_document_scores, setup)

    utils.save_jsonl(span_records, analysis_folder / f"extreme_spans_{timestamp}.jsonl")
    label(span_significance, setup).to_csv(analysis_folder / f"span_significance_tests_{timestamp}.csv", index=False)
    label(rewording_agreement, setup).to_csv(analysis_folder / f"rewording_agreement_{timestamp}.csv", index=False)
    label(per_document_scores, setup).to_csv(
        analysis_folder / f"{SECONDARY_FILE_PREFIX}per_document_scores_{timestamp}.csv", index=False)
    label(significance_table, setup).to_csv(
        analysis_folder / f"{SECONDARY_FILE_PREFIX}per_document_significance_tests_{timestamp}.csv", index=False)
    label(scores_by_spec_variable, setup).to_csv(
        analysis_folder / f"{SECONDARY_FILE_PREFIX}scores_by_spec_variable_{timestamp}.csv", index=False)
    label(whole_document_check, setup).to_csv(
        analysis_folder / f"{SECONDARY_FILE_PREFIX}whole_document_control_check_{timestamp}.csv", index=False)

    changed_span_summary = None
    if not skip_changed_spans:
        changed_span_summary, _ = changed_spans.write_changed_span_outputs(run, analysis_folder, timestamp)
    verdicts = {"model_type": setup.model_type, "model_name": setup.model_name, "model_revision": setup.model_revision,
                "persona_name": setup.persona_name, "pass_criteria": setup.pass_criteria,
                "whole_document": whole_document_result,
                "changed_span": changed_span_summary["verdicts"] if changed_span_summary else None}
    (analysis_folder / f"control_verdicts_{timestamp}.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    (analysis_folder / f"analysis_notes_{timestamp}.md").write_text(
        analysis_notes_text(setup, window_size, windows_per_direction, number_of_permutations), encoding="utf-8")
    summary = {"model_type": setup.model_type, "model_name": setup.model_name, "model_revision": setup.model_revision,
               "persona_set_name": setup.persona_set_name, "persona_name": setup.persona_name,
               "run_folder": str(run_folder), "converted_from_legacy_run_format": run.is_legacy_format,
               "number_of_documents": len(run.documents), "number_of_conditions": len(run.conditions),
               "window_size_tokens": window_size, "windows_per_direction": windows_per_direction,
               "number_of_permutations": number_of_permutations, "permutation_seed": permutation_seed}
    (analysis_folder / f"analysis_summary_{timestamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Analysis of {setup.model_label} written to {analysis_folder}")
    return analysis_folder


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--window-size-tokens", type=int, default=DEFAULT_WINDOW_SIZE_TOKENS)
    parser.add_argument("--windows-per-direction", type=int, default=DEFAULT_WINDOWS_PER_DIRECTION)
    parser.add_argument("--number-of-permutations", type=int, default=DEFAULT_NUMBER_OF_PERMUTATIONS)
    parser.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    parser.add_argument("--skip-changed-spans", action="store_true", help="Leave out the changed-span analysis")
    parser.add_argument("--legacy-run-persona-set", default=None,
                        help="Only for legacy-format run folders: the persona set to pair them with")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    return run_analysis(arguments.run_folder, arguments.window_size_tokens, arguments.windows_per_direction,
                        arguments.number_of_permutations, arguments.permutation_seed,
                        legacy_run_persona_set=arguments.legacy_run_persona_set,
                        skip_changed_spans=arguments.skip_changed_spans)


if __name__ == "__main__":
    main()
