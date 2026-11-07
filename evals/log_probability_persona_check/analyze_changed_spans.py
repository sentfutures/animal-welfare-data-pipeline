#!/usr/bin/env python3
"""Changed-span analysis of the positive and neutral control rewrites (local, free; no API, no model).

For each rewritten document, the rewrite's tokens are aligned to its source's tokens with difflib.SequenceMatcher.
Rewrite tokens outside every matching block are "changed" (inserted or reworded), rewrite tokens inside one are
"unchanged", and source tokens outside every matching block are "replaced". The score of a span is the mean over its
tokens of the persona-minus-control per-token difference (persona minus generic, control being the topic control
personas pooled). Direction names, group names and pass criteria come from the persona set configuration recorded in
the run configuration.

Writes, into <run folder>/analysis_<timestamp>/ (or --analysis-folder):
  changed_span_<persona>_minus_<control>_<timestamp>.csv   per rewrite: token counts and the score on changed, unchanged
                                                           and replaced text
  changed_span_by_persona_<timestamp>.csv                  per rewrite direction and persona: persona-minus-generic on
                                                           changed minus unchanged and changed minus replaced text
  total_log_likelihood_differences_<timestamp>.csv         per document: total log probability under the persona minus
                                                           under the controls
  changed_span_summary_<timestamp>.json                    counts of rewrites moving the expected way, Wilcoxon p-values
                                                           and the changed-span verdicts against the pass criteria

Usage, from the repository root:
    python evals/log_probability_persona_check/analyze_changed_spans.py --run-folder <run folder>
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check import analysis_run_loading as loading  # noqa: E402
from evals.log_probability_persona_check.analysis_run_loading import (  # noqa: E402
    DIRECTION_ROLE_NEUTRAL, DIRECTION_ROLE_TOWARD, EXPECTED_SIGN_BY_DIRECTION_ROLE)
from evals.log_probability_persona_check.build_scoring_inputs import timestamp_now  # noqa: E402

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_NOT_EVALUABLE = "not evaluable"
VERDICT_NO_CRITERION = "no criterion pre-registered"


def changed_token_masks(source_token_ids, rewrite_token_ids):
    matcher = difflib.SequenceMatcher(None, source_token_ids, rewrite_token_ids, autojunk=False)
    rewrite_changed = np.ones(len(rewrite_token_ids), bool)
    source_replaced = np.ones(len(source_token_ids), bool)
    for block in matcher.get_matching_blocks():
        rewrite_changed[block.b:block.b + block.size] = False
        source_replaced[block.a:block.a + block.size] = False
    return rewrite_changed, source_replaced


def masked_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else float("nan")


def nan_safe_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")


def wilcoxon_p_value(values):
    values = [value for value in values if not np.isnan(value)]
    return float(wilcoxon(values).pvalue) if len(values) > 1 and any(values) else float("nan")


def judge_count_against_criterion(number_moving, number_of_rewrites, minimum_required, number_pre_registered):
    """Verdict for one direction: pass when at least `minimum_required` of the rewrites moved the expected way. The
    criterion was fixed for `number_pre_registered` rewrites, so a different number of rewrites cannot be judged."""
    result = {"number_of_rewrites": number_of_rewrites, "number_moving_expected_way": number_moving,
              "minimum_required": minimum_required, "number_of_rewrites_pre_registered": number_pre_registered}
    if number_of_rewrites == 0:
        result.update(verdict=VERDICT_NOT_EVALUABLE, reason="no rewrites of this direction in the run")
    elif number_of_rewrites != number_pre_registered:
        result.update(verdict=VERDICT_NOT_EVALUABLE,
                      reason=f"{number_of_rewrites} rewrites were scored; the criterion was fixed for {number_pre_registered}")
    else:
        result.update(verdict=VERDICT_PASS if number_moving >= minimum_required else VERDICT_FAIL, reason="")
    return result


def overall_verdict(per_direction_results):
    verdicts = [result["verdict"] for result in per_direction_results.values()]
    if not verdicts or VERDICT_NOT_EVALUABLE in verdicts:
        return VERDICT_NOT_EVALUABLE
    return VERDICT_PASS if all(verdict == VERDICT_PASS for verdict in verdicts) else VERDICT_FAIL


def build_changed_span_tables(run):
    """Returns (span_table, persona_table, total_table) as DataFrames."""
    setup = run.setup
    persona_name, pooled_name = setup.persona_name, setup.pooled_control_persona_name
    control_persona_names = run.persona_names_by_group[setup.control_group]
    columns_prefix = f"{persona_name}_minus_{setup.control_group}_per_token"

    same_values_persona_names = run.persona_names_by_group.get(setup.same_values_group, [])

    def with_pooled_control(scores):
        pooled = dict(scores.per_token_difference_by_persona)
        pooled[pooled_name] = np.mean([pooled[name] for name in control_persona_names], axis=0)
        if same_values_persona_names:
            pooled[setup.pooled_same_values_persona_name] = np.mean(
                [pooled[name] for name in same_values_persona_names], axis=0)
        return pooled

    per_token = {document_id: (scores.token_ids, with_pooled_control(scores))
                 for document_id, scores in run.scores_by_document.items() if scores.number_of_tokens > 0}
    span_rows, persona_values = [], {}
    for document in run.documents:
        direction = document.get("rewrite_direction")
        if not direction:
            continue
        if document["document_id"] not in per_token or document["source_document_id"] not in per_token:
            continue
        rewrite_ids, rewrite_scores = per_token[document["document_id"]]
        source_ids, source_scores = per_token[document["source_document_id"]]
        rewrite_changed, source_replaced = changed_token_masks(source_ids, rewrite_ids)
        rewrite_contrast = rewrite_scores[persona_name] - rewrite_scores[pooled_name]
        source_contrast = source_scores[persona_name] - source_scores[pooled_name]
        changed = masked_mean(rewrite_contrast, rewrite_changed)
        unchanged = masked_mean(rewrite_contrast, ~rewrite_changed)
        replaced = masked_mean(source_contrast, source_replaced)
        span_rows.append({
            "document_id": document["document_id"], "source_document_id": document["source_document_id"],
            "rewrite_direction": direction, "rewrite_tokens": len(rewrite_ids),
            "changed_tokens": int(rewrite_changed.sum()), "replaced_source_tokens": int(source_replaced.sum()),
            f"{columns_prefix}_changed": changed, f"{columns_prefix}_unchanged": unchanged,
            f"{columns_prefix}_replaced_source_text": replaced,
            "changed_minus_unchanged": changed - unchanged, "changed_minus_replaced": changed - replaced})
        for name, scores in rewrite_scores.items():
            changed_persona = masked_mean(scores, rewrite_changed)
            entry = persona_values.setdefault((direction, name), {"unchanged": [], "replaced": []})
            entry["unchanged"].append(changed_persona - masked_mean(scores, ~rewrite_changed))
            entry["replaced"].append(changed_persona - masked_mean(source_scores[name], source_replaced))
    span_table = pd.DataFrame(span_rows)
    persona_table = pd.DataFrame([{
        "rewrite_direction": direction, "persona": name,
        "mean_changed_minus_unchanged": nan_safe_mean(values["unchanged"]),
        "rewrites_where_changed_above_unchanged": int(np.sum(np.array(values["unchanged"]) > 0)),
        "mean_changed_minus_replaced": nan_safe_mean(values["replaced"]),
        "rewrites_where_changed_above_replaced": int(np.sum(np.array(values["replaced"]) > 0)),
        "number_of_rewrites": len(values["unchanged"])}
        for (direction, name), values in persona_values.items()])
    if len(persona_table):
        persona_table = persona_table.sort_values(["rewrite_direction", "mean_changed_minus_unchanged"],
                                                  ascending=[True, False], kind="stable")
    total_table = pd.DataFrame([{
        "document_id": document_id,
        f"total_log_probability_{persona_name}_minus_{setup.control_group}":
            scores.total_log_probability_by_group[setup.target_group]
            - scores.total_log_probability_by_group[setup.control_group]}
        for document_id, scores in run.scores_by_document.items()])
    return span_table, persona_table, total_table


def summarize_changed_spans(run, span_table, total_table):
    """Returns the summary dict: per rewrite direction counts and p-values, plus the changed-span verdicts."""
    setup = run.setup
    total_column = f"total_log_probability_{setup.persona_name}_minus_{setup.control_group}"
    total_by_document = dict(zip(total_table.document_id, total_table[total_column]))
    summary = {"model_type": setup.model_type, "model_name": setup.model_name, "model_revision": setup.model_revision,
               "persona_name": setup.persona_name, "run_folder": str(run.run_folder), "directions": {}}
    counts_by_role = {}
    for role, direction in setup.direction_names_by_role.items():
        rows = span_table[span_table.rewrite_direction == direction] if len(span_table) else span_table
        if len(rows) == 0:
            continue
        expected_sign = EXPECTED_SIGN_BY_DIRECTION_ROLE.get(role)
        sign_for_counting = expected_sign if expected_sign else 1  # neutral: count moves toward the persona
        changed_minus_unchanged = rows.changed_minus_unchanged.to_numpy(dtype=float)
        total_moves = [total_by_document[row.document_id] - total_by_document[row.source_document_id]
                       for row in rows.itertuples()]
        entry = {
            "direction_role": role, "expected_sign": expected_sign, "number_of_rewrites": len(rows),
            "rewrites_with_no_changed_tokens": int((rows.changed_tokens == 0).sum()),
            "changed_minus_unchanged_positive": int(np.nansum(changed_minus_unchanged > 0)),
            "changed_minus_unchanged_number_moving_expected_way":
                int(np.nansum(changed_minus_unchanged * sign_for_counting > 0)),
            "changed_minus_unchanged_mean": float(np.nanmean(changed_minus_unchanged))
            if not np.all(np.isnan(changed_minus_unchanged)) else float("nan"),
            "changed_minus_unchanged_wilcoxon_p_two_sided": wilcoxon_p_value(changed_minus_unchanged),
            "changed_minus_replaced_positive": int(np.nansum(rows.changed_minus_replaced > 0)),
            "total_log_probability_move_in_expected_direction":
                int(sum(np.sign(move) == sign_for_counting for move in total_moves)),
            "total_log_probability_move_wilcoxon_p_two_sided": wilcoxon_p_value(total_moves)}
        summary["directions"][direction] = entry
        counts_by_role[role] = entry
    summary["verdicts"] = judge_changed_span_criteria(setup, counts_by_role)
    return summary


def judge_changed_span_criteria(setup, counts_by_role):
    criteria = setup.pass_criteria
    verdicts = {}
    positive = criteria.get("changed_span")
    if positive is None:
        verdicts["positive_control"] = {"verdict": VERDICT_NO_CRITERION, "per_direction": {}}
    else:
        per_direction = {}
        for role in positive["applies_to_directions"]:
            entry = counts_by_role.get(role, {"number_of_rewrites": 0, "changed_minus_unchanged_number_moving_expected_way": 0})
            result = judge_count_against_criterion(
                entry["changed_minus_unchanged_number_moving_expected_way"], entry["number_of_rewrites"],
                positive["minimum_number_moving_expected_way"], positive["number_of_rewrites_tested_per_direction"])
            result["direction_name"] = setup.direction_names_by_role[role]
            per_direction[role] = result
        verdicts["positive_control"] = {"verdict": overall_verdict(per_direction), "per_direction": per_direction,
                                        "criterion_status": criteria.get("status")}
    neutral_criterion = (criteria.get("neutral_control") or {}).get("changed_span")
    neutral = counts_by_role.get(DIRECTION_ROLE_NEUTRAL)
    if neutral_criterion is None:
        verdicts["neutral_control"] = {
            "verdict": VERDICT_NO_CRITERION,
            "number_of_rewrites": neutral["number_of_rewrites"] if neutral else 0,
            "number_moving_toward_persona": neutral["changed_minus_unchanged_number_moving_expected_way"] if neutral else None}
    else:
        number_of_rewrites = neutral["number_of_rewrites"] if neutral else 0
        number_toward = neutral["changed_minus_unchanged_number_moving_expected_way"] if neutral else 0
        maximum, pre_registered = (neutral_criterion["maximum_number_moving_toward_persona"],
                                   neutral_criterion["number_of_rewrites_tested"])
        result = {"number_of_rewrites": number_of_rewrites, "number_moving_toward_persona": number_toward,
                  "maximum_allowed": maximum, "number_of_rewrites_pre_registered": pre_registered}
        if number_of_rewrites == 0:
            result.update(verdict=VERDICT_NOT_EVALUABLE, reason="no neutral control rewrites in the run")
        elif number_of_rewrites != pre_registered:
            result.update(verdict=VERDICT_NOT_EVALUABLE,
                          reason=f"{number_of_rewrites} rewrites were scored; the criterion was fixed for {pre_registered}")
        else:
            result.update(verdict=VERDICT_PASS if number_toward <= maximum else VERDICT_FAIL, reason="")
        verdicts["neutral_control"] = result
    return verdicts


def write_changed_span_outputs(run, analysis_folder, timestamp):
    """Runs the analysis on a LoadedRun and writes its files. Returns (summary dict, span_table)."""
    setup = run.setup
    span_table, persona_table, total_table = build_changed_span_tables(run)
    label = loading.insert_model_labels
    if len(span_table):
        label(span_table, setup).to_csv(
            analysis_folder / f"changed_span_{setup.persona_name}_minus_{setup.control_group}_{timestamp}.csv", index=False)
        label(persona_table, setup).to_csv(analysis_folder / f"changed_span_by_persona_{timestamp}.csv", index=False)
    label(total_table, setup).to_csv(analysis_folder / f"total_log_likelihood_differences_{timestamp}.csv", index=False)
    summary = summarize_changed_spans(run, span_table, total_table)
    (analysis_folder / f"changed_span_summary_{timestamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, span_table


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--analysis-folder", type=Path, default=None,
                        help="Existing or new folder to write into (default: <run folder>/analysis_<timestamp>)")
    parser.add_argument("--legacy-run-persona-set", default=None,
                        help="Only for legacy-format run folders: the persona set to pair them with")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    timestamp = timestamp_now()
    run = loading.load_run(arguments.run_folder, arguments.legacy_run_persona_set)
    analysis_folder = arguments.analysis_folder or arguments.run_folder / f"analysis_{timestamp}"
    analysis_folder.mkdir(parents=True, exist_ok=True)
    summary, _ = write_changed_span_outputs(run, analysis_folder, timestamp)
    print(json.dumps(summary, indent=2))
    print(f"Written to {analysis_folder}")
    return analysis_folder


if __name__ == "__main__":
    main()
