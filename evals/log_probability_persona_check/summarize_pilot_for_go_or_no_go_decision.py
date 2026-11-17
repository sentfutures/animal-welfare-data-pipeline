#!/usr/bin/env python3
"""Facts for the go or no-go decision after a pilot scoring run of the log-probability persona check (local and free:
reads files only).

Reads, from the pilot run folder: the run configuration, scoring_outputs/ (environment records, progress logs,
completion records, full-logits verification result), the newest analysis_<timestamp>/ folder (run
analyze_persona_scores.py first) and the pod cost logs. Writes pilot_go_or_no_go_summary_<timestamp>.md and .json into
the run folder and prints the markdown. It reports facts and whether each explicit criterion is met; it makes no
recommendation.

The full-run projection is a range. Its size is the pilot's mean tokens per original document times the number of
documents in the full corpus (--full-run-number-of-documents, or counted from the run's corpus file) times the number
of conditions. Throughput is taken from the scorer sessions: the range runs from the slowest of (slowest session,
mean minus --throughput-uncertainty-fraction) to the fastest of (fastest session, mean plus that fraction). The
fraction is an assumption, not a measurement. Setup time per pod is the pilot's (pod running time minus scoring time).

Usage, from the repository root:
    python evals/log_probability_persona_check/summarize_pilot_for_go_or_no_go_decision.py --run-folder <pilot run folder>
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check import analysis_run_loading as loading  # noqa: E402
from evals.log_probability_persona_check.build_scoring_inputs import timestamp_now  # noqa: E402
from evals.log_probability_persona_check.legacy_run_format_compatibility_reader import (  # noqa: E402
    rename_legacy_condition_groups)
from shared import utils  # noqa: E402

DEFAULT_THROUGHPUT_UNCERTAINTY_FRACTION = 0.25
DEFAULT_MINIMUM_MEAN_PAIRWISE_SPEARMAN = 0.5
DEFAULT_MAXIMUM_OUTLIER_FRACTION = 0.10
PROGRESS_LINE_PATTERN = re.compile(r"elapsed (\d+)h(\d+)m(\d+)s;.*?([\d,]+) scored document tokens/s")
MET, NOT_MET, NOT_EVALUABLE = "MET", "NOT MET", "NOT EVALUABLE"


def newest_file(folder, pattern):
    paths = sorted(Path(folder).glob(pattern))
    return paths[-1] if paths else None


def read_json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_hours_and_minutes(seconds):
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def number_text(value, template="{:.3f}"):
    return "not available" if value is None or (isinstance(value, float) and math.isnan(value)) else template.format(value)


# ---------------------------------------------------------------------------------------------------
# Facts from the run folder
# ---------------------------------------------------------------------------------------------------

def read_scoring_sessions(scoring_outputs):
    """One {elapsed_seconds, scored_tokens_per_second} per scorer session: the scoring environment record's
    throughput when it has one, otherwise the last line of the matching progress log."""
    sessions = []
    for path in sorted(scoring_outputs.glob("scoring_environment_*.json")):
        throughput = read_json_file(path).get("throughput")
        if throughput and throughput.get("scoring_seconds"):
            sessions.append({"source_file": path.name, "elapsed_seconds": float(throughput["scoring_seconds"]),
                             "scored_tokens_per_second": float(throughput["document_tokens_scored_per_second"])})
    if sessions:
        return sessions
    for path in sorted(scoring_outputs.glob("scoring_progress_log_*.txt")):
        matches = [PROGRESS_LINE_PATTERN.search(line) for line in path.read_text(encoding="utf-8").splitlines()]
        matches = [match for match in matches if match]
        if matches:
            hours, minutes, seconds, rate = matches[-1].groups()
            sessions.append({"source_file": path.name, "elapsed_seconds": int(hours) * 3600 + int(minutes) * 60 + int(seconds),
                             "scored_tokens_per_second": float(rate.replace(",", ""))})
    return sessions


def read_pod_cost_logs(run_folder):
    logs = []
    for path in sorted(Path(run_folder).glob("pod_cost_log_*.json")):
        record = read_json_file(path)
        record["file_name"] = path.name
        record["running_seconds"] = (
            (datetime.fromisoformat(record["stop_time"]) - datetime.fromisoformat(record["start_time"])).total_seconds()
            if record.get("start_time") and record.get("stop_time") else None)
        logs.append(record)
    mark_logs_covered_by_a_later_attach_log(logs)
    return logs


def mark_logs_covered_by_a_later_attach_log(logs):
    """A log written with --attach-to-existing-pod counts from the start of the pod's earlier log, so it already holds
    the time and cost of every earlier log of the same pod that lies inside its own period. Those earlier logs get
    `covered_by_a_later_attach_log: true` and are left out of the totals; logs of separate uses of a pod (a reuse)
    are not covered by an attach and still add up."""
    for position, log in enumerate(logs):
        log["covered_by_a_later_attach_log"] = False
        if not (log.get("start_time") and log.get("stop_time")):
            continue
        for later_log in logs[position + 1:]:
            if (later_log.get("attached_to_existing_pod") and later_log.get("pod_id") == log.get("pod_id")
                    and later_log.get("start_time") and later_log.get("stop_time")
                    and later_log["start_time"] <= log["start_time"] and log["stop_time"] <= later_log["stop_time"]):
                log["covered_by_a_later_attach_log"] = True


def count_full_run_documents(run_configuration, explicit_count, corpus_path):
    """(number of documents, description of the source) or (None, reason)."""
    if explicit_count:
        return explicit_count, "given on the command line"
    corpus_path = corpus_path or (run_configuration.get("document_selection") or {}).get("corpus_path")
    if corpus_path and Path(corpus_path).exists():
        return len(utils.load_jsonl(corpus_path)), f"records in {Path(corpus_path).name}"
    return None, "no corpus file found; pass --full-run-number-of-documents"


def project_full_run(facts, throughput_uncertainty_fraction):
    """Fills facts['full_run'] with a low and a high estimate of scoring time, total time and cost."""
    full_run = facts["full_run"]
    sessions, mean_rate = facts["scoring_sessions"], facts["scored_tokens_per_second"]
    mean_tokens = facts["pilot_mean_tokens_per_original_document"]
    ready = (full_run["number_of_documents"] and mean_tokens and mean_rate and facts["pod_cost_per_hour_dollars"] is not None
             and facts["setup_seconds_per_pod"] is not None)
    if not ready:
        full_run["projection_available"] = False
        return
    rates = [session["scored_tokens_per_second"] for session in sessions]
    slowest = min(min(rates), mean_rate * (1 - throughput_uncertainty_fraction))
    fastest = max(max(rates), mean_rate * (1 + throughput_uncertainty_fraction))
    full_run["document_tokens"] = full_run["number_of_documents"] * mean_tokens
    full_run["scored_tokens"] = full_run["document_tokens"] * facts["number_of_conditions"]
    full_run["projection_available"] = True
    full_run["throughput_range_scored_tokens_per_second"] = [slowest, fastest]
    for label, rate in (("low", fastest), ("high", slowest)):
        scoring_seconds = full_run["scored_tokens"] / rate
        total_seconds = scoring_seconds + facts["setup_seconds_per_pod"]
        full_run[label] = {"scoring_seconds": scoring_seconds, "total_seconds": total_seconds,
                           "cost_dollars": total_seconds / 3600 * facts["pod_cost_per_hour_dollars"]}


def summarize_rewording_agreement(rewording_agreement):
    personas = []
    for persona_name, rows in rewording_agreement.groupby("persona_name", sort=False):
        first = rows.iloc[0]
        outliers = rows[rows["rewording_is_outlier"].astype(bool)]
        personas.append({
            "persona_name": persona_name, "condition_group": first["condition_group"],
            "number_of_rewordings": int(len(rows)),
            "mean_pairwise_spearman": float(first["persona_mean_pairwise_spearman_across_documents"]),
            "intraclass_correlation": float(first["persona_intraclass_correlation_one_way_single_rewording"]),
            "within_to_across_sd_ratio": float(first["persona_within_to_across_sd_ratio"]),
            "outliers": [{"condition_id": row["condition_id"],
                          "leave_one_out_spearman": float(row["rewording_spearman_with_leave_one_out_mean_of_other_rewordings"])}
                         for _, row in outliers.iterrows()]})
    return personas


def collect_facts(run_folder, full_run_number_of_documents=None, corpus_path=None,
                  throughput_uncertainty_fraction=DEFAULT_THROUGHPUT_UNCERTAINTY_FRACTION,
                  minimum_mean_pairwise_spearman=DEFAULT_MINIMUM_MEAN_PAIRWISE_SPEARMAN,
                  maximum_outlier_fraction=DEFAULT_MAXIMUM_OUTLIER_FRACTION, total_budget_dollars=None,
                  rewrite_generation_cost_dollars=None, legacy_run_persona_set=None):
    run_folder = Path(run_folder)
    scoring_outputs = run_folder / "scoring_outputs"
    configurations, is_legacy_format = loading.read_run_configurations(run_folder, legacy_run_persona_set)
    setup = loading.build_setup(configurations, run_folder)
    conditions = utils.load_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    if is_legacy_format:
        conditions = rename_legacy_condition_groups(conditions, setup.configuration)
    documents = utils.load_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    records = utils.load_jsonl(scoring_outputs / "scoring_completion_records.jsonl")
    persona_conditions_by_group = {}
    for condition in conditions:
        if condition["condition_group"] in setup.persona_group_names:
            counts = persona_conditions_by_group.setdefault(condition["condition_group"], {})
            counts[condition["persona_name"]] = counts.get(condition["persona_name"], 0) + 1
    facts = {"run_folder": str(run_folder), "persona_conditions_by_group": persona_conditions_by_group, "persona_set_name": setup.persona_set_name, "persona_name": setup.persona_name,
             "model_type": setup.model_type, "model_name": setup.model_name, "model_revision": setup.model_revision,
             "number_of_documents_in_run": len(documents), "number_of_conditions": len(conditions),
             "number_of_documents_scored": len({record["document_id"] for record in records}),
             "scored_tokens_in_completion_records": sum(record["number_of_document_tokens"] for record in records)}
    environment_path = newest_file(scoring_outputs, "scoring_environment_*.json")
    facts["environment"] = read_json_file(environment_path) if environment_path else None
    verification_path = newest_file(scoring_outputs, "full_logits_verification_*.json")
    facts["verification"] = read_json_file(verification_path) if verification_path else None

    original_ids = {document["document_id"] for document in documents if document.get("rewrite_direction") is None}
    original_token_counts = [record["number_of_document_tokens"] for record in records
                             if record["condition_id"] == loading.GENERIC_CONDITION_ID and record["document_id"] in original_ids]
    facts["pilot_mean_tokens_per_original_document"] = (sum(original_token_counts) / len(original_token_counts)
                                                        if original_token_counts else None)
    sessions = read_scoring_sessions(scoring_outputs)
    scoring_seconds = sum(session["elapsed_seconds"] for session in sessions)
    facts["scoring_sessions"] = sessions
    facts["scoring_seconds"] = scoring_seconds
    facts["scored_tokens_per_second"] = (sum(s["elapsed_seconds"] * s["scored_tokens_per_second"] for s in sessions)
                                         / scoring_seconds if scoring_seconds > 0 else None)

    pod_logs = read_pod_cost_logs(run_folder)
    facts["pod_cost_logs"] = pod_logs
    counted_logs = [log for log in pod_logs if not log["covered_by_a_later_attach_log"]]
    facts["pod_cost_spent_dollars"] = sum(log.get("total_cost_dollars") or 0.0 for log in counted_logs)
    prices = [log["hourly_price_dollars"] for log in pod_logs if log.get("hourly_price_dollars") is not None]
    facts["pod_cost_per_hour_dollars"] = prices[-1] if prices else None
    running_seconds = [log["running_seconds"] for log in counted_logs if log["running_seconds"] is not None]
    facts["pod_running_seconds"] = sum(running_seconds) if running_seconds else None
    # Everything on a pod that is not scoring (boot, upload, install, model load, pulls), per pod, so a resume on a
    # second pod does not double the allowance.
    facts["setup_seconds_per_pod"] = (max(0.0, facts["pod_running_seconds"] - scoring_seconds) / len(running_seconds)
                                      if running_seconds else None)
    facts["rewrite_generation_cost_dollars"] = rewrite_generation_cost_dollars
    facts["total_spent_so_far_dollars"] = facts["pod_cost_spent_dollars"] + (rewrite_generation_cost_dollars or 0.0)

    number_of_documents, source = count_full_run_documents(configurations[0], full_run_number_of_documents, corpus_path)
    facts["full_run"] = {"number_of_documents": number_of_documents, "number_of_documents_source": source}
    project_full_run(facts, throughput_uncertainty_fraction)

    analysis_folder = newest_file(run_folder, "analysis_*")
    facts["analysis_folder"] = str(analysis_folder) if analysis_folder else None
    facts["verdicts"], facts["rewording_agreement"], facts["span_significance"] = None, [], None
    if analysis_folder:
        verdicts_path = newest_file(analysis_folder, "control_verdicts_*.json")
        facts["verdicts"] = read_json_file(verdicts_path) if verdicts_path else None
        agreement_path = newest_file(analysis_folder, "rewording_agreement_*.csv")
        if agreement_path:
            facts["rewording_agreement"] = summarize_rewording_agreement(pd.read_csv(agreement_path))
        span_path = newest_file(analysis_folder, "span_significance_tests_*.csv")
        if span_path:
            table = pd.read_csv(span_path)
            originals = table[~table["q_value_is_uncorrected_rewrite"].astype(bool)]
            facts["span_significance"] = {
                "number_of_original_documents_tested": int(len(originals)),
                "highest_window_q_below_0.05": int((originals["highest_window_q_value"] < 0.05).sum()),
                "lowest_window_q_below_0.05": int((originals["lowest_window_q_value"] < 0.05).sum())}
    facts["criteria"] = evaluate_criteria(facts, setup, minimum_mean_pairwise_spearman, maximum_outlier_fraction,
                                          total_budget_dollars)
    return facts


# ---------------------------------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------------------------------

def status_of_verdict(verdict):
    return {"pass": MET, "fail": NOT_MET}.get(verdict, NOT_EVALUABLE)


def evaluate_criteria(facts, setup, minimum_mean_pairwise_spearman, maximum_outlier_fraction, total_budget_dollars):
    """List of (description, status, detail). The control criteria are the pre-registered ones in the run configuration;
    the rewording agreement and budget thresholds are the summary's own and are named as such."""
    criteria = []
    verification = facts["verification"]
    if verification is None:
        criteria.append(("Full-logits verification passed", NOT_EVALUABLE, "no verification result file found"))
    else:
        criteria.append(("Full-logits verification passed", MET if verification["passed"] else NOT_MET,
                         f"maximum absolute difference {verification['maximum_absolute_difference']:.5f}, mean "
                         f"{verification['mean_absolute_difference']:.5f}"))
    verdicts = facts["verdicts"]
    for level, key in (("whole-document", "whole_document"), ("changed-span", "changed_span")):
        block = ((verdicts or {}).get(key) or {}).get("positive_control")
        for role, result in (block or {}).get("per_direction", {}).items():
            criteria.append((f"Positive control, {level}: at least {result['minimum_required']} of "
                             f"{result['number_of_rewrites_pre_registered']} '{result['direction_name']}' rewrites move the "
                             f"expected way (pre-registered)", status_of_verdict(result["verdict"]),
                             f"{result['number_moving_expected_way']} of {result['number_of_rewrites']}"
                             + (f"; {result['reason']}" if result["reason"] else "")))
        if not block or not block.get("per_direction"):
            criteria.append((f"Positive control, {level}", NOT_EVALUABLE,
                             "no analysis result" if block is None else block.get("verdict", "")))
    neutral = ((verdicts or {}).get("changed_span") or {}).get("neutral_control")
    if neutral is None:
        criteria.append(("Neutral control, changed-span", NOT_EVALUABLE, "no analysis result"))
    elif "maximum_allowed" not in neutral:
        criteria.append(("Neutral control, changed-span: no pre-registered criterion in the persona set configuration",
                         NOT_EVALUABLE, f"{neutral['number_of_rewrites']} neutral rewrites scored; "
                         f"{neutral['number_moving_toward_persona']} move toward the persona"))
    else:
        criteria.append((f"Neutral control, changed-span: at most {neutral['maximum_allowed']} of "
                         f"{neutral['number_of_rewrites_pre_registered']} neutral rewrites move toward the persona (pre-registered)",
                         status_of_verdict(neutral["verdict"]),
                         f"{neutral['number_moving_toward_persona']} of {neutral['number_of_rewrites']}"
                         + (f"; {neutral['reason']}" if neutral.get("reason") else "")))
    target = next((p for p in facts["rewording_agreement"] if p["condition_group"] == setup.target_group), None)
    description = (f"{setup.persona_name} rewording agreement not flagged as poor (summary threshold, not pre-registered: "
                   f"mean pairwise Spearman at least {minimum_mean_pairwise_spearman} and at most "
                   f"{maximum_outlier_fraction:.0%} of rewordings outliers)")
    if target is None:
        criteria.append((description, NOT_EVALUABLE, "no rewording agreement result"))
    else:
        outlier_fraction = len(target["outliers"]) / target["number_of_rewordings"]
        spearman = target["mean_pairwise_spearman"]
        met = not math.isnan(spearman) and spearman >= minimum_mean_pairwise_spearman and outlier_fraction <= maximum_outlier_fraction
        criteria.append((description, MET if met else NOT_MET,
                         f"mean pairwise Spearman {number_text(spearman)}; {len(target['outliers'])} of "
                         f"{target['number_of_rewordings']} rewordings are outliers"))
    if total_budget_dollars is not None:
        full_run = facts["full_run"]
        description = f"Projected full-run cost plus spent so far under {total_budget_dollars:.0f} dollars (high estimate)"
        if not full_run.get("projection_available"):
            criteria.append((description, NOT_EVALUABLE, "projection not possible (missing size, throughput, price or pod timing)"))
        else:
            total = full_run["high"]["cost_dollars"] + facts["total_spent_so_far_dollars"]
            criteria.append((description, MET if total < total_budget_dollars else NOT_MET,
                             f"{full_run['high']['cost_dollars']:.2f} projected + {facts['total_spent_so_far_dollars']:.2f} "
                             f"spent = {total:.2f} dollars"))
    return criteria


# ---------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------

def render_verdict_lines(label, block):
    lines = []
    if not block:
        return [f"- {label}: no result"]
    for role, result in block.get("per_direction", {}).items():
        lines.append(f"- {label}, '{result['direction_name']}' rewrites: {result['number_moving_expected_way']} of "
                     f"{result['number_of_rewrites']} move the expected way (at least {result['minimum_required']} needed "
                     f"for {result['number_of_rewrites_pre_registered']} rewrites): {result['verdict']}"
                     + (f" ({result['reason']})" if result["reason"] else ""))
    if not block.get("per_direction"):
        lines.append(f"- {label}: {block.get('verdict')}")
    return lines


def render_markdown(facts):
    lines = [f"# Pilot go or no-go summary: {facts['persona_set_name']}, {facts['model_type']} model {facts['model_name']}", "",
             f"Run folder: `{facts['run_folder']}`", f"Analysis folder: `{facts['analysis_folder']}`",
             f"Model revision: `{facts['model_revision']}`", ""]
    environment = facts["environment"] or {}
    lines += ["## Run", f"- Persona: {facts['persona_name']} (persona set {facts['persona_set_name']})",
              f"- GPU: {environment.get('gpu_name')}; dtype {environment.get('dtype')}; torch {environment.get('torch_version')}; "
              f"transformers {environment.get('transformers_version')}",
              f"- Documents scored: {facts['number_of_documents_scored']} of {facts['number_of_documents_in_run']}; "
              f"conditions: {facts['number_of_conditions']}"]
    for group, counts in facts["persona_conditions_by_group"].items():
        lines.append(f"- Condition group `{group}` (pooled over its personas): {sum(counts.values())} conditions; "
                     f"by persona: " + ", ".join(f"{name} {count}" for name, count in counts.items()))
    lines.append("")

    verification = facts["verification"]
    lines.append("## (a) Full-logits verification")
    if verification is None:
        lines.append("- No verification result file in scoring_outputs/.")
    else:
        lines.append(f"- Passed: {verification['passed']} (document {verification['document_id']}, condition "
                     f"{verification['condition_id']}, {verification['number_of_document_tokens']} tokens)")
        lines.append(f"- Maximum absolute difference {verification['maximum_absolute_difference']:.5f} (allowed "
                     f"{verification['maximum_allowed_maximum_difference']}); mean absolute difference "
                     f"{verification['mean_absolute_difference']:.5f} (allowed {verification['maximum_allowed_mean_difference']})")
    lines.append("")

    full_run = facts["full_run"]
    lines += ["## (b) Throughput, cost and full-run projection",
              f"- Scoring time: {format_hours_and_minutes(facts['scoring_seconds'])} over {len(facts['scoring_sessions'])} scorer session(s)",
              f"- Measured throughput: {number_text(facts['scored_tokens_per_second'], '{:,.0f}')} scored tokens per second "
              f"(document tokens x conditions); scored tokens in completion records: {facts['scored_tokens_in_completion_records']:,}",
              f"- Pod cost per hour: {number_text(facts['pod_cost_per_hour_dollars'], '{:.2f}')} dollars; pod cost so far: "
              f"{facts['pod_cost_spent_dollars']:.2f} dollars; rewrite generation cost: "
              + ("not given" if facts["rewrite_generation_cost_dollars"] is None
                 else f"{facts['rewrite_generation_cost_dollars']:.2f} dollars"),
              f"- Pilot setup duration per pod (pod running time minus scoring time): "
              f"{number_text(facts['setup_seconds_per_pod'], '{:.0f}')} seconds",
              f"- Pilot mean tokens per original document: {number_text(facts['pilot_mean_tokens_per_original_document'], '{:,.0f}')}",
              f"- Full run: {full_run['number_of_documents'] or 'unknown number of'} documents ({full_run['number_of_documents_source']}) "
              f"x {facts['number_of_conditions']} conditions"]
    if not full_run.get("projection_available"):
        lines.append("- Full-run projection not possible: document count, token count, throughput, price or pod timing missing")
    else:
        low, high = full_run["low"], full_run["high"]
        slowest, fastest = full_run["throughput_range_scored_tokens_per_second"]
        lines.append(f"- Scored tokens {full_run['scored_tokens']:,.0f} (pilot mean tokens per original document times the full document count)")
        lines.append(f"- Throughput range used: {slowest:,.0f} to {fastest:,.0f} scored tokens per second")
        lines.append(f"- Projected full-run time: {format_hours_and_minutes(low['total_seconds'])} to "
                     f"{format_hours_and_minutes(high['total_seconds'])} (scoring {format_hours_and_minutes(low['scoring_seconds'])} to "
                     f"{format_hours_and_minutes(high['scoring_seconds'])} plus setup)")
        lines.append(f"- Projected full-run cost: {low['cost_dollars']:.2f} to {high['cost_dollars']:.2f} dollars")
    lines.append("")

    verdicts = facts["verdicts"] or {}
    lines.append("## (c) Positive and neutral controls (pre-registered criteria from the persona set configuration)")
    if not verdicts:
        lines.append("- No control verdicts in the analysis folder")
    else:
        if verdicts.get("pass_criteria", {}).get("status"):
            lines.append(f"- Criteria status: {verdicts['pass_criteria']['status']}")
        lines += render_verdict_lines("Whole document", (verdicts.get("whole_document") or {}).get("positive_control"))
        lines += render_verdict_lines("Changed span", ((verdicts.get("changed_span") or {}).get("positive_control")))
        neutral = (verdicts.get("changed_span") or {}).get("neutral_control")
        if neutral:
            if "maximum_allowed" in neutral:
                lines.append(f"- Neutral control, changed span: {neutral['number_moving_toward_persona']} of "
                             f"{neutral['number_of_rewrites']} rewrites move toward the persona (at most {neutral['maximum_allowed']} "
                             f"allowed for {neutral['number_of_rewrites_pre_registered']} rewrites): {neutral['verdict']}"
                             + (f" ({neutral['reason']})" if neutral.get("reason") else ""))
            else:
                lines.append(f"- Neutral control, changed span: {neutral['verdict']}; {neutral['number_of_rewrites']} rewrites scored")
    lines.append("")

    lines.append("## (d) Rewording agreement (across all documents in the run, rewrites included)")
    lines.append("| Persona | Group | Rewordings | Mean pairwise Spearman | ICC | Within/across SD ratio | Outlier rewordings |")
    lines.append("|---|---|---|---|---|---|---|")
    for persona in facts["rewording_agreement"]:
        outliers = ", ".join(f"{o['condition_id']} ({o['leave_one_out_spearman']:.2f})" for o in persona["outliers"]) or "none"
        lines.append(f"| {persona['persona_name']} | {persona['condition_group']} | {persona['number_of_rewordings']} | "
                     f"{number_text(persona['mean_pairwise_spearman'])} | {number_text(persona['intraclass_correlation'])} | "
                     f"{number_text(persona['within_to_across_sd_ratio'])} | {outliers} |")
    lines.append("")

    lines.append("## (e) Span significance (permutation test over windows, original documents, Benjamini-Hochberg q < 0.05)")
    span = facts["span_significance"]
    if span is None:
        lines.append("- No span significance result in the analysis folder")
    else:
        lines.append(f"- Original documents tested: {span['number_of_original_documents_tested']}; highest window significant in "
                     f"{span['highest_window_q_below_0.05']}, lowest window significant in {span['lowest_window_q_below_0.05']}")
    lines.append("")

    lines.append("## (f) Pods")
    for log in facts["pod_cost_logs"]:
        lines.append(f"- Pod {log.get('pod_id')} ({log.get('file_name')}): GPU {log.get('gpu_type')}; "
                     f"{number_text(log.get('hourly_price_dollars'), '{:.2f}')} dollars per hour; cost "
                     f"{number_text(log.get('total_cost_dollars'), '{:.2f}')} dollars; termination reason {log.get('termination_reason')}"
                     + ("; included in the totals of the later log that attached to this pod, not counted again"
                        if log["covered_by_a_later_attach_log"] else ""))
    if not facts["pod_cost_logs"]:
        lines.append("- No pod cost log found")
    lines.append("")

    lines.append("## (g) Criteria")
    for description, status, detail in facts["criteria"]:
        lines.append(f"- {status}: {description}; {detail}")
    lines.append("")
    return "\n".join(lines)


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-folder", required=True, type=Path)
    parser.add_argument("--full-run-number-of-documents", type=int, default=None,
                        help="Documents in the full corpus (default: counted from the run's corpus file)")
    parser.add_argument("--corpus-path", default=None, help="Corpus file to count (default: the run configuration's)")
    parser.add_argument("--throughput-uncertainty-fraction", type=float, default=DEFAULT_THROUGHPUT_UNCERTAINTY_FRACTION)
    parser.add_argument("--minimum-mean-pairwise-spearman", type=float, default=DEFAULT_MINIMUM_MEAN_PAIRWISE_SPEARMAN)
    parser.add_argument("--maximum-outlier-fraction", type=float, default=DEFAULT_MAXIMUM_OUTLIER_FRACTION)
    parser.add_argument("--total-budget-dollars", type=float, default=None,
                        help="Adds a budget criterion: projected high cost plus spent so far under this")
    parser.add_argument("--rewrite-generation-cost-dollars", type=float, default=None,
                        help="Cost of the control rewrite calls, to include in the spent total")
    parser.add_argument("--legacy-run-persona-set", default=None,
                        help="Only for legacy-format run folders: the persona set to pair them with")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    facts = collect_facts(
        arguments.run_folder, arguments.full_run_number_of_documents, arguments.corpus_path,
        arguments.throughput_uncertainty_fraction, arguments.minimum_mean_pairwise_spearman,
        arguments.maximum_outlier_fraction, arguments.total_budget_dollars, arguments.rewrite_generation_cost_dollars,
        arguments.legacy_run_persona_set)
    markdown = render_markdown(facts)
    timestamp = timestamp_now()
    output_path = arguments.run_folder / f"pilot_go_or_no_go_summary_{timestamp}.md"
    output_path.write_text(markdown, encoding="utf-8")
    (arguments.run_folder / f"pilot_go_or_no_go_summary_{timestamp}.json").write_text(
        json.dumps(facts, indent=2, default=str), encoding="utf-8")
    print(markdown)
    print(f"Written to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
