#!/usr/bin/env python3
"""Post-process a plausibility audit run folder into per-document scores, per-problem rows and summaries.

Offline: makes no model calls.

Joins (offline, never sent to the model) each document's language, type and variables, and verifies each
quote against the document. A document whose judgment failed gets empty score cells, never default scores.
Inapplicable axes are excluded from every mean.

Usage: python evals/plausibility_audit_postprocess_sdf.py <run_folder>
"""

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.model_judged_audit_support import (  # noqa: E402
    read_all_records, read_jsonl, timestamp_now, verify_quote, write_jsonl, write_rows_to_csv)
from evals.plausibility_audit_sdf import (  # noqa: E402
    PROMPT_DIRECTORY, SCORE_VALUES, SEVERITY_VALUES, instructions_file_path, load_existing_records,
    parse_axis_keys, read_text)

SEVERITY_RANK = {severity: rank for rank, severity in enumerate(SEVERITY_VALUES)}


def parse_variables(raw_variables):
    """Variables come as a dict or as a Python-dict string. Parsed with ast.literal_eval, never eval."""
    if isinstance(raw_variables, dict):
        return raw_variables
    if isinstance(raw_variables, str):
        try:
            parsed = ast.literal_eval(raw_variables)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"cannot parse variables string: {raw_variables[:80]!r}") from error
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"variables is neither a dict nor a dict string: {type(raw_variables).__name__}")


def worst_severity(problems):
    """Most severe severity among the problems, or None when there are none."""
    if not problems:
        return None
    return max((problem["severity"] for problem in problems), key=lambda value: SEVERITY_RANK[value])


def document_context(document):
    """Offline-only joined fields for a document: language, type and every variable as a column."""
    variables = parse_variables(document["variables"])
    context = {"language": document.get("language"),
               "document_type": variables.get("document_type", document.get("type_name"))}
    for name, value in variables.items():
        context[name] = value
    return context


def axis_columns(axis_key):
    return [f"{axis_key}__score", f"{axis_key}__applicable", f"{axis_key}__problem_count",
            f"{axis_key}__worst_severity"]


def build_per_document_rows(records, error_records, documents_by_id, axis_keys):
    """One row per attempted document. Failed documents have judgment_status 'failed' and empty cells."""
    records_by_document = {record["doc_id"]: record for record in records}
    attempted_ids = sorted(records_by_document.keys() | {error["doc_id"] for error in error_records})
    rows = []
    for doc_id in attempted_ids:
        row = {"doc_id": doc_id}
        row.update(document_context(documents_by_id[doc_id]))
        record = records_by_document.get(doc_id)
        row["judgment_status"] = "judged" if record else "failed"
        row["overall_plausibility"] = record["parsed_result"]["overall_plausibility"] if record else None
        for axis_key in axis_keys:
            axis = record["parsed_result"]["axes"][axis_key] if record else None
            values = ((axis["score"], axis["applicable"], len(axis["problems"]), worst_severity(axis["problems"]))
                      if axis else (None, None, None, None))
            row.update(dict(zip(axis_columns(axis_key), values)))
        rows.append(row)
    return rows


def build_problem_rows(records, documents_by_id):
    rows = []
    for record in records:
        document = documents_by_id[record["doc_id"]]
        context = document_context(document)
        for axis_key, axis in record["parsed_result"]["axes"].items():
            for problem_index, problem in enumerate(axis["problems"]):
                rows.append(build_quoted_row(record["doc_id"], document, context,
                                             {"axis": axis_key, "problem_index": problem_index}, problem))
    return rows


def build_other_implausibility_rows(records, documents_by_id):
    rows = []
    for record in records:
        document = documents_by_id[record["doc_id"]]
        context = document_context(document)
        for entry_index, entry in enumerate(record["parsed_result"]["other_implausibilities"]):
            rows.append(build_quoted_row(record["doc_id"], document, context, {"entry_index": entry_index}, entry))
    return rows


def build_quoted_row(doc_id, document, context, position_fields, entry):
    found, exact, fuzzy_score, spans_break = verify_quote(entry["quote"], document["content"])
    row = {"doc_id": doc_id}
    row.update(position_fields)
    row.update(entry)
    row.update({"quote_found_in_document": found, "quote_found_exactly_after_normalization": exact,
                "quote_fuzzy_match_score": fuzzy_score, "quote_spans_paragraph_break": spans_break})
    row.update(context)
    return row


def mean_or_none(values):
    return round(sum(values) / len(values), 4) if values else None


def build_per_axis_summary(records, axis_keys):
    summary = []
    for axis_key in axis_keys:
        axes = [record["parsed_result"]["axes"][axis_key] for record in records]
        applicable_axes = [axis for axis in axes if axis["applicable"]]
        score_counts = Counter(axis["score"] for axis in applicable_axes)
        severity_counts = Counter(problem["severity"] for axis in axes for problem in axis["problems"])
        entry = {"axis": axis_key, "documents_judged": len(axes), "documents_applicable": len(applicable_axes),
                 "mean_score_among_applicable": mean_or_none([axis["score"] for axis in applicable_axes])}
        for score in SCORE_VALUES:
            entry[f"applicable_documents_with_score_{score}"] = score_counts[score]
        for severity in SEVERITY_VALUES:
            entry[f"problems_with_severity_{severity}"] = severity_counts[severity]
        summary.append(entry)
    return summary


def build_per_variable_value_axis_means(records, documents_by_id, axis_keys):
    """For each variable and value: judged documents, overall mean, and per axis the mean score among
    applicable documents (plus how many were applicable). Failed documents are not included."""
    groups = {}
    for record in records:
        for name, value in parse_variables(documents_by_id[record["doc_id"]]["variables"]).items():
            groups.setdefault((name, str(value)), []).append(record["parsed_result"])
    rows = []
    for (name, value), results in sorted(groups.items()):
        row = {"variable": name, "value": value, "document_count": len(results),
               "overall_plausibility_mean": mean_or_none([result["overall_plausibility"] for result in results])}
        for axis_key in axis_keys:
            applicable_scores = [result["axes"][axis_key]["score"] for result in results
                                 if result["axes"][axis_key]["applicable"]]
            row[f"{axis_key}__applicable_document_count"] = len(applicable_scores)
            row[f"{axis_key}__mean_score_among_applicable"] = mean_or_none(applicable_scores)
        rows.append(row)
    return rows


def write_jsonl_and_csv(run_folder, base_name, timestamp, rows, preferred_columns=()):
    jsonl_path = Path(run_folder) / f"{base_name}_{timestamp}.jsonl"
    csv_path = Path(run_folder) / f"{base_name}_{timestamp}.csv"
    write_jsonl(jsonl_path, rows)
    write_rows_to_csv(csv_path, rows, preferred_columns)
    return jsonl_path, csv_path


def postprocess_run_folder(run_folder):
    """Writes every output file into the run folder; returns a dict of label -> path, and the row lists."""
    run_folder = Path(run_folder)
    configuration = json.loads(sorted(run_folder.glob("run_configuration_*.json"))[0].read_text(encoding="utf-8"))
    axis_keys = configuration.get("axis_keys") or parse_axis_keys(read_text(instructions_file_path(PROMPT_DIRECTORY)))
    documents_by_id = {d["doc_id"]: d for d in read_jsonl(configuration["corpus_path"])}
    records = load_existing_records(run_folder)
    error_records = read_all_records(run_folder, "errors_*.jsonl")
    judged_ids = {record["doc_id"] for record in records}
    error_records = [error for error in error_records if error["doc_id"] not in judged_ids]  # later success wins
    timestamp = timestamp_now()

    per_document_rows = build_per_document_rows(records, error_records, documents_by_id, axis_keys)
    problem_rows = build_problem_rows(records, documents_by_id)
    other_rows = build_other_implausibility_rows(records, documents_by_id)
    axis_summary_rows = build_per_axis_summary(records, axis_keys)
    variable_rows = build_per_variable_value_axis_means(records, documents_by_id, axis_keys)

    paths = {}
    paths["per_document_scores_jsonl"], paths["per_document_scores_csv"] = write_jsonl_and_csv(
        run_folder, "per_document_plausibility_scores", timestamp, per_document_rows,
        ["doc_id", "language", "document_type", "judgment_status", "overall_plausibility"])
    problem_columns = ["doc_id", "axis", "severity", "quote", "english_gloss", "explanation",
                       "quote_found_in_document", "quote_spans_paragraph_break", "language", "document_type"]
    paths["problems_jsonl"], paths["problems_csv"] = write_jsonl_and_csv(
        run_folder, "plausibility_problems", timestamp, problem_rows, problem_columns)
    paths["other_implausibilities_jsonl"], paths["other_implausibilities_csv"] = write_jsonl_and_csv(
        run_folder, "other_implausibilities", timestamp,
        sorted(other_rows, key=lambda row: row["suggested_axis_name"].lower()),
        ["suggested_axis_name", "description", "doc_id", "severity", "quote", "english_gloss", "explanation",
         "quote_found_in_document", "quote_spans_paragraph_break", "language", "document_type"])
    paths["per_axis_summary_csv"] = run_folder / f"per_axis_summary_{timestamp}.csv"
    write_rows_to_csv(paths["per_axis_summary_csv"], axis_summary_rows)
    paths["per_variable_value_axis_means_csv"] = run_folder / f"per_variable_value_axis_means_{timestamp}.csv"
    write_rows_to_csv(paths["per_variable_value_axis_means_csv"], variable_rows, ["variable", "value", "document_count"])
    return paths, {"per_document": per_document_rows, "problems": problem_rows, "other": other_rows,
                   "axis_summary": axis_summary_rows, "variable_means": variable_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_folder")
    arguments = parser.parse_args()
    output_paths, output_rows = postprocess_run_folder(arguments.run_folder)
    print(f"{len(output_rows['per_document'])} documents, {len(output_rows['problems'])} problems, "
          f"{len(output_rows['other'])} other implausibilities")
    for label, path in output_paths.items():
        print(f"{label} -> {path}")
