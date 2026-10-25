#!/usr/bin/env python3
"""Merges a finished run (the base run) with a persona-subset run (built with build_scoring_inputs.py
--only-personas) into a new pipeline-format run folder that holds every condition of the persona set, so that
analyze_persona_scores.py runs on it unchanged. Local and free: reads and writes score files only.

Why: when a persona set gains conditions (for the preachiness set, two more same-values personas), the conditions the
base run already scored need not be scored again. The subset run scores `generic` and the missing conditions, with
the same model, revision, prompts and documents.

Usage (from the repository root):
  python evals/log_probability_persona_check/merge_persona_subset_run_into_full_run.py \\
      --base-run-folder <finished run> --subset-run-folder <subset run> --output-run-folder <new folder> \\
      [--base-run-legacy-persona-set preachiness]

The base run may be in the pipeline format or, with --base-run-legacy-persona-set, the legacy format (read through
legacy_run_format_compatibility_reader.py). Nothing in either source run is changed.

What the merged run holds:
- documents: the base run's documents, in its order (the subset run may hold more; those are left out and listed in the
  merge log). Every one must be in the subset run, with identical document token ids and content, or the merge is refused.
- conditions: every condition of the base run, then the subset run's conditions that the base lacks (so the full persona
  set's order when the base holds a prefix of it). A condition both runs hold with a different system prompt is refused.
- log probabilities: base rows unchanged. A row new from the subset run is re-expressed against the base run's generic
  row, `new_condition - new_generic + base_generic`, so that every score (condition minus generic) equals the one its own
  run measured, and the baseline of all 72 conditions is the base run's generic. The merged values are therefore not raw
  log probabilities of the new conditions: they are base-run generic plus the subset run's own difference.
- merge_log_<timestamp>.json and .md: per document and overall, the maximum and mean absolute difference between the two
  runs' `generic` log probabilities, and between their log probabilities and generic-relative scores for every condition
  both runs hold other than generic (for the preachiness rerun: the five proportionate_welfare rewordings, scored in both).
  The same model scored the same tokens, so these show how reproducible the two runs are (batching, GPU, library
  versions); large differences mean the runs are not comparable.
- run_configuration_<timestamp>.json: the subset run's configuration (current persona set, full condition count), with the
  merge recorded under `merged_from` (both source runs' folders, formats, configuration files, models, document counts).
  No scoring_environment record is written: nothing was scored.

The merged run folder must not exist yet. The merge is quick and deterministic; to redo it, delete the output folder.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check import analysis_run_loading  # noqa: E402
from evals.log_probability_persona_check.persona_sets import sha256_of_file  # noqa: E402
from evals.log_probability_persona_check.legacy_run_format_compatibility_reader import (  # noqa: E402
    rename_legacy_condition_groups)
from shared import utils  # noqa: E402

GENERIC_CONDITION_ID = analysis_run_loading.GENERIC_CONDITION_ID
MODEL_IDENTITY_KEYS = ("model_type", "model_name", "model_revision")
NPZ_SUBFOLDER = Path("scoring_outputs") / "per_token_log_probabilities"


class RunsCannotBeMergedError(ValueError):
    """The two runs differ in a way that makes merging their scores meaningless."""


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


# ---------------------------------------------------------------- loading the two runs


def load_run_description(run_folder, legacy_persona_set=None):
    """Everything the merge needs from one run folder except the score arrays: the first run configuration (converted
    from the legacy format when `legacy_persona_set` is given), conditions (legacy group names renamed),
    documents, and the paths of its configuration files."""
    run_folder = Path(run_folder)
    configurations, is_legacy_format = analysis_run_loading.read_run_configurations(run_folder, legacy_persona_set)
    configuration = configurations[0]
    conditions = utils.load_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    if is_legacy_format:
        conditions = rename_legacy_condition_groups(conditions, configuration["persona_set"]["configuration"])
    return {
        "run_folder": run_folder, "configuration": configuration, "is_legacy_format": is_legacy_format,
        "configuration_files": [path.name for path in sorted(run_folder.glob("run_configuration_*.json"))],
        "conditions": conditions, "documents": utils.load_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")}


def check_model_identity_matches(base_description, subset_description):
    base_configuration, subset_configuration = base_description["configuration"], subset_description["configuration"]
    for key in MODEL_IDENTITY_KEYS:
        if base_configuration.get(key) != subset_configuration.get(key):
            raise RunsCannotBeMergedError(
                f"The runs differ in {key}: base run {base_configuration.get(key)!r}, subset run "
                f"{subset_configuration.get(key)!r}. Scores of different models are not comparable.")
    if base_configuration["persona_set"]["name"] != subset_configuration["persona_set"]["name"]:
        raise RunsCannotBeMergedError(
            f"The runs use different persona sets: {base_configuration['persona_set']['name']!r} and "
            f"{subset_configuration['persona_set']['name']!r}")
    # The chat template keyword arguments change the rendered prompt; the legacy format recorded none (and needed none).
    for description in (base_description, subset_description):
        description["chat_template_keyword_arguments"] = description["configuration"].get(
            "chat_template_keyword_arguments", {})
    if base_description["chat_template_keyword_arguments"] != subset_description["chat_template_keyword_arguments"]:
        raise RunsCannotBeMergedError(
            f"The runs used different chat template keyword arguments: "
            f"{base_description['chat_template_keyword_arguments']} and "
            f"{subset_description['chat_template_keyword_arguments']}")


def plan_merged_conditions(base_conditions, subset_conditions):
    """Returns (merged conditions, new condition ids, overlapping condition ids other than generic). Raises
    RunsCannotBeMergedError if the subset run has no generic condition, or a condition both hold differs in its system
    prompt, or the runs are not otherwise consistent."""
    base_by_id = {condition["condition_id"]: condition for condition in base_conditions}
    subset_ids = [condition["condition_id"] for condition in subset_conditions]
    if GENERIC_CONDITION_ID not in base_by_id or GENERIC_CONDITION_ID not in subset_ids:
        raise RunsCannotBeMergedError("Both runs need a 'generic' condition: it is the baseline of every score")
    overlapping, new_conditions = [], []
    for condition in subset_conditions:
        condition_id = condition["condition_id"]
        if condition_id in base_by_id:
            if base_by_id[condition_id]["system_prompt"] != condition["system_prompt"]:
                raise RunsCannotBeMergedError(
                    f"Condition {condition_id} has a different system prompt in the two runs: "
                    f"{base_by_id[condition_id]['system_prompt']!r} vs {condition['system_prompt']!r}")
            if condition_id != GENERIC_CONDITION_ID:
                overlapping.append(condition_id)
        else:
            new_conditions.append(condition)
    return list(base_conditions) + new_conditions, [c["condition_id"] for c in new_conditions], overlapping


def plan_merged_documents(base_documents, subset_documents):
    """The subset run's records of the base run's documents, in the base run's order, and the ids of the documents only
    the subset run holds. Raises RunsCannotBeMergedError if the subset run lacks a base document or the content differs."""
    subset_by_id = {document["document_id"]: document for document in subset_documents}
    missing = [document["document_id"] for document in base_documents if document["document_id"] not in subset_by_id]
    if missing:
        raise RunsCannotBeMergedError(f"The subset run lacks {len(missing)} of the base run's documents, "
                                      f"for example {missing[:5]}")
    merged_documents = []
    for base_document in base_documents:
        subset_document = subset_by_id[base_document["document_id"]]
        if base_document["content"] != subset_document["content"]:
            raise RunsCannotBeMergedError(
                f"Document {base_document['document_id']} has different content in the two runs")
        merged_documents.append(subset_document)
    base_ids = {document["document_id"] for document in base_documents}
    return merged_documents, [d["document_id"] for d in subset_documents if d["document_id"] not in base_ids]


# ---------------------------------------------------------------- scores


def load_scores(run_folder, document_id):
    """(document token ids, condition ids, log probabilities float64) of one document's .npz."""
    path = Path(run_folder) / NPZ_SUBFOLDER / f"{document_id}.npz"
    if not path.is_file():
        raise RunsCannotBeMergedError(f"Missing scoring output for document {document_id}: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return ([int(value) for value in archive["document_token_ids"]], [str(value) for value in archive["condition_ids"]],
                np.asarray(archive["log_probabilities"], dtype=np.float64), [str(v) for v in archive["document_token_strings"]])


def row_of(condition_ids, condition_id, run_label, document_id):
    if condition_id not in condition_ids:
        raise RunsCannotBeMergedError(f"The {run_label} run's output for {document_id} has no condition {condition_id}")
    return condition_ids.index(condition_id)


def absolute_difference_summary(differences):
    """Maximum and mean absolute difference over one array's tokens (plus the token count)."""
    absolute = np.abs(differences)
    return {"maximum_absolute_difference": float(absolute.max()), "mean_absolute_difference": float(absolute.mean()),
            "number_of_tokens": int(absolute.size)}


def merge_one_document(document_id, base_folder, subset_folder, merged_conditions, new_condition_ids,
                       overlapping_condition_ids):
    """Returns (merged log probabilities float32 (merged conditions, tokens), document token ids, token strings,
    differences record). Refuses when the two runs' document token ids differ."""
    base_token_ids, base_condition_ids, base_log_probabilities, base_token_strings = load_scores(base_folder, document_id)
    subset_token_ids, subset_condition_ids, subset_log_probabilities, _ = load_scores(subset_folder, document_id)
    if base_token_ids != subset_token_ids:
        raise RunsCannotBeMergedError(
            f"Document {document_id} has different token ids in the two runs ({len(base_token_ids)} vs "
            f"{len(subset_token_ids)} tokens): the runs were tokenized differently, so their scores cannot be combined")
    base_generic = base_log_probabilities[row_of(base_condition_ids, GENERIC_CONDITION_ID, "base", document_id)]
    subset_generic = subset_log_probabilities[row_of(subset_condition_ids, GENERIC_CONDITION_ID, "subset", document_id)]

    merged = np.zeros((len(merged_conditions), len(base_token_ids)), dtype=np.float64)
    for merged_row, condition in enumerate(merged_conditions):
        condition_id = condition["condition_id"]
        if condition_id in new_condition_ids:
            subset_row = row_of(subset_condition_ids, condition_id, "subset", document_id)
            merged[merged_row] = subset_log_probabilities[subset_row] - subset_generic + base_generic
        else:
            merged[merged_row] = base_log_probabilities[row_of(base_condition_ids, condition_id, "base", document_id)]

    differences = {"generic": absolute_difference_summary(base_generic - subset_generic)}
    for condition_id in overlapping_condition_ids:
        base_row = base_log_probabilities[row_of(base_condition_ids, condition_id, "base", document_id)]
        subset_row = subset_log_probabilities[row_of(subset_condition_ids, condition_id, "subset", document_id)]
        differences[condition_id] = {
            "log_probability": absolute_difference_summary(base_row - subset_row),
            "score_relative_to_generic": absolute_difference_summary((base_row - base_generic) - (subset_row - subset_generic))}
    return merged.astype(np.float32), np.array(base_token_ids, dtype=np.int32), base_token_strings, differences


def combine_differences(summaries):
    """Overall maximum, token-weighted mean and the document with the largest maximum, over per-document summaries
    {document_id: {maximum_absolute_difference, mean_absolute_difference, number_of_tokens}}."""
    total_tokens = sum(summary["number_of_tokens"] for summary in summaries.values())
    worst_document_id = max(summaries, key=lambda document_id: summaries[document_id]["maximum_absolute_difference"])
    return {"maximum_absolute_difference": summaries[worst_document_id]["maximum_absolute_difference"],
            "document_with_maximum": worst_document_id,
            "mean_absolute_difference": sum(summary["mean_absolute_difference"] * summary["number_of_tokens"]
                                            for summary in summaries.values()) / total_tokens,
            "mean_of_per_document_mean_absolute_differences": float(np.mean(
                [summary["mean_absolute_difference"] for summary in summaries.values()])),
            "number_of_documents": len(summaries), "number_of_tokens": total_tokens}


def summarize_differences(differences_by_document, overlapping_condition_ids):
    """Overall figures, and the same grouped by persona (the persona name is the condition id up to '__rewording_')."""
    overall = {"generic": combine_differences({d: r["generic"] for d, r in differences_by_document.items()})}
    for condition_id in overlapping_condition_ids:
        overall[condition_id] = {kind: combine_differences(
            {d: r[condition_id][kind] for d, r in differences_by_document.items()})
            for kind in ("log_probability", "score_relative_to_generic")}
    personas = {}
    for condition_id in overlapping_condition_ids:
        personas.setdefault(condition_id.split("__rewording_")[0], []).append(condition_id)
    by_persona = {}
    for persona_name, condition_ids in personas.items():
        by_persona[persona_name] = {}
        for kind in ("log_probability", "score_relative_to_generic"):
            pooled = {}
            for document_id, record in differences_by_document.items():
                summaries = [record[condition_id][kind] for condition_id in condition_ids]
                pooled[document_id] = {
                    "maximum_absolute_difference": max(s["maximum_absolute_difference"] for s in summaries),
                    "mean_absolute_difference": float(np.mean([s["mean_absolute_difference"] for s in summaries])),
                    "number_of_tokens": summaries[0]["number_of_tokens"] * len(summaries)}
            by_persona[persona_name][kind] = combine_differences(pooled)
    return {"overall": overall, "by_overlapping_persona": by_persona}


# ---------------------------------------------------------------- output files


def write_merge_log(output_folder, timestamp, record):
    json_path = output_folder / f"merge_log_{timestamp}.json"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    overall = record["differences_between_the_two_runs"]["overall"]
    lines = [f"# Merge log {timestamp}", "",
             f"Base run: `{record['base_run']['run_folder']}` ({record['base_run']['format']}, "
             f"{record['base_run']['number_of_conditions']} conditions)",
             f"Subset run: `{record['subset_run']['run_folder']}` ({record['subset_run']['number_of_conditions']} conditions)",
             f"Merged run: `{output_folder}`: {record['number_of_documents_merged']} documents, "
             f"{record['number_of_conditions_merged']} conditions ({len(record['new_condition_ids'])} taken from the subset run, "
             f"re-expressed relative to the base run's generic)", "",
             "## Reproducibility of the two runs (same model, same tokens)", "",
             "| Quantity | Maximum absolute difference | Mean absolute difference |", "|---|---|---|"]
    generic = overall["generic"]
    lines.append(f"| generic log probability | {generic['maximum_absolute_difference']:.4f} | "
                 f"{generic['mean_absolute_difference']:.4f} |")
    for persona_name, kinds in record["differences_between_the_two_runs"]["by_overlapping_persona"].items():
        for kind, label in (("log_probability", "log probability"), ("score_relative_to_generic", "score relative to generic")):
            lines.append(f"| {persona_name} rewordings, {label} | {kinds[kind]['maximum_absolute_difference']:.4f} | "
                         f"{kinds[kind]['mean_absolute_difference']:.4f} |")
    lines += ["", "Differences are over document tokens (maximum over all tokens of all documents; mean weighted by tokens). "
              "Per-document figures are in the JSON file."]
    (output_folder / f"merge_log_{timestamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def build_merged_configuration(base_description, subset_description, output_folder, timestamp, merged_documents,
                               merged_conditions, new_condition_ids, documents_only_in_subset_run):
    configuration = json.loads(json.dumps(subset_description["configuration"]))
    configuration["run_folder_name"] = output_folder.name
    configuration["created_timestamp"] = timestamp
    configuration["resumed_from_existing_folder"] = False
    configuration["number_of_conditions"] = len(merged_conditions)
    configuration["condition_subset"] = None
    document_selection = configuration["document_selection"]
    document_selection["document_ids"] = [document["document_id"] for document in merged_documents]
    document_selection["number_of_extra_documents"] = sum(1 for d in merged_documents if d["rewrite_direction"])
    document_selection["number_of_corpus_documents"] = len(merged_documents) - document_selection["number_of_extra_documents"]
    document_selection["maximum_documents"] = None
    document_selection["note"] = ("The documents of the base run; the subset run's other documents are not in the merged "
                                  "run (see merged_from.documents_only_in_subset_run)")

    def describe(description, role):
        return {"run_folder": str(description["run_folder"]),
                "format": "legacy" if description["is_legacy_format"] else "pipeline",
                "run_configuration_files": description["configuration_files"],
                "model_type": description["configuration"].get("model_type"),
                "model_name": description["configuration"]["model_name"],
                "model_revision": description["configuration"]["model_revision"],
                "number_of_documents": len(description["documents"]), "number_of_conditions": len(description["conditions"]),
                "role": role}

    configuration["merged_from"] = {
        "base_run": describe(base_description, "supplies its own conditions unchanged and the generic baseline of all rows"),
        "subset_run": describe(subset_description, "supplies the conditions the base run lacks, re-expressed against the base run's generic"),
        "new_condition_ids_from_subset_run": new_condition_ids,
        "documents_only_in_subset_run": documents_only_in_subset_run,
        "merged_log_probability_formula": "new_condition - new_generic + base_generic",
        "note": "No model was run for this folder; the scores are the two source runs' scores combined."}
    return configuration


def merge_runs(base_run_folder, subset_run_folder, output_run_folder, base_run_legacy_persona_set=None,
               validate_with_analysis_loader=True):
    """Writes the merged run folder and returns its path. Raises RunsCannotBeMergedError on any inconsistency (the
    output folder is then removed again, because a half-written merge would look like a finished run)."""
    output_folder = Path(output_run_folder)
    if output_folder.exists():
        raise SystemExit(f"{output_folder} already exists; the merge writes a new folder")
    base_description = load_run_description(base_run_folder, base_run_legacy_persona_set)
    subset_description = load_run_description(subset_run_folder)
    if subset_description["is_legacy_format"]:
        raise RunsCannotBeMergedError("The subset run must be a pipeline-format run")
    check_model_identity_matches(base_description, subset_description)
    merged_conditions, new_condition_ids, overlapping_condition_ids = plan_merged_conditions(
        base_description["conditions"], subset_description["conditions"])
    if not new_condition_ids:
        raise RunsCannotBeMergedError("The subset run has no condition the base run lacks; there is nothing to merge")
    merged_documents, documents_only_in_subset_run = plan_merged_documents(
        base_description["documents"], subset_description["documents"])
    new_condition_id_set = set(new_condition_ids)

    timestamp = timestamp_now()
    npz_directory = output_folder / NPZ_SUBFOLDER
    try:
        npz_directory.mkdir(parents=True)
        differences_by_document, completion_records = {}, []
        for number, document in enumerate(merged_documents, start=1):
            document_id = document["document_id"]
            merged, token_ids, token_strings, differences = merge_one_document(
                document_id, base_description["run_folder"], subset_description["run_folder"], merged_conditions,
                new_condition_id_set, overlapping_condition_ids)
            np.savez_compressed(npz_directory / f"{document_id}.npz", document_token_ids=token_ids,
                                document_token_strings=np.array(token_strings),
                                condition_ids=np.array([c["condition_id"] for c in merged_conditions]),
                                log_probabilities=merged)
            differences_by_document[document_id] = differences
            for row, condition in enumerate(merged_conditions):
                completion_records.append({
                    "document_id": document_id, "condition_id": condition["condition_id"],
                    "number_of_document_tokens": int(merged.shape[1]),
                    "sum_log_probability": float(merged[row].astype(np.float64).sum()),
                    "mean_log_probability": float(merged[row].astype(np.float64).mean())})
            if number % 50 == 0 or number == len(merged_documents):
                print(f"Merged {number}/{len(merged_documents)} documents", flush=True)
        scoring_inputs_directory = output_folder / "scoring_inputs"
        scoring_inputs_directory.mkdir()
        utils.save_jsonl(merged_documents, scoring_inputs_directory / "documents.jsonl")
        utils.save_jsonl(merged_conditions, scoring_inputs_directory / "conditions.jsonl")
        utils.save_jsonl(completion_records, output_folder / "scoring_outputs" / "scoring_completion_records.jsonl")
        configuration = build_merged_configuration(
            base_description, subset_description, output_folder, timestamp, merged_documents, merged_conditions,
            new_condition_ids, documents_only_in_subset_run)
        configuration["scoring_input_file_sha256"] = {
            "documents.jsonl": sha256_of_file(scoring_inputs_directory / "documents.jsonl"),
            "conditions.jsonl": sha256_of_file(scoring_inputs_directory / "conditions.jsonl")}
        (output_folder / f"run_configuration_{timestamp}.json").write_text(
            json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8")
        merge_log_record = {
            "merge_timestamp": timestamp,
            "base_run": configuration["merged_from"]["base_run"], "subset_run": configuration["merged_from"]["subset_run"],
            "number_of_documents_merged": len(merged_documents), "number_of_conditions_merged": len(merged_conditions),
            "new_condition_ids": new_condition_ids, "overlapping_condition_ids_other_than_generic": overlapping_condition_ids,
            "documents_only_in_subset_run": documents_only_in_subset_run,
            "differences_between_the_two_runs": summarize_differences(differences_by_document, overlapping_condition_ids),
            "differences_per_document": differences_by_document}
        write_merge_log(output_folder, timestamp, merge_log_record)
        if validate_with_analysis_loader:
            analysis_run_loading.load_run(output_folder)
            print("Validated: the analysis loader reads the merged run folder")
    except BaseException:
        import shutil
        shutil.rmtree(output_folder, ignore_errors=True)
        raise
    print(f"Merged run folder: {output_folder}")
    return output_folder


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-run-folder", required=True, help="The finished run whose conditions are kept as they are")
    parser.add_argument("--subset-run-folder", required=True,
                        help="The persona-subset run (build_scoring_inputs.py --only-personas) with the missing conditions")
    parser.add_argument("--output-run-folder", required=True, help="New folder; must not exist")
    parser.add_argument("--base-run-legacy-persona-set", default=None,
                        help="Set when the base run is in the legacy format: the persona set to pair it with")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Do not read the merged folder back with the analysis loader at the end")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    try:
        return merge_runs(arguments.base_run_folder, arguments.subset_run_folder, arguments.output_run_folder,
                          arguments.base_run_legacy_persona_set, not arguments.skip_validation)
    except RunsCannotBeMergedError as error:
        raise SystemExit(f"Refusing to merge: {error}")


if __name__ == "__main__":
    main()
