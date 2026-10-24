#!/usr/bin/env python3
"""LLM classifier that audits an SDF corpus for "shoehorned" ethical considerations (paid).

Calls go through shared/openai_api.py (cost log, model-call log) and keys through shared/api_keys.py.

For every (document, pass) pair, the model receives ONLY the instructions
(prompts/tools/shoehorning_audit_shared_header.txt + the pass file) and the document's `content` text.
Nothing else about the document (description, variables, language, doc_id, scores) is ever sent.

Runs are resumable: finished (doc_id, pass) pairs are read back from the run folder. Run
evals/shoehorning_audit_postprocess_sdf.py on the run folder afterwards.

Usage example:
    python evals/shoehorning_audit_sdf.py --input outputs/sdf/latest --run-name full_audit_medium_effort
    python evals/shoehorning_audit_sdf.py --input outputs/sdf/latest --run-name x --estimate-only
    python evals/shoehorning_audit_sdf.py --input outputs/sdf/latest --run-name x \\
        --resume-folder outputs/sdf/latest/audit/shoehorning_audit_runs/x_2026-10-07_12-00

Run folders go to <run>/audit/shoehorning_audit_runs/ (--runs-directory to change); every partial run
(--estimate-only, --document-ids, --document-limit) goes to its small_sample_checks subfolder.
"""

import argparse
import concurrent.futures
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.model_judged_audit_support import (  # noqa: E402
    DEFAULT_INPUT, MAXIMUM_OUTPUT_TOKENS, AppendOnlyJsonlWriter, RunningUsageTotals,
    StructuredOutputParseFailure, TimedProgressReporter, build_array_schema, build_strict_object_schema,
    check_resume_is_compatible, choose_run_folder, choose_runs_parent_directory, choose_stratified_sample,
    compute_cost_in_dollars, corpus_path_of, find_completed_keys, format_duration, read_all_records,
    read_jsonl, request_structured_output, run_directory_of, select_documents, sha256_of_file, timestamp_now,
    usage_paid_before_transport_failure, write_run_configuration)
from shared import api, openai_api  # noqa: E402
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
PROMPT_DIRECTORY = REPO_ROOT / "prompts" / "tools"

MODEL_NAME = "gpt-5.6-luna"
STAGE_TAG = "eval_shoehorning_audit"
SAMPLING_TEMPERATURE = openai_api.SAMPLING_TEMPERATURE
ALL_PASS_NAMES = ["moral_considerations", "field_distinctions", "epistemic_notes"]
SHARED_HEADER_FILE_NAME = "shoehorning_audit_shared_header.txt"
MAXIMUM_OCCURRENCES_PER_ITEM = 3

# Set by probing the API; see the report. If strict mode rejects an empty sub_flags
# object, flip this to False and sub_flags is omitted for items that have none.
INCLUDE_EMPTY_SUB_FLAGS_OBJECT_FOR_ITEMS_WITHOUT_SUB_FLAGS = True

MOTIVATION_VALUES = ["organic", "staged", "unmotivated"]
INTEGRATION_VALUES = ["woven", "aside"]


# ----------------------------------------------------------------------------
# Prompt parsing and schema construction
# ----------------------------------------------------------------------------

def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def pass_file_path(pass_name, prompt_directory=PROMPT_DIRECTORY):
    return Path(prompt_directory) / f"shoehorning_audit_pass_{pass_name}.txt"


def parse_pass_items(pass_file_text):
    """Return an ordered list of {"item_key": str, "sub_flag_names": [str, ...]}."""
    parsed_items = []
    current_item = None
    inside_sub_flag_list = False
    for line in pass_file_text.splitlines():
        header_match = re.match(r"^###\s+(item_\d+_\w+)\s*$", line)
        if header_match:
            current_item = {"item_key": header_match.group(1), "sub_flag_names": []}
            parsed_items.append(current_item)
            inside_sub_flag_list = False
            continue
        if current_item is None:
            continue
        if line.startswith("Sub-flags:"):
            inside_sub_flag_list = line[len("Sub-flags:"):].strip().lower() != "none"
            continue
        if inside_sub_flag_list:
            flag_match = re.match(r"^-\s+(\w+):", line)
            if flag_match:
                current_item["sub_flag_names"].append(flag_match.group(1))
            elif line.strip() and not line.startswith("-"):
                inside_sub_flag_list = False
    return parsed_items


def build_instructions_text(pass_name, prompt_directory=PROMPT_DIRECTORY):
    header_text = read_text(Path(prompt_directory) / SHARED_HEADER_FILE_NAME)
    pass_text = read_text(pass_file_path(pass_name, prompt_directory))
    return header_text + "\n\n" + pass_text


def build_occurrence_schema(sub_flag_names):
    properties = {
        "quote": {"type": "string"},
        "english_gloss": {"type": ["string", "null"]},
        "context": {"type": "string"},
        "motivation_reasoning": {"type": "string"},
        "motivation": {"type": "string", "enum": MOTIVATION_VALUES},
        "integration_reasoning": {"type": "string"},
        "integration": {"type": "string", "enum": INTEGRATION_VALUES},
    }
    if sub_flag_names or INCLUDE_EMPTY_SUB_FLAGS_OBJECT_FOR_ITEMS_WITHOUT_SUB_FLAGS:
        properties["sub_flags"] = build_strict_object_schema(
            {name: {"type": "boolean"} for name in sub_flag_names})
    return build_strict_object_schema(properties)  # dict order = generation order


def build_other_considerations_schema(use_maximum_items=True):
    entry_schema = build_strict_object_schema({
        "suggested_name": {"type": "string"},
        "description": {"type": "string"},
        "quote": {"type": "string"},
        "english_gloss": {"type": ["string", "null"]},
        "context": {"type": "string"},
        "motivation_reasoning": {"type": "string"},
        "motivation": {"type": "string", "enum": ["staged", "unmotivated"]},
        "integration_reasoning": {"type": "string"},
        "integration": {"type": "string", "enum": INTEGRATION_VALUES},
    })
    return build_array_schema(entry_schema, MAXIMUM_OCCURRENCES_PER_ITEM if use_maximum_items else None)


def build_pass_schema(parsed_items, use_maximum_items=True):
    maximum_items = MAXIMUM_OCCURRENCES_PER_ITEM if use_maximum_items else None
    item_properties = {}
    for item in parsed_items:
        occurrences_schema = build_array_schema(build_occurrence_schema(item["sub_flag_names"]), maximum_items)
        item_properties[item["item_key"]] = build_strict_object_schema({"occurrences": occurrences_schema})
    return build_strict_object_schema({
        "items": build_strict_object_schema(item_properties),
        "other_considerations": build_other_considerations_schema(use_maximum_items),
    })


class PassDefinition:
    def __init__(self, pass_name, prompt_directory=PROMPT_DIRECTORY):
        self.pass_name = pass_name
        self.instructions_text = build_instructions_text(pass_name, prompt_directory)
        self.parsed_items = parse_pass_items(read_text(pass_file_path(pass_name, prompt_directory)))
        self.schema = build_pass_schema(self.parsed_items)
        self.schema_name = f"shoehorning_audit_{pass_name}"


def build_request_arguments(pass_definition, document_content, reasoning_effort):
    """The complete payload sent to client.responses.create by shared.openai_api.call_openai.
    Only instructions and content go in."""
    return openai_api.build_request_arguments(
        MODEL_NAME, pass_definition.instructions_text, document_content,
        max_output_tokens=MAXIMUM_OUTPUT_TOKENS, reasoning_effort=reasoning_effort,
        json_schema=pass_definition.schema, schema_name=pass_definition.schema_name,
        prompt_cache_key=pass_definition.schema_name)


def validate_parsed_result(parsed_result, pass_definition):
    """Raise ValueError if the parsed JSON does not have the expected item keys."""
    if not isinstance(parsed_result, dict) or not isinstance(parsed_result.get("items"), dict):
        raise ValueError("missing top-level 'items' object")
    expected_keys = [item["item_key"] for item in pass_definition.parsed_items]
    missing_keys = [key for key in expected_keys if key not in parsed_result["items"]]
    if missing_keys:
        raise ValueError(f"missing item keys: {missing_keys}")
    if not isinstance(parsed_result.get("other_considerations"), list):
        raise ValueError("missing other_considerations list")
    for key in expected_keys:
        if not isinstance(parsed_result["items"][key].get("occurrences"), list):
            raise ValueError(f"item {key} has no occurrences list")


def classify_one_document(pass_definition, document, reasoning_effort):
    """Returns (record, error_record). Retries once on parse/schema failure."""
    try:
        result = request_structured_output(
            MODEL_NAME, pass_definition.instructions_text, document["content"],
            pass_definition.schema_name, pass_definition.schema, reasoning_effort,
            stage=STAGE_TAG, item_id=f"{document['doc_id']}__{pass_definition.pass_name}",
            validate_parsed_result=lambda parsed: validate_parsed_result(parsed, pass_definition),
            prompt_cache_key=pass_definition.schema_name)
    except StructuredOutputParseFailure as failure:
        return None, {
            "doc_id": document["doc_id"], "pass": pass_definition.pass_name,
            "error_kind": "parse_or_schema_failure", "description": failure.description,
            "raw_output_text": failure.raw_output_text, "usage": failure.usage,
            "latency_seconds": failure.latency_seconds,
        }
    return {
        "doc_id": document["doc_id"],
        "pass": pass_definition.pass_name,
        "model": MODEL_NAME,
        "reasoning_effort": reasoning_effort,
        "parsed_result": result["parsed_result"],
        "raw_output_text": result["raw_output_text"],
        "usage": result["usage"],
        "parse_attempts": result["parse_attempts"],
        "latency_seconds": result["latency_seconds"],
    }, None


# ----------------------------------------------------------------------------
# Run folder, resume, configuration
# ----------------------------------------------------------------------------

def load_corpus(corpus_path):
    return read_jsonl(corpus_path)


def find_completed_pairs(run_folder):
    """Set of (doc_id, pass) already classified in this run folder."""
    return find_completed_keys(run_folder, "classifications_*.jsonl", ["doc_id", "pass"])


def load_existing_records(run_folder):
    return read_all_records(run_folder, "classifications_*.jsonl")


def build_run_configuration(pass_names, reasoning_effort, corpus_path, prompt_directory=PROMPT_DIRECTORY):
    hashes = {SHARED_HEADER_FILE_NAME: sha256_of_file(Path(prompt_directory) / SHARED_HEADER_FILE_NAME)}
    for pass_name in pass_names:
        hashes[pass_file_path(pass_name, prompt_directory).name] = sha256_of_file(
            pass_file_path(pass_name, prompt_directory))
    input_price, cached_input_price, output_price = openai_api._PRICING[MODEL_NAME]
    return {
        "model": MODEL_NAME,
        "reasoning_effort": reasoning_effort,
        "temperature": SAMPLING_TEMPERATURE,
        "passes": list(pass_names),
        "prompt_file_sha256": hashes,
        "corpus_path": str(Path(corpus_path).resolve()),
        "corpus_sha256": sha256_of_file(corpus_path),
        "prices_per_million_tokens": {
            "input": input_price,
            "cached_input": cached_input_price,
            "output": output_price},
    }


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

class RunReporter:
    """Writes classifications, errors, progress reports and the usage log for one run."""

    def __init__(self, run_folder, timestamp, total_new_tasks, already_done_count, prior_records):
        self.run_folder = Path(run_folder)
        self.timestamp = timestamp
        self.record_writer = AppendOnlyJsonlWriter()
        self.usage_totals = RunningUsageTotals(MODEL_NAME)
        for record in prior_records:
            self.usage_totals.add_usage(record["usage"])
        self.progress_log_path = self.run_folder / f"progress_log_{timestamp}.txt"
        self.errors_path = self.run_folder / f"errors_{timestamp}.jsonl"
        self.usage_log_path = self.run_folder / f"usage_and_cost_log_{timestamp}.json"
        self.progress_reporter = TimedProgressReporter(
            self.progress_log_path, total_new_tasks, already_done_count,
            status_suffix_function=self.usage_totals.status_text)

    @property
    def completed_new_tasks(self):
        return self.progress_reporter.completed_new_tasks

    @property
    def failed_new_tasks(self):
        return self.progress_reporter.failed_new_tasks

    def record_success(self, record):
        self.record_writer.append(
            self.run_folder / f"classifications_{record['pass']}_{self.timestamp}.jsonl", record)
        self.usage_totals.add_usage(record["usage"])
        self.progress_reporter.note_task_finished(failed=False)

    def record_failure(self, error_record):
        self.record_writer.append(self.errors_path, error_record)
        self.usage_totals.add_usage(error_record["usage"])
        self.progress_reporter.note_task_finished(failed=True)

    def finish(self):
        self.progress_reporter.finish()
        self.usage_totals.write_log_file(self.usage_log_path, "totals include records from before any resume")


def run_classification(run_folder, timestamp, pass_definitions, documents, reasoning_effort,
                       worker_count, already_completed_pairs, prior_records):
    """Classify all not-yet-done (document, pass) pairs. Returns the reporter."""
    tasks = [(document, pass_definition)
             for document in documents for pass_definition in pass_definitions
             if (document["doc_id"], pass_definition.pass_name) not in already_completed_pairs]
    reporter = RunReporter(run_folder, timestamp, len(tasks), len(already_completed_pairs), prior_records)
    print(f"{len(tasks)} calls to make ({len(already_completed_pairs)} pairs already done).", flush=True)

    def work(task):
        document, pass_definition = task
        try:
            record, error_record = classify_one_document(pass_definition, document, reasoning_effort)
        except Exception as error:  # noqa: BLE001 - transport failure after all retries
            record, error_record = None, {
                "doc_id": document["doc_id"], "pass": pass_definition.pass_name,
                "error_kind": "transport_failure", "description": repr(error),
                "raw_output_text": None,
                "usage": usage_paid_before_transport_failure(error),
                "latency_seconds": None}
        if record:
            reporter.record_success(record)
        else:
            reporter.record_failure(error_record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(work, tasks))
    reporter.finish()
    return reporter


# ----------------------------------------------------------------------------
# Estimate-only
# ----------------------------------------------------------------------------

def extrapolate_estimate(sample_records, sample_documents, all_documents, pass_names, error_records=()):
    """Scale uncached input by content characters, output by document count; cached per call.

    Each pass is scaled over the sample documents it was judged successfully for; the tokens of
    failed attempts count towards its totals, as they are paid for. Passes with no successful sample
    call are listed in passes_without_a_successful_sample_call and add nothing to the totals.
    Returns None if no sample call succeeded.
    """
    if not sample_records:
        return None
    sample_ids = {d["doc_id"] for d in sample_documents}
    corpus_characters = sum(len(d["content"]) for d in all_documents)
    judged_document_ids = {r["doc_id"] for r in sample_records}
    judged_documents = [d for d in sample_documents if d["doc_id"] in judged_document_ids]
    character_scale = corpus_characters / sum(len(d["content"]) for d in judged_documents)
    document_scale = len(all_documents) / len(judged_documents)
    per_pass = {}
    passes_without_a_successful_sample_call = []
    for pass_name in pass_names:
        successes = [r for r in sample_records if r["pass"] == pass_name]
        if not successes:
            passes_without_a_successful_sample_call.append(pass_name)
            continue
        records = successes + [e for e in error_records if e["pass"] == pass_name and e["doc_id"] in sample_ids]
        judged_for_pass = {r["doc_id"] for r in successes}
        pass_documents = [d for d in sample_documents if d["doc_id"] in judged_for_pass]
        pass_character_scale = corpus_characters / sum(len(d["content"]) for d in pass_documents)
        pass_document_scale = len(all_documents) / len(pass_documents)
        uncached = sum(r["usage"]["input_tokens"] - r["usage"]["cached_input_tokens"] for r in records)
        cached = sum(r["usage"]["cached_input_tokens"] for r in records)
        output = sum(r["usage"]["output_tokens"] for r in records)
        total_uncached = uncached * pass_character_scale
        total_cached = cached * pass_document_scale
        total_output = output * pass_document_scale
        total_seconds = sum(r["latency_seconds"] or 0 for r in records) * pass_document_scale
        per_pass[pass_name] = {
            "uncached_input_tokens": total_uncached, "cached_input_tokens": total_cached,
            "output_tokens": total_output, "total_call_seconds": total_seconds,
            "cost_dollars": compute_cost_in_dollars(
                MODEL_NAME, total_uncached + total_cached, total_cached, total_output)}
    return {"character_scale": character_scale, "document_scale": document_scale,
            "per_pass": per_pass,
            "passes_without_a_successful_sample_call": passes_without_a_successful_sample_call,
            "total_cost_dollars": sum(p["cost_dollars"] for p in per_pass.values()),
            "total_call_seconds_across_all_calls": sum(p["total_call_seconds"] for p in per_pass.values()),
            "note": ("Sample uncached input includes the instruction prefix on cache-missing calls, "
                     "so uncached input (and cost) is likely overestimated; scaled by corpus chars / "
                     "sample chars. Cached input and output scaled by document count.")}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="SDF run directory or sdf_corpus.jsonl path")
    parser.add_argument("--run-name", required=True, help="Descriptive name for this run")
    parser.add_argument("--runs-directory", default=None,
                        help="Where run folders go (default: <run>/audit/shoehorning_audit_runs)")
    parser.add_argument("--resume-folder", default=None, help="Existing run folder to resume")
    parser.add_argument("--passes", nargs="+", default=ALL_PASS_NAMES, choices=ALL_PASS_NAMES)
    parser.add_argument("--document-limit", type=int, default=None, help="Use only the first N documents")
    parser.add_argument("--document-ids", nargs="+", default=None, help="Use only these doc_ids")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-sample-size", type=int, default=6)
    add_api_key_source_flags(parser)
    return parser.parse_args(argv)


def default_runs_directory(input_argument):
    run_directory = run_directory_of(input_argument)
    audit_folder = (run_directory if run_directory is not None else Path(input_argument).parent) / "audit"
    return audit_folder / "shoehorning_audit_runs"


def main(argv=None):
    arguments = parse_arguments(argv)
    apply_api_key_source_flags(arguments)
    corpus_path = corpus_path_of(arguments.input)
    all_documents = read_jsonl(corpus_path)
    documents = select_documents(all_documents, arguments.document_ids, arguments.document_limit, corpus_path)

    runs_directory = Path(arguments.runs_directory) if arguments.runs_directory else default_runs_directory(
        arguments.input)
    run_kind, runs_parent_directory = choose_runs_parent_directory(arguments, runs_directory)
    run_folder, is_resume = choose_run_folder(runs_parent_directory, arguments.run_name, arguments.resume_folder)
    print(f"Run kind: {run_kind}{' (resuming)' if is_resume else ''}; run folder: {run_folder}")
    timestamp = timestamp_now()
    configuration = build_run_configuration(arguments.passes, arguments.reasoning_effort, corpus_path)
    if is_resume:
        check_resume_is_compatible(run_folder, configuration, compared_settings=("reasoning_effort",))
    write_run_configuration(run_folder, timestamp, configuration)

    pass_definitions = [PassDefinition(name) for name in arguments.passes]
    for pass_definition in pass_definitions:
        (run_folder / f"instructions_sent_{pass_definition.pass_name}_{timestamp}.txt").write_text(
            pass_definition.instructions_text, encoding="utf-8")

    already_completed_pairs = find_completed_pairs(run_folder)
    prior_records = load_existing_records(run_folder) + read_all_records(run_folder, "errors_*.jsonl")
    api.init_cost_log_only(run_folder / f"model_call_cost_log_{timestamp}.jsonl")

    if arguments.estimate_only:
        documents = choose_stratified_sample(documents, arguments.estimate_sample_size)
        print("Estimate sample: " + ", ".join(f"{d['doc_id']} ({d['language']}, {len(d['content'])} chars)"
                                              for d in documents), flush=True)

    run_classification(run_folder, timestamp, pass_definitions, documents, arguments.reasoning_effort,
                       arguments.workers, already_completed_pairs, prior_records)

    if arguments.estimate_only:
        sample_ids = {d["doc_id"] for d in documents}
        sample_records = [r for r in load_existing_records(run_folder)
                          if r["doc_id"] in sample_ids and r["pass"] in arguments.passes]
        error_records = [e for e in read_all_records(run_folder, "errors_*.jsonl") if e["pass"] in arguments.passes]
        estimate = extrapolate_estimate(sample_records, documents, all_documents, arguments.passes, error_records)
        if estimate is None:
            print("ESTIMATE full corpus x passes: no estimate is available, every sample call failed "
                  f"(see the error records in {run_folder})")
            print(f"Run folder: {run_folder}")
            return run_folder
        estimate["sample_doc_ids"] = sorted(sample_ids)
        estimate["corpus_document_count"] = len(all_documents)
        estimate["workers_assumed_for_wall_clock"] = arguments.workers
        estimate["estimated_wall_clock_seconds"] = (
            estimate["total_call_seconds_across_all_calls"] / arguments.workers)
        (run_folder / f"estimate_{timestamp}.json").write_text(json.dumps(estimate, indent=2), encoding="utf-8")
        print(f"ESTIMATE full corpus x passes: ${estimate['total_cost_dollars']:.2f}, about "
              f"{format_duration(estimate['estimated_wall_clock_seconds'])} with {arguments.workers} workers")
        for pass_name, values in estimate["per_pass"].items():
            print(f"  {pass_name}: ${values['cost_dollars']:.2f}")
        for pass_name in estimate["passes_without_a_successful_sample_call"]:
            print(f"  {pass_name}: no estimate is available, every sample call failed (not in the total)")
    print(f"Run folder: {run_folder}")
    return run_folder


if __name__ == "__main__":
    main()
