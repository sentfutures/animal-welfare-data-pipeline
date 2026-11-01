#!/usr/bin/env python3
"""Per-variable persistence check for an SDF corpus (paid LLM pass, one call per document).

Each SDF document is planned from a hand of dealt variable values (document type,
culture, tone, resolution, ...) and, for some documents, assigned considerations.
This eval asks a judge, for every dealt variable and every assigned consideration,
whether that value PERSISTED in the finished document: brief reasoning, then a
yes/no call, returned as typed JSON.

It is deliberately not a conformance check against the document's free-text
description. The judge is never shown the description (nor the doc_id, scores, or
the considerations spec, which carries per-document reasons): only the dealt
values, one line on what each variable means, the assigned considerations (label
and gloss), and the document text. The definition of "persisted" lives in
prompts/tools/variable_persistence_judge.txt so it is easy to edit.

Usage
-----
    python evals/variable_persistence_sdf.py --input outputs/sdf/latest
    python evals/variable_persistence_sdf.py --input outputs/sdf/latest --sample 40 --seed 1
    python evals/variable_persistence_sdf.py --input outputs/sdf/latest --doc-ids matrix_000003 matrix_000011
    python evals/variable_persistence_sdf.py --input outputs/sdf/latest --dry-run     # no calls, no key
    python evals/variable_persistence_sdf.py --input outputs/sdf/latest --resume <folder>

``--input`` is a run directory or a corpus ``sdf_corpus.jsonl``. Outputs go to
``<run>/audit/variable_persistence_<YYYY-MM-DD_HH-MM>/``, or for ``--sample`` and ``--doc-ids`` runs to
``<run>/audit/small_sample_checks/variable_persistence_<YYYY-MM-DD_HH-MM>/`` (or ``--output-dir``):

    persistence_judgments_<timestamp>.jsonl        one row per document, appended as each completes
    per_variable_persistence_summary_<timestamp>.csv
    failed_persistence_checks_<timestamp>.jsonl    every "no", with doc_id, variable, dealt value, reasoning
    persistence_cost_log_<timestamp>.jsonl         one row per API call
    run_configuration_<timestamp>.json             judge prompt and variable-meanings hashes, model, corpus hash

A rerun with ``--resume <folder>`` skips documents already judged; documents whose
calls failed are retried. A resume is refused if the corpus, the judge prompt, the
variable-meanings file or the model differs from what the folder recorded (or if the
folder recorded none).

Config (``sdf`` section): ``variable_persistence_model`` (falls back to
``score_model``, then ``plan_model``, then the global ``model``),
``variable_persistence_max_attempts`` (default 3); the global ``workers`` sets the
thread pool.
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.audit_sdf import resolve_input
from evals.model_judged_audit_support import (
    SMALL_SAMPLE_CHECKS_DIRECTORY_NAME, check_resume_is_compatible, corpus_path_of, sha256_of_file,
    write_run_configuration)
from sdf_pipeline import compose_prompts as cp
from sdf_pipeline import considerations_catalog
from shared import api, utils
from shared.api_keys import add_api_key_source_flags, apply_api_key_source_flags

REPO_ROOT = Path(__file__).parent.parent
PROMPT_PATH = REPO_ROOT / "prompts" / "tools" / "variable_persistence_judge.txt"
MEANINGS_PATH = REPO_ROOT / "prompts" / "tools" / "variable_persistence_variable_meanings.txt"
DEFAULT_PROMPTS_DIR = REPO_ROOT / "prompts" / "sdf"

STAGE_TAG = "eval_variable_persistence"
PROGRESS_INTERVAL_SECONDS = 180   # progress lines are printed no more often than this
DEFAULT_MAX_ATTEMPTS = 3
CONSIDERATIONS_KEY = "considerations"
VERDICTS = ("yes", "no")
JUDGE_MARKER = "=== JUDGE INSTRUCTIONS ==="
CONSIDERATIONS_MARKER = "=== CONSIDERATIONS SECTION ==="

# Output allowance: brief reasoning per item. Generous so a truncation means a runaway answer.
BASE_MAX_TOKENS = 1000
TOKENS_PER_ITEM = 200
# Rough per-item output estimate used only by --dry-run (reasoning sentence + verdict + JSON syntax).
ESTIMATED_OUTPUT_TOKENS_PER_ITEM = 90
ESTIMATED_CHARACTERS_PER_TOKEN = 3.5

SUMMARY_COLUMNS = ["kind", "name", "documents_judged", "yes_count", "no_count", "share_no"]


class PersistenceAnswerError(ValueError):
    """The judge's answer was unusable (truncated, malformed, incomplete)."""


# ---------------------------------------------------------------- prompt pieces

def load_prompt_sections(prompt_path: Path = PROMPT_PATH) -> tuple[str, str]:
    """(judge template, considerations-section template) split on the marker lines.
    Text before the first marker (the file's header comments) is ignored."""
    text = Path(prompt_path).read_text(encoding="utf-8")
    if JUDGE_MARKER not in text or CONSIDERATIONS_MARKER not in text:
        raise ValueError(f"{Path(prompt_path).name} must contain the marker lines "
                         f"{JUDGE_MARKER!r} and {CONSIDERATIONS_MARKER!r}")
    after_judge = text.split(JUDGE_MARKER, 1)[1]
    judge, considerations = after_judge.split(CONSIDERATIONS_MARKER, 1)
    return judge.strip("\n"), considerations.strip("\n")


def load_variable_meanings(meanings_path: Path = MEANINGS_PATH) -> dict[str, str]:
    """{variable name: one-line meaning}, in file order (which is the judge's variable order)."""
    meanings: dict[str, str] = {}
    for line in Path(meanings_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, meaning = stripped.partition(":")
        if not separator or not meaning.strip() or not re.fullmatch(r"[a-z][a-z0-9_]*", name.strip()):
            raise ValueError(f"{Path(meanings_path).name}: line {line!r} must read 'variable_name: meaning'")
        meanings[name.strip()] = meaning.strip()
    return meanings


def find_prompts_dir(run_directory: Path | None) -> Path:
    """The run's own prompt snapshot when it has one (so deal rules and the considerations
    catalog match the run), else the repo's prompts/sdf."""
    if run_directory is not None:
        snapshot = run_directory / "inputs" / "prompts"
        if (snapshot / "variables.txt").is_file() and (snapshot / "considerations.txt").is_file():
            return snapshot
    return DEFAULT_PROMPTS_DIR


def run_directory_of(input_argument: str) -> Path | None:
    """The run directory an --input points at, if it can be told (a run dir, or a corpus in <run>/final/)."""
    path = Path(input_argument)
    if path.is_dir():
        return path
    if path.parent.name == "final":
        return path.parent.parent
    return None


def _load_catalog(prompts_dir: Path):
    """The considerations catalog (for glosses), from the run's snapshot, else the repo's own file.
    A snapshot written before a catalog validation rule existed can fail to load; then the repo
    catalog supplies the glosses, and failing that a label is judged on its label alone (None)."""
    for candidate in dict.fromkeys([prompts_dir, DEFAULT_PROMPTS_DIR]):
        try:
            return considerations_catalog.load_considerations(candidate)
        except (ValueError, OSError, KeyError) as error:
            print(f"  NOTICE: could not load the considerations catalog from {candidate} ({error})")
    return None


class JudgeContext:
    """Everything loaded once per run that the per-document steps need."""

    def __init__(self, prompts_dir: Path):
        self.judge_template, self.considerations_template = load_prompt_sections()
        self.variable_meanings = load_variable_meanings()
        _, _, value_ids = cp.load_variables(prompts_dir / cp.DEFAULT_VARIABLES.name)
        self.deal_rules = cp.load_deal_rules(prompts_dir / cp.DEAL_CONSTRAINTS_FILENAME, value_ids)
        self.catalog = _load_catalog(prompts_dir)


def judged_variables(record: dict, context: JudgeContext) -> dict[str, str]:
    """{variable name: dealt value text} for the variables the judge is asked about: those the
    record carries with a non-empty value, minus any the deal rules omit from the planning prompt
    for this hand (an omitted variable was never put to the planner, so it cannot have been asked
    to persist)."""
    dealt = record.get("variables") or {}
    omitted = cp.deal_constraints.render_rule_variables_to_omit(context.deal_rules, dealt)
    return {
        name: str(dealt[name]).strip()
        for name in context.variable_meanings
        if name not in omitted and isinstance(dealt.get(name), str) and dealt[name].strip()
    }


def assigned_considerations(record: dict, context: JudgeContext) -> list[dict]:
    """[{label, gloss}] for the record's assigned considerations, moral then epistemic. The gloss
    comes from the catalog; a label the catalog no longer holds is judged on its label alone."""
    dealt = record.get("variables") or {}
    labels = list(dealt.get("moral_considerations") or []) + list(dealt.get("epistemic_considerations") or [])
    considerations = []
    for label in labels:
        try:
            gloss = context.catalog.entry_by_label(label).gloss if context.catalog else ""
        except KeyError:
            gloss = ""
        considerations.append({"label": label, "gloss": gloss})
    return considerations


def _fill(template: str, **values: str) -> str:
    """Substitute placeholders; substituted text (the document) is never re-parsed, so its braces are safe."""
    return template.format(**values)


def render_prompt(record: dict, context: JudgeContext) -> str:
    """The judge prompt: instructions, variable values with meanings, considerations, document text.
    Reads only the record's variables and content, never its description or ids."""
    variables = judged_variables(record, context)
    variable_block = "\n\n".join(
        f"{name}\n  Meaning: {context.variable_meanings[name]}\n  Dealt value: {value}"
        for name, value in variables.items()
    )
    considerations = assigned_considerations(record, context)
    considerations_section = ""
    if considerations:
        considerations_block = "\n".join(
            f"- {item['label']}" + (f": {item['gloss']}" if item["gloss"] else "") for item in considerations)
        considerations_section = "\n" + _fill(
            context.considerations_template, considerations_block=considerations_block) + "\n"
    return _fill(
        context.judge_template,
        variable_block=variable_block,
        considerations_section=considerations_section,
        document=record.get("content") or "",
    )


# ---------------------------------------------------------------- schema and parsing

def _verdict_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "persisted": {"type": "string", "enum": list(VERDICTS)},
        },
        "required": ["reasoning", "persisted"],
        "additionalProperties": False,
    }


def build_schema(variable_names: list[str], consideration_labels: list[str]) -> dict:
    """The answer schema for one document: one required object per dealt variable (reasoning
    first, then the call) and, when considerations were assigned, a required array of
    per-consideration items whose label is restricted to this document's assigned labels."""
    properties = {name: _verdict_schema() for name in variable_names}
    if consideration_labels:
        item = _verdict_schema()
        item["properties"] = {"label": {"type": "string", "enum": list(consideration_labels)}, **item["properties"]}
        item["required"] = ["label", "reasoning", "persisted"]
        properties[CONSIDERATIONS_KEY] = {"type": "array", "items": item}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _checked_verdict(where: str, entry) -> dict:
    if not isinstance(entry, dict):
        raise PersistenceAnswerError(f"{where}: not an object")
    reasoning, persisted = entry.get("reasoning"), entry.get("persisted")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise PersistenceAnswerError(f"{where}: missing reasoning")
    if persisted not in VERDICTS:
        raise PersistenceAnswerError(f"{where}: persisted must be 'yes' or 'no', got {persisted!r}")
    return {"reasoning": reasoning.strip(), "persisted": persisted}


def parse_answer(raw: str, variable_names: list[str], consideration_labels: list[str]) -> tuple[dict, list[dict]]:
    """Validate the judge's JSON; return ({variable: verdict}, [consideration verdicts in the
    assigned order]). Raises PersistenceAnswerError on anything unusable: unparseable JSON, a
    missing variable, a bad verdict, a missing/duplicate/unassigned consideration label."""
    try:
        parsed = utils.extract_json_object(raw)
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise PersistenceAnswerError(f"answer is not a JSON object ({error})") from error
    if not isinstance(parsed, dict):
        raise PersistenceAnswerError("answer is not a JSON object")
    variable_verdicts = {}
    for name in variable_names:
        if name not in parsed:
            raise PersistenceAnswerError(f"variable {name!r} is missing")
        variable_verdicts[name] = _checked_verdict(f"variable {name!r}", parsed[name])

    consideration_verdicts: list[dict] = []
    if consideration_labels:
        items = parsed.get(CONSIDERATIONS_KEY)
        if not isinstance(items, list):
            raise PersistenceAnswerError("considerations array is missing")
        by_label: dict[str, dict] = {}
        for item in items:
            label = item.get("label") if isinstance(item, dict) else None
            if label not in consideration_labels:
                raise PersistenceAnswerError(f"consideration label {label!r} was not assigned to this document")
            if label in by_label:
                raise PersistenceAnswerError(f"consideration {label!r} appears more than once")
            by_label[label] = {"label": label, **_checked_verdict(f"consideration {label!r}", item)}
        missing = [label for label in consideration_labels if label not in by_label]
        if missing:
            raise PersistenceAnswerError(f"consideration(s) missing: {missing}")
        consideration_verdicts = [by_label[label] for label in consideration_labels]
    return variable_verdicts, consideration_verdicts


# ---------------------------------------------------------------- one document

def judge_document(record: dict, context: JudgeContext, model: str | None, max_attempts: int) -> dict:
    """Judge one document; return its judgments row (status 'ok' or 'failed'). Never raises:
    a failing call or an answer still unusable after max_attempts is recorded, not fatal."""
    doc_id = record.get("doc_id")
    variables = judged_variables(record, context)
    considerations = assigned_considerations(record, context)
    labels = [item["label"] for item in considerations]
    if not variables:
        return {"doc_id": doc_id, "status": "failed", "attempts": 0, "error": "record has no dealt variables"}
    prompt = render_prompt(record, context)
    schema = build_schema(list(variables), labels)
    max_tokens = BASE_MAX_TOKENS + TOKENS_PER_ITEM * (len(variables) + len(labels))
    problem = ""
    # The API's structured output refuses some documents outright (stop_reason "refusal", no
    # output) although the same prompt without it is answered. After such a refusal the schema
    # goes into the prompt instead, and parse_answer enforces it in code.
    enforce_schema_through_api = True
    for attempt in range(1, max_attempts + 1):
        if enforce_schema_through_api:
            call_prompt, call_schema = prompt, schema
        else:
            call_prompt = (f"{prompt}\n\nAnswer with only a JSON object that matches this JSON schema:\n"
                           f"{json.dumps(schema, ensure_ascii=False)}")
            call_schema = None
        try:
            raw, stop_reason = api.call_claude(
                call_prompt, model=model, max_tokens=max_tokens, stage=STAGE_TAG, item_id=doc_id,
                return_stop_reason=True, json_schema=call_schema)
        except Exception as error:  # one failing call must not discard its siblings' work
            return {"doc_id": doc_id, "status": "failed", "attempts": attempt,
                    "error": f"API call failed ({type(error).__name__}: {error})"}
        try:
            if stop_reason == "refusal" and enforce_schema_through_api:
                enforce_schema_through_api = False
                raise PersistenceAnswerError("refused with the API-enforced schema")
            if stop_reason != "end_turn":
                raise PersistenceAnswerError(f"truncated answer (stop_reason={stop_reason})")
            variable_verdicts, consideration_verdicts = parse_answer(raw, list(variables), labels)
        except PersistenceAnswerError as error:
            problem = str(error)
            continue
        glosses = {item["label"]: item["gloss"] for item in considerations}
        row = {
            "doc_id": doc_id, "status": "ok", "attempts": attempt, "model": model,
            "schema_enforcement": "api_structured_output" if enforce_schema_through_api else "prompt_and_code_validation",
            "variables": {name: {"dealt_value": variables[name], **verdict}
                          for name, verdict in variable_verdicts.items()},
        }
        if considerations:
            row[CONSIDERATIONS_KEY] = [{**verdict, "gloss": glosses[verdict["label"]]}
                                       for verdict in consideration_verdicts]
        return row
    return {"doc_id": doc_id, "status": "failed", "attempts": max_attempts,
            "error": f"unusable answer ({problem}) after {max_attempts} attempts"}


# ---------------------------------------------------------------- summaries

def latest_ok_rows(rows: list[dict]) -> dict[str, dict]:
    """{doc_id: judgments row} for the documents with an 'ok' row (a later ok row replaces an earlier one)."""
    return {row["doc_id"]: row for row in rows if row.get("status") == "ok"}


def _summary_row(kind: str, name: str, yes_count: int, no_count: int) -> dict:
    judged = yes_count + no_count
    return {"kind": kind, "name": name, "documents_judged": judged, "yes_count": yes_count,
            "no_count": no_count, "share_no": round(no_count / judged, 4) if judged else ""}


def summarize(ok_rows: dict[str, dict], variable_order: list[str]) -> list[dict]:
    """Summary rows: one per variable, one per consideration label (sorted), then one for all
    considerations pooled. Each counts verdicts, not documents, for the considerations rows."""
    variable_counts = {name: [0, 0] for name in variable_order}
    label_counts: dict[str, list[int]] = {}
    for row in ok_rows.values():
        for name, entry in row["variables"].items():
            variable_counts.setdefault(name, [0, 0])[entry["persisted"] == "no"] += 1
        for entry in row.get(CONSIDERATIONS_KEY, []):
            label_counts.setdefault(entry["label"], [0, 0])[entry["persisted"] == "no"] += 1
    rows = [_summary_row("variable", name, *counts) for name, counts in variable_counts.items()
            if counts[0] + counts[1]]
    rows += [_summary_row("consideration", label, *label_counts[label]) for label in sorted(label_counts)]
    if label_counts:
        rows.append(_summary_row("considerations_overall", "all considerations",
                                 sum(c[0] for c in label_counts.values()),
                                 sum(c[1] for c in label_counts.values())))
    return rows


def failed_checks(ok_rows: dict[str, dict]) -> list[dict]:
    """Every 'no': doc_id, kind, variable (or consideration label), dealt value, reasoning."""
    failures = []
    for doc_id, row in ok_rows.items():
        for name, entry in row["variables"].items():
            if entry["persisted"] == "no":
                failures.append({"doc_id": doc_id, "kind": "variable", "variable": name,
                                 "dealt_value": entry["dealt_value"], "reasoning": entry["reasoning"]})
        for entry in row.get(CONSIDERATIONS_KEY, []):
            if entry["persisted"] == "no":
                value = f"{entry['label']}: {entry['gloss']}" if entry.get("gloss") else entry["label"]
                failures.append({"doc_id": doc_id, "kind": "consideration", "variable": entry["label"],
                                 "dealt_value": value, "reasoning": entry["reasoning"]})
    return failures


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary_table(summary_rows: list[dict], documents_judged: int) -> None:
    print(f"\nVARIABLE PERSISTENCE ({documents_judged} documents judged)")
    print(f"   {'item':<52}{'judged':>8}{'yes':>6}{'no':>6}{'share no':>10}")
    for row in summary_rows:
        name = row["name"] if row["kind"] == "variable" else f"[{row['kind'].split('_')[0]}] {row['name']}"
        share = f"{row['share_no']:.0%}" if row["share_no"] != "" else "n/a"
        print(f"   {name[:52]:<52}{row['documents_judged']:>8}{row['yes_count']:>6}{row['no_count']:>6}{share:>10}")


# ---------------------------------------------------------------- dry run

def estimate_cost(prompts: list[str], items_per_document: list[int], model: str) -> dict:
    """Rough dollar estimate from the repo's pricing table: input from prompt length, output from a
    fixed per-item allowance. Models missing from the table are priced at Sonnet rates, as api.py does."""
    prices = api._PRICING.get(model)
    priced_as_fallback = prices is None
    prices = prices or (3.00, 15.00)
    input_tokens = sum(len(prompt) for prompt in prompts) / ESTIMATED_CHARACTERS_PER_TOKEN
    output_tokens = sum(items * ESTIMATED_OUTPUT_TOKENS_PER_ITEM + 20 for items in items_per_document)
    return {
        "input_tokens": round(input_tokens), "output_tokens": round(output_tokens),
        "cost_usd": round(input_tokens / 1e6 * prices[0] + output_tokens / 1e6 * prices[1], 2),
        "priced_as_fallback": priced_as_fallback, "prices_per_million": prices,
    }


def dry_run(records: list[dict], context: JudgeContext, model: str | None, config: dict) -> None:
    """Print the first document's prompt and schema, the call count and a rough cost. No calls, no key."""
    resolved_model = model or config.get("model", "claude-sonnet-5")
    first = records[0]
    variables = judged_variables(first, context)
    labels = [item["label"] for item in assigned_considerations(first, context)]
    print("\n=== RENDERED PROMPT (first document) ===")
    print(render_prompt(first, context))
    print("\n=== ANSWER SCHEMA (first document) ===")
    print(json.dumps(build_schema(list(variables), labels), indent=2))
    prompts = [render_prompt(record, context) for record in records]
    items = [len(judged_variables(r, context)) + len(assigned_considerations(r, context)) for r in records]
    estimate = estimate_cost(prompts, items, resolved_model)
    print(f"\n=== DRY RUN: {len(records)} calls (one per document) with model {resolved_model} ===")
    note = (f" (model not in pricing table; priced at ${estimate['prices_per_million'][0]}/"
            f"${estimate['prices_per_million'][1]} per million)") if estimate["priced_as_fallback"] else ""
    print(f"  estimated input ~{estimate['input_tokens']:,} tokens, output ~{estimate['output_tokens']:,} tokens, "
          f"cost ~${estimate['cost_usd']:.2f}{note}")
    print("  (rough: assumes one attempt per document, "
          f"{ESTIMATED_CHARACTERS_PER_TOKEN} characters per token, no hidden reasoning tokens)")


# ---------------------------------------------------------------- run

def resolve_model(config: dict) -> str | None:
    sdf = config.get("sdf", {})
    return (sdf.get("variable_persistence_model") or sdf.get("score_model") or sdf.get("plan_model") or None)


def select_records(records: list[dict], doc_ids: list[str] | None, sample: int | None, seed: int) -> list[dict]:
    """Restrict to the named documents, then to a seeded sample (corpus order is kept)."""
    if doc_ids:
        wanted = set(doc_ids)
        records = [r for r in records if r.get("doc_id") in wanted]
        unknown = wanted - {r.get("doc_id") for r in records}
        if unknown:
            raise SystemExit(f"doc ids not found in the corpus: {sorted(unknown)}")
    if sample is not None and sample < len(records):
        chosen = set(random.Random(seed).sample(range(len(records)), sample))
        records = [r for i, r in enumerate(records) if i in chosen]
    return records


def output_names(timestamp: str) -> dict[str, str]:
    return {
        "judgments": f"persistence_judgments_{timestamp}.jsonl",
        "summary": f"per_variable_persistence_summary_{timestamp}.csv",
        "failed": f"failed_persistence_checks_{timestamp}.jsonl",
        "cost": f"persistence_cost_log_{timestamp}.jsonl",
    }


def existing_timestamp(folder: Path) -> str | None:
    """The timestamp of the judgments file already in a --resume folder (the latest, if several)."""
    found = sorted(folder.glob("persistence_judgments_*.jsonl"))
    return found[-1].stem.removeprefix("persistence_judgments_") if found else None


def report_directory_of(input_argument: str) -> Path:
    """<run>/audit for a run directory or a corpus in <run>/final/; next to a loose corpus file otherwise."""
    run_directory = run_directory_of(input_argument)
    return (run_directory if run_directory is not None else Path(input_argument).parent) / "audit"


def default_output_directory(report_directory: Path, timestamp: str, is_small_sample: bool) -> Path:
    """Full runs go in the report directory; --sample and --doc-ids runs in its small_sample_checks subfolder."""
    parent = report_directory / SMALL_SAMPLE_CHECKS_DIRECTORY_NAME if is_small_sample else report_directory
    return parent / f"variable_persistence_{timestamp}"


def build_run_configuration(corpus_path: Path, model: str | None) -> dict:
    """What a resume must match: the judge prompt, the variable meanings, the model and the corpus."""
    return {
        "model": model,
        "prompt_file_sha256": {PROMPT_PATH.name: sha256_of_file(PROMPT_PATH),
                               MEANINGS_PATH.name: sha256_of_file(MEANINGS_PATH)},
        "corpus_path": str(Path(corpus_path).resolve()),
        "corpus_sha256": sha256_of_file(corpus_path),
    }


def record_run_configuration(output_dir: Path, timestamp: str, is_resume: bool, configuration: dict) -> None:
    """Write the run's configuration file; on a resume, first refuse any change to what it recorded."""
    if is_resume:
        check_resume_is_compatible(output_dir, configuration, compared_settings=("model",))
    write_run_configuration(output_dir, timestamp, configuration)


def run_judging(records: list[dict], context: JudgeContext, config: dict, output_dir: Path, timestamp: str) -> None:
    names = output_names(timestamp)
    judgments_path = output_dir / names["judgments"]
    existing = utils.load_jsonl(judgments_path) if judgments_path.exists() else []
    done = set(latest_ok_rows(existing))
    pending = [r for r in records if r.get("doc_id") not in done]
    print(f"  {len(records)} documents selected, {len(records) - len(pending)} already judged, "
          f"{len(pending)} to judge")

    model = resolve_model(config)
    sdf = config.get("sdf", {})
    max_attempts = int(sdf.get("variable_persistence_max_attempts", DEFAULT_MAX_ATTEMPTS))
    workers = max(int(config.get("workers", 1)), 1)

    if pending:
        api.init(config["_config_path"], cost_log_path=output_dir / names["cost"])
        judged = failed = 0
        started = last_progress = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(judge_document, record, context, model, max_attempts) for record in pending]
            for future in as_completed(futures):
                row = future.result()
                utils.append_jsonl(row, judgments_path)  # only this thread writes
                if row["status"] == "ok":
                    judged += 1
                else:
                    failed += 1
                    print(f"  {row['doc_id']}: {row['error']} (will be retried on --resume)")
                if time.monotonic() - last_progress >= PROGRESS_INTERVAL_SECONDS:
                    last_progress = time.monotonic()
                    print(f"  Progress: {judged} judged, {failed} failed of {len(pending)} "
                          f"({last_progress - started:.1f}s elapsed)")
        print(f"  Judging finished: {judged} judged, {failed} failed ({time.monotonic() - started:.1f}s), "
              f"cost ${api.get_total_cost():.4f}")

    ok_rows = latest_ok_rows(utils.load_jsonl(judgments_path)) if judgments_path.exists() else {}
    selected_ids = {r.get("doc_id") for r in records}
    ok_rows = {doc_id: row for doc_id, row in ok_rows.items() if doc_id in selected_ids}
    if not ok_rows:
        raise SystemExit("no documents judged (every call failed or was unusable) — nothing to summarize")
    summary_rows = summarize(ok_rows, list(context.variable_meanings))
    write_summary_csv(summary_rows, output_dir / names["summary"])
    failures = failed_checks(ok_rows)
    with open(output_dir / names["failed"], "w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    print_summary_table(summary_rows, len(ok_rows))
    print(f"\n  {len(failures)} 'no' verdicts written to {output_dir / names['failed']}")
    print(f"  Outputs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="outputs/sdf/latest", help="Run directory or sdf_corpus.jsonl path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample", type=int, default=None, help="Judge a seeded random sample of N documents")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --sample")
    parser.add_argument("--doc-ids", nargs="+", default=None, help="Judge only these doc_ids")
    parser.add_argument("--output-dir", default=None, help="Output folder (default: <run>/audit/variable_persistence_<timestamp>, or under audit/small_sample_checks for --sample and --doc-ids)")
    parser.add_argument("--resume", default=None, help="Continue an earlier output folder, skipping documents already judged")
    parser.add_argument("--dry-run", action="store_true",
                        help="Make no calls and need no key: print the first prompt and schema, the call count and a cost estimate")
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)

    records, _ = resolve_input(args.input)
    if not records:
        raise SystemExit("Corpus is empty — nothing to judge.")
    config = utils.load_config(args.config)
    config["_config_path"] = args.config
    context = JudgeContext(find_prompts_dir(run_directory_of(args.input)))
    records = select_records(records, args.doc_ids, args.sample, args.seed)
    if not records:
        raise SystemExit("No documents selected.")

    if args.dry_run:
        dry_run(records, context, resolve_model(config), config)
        return

    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.is_dir():
            raise SystemExit(f"--resume folder not found: {output_dir}")
        timestamp = existing_timestamp(output_dir) or datetime.now().strftime("%Y-%m-%d_%H-%M")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = Path(args.output_dir) if args.output_dir else default_output_directory(
            report_directory_of(args.input), timestamp, is_small_sample=bool(args.sample or args.doc_ids))
    utils.ensure_dir(output_dir)
    corpus_path = corpus_path_of(args.input)
    record_run_configuration(output_dir, timestamp, bool(args.resume),
                             build_run_configuration(corpus_path, resolve_model(config) or config.get("model")))
    print(f"=== SDF variable persistence check: {args.input} ({len(records)} documents) ===")
    run_judging(records, context, config, output_dir, timestamp)


if __name__ == "__main__":
    main()
