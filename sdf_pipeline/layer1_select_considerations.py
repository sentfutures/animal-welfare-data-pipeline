"""Layer 1 selector: the model scores, code assigns.

Runs after the implausibility screen, planning and the description rewrite, and
before drafting. The planner writes a scenario description with no consideration
attached; this stage then makes ONE scoring call per eligible document. The model
sees every eligible entry of both lists (moral and epistemic, from
prompts/sdf/considerations.txt; label plus gloss), in an order shuffled per
document with a seeded RNG, and no target or fullness information, and scores
each entry 1 to 5 for how naturally the scenario raises it (the scale is in
prompts/sdf/layer1_select_considerations.txt), with one short sentence on where
it arises for scores of 3 or more. Which entry each document actually gets is
then decided in code (considerations_assignment.py): a linear programme over all
scores that respects every entry's target count and the per-document caps.

Eligibility: documents with a skipped resolution (the no-welfare-stake arm) get
nothing; digital-minds-only entries are offered only on digital-mind documents.

Answer format: a JSON schema built per call (``build_scoring_schema``) with, per
list, an array of {label, score, where_it_arises} items, where ``label`` is an
enum of that list's offered labels (listed in the shuffled order), ``score`` is
an integer enum 1-5. Structured outputs cannot fix the array length or the label
at each position (no usable prefixItems / minItems above 1), so
``parse_scoring_answer`` checks in code that every label appears exactly once with
a valid score. Arbitrary labels are deliberately NOT used as JSON property names
(spaces and typographic quotes in property names are not guaranteed to be
supported). The model never reproduces ids.

Files in the layer 1 stage folder:
- consideration_scores.jsonl: one row per scored document ({prompt_id, scores:
  {list: {label: {score, where_it_arises}}}, attempts}), appended as each answer
  arrives; it is the checkpoint, and a rerun or ``--resume`` scores only the
  documents without a row. An unusable answer is retried up to
  ``sdf.max_scoring_attempts`` times (default 3) and then recorded in
  consideration_scoring_failures.jsonl, never crashing the run; that document is
  scored on the next resume.
- consideration_selections.jsonl: the assignment, rewritten after every run, one
  row per scored document ({prompt_id, outcomes: [{list, label, reason, status,
  score}]}); this is what ``attach_selections`` reads when a run resumes past layer 1.
- consideration_coverage.json: per entry target_count, achieved, documents
  scoring at or above the threshold and score histogram, plus documents with no consideration.

The assigned entries are assembled in code into ``considerations_spec``, a text
block stored on the plan record next to ``description`` and carried through the
draft, rewrite and final records (empty string when a document has none). Each
plan's ``variables`` also gains ``moral_considerations`` and
``epistemic_considerations``: lists of the assigned labels, for analysis.

Config (``sdf`` section): ``select_enabled`` (default true), ``select_model``
(falls back to ``plan_model``, then the global model), ``select_skip_no_stake``
(default true), ``max_scoring_attempts`` (3), ``minimum_assignable_score`` (3),
``max_moral_considerations_per_document`` (3),
``max_epistemic_considerations_per_document`` (3),
``max_considerations_per_document`` (4), ``score_weights`` ({3: 1, 4: 3, 5: 9}),
``penalty_step_per_further_consideration_in_a_list`` (2); the calls use ``workers`` threads.
"""

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdf_pipeline import considerations_assignment as assignment
from sdf_pipeline.considerations_catalog import (
    CONSIDERATIONS_FILENAME, EPISTEMIC_LIST, LIST_NAMES, MORAL_LIST,
    Consideration, ConsiderationCatalog, load_considerations,
)
from sdf_pipeline.considerations_coverage import corpus_coverage
from shared import api, utils
from shared.matrix import split_sections

STAGE_TAG = "layer1_select"
TEMPLATE_FILENAME = "layer1_select_considerations.txt"
SCORES_FILENAME = "consideration_scores.jsonl"
FAILURES_FILENAME = "consideration_scoring_failures.jsonl"
SELECTIONS_FILENAME = "consideration_selections.jsonl"
COVERAGE_FILENAME = "consideration_coverage.json"
DEFAULT_MAX_SCORING_ATTEMPTS = 3
PROGRESS_INTERVAL_SECONDS = 180   # progress lines are printed no more often than this
_BASE_MAX_TOKENS = 400
_TOKENS_PER_ENTRY = 120           # label, score and one short sentence; bounds api-backend cost

NO_ENTRIES_MARKER = "(no entries offered)"
STATUS_ACCEPTED = "accepted"
STATUS_NONE = "none"


class ScoringAnswerError(ValueError):
    """The model's scoring answer was truncated, malformed, or did not score every label exactly once."""


@dataclass(frozen=True)
class DocumentFacts:
    """What the selector is shown about one document (all of it already in the plan record)."""
    document_description: str
    document_type: str
    centrality: str
    sentient_category: str


# ---------------------------------------------------------------- eligibility, prompt and schema

def load_selection_template(prompts_dir: Path) -> str | None:
    """The selector template, or None for a run whose snapshot predates the selector."""
    path = Path(prompts_dir) / TEMPLATE_FILENAME
    return path.read_text(encoding="utf-8") if path.is_file() else None


def eligible_entries_for_document(catalog: ConsiderationCatalog, variables: dict) -> dict[str, list[Consideration]]:
    """The entries this document may be scored on, per list, in catalog order."""
    is_digital = catalog.is_digital_mind_document(variables)
    return {
        list_name: [e for e in catalog.entries_of(list_name) if is_digital or not e.digital_minds_only]
        for list_name in LIST_NAMES
    }


def shuffled_offer(offered: dict[str, list[Consideration]], seed, prompt_id: str) -> dict[str, list[Consideration]]:
    """Each list in an order shuffled by an RNG seeded from the run seed and the prompt id (deterministic)."""
    shuffled = {}
    for list_name in LIST_NAMES:
        entries = list(offered.get(list_name) or [])
        random.Random(f"{seed}:{STAGE_TAG}:{prompt_id}:{list_name}").shuffle(entries)
        shuffled[list_name] = entries
    return shuffled


def render_entries(entries: list[Consideration]) -> str:
    return "\n".join(f"- {entry.display}" for entry in entries) if entries else NO_ENTRIES_MARKER


def build_scoring_prompt(
    template: str, facts: DocumentFacts, offered: dict[str, list[Consideration]],
) -> tuple[str | None, str]:
    """Render (system, user) for one document; ``offered`` is in the order to show."""
    rendered = template.format(
        document_type=facts.document_type,
        centrality=facts.centrality,
        sentient_category=facts.sentient_category,
        document_description=facts.document_description.strip(),
        moral_list=render_entries(offered.get(MORAL_LIST, [])),
        epistemic_list=render_entries(offered.get(EPISTEMIC_LIST, [])),
    )
    return split_sections(rendered)


def build_scoring_schema(offered: dict[str, list[Consideration]]) -> dict:
    """The answer schema for one call: for each list that has entries, an array of
    {label, score, where_it_arises} items whose ``label`` enum is the offered labels
    in the shuffled order and whose ``score`` is an integer enum 1-5. A list with no
    entries is left out of the schema."""
    def list_schema(entries: list[Consideration]) -> dict:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": [entry.label for entry in entries]},
                    "score": {"type": "integer", "enum": list(assignment.SCORE_VALUES)},
                    "where_it_arises": {"type": "string"},
                },
                "required": ["label", "score", "where_it_arises"],
                "additionalProperties": False,
            },
        }
    names = [name for name in LIST_NAMES if offered.get(name)]
    return {
        "type": "object",
        "properties": {name: list_schema(offered[name]) for name in names},
        "required": names,
        "additionalProperties": False,
    }


def parse_scoring_answer(raw: str, offered: dict[str, list[Consideration]]) -> dict[str, dict[str, dict]]:
    """{list_name: {label: {"score", "where_it_arises"}}} in the order the model answered.

    Raises ScoringAnswerError unless the answer is one JSON object holding, for every
    offered list, items that name each offered label exactly once with an integer score
    1-5 (and, for a score of 3 or more, a non-empty ``where_it_arises``), and nothing
    for a list that was not offered. ``where_it_arises`` is blanked for scores below 3."""
    try:
        answer = json.loads(raw or "")
    except ValueError as error:
        raise ScoringAnswerError(f"no usable JSON object ({error})") from error
    if not isinstance(answer, dict):
        raise ScoringAnswerError("answer is not a JSON object")
    parsed: dict[str, dict[str, dict]] = {}
    for list_name in LIST_NAMES:
        entries = offered.get(list_name) or []
        items = answer.get(list_name)
        if not entries:
            if items:
                raise ScoringAnswerError(f"{list_name}: scores given for a list that was not offered")
            parsed[list_name] = {}
            continue
        if not isinstance(items, list):
            raise ScoringAnswerError(f"{list_name}: missing or not an array")
        offered_labels = {entry.label for entry in entries}
        scored: dict[str, dict] = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("label"), str):
                raise ScoringAnswerError(f"{list_name}: malformed item {item!r}")
            label = item["label"].strip()
            if label not in offered_labels:
                raise ScoringAnswerError(f"{list_name}: {item['label']!r} is not on the list shown")
            if label in scored:
                raise ScoringAnswerError(f"{list_name}: {label!r} scored more than once")
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or score not in assignment.SCORE_VALUES:
                raise ScoringAnswerError(f"{list_name}: {label!r} has an invalid score {score!r}")
            where = item.get("where_it_arises")
            where = " ".join(where.split()) if isinstance(where, str) else ""
            if score >= 3 and not where:
                raise ScoringAnswerError(f"{list_name}: {label!r} scored {score} without saying where it arises")
            scored[label] = {"score": score, "where_it_arises": where if score >= 3 else ""}
        missing = sorted(offered_labels - set(scored))
        if missing:
            raise ScoringAnswerError(f"{list_name}: no score for {missing}")
        parsed[list_name] = scored
    return parsed


def score_document(
    facts: DocumentFacts,
    offered: dict[str, list[Consideration]],
    *,
    template: str,
    model: str | None = None,
    item_id: str | None = None,
    max_attempts: int = DEFAULT_MAX_SCORING_ATTEMPTS,
) -> tuple[dict[str, dict[str, dict]], int]:
    """Ask the model to score one document's offered entries; return (scores, attempts used).

    An unusable answer (truncated, malformed, a label missing or repeated, a bad score)
    is retried up to ``max_attempts`` times, then ScoringAnswerError is raised with the
    last problem. API errors propagate. Tags the call ``layer1_select`` for the cost log."""
    system, user = build_scoring_prompt(template, facts, offered)
    schema = build_scoring_schema(offered)
    entry_count = sum(len(entries) for entries in offered.values())
    problem = ""
    for attempt in range(1, max_attempts + 1):
        raw, stop_reason = api.call_claude(
            user,
            system_prompt=system or "",
            model=model,
            max_tokens=_BASE_MAX_TOKENS + _TOKENS_PER_ENTRY * entry_count,
            stage=STAGE_TAG,
            item_id=item_id,
            return_stop_reason=True,
            json_schema=schema,
        )
        try:
            if stop_reason != "end_turn":
                raise ScoringAnswerError(f"truncated answer (stop_reason={stop_reason})")
            return parse_scoring_answer(raw, offered), attempt
        except ScoringAnswerError as error:
            problem = str(error)
    raise ScoringAnswerError(f"{problem} (after {max_attempts} attempts)")


# ---------------------------------------------------------------- the considerations spec

def assemble_considerations_spec(picks: list) -> str:
    """The text block handed to later prompts: one line per pick, 'Label (gloss) —
    the selector's sentence on where it arises', moral picks first and each list
    in catalog order. Empty string when there are no picks."""
    return "\n".join(
        f"{entry.label} ({entry.gloss}) — {reason}" if entry.gloss else f"{entry.label} — {reason}"
        for entry, reason in picks
    )


# ---------------------------------------------------------------- stage

def selection_enabled(config: dict) -> bool:
    return bool(config["sdf"].get("select_enabled", True))


def eligible_plans(config: dict, catalog: ConsiderationCatalog, plans: list[dict]) -> list[dict]:
    """Plans with a description, minus the skipped (no-welfare-stake) resolutions."""
    skip_no_stake = bool(config["sdf"].get("select_skip_no_stake", True))
    return [
        p for p in plans
        if p.get("description") and not (skip_no_stake and catalog.is_skipped_resolution(p["variables"]))
    ]


def digital_mind_flags(catalog: ConsiderationCatalog, eligible: list[dict]) -> dict[str, bool]:
    return {p["prompt_id"]: catalog.is_digital_mind_document(p["variables"]) for p in eligible}


def _selection_rows(
    catalog: ConsiderationCatalog, score_rows: dict[str, dict], assigned: dict[str, list[str]],
) -> list[dict]:
    """One row per scored document: one accepted outcome per assigned entry, or a single
    "none" outcome per list with nothing assigned."""
    rows = []
    for prompt_id in sorted(score_rows):
        scores = score_rows[prompt_id]["scores"]
        outcomes = []
        for list_name in LIST_NAMES:
            labels = [label for label in assigned.get(prompt_id, []) if label in scores.get(list_name, {})]
            for label in labels:
                item = scores[list_name][label]
                outcomes.append({"list": list_name, "label": label, "reason": item["where_it_arises"],
                                 "status": STATUS_ACCEPTED, "score": item["score"]})
            if not labels:
                outcomes.append({"list": list_name, "label": None, "reason": "", "status": STATUS_NONE})
        rows.append({"prompt_id": prompt_id, "outcomes": outcomes})
    return rows


def _attach(catalog: ConsiderationCatalog, selection_rows: list[dict], plans: list[dict]) -> list[dict]:
    """Copies of ``plans`` with each described plan's assigned entries attached (plans without a
    description are INCOHERENT and pass through unchanged)."""
    by_document = {row["prompt_id"]: row["outcomes"] for row in selection_rows}
    attached = []
    for plan in plans:
        if not plan.get("description"):
            attached.append(plan)
            continue
        entries = sorted(
            ((catalog.entry_by_label(o["label"]), o["reason"])
             for o in by_document.get(plan["prompt_id"], []) if o["status"] == STATUS_ACCEPTED),
            key=lambda pair: (LIST_NAMES.index(pair[0].list_name),
                              catalog.entries_of(pair[0].list_name).index(pair[0])),
        )
        variables = {
            **plan["variables"],
            "moral_considerations": [e.label for e, _ in entries if e.list_name == MORAL_LIST],
            "epistemic_considerations": [e.label for e, _ in entries if e.list_name == EPISTEMIC_LIST],
        }
        attached.append({**plan, "variables": variables, "considerations_spec": assemble_considerations_spec(entries)})
    return attached


def attach_selections(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Plans with the saved selections attached, without any calls. Used when a run
    resumes past layer 1 and plans are re-read from disk. A disabled selector, a
    snapshot without the selector files, or a run with no saved selections returns
    ``plans`` unchanged."""
    if not selection_enabled(config):
        return plans
    if load_selection_template(prompts_dir) is None or not (Path(prompts_dir) / CONSIDERATIONS_FILENAME).is_file():
        return plans
    rows = utils.load_jsonl(Path(output_dir) / SELECTIONS_FILENAME)
    if not rows:
        return plans
    return _attach(load_considerations(prompts_dir), rows, plans)


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Score every eligible document, assign considerations in code, and return ``plans``
    with the assigned entries attached. A disabled selector, or a run whose prompt
    snapshot lacks the template or the considerations file, makes no calls and returns
    ``plans`` unchanged."""
    sdf = config["sdf"]
    if not selection_enabled(config):
        print("  Consideration selection disabled (sdf.select_enabled: false)")
        return plans
    template = load_selection_template(prompts_dir)
    if template is None or not (Path(prompts_dir) / CONSIDERATIONS_FILENAME).is_file():
        print(f"  No {TEMPLATE_FILENAME} or {CONSIDERATIONS_FILENAME} in this run's prompt snapshot "
              "— consideration selection skipped")
        return plans

    catalog = load_considerations(prompts_dir)
    model = sdf.get("select_model") or sdf.get("plan_model")
    seed = sdf.get("seed", 0)
    max_attempts = int(sdf.get("max_scoring_attempts", DEFAULT_MAX_SCORING_ATTEMPTS))
    workers = max(1, int(config.get("workers", 1)))
    utils.ensure_dir(output_dir)
    scores_path = output_dir / SCORES_FILENAME

    eligible = eligible_plans(config, catalog, plans)
    plan_by_id = {p["prompt_id"]: p for p in eligible}
    score_rows = {r["prompt_id"]: r for r in utils.load_jsonl(scores_path) if r["prompt_id"] in plan_by_id}
    pending = sorted(pid for pid in plan_by_id if pid not in score_rows)
    print(f"  Scoring considerations for {len(eligible)} specifications: "
          f"{len(score_rows)} already scored, {len(pending)} to score")

    write_lock = threading.Lock()

    def score_one(prompt_id: str):
        plan = plan_by_id[prompt_id]
        variables = plan["variables"]
        facts = DocumentFacts(
            document_description=plan["description"],
            document_type=variables.get("document_type", ""),
            centrality=variables.get("centrality", ""),
            sentient_category=variables.get("sentient_category", ""))
        offered = shuffled_offer(eligible_entries_for_document(catalog, variables), seed, prompt_id)
        try:
            scores, attempts = score_document(
                facts, offered, template=template, model=model, item_id=prompt_id, max_attempts=max_attempts)
            row = {"prompt_id": prompt_id, "scores": scores, "attempts": attempts}
            with write_lock:
                utils.append_jsonl(row, scores_path)   # checkpoint the moment the answer arrives
            return prompt_id, row, None
        except ScoringAnswerError as error:
            return prompt_id, None, f"unusable answer ({error})"
        except Exception as error:  # one failing call must not discard its siblings' work
            return prompt_id, None, f"API call failed ({type(error).__name__}: {error})"

    answered = failed = 0
    started = last_progress = time.monotonic()
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_one, prompt_id) for prompt_id in pending]
            for future in as_completed(futures):
                prompt_id, row, problem = future.result()
                if row is not None:
                    score_rows[prompt_id] = row
                    answered += 1
                else:
                    failed += 1
                    utils.append_jsonl({"prompt_id": prompt_id, "problem": problem}, output_dir / FAILURES_FILENAME)
                    print(f"  {prompt_id}: {problem} — will be rescored on resume")
                if time.monotonic() - last_progress >= PROGRESS_INTERVAL_SECONDS:
                    last_progress = time.monotonic()
                    print(f"  Scoring progress: {answered} answered, {failed} failed of {len(pending)} "
                          f"({last_progress - started:.1f}s elapsed)")
        print(f"  Scoring: {answered} answered, {failed} failed ({time.monotonic() - started:.1f}s)")
        if failed == len(pending):
            raise SystemExit(
                "layer1_select: every pending scoring call failed — this is systemic "
                "(auth, backend, or network), not per-document; fix and --resume.")

    limits = assignment.limits_from_config(config)
    document_is_digital = digital_mind_flags(catalog, eligible)
    flat = assignment.flat_scores({pid: row for pid, row in score_rows.items()})
    assigned = assignment.assign_considerations(flat, catalog, document_is_digital, limits, seed)
    selection_rows = _selection_rows(catalog, score_rows, assigned)
    (output_dir / SELECTIONS_FILENAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection_rows), encoding="utf-8")
    report = assignment.coverage_report(flat, catalog, document_is_digital, assigned, limits)
    (output_dir / COVERAGE_FILENAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_coverage_report(report)
    return _attach(catalog, selection_rows, plans)


# ---------------------------------------------------------------- reporting

def print_coverage_report(report: dict) -> None:
    print(f"  Considerations coverage over {report['eligible_documents']} eligible documents "
          f"({report['digital_mind_documents']} digital-mind):")
    for entry in report["entries"]:
        scoring = entry.get("documents_scoring_at_least_threshold")
        suffix = f", {scoring} documents scored at or above the threshold" if scoring is not None else ""
        print(f"    [{entry['list']}] {entry['label']}: {entry['achieved']}/{entry['target_count']}{suffix}")
    total = report["eligible_documents"]
    print(f"    no consideration: {report['documents_with_no_consideration']}/{total}, "
          f"one list only: {report['documents_with_only_one_list']}/{total}, "
          f"both lists: {report['documents_with_both_lists']}/{total}, "
          f"two or more from one list: {report['documents_with_two_or_more_in_one_list']}")


def load_considerations_outputs(run_dir: Path) -> dict | None:
    """The selector's results for a run, or None when it never ran (older runs).

    Returns {"catalog", "rows", "report"}: the catalog loaded from the run's own
    prompt snapshot, the saved selection rows, and the coverage report written at
    the end of layer 1 (None if the stage never finished)."""
    stage_dir = utils.sdf_stage_dir(run_dir, 1)
    rows = utils.load_jsonl(stage_dir / SELECTIONS_FILENAME)
    snapshot = Path(run_dir) / "inputs" / "prompts"
    if not rows or not (snapshot / CONSIDERATIONS_FILENAME).is_file():
        return None
    report_path = stage_dir / COVERAGE_FILENAME
    return {
        "catalog": load_considerations(snapshot),
        "rows": rows,
        "report": json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None,
    }


def corpus_coverage_report(outputs: dict, corpus: list[dict]) -> dict:
    """The coverage report measured on the final corpus documents."""
    return corpus_coverage(outputs["catalog"], [r["variables"] for r in corpus])
