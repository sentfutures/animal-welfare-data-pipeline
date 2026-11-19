"""Layer 1 selector: the model scores, code assigns. One scoring call per eligible document scores every offered entry 1-5 (entries shuffled per
document with a seeded RNG; the model never sees targets and never reproduces ids); a linear programme in ``considerations_assignment.py`` then assigns over all scores.
Rows are appended to consideration_scores.jsonl (that finishes a document); an unusable answer is retried up to ``sdf.max_scoring_attempts``, then recorded in consideration_scoring_failures.jsonl."""


import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdf_pipeline import considerations_assignment as assignment
from sdf_pipeline.considerations_catalog import (
    CONSIDERATIONS_FILENAME, EPISTEMIC_LIST, LIST_NAMES, MORAL_LIST,
    Consideration, ConsiderationCatalog, load_considerations,
)
from sdf_pipeline.considerations_coverage import corpus_coverage
from sdf_pipeline.shared_step_machinery import structured_answers
from sdf_pipeline.shared_step_machinery.step_record import LAYER1, Run, StageFile, Step, loaded_once_per_run
from sdf_pipeline.shared_step_machinery.step_runner import find_pending, maximum_attempts, run_step
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, StepPolicy, StepProgress,
    WhatAFailedDocumentDoes, WhereAnErrorIsAThrownCall, line_for_document_left_to_retry, reason_without_error_prefix)
from shared import utils
from shared.matrix import split_sections

STAGE_TAG = "layer1_select"
TEMPLATE_FILENAME = "layer1_select_considerations.txt"
SCORES_FILENAME = "consideration_scores.jsonl"
FAILURES_FILENAME = "consideration_scoring_failures.jsonl"
SELECTIONS_FILENAME = "consideration_selections.jsonl"
COVERAGE_FILENAME = "consideration_coverage.json"
DEFAULT_MAX_SCORING_ATTEMPTS = 3
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

    @classmethod
    def from_plan(cls, plan: dict) -> "DocumentFacts":
        """The facts of a plan record: its description and the three dealt values the selector shows."""
        variables = plan["variables"]
        return cls(
            document_description=plan["description"],
            document_type=variables.get("document_type", ""),
            centrality=variables.get("centrality", ""),
            sentient_category=variables.get("sentient_category", ""))


# ---------------------------------------------------------------- eligibility, prompt and schema

def load_selection_template(prompts_dir: Path) -> str | None:
    """The selector template, or None for a run whose snapshot predates the selector."""
    return structured_answers.read_template_if_present(prompts_dir, TEMPLATE_FILENAME)


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
    answer = structured_answers.parse_json_answer(
        raw, ScoringAnswerError, caught_exceptions=(ValueError,), failure_message="no usable JSON object ({error})")
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


def scoring_max_tokens(offered: dict[str, list[Consideration]]) -> int:
    """The token limit of one scoring call: a base plus an allowance per offered entry."""
    return _BASE_MAX_TOKENS + _TOKENS_PER_ENTRY * sum(len(entries) for entries in offered.values())


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


def assign_considerations_to_documents(
        config: dict, catalog: ConsiderationCatalog, eligible: list[dict], score_rows: list[dict]) -> tuple[list[dict], dict]:
    """``(selection rows, coverage report)`` of the programme over every scored document; ``eligible`` are the plans offered considerations."""
    seed = config["sdf"].get("seed", 0)
    limits = assignment.limits_from_config(config)
    document_is_digital = digital_mind_flags(catalog, eligible)
    rows_by_document_id = {row["prompt_id"]: row for row in score_rows}
    flat_scores = assignment.flat_scores(rows_by_document_id)
    assigned_labels_by_document_id = assignment.assign_considerations(flat_scores, catalog, document_is_digital, limits, seed)
    selection_rows = _selection_rows(catalog, rows_by_document_id, assigned_labels_by_document_id)
    coverage_report = assignment.coverage_report(
        flat_scores, catalog, document_is_digital, assigned_labels_by_document_id, limits)
    return selection_rows, coverage_report


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


SCORES = StageFile(LAYER1, SCORES_FILENAME, "prompt_id")


@loaded_once_per_run
def _catalog(run: Run) -> ConsiderationCatalog:
    return load_considerations(run.prompts_directory)


@loaded_once_per_run
def _template(run: Run) -> str:
    return load_selection_template(run.prompts_directory)


def offered_entries(plan: dict, run: Run) -> dict[str, list[Consideration]]:
    """The entries offered for the plan, each list in its shuffled order: a pure function of the catalog, the plan's
    variables, the run's seed and the prompt id."""
    return shuffled_offer(
        eligible_entries_for_document(_catalog(run), plan["variables"]), run.config["sdf"].get("seed", 0), plan["prompt_id"])


def applies(plan: dict, run: Run) -> bool:
    """A plan with a description, outside the no-welfare-stake arm (when that is skipped)."""
    skip_no_stake = bool(run.config["sdf"].get("select_skip_no_stake", True))
    return bool(plan.get("description")) and not (skip_no_stake and _catalog(run).is_skipped_resolution(plan["variables"]))


def build_request(documents: list, run: Run) -> ModelRequest:
    """One scoring call for the plan record (``prompt_id``, ``description``, ``variables``)."""
    (plan,) = documents
    offered = offered_entries(plan, run)
    system, user = build_scoring_prompt(_template(run), DocumentFacts.from_plan(plan), offered)
    return STEP.model_request(
        plan, user, system, run, max_tokens=scoring_max_tokens(offered), json_schema=build_scoring_schema(offered))


def read_reply(documents: list, reply: ModelReply, run: Run) -> dict:
    """An unusable answer (truncated, malformed, a label missing or repeated, a bad score) is asked again, the same
    request, up to ``sdf.max_scoring_attempts`` times in one run, then left behind with the last problem."""
    (plan,) = documents
    try:
        structured_answers.raise_if_answer_was_truncated(reply.stop_reason, ScoringAnswerError)
        scores = parse_scoring_answer(reply.text, offered_entries(plan, run))
    except ScoringAnswerError as error:
        problem = str(error)
        if reply.attempt >= maximum_attempts(STEP, run):
            problem = f"{problem} (after {maximum_attempts(STEP, run)} attempts)"
        return {plan["prompt_id"]: Failed(FailureKind.UNUSABLE_ANSWER, problem)}
    return {plan["prompt_id"]: Accepted({"scores": scores, "attempts": reply.attempt})}


def _problem_text(failed: Failed) -> str:
    if failed.kind is FailureKind.UNUSABLE_ANSWER:
        return f"unusable answer ({failed.reason})"
    return f"API call failed ({reason_without_error_prefix(failed.reason)})"


def records(documents: list, outcomes: dict, run: Run) -> OutcomeRecords:
    """An accepted score row (in the scores file, which finishes the plan) or, for a failure, its problem in the
    failures file."""
    (plan,), (outcome,) = documents, outcomes.values()
    if isinstance(outcome, Failed):
        return OutcomeRecords(((FAILURES_FILENAME, {"prompt_id": plan["prompt_id"], "problem": _problem_text(outcome)}),))
    return OutcomeRecords(((SCORES_FILENAME, {
        "prompt_id": plan["prompt_id"], "scores": outcome.update["scores"], "attempts": outcome.update["attempts"]}),))


def progress_line(documents: list, outcomes: dict, progress: StepProgress, run: Run) -> str | None:
    (plan,), (outcome,) = documents, outcomes.values()
    lines = []
    if isinstance(outcome, Failed):
        lines.append(line_for_document_left_to_retry(
            plan["prompt_id"], _problem_text(outcome), ending="will be rescored on resume"))
    if progress.clock.is_due():
        elapsed_seconds = progress.clock.mark_printed()
        lines.append(f"  Scoring progress: {progress.answered} answered, {progress.failed} failed of "
                     f"{progress.pending_count} ({elapsed_seconds:.1f}s elapsed)")
    return "\n".join(lines) if lines else None


def after_run(plans: list, run_result, run: Run) -> None:
    """The summary line, then the systemic stop when every pending call failed (thrown or unusable). Nothing is
    printed for a run with nothing to score."""
    if not run_result.pending_document_ids:
        return
    progress = run_result.progress
    print(f"  Scoring: {progress.answered} answered, {progress.failed} failed ({progress.clock.seconds_since_start():.1f}s)")
    if len(run_result.failed_document_ids) == len(run_result.pending_document_ids):
        raise SystemExit(STEP.systemic_stop_message)


def stop_message(plans: list, run_result, run: Run) -> str:
    return (f"{STAGE_TAG}: {len(run_result.failed_document_ids)} of {len(run_result.pending_document_ids)} "
            "scoring calls failed — stopping before the "
            "assignment, because the assignment is solved over every document together and a "
            "document drafted without scores would be reassigned on resume; fix and --resume.")


STEP = Step(
    name="consideration_scoring", folder_key=LAYER1, stage_tag=STAGE_TAG, model_setting_keys=("select_model", "plan_model"),
    applies=applies, build_request=build_request, read_reply=read_reply, records=records, writes=(SCORES,),
    writes_as_replies_arrive=True,
    policy=StepPolicy(
        maximum_attempts_in_one_run=DEFAULT_MAX_SCORING_ATTEMPTS, maximum_attempts_setting="max_scoring_attempts",
        # ``after_run`` raises the systemic stop (after the summary line), so the runner's own check is switched off.
        failure_kinds_counting_towards_the_systemic_stop=frozenset(),
        failed_document=WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP,
        errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND
        | WhereAnErrorIsAThrownCall.INTERPRET),
    call_noun="scoring call", progress_line=progress_line, after_run=after_run, stop_message=stop_message)


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Score every eligible document, assign considerations in code, and return ``plans`` with the assigned entries
    attached. A disabled selector, or a prompt snapshot without the template or the considerations file, makes no
    calls and returns ``plans`` unchanged."""
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
    utils.ensure_dir(output_dir)
    scores_path = output_dir / SCORES_FILENAME

    run_of_scoring = Run(config, prompts_dir, {LAYER1: output_dir})
    eligible = eligible_plans(config, catalog, plans)
    plan_by_id = {p["prompt_id"]: p for p in eligible}
    documents_in_scoring_order = [plan_by_id[prompt_id] for prompt_id in sorted(plan_by_id)]
    _, pending = find_pending(STEP, run_of_scoring, documents_in_scoring_order)
    print(f"  Scoring considerations for {len(eligible)} specifications: "
          f"{len(eligible) - len(pending)} already scored, {len(pending)} to score")
    run_step(STEP, run_of_scoring, documents_in_scoring_order)  # stops the run before the assignment if any document failed

    score_rows = [r for r in utils.load_jsonl(scores_path) if r["prompt_id"] in plan_by_id]
    selection_rows, report = assign_considerations_to_documents(config, catalog, eligible, score_rows)
    (output_dir / SELECTIONS_FILENAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection_rows), encoding="utf-8")
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
